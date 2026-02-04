"""
Distributed multi-GPU benchmark for matching problem.
Edit the CONFIG section below to change parameters.

Usage:
    torchrun --nproc_per_node=<num_gpus> benchmark/run_matching_benchmark_dist.py
"""

import csv
import time

import torch
from generate_synthetic_data import generate_synthetic_matching_input_args

from dualip.objectives.matching import MatchingInputArgs, MatchingSolverDualObjectiveFunctionDistributed
from dualip.optimizers.agd import AcceleratedGradientDescent
from dualip.preprocessing.precondition import jacobi_precondition
from dualip.utils.dist_utils import global_to_local_projection_map, split_tensors_to_devices

# =============================================================================
# CONFIG - Edit these values
# =============================================================================

# Data parameters (fixed across all runs)
NUM_SOURCES = 25_000_000
NUM_DESTINATIONS = 10_000
TARGET_SPARSITY = 0.001
SEED = 42

# Distributed parameters
NUM_GPUS = 2  # Number of GPUs to use for distributed training

# Solver parameters (fixed across all runs)
GAMMA = 1e-3
MAX_ITER = 1000
INITIAL_STEP_SIZE = 1e-3
MAX_STEP_SIZE = 1e-1

# Ablation toggles
USE_PRECONDITIONING = False  # Jacobi preconditioning


def get_output_filename():
    """Generate informative filename based on parameters."""
    parts = [
        f"s{NUM_SOURCES//1_000_000}M",
        f"d{NUM_DESTINATIONS//1_000}K",
        f"sp{TARGET_SPARSITY}",
        f"g{GAMMA}",
        f"iter{MAX_ITER}",
    ]
    if USE_PRECONDITIONING:
        parts.append("precon")
    return "_".join(parts) + ".csv"


# =============================================================================
# END CONFIG
# =============================================================================


def run_benchmark():
    rng = None  # np.random.default_rng(SEED)

    # Generate data BEFORE initializing distributed
    # (Each process will generate identical data with same seed)
    print("\n[1/4] Generating data...")
    t0 = time.perf_counter()
    # Use CPU temporarily for data generation
    temp_device = "cpu"
    input_args: MatchingInputArgs = generate_synthetic_matching_input_args(
        num_sources=NUM_SOURCES,
        num_destinations=NUM_DESTINATIONS,
        target_sparsity=TARGET_SPARSITY,
        device=temp_device,
        rng=rng,
    )
    data_time = time.perf_counter() - t0
    print(f"      {data_time:.3f}s | NNZ: {input_args.A._nnz()}")

    # Preconditioning (before splitting)
    if USE_PRECONDITIONING:
        print("      Applying preconditioning...")
        jacobi_precondition(input_args.A, input_args.b_vec)

    # Split data BEFORE initializing process group
    print("[2/4] Splitting data...")
    compute_devices = [f"cuda:{i}" for i in range(NUM_GPUS)]

    A_splits, c_splits, split_index_map = split_tensors_to_devices(input_args.A, input_args.c, compute_devices)

    # NOW initialize distributed
    print("[3/4] Initializing distributed...")
    torch.distributed.init_process_group(backend="nccl")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    # Each rank sets its device
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    if rank == 0:
        print("=" * 60)
        print("CONFIG")
        print("=" * 60)
        print(f"  Data: {NUM_SOURCES} sources x {NUM_DESTINATIONS} destinations")
        print(f"  Sparsity: {TARGET_SPARSITY}")
        print(f"  Seed: {SEED}")
        print(f"  World size: {world_size}")
        print(f"  Preconditioning: {USE_PRECONDITIONING}")
        print(f"  Gamma: {GAMMA}, Max iter: {MAX_ITER}")
        print("=" * 60)

    # Each rank takes its partition
    A_local = A_splits[rank].to(device)
    c_local = c_splits[rank].to(device)
    pm_local = global_to_local_projection_map(input_args.projection_map, split_index_map[rank])

    # Create local input args (b_vec=None for local partition)
    local_input_args = MatchingInputArgs(
        A=A_local, c=c_local, projection_map=pm_local, b_vec=None, equality_mask=input_args.equality_mask
    )

    # Create distributed objective with local data
    # host_device is cuda:0 for aggregation (all ranks send results there)
    if rank == 0:
        print("[4/4] Creating objective...")
    t0 = time.perf_counter()
    objective = MatchingSolverDualObjectiveFunctionDistributed(
        local_matching_input_args=local_input_args,
        b_vec=input_args.b_vec,
        gamma=GAMMA,
        host_device="cuda:0",
    )
    obj_time = time.perf_counter() - t0
    if rank == 0:
        print(f"      {obj_time:.3f}s")

    # Create solver
    if rank == 0:
        print("Running solver...")
    solver = AcceleratedGradientDescent(
        max_iter=MAX_ITER,
        gamma=GAMMA,
        initial_step_size=INITIAL_STEP_SIZE,
        max_step_size=MAX_STEP_SIZE,
        save_primal=False,  # Not yet supported in distributed mode
    )

    # Initialize dual variables on each rank's device (for broadcast to work)
    initial_dual = torch.zeros_like(input_args.b_vec).to(device)

    # Synchronize all ranks before timing to ensure fair measurement
    torch.distributed.barrier()

    # Only rank 0 measures time
    if rank == 0:
        t0 = time.perf_counter()

    result = solver.maximize(objective, initial_dual, rank=rank)

    # Synchronize all ranks after solver completes
    torch.distributed.barrier()

    # Rank 0 calculates elapsed time
    if rank == 0:
        solve_time = time.perf_counter() - t0

    # Only rank 0 prints and saves results
    if rank == 0:
        # Results
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"  Solve time: {solve_time:.3f}s ({solve_time/MAX_ITER*1000:.2f} ms/iter)")
        print(f"  Dual objective: {result.dual_objective:.6f}")
        if (
            hasattr(result.objective_result, "primal_objective")
            and result.objective_result.primal_objective is not None
        ):
            print(f"  Primal objective: {result.objective_result.primal_objective.item():.6f}")
        print(f"  Reg penalty: {result.objective_result.reg_penalty.item():.6f}")
        max_slack = result.objective_result.max_pos_slack
        max_slack = max_slack.item() if hasattr(max_slack, "item") else max_slack
        print(f"  Max positive slack: {max_slack:.6e}")
        print(f"  Sum positive slack: {result.objective_result.sum_pos_slack.item():.6e}")
        print("=" * 60)

        # Save dual objective curve to CSV
        filename = get_output_filename()
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "dual_objective"])
            for i, obj in enumerate(result.dual_objective_log, 1):
                writer.writerow([i, obj])
        print(f"\nDual objective curve saved to: {filename}")

    return result


if __name__ == "__main__":
    run_benchmark()
