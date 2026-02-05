"""
Distributed multi-GPU benchmark for matching problem.
Edit the CONFIG section below to change parameters.

Usage:
    torchrun --nproc_per_node=<num_gpus> benchmark/run_matching_benchmark_dist.py
"""

import time

import torch

import config
from benchmark_utils import generate_benchmark_data, get_output_filename, print_config, print_results, save_dual_curve

from dualip.objectives.matching import MatchingInputArgs, MatchingSolverDualObjectiveFunctionDistributed
from dualip.optimizers.agd import AcceleratedGradientDescent
from dualip.utils.dist_utils import global_to_local_projection_map, split_tensors_to_devices

# =============================================================================
# CONFIG - Edit these values
# =============================================================================

# Gamma setting
GAMMA = 1e-3

# =============================================================================
# END CONFIG
# =============================================================================


def run_benchmark():
    rng = None  # np.random.default_rng(config.SEED)

    # Generate data BEFORE initializing distributed
    # (Each process will generate identical data with same seed)
    #
    # NOTE: This approach is feasible for small-to-medium datasets where full data
    # generation on each process is acceptable. For large-scale datasets, this results
    # in redundant computation (N processes doing the same work). In production scenarios
    # with very large datasets, consider:
    # 1. Pre-generating and saving data to disk, then loading shards per process
    # 2. Having only rank 0 generate data and broadcast to other ranks
    # 3. Generating sharded data directly per process (different indices per rank)
    print("\n[1/4] Generating data...")
    input_args, data_time = generate_benchmark_data(
        num_sources=config.NUM_SOURCES,
        num_destinations=config.NUM_DESTINATIONS,
        target_sparsity=config.TARGET_SPARSITY,
        device="cpu",  # Use CPU for data generation
        use_preconditioning=config.USE_PRECONDITIONING,
        rng=rng,
    )

    # Initialize distributed
    print("[2/4] Initializing distributed...")
    torch.distributed.init_process_group(backend="nccl")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    # Each rank sets its device
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    # Print configuration (rank 0 only)
    if rank == 0:
        print_config(
            num_sources=config.NUM_SOURCES,
            num_destinations=config.NUM_DESTINATIONS,
            target_sparsity=config.TARGET_SPARSITY,
            seed=config.SEED,
            gamma=GAMMA,
            max_iter=config.MAX_ITER,
            use_preconditioning=config.USE_PRECONDITIONING,
            world_size=world_size,
        )
        print("[3/4] Splitting data on CPU...")

    # Split data on CPU ONLY (don't move to GPUs)
    # Pass CPU devices to split data while keeping it on CPU
    cpu_devices = ["cpu"] * world_size
    A_splits, c_splits, split_index_map = split_tensors_to_devices(input_args.A, input_args.c, cpu_devices)

    # Each rank takes ONLY its own partition and moves to its own GPU
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
        batching=config.BATCHING,
    )
    obj_time = time.perf_counter() - t0
    if rank == 0:
        print(f"      {obj_time:.3f}s")

    # Create solver
    if rank == 0:
        print("Running solver...")
    solver = AcceleratedGradientDescent(
        max_iter=config.MAX_ITER,
        gamma=GAMMA,
        initial_step_size=config.INITIAL_STEP_SIZE,
        max_step_size=config.MAX_STEP_SIZE,
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
        print_results(result, solve_time, config.MAX_ITER)

        # Save dual objective curve to CSV
        filename = get_output_filename(
            num_sources=config.NUM_SOURCES,
            num_destinations=config.NUM_DESTINATIONS,
            target_sparsity=config.TARGET_SPARSITY,
            gamma=GAMMA,
            max_iter=config.MAX_ITER,
            use_preconditioning=config.USE_PRECONDITIONING,
        )
        save_dual_curve(result, filename)

    return result


if __name__ == "__main__":
    run_benchmark()
