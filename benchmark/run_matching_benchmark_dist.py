"""
Distributed multi-GPU benchmark for matching problem.
Edit the CONFIG section below to change parameters.

Usage:
    torchrun --nproc_per_node=<num_gpus> benchmark/run_matching_benchmark_dist.py [--cache-dir DIR]
"""

import argparse
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


def run_benchmark(cache_dir: str | None = None):
    # Initialize distributed FIRST to get rank
    torch.distributed.init_process_group(backend="nccl")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    # Each rank sets its device
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Only rank 0 generates and splits data to avoid redundant computation
    if rank == 0:
        print("[2/4] Generating data on rank 0...")
        input_args, data_time = generate_benchmark_data(
            num_sources=config.NUM_SOURCES,
            num_destinations=config.NUM_DESTINATIONS,
            target_sparsity=config.TARGET_SPARSITY,
            device="cpu",  # Use CPU for data generation
            dtype=config.DTYPE,
            seed=config.SEED,
            use_preconditioning=config.USE_PRECONDITIONING,
            cache_dir=cache_dir,
        )

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

        print("[3/4] Splitting data on rank 0...")
        # Split data on CPU ONLY (don't move to GPUs)
        cpu_devices = ["cpu"] * world_size
        A_splits, c_splits, split_index_map = split_tensors_to_devices(input_args.A, input_args.c, cpu_devices)

        # Create local input args for each rank (b_vec=None for local partitions)
        local_input_args_list = []
        for r in range(world_size):
            pm_local = global_to_local_projection_map(input_args.projection_map, split_index_map[r])
            local_input_args_list.append(
                MatchingInputArgs(
                    A=A_splits[r],
                    c=c_splits[r],
                    projection_map=pm_local,
                    b_vec=None,
                    equality_mask=input_args.equality_mask,
                )
            )
        b_vec_cpu = input_args.b_vec

        print("      Scattering partitions to all ranks...")
    else:
        local_input_args_list = None
        b_vec_cpu = None

    # Scatter: each rank receives only its own partition
    local_input_args_recv = [None]
    torch.distributed.scatter_object_list(local_input_args_recv, local_input_args_list, src=0)
    local_input_args_cpu = local_input_args_recv[0]

    # Broadcast b_vec to all ranks (small, shared across all ranks)
    b_vec_list = [b_vec_cpu]
    torch.distributed.broadcast_object_list(b_vec_list, src=0)
    b_vec_cpu = b_vec_list[0]

    # Move local partition to GPU
    local_input_args = MatchingInputArgs(
        A=local_input_args_cpu.A.to(device),
        c=local_input_args_cpu.c.to(device),
        projection_map=local_input_args_cpu.projection_map,
        b_vec=None,
        equality_mask=local_input_args_cpu.equality_mask,
    )

    # Create distributed objective with local data
    # host_device is cuda:0 for aggregation (all ranks send results there)

    objective = MatchingSolverDualObjectiveFunctionDistributed(
        local_matching_input_args=local_input_args,
        b_vec=b_vec_cpu,
        gamma=GAMMA,
        host_device="cuda:0",
        batching=config.BATCHING,
    )

    # Create solver
    solver = AcceleratedGradientDescent(
        max_iter=config.MAX_ITER,
        gamma=GAMMA,
        initial_step_size=config.INITIAL_STEP_SIZE,
        max_step_size=config.MAX_STEP_SIZE,
        save_primal=False,  # Not yet supported in distributed mode
    )

    # Initialize dual variables on each rank's device (for broadcast to work)
    initial_dual = torch.zeros_like(b_vec_cpu).to(device)

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

    # Clean up distributed process group
    torch.distributed.destroy_process_group()

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed multi-GPU benchmark for matching problem")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for cached data (default: ./benchmark_data)",
    )
    parser.add_argument(
        "--num-sources",
        type=int,
        default=None,
        help=f"Number of sources (default: {config.NUM_SOURCES})",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help=f"Maximum iterations (default: {config.MAX_ITER})",
    )
    args = parser.parse_args()

    # Override config if specified
    if args.num_sources is not None:
        config.NUM_SOURCES = args.num_sources
    if args.max_iter is not None:
        config.MAX_ITER = args.max_iter

    run_benchmark(cache_dir=args.cache_dir)
