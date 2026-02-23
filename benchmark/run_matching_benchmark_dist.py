"""
Distributed multi-GPU benchmark for matching problem.
Edit the CONFIG section below to change parameters.

Usage:
    torchrun --nproc_per_node=<num_gpus> benchmark/run_matching_benchmark_dist.py [--cache-dir DIR]
"""

import argparse
import json
import time

import config
import torch
from benchmark_utils import generate_benchmark_data, get_output_filename, print_config, print_results, save_dual_curve
from generate_synthetic_data import load_local_partition_from_cache

from dualip.objectives.matching import MatchingInputArgs, MatchingSolverDualObjectiveFunctionDistributed
from dualip.optimizers.agd import AcceleratedGradientDescent

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

    # Per-rank loading does not support preconditioning (it modifies A and
    # b_vec in-place after generation, which is not reflected in the cache).
    if config.USE_PRECONDITIONING:
        raise NotImplementedError(
            "Per-rank loading from disk cache does not support preconditioning. "
            "Set config.USE_PRECONDITIONING = False or use the single-GPU benchmark."
        )

    # Rank 0 ensures the cache exists on disk (generates if missing)
    if rank == 0:
        print("[2/4] Ensuring cache exists (rank 0)...")
        input_args, data_time = generate_benchmark_data(
            num_sources=config.NUM_SOURCES,
            num_destinations=config.NUM_DESTINATIONS,
            target_sparsity=config.TARGET_SPARSITY,
            device="cpu",
            dtype=config.DTYPE,
            seed=config.SEED,
            use_preconditioning=False,
            cache_dir=cache_dir,
        )
        del input_args  # free CPU memory before other ranks start loading

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

    # Wait for cache to be fully written before other ranks read it
    torch.distributed.barrier()

    # Each rank loads only its own partition from the memmap cache
    if rank == 0:
        print("[3/4] Loading local partitions from cache...")
    local_input_args_cpu = load_local_partition_from_cache(
        num_sources=config.NUM_SOURCES,
        num_destinations=config.NUM_DESTINATIONS,
        target_sparsity=config.TARGET_SPARSITY,
        dtype=config.DTYPE,
        seed=config.SEED,
        rank=rank,
        world_size=world_size,
        cache_dir=cache_dir,
    )
    b_vec_cpu = local_input_args_cpu.b_vec

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
    metrics = None
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

        # Create metrics dictionary
        metrics = {
            "num_gpus": world_size,
            "num_sources": config.NUM_SOURCES,
            "num_destinations": config.NUM_DESTINATIONS,
            "target_sparsity": config.TARGET_SPARSITY,
            "max_iter": config.MAX_ITER,
            "solve_time": solve_time,
            "dual_objective": float(result.dual_objective),
            "primal_objective": (
                float(result.objective_result.primal_objective.item())
                if result.objective_result.primal_objective is not None
                else None
            ),
            "reg_penalty": float(result.objective_result.reg_penalty.item()),
            "max_pos_slack": float(
                result.objective_result.max_pos_slack.item()
                if hasattr(result.objective_result.max_pos_slack, "item")
                else result.objective_result.max_pos_slack
            ),
            "sum_pos_slack": float(result.objective_result.sum_pos_slack.item()),
        }

    # Clean up distributed process group
    torch.distributed.destroy_process_group()

    return metrics


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
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Save metrics to JSON file (only rank 0)",
    )
    args = parser.parse_args()

    # Override config if specified
    if args.num_sources is not None:
        config.NUM_SOURCES = args.num_sources
    if args.max_iter is not None:
        config.MAX_ITER = args.max_iter

    metrics = run_benchmark(cache_dir=args.cache_dir)

    # Save metrics to JSON if requested (only rank 0 has metrics)
    if args.json_output and metrics is not None:
        with open(args.json_output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n📊 Metrics saved to {args.json_output}")
