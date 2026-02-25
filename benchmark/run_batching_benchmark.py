"""
Benchmark batching vs no-batching on both CPU and GPU.

Runs the solver for each combination of (device, batching) across
a range of problem sizes and records solve times to a CSV.

Usage:
    python run_batching_benchmark.py [--cache-dir DIR] [--output-dir DIR]
    python run_batching_benchmark.py --sizes 1000000 5000000 10000000
    python run_batching_benchmark.py --cpu-only
    python run_batching_benchmark.py --gpu-only
"""

import argparse
import csv
import os
import time

import config
import torch
from benchmark_utils import generate_benchmark_data

from dualip.objectives.matching import MatchingSolverDualObjectiveFunction
from dualip.optimizers.agd import AcceleratedGradientDescent

DEFAULT_SIZES = [1_000_000, 5_000_000, 10_000_000, 25_000_000]
RESULTS_FILENAME = "batching_results.csv"


def run_single(num_sources: int, device: torch.device, batching: bool, cache_dir: str | None = None):
    """Run a single benchmark configuration and return solve_time."""
    device_label = "GPU" if device.type == "cuda" else "CPU"
    batch_label = "batching" if batching else "no_batching"
    print(f"\n{'='*60}")
    print(f"  {device_label} | {batch_label} | {num_sources:,} sources")
    print(f"{'='*60}")

    # Generate data on the target device
    input_args, _ = generate_benchmark_data(
        num_sources=num_sources,
        num_destinations=config.NUM_DESTINATIONS,
        target_sparsity=config.TARGET_SPARSITY,
        device=device,
        dtype=config.DTYPE,
        seed=config.SEED,
        use_preconditioning=config.USE_PRECONDITIONING,
        cache_dir=cache_dir,
    )

    # Create objective
    objective = MatchingSolverDualObjectiveFunction(
        matching_input_args=input_args,
        gamma=1e-3,
        batching=batching,
    )

    # Create solver
    solver = AcceleratedGradientDescent(
        max_iter=config.MAX_ITER,
        gamma=1e-3,
        initial_step_size=config.INITIAL_STEP_SIZE,
        max_step_size=config.MAX_STEP_SIZE,
        save_primal=False,
    )

    initial_dual = torch.zeros_like(input_args.b_vec)

    # Solve and time
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = solver.maximize(objective, initial_dual)
    if device.type == "cuda":
        torch.cuda.synchronize()
    solve_time = time.perf_counter() - t0

    print(f"  Solve time: {solve_time:.1f}s ({solve_time / config.MAX_ITER * 1000:.2f} ms/iter)")
    print(f"  Dual objective: {result.dual_objective:.2f}")

    return solve_time, float(result.dual_objective)


def run_batching_benchmark(
    sizes: list[int],
    run_cpu: bool = True,
    run_gpu: bool = True,
    cache_dir: str | None = None,
    output_dir: str = ".",
):
    """Run the full batching benchmark grid and write results to CSV."""
    has_gpu = torch.cuda.is_available()
    if run_gpu and not has_gpu:
        print("WARNING: CUDA not available, skipping GPU runs")
        run_gpu = False

    devices = []
    if run_cpu:
        devices.append(torch.device("cpu"))
    if run_gpu:
        devices.append(torch.device("cuda:0"))

    if not devices:
        print("No devices selected. Use --cpu-only or --gpu-only, or run without flags for both.")
        return

    results = []
    for num_sources in sizes:
        for device in devices:
            for batching in [False, True]:
                solve_time, dual_obj = run_single(
                    num_sources=num_sources,
                    device=device,
                    batching=batching,
                    cache_dir=cache_dir,
                )
                results.append({
                    "num_sources": num_sources,
                    "device": device.type,
                    "batching": batching,
                    "solve_time": solve_time,
                    "dual_objective": dual_obj,
                    "max_iter": config.MAX_ITER,
                })

    # Write CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, RESULTS_FILENAME)
    fieldnames = ["num_sources", "device", "batching", "solve_time", "dual_objective", "max_iter"]

    # Append if file exists, otherwise create with header
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {csv_path}")
    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark batching vs no-batching on CPU/GPU")
    parser.add_argument("--cache-dir", type=str, default=None, help="Directory for cached data")
    parser.add_argument(
        "--output-dir", type=str, default=".", help="Directory for output CSV (default: cwd)"
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=DEFAULT_SIZES,
        help=f"Problem sizes (num_sources) to benchmark (default: {DEFAULT_SIZES})",
    )
    parser.add_argument("--cpu-only", action="store_true", help="Only run CPU benchmarks")
    parser.add_argument("--gpu-only", action="store_true", help="Only run GPU benchmarks")
    args = parser.parse_args()

    run_cpu = not args.gpu_only
    run_gpu = not args.cpu_only

    run_batching_benchmark(
        sizes=sorted(args.sizes),
        run_cpu=run_cpu,
        run_gpu=run_gpu,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
    )
