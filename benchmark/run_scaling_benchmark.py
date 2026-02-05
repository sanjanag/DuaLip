"""
Systematic scaling benchmark for matching problem.

Runs benchmarks across different problem sizes (1M to 100M sources) and
GPU counts (1 to 4 GPUs) with 10k iterations to measure performance and
verify correctness across configurations.

Usage:
    python run_scaling_benchmark.py [--cache-dir DIR] [--output results.csv]

Output CSV columns:
    - num_gpus: Number of GPUs used
    - num_sources: Number of source nodes
    - num_destinations: Number of destination nodes
    - target_sparsity: Sparsity of constraint matrix
    - max_iter: Maximum iterations
    - solve_time: Wall-clock time for solver (seconds)
    - dual_objective: Final dual objective value
    - primal_objective: Final primal objective value (if available)
    - reg_penalty: Regularization penalty
    - max_pos_slack: Maximum positive constraint violation
    - sum_pos_slack: Sum of positive constraint violations
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


# Benchmark configuration
SOURCE_SIZES = [
    25_000_000,     # 25M
    50_000_000,     # 50M
    75_000_000,     # 75M
    100_000_000,    # 100M
    125_000_000,    # 125M
    150_000_000,    # 150M
    175_000_000,    # 175M
    200_000_000,    # 200M
    225_000_000,    # 225M
    250_000_000,    # 250M
]

NUM_DESTINATIONS = 10_000
TARGET_SPARSITY = 0.001
SEED = 42
MAX_ITER = 10_000
GPU_COUNTS = [1, 2, 3, 4]

# Fixed solver parameters
GAMMA = 1e-3
INITIAL_STEP_SIZE = 1e-3
MAX_STEP_SIZE = 1e-1


def run_single_gpu_benchmark(num_sources: int, cache_dir: str | None = None) -> dict | None:
    """
    Run single-GPU benchmark for given problem size.
    """
    print(f"\n{'='*80}")
    print(f"Running SINGLE GPU benchmark: {num_sources:,} sources")
    print(f"{'='*80}")

    # Use a temporary JSON file for metrics
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json_file = f.name

    try:
        cmd = [
            sys.executable,
            "run_matching_benchmark.py",
            "--num-sources", str(num_sources),
            "--max-iter", str(MAX_ITER),
            "--json-output", json_file,
        ]
        if cache_dir:
            cmd.extend(["--cache-dir", cache_dir])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
        )

        if result.returncode != 0:
            print(f"ERROR: Benchmark failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return None

        # Read metrics from JSON file
        with open(json_file, 'r') as f:
            metrics = json.load(f)

        return metrics

    except Exception as e:
        print(f"ERROR running benchmark: {e}")
        return None
    finally:
        # Clean up temp file
        Path(json_file).unlink(missing_ok=True)


def run_distributed_benchmark(num_sources: int, num_gpus: int, cache_dir: str | None = None) -> dict | None:
    """
    Run distributed benchmark for given problem size and GPU count via torchrun.
    """
    print(f"\n{'='*80}")
    print(f"Running {num_gpus} GPU benchmark: {num_sources:,} sources")
    print(f"{'='*80}")

    # Use a temporary JSON file for metrics
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json_file = f.name

    try:
        cmd = [
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            "run_matching_benchmark_dist.py",
            "--num-sources", str(num_sources),
            "--max-iter", str(MAX_ITER),
            "--json-output", json_file,
        ]
        if cache_dir:
            cmd.extend(["--cache-dir", cache_dir])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
        )

        if result.returncode != 0:
            print(f"ERROR: Benchmark failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return None

        # Read metrics from JSON file
        with open(json_file, 'r') as f:
            metrics = json.load(f)

        return metrics

    except Exception as e:
        print(f"ERROR running benchmark: {e}")
        return None
    finally:
        # Clean up temp file
        Path(json_file).unlink(missing_ok=True)


def run_all_benchmarks(cache_dir: str | None = None, output_file: str = "scaling_results.csv"):
    """
    Run all benchmark combinations and save results to CSV.
    """
    results = []

    print(f"\n{'#'*80}")
    print(f"# SCALING BENCHMARK")
    print(f"# Source sizes: {[f'{s:,}' for s in SOURCE_SIZES]}")
    print(f"# GPU counts: {GPU_COUNTS}")
    print(f"# Max iterations: {MAX_ITER:,}")
    print(f"# Output: {output_file}")
    print(f"{'#'*80}\n")

    for num_sources in SOURCE_SIZES:
        for num_gpus in GPU_COUNTS:
            try:
                if num_gpus == 1:
                    result = run_single_gpu_benchmark(num_sources, cache_dir)
                else:
                    result = run_distributed_benchmark(num_sources, num_gpus, cache_dir)

                if result:
                    results.append(result)

                    # Print summary
                    print(f"\n✓ Completed: {num_sources:,} sources, {num_gpus} GPU(s)")
                    print(f"  Solve time: {result['solve_time']:.3f}s")
                    print(f"  Dual obj: {result['dual_objective']:.6f}")
                    print(f"  Max slack: {result['max_pos_slack']:.6e}")

                    # Save intermediate results
                    save_results(results, output_file)
                else:
                    print(f"\n✗ FAILED: {num_sources:,} sources, {num_gpus} GPU(s)")

            except Exception as e:
                print(f"\n✗ ERROR: {num_sources:,} sources, {num_gpus} GPU(s)")
                print(f"  Exception: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n{'#'*80}")
    print(f"# BENCHMARK COMPLETE")
    print(f"# Results saved to: {output_file}")
    print(f"{'#'*80}\n")

    return results


def save_results(results: list[dict], output_file: str):
    """Save results to CSV file."""
    if not results:
        return

    fieldnames = [
        "num_gpus",
        "num_sources",
        "num_destinations",
        "target_sparsity",
        "max_iter",
        "solve_time",
        "dual_objective",
        "primal_objective",
        "reg_penalty",
        "max_pos_slack",
        "sum_pos_slack",
    ]

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n💾 Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run scaling benchmarks across problem sizes and GPU counts")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for cached data (default: ./benchmark_data)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="scaling_results.csv",
        help="Output CSV file for results (default: scaling_results.csv)",
    )
    args = parser.parse_args()

    run_all_benchmarks(cache_dir=args.cache_dir, output_file=args.output)
