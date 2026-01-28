import argparse
import csv
import os
import time
import traceback

import torch
from run_matching_benchmark import run_benchmark as run_benchmark_single
from run_matching_dist_benchmark import run_benchmark as run_benchmark_dist


def parse_args():
    parser = argparse.ArgumentParser(
        description="Memory testing for matching problem.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base_num_sources", type=int, default=25_000_000, help="Base number of sources")
    parser.add_argument("--num_compute_devices", type=int, default=1, help="Number of compute devices")
    parser.add_argument("--host_device", type=str, default="cuda:0", help="Host device")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float64"],
        help="Floating point precision",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="./benchmark_data",
        help="Base directory for cache and results (subdirs created automatically based on params)",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=1000,
        help="Maximum number of iterations for the solver",
    )
    return parser.parse_args()


if __name__ == "__main__":
    start_time = time.time()
    args = parse_args()
    BASE_NUM_SOURCES = args.base_num_sources
    host_device = args.host_device

    # Convert dtype string to torch.dtype
    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
    }
    dtype = dtype_map[args.dtype]

    # Create base directory
    os.makedirs(args.base_dir, exist_ok=True)

    # Output file for aggregate results (in base_dir root)
    dtype_str = args.dtype
    output_file = os.path.join(args.base_dir, f"memory_testing_results_{dtype_str}.csv")

    # Load existing results to skip already completed experiments
    completed_experiments = set()
    if os.path.exists(output_file):
        print(f"Loading existing results from {output_file}...")
        with open(output_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                num_sources = int(row["num_sources"])
                num_compute_devices = int(row["num_compute_devices"])
                completed_experiments.add((num_sources, num_compute_devices))
        print(f"Found {len(completed_experiments)} completed experiments")

    print("\n" + "=" * 70)
    print(f"MEMORY TESTING: {BASE_NUM_SOURCES:,} base sources, {dtype_str}, {args.max_iter} max_iter")
    print("5 iters x 4 devices")
    print(f"Output: {output_file}")
    print("=" * 70 + "\n")

    # Track stats
    total_runs = 0
    success_count = 0
    skipped_count = 0

    for i in range(1, 6):
        num_sources = BASE_NUM_SOURCES * i
        print(f"\n[Iteration {i}/5] Testing {num_sources:,} sources")
        for num_compute_devices in [1, 2, 3, 4]:
            # Check if this experiment has already been completed
            if (num_sources, num_compute_devices) in completed_experiments:
                skipped_count += 1
                print(f"  [SKIP] dev={num_compute_devices}, sources={num_sources:,} (already completed)")
                continue

            total_runs += 1
            print(f"  [{total_runs}/20] dev={num_compute_devices}...", end=" ", flush=True)
            try:
                if num_compute_devices == 1:
                    result_metrics = run_benchmark_single(
                        num_sources=num_sources,
                        dtype=dtype,
                        base_dir=args.base_dir,
                        max_iter=args.max_iter,
                    )
                else:
                    result_metrics = run_benchmark_dist(
                        num_sources=num_sources,
                        num_compute_devices=num_compute_devices,
                        host_device=host_device,
                        dtype=dtype,
                        base_dir=args.base_dir,
                        max_iter=args.max_iter,
                    )

                # Add iteration and num_sources to the metrics
                result_metrics["num_sources"] = num_sources
                result_metrics["num_compute_devices"] = num_compute_devices
                result_metrics["dtype"] = dtype_str

                # Save to CSV immediately after execution
                file_exists = os.path.exists(output_file) and os.path.getsize(output_file) > 0
                with open(output_file, "a", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "num_sources",
                            "num_compute_devices",
                            "dtype",
                            "solve_time",
                            "avg_iter_time",
                            "dual_objective",
                            "reg_penalty",
                            "max_pos_slack",
                            "sum_pos_slack",
                            "dual_objective_file",
                        ],
                    )
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(result_metrics)

                success_count += 1
                print(
                    f"✓ {result_metrics['solve_time']:.1f}s "
                    f"({result_metrics['avg_iter_time']*1000:.1f}ms/iter, "
                    f"obj={result_metrics['dual_objective']:.4f})"
                )
            except Exception as e:
                print(f"✗ FAILED: {str(e)}")
                traceback.print_exc()

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"COMPLETE: {success_count}/{total_runs} successful, {skipped_count} skipped ({elapsed/60:.1f} min)")
    print(f"Results: {output_file}")
    print("=" * 70 + "\n")
