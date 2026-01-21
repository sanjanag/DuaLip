import argparse
import csv
import os
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    BASE_NUM_SOURCES = args.base_num_sources
    num_compute_devices = args.num_compute_devices
    host_device = args.host_device
    output_file = "memory_testing_results.csv"

    # Create file with header
    file_exists = os.path.exists(output_file)

    for i in range(1, 11):
        print(f"Iteration {i}:")
        num_sources = BASE_NUM_SOURCES * i
        if num_compute_devices == 1:
            result_metrics = run_benchmark_single(num_sources=num_sources)
        else:
            result_metrics = run_benchmark_dist(
                num_sources=num_sources, num_compute_devices=num_compute_devices, host_device=host_device
            )

        # Add iteration and num_sources to the metrics
        result_metrics["num_sources"] = num_sources

        # Save to CSV immediately after execution
        with open(output_file, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "num_sources",
                    "solve_time",
                    "dual_objective",
                    "reg_penalty",
                    "max_pos_slack",
                    "sum_pos_slack",
                ],
            )
            if i == 1:
                writer.writeheader()
            writer.writerow(result_metrics)

        print(f"Results saved to: {output_file}")
