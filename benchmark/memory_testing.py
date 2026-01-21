import csv
import os
from run_matching_dist_benchmark import run_benchmark

if __name__ == "__main__":
    BASE_NUM_SOURCES = 25_000_000
    output_file = "memory_testing_results.csv"

    # Create file with header
    file_exists = os.path.exists(output_file)

    for i in range(1, 11):
        print(f"Iteration {i}:")
        num_sources = BASE_NUM_SOURCES * i
        result_metrics = run_benchmark(num_sources=num_sources, num_compute_devices=2, host_device="cuda:0")

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
