#!/usr/bin/env python3
"""Plot average iteration time vs number of sources from memory testing results."""

import csv
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
data = {"1_device": {"num_sources": [], "avg_iter_time": [], "solve_time": []}, "2_devices": {"num_sources": [], "avg_iter_time": [], "solve_time": []}}

with open("memory_testing_results.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        num_sources = int(row["num_sources"])
        num_devices = int(row["num_compute_devices"])
        avg_iter_time = float(row["avg_iter_time"]) * 1000  # Convert to milliseconds
        solve_time = float(row["solve_time"])

        key = f"{num_devices}_device{'s' if num_devices > 1 else ''}"
        data[key]["num_sources"].append(num_sources / 1_000_000)  # Convert to millions
        data[key]["avg_iter_time"].append(avg_iter_time)
        data[key]["solve_time"].append(solve_time)

# Create the plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Average Iteration Time
ax1.plot(
    data["1_device"]["num_sources"],
    data["1_device"]["avg_iter_time"],
    marker="o",
    linewidth=2,
    markersize=8,
    label="1 Device",
)
ax1.plot(
    data["2_devices"]["num_sources"],
    data["2_devices"]["avg_iter_time"],
    marker="s",
    linewidth=2,
    markersize=8,
    label="2 Devices",
)
ax1.set_xlabel("Number of Sources (Millions)", fontsize=12)
ax1.set_ylabel("Average Iteration Time (ms)", fontsize=12)
ax1.set_title("Average Iteration Time vs Problem Size", fontsize=14, fontweight="bold")
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Plot 2: Total Solve Time
ax2.plot(
    data["1_device"]["num_sources"],
    data["1_device"]["solve_time"],
    marker="o",
    linewidth=2,
    markersize=8,
    label="1 Device",
)
ax2.plot(
    data["2_devices"]["num_sources"],
    data["2_devices"]["solve_time"],
    marker="s",
    linewidth=2,
    markersize=8,
    label="2 Devices",
)
ax2.set_xlabel("Number of Sources (Millions)", fontsize=12)
ax2.set_ylabel("Total Solve Time (s)", fontsize=12)
ax2.set_title("Total Solve Time vs Problem Size", fontsize=14, fontweight="bold")
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# Save the plot
plt.savefig("memory_results_plot.png", dpi=150, bbox_inches="tight")
print("Plot saved to: memory_results_plot.png")

# Also display speedup information
print("\nAvg Iteration Time Speedup (1 device / 2 devices):")
for i in range(len(data["1_device"]["num_sources"])):
    sources = data["1_device"]["num_sources"][i]
    time_1_dev = data["1_device"]["avg_iter_time"][i]
    time_2_dev = data["2_devices"]["avg_iter_time"][i]
    speedup = time_1_dev / time_2_dev
    print(f"  {sources:.0f}M sources: {speedup:.2f}x speedup ({time_1_dev:.2f}ms → {time_2_dev:.2f}ms)")

print("\nTotal Solve Time Speedup (1 device / 2 devices):")
for i in range(len(data["1_device"]["num_sources"])):
    sources = data["1_device"]["num_sources"][i]
    time_1_dev = data["1_device"]["solve_time"][i]
    time_2_dev = data["2_devices"]["solve_time"][i]
    speedup = time_1_dev / time_2_dev
    print(f"  {sources:.0f}M sources: {speedup:.2f}x speedup ({time_1_dev:.2f}s → {time_2_dev:.2f}s)")
