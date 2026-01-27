import pandas as pd
import matplotlib.pyplot as plt

# Read data
df = pd.read_csv("memory_testing_results.csv")

# Plot 1: Solve time vs problem size
fig1 = plt.figure(figsize=(10, 6))
devices = sorted(df["num_compute_devices"].unique())
markers = ["o", "s", "^", "D", "P", "X"]
colors = plt.cm.tab10(range(len(devices)))  # Get distinct colors
for i, device in enumerate(devices):
    data = df[df["num_compute_devices"] == device]
    plt.plot(
        data["num_sources"] / 1e6,
        data["solve_time"],
        marker=markers[i % len(markers)],
        linestyle="--",
        label=f"{device} devices",
        markersize=10,
        linewidth=1.5,
        alpha=0.5,  # Make line more transparent
        color=colors[i],
        markerfacecolor=colors[i],
        # markeredgecolor="black",
        markeredgewidth=1.5,
        zorder=10 - i,
    )
plt.xlabel("Number of Sources (millions)")
plt.ylabel("Solve Time (s)")
plt.title("Solve Time")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("solve_time.png", dpi=150)
print("Plot saved to solve_time.png")
plt.close()

# Plot 2: Dual objective (box plot by devices)
fig2 = plt.figure(figsize=(10, 6))
box_data = []
box_labels = []
for sources in sorted(df["num_sources"].unique()):
    data = df[df["num_sources"] == sources]["dual_objective"].values
    box_data.append(data)
    box_labels.append(f"{sources/1e6:.0f}M")

plt.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
plt.xlabel("Number of Sources")
plt.ylabel("Dual Objective")
plt.title("Dual Objective Distribution\n(Each box shows variation across device counts)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("dual_objective.png", dpi=150)
print("Plot saved to dual_objective.png")
plt.close()

# Plot 3: Max positive slack (box plot by devices)
fig3 = plt.figure(figsize=(10, 6))
box_data_max = []
box_labels_max = []
for sources in sorted(df["num_sources"].unique()):
    data = df[df["num_sources"] == sources]["max_pos_slack"].values
    box_data_max.append(data)
    box_labels_max.append(f"{sources/1e6:.0f}M")

plt.boxplot(box_data_max, tick_labels=box_labels_max, patch_artist=True)
plt.xlabel("Number of Sources")
plt.ylabel("Max Positive Slack")
plt.title("Max Positive Slack Distribution\n(Each box shows variation across device counts)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("max_pos_slack.png", dpi=150)
print("Plot saved to max_pos_slack.png")
plt.close()

# Plot 4: Sum positive slack (box plot by devices)
fig4 = plt.figure(figsize=(10, 6))
box_data_sum = []
box_labels_sum = []
for sources in sorted(df["num_sources"].unique()):
    data = df[df["num_sources"] == sources]["sum_pos_slack"].values
    box_data_sum.append(data)
    box_labels_sum.append(f"{sources/1e6:.0f}M")

plt.boxplot(box_data_sum, tick_labels=box_labels_sum, patch_artist=True)
plt.xlabel("Number of Sources")
plt.ylabel("Sum Positive Slack")
plt.title("Sum Positive Slack Distribution\n(Each box shows variation across device counts)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("sum_pos_slack.png", dpi=150)
print("Plot saved to sum_pos_slack.png")
plt.close()

print("\nAll plots saved successfully!")
