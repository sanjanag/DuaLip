#!/usr/bin/env python3
"""Plot convergence ablations: preconditioning and gamma decay.

Usage:
    python plot_convergence.py --dir convergence_results -o convergence_plot.pdf
"""

import argparse
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -- Publication style --------------------------------------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 1.4,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.pad": 4,
    "ytick.major.pad": 4,
})

# -- Estimated optimal dual objectives ---------------------------------------
L_HAT_PRECON = -1_264_000
L_HAT_DECAY = -1_100_000

# Each ablation: (csv_a, csv_b, label_a, label_b, l_hat, title)
ABLATIONS = [
    ("precon_dual_objective.csv", "no_precon_dual_objective.csv",
     "With preconditioning", "Without preconditioning", L_HAT_PRECON,
     "Preconditioning Ablation"),
    ("gamma_decay_dual_objective.csv", "no_gamma_decay_dual_objective.csv",
     r"With $\gamma$ decay", r"Without $\gamma$ decay", L_HAT_DECAY,
     r"$\gamma$ Decay Ablation"),
]

COLORS = ["#1f77b4", "#d62728"]  # blue, red — high contrast for print/grayscale


def plot_ablation(path_a, path_b, label_a, label_b, l_hat, title, output=None):
    """Plot a single ablation and save or show it."""
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    all_log_gap = []
    for path, label, color, ls in [
        (path_a, label_a, COLORS[0], "-"),
        (path_b, label_b, COLORS[1], "--"),
    ]:
        df = pd.read_csv(path)
        gap = np.abs(df["dual_objective"].values - l_hat)
        gap = np.maximum(gap, 1e-10)
        log_gap = np.log(gap)
        all_log_gap.append(log_gap)
        ax.plot(df["iteration"].values, log_gap, color=color, linestyle=ls,
                label=label)

    # Fit y-axis tightly to data with some padding
    all_vals = np.concatenate(all_log_gap)
    y_min, y_max = np.min(all_vals), np.max(all_vals)
    margin = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - margin, y_max + margin)

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\log\,|L - \hat{L}|$")
    ax.set_title(title)
    ax.legend(frameon=True, fancybox=False, edgecolor="0.7")
    ax.grid(True, alpha=0.2, linewidth=0.4)
    plt.tight_layout()

    if output:
        fig.savefig(output, dpi=300, bbox_inches="tight")
        print(f"Saved to {output}")
    else:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=".", help="Directory with CSV files")
    parser.add_argument("--output", "-o",
                        help="Output directory for plots (omit to show)")
    args = parser.parse_args()

    for csv_a, csv_b, label_a, label_b, l_hat, title in ABLATIONS:
        path_a = os.path.join(args.dir, csv_a)
        path_b = os.path.join(args.dir, csv_b)
        if not (os.path.exists(path_a) and os.path.exists(path_b)):
            continue

        if args.output:
            # Derive filename from first CSV name (e.g. precon_dual_objective -> precon)
            stem = csv_a.replace("_dual_objective.csv", "")
            ext = os.path.splitext(args.output)[1] if os.path.splitext(args.output)[1] else ".png"
            out_path = os.path.join(
                os.path.dirname(args.output) or ".",
                f"{stem}_plot{ext}",
            )
        else:
            out_path = None

        plot_ablation(path_a, path_b, label_a, label_b, l_hat, title, out_path)


if __name__ == "__main__":
    main()
