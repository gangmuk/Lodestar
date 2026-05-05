#!/usr/bin/env python3
"""Paper figure for heterogeneous-GPU routing results.

Usage:
  python plot_heterogeneous_routing.py <experiment_dir>
         [--output PATH]
         [--ylim-ttft MIN MAX]
         [--ylim-count MIN MAX]

Reads <experiment_dir>/filtered-aibrix-gateway-plugins-processed.log.csv and
writes a PDF. By default the PDF is saved as
  <experiment_dir>/figure_c_avg_ttft_per_pod.pdf
Use --output to override the destination path. Use --ylim-ttft / --ylim-count
to pin the left (TTFT) and right (# requests) y-axis ranges.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def resolve_paths(argv):
    parser = argparse.ArgumentParser(
        prog="plot_heterogeneous_routing.py",
        description="Plot mean TTFT per pod for a heterogeneous-GPU routing experiment.",
    )
    parser.add_argument(
        "experiment_dir",
        help="directory containing filtered-aibrix-gateway-plugins-processed.log.csv",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="output PDF path (default: <experiment_dir>/figure_c_avg_ttft_per_pod.pdf)",
    )
    parser.add_argument(
        "--ylim-ttft",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=None,
        help="y-axis range (ms) for the Mean TTFT bars (default: auto)",
    )
    parser.add_argument(
        "--ylim-count",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=None,
        help="y-axis range for the # requests routed line (default: auto)",
    )
    args = parser.parse_args(argv[1:])

    target_dir = os.path.abspath(args.experiment_dir)
    if not os.path.isdir(target_dir):
        sys.stderr.write(f"error: not a directory: {target_dir}\n")
        sys.exit(2)
    csv_path = os.path.join(target_dir, "filtered-aibrix-gateway-plugins-processed.log.csv")
    if not os.path.isfile(csv_path):
        sys.stderr.write(f"error: CSV not found: {csv_path}\n")
        sys.exit(2)

    if args.output is None:
        out_path = os.path.join(target_dir, "figure_c_avg_ttft_per_pod.pdf")
    else:
        out_path = os.path.abspath(args.output)
        out_dir = os.path.dirname(out_path) or "."
        if not os.path.isdir(out_dir):
            sys.stderr.write(f"error: output directory does not exist: {out_dir}\n")
            sys.exit(2)

    ylim_ttft = tuple(args.ylim_ttft) if args.ylim_ttft else None
    ylim_count = tuple(args.ylim_count) if args.ylim_count else None
    return target_dir, csv_path, out_path, ylim_ttft, ylim_count

A30_PODS = [f"pod_{i:04d}" for i in range(0, 7)]
V100_PODS = [f"pod_{i:04d}" for i in range(7, 15)]
ALL_PODS = A30_PODS + V100_PODS
POD_GPU = {p: "NVIDIA-A30" for p in A30_PODS}
POD_GPU.update({p: "Tesla-V100" for p in V100_PODS})

A30_COLOR = "#1b9e77"   # ColorBrewer Dark2 teal
V100_COLOR = "#d95f02"  # ColorBrewer Dark2 orange

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 6,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.grid": False,
})


def load_data(csv_path):
    df = pd.read_csv(
        csv_path,
        usecols=["selected_pod", "request_start_time", "ttft", "e2e_latency"],
    )
    df = df.dropna(subset=["selected_pod", "request_start_time", "ttft"])
    df["gpu"] = df["selected_pod"].map(POD_GPU)
    df = df.dropna(subset=["gpu"]).copy()
    t0 = df["request_start_time"].min()
    # request_start_time is in microseconds
    df["t_sec"] = (df["request_start_time"] - t0) / 1e6
    return df


def fig_c(df, out_path, ylim_ttft=None, ylim_count=None):
    stats = (
        df.groupby("selected_pod")["ttft"]
        .agg(["mean", "count"])
        .reindex(ALL_PODS)
        .sort_values("mean", ascending=False)
    )
    sorted_pods = list(stats.index)
    colors = [A30_COLOR if POD_GPU[p] == "NVIDIA-A30" else V100_COLOR
              for p in sorted_pods]

    fig, ax = plt.subplots(figsize=(4.0, 2.5))
    xs = np.arange(len(sorted_pods))
    bars = ax.bar(xs, stats["mean"].values, color=colors,
                  edgecolor="k", lw=0.5, zorder=2)

    ymax = stats["mean"].max()
    short = {"NVIDIA-A30": "A30", "Tesla-V100": "V100"}
    ax.set_xticks(xs)
    ax.set_xticklabels([short[POD_GPU[p]] for p in sorted_pods],
                       rotation=45, ha="right")
    ax.set_xlabel("Instances", fontsize=12)
    ax.set_ylabel("Mean TTFT (ms)", fontsize=12)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    if ylim_ttft is not None:
        ax.set_ylim(*ylim_ttft)
    else:
        ax.set_ylim(0, ymax * 1.1)

    ax2 = ax.twinx()
    ax2.plot(xs, stats["count"].values,
             color="black", marker="o", ls="--",
             lw=1.2, markersize=5, zorder=3)
    ax2.set_ylabel("# req routed", fontsize=12)
    ax2.tick_params(axis="y", labelsize=11)
    cnt_max = stats["count"].max()
    if ylim_count is not None:
        ax2.set_ylim(*ylim_count)
    else:
        ax2.set_ylim(0, cnt_max * 1.15)

    legend_handles = [
        Patch(color=A30_COLOR, label="NVIDIA-A30"),
        Patch(color=V100_COLOR, label="Tesla-V100"),
        Line2D([0], [0], color="black", marker="o", ls="--",
               label="# requests"),
    ]
    ax.legend(handles=legend_handles,
              loc="lower center",
              bbox_to_anchor=(0.5, 1.02),
              ncol=3, frameon=True, framealpha=0.95,
              columnspacing=1.0, handletextpad=0.4, borderpad=0.3,
              fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main(argv):
    target_dir, csv_path, out_c, ylim_ttft, ylim_count = resolve_paths(argv)
    df = load_data(csv_path)
    fig_c(df, out_c, ylim_ttft=ylim_ttft, ylim_count=ylim_count)


if __name__ == "__main__":
    main(sys.argv)
