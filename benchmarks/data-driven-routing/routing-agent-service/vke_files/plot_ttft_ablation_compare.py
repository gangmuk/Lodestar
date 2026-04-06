#!/usr/bin/env python3
"""
Compare two experiment runs using per-request TTFT.

Outputs:
  1) One bar chart comparing Avg TTFT and P99 TTFT for exp1 vs exp2.
  2) One time-series graph of sliding-window mean TTFT (default window=500).

Input format:
  Uses per-request CSV: `filtered-aibrix-gateway-plugins-processed.log.csv`
  and expects a numeric `ttft` column.

Typical usage (directory names under base_dir):
  python plot_ttft_ablation_compare.py \
    --base-dir /path/to/without_bitsandbytes \
    --exp1-dir contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_random_no_candidate_filtering-iter1-onlinelearning_1-20260329_033314 \
    --exp2-dir contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_random-iter1-onlinelearning_1-20260330_233613
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


DEFAULT_CSV_NAME = "filtered-aibrix-gateway-plugins-processed.log.csv"


def _resolve_csv_path(base_dir: str, exp_dir_or_csv: str, csv_name: str) -> str:
    """
    Resolve an experiment CSV path.

    - If `exp_dir_or_csv` already ends with `.csv`, treat it as a file path.
    - Otherwise treat it as a directory under `base_dir` and append `csv_name`.
    """
    # Allow passing explicit CSV paths.
    if exp_dir_or_csv.lower().endswith(".csv"):
        csv_path = exp_dir_or_csv
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        return csv_path

    # Otherwise treat as directory name.
    candidate = os.path.join(base_dir, exp_dir_or_csv, csv_name)
    if not os.path.exists(candidate):
        raise FileNotFoundError(
            "CSV not found for experiment directory.\n"
            f"  base_dir: {base_dir}\n"
            f"  exp: {exp_dir_or_csv}\n"
            f"  expected: {candidate}"
        )
    return candidate


def load_ttft_series(csv_path: str) -> np.ndarray:
    """Load and return a 1D numpy array of TTFT values (drop NaNs)."""
    df = pd.read_csv(csv_path)
    if "ttft" not in df.columns:
        raise KeyError(f"Missing `ttft` column in CSV: {csv_path}")
    ttft = pd.to_numeric(df["ttft"], errors="coerce").dropna().to_numpy(dtype=float)
    if ttft.size == 0:
        raise ValueError(f"No valid TTFT values in CSV: {csv_path}")
    return ttft


def compute_sliding_window_mean(
    values: np.ndarray, window_size: int, step: int
) -> Tuple[List[int], List[float]]:
    """Compute sliding-window mean over request index order."""
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    if step <= 0:
        raise ValueError("step must be > 0")

    n = int(values.shape[0])
    if n < window_size:
        # Fall back to one window using all data.
        window_size = n

    centers: List[int] = []
    means: List[float] = []

    last_start = n - window_size
    for start in range(0, last_start + 1, step):
        end = start + window_size
        window = values[start:end]
        if window.size == 0:
            continue
        centers.append(start + (window_size // 2))
        means.append(float(np.mean(window)))

    return centers, means


def plot_bar_avg_p99(
    avg1: float,
    p99_1: float,
    avg2: float,
    p99_2: float,
    label1: str,
    label2: str,
    out_path_pdf: str,
) -> None:
    """
    Plot Avg TTFT and P99 TTFT as grouped bars.

    X-axis uses experiment indices (1 and 2). Legend entries are formatted:
      "1: <label1>", "2: <label2>"
    """
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.titlesize": 24,
            "axes.labelsize": 23,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "legend.fontsize": 22,
        }
    )

    x = np.arange(2)
    w = 0.34
    exp_colors = ["#1f77b4", "#ff7f0e"]  # exp1/exp2 colors; used consistently

    avg_vals = [avg1, avg2]
    p99_vals = [p99_1, p99_2]
    max_avg = float(max(avg_vals))
    max_p99 = float(max(p99_vals))
    max_avg = max(max_avg, 1.0)
    max_p99 = max(max_p99, 1.0)

    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    ax2 = ax.twinx()

    # Avg/P99 offset within each experiment index.
    b_avg = ax.bar(
        x - w / 2,
        avg_vals,
        width=w,
        color=exp_colors,
        alpha=0.95,
        edgecolor="black",
    )
    b_p99 = ax2.bar(
        x + w / 2,
        p99_vals,
        width=w,
        color=exp_colors,
        alpha=0.55,
        edgecolor="black",
    )

    ax.set_title("TTFT Ablation (Avg vs P99)")
    ax.set_ylabel("Avg TTFT (ms)", fontsize=22, color="black")
    ax.set_xticks(x)
    ax.set_xticklabels(["1", "2"])
    ax.tick_params(axis="x", labelsize=20)
    ax.grid(axis="y", alpha=0.25)

    # Reserve extra headroom so rotated numeric labels never get clipped.
    # (avg left axis headroom, p99 right axis headroom).
    ax.set_ylim(0, max_avg * 2.35)
    ax2.set_ylim(0, max_p99 * 1.50)

    ax2.set_ylabel("P99 TTFT (ms)", fontsize=22, color="black")
    ax.tick_params(axis="y", labelsize=20, colors="black")
    ax2.tick_params(axis="y", labelsize=20, colors="black")

    # Annotate rotated values on their respective axes.
    # Keep labels just above bar tops; too-large offsets can collide with
    # the legend box near the top-left.
    label_offset_avg = max_avg * 0.0031
    label_offset_p99 = max_p99 * 0.0031
    for bar in b_avg:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + label_offset_avg,
            f"{h:.0f}",
            ha="center",
            va="bottom",
            rotation=90,
                fontsize=15,
            fontweight="bold",
        )
    for bar in b_p99:
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            h + label_offset_p99,
            f"{h:.0f}",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=15,
            fontweight="bold",
        )

    # Legend: experiment indices -> experiment names.
    legend_handles = [
        Patch(facecolor=exp_colors[0], edgecolor="black", label=f"1: {label1}"),
        Patch(facecolor=exp_colors[1], edgecolor="black", label=f"2: {label2}"),
    ]
    # Legend location: left top.
    ax.legend(handles=legend_handles, loc="upper left", framealpha=0.9)
    fig.tight_layout()

    fig.savefig(out_path_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_sliding_window(
    centers1: List[int],
    means1: List[float],
    centers2: List[int],
    means2: List[float],
    label1: str,
    label2: str,
    window_size: int,
    step: int,
    out_path_pdf: str,
) -> None:
    """Plot sliding-window mean TTFT vs request index (window mean)."""
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.titlesize": 22,
            "axes.labelsize": 22,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 18,
        }
    )

    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    ax.plot(centers1, means1, linewidth=2.0, label=label1, color="#1f77b4")
    ax.plot(centers2, means2, linewidth=2.0, label=label2, color="#ff7f0e")

    ax.set_title(f"Sliding-window mean TTFT (window={window_size}, step={step})")
    ax.set_xlabel("Request index", fontsize=22)
    ax.set_ylabel("Mean TTFT (ms)", fontsize=22)
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path_pdf, bbox_inches="tight")
    plt.close(fig)


def short_label(s: str) -> str:
    """Make labels shorter for plots."""
    # Keep directory part only (if user passed a full path).
    base = os.path.basename(s.rstrip("/"))
    # Drop timestamp suffix if it looks like -YYYYMMDD_...
    # (leave -iter1 and onlinelearning_1 intact)
    if "-" in base:
        # Split by last "-<digits...>" segment when it looks timestamp-like.
        parts = base.rsplit("-", 2)
        if len(parts) == 3 and parts[-1].replace("_", "").isdigit():
            return "-".join(parts[:-1])
    return base


def display_name_for_ablation(exp_dir: str) -> str:
    """
    Map the two ablation experiment directory names to short paper labels.
    """
    s = exp_dir.lower()
    if "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_random_no_candidate_filtering" in s:
        return "Quicksilver-without-k-candidate-filtering"
    if "contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_random" in s:
        # Must be checked after the no_candidate_filtering case.
        return "Quicksilver"
    return short_label(exp_dir)


def plot_combined_bar_and_timeseries(
    avg1: float,
    p99_1: float,
    avg2: float,
    p99_2: float,
    label1: str,
    label2: str,
    centers1: List[int],
    means1: List[float],
    centers2: List[int],
    means2: List[float],
    window_size: int,
    step: int,
    out_path_pdf: str,
) -> None:
    """Plot bar (Avg+P99, twin y-axis) and sliding-window time series side-by-side."""
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.titlesize": 22,
            "axes.labelsize": 22,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 21,
        }
    )

    fig, (bar_ax, ts_ax) = plt.subplots(1, 2, figsize=(18.0, 5.6))
    # Make room for y-labels while keeping subplots in the same row.
    fig.subplots_adjust(wspace=0.42)

    x = np.arange(2)
    w = 0.34
    exp_colors = ["#1f77b4", "#ff7f0e"]

    max_avg = max(float(avg1), float(avg2), 1.0)
    max_p99 = max(float(p99_1), float(p99_2), 1.0)

    bar_ax2 = bar_ax.twinx()

    # Bars
    b_avg = bar_ax.bar(
        x - w / 2,
        [avg1, avg2],
        width=w,
        color=exp_colors,
        alpha=0.95,
        edgecolor="black",
    )
    b_p99 = bar_ax2.bar(
        x + w / 2,
        [p99_1, p99_2],
        width=w,
        color=exp_colors,
        alpha=0.55,
        edgecolor="black",
    )

    bar_ax.set_title("TTFT Ablation (Avg vs P99)")
    bar_ax.set_ylabel("Avg TTFT (ms)", fontsize=22, color="black")
    bar_ax.set_xticks(x)
    bar_ax.set_xticklabels(["1", "2"])
    bar_ax.tick_params(axis="x", labelsize=20)
    bar_ax.grid(axis="y", alpha=0.25)

    bar_ax2.set_ylabel("P99 TTFT (ms)", fontsize=22, color="black")
    bar_ax.tick_params(axis="y", labelsize=20, colors="black")
    bar_ax2.tick_params(axis="y", labelsize=20, colors="black")

    # Headroom for rotated numeric annotations.
    bar_ax.set_ylim(0, max_avg * 2.35)
    bar_ax2.set_ylim(0, max_p99 * 1.50)

    label_offset_avg = max_avg * 0.0031
    label_offset_p99 = max_p99 * 0.0031

    for bar in b_avg:
        h = bar.get_height()
        bar_ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + label_offset_avg,
            f"{h:.0f}",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=15,
            fontweight="bold",
        )
    for bar in b_p99:
        h = bar.get_height()
        bar_ax2.text(
            bar.get_x() + bar.get_width() / 2,
            h + label_offset_p99,
            f"{h:.0f}",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=15,
            fontweight="bold",
        )

    # Time series subplot
    ts_ax.plot(centers1, means1, linewidth=2.0, label=label1, color="#1f77b4")
    ts_ax.plot(centers2, means2, linewidth=2.0, label=label2, color="#ff7f0e")
    ts_ax.set_title(f"Sliding-window mean TTFT")
    ts_ax.set_xlabel("Request index", fontsize=22)
    ts_ax.set_ylabel("Mean TTFT (ms)", fontsize=22)
    # Put y-axis label/ticks on the right for the time-series subplot.
    ts_ax.yaxis.set_label_position("right")
    ts_ax.yaxis.tick_right()
    ts_ax.tick_params(axis="x", labelsize=20)
    ts_ax.tick_params(axis="y", labelsize=20, labelleft=False, labelright=True)
    ts_ax.grid(alpha=0.25)

    # Single shared legend for both subplots; place outside at bottom-center.
    legend_handles = [
        Patch(facecolor=exp_colors[0], edgecolor="black", label=f"1: {label1}"),
        Patch(facecolor=exp_colors[1], edgecolor="black", label=f"2: {label2}"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        framealpha=0.9,
    )

    # Use subplots_adjust (not tight_layout) so `wspace` actually takes effect.
    # Reserve more bottom margin for the (now larger) shared legend.
    # Also avoid `bbox_inches="tight"` here since it can shrink the canvas
    # and cause the legend to overlap the subplots.
    fig.subplots_adjust(bottom=0.28, top=0.92, wspace=0.42)
    fig.savefig(out_path_pdf)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation plot: Avg TTFT/P99 + sliding-window mean TTFT.")
    parser.add_argument("--base-dir", required=True, help="Directory containing the two experiment subdirs.")
    parser.add_argument("--exp1-dir", required=True, help="Experiment subdir name under base-dir, or direct CSV path.")
    parser.add_argument("--exp2-dir", required=True, help="Experiment subdir name under base-dir, or direct CSV path.")
    parser.add_argument("--csv-name", default=DEFAULT_CSV_NAME, help=f"Per-request CSV name (default: {DEFAULT_CSV_NAME})")
    parser.add_argument("--window-size", type=int, default=500, help="Sliding window size (default: 500)")
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Sliding step size. Default: window_size//5 (at least 1).",
    )
    parser.add_argument("--out-dir", default=None, help="Output directory (default: base-dir)")

    args = parser.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else base_dir
    os.makedirs(out_dir, exist_ok=True)

    step = args.step
    if step is None:
        step = max(1, args.window_size // 5)

    csv1 = _resolve_csv_path(base_dir, args.exp1_dir, args.csv_name)
    csv2 = _resolve_csv_path(base_dir, args.exp2_dir, args.csv_name)

    label1 = display_name_for_ablation(args.exp1_dir)
    label2 = display_name_for_ablation(args.exp2_dir)

    ttft1 = load_ttft_series(csv1)
    ttft2 = load_ttft_series(csv2)

    avg1 = float(np.mean(ttft1))
    p99_1 = float(np.percentile(ttft1, 99))
    avg2 = float(np.mean(ttft2))
    p99_2 = float(np.percentile(ttft2, 99))

    centers1, means1 = compute_sliding_window_mean(ttft1, window_size=args.window_size, step=step)
    centers2, means2 = compute_sliding_window_mean(ttft2, window_size=args.window_size, step=step)

    out_base = os.path.join(out_dir, "ttft_ablation_compare")
    plot_bar_avg_p99(
        avg1=avg1,
        p99_1=p99_1,
        avg2=avg2,
        p99_2=p99_2,
        label1=label1,
        label2=label2,
        out_path_pdf=out_base + ".bar.pdf",
    )

    plot_sliding_window(
        centers1=centers1,
        means1=means1,
        centers2=centers2,
        means2=means2,
        label1=label1,
        label2=label2,
        window_size=args.window_size,
        step=step,
        out_path_pdf=out_base + ".timeseries.pdf",
    )

    plot_combined_bar_and_timeseries(
        avg1=avg1,
        p99_1=p99_1,
        avg2=avg2,
        p99_2=p99_2,
        label1=label1,
        label2=label2,
        centers1=centers1,
        means1=means1,
        centers2=centers2,
        means2=means2,
        window_size=args.window_size,
        step=step,
        out_path_pdf=out_base + ".combined.pdf",
    )

    print("=== TTFT ablation comparison ===")
    print(f"exp1: {label1}")
    print(f"  avg_ttft_ms: {avg1:.3f}")
    print(f"  p99_ttft_ms: {p99_1:.3f}")
    print(f"exp2: {label2}")
    print(f"  avg_ttft_ms: {avg2:.3f}")
    print(f"  p99_ttft_ms: {p99_2:.3f}")
    print(
        "Saved:\n"
        f"  {out_base}.bar.pdf\n"
        f"  {out_base}.timeseries.pdf\n"
        f"  {out_base}.combined.pdf"
    )


if __name__ == "__main__":
    main()

