#!/usr/bin/env python3
"""Plot TTFT sliding window (mean, p50, p99) across experiment subdirs."""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

WINDOW_SIZE = 1000
STEP_SIZE = 200  # slide by 200 requests

plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 14,
})

if len(sys.argv) < 2:
    print("Usage: python plot_ttft_sliding_window.py <experiment_parent_dir>")
    sys.exit(1)
BASE_DIR = os.path.abspath(sys.argv[1])
CSV_NAME = "filtered-aibrix-gateway-plugins-processed.log.csv"

# Routing policy color families (same as compare_routing_strategies.py)
STRATEGY_COLOR_FAMILIES = {
    'random':        ['#2ca02c', '#32cd32', '#00ff00', '#00ff7f', '#98df8a'],
    'least_request': ['#008b8b', '#20b2aa', '#48d1cc', '#40e0d0', '#00ced1'],
    'least_kv_cache':['#d2691e', '#cd853f', '#daa520', '#b8860b', '#f4a460'],
    'least_latency': ['#483d8b', '#6a5acd', '#7b68ee', '#9370db', '#8470ff'],
    'prefix_cache_1':['#1f77b4', '#4682b4', '#6495ed', '#aec7e8', '#87ceeb'],
    'prefix_cache_2':['#006400', '#228b22', '#32cd32', '#00ff00', '#7cfc00'],
    'preble':        ['#ff8c00', '#ffa500', '#ffd700', '#ff6347', '#ff4500'],
    'contextual_bandit': ['#ff0000', '#dc143c', '#ff6347', '#ff4500', '#ff7f50'],
    'rl_naive':      ['#4169e1', '#483d8b', '#6a5acd', '#7b68ee', '#9370db'],
    'latency_predictor_e2e_latency':  ['#8b008b', '#ba55d3', '#9932cc', '#8a2be2', '#c71585'],
    'latency_predictor_ttft':         ['#ff1493', '#ff69b4', '#dc143c', '#ff00ff', '#da70d6'],
    'latency_predictor_avg_tpot':     ['#8b0000', '#b22222', '#cd5c5c', '#f08080', '#fa8072'],
    'other':         ['#7f7f7f', '#696969', '#a9a9a9', '#c7c7c7', '#d3d3d3'],
}

def categorize_strategy(dirname):
    """Map a directory name to a routing policy category."""
    s = dirname.lower()
    for key in ['latency_predictor_e2e_latency', 'latency_predictor_ttft',
                'latency_predictor_avg_tpot', 'rl_naive',
                'prefix_cache_1', 'prefix_cache_2', 'preble',
                'contextual_bandit',
                'random', 'least_kv_cache', 'least_latency',
                'least_request']:
        if key in s:
            return key
    return 'other'


def get_short_name(dirname):
    """Extract a readable short name from the experiment directory name."""
    # Remove timestamp suffix like -20260312_163840
    parts = dirname.rsplit("-", 2)
    if len(parts) >= 3 and parts[-1].replace("_", "").isdigit():
        dirname = "-".join(parts[:-1])
    # Further shorten: remove common prefixes
    name = dirname
    for prefix in ["contextual_bandit_perpodmodel_checkpoint_ttft_negative_linear_"]:
        if name.startswith(prefix):
            name = "cb_" + name[len(prefix):]
    # Truncate iter info for readability
    if "-iter" in name:
        base, _, rest = name.partition("-iter")
        name = base + "-iter" + rest.split("-")[0]
    return name


def assign_colors(dir_names):
    """Assign colors to experiments based on their routing policy category."""
    category_counters = {}
    color_map = {}
    for dirname in dir_names:
        cat = categorize_strategy(dirname)
        idx = category_counters.get(cat, 0)
        family = STRATEGY_COLOR_FAMILIES.get(cat, STRATEGY_COLOR_FAMILIES['other'])
        color_map[dirname] = family[idx % len(family)]
        category_counters[cat] = idx + 1
    return color_map


def compute_sliding_window_stats(ttft_values, window_size=WINDOW_SIZE, step=STEP_SIZE):
    """Compute sliding window mean, p50, p99 over request indices."""
    n = len(ttft_values)
    if n < window_size:
        print(f"  Warning: only {n} requests, less than window size {window_size}")
        window_size = n

    centers = []
    means = []
    p50s = []
    p99s = []

    for start in range(0, n - window_size + 1, step):
        end = start + window_size
        window = ttft_values[start:end]
        centers.append(start + window_size // 2)
        means.append(np.mean(window))
        p50s.append(np.percentile(window, 50))
        p99s.append(np.percentile(window, 99))

    return centers, means, p50s, p99s


def main():
    # Find all experiment subdirs with the CSV
    experiment_dirs = sorted(glob.glob(os.path.join(BASE_DIR, "*", CSV_NAME)))
    if not experiment_dirs:
        print("No experiment CSVs found!")
        sys.exit(1)

    print(f"Found {len(experiment_dirs)} experiments")

    # Load data; key by dirname to avoid collisions from identical short names
    experiments = {}  # dirname -> ttft
    dir_names = []    # original dir names in order
    display_names = {}  # dirname -> short display label
    for csv_path in experiment_dirs:
        dirname = os.path.basename(os.path.dirname(csv_path))
        short_name = get_short_name(dirname)
        df = pd.read_csv(csv_path)
        if "ttft" not in df.columns:
            print(f"  Skipping {short_name}: no 'ttft' column")
            continue
        ttft = df["ttft"].dropna().values.astype(float)
        print(f"  {short_name}: {len(ttft)} requests")
        experiments[dirname] = ttft
        dir_names.append(dirname)
        display_names[dirname] = short_name

    if not experiments:
        print("No valid data found!")
        sys.exit(1)

    # Disambiguate display names that collide by appending timestamp
    name_counts = {}
    for dn, sn in display_names.items():
        name_counts.setdefault(sn, []).append(dn)
    for sn, dns in name_counts.items():
        if len(dns) > 1:
            for dn in dns:
                # Extract timestamp from dirname (last two dash-separated parts)
                parts = dn.rsplit("-", 2)
                if len(parts) >= 3 and parts[-1].replace("_", "").isdigit():
                    ts = parts[-2] + "_" + parts[-1]
                    display_names[dn] = f"{sn}-{ts}"

    # Assign colors by routing policy category
    color_map = assign_colors(dir_names)

    # Compute all stats first
    all_stats = {}  # dirname -> (centers, means, p50s, p99s)
    for dirname, ttft in experiments.items():
        all_stats[dirname] = compute_sliding_window_stats(ttft)

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 18), sharex=True)
    stat_names = ["Mean TTFT", "P50 TTFT", "P99 TTFT"]
    stat_indices = [1, 2, 3]  # index into (centers, means, p50s, p99s)

    for dirname in experiments:
        centers, means, p50s, p99s = all_stats[dirname]
        c = color_map[dirname]
        label = display_names[dirname]
        for ax, values in zip(axes, [means, p50s, p99s]):
            ax.plot(centers, values, label=label, color=c, alpha=0.8, linewidth=1.5)

    # Add colored background strip showing which policy has the lowest value
    for ax, stat_idx in zip(axes, stat_indices):
        # Build a lookup: center -> {name: value} for all experiments
        center_values = {}
        for dirname in experiments:
            centers = all_stats[dirname][0]
            values = all_stats[dirname][stat_idx]
            for c_val, v_val in zip(centers, values):
                if c_val not in center_values:
                    center_values[c_val] = {}
                center_values[c_val][dirname] = v_val

        # Only compare at centers where at least 2 experiments have data
        comparable_centers = sorted(c for c, d in center_values.items() if len(d) >= 2)
        if not comparable_centers:
            continue

        # For each center, find the winner
        half_step = STEP_SIZE / 2
        ylim_low = ax.get_ylim()[0]  # will set after lines are drawn
        # Track contiguous winner regions for labeling
        winner_runs = []  # list of (start_center, end_center, winner_dirname)
        prev_winner = None
        for center in comparable_centers:
            vals = center_values[center]
            winner = min(vals, key=vals.get)
            ax.axvspan(center - half_step, center + half_step,
                       ymin=0, ymax=0.06,  # bottom 6% of the axes
                       color=color_map[winner], alpha=1.0, linewidth=0)
            if winner == prev_winner:
                winner_runs[-1] = (winner_runs[-1][0], center, winner)
            else:
                winner_runs.append((center, center, winner))
            prev_winner = winner

        # Draw vertical boundary lines where the winner changes
        for i in range(1, len(winner_runs)):
            boundary_x = (winner_runs[i-1][1] + winner_runs[i][0]) / 2
            ax.axvline(boundary_x, ymin=0, ymax=0.06, color='black', linewidth=2, zorder=5)

        # Place rotated policy name at the center of each contiguous winner region
        for run_start, run_end, winner_dn in winner_runs:
            mid_x = (run_start + run_end) / 2
            ax.text(mid_x, 0.08, display_names[winner_dn],
                    transform=ax.get_xaxis_transform(),
                    ha='left', va='bottom', fontsize=11, rotation=45,
                    rotation_mode='anchor',
                    color=color_map[winner_dn], fontweight='bold',
                    clip_on=True)

    for ax, stat_name in zip(axes, stat_names):
        ax.set_ylabel(f"{stat_name} (ms)")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{stat_name} (sliding window={WINDOW_SIZE}, step={STEP_SIZE})")

    # Single legend above the top axes, centered
    handles, labels = axes[0].get_legend_handles_labels()
    leg = axes[0].legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.15),
                         ncol=min(len(labels), 3), borderaxespad=0, frameon=True)
    for line in leg.get_lines():
        line.set_linewidth(8.0)

    axes[-1].set_xlabel("Request Index")
    plt.tight_layout()

    out_path = os.path.join(BASE_DIR, "ttft_sliding_window_comparison.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    print(f"\nSaved to {out_path}")

    out_png = os.path.join(BASE_DIR, "ttft_sliding_window_comparison.png")
    plt.savefig(out_png, bbox_inches="tight", dpi=150)
    print(f"Saved to {out_png}")


if __name__ == "__main__":
    main()
