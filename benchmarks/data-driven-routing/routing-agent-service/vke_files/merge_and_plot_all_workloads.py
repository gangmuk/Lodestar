#!/usr/bin/env python3
"""
Merge routing_strategy_metrics.csv files from multiple workloads and create comparison plots.

Usage:
    python merge_and_plot_all_workloads.py <base_dir> [--output-dir <output_dir>]

Example:
    python merge_and_plot_all_workloads.py /path/to/workload-and-experiment_results/NVIDIA-A10/maxTokens_1-maxTokensStd_0
"""

import os
import sys
import glob
import argparse
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.gridspec import GridSpec

# Font sizes
maintitle_fontsize = 24
subtitle_fontsize = 18
legend_fontsize = 10
ylabel_fontsize = 14
tick_fontsize = 10
value_fontsize = 7


# Color families for routing policy categories
POLICY_COLOR_FAMILIES = {
    'rl_naive': ['#4169e1', '#483d8b', '#6a5acd', '#7b68ee', '#9370db'],  # Slate blue family
    'latency_predictor_e2e_latency': ['#8b008b', '#ba55d3', '#9932cc', '#8a2be2', '#c71585'],  # Purple family
    'latency_predictor_ttft': ['#ff1493', '#ff69b4', '#dc143c', '#ff00ff', '#da70d6'],  # Pink/Magenta family
    'latency_predictor_avg_tpot': ['#8b0000', '#b22222', '#cd5c5c', '#f08080', '#fa8072'],  # Dark red/Coral family
    'prefix_cache_1': ['#1f77b4', '#4682b4', '#6495ed', '#aec7e8', '#87ceeb'],  # Blue family
    'prefix_cache_2': ['#006400', '#228b22', '#32cd32', '#00ff00', '#7cfc00'],  # Dark green/Lime family
    'preble': ['#ff8c00', '#ffa500', '#ffd700', '#ff6347', '#ff4500'],  # Orange/Gold family
    'random': ['#2ca02c', '#32cd32', '#00ff00', '#00ff7f', '#98df8a'],  # Light green family
    'least_kv_cache': ['#d2691e', '#cd853f', '#daa520', '#b8860b', '#f4a460'],  # Brown/Tan family
    'least_latency': ['#483d8b', '#6a5acd', '#7b68ee', '#9370db', '#8470ff'],  # Slate blue family
    'least_request': ['#008b8b', '#20b2aa', '#48d1cc', '#40e0d0', '#00ced1'],  # Cyan/Teal family
    # Contextual bandit variants use the same red family as compare_routing_strategies.py
    'contextual_bandit_quantile_based_perpodmodel_advanced': ['#ff0000', '#dc143c', '#ff6347', '#ff4500', '#ff7f50'],  # Red family
    'contextual_bandit_quantile_based_perpodmodel': ['#ff0000', '#dc143c', '#ff6347', '#ff4500', '#ff7f50'],  # Red family
    'contextual_bandit_perpodmodel_policygradient_throughput_based': ['#ff0000', '#dc143c', '#ff6347', '#ff4500', '#ff7f50'],  # Red family
    'contextual_bandit_perpodmodel_policygradient': ['#ff0000', '#dc143c', '#ff6347', '#ff4500', '#ff7f50'],  # Red family
    'contextual_bandit_negative_linear_perpodmodel': ['#ff0000', '#dc143c', '#ff6347', '#ff4500', '#ff7f50'],  # Red family
    'contextual_bandit_negative_squared_perpodmodel': ['#ff0000', '#dc143c', '#ff6347', '#ff4500', '#ff7f50'],  # Red family
    'contextual_bandit': ['#ff0000', '#dc143c', '#ff6347', '#ff4500', '#ff7f50'],  # Red family
}

DEFAULT_COLORS = ['#7f7f7f', '#696969', '#a9a9a9', '#c7c7c7', '#d3d3d3']  # Gray family

# Preferred ordering for routing policies in plots.
PREFERRED_POLICY_ORDER = [
    'random',
    'least_request',
    'least_latency',
    'prefix_cache_1',
    'contextual_bandit',
]

PREFERRED_WORKLOAD_ORDER = [
    'SharingRatio71%',
    'SharingRatio47%',
    'SharingRatio28%',
    'SharingRatio9%',
]


def order_policies(policies):
    """Order policies with preferred ones first, then remaining alphabetically."""
    preferred = []
    preferred_set = set()
    for name in PREFERRED_POLICY_ORDER:
        if name in policies:
            preferred.append(name)
            preferred_set.add(name)
    remaining = sorted([p for p in policies if p not in preferred_set])
    return preferred + remaining


def order_workloads(workloads):
    """Order workloads with preferred SharingRatio groups first, then remaining."""
    preferred = []
    preferred_set = set()
    for name in PREFERRED_WORKLOAD_ORDER:
        for workload in workloads:
            if name in workload and workload not in preferred_set:
                preferred.append(workload)
                preferred_set.add(workload)
    remaining = sorted([w for w in workloads if w not in preferred_set])
    return preferred + remaining


def categorize_policy(policy_name):
    """Categorize a policy name into one of the predefined categories."""
    policy_lower = policy_name.lower()

    # Check each category in order of specificity (more specific first)
    if 'rl_naive' in policy_lower:
        return 'rl_naive'
    elif 'latency_predictor_e2e' in policy_lower:
        return 'latency_predictor_e2e_latency'
    elif 'latency_predictor_ttft' in policy_lower:
        return 'latency_predictor_ttft'
    elif 'latency_predictor_avg_tpot' in policy_lower:
        return 'latency_predictor_avg_tpot'
    elif 'prefix_cache_1' in policy_lower:
        return 'prefix_cache_1'
    elif 'prefix_cache_2' in policy_lower:
        return 'prefix_cache_2'
    elif 'preble' in policy_lower:
        return 'preble'
    elif 'random' in policy_lower:
        return 'random'
    elif 'least_kv_cache' in policy_lower:
        return 'least_kv_cache'
    elif 'least_latency' in policy_lower:
        return 'least_latency'
    elif 'least_request' in policy_lower:
        return 'least_request'
    # Contextual bandit variants (more specific patterns first)
    elif 'contextual_bandit_quantile_based_perpodmodel_advanced' in policy_lower:
        return 'contextual_bandit_quantile_based_perpodmodel_advanced'
    elif 'contextual_bandit_quantile_based_perpodmodel' in policy_lower:
        return 'contextual_bandit_quantile_based_perpodmodel'
    elif 'contextual_bandit_perpodmodel_policygradient_throughput_based' in policy_lower:
        return 'contextual_bandit_perpodmodel_policygradient_throughput_based'
    elif 'contextual_bandit_perpodmodel_policygradient' in policy_lower:
        return 'contextual_bandit_perpodmodel_policygradient'
    elif 'contextual_bandit_negative_linear_perpodmodel' in policy_lower:
        return 'contextual_bandit_negative_linear_perpodmodel'
    elif 'contextual_bandit_negative_squared_perpodmodel' in policy_lower:
        return 'contextual_bandit_negative_squared_perpodmodel'
    elif 'contextual_bandit' in policy_lower:
        return 'contextual_bandit'
    else:
        return None


def generate_policy_colors(policies):
    """Generate colors for each policy using color families based on category."""
    colors = {}
    category_counts = {}  # Track how many policies we've seen in each category

    for policy in sorted(policies):
        category = categorize_policy(policy)

        if category and category in POLICY_COLOR_FAMILIES:
            # Get the index within this category
            if category not in category_counts:
                category_counts[category] = 0
            idx = category_counts[category]
            category_counts[category] += 1

            # Get color from the family, cycling if needed
            color_family = POLICY_COLOR_FAMILIES[category]
            colors[policy] = color_family[idx % len(color_family)]
        else:
            # Use gray family for unknown categories
            if 'unknown' not in category_counts:
                category_counts['unknown'] = 0
            idx = category_counts['unknown']
            category_counts['unknown'] += 1
            colors[policy] = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]

    return colors


def get_policy_color(policy, color_map):
    """Get color for a routing policy from the color map."""
    return color_map.get(policy, '#7f7f7f')


def extract_routing_policy(strategy_full_name):
    """
    Extract routing policy from strategy_full_name.

    Examples:
    - "contextual_bandit_perpodmodel_policygradient_throughput_based-iter4-20251224_114736"
      -> "contextual_bandit_perpodmodel_policygradient_throughput_based"
    - "least_request-iter1--20251122_123525" -> "least_request"
    - "random-iter1--20251122_131129" -> "random"
    """
    # Pattern: everything before "-iter" or before "-YYYYMMDD_HHMMSS"
    # First try to match -iter pattern
    match = re.match(r'^(.+?)-iter\d+', strategy_full_name)
    if match:
        return match.group(1)

    # Try to match timestamp pattern directly
    match = re.match(r'^(.+?)-\d{8}_\d{6}', strategy_full_name)
    if match:
        return match.group(1)

    # Fallback: return as-is
    return strategy_full_name


def find_metrics_files(base_dir):
    """Find all routing_strategy_metrics.csv files under base_dir."""
    pattern = os.path.join(base_dir, "**", "routing_strategy_metrics.csv")
    files = glob.glob(pattern, recursive=True)
    return files


def merge_metrics_files(files):
    """Merge all metrics CSV files into a single DataFrame."""
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
            print(f"Loaded {len(df)} rows from {f}")
        except Exception as e:
            print(f"Warning: Could not read {f}: {e}")

    if not dfs:
        return None

    merged = pd.concat(dfs, ignore_index=True)
    # Normalize workload to a consistent string column to avoid mixed-type sorting errors.
    if 'workload' in merged.columns:
        merged['workload'] = merged['workload'].fillna("unknown").astype(str)
    print(f"\nMerged {len(merged)} total rows from {len(dfs)} files")
    return merged


def plot_individual_results(df, metric, ylabel, title, output_path):
    """Create a grouped bar chart showing each individual experiment result."""

    # Get unique workloads
    workloads = order_workloads(df['workload'].unique())
    n_workloads = len(workloads)

    # Create figure - wider to accommodate many bars
    fig, ax = plt.subplots(figsize=(max(16, n_workloads * 4), 10))

    # For each workload, get all experiments
    all_positions = []
    all_values = []
    all_colors = []
    all_labels = []
    group_centers = []

    current_pos = 0
    bar_width = 0.8
    group_gap = 2  # Gap between workload groups

    for workload in workloads:
        workload_data = df[df['workload'] == workload].copy()
        workload_data = workload_data.copy()
        workload_data['routing_policy'] = pd.Categorical(
            workload_data['routing_policy'],
            categories=order_policies(workload_data['routing_policy'].unique()),
            ordered=True
        )
        workload_data = workload_data.sort_values('routing_policy')

        group_start = current_pos

        for idx, (_, row) in enumerate(workload_data.iterrows()):
            all_positions.append(current_pos)
            val = row[metric] if pd.notna(row[metric]) else 0
            all_values.append(val)
            all_colors.append(get_policy_color(row['routing_policy']))
            # Short label: routing_policy
            all_labels.append(row['routing_policy'])
            current_pos += bar_width + 0.1

        group_centers.append((group_start + current_pos - bar_width - 0.1) / 2)
        current_pos += group_gap

    # Plot bars
    bars = ax.bar(all_positions, all_values, bar_width, color=all_colors, edgecolor='black', linewidth=0.5)

    # Add value labels on bars
    for bar, val, label in zip(bars, all_values, all_labels):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                   f'{val:.0f}', ha='center', va='bottom', fontsize=value_fontsize, rotation=90)

    # Add x-axis labels for each bar (routing policy)
    ax.set_xticks(all_positions)
    ax.set_xticklabels(all_labels, rotation=90, ha='center', fontsize=tick_fontsize - 2)

    # Add workload labels below
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(group_centers)
    ax2.set_xticklabels(workloads, fontsize=tick_fontsize)
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 60))

    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    ax.set_title(title, fontsize=subtitle_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def plot_averaged_results(df, metric, ylabel, title, output_path):
    """Create a grouped bar chart with averaged results per routing policy."""

    # Get unique workloads and policies
    workloads = order_workloads(df['workload'].unique())
    policies = order_policies(df['routing_policy'].unique())

    n_workloads = len(workloads)
    n_policies = len(policies)

    # Create figure
    fig, ax = plt.subplots(figsize=(max(14, n_workloads * 2.5), 10))

    # Bar positioning
    bar_width = 0.8 / n_policies
    x = np.arange(n_workloads)

    # Plot bars for each policy
    for i, policy in enumerate(policies):
        policy_data = df[df['routing_policy'] == policy]
        values = []
        stds = []
        for workload in workloads:
            workload_policy_data = policy_data[policy_data['workload'] == workload]
            if len(workload_policy_data) > 0:
                values.append(workload_policy_data[metric].mean())
                stds.append(workload_policy_data[metric].std() if len(workload_policy_data) > 1 else 0)
            else:
                values.append(0)
                stds.append(0)

        offset = (i - n_policies / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, values, bar_width,
                      label=policy, color=get_policy_color(policy),
                      edgecolor='black', linewidth=0.5,
                      yerr=stds, capsize=2, error_kw={'linewidth': 0.5})

        # Add value labels on bars
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                       f'{val:.0f}', ha='center', va='bottom', fontsize=value_fontsize, rotation=90)

    ax.set_xlabel('Workload', fontsize=ylabel_fontsize)
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    ax.set_title(title, fontsize=subtitle_fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(workloads, rotation=45, ha='right', fontsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    ax.legend(fontsize=legend_fontsize, loc='upper left', bbox_to_anchor=(1.02, 1))
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def plot_all_metrics_summary(df, output_dir, averaged=True):
    """Create a comprehensive summary plot with one workload per row.

    Each row shows bar chart with avg, p99, p999 grouped by routing policy,
    similar to plot_single_metric_comparison in compare_routing_strategies.py.
    """
    from matplotlib.patches import Patch

    # Get unique workloads and policies
    workloads = order_workloads(df['workload'].unique())
    policies = order_policies(df['routing_policy'].unique())

    n_workloads = len(workloads)
    n_policies = len(policies)

    # Generate colors for each policy
    policy_colors = generate_policy_colors(policies)

    # Create figure with one row per workload
    fig = plt.figure(figsize=(max(18, n_policies * 1.2), 6 * n_workloads))
    gs = GridSpec(n_workloads, 1, figure=fig, hspace=0.6)

    for workload_idx, workload in enumerate(workloads):
        ax = fig.add_subplot(gs[workload_idx, 0])

        # Filter data for this workload
        workload_df = df[df['workload'] == workload]

        # Get policies that have data for this workload
        available_policies = [p for p in policies if p in workload_df['routing_policy'].values]

        if not available_policies:
            ax.text(0.5, 0.5, f'No data for {workload}', ha='center', va='center')
            continue

        # Bar positioning: 3 bars (avg, p99, p999) per policy
        bar_width = 0.25
        group_width = 3 * bar_width + 0.3  # Space between groups
        group_centers = np.arange(len(available_policies)) * group_width

        # Collect metrics for each policy
        avg_values = []
        p99_values = []
        p999_values = []

        for policy in available_policies:
            policy_data = workload_df[workload_df['routing_policy'] == policy]
            if len(policy_data) > 0:
                avg_values.append(policy_data['avg_ttft'].mean())
                p99_values.append(policy_data['p99_ttft'].mean())
                p999_values.append(policy_data['p999_ttft'].mean())
            else:
                avg_values.append(0)
                p99_values.append(0)
                p999_values.append(0)

        # Get max value for y-axis scaling
        max_value = max(max(avg_values or [0]), max(p99_values or [0]), max(p999_values or [0]))

        # Plot bars for each policy (avg, p99, p999)
        for i, policy in enumerate(available_policies):
            strategy_color = policy_colors[policy]
            group_center = group_centers[i]

            # Calculate positions for the 3 bars in each group
            offset_start = -bar_width

            # Create bars with slight color variations (like original script)
            for j, (value, label, alpha) in enumerate([
                (avg_values[i], 'Avg', 0.9),
                (p99_values[i], 'P99', 0.7),
                (p999_values[i], 'P999', 0.5)
            ]):
                pos = group_center + offset_start + j * bar_width
                ax.bar(pos, value, bar_width, color=strategy_color,
                       edgecolor='black', linewidth=0.8, alpha=alpha)

                # Add value labels on top of bars
                if value > 0:
                    ax.text(pos, value + max_value * 0.02,
                           f'{value:.0f}', rotation=90, ha='center', va='bottom',
                           fontsize=10, fontweight='bold')

        # Set up x-axis with policy names (like original: "policy_name\n(timestamp)")
        ax.set_xticks(group_centers)

        # Create labels: policy name + timestamp if available
        strategy_labels = []
        for policy in available_policies:
            # Get full strategy name from df if available
            policy_data = workload_df[workload_df['routing_policy'] == policy]
            if len(policy_data) > 0 and 'strategy_full_name' in policy_data.columns:
                full_name = policy_data['strategy_full_name'].iloc[0]
                parts = full_name.split('-')
                if len(parts) >= 2:
                    # Policy name + timestamp in parentheses
                    label = f"{parts[0]}\n({parts[-1]})"
                else:
                    label = policy
            else:
                label = policy
            strategy_labels.append(label)

        ax.set_xticklabels(strategy_labels, fontsize=10, rotation=45, ha='right')

        # Add legend for Avg/P99/P999
        legend_elements = [
            Patch(facecolor='gray', edgecolor='black', alpha=0.9, label='Avg'),
            Patch(facecolor='gray', edgecolor='black', alpha=0.7, label='P99'),
            Patch(facecolor='gray', edgecolor='black', alpha=0.5, label='P999')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12, ncol=3)

        # Styling
        ax.set_ylabel('TTFT (ms)', fontsize=ylabel_fontsize)
        # Use workload name as title
        ax.set_title(f'TTFT Latency Comparison - {workload}', fontsize=subtitle_fontsize)
        ax.tick_params(axis='y', labelsize=tick_fontsize)
        ax.tick_params(axis='x', labelsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max_value * 1.4)

    suffix = "(Averaged)" if averaged else "(Individual)"
    plt.suptitle(f'Routing Strategy Performance Across All Workloads {suffix}', fontsize=maintitle_fontsize, y=0.99)

    filename = "all_workloads_summary_averaged.pdf" if averaged else "all_workloads_summary_individual.pdf"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Merge and plot routing metrics across workloads')
    parser.add_argument('base_dir', help='Base directory to search for routing_strategy_metrics.csv files')
    parser.add_argument('--output-dir', '-o', default=None,
                        help='Output directory for merged CSV and plots (default: base_dir)')

    args = parser.parse_args()

    base_dir = args.base_dir
    output_dir = args.output_dir if args.output_dir else base_dir

    # Find all metrics files
    print(f"Searching for metrics files in {base_dir}...")
    files = find_metrics_files(base_dir)

    if not files:
        print(f"No routing_strategy_metrics.csv files found in {base_dir}")
        sys.exit(1)

    print(f"Found {len(files)} metrics files")

    # Merge files
    df = merge_metrics_files(files)
    if df is None or len(df) == 0:
        print("No data to process")
        sys.exit(1)

    # Re-extract routing_policy from strategy_full_name using the correct logic
    print("\nRe-extracting routing policies from strategy names...")
    df['routing_policy'] = df['strategy_full_name'].apply(extract_routing_policy)

    # Save merged CSV
    os.makedirs(output_dir, exist_ok=True)
    merged_csv_path = os.path.join(output_dir, "merged_routing_metrics.csv")
    df.to_csv(merged_csv_path, index=False)
    print(f"\nSaved merged CSV to {merged_csv_path}")

    # Print summary
    print("\n--- Data Summary ---")
    print(f"Workloads: {sorted(df['workload'].unique().tolist())}")
    print(f"Routing policies: {sorted(df['routing_policy'].unique().tolist())}")
    print(f"Total rows: {len(df)}")

    # Show counts per policy
    print("\nExperiments per routing policy:")
    for policy in sorted(df['routing_policy'].unique()):
        count = len(df[df['routing_policy'] == policy])
        print(f"  {policy}: {count}")

    # Create summary plot only
    print("\n--- Generating Summary Plot ---")
    plot_all_metrics_summary(df, output_dir, averaged=True)

    print("\nDone!")


if __name__ == "__main__":
    main()
