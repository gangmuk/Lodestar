#!/usr/bin/env python3
"""
Merge routing_strategy_metrics_from_client_log.csv files from multiple workloads and create comparison plots.

Usage:
    python merge_and_plot_all_workloads_from_client_log.py <base_dir> [--output-dir <output_dir>]

Example:
    python merge_and_plot_all_workloads_from_client_log.py /path/to/workload-and-experiment_results/NVIDIA-A10/maxTokens_1-maxTokensStd_0
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
from matplotlib.patches import Patch
from matplotlib.backends.backend_pdf import PdfPages

# Font sizes
maintitle_fontsize = 24
subtitle_fontsize = 18
legend_fontsize = 10
ylabel_fontsize = 14
tick_fontsize = 10
value_fontsize = 7


# Color families for routing policy categories
POLICY_COLOR_FAMILIES = {
    'rl_naive': ['#4169e1', '#483d8b', '#6a5acd', '#7b68ee', '#9370db'],
    'latency_predictor_e2e_latency': ['#8b008b', '#ba55d3', '#9932cc', '#8a2be2', '#c71585'],
    'latency_predictor_ttft': ['#ff1493', '#ff69b4', '#dc143c', '#ff00ff', '#da70d6'],
    'latency_predictor_avg_tpot': ['#8b0000', '#b22222', '#cd5c5c', '#f08080', '#fa8072'],
    'prefix_cache_1': ['#1f77b4', '#4682b4', '#6495ed', '#aec7e8', '#87ceeb'],
    'prefix_cache_2': ['#006400', '#228b22', '#32cd32', '#00ff00', '#7cfc00'],
    'preble': ['#ff8c00', '#ffa500', '#ffd700', '#ff6347', '#ff4500'],
    'random': ['#2ca02c', '#32cd32', '#00ff00', '#00ff7f', '#98df8a'],
    'least_kv_cache': ['#d2691e', '#cd853f', '#daa520', '#b8860b', '#f4a460'],
    'least_latency': ['#483d8b', '#6a5acd', '#7b68ee', '#9370db', '#8470ff'],
    'least_request': ['#008b8b', '#20b2aa', '#48d1cc', '#40e0d0', '#00ced1'],
    'prefix_hit_threshold_or_least_request': ['#556b2f', '#6b8e23', '#808000', '#9acd32', '#bdb76b'],  # Olive/green-yellow family
    'contextual_bandit_quantile_based_perpodmodel_advanced': ['#ff0000', '#dc143c', '#ff6347', '#ff4500', '#ff7f50'],
    'contextual_bandit_quantile_based_perpodmodel': ['#ff0000', '#dc143c', '#ff6347', '#ff4500', '#ff7f50'],
    'contextual_bandit_perpodmodel_policygradient_throughput_based': ['#ff0000', '#dc143c', '#ff6347', '#ff4500', '#ff7f50'],
    'contextual_bandit_perpodmodel_policygradient': ['#ff0000', '#dc143c', '#ff6347', '#ff4500', '#ff7f50'],
    'contextual_bandit_negative_linear_perpodmodel': ['#ff0000', '#dc143c', '#ff6347', '#ff4500', '#ff7f50'],
    'contextual_bandit_negative_squared_perpodmodel': ['#ff0000', '#dc143c', '#ff6347', '#ff4500', '#ff7f50'],
    'contextual_bandit': ['#ff0000', '#dc143c', '#ff6347', '#ff4500', '#ff7f50'],
}

DEFAULT_COLORS = ['#7f7f7f', '#696969', '#a9a9a9', '#c7c7c7', '#d3d3d3']

# Preferred ordering for routing policies in plots
# Order: random -> least_request -> prefix_cache -> contextual_bandit
PREFERRED_POLICY_ORDER = [
    'random',
    'least_request',
    'prefix_cache_1',
    'prefix_cache_2',
    'prefix_cache',
    'contextual_bandit',
]


def extract_rps_from_workload(workload):
    """Extract RPS number from workload string for sorting.

    Examples:
    - "gangmuk-prefix/SharingRatio71%/rps4-benchmark/without_bitsandbytes" -> 4
    - "mooncake/conversation-2/rps15-benchmark/without_bitsandbytes" -> 15
    """
    match = re.search(r'rps([\d,]+)', workload.lower())
    if match:
        # For multi-RPS like "rps6,11,8", return the first value for sorting
        return int(match.group(1).split(',')[0])
    return 9999  # Put workloads without RPS at the end


def extract_sharing_ratio(workload):
    """Extract SharingRatio percentage from workload string for sorting.

    Examples:
    - "gangmuk-prefix/SharingRatio71%/rps4" -> 71
    - "gangmuk-prefix/SharingRatio9%/rps6" -> 9
    - "MixedSharingRatio10_30_50_70%" -> average or first value
    """
    # Handle MixedSharingRatio
    match = re.search(r'MixedSharingRatio(\d+)', workload)
    if match:
        return int(match.group(1))
    # Handle regular SharingRatio
    match = re.search(r'SharingRatio(\d+)', workload)
    if match:
        return int(match.group(1))
    return 9999  # Put workloads without SharingRatio at the end


def extract_workload_category(workload):
    """Extract workload category from workload string.

    Examples:
    - "gangmuk-prefix/SharingRatio71%/rps4" -> "gangmuk-prefix"
    - "mooncake/conversation-2/rps15" -> "mooncake"
    - "azure/azure_code_poisson/rps25" -> "azure"
    """
    # Split by '/' and get the first meaningful part
    parts = workload.split('/')
    for part in parts:
        part_lower = part.lower()
        if 'gangmuk' in part_lower or 'prefix' in part_lower:
            return 'gangmuk-prefix'
        elif 'mooncake' in part_lower:
            return 'mooncake'
        elif 'azure' in part_lower:
            return 'azure'
    # If no known category, return the first part
    return parts[0] if parts else 'unknown'


def extract_mooncake_subcategory(workload):
    """Extract mooncake subcategory (conversation-2, toolagent-2, etc.) for sorting.

    Examples:
    - "mooncake/conversation-2/rps15" -> 0 (conversation first)
    - "mooncake/toolagent-2/rps20" -> 1 (toolagent second)
    """
    workload_lower = workload.lower()
    if 'conversation' in workload_lower:
        return 0
    elif 'toolagent' in workload_lower:
        return 1
    else:
        return 99


def get_workload_sort_key(workload):
    """Get hierarchical sort key for a workload.

    Sort order:
    1. Workload category (gangmuk-prefix, mooncake, azure, etc.)
    2. For gangmuk-prefix: SharingRatio (71%, 47%, 28%, 9% - descending order)
       For mooncake: subcategory (conversation-2 first, then toolagent-2)
    3. RPS (4, 5, 6, 7, etc. - ascending order)
    """
    category = extract_workload_category(workload)

    # Category priority: gangmuk-prefix first, then mooncake, then azure, then others
    category_priority = {
        'gangmuk-prefix': 0,
        'mooncake': 1,
        'azure': 2,
    }
    cat_order = category_priority.get(category, 99)

    # Secondary sort key depends on category
    if category == 'mooncake':
        # For mooncake: conversation-2 first, then toolagent-2
        secondary_order = extract_mooncake_subcategory(workload)
    else:
        # For gangmuk-prefix and others: SharingRatio (higher percentages first)
        sharing_ratio = extract_sharing_ratio(workload)
        secondary_order = -sharing_ratio if sharing_ratio < 9999 else 9999

    # RPS: lower first (ascending)
    rps = extract_rps_from_workload(workload)

    return (cat_order, secondary_order, rps, workload)


def extract_datetime_from_name(name):
    """Extract datetime string from strategy name for sorting.

    Looks for patterns like YYYYMMDD_HHMMSS or similar date-time formats.
    Returns a sortable string (empty string if not found, which sorts first).
    """
    # Pattern: YYYYMMDD_HHMMSS or YYYYMMDD-HHMMSS
    match = re.search(r'(\d{8}[_-]\d{6})', name)
    if match:
        return match.group(1)
    # Pattern: YYYY-MM-DD_HH-MM-SS or similar
    match = re.search(r'(\d{4}-\d{2}-\d{2}[_-]\d{2}-\d{2}-\d{2})', name)
    if match:
        return match.group(1)
    # Pattern: just a date YYYYMMDD
    match = re.search(r'(\d{8})', name)
    if match:
        return match.group(1)
    return ""


def get_policy_sort_key(policy):
    """Get sort key for a policy: (priority, datetime, name).

    Priority order: random(0) -> least_request(1) -> prefix_cache(2) -> contextual_bandit(3) -> others(4+)
    Within same priority, sort by datetime (past to recent).
    """
    policy_lower = policy.lower()
    datetime_str = extract_datetime_from_name(policy)

    if 'random' in policy_lower and 'contextual_bandit' not in policy_lower:
        return (0, datetime_str, policy)
    elif 'prefix_hit_threshold_or_least_request' in policy_lower:
        return (1, datetime_str, policy)
    elif 'least_request' in policy_lower:
        return (2, datetime_str, policy)
    elif 'prefix_cache' in policy_lower:
        return (3, datetime_str, policy)
    elif 'contextual_bandit' in policy_lower and 'onlinelearning_0' in policy_lower:
        return (4, datetime_str, policy)
    elif 'contextual_bandit' in policy_lower and 'onlinelearning_1' in policy_lower and 'random' in policy_lower:
        return (5, datetime_str, policy)
    elif 'contextual_bandit' in policy_lower:
        return (6, datetime_str, policy)
    elif 'least_latency' in policy_lower:
        return (7, datetime_str, policy)
    elif 'least_kv_cache' in policy_lower:
        return (8, datetime_str, policy)
    else:
        return (9, datetime_str, policy)


def order_policies(policies):
    """Order policies: random -> least_request -> prefix_cache -> contextual_bandit.

    Within same routing policy category, sort by datetime (past to recent).
    """
    return sorted(policies, key=get_policy_sort_key)


def order_workloads(workloads):
    """Order workloads hierarchically.

    Sort order:
    1. Workload category (gangmuk-prefix, mooncake, azure, etc.)
    2. SharingRatio (71%, 47%, 28%, 9% - descending)
    3. RPS (4, 5, 6, 7, etc. - ascending)
    """
    return sorted(workloads, key=get_workload_sort_key)


def categorize_policy(policy_name):
    """Categorize a policy name into one of the predefined categories."""
    policy_lower = policy_name.lower()

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
    elif 'prefix_hit_threshold_or_least_request' in policy_lower:
        return 'prefix_hit_threshold_or_least_request'
    elif 'least_request' in policy_lower:
        return 'least_request'
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
    category_counts = {}

    for policy in sorted(policies):
        category = categorize_policy(policy)

        if category and category in POLICY_COLOR_FAMILIES:
            if category not in category_counts:
                category_counts[category] = 0
            idx = category_counts[category]
            category_counts[category] += 1

            color_family = POLICY_COLOR_FAMILIES[category]
            colors[policy] = color_family[idx % len(color_family)]
        else:
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
    Extract routing policy from strategy_full_name, preserving the
    onlinelearning_X suffix when present.

    Examples:
    - "contextual_bandit_perpodmodel_checkpoint_negative_linear-iter3-onlinelearning_1-20260202_203318"
      -> "contextual_bandit_perpodmodel_checkpoint_negative_linear-onlinelearning_1"
    - "least_request-iter3-onlinelearning_0-20260202_193433"
      -> "least_request-onlinelearning_0"
    - "random-iter1--20251122_131129" -> "random"
    """
    # Extract onlinelearning_X suffix if present
    ol_match = re.search(r'(onlinelearning_\d+)', strategy_full_name)
    ol_suffix = '-' + ol_match.group(1) if ol_match else ''

    # Pattern: everything before "-iter" or before "-YYYYMMDD_HHMMSS"
    match = re.match(r'^(.+?)-iter\d+', strategy_full_name)
    if match:
        return match.group(1) + ol_suffix

    match = re.match(r'^(.+?)-\d{8}_\d{6}', strategy_full_name)
    if match:
        return match.group(1) + ol_suffix

    return strategy_full_name


def find_metrics_files(base_dir, target_dirs=None):
    """Find routing_strategy_metrics_from_client_log.csv files.

    If target_dirs is provided, only look in those specific directories.
    Otherwise, search recursively under base_dir.
    """
    files = []

    if target_dirs:
        # Only look in specified directories
        for target_dir in target_dirs:
            csv_path = os.path.join(target_dir, "routing_strategy_metrics_from_client_log.csv")
            if os.path.exists(csv_path):
                files.append(csv_path)
            else:
                print(f"Warning: No CSV found in {target_dir}")
    else:
        # Search recursively under base_dir
        pattern = os.path.join(base_dir, "**", "routing_strategy_metrics_from_client_log.csv")
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
    if 'workload' in merged.columns:
        merged['workload'] = merged['workload'].fillna("unknown").astype(str)
    print(f"\nMerged {len(merged)} total rows from {len(dfs)} files")
    return merged


def get_short_experiment_label(strategy_full_name):
    """Create a short label for an experiment from strategy_full_name.

    Uses the same format as the averaged version:
    - "prefix_cache_1-iter4-20260217_012930" -> "prefix_cache_1\n(20260217_012930)"
    """
    parts = strategy_full_name.split('-')
    if len(parts) >= 2:
        return f"{parts[0]}\n({parts[-1]})"
    else:
        return strategy_full_name


def plot_all_metrics_summary(df, output_dir, pdf_pages=None):
    """Create a comprehensive summary plot with one workload per row.

    Each row shows bar chart with avg, p99, p999 for each individual experiment.
    
    Args:
        df: DataFrame with metrics data
        output_dir: Output directory (used only if pdf_pages is None)
        pdf_pages: Optional PdfPages object to save to multi-page PDF
    """
    workloads = order_workloads(df['workload'].unique())
    policies = order_policies(df['routing_policy'].unique())

    n_workloads = len(workloads)

    # Generate colors for each policy category
    policy_colors = generate_policy_colors(policies)

    # Calculate max number of experiments per workload for figure sizing
    max_experiments = 0
    for workload in workloads:
        workload_df = df[df['workload'] == workload]
        max_experiments = max(max_experiments, len(workload_df))

    # Create figure with one row per workload
    # Increased width to accommodate more bars
    fig = plt.figure(figsize=(max(18, max_experiments * 1.5), 7 * n_workloads))
    gs = GridSpec(n_workloads, 1, figure=fig, hspace=0.8)

    for workload_idx, workload in enumerate(workloads):
        ax = fig.add_subplot(gs[workload_idx, 0])

        # Filter data for this workload
        workload_df = df[df['workload'] == workload]

        if len(workload_df) == 0:
            ax.text(0.5, 0.5, f'No data for {workload}', ha='center', va='center')
            continue

        # Sort experiments by policy order, then datetime
        experiments = workload_df['strategy_full_name'].unique().tolist()
        experiments = sorted(experiments, key=get_policy_sort_key)

        # Bar positioning: 3 bars (avg, p99, p999) per experiment
        bar_width = 0.25
        group_width = 3 * bar_width + 0.3
        group_centers = np.arange(len(experiments)) * group_width

        # Collect metrics for each experiment
        avg_values = []
        p99_values = []
        p999_values = []
        num_requests_values = []
        experiment_policies = []  # To get color for each experiment

        for exp in experiments:
            exp_data = workload_df[workload_df['strategy_full_name'] == exp]
            if len(exp_data) > 0:
                avg_values.append(exp_data['avg_ttft'].values[0])
                p99_values.append(exp_data['p99_ttft'].values[0])
                p999_values.append(exp_data['p999_ttft'].values[0])
                # Get num_requests if available
                if 'num_requests' in exp_data.columns:
                    num_requests_values.append(int(exp_data['num_requests'].values[0]))
                else:
                    num_requests_values.append(0)
                # Get routing policy for color
                experiment_policies.append(exp_data['routing_policy'].values[0])
            else:
                avg_values.append(0)
                p99_values.append(0)
                p999_values.append(0)
                num_requests_values.append(0)
                experiment_policies.append('unknown')

        # Get max value for y-axis scaling
        max_value = max(max(avg_values or [0]), max(p99_values or [0]), max(p999_values or [0]))

        # Plot bars for each experiment (avg, p99, p999)
        for i, exp in enumerate(experiments):
            # Get color from the routing policy
            policy = experiment_policies[i]
            strategy_color = policy_colors.get(policy, '#7f7f7f')
            group_center = group_centers[i]

            offset_start = -bar_width

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
                           fontsize=8, fontweight='bold')

        # Set up x-axis with experiment names
        ax.set_xticks(group_centers)

        strategy_labels = [get_short_experiment_label(exp) for exp in experiments]

        ax.set_xticklabels(strategy_labels, fontsize=8, rotation=45, ha='right')

        # Add legend for Avg/P99/P999
        legend_elements = [
            Patch(facecolor='gray', edgecolor='black', alpha=0.9, label='Avg'),
            Patch(facecolor='gray', edgecolor='black', alpha=0.7, label='P99'),
            Patch(facecolor='gray', edgecolor='black', alpha=0.5, label='P999')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12, ncol=3)

        # Add num_requests annotation below each experiment group
        has_num_requests = False
        for i, exp in enumerate(experiments):
            if num_requests_values[i] > 0:
                has_num_requests = True
                group_center = group_centers[i]
                ax.text(group_center, -max_value * 0.12, f'n={num_requests_values[i]}',
                       ha='center', va='top', fontsize=7, style='italic', color='gray')

        # Styling
        ax.set_ylabel('TTFT (ms)', fontsize=ylabel_fontsize)
        ax.set_title(f'TTFT Latency Comparison - {workload}', fontsize=subtitle_fontsize)
        ax.tick_params(axis='y', labelsize=tick_fontsize)
        ax.tick_params(axis='x', labelsize=8)
        ax.grid(axis='y', alpha=0.3)
        # Adjust y-axis limits to accommodate num_requests text
        if has_num_requests:
            ax.set_ylim(-max_value * 0.18, max_value * 1.4)
        else:
            ax.set_ylim(0, max_value * 1.4)

    # Reduced top margin to bring subfigures closer to suptitle
    plt.subplots_adjust(top=0.95, bottom=0.05)
    plt.suptitle(f'TTFT Latency Performance Across All Workloads (Individual Experiments) - From Client Logs',
                 fontsize=maintitle_fontsize, y=0.98)

    if pdf_pages:
        pdf_pages.savefig(fig, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        filename = "all_workloads_summary_from_client_log_individual.pdf"
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved {output_path}")


def plot_metric_comparison(df, output_dir, metric_col, metric_name, ylabel, pdf_pages=None):
    """Create a comparison plot for a specific metric (TTFT, TPOT, or E2E).

    Each experiment is shown as a separate bar (not averaged by policy).
    
    Args:
        df: DataFrame with metrics data
        output_dir: Output directory (used only if pdf_pages is None)
        metric_col: Column name suffix (e.g., 'tpot', 'end_to_end')
        metric_name: Display name for the metric
        ylabel: Y-axis label
        pdf_pages: Optional PdfPages object to save to multi-page PDF
    """
    workloads = order_workloads(df['workload'].unique())
    policies = order_policies(df['routing_policy'].unique())

    n_workloads = len(workloads)

    policy_colors = generate_policy_colors(policies)

    # Calculate max number of experiments per workload for figure sizing
    max_experiments = 0
    for workload in workloads:
        workload_df = df[df['workload'] == workload]
        max_experiments = max(max_experiments, len(workload_df))

    # Increased width to accommodate more bars
    fig = plt.figure(figsize=(max(18, max_experiments * 1.5), 7 * n_workloads))
    gs = GridSpec(n_workloads, 1, figure=fig, hspace=0.8)

    for workload_idx, workload in enumerate(workloads):
        ax = fig.add_subplot(gs[workload_idx, 0])

        workload_df = df[df['workload'] == workload]

        if len(workload_df) == 0:
            ax.text(0.5, 0.5, f'No data for {workload}', ha='center', va='center')
            continue

        # Sort experiments by policy order, then datetime
        experiments = workload_df['strategy_full_name'].unique().tolist()
        experiments = sorted(experiments, key=get_policy_sort_key)

        bar_width = 0.25
        group_width = 3 * bar_width + 0.3
        group_centers = np.arange(len(experiments)) * group_width

        avg_values = []
        p99_values = []
        p999_values = []
        num_requests_values = []
        experiment_policies = []

        for exp in experiments:
            exp_data = workload_df[workload_df['strategy_full_name'] == exp]
            if len(exp_data) > 0:
                avg_values.append(exp_data[f'avg_{metric_col}'].values[0])
                p99_values.append(exp_data[f'p99_{metric_col}'].values[0])
                # P999 might not exist for all metrics
                if f'p999_{metric_col}' in exp_data.columns:
                    p999_values.append(exp_data[f'p999_{metric_col}'].values[0])
                else:
                    p999_values.append(0)
                # Get num_requests if available
                if 'num_requests' in exp_data.columns:
                    num_requests_values.append(int(exp_data['num_requests'].values[0]))
                else:
                    num_requests_values.append(0)
                # Get routing policy for color
                experiment_policies.append(exp_data['routing_policy'].values[0])
            else:
                avg_values.append(0)
                p99_values.append(0)
                p999_values.append(0)
                num_requests_values.append(0)
                experiment_policies.append('unknown')

        max_value = max(max(avg_values or [0]), max(p99_values or [0]), max(p999_values or [0]))

        for i, exp in enumerate(experiments):
            policy = experiment_policies[i]
            strategy_color = policy_colors.get(policy, '#7f7f7f')
            group_center = group_centers[i]
            offset_start = -bar_width

            for j, (value, label, alpha) in enumerate([
                (avg_values[i], 'Avg', 0.9),
                (p99_values[i], 'P99', 0.7),
                (p999_values[i], 'P999', 0.5)
            ]):
                if value > 0:  # Only plot if we have data
                    pos = group_center + offset_start + j * bar_width
                    ax.bar(pos, value, bar_width, color=strategy_color,
                           edgecolor='black', linewidth=0.8, alpha=alpha)

                    ax.text(pos, value + max_value * 0.02,
                           f'{value:.0f}', rotation=90, ha='center', va='bottom',
                           fontsize=8, fontweight='bold')

        ax.set_xticks(group_centers)

        strategy_labels = [get_short_experiment_label(exp) for exp in experiments]

        ax.set_xticklabels(strategy_labels, fontsize=8, rotation=45, ha='right')

        legend_elements = [
            Patch(facecolor='gray', edgecolor='black', alpha=0.9, label='Avg'),
            Patch(facecolor='gray', edgecolor='black', alpha=0.7, label='P99'),
            Patch(facecolor='gray', edgecolor='black', alpha=0.5, label='P999')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12, ncol=3)

        # Add num_requests annotation below each experiment group
        has_num_requests = False
        for i, exp in enumerate(experiments):
            if num_requests_values[i] > 0:
                has_num_requests = True
                group_center = group_centers[i]
                ax.text(group_center, -max_value * 0.12, f'n={num_requests_values[i]}',
                       ha='center', va='top', fontsize=7, style='italic', color='gray')

        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
        ax.set_title(f'{metric_name} Comparison - {workload}', fontsize=subtitle_fontsize)
        ax.tick_params(axis='y', labelsize=tick_fontsize)
        ax.tick_params(axis='x', labelsize=8)
        ax.grid(axis='y', alpha=0.3)
        # Adjust y-axis limits to accommodate num_requests text
        if has_num_requests:
            ax.set_ylim(-max_value * 0.18, max_value * 1.4)
        else:
            ax.set_ylim(0, max_value * 1.4)

    # Reduced top margin to bring subfigures closer to suptitle
    plt.subplots_adjust(top=0.95, bottom=0.05)
    plt.suptitle(f'{metric_name} Performance Across All Workloads (Individual Experiments) - From Client Logs',
                 fontsize=maintitle_fontsize, y=0.98)

    if pdf_pages:
        pdf_pages.savefig(fig, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        filename = f"all_workloads_{metric_col}_from_client_log_individual.pdf"
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Merge and plot routing metrics from client logs across workloads')
    parser.add_argument('base_dir', help='Base directory for output (and recursive search if --target-dirs-file not provided)')
    parser.add_argument('--output-dir', '-o', default=None,
                        help='Output directory for merged CSV and plots (default: base_dir)')
    parser.add_argument('--target-dirs-file', '-t', default=None,
                        help='File containing list of target directories (one per line). If provided, only these directories will be used instead of recursive search.')

    args = parser.parse_args()

    base_dir = args.base_dir
    output_dir = args.output_dir if args.output_dir else base_dir

    # Load target directories from file if provided
    target_dirs = None
    if args.target_dirs_file:
        if os.path.exists(args.target_dirs_file):
            with open(args.target_dirs_file, 'r') as f:
                target_dirs = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            print(f"Using {len(target_dirs)} target directories from {args.target_dirs_file}")
        else:
            print(f"Error: Target dirs file not found: {args.target_dirs_file}")
            sys.exit(1)

    # Find metrics files
    if target_dirs:
        print(f"Looking for routing_strategy_metrics_from_client_log.csv in specified directories...")
    else:
        print(f"Searching recursively for routing_strategy_metrics_from_client_log.csv files in {base_dir}...")

    files = find_metrics_files(base_dir, target_dirs)

    if not files:
        print(f"No routing_strategy_metrics_from_client_log.csv files found")
        sys.exit(1)

    print(f"Found {len(files)} metrics files")

    # Merge files
    df = merge_metrics_files(files)
    if df is None or len(df) == 0:
        print("No data to process")
        sys.exit(1)

    # Re-extract routing_policy from strategy_full_name
    print("\nRe-extracting routing policies from strategy names...")
    df['routing_policy'] = df['strategy_full_name'].apply(extract_routing_policy)

    # Save merged CSV
    os.makedirs(output_dir, exist_ok=True)
    merged_csv_path = os.path.join(output_dir, "merged_routing_metrics_from_client_log.csv")
    df.to_csv(merged_csv_path, index=False)
    print(f"\nSaved merged CSV to {merged_csv_path}")

    # Print summary
    print("\n--- Data Summary ---")
    print(f"Workloads: {sorted(df['workload'].unique().tolist())}")
    print(f"Routing policies: {sorted(df['routing_policy'].unique().tolist())}")
    print(f"Total rows: {len(df)}")

    print("\nExperiments per routing policy:")
    for policy in sorted(df['routing_policy'].unique()):
        count = len(df[df['routing_policy'] == policy])
        print(f"  {policy}: {count}")

    # Create plots
    print("\n--- Generating Plots ---")

    # Create a single PDF file with all plots
    pdf_filename = "all_workloads_from_client_log_individual.pdf"
    pdf_path = os.path.join(output_dir, pdf_filename)
    
    with PdfPages(pdf_path) as pdf:
        # Summary plot (TTFT) - each experiment as separate bar
        plot_all_metrics_summary(df, output_dir, pdf_pages=pdf)

        # Individual metric plots - each experiment as separate bar
        if 'avg_tpot' in df.columns:
            plot_metric_comparison(df, output_dir, 'tpot', 'TPOT', 'TPOT (ms)', pdf_pages=pdf)

        if 'avg_end_to_end' in df.columns:
            plot_metric_comparison(df, output_dir, 'end_to_end', 'End-to-End Latency', 'E2E Latency (ms)', pdf_pages=pdf)

    print(f"Saved combined PDF to {pdf_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
