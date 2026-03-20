#!/usr/bin/env python3
"""
Compare routing strategies using client.log.txt files.
Generates comprehensive comparison plots with CDFs, bar charts, and time series.
"""

import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import argparse
from pathlib import Path
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

# Font sizes
maintitle_fontsize = 30
subtitle_fontsize = 26
legend_fontsize = 22
text_fontsize = 14
ylabel_fontsize = 22
tick_fontsize = 22

# Routing strategy types
rl_naive_routing = "rl_naive"
prefix_cache_1_routing = "prefix_cache_1"
prefix_cache_2_routing = "prefix_cache_2"
preble_routing = "preble"
e2e_latency_predictor_routing = "latency_predictor_e2e_latency"
ttft_latency_predictor_routing = "latency_predictor_ttft"
avg_tpot_latency_predictor_routing = "latency_predictor_avg_tpot"
random_routing = "random"
least_kv_cache_routing = "least_kv_cache"
least_latency_routing = "least_latency"
least_request_routing = "least_request"
contextual_bandit_routing = "contextual_bandit"
prefix_hit_threshold_or_least_request_routing = "prefix_hit_threshold_or_least_request"


INPUT_LENGTH_GROUPS = ['Short (0-1K)', 'Medium (1K-5K)', 'Long (5K+)']
INPUT_LENGTH_COLORS = {
    'Short (0-1K)': '#2ca02c',
    'Medium (1K-5K)': '#1f77b4',
    'Long (5K+)': '#d62728',
}

def categorize_input_length(tokens):
    """Categorize input tokens into short/medium/long groups."""
    if tokens is None or (isinstance(tokens, float) and np.isnan(tokens)):
        return None
    if tokens < 1000:
        return 'Short (0-1K)'
    elif tokens < 5000:
        return 'Medium (1K-5K)'
    else:
        return 'Long (5K+)'

def categorize_strategy(strategy_name):
    """Categorize strategy name into one of the predefined routing types."""
    strategy_lower = strategy_name.lower()
    if rl_naive_routing in strategy_lower:
        return rl_naive_routing
    elif e2e_latency_predictor_routing in strategy_lower:
        return e2e_latency_predictor_routing
    elif ttft_latency_predictor_routing in strategy_lower:
        return ttft_latency_predictor_routing
    elif avg_tpot_latency_predictor_routing in strategy_lower:
        return avg_tpot_latency_predictor_routing
    elif prefix_cache_1_routing in strategy_lower:
        return prefix_cache_1_routing
    elif prefix_cache_2_routing in strategy_lower:
        return prefix_cache_2_routing
    elif preble_routing in strategy_lower:
        return preble_routing
    elif contextual_bandit_routing in strategy_lower:
        return contextual_bandit_routing
    elif random_routing in strategy_lower:
        return random_routing
    elif least_kv_cache_routing in strategy_lower:
        return least_kv_cache_routing
    elif least_latency_routing in strategy_lower:
        return least_latency_routing
    elif prefix_hit_threshold_or_least_request_routing in strategy_lower:
        return prefix_hit_threshold_or_least_request_routing
    elif least_request_routing in strategy_lower:
        return least_request_routing
    else:
        return strategy_name


def parse_strategy_name(filepath):
    """Extract the routing strategy name from the filepath - directory name before the log file."""
    # Get the parent directory name (the one containing client.log.txt)
    parent_dir = os.path.basename(os.path.dirname(filepath))
    return parent_dir


def is_ml_strategy(strategy_name):
    """Return True for ML-based routing policies."""
    strategy_lower = strategy_name.lower()
    ml_markers = ['latency_predictor', 'contextual_bandit', 'rl']
    return any(marker in strategy_lower for marker in ml_markers)


def find_client_log_files(base_dir):
    """Find all client.log.txt files in the base directory - only one level deep."""
    log_files = []
    
    # Search in the base directory itself
    pattern = os.path.join(base_dir, "client.log.txt")
    log_files.extend(glob.glob(pattern))
    
    # Search in immediate subdirectories (one level deep)
    pattern = os.path.join(base_dir, "*", "client.log.txt")
    log_files.extend(glob.glob(pattern))

    # Search two levels deep
    pattern = os.path.join(base_dir, "*", "*", "client.log.txt")
    log_files.extend(glob.glob(pattern))

    return log_files


def parse_log_file(log_file_path):
    """
    Parse client.log.txt to extract metrics.
    
    Returns:
        DataFrame with columns: request_id, ttft, avg_tpot, e2e, iteration, timestamp, start_time, end_time
    """
    data = []
    
    print(f"  Parsing client.log.txt...")
    with open(log_file_path, 'r') as f:
        for line in f:
            # Look for lines with metric information
            if 'TTFT:' in line and 'Avg_tpot:' in line and 'E2E:' in line:
                try:
                    # Parse the line
                    parts = line.split('[Req ')[1].split(']')[0]
                    req_info = parts.split('(')[0]
                    req_id = int(req_info.split('/')[0])
                    
                    iter_part = line.split('iter ')[1].split(':')[0]
                    iteration = int(iter_part.split('/')[0])
                    
                    # Extract metrics
                    ttft_str = line.split('TTFT: ')[1].split('ms')[0]
                    ttft = float(ttft_str)

                    tpot_str = line.split('Avg_tpot: ')[1].split('ms')[0]
                    avg_tpot = float(tpot_str)

                    e2e_str = line.split('E2E: ')[1].split('ms')[0]
                    e2e = float(e2e_str)

                    # Extract input token count
                    input_tokens = None
                    if 'Input: ' in line:
                        try:
                            input_str = line.split('Input: ')[1].split(',')[0]
                            input_tokens = int(input_str)
                        except (ValueError, IndexError):
                            pass

                    # Use sequential counter as timestamp
                    timestamp = len(data)

                    data.append({
                        'request_id': req_id,
                        'ttft': ttft,
                        'avg_tpot': avg_tpot,
                        'e2e': e2e,
                        'iteration': iteration,
                        'timestamp': timestamp,
                        'input_tokens': input_tokens,
                    })
                except Exception as e:
                    continue
    
    df = pd.DataFrame(data)
    
    # Create synthetic time based on order of completion
    if len(df) > 0:
        df = df.sort_values('timestamp').reset_index(drop=True)
        # Assign time assuming roughly even spacing
        df['end_time'] = df.index * 0.05  # Approximate 20 RPS = 0.05s between requests
        df['start_time'] = df['end_time'] - (df['e2e'] / 1000.0)
        df['relative_time'] = df['end_time']  # For compatibility with plotting functions
    
    return df


def calculate_performance_metrics(df, iteration_from=None):
    """Calculate the performance metrics for a dataframe."""
    metrics = {}

    # Calculate TTFT metrics if available
    if 'ttft' in df.columns:
        metrics['avg_ttft'] = df['ttft'].mean()
        metrics['p99_ttft'] = df['ttft'].quantile(0.99)
        metrics['p999_ttft'] = df['ttft'].quantile(0.999)

    # Calculate TPOT metrics if available
    if 'avg_tpot' in df.columns:
        metrics['avg_tpot'] = df['avg_tpot'].mean()
        metrics['p99_tpot'] = df['avg_tpot'].quantile(0.99)
        metrics['p999_tpot'] = df['avg_tpot'].quantile(0.999)

    # Calculate end-to-end latency metrics if available
    if 'e2e' in df.columns:
        metrics['avg_end_to_end'] = df['e2e'].mean()
        metrics['p99_end_to_end'] = df['e2e'].quantile(0.99)
        metrics['p999_end_to_end'] = df['e2e'].quantile(0.999)

    # Calculate throughput
    if len(df) > 0 and 'relative_time' in df.columns:
        total_duration = df['relative_time'].max() - df['relative_time'].min()
        if total_duration > 0:
            metrics['throughput_rps'] = len(df) / total_duration
        else:
            metrics['throughput_rps'] = 0
    
    return metrics


def process_log_file(file_path, warmup_seconds=None, cut_last_seconds=None, iteration_from=None, iteration_upto=None, upto_request=None):
    """Process a single client.log.txt file and return its performance metrics and DataFrame."""
    print(f"Processing {file_path}...")

    # Parse the log file
    df = parse_log_file(file_path)

    if df.empty:
        print(f"  Warning: No data found in {file_path}")
        return None, None

    # Filter out warm-up period and cut last seconds if specified
    if len(df) > 0 and 'relative_time' in df.columns:
        min_time = df['relative_time'].min()
        max_time = df['relative_time'].max()
        original_count = len(df)

        if warmup_seconds is not None:
            df = df[df['relative_time'] >= min_time + warmup_seconds]
        if cut_last_seconds is not None:
            df = df[df['relative_time'] <= max_time - cut_last_seconds]

        filtered_count = len(df)
        if original_count != filtered_count:
            print(f"  Filtered: {original_count} -> {filtered_count} requests")

    # Extract strategy name from the file path
    strategy = parse_strategy_name(file_path)

    # Apply iteration-based filtering only for ML-based policies
    if is_ml_strategy(strategy) and 'iteration' in df.columns:
        df['iteration'] = pd.to_numeric(df['iteration'], errors='coerce')
        before_count = len(df)

        # Apply iteration_from filter
        if iteration_from is not None and iteration_from > 0:
            filtered_df = df[df['iteration'] >= iteration_from]
            if filtered_df.empty and not df.empty:
                # Fallback: use the max available iteration to avoid empty plots
                max_iter = df['iteration'].max()
                print(f"  Warning: no rows with iteration >= {iteration_from} (max iteration = {max_iter}). Falling back to iteration == {max_iter}")
                df = df[df['iteration'] == max_iter]
            else:
                df = filtered_df

        # Apply iteration_upto filter
        if iteration_upto is not None:
            filtered_df = df[df['iteration'] <= iteration_upto]
            if filtered_df.empty and not df.empty:
                # Fallback: use the min available iteration to avoid empty plots
                min_iter = df['iteration'].min()
                print(f"  Warning: no rows with iteration <= {iteration_upto} (min iteration = {min_iter}). Falling back to iteration == {min_iter}")
                df = df[df['iteration'] == min_iter]
            else:
                df = filtered_df

        after_count = len(df)
        if before_count != after_count:
            filter_desc = []
            if iteration_from is not None and iteration_from > 0:
                filter_desc.append(f">= {iteration_from}")
            if iteration_upto is not None:
                filter_desc.append(f"<= {iteration_upto}")
            print(f"  Iteration filter: {before_count} -> {after_count} rows (iteration {' and '.join(filter_desc)})")

    # Limit to the first N requests by arrival order if requested
    if upto_request is not None and len(df) > upto_request:
        df = df.iloc[:upto_request]
        print(f"  Truncated to first {upto_request} requests (--upto_request)")

    # Calculate performance metrics
    metrics = calculate_performance_metrics(df, iteration_from)
    metrics['strategy'] = strategy
    metrics['file_path'] = file_path
    metrics['num_requests'] = len(df)
    
    print(f"  Strategy: {strategy}, Requests: {len(df)}")
    print(f"  Metrics: avg_ttft={metrics.get('avg_ttft', 0):.2f}ms, "
          f"avg_tpot={metrics.get('avg_tpot', 0):.2f}ms")
    
    return metrics, df


def extract_datetime_from_strategy(strategy_name):
    """Extract datetime string from strategy name for sorting.

    Looks for patterns like YYYYMMDD_HHMMSS or similar date-time formats.
    Returns a sortable string (empty string if not found, which sorts first).
    """
    # Common datetime patterns in strategy names
    import re
    # Pattern: YYYYMMDD_HHMMSS or YYYYMMDD-HHMMSS
    match = re.search(r'(\d{8}[_-]\d{6})', strategy_name)
    if match:
        return match.group(1)
    # Pattern: YYYY-MM-DD_HH-MM-SS or similar
    match = re.search(r'(\d{4}-\d{2}-\d{2}[_-]\d{2}-\d{2}-\d{2})', strategy_name)
    if match:
        return match.group(1)
    # Pattern: just a date YYYYMMDD
    match = re.search(r'(\d{8})', strategy_name)
    if match:
        return match.group(1)
    return ""


def get_strategy_priority(strategy_name):
    """Define strategy ordering for consistent plots.

    Order: random -> least_request -> prefix_cache -> contextual_bandit
    Within same routing policy, sort by datetime (past to recent).
    """
    strategy_lower = strategy_name.lower()
    datetime_str = extract_datetime_from_strategy(strategy_name)

    # Primary order: random(0) -> least_request(1) -> prefix_cache(2) -> contextual_bandit(3) -> others(4+)
    if contextual_bandit_routing in strategy_lower:
        if 'random' in strategy_lower:
            return (7, 0, datetime_str, strategy_name)  # random init first
        elif 'onlinelearning' in strategy_lower:
            return (7, 2, datetime_str, strategy_name)  # online learning last
        else:
            return (7, 1, datetime_str, strategy_name)  # no online learning middle
    elif random_routing in strategy_lower:
        return (0, datetime_str, strategy_name)
    elif least_request_routing in strategy_lower:
        return (1, datetime_str, strategy_name)
    elif least_kv_cache_routing in strategy_lower:
        return (2, datetime_str, strategy_name)
    elif least_latency_routing in strategy_lower:
        return (3, datetime_str, strategy_name)
    elif prefix_cache_1_routing in strategy_lower:
        return (4, datetime_str, strategy_name)
    elif prefix_cache_2_routing in strategy_lower:
        return (5, datetime_str, strategy_name)
    elif preble_routing in strategy_lower:
        return (6, datetime_str, strategy_name)
    elif rl_naive_routing in strategy_lower:
        return (8, datetime_str, strategy_name)
    elif e2e_latency_predictor_routing in strategy_lower:
        return (9, datetime_str, strategy_name)
    elif ttft_latency_predictor_routing in strategy_lower:
        return (10, datetime_str, strategy_name)
    elif avg_tpot_latency_predictor_routing in strategy_lower:
        return (11, datetime_str, strategy_name)
    else:
        return (12, datetime_str, strategy_name)


def get_strategy_color(strategy_name, index_in_category):
    """Get color for strategy based on category."""
    if rl_naive_routing in strategy_name.lower():
        base_colors = ['#4169e1', '#483d8b', '#6a5acd', '#7b68ee', '#9370db']
    elif e2e_latency_predictor_routing in strategy_name.lower():
        base_colors = ['#8b008b','#ba55d3', '#9932cc', '#8a2be2', '#c71585']
    elif ttft_latency_predictor_routing in strategy_name.lower():
        base_colors = ['#ff1493', '#ff69b4', '#dc143c', '#ff00ff', '#da70d6']
    elif avg_tpot_latency_predictor_routing in strategy_name.lower():
        base_colors = ['#8b0000', '#b22222', '#cd5c5c', '#f08080', '#fa8072']
    elif prefix_cache_1_routing in strategy_name.lower():
        base_colors = ['#1f77b4', '#4682b4', '#6495ed', '#aec7e8', '#87ceeb']
    elif prefix_cache_2_routing in strategy_name.lower():
        base_colors = ['#006400', '#228b22', '#32cd32', '#00ff00', '#7cfc00']
    elif preble_routing in strategy_name.lower():
        base_colors = ['#ff8c00', '#ffa500', '#ffd700', '#ff6347', '#ff4500']
    elif contextual_bandit_routing in strategy_name.lower():
        base_colors = ['#ff0000', '#dc143c', '#ff6347', '#ff4500', '#ff7f50']
    elif random_routing in strategy_name.lower():
        base_colors = ['#2ca02c', '#32cd32', '#00ff00', '#00ff7f', '#98df8a']
    elif least_kv_cache_routing in strategy_name.lower():
        base_colors = ['#d2691e', '#cd853f', '#daa520', '#b8860b', '#f4a460']
    elif least_latency_routing in strategy_name.lower():
        base_colors = ['#483d8b', '#6a5acd', '#7b68ee', '#9370db', '#8470ff']
    elif prefix_hit_threshold_or_least_request_routing in strategy_name.lower():
        base_colors = ['#556b2f', '#6b8e23', '#808000', '#9acd32', '#bdb76b']  # Olive/green-yellow family
    elif least_request_routing in strategy_name.lower():
        base_colors = ['#008b8b', '#20b2aa', '#48d1cc', '#40e0d0', '#00ced1']
    else:
        base_colors = ['#7f7f7f', '#696969', '#a9a9a9', '#c7c7c7', '#d3d3d3']
    
    return base_colors[index_in_category % len(base_colors)]


def plot_latency_cdf(ax, csv_data_dict, strategy_order, color_dict, column, title, xlabel):
    """Plot CDF for a given latency column for each strategy."""
    for strategy in strategy_order:
        if strategy in csv_data_dict and column in csv_data_dict[strategy].columns:
            data = csv_data_dict[strategy][column].dropna().sort_values()
            if len(data) == 0:
                continue
            y = np.linspace(0, 1, len(data))
            # Shorten strategy name for legend
            legend_label = strategy.split('-')[0]
            ax.plot(data, y, label=legend_label, color=color_dict[strategy], linewidth=2, alpha=0.8)
    
    ax.set_title(title, fontsize=subtitle_fontsize)
    ax.set_xlabel(xlabel, fontsize=ylabel_fontsize)
    ax.set_ylabel('CDF', fontsize=ylabel_fontsize)
    ax.legend(fontsize=16, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=tick_fontsize)


def plot_latency_timeseries(ax, csv_data_dict, strategy_order, color_dict, column, title, ylabel):
    """Plot time series with 1-second window averages for a given latency column."""
    for strategy in strategy_order:
        if strategy in csv_data_dict and column in csv_data_dict[strategy].columns:
            df = csv_data_dict[strategy]
            
            # Create 1-second time bins
            df_copy = df.copy()
            df_copy['time_bin'] = np.floor(df_copy['relative_time']).astype(int)
            
            # Calculate average latency for each 1-second bin
            latency_stats = df_copy.groupby('time_bin')[column].agg(['mean', 'count']).reset_index()
            latency_stats = latency_stats[latency_stats['count'] > 0]
            
            # Shorten strategy name for legend
            legend_label = strategy.split('-')[0]
            
            # Plot the time series
            ax.plot(latency_stats['time_bin'], latency_stats['mean'], 
                   color=color_dict[strategy], linewidth=2, alpha=0.8, label=legend_label)
    
    ax.set_title(title, fontsize=subtitle_fontsize)
    ax.set_xlabel('Time (seconds)', fontsize=ylabel_fontsize)
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    ax.legend(fontsize=16, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=tick_fontsize)


def plot_single_metric_comparison(ax, metrics_df, strategy_order, color_dict, metric_type, title):
    """Plot bar chart with single metric: avg on left y-axis, p99/p999 on right y-axis."""
    strategies = [s for s in strategy_order if s in metrics_df['strategy'].values]
    n_strategies = len(strategies)
    if n_strategies == 0:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax.set_xticks([])
        ax.set_yticks([])
        return

    metrics_indexed = metrics_df.set_index('strategy')

    # Get the appropriate metrics based on metric_type
    if metric_type == 'ttft':
        avg_values = [metrics_indexed.loc[s, 'avg_ttft'] if 'avg_ttft' in metrics_df.columns else 0 for s in strategies]
        p99_values = [metrics_indexed.loc[s, 'p99_ttft'] if 'p99_ttft' in metrics_df.columns else 0 for s in strategies]
        p999_values = [metrics_indexed.loc[s, 'p999_ttft'] if 'p999_ttft' in metrics_df.columns else 0 for s in strategies]
        ylabel_text = 'TTFT (ms)'
    elif metric_type == 'tpot':
        avg_values = [metrics_indexed.loc[s, 'avg_tpot'] if 'avg_tpot' in metrics_df.columns else 0 for s in strategies]
        p99_values = [metrics_indexed.loc[s, 'p99_tpot'] if 'p99_tpot' in metrics_df.columns else 0 for s in strategies]
        p999_values = [metrics_indexed.loc[s, 'p999_tpot'] if 'p999_tpot' in metrics_df.columns else 0 for s in strategies]
        ylabel_text = 'Avg TPOT (ms)'
    else:  # e2e
        avg_values = [metrics_indexed.loc[s, 'avg_end_to_end'] if 'avg_end_to_end' in metrics_df.columns else 0 for s in strategies]
        p99_values = [metrics_indexed.loc[s, 'p99_end_to_end'] if 'p99_end_to_end' in metrics_df.columns else 0 for s in strategies]
        p999_values = [metrics_indexed.loc[s, 'p999_end_to_end'] if 'p999_end_to_end' in metrics_df.columns else 0 for s in strategies]
        ylabel_text = 'End-to-End Latency (ms)'

    # Compute max values for each y-axis
    avg_arr = np.array(avg_values, dtype=float)
    avg_finite = avg_arr[np.isfinite(avg_arr)]
    tail_arr = np.array(p99_values + p999_values, dtype=float)
    tail_finite = tail_arr[np.isfinite(tail_arr)]

    if avg_finite.size == 0 and tail_finite.size == 0:
        ax.text(0.5, 0.5, 'No valid data available', ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax.set_xticks([])
        ax.set_yticks([])
        return

    max_avg = float(np.max(avg_finite)) if avg_finite.size > 0 else 1.0
    max_tail = float(np.max(tail_finite)) if tail_finite.size > 0 else 1.0

    # Create right y-axis for P99/P999
    ax2 = ax.twinx()

    # Create bar positions
    num_bars = 3
    bar_width = 0.25
    group_width = num_bars * bar_width + 0.3
    group_centers = np.arange(n_strategies) * group_width

    # Plot bars for each metric
    for i, strategy in enumerate(strategies):
        strategy_color = color_dict[strategy]
        group_center = group_centers[i]

        offset_start = -(num_bars - 1) * bar_width / 2

        # Use hatch patterns
        sl = strategy.lower()
        hatch_pattern = 'xx' if 'onlinelearning_0' in sl else ('//' if contextual_bandit_routing in sl and 'random' in sl else None)

        # Avg on left axis
        pos = group_center + offset_start + 0 * bar_width
        if avg_values[i] > 0:
            ax.bar(pos, avg_values[i], bar_width, color=strategy_color,
                   edgecolor='black', linewidth=0.8, alpha=0.9, hatch=hatch_pattern)
            if np.isfinite(avg_values[i]):
                ax.text(pos, avg_values[i] + max_avg * 0.02,
                       f'{avg_values[i]:.0f}', rotation=90, ha='center', va='bottom',
                       fontsize=10, fontweight='bold', color='#222266')

        # P99, P999 on right axis
        right_bars = [
            (p99_values[i], 'P99', 0.7, 1),
            (p999_values[i], 'P999', 0.5, 2),
        ]
        for value, label, alpha, j in right_bars:
            pos = group_center + offset_start + j * bar_width
            if value > 0:
                ax2.bar(pos, value, bar_width, color=strategy_color,
                       edgecolor='black', linewidth=0.8, alpha=alpha, hatch=hatch_pattern)
                if np.isfinite(value):
                    ax2.text(pos, value + max_tail * 0.02,
                            f'{value:.0f}', rotation=90, ha='center', va='bottom',
                            fontsize=10, fontweight='bold', color='#662222')

    # Set up x-axis with strategy names
    ax.set_xticks(group_centers)

    strategy_labels = []
    for s in strategies:
        parts = s.split('-')
        if len(parts) >= 2:
            label = f"{parts[0]}\n({parts[-1]})"
        else:
            label = s
        strategy_labels.append(label)

    ax.set_xticklabels(strategy_labels, fontsize=10, rotation=45, ha='right')

    # Add legend for Avg/P99/P999 with axis indication
    legend_elements = [
        Patch(facecolor='gray', edgecolor='black', alpha=0.9, label='Avg (left)'),
        Patch(facecolor='gray', edgecolor='black', alpha=0.7, label='P99 (right)'),
        Patch(facecolor='gray', edgecolor='black', alpha=0.5, label='P999 (right)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=14, ncol=3)

    # Add num_requests annotation below each strategy group
    has_num_requests = False
    for i, strategy in enumerate(strategies):
        if 'num_requests' in metrics_df.columns:
            num_requests = metrics_indexed.loc[strategy, 'num_requests'] if 'num_requests' in metrics_indexed.columns else 0
            if pd.notna(num_requests) and num_requests > 0:
                has_num_requests = True
                group_center = group_centers[i]
                ax.text(group_center, -max_avg * 0.15, f'n={int(num_requests)}',
                       ha='center', va='top', fontsize=9, style='italic', color='gray')

    # Styling
    ax.set_ylabel(f'{ylabel_text} — Avg', fontsize=ylabel_fontsize, color='#222266')
    ax2.set_ylabel(f'{ylabel_text} — P99/P999', fontsize=ylabel_fontsize, color='#662222')
    ax.set_title(title, fontsize=subtitle_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize, labelcolor='#222266')
    ax2.tick_params(axis='y', labelsize=tick_fontsize, labelcolor='#662222')
    ax.tick_params(axis='x', labelsize=10)
    ax.grid(axis='y', alpha=0.3)
    # Set y-axis limits (both start from 0)
    ax.set_ylim(0, max(max_avg * 1.6, 1.0))
    ax2.set_ylim(0, max(max_tail * 1.6, 1.0))


def plot_metric_by_token_range(ax, csv_data_dict, strategy_order, color_dict, metric_column, title, ylabel_text):
    """Plot bar chart with avg/p99/p999 for each input token range, grouped by strategy."""
    strategies = [s for s in strategy_order if s in csv_data_dict]
    n_strategies = len(strategies)
    if n_strategies == 0:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax.set_xticks([])
        ax.set_yticks([])
        return

    input_col = 'input_tokens'
    # Check if the column exists in any strategy
    has_input = any(input_col in csv_data_dict[s].columns and csv_data_dict[s][input_col].notna().any()
                    for s in strategies)
    if not has_input:
        ax.text(0.5, 0.5, 'No input token data available', ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax.set_xticks([])
        ax.set_yticks([])
        return

    groups = INPUT_LENGTH_GROUPS
    n_groups = len(groups)
    stat_labels = ['Avg', 'P99', 'P999']
    n_stats = len(stat_labels)

    # Compute stats per strategy per group (including 'All' aggregated)
    all_groups = groups + ['All']
    all_bar_values = []
    for strategy in strategies:
        df = csv_data_dict[strategy]
        if metric_column not in df.columns or input_col not in df.columns:
            continue
        df_tmp = df[[metric_column, input_col]].dropna()
        df_tmp = df_tmp.copy()
        df_tmp['_group'] = df_tmp[input_col].apply(categorize_input_length)
        for group in groups:
            gdf = df_tmp[df_tmp['_group'] == group][metric_column]
            if len(gdf) == 0:
                all_bar_values.append((strategy, group, 'Avg', 0))
                all_bar_values.append((strategy, group, 'P99', 0))
                all_bar_values.append((strategy, group, 'P999', 0))
            else:
                all_bar_values.append((strategy, group, 'Avg', gdf.mean()))
                all_bar_values.append((strategy, group, 'P99', gdf.quantile(0.99)))
                all_bar_values.append((strategy, group, 'P999', gdf.quantile(0.999)))
        # Aggregated (All) stats
        all_data = df_tmp[metric_column]
        if len(all_data) == 0:
            all_bar_values.append((strategy, 'All', 'Avg', 0))
            all_bar_values.append((strategy, 'All', 'P99', 0))
            all_bar_values.append((strategy, 'All', 'P999', 0))
        else:
            all_bar_values.append((strategy, 'All', 'Avg', all_data.mean()))
            all_bar_values.append((strategy, 'All', 'P99', all_data.quantile(0.99)))
            all_bar_values.append((strategy, 'All', 'P999', all_data.quantile(0.999)))

    if not all_bar_values:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    n_all_groups = len(all_groups)
    bar_width = 0.15
    sub_group_width = n_stats * bar_width + 0.05
    group_block_width = n_all_groups * sub_group_width + 0.4
    strategy_centers = np.arange(n_strategies) * group_block_width

    stat_alphas = {'Avg': 0.9, 'P99': 0.7, 'P999': 0.5}
    all_group_colors = dict(INPUT_LENGTH_COLORS)
    all_group_colors['All'] = '#555555'

    # Compute separate max values for left (Avg) and right (P99/P999) axes
    avg_vals = [v[3] for v in all_bar_values if v[2] == 'Avg' and np.isfinite(v[3])]
    tail_vals = [v[3] for v in all_bar_values if v[2] in ('P99', 'P999') and np.isfinite(v[3])]
    max_avg = max(avg_vals, default=1)
    max_tail = max(tail_vals, default=1)

    # Create right y-axis for P99/P999
    ax2 = ax.twinx()

    for si, strategy in enumerate(strategies):
        base_x = strategy_centers[si] - (n_all_groups * sub_group_width) / 2

        for gi, group in enumerate(all_groups):
            group_color = all_group_colors[group]
            sub_base = base_x + gi * sub_group_width

            # Use hatch patterns: '///' for onlinelearning_0, 'xx' for contextual_bandit with random init
            sl = strategy.lower()
            hatch_pattern = 'xx' if 'onlinelearning_0' in sl else ('//' if contextual_bandit_routing in sl and 'random' in sl else None)

            for stat_idx, stat_label in enumerate(stat_labels):
                val = 0
                for v in all_bar_values:
                    if v[0] == strategy and v[1] == group and v[2] == stat_label:
                        val = v[3]
                        break
                pos = sub_base + stat_idx * bar_width
                alpha = stat_alphas[stat_label]

                # Avg on left axis, P99/P999 on right axis
                if stat_label == 'Avg':
                    ax.bar(pos, val, bar_width, color=group_color, edgecolor='black',
                           linewidth=0.5, alpha=alpha, hatch=hatch_pattern)
                    if np.isfinite(val) and val > 0:
                        ax.text(pos, val + max_avg * 0.01, f'{val:.0f}', rotation=90,
                                ha='center', va='bottom', fontsize=7, fontweight='bold', color='#222266')
                else:
                    ax2.bar(pos, val, bar_width, color=group_color, edgecolor='black',
                            linewidth=0.5, alpha=alpha, hatch=hatch_pattern)
                    if np.isfinite(val) and val > 0:
                        ax2.text(pos, val + max_tail * 0.01, f'{val:.0f}', rotation=90,
                                 ha='center', va='bottom', fontsize=7, fontweight='bold', color='#662222')

    # Draw vertical separator lines between experiments
    for si in range(n_strategies - 1):
        sep_x = (strategy_centers[si] + strategy_centers[si + 1]) / 2
        ax.axvline(x=sep_x, color='gray', linestyle='--', linewidth=1.0, alpha=0.6)

    ax.set_xticks(strategy_centers)
    strategy_labels = []
    for s in strategies:
        parts = s.split('-')
        if len(parts) >= 2:
            label = f"{parts[0]}\n({parts[-1]})"
        else:
            label = s
        strategy_labels.append(label)
    ax.set_xticklabels(strategy_labels, fontsize=10, rotation=45, ha='right')

    # Add count annotations
    for si, strategy in enumerate(strategies):
        df = csv_data_dict[strategy]
        if input_col in df.columns:
            df_tmp = df[[input_col]].dropna().copy()
            df_tmp['_group'] = df_tmp[input_col].apply(categorize_input_length)
            counts = []
            for group in groups:
                n = (df_tmp['_group'] == group).sum()
                counts.append(f"{group.split('(')[0].strip()[0]}:{n}")
            total_n = len(df_tmp)
            counts.append(f"All:{total_n}")
            ax.text(strategy_centers[si], -max_avg * 0.12,
                    ', '.join(counts), ha='center', va='top', fontsize=7, color='gray')

    group_legend = [Patch(facecolor=all_group_colors[g], edgecolor='black', alpha=0.8,
                          label=g) for g in groups]  # Exclude 'All' gray patch
    ax.legend(handles=group_legend, loc='upper left', fontsize=10, ncol=len(group_legend))

    ax.set_ylabel(f'{ylabel_text} — Avg', fontsize=ylabel_fontsize, color='#222266')
    ax2.set_ylabel(f'{ylabel_text} — P99/P999', fontsize=ylabel_fontsize, color='#662222')
    ax.set_title(title, fontsize=subtitle_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize, labelcolor='#222266')
    ax2.tick_params(axis='y', labelsize=tick_fontsize, labelcolor='#662222')
    ax.tick_params(axis='x', labelsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(max_avg * 1.6, 1.0))
    ax2.set_ylim(0, max(max_tail * 1.6, 1.0))


def plot_routing_comparison(metrics_list, base_dir, csv_data_dict=None):
    """Create comparison plots across routing strategies."""
    if not metrics_list:
        print("No metrics to plot.")
        return
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics_list)
    
    # Sort strategies by priority
    all_strategies = metrics_df['strategy'].tolist()
    strategy_order = sorted(all_strategies, key=get_strategy_priority)

    # Create color dictionary
    color_dict = {}
    category_counts = {
        rl_naive_routing: 0, prefix_cache_1_routing: 0, prefix_cache_2_routing: 0,
        preble_routing: 0, e2e_latency_predictor_routing: 0, ttft_latency_predictor_routing: 0,
        avg_tpot_latency_predictor_routing: 0, random_routing: 0, least_kv_cache_routing: 0,
        least_latency_routing: 0, least_request_routing: 0,
        contextual_bandit_routing: 0,
        prefix_hit_threshold_or_least_request_routing: 0,
        'other': 0
    }

    for strategy in strategy_order:
        category = categorize_strategy(strategy)
        if category in category_counts:
            color_dict[strategy] = get_strategy_color(strategy, category_counts[category])
            category_counts[category] += 1
        else:
            color_dict[strategy] = get_strategy_color(strategy, category_counts['other'])
            category_counts['other'] += 1
    
    # Create figure with GridSpec
    # 8 rows: CDFs, TTFT bar, TTFT by token range, TPOT bar, TPOT by token range, E2E bar, time series x2
    fig = plt.figure(figsize=(18, 48))
    gs = GridSpec(8, 9, figure=fig,
                  height_ratios=[1, 1.5, 1.5, 1.5, 1.5, 1.5, 1, 1],
                  hspace=1.0,
                  wspace=0.35)
    
    # Plot components if we have CSV data
    if csv_data_dict:
        # Row 0: CDFs
        ax = fig.add_subplot(gs[0, :4])
        plot_latency_cdf(ax, csv_data_dict, strategy_order, color_dict, 'ttft', 'TTFT Latency CDF', 'TTFT (ms)')

        ax = fig.add_subplot(gs[0, 5:])
        plot_latency_cdf(ax, csv_data_dict, strategy_order, color_dict, 'avg_tpot', 'Avg TPOT Latency CDF', 'Avg TPOT (ms)')

        # Row 1: TTFT bar chart
        ax = fig.add_subplot(gs[1, :])
        plot_single_metric_comparison(ax, metrics_df, strategy_order, color_dict, 'ttft', 'TTFT Latency Comparison (Avg, P99, P999)')

        # Row 2: TTFT by input token range
        ax = fig.add_subplot(gs[2, :])
        plot_metric_by_token_range(ax, csv_data_dict, strategy_order, color_dict, 'ttft',
                                   'TTFT by Input Token Range (Avg, P99, P999)', 'TTFT (ms)')

        # Row 3: TPOT bar chart
        ax = fig.add_subplot(gs[3, :])
        plot_single_metric_comparison(ax, metrics_df, strategy_order, color_dict, 'tpot', 'Avg TPOT Latency Comparison (Avg, P99, P999)')

        # Row 4: TPOT by input token range
        ax = fig.add_subplot(gs[4, :])
        plot_metric_by_token_range(ax, csv_data_dict, strategy_order, color_dict, 'avg_tpot',
                                   'Avg TPOT by Input Token Range (Avg, P99, P999)', 'Avg TPOT (ms)')

        # Row 5: E2E bar chart
        ax = fig.add_subplot(gs[5, :])
        plot_single_metric_comparison(ax, metrics_df, strategy_order, color_dict, 'e2e', 'End-to-End Latency Comparison (Avg, P99)')

        # Row 6-7: Time series
        ax = fig.add_subplot(gs[6, :])
        plot_latency_timeseries(ax, csv_data_dict, strategy_order, color_dict, 'ttft', 'TTFT Time Series (1s averages)', 'TTFT (ms)')

        ax = fig.add_subplot(gs[7, :])
        plot_latency_timeseries(ax, csv_data_dict, strategy_order, color_dict, 'avg_tpot', 'Avg TPOT Time Series (1s averages)', 'Avg TPOT (ms)')
    else:
        # Placeholder if no CSV data
        for row_idx, plot_cols in [(0, [slice(None, 4), slice(5, None)]),
                                   (1, [slice(None)]), (2, [slice(None)]),
                                   (3, [slice(None)]), (4, [slice(None)]), (5, [slice(None)]),
                                   (6, [slice(None)]), (7, [slice(None)])]:
            if row_idx == 0:
                for col_slice in plot_cols:
                    ax = fig.add_subplot(gs[row_idx, col_slice])
                    ax.text(0.5, 0.5, 'No data available',
                           ha='center', va='center', fontsize=12,
                           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                    ax.set_xticks([])
                    ax.set_yticks([])
            else:
                ax = fig.add_subplot(gs[row_idx, :])
                ax.text(0.5, 0.5, 'No data available', 
                       ha='center', va='center', fontsize=12, 
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                ax.set_xticks([])
                ax.set_yticks([])
    
    # Reduced top margin to bring subfigures closer to top; increased bottom for labels
    plt.subplots_adjust(top=0.98, bottom=0.06, left=0.05, right=0.95)
    
    # Save the figure
    output_file = f"{base_dir}/routing_strategy_comparison_client.pdf"
    # plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.savefig(output_file, bbox_inches='tight', dpi=600)
    print(f"** Saved comparison plot to {output_file}")
    plt.close()


def export_metrics_to_csv(all_metrics, base_dir):
    """Export performance metrics to a CSV file."""
    if not all_metrics:
        print("No metrics to export.")
        return None

    # Extract workload identifier from base_dir
    # The directory tree may optionally contain a model name level between the GPU type
    # and the output-token config, e.g.:
    #   NVIDIA-A30/llama-3-8b-instruct/maxTokens_1-maxTokensStd_0/gangmuk-prefix/SharingRatio9%/rps5-benchmark/...
    #   NVIDIA-A10/maxTokens_1-maxTokensStd_0/gangmuk-prefix/SharingRatio71%/rps7-benchmark/...
    # We include the full relative path so the workload identifier is unambiguous.
    workload = ""
    if "workload-and-experiment_results" in base_dir:
        parts = base_dir.split("workload-and-experiment_results")
        if len(parts) > 1:
            full_path = parts[1].lstrip("/")
            workload = full_path

    # Define columns
    csv_data = []
    for metrics in all_metrics:
        row = {
            'workload': workload,
            'routing_policy': categorize_strategy(metrics['strategy']),
            'strategy_full_name': metrics['strategy'],
            'num_requests': metrics.get('num_requests', 0),
            'avg_ttft': metrics.get('avg_ttft', ''),
            'p99_ttft': metrics.get('p99_ttft', ''),
            'p999_ttft': metrics.get('p999_ttft', ''),
            'avg_tpot': metrics.get('avg_tpot', ''),
            'p99_tpot': metrics.get('p99_tpot', ''),
            'p999_tpot': metrics.get('p999_tpot', ''),
            'avg_end_to_end': metrics.get('avg_end_to_end', ''),
            'p99_end_to_end': metrics.get('p99_end_to_end', ''),
            'p999_end_to_end': metrics.get('p999_end_to_end', ''),
            'throughput_rps': metrics.get('throughput_rps', ''),
        }
        csv_data.append(row)

    # Save to CSV
    csv_filepath = os.path.join(base_dir, "routing_strategy_metrics_from_client_log.csv")
    df_export = pd.DataFrame(csv_data)
    df_export.to_csv(csv_filepath, index=False)
    print(f"** Saved metrics CSV to {csv_filepath}")
    return csv_filepath


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare routing strategies using client.log.txt files')
    parser.add_argument('base_directory', help='Base directory containing subdirectories with client.log.txt files')
    parser.add_argument('warmup_seconds', nargs='?', type=int, default=None,
                       help='Seconds to exclude from start for warmup')
    parser.add_argument('cut_last_seconds', nargs='?', type=int, default=None,
                       help='Seconds to exclude from end')
    parser.add_argument('--iteration-from', type=int, default=0,
                       help='Only include rows with iteration >= this value for ML policies')
    parser.add_argument('--iteration-upto', type=int, default=None,
                       help='Only include rows with iteration <= this value for ML policies')
    parser.add_argument('--upto_request', type=int, default=None,
                       help='If given, only plot the first N requests (by arrival order) from each log file')

    args = parser.parse_args()

    base_dir = args.base_directory
    warmup_seconds = args.warmup_seconds
    cut_last_seconds = args.cut_last_seconds
    iteration_from = args.iteration_from
    iteration_upto = args.iteration_upto
    upto_request = args.upto_request

    # Validate iteration range
    if iteration_upto is not None and iteration_from > 0 and iteration_upto < iteration_from:
        print(f"Error: --iteration-upto ({iteration_upto}) cannot be smaller than --iteration-from ({iteration_from})")
        sys.exit(1)

    print(f"Searching for client.log.txt files in {base_dir}...")
    if warmup_seconds is not None:
        print(f"Warmup seconds: {warmup_seconds}")
    if cut_last_seconds is not None:
        print(f"Cut last seconds: {cut_last_seconds}")
    if iteration_from > 0:
        print(f"Iteration filter (ML policies only): >= {iteration_from}")
    if iteration_upto is not None:
        print(f"Iteration filter (ML policies only): <= {iteration_upto}")
    if upto_request is not None:
        print(f"upto_request: plotting only the first {upto_request} requests")
    
    log_files = find_client_log_files(base_dir)
    print(f"Found {len(log_files)} client.log.txt files.\n")
    
    if not log_files:
        print(f"No client.log.txt files found in {base_dir}")
        sys.exit(1)
    
    # Process each log file
    all_metrics = []
    csv_data_dict = {}
    
    for log_file in log_files:
        metrics, df = process_log_file(log_file, warmup_seconds, cut_last_seconds, iteration_from, iteration_upto, upto_request)
        if metrics and df is not None and not df.empty:
            all_metrics.append(metrics)
            csv_data_dict[metrics['strategy']] = df
    
    if not all_metrics:
        print("No valid metrics found!")
        sys.exit(1)
    
    print(f"\nSuccessfully processed {len(all_metrics)} strategies:")
    for m in all_metrics:
        print(f"  - {m['strategy']}: {m['num_requests']} requests")
    
    # Export metrics to CSV
    export_metrics_to_csv(all_metrics, base_dir)
    
    # Create comparison plot
    print("\nGenerating comparison plot...")
    plot_routing_comparison(all_metrics, base_dir, csv_data_dict)
    