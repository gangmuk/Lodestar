import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import json
from datetime import datetime
import logging
import argparse
import csv
# import training.preprocess as preprocess
import preprocess
from matplotlib.gridspec import GridSpec

maintitle_fontsize = 30
subtitle_fontsize = 26
legend_fontsize = 22
text_fontsize = 14
ylabel_fontsize = 22
tick_fontsize = 22

rl_naive_routing="rl_naive" # none
prefix_cache_1_routing="prefix_cache_1"
prefix_cache_2_routing="prefix_cache_2"
preble_routing="preble"
e2e_latency_predictor_routing="latency_predictor_e2e_latency"
ttft_latency_predictor_routing="latency_predictor_ttft"
avg_tpot_latency_predictor_routing="latency_predictor_avg_tpot"
random_routing="random"

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
    elif random_routing in strategy_lower:
        return random_routing
    else:
        return strategy_name  # Return original if no category match

def parse_strategy_name(filepath):
    """Extract the routing strategy name from the filepath."""
    # Pattern matches directories like "latency-prediction-based:none" from the path
    pattern = r"/([^/]+:[^/]+)/"
    match = re.search(pattern, filepath)
    if match:
        return match.group(1)
    else:
        # Fallback to the immediate parent directory if the pattern doesn't match
        return os.path.basename(os.path.dirname(filepath))

def find_log_files(base_dir):
    """Find all filtered log CSV files in the base directory - only one level deep."""
    log_files = []
    
    # Search in the base directory itself
    pattern = os.path.join(base_dir, "filtered-aibrix-gateway-plugins.log.csv")
    log_files.extend(glob.glob(pattern))
    
    # Search in immediate subdirectories (one level deep)
    pattern = os.path.join(base_dir, "*", "filtered-aibrix-gateway-plugins.log.csv")
    log_files.extend(glob.glob(pattern))
    
    return log_files

def analyze_llm_inference_logs(df):
    """Process the dataframe to calculate basic statistics - modified from your existing function."""
    if df.empty:
        print("No valid data found in the log file.")
        return df
    
    # Calculate experiment duration
    if 'request_start_time' in df.columns and 'request_end_time' in df.columns:
        start_time = df['request_start_time'].min()
        end_time = df['request_end_time'].max()
        experiment_duration = (end_time - start_time) / 1000000
        print(f"Experiment duration: {experiment_duration:.2f} seconds")
    
    if 'selectedpod' in df.columns:
        df['selectedpod'] = df['selectedpod'].str.split(':').str[0]

    # Process other metrics as in the original function
    # The rest of the processing is the same as in your original function
    # This is a simplified version focusing on the metrics we need
    
    return df

# def calculate_performance_metrics(df):
#     """Calculate the performance metrics for a dataframe."""
#     metrics = {}
    
#     # Calculate TTFT metrics if available
#     if 'ttft' in df.columns:
#         metrics['avg_ttft'] = df['ttft'].mean()
#         metrics['p99_ttft'] = df['ttft'].quantile(0.99)
    
#     # Calculate TPOT metrics if available
#     if 'avg_tpot' in df.columns:
#         metrics['avg_tpot'] = df['avg_tpot'].mean()
#         metrics['p99_tpot'] = df['avg_tpot'].quantile(0.99)
    
#     # Calculate throughput
#     if 'normalized_start_time' in df.columns and 'request_end_time' in df.columns:
#         # Calculate duration in seconds
#         total_duration = (df['request_end_time'].max() - df['request_start_time'].min()) / 1000000
#         if total_duration > 0:
#             metrics['throughput_rps'] = len(df) / total_duration
#         else:
#             metrics['throughput_rps'] = 0
    
#     # Calculate output token throughput if available
#     if 'normalized_start_time' in df.columns and 'numOutputTokens' in df.columns:
#         total_output_tokens = df['numOutputTokens'].sum()
#         if 'request_end_time' in df.columns and 'request_start_time' in df.columns:
#             total_duration = (df['request_end_time'].max() - df['request_start_time'].min()) / 1000000
#             if total_duration > 0:
#                 metrics['throughput_tps'] = total_output_tokens / total_duration
#             else:
#                 metrics['throughput_tps'] = 0
    
#     return metrics


def calculate_performance_metrics(df):
    """Calculate the performance metrics for a dataframe."""
    metrics = {}
    
    # Calculate TTFT metrics if available
    if 'ttft' in df.columns:
        metrics['avg_ttft'] = df['ttft'].mean()
        metrics['p99_ttft'] = df['ttft'].quantile(0.99)
    
    # Calculate TPOT metrics if available
    if 'avg_tpot' in df.columns:
        metrics['avg_tpot'] = df['avg_tpot'].mean()
        metrics['p99_tpot'] = df['avg_tpot'].quantile(0.99)

    # Calculate end-to-end latency metrics if available
    if 'request_start_time' in df.columns and 'request_end_time' in df.columns:
        df['end_to_end_latency'] = (df['request_end_time'] - df['request_start_time']) / 1000  # Convert to milliseconds
        metrics['avg_end_to_end'] = df['end_to_end_latency'].mean()
        metrics['p99_end_to_end'] = df['end_to_end_latency'].quantile(0.99)
    
    # Calculate throughput metrics - per-second RPS calculation and averaging
    # This accounts for different experiment durations and gives second-by-second breakdown
    if 'request_start_time' in df.columns and 'request_end_time' in df.columns:
        # Convert timestamps to seconds relative to experiment start
        experiment_start = df['request_start_time'].min()
        df_copy = df.copy()
        df_copy['start_seconds'] = (df_copy['request_start_time'] - experiment_start) / 1_000_000
        df_copy['end_seconds'] = (df_copy['request_end_time'] - experiment_start) / 1_000_000

        # Create 1-second bins from experiment start to end
        experiment_duration = df_copy['end_seconds'].max()
        time_bins = range(0, int(experiment_duration) + 2)  # +2 to cover the last partial second

        per_second_rps = []
        per_second_tps = []

        for second in time_bins:
            # Count requests that STARTED in this second (arrival rate)
            requests_in_second = df_copy[(df_copy['start_seconds'] >= second) &
                                       (df_copy['start_seconds'] < second + 1)]
            rps = len(requests_in_second)
            per_second_rps.append(rps)

            # Sum tokens from requests that STARTED in this second
            if 'numOutputTokens' in df.columns and len(requests_in_second) > 0:
                tps = requests_in_second['numOutputTokens'].sum()
            else:
                tps = 0
            per_second_tps.append(tps)

        # Calculate average RPS across all seconds (weighted by actual experiment duration)
        if per_second_rps:
            # Only consider seconds that are within the actual experiment duration
            valid_seconds = min(len(per_second_rps), int(experiment_duration) + 1)
            metrics['throughput_rps'] = sum(per_second_rps[:valid_seconds]) / valid_seconds

            if 'numOutputTokens' in df.columns:
                metrics['throughput_tps'] = sum(per_second_tps[:valid_seconds]) / valid_seconds
            else:
                metrics['throughput_tps'] = 0
        else:
            metrics['throughput_rps'] = 0
            metrics['throughput_tps'] = 0
    else:
        # Fallback: use relative_time if available
        if 'relative_time' in df.columns:
            total_duration = df['relative_time'].max() - df['relative_time'].min()
            if total_duration > 0:
                metrics['throughput_rps'] = len(df) / total_duration
                if 'numOutputTokens' in df.columns:
                    metrics['throughput_tps'] = df['numOutputTokens'].sum() / total_duration
                else:
                    metrics['throughput_tps'] = 0
            else:
                metrics['throughput_rps'] = 0
                metrics['throughput_tps'] = 0
        else:
            # Final fallback
            metrics['throughput_rps'] = 0
            metrics['throughput_tps'] = 0
    
    return metrics

def average_metrics_by_category(all_metrics, average_duplicates=False):
    """Group and average metrics by routing category if average_duplicates is True."""
    if not average_duplicates:
        return all_metrics
    
    # Group metrics by category
    category_groups = {}
    for metrics in all_metrics:
        category = categorize_strategy(metrics['strategy'])
        if category not in category_groups:
            category_groups[category] = []
        category_groups[category].append(metrics)
    
    # Average metrics within each category
    averaged_metrics = []
    for category, metrics_list in category_groups.items():
        if len(metrics_list) == 1:
            # Single experiment - just rename strategy to category
            avg_metrics = metrics_list[0].copy()
            avg_metrics['strategy'] = category
            avg_metrics['experiment_count'] = 1
            averaged_metrics.append(avg_metrics)
        else:
            # Multiple experiments - calculate averages and ranges
            avg_metrics = {'strategy': category, 'experiment_count': len(metrics_list)}
            
            # Get all numeric metrics to average
            numeric_metrics = ['avg_ttft', 'p99_ttft', 'avg_tpot', 'p99_tpot', 
                             'avg_end_to_end', 'p99_end_to_end', 'num_requests', 
                             'throughput_rps', 'throughput_tps']
            
            for metric in numeric_metrics:
                if metric in metrics_list[0]:
                    values = [m[metric] for m in metrics_list if metric in m]
                    if values:
                        avg_metrics[metric] = np.mean(values)
                        avg_metrics[f'{metric}_min'] = np.min(values)
                        avg_metrics[f'{metric}_max'] = np.max(values)
                        avg_metrics[f'{metric}_std'] = np.std(values)
            
            # Keep first file_path as representative
            avg_metrics['file_path'] = metrics_list[0]['file_path']
            averaged_metrics.append(avg_metrics)
    
    return averaged_metrics

def export_metrics_to_csv(all_metrics, base_dir, output_dir="../workload-and-experiment_results"):
    """Export performance metrics to an aggregated CSV file."""
    if not all_metrics:
        print("No metrics to export.")
        return

    # Resolve relative path from current working directory
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.getcwd(), output_dir)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Use a single aggregated CSV file
    csv_filename = "aggregated_summary.csv"
    csv_filepath = os.path.join(output_dir, csv_filename)

    # Extract group from base_dir (part after workload-and-experiment_results)
    group = ""
    if "workload-and-experiment_results" in base_dir:
        # Split by workload-and-experiment_results and take everything after it
        parts = base_dir.split("workload-and-experiment_results")
        if len(parts) > 1:
            group = parts[1].lstrip("/")

    # Define the metrics we want to export
    metric_columns = [
        'avg_ttft', 'p99_ttft', 'avg_tpot', 'p99_tpot',
        'avg_end_to_end', 'p99_end_to_end', 'num_requests',
        'throughput_rps', 'throughput_tps'
    ]

    print(f"Exporting metrics to aggregated file: {csv_filepath}")

    # Read existing data if file exists
    existing_data = {}
    if os.path.exists(csv_filepath):
        try:
            with open(csv_filepath, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    filename = row.get('filename', '')
                    if filename:
                        existing_data[filename] = row
        except Exception as e:
            print(f"Warning: Could not read existing CSV file: {e}")

    # Prepare new data for current run
    for metrics in all_metrics:
        strategy_name = metrics.get('strategy', '')
        row = {
            'filename': strategy_name,
            'routing_policy': categorize_strategy(strategy_name),
            'group': group,
        }

        # Add all metric values
        for metric in metric_columns:
            row[metric] = metrics.get(metric, '')

        # Add additional info if available (from averaging)
        if 'experiment_count' in metrics:
            row['experiment_count'] = metrics['experiment_count']
        if f'avg_ttft_std' in metrics:
            row['avg_ttft_std'] = metrics.get('avg_ttft_std', '')
            row['avg_ttft_min'] = metrics.get('avg_ttft_min', '')
            row['avg_ttft_max'] = metrics.get('avg_ttft_max', '')

        # Update existing data with new data (always update group for current run)
        if strategy_name in existing_data:
            # Preserve any existing fields not in current row, but update group
            existing_row = existing_data[strategy_name]
            existing_row.update(row)
            existing_data[strategy_name] = existing_row
        else:
            existing_data[strategy_name] = row

    # Write aggregated data back to CSV
    if existing_data:
        # Get all fieldnames from existing data
        all_fieldnames = set()
        for row in existing_data.values():
            all_fieldnames.update(row.keys())

        # Ensure consistent field order
        fieldnames = ['filename', 'routing_policy', 'group'] + metric_columns
        extra_fields = sorted(all_fieldnames - set(fieldnames))
        fieldnames.extend(extra_fields)

        with open(csv_filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in existing_data.values():
                writer.writerow(row)

        print(f"Successfully exported {len(existing_data)} total strategy results to {csv_filepath} ({len(all_metrics)} updated from current run)")

def normalize_time(df):
    first_request_start_time = df['request_start_time'].min()
    df['normalized_start_time'] = df['request_start_time'] - first_request_start_time
    df['normalized_end_time'] = df['request_end_time'] - first_request_start_time
    df['normalized_start_time'] /= 1_000_000
    df['normalized_end_time'] /= 1_000_000
    
    
    if 'log_window_start_time' in df.columns:
        df['log_window_start_time'] = df['log_window_start_time'] - first_request_start_time
        df['log_window_start_time'] /= 1_000_000
    if 'log_window_end_time' in df.columns:
        df['log_window_end_time'] = df['log_window_end_time'] - first_request_start_time
        df['log_window_end_time'] /= 1_000_000

    df.loc[:, 'normalized_start_time'] = df['normalized_start_time'] - df['normalized_start_time'].min()
    df.loc[:, 'normalized_end_time'] = df['normalized_end_time'] - df['normalized_start_time'].min()
    df = df.sort_values(by='normalized_start_time', ascending=True)
    df['time_bucket'] = df['normalized_start_time'].astype(int)
    df = df[['normalized_start_time', 'time_bucket', 'normalized_end_time'] + [col for col in df.columns if col != 'normalized_start_time' and col != 'normalized_end_time' and col != 'time_bucket']]
    df.reset_index(drop=True, inplace=True)
    return df

def process_log_file(file_path, warmup_seconds, cut_last_seconds):
    """Process a single log file and return its performance metrics AND the processed DataFrame."""
    print(f"Processing {file_path}...")
    df, json_columns = preprocess.parse_log_file(file_path)
    df = preprocess.parse_json_columns(df, json_columns)
    df = normalize_time(df)
    df = analyze_llm_inference_logs(df)
    
    # Filter out warm-up period (first 30 seconds)
    if len(df) > 0:
        # Check which timestamp column is available
        time_column = None
        if 'normalized_start_time' in df.columns:
            time_column = 'normalized_start_time'
        elif 'request_start_time' in df.columns:
            time_column = 'request_start_time'
        
        if time_column:
            min_time = df[time_column].min()
            original_count = len(df)
            if warmup_seconds != None:
                df = df[df[time_column] >= min_time + warmup_seconds]
            if cut_last_seconds != None:
                df = df[df[time_column] < min_time + cut_last_seconds]
            filtered_count = len(df)
            
            print(f"  Filtered out {original_count - filtered_count} requests from warm-up period (using {time_column})")
            print(f"  Remaining requests: {filtered_count}")
        else:
            print(f"  Warning: No suitable timestamp column found for warm-up filtering")
    
    # ADD: Create relative_time column for plotting (if not already present)
    if 'normalized_start_time' in df.columns and 'relative_time' not in df.columns:
        df['relative_time'] = df['normalized_start_time']
    elif 'request_start_time' in df.columns and 'relative_time' not in df.columns:
        # Convert to relative time in seconds
        min_time = df['request_start_time'].min()
        df['relative_time'] = (df['request_start_time'] - min_time) / 1000000
    
    # Extract strategy name from the file path
    strategy = parse_strategy_name(file_path)
    
    # Calculate performance metrics on filtered data
    metrics = calculate_performance_metrics(df)
    metrics['strategy'] = strategy
    metrics['file_path'] = file_path
    metrics['num_requests'] = len(df)
    
    # RETURN both metrics and the processed DataFrame
    return metrics, df



def calculate_ttft_reward(ttft, slo_ttft=500):
    """Calculate TTFT reward based on the given formula"""
    if ttft <= 0:
        return 0.5
    elif 0 < ttft <= slo_ttft:
        return 0.5 - 0.4 * (ttft / slo_ttft)
    else:  # ttft > slo_ttft
        return -0.1 - 0.4 * min(1, (ttft - slo_ttft) / slo_ttft)

def calculate_tpot_reward(tpot, slo_tpot=50):
    """Calculate TPOT reward based on the given formula"""
    if tpot <= 0:
        return -0.5
    elif 0 < tpot <= slo_tpot:
        return 0.1 + 0.4 * (1 - tpot / slo_tpot)
    else:  # tpot > slo_tpot
        return -0.1 - 0.4 * min(1, (tpot - slo_tpot) / slo_tpot)

def calculate_slo_satisfaction(df, slo_ttft=500, slo_tpot=50):
    """Calculate SLO satisfaction statistics"""
    ttft_satisfied = (df['ttft'] <= slo_ttft).sum()
    tpot_satisfied = (df['avg_tpot'] <= slo_tpot).sum()
    both_satisfied = ((df['ttft'] <= slo_ttft) & (df['avg_tpot'] <= slo_tpot)).sum()
    
    total_requests = len(df)
    
    return {
        'ttft_satisfied': ttft_satisfied,
        'tpot_satisfied': tpot_satisfied,
        'both_satisfied': both_satisfied,
        'total_requests': total_requests,
        'ttft_satisfaction_rate': ttft_satisfied / total_requests * 100 if total_requests > 0 else 0,
        'tpot_satisfaction_rate': tpot_satisfied / total_requests * 100 if total_requests > 0 else 0,
        'both_satisfaction_rate': both_satisfied / total_requests * 100 if total_requests > 0 else 0
    }

# Sort strategies by avg_ttft for consistent ordering
def get_strategy_priority(strategy_name):
    if rl_naive_routing in strategy_name.lower():
        return (0, strategy_name)  # First priority
    elif e2e_latency_predictor_routing in strategy_name.lower():
        return (1, strategy_name)  # Second priority
    elif ttft_latency_predictor_routing in strategy_name.lower():
        return (2, strategy_name)  # Second priority
    elif avg_tpot_latency_predictor_routing in strategy_name.lower():
        return (3, strategy_name)  # Third priority
    elif prefix_cache_1_routing in strategy_name.lower():
        return (4, strategy_name)  # Second priority
    elif prefix_cache_2_routing in strategy_name.lower():
        return (5, strategy_name)  # Third priority
    elif preble_routing in strategy_name.lower():
        return (6, strategy_name)  # Second priority
    elif random_routing in strategy_name.lower():
        return (7, strategy_name)  # Third priority
    else:
        return (8, strategy_name)  # Others last


# Set up colors by category
def get_strategy_color(strategy_name, index_in_category):
    """Get color for strategy based on category and index within category"""
    if rl_naive_routing in strategy_name.lower():
        base_colors = ['#ff0000', '#dc143c', '#ff6347', '#ff4500', '#ff7f50']  # Red family
    elif e2e_latency_predictor_routing in strategy_name.lower():
        base_colors = ['#8b008b','#ba55d3', '#9932cc', '#8a2be2',  '#c71585']  # Purple family
    elif ttft_latency_predictor_routing in strategy_name.lower():
        base_colors = ['#ff1493', '#ff69b4', '#dc143c', '#ff00ff', '#da70d6']  # Pink/Magenta family
    elif avg_tpot_latency_predictor_routing in strategy_name.lower():
        base_colors = ['#8b0000', '#b22222', '#cd5c5c', '#f08080', '#fa8072']  # Dark red/Coral family
    elif prefix_cache_1_routing in strategy_name.lower():
        base_colors = ['#1f77b4', '#4682b4', '#6495ed', '#aec7e8', '#87ceeb']  # Blue family
    elif prefix_cache_2_routing in strategy_name.lower():
        base_colors = ['#006400', '#228b22', '#32cd32', '#00ff00', '#7cfc00']  # Dark green/Lime family
    elif preble_routing in strategy_name.lower():
        base_colors = ['#ff8c00', '#ffa500', '#ffd700', '#ff6347', '#ff4500']  # Orange/Gold family
    elif random_routing in strategy_name.lower():
        base_colors = ['#2ca02c', '#32cd32', '#00ff00', '#00ff7f', '#98df8a']  # Light green family
    else:
        base_colors = ['#7f7f7f', '#696969', '#a9a9a9', '#c7c7c7', '#d3d3d3']  # Gray family
    # Use modulo to cycle through colors if more strategies than colors
    return base_colors[index_in_category % len(base_colors)]

    
def plot_routing_comparison(metrics_list, base_dir, slo_ttft, slo_tpot, csv_data_dict=None):
    """Create bar charts comparing performance metrics across routing strategies."""
    if not metrics_list:
        print("No metrics to plot.")
        return
    
    # Convert to DataFrame for easier plotting
    metrics_df = pd.DataFrame(metrics_list)
    

    # Sort strategies by custom priority
    all_strategies = metrics_df['strategy'].tolist()
    strategy_order = sorted(all_strategies, key=get_strategy_priority)

    # Create color dictionary with grouped coloring
    color_dict = {}
    category_counts = {rl_naive_routing: 0, prefix_cache_1_routing: 0, prefix_cache_2_routing: 0, preble_routing: 0, e2e_latency_predictor_routing: 0, ttft_latency_predictor_routing: 0, avg_tpot_latency_predictor_routing: 0, random_routing: 0, 'other': 0}

    for strategy in strategy_order:
        if rl_naive_routing in strategy.lower():
            color_dict[strategy] = get_strategy_color(strategy, category_counts[rl_naive_routing])
            category_counts[rl_naive_routing] += 1
        elif e2e_latency_predictor_routing in strategy.lower():
            color_dict[strategy] = get_strategy_color(strategy, category_counts[e2e_latency_predictor_routing])
            category_counts[e2e_latency_predictor_routing] += 1
        elif ttft_latency_predictor_routing in strategy.lower():
            color_dict[strategy] = get_strategy_color(strategy, category_counts[ttft_latency_predictor_routing])
            category_counts[ttft_latency_predictor_routing] += 1
        elif avg_tpot_latency_predictor_routing in strategy.lower():
            color_dict[strategy] = get_strategy_color(strategy, category_counts[avg_tpot_latency_predictor_routing])
            category_counts[avg_tpot_latency_predictor_routing] += 1
        elif prefix_cache_1_routing in strategy.lower():
            color_dict[strategy] = get_strategy_color(strategy, category_counts[prefix_cache_1_routing])
            category_counts[prefix_cache_1_routing] += 1
        elif prefix_cache_2_routing in strategy.lower():
            color_dict[strategy] = get_strategy_color(strategy, category_counts[prefix_cache_2_routing])
            category_counts[prefix_cache_2_routing] += 1
        elif preble_routing in strategy.lower():
            color_dict[strategy] = get_strategy_color(strategy, category_counts[preble_routing])
            category_counts[preble_routing] += 1
        elif random_routing in strategy.lower():
            color_dict[strategy] = get_strategy_color(strategy, category_counts[random_routing])
            category_counts[random_routing] += 1
        else:
            color_dict[strategy] = get_strategy_color(strategy, category_counts['other'])
            category_counts['other'] += 1
    
    # Create figure with custom GridSpec for better control
    fig = plt.figure(figsize=(36, 24))  # Increased width for 9 columns

    # MODIFIED GridSpec: 6 rows (bar charts, rewards, CDFs, avg)
    gs = GridSpec(6, 9, figure=fig,
                  height_ratios=[0.8, 1, 1, 1, 1, 1],
                  hspace=0.6,
                  wspace=0.35)
    
    fig.suptitle('Routing Strategy Performance Comparison', fontsize=maintitle_fontsize, y=0.96)
    
    # FIRST ROW: All 8 bar chart plots
    # Plot 1: Average TTFT
    if 'avg_ttft' in metrics_df.columns:
        ax = fig.add_subplot(gs[0, 0])
        plot_metric_bar(ax, metrics_df, 'avg_ttft', 'Average TTFT', strategy_order, color_dict)

    # Plot 2: P99 TTFT
    if 'p99_ttft' in metrics_df.columns:
        ax = fig.add_subplot(gs[0, 1])
        plot_metric_bar(ax, metrics_df, 'p99_ttft', 'P99 TTFT', strategy_order, color_dict)

    # Plot 3: Average TPOT
    if 'avg_tpot' in metrics_df.columns:
        ax = fig.add_subplot(gs[0, 2])
        plot_metric_bar(ax, metrics_df, 'avg_tpot', 'Average TPOT', strategy_order, color_dict)

    # Plot 4: P99 TPOT
    if 'p99_tpot' in metrics_df.columns:
        ax = fig.add_subplot(gs[0, 3])
        plot_metric_bar(ax, metrics_df, 'p99_tpot', 'P99 TPOT', strategy_order, color_dict)

    # Plot 5: End-to-End
    if 'avg_end_to_end' in metrics_df.columns:
        ax = fig.add_subplot(gs[0, 4])
        plot_metric_bar(ax, metrics_df, 'avg_end_to_end', 'Average End-to-End', strategy_order, color_dict)

    # Plot 6: P99 End-to-End
    if 'p99_end_to_end' in metrics_df.columns:
        ax = fig.add_subplot(gs[0, 5])
        plot_metric_bar(ax, metrics_df, 'p99_end_to_end', 'P99 End-to-End', strategy_order, color_dict)

    # Plot 7: Total Requests
    if 'num_requests' in metrics_df.columns:
        ax = fig.add_subplot(gs[0, 6])
        plot_metric_bar(ax, metrics_df, 'num_requests', 'Total Requests', strategy_order, color_dict)

    # Plot 8: Token Throughput
    if 'throughput_tps' in metrics_df.columns:
        ax = fig.add_subplot(gs[0, 7])
        plot_metric_bar(ax, metrics_df, 'throughput_tps', 'Throughput (Tokens/sec)', strategy_order, color_dict)
    
    # NEW REWARD PLOTS - Each occupying a full row with more spacing
    if csv_data_dict:
        # Calculate rewards for each strategy
        for strategy in strategy_order:
            if strategy in csv_data_dict:
                df = csv_data_dict[strategy]
                df['ttft_reward'] = df['ttft'].apply(lambda x: calculate_ttft_reward(x, slo_ttft))
                df['tpot_reward'] = df['avg_tpot'].apply(lambda x: calculate_tpot_reward(x, slo_tpot))
                df['total_reward'] = df['ttft_reward'] + df['tpot_reward']

        # Plot 7: TTFT Reward Time Series (full width)
        ax = fig.add_subplot(gs[1, :])  # Full width of row 1
        plot_reward_timeseries(ax, csv_data_dict, 'ttft_reward', 'TTFT Reward', 
                      strategy_order, color_dict, slo_ttft, 'TTFT')

        # Plot 8: TPOT Reward Time Series (full width)
        ax = fig.add_subplot(gs[2, :])  # Full width of row 2
        plot_reward_timeseries(ax, csv_data_dict, 'tpot_reward', 'TPOT Reward', 
                      strategy_order, color_dict, slo_tpot, 'TPOT')

        # Plot 9: Total Reward Time Series (full width)
        ax = fig.add_subplot(gs[3, :])  # Full width of row 3
        plot_reward_timeseries(ax, csv_data_dict, 'total_reward', 'Total Reward', 
                      strategy_order, color_dict, None, 'Total')

        # Plot 10: TTFT Latency CDF (left third of row 4)
        ax = fig.add_subplot(gs[4, :3])
        plot_latency_cdf(ax, csv_data_dict, strategy_order, color_dict, 'ttft', 'TTFT Latency CDF', 'TTFT (ms)')

        # Plot 11: Avg TPOT Latency CDF (middle third of row 4)
        ax = fig.add_subplot(gs[4, 3:6])
        plot_latency_cdf(ax, csv_data_dict, strategy_order, color_dict, 'avg_tpot', 'Avg TPOT Latency CDF', 'Avg TPOT (ms)')

        # Plot 12: End-to-End Latency CDF (right third of row 4)
        ax = fig.add_subplot(gs[4, 6:])
        plot_latency_cdf(ax, csv_data_dict, strategy_order, color_dict, 'end_to_end_latency', 'End-to-End Latency CDF', 'End-to-End (ms)')

        # Plot 13: Average TTFT, TPOT, and End-to-End Comparison (full width, now row 5)
        ax = fig.add_subplot(gs[5, :])
        plot_triple_axis_comparison(ax, metrics_df, strategy_order, color_dict)
    else:
        # If no CSV data provided, show placeholder text for reward plots
        for row in [1, 2, 3, 4, 5]:
            if row == 4:
                # CDFs: left, middle, and right thirds
                ax = fig.add_subplot(gs[4, :3])
                ax.text(0.5, 0.5, 'No time series data available\n(csv_data_dict not provided)',
                        ha='center', va='center', fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                ax.set_xticks([])
                ax.set_yticks([])
                ax = fig.add_subplot(gs[4, 3:6])
                ax.text(0.5, 0.5, 'No time series data available\n(csv_data_dict not provided)',
                        ha='center', va='center', fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                ax.set_xticks([])
                ax.set_yticks([])
                ax = fig.add_subplot(gs[4, 6:])
                ax.text(0.5, 0.5, 'No time series data available\n(csv_data_dict not provided)',
                        ha='center', va='center', fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax = fig.add_subplot(gs[row, :])
                ax.text(0.5, 0.5, 'No time series data available\n(csv_data_dict not provided)', 
                        ha='center', va='center', fontsize=12, 
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                ax.set_xticks([])
                ax.set_yticks([])
    
    # Create a single shared legend for all plots
    handles = [plt.Rectangle((0,0), 1, 1, color=color_dict[s]) for s in strategy_order]
    legend_labels = [s for s in strategy_order]
    
    # Place the legend at the bottom
    # fig.legend(handles, legend_labels, 
    #           loc='upper right', 
    #           bbox_to_anchor=(1.35, 0.5),
    #           fontsize=legend_fontsize, 
    #           ncol=1)
    
    # MODIFIED layout parameters for better spacing
    plt.subplots_adjust(top=0.93, bottom=0.08, left=0.04, right=0.96)
    
    # Save the figure
    output_file = f"{base_dir}/routing_strategy_comparison.pdf"
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"** Saved comparison plot to {output_file}")
    
    
def plot_reward_timeseries(ax, csv_data_dict, reward_column, title, strategy_order, color_dict, slo_value, metric_type):
    """Plot reward time series with 1-second granularity trend lines for all strategies."""
    
    for strategy in strategy_order:
        if strategy in csv_data_dict and reward_column in csv_data_dict[strategy].columns:
            df = csv_data_dict[strategy]
            color = color_dict[strategy]

            # ax.scatter(df['relative_time'], df[reward_column], 
            #           s=8, alpha=0.3, color=color, 
            #           label=f'{strategy} (individual)', zorder=1)
            
            # Create 1-second time bins
            df_copy = df.copy()
            df_copy['time_bin'] = np.floor(df_copy['relative_time']).astype(int)
            
            # Calculate average reward and statistics for each 1-second bin
            reward_stats = df_copy.groupby('time_bin')[reward_column].agg(['mean', 'std', 'count']).reset_index()
            reward_stats = reward_stats[reward_stats['count'] > 0]  # Only bins with data
            
            # Plot the trend line with slightly larger markers for full-width plots
            ax.plot(reward_stats['time_bin'], reward_stats['mean'], color=color, linewidth=2, alpha=0.8, label=strategy)
            
            # # Optional: Add confidence bands (standard error) - FIXED VERSION
            # if len(reward_stats) > 1:
            #     # Fill NaN std values with 0 (for bins with only 1 data point)
            #     reward_stats['std'] = reward_stats['std'].fillna(0)
                
            #     # Calculate standard error
            #     reward_se = reward_stats['std'] / np.sqrt(reward_stats['count'])
                
            #     # Add subtle confidence bands - now all arrays have the same length
            #     ax.fill_between(reward_stats['time_bin'], 
            #                   reward_stats['mean'] - reward_se, 
            #                   reward_stats['mean'] + reward_se,
            #                   color=color, alpha=0.15)
    
    # Set title with SLO info
    if slo_value:
        full_title = f'{title} Trend Over Time (1s averages) - SLO: {metric_type} ≤ {slo_value}ms'
    else:
        full_title = f'{title} Trend Over Time (1s averages)'
    
    ax.set_title(full_title, fontsize=subtitle_fontsize+2, pad=15)  # Slightly larger title
    ax.set_xlabel('Time (seconds)', fontsize=ylabel_fontsize)
    ax.set_ylabel(f'Average {title}', fontsize=ylabel_fontsize)
    
    # Set y-axis limits based on reward type
    if 'total' in reward_column.lower():
        ax.set_ylim(-1.2, 1.2)
    else:
        ax.set_ylim(-0.6, 0.6)
    
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    
    # # Position legend in upper right
    # ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
    
    # Add some additional reference lines for better interpretation
    if 'total' not in reward_column.lower():
        # Add SLO satisfaction threshold lines for individual rewards
        if metric_type == 'TTFT':
            ax.axhline(y=0.1, color='green', linestyle=':', alpha=0.7, linewidth=1.5)
        elif metric_type == 'TPOT':
            ax.axhline(y=0.5, color='green', linestyle=':', alpha=0.7, linewidth=1.5)


# New function to plot triple-axis comparison (TTFT, TPOT, End-to-End)
def plot_triple_axis_comparison(ax, metrics_df, strategy_order, color_dict):
    """Plot bar chart with triple y-axis: left for avg TTFT, middle for avg TPOT, right for avg end-to-end across strategies."""
    strategies = [s for s in strategy_order if s in metrics_df['strategy'].values]
    label_list = []
    for s in strategies:
        len_s = len(s)
        # label_list.append(f"{s[:len_s//2]}\n{s[len_s//2:]}")
        label_list.append(f"{s.split('-')[0]}")
    n_strategies = len(strategies)
    avg_ttft = [metrics_df.set_index('strategy').loc[s, 'avg_ttft'] if 'avg_ttft' in metrics_df.columns else 0 for s in strategies]
    avg_tpot = [metrics_df.set_index('strategy').loc[s, 'avg_tpot'] if 'avg_tpot' in metrics_df.columns else 0 for s in strategies]
    avg_e2e = [metrics_df.set_index('strategy').loc[s, 'avg_end_to_end'] if 'avg_end_to_end' in metrics_df.columns else 0 for s in strategies]
    x = np.arange(n_strategies)
    bar_width = 0.5
    strategy_colors = [color_dict[s] for s in strategies]

    # Bars for TTFT (left y-axis)
    bars1 = ax.bar(x - bar_width/3, avg_ttft, bar_width/3, label='Avg TTFT (ms)', color=strategy_colors, alpha=0.9, edgecolor='black', linewidth=1)
    ax.set_ylabel('Avg TTFT (ms)', fontsize=ylabel_fontsize, color='#222266')
    ax.tick_params(axis='y', labelcolor='#222266', labelsize=tick_fontsize)

    # Add value labels for TTFT
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{height:.0f}', rotation=90, ha='center', va='bottom', fontsize=12, fontweight='bold', color='#222266')

    # Twin axis for TPOT (middle y-axis)
    ax2 = ax.twinx()
    bars2 = ax2.bar(x, avg_tpot, bar_width/3, label='Avg TPOT (ms)', color=strategy_colors, alpha=0.6, edgecolor='black', linewidth=1)
    ax2.set_ylabel('Avg TPOT (ms)', fontsize=ylabel_fontsize, color='#226622')
    ax2.tick_params(axis='y', labelcolor='#226622', labelsize=tick_fontsize)

    # Add value labels for TPOT
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{height:.0f}', rotation=90, ha='center', va='bottom', fontsize=12, color='#226622')

    # Twin axis for End-to-End (right y-axis)
    ax3 = ax.twinx()
    # Offset the third axis to the right
    ax3.spines["right"].set_position(("axes", 1.1))
    bars3 = ax3.bar(x + bar_width/3, avg_e2e, bar_width/3, label='Avg End-to-End (ms)', color=strategy_colors, alpha=0.3, edgecolor='black', linewidth=1)
    ax3.set_ylabel('Avg End-to-End (ms)', fontsize=ylabel_fontsize, color='#662222')
    ax3.tick_params(axis='y', labelcolor='#662222', labelsize=tick_fontsize)

    # Add value labels for End-to-End
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{height:.0f}', rotation=90, ha='center', va='bottom', fontsize=12, color='#662222')

    # X-axis and title
    ax.set_title('Average TTFT (left), TPOT (middle), End-to-End (right) Comparison', fontsize=subtitle_fontsize)
    # ax.set_xlabel('Routing Strategy', fontsize=ylabel_fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(label_list, rotation=45, ha='right', fontsize=tick_fontsize)
    ax.grid(axis='y', alpha=0.3)
    ax.set_zorder(2)
    ax.patch.set_visible(False)
    # Set y-axis limits with padding
    max_ttft = max(avg_ttft or [0])
    max_tpot = max(avg_tpot or [0])
    max_e2e = max(avg_e2e or [0])
    ax.set_ylim(0, max_ttft * 1.4 if max_ttft > 0 else 1)
    ax2.set_ylim(0, max_tpot * 1.4 if max_tpot > 0 else 1)
    ax3.set_ylim(0, max_e2e * 1.4 if max_e2e > 0 else 1)
    
    
    ## Custom legend
    # lines = [bars1, bars2]
    # labels = ['Avg TTFT (ms)', 'Avg TPOT (ms)']
    # ax.legend(lines, labels, fontsize=legend_fontsize, loc='upper right')


def plot_metric_bar(ax, metrics_df, metric, title, strategy_order, color_dict):
    """Helper function to create a bar chart for a specific metric - optimized for narrow plots."""
    if metric not in metrics_df.columns:
        ax.text(0.5, 0.5, f"No data for {metric}", horizontalalignment='center', verticalalignment='center')
        ax.set_title(title, fontsize=subtitle_fontsize-2)  # Slightly smaller title for narrow plots
        return
    
    # Sort by strategy order - only include strategies that exist in the data
    available_strategies = [s for s in strategy_order if s in metrics_df['strategy'].values]
    plot_data = metrics_df.set_index('strategy').loc[available_strategies, :]
    
    # Use simple index numbers for x-axis
    bar_positions = np.arange(len(plot_data))
    
    # Check if we have min/max data for error bars (from averaging)
    has_error_bars = f'{metric}_min' in plot_data.columns and f'{metric}_max' in plot_data.columns
    
    if has_error_bars:
        # Calculate error bar values (distance from mean to min/max)
        lower_errors = plot_data[metric] - plot_data[f'{metric}_min']
        upper_errors = plot_data[f'{metric}_max'] - plot_data[metric]
        error_bars = [lower_errors.values, upper_errors.values]
        
        # Create bar chart with error bars
        bars = ax.bar(bar_positions, plot_data[metric], 
                      yerr=error_bars,
                      color=[color_dict[s] for s in available_strategies],
                      width=0.8,
                      capsize=3,
                      error_kw={'linewidth': 1.5, 'capthick': 1.5}
                      )
        max_bar_height = plot_data[f'{metric}_max'].max()
    else:
        # Create bar chart without error bars
        bars = ax.bar(bar_positions, plot_data[metric], 
                      color=[color_dict[s] for s in available_strategies],
                      width=0.8,
                      )
        max_bar_height = plot_data[metric].max()
    
    # Calculate relative performance for latency metrics (lower is better)
    # For throughput metrics (higher is better), we'll calculate inverse ratios
    is_latency_metric = any(word in metric.lower() for word in ['ttft', 'tpot'])
    is_throughput_metric = 'throughput' in metric.lower()
    
    if is_latency_metric:
        # For latency: find minimum (best) value and calculate degradation
        min_value = plot_data[metric].min()
        relative_values = plot_data[metric] / min_value
    elif is_throughput_metric:
        # For throughput: find maximum (best) value and calculate degradation
        max_value = plot_data[metric].max()
        relative_values = max_value / plot_data[metric]
    else:
        # Default case: treat as latency metric
        min_value = plot_data[metric].min()
        relative_values = plot_data[metric] / min_value

    # Add value labels on top of each bar with relative performance
    for i, bar in enumerate(bars):
        height = bar.get_height()
        relative_perf = relative_values.iloc[i]
        
        # Add experiment count info if available
        experiment_count = ""
        if 'experiment_count' in plot_data.columns:
            count = plot_data['experiment_count'].iloc[i]
            if count > 1:
                experiment_count = f" (n={count})"
        
        annotation_text = f'{height:.0f} ({relative_perf:.1f}x){experiment_count}'
        
        # Calculate dynamic offset to prevent overflow
        text_y_position = height if not has_error_bars else plot_data[f'{metric}_max'].iloc[i]
        ax.annotate(annotation_text,
                    xy=(bar.get_x() + bar.get_width() / 2, text_y_position),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', rotation=90,
                    fontsize=text_fontsize-2, color='black')  # Smaller font
    
    ax.set_ylim(0, max_bar_height * 2.2)  # Provide more extra space above bars

    # Set chart titles and labels - adjusted for narrow plots
    ax.set_title(title, fontsize=subtitle_fontsize-4, pad=8)  # Even smaller title
    # Add ylabel only for the leftmost chart (Average TTFT)
    if metric == 'avg_ttft':
        ax.set_ylabel('millisecond', fontsize=ylabel_fontsize-2)
    
    # Remove x-axis ticks and labels
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_xlabel('')
    
    # Optimize y-axis ticks
    ax.tick_params(axis='y', labelsize=tick_fontsize-2)  # Smaller tick labels
    if len(ax.get_yticks()) > 4:  # Fewer ticks for narrow plots
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    
    ax.grid(axis='y', alpha=0.3)


# New function to plot CDFs for TTFT and avg TPOT
def plot_latency_cdf(ax, csv_data_dict, strategy_order, color_dict, column, title, xlabel):
    """Plot CDF for a given latency column for each strategy."""
    for strategy in strategy_order:
        if strategy in csv_data_dict and column in csv_data_dict[strategy].columns:
            data = csv_data_dict[strategy][column].dropna().sort_values()
            if len(data) == 0:
                continue
            y = np.linspace(0, 1, len(data))
            ax.plot(data, y, label=strategy, color=color_dict[strategy], linewidth=2, alpha=0.8)
    ax.set_title(title, fontsize=subtitle_fontsize)
    ax.set_xlabel(xlabel, fontsize=ylabel_fontsize)
    ax.set_ylabel('CDF', fontsize=ylabel_fontsize)
    # ax.legend(fontsize=legend_fontsize, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=tick_fontsize)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare routing strategies performance')
    parser.add_argument('base_directory', help='Base directory containing log files')
    parser.add_argument('warmup_seconds', nargs='?', type=int, default=None, 
                       help='Seconds to exclude from start for warmup')
    parser.add_argument('cut_last_seconds', nargs='?', type=int, default=None,
                       help='Seconds to exclude from end')
    parser.add_argument('--average-duplicates', action='store_true',
                       help='Average multiple experiments for the same routing policy')
    
    args = parser.parse_args()
    
    base_dir = args.base_directory
    warmup_seconds = args.warmup_seconds
    cut_last_seconds = args.cut_last_seconds
    average_duplicates = args.average_duplicates
    
    print(f"Searching for log files in {base_dir}...")
    if warmup_seconds is not None:
        print(f"warmup_seconds: {warmup_seconds} seconds")
    if cut_last_seconds is not None:
        print(f"cut_last_seconds: {cut_last_seconds} seconds")
    if average_duplicates:
        print("Will average multiple experiments for the same routing policy")
    
    slo_ttft = 1000
    slo_tpot = 50
    
    log_files = find_log_files(base_dir)
    print(f"Found {len(log_files)} log files.")
    
    if not log_files:
        print(f"No log files found in {base_dir}")
        sys.exit(1)
    
    # Process each log file - MODIFIED to collect both metrics and DataFrames
    all_metrics = []
    csv_data_dict = {}  # ADD: Dictionary to store DataFrames
    
    for log_file in log_files:
        result = process_log_file(log_file, warmup_seconds, cut_last_seconds)
        if result:
            metrics, df = result  # UNPACK both metrics and DataFrame
            all_metrics.append(metrics)
            csv_data_dict[metrics['strategy']] = df  # STORE DataFrame by strategy name
    
    # Apply averaging if requested
    if average_duplicates:
        print(f"Original strategies: {[m['strategy'] for m in all_metrics]}")
        all_metrics = average_metrics_by_category(all_metrics, average_duplicates)
        print(f"After averaging: {[m['strategy'] for m in all_metrics]}")
        
        # For CSV data, we'll keep the first experiment's data for each category
        # (Time series averaging is more complex and not implemented in this version)
        if csv_data_dict:
            category_csv_dict = {}
            for strategy, df in csv_data_dict.items():
                category = categorize_strategy(strategy)
                if category not in category_csv_dict:
                    category_csv_dict[category] = df
            csv_data_dict = category_csv_dict
    
    # ADD: Debug information
    print(f"CSV data dict keys: {list(csv_data_dict.keys())}")
    if csv_data_dict:
        first_key = list(csv_data_dict.keys())[0]
        sample_df = csv_data_dict[first_key]
        print(f"Sample DataFrame columns: {list(sample_df.columns)}")
        print(f"Sample DataFrame shape: {sample_df.shape}")

        # Check for required columns
        required_cols = ['ttft', 'avg_tpot', 'relative_time']
        missing_cols = [col for col in required_cols if col not in sample_df.columns]
        if missing_cols:
            print(f"WARNING: Missing required columns: {missing_cols}")
            print(f"Available columns: {list(sample_df.columns)}")

    # Export metrics to CSV
    export_metrics_to_csv(all_metrics, base_dir)

    # Plot the comparison - MODIFIED to pass csv_data_dict
    plot_routing_comparison(all_metrics, base_dir, slo_ttft, slo_tpot, csv_data_dict)