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
from matplotlib.patches import Patch

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
least_kv_cache_routing="least_kv_cache"
least_latency_routing="least_latency"
least_request_routing="least_request"
contextual_bandit_routing="contextual_bandit"

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
    elif least_kv_cache_routing in strategy_lower:
        return least_kv_cache_routing
    elif least_latency_routing in strategy_lower:
        return least_latency_routing
    elif least_request_routing in strategy_lower:
        return least_request_routing
    elif contextual_bandit_routing in strategy_lower:
        return contextual_bandit_routing
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

def is_ml_strategy(strategy_name):
    """Return True for ML-based routing policies (latency predictor, RL, contextual bandit, etc.)."""
    strategy_lower = strategy_name.lower()
    ml_markers = [
        'latency_predictor',
        'contextual_bandit',
        'rl',
    ]
    return any(marker in strategy_lower for marker in ml_markers)

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
        # Debug: Print sample values and statistics
        print(f"  TTFT column statistics:")
        print(f"    - Sample values (first 5): {df['ttft'].head().tolist()}")
        print(f"    - Min: {df['ttft'].min():.2f}, Max: {df['ttft'].max():.2f}")
        print(f"    - Mean (before any conversion): {df['ttft'].mean():.2f}")
        print(f"    - Median: {df['ttft'].median():.2f}")
        
        metrics['avg_ttft'] = df['ttft'].mean()
        metrics['p99_ttft'] = df['ttft'].quantile(0.99)
        metrics['p999_ttft'] = df['ttft'].quantile(0.999)
        
        print(f"    - Calculated avg_ttft: {metrics['avg_ttft']:.2f} ms")
        print(f"    - Calculated p99_ttft: {metrics['p99_ttft']:.2f} ms")
        print(f"    - Calculated p999_ttft: {metrics['p999_ttft']:.2f} ms")

    # Calculate TPOT metrics if available
    if 'avg_tpot' in df.columns:
        # Debug: Print sample values and statistics
        print(f"  TPOT column statistics:")
        print(f"    - Sample values (first 5): {df['avg_tpot'].head().tolist()}")
        print(f"    - Min: {df['avg_tpot'].min():.2f}, Max: {df['avg_tpot'].max():.2f}")
        print(f"    - Mean (before any conversion): {df['avg_tpot'].mean():.2f}")
        print(f"    - Median: {df['avg_tpot'].median():.2f}")
        
        metrics['avg_tpot'] = df['avg_tpot'].mean()
        metrics['p99_tpot'] = df['avg_tpot'].quantile(0.99)
        metrics['p999_tpot'] = df['avg_tpot'].quantile(0.999)
        
        print(f"    - Calculated avg_tpot: {metrics['avg_tpot']:.2f} ms")
        print(f"    - Calculated p99_tpot: {metrics['p99_tpot']:.2f} ms")
        print(f"    - Calculated p999_tpot: {metrics['p999_tpot']:.2f} ms")

    # Calculate end-to-end latency metrics if available
    if 'request_start_time' in df.columns and 'request_end_time' in df.columns:
        df['end_to_end_latency'] = (df['request_end_time'] - df['request_start_time']) / 1000  # Convert to milliseconds
        metrics['avg_end_to_end'] = df['end_to_end_latency'].mean()
        metrics['p99_end_to_end'] = df['end_to_end_latency'].quantile(0.99)

    # Calculate end-to-end overhead metrics if available
    if 'endToEndOverhead' in df.columns:
        metrics['avg_end_to_end_overhead'] = df['endToEndOverhead'].mean()
        metrics['p50_end_to_end_overhead'] = df['endToEndOverhead'].quantile(0.50)
        metrics['p99_end_to_end_overhead'] = df['endToEndOverhead'].quantile(0.99)
        metrics['p999_end_to_end_overhead'] = df['endToEndOverhead'].quantile(0.999)
    
    # Calculate throughput metrics - per-second RPS calculation and averaging
    # Prefer normalized/relative start times to avoid outlier end timestamps.
    start_seconds = None
    if 'normalized_start_time' in df.columns:
        start_seconds = df['normalized_start_time'].astype(float)
    elif 'request_start_time' in df.columns:
        start_seconds = (df['request_start_time'] - df['request_start_time'].min()) / 1_000_000
    elif 'relative_time' in df.columns:
        start_seconds = df['relative_time'].astype(float)

    if start_seconds is not None and len(start_seconds) > 0:
        df_copy = df.copy()
        df_copy['start_seconds'] = pd.to_numeric(start_seconds, errors='coerce')
        df_copy = df_copy.replace([np.inf, -np.inf], np.nan).dropna(subset=['start_seconds'])

        if df_copy.empty:
            metrics['throughput_rps'] = 0
            metrics['throughput_tps'] = 0
        else:
            max_second = int(np.floor(df_copy['start_seconds'].max()))
            effective_max_second = max_second

            # Guard against extreme outliers that would create huge zero-filled ranges.
            if max_second > len(df_copy) * 10:
                p99_9 = np.nanpercentile(df_copy['start_seconds'], 99.9)
                effective_max_second = int(np.floor(p99_9))
                if effective_max_second < 0:
                    effective_max_second = 0

            total_seconds = max(effective_max_second + 1, 1)
            if effective_max_second < max_second:
                in_window = df_copy['start_seconds'] <= effective_max_second
                total_requests = int(in_window.sum())
                if 'numOutputTokens' in df_copy.columns:
                    total_tokens = df_copy.loc[in_window, 'numOutputTokens'].sum()
                else:
                    total_tokens = 0
            else:
                total_requests = len(df_copy)
                if 'numOutputTokens' in df_copy.columns:
                    total_tokens = df_copy['numOutputTokens'].sum()
                else:
                    total_tokens = 0

            metrics['throughput_rps'] = total_requests / total_seconds
            metrics['throughput_tps'] = total_tokens / total_seconds
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
            numeric_metrics = ['avg_ttft', 'p99_ttft', 'p999_ttft', 'avg_tpot', 'p99_tpot', 'p999_tpot',
                             'avg_end_to_end', 'p99_end_to_end',
                             'avg_end_to_end_overhead', 'p50_end_to_end_overhead',
                             'p99_end_to_end_overhead', 'p999_end_to_end_overhead',
                             'num_requests',
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

def export_metrics_to_csv(all_metrics, base_dir):
    """Export performance metrics to a CSV file in the same directory as the PDF."""
    if not all_metrics:
        print("No metrics to export.")
        return None

    # Extract workload identifier from base_dir for the workload column
    # e.g., "NVIDIA-A10/maxTokens_1-maxTokensStd_0/SharingRatio71%/rps7" -> "SharingRatio71%/rps7"
    workload = ""
    if "workload-and-experiment_results" in base_dir:
        parts = base_dir.split("workload-and-experiment_results")
        if len(parts) > 1:
            # Get the path after workload-and-experiment_results
            full_path = parts[1].lstrip("/")
            # Extract just the SharingRatio and rps parts
            path_parts = full_path.split("/")
            # Find the SharingRatio part and everything after
            for i, part in enumerate(path_parts):
                if "SharingRatio" in part or "MixedSharingRatio" in part:
                    workload = "/".join(path_parts[i:])
                    break
            if not workload:
                workload = full_path

    # Define the metrics we want to export
    metric_columns = [
        'avg_ttft', 'p99_ttft', 'p999_ttft',
        'avg_tpot', 'p99_tpot', 'p999_tpot',
        'avg_end_to_end', 'p99_end_to_end',
        'avg_end_to_end_overhead', 'p50_end_to_end_overhead',
        'p99_end_to_end_overhead', 'p999_end_to_end_overhead',
        'num_requests', 'throughput_rps', 'throughput_tps'
    ]

    # Save CSV file in the same directory as the PDF
    csv_filepath = os.path.join(base_dir, "routing_strategy_metrics_client.csv")

    rows = []
    for metrics in all_metrics:
        strategy_name = metrics.get('strategy', '')
        row = {
            'workload': workload,
            'routing_policy': categorize_strategy(strategy_name),
            'strategy_full_name': strategy_name,
        }

        # Add all metric values
        for metric in metric_columns:
            row[metric] = metrics.get(metric, '')

        # Add additional info if available (from averaging)
        if 'experiment_count' in metrics:
            row['experiment_count'] = metrics['experiment_count']
        if 'avg_ttft_std' in metrics:
            row['avg_ttft_std'] = metrics.get('avg_ttft_std', '')
            row['avg_ttft_min'] = metrics.get('avg_ttft_min', '')
            row['avg_ttft_max'] = metrics.get('avg_ttft_max', '')

        rows.append(row)

    # Write CSV file
    if rows:
        # Define fieldnames
        fieldnames = ['workload', 'routing_policy', 'strategy_full_name'] + metric_columns
        # Add optional fields if present
        optional_fields = ['experiment_count', 'avg_ttft_std', 'avg_ttft_min', 'avg_ttft_max']
        for field in optional_fields:
            if any(field in row for row in rows):
                fieldnames.append(field)

        with open(csv_filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        print(f"** Saved metrics CSV to {csv_filepath}")
        return csv_filepath

    return None

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

def process_log_file(file_path, warmup_seconds, cut_last_seconds, iteration_from):
    """Process a single log file and return its performance metrics AND the processed DataFrame."""
    print(f"Processing {file_path}...")
    df, json_columns = preprocess.parse_log_file(file_path)
    df = preprocess.parse_json_columns(df, json_columns)
    
    # Filter out anomalous timestamps (outliers) before normalization
    if len(df) > 10 and 'request_start_time' in df.columns:
        import statistics
        valid_start_times = df['request_start_time'].dropna()
        if len(valid_start_times) > 10:
            median_time = statistics.median(valid_start_times)
            # Filter out times that are more than 10x away from the median
            outlier_threshold_low = median_time / 10
            outlier_threshold_high = median_time * 10
            before_count = len(df)
            df = df[(df['request_start_time'] >= outlier_threshold_low) & 
                    (df['request_start_time'] <= outlier_threshold_high)]
            after_count = len(df)
            if before_count > after_count:
                print(f"  Warning: Filtered out {before_count - after_count} entries with anomalous timestamps (likely clock sync issues)")
    
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

    # Apply iteration-based filtering only for ML-based policies
    if iteration_from is not None and iteration_from > 0 and is_ml_strategy(strategy):
        if 'iteration' in df.columns:
            df['iteration'] = pd.to_numeric(df['iteration'], errors='coerce')
            before_count = len(df)
            df = df[df['iteration'] >= iteration_from]
            after_count = len(df)
            print(f"  Iteration filter applied: {before_count - after_count} rows removed (iteration >= {iteration_from})")
        else:
            print("  Warning: iteration column not found; skipping iteration filter")

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
    elif least_kv_cache_routing in strategy_name.lower():
        return (8, strategy_name)
    elif least_latency_routing in strategy_name.lower():
        return (9, strategy_name)
    elif least_request_routing in strategy_name.lower():
        return (10, strategy_name)
    elif contextual_bandit_routing in strategy_name.lower():
        return (11, strategy_name)
    else:
        return (11, strategy_name)  # Others last


# Set up colors by category
def get_strategy_color(strategy_name, index_in_category):
    """Get color for strategy based on category and index within category"""
    if rl_naive_routing in strategy_name.lower():
        base_colors = ['#4169e1', '#483d8b', '#6a5acd', '#7b68ee', '#9370db']  # Slate blue family
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
    elif least_kv_cache_routing in strategy_name.lower():
        base_colors = ['#d2691e', '#cd853f', '#daa520', '#b8860b', '#f4a460']  # Brown/Tan family
    elif least_latency_routing in strategy_name.lower():
        base_colors = ['#483d8b', '#6a5acd', '#7b68ee', '#9370db', '#8470ff']  # Slate blue family
    elif least_request_routing in strategy_name.lower():
        base_colors = ['#008b8b', '#20b2aa', '#48d1cc', '#40e0d0', '#00ced1']  # Cyan/Teal family
    elif contextual_bandit_routing in strategy_name.lower():
        base_colors = ['#ff0000', '#dc143c', '#ff6347', '#ff4500', '#ff7f50']  # Red family
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
    category_counts = {rl_naive_routing: 0, prefix_cache_1_routing: 0, prefix_cache_2_routing: 0, preble_routing: 0, e2e_latency_predictor_routing: 0, ttft_latency_predictor_routing: 0, avg_tpot_latency_predictor_routing: 0, random_routing: 0, least_kv_cache_routing: 0, least_latency_routing: 0, least_request_routing: 0, contextual_bandit_routing: 0, 'other': 0}

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
        elif least_kv_cache_routing in strategy.lower():
            color_dict[strategy] = get_strategy_color(strategy, category_counts[least_kv_cache_routing])
            category_counts[least_kv_cache_routing] += 1
        elif least_latency_routing in strategy.lower():
            color_dict[strategy] = get_strategy_color(strategy, category_counts[least_latency_routing])
            category_counts[least_latency_routing] += 1
        elif least_request_routing in strategy.lower():
            color_dict[strategy] = get_strategy_color(strategy, category_counts[least_request_routing])
            category_counts[least_request_routing] += 1
        elif contextual_bandit_routing in strategy.lower():
            color_dict[strategy] = get_strategy_color(strategy, category_counts[contextual_bandit_routing])
            category_counts[contextual_bandit_routing] += 1
        else:
            color_dict[strategy] = get_strategy_color(strategy, category_counts['other'])
            category_counts['other'] += 1
    
    # Create figure with custom GridSpec for better control
    fig = plt.figure(figsize=(18, 32))  # Increased height for 6 rows with more spacing

    # MODIFIED GridSpec: 6 rows (CDFs, TTFT bar chart, TPOT bar chart, overhead bar chart, time series graphs)
    # Increased height_ratio for bar charts and increased hspace for better spacing
    gs = GridSpec(6, 9, figure=fig,
                  height_ratios=[1, 1.5, 1.5, 1.5, 1, 1],
                  hspace=0.9,
                  wspace=0.35)
    
    # fig.suptitle('Routing Strategy Performance Comparison', fontsize=maintitle_fontsize, y=0.98)
    
    # NEW: CDF plots, bar charts, and time series graphs
    if csv_data_dict:
        # Plot 1: TTFT Latency CDF (left half of row 0)
        ax = fig.add_subplot(gs[0, :4])
        plot_latency_cdf(ax, csv_data_dict, strategy_order, color_dict, 'ttft', 'TTFT Latency CDF', 'TTFT (ms)')

        # Plot 2: Avg TPOT Latency CDF (right half of row 0)
        ax = fig.add_subplot(gs[0, 5:])
        plot_latency_cdf(ax, csv_data_dict, strategy_order, color_dict, 'avg_tpot', 'Avg TPOT Latency CDF', 'Avg TPOT (ms)')

        # Plot 3: TTFT Bar Chart (full width, row 1)
        ax = fig.add_subplot(gs[1, :])
        plot_single_metric_comparison(ax, metrics_df, strategy_order, color_dict, 'ttft', 'TTFT Latency Comparison (Avg, P99, P999)')
        
        # Plot 4: TPOT Bar Chart (full width, row 2)
        ax = fig.add_subplot(gs[2, :])
        plot_single_metric_comparison(ax, metrics_df, strategy_order, color_dict, 'tpot', 'Avg TPOT Latency Comparison (Avg, P99, P999)')
        
        # Plot 5: End-to-End Overhead Bar Chart (full width, row 3)
        ax = fig.add_subplot(gs[3, :])
        plot_single_metric_comparison(ax, metrics_df, strategy_order, color_dict, 'end_to_end_overhead',
                                      'End-to-End Overhead Comparison (Avg, P50, P99, P999)')

        # Plot 6: TTFT Time Series (full width, row 4)
        ax = fig.add_subplot(gs[4, :])
        plot_latency_timeseries(ax, csv_data_dict, strategy_order, color_dict, 'ttft', 'TTFT Time Series (1s averages)', 'TTFT (ms)')
        
        # Plot 7: Avg TPOT Time Series (full width, row 5)
        ax = fig.add_subplot(gs[5, :])
        plot_latency_timeseries(ax, csv_data_dict, strategy_order, color_dict, 'avg_tpot', 'Avg TPOT Time Series (1s averages)', 'Avg TPOT (ms)')
    else:
        # If no CSV data provided, show placeholder text for all plots
        for row_idx, plot_cols in [(0, [slice(None, 4), slice(5, None)]), (1, [slice(None)]), (2, [slice(None)]),
                                   (3, [slice(None)]), (4, [slice(None)]), (5, [slice(None)])]:
            if row_idx == 0:
                for col_slice in plot_cols:
                    ax = fig.add_subplot(gs[row_idx, col_slice])
                    ax.text(0.5, 0.5, 'No data available\n(csv_data_dict not provided)',
                           ha='center', va='center', fontsize=12,
                           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                    ax.set_xticks([])
                    ax.set_yticks([])
            else:
                ax = fig.add_subplot(gs[row_idx, :])
                ax.text(0.5, 0.5, 'No data available\n(csv_data_dict not provided)', 
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
    
    # MODIFIED layout parameters for better spacing (more bottom space for large strategy labels)
    plt.subplots_adjust(top=0.96, bottom=0.08, left=0.05, right=0.95)
    
    # Save the figure
    output_file = f"{base_dir}/routing_strategy_comparison_client.pdf"
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
    
    # # Set title with SLO info
    # if slo_value:
    #     full_title = f'{title} Trend Over Time (1s averages) - SLO: {metric_type} ≤ {slo_value}ms'
    # else:
    #     full_title = f'{title} Trend Over Time (1s averages)'
    
    # ax.set_title(full_title, fontsize=subtitle_fontsize+2, pad=15)  # Slightly larger title
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


# New function to plot single metric comparison (TTFT or TPOT) with avg, p99, p999
def plot_single_metric_comparison(ax, metrics_df, strategy_order, color_dict, metric_type, title):
    """Plot bar chart with single metric (TTFT or TPOT) showing avg, p99, p999 grouped by strategy."""
    strategies = [s for s in strategy_order if s in metrics_df['strategy'].values]
    n_strategies = len(strategies)
    if n_strategies == 0:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax.set_xticks([])
        ax.set_yticks([])
        return

    # Extract metrics for each strategy
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
    else:  # end-to-end overhead
        avg_values = [metrics_indexed.loc[s, 'avg_end_to_end_overhead'] if 'avg_end_to_end_overhead' in metrics_df.columns else 0 for s in strategies]
        p99_values = [metrics_indexed.loc[s, 'p99_end_to_end_overhead'] if 'p99_end_to_end_overhead' in metrics_df.columns else 0 for s in strategies]
        p999_values = [metrics_indexed.loc[s, 'p999_end_to_end_overhead'] if 'p999_end_to_end_overhead' in metrics_df.columns else 0 for s in strategies]
        ylabel_text = 'End-to-End Overhead (ms)'

    # Get max value for y-axis scaling (ignore NaN/Inf)
    all_values = np.array((avg_values or []) + (p99_values or []) + (p999_values or []), dtype=float)
    finite_values = all_values[np.isfinite(all_values)]
    if finite_values.size == 0:
        ax.text(0.5, 0.5, 'No valid data available', ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax.set_xticks([])
        ax.set_yticks([])
        return
    max_value = float(np.max(finite_values))

    # Create bar positions - 3 bars per strategy (TTFT/TPOT) or 4 bars (overhead)
    num_bars = 4 if metric_type == 'end_to_end_overhead' else 3
    bar_width = 0.2 if metric_type == 'end_to_end_overhead' else 0.25
    group_width = num_bars * bar_width + 0.3  # Space between groups
    group_centers = np.arange(n_strategies) * group_width

    # Plot bars for each metric (avg, p99, p999)
    for i, strategy in enumerate(strategies):
        strategy_color = color_dict[strategy]
        group_center = group_centers[i]

        # Calculate positions for the 3 bars in each group
        offset_start = -(num_bars - 1) * bar_width / 2

        # Create bars with slight color variations
        bar_sets = [
            (avg_values[i], 'Avg', 0.9),
            (p99_values[i], 'P99', 0.7),
            (p999_values[i], 'P999', 0.5),
        ]
        if metric_type == 'end_to_end_overhead':
            bar_sets = [
                (avg_values[i], 'Avg', 0.9),
                (metrics_indexed.loc[strategy, 'p50_end_to_end_overhead']
                 if 'p50_end_to_end_overhead' in metrics_df.columns else 0, 'P50', 0.75),
                (p99_values[i], 'P99', 0.6),
                (p999_values[i], 'P999', 0.45),
            ]

        for j, (value, label, alpha) in enumerate(bar_sets):
            pos = group_center + offset_start + j * bar_width
            ax.bar(pos, value, bar_width, color=strategy_color,
                   edgecolor='black', linewidth=0.8, alpha=alpha)

            # Add value labels on top of bars
            if np.isfinite(value):
                ax.text(pos, value + max_value * 0.02,
                       f'{value:.0f}', rotation=90, ha='center', va='bottom',
                       fontsize=10, fontweight='bold')

    # Set up x-axis with strategy names
    ax.set_xticks(group_centers)
    
    # Create shorter, multi-line labels for better readability
    strategy_labels = []
    for s in strategies:
        parts = s.split('-')
        if len(parts) >= 2:
            # First line: routing strategy name
            # Second line: timestamp
            label = f"{parts[0]}\n({parts[-1]})"
        else:
            label = s
        strategy_labels.append(label)
    
    ax.set_xticklabels(strategy_labels, fontsize=10, rotation=45, ha='right')

    # Add legend for Avg/P99/P999
    if metric_type == 'end_to_end_overhead':
        legend_elements = [
            Patch(facecolor='gray', edgecolor='black', alpha=0.9, label='Avg'),
            Patch(facecolor='gray', edgecolor='black', alpha=0.75, label='P50'),
            Patch(facecolor='gray', edgecolor='black', alpha=0.6, label='P99'),
            Patch(facecolor='gray', edgecolor='black', alpha=0.45, label='P999')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=14, ncol=4)
    else:
        legend_elements = [
            Patch(facecolor='gray', edgecolor='black', alpha=0.9, label='Avg'),
            Patch(facecolor='gray', edgecolor='black', alpha=0.7, label='P99'),
            Patch(facecolor='gray', edgecolor='black', alpha=0.5, label='P999')
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
                # Position text below x-axis
                ax.text(group_center, -max_value * 0.15, f'n={int(num_requests)}',
                       ha='center', va='top', fontsize=9, style='italic', color='gray')

    # Styling
    ax.set_ylabel(ylabel_text, fontsize=ylabel_fontsize)
    ax.set_title(title, fontsize=subtitle_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    ax.tick_params(axis='x', labelsize=10)
    ax.grid(axis='y', alpha=0.3)
    # Adjust y-axis limits to accommodate num_requests text if present
    if has_num_requests:
        ax.set_ylim(-max_value * 0.2, max(max_value * 1.4, 1.0))
    else:
        ax.set_ylim(0, max(max_value * 1.4, 1.0))


# New function to plot dual-axis comparison (TTFT, TPOT) with avg, p99, p999
def plot_triple_axis_comparison(ax, metrics_df, strategy_order, color_dict):
    """Plot bar chart with TTFT and TPOT (avg, p99, p999) grouped by strategy - dual y-axes (left: TTFT, right: TPOT)."""
    strategies = [s for s in strategy_order if s in metrics_df['strategy'].values]
    n_strategies = len(strategies)

    # Extract metrics for each strategy
    metrics_indexed = metrics_df.set_index('strategy')

    # TTFT metrics
    avg_ttft = [metrics_indexed.loc[s, 'avg_ttft'] if 'avg_ttft' in metrics_df.columns else 0 for s in strategies]
    p99_ttft = [metrics_indexed.loc[s, 'p99_ttft'] if 'p99_ttft' in metrics_df.columns else 0 for s in strategies]
    p999_ttft = [metrics_indexed.loc[s, 'p999_ttft'] if 'p999_ttft' in metrics_df.columns else 0 for s in strategies]

    # TPOT metrics
    avg_tpot = [metrics_indexed.loc[s, 'avg_tpot'] if 'avg_tpot' in metrics_df.columns else 0 for s in strategies]
    p99_tpot = [metrics_indexed.loc[s, 'p99_tpot'] if 'p99_tpot' in metrics_df.columns else 0 for s in strategies]
    p999_tpot = [metrics_indexed.loc[s, 'p999_tpot'] if 'p999_tpot' in metrics_df.columns else 0 for s in strategies]

    # Get max values for each y-axis (across all percentiles)
    max_ttft = max(max(avg_ttft or [0]), max(p99_ttft or [0]), max(p999_ttft or [0]))
    max_tpot = max(max(avg_tpot or [0]), max(p99_tpot or [0]), max(p999_tpot or [0]))

    # Create bar positions - 6 bars per strategy (3 TTFT + 3 TPOT) with spacing between groups
    bar_width = 0.12  # Narrower bars to fit 6 bars per strategy
    group_width = 6 * bar_width + 0.5  # Space between groups
    group_centers = np.arange(n_strategies) * group_width

    # Create x positions for each metric within each group
    all_positions = []
    all_values_ttft = []
    all_values_tpot = []
    all_colors = []
    all_labels = []

    for i, strategy in enumerate(strategies):
        strategy_color = color_dict[strategy]
        group_center = group_centers[i]

        # Calculate positions for the 6 bars in each group
        # Left side: 3 TTFT bars, Right side: 3 TPOT bars
        offset_start = -2.5 * bar_width

        # TTFT bars (avg, p99, p999)
        for j, (value, label) in enumerate([(avg_ttft[i], 'Avg'), (p99_ttft[i], 'P99'), (p999_ttft[i], 'P999')]):
            pos = group_center + offset_start + j * bar_width
            all_positions.append(pos)
            all_values_ttft.append(value)
            all_values_tpot.append(0)  # Placeholder for secondary axis
            all_colors.append(strategy_color)
            all_labels.append(f'TTFT\n{label}')

        # TPOT bars (avg, p99, p999)
        for j, (value, label) in enumerate([(avg_tpot[i], 'Avg'), (p99_tpot[i], 'P99'), (p999_tpot[i], 'P999')]):
            pos = group_center + offset_start + (3 + j) * bar_width
            all_positions.append(pos)
            all_values_ttft.append(0)  # Placeholder for primary axis
            all_values_tpot.append(value)
            all_colors.append(strategy_color)
            all_labels.append(f'TPOT\n{label}')

    # Create TTFT bars on left y-axis
    ttft_bars = []
    for pos, value, color in zip(all_positions, all_values_ttft, all_colors):
        if value > 0:
            bar = ax.bar(pos, value, bar_width, color=color,
                        edgecolor='black', linewidth=0.8, alpha=0.9)
            ttft_bars.append(bar)

            # Add value labels for TTFT
            ax.text(pos, value + max_ttft * 0.03,
                   f'{value:.0f}', rotation=90, ha='center', va='bottom',
                   fontsize=10, color='#222266')

    # Create TPOT bars on right y-axis
    ax2 = ax.twinx()
    tpot_bars = []
    for pos, value, color in zip(all_positions, all_values_tpot, all_colors):
        if value > 0:
            bar = ax2.bar(pos, value, bar_width, color=color,
                         edgecolor='black', linewidth=0.8, alpha=0.6)
            tpot_bars.append(bar)

            # Add value labels for TPOT
            ax2.text(pos, value + max_tpot * 0.03,
                    f'{value:.0f}', rotation=90, ha='center', va='bottom',
                    fontsize=10, color='#226622')

    # Set up x-axis with metric names on each bar
    ax.set_xticks(all_positions)
    ax.set_xticklabels(all_labels, fontsize=8, rotation=45, ha='right')

    # Add strategy names as text annotations below the bars
    strategy_labels = [s.split('-')[0] + "\n(" + s.split('-')[-1] + ")" for s in strategies]
    for i, (center, label) in enumerate(zip(group_centers, strategy_labels)):
        ax.text(center, -0.18, label, ha='center', va='top', fontsize=12,
                transform=ax.get_xaxis_transform(), rotation=45)

    # Styling
    ax.set_ylabel('TTFT (ms)', fontsize=ylabel_fontsize, color='#222266')
    ax2.set_ylabel('Avg TPOT (ms)', fontsize=ylabel_fontsize, color='#226622')
    ax.set_title('Average TTFT and TPOT Latency Comparison (Avg, P99, P999)', fontsize=subtitle_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize, labelcolor='#222266')
    ax2.tick_params(axis='y', labelsize=tick_fontsize, labelcolor='#226622')
    ax2.tick_params(axis='x')
    ax.tick_params(axis='x', which='both', length=5)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max_ttft * 1.6)
    ax2.set_ylim(0, max_tpot * 1.6)

    # Add legend
    legend_elements = [Patch(facecolor=color_dict[s], edgecolor='black',
                             label=s.split('-')[0] + "-" + s.split('-')[-1]) for s in strategies]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12, ncol=min(3, n_strategies))


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
            # Shorten strategy name for legend
            legend_label = strategy.split('-')[0]
            ax.plot(data, y, label=legend_label, color=color_dict[strategy], linewidth=2, alpha=0.8)
    ax.set_title(title, fontsize=subtitle_fontsize)
    ax.set_xlabel(xlabel, fontsize=ylabel_fontsize)
    ax.set_ylabel('CDF', fontsize=ylabel_fontsize)
    ax.legend(fontsize=16, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=tick_fontsize)


# New function to plot latency time series with 1-second averages
def plot_latency_timeseries(ax, csv_data_dict, strategy_order, color_dict, column, title, ylabel):
    """Plot time series with 1-second window averages for a given latency column for each strategy."""
    for strategy in strategy_order:
        if strategy in csv_data_dict and column in csv_data_dict[strategy].columns:
            df = csv_data_dict[strategy]
            
            # Create 1-second time bins
            df_copy = df.copy()
            df_copy['time_bin'] = np.floor(df_copy['relative_time']).astype(int)
            
            # Calculate average latency for each 1-second bin
            latency_stats = df_copy.groupby('time_bin')[column].agg(['mean', 'count']).reset_index()
            latency_stats = latency_stats[latency_stats['count'] > 0]  # Only bins with data
            
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare routing strategies performance')
    parser.add_argument('base_directory', help='Base directory containing log files')
    parser.add_argument('warmup_seconds', nargs='?', type=int, default=None, 
                       help='Seconds to exclude from start for warmup')
    parser.add_argument('cut_last_seconds', nargs='?', type=int, default=None,
                       help='Seconds to exclude from end')
    parser.add_argument('--iteration-from', type=int, default=0,
                       help='Only include rows with iteration >= this value for ML policies')
    parser.add_argument('--average-duplicates', action='store_true',
                       help='Average multiple experiments for the same routing policy')
    
    args = parser.parse_args()
    
    base_dir = args.base_directory
    warmup_seconds = args.warmup_seconds
    cut_last_seconds = args.cut_last_seconds
    iteration_from = args.iteration_from
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
        result = process_log_file(log_file, warmup_seconds, cut_last_seconds, iteration_from)
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
    # print(f"CSV data dict keys: {list(csv_data_dict.keys())}")
    if csv_data_dict:
        first_key = list(csv_data_dict.keys())[0]
        sample_df = csv_data_dict[first_key]
        # print(f"Sample DataFrame columns: {list(sample_df.columns)}")
        # print(f"Sample DataFrame shape: {sample_df.shape}")

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