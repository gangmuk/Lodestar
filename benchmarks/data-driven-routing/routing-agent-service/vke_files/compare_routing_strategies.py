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
# import training.preprocess as preprocess
import preprocess
from matplotlib.gridspec import GridSpec

maintitle_fontsize = 30
subtitle_fontsize = 26
legend_fontsize = 22
text_fontsize = 14
ylabel_fontsize = 22
tick_fontsize = 22

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
    
    # Calculate throughput metrics CORRECTLY - per-second windowing approach
    if 'relative_time' in df.columns:
        # Create 1-second time bins
        df_copy = df.copy()
        df_copy['time_bin'] = np.floor(df_copy['relative_time']).astype(int)
        
        # Calculate per-second metrics
        per_second_stats = df_copy.groupby('time_bin').agg({
            'relative_time': 'count',  # Requests per second
            'numOutputTokens': 'sum' if 'numOutputTokens' in df_copy.columns else lambda x: 0
        }).reset_index()
        
        per_second_stats.columns = ['time_bin', 'requests_per_second', 'tokens_per_second']
        
        # Calculate average throughput across all 1-second windows
        metrics['throughput_rps'] = per_second_stats['requests_per_second'].mean()
        
        if 'numOutputTokens' in df.columns:
            metrics['throughput_tps'] = per_second_stats['tokens_per_second'].mean()
        else:
            metrics['throughput_tps'] = 0
            
        # Optional: Also calculate the experiment-wide average for comparison
        total_duration = df['relative_time'].max() - df['relative_time'].min()
        if total_duration > 0:
            metrics['overall_rps'] = len(df) / total_duration
            if 'numOutputTokens' in df.columns:
                metrics['overall_tps'] = df['numOutputTokens'].sum() / total_duration
    else:
        # Fallback to old method if relative_time not available
        if 'normalized_start_time' in df.columns and 'request_end_time' in df.columns:
            total_duration = (df['request_end_time'].max() - df['request_start_time'].min()) / 1000000
            if total_duration > 0:
                metrics['throughput_rps'] = len(df) / total_duration
                if 'numOutputTokens' in df.columns:
                    metrics['throughput_tps'] = df['numOutputTokens'].sum() / total_duration
            else:
                metrics['throughput_rps'] = 0
                metrics['throughput_tps'] = 0
    
    return metrics

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
    
def plot_routing_comparison(metrics_list, base_dir, slo_ttft, slo_tpot, csv_data_dict=None):
    """Create bar charts comparing performance metrics across routing strategies."""
    if not metrics_list:
        print("No metrics to plot.")
        return
    
    # Convert to DataFrame for easier plotting
    metrics_df = pd.DataFrame(metrics_list)
    
    # Sort strategies by avg_ttft for consistent ordering
    def get_strategy_priority(strategy_name):
        if 'none' in strategy_name.lower():
            return (0, strategy_name)  # First priority
        elif 'prefix-cache' in strategy_name.lower():
            return (1, strategy_name)  # Second priority
        elif 'random' in strategy_name.lower():
            return (2, strategy_name)  # Third priority
        else:
            return (3, strategy_name)  # Others last

    # Sort strategies by custom priority
    all_strategies = metrics_df['strategy'].tolist()
    strategy_order = sorted(all_strategies, key=get_strategy_priority)

    # Set up colors by category
    def get_strategy_color(strategy_name, strategies_in_category, index_in_category):
        """Get color for strategy based on category and index within category"""
        if 'none' in strategy_name.lower():
            # Purple family
            # base_colors = ['#8b008b','#ba55d3', '#9932cc', '#8a2be2',  '#c71585']
            # Red family
            base_colors = ['#ff0000', '#dc143c', '#ff6347', '#ff4500', '#ff7f50']
        elif 'prefix-cache' in strategy_name.lower():
            # Blue family
            base_colors = ['#1f77b4', '#4682b4', '#6495ed', '#aec7e8', '#87ceeb']
            # Orange family  
            # base_colors = ['#ff7f0e', '#ffa500', '#ff8c00', '#ffbb78', '#ffb347']
        elif 'random' in strategy_name.lower():
            # Green family
            base_colors = ['#2ca02c', '#32cd32', '#00ff00', '#00ff7f', '#98df8a']
        else:
            # Gray family for others
            base_colors = ['#7f7f7f', '#696969', '#a9a9a9', '#c7c7c7', '#d3d3d3']
        
        # Use modulo to cycle through colors if more strategies than colors
        return base_colors[index_in_category % len(base_colors)]

    # Create color dictionary with grouped coloring
    color_dict = {}
    category_counts = {'none': 0, 'prefix-cache': 0, 'random': 0, 'other': 0}

    for strategy in strategy_order:
        if 'none' in strategy.lower():
            color_dict[strategy] = get_strategy_color(strategy, None, category_counts['none'])
            category_counts['none'] += 1
        elif 'prefix-cache' in strategy.lower():
            color_dict[strategy] = get_strategy_color(strategy, None, category_counts['prefix-cache'])
            category_counts['prefix-cache'] += 1
        elif 'random' in strategy.lower():
            color_dict[strategy] = get_strategy_color(strategy, None, category_counts['random'])
            category_counts['random'] += 1
        else:
            color_dict[strategy] = get_strategy_color(strategy, None, category_counts['other'])
            category_counts['other'] += 1
    
    # Create figure with custom GridSpec for better control
    fig = plt.figure(figsize=(24, 24))  # Adjusted height for 6 rows

    # MODIFIED GridSpec: 6 rows (bar charts, rewards, CDFs, avg)
    gs = GridSpec(6, 6, figure=fig,
                  height_ratios=[0.8, 1, 1, 1, 1, 1],
                  hspace=0.6,
                  wspace=0.35)
    
    fig.suptitle('Routing Strategy Performance Comparison', fontsize=maintitle_fontsize, y=0.96)
    
    # FIRST ROW: All 6 bar chart plots
    # Plot 1: Average TTFT
    if 'avg_ttft' in metrics_df.columns:
        ax = fig.add_subplot(gs[0, 0])
        plot_metric_bar(ax, metrics_df, 'avg_ttft', 'Average TTFT (ms)', 
                        strategy_order, color_dict)

    # Plot 2: P99 TTFT
    if 'p99_ttft' in metrics_df.columns:
        ax = fig.add_subplot(gs[0, 1])
        plot_metric_bar(ax, metrics_df, 'p99_ttft', 'P99 TTFT (ms)', 
                        strategy_order, color_dict)

    # Plot 3: Average TPOT
    if 'avg_tpot' in metrics_df.columns:
        ax = fig.add_subplot(gs[0, 2])
        plot_metric_bar(ax, metrics_df, 'avg_tpot', 'Average TPOT (ms)', 
                        strategy_order, color_dict)

    # Plot 4: P99 TPOT
    if 'p99_tpot' in metrics_df.columns:
        ax = fig.add_subplot(gs[0, 3])
        plot_metric_bar(ax, metrics_df, 'p99_tpot', 'P99 TPOT (ms)', 
                        strategy_order, color_dict)

    # Plot 5: Throughput (Requests per Second)
    if 'throughput_rps' in metrics_df.columns:
        ax = fig.add_subplot(gs[0, 4])
        plot_metric_bar(ax, metrics_df, 'throughput_rps', 'Throughput (Requests/sec)', 
                        strategy_order, color_dict)

    # Plot 6: Token Throughput
    if 'throughput_tps' in metrics_df.columns:
        ax = fig.add_subplot(gs[0, 5])
        plot_metric_bar(ax, metrics_df, 'throughput_tps', 'Throughput (Tokens/sec)', 
                        strategy_order, color_dict)
    
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

        # Plot 10: TTFT Latency CDF (left half of row 4)
        ax = fig.add_subplot(gs[4, :3])
        plot_latency_cdf(ax, csv_data_dict, strategy_order, color_dict, 'ttft', 'TTFT Latency CDF', 'TTFT (ms)')

        # Plot 11: Avg TPOT Latency CDF (right half of row 4)
        ax = fig.add_subplot(gs[4, 3:])
        plot_latency_cdf(ax, csv_data_dict, strategy_order, color_dict, 'avg_tpot', 'Avg TPOT Latency CDF', 'Avg TPOT (ms)')

        # Plot 12: Average TTFT and Average TPOT Comparison (full width, now row 5)
        ax = fig.add_subplot(gs[5, :])
        plot_avg_ttft_tpot_comparison(ax, metrics_df, strategy_order, color_dict)
    else:
        # If no CSV data provided, show placeholder text for reward plots
        for row in [1, 2, 3, 4, 5]:
            if row == 4:
                # CDFs: left and right
                ax = fig.add_subplot(gs[4, :3])
                ax.text(0.5, 0.5, 'No time series data available\n(csv_data_dict not provided)', 
                        ha='center', va='center', fontsize=12, 
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                ax.set_xticks([])
                ax.set_yticks([])
                ax = fig.add_subplot(gs[4, 3:])
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


# New function to plot average TTFT and average TPOT comparison
def plot_avg_ttft_tpot_comparison(ax, metrics_df, strategy_order, color_dict):
    """Plot bar chart with double y-axis: left for avg TTFT, right for avg TPOT across strategies."""
    strategies = [s for s in strategy_order if s in metrics_df['strategy'].values]
    label_list = []
    for s in strategies:
        len_s = len(s)
        label_list.append(f"{s[:len_s//2]}\n{s[len_s//2:]}")
    n_strategies = len(strategies)
    avg_ttft = [metrics_df.set_index('strategy').loc[s, 'avg_ttft'] if 'avg_ttft' in metrics_df.columns else 0 for s in strategies]
    avg_tpot = [metrics_df.set_index('strategy').loc[s, 'avg_tpot'] if 'avg_tpot' in metrics_df.columns else 0 for s in strategies]
    x = np.arange(n_strategies)
    bar_width = 0.6
    strategy_colors = [color_dict[s] for s in strategies]

    # Bars for TTFT (left y-axis)
    bars1 = ax.bar(x - bar_width/4, avg_ttft, bar_width/2, label='Avg TTFT (ms)', color=strategy_colors, alpha=0.9, edgecolor='black', linewidth=1)
    ax.set_ylabel('Avg TTFT (ms)', fontsize=ylabel_fontsize, color='#222266')
    ax.tick_params(axis='y', labelcolor='#222266', labelsize=tick_fontsize)

    # Add value labels for TTFT
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{height:.0f}', rotation=90, ha='center', va='bottom', fontsize=14, fontweight='bold', color='#222266')

    # Twin axis for TPOT (right y-axis)
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + bar_width/4, avg_tpot, bar_width/2, label='Avg TPOT (ms)', color=strategy_colors, alpha=0.5, edgecolor='black', linewidth=1)
    ax2.set_ylabel('Avg TPOT (ms)', fontsize=ylabel_fontsize, color='#226622')
    ax2.tick_params(axis='y', labelcolor='#226622', labelsize=tick_fontsize)

    # Add value labels for TPOT
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{height:.0f}', rotation=90, ha='center', va='bottom', fontsize=14, color='#226622')

    # X-axis and title
    ax.set_title('Average TTFT (left) and TPOT (right) Comparison', fontsize=subtitle_fontsize)
    ax.set_xlabel('Routing Strategy', fontsize=ylabel_fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(label_list, rotation=45, ha='right', fontsize=tick_fontsize)
    ax.grid(axis='y', alpha=0.3)
    ax.set_zorder(2)
    ax.patch.set_visible(False)
    # Set y-axis limits with padding
    max_ttft = max(avg_ttft or [0])
    max_tpot = max(avg_tpot or [0])
    ax.set_ylim(0, max_ttft * 1.4 if max_ttft > 0 else 1)
    ax2.set_ylim(0, max_tpot * 1.4 if max_tpot > 0 else 1)
    
    
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
    
    # Sort by strategy order
    plot_data = metrics_df.set_index('strategy').loc[strategy_order, [metric]]
    
    # Use simple index numbers for x-axis
    bar_positions = np.arange(len(plot_data))
    
    # Create bar chart with bars
    bars = ax.bar(bar_positions, plot_data[metric], 
                  color=[color_dict[s] for s in strategy_order],
                  width=0.8,
                  )
    
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
    
    # Set y-axis limits first to provide space for text labels
    max_bar_height = plot_data[metric].max()

    # Add value labels on top of each bar with relative performance
    for i, bar in enumerate(bars):
        height = bar.get_height()
        relative_perf = relative_values.iloc[i]
        
        # # Format the annotation text - smaller font for narrow plots
        # if relative_perf == 1.0:
        #     annotation_text = f'{height:.0f}\n(1x)'
        # else:
        annotation_text = f'{height:.0f} ({relative_perf:.1f}x)'
        
        # Calculate dynamic offset to prevent overflow
        # y_max = ax.get_ylim()[1]
        ax.annotate(annotation_text,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 0),
                    textcoords="offset points",
                    ha='center', va='bottom', rotation=90,
                    fontsize=text_fontsize-2, color='black')  # Smaller font
    
    ax.set_ylim(0, max_bar_height * 1.6)  # Provide 40% extra space above bars
    
    # Set chart titles and labels - adjusted for narrow plots
    ax.set_title(title, fontsize=subtitle_fontsize-2, pad=8)  # Smaller title and padding
    ax.set_ylabel(title.split('(')[0].strip(), fontsize=ylabel_fontsize-2)  # Smaller ylabel
    
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
    if len(sys.argv) < 2:
        print("Usage: python compare_routing_strategies.py <base_directory>")
        print("Example: python compare_routing_strategies.py filtered_logs/chatbot-simulation")
        sys.exit(1)

    base_dir = sys.argv[1]
    print(f"Searching for log files in {base_dir}...")

    warmup_seconds = None
    cut_last_seconds = None
    if len(sys.argv) >= 3:
        warmup_seconds = int(sys.argv[2])
        print(f"warmup_seconds: {warmup_seconds} seconds")
    if len(sys.argv) >= 4:
        cut_last_seconds = int(sys.argv[3])
        print(f"cut_last_seconds: {cut_last_seconds} seconds")
    
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
    
    # Plot the comparison - MODIFIED to pass csv_data_dict
    plot_routing_comparison(all_metrics, base_dir, slo_ttft, slo_tpot, csv_data_dict)