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
import training.preprocess as preprocess

maintitle_fontsize = 20
subtitle_fontsize = 18
legend_fontsize = 14
text_fontsize = 14
ylabel_fontsize = 14
tick_fontsize = 14

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
    """Recursively find all filtered log CSV files in the base directory."""
    pattern = os.path.join(base_dir, "**", "filtered-aibrix-gateway-plugins.log.csv")
    return glob.glob(pattern, recursive=True)

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
    
    # Calculate throughput
    if 'normalized_start_time' in df.columns and 'request_end_time' in df.columns:
        # Calculate duration in seconds
        total_duration = (df['request_end_time'].max() - df['request_start_time'].min()) / 1000000
        if total_duration > 0:
            metrics['throughput_rps'] = len(df) / total_duration
        else:
            metrics['throughput_rps'] = 0
    
    # Calculate output token throughput if available
    if 'normalized_start_time' in df.columns and 'numOutputTokens' in df.columns:
        total_output_tokens = df['numOutputTokens'].sum()
        if 'request_end_time' in df.columns and 'request_start_time' in df.columns:
            total_duration = (df['request_end_time'].max() - df['request_start_time'].min()) / 1000000
            if total_duration > 0:
                metrics['throughput_tps'] = total_output_tokens / total_duration
            else:
                metrics['throughput_tps'] = 0
    
    return metrics
def process_log_file(file_path, warmup_seconds, cut_last_seconds):
    """Process a single log file and return its performance metrics AND the processed DataFrame."""
    print(f"Processing {file_path}...")
    df, json_columns = preprocess.parse_log_file(file_path)
    df = preprocess.parse_json_columns(df, json_columns)
    df = preprocess.normalize_time(df)
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
    if 'avg_ttft' in metrics_df.columns:
        strategy_order = metrics_df.sort_values('avg_ttft')['strategy'].tolist()
    else:
        strategy_order = metrics_df['strategy'].tolist()
    
    # Set up colors for each strategy
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(strategy_order)))
    color_dict = dict(zip(strategy_order, colors))
    
    # Create figure with custom GridSpec for better control
    fig = plt.figure(figsize=(24, 20))  # INCREASED height from 16 to 20
    
    # MODIFIED GridSpec with larger spacing between time series plots
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(5, 6, figure=fig, 
                  height_ratios=[0.8, 1, 1, 1, 1],  # Made bar charts slightly shorter
                  hspace=0.6,  # INCREASED from 0.4 to 0.6 for more vertical spacing
                  wspace=0.25)
    
    fig.suptitle('Routing Strategy Performance Comparison', fontsize=maintitle_fontsize, y=0.96)
    
    # ... rest of your existing subplot code remains the same ...
    
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
        # Calculate rewards and SLO satisfaction for each strategy
        strategy_slo_stats = {}
        for strategy in strategy_order:
            if strategy in csv_data_dict:
                df = csv_data_dict[strategy]
                # Add reward columns
                df['ttft_reward'] = df['ttft'].apply(lambda x: calculate_ttft_reward(x, slo_ttft))
                df['tpot_reward'] = df['avg_tpot'].apply(lambda x: calculate_tpot_reward(x, slo_tpot))
                df['total_reward'] = df['ttft_reward'] + df['tpot_reward']
                
                # Calculate SLO satisfaction
                strategy_slo_stats[strategy] = calculate_slo_satisfaction(df, slo_ttft, slo_tpot)
        
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
        
        # Plot 10: SLO Satisfaction Comparison (full width)
        ax = fig.add_subplot(gs[4, :])  # Full width of row 4
        plot_slo_satisfaction_comparison(ax, strategy_slo_stats, strategy_order, 
                                        color_dict, slo_ttft, slo_tpot)
    else:
        # If no CSV data provided, show placeholder text for reward plots
        for row in [1, 2, 3, 4]:
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
    fig.legend(handles, legend_labels, 
              loc='lower center', 
              bbox_to_anchor=(0.5, 0.02),
              fontsize=legend_fontsize, ncol=len(strategy_order), 
              title="Routing Strategies")
    
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

            ax.scatter(df['relative_time'], df[reward_column], 
                      s=8, alpha=0.3, color=color, 
                      label=f'{strategy} (individual)', zorder=1)
            
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
    
    # Position legend in upper right
    ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
    
    # Add some additional reference lines for better interpretation
    if 'total' not in reward_column.lower():
        # Add SLO satisfaction threshold lines for individual rewards
        if metric_type == 'TTFT':
            ax.axhline(y=0.1, color='green', linestyle=':', alpha=0.7, linewidth=1.5)
        elif metric_type == 'TPOT':
            ax.axhline(y=0.5, color='green', linestyle=':', alpha=0.7, linewidth=1.5)

def plot_slo_satisfaction_comparison(ax, strategy_slo_stats, strategy_order, color_dict, slo_ttft, slo_tpot):
    """Plot grouped bar chart comparing SLO satisfaction across strategies."""
    
    if not strategy_slo_stats:
        ax.text(0.5, 0.5, 'No SLO satisfaction data available', 
               ha='center', va='center', fontsize=12)
        ax.set_title('SLO Satisfaction Comparison', fontsize=subtitle_fontsize)
        return
    
    # Prepare data
    strategies = [s for s in strategy_order if s in strategy_slo_stats]
    n_strategies = len(strategies)
    
    ttft_counts = [strategy_slo_stats[s]['ttft_satisfied'] for s in strategies]
    tpot_counts = [strategy_slo_stats[s]['tpot_satisfied'] for s in strategies]
    both_counts = [strategy_slo_stats[s]['both_satisfied'] for s in strategies]
    
    # Set up bar positions
    x = np.arange(n_strategies)
    width = 0.1
    
    # Create grouped bars
    bars1 = ax.bar(x - width, ttft_counts, width, label='TTFT SLO', 
                   color='blue', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x, tpot_counts, width, label='TPOT SLO', 
                   color='green', alpha=0.7, edgecolor='black')
    bars3 = ax.bar(x + width, both_counts, width, label='Both SLOs', 
                   color='red', alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    def add_value_labels(bars, stat_key):
        for i, bar in enumerate(bars):
            height = bar.get_height()
            strategy = strategies[i]
            rate = strategy_slo_stats[strategy][stat_key]
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{int(height)}\n({rate:.1f}%)', 
                   ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    add_value_labels(bars1, 'ttft_satisfaction_rate')
    add_value_labels(bars2, 'tpot_satisfaction_rate')
    add_value_labels(bars3, 'both_satisfaction_rate')
    
    # Customize the plot
    ax.set_title(f'SLO Satisfaction Comparison\n(TTFT≤{slo_ttft}ms, TPOT≤{slo_tpot}ms)', 
                fontsize=subtitle_fontsize)
    ax.set_xlabel('Routing Strategy', fontsize=ylabel_fontsize)
    ax.set_ylabel('Number of Requests', fontsize=ylabel_fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45, ha='right', fontsize=tick_fontsize)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    
    # Set y-axis limit with padding
    if ttft_counts or tpot_counts or both_counts:
        max_requests = max(max(ttft_counts or [0]), max(tpot_counts or [0]), max(both_counts or [0]))
        ax.set_ylim(0, max_requests * 1.15)


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
                  color=[color_dict[s] for s in strategy_order])
    
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
        
        # Format the annotation text - smaller font for narrow plots
        if relative_perf == 1.0:
            annotation_text = f'{height:.1f}\n(1x)'
        else:
            annotation_text = f'{height:.1f}\n({relative_perf:.1f}x)'
        
        # Calculate dynamic offset to prevent overflow
        y_max = ax.get_ylim()[1]
        if height > y_max * 0.85:  # If bar is too tall, put text inside the bar
            y_offset = -25
            va_alignment = 'top'
            text_color = 'magenta'
        else:
            y_offset = 3
            va_alignment = 'bottom'
            text_color = 'magenta'
        
        ax.annotate(annotation_text,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, y_offset),
                    textcoords="offset points",
                    ha='center', va=va_alignment,
                    fontsize=text_fontsize-2, color=text_color)  # Smaller font
    
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