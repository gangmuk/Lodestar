import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import json
from datetime import datetime
import sys
import pandas as pd
import json
import logging
import time


logger = logging.getLogger(__name__)
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def parse_log_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Check if this is a metrics line
            if "latency_metrics" not in line:
                logger.error(f"Invalid line. {line}")
                assert False
            if "**@" in line:
                line = line.split("**@latency_metrics@")[1]
            parts = line.split('@')
            row = {}
            json_columns = list()
            column_names = list()
            for i in range(0, len(parts), 2):
                column_name = parts[i]
                column_names.append(column_name)
                value = parts[i+1]
                if value.startswith('{') and value.endswith('}'):
                    try:
                        json_columns.append(column_name)
                        row[column_name] = json.loads(value) # this is going to be dictionary
                    except json.JSONDecodeError:
                        logger.error(f"Error decoding JSON: {value}")
                else:
                    try:
                        row[column_name] = int(value)
                    except ValueError:
                        try:
                            row[column_name] = float(value)
                        except ValueError:
                            row[column_name] = value
            data.append(row)
    def parse_json_columns(df, json_columns):
        for col in json_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        return df
    df = pd.DataFrame(data, columns=column_names)
    df = parse_json_columns(df, json_columns)

    return df


def analyze_llm_inference_logs(df):
    if df.empty:
        print("No valid data found in the log file.")
        return
    
    print("Available columns:", df.columns)
    
    # Basic statistics
    print(f"Total requests: {len(df)}")
    
    # Calculate experiment duration
    if 'request_start_time' in df.columns and 'request_end_time' in df.columns:
        start_time = df['request_start_time'].min()
        end_time = df['request_end_time'].max()
        print(f"Experiment duration: {(end_time - start_time) / 1000000:.2f} seconds")
    else:
        print("Start and end time columns not found.")
    
    df['selectedpod'] = df['selectedpod'].str.split(':').str[0]

    # Process KV cache hit ratios
    if 'allPodsKvCacheHitRatios' in df.columns:
        df['selected_pod_kv_cache_hit_ratio'] = df.apply(
            lambda row: row['allPodsKvCacheHitRatios'].get(row['selectedpod'], 0) 
            if isinstance(row['allPodsKvCacheHitRatios'], dict) else 0, 
            axis=1
        )
    
    # Process GPU KV cache usage
    if 'vllmGPUKVCacheUsage' in df.columns:
        df['selected_pod_vllm_gpu_kv_cache_usage'] = df.apply(
            lambda row: row['vllmGPUKVCacheUsage'].get(row['selectedpod'], 0) 
            if isinstance(row['vllmGPUKVCacheUsage'], dict) else 0, 
            axis=1
        )
    
    # Process CPU KV cache usage
    if 'vllmCPUKVCacheUsage' in df.columns:
        df['selected_pod_vllm_cpu_kv_cache_usage'] = df.apply(
            lambda row: row['vllmCPUKVCacheUsage'].get(row['selectedpod'], 0) 
            if isinstance(row['vllmCPUKVCacheUsage'], dict) else 0, 
            axis=1
        )
    
    # Process inflight requests
    if 'numInflightRequestsAllPods' in df.columns:
        df['total_num_inflight_requests'] = df.apply(
            lambda row: sum(row['numInflightRequestsAllPods'].values()) 
            if isinstance(row['numInflightRequestsAllPods'], dict) else 0, 
            axis=1
        )
        df['selected_pod_num_inflight_requests'] = df.apply(
            lambda row: row['numInflightRequestsAllPods'].get(row['selectedpod'], 0) 
            if isinstance(row['numInflightRequestsAllPods'], dict) else 0, 
            axis=1
        )
    
    # Process running requests
    if 'vllmNumRequestsRunning' in df.columns:
        df['total_vllm_num_running_requests'] = df.apply(
            lambda row: sum(row['vllmNumRequestsRunning'].values()) 
            if isinstance(row['vllmNumRequestsRunning'], dict) else 0, 
            axis=1
        )
        df['selected_pod_num_running_requests'] = df.apply(
            lambda row: row['vllmNumRequestsRunning'].get(row['selectedpod'], 0) 
            if isinstance(row['vllmNumRequestsRunning'], dict) else 0, 
            axis=1
        )
    
    # Process waiting requests
    if 'vllmNumRequestsWaiting' in df.columns:
        df['total_vllm_num_waiting_requests'] = df.apply(
            lambda row: sum(row['vllmNumRequestsWaiting'].values()) 
            if isinstance(row['vllmNumRequestsWaiting'], dict) else 0, 
            axis=1
        )
        df['selected_pod_num_waiting_requests'] = df.apply(
            lambda row: row['vllmNumRequestsWaiting'].get(row['selectedpod'], 0) 
            if isinstance(row['vllmNumRequestsWaiting'], dict) else 0, 
            axis=1
        )
    
    # Map column names from new format to old format
    column_mapping = {
        'ttft': 'gateway_side_ttft',
        'avg_tpot': 'gateway_side_tpot',
        'e2e': 'gateway_side_e2e_latency',
        'numInputTokens': 'prompt_tokens',
        'numOutputTokens': 'output_tokens',
        'numTotalTokens': 'total_tokens'
    }
    
    # Rename columns
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df[new_col] = df[old_col]
    
    # Create plots as in the original script    
    return df

def analyze_pod_metrics_last_second(df):
    """Extract and analyze pod metrics from the last second window"""
    
    if 'podMetricsLastSecond' not in df.columns:
        print("No pod metrics last second data found in the DataFrame")
        return df
    
    # Initialize new columns for the selected pod metrics
    metrics_to_extract = [
        'avg_ttft_ms', 'min_ttft_ms', 'max_ttft_ms', 'p50_ttft_ms', 'p90_ttft_ms', 'p95_ttft_ms', 'p99_ttft_ms',
        'avg_tpot_ms', 'min_tpot_ms', 'max_tpot_ms', 'p50_tpot_ms', 'p90_tpot_ms', 'p95_tpot_ms', 'p99_tpot_ms',
        'early_tokens_tpot_ms', 'mid_tokens_tpot_ms', 'late_tokens_tpot_ms',
        'ttft_samples', 'tpot_samples', 'total_requests', 'total_tokens'
    ]
    
    for metric in metrics_to_extract:
        df[f'selected_pod_{metric}'] = None
        print(f"Adding column: selected_pod_{metric}")
    
    # Also calculate cluster-wide averages
    for metric in metrics_to_extract:
        df[f'cluster_avg_{metric}'] = None
    
    # Process each row
    for idx, row in df.iterrows():
        selected_pod = row['selectedpod']
        pod_metrics = row['podMetricsLastSecond']
        
        # Extract selected pod metrics
        if selected_pod in pod_metrics:
            for metric in metrics_to_extract:
                if metric in pod_metrics[selected_pod]:
                    df.at[idx, f'selected_pod_{metric}'] = pod_metrics[selected_pod][metric]
        
        # Calculate cluster-wide averages for numeric metrics
        for metric in metrics_to_extract:
            values = []
            for pod, metrics in pod_metrics.items():
                if metric in metrics and isinstance(metrics[metric], (int, float)) and metrics[metric] > 0:
                    values.append(metrics[metric])
            
            if values:
                df.at[idx, f'cluster_avg_{metric}'] = sum(values) / len(values)
    
    # Calculate additional metrics
    # TTFT variance across cluster
    df['ttft_variance_across_cluster'] = df.apply(
        lambda row: calculate_variance(row['podMetricsLastSecond'], 'avg_ttft_ms'),
        axis=1
    )
    
    # TPOT variance across cluster
    df['tpot_variance_across_cluster'] = df.apply(
        lambda row: calculate_variance(row['podMetricsLastSecond'], 'avg_tpot_ms'),
        axis=1
    )
    
    # Total active pods (pods handling requests)
    df['active_pods_count'] = df.apply(
        lambda row: sum(1 for pod, metrics in row['podMetricsLastSecond'].items() 
                      if 'total_requests' in metrics and metrics['total_requests'] > 0),
        axis=1
    )
    
    return df

def calculate_variance(pod_metrics, metric_name):
    """Calculate variance of a metric across all pods"""
    values = []
    for pod, metrics in pod_metrics.items():
        if metric_name in metrics and isinstance(metrics[metric_name], (int, float)) and metrics[metric_name] > 0:
            values.append(metrics[metric_name])
    
    if len(values) <= 1:
        return 0
    
    mean = sum(values) / len(values)
    variance = sum((x - mean)**2 for x in values) / len(values)
    return variance



def create_simple_rps_plots(df):
    """Create simplified RPS (Requests Per Second) plots with all subfigures in one row"""
    
    # Determine which plots we can create based on available columns
    plots_to_create = []
    
    if 'normalized_start_time' in df.columns:
        plots_to_create.append('rps')
    
    if 'normalized_start_time' in df.columns and 'selectedpod' in df.columns:
        plots_to_create.append('pod_rps')
    
    if 'normalized_start_time' in df.columns and 'prompt_tokens' in df.columns:
        plots_to_create.append('prompt_tps')
    
    if 'normalized_start_time' in df.columns and 'output_tokens' in df.columns:
        plots_to_create.append('output_tps')
    
    if not plots_to_create:
        return
    
    df_copy = df.copy()    
    # Get max time for consistent x-axis limits
    max_time = df_copy['normalized_end_time'].max() if 'normalized_end_time' in df_copy.columns else None
    if max_time is not None and np.isnan(max_time):
        max_time = None
    
    # Create subplots in a single row
    fig, axes = plt.subplots(1, len(plots_to_create), figsize=(6*len(plots_to_create), 5))
    fig.suptitle('Throughput Metrics Over Time', fontsize=16)
    
    # Make axes iterable if only one plot
    if len(plots_to_create) == 1:
        axes = [axes]
    
    # Create each plot
    for i, plot_type in enumerate(plots_to_create):
        ax = axes[i]
        
        if plot_type == 'rps':
            # Basic RPS plot
            rps = df_copy.groupby('time_bucket').size().reset_index(name='count')
            ax.plot(rps['time_bucket'], rps['count'])
            ax.set_title('Requests per Second')
            ax.set_ylabel('Requests per second (RPS)')
            if len(rps) > 0:
                ax.set_ylim(0, rps['count'].max() * 1.1)
                
        elif plot_type == 'pod_rps':
            # RPS by pod
            pod_rps = df_copy.groupby(['time_bucket', 'selectedpod']).size().reset_index(name='count')
            for pod in pod_rps['selectedpod'].unique():
                pod_data = pod_rps[pod_rps['selectedpod'] == pod]
                ax.plot(pod_data['time_bucket'], pod_data['count'], label=pod)
            ax.set_title('RPS by Pod')
            ax.set_ylabel('Requests per second (RPS)')
            ax.legend()
            if len(pod_rps) > 0:
                ax.set_ylim(0, pod_rps['count'].max() * 1.1)
                
        elif plot_type == 'prompt_tps':
            # Prompt tokens per second
            prompt_tps = df_copy.groupby('time_bucket')['prompt_tokens'].sum().reset_index(name='count')
            ax.plot(prompt_tps['time_bucket'], prompt_tps['count'])
            ax.set_title('Prompt Tokens per Second')
            ax.set_ylabel('Tokens per Second')
            if len(prompt_tps) > 0:
                ax.set_ylim(0, prompt_tps['count'].max() * 1.1)
                
        elif plot_type == 'output_tps':
            # Output tokens per second
            output_tps = df_copy.groupby('time_bucket')['output_tokens'].sum().reset_index(name='count')
            ax.plot(output_tps['time_bucket'], output_tps['count'])
            ax.set_title('Output Tokens per Second')
            ax.set_ylabel('Tokens per Second')
            if len(output_tps) > 0:
                ax.set_ylim(0, output_tps['count'].max() * 1.1)
        
        # Common settings for all plots
        ax.set_xlabel('Time (s)')
        ax.grid(True)
        
        # Set x-axis limits if we have valid max_time
        if max_time is not None:
            ax.set_xlim(0, max_time)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Adjust for the suptitle
    plt.savefig("rps_plots.pdf")
    print(f"* Saved RPS plots to rps_plots.pdf")
    plt.show()

def create_e2e_latency_correlation_plots(df):
    """Create scatter plots for correlations between E2E latency and different metrics as subfigures with 4 per row"""
    
    # Define the correlation plots we want to create
    correlation_plots = []
    
    # Plot 1: E2E latency vs KV Cache Hit Ratio
    if 'gateway_side_e2e_latency' in df.columns and 'selected_pod_kv_cache_hit_ratio' in df.columns:
        correlation_plots.append({
            'x': 'selected_pod_kv_cache_hit_ratio',
            'y': 'gateway_side_e2e_latency',
            'title': 'E2E Latency vs KV Cache Hit Ratio',
            'xlabel': 'KV Cache Hit Ratio',
            'ylabel': 'Gateway Side E2E Latency (ms)',
            'xlim': (0, 1)
        })
    
    # Plot 2: E2E latency vs GPU KV Cache Usage
    if 'gateway_side_e2e_latency' in df.columns and 'selected_pod_vllm_gpu_kv_cache_usage' in df.columns:
        correlation_plots.append({
            'x': 'selected_pod_vllm_gpu_kv_cache_usage',
            'y': 'gateway_side_e2e_latency',
            'title': 'E2E Latency vs GPU KV Cache Usage',
            'xlabel': 'vLLM GPU Cache Usage %',
            'ylabel': 'Gateway Side E2E Latency (ms)',
            'xlim': (0, 1)
        })
    
    # Plot 3: E2E Latency vs Number of Running Requests
    if 'gateway_side_e2e_latency' in df.columns and 'selected_pod_num_running_requests' in df.columns:
        correlation_plots.append({
            'x': 'selected_pod_num_running_requests',
            'y': 'gateway_side_e2e_latency',
            'title': 'E2E Latency vs Running Requests',
            'xlabel': 'Number of Running Requests',
            'ylabel': 'E2E Latency (ms)',
            'xlim': (0, None)
        })
    
    # Plot 4: E2E Latency vs Number of Waiting Requests
    if 'gateway_side_e2e_latency' in df.columns and 'selected_pod_num_waiting_requests' in df.columns:
        correlation_plots.append({
            'x': 'selected_pod_num_waiting_requests',
            'y': 'gateway_side_e2e_latency',
            'title': 'E2E Latency vs Waiting Requests',
            'xlabel': 'Number of Waiting Requests',
            'ylabel': 'E2E Latency (ms)',
            'xlim': (0, None)
        })
    
    # Plot 5: E2E Latency vs Number of Input Tokens
    if 'gateway_side_e2e_latency' in df.columns and 'prompt_tokens' in df.columns:
        correlation_plots.append({
            'x': 'prompt_tokens',
            'y': 'gateway_side_e2e_latency',
            'title': 'E2E Latency vs Input Tokens',
            'xlabel': 'Number of Input Tokens',
            'ylabel': 'E2E Latency (ms)',
            'xlim': (0, None)
        })
    
    # Plot 6: E2E Latency vs Number of Output Tokens
    if 'gateway_side_e2e_latency' in df.columns and 'output_tokens' in df.columns:
        correlation_plots.append({
            'x': 'output_tokens',
            'y': 'gateway_side_e2e_latency',
            'title': 'E2E Latency vs Output Tokens',
            'xlabel': 'Number of Output Tokens',
            'ylabel': 'E2E Latency (ms)',
            'xlim': (0, None)
        })
    
    # Plot 7: E2E Latency vs Total Tokens
    if 'gateway_side_e2e_latency' in df.columns and 'total_tokens' in df.columns:
        correlation_plots.append({
            'x': 'total_tokens',
            'y': 'gateway_side_e2e_latency',
            'title': 'E2E Latency vs Total Tokens',
            'xlabel': 'Total Number of Tokens',
            'ylabel': 'E2E Latency (ms)',
            'xlim': (0, None)
        })
    
    # If no correlation plots are available, return
    if not correlation_plots:
        return
    
    # Calculate number of rows needed (4 plots per row)
    num_plots = len(correlation_plots)
    num_rows = (num_plots + 3) // 4  # Ceiling division
    
    # Create the figure and subplots
    fig, axes = plt.subplots(num_rows, 4, figsize=(20, 5 * num_rows))
    fig.suptitle('E2E Latency Correlation Analysis', fontsize=16)
    
    # Make axes a 2D array even if it's a 1D array (single row) or a single subplot
    if num_plots == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = np.array([axes])
    
    # Create the correlation plots
    for i, plot_config in enumerate(correlation_plots):
        row = i // 4
        col = i % 4
        
        ax = axes[row, col]
        
        x_data = df[plot_config['x']]
        y_data = df[plot_config['y']]
        
        # Add a scatter plot
        ax.scatter(x_data, y_data, alpha=0.5)
        
        # # Add trend line if there are enough points
        # if len(x_data) > 1:
        #     # Calculate trend line
        #     z = np.polyfit(x_data, y_data, 1)
        #     p = np.poly1d(z)
            
        #     # Add trend line to plot
        #     x_trend = np.linspace(min(x_data), max(x_data), 100)
        #     y_trend = p(x_trend)
        #     ax.plot(x_trend, y_trend, 'r--', alpha=0.7)
            
        #     # Add correlation coefficient
        #     corr = np.corrcoef(x_data, y_data)[0, 1]
        #     ax.annotate(f"Correlation: {corr:.2f}", 
        #                 xy=(0.05, 0.95), 
        #                 xycoords='axes fraction',
        #                 backgroundcolor='white',
        #                 alpha=0.8)
        
        ax.set_title(plot_config['title'])
        ax.set_xlabel(plot_config['xlabel'])
        ax.set_ylabel(plot_config['ylabel'])
        
        xlim = plot_config['xlim']
        if xlim[0] is not None:
            if xlim[1] is not None:
                ax.set_xlim(xlim[0], xlim[1])
            else:
                ax.set_xlim(left=xlim[0])
        
        ax.grid(True, alpha=0.3)
    
    # Hide any unused subplots
    for i in range(num_plots, num_rows * 4):
        row = i // 4
        col = i % 4
        fig.delaxes(axes[row, col])
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust for the suptitle
    plt.savefig("e2e_latency_correlation_plots.pdf")
    print(f"* Saved E2E latency correlation plots to e2e_latency_correlation_plots.pdf")
    plt.show()

    
def create_ttft_correlation_plots(df):
    """Create scatter plots for correlations between TTFT and different metrics as subfigures with 4 per row"""
    
    # Define the correlation plots we want to create
    correlation_plots = []
    
    # Plot 1: TTFT vs KV Cache Hit Ratio
    if 'gateway_side_ttft' in df.columns and 'selected_pod_kv_cache_hit_ratio' in df.columns:
        correlation_plots.append({
            'x': 'selected_pod_kv_cache_hit_ratio',
            'y': 'gateway_side_ttft',
            'title': 'TTFT vs KV Cache Hit Ratio',
            'xlabel': 'KV Cache Hit Ratio',
            'ylabel': 'TTFT (ms)',
            'xlim': (0, 1)
        })
    
    # Plot 2: TTFT vs GPU KV Cache Usage
    if 'gateway_side_ttft' in df.columns and 'selected_pod_vllm_gpu_kv_cache_usage' in df.columns:
        correlation_plots.append({
            'x': 'selected_pod_vllm_gpu_kv_cache_usage',
            'y': 'gateway_side_ttft',
            'title': 'TTFT vs GPU KV Cache Usage',
            'xlabel': 'vLLM GPU Cache Usage %',
            'ylabel': 'TTFT (ms)',
            'xlim': (0, 1)
        })
    
    # Plot 3: TTFT vs Number of Running Requests
    if 'gateway_side_ttft' in df.columns and 'selected_pod_num_running_requests' in df.columns:
        correlation_plots.append({
            'x': 'selected_pod_num_running_requests',
            'y': 'gateway_side_ttft',
            'title': 'TTFT vs Running Requests',
            'xlabel': 'Number of Running Requests',
            'ylabel': 'TTFT (ms)',
            'xlim': (0, None)
        })
    
    # Plot 4: TTFT vs Number of Waiting Requests
    if 'gateway_side_ttft' in df.columns and 'selected_pod_num_waiting_requests' in df.columns:
        correlation_plots.append({
            'x': 'selected_pod_num_waiting_requests',
            'y': 'gateway_side_ttft',
            'title': 'TTFT vs Waiting Requests',
            'xlabel': 'Number of Waiting Requests',
            'ylabel': 'TTFT (ms)',
            'xlim': (0, None)
        })
    
    # Plot 5: TTFT vs Input Token Length
    if 'gateway_side_ttft' in df.columns and 'prompt_tokens' in df.columns:
        correlation_plots.append({
            'x': 'prompt_tokens',
            'y': 'gateway_side_ttft',
            'title': 'TTFT vs Input Token Length',
            'xlabel': 'Input Token Length',
            'ylabel': 'TTFT (ms)',
            'xlim': (0, None)
        })
    
    # Plot 6: TTFT vs Output Token Length
    if 'gateway_side_ttft' in df.columns and 'output_tokens' in df.columns:
        correlation_plots.append({
            'x': 'output_tokens',
            'y': 'gateway_side_ttft',
            'title': 'TTFT vs Output Token Length',
            'xlabel': 'Output Token Length',
            'ylabel': 'TTFT (ms)',
            'xlim': (0, None)
        })
    
    # Plot 7: TTFT vs Total Token Length
    if 'gateway_side_ttft' in df.columns and 'total_tokens' in df.columns:
        correlation_plots.append({
            'x': 'total_tokens',
            'y': 'gateway_side_ttft',
            'title': 'TTFT vs Total Token Length',
            'xlabel': 'Total Token Length',
            'ylabel': 'TTFT (ms)',
            'xlim': (0, None)
        })
    
    # If no correlation plots are available, return
    if not correlation_plots:
        return
    
    # Calculate number of rows needed (4 plots per row)
    num_plots = len(correlation_plots)
    num_rows = (num_plots + 3) // 4  # Ceiling division
    
    # Create the figure and subplots
    fig, axes = plt.subplots(num_rows, 4, figsize=(20, 5 * num_rows))
    fig.suptitle('TTFT Correlation Analysis', fontsize=16)
    
    # Make axes a 2D array even if it's a 1D array (single row) or a single subplot
    if num_plots == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = np.array([axes])
    
    # Create the correlation plots
    for i, plot_config in enumerate(correlation_plots):
        row = i // 4
        col = i % 4
        
        ax = axes[row, col]
        
        x_data = df[plot_config['x']]
        y_data = df[plot_config['y']]
        
        # Add a scatter plot
        ax.scatter(x_data, y_data, alpha=0.5)
        ax.set_title(plot_config['title'])
        ax.set_xlabel(plot_config['xlabel'])
        ax.set_ylabel(plot_config['ylabel'])
        
        xlim = plot_config['xlim']
        if xlim[0] is not None:
            if xlim[1] is not None:
                ax.set_xlim(xlim[0], xlim[1])
            else:
                ax.set_xlim(left=xlim[0])
        
        ax.grid(True, alpha=0.3)
    
    # Hide any unused subplots
    for i in range(num_plots, num_rows * 4):
        row = i // 4
        col = i % 4
        fig.delaxes(axes[row, col])
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust for the suptitle
    plt.savefig("ttft_correlation_plots.pdf")
    print(f"* Saved TTFT correlation plots to ttft_correlation_plots.pdf")
    plt.show()


def create_cdf_plots(df):
    """Create Cumulative Distribution Function (CDF) plots for various metrics as subfigures"""
    
    # Check if we have the required columns
    available_plots = []
    if 'gateway_side_ttft' in df.columns:
        available_plots.append('ttft')
    if 'gateway_side_tpot' in df.columns:
        available_plots.append('tpot')
    if 'gateway_side_e2e_latency' in df.columns:
        available_plots.append('e2e')
    
    if not available_plots:
        return
    
    # Create the figure and subplots
    fig, axes = plt.subplots(1, len(available_plots), figsize=(5*len(available_plots), 5))
    fig.suptitle('Latency Distribution CDFs', fontsize=16)
    
    # If only one plot, make axes iterable
    if len(available_plots) == 1:
        axes = [axes]
    
    # Plot index
    plot_idx = 0
    
    # TTFT CDF
    if 'ttft' in available_plots:
        ax = axes[plot_idx]
        ttft = df['gateway_side_ttft'].astype(int)
        ttft_sorted = np.sort(ttft)
        y = np.arange(1, len(ttft_sorted) + 1) / len(ttft_sorted)
        ax.plot(ttft_sorted, y, linewidth=2, label='TTFT CDF')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('TTFT (ms)', fontsize=12)
        ax.set_ylabel('Cumulative Probability', fontsize=12)
        ax.set_title('Time To First Token (TTFT)', fontsize=14)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.legend()
        plot_idx += 1
    
    # TPOT CDF
    if 'tpot' in available_plots:
        ax = axes[plot_idx]
        tpot = df['gateway_side_tpot'].astype(int)
        tpot_sorted = np.sort(tpot)
        y = np.arange(1, len(tpot_sorted) + 1) / len(tpot_sorted)
        ax.plot(tpot_sorted, y, linewidth=2, label='TPOT CDF')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('TPOT (ms)', fontsize=12)
        ax.set_ylabel('Cumulative Probability', fontsize=12)
        ax.set_title('Time Per Output Token (TPOT)', fontsize=14)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.legend()
        plot_idx += 1
    
    # E2E Latency CDF
    if 'e2e' in available_plots:
        ax = axes[plot_idx]
        e2e_latency = df['gateway_side_e2e_latency'].astype(int)
        e2e_latency_sorted = np.sort(e2e_latency)
        y = np.arange(1, len(e2e_latency_sorted) + 1) / len(e2e_latency_sorted)
        ax.plot(e2e_latency_sorted, y, linewidth=2, label='E2E Latency CDF')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('E2E Latency (ms)', fontsize=12)
        ax.set_ylabel('Cumulative Probability', fontsize=12)
        ax.set_title('End-to-End Latency', fontsize=14)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.legend()
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Adjust for the suptitle
    plt.savefig("cdf_plots.pdf")
    print(f"* Saved CDF plots to cdf_plots.pdf")
    plt.show()


def create_tpot_correlation_plots(df):
    """Create scatter plots for correlations between TPOT and different metrics as subfigures with 4 per row"""
    
    # Define the correlation plots we want to create
    correlation_plots = []
    
    # Plot 1: TPOT vs KV Cache Hit Ratio
    if 'gateway_side_tpot' in df.columns and 'selected_pod_kv_cache_hit_ratio' in df.columns:
        correlation_plots.append({
            'x': 'selected_pod_kv_cache_hit_ratio',
            'y': 'gateway_side_tpot',
            'title': 'TPOT vs KV Cache Hit Ratio',
            'xlabel': 'KV Cache Hit Ratio',
            'ylabel': 'Time Per Output Token (ms)',
            'xlim': (0, 1)
        })
    
    # Plot 2: TPOT vs GPU KV Cache Usage
    if 'gateway_side_tpot' in df.columns and 'selected_pod_vllm_gpu_kv_cache_usage' in df.columns:
        correlation_plots.append({
            'x': 'selected_pod_vllm_gpu_kv_cache_usage',
            'y': 'gateway_side_tpot',
            'title': 'TPOT vs GPU KV Cache Usage',
            'xlabel': 'vLLM GPU Cache Usage %',
            'ylabel': 'Time Per Output Token (ms)',
            'xlim': (0, 1)
        })
    
    # Plot 3: TPOT vs Number of Running Requests
    if 'gateway_side_tpot' in df.columns and 'selected_pod_num_running_requests' in df.columns:
        correlation_plots.append({
            'x': 'selected_pod_num_running_requests',
            'y': 'gateway_side_tpot',
            'title': 'TPOT vs Running Requests',
            'xlabel': 'Number of Running Requests',
            'ylabel': 'Time Per Output Token (ms)',
            'xlim': (0, None)
        })
    
    # Plot 4: TPOT vs Number of Waiting Requests
    if 'gateway_side_tpot' in df.columns and 'selected_pod_num_waiting_requests' in df.columns:
        correlation_plots.append({
            'x': 'selected_pod_num_waiting_requests',
            'y': 'gateway_side_tpot',
            'title': 'TPOT vs Waiting Requests',
            'xlabel': 'Number of Waiting Requests',
            'ylabel': 'Time Per Output Token (ms)',
            'xlim': (0, None)
        })
    
    # Plot 5: TPOT vs Input Token Length
    if 'gateway_side_tpot' in df.columns and 'prompt_tokens' in df.columns:
        correlation_plots.append({
            'x': 'prompt_tokens',
            'y': 'gateway_side_tpot',
            'title': 'TPOT vs Input Token Length',
            'xlabel': 'Input Token Length',
            'ylabel': 'Time Per Output Token (ms)',
            'xlim': (0, None)
        })
    
    # Plot 6: TPOT vs Output Token Length
    if 'gateway_side_tpot' in df.columns and 'output_tokens' in df.columns:
        correlation_plots.append({
            'x': 'output_tokens',
            'y': 'gateway_side_tpot',
            'title': 'TPOT vs Output Token Length',
            'xlabel': 'Output Token Length',
            'ylabel': 'Time Per Output Token (ms)',
            'xlim': (0, None)
        })
    
    # Plot 7: TPOT vs Total Token Length
    if 'gateway_side_tpot' in df.columns and 'total_tokens' in df.columns:
        correlation_plots.append({
            'x': 'total_tokens',
            'y': 'gateway_side_tpot',
            'title': 'TPOT vs Total Token Length',
            'xlabel': 'Total Token Length',
            'ylabel': 'Time Per Output Token (ms)',
            'xlim': (0, None)
        })
    
    # If no correlation plots are available, return
    if not correlation_plots:
        return
    
    # Calculate number of rows needed (4 plots per row)
    num_plots = len(correlation_plots)
    num_rows = (num_plots + 3) // 4  # Ceiling division
    
    # Create the figure and subplots
    fig, axes = plt.subplots(num_rows, 4, figsize=(20, 5 * num_rows))
    fig.suptitle('TPOT Correlation Analysis', fontsize=16)
    
    # Make axes a 2D array even if it's a 1D array (single row) or a single subplot
    if num_plots == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = np.array([axes])
    
    # Create the correlation plots
    for i, plot_config in enumerate(correlation_plots):
        row = i // 4
        col = i % 4
        
        ax = axes[row, col]
        
        x_data = df[plot_config['x']]
        y_data = df[plot_config['y']]
        
        # Add a scatter plot with a trend line
        ax.scatter(x_data, y_data, alpha=0.5)        
        ax.set_title(plot_config['title'])
        ax.set_xlabel(plot_config['xlabel'])
        ax.set_ylabel(plot_config['ylabel'])
        
        xlim = plot_config['xlim']
        if xlim[0] is not None:
            if xlim[1] is not None:
                ax.set_xlim(xlim[0], xlim[1])
            else:
                ax.set_xlim(left=xlim[0])
        
        ax.grid(True, alpha=0.3)
    
    # Hide any unused subplots
    for i in range(num_plots, num_rows * 4):
        row = i // 4
        col = i % 4
        fig.delaxes(axes[row, col])
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust for the suptitle
    plt.savefig("tpot_correlation_plots.pdf")
    print(f"* Saved TPOT correlation plots to tpot_correlation_plots.pdf")
    plt.show()


def create_pod_metrics_distribution_plots(df):
    """Create plots showing the distribution of metrics across pods"""
    
    # Check if we have pod metrics data
    if 'podMetricsLastSecond' not in df.columns:
        print("No pod metrics data available for plotting")
        return
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Pod Metrics Distribution Analysis', fontsize=16)
    
    # Plot 1: TTFT distribution over time with percentiles
    ax = axes[0, 0]
    ax.plot(df['normalized_start_time'], df['selected_pod_avg_ttft_ms'], label='Avg TTFT', color='blue')
    ax.plot(df['normalized_start_time'], df['selected_pod_p50_ttft_ms'], label='P50 TTFT', color='green', linestyle='--')
    ax.plot(df['normalized_start_time'], df['selected_pod_p90_ttft_ms'], label='P90 TTFT', color='orange', linestyle='--')
    ax.plot(df['normalized_start_time'], df['selected_pod_p99_ttft_ms'], label='P99 TTFT', color='red', linestyle='--')
    ax.set_title('TTFT Distribution Over Time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('TTFT (ms)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: TPOT distribution over time with percentiles
    ax = axes[0, 1]
    ax.plot(df['normalized_start_time'], df['selected_pod_avg_tpot_ms'], label=f'Avg TPOT', color='blue')
    ax.plot(df['normalized_start_time'], df['selected_pod_p50_tpot_ms'], label='P50 TPOT', color='green', linestyle='--')
    ax.plot(df['normalized_start_time'], df['selected_pod_p90_tpot_ms'], label='P90 TPOT', color='orange', linestyle='--')
    ax.plot(df['normalized_start_time'], df['selected_pod_p99_tpot_ms'], label='P99 TPOT', color='red', linestyle='--')
    ax.set_title('TPOT Distribution Over Time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('TPOT (ms)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 3: TPOT by token position (early, mid, late)
    ax = axes[1, 0]
    ax.plot(df['normalized_start_time'], df['selected_pod_early_tokens_tpot_ms'], label='Early Tokens TPOT', color='green')
    ax.plot(df['normalized_start_time'], df['selected_pod_mid_tokens_tpot_ms'], label='Mid Tokens TPOT', color='blue')
    ax.plot(df['normalized_start_time'], df['selected_pod_late_tokens_tpot_ms'], label='Late Tokens TPOT', color='purple')
    ax.set_title('TPOT by Token Position')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('TPOT (ms)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 4: Variance across cluster
    ax = axes[1, 1]
    ax.plot(df['normalized_start_time'], df['ttft_variance_across_cluster'], label='TTFT Variance', color='red')
    ax.plot(df['normalized_start_time'], df['tpot_variance_across_cluster'], label='TPOT Variance', color='blue')
    ax.set_title('Latency Variance Across Cluster')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Variance')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig("pod_metrics_distribution_plots.pdf")
    print(f"* Saved Pod Metrics Distribution plots to pod_metrics_distribution_plots.pdf")
    plt.show()


def create_pod_metrics_correlation_plots(df):
    correlation_plots = []
    
    # Plot 1: Pod Load vs TTFT
    if 'selected_pod_total_requests' in df.columns and 'selected_pod_avg_ttft_ms' in df.columns:
        correlation_plots.append({
            'x': 'selected_pod_total_requests',
            'y': 'selected_pod_avg_ttft_ms',
            'title': 'Pod Load vs TTFT',
            'xlabel': 'Pod Total Requests',
            'ylabel': 'Average TTFT (ms)',
            'xlim': (0, None)
        })
    
    # Plot 2: Pod Load vs TPOT
    if 'selected_pod_total_requests' in df.columns and 'selected_pod_avg_tpot_ms' in df.columns:
        correlation_plots.append({
            'x': 'selected_pod_total_requests',
            'y': 'selected_pod_avg_tpot_ms',
            'title': 'Pod Load vs TPOT',
            'xlabel': 'Pod Total Requests',
            'ylabel': 'Average TPOT (ms)',
            'xlim': (0, None)
        })
    
    # Plot 3: Active Pods Count vs TTFT
    if 'active_pods_count' in df.columns and 'selected_pod_avg_ttft_ms' in df.columns:
        correlation_plots.append({
            'x': 'active_pods_count',
            'y': 'selected_pod_avg_ttft_ms',
            'title': 'Active Pods vs TTFT',
            'xlabel': 'Number of Active Pods',
            'ylabel': 'Average TTFT (ms)',
            'xlim': (0, None)
        })
    
    # Plot 4: Active Pods Count vs TPOT
    if 'active_pods_count' in df.columns and 'selected_pod_avg_tpot_ms' in df.columns:
        correlation_plots.append({
            'x': 'active_pods_count',
            'y': 'selected_pod_avg_tpot_ms',
            'title': 'Active Pods vs TPOT',
            'xlabel': 'Number of Active Pods',
            'ylabel': 'Average TPOT (ms)',
            'xlim': (0, None)
        })
    
    # Plot 5: TTFT P99/P50 Ratio Over Time
    df['ttft_p99_p50_ratio'] = df['selected_pod_p99_ttft_ms'] / df['selected_pod_p50_ttft_ms']
    correlation_plots.append({
        'x': 'normalized_start_time',
        'y': 'ttft_p99_p50_ratio',
        'title': 'TTFT P99/P50 Ratio Over Time',
        'xlabel': 'Time (s)',
        'ylabel': 'P99/P50 Ratio',
        'xlim': (0, None)
    })
    
    # Plot 6: TPOT P99/P50 Ratio Over Time
    df['tpot_p99_p50_ratio'] = df['selected_pod_p99_tpot_ms'] / df['selected_pod_p50_tpot_ms']
    correlation_plots.append({
        'x': 'normalized_start_time',
        'y': 'tpot_p99_p50_ratio',
        'title': 'TPOT P99/P50 Ratio Over Time',
        'xlabel': 'Time (s)',
        'ylabel': 'P99/P50 Ratio',
        'xlim': (0, None)
    })
    
    # If no correlation plots are available, return
    if not correlation_plots:
        return
    
    # Calculate number of rows needed (3 plots per row)
    num_plots = len(correlation_plots)
    num_rows = (num_plots + 2) // 3  # Ceiling division
    
    # Create the figure and subplots
    fig, axes = plt.subplots(num_rows, 3, figsize=(18, 6 * num_rows))
    fig.suptitle('Pod Metrics Correlation Analysis', fontsize=16)
    
    # Make axes a 2D array even if it's a 1D array (single row) or a single subplot
    if num_plots == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = np.array([axes])
    
    # Create the correlation plots
    for i, plot_config in enumerate(correlation_plots):
        row = i // 3
        col = i % 3
        
        ax = axes[row, col]
        
        x_data = df[plot_config['x']]
        y_data = df[plot_config['y']]
        
        # Add a scatter plot
        ax.scatter(x_data, y_data, alpha=0.5)
        ax.set_title(plot_config['title'])
        ax.set_xlabel(plot_config['xlabel'])
        ax.set_ylabel(plot_config['ylabel'])
        
        xlim = plot_config['xlim']
        if xlim[0] is not None:
            if xlim[1] is not None:
                ax.set_xlim(xlim[0], xlim[1])
            else:
                ax.set_xlim(left=xlim[0])
        
        ax.grid(True, alpha=0.3)
    
    # Hide any unused subplots
    for i in range(num_plots, num_rows * 3):
        row = i // 3
        col = i % 3
        fig.delaxes(axes[row, col])
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust for the suptitle
    plt.savefig("pod_metrics_correlation_plots.pdf")
    print(f"* Saved Pod Metrics Correlation plots to pod_metrics_correlation_plots.pdf")
    plt.show()


def create_pod_metrics_histograms(df):
    """Create histograms for pod metrics distributions"""
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Pod Metrics Histograms', fontsize=16)
    
    # Plot 1: TTFT P50 histogram
    ax = axes[0, 0]
    ax.hist(df['selected_pod_p50_ttft_ms'].dropna(), bins=20, alpha=0.7)
    ax.set_title('TTFT P50 Distribution')
    ax.set_xlabel('TTFT P50 (ms)')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: TTFT P99 histogram
    ax = axes[0, 1]
    ax.hist(df['selected_pod_p99_ttft_ms'].dropna(), bins=20, alpha=0.7)
    ax.set_title('TTFT P99 Distribution')
    ax.set_xlabel('TTFT P99 (ms)')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: TPOT P50 histogram
    ax = axes[1, 0]
    ax.hist(df['selected_pod_p50_tpot_ms'].dropna(), bins=20, alpha=0.7)
    ax.set_title('TPOT P50 Distribution')
    ax.set_xlabel('TPOT P50 (ms)')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: TPOT P99 histogram
    ax = axes[1, 1]
    ax.hist(df['selected_pod_p99_tpot_ms'].dropna(), bins=20, alpha=0.7)
    ax.set_title('TPOT P99 Distribution')
    ax.set_xlabel('TPOT P99 (ms)')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig("pod_metrics_histograms.pdf")
    print(f"* Saved Pod Metrics Histograms to pod_metrics_histograms.pdf")
    plt.show()


if __name__ == "__main__":
    cutoff_time=0
    output_dir="output/p1024_s128_rps5-p2048_s128_rps5-p4096_s128_rps5-maxtokens100-20250429_155432"
    log_file =f"{output_dir}/gateway-plugins.log.csv"

    ts = time.time()
    df = parse_log_file(log_file)
    print(f"parse_log_file took {time.time() - ts} seconds")

    ts = time.time()
    first_request_start_time = df['request_start_time'].min()
    df['normalized_start_time'] = df['request_start_time'] - first_request_start_time
    df['normalized_end_time'] = df['request_end_time'] - first_request_start_time
    df['normalized_start_time'] /= 1_000_000
    df['normalized_end_time'] /= 1_000_000
    df['log_window_start_time'] = df['log_window_start_time'] - first_request_start_time
    df['log_window_start_time'] /= 1_000_000
    df['log_window_end_time'] = df['log_window_end_time'] - first_request_start_time
    df['log_window_end_time'] /= 1_000_000
    df = df[df['normalized_start_time'] > cutoff_time]
    df['normalized_start_time'] = df['normalized_start_time'] - df['normalized_start_time'].min()
    df['normalized_end_time'] = df['normalized_end_time'] - df['normalized_start_time'].min()
    df = df.sort_values(by='normalized_start_time', ascending=True)
    df['time_bucket'] = df['normalized_start_time'].astype(int)
    df = df[['normalized_start_time', 'time_bucket', 'normalized_end_time'] + [col for col in df.columns if col != 'normalized_start_time' and col != 'normalized_end_time' and col != 'time_bucket']]
    df.reset_index(drop=True, inplace=True)
    print(f"time normalization took {time.time() - ts} seconds")

    ts = time.time()
    df.to_csv(f"{output_dir}/parsed-gateway-plugins.log.csv", index=False)
    print(f"write to csv took {time.time() - ts} seconds")
    # display(df.head())
    # display(df.tail())
    print(df["podMetricsLastSecond"][10].keys())
    print(df["podMetricsLastSecond"][12].keys())

    # ts = time.time()
    df = analyze_llm_inference_logs(df)
    # print(f"analyze_llm_inference_logs took {time.time() - ts} seconds")

    # ts = time.time()
    df = analyze_pod_metrics_last_second(df)
    # print(f"analyze_pod_metrics_last_second took {time.time() - ts} seconds")

    # ts = time.time()
    create_simple_rps_plots(df)
    # print(f"create_simple_rps_plots took {time.time() - ts} seconds")

    # ts = time.time()
    create_cdf_plots(df)
    # print(f"create_cdf_plots took {time.time() - ts} seconds")
    
    # ts = time.time()
    create_e2e_latency_correlation_plots(df)
    # print(f"create_e2e_latency_correlation_plots took {time.time() - ts} seconds")

    # ts = time.time()
    create_ttft_correlation_plots(df)
    # print(f"create_ttft_correlation_plots took {time.time() - ts} seconds")
    
    # ts = time.time()
    create_tpot_correlation_plots(df)
    # print(f"create_tpot_correlation_plots took {time.time() - ts} seconds")

    # ts = time.time()
    create_pod_metrics_distribution_plots(df)
    # print(f"create_pod_metrics_distribution_plots took {time.time() - ts} seconds")

    # ts = time.time()
    create_pod_metrics_correlation_plots(df)
    # print(f"create_pod_metrics_correlation_plots took {time.time() - ts} seconds")

    # ts = time.time()
    create_pod_metrics_histograms(df)
    # print(f"create_pod_metrics_histograms took {time.time() - ts} seconds")