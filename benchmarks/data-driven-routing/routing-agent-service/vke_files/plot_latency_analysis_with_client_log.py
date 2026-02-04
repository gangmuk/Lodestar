#!/usr/bin/env python3
"""
Professional latency analysis plotting script.
Generates comprehensive analysis with time series and CDF plots in one figure.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import argparse
import sys


def parse_log_file(log_file_path):
    """
    Parse client.log.txt to extract metrics.
    
    Returns:
        DataFrame with columns: request_id, ttft, avg_tpot, e2e, iteration, timestamp
    """
    data = []
    
    print("Parsing client.log.txt...")
    with open(log_file_path, 'r') as f:
        for line in f:
            # Look for lines with metric information
            if 'TTFT:' in line and 'Avg_tpot:' in line and 'E2E:' in line:
                try:
                    # Parse the line
                    # Extract request number, iteration, and metrics
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
                    
                    # Use sequential counter as timestamp
                    timestamp = len(data)
                    
                    data.append({
                        'request_id': req_id,
                        'ttft': ttft,
                        'avg_tpot': avg_tpot,
                        'e2e': e2e,
                        'iteration': iteration,
                        'timestamp': timestamp
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
    
    return df


def calculate_sliding_window_metrics(df, window_size=1.0, p99_window_size=60.0):
    """
    Calculate sliding window averages for RPS, TTFT, and TPOT.
    """
    max_time = df['end_time'].max()
    time_bins = np.arange(0, max_time + window_size, window_size)
    
    results = []
    
    for i in range(len(time_bins) - 1):
        window_start = time_bins[i]
        window_end = time_bins[i + 1]
        window_center = (window_start + window_end) / 2
        
        # Get requests that ended in this window
        window_df = df[(df['end_time'] >= window_start) & (df['end_time'] < window_end)]
        
        # Get requests for P99 calculation (larger window)
        p99_start = max(0, window_center - p99_window_size / 2)
        p99_end = window_center + p99_window_size / 2
        p99_df = df[(df['end_time'] >= p99_start) & (df['end_time'] < p99_end)]
        
        if len(window_df) > 0:
            rps = len(window_df) / window_size
            ttft_avg = window_df['ttft'].mean()
            tpot_avg = window_df['avg_tpot'].mean()
            
            ttft_p99 = p99_df['ttft'].quantile(0.99) if len(p99_df) > 0 else np.nan
            tpot_p99 = p99_df['avg_tpot'].quantile(0.99) if len(p99_df) > 0 else np.nan
            
            iterations = window_df['iteration'].unique()
            
            results.append({
                'time': window_center,
                'rps': rps,
                'ttft_avg': ttft_avg,
                'tpot_avg': tpot_avg,
                'ttft_p99': ttft_p99,
                'tpot_p99': tpot_p99,
                'iterations': iterations
            })
    
    return pd.DataFrame(results)


def find_iteration_transitions(df):
    """Find iteration transition points."""
    transitions = []
    
    for iteration in sorted(df['iteration'].unique()):
        iter_df = df[df['iteration'] == iteration]
        if len(iter_df) > 0:
            end_time = iter_df['end_time'].max()
            transitions.append(end_time)
    
    # Remove the last transition
    if len(transitions) > 1:
        transitions = transitions[:-1]
    
    return transitions


def calculate_cdf(data):
    """Calculate CDF for a given data array."""
    sorted_data = np.sort(data)
    cdf_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, cdf_values


def plot_comprehensive_analysis(df, window_df, transitions, output_path):
    """
    Create comprehensive analysis plot with time series (top) and CDF (bottom).
    """
    # Set up the style
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    
    # Create figure with gridspec for custom layout
    fig = plt.figure(figsize=(18, 16))
    # 9 rows, 3 columns: 3 rows for time series (full width), 1 row for CDF, 3 rows for bar charts (one per row)
    gs = gridspec.GridSpec(9, 3, figure=fig, height_ratios=[0.6, 0.6, 0.6, 0.1, 0.7, 0.1, 0.5, 0.5, 0.5], 
                          hspace=0.4, wspace=0.3, left=0.06, right=0.98, top=0.96, bottom=0.04)
    
    # Color schemes
    color_rps = '#3366CC'
    color_ttft = '#CC3333'
    color_ttft_p99 = '#FF6B6B'
    color_tpot = '#CC3333'
    color_tpot_p99 = '#FF6B6B'
    color_transition = '#9B59B6'  # Purple color for iteration boundaries
    
    # Colors for iterations in CDF
    iter_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    total_requests = len(df)
    iterations = sorted(df['iteration'].unique())
    
    # ========================================================================
    # TIME SERIES PLOTS (Top 3 Rows - Full Width)
    # ========================================================================
    
    # Create axes for time series (3 rows, full width)
    ax_rps = plt.subplot(gs[0, :])
    ax_ttft = plt.subplot(gs[1, :])
    ax_tpot = plt.subplot(gs[2, :])
    
    # Plot 1: RPS
    ax_rps.plot(window_df['time'], window_df['rps'], color=color_rps, linewidth=0.8, alpha=0.85)
    ax_rps.set_ylabel('Requests/sec', fontsize=10)
    ax_rps.set_title(f'Requests Per Second (RPS) - Total Requests: {total_requests}', 
                     fontsize=11, fontweight='bold', pad=8)
    ax_rps.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
    ax_rps.set_xlim(0, window_df['time'].max())
    ax_rps.set_ylim(bottom=0)
    ax_rps.set_facecolor('#F5F5F5')
    
    for trans_time in transitions:
        ax_rps.axvline(x=trans_time, color=color_transition, linestyle='--', linewidth=1.2, alpha=0.6)
    
    # Plot 2: TTFT
    ax_ttft.plot(window_df['time'], window_df['ttft_avg'], color=color_ttft, 
                 linewidth=0.8, alpha=0.85, label='Avg (1s window)')
    ax_ttft.plot(window_df['time'], window_df['ttft_p99'], color=color_ttft_p99, 
                 linewidth=1.0, alpha=0.7, linestyle='-.', label='P99 (60s window)')
    ax_ttft.set_ylabel('TTFT (ms)', fontsize=10)
    ax_ttft.set_title('Time to First Token (TTFT) - Sliding Window Average', 
                      fontsize=11, fontweight='bold', pad=8)
    ax_ttft.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
    ax_ttft.set_xlim(0, window_df['time'].max())
    ax_ttft.set_ylim(bottom=0)
    ax_ttft.set_facecolor('#F5F5F5')
    ax_ttft.legend(loc='upper left', fontsize=8, framealpha=0.9)
    
    for trans_time in transitions:
        ax_ttft.axvline(x=trans_time, color=color_transition, linestyle='--', linewidth=1.2, alpha=0.6)
    
    # Plot 3: TPOT
    ax_tpot.plot(window_df['time'], window_df['tpot_avg'], color=color_tpot, 
                 linewidth=0.8, alpha=0.85, label='Avg (1s window)')
    ax_tpot.plot(window_df['time'], window_df['tpot_p99'], color=color_tpot_p99, 
                 linewidth=1.0, alpha=0.7, linestyle='-.', label='P99 (60s window)')
    ax_tpot.set_ylabel('TPOT (ms)', fontsize=10)
    ax_tpot.set_xlabel('Time (sec)', fontsize=10)
    ax_tpot.set_title('Time Per Output Token (TPOT) - Sliding Window Average', 
                      fontsize=11, fontweight='bold', pad=8)
    ax_tpot.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
    ax_tpot.set_xlim(0, window_df['time'].max())
    ax_tpot.set_ylim(bottom=0)
    ax_tpot.set_facecolor('#F5F5F5')
    ax_tpot.legend(loc='upper right', fontsize=8, framealpha=0.9)
    
    for trans_time in transitions:
        ax_tpot.axvline(x=trans_time, color=color_transition, linestyle='--', linewidth=1.2, alpha=0.6)
    
    # ========================================================================
    # CDF PLOTS (Bottom Row - 3 Columns)
    # ========================================================================
    
    ax_cdf_ttft = plt.subplot(gs[4, 0])
    ax_cdf_tpot = plt.subplot(gs[4, 1])
    ax_cdf_e2e = plt.subplot(gs[4, 2])
    
    cdf_axes = [ax_cdf_ttft, ax_cdf_tpot, ax_cdf_e2e]
    metrics = [
        {'column': 'ttft', 'title': 'TTFT CDF', 'xlabel': 'TTFT (ms)'},
        {'column': 'avg_tpot', 'title': 'TPOT CDF', 'xlabel': 'TPOT (ms)'},
        {'column': 'e2e', 'title': 'End-to-End CDF', 'xlabel': 'E2E Latency (ms)'}
    ]
    
    for ax, metric_config in zip(cdf_axes, metrics):
        metric = metric_config['column']
        
        # Plot CDF for each iteration
        for iter_idx, iteration in enumerate(iterations):
            iter_df = df[df['iteration'] == iteration]
            data = iter_df[metric].dropna()
            
            if len(data) > 0:
                sorted_data, cdf_values = calculate_cdf(data)
                p50 = np.percentile(data, 50)
                p99 = np.percentile(data, 99)
                
                color = iter_colors[iter_idx % len(iter_colors)]
                label = f'Iter {iteration} (n={len(data)}, P50={p50:.0f}, P99={p99:.0f})'
                ax.plot(sorted_data, cdf_values * 100, 
                       color=color, linewidth=1.5, alpha=0.85, label=label)
        
        # Formatting
        ax.set_xlabel(metric_config['xlabel'], fontsize=10, fontweight='bold')
        ax.set_ylabel('CDF (%)', fontsize=10, fontweight='bold')
        ax.set_title(metric_config['title'], fontsize=11, fontweight='bold', pad=8)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_ylim(0, 100)
        ax.set_xlim(left=0)
        ax.legend(loc='lower right', fontsize=8, framealpha=0.95)
        ax.set_facecolor('#F5F5F5')
        
        # Add percentile lines
        ax.axhline(y=50, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax.axhline(y=95, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax.axhline(y=99, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    # ========================================================================
    # BAR CHARTS (3 Rows - One per row, full width)
    # ========================================================================
    
    ax_bar_ttft = plt.subplot(gs[6, :])
    ax_bar_tpot = plt.subplot(gs[7, :])
    ax_bar_e2e = plt.subplot(gs[8, :])
    
    bar_axes = [ax_bar_ttft, ax_bar_tpot, ax_bar_e2e]
    bar_metrics = [
        {'column': 'ttft', 'title': 'TTFT Statistics', 'ylabel': 'TTFT (ms)'},
        {'column': 'avg_tpot', 'title': 'TPOT Statistics', 'ylabel': 'TPOT (ms)'},
        {'column': 'e2e', 'title': 'End-to-End Statistics', 'ylabel': 'E2E Latency (ms)'}
    ]
    
    percentiles = ['Avg', 'P50', 'P90', 'P99']
    x_pos = np.arange(len(percentiles))
    bar_width = 0.05  # Thin bars with space between metrics
    
    for ax, metric_config in zip(bar_axes, bar_metrics):
        metric = metric_config['column']
        
        # Calculate statistics for each iteration
        iteration_stats = []
        for iteration in iterations:
            iter_df = df[df['iteration'] == iteration]
            data = iter_df[metric].dropna()
            
            if len(data) > 0:
                stats = [
                    data.mean(),  # Avg
                    np.percentile(data, 50),  # P50
                    np.percentile(data, 90),  # P90
                    np.percentile(data, 99)  # P99
                ]
                iteration_stats.append(stats)
        
        # Calculate statistics for all iterations combined
        all_data = df[metric].dropna()
        if len(all_data) > 0:
            all_stats = [
                all_data.mean(),  # Avg
                np.percentile(all_data, 50),  # P50
                np.percentile(all_data, 90),  # P90
                np.percentile(all_data, 99)  # P99
            ]
        
        # Plot bars for each iteration
        num_groups = len(iterations) + 1  # iterations + All
        bars_list = []
        
        for iter_idx, (iteration, stats) in enumerate(zip(iterations, iteration_stats)):
            offset = (iter_idx - num_groups/2 + 0.5) * bar_width
            color = iter_colors[iter_idx % len(iter_colors)]
            bars = ax.bar(x_pos + offset, stats, bar_width, alpha=0.8, 
                         color=color, label=f'Iter {iteration}')
            bars_list.append((bars, stats))
        
        # Plot bars for all iterations combined
        offset = (len(iterations) - num_groups/2 + 0.5) * bar_width
        bars = ax.bar(x_pos + offset, all_stats, bar_width, alpha=0.9, 
                     color='#34495E', label='All', edgecolor='black', linewidth=1)
        bars_list.append((bars, all_stats))
        
        # Add value labels on top of each bar
        for bars, stats in bars_list:
            for bar, value in zip(bars, stats):
                height = bar.get_height()
                # Format value based on magnitude
                if value < 10:
                    label_text = f'{value:.1f}'
                else:
                    label_text = f'{int(value)}'
                
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       label_text,
                       ha='center', va='bottom', fontsize=7, rotation=90,
                       fontweight='normal')
        
        # Formatting
        # ax.set_xlabel('Percentile', fontsize=10, fontweight='bold')
        ax.set_ylabel(metric_config['ylabel'], fontsize=10, fontweight='bold')
        ax.set_title(metric_config['title'], fontsize=11, fontweight='bold', pad=8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(percentiles, fontsize=9)
        ax.legend(loc='upper left', fontsize=8, framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')
        ax.set_facecolor('#F5F5F5')
        ax.set_ylim(bottom=0)
    
    # Add overall title
    fig.suptitle('Comprehensive Latency Analysis - Time Series, CDF, and Statistics', 
                fontsize=15, fontweight='bold', y=0.985)
    
    # Save figure
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Plot saved to: {output_path}")
    
    plt.close()


def parse_path_metadata(file_path):
    """
    Parse metadata from the directory path.
    Expected format: .../NVIDIA-A30/maxTokens_1-maxTokensStd_0/mooncake/toolagent-2/rps20-benchmark/...
    
    Returns:
        dict with gpu_type, output_distribution, workload_category, workload_name, load
    """
    path_parts = Path(file_path).parts
    
    metadata = {
        'gpu_type': '',
        'output_distribution': '',
        'workload_category': '',
        'workload_name': '',
        'load': ''
    }
    
    try:
        # Find NVIDIA-* part for gpu_type
        for i, part in enumerate(path_parts):
            if part.startswith('NVIDIA-'):
                metadata['gpu_type'] = part
                
                # Next part should be output distribution (e.g., maxTokens_1-maxTokensStd_0)
                if i + 1 < len(path_parts):
                    metadata['output_distribution'] = path_parts[i + 1]
                
                # Next should be workload category (e.g., mooncake)
                if i + 2 < len(path_parts):
                    metadata['workload_category'] = path_parts[i + 2]
                
                # Next should be workload name (e.g., toolagent-2)
                if i + 3 < len(path_parts):
                    metadata['workload_name'] = path_parts[i + 3]
                
                # Next should be load (e.g., rps20-benchmark)
                if i + 4 < len(path_parts):
                    metadata['load'] = path_parts[i + 4]
                
                break
    except Exception as e:
        print(f"Warning: Could not parse path metadata: {e}")
    
    return metadata


def save_latency_metrics_csv(df, output_dir, input_path):
    """
    Save comprehensive latency metrics to a single CSV file.
    
    Args:
        df: DataFrame with latency metrics
        output_dir: Directory to save CSV file
        input_path: Input file path for metadata extraction
    """
    output_dir = Path(output_dir)
    
    # Parse metadata from path
    metadata = parse_path_metadata(input_path)
    
    metrics = ['ttft', 'avg_tpot', 'e2e']
    metric_names = ['TTFT', 'TPOT', 'E2E']
    percentiles = [50, 75, 90, 95, 99, 99.9]
    
    iterations = sorted(df['iteration'].unique())
    
    # Prepare comprehensive data for CSV
    csv_data = []
    
    # Per iteration statistics
    for iteration in iterations:
        iter_df = df[df['iteration'] == iteration]
        row = {
            'gpu_type': metadata['gpu_type'],
            'output_distribution': metadata['output_distribution'],
            'workload_category': metadata['workload_category'],
            'workload_name': metadata['workload_name'],
            'load': metadata['load'],
            'Iteration': f'Iter_{iteration}',
            'Count': len(iter_df)
        }
        
        for metric, metric_name in zip(metrics, metric_names):
            data = iter_df[metric].dropna()
            if len(data) > 0:
                row[f'{metric_name}_Mean'] = round(data.mean(), 2)
                for p in percentiles:
                    p_label = f'P{p}'.replace('.', '_')
                    row[f'{metric_name}_{p_label}'] = round(np.percentile(data, p), 2)
                row[f'{metric_name}_Max'] = round(data.max(), 2)
        
        csv_data.append(row)
    
    # Aggregated statistics
    row = {
        'gpu_type': metadata['gpu_type'],
        'output_distribution': metadata['output_distribution'],
        'workload_category': metadata['workload_category'],
        'workload_name': metadata['workload_name'],
        'load': metadata['load'],
        'Iteration': 'All',
        'Count': len(df)
    }
    for metric, metric_name in zip(metrics, metric_names):
        data = df[metric].dropna()
        if len(data) > 0:
            row[f'{metric_name}_Mean'] = round(data.mean(), 2)
            for p in percentiles:
                p_label = f'P{p}'.replace('.', '_')
                row[f'{metric_name}_{p_label}'] = round(np.percentile(data, p), 2)
            row[f'{metric_name}_Max'] = round(data.max(), 2)
    
    csv_data.append(row)
    
    # Create DataFrame and save to single CSV
    stats_df = pd.DataFrame(csv_data)
    csv_path = output_dir / 'latency_metrics.csv'
    stats_df.to_csv(csv_path, index=False)
    print(f"✓ Saved comprehensive latency metrics to: {csv_path}")


def print_statistics(df):
    """Print detailed statistics."""
    print("\n" + "="*80)
    print("DETAILED STATISTICS BY ITERATION")
    print("="*80)
    
    metrics = [
        ('ttft', 'TTFT (ms)'),
        ('avg_tpot', 'TPOT (ms)'),
        ('e2e', 'End-to-End (ms)')
    ]
    
    iterations = sorted(df['iteration'].unique())
    
    for metric_col, metric_name in metrics:
        print(f"\n{metric_name}")
        print("-" * 80)
        print(f"{'Iter':<6} {'Count':<8} {'Mean':<10} {'Median':<10} {'P95':<10} {'P99':<10} {'Max':<10}")
        print("-" * 80)
        
        for iteration in iterations:
            iter_df = df[df['iteration'] == iteration]
            data = iter_df[metric_col].dropna()
            
            if len(data) > 0:
                count = len(data)
                mean_val = data.mean()
                median_val = np.percentile(data, 50)
                p95 = np.percentile(data, 95)
                p99 = np.percentile(data, 99)
                max_val = data.max()
                
                print(f"{iteration:<6} {count:<8} {mean_val:<10.2f} {median_val:<10.2f} "
                     f"{p95:<10.2f} {p99:<10.2f} {max_val:<10.2f}")
        
        # Overall
        data = df[metric_col].dropna()
        if len(data) > 0:
            count = len(data)
            mean_val = data.mean()
            median_val = np.percentile(data, 50)
            p95 = np.percentile(data, 95)
            p99 = np.percentile(data, 99)
            max_val = data.max()
            
            print("-" * 80)
            print(f"{'ALL':<6} {count:<8} {mean_val:<10.2f} {median_val:<10.2f} "
                 f"{p95:<10.2f} {p99:<10.2f} {max_val:<10.2f}")
    
    print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Generate comprehensive latency analysis plots from client.log.txt'
    )
    parser.add_argument(
        'input',
        type=str,
        default='client.log.txt',
        help='Path to client.log.txt file (default: client.log.txt)'
    )
    parser.add_argument(
        '--window',
        type=float,
        default=1.0,
        help='Sliding window size in seconds (default: 1.0)'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Print detailed statistics'
    )
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return
    
    output_path = Path(args.input).parent / 'latency_client_log.pdf'
    print(f"Output path: {output_path}")
    df = parse_log_file(args.input)
    print(f"✓ Loaded {len(df)} requests")
    
    # Show iteration breakdown
    if 'iteration' in df.columns:
        iterations = sorted(df['iteration'].unique())
        print(f"\nIterations found: {iterations}")
        for iteration in iterations:
            count = len(df[df['iteration'] == iteration])
            print(f"  Iteration {iteration}: {count} requests")
    
    # Print statistics if requested
    if args.stats:
        print_statistics(df)
    
    # Calculate metrics
    print(f"\nCalculating sliding window metrics (window: {args.window}s, P99 window: 60s)...")
    window_df = calculate_sliding_window_metrics(df, window_size=args.window, p99_window_size=60.0)
    print(f"✓ Created {len(window_df)} time windows")
    
    print("\nFinding iteration transitions...")
    transitions = find_iteration_transitions(df)
    print(f"✓ Found {len(transitions)} iteration transitions at times: {[f'{t:.2f}' for t in transitions]}")
    
    print("\nGenerating comprehensive plot...")
    plot_comprehensive_analysis(df, window_df, transitions, output_path)
    
    # Save CSV files with latency metrics
    print("\nSaving latency metrics to CSV files...")
    output_dir = Path(args.input).parent
    save_latency_metrics_csv(df, output_dir, args.input)
    
    print("\n" + "="*80)
    print("           ✅ SUCCESS! Analysis complete.")
    print("="*80)


if __name__ == '__main__':
    main()

