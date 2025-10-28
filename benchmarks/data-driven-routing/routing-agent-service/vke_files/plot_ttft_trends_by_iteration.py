#!/usr/bin/env python3

import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import argparse
from plot_latency_timeseries import parse_log_file

def plot_ttft_trends_by_iteration(df, output_path):
    """Plot TTFT trends by iterations with dual y-axis for Avg and P99"""
    linewidth = 1.5
    alpha = 0.7
    
    # Get unique iterations
    unique_iterations = sorted(df['iteration'].unique())
    
    # Calculate iteration statistics
    iter_stats = []
    for iteration in unique_iterations:
        subset = df[df['iteration'] == iteration]
        if len(subset) > 0:
            iter_stats.append({
                'iteration': iteration,
                'avg_ttft': subset['ttft'].mean(),
                'p99_ttft': np.percentile(subset['ttft'], 99),
                'count': len(subset)
            })
    
    if not iter_stats:
        print("No iteration statistics to plot")
        return
    
    iter_vals = [s['iteration'] for s in iter_stats]
    avg_ttft_vals = [s['avg_ttft'] for s in iter_stats]
    p99_ttft_vals = [s['p99_ttft'] for s in iter_stats]
    count_vals = [s['count'] for s in iter_stats]
    
    # Compute overall statistics for legend
    overall_avg_ttft = df['ttft'].mean()
    overall_p99_ttft = np.percentile(df['ttft'], 99)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Plot average TTFT (left axis)
    ax.plot(iter_vals, avg_ttft_vals, marker='o', linestyle='-', color='blue', 
            linewidth=linewidth, alpha=alpha, label=f'Avg: {overall_avg_ttft:.1f}ms')
    
    # Add value labels on each dot for average (above the dot)
    for it, val in zip(iter_vals, avg_ttft_vals):
        ax.text(it, val, f'{val:.0f}', ha='center', va='bottom', 
                fontsize=12, color='blue')
    
    ax.set_xlabel('Iteration', fontsize=16)
    ax.set_ylabel('Average TTFT (ms)', fontsize=16, color='blue')
    ax.tick_params(axis='y', labelcolor='blue', labelsize=14)
    ax.tick_params(axis='x', labelsize=14)
    
    # Create right axis for P99
    ax_right = ax.twinx()
    ax_right.plot(iter_vals, p99_ttft_vals, marker='x', linestyle='--', color='tab:red', 
                  linewidth=linewidth, alpha=alpha, label=f'P99: {overall_p99_ttft:.1f}ms')
    
    # Add value labels on each dot for P99 (below the dot, centered)
    for it, val in zip(iter_vals, p99_ttft_vals):
        ax_right.text(it, val, f'{val:.0f}', ha='center', va='top', # slightly above the top of the dot 
                      fontsize=12, color='tab:red')
    
    ax_right.set_ylabel('P99 TTFT (ms)', fontsize=16, color='tab:red')
    ax_right.tick_params(axis='y', labelcolor='tab:red', labelsize=14)
    
    # Set y-limits
    ax.set_ylim(0, max(avg_ttft_vals) * 1.4)
    ax_right.set_ylim(0, max(p99_ttft_vals) * 1.4)
    
    # Add grid
    ax.grid(True, alpha=alpha)
    
    # Add legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_right.get_legend_handles_labels()
    # ax.legend(lines1 + lines2, labels1 + labels2, fontsize=14, loc='upper right')
    
    plt.tight_layout()
    
    # Save figure (PDF only)
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved plot to: {pdf_path}")
    
    return fig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot TTFT trends by iterations')
    parser.add_argument('log_file', type=str, help='Path to the log file')
    parser.add_argument('--skip-first-seconds', type=float, default=30, 
                        help='Skip/truncate the first X seconds of data (default: 30s)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for the plot (default: same directory as log file)')
    
    args = parser.parse_args()
    
    # Parse log file
    print(f"Parsing log file: {args.log_file}")
    data = parse_log_file(args.log_file)
    
    if not data:
        print(f"Error: No valid latency metrics found in {args.log_file}")
        exit(1)
    
    print(f"Found {len(data)} log entries")
    
    # Filter out first X seconds if specified
    if args.skip_first_seconds > 0:
        original_count = len(data)
        data = [entry for entry in data if entry.get('relative_time', 0) >= args.skip_first_seconds]
        filtered_count = original_count - len(data)
        print(f"Skipped first {args.skip_first_seconds} seconds: removed {filtered_count} entries, {len(data)} entries remaining")
        
        if not data:
            print(f"Error: No data remaining after skipping first {args.skip_first_seconds} seconds.")
            exit(1)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Filter out negative iterations
    if 'iteration' in df.columns:
        original_count = len(df)
        df = df[df['iteration'] >= 0]
        filtered_count = original_count - len(df)
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} rows with negative iteration values")
    
    if len(df) == 0:
        print("Error: No valid data after filtering")
        exit(1)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        log_dir = args.log_file.rsplit('/', 1)[0]
        output_path = f"{log_dir}/ttft_trends_by_iteration.png"
    
    # Create plot
    plot_ttft_trends_by_iteration(df, output_path)
    
    print(f"\nTotal iterations: {df['iteration'].nunique()}")
    print(f"Total requests: {len(df)}")
    print(f"Overall Avg TTFT: {df['ttft'].mean():.1f} ms")
    print(f"Overall P99 TTFT: {np.percentile(df['ttft'], 99):.1f} ms")

