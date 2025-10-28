#!/usr/bin/env python3

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import json
import argparse
import math
from plot_latency_timeseries import parse_log_file

def plot_actual_vs_predicted_by_iteration(df, routing_policy, output_path):
    """Plot actual vs predicted latency scatter plot, colored by iterations"""
    
    linewidth = 1.5
    alpha = 0.7
    
    # Determine the target latency metric based on routing policy
    if 'latency_predictor_ttft' in routing_policy:
        actual_col = 'ttft'
        metric_name = 'TTFT'
    elif 'latency_predictor_avg_tpot' in routing_policy:
        actual_col = 'avg_tpot'
        metric_name = 'TPOT'
    elif 'latency_predictor_e2e_latency' in routing_policy:
        actual_col = 'e2e'
        metric_name = 'E2E Latency'
    else:
        actual_col = 'e2e'
        metric_name = 'E2E Latency'
    
    # Filter out entries where predicted latency is None or 0
    valid_predictions = df[(df['chosen_pod_predicted_latency'].notna()) &
                          (df['chosen_pod_predicted_latency'] > 0) &
                          (df[actual_col].notna()) &
                          (df[actual_col] > 0)]
    
    if valid_predictions.empty:
        print("Error: No valid prediction data available")
        return None
    
    # Get unique iterations
    unique_iterations = sorted(valid_predictions['iteration'].unique())
    
    # Define colors for iterations
    iteration_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_iterations)))
    iteration_color_map = dict(zip(unique_iterations, iteration_colors))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Scatter plot of actual vs predicted, colored by iteration
    for iteration in unique_iterations:
        subset = valid_predictions[valid_predictions['iteration'] == iteration]
        if len(subset) > 0:
            # Calculate MAE and MAPE for this iteration
            mae = (subset[actual_col] - subset['chosen_pod_predicted_latency']).abs().mean()
            mape = ((subset[actual_col] - subset['chosen_pod_predicted_latency']).abs() / subset[actual_col]).mean() * 100
            
            ax.scatter(subset[actual_col], subset['chosen_pod_predicted_latency'],
                      s=15, color=iteration_color_map[iteration], alpha=0.5, marker='.',
                      label=f'Iter {iteration}')
    
    # Add diagonal line for perfect prediction
    max_val = max(valid_predictions[actual_col].max(), valid_predictions['chosen_pod_predicted_latency'].max())
    min_val = min(valid_predictions[actual_col].min(), valid_predictions['chosen_pod_predicted_latency'].min())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=linewidth+0.5, alpha=alpha, label='Perfect Prediction', zorder=100)
    
    # Set same range for x and y axes, starting from 0
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val*0.7)
    
    # Set same grid intervals for both axes to create square grid cells
    n_grid_lines = 5  # Reduced number of grid lines for smaller figure
    grid_interval = math.ceil(max_val / n_grid_lines / 100) * 100
    
    tick_positions = [i * grid_interval for i in range(0, int(max_val / grid_interval) + 2)]
    tick_positions = [pos for pos in tick_positions if pos <= max_val * 1.05]
    
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels([f'{int(tick)}' for tick in tick_positions], fontsize=14)
    ax.set_yticklabels([f'{int(tick)}' for tick in tick_positions], fontsize=14)
    
    ax.xaxis.set_major_locator(ticker.FixedLocator(tick_positions))
    ax.yaxis.set_major_locator(ticker.FixedLocator(tick_positions))
    
    # Labels and title
    ax.set_xlabel(f'Actual {metric_name} (ms)', fontsize=16)
    ax.set_ylabel(f'Predicted {metric_name} (ms)', fontsize=16)
    # ax.set_title(f'Actual vs Predicted {metric_name} by Iterations', fontsize=18, fontweight='bold', pad=15)
    
    # Grid
    ax.grid(True, alpha=0.3, which='major')
    
    # Legend with appropriate font size
    # ax.legend(fontsize=10, loc='upper left', title='Iteration', title_fontsize=11, ncol=1)
    ax.legend(fontsize=12, loc='upper left', ncol=1)
    
    # Rotate x-axis labels
    ax.tick_params(axis='x', rotation=45, labelsize=14)
    
    # Set aspect ratio to be equal (square plot)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    # Save figure (PDF only)
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved plot to: {pdf_path}")
    
    return fig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot actual vs predicted latency by iterations')
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
    
    # Determine routing policy from log file path
    routing_policy = args.log_file.split('/')[-2].split('-')[0]
    print(f"Routing policy: {routing_policy}")
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        log_dir = args.log_file.rsplit('/', 1)[0]
        output_path = f"{log_dir}/actual_vs_predicted_by_iteration.png"
    
    # Create plot
    fig = plot_actual_vs_predicted_by_iteration(df, routing_policy, output_path)
    
    if fig:
        print(f"\nTotal iterations: {df['iteration'].nunique()}")
        print(f"Total requests: {len(df)}")
        
        # Print iteration-wise statistics
        if 'latency_predictor_ttft' in routing_policy:
            actual_col = 'ttft'
        elif 'latency_predictor_avg_tpot' in routing_policy:
            actual_col = 'avg_tpot'
        else:
            actual_col = 'e2e'
        
        valid_pred = df[(df['chosen_pod_predicted_latency'].notna()) & 
                        (df['chosen_pod_predicted_latency'] > 0) &
                        (df[actual_col].notna()) & (df[actual_col] > 0)]
        
        print("\nIteration-wise Statistics:")
        for iteration in sorted(valid_pred['iteration'].unique()):
            subset = valid_pred[valid_pred['iteration'] == iteration]
            mae = (subset[actual_col] - subset['chosen_pod_predicted_latency']).abs().mean()
            mape = ((subset[actual_col] - subset['chosen_pod_predicted_latency']).abs() / subset[actual_col]).mean() * 100
            print(f"  Iteration {iteration}: MAE={mae:.1f} ms, MAPE={mape:.1f}%, n={len(subset)}")

