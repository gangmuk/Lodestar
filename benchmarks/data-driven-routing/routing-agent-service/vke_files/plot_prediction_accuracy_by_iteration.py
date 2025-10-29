#!/usr/bin/env python3

import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import argparse
from plot_latency_timeseries import parse_log_file

def plot_prediction_accuracy_by_iteration(df, routing_policy, output_path):
    """Plot prediction accuracy (MAE and MAPE) by iterations"""
    
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
    
    # Calculate accuracy metrics for each iteration
    mae_values = []
    mape_values = []
    count_values = []
    iterations_with_data = []
    
    for iteration in unique_iterations:
        subset = valid_predictions[valid_predictions['iteration'] == iteration]
        if len(subset) > 0:
            mae = (subset[actual_col] - subset['chosen_pod_predicted_latency']).abs().mean()
            mape = ((subset[actual_col] - subset['chosen_pod_predicted_latency']).abs() / subset[actual_col]).mean() * 100
            
            mae_values.append(mae)
            mape_values.append(mape)
            count_values.append(len(subset))
            iterations_with_data.append(iteration)
    
    if not mae_values:
        print("Error: No iteration statistics to plot")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Create bar chart for MAE only
    x = np.arange(len(iterations_with_data))
    width = 0.6
    
    # Plot MAE bars - all in navy/blue color tone
    bars = ax.bar(x, mae_values, width, 
                  color='steelblue',
                  alpha=0.8, edgecolor='gray', label='MAE (ms)')
    
    # Add value labels on bars
    for i, (bar, mae) in enumerate(zip(bars, mae_values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mae_values)*0.02,
                f'{mae:.0f}', ha='center', va='bottom', fontsize=14, color='navy', rotation=45, fontweight='bold')
    
    # Set labels and title
    ax.set_xlabel('Iteration', fontsize=16, fontweight='bold')
    ax.set_ylabel('MAE (ms)', fontsize=16, color='navy', fontweight='bold')
    
    # Set x-axis
    ax.set_xticks(x)
    ax.set_xticklabels([f'{it}' for it in iterations_with_data], fontsize=14)
    
    # Set y-axis colors
    ax.tick_params(axis='y', labelcolor='navy', labelsize=14)
    ax.tick_params(axis='x', labelsize=14)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    ax.legend(fontsize=14, loc='upper right')
    
    # Set y-limits with padding
    ax.set_ylim(0, max(mae_values) * 1.4)
    
    plt.tight_layout()
    
    # Save figure (PDF only)
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved plot to: {pdf_path}")
    
    return fig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot prediction accuracy by iterations')
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
        output_path = f"{log_dir}/prediction_accuracy_by_iteration.png"
    
    # Create plot
    fig = plot_prediction_accuracy_by_iteration(df, routing_policy, output_path)
    
    if fig:
        print(f"\nTotal iterations: {df['iteration'].nunique()}")
        print(f"Total requests: {len(df)}")
        # Print overall MAE and MAPE
        if 'latency_predictor_ttft' in routing_policy:
            actual_col = 'ttft'
        elif 'latency_predictor_avg_tpot' in routing_policy:
            actual_col = 'avg_tpot'
        else:
            actual_col = 'e2e'
        
        valid_pred = df[(df['chosen_pod_predicted_latency'].notna()) & 
                        (df['chosen_pod_predicted_latency'] > 0) &
                        (df[actual_col].notna()) & (df[actual_col] > 0)]
        if not valid_pred.empty:
            overall_mae = (valid_pred[actual_col] - valid_pred['chosen_pod_predicted_latency']).abs().mean()
            overall_mape = ((valid_pred[actual_col] - valid_pred['chosen_pod_predicted_latency']).abs() / valid_pred[actual_col]).mean() * 100
            print(f"Overall MAE: {overall_mae:.1f} ms")
            print(f"Overall MAPE: {overall_mape:.1f}%")

