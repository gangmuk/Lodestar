#!/usr/bin/env python3
"""
Plot prediction accuracy scatter plot from CSV file.

Usage:
    python plot_prediction_accuracy.py <csv_file> [output_pdf] [latency_metric]

Example:
    python plot_prediction_accuracy.py prediction_accuracy_data-5000.csv
    python plot_prediction_accuracy.py prediction_accuracy_data-5000.csv custom_scatter.pdf TTFT
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

def plot_prediction_accuracy(csv_file, output_pdf=None, latency_metric='TTFT'):
    """
    Create prediction accuracy scatter plot from CSV file.
    
    Args:
        csv_file: Path to prediction_accuracy_data CSV file
        output_pdf: Output PDF path (default: same directory as CSV)
        latency_metric: Metric name for labels (default: 'TTFT')
    """
    # Read CSV file
    if not os.path.exists(csv_file):
        print(f"ERROR: CSV file not found: {csv_file}")
        sys.exit(1)
    
    print(f"Reading data from: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Verify required columns
    required_cols = ['actual', 'predicted', 'error']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns: {missing_cols}")
        print(f"Available columns: {df.columns.tolist()}")
        sys.exit(1)
    
    print(f"Loaded {len(df)} data points")
    
    # Set output path
    if output_pdf is None:
        csv_dir = os.path.dirname(csv_file)
        csv_basename = os.path.splitext(os.path.basename(csv_file))[0]
        output_pdf = os.path.join(csv_dir, f'{csv_basename}_scatter_plot.pdf')
    
    # Create figure with larger size
    fig = plt.figure(figsize=(6, 5.3))
    
    # Set global font sizes - MUCH BIGGER
    plt.rcParams.update({
        'font.size': 18,           # Base font size
        'axes.titlesize': 24,      # Title font size
        'axes.labelsize': 22,      # Axis label font size
        'xtick.labelsize': 18,     # X-axis tick label size
        'ytick.labelsize': 18,     # Y-axis tick label size
        'legend.fontsize': 20,     # Legend font size
        'figure.titlesize': 26     # Figure title size
    })
    
    # Create scatter plot
    plt.scatter(df['actual'], df['predicted'], alpha=0.6, s=50, 
               color='steelblue', edgecolors='black', linewidth=0.8)
    
    # Perfect prediction line
    min_val = min(df['actual'].min(), df['predicted'].min())
    max_val = max(df['actual'].max(), df['predicted'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', 
            alpha=0.8, linewidth=3, label='Perfect Prediction')
    
    # Labels and title with larger fonts
    plt.xlim(left=-0.5, right=30000)
    plt.ylim(bottom=-0.5, top=30000)
    plt.xlabel(f'Actual {latency_metric} (ms)', fontsize=20, fontweight='bold')
    plt.ylabel(f'Predicted {latency_metric} (ms)', fontsize=20, fontweight='bold')
    # plt.title(f'{latency_metric} Prediction Accuracy', fontsize=28, fontweight='bold', pad=20)
    plt.title(f'Prediction Accuracy', fontsize=20, fontweight='bold', pad=12)
    plt.legend(fontsize=16, loc='upper left')
    plt.grid(True, alpha=0.3, linewidth=1.5)
    
    # Calculate metrics
    corr = np.corrcoef(df['actual'], df['predicted'])[0, 1]
    mae = mean_absolute_error(df['actual'], df['predicted'])
    r2 = r2_score(df['actual'], df['predicted'])
    mse = mean_squared_error(df['actual'], df['predicted'])
    rmse = np.sqrt(mse)
    
    # Add statistics box with larger font
    stats_text = f'Correlation: {corr:.3f}\n'
    stats_text += f'R²: {r2:.3f}\n'
    stats_text += f'MAE: {mae:.2f}\n'
    stats_text += f'RMSE: {rmse:.2f}\n'
    stats_text += f'Samples: {len(df):,}'
    
    # plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
    #         verticalalignment='top', fontsize=12,
    #         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.85, pad=1))
    
    # Make tick labels larger and bolder
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=18, width=2, length=8)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_pdf, dpi=150, bbox_inches='tight')
    print(f"\n{'='*70}")
    print(f"PLOT SAVED: {output_pdf}")
    print(f"{'='*70}")
    print(f"\nStatistics:")
    print(f"  Correlation: {corr:.4f}")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MAE: {mae:.2f} ms")
    print(f"  RMSE: {rmse:.2f} ms")
    print(f"  Samples: {len(df):,}")
    print(f"{'='*70}\n")
    
    plt.close()
    
    return output_pdf

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_prediction_accuracy.py <csv_file> [output_pdf] [latency_metric]")
        print("\nExample:")
        print("  python plot_prediction_accuracy.py prediction_accuracy_data-5000.csv")
        print("  python plot_prediction_accuracy.py prediction_accuracy_data-5000.csv custom_plot.pdf TTFT")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    output_pdf = sys.argv[2] if len(sys.argv) > 2 else None
    latency_metric = sys.argv[3] if len(sys.argv) > 3 else 'TTFT'
    
    plot_prediction_accuracy(csv_file, output_pdf, latency_metric)

if __name__ == "__main__":
    main()

