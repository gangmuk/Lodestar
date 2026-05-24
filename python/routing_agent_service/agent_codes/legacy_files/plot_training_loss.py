#!/usr/bin/env python3
"""
Plot training loss curves from CSV file.

Usage:
    python plot_training_loss.py <csv_file> [output_pdf] [latency_metric]

Example:
    python plot_training_loss.py training_loss_data-5000.csv
    python plot_training_loss.py training_loss_data-5000.csv custom_loss.pdf TTFT
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_training_loss(csv_file, output_pdf=None, latency_metric='TTFT'):
    """
    Create training loss plot from CSV file.
    
    Args:
        csv_file: Path to training_loss_data CSV file
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
    required_cols = ['epoch', 'train_loss']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns: {missing_cols}")
        print(f"Available columns: {df.columns.tolist()}")
        sys.exit(1)
    
    print(f"Loaded {len(df)} epochs")
    
    # Set output path
    if output_pdf is None:
        csv_dir = os.path.dirname(csv_file)
        csv_basename = os.path.splitext(os.path.basename(csv_file))[0]
        output_pdf = os.path.join(csv_dir, f'{csv_basename}_loss_plot.pdf')
    
    # Create figure with larger size
    fig = plt.figure(figsize=(10, 8))
    
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
    
    # Plot training loss
    plt.plot(df['epoch'], df['train_loss'], 'b-', linewidth=3, 
            marker='o', markersize=8, label='Training Loss',
            markerfacecolor='lightblue', markeredgecolor='navy', markeredgewidth=2)
    
    # Plot validation loss if available
    has_val_loss = 'val_loss' in df.columns and df['val_loss'].notna().any()
    if has_val_loss:
        plt.plot(df['epoch'], df['val_loss'], 'r-', linewidth=3,
                marker='s', markersize=8, label='Validation Loss',
                markerfacecolor='lightcoral', markeredgecolor='darkred', markeredgewidth=2)
    
    # Labels and title with larger fonts
    plt.xlim(left=-0.5)
    plt.ylim(bottom=0)
    plt.xlabel('Epoch', fontsize=30, fontweight='bold')
    plt.ylabel('MSE Loss', fontsize=30, fontweight='bold')
    # plt.title(f'{latency_metric} Prediction Loss', fontsize=28, fontweight='bold', pad=20)
    plt.title(f'Prediction Loss', fontsize=28, fontweight='bold', pad=20)
    plt.legend(fontsize=28, loc='upper right', framealpha=0.9)
    plt.grid(True, alpha=0.3, linewidth=1.5)
    
    # Make tick labels larger and bolder
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=28, width=2, length=8)
    
    # Calculate and display final loss values
    final_train = df['train_loss'].iloc[-1]
    initial_train = df['train_loss'].iloc[0]
    improvement = ((initial_train - final_train) / initial_train) * 100
    
    loss_text = f'Initial Train Loss: {initial_train:.1f}\n'
    loss_text += f'Final Train Loss: {final_train:.1f}\n'
    loss_text += f'Improvement: {improvement:.1f}%'
    
    if has_val_loss:
        final_val = df['val_loss'].iloc[-1]
        loss_text += f'\n\nFinal Val Loss: {final_val:.1f}'
        
        # Calculate train/val gap
        gap = abs(final_val - final_train)
        gap_pct = (gap / final_train) * 100
        loss_text += f'\nTrain/Val Gap: {gap_pct:.1f}%'
        
        if gap_pct > 30:
            loss_text += '\n⚠️ High variance'
        elif gap_pct > 15:
            loss_text += '\n⚠️ Some overfitting'
        else:
            loss_text += '\n✓ Good fit'
    
    loss_text += f'\n\nTotal Epochs: {len(df)}'
    
    # plt.text(0.98, 0.98, loss_text, transform=plt.gca().transAxes,
    #         verticalalignment='top', horizontalalignment='right', fontsize=18,
    #         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, pad=1.2))
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"\n{'='*70}")
    print(f"PLOT SAVED: {output_pdf}")
    print(f"{'='*70}")
    print(f"\nTraining Summary:")
    print(f"  Total Epochs: {len(df)}")
    print(f"  Initial Train Loss: {initial_train:.2f}")
    print(f"  Final Train Loss: {final_train:.2f}")
    print(f"  Improvement: {improvement:.1f}%")
    
    if has_val_loss:
        print(f"\nValidation Summary:")
        print(f"  Final Val Loss: {final_val:.2f}")
        print(f"  Train/Val Gap: {gap_pct:.1f}%")
        
        if gap_pct > 30:
            print(f"  Status: ⚠️ High variance detected")
        elif gap_pct > 15:
            print(f"  Status: ⚠️ Some overfitting")
        else:
            print(f"  Status: ✓ Good fit")
    
    print(f"{'='*70}\n")
    
    plt.close()
    
    return output_pdf

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_training_loss.py <csv_file> [output_pdf] [latency_metric]")
        print("\nExample:")
        print("  python plot_training_loss.py training_loss_data-5000.csv")
        print("  python plot_training_loss.py training_loss_data-5000.csv custom_plot.pdf TTFT")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    output_pdf = sys.argv[2] if len(sys.argv) > 2 else None
    latency_metric = sys.argv[3] if len(sys.argv) > 3 else 'TTFT'
    
    plot_training_loss(csv_file, output_pdf, latency_metric)

if __name__ == "__main__":
    main()

