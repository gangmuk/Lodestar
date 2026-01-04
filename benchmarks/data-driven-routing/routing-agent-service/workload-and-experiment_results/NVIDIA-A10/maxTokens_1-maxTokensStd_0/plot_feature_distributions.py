#!/usr/bin/env python3
"""
Plot distribution (PDF) of each feature from the CSV file.
Each feature is plotted in a separate subfigure.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# Set style for professional-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.titlesize'] = 14


def identify_numeric_features(df):
    """Identify numeric features, excluding categorical and boolean columns."""
    numeric_features = []
    exclude_cols = ['request_id', 'selected_pod', 'subAlgorithm', 'source_file']
    
    for col in df.columns:
        # Skip excluded columns
        if col in exclude_cols:
            continue
        
        # Skip columns that end with '-GPU' (categorical)
        if col.endswith('-GPU'):
            continue
        
        # Skip boolean columns (check for True/False values)
        if df[col].dtype == 'bool' or df[col].dtype == 'object':
            # Check if it contains only boolean-like values
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) <= 2:
                # Check if values are boolean-like
                str_vals = [str(v).lower() for v in unique_vals]
                if all(v in ['true', 'false', '1', '0', 'yes', 'no'] for v in str_vals):
                    continue
        
        # Check if column is numeric
        if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            numeric_features.append(col)
        elif df[col].dtype == 'object':
            # Try to convert to numeric
            try:
                pd.to_numeric(df[col], errors='raise')
                numeric_features.append(col)
            except:
                continue
    
    return numeric_features


def plot_feature_distributions(df, numeric_features, output_pdf_path):
    """Plot PDF distributions for all numeric features."""
    
    n_features = len(numeric_features)
    
    # Calculate grid dimensions
    cols_per_row = 4
    n_rows = int(np.ceil(n_features / cols_per_row))
    
    # Create figure with appropriate size
    fig = plt.figure(figsize=(16, 4 * n_rows))
    gs = gridspec.GridSpec(n_rows, cols_per_row, figure=fig, 
                          hspace=0.4, wspace=0.3,
                          left=0.05, right=0.98, top=0.96, bottom=0.04)
    
    fig.suptitle('Feature Distribution Analysis (PDF)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    for idx, feature in enumerate(numeric_features):
        row = idx // cols_per_row
        col = idx % cols_per_row
        ax = fig.add_subplot(gs[row, col])
        
        # Get data for this feature
        data = df[feature].dropna()
        
        if len(data) == 0:
            ax.text(0.5, 0.5, 'No data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(feature, fontsize=10, fontweight='bold')
            continue
        
        # Remove infinite values
        data = data[np.isfinite(data)]
        
        if len(data) == 0:
            ax.text(0.5, 0.5, 'No valid data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(feature, fontsize=10, fontweight='bold')
            continue
        
        # Plot histogram with KDE overlay for smooth PDF
        try:
            # Use seaborn for better-looking distributions
            sns.histplot(data=data, ax=ax, kde=True, stat='density', 
                        color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Add statistics text
            mean_val = data.mean()
            median_val = data.median()
            std_val = data.std()
            
            stats_text = f'Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd: {std_val:.2f}'
            ax.text(0.98, 0.98, stats_text, 
                   transform=ax.transAxes, 
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=7)
            
        except Exception as e:
            # Fallback to simple histogram if KDE fails
            ax.hist(data, bins=50, density=True, alpha=0.7, 
                   color='steelblue', edgecolor='black', linewidth=0.5)
            mean_val = data.mean()
            median_val = data.median()
            std_val = data.std()
            
            stats_text = f'Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd: {std_val:.2f}'
            ax.text(0.98, 0.98, stats_text, 
                   transform=ax.transAxes, 
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=7)
        
        # Formatting
        ax.set_title(feature, fontsize=10, fontweight='bold', pad=10)
        ax.set_xlabel('Value', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Rotate x-axis labels if needed
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Save to PDF
    plt.savefig(output_pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Plot saved to: {output_pdf_path}")
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_feature_distributions.py <input_csv_file> [output_pdf_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Default output filename if not provided
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        # Generate output filename from input filename
        if input_file.endswith('.csv'):
            output_file = input_file[:-4] + '_feature_distributions.pdf'
        else:
            output_file = input_file + '_feature_distributions.pdf'
    
    print(f"Reading CSV file: {input_file}")
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)
    
    # Identify numeric features
    numeric_features = identify_numeric_features(df)
    print(f"Found {len(numeric_features)} numeric features to plot")
    
    if len(numeric_features) == 0:
        print("No numeric features found to plot!")
        sys.exit(1)
    
    # Plot distributions
    print("Generating plots...")
    plot_feature_distributions(df, numeric_features, output_file)
    print("Done!")


if __name__ == "__main__":
    main()

