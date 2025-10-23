#!/usr/bin/env python3
"""
Corrected Preprocessing Overhead Analysis
Shows individual preprocessing components stacked, with unified inference as a line
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("Set2")

def load_preprocess_data():
    """Load preprocessing overhead data from the three CSV files"""
    data_files = [
        ('5 RPS', 'overhead_results/rps_5_dur_10s_20251023_205404_components.csv'),
        ('10 RPS', 'overhead_results/rps_10_dur_10s_20251023_205414_components.csv'),
        ('100 RPS', 'overhead_results/rps_100_dur_10s_20251023_205425_components.csv')
    ]
    
    all_data = {}
    
    for rps_label, file_path in data_files:
        if Path(file_path).exists():
            df = pd.read_csv(file_path)
            all_data[rps_label] = df
            print(f"Loaded {rps_label}: {len(df)} components")
        else:
            print(f"Warning: File not found: {file_path}")
    
    return all_data

def create_corrected_preprocess_stacked_chart(all_data):
    """Create a stacked bar chart with individual components + unified inference as line"""
    # Individual preprocessing components (excluding unified inference)
    individual_components = [
        'preprocess_json_parse_overhead',
        'preprocess_numeric_conversion_overhead', 
        'preprocess_get_value_overhead',
        'preprocess_create_df_overhead',
        'preprocess_pod_index_overhead'
    ]
    
    # Unified inference (end-to-end preprocessing)
    unified_component = 'preprocess_preprocess_unified_inference'
    
    # Prepare data
    rps_levels = []
    individual_data = {comp: [] for comp in individual_components}
    unified_data = []
    
    for rps_label, df in all_data.items():
        rps_levels.append(rps_label)
        
        # Get individual components
        for comp in individual_components:
            comp_row = df[df['Component'] == comp]
            if not comp_row.empty:
                individual_data[comp].append(comp_row['Avg_ms'].iloc[0])
            else:
                individual_data[comp].append(0)
        
        # Get unified inference
        unified_row = df[df['Component'] == unified_component]
        if not unified_row.empty:
            unified_data.append(unified_row['Avg_ms'].iloc[0])
        else:
            unified_data.append(0)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define colors for individual components
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
    
    # Create stacked bars for individual components
    bottom = np.zeros(len(rps_levels))
    
    for i, (comp, values) in enumerate(individual_data.items()):
        display_name = comp.replace('preprocess_', '').replace('_overhead', '').replace('_', ' ').title()
        if display_name.startswith('Json Parse'):
            display_name = 'JSON Parse'
        elif display_name.startswith('Numeric Conversion'):
            display_name = 'Numeric Conversion'
        elif display_name.startswith('Get Value'):
            display_name = 'Get Value'
        elif display_name.startswith('Create Df'):
            display_name = 'Create DataFrame'
        elif display_name.startswith('Pod Index'):
            display_name = 'Pod Index'
        
        ax.bar(rps_levels, values, bottom=bottom, label=display_name, color=colors[i])
        bottom += values
    
    # Add unified inference as a line
    ax.plot(rps_levels, unified_data, marker='o', linewidth=4, markersize=10, 
            color='#FF9FF3', label='Unified Inference (End-to-End)', linestyle='--')
    
    ax.set_xlabel('Requests Per Second (RPS)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Preprocessing Overhead (ms)', fontsize=14, fontweight='bold')
    ax.set_title('Preprocessing Component Breakdown\n(Individual Components Stacked + Unified Inference)', 
                fontsize=16, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add total values on top of stacked bars
    for i, total in enumerate(bottom):
        ax.text(i, total + max(bottom) * 0.02, f'{total:.1f}ms', 
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add unified inference values
    for i, unified_val in enumerate(unified_data):
        ax.text(i, unified_val + max(max(bottom), max(unified_data)) * 0.05, 
               f'{unified_val:.1f}ms', ha='center', va='bottom', 
               fontsize=11, fontweight='bold', color='#FF9FF3')
    
    plt.tight_layout()
    plt.savefig('preprocess_corrected_stacked.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_corrected_summary(all_data):
    """Print corrected summary of preprocessing components"""
    print("\n" + "="*80)
    print("CORRECTED PREPROCESSING OVERHEAD ANALYSIS")
    print("="*80)
    
    individual_components = [
        'preprocess_json_parse_overhead',
        'preprocess_numeric_conversion_overhead', 
        'preprocess_get_value_overhead',
        'preprocess_create_df_overhead',
        'preprocess_pod_index_overhead'
    ]
    
    unified_component = 'preprocess_preprocess_unified_inference'
    
    for rps_label, df in all_data.items():
        print(f"\n{rps_label}:")
        print("-" * 50)
        
        # Individual components
        total_individual = 0
        print("  Individual Components:")
        for comp in individual_components:
            comp_row = df[df['Component'] == comp]
            if not comp_row.empty:
                overhead = comp_row['Avg_ms'].iloc[0]
                total_individual += overhead
                comp_name = comp.replace('preprocess_', '').replace('_overhead', '').replace('_', ' ').title()
                print(f"    {comp_name}: {overhead:.2f} ms")
        
        print(f"    {'Total Individual':<20}: {total_individual:.2f} ms")
        
        # Unified inference
        unified_row = df[df['Component'] == unified_component]
        if not unified_row.empty:
            unified_overhead = unified_row['Avg_ms'].iloc[0]
            print(f"  {'Unified Inference (E2E)':<20}: {unified_overhead:.2f} ms")
            
            # Calculate overhead difference
            overhead_diff = unified_overhead - total_individual
            print(f"  {'Overhead Difference':<20}: {overhead_diff:.2f} ms")
    
    # Calculate scaling factors
    rps_3_data = all_data.get('3 RPS')
    rps_100_data = all_data.get('100 RPS')
    
    if rps_3_data is not None and rps_100_data is not None:
        print("\n" + "="*80)
        print("PREPROCESSING SCALING ANALYSIS (3 RPS → 100 RPS):")
        print("="*80)
        
        print("Individual Components:")
        for comp in individual_components:
            comp_3 = rps_3_data[rps_3_data['Component'] == comp]
            comp_100 = rps_100_data[rps_100_data['Component'] == comp]
            
            if not comp_3.empty and not comp_100.empty:
                val_3 = comp_3['Avg_ms'].iloc[0]
                val_100 = comp_100['Avg_ms'].iloc[0]
                scaling_factor = val_100 / val_3 if val_3 > 0 else float('inf')
                
                comp_name = comp.replace('preprocess_', '').replace('_overhead', '').replace('_', ' ').title()
                print(f"  {comp_name:<25}: {val_3:.2f}ms → {val_100:.2f}ms ({scaling_factor:.1f}x)")
        
        # Unified inference scaling
        unified_3 = rps_3_data[rps_3_data['Component'] == unified_component]
        unified_100 = rps_100_data[rps_100_data['Component'] == unified_component]
        
        if not unified_3.empty and not unified_100.empty:
            val_3 = unified_3['Avg_ms'].iloc[0]
            val_100 = unified_100['Avg_ms'].iloc[0]
            scaling_factor = val_100 / val_3 if val_3 > 0 else float('inf')
            
            print(f"  {'Unified Inference (E2E)':<25}: {val_3:.2f}ms → {val_100:.2f}ms ({scaling_factor:.1f}x)")

def main():
    """Main function to run the corrected preprocessing analysis"""
    print("Loading preprocessing overhead data...")
    all_data = load_preprocess_data()
    
    if not all_data:
        print("No data files found!")
        return
    
    print("\nCreating corrected preprocessing visualization...")
    
    # Create corrected chart
    create_corrected_preprocess_stacked_chart(all_data)
    
    # Print summary
    print_corrected_summary(all_data)
    
    print("\nCorrected preprocessing visualization complete! Generated file:")
    print("- preprocess_corrected_stacked.png")

if __name__ == "__main__":
    main()
