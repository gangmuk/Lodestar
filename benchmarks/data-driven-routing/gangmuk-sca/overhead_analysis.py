#!/usr/bin/env python3
"""
Overhead Analysis Visualization Script
Creates bar charts showing how different component overheads scale with RPS
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_overhead_data():
    """Load overhead data from the three CSV files"""
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

def create_main_components_chart(all_data):
    """Create a chart showing main component overheads"""
    # Main components to focus on (excluding sub-components)
    main_components = [
        'handle_infer_end_to_end',
        'handle_infer_preprocess_overhead', 
        'handle_infer_normalize',
        'handle_infer_encode',
        'handle_infer_calling_infer_from_tensor',
        'handle_infer_remaining_work',
        'encode_end_to_end',
        'preprocess_preprocess_unified_inference',
        'infer_from_tensor_model_inference'
    ]
    
    # Prepare data for plotting
    rps_levels = []
    component_data = {comp: [] for comp in main_components}
    
    for rps_label, df in all_data.items():
        rps_levels.append(rps_label)
        for comp in main_components:
            comp_row = df[df['Component'] == comp]
            if not comp_row.empty:
                component_data[comp].append(comp_row['Avg_ms'].iloc[0])
            else:
                component_data[comp].append(0)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(rps_levels))
    width = 0.08
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(main_components)))
    
    for i, (comp, values) in enumerate(component_data.items()):
        # Clean up component names for display
        display_name = comp.replace('handle_infer_', '').replace('_', ' ').title()
        if display_name.startswith('End To End'):
            display_name = 'End-to-End'
        elif display_name.startswith('Preprocess Overhead'):
            display_name = 'Preprocess'
        elif display_name.startswith('Calling Infer From Tensor'):
            display_name = 'Model Call'
        elif display_name.startswith('Remaining Work'):
            display_name = 'Remaining'
        
        ax.bar(x + i * width, values, width, label=display_name, color=colors[i])
    
    ax.set_xlabel('Requests Per Second (RPS)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Overhead (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Component Overhead Scaling with RPS', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(main_components) - 1) / 2)
    ax.set_xticklabels(rps_levels)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (comp, values) in enumerate(component_data.items()):
        for j, v in enumerate(values):
            if v > 0:  # Only show labels for non-zero values
                ax.text(j + i * width, v + max(values) * 0.01, f'{v:.1f}', 
                       ha='center', va='bottom', fontsize=8, rotation=90)
    
    plt.tight_layout()
    plt.savefig('overhead_main_components.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_preprocessing_breakdown(all_data):
    """Create a detailed chart for preprocessing components"""
    preprocess_components = [
        'preprocess_json_parse_overhead',
        'preprocess_numeric_conversion_overhead', 
        'preprocess_create_df_overhead',
        'preprocess_preprocess_unified_inference'
    ]
    
    # Prepare data
    rps_levels = []
    component_data = {comp: [] for comp in preprocess_components}
    
    for rps_label, df in all_data.items():
        rps_levels.append(rps_label)
        for comp in preprocess_components:
            comp_row = df[df['Component'] == comp]
            if not comp_row.empty:
                component_data[comp].append(comp_row['Avg_ms'].iloc[0])
            else:
                component_data[comp].append(0)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(rps_levels))
    width = 0.2
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for i, (comp, values) in enumerate(component_data.items()):
        display_name = comp.replace('preprocess_', '').replace('_overhead', '').replace('_', ' ').title()
        if display_name.startswith('Preprocess Unified Inference'):
            display_name = 'Unified Inference'
        
        ax.bar(x + i * width, values, width, label=display_name, color=colors[i])
    
    ax.set_xlabel('Requests Per Second (RPS)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Overhead (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Preprocessing Component Breakdown', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(rps_levels)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (comp, values) in enumerate(component_data.items()):
        for j, v in enumerate(values):
            if v > 0:
                ax.text(j + i * width, v + max(values) * 0.01, f'{v:.1f}', 
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('overhead_preprocessing_breakdown.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_encoding_breakdown(all_data):
    """Create a detailed chart for encoding components"""
    encoding_components = [
        'encode_prepare_for_encoding',
        'encode_prepare_for_encoding.extract_pod_columns',
        'encode_prepare_for_encoding.extract_actions', 
        'encode_prepare_for_encoding.positional_encoding',
        'encode_post_process',
        'encode_end_to_end'
    ]
    
    # Prepare data
    rps_levels = []
    component_data = {comp: [] for comp in encoding_components}
    
    for rps_label, df in all_data.items():
        rps_levels.append(rps_label)
        for comp in encoding_components:
            comp_row = df[df['Component'] == comp]
            if not comp_row.empty:
                component_data[comp].append(comp_row['Avg_ms'].iloc[0])
            else:
                component_data[comp].append(0)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(rps_levels))
    width = 0.15
    
    colors = ['#FF9F43', '#10AC84', '#EE5A24', '#0984E3', '#A29BFE', '#FD79A8']
    
    for i, (comp, values) in enumerate(component_data.items()):
        if comp == 'encode_prepare_for_encoding':
            display_name = 'Prepare for Encoding'
        elif 'extract_pod_columns' in comp:
            display_name = 'Extract Pod Columns'
        elif 'extract_actions' in comp:
            display_name = 'Extract Actions'
        elif 'positional_encoding' in comp:
            display_name = 'Positional Encoding'
        elif comp == 'encode_post_process':
            display_name = 'Post Process'
        elif comp == 'encode_end_to_end':
            display_name = 'End-to-End'
        else:
            display_name = comp.replace('encode_', '').replace('_', ' ').title()
        
        ax.bar(x + i * width, values, width, label=display_name, color=colors[i])
    
    ax.set_xlabel('Requests Per Second (RPS)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Overhead (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Encoding Component Breakdown', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(rps_levels)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (comp, values) in enumerate(component_data.items()):
        for j, v in enumerate(values):
            if v > 0:
                ax.text(j + i * width, v + max(values) * 0.01, f'{v:.1f}', 
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('overhead_encoding_breakdown.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_scaling_analysis(all_data):
    """Create a scaling analysis showing the dramatic increase at high RPS"""
    # Focus on the most problematic components
    key_components = [
        'handle_infer_end_to_end',
        'handle_infer_preprocess_overhead',
        'handle_infer_calling_infer_from_tensor',
        'preprocess_preprocess_unified_inference',
        'preprocess_json_parse_overhead',
        'preprocess_create_df_overhead'
    ]
    
    # Prepare data
    rps_values = [3, 5, 100]  # Actual RPS values
    component_data = {comp: [] for comp in key_components}
    
    for rps_label, df in all_data.items():
        for comp in key_components:
            comp_row = df[df['Component'] == comp]
            if not comp_row.empty:
                component_data[comp].append(comp_row['Avg_ms'].iloc[0])
            else:
                component_data[comp].append(0)
    
    # Create log-scale plot to show the dramatic scaling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Linear scale
    for comp, values in component_data.items():
        display_name = comp.replace('handle_infer_', '').replace('preprocess_', '').replace('_overhead', '').replace('_', ' ').title()
        if display_name.startswith('End To End'):
            display_name = 'End-to-End'
        elif display_name.startswith('Preprocess Overhead'):
            display_name = 'Preprocess'
        elif display_name.startswith('Calling Infer From Tensor'):
            display_name = 'Model Call'
        elif display_name.startswith('Preprocess Unified Inference'):
            display_name = 'Unified Inference'
        elif display_name.startswith('Json Parse'):
            display_name = 'JSON Parse'
        elif display_name.startswith('Create Df'):
            display_name = 'Create DataFrame'
        
        ax1.plot(rps_values, values, marker='o', linewidth=2, label=display_name)
    
    ax1.set_xlabel('Requests Per Second (RPS)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Overhead (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Component Overhead Scaling (Linear Scale)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Log scale
    for comp, values in component_data.items():
        display_name = comp.replace('handle_infer_', '').replace('preprocess_', '').replace('_overhead', '').replace('_', ' ').title()
        if display_name.startswith('End To End'):
            display_name = 'End-to-End'
        elif display_name.startswith('Preprocess Overhead'):
            display_name = 'Preprocess'
        elif display_name.startswith('Calling Infer From Tensor'):
            display_name = 'Model Call'
        elif display_name.startswith('Preprocess Unified Inference'):
            display_name = 'Unified Inference'
        elif display_name.startswith('Json Parse'):
            display_name = 'JSON Parse'
        elif display_name.startswith('Create Df'):
            display_name = 'Create DataFrame'
        
        ax2.plot(rps_values, values, marker='o', linewidth=2, label=display_name)
    
    ax2.set_xlabel('Requests Per Second (RPS)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Overhead (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('Component Overhead Scaling (Log Scale)', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('overhead_scaling_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_summary_statistics(all_data):
    """Print summary statistics about the scaling behavior"""
    print("\n" + "="*80)
    print("OVERHEAD SCALING ANALYSIS SUMMARY")
    print("="*80)
    
    for rps_label, df in all_data.items():
        print(f"\n{rps_label}:")
        print("-" * 40)
        
        # Get end-to-end latency
        e2e_row = df[df['Component'] == 'handle_infer_end_to_end']
        if not e2e_row.empty:
            e2e_latency = e2e_row['Avg_ms'].iloc[0]
            print(f"End-to-End Latency: {e2e_latency:.2f} ms")
        
        # Get top 5 components by overhead
        top_components = df.nlargest(5, 'Avg_ms')[['Component', 'Avg_ms']]
        print("Top 5 Components by Overhead:")
        for _, row in top_components.iterrows():
            comp_name = row['Component'].replace('handle_infer_', '').replace('preprocess_', '').replace('_overhead', '')
            print(f"  {comp_name}: {row['Avg_ms']:.2f} ms")
    
    print("\n" + "="*80)
    print("SCALING OBSERVATIONS:")
    print("="*80)
    
    # Calculate scaling factors
    rps_3_data = all_data.get('3 RPS')
    rps_100_data = all_data.get('100 RPS')
    
    if rps_3_data is not None and rps_100_data is not None:
        print("\nScaling from 3 RPS to 100 RPS (33x increase):")
        
        key_components = [
            'handle_infer_end_to_end',
            'handle_infer_preprocess_overhead',
            'handle_infer_calling_infer_from_tensor',
            'preprocess_preprocess_unified_inference',
            'preprocess_json_parse_overhead'
        ]
        
        for comp in key_components:
            comp_3 = rps_3_data[rps_3_data['Component'] == comp]
            comp_100 = rps_100_data[rps_100_data['Component'] == comp]
            
            if not comp_3.empty and not comp_100.empty:
                val_3 = comp_3['Avg_ms'].iloc[0]
                val_100 = comp_100['Avg_ms'].iloc[0]
                scaling_factor = val_100 / val_3 if val_3 > 0 else float('inf')
                
                comp_name = comp.replace('handle_infer_', '').replace('preprocess_', '').replace('_overhead', '')
                print(f"  {comp_name}: {val_3:.2f}ms → {val_100:.2f}ms ({scaling_factor:.1f}x)")

def main():
    """Main function to run the analysis"""
    print("Loading overhead data...")
    all_data = load_overhead_data()
    
    if not all_data:
        print("No data files found!")
        return
    
    print("\nCreating visualizations...")
    
    # Create all charts
    create_main_components_chart(all_data)
    create_preprocessing_breakdown(all_data)
    create_encoding_breakdown(all_data)
    create_scaling_analysis(all_data)
    
    # Print summary
    print_summary_statistics(all_data)
    
    print("\nVisualization complete! Generated files:")
    print("- overhead_main_components.png")
    print("- overhead_preprocessing_breakdown.png") 
    print("- overhead_encoding_breakdown.png")
    print("- overhead_scaling_analysis.png")

if __name__ == "__main__":
    main()
