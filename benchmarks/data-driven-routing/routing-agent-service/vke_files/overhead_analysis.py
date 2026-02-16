#!/usr/bin/env python3
"""
Overhead Analysis and Visualization Tool
Analyzes routing agent overhead logs and generates publication-quality plots.
"""
import argparse
import pathlib
import re
import statistics
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
})

OVERHEAD_RE = re.compile(r'overhead_log: oh, (.*)')


def parse_overhead_entries(text: str) -> List[Dict[str, float]]:
    """Parse overhead log entries from text."""
    entries: List[Dict[str, float]] = []
    for match in OVERHEAD_RE.finditer(text):
        line = match.group(1)
        parts = [p.strip() for p in line.split(',')]
        values: Dict[str, float] = {}
        for part in parts:
            if ': ' not in part:
                continue
            key, value = part.split(': ', 1)
            if value.endswith('ms'):
                try:
                    val = float(value[:-2])
                    # Filter out placeholder values
                    if val >= 0:
                        values[key] = val
                except ValueError:
                    continue
        if values:
            entries.append(values)
    return entries


def percentile(sorted_vals: List[float], p: float) -> float:
    """Calculate percentile from sorted values."""
    if not sorted_vals:
        return float('nan')
    idx = int(p * (len(sorted_vals) - 1))
    return sorted_vals[idx]


def summarize(entries: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Compute statistics for each metric."""
    all_keys = sorted({k for e in entries for k in e.keys()})
    summary: Dict[str, Dict[str, float]] = {}
    for key in all_keys:
        vals = [e[key] for e in entries if key in e]
        if not vals:
            continue
        vals_sorted = sorted(vals)
        summary[key] = {
            "count": float(len(vals)),
            "mean": statistics.mean(vals),
            "median": statistics.median(vals_sorted),
            "p90": percentile(vals_sorted, 0.90),
            "p95": percentile(vals_sorted, 0.95),
            "p99": percentile(vals_sorted, 0.99),
            "max": max(vals_sorted),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        }
    return summary


def categorize_metrics(summary: Dict[str, Dict[str, float]]) -> Dict[str, List[str]]:
    """Categorize metrics into logical groups."""
    categories = {
        'End-to-End': [],
        'Preprocessing': [],
        'Encoding': [],
        'Inference': [],
        'Other': [],
    }
    
    for metric in summary.keys():
        if 'end_to_end' in metric:
            categories['End-to-End'].append(metric)
        elif 'preprocess' in metric or 'normalize' in metric:
            categories['Preprocessing'].append(metric)
        elif 'encode' in metric:
            categories['Encoding'].append(metric)
        elif 'infer' in metric or 'inference' in metric:
            categories['Inference'].append(metric)
        else:
            categories['Other'].append(metric)
    
    return categories


def calculate_contributions(summary: Dict[str, Dict[str, float]], 
                          stat: str = 'mean') -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Calculate contribution percentages for top-level components.
    Returns (component_values, component_percentages)
    """
    # Define the main components that sum to end-to-end
    main_components = [
        'handle_infer_preprocess_overhead',
        'handle_infer_normalize', 
        'handle_infer_encode',
        'handle_infer_contextual_bandit_infer',
        'handle_infer_remaining_work',
    ]
    
    total_key = 'handle_infer_end_to_end'
    
    if total_key not in summary:
        return {}, {}
    
    total_time = summary[total_key][stat]
    
    component_values = {}
    component_percentages = {}
    
    for comp in main_components:
        if comp in summary:
            val = summary[comp][stat]
            component_values[comp] = val
            component_percentages[comp] = (val / total_time * 100) if total_time > 0 else 0
    
    # Calculate "other" component
    accounted = sum(component_values.values())
    other = max(0, total_time - accounted)
    component_values['other'] = other
    component_percentages['other'] = (other / total_time * 100) if total_time > 0 else 0
    
    return component_values, component_percentages


def print_analysis(summary: Dict[str, Dict[str, float]]) -> None:
    """Print detailed text analysis."""
    print("=" * 80)
    print("OVERHEAD ANALYSIS REPORT")
    print("=" * 80)
    print(f"\nTotal requests analyzed: {int(summary[list(summary.keys())[0]]['count'])}")
    
    # Main components analysis
    print("\n" + "=" * 80)
    print("MAIN COMPONENT BREAKDOWN (Mean)")
    print("=" * 80)
    
    comp_values, comp_pct = calculate_contributions(summary, 'mean')
    
    if comp_values:
        total = summary['handle_infer_end_to_end']['mean']
        print(f"\nTotal End-to-End Time: {total:.2f} ms\n")
        
        # Sort by value
        sorted_comps = sorted(comp_values.items(), key=lambda x: x[1], reverse=True)
        
        for comp, value in sorted_comps:
            pct = comp_pct[comp]
            comp_name = comp.replace('handle_infer_', '').replace('_', ' ').title()
            print(f"  {comp_name:40s}: {value:8.2f} ms ({pct:5.1f}%)")
    
    # Detailed breakdown by category
    print("\n" + "=" * 80)
    print("DETAILED METRICS BY CATEGORY")
    print("=" * 80)
    
    categories = categorize_metrics(summary)
    
    for cat_name, metrics in categories.items():
        if not metrics:
            continue
            
        print(f"\n{cat_name}:")
        print("-" * 80)
        
        # Sort by mean value
        metrics_sorted = sorted(metrics, key=lambda m: summary[m]['mean'], reverse=True)
        
        for metric in metrics_sorted[:10]:  # Top 10 per category
            s = summary[metric]
            short_name = metric.replace('handle_infer_', '').replace('encode_', '').replace('infer_from_tensor_', '')
            print(f"  {short_name:45s}: mean={s['mean']:7.2f} ms  "
                  f"p50={s['median']:6.1f}  p90={s['p90']:6.1f}  p99={s['p99']:6.1f}  max={s['max']:6.1f}")
    
    # Top overall contributors
    print("\n" + "=" * 80)
    print("TOP 15 CONTRIBUTORS (by Mean)")
    print("=" * 80)
    
    all_metrics = sorted(summary.items(), key=lambda x: x[1]['mean'], reverse=True)
    print(f"\n{'Metric':<50s} {'Mean':>8s} {'P50':>8s} {'P90':>8s} {'P99':>8s} {'Max':>8s}")
    print("-" * 80)
    
    for metric, stats in all_metrics[:15]:
        short_name = metric[:48] if len(metric) <= 48 else metric[:45] + "..."
        print(f"{short_name:<50s} {stats['mean']:8.2f} {stats['median']:8.1f} "
              f"{stats['p90']:8.1f} {stats['p99']:8.1f} {stats['max']:8.1f}")


def create_plots(summary: Dict[str, Dict[str, float]], output_path: pathlib.Path) -> None:
    """Generate publication-quality plots."""
    
    with PdfPages(output_path) as pdf:
        
        # Page 1: Main Component Breakdown
        fig = plt.figure(figsize=(12, 8))
        
        # Pie chart
        ax1 = plt.subplot(2, 2, 1)
        comp_values, comp_pct = calculate_contributions(summary, 'mean')
        
        if comp_values:
            # Prepare data
            labels = []
            values = []
            colors = plt.cm.Set3(np.linspace(0, 1, len(comp_values)))
            
            sorted_comps = sorted(comp_values.items(), key=lambda x: x[1], reverse=True)
            for comp, value in sorted_comps:
                label = comp.replace('handle_infer_', '').replace('_', ' ').title()
                labels.append(label)
                values.append(value)
            
            wedges, texts, autotexts = ax1.pie(values, labels=labels, autopct='%1.1f%%',
                                                colors=colors, startangle=90)
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(9)
                autotext.set_weight('bold')
            
            ax1.set_title('Component Contribution to\nEnd-to-End Latency (Mean)', fontsize=12, weight='bold')
        
        # Bar chart - mean values
        ax2 = plt.subplot(2, 2, 2)
        if comp_values:
            sorted_comps = sorted(comp_values.items(), key=lambda x: x[1], reverse=True)
            names = [c[0].replace('handle_infer_', '').replace('_', '\n') for c in sorted_comps]
            vals = [c[1] for c in sorted_comps]
            
            bars = ax2.barh(range(len(names)), vals, color=colors)
            ax2.set_yticks(range(len(names)))
            ax2.set_yticklabels(names, fontsize=9)
            ax2.set_xlabel('Time (ms)', fontsize=11)
            ax2.set_title('Mean Latency by Component', fontsize=12, weight='bold')
            ax2.grid(axis='x', alpha=0.3, linestyle='--')
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, vals)):
                ax2.text(val + max(vals)*0.01, i, f'{val:.1f}ms', 
                        va='center', fontsize=8)
        
        # Stacked bar - comparing percentiles
        ax3 = plt.subplot(2, 1, 2)
        
        main_components = [
            'handle_infer_preprocess_overhead',
            'handle_infer_normalize',
            'handle_infer_encode',
            'handle_infer_contextual_bandit_infer',
        ]
        
        stats_to_plot = ['mean', 'median', 'p90', 'p99']
        x_pos = np.arange(len(stats_to_plot))
        
        bottoms = np.zeros(len(stats_to_plot))
        colors_stack = plt.cm.Set2(np.linspace(0, 1, len(main_components)))
        
        for idx, comp in enumerate(main_components):
            if comp in summary:
                values = [summary[comp][stat] for stat in stats_to_plot]
                label = comp.replace('handle_infer_', '').replace('_', ' ').title()
                ax3.bar(x_pos, values, bottom=bottoms, label=label, 
                       color=colors_stack[idx], width=0.6)
                bottoms += values
        
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(['Mean', 'Median (P50)', 'P90', 'P99'], fontsize=10)
        ax3.set_ylabel('Time (ms)', fontsize=11)
        ax3.set_title('Latency Distribution Across Percentiles', fontsize=12, weight='bold')
        ax3.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax3.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 2: Detailed Component Analysis
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Encoding breakdown
        ax = axes[0, 0]
        encode_metrics = [m for m in summary.keys() if 'encode_prepare_for_encoding.' in m]
        if encode_metrics:
            encode_sorted = sorted(encode_metrics, 
                                  key=lambda m: summary[m]['mean'], reverse=True)[:8]
            names = [m.replace('encode_prepare_for_encoding.', '') for m in encode_sorted]
            values = [summary[m]['mean'] for m in encode_sorted]
            
            colors_encode = plt.cm.Pastel1(np.linspace(0, 1, len(names)))
            bars = ax.barh(range(len(names)), values, color=colors_encode)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=9)
            ax.set_xlabel('Mean Time (ms)', fontsize=10)
            ax.set_title('Encoding Sub-Components', fontsize=11, weight='bold')
            ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Inference breakdown
        ax = axes[0, 1]
        infer_metrics = [m for m in summary.keys() if 'infer_from_tensor' in m]
        if infer_metrics:
            infer_sorted = sorted(infer_metrics,
                                 key=lambda m: summary[m]['mean'], reverse=True)
            names = [m.replace('infer_from_tensor_', '') for m in infer_sorted]
            values = [summary[m]['mean'] for m in infer_sorted]
            
            colors_infer = plt.cm.Pastel2(np.linspace(0, 1, len(names)))
            bars = ax.barh(range(len(names)), values, color=colors_infer)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=9)
            ax.set_xlabel('Mean Time (ms)', fontsize=10)
            ax.set_title('Inference Sub-Components', fontsize=11, weight='bold')
            ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Preprocessing breakdown
        ax = axes[1, 0]
        preprocess_metrics = [m for m in summary.keys() if 'preprocess_' in m and m != 'handle_infer_preprocess_overhead']
        if preprocess_metrics:
            preprocess_sorted = sorted(preprocess_metrics,
                                      key=lambda m: summary[m]['mean'], reverse=True)[:8]
            names = [m.replace('preprocess_', '') for m in preprocess_sorted]
            values = [summary[m]['mean'] for m in preprocess_sorted]
            
            colors_pre = plt.cm.Set3(np.linspace(0, 1, len(names)))
            bars = ax.barh(range(len(names)), values, color=colors_pre)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=8)
            ax.set_xlabel('Mean Time (ms)', fontsize=10)
            ax.set_title('Preprocessing Sub-Components', fontsize=11, weight='bold')
            ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Percentile comparison for top metrics
        ax = axes[1, 1]
        top_metrics = sorted(summary.items(), key=lambda x: x[1]['mean'], reverse=True)[:6]
        
        x = np.arange(len(top_metrics))
        width = 0.15
        
        percentiles = ['mean', 'median', 'p90', 'p95', 'p99']
        colors_pct = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A0572']
        
        for i, pct in enumerate(percentiles):
            values = [m[1][pct] for m in top_metrics]
            ax.bar(x + i*width, values, width, label=pct.upper(), color=colors_pct[i])
        
        ax.set_xlabel('Component', fontsize=10)
        ax.set_ylabel('Time (ms)', fontsize=10)
        ax.set_title('Latency Distribution for Top Components', fontsize=11, weight='bold')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels([m[0].replace('handle_infer_', '')[:15] for m in top_metrics], 
                          rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=8, ncol=5, loc='upper left')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 3: Heatmap and statistical analysis
        fig = plt.figure(figsize=(12, 10))
        
        # Heatmap of top metrics across percentiles
        ax1 = plt.subplot(2, 1, 1)
        
        top_n = 15
        top_metrics = sorted(summary.items(), key=lambda x: x[1]['mean'], reverse=True)[:top_n]
        
        stats_cols = ['mean', 'median', 'p90', 'p95', 'p99', 'max']
        heatmap_data = np.zeros((len(top_metrics), len(stats_cols)))
        
        for i, (metric, stats) in enumerate(top_metrics):
            for j, stat in enumerate(stats_cols):
                heatmap_data[i, j] = stats[stat]
        
        # Normalize for better visualization
        heatmap_data_norm = heatmap_data / heatmap_data.max(axis=1, keepdims=True)
        
        im = ax1.imshow(heatmap_data_norm, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        
        ax1.set_xticks(np.arange(len(stats_cols)))
        ax1.set_yticks(np.arange(len(top_metrics)))
        ax1.set_xticklabels([s.upper() for s in stats_cols], fontsize=10)
        ax1.set_yticklabels([m[0].replace('handle_infer_', '')[:35] for m in top_metrics], 
                           fontsize=8)
        
        # Add text annotations with actual values
        for i in range(len(top_metrics)):
            for j in range(len(stats_cols)):
                text = ax1.text(j, i, f'{heatmap_data[i, j]:.0f}',
                               ha="center", va="center", color="black" if heatmap_data_norm[i, j] < 0.5 else "white",
                               fontsize=7)
        
        ax1.set_title('Latency Heatmap: Top Contributors (ms)', fontsize=12, weight='bold', pad=10)
        
        cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        cbar.set_label('Normalized Intensity', rotation=270, labelpad=15, fontsize=10)
        
        # Coefficient of Variation analysis
        ax2 = plt.subplot(2, 1, 2)
        
        cv_data = []
        for metric, stats in summary.items():
            if stats['mean'] > 1:  # Only consider metrics with mean > 1ms
                cv = (stats['std'] / stats['mean']) * 100 if stats['mean'] > 0 else 0
                cv_data.append((metric, cv, stats['mean']))
        
        cv_sorted = sorted(cv_data, key=lambda x: x[1], reverse=True)[:15]
        
        names = [c[0].replace('handle_infer_', '').replace('encode_', '').replace('infer_from_tensor_', '')[:30] 
                for c in cv_sorted]
        cvs = [c[1] for c in cv_sorted]
        means = [c[2] for c in cv_sorted]
        
        # Create color map based on mean value
        norm = plt.Normalize(vmin=min(means), vmax=max(means))
        colors_cv = plt.cm.viridis(norm(means))
        
        bars = ax2.barh(range(len(names)), cvs, color=colors_cv)
        ax2.set_yticks(range(len(names)))
        ax2.set_yticklabels(names, fontsize=8)
        ax2.set_xlabel('Coefficient of Variation (%)', fontsize=10)
        ax2.set_title('Latency Variability Analysis (CV = σ/μ × 100%)', 
                     fontsize=12, weight='bold')
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add colorbar for mean values
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])
        cbar2 = plt.colorbar(sm, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label('Mean Latency (ms)', rotation=270, labelpad=15, fontsize=9)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Metadata page
        d = pdf.infodict()
        d['Title'] = 'Routing Agent Overhead Analysis'
        d['Author'] = 'Overhead Analysis Tool'
        d['Subject'] = 'Performance Analysis'
        d['Keywords'] = 'Latency, Performance, Routing Agent'


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Analyze overhead logs and generate publication-quality visualizations.'
    )
    parser.add_argument('log_path', type=pathlib.Path, 
                       help='Path to routing-agent service log file')
    parser.add_argument('--output', '-o', type=pathlib.Path, 
                       default=None,
                       help='Output PDF path (default: same directory as log, named overhead_analysis.pdf)')
    
    args = parser.parse_args()
    
    # Read and parse log
    print("Reading log file...")
    text = args.log_path.read_text(errors="ignore")
    entries = parse_overhead_entries(text)
    
    if not entries:
        print("ERROR: No overhead log entries found!")
        return
    
    print(f"Found {len(entries)} requests with overhead data")
    
    # Compute statistics
    print("Computing statistics...")
    summary = summarize(entries)
    
    # Print analysis
    print_analysis(summary)
    
    # Generate plots
    if args.output:
        output_path = args.output
    else:
        output_path = args.log_path.parent / 'overhead_analysis.pdf'
    
    print(f"\nGenerating plots...")
    create_plots(summary, output_path)
    
    print(f"\n{'='*80}")
    print(f"✓ Analysis complete! PDF saved to: {output_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()




















