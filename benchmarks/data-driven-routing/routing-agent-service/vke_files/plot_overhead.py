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
import numpy as np

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

# Metrics to exclude from all analysis (one-off reload events, not per-request overhead).
EXCLUDED_METRICS = {
    'infer_from_tensor_get_agent_reload',
    'infer_from_tensor_get_agent_async_reload_started',
}

# Additive components that sum to end-to-end.
# NOTE: calling_infer_from_tensor is the additive component (wraps tensor transfer + inference).
#       contextual_bandit_infer is an inner sub-measurement, NOT the additive one.
# NOTE: replace_podid_overhead is a subset of request_prepare — don't list both.
#       contextual_bandit_infer is a subset of calling_infer_from_tensor — don't list both.
MAIN_ADDITIVE_COMPONENTS = [
    'handle_infer_request_prepare',
    'handle_infer_preprocess_overhead',
    'handle_infer_distribution_monitor',
    'handle_infer_ood_check',
    'handle_infer_normalize',
    'handle_infer_encode',
    'handle_infer_contextual_bandit_write_lock',
    'handle_infer_contextual_bandit_create',
    'handle_infer_calling_infer_from_tensor',
    'handle_infer_remaining_work',
]

# Nice display names for main components
COMPONENT_DISPLAY_NAMES = {
    'handle_infer_request_prepare': 'Request Prepare',
    'handle_infer_preprocess_overhead': 'Preprocess',
    'handle_infer_distribution_monitor': 'Distribution Monitor',
    'handle_infer_ood_check': 'OOD Check',
    'handle_infer_normalize': 'Normalize',
    'handle_infer_encode': 'Encode',
    'handle_infer_contextual_bandit_write_lock': 'CB Write Lock',
    'handle_infer_contextual_bandit_create': 'CB Create',
    'handle_infer_calling_infer_from_tensor': 'Calling Infer From Tensor',
    'handle_infer_remaining_work': 'Remaining Work',
    'other': 'Other',
}


def parse_overhead_entries(text: str) -> List[Dict[str, float]]:
    """Parse overhead log entries from text."""
    entries: List[Dict[str, float]] = []
    for line in text.splitlines():
        match = OVERHEAD_RE.search(line)
        if not match:
            continue
        payload = match.group(1)
        parts = [p.strip() for p in payload.split(',')]
        values: Dict[str, float] = {}
        for part in parts:
            if ': ' not in part:
                continue
            key, value = part.split(': ', 1)
            if key in EXCLUDED_METRICS:
                continue
            if value.endswith('ms'):
                try:
                    val = float(value[:-2])
                    if val >= 0:  # Filter out -1000ms placeholder values
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
    """Categorize metrics into logical groups based on prefix."""
    categories = {
        'Handle Infer (End-to-End)': [],
        'Handle Infer (Pipeline Stages)': [],
        'Preprocessing': [],
        'Encoding': [],
        'Infer From Tensor': [],
        'Other': [],
    }

    for metric in summary.keys():
        if metric == 'handle_infer_end_to_end':
            categories['Handle Infer (End-to-End)'].append(metric)
        elif metric.startswith('handle_infer_'):
            categories['Handle Infer (Pipeline Stages)'].append(metric)
        elif metric.startswith('preprocess_'):
            categories['Preprocessing'].append(metric)
        elif metric.startswith('encode_'):
            categories['Encoding'].append(metric)
        elif metric.startswith('infer_from_tensor_'):
            categories['Infer From Tensor'].append(metric)
        else:
            categories['Other'].append(metric)

    return categories


def calculate_contributions(summary: Dict[str, Dict[str, float]],
                          stat: str = 'mean') -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Calculate contribution percentages for top-level additive components.
    Returns (component_values, component_percentages)
    """
    total_key = 'handle_infer_end_to_end'

    if total_key not in summary:
        return {}, {}

    total_time = summary[total_key][stat]

    component_values = {}
    component_percentages = {}

    for comp in MAIN_ADDITIVE_COMPONENTS:
        if comp in summary:
            val = summary[comp][stat]
            component_values[comp] = val
            component_percentages[comp] = (val / total_time * 100) if total_time > 0 else 0

    # Calculate "other" component (unaccounted time)
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

        # Sort by value, filter out zero-value components
        sorted_comps = sorted(comp_values.items(), key=lambda x: x[1], reverse=True)

        for comp, value in sorted_comps:
            pct = comp_pct[comp]
            comp_name = COMPONENT_DISPLAY_NAMES.get(comp, comp)
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
            print(f"  {metric:45s}: mean={s['mean']:7.2f} ms  "
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


def create_plot(summary: Dict[str, Dict[str, float]],
                entries: List[Dict[str, float]],
                output_path: pathlib.Path) -> None:
    """Generate a single PDF with stacked bar breakdown and per-request histogram."""
    plot_components = [c for c in MAIN_ADDITIVE_COMPONENTS if c in summary and summary[c]['mean'] > 0.01]

    total_key = 'handle_infer_end_to_end'
    stats_to_plot = ['p99', 'p95', 'p90', 'median', 'mean']
    stat_labels = ['P99', 'P95', 'P90', 'Median', 'Mean']

    # Build data matrix: rows = stats, cols = components + other
    data = []
    for stat in stats_to_plot:
        row = [summary[c][stat] if c in summary else 0.0 for c in plot_components]
        accounted = sum(row)
        total = summary[total_key][stat] if total_key in summary else accounted
        row.append(max(0, total - accounted))
        data.append(row)
    data = np.array(data)

    all_labels = [COMPONENT_DISPLAY_NAMES.get(c, c) for c in plot_components] + ['Other']

    # Drop components that are negligible across all stats (< 1% of max total)
    max_total = data.sum(axis=1).max()
    keep = data.max(axis=0) >= 0.005 * max_total
    data = data[:, keep]
    all_labels = [l for l, k in zip(all_labels, keep) if k]

    n_stats = len(stats_to_plot)
    n_comps = data.shape[1]

    cmap = plt.colormaps.get_cmap('tab10').resampled(max(n_comps, 1))
    colors = [cmap(i) for i in range(n_comps)]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 9),
                                        gridspec_kw={'height_ratios': [1, 0.7, 0.7]})

    # --- Top: stacked bar breakdown ---
    y_pos = np.arange(n_stats)
    bar_height = 0.55
    lefts = np.zeros(n_stats)

    bars_for_legend = []
    for ci in range(n_comps):
        vals = data[:, ci]
        b = ax1.barh(y_pos, vals, left=lefts, height=bar_height,
                      color=colors[ci], edgecolor='white', linewidth=0.4)
        bars_for_legend.append(b)
        for si in range(n_stats):
            seg_w = vals[si]
            row_total = data[si].sum()
            if row_total > 0 and seg_w / row_total > 0.04:
                cx = lefts[si] + seg_w / 2
                ax1.text(cx, y_pos[si], f'{seg_w:.1f}',
                         ha='center', va='center', fontsize=7, color='white',
                         fontweight='bold')
        lefts += vals

    totals_bar = data.sum(axis=1)
    for si in range(n_stats):
        ax1.text(totals_bar[si] + max_total * 0.01, y_pos[si],
                 f'{totals_bar[si]:.1f} ms', ha='left', va='center', fontsize=9)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(stat_labels)
    ax1.set_xlabel('Latency (ms)')
    ax1.set_xlim(0, max_total * 1.18)
    ax1.invert_yaxis()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='x', alpha=0.25, linestyle='--')

    ax1.legend([b[0] for b in bars_for_legend], all_labels,
               loc='upper center', bbox_to_anchor=(0.5, -0.22),
               ncol=min(4, n_comps), fontsize=8, frameon=False,
               columnspacing=1.0, handlelength=1.2)

    # --- Bottom: overhead distribution histogram ---
    per_req = [e[total_key] for e in entries if total_key in e]
    if per_req:
        mean_val = statistics.mean(per_req)
        ax2.hist(per_req, bins='auto', color='#4C72B0', edgecolor='white',
                 linewidth=0.5, alpha=0.85)
        ax2.axvline(mean_val, color='#C44E52', linestyle='--', linewidth=1.2,
                     label=f'Mean: {mean_val:.2f} ms  (n={len(per_req)})')
        ax2.set_xlabel('End-to-End Overhead (ms)')
        ax2.set_ylabel('Count')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.grid(axis='y', alpha=0.25, linestyle='--')
        ax2.legend(fontsize=9, frameon=False)

        # --- Bottom: CDF ---
        sorted_vals = np.sort(per_req)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax3.plot(sorted_vals, cdf, color='#4C72B0', linewidth=1.5)
        ax3.axvline(mean_val, color='#C44E52', linestyle='--', linewidth=1.2,
                     label=f'Mean: {mean_val:.2f} ms')
        ax3.set_xlabel('End-to-End Overhead (ms)')
        ax3.set_ylabel('CDF')
        ax3.set_ylim(0, 1.05)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.grid(alpha=0.25, linestyle='--')
        ax3.legend(fontsize=9, frameon=False)

    plt.tight_layout()
    fig.savefig(str(output_path), bbox_inches='tight')
    plt.close()
    print(f"  Saved figure to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Analyze overhead logs and generate publication-quality visualizations.'
    )
    parser.add_argument('log_path', type=pathlib.Path,
                       help='Path to routing-agent service log file')
    parser.add_argument('--output', '-o', type=pathlib.Path,
                       default=None,
                       help='Output PDF path (default: same directory as log, named overhead_analysis.pdf)')
    parser.add_argument('--skip-first', type=int, default=5,
                       help='Number of initial overhead entries to skip (default: 5 for warmup)')

    args = parser.parse_args()

    # Read and parse log
    print("Reading log file...")
    text = args.log_path.read_text(errors="ignore")
    entries = parse_overhead_entries(text)

    if not entries:
        print("ERROR: No overhead log entries found!")
        return

    print(f"Found {len(entries)} requests with overhead data")

    if args.skip_first > 0:
        skipped = min(args.skip_first, len(entries))
        entries = entries[skipped:]
        print(f"Skipped first {skipped} entries (warmup), {len(entries)} remaining")

    # Compute statistics
    print("Computing statistics...")
    summary = summarize(entries)

    # Print analysis
    print_analysis(summary)

    # Generate plot
    if args.output:
        output_path = args.output
    else:
        output_path = args.log_path.parent / 'overhead_analysis.pdf'

    print(f"\nGenerating plot...")
    create_plot(summary, entries, output_path)

    print(f"\n{'='*80}")
    print(f"\u2713 Analysis complete! PDF saved to: {output_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
