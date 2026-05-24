#!/usr/bin/env python3
"""
Overhead Analysis and Visualization Tool
Analyzes routing agent overhead logs and generates publication-quality plots.

Usage:
    python plot_overhead.py <experiment_dir>

Expects the directory to contain:
    - all-routing-agent-service.log.txt   (required)
    - filtered-aibrix-gateway-plugins.log.csv  (optional, adds VLLMScrapingOverhead)
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
# NOTE: gateway_vllm_scraping_overhead comes from the gateway log, not the agent log.
MAIN_ADDITIVE_COMPONENTS = [
    'gateway_vllm_scraping_overhead',
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
    'gateway_vllm_scraping_overhead': 'VLLM Scraping (Gateway)',
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


def parse_gateway_overhead(text: str) -> List[Dict[str, float]]:
    """Parse overhead fields from the gateway log CSV.

    Each line contains @-delimited key-value pairs like:
        ...EndToEndOverhead@15@VLLMScrapingOverhead@7@...
    Values of -1 indicate not-measured and are filtered out.
    """
    field_map = {
        'EndToEndOverhead': 'gateway_end_to_end_overhead',
        'VLLMScrapingOverhead': 'gateway_vllm_scraping_overhead',
        'FeaturePrepOverhead': 'gateway_feature_prep_overhead',
        'HTTPRoundTripOverhead': 'gateway_http_round_trip_overhead',
    }
    entries: List[Dict[str, float]] = []
    for line in text.splitlines():
        if '**@latency_metrics@' not in line:
            continue
        values: Dict[str, float] = {}
        for gw_key, internal_key in field_map.items():
            marker = f'{gw_key}@'
            idx = line.find(marker)
            if idx == -1:
                continue
            start = idx + len(marker)
            end = line.find('@', start)
            val_str = line[start:end] if end != -1 else line[start:].strip()
            try:
                val = float(val_str)
                if val >= 0:  # Filter -1 placeholders
                    values[internal_key] = val
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
        'Gateway': [],
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
        elif metric.startswith('gateway_'):
            categories['Gateway'].append(metric)
        else:
            categories['Other'].append(metric)

    return categories


def calculate_contributions(summary: Dict[str, Dict[str, float]],
                          stat: str = 'mean') -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Calculate contribution percentages for top-level additive components.
    Total is the sum of all present components (no "Other" bucket).
    Returns (component_values, component_percentages)
    """
    component_values = {}
    for comp in MAIN_ADDITIVE_COMPONENTS:
        if comp in summary:
            component_values[comp] = summary[comp][stat]

    if not component_values:
        return {}, {}

    total_time = sum(component_values.values())
    component_percentages = {
        comp: (val / total_time * 100) if total_time > 0 else 0
        for comp, val in component_values.items()
    }

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
        total = sum(comp_values.values())
        print(f"\nTotal Overhead (sum of components): {total:.2f} ms\n")

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


def _draw_stacked_bar(ax, stat_key: str, label: str,
                      summary: Dict[str, Dict[str, float]],
                      plot_components: List[str],
                      all_labels: List[str],
                      colors: List) -> List:
    """Draw a single horizontal stacked bar for one statistic and return bar handles."""
    vals = [summary[c][stat_key] if c in summary else 0.0 for c in plot_components]
    total = sum(vals)

    bars_for_legend = []
    left = 0.0
    for ci, v in enumerate(vals):
        b = ax.barh(0, v, left=left, height=0.55,
                     color=colors[ci], edgecolor='white', linewidth=0.4)
        bars_for_legend.append(b)
        if total > 0 and v / total > 0.04:
            ax.text(left + v / 2, 0, f'{v:.1f}',
                    ha='center', va='center', fontsize=7, color='white',
                    fontweight='bold')
        left += v

    ax.text(total + total * 0.02, 0,
            f'{total:.1f} ms', ha='left', va='center', fontsize=9)

    ax.set_yticks([0])
    ax.set_yticklabels([label])
    ax.set_xlabel('Latency (ms)')
    ax.set_xlim(0, total * 1.18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.25, linestyle='--')

    return bars_for_legend


def create_plot(summary: Dict[str, Dict[str, float]],
                agent_entries: List[Dict[str, float]],
                gateway_entries: List[Dict[str, float]],
                output_path: pathlib.Path) -> None:
    """Generate a PDF with separate Mean/P99 stacked bars, histogram, and CDF."""
    plot_components = [c for c in MAIN_ADDITIVE_COMPONENTS if c in summary and summary[c]['mean'] > 0.01]
    all_labels = [COMPONENT_DISPLAY_NAMES.get(c, c) for c in plot_components]

    n_comps = len(plot_components)
    cmap = plt.colormaps.get_cmap('tab10').resampled(max(n_comps, 1))
    colors = [cmap(i) for i in range(n_comps)]

    fig, (ax_mean, ax_p99, ax_hist, ax_cdf) = plt.subplots(
        4, 1, figsize=(7, 8),
        gridspec_kw={'height_ratios': [0.45, 0.45, 0.7, 0.7]})

    # --- Mean stacked bar (own x-scale) ---
    bars = _draw_stacked_bar(ax_mean, 'mean', 'Mean', summary,
                             plot_components, all_labels, colors)

    # --- P99 stacked bar (own x-scale) ---
    _draw_stacked_bar(ax_p99, 'p99', 'P99', summary,
                      plot_components, all_labels, colors)

    # Shared legend between the two bars
    ax_mean.legend([b[0] for b in bars], all_labels,
                   loc='upper center', bbox_to_anchor=(0.5, -0.35),
                   ncol=min(4, n_comps), fontsize=8, frameon=False,
                   columnspacing=1.0, handlelength=1.2)

    # --- Histogram and CDF use agent handle_infer_end_to_end (per-request) ---
    agent_total_key = 'handle_infer_end_to_end'
    per_req = [e[agent_total_key] for e in agent_entries if agent_total_key in e]

    if per_req:
        mean_val = statistics.mean(per_req)

        ax_hist.hist(per_req, bins='auto', color='#4C72B0', edgecolor='white',
                     linewidth=0.5, alpha=0.85)
        ax_hist.axvline(mean_val, color='#C44E52', linestyle='--', linewidth=1.2,
                        label=f'Mean: {mean_val:.2f} ms  (n={len(per_req)})')
        ax_hist.set_xlabel('Routing Agent End-to-End Overhead (ms)')
        ax_hist.set_ylabel('Count')
        ax_hist.spines['top'].set_visible(False)
        ax_hist.spines['right'].set_visible(False)
        ax_hist.grid(axis='y', alpha=0.25, linestyle='--')
        ax_hist.legend(fontsize=9, frameon=False)

        sorted_vals = np.sort(per_req)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax_cdf.plot(sorted_vals, cdf, color='#4C72B0', linewidth=1.5)
        ax_cdf.axvline(mean_val, color='#C44E52', linestyle='--', linewidth=1.2,
                       label=f'Mean: {mean_val:.2f} ms')
        ax_cdf.set_xlabel('Routing Agent End-to-End Overhead (ms)')
        ax_cdf.set_ylabel('CDF')
        ax_cdf.set_ylim(0, 1.05)
        ax_cdf.spines['top'].set_visible(False)
        ax_cdf.spines['right'].set_visible(False)
        ax_cdf.grid(alpha=0.25, linestyle='--')
        ax_cdf.legend(fontsize=9, frameon=False)

    plt.tight_layout()
    fig.savefig(str(output_path), bbox_inches='tight')
    plt.close()
    print(f"  Saved figure to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Analyze overhead logs and generate publication-quality visualizations.'
    )
    parser.add_argument('experiment_dir', type=pathlib.Path,
                       help='Experiment directory containing log files')
    parser.add_argument('--output', '-o', type=pathlib.Path,
                       default=None,
                       help='Output PDF path (default: <experiment_dir>/overhead_analysis.pdf)')
    parser.add_argument('--skip-first', type=int, default=5,
                       help='Number of initial overhead entries to skip (default: 5 for warmup)')

    args = parser.parse_args()
    exp_dir = args.experiment_dir

    # --- Routing agent service log (required) ---
    agent_log = exp_dir / 'all-routing-agent-service.log.txt'
    if not agent_log.exists():
        print(f"ERROR: routing agent log not found: {agent_log}")
        return

    print(f"Reading {agent_log.name}...")
    agent_entries = parse_overhead_entries(agent_log.read_text(errors="ignore"))
    if not agent_entries:
        print("ERROR: No overhead log entries found!")
        return
    print(f"Found {len(agent_entries)} requests with overhead data")

    if args.skip_first > 0:
        skipped = min(args.skip_first, len(agent_entries))
        agent_entries = agent_entries[skipped:]
        print(f"Skipped first {skipped} entries (warmup), {len(agent_entries)} remaining")

    # --- Gateway log (optional, adds VLLMScrapingOverhead) ---
    gateway_log = exp_dir / 'filtered-aibrix-gateway-plugins.log.csv'
    gateway_entries: List[Dict[str, float]] = []
    if gateway_log.exists():
        print(f"Reading {gateway_log.name}...")
        gateway_entries = parse_gateway_overhead(gateway_log.read_text(errors="ignore"))
        print(f"Found {len(gateway_entries)} gateway entries with overhead data")
        if args.skip_first > 0:
            gw_skipped = min(args.skip_first, len(gateway_entries))
            gateway_entries = gateway_entries[gw_skipped:]
            print(f"Skipped first {gw_skipped} gateway entries (warmup), {len(gateway_entries)} remaining")
    else:
        print(f"No gateway log found at {gateway_log}, skipping VLLMScrapingOverhead")

    # Compute statistics
    print("Computing statistics...")
    summary = summarize(agent_entries)

    if gateway_entries:
        gateway_summary = summarize(gateway_entries)
        summary.update(gateway_summary)

    # Print analysis
    print_analysis(summary)

    # Generate plot
    output_path = args.output or (exp_dir / 'overhead_analysis.pdf')
    print(f"\nGenerating plot...")
    create_plot(summary, agent_entries, gateway_entries, output_path)

    print(f"\n{'='*80}")
    print(f"\u2713 Analysis complete! PDF saved to: {output_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
