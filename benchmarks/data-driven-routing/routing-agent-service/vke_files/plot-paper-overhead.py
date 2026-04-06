#!/usr/bin/env python3
"""
Paper-quality overhead comparison across RPS levels.

Usage:
    python plot-paper-overhead.py <dir>

Recursively finds all-routing-agent-service.log.txt files under <dir>,
extracts RPS from directory names (e.g. rps50-benchmark), parses overhead
logs, saves a CSV summary, and generates overhead.pdf with two side-by-side
figures:
  Left:  Mean and P99 overhead trend line vs RPS
  Right: Single stacked bar showing mean overhead breakdown (averaged across all RPS)
"""
import argparse
import csv
import pathlib
import re
import statistics
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Font size config — change BASE_FONT_SIZE to scale everything uniformly
# ---------------------------------------------------------------------------
BASE_FONT_SIZE = 15

plt.rcParams.update({
    'font.size': BASE_FONT_SIZE,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': BASE_FONT_SIZE + 1,
    'axes.titlesize': BASE_FONT_SIZE + 2,
    'xtick.labelsize': BASE_FONT_SIZE - 1,
    'ytick.labelsize': BASE_FONT_SIZE - 1,
    'legend.fontsize': BASE_FONT_SIZE - 1,
    'figure.titlesize': BASE_FONT_SIZE + 3,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
})

OVERHEAD_RE = re.compile(r'overhead_log: oh, (.*)')
RPS_RE = re.compile(r'rps(\d+)-benchmark')

EXCLUDED_METRICS = {
    'infer_from_tensor_get_agent_reload',
    'infer_from_tensor_get_agent_async_reload_started',
}

SKIP_FIRST = 5  # warmup requests to skip

# Fine-grained breakdown components for the stacked bar.
# These are the additive pieces of handle_infer_end_to_end, but with
# "Calling Infer From Tensor" expanded into its sub-components and
# encoding shown separately.
BREAKDOWN_COMPONENTS = [
    # Preprocessing
    'handle_infer_preprocess_overhead',
    # Normalize
    'handle_infer_normalize',
    # Encoding
    'handle_infer_encode',
    # Infer from tensor sub-components (fine-grained)
    'infer_from_tensor_get_agent',
    'infer_from_tensor_tensor_transfer',
    'infer_from_tensor_inference',
    'infer_from_tensor_batch_format',
    'infer_from_tensor_result_formatting',
    # Other handle_infer stages
    # NOTE: replace_podid_overhead is a subset of request_prepare — don't list both.
    'handle_infer_request_prepare',
    'handle_infer_distribution_monitor',
    'handle_infer_ood_check',
    'handle_infer_contextual_bandit_write_lock',
    'handle_infer_contextual_bandit_create',
    'handle_infer_remaining_work',
]

BREAKDOWN_DISPLAY_NAMES = {
    'handle_infer_preprocess_overhead': 'Preprocess',
    'handle_infer_normalize': 'Normalize',
    'handle_infer_encode': 'Encode',
    'infer_from_tensor_get_agent': 'Get Agent',
    'infer_from_tensor_tensor_transfer': 'Tensor Transfer',
    'infer_from_tensor_inference': 'NN Inference',
    'infer_from_tensor_batch_format': 'Batch Format',
    'infer_from_tensor_result_formatting': 'Result Format',
    'handle_infer_request_prepare': 'Request Prepare',
    'handle_infer_distribution_monitor': 'Distribution Monitor',
    'handle_infer_ood_check': 'OOD Check',
    'handle_infer_contextual_bandit_write_lock': 'CB Write Lock',
    'handle_infer_contextual_bandit_create': 'CB Create',
    'handle_infer_remaining_work': 'Remaining Work',
    'other': 'Other',
}


def parse_overhead_entries(text: str) -> List[Dict[str, float]]:
    entries: List[Dict[str, float]] = []
    for line in text.splitlines():
        match = OVERHEAD_RE.search(line)
        if not match:
            continue
        parts = [p.strip() for p in match.group(1).split(',')]
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
                    if val >= 0:
                        values[key] = val
                except ValueError:
                    continue
        if values:
            entries.append(values)
    return entries


def percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return float('nan')
    idx = int(p * (len(sorted_vals) - 1))
    return sorted_vals[idx]


def extract_rps(path: pathlib.Path) -> int | None:
    for part in path.parts:
        m = RPS_RE.search(part)
        if m:
            return int(m.group(1))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description='Plot overhead comparison across RPS levels.')
    parser.add_argument('dir', type=pathlib.Path, help='Root directory to search for log files')
    args = parser.parse_args()

    root = args.dir.resolve()
    log_files = sorted(root.rglob('all-routing-agent-service.log.txt'))

    if not log_files:
        print(f"No all-routing-agent-service.log.txt files found under {root}")
        return

    print(f"Found {len(log_files)} log file(s)")

    # Per-RPS data: {rps: {metric: [values across requests]}}
    rps_data: Dict[int, Dict[str, List[float]]] = {}

    for log_path in log_files:
        rps = extract_rps(log_path)
        if rps is None:
            print(f"  Skipping {log_path} (cannot extract RPS)")
            continue

        text = log_path.read_text(errors='ignore')
        entries = parse_overhead_entries(text)
        entries = entries[SKIP_FIRST:]

        if not entries:
            print(f"  Skipping {log_path} (no entries after warmup skip)")
            continue

        print(f"  RPS {rps}: {len(entries)} requests from {log_path.parent.name}")

        if rps not in rps_data:
            rps_data[rps] = {}
        for entry in entries:
            for key, val in entry.items():
                rps_data[rps].setdefault(key, []).append(val)

    if not rps_data:
        print("No valid data found.")
        return

    sorted_rps = sorted(rps_data.keys())
    total_key = 'handle_infer_end_to_end'

    # --- Build CSV and collect plot data ---
    csv_rows = []
    plot_mean = []
    plot_p99 = []

    # Accumulate all entries across RPS for the single breakdown bar
    all_metrics: Dict[str, List[float]] = {}

    for rps in sorted_rps:
        metrics = rps_data[rps]
        if total_key not in metrics:
            continue

        e2e = sorted(metrics[total_key])
        mean_val = statistics.mean(e2e)
        p50_val = percentile(e2e, 0.50)
        p90_val = percentile(e2e, 0.90)
        p99_val = percentile(e2e, 0.99)
        max_val = max(e2e)
        n = len(e2e)

        plot_mean.append(mean_val)
        plot_p99.append(p99_val)

        row = {'rps': rps, 'n': n, 'mean': round(mean_val, 3),
               'p50': round(p50_val, 3), 'p90': round(p90_val, 3),
               'p99': round(p99_val, 3), 'max': round(max_val, 3)}

        for comp in BREAKDOWN_COMPONENTS:
            if comp in metrics:
                v = statistics.mean(metrics[comp])
            else:
                v = 0.0
            display = BREAKDOWN_DISPLAY_NAMES.get(comp, comp)
            row[display] = round(v, 3)

        csv_rows.append(row)

        # Merge into all_metrics for the single breakdown bar
        for key, vals in metrics.items():
            all_metrics.setdefault(key, []).extend(vals)

    # Save CSV
    csv_path = root / 'overhead_summary.csv'
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nSaved CSV to {csv_path}")

    # --- Compute single breakdown bar (averaged across all requests) ---
    total_mean = statistics.mean(all_metrics[total_key]) if total_key in all_metrics else 0
    comp_means = {}
    accounted = 0.0
    for comp in BREAKDOWN_COMPONENTS:
        if comp in all_metrics:
            v = statistics.mean(all_metrics[comp])
        else:
            v = 0.0
        comp_means[comp] = v
        accounted += v
    comp_means['other'] = max(0, total_mean - accounted)

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5),
                                    gridspec_kw={'height_ratios': [2, 1]})

    # Left: trend line — Mean and P99 vs RPS
    ax1.plot(sorted_rps, plot_mean, 'o-', color='#4C72B0', label='Mean', markersize=7)
    ax1.plot(sorted_rps, plot_p99, 's--', color='#C44E52', label='P99', markersize=7)
    for i, rps in enumerate(sorted_rps):
        ax1.text(rps, plot_mean[i] + max(plot_p99) * 0.03, f'{plot_mean[i]:.1f}',
                 ha='center', va='bottom', fontsize=BASE_FONT_SIZE - 3)
        ax1.text(rps, plot_p99[i] + max(plot_p99) * 0.03, f'{plot_p99[i]:.1f}',
                 ha='center', va='bottom', fontsize=BASE_FONT_SIZE - 3)
    ax1.set_xlabel('RPS')
    ax1.set_ylabel('Overhead (ms)')
    ax1.set_xticks(sorted_rps)
    ax1.set_ylim(bottom=0)
    ax1.legend(frameon=False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(axis='y', alpha=0.25, linestyle='--')

    # Right: single horizontal stacked bar — breakdown
    all_comps = BREAKDOWN_COMPONENTS + ['other']
    bar_total = sum(comp_means.values())
    keep_comps = [c for c in all_comps if comp_means.get(c, 0) >= 0.005 * bar_total]

    cmap = plt.colormaps.get_cmap('tab10').resampled(max(len(keep_comps), 1))
    colors = [cmap(i) for i in range(len(keep_comps))]

    left = 0.0
    bars_for_legend = []
    for ci, comp in enumerate(keep_comps):
        val = comp_means[comp]
        b = ax2.barh(0, val, left=left, height=0.3, color=colors[ci],
                      edgecolor='white', linewidth=0.4)
        bars_for_legend.append(b)
        if bar_total > 0 and val / bar_total > 0.04:
            ax2.text(left + val / 2, 0, f'{val:.1f}',
                     ha='center', va='center', fontsize=BASE_FONT_SIZE - 4,
                     color='white', fontweight='bold')
        left += val

    ax2.text(left + bar_total * 0.01, 0, f'{left:.1f} ms',
             ha='left', va='center', fontsize=BASE_FONT_SIZE - 2)

    ax2.set_xlabel('Latency (ms)', labelpad=18)
    ax2.set_yticks([0])
    ax2.set_yticklabels(['Mean'])
    ax2.set_xlim(0, bar_total * 1.18)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='x', alpha=0.25, linestyle='--')

    labels = [BREAKDOWN_DISPLAY_NAMES.get(c, c) for c in keep_comps]
    ax2.legend([b[0] for b in bars_for_legend], labels,
               loc='upper center', bbox_to_anchor=(0.5, -0.18),
               ncol=min(4, len(keep_comps)), fontsize=BASE_FONT_SIZE - 3,
               frameon=False, columnspacing=1.0, handlelength=1.2)

    plt.tight_layout()
    pdf_path = root / 'overhead.pdf'
    fig.savefig(str(pdf_path), bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {pdf_path}")


if __name__ == '__main__':
    main()
