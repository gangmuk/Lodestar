#!/usr/bin/env python3
"""
Plot trend-line graphs from llama3_maxtoken1_all_workloads_benchmark.csv.

For each workload group (same workload_category + workload_name, varying RPS),
creates line plots with:
  - X-axis: RPS (low to high)
  - Different lines: different routing policies
  - 3 subplots per page: Avg, P99, P999 E2E latency

Multiple experiments (same policy + load) are averaged.

Usage:
    python plot_trendline_from_benchmark_csv.py [csv_file] [--output-dir <dir>]

Example:
    python plot_trendline_from_benchmark_csv.py llama3_maxtoken1_all_workloads_benchmark.csv
"""

import os
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ── Font sizes ──────────────────────────────────────────────────────────────
TITLE_FONTSIZE = 22
SUBTITLE_FONTSIZE = 18
AXIS_LABEL_FONTSIZE = 16
TICK_FONTSIZE = 14
LEGEND_FONTSIZE = 14
SUBFIG_LEGEND_FONTSIZE = 11

# ── Line style rotation ────────────────────────────────────────────────────
LINE_STYLES = ['-', '--', '-.', ':']
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']

# ── Policy color families ──────────────────────────────────────────────────
POLICY_COLOR_FAMILIES = {
    'random':          ['#2ca02c', '#32cd32', '#00ff00', '#00ff7f', '#98df8a'],
    'least_request':   ['#008b8b', '#20b2aa', '#48d1cc', '#40e0d0', '#00ced1'],
    'prefix_cache':    ['#1f77b4', '#4682b4', '#6495ed', '#aec7e8', '#87ceeb'],
    'prefix_hit_threshold_or_least_request': [
        '#556b2f', '#6b8e23', '#808000', '#9acd32', '#bdb76b',
    ],
    'contextual_bandit': [
        '#ff0000', '#dc143c', '#ff6347', '#ff4500', '#ff7f50',
    ],
}
DEFAULT_COLORS = ['#7f7f7f', '#696969', '#a9a9a9', '#c7c7c7', '#d3d3d3']


# ── Policy helpers ─────────────────────────────────────────────────────────

def categorize_policy(policy: str) -> str:
    p = policy.lower()
    if 'prefix_hit_threshold_or_least_request' in p:
        return 'prefix_hit_threshold_or_least_request'
    if 'contextual_bandit' in p:
        return 'contextual_bandit'
    if 'prefix_cache' in p:
        return 'prefix_cache'
    if 'least_request' in p:
        return 'least_request'
    if 'random' in p:
        return 'random'
    return ''


def get_policy_sort_key(policy: str):
    p = policy.lower()
    if 'random' in p:
        return (0, policy)
    if 'prefix_hit_threshold_or_least_request' in p:
        return (1, policy)
    if 'least_request' in p:
        return (2, policy)
    if 'prefix_cache' in p:
        return (3, policy)
    if 'contextual_bandit' in p:
        return (4, policy)
    return (5, policy)


def order_policies(policies):
    return sorted(policies, key=get_policy_sort_key)


def generate_policy_colors(policies):
    colors = {}
    category_counts = {}
    for policy in sorted(policies):
        category = categorize_policy(policy)
        if category and category in POLICY_COLOR_FAMILIES:
            idx = category_counts.get(category, 0)
            category_counts[category] = idx + 1
            family = POLICY_COLOR_FAMILIES[category]
            colors[policy] = family[idx % len(family)]
        else:
            idx = category_counts.get('unknown', 0)
            category_counts['unknown'] = idx + 1
            colors[policy] = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
    return colors


# ── RPS extraction ─────────────────────────────────────────────────────────

def extract_rps(load_str: str) -> int:
    """Extract integer RPS from load column like 'rps10-benchmark'."""
    m = re.search(r'rps(\d+)', load_str, re.IGNORECASE)
    return int(m.group(1)) if m else 0


# ── Plotting ───────────────────────────────────────────────────────────────

def _plot_trendline_subplot(
    ax, df_group, rps_list, policies, policy_colors,
    value_col, ylabel, title,
):
    """Plot trend lines on one Axes for one stat column."""
    policy_style_idx = {}

    for pi, policy in enumerate(policies):
        color = policy_colors.get(policy, '#7f7f7f')
        if color not in policy_style_idx:
            policy_style_idx[color] = 0
        style_idx = policy_style_idx[color]
        policy_style_idx[color] += 1

        linestyle = LINE_STYLES[style_idx % len(LINE_STYLES)]
        marker = MARKERS[pi % len(MARKERS)]

        xs, ys = [], []
        for rps in rps_list:
            rows = df_group[
                (df_group['rps'] == rps) & (df_group['routing_policy'] == policy)
            ]
            if len(rows) == 0:
                continue
            val = rows[value_col].mean()
            if pd.notna(val) and val > 0:
                xs.append(rps)
                ys.append(val)

        if xs:
            val_strs = [f'{y:.0f}({x})' for x, y in zip(xs, ys)]
            subfig_label = ', '.join(val_strs)
            line, = ax.plot(
                xs, ys, color=color, linestyle=linestyle, marker=marker,
                markersize=7, linewidth=2.2, label=subfig_label,
            )
            line._policy_name = policy

    ax.set_ylim(0, None)
    ax.set_xlabel('RPS', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=SUBTITLE_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.set_xticks(rps_list)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=SUBFIG_LEGEND_FONTSIZE, loc='best', framealpha=0.8)


def plot_trendlines(df, output_dir, output_filename):
    """Generate trend-line plots and save to a single PDF."""
    df = df.copy()
    df['rps'] = df['load'].apply(extract_rps)
    df['group_key'] = df['workload_category'] + '/' + df['workload_name']

    stats = [
        ('avg_ms', 'E2E Latency (ms)', 'Avg E2E Latency'),
        ('p99_ms', 'E2E Latency (ms)', 'P99 E2E Latency'),
        ('p999_ms', 'E2E Latency (ms)', 'P999 E2E Latency'),
    ]

    groups = {}
    for gk, gdf in df.groupby('group_key'):
        rps_sorted = sorted(gdf['rps'].unique())
        if len(rps_sorted) >= 2:
            groups[gk] = (rps_sorted, gdf)

    all_policies = order_policies(df['routing_policy'].unique())
    policy_colors = generate_policy_colors(all_policies)

    pdf_path = os.path.join(output_dir, output_filename)

    with PdfPages(pdf_path) as pdf:
        for gk in sorted(groups.keys()):
            rps_list, df_group = groups[gk]

            # Only keep policies that appear in this group
            group_policies = order_policies(df_group['routing_policy'].unique())

            fig, axes = plt.subplots(1, 3, figsize=(24, 8), squeeze=False)

            fig.suptitle(f'Trend Lines — {gk}', fontsize=TITLE_FONTSIZE, y=0.98)

            for ci, (col, ylabel, title) in enumerate(stats):
                _plot_trendline_subplot(
                    axes[0][ci], df_group, rps_list,
                    group_policies, policy_colors, col, ylabel, title,
                )

            # Bottom legend with policy names (below subplots, no overlap with title)
            seen = {}
            for ax in axes[0]:
                for line in ax.get_lines():
                    name = getattr(line, '_policy_name', None)
                    if name and name not in seen:
                        seen[name] = line

            if seen:
                fig.legend(
                    list(seen.values()), list(seen.keys()),
                    loc='lower center',
                    ncol=min(len(seen), 4),
                    fontsize=LEGEND_FONTSIZE,
                    bbox_to_anchor=(0.5, -0.02),
                    framealpha=0.9,
                )

            fig.tight_layout(rect=[0, 0.08, 1, 0.95])
            pdf.savefig(fig, bbox_inches='tight', dpi=300)
            plt.close(fig)

    print(f"Saved trend-line PDF to {pdf_path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Plot trend-line graphs from benchmark CSV'
    )
    parser.add_argument(
        'csv_file',
        nargs='?',
        default='llama3_maxtoken1_all_workloads_benchmark.csv',
        help='Input CSV file (default: llama3_maxtoken1_all_workloads_benchmark.csv)',
    )
    parser.add_argument(
        '--output-dir', '-o', default=None,
        help='Output directory for PDF (default: same dir as CSV)',
    )
    parser.add_argument(
        '--output-filename', '-f',
        default='llama3_maxtoken1_trendline.pdf',
        help='Output PDF filename (default: llama3_maxtoken1_trendline.pdf)',
    )
    args = parser.parse_args()

    csv_path = args.csv_file
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        return

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(csv_path))
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"  Workload categories: {sorted(df['workload_category'].unique())}")
    print(f"  Workload names:      {sorted(df['workload_name'].unique())}")
    print(f"  Routing policies:    {sorted(df['routing_policy'].unique())}")
    print(f"  Loads:               {sorted(df['load'].unique())}")

    # Summary of groups
    df_tmp = df.copy()
    df_tmp['rps'] = df_tmp['load'].apply(extract_rps)
    df_tmp['group_key'] = df_tmp['workload_category'] + '/' + df_tmp['workload_name']
    print(f"\nWorkload groups:")
    for gk in sorted(df_tmp['group_key'].unique()):
        rps_vals = sorted(df_tmp[df_tmp['group_key'] == gk]['rps'].unique())
        print(f"  {gk} — RPS: {rps_vals}")

    plot_trendlines(df, output_dir, args.output_filename)
    print("Done!")


if __name__ == "__main__":
    main()
