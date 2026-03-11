#!/usr/bin/env python3
"""
Plot trend line graphs from routing_strategy_metrics_from_client_log.csv files.

For each workload group (same category + sharing ratio / subcategory, varying RPS),
creates line plots with:
  - X-axis: RPS (low to high, left to right)
  - Different lines: different routing policies
  - Separate subfigures for avg, p99, p999 of TTFT, TPOT, and E2E latency

Usage:
    python trendline_plot_from_client_log.py <base_dir> [--output-dir <output_dir>]
    python trendline_plot_from_client_log.py <base_dir> --target-dirs-file <file>

Example:
    python trendline_plot_from_client_log.py /path/to/workload-and-experiment_results/NVIDIA-A30
"""

import os
import sys
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# Reuse helpers from the bar chart script
from merge_and_plot_all_workloads_from_client_log import (
    find_metrics_files,
    merge_metrics_files,
    extract_routing_policy,
    extract_rps_from_workload,
    order_policies,
    generate_policy_colors,
    get_policy_sort_key,
    POLICY_COLOR_FAMILIES,
    DEFAULT_COLORS,
)

# ── Font sizes ──────────────────────────────────────────────────────────────
TITLE_FONTSIZE = 22
SUBTITLE_FONTSIZE = 18
AXIS_LABEL_FONTSIZE = 16
TICK_FONTSIZE = 14
LEGEND_FONTSIZE = 14
SUBFIG_LEGEND_FONTSIZE = 11

# ── Line style rotation for policies sharing a color family ─────────────────
def _shade_color(color, factor):
    """Lighten (factor > 1) or darken (factor < 1) a color.

    Returns an RGB tuple.  *color* can be anything matplotlib accepts.
    """
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(color)
    # Blend towards white (lighten) or black (darken)
    if factor >= 1.0:
        # lighten: interpolate towards white
        t = factor - 1.0  # 0 = no change, 1 = full white
        t = min(t, 1.0)
        return (r + (1 - r) * t, g + (1 - g) * t, b + (1 - b) * t)
    else:
        # darken: scale towards black
        return (r * factor, g * factor, b * factor)


LINE_STYLES = ['-', '--', '-.', ':']
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']


# ── Workload grouping helpers ───────────────────────────────────────────────

def _remove_rps_segment(workload: str) -> str:
    """Remove the rps<N>-benchmark (or rps<N>) path segment from a workload string.

    Examples:
        ".../SharingRatio71%/rps10-benchmark/without_bitsandbytes"
        -> ".../SharingRatio71%/without_bitsandbytes"

        ".../mooncake/conversation-2/rps50-benchmark/without_bitsandbytes"
        -> ".../mooncake/conversation-2/without_bitsandbytes"
    """
    # Remove path segments that match rps<digits>... (whole segment)
    parts = workload.split('/')
    filtered = [p for p in parts if not re.match(r'^rps\d+', p, re.IGNORECASE)]
    return '/'.join(filtered)


def group_workloads_by_category(workloads):
    """Group workload strings that differ only by the RPS segment.

    Returns:
        dict: {group_label: [(rps_int, workload_string), ...]}
              sorted by rps within each group.
    """
    groups = {}
    for w in workloads:
        key = _remove_rps_segment(w)
        rps = extract_rps_from_workload(w)
        groups.setdefault(key, []).append((rps, w))

    # Sort each group by rps ascending
    for key in groups:
        groups[key].sort(key=lambda x: x[0])

    return groups


def _short_group_label(group_key: str) -> str:
    """Create a shorter human-readable label from the group key."""
    parts = group_key.split('/')
    # Keep only informative parts (drop GPU, model, output-length config)
    informative = []
    for p in parts:
        p_lower = p.lower()
        # Skip generic path segments
        if any(skip in p_lower for skip in [
            'nvidia', 'qwen', 'llama', 'use_given', 'maxtokens',
            'without_bitsandbytes', 'with_bitsandbytes', 'benchmark',
        ]):
            continue
        if p.strip():
            informative.append(p)
    return '/'.join(informative) if informative else group_key


# ── Plotting ────────────────────────────────────────────────────────────────

def _plot_trendlines_for_group(
    ax, df_group, rps_workload_pairs, policies, policy_colors,
    stat_prefix, ylabel, title, ylim_upper=None
):
    """Plot trend lines on a single axes for one workload group and one stat.

    Args:
        ax: matplotlib Axes
        df_group: DataFrame filtered to the rows belonging to this workload group
        rps_workload_pairs: [(rps_int, workload_str), ...] sorted by rps
        policies: ordered list of routing policies
        policy_colors: {policy: color}
        stat_prefix: e.g. 'avg_ttft', 'p99_tpot', 'p999_end_to_end'
        ylabel: label for y-axis
        title: subplot title
        ylim_upper: upper limit for y-axis
    """
    if stat_prefix not in df_group.columns:
        ax.set_visible(False)
        return

    rps_values = [r for r, _ in rps_workload_pairs]
    workload_at_rps = {r: w for r, w in rps_workload_pairs}

    policy_style_idx = {}  # track style index per color to distinguish same-color policies

    for pi, policy in enumerate(policies):
        color = policy_colors.get(policy, '#7f7f7f')
        if color not in policy_style_idx:
            policy_style_idx[color] = 0
        style_idx = policy_style_idx[color]
        policy_style_idx[color] += 1

        linestyle = LINE_STYLES[style_idx % len(LINE_STYLES)]
        marker = MARKERS[pi % len(MARKERS)]

        # Collect per-RPS experiment values (one entry per experiment),
        # sorted by datetime in strategy_full_name so run1 is the earliest.
        rps_to_vals = {}
        for rps in rps_values:
            w = workload_at_rps[rps]
            rows = df_group[
                (df_group['workload'] == w) & (df_group['routing_policy'] == policy)
            ].copy()
            # Extract datetime suffix (YYYYMMDD_HHMMSS) for sorting
            rows['_sort_dt'] = rows['strategy_full_name'].str.extract(
                r'(\d{8}_\d{6})$', expand=False
            ).fillna('')
            rows = rows.sort_values('_sort_dt')
            vals = rows[stat_prefix].dropna().tolist()
            vals = [v for v in vals if v > 0]
            rps_to_vals[rps] = vals

        max_experiments = max((len(v) for v in rps_to_vals.values()), default=0)

        for exp_idx in range(max_experiments):
            xs = []
            ys = []
            for rps in rps_values:
                vals = rps_to_vals[rps]
                if exp_idx < len(vals):
                    xs.append(rps)
                    ys.append(vals[exp_idx])

            if xs:
                val_strs = [f'{y:.0f}({x})' for x, y in zip(xs, ys)]
                if max_experiments > 1:
                    subfig_label = f'run{exp_idx+1}: ' + ', '.join(val_strs)
                    # Gradation: run1 = base color, later runs progressively lighter
                    shade_factor = 1.0 + exp_idx * (0.6 / max(max_experiments - 1, 1))
                    run_color = _shade_color(color, shade_factor)
                else:
                    subfig_label = ', '.join(val_strs)
                    run_color = color

                line, = ax.plot(xs, ys, color=run_color, linestyle=linestyle, marker=marker,
                               markersize=14, linewidth=2.2, label=subfig_label,
                               markerfacecolor='none', markeredgewidth=2)
                line._policy_name = policy  # used by top-level legend

    # Set y-axis upper limit to max data value + 500
    all_vals = df_group[stat_prefix].dropna()
    all_vals = all_vals[all_vals > 0]
    if ylim_upper is not None:
        ax.set_ylim(0, ylim_upper)
    # elif len(all_vals) > 0:
    #     ax.set_ylim(0, all_vals.max()*1.2)
    # else:
    #     ax.set_ylim(0, None)
    ax.set_ylim(0, None)

    ax.set_xlabel('RPS', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=SUBTITLE_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.set_xticks(rps_values)
    ax.grid(alpha=0.3)
    # Per-subfigure legend with latency values
    ax.legend(fontsize=SUBFIG_LEGEND_FONTSIZE, loc='best', framealpha=0.8)


def plot_trendlines(df, output_dir):
    """Generate all trend-line plots and save to a single PDF.

    Layout: one page per workload group.
    Each page has a grid of subplots:
        rows = metrics (TTFT, TPOT, E2E)
        cols = statistics (avg, p99, p999)
    """
    workloads = df['workload'].unique().tolist()
    groups = group_workloads_by_category(workloads)

    # Sort group keys for deterministic output
    sorted_group_keys = sorted(groups.keys())

    policies = order_policies(df['routing_policy'].unique())
    policy_colors = generate_policy_colors(policies)

    # Define metrics to plot: (column_stem, display_name, ylabel)
    metrics = [
        ('ttft', 'TTFT', 'TTFT (ms)'),
        ('tpot', 'TPOT', 'TPOT (ms)'),
        ('end_to_end', 'End-to-End Latency', 'E2E Latency (ms)'),
    ]

    stats = [
        ('avg', 'Avg'),
        ('p99', 'P99'),
        ('p999', 'P999'),
    ]

    pdf_path = os.path.join(output_dir, 'trendline_from_client_log.pdf')

    with PdfPages(pdf_path) as pdf:
        for gk in sorted_group_keys:
            rps_workload_pairs = groups[gk]
            if len(rps_workload_pairs) == 1:
                print(f"Group '{gk}' has only 1 RPS point — plotting as dot(s)")

            group_workloads = [w for _, w in rps_workload_pairs]
            df_group = df[df['workload'].isin(group_workloads)]

            # Determine which metrics are available
            available_metrics = []
            for col_stem, display, ylabel in metrics:
                if f'avg_{col_stem}' in df_group.columns:
                    available_metrics.append((col_stem, display, ylabel))

            if not available_metrics:
                continue

            n_rows = len(available_metrics)
            n_cols = len(stats)

            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(6 * n_cols, 4.5 * n_rows),
                squeeze=False,
            )

            short_label = _short_group_label(gk)
            fig.suptitle(f'Trend Lines — {short_label}', fontsize=TITLE_FONTSIZE, y=1.02)

            for ri, (col_stem, display, ylabel) in enumerate(available_metrics):
                for ci, (stat, stat_label) in enumerate(stats):
                    stat_col = f'{stat}_{col_stem}'
                    # ylim_upper = min(df_group[stat_col].max())
                    ylim_upper = None
                    _plot_trendlines_for_group(
                        axes[ri][ci],
                        df_group,
                        rps_workload_pairs,
                        policies,
                        policy_colors,
                        stat_col,
                        ylabel,
                        f'{stat_label} {display}',
                        ylim_upper=ylim_upper,
                    )

            # Shared routing-policy legend at the top of the page
            # Collect one handle per policy (use policy name only, not the per-subfig label)
            seen_policies = {}
            for ax_row in axes:
                for ax in ax_row:
                    for line in ax.get_lines():
                        name = getattr(line, '_policy_name', None)
                        if name and name not in seen_policies:
                            seen_policies[name] = line

            if seen_policies:
                top_handles = list(seen_policies.values())
                top_labels = list(seen_policies.keys())
                fig.legend(
                    top_handles, top_labels,
                    loc='upper center',
                    ncol=min(len(top_handles), 6),
                    fontsize=LEGEND_FONTSIZE,
                    bbox_to_anchor=(0.5, 1.06),
                    framealpha=0.9,
                )

            fig.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig, bbox_inches='tight', dpi=300)
            plt.close(fig)

    print(f"Saved trend-line PDF to {pdf_path}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Plot trend-line graphs from routing metrics (client logs)'
    )
    parser.add_argument(
        'base_dir',
        help='Base directory for output (and recursive search if --target-dirs-file not provided)',
    )
    parser.add_argument(
        '--output-dir', '-o', default=None,
        help='Output directory for plots (default: base_dir)',
    )
    parser.add_argument(
        '--target-dirs-file', '-t', default=None,
        help='File containing list of target directories (one per line).',
    )

    args = parser.parse_args()

    base_dir = args.base_dir
    output_dir = args.output_dir if args.output_dir else base_dir

    # Load target directories from file if provided
    target_dirs = None
    if args.target_dirs_file:
        if os.path.exists(args.target_dirs_file):
            with open(args.target_dirs_file, 'r') as f:
                target_dirs = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            print(f"Using {len(target_dirs)} target directories from {args.target_dirs_file}")
        else:
            print(f"Error: Target dirs file not found: {args.target_dirs_file}")
            sys.exit(1)

    # Find and merge metrics files
    files = find_metrics_files(base_dir, target_dirs)
    if not files:
        print("No routing_strategy_metrics_from_client_log.csv files found")
        sys.exit(1)

    print(f"Found {len(files)} metrics files")
    df = merge_metrics_files(files)
    if df is None or len(df) == 0:
        print("No data to process")
        sys.exit(1)

    # Re-extract routing_policy
    df['routing_policy'] = df['strategy_full_name'].apply(extract_routing_policy)

    os.makedirs(output_dir, exist_ok=True)

    # Print summary
    print(f"\nWorkloads: {sorted(df['workload'].unique().tolist())}")
    print(f"Routing policies: {sorted(df['routing_policy'].unique().tolist())}")
    print(f"Total rows: {len(df)}")

    # Group info
    groups = group_workloads_by_category(df['workload'].unique().tolist())
    print(f"\nWorkload groups ({len(groups)}):")
    for gk in sorted(groups.keys()):
        rps_list = [r for r, _ in groups[gk]]
        print(f"  {_short_group_label(gk)}  — RPS points: {rps_list}")

    # Generate plots
    plot_trendlines(df, output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
