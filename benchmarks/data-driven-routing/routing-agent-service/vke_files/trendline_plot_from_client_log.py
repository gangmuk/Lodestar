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

def _plot_bars_twin_y(ax, df_group, rps_workload_pair, policies, policy_colors):
    """Plot Avg TTFT and P99 TTFT for each policy in one subplot with twin y-axes.

    Layout per policy group: [Avg bar | P99 bar]
    Left y-axis  (ax)  → Avg TTFT  (solid bars)
    Right y-axis (ax2) → P99 TTFT  (hatched bars)
    """
    rps, workload = rps_workload_pair
    avg_col, p99_col = 'avg_ttft', 'p99_ttft'

    has_avg = avg_col in df_group.columns
    has_p99 = p99_col in df_group.columns
    if not has_avg and not has_p99:
        ax.set_visible(False)
        return

    avg_vals, p99_vals, bar_colors = [], [], []
    for policy in policies:
        rows = df_group[
            (df_group['workload'] == workload) & (df_group['routing_policy'] == policy)
        ]
        def _mean(col):
            if col not in df_group.columns:
                return 0
            vs = [v for v in rows[col].dropna().tolist() if v > 0]
            return np.mean(vs) if vs else 0

        avg_vals.append(_mean(avg_col))
        p99_vals.append(_mean(p99_col))
        bar_colors.append(policy_colors.get(policy, '#7f7f7f'))

    if not any(v > 0 for v in avg_vals + p99_vals):
        ax.set_visible(False)
        return

    ax2 = ax.twinx()

    n = len(policies)
    bar_w = 0.35
    group_centers = np.arange(n)
    avg_x = group_centers - bar_w / 2
    p99_x = group_centers + bar_w / 2

    # ── Avg bars on left axis ────────────────────────────────────────────────
    avg_bars = []
    for i, (x, val, color) in enumerate(zip(avg_x, avg_vals, bar_colors)):
        b = ax.bar(x, val, width=bar_w, color=color, edgecolor='black', linewidth=0.8)
        b[0]._policy_name = policies[i]
        avg_bars.append(b[0])

    # ── P99 bars on right axis (hatched) ─────────────────────────────────────
    p99_bars = []
    for i, (x, val, color) in enumerate(zip(p99_x, p99_vals, bar_colors)):
        b = ax2.bar(x, val, width=bar_w, color=color, edgecolor='black',
                    linewidth=0.8, hatch='//', alpha=0.75)
        b[0]._policy_name = policies[i]
        p99_bars.append(b[0])

    # ── Annotations ─────────────────────────────────────────────────────────
    max_avg = max(avg_vals) if avg_vals else 1
    max_p99 = max(p99_vals) if p99_vals else 1

    def _annotate(axis, bars, vals, max_val):
        for bar_obj, val in zip(bars, vals):
            if val > 0:
                axis.text(
                    bar_obj.get_x() + bar_obj.get_width() / 2,
                    bar_obj.get_height() + max_val * 0.02,
                    f'{val:.0f}',
                    ha='center', va='bottom',
                    fontsize=SUBFIG_LEGEND_FONTSIZE, rotation=45,
                )

    _annotate(ax,  avg_bars, avg_vals, max_avg)
    _annotate(ax2, p99_bars, p99_vals, max_p99)

    ax.set_ylim(0, max_avg * 1.4)
    ax2.set_ylim(0, max_p99 * 1.4)

    ax.set_xticks(group_centers)
    ax.set_xticklabels(policies, fontsize=TICK_FONTSIZE - 2, rotation=30, ha='right')
    ax.set_ylabel('Avg TTFT (ms)', fontsize=AXIS_LABEL_FONTSIZE, color='black')
    ax2.set_ylabel('P99 TTFT (ms)', fontsize=AXIS_LABEL_FONTSIZE, color='dimgray')
    ax2.tick_params(axis='y', labelcolor='dimgray')
    ax.set_title(f'TTFT  (RPS {rps})', fontsize=SUBTITLE_FONTSIZE)
    ax.tick_params(axis='y', labelsize=TICK_FONTSIZE)
    ax.grid(axis='y', alpha=0.3)

    # Mini legend inside the subplot distinguishing Avg vs P99 style
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor='gray', edgecolor='black', label='Avg (left axis)'),
        Patch(facecolor='gray', edgecolor='black', hatch='//', alpha=0.75,
              label='P99 (right axis)'),
    ]
    ax.legend(handles=legend_handles, fontsize=SUBFIG_LEGEND_FONTSIZE - 1,
              loc='upper left', framealpha=0.8)


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


def plot_trendlines(df, output_dir, exclude_patterns=None):
    """Generate TTFT bar-chart plots and save to a single PDF.

    Layout: one page per workload group.
    One row of subplots — one subplot per RPS point.
    Each subplot shows Avg TTFT (left y-axis) and P99 TTFT (right y-axis)
    as paired bars grouped by routing policy.

    Args:
        exclude_patterns: list of substrings — any routing policy whose name
                          contains one of these is omitted from the plots.
    """
    exclude_patterns = exclude_patterns or []

    workloads = df['workload'].unique().tolist()
    groups = group_workloads_by_category(workloads)

    sorted_group_keys = sorted(groups.keys())
    all_policies = order_policies(df['routing_policy'].unique())
    policies = [
        p for p in all_policies
        if not any(pat in p for pat in exclude_patterns)
    ]
    if exclude_patterns:
        excluded = [p for p in all_policies if p not in policies]
        if excluded:
            print(f"  Excluded policies: {excluded}")
    policy_colors = generate_policy_colors(policies)

    pdf_path = os.path.join(output_dir, 'trendline_from_client_log.pdf')

    with PdfPages(pdf_path) as pdf:
        for gk in sorted_group_keys:
            rps_workload_pairs = groups[gk]

            group_workloads = [w for _, w in rps_workload_pairs]
            df_group = df[df['workload'].isin(group_workloads)]

            if 'avg_ttft' not in df_group.columns and 'p99_ttft' not in df_group.columns:
                continue

            # 1 row, one subplot per RPS point
            n_cols = len(rps_workload_pairs)
            fig, axes = plt.subplots(
                1, n_cols,
                figsize=(7 * n_cols, 6),
                squeeze=False,
            )

            short_label = _short_group_label(gk)
            # Title sits at the very top; legend goes just below it
            fig.suptitle(f'TTFT — {short_label}', fontsize=TITLE_FONTSIZE, y=1.10)

            for ci, rps_pair in enumerate(rps_workload_pairs):
                _plot_bars_twin_y(
                    axes[0][ci], df_group, rps_pair, policies, policy_colors,
                )

            # Shared policy-color legend placed between title and subplots
            seen_policies = {}
            for ax in axes[0]:
                for patch in ax.patches:
                    name = getattr(patch, '_policy_name', None)
                    if name and name not in seen_policies:
                        seen_policies[name] = patch

            if seen_policies:
                fig.legend(
                    list(seen_policies.values()), list(seen_policies.keys()),
                    loc='upper center',
                    ncol=min(len(seen_policies), 6),
                    fontsize=LEGEND_FONTSIZE,
                    bbox_to_anchor=(0.5, 1.02),
                    framealpha=0.9,
                )

            # Reserve top margin so title + legend don't overlap the subplots
            fig.tight_layout(rect=[0, 0, 1, 0.88])
            pdf.savefig(fig, bbox_inches='tight', dpi=300)
            plt.close(fig)

    print(f"Saved bar-chart PDF to {pdf_path}")


# ── CSV export ──────────────────────────────────────────────────────────────

def export_performance_csv(df, groups, output_dir):
    """Export a tidy performance summary CSV.

    Each row represents one (workload_group, rps, routing_policy, run) combination.
    Columns: workload_group, rps, routing_policy, run,
             avg_ttft, p99_ttft, p999_ttft,
             avg_tpot, p99_tpot, p999_tpot,
             avg_end_to_end, p99_end_to_end, p999_end_to_end
    """
    stat_cols = [
        'avg_ttft', 'p99_ttft', 'p999_ttft',
        'avg_tpot', 'p99_tpot', 'p999_tpot',
        'avg_end_to_end', 'p99_end_to_end', 'p999_end_to_end',
    ]
    # Only keep columns that actually exist in the dataframe
    stat_cols = [c for c in stat_cols if c in df.columns]

    rows = []
    for gk, rps_workload_pairs in groups.items():
        short_label = _short_group_label(gk)
        group_workloads = [w for _, w in rps_workload_pairs]
        df_group = df[df['workload'].isin(group_workloads)]

        for rps, workload in rps_workload_pairs:
            df_rps = df_group[df_group['workload'] == workload].copy()
            # Sort runs by datetime suffix so run indices are stable
            df_rps['_sort_dt'] = df_rps['strategy_full_name'].str.extract(
                r'(\d{8}_\d{6})$', expand=False
            ).fillna('')
            df_rps = df_rps.sort_values(['routing_policy', '_sort_dt'])

            for policy, df_policy in df_rps.groupby('routing_policy', sort=False):
                df_policy = df_policy.reset_index(drop=True)
                for run_idx, (_, row) in enumerate(df_policy.iterrows()):
                    record = {
                        'workload_group': short_label,
                        'rps': rps,
                        'routing_policy': policy,
                        'run': run_idx + 1,
                        'strategy_full_name': row.get('strategy_full_name', ''),
                    }
                    for col in stat_cols:
                        val = row.get(col, None)
                        record[col] = round(float(val), 2) if pd.notna(val) and val > 0 else None
                    rows.append(record)

    if not rows:
        print("No data to export to CSV")
        return

    out_df = pd.DataFrame(rows)
    # Sort for readability
    out_df = out_df.sort_values(['workload_group', 'rps', 'routing_policy', 'run']).reset_index(drop=True)

    csv_path = os.path.join(output_dir, 'performance_summary.csv')
    out_df.to_csv(csv_path, index=False)
    print(f"Saved performance summary CSV to {csv_path}  ({len(out_df)} rows)")


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
    parser.add_argument(
        '--exclude', '-e', nargs='+', default=[],
        metavar='PATTERN',
        help='Exclude routing policies whose name contains any of these substrings '
             '(plot only; excluded policies are still saved in the CSV). '
             'Example: --exclude e2e_latency_negative_linear',
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

    # Export performance summary CSV (full data, no exclusions)
    export_performance_csv(df, groups, output_dir)

    # Generate plots (with optional policy exclusions)
    exclude_patterns = args.exclude
    if exclude_patterns:
        print(f"\nExcluding from plots policies matching: {exclude_patterns}")
    plot_trendlines(df, output_dir, exclude_patterns=exclude_patterns)
    print("Done!")


if __name__ == "__main__":
    main()
