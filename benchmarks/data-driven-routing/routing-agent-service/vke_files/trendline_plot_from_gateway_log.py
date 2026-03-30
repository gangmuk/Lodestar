#!/usr/bin/env python3
"""
Plot trend line graphs from routing_strategy_metrics_gateway.csv files
(produced by compare_routing_strategies.py from gateway logs).

For each workload group (same category + sharing ratio / subcategory, varying RPS),
creates bar plots with:
  - One subplot per RPS point
  - Avg TTFT on left y-axis, P99 TTFT on right y-axis
  - Bars grouped by routing policy

Usage:
    python trendline_plot_from_gateway_log.py <base_dir> [--output-dir <output_dir>]
    python trendline_plot_from_gateway_log.py <base_dir> --target-dirs-file <file>

Example:
    python trendline_plot_from_gateway_log.py /path/to/workload-and-experiment_results/NVIDIA-A30
"""

import os
import sys
import re
import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import preprocess

# Reuse non-color helpers from the bar chart script
from merge_and_plot_all_workloads_from_client_log import (
    extract_routing_policy,
    extract_rps_from_workload,
    order_policies,
    get_policy_sort_key,
)

# ── Color logic mirrored from compare_routing_strategies.py ─────────────────
# Each entry is (keyword_or_tuple, palette).
#   - str keyword: matches if keyword is in the policy name.
#   - tuple of str: matches if ALL keywords are in the policy name.
# More-specific (compound) rules MUST come before broader ones so that e.g.
# "contextual_bandit_..._random" hits the CB-random teal palette, not the
# generic contextual_bandit red palette.
# Palette design: each family occupies a distinct hue band so no two
# families can be confused even at a glance.  Shades within a family
# are ordered dark → light for multi-run gradation.
#
# Hue allocation (approximate):
#   rl_naive .................. royal blue    (hue 225)
#   latency_pred_e2e ......... magenta/plum  (hue 300)
#   latency_pred_ttft ........ hot pink      (hue 330)
#   latency_pred_tpot ........ dark red      (hue   0)
#   prefix_cache_1 ........... steel blue    (hue 210)
#   prefix_cache_2 ........... forest green  (hue 120)
#   preble ................... orange        (hue  30)
#   CB + random .............. teal-green    (hue 160)  ← distinct from least_request cyan
#   onlinelearning_0 ......... purple        (hue 280)
#   contextual_bandit (other)  red           (hue  10)
#   prefix_hit_thresh ........ olive/khaki   (hue  80)
#   least_kv_cache ........... brown/sienna  (hue  25)
#   least_latency ............ slate indigo  (hue 250)
#   least_request ............ cyan          (hue 190)  ← clearly bluer than CB-random teal
#   random ................... lime green    (hue 100)
_COLOR_RULES = [
    ('rl_naive',                          ['#2b5cd9', '#3a6be8', '#4a7af5', '#6b93f7', '#8dabf9']),
    ('latency_predictor_e2e_latency',     ['#8b008b', '#a020a0', '#b838b8', '#c850d0', '#d870e8']),
    ('latency_predictor_ttft',            ['#e91e63', '#f06292', '#f48fb1', '#f8bbd0', '#fce4ec']),
    ('latency_predictor_avg_tpot',        ['#8b0000', '#b71c1c', '#d32f2f', '#e57373', '#ef9a9a']),
    ('prefix_cache_1',                    ['#1565c0', '#1976d2', '#2196f3', '#64b5f6', '#90caf9']),
    ('prefix_cache_2',                    ['#1b5e20', '#2e7d32', '#43a047', '#66bb6a', '#a5d6a7']),
    ('preble',                            ['#e65100', '#f57c00', '#ff9800', '#ffb74d', '#ffcc80']),
    # — CB random (orange) — matches compare_routing_strategies.py; must come BEFORE generic contextual_bandit
    (('contextual_bandit', 'random'),     ['#ff8c00', '#ffa500', '#ff7f00', '#e08600', '#ffb347']),
    # — onlinelearning_0 (purple hue ~280) — must come BEFORE generic contextual_bandit
    ('onlinelearning_0',                  ['#6a1b9a', '#7b1fa2', '#9c27b0', '#ab47bc', '#ce93d8']),
    ('contextual_bandit',                 ['#c62828', '#d32f2f', '#e53935', '#ef5350', '#ff7043']),
    ('prefix_hit_threshold_or_least_request', ['#5d6b1f', '#7c8b23', '#9e9d24', '#c0ca33', '#d4e157']),
    ('least_kv_cache',                    ['#8d6e00', '#a68500', '#bf9b30', '#d4ad48', '#e8c560']),
    ('least_latency',                     ['#4527a0', '#5e35b1', '#7e57c2', '#9575cd', '#b39ddb']),
    ('least_request',                     ['#006064', '#00838f', '#0097a7', '#00acc1', '#4dd0e1']),
    ('random',                            ['#33691e', '#558b2f', '#689f38', '#8bc34a', '#aed581']),
]
_DEFAULT_COLORS = ['#7f7f7f', '#696969', '#a9a9a9', '#c7c7c7', '#d3d3d3']


def _match_rule(keyword, policy_lower):
    """Check if a rule keyword (str or tuple of str) matches a lowered policy name."""
    if isinstance(keyword, tuple):
        return all(k in policy_lower for k in keyword)
    return keyword in policy_lower


def _rule_key(keyword):
    """Return a hashable family key for a rule keyword (str or tuple)."""
    if isinstance(keyword, tuple):
        return '&'.join(keyword)
    return keyword


def _get_base_color(policy_name: str, index_in_family: int = 0) -> str:
    """Return the base bar color for a policy, matching compare_routing_strategies.py."""
    pl = policy_name.lower()
    for keyword, palette in _COLOR_RULES:
        if _match_rule(keyword, pl):
            return palette[index_in_family % len(palette)]
    return _DEFAULT_COLORS[index_in_family % len(_DEFAULT_COLORS)]


def generate_policy_colors(policies):
    """Assign a base color to each policy using the same palette as compare_routing_strategies.py."""
    family_counts = {}
    colors = {}
    for policy in policies:
        pl = policy.lower()
        family = next((_rule_key(kw) for kw, _ in _COLOR_RULES if _match_rule(kw, pl)), 'default')
        idx = family_counts.get(family, 0)
        family_counts[family] = idx + 1
        colors[policy] = _get_base_color(policy, idx)
    return colors

RAW_LOG_NAME = "filtered-aibrix-gateway-plugins.log.csv"
GATEWAY_CSV_NAME = "routing_strategy_metrics_gateway.csv"

# Number of initial requests to skip for CB-random "post-warmup" annotation.
# The first N requests use a fallback policy until the online model is trained.
CB_RANDOM_WARMUP_REQUESTS = 5000


def find_gateway_metrics_files(base_dir, target_dirs=None):
    """Find routing_strategy_metrics_gateway.csv files."""
    files = []
    if target_dirs:
        for d in target_dirs:
            csv_path = os.path.join(d, GATEWAY_CSV_NAME)
            if os.path.exists(csv_path):
                files.append(csv_path)
            else:
                print(f"Warning: No CSV found in {d}")
    else:
        pattern = os.path.join(base_dir, "**", GATEWAY_CSV_NAME)
        files = glob.glob(pattern, recursive=True)
    return sorted(files)


def merge_gateway_metrics_files(files):
    """Load and merge gateway CSV files into a single DataFrame."""
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            print(f"Loaded {len(df)} rows from {f}")
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")
    if not dfs:
        return None
    merged = pd.concat(dfs, ignore_index=True)
    print(f"\nMerged {len(merged)} total rows from {len(dfs)} files")
    return merged


def _build_csv_dir_map(csv_files):
    """Build a map from CSV file path to its parent directory.

    Returns: {csv_path: parent_dir}
    """
    return {f: os.path.dirname(f) for f in csv_files}


def load_raw_request_data(csv_files, df_aggregated):
    """Load per-request data from raw gateway log files.

    For each strategy_full_name in df_aggregated, find the corresponding raw
    log file in the same directory tree as the aggregated CSV.

    Returns:
        dict: {strategy_full_name: DataFrame} with per-request data including
              'ttft' and 'request_index' columns.
    """
    csv_dirs = _build_csv_dir_map(csv_files)
    raw_data = {}

    # Collect unique (csv_parent_dir, strategy_full_name) pairs
    seen = set()
    for _, row in df_aggregated.iterrows():
        sfn = row.get('strategy_full_name', '')
        if not sfn or sfn in raw_data:
            continue

        # Find which CSV directory this strategy belongs to
        for csv_path, parent_dir in csv_dirs.items():
            log_path = os.path.join(parent_dir, sfn, RAW_LOG_NAME)
            if os.path.exists(log_path) and log_path not in seen:
                seen.add(log_path)
                try:
                    df_raw, _ = preprocess.parse_log_file(log_path)
                    if 'ttft' in df_raw.columns and 'request_start_time' in df_raw.columns:
                        df_raw['ttft'] = pd.to_numeric(df_raw['ttft'], errors='coerce')
                        df_raw = df_raw.dropna(subset=['ttft'])
                        df_raw = df_raw.sort_values('request_start_time').reset_index(drop=True)
                        df_raw['request_index'] = range(len(df_raw))
                        raw_data[sfn] = df_raw
                except Exception as e:
                    print(f"  Warning: Failed to load raw log {log_path}: {e}")
                break

    print(f"Loaded per-request data for {len(raw_data)} strategies")
    return raw_data


def compute_windowed_avg_ttft(df_raw, window_size=1000):
    """Compute non-overlapping window average TTFT.

    Args:
        df_raw: DataFrame with 'ttft' and 'request_index' columns, sorted by request order.
        window_size: number of requests per window (non-overlapping).

    Returns:
        (window_centers, window_avgs): lists of x-positions (request index of window center)
                                        and corresponding average TTFT values.
    """
    ttft_vals = df_raw['ttft'].values
    n = len(ttft_vals)
    centers = []
    avgs = []
    for start in range(0, n, window_size):
        end = min(start + window_size, n)
        chunk = ttft_vals[start:end]
        avgs.append(np.mean(chunk))
        centers.append((start + end - 1) / 2.0)
    return centers, avgs


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

def _plot_bars_twin_y(ax, df_group, rps_workload_pair, policies, policy_colors,
                      raw_data=None):
    """Plot Avg TTFT and P99 TTFT for each policy in one subplot with twin y-axes.

    Layout per policy group: [Avg bar | P99 bar]
    Left y-axis  (ax)  → Avg TTFT  (solid bars)
    Right y-axis (ax2) → P99 TTFT  (hatched bars)

    For contextual_bandit+random policies, a horizontal diamond marker is drawn
    on each bar showing the avg/p99 TTFT computed only from requests after the
    first CB_RANDOM_WARMUP_REQUESTS (post-warmup, i.e. after the online model
    has been trained).
    """
    rps, workload = rps_workload_pair
    avg_col, p99_col = 'avg_ttft', 'p99_ttft'

    has_avg = avg_col in df_group.columns
    has_p99 = p99_col in df_group.columns
    if not has_avg and not has_p99:
        ax.set_visible(False)
        return

    avg_vals, avg_errs, p99_vals, p99_errs, base_colors = [], [], [], [], []
    for policy in policies:
        rows = df_group[
            (df_group['workload'] == workload) & (df_group['routing_policy'] == policy)
        ]
        def _mean_std(col):
            if col not in df_group.columns:
                return 0, 0
            vs = [v for v in rows[col].dropna().tolist() if v > 0]
            if not vs:
                return 0, 0
            return np.mean(vs), (np.std(vs, ddof=1) if len(vs) > 1 else 0)

        avg_mean, avg_std = _mean_std(avg_col)
        p99_mean, p99_std = _mean_std(p99_col)
        avg_vals.append(avg_mean)
        avg_errs.append(avg_std)
        p99_vals.append(p99_mean)
        p99_errs.append(p99_std)
        base_colors.append(policy_colors.get(policy, '#7f7f7f'))

    if not any(v > 0 for v in avg_vals + p99_vals):
        ax.set_visible(False)
        return

    # P99 uses a lightened version of the same base color (no hatch)
    p99_colors = [_shade_color(c, 1.5) for c in base_colors]

    ax2 = ax.twinx()

    n = len(policies)
    bar_w = 0.35
    group_centers = np.arange(n)
    avg_x = group_centers - bar_w / 2
    p99_x = group_centers + bar_w / 2

    # ── Determine hatch pattern per policy ────────────────────────────────────
    def _hatch_for_policy(policy_name):
        pl = policy_name.lower()
        if 'onlinelearning_0' in pl:
            return 'xx'
        if 'contextual_bandit' in pl and 'random' in pl:
            return '//'
        return None

    # ── Avg bars on left axis (full color) ───────────────────────────────────
    avg_bars = []
    for i, (x, val, color) in enumerate(zip(avg_x, avg_vals, base_colors)):
        hatch = _hatch_for_policy(policies[i])
        err = avg_errs[i] if avg_errs[i] > 0 else None
        b = ax.bar(x, val, width=bar_w, color=color, edgecolor='black', linewidth=0.8,
                   hatch=hatch, yerr=err, capsize=3, error_kw=dict(elinewidth=1.2, capthick=1.2, color='black'))
        b[0]._policy_name = policies[i]
        avg_bars.append(b[0])

    # ── P99 bars on right axis (lightened shade) ─────────────────────────────
    p99_bars = []
    for i, (x, val, color) in enumerate(zip(p99_x, p99_vals, p99_colors)):
        hatch = _hatch_for_policy(policies[i])
        err = p99_errs[i] if p99_errs[i] > 0 else None
        b = ax2.bar(x, val, width=bar_w, color=color, edgecolor='black', linewidth=0.8,
                    hatch=hatch, yerr=err, capsize=3, error_kw=dict(elinewidth=1.2, capthick=1.2, color='black'))
        b[0]._policy_name = policies[i]
        p99_bars.append(b[0])

    # ── Annotations ─────────────────────────────────────────────────────────
    max_avg = max(v + e for v, e in zip(avg_vals, avg_errs)) if avg_vals else 1
    max_p99 = max(v + e for v, e in zip(p99_vals, p99_errs)) if p99_vals else 1

    def _annotate(axis, bars, vals, errs, max_val):
        for bar_obj, val, err in zip(bars, vals, errs):
            if val > 0:
                top = val + err  # place text above error bar cap
                axis.text(
                    bar_obj.get_x() + bar_obj.get_width() / 2,
                    top + max_val * 0.02,
                    f'{val:.0f}',
                    ha='center', va='bottom',
                    fontsize=SUBFIG_LEGEND_FONTSIZE, rotation=90,
                )

    _annotate(ax,  avg_bars, avg_vals, avg_errs, max_avg)
    _annotate(ax2, p99_bars, p99_vals, p99_errs, max_p99)

    # ── Post-warmup annotation for CB-random policies ────────────────────
    # For contextual_bandit+random, compute avg/p99 TTFT from requests after
    # the first CB_RANDOM_WARMUP_REQUESTS.  Draw a bold horizontal line
    # across the bar at the post-warmup value so the viewer can see:
    #   bar top  = all-requests metric  (includes warmup/fallback penalty)
    #   line     = post-warmup metric   (after online model kicks in)
    #   gap      = warmup penalty
    if raw_data:
        for i, policy in enumerate(policies):
            pl = policy.lower()
            if not ('contextual_bandit' in pl and 'random' in pl):
                continue
            # Find strategy_full_names for this policy + workload
            rows = df_group[
                (df_group['workload'] == workload) & (df_group['routing_policy'] == policy)
            ]
            sfn_list = rows['strategy_full_name'].dropna().tolist()
            # Collect post-warmup TTFT values across all runs
            post_warmup_ttfts = []
            for sfn in sfn_list:
                if sfn not in raw_data:
                    continue
                df_raw = raw_data[sfn]
                if 'ttft' not in df_raw.columns:
                    continue
                post = df_raw.iloc[CB_RANDOM_WARMUP_REQUESTS:]
                if len(post) > 0:
                    post_warmup_ttfts.append(post['ttft'].values)
            if not post_warmup_ttfts:
                continue
            all_ttft = np.concatenate(post_warmup_ttfts)
            pw_avg = np.mean(all_ttft)
            pw_p99 = np.percentile(all_ttft, 99)

            # Horizontal line across the Avg bar (left axis)
            # avg_x[i] is the left edge of the bar; bar spans [avg_x[i], avg_x[i] + bar_w]
            ax.hlines(pw_avg, avg_x[i], avg_x[i] + bar_w,
                      colors='black', linewidths=2.5, zorder=10)
            ax.text(avg_x[i] + bar_w / 2, pw_avg + max_avg * 0.02,
                    f'{pw_avg:.0f}',
                    ha='center', va='bottom', fontsize=SUBFIG_LEGEND_FONTSIZE - 1,
                    color='black', fontweight='bold')

            # Horizontal line across the P99 bar (right axis)
            # p99_x[i] is the left edge of the bar; bar spans [p99_x[i], p99_x[i] + bar_w]
            ax2.hlines(pw_p99, p99_x[i], p99_x[i] + bar_w,
                       colors='black', linewidths=2.5, zorder=10)
            ax2.text(p99_x[i] + bar_w / 2, pw_p99 + max_p99 * 0.02,
                     f'{pw_p99:.0f}',
                     ha='center', va='bottom', fontsize=SUBFIG_LEGEND_FONTSIZE - 1,
                     color='black', fontweight='bold')

    ax.set_ylim(0, max_avg * 1.4)
    ax2.set_ylim(0, max_p99 * 1.4)

    # Use numeric indices on x-axis; the top-level color legend identifies each policy.
    ax.set_xticks(group_centers)
    ax.set_xticklabels([str(i + 1) for i in range(n)], fontsize=TICK_FONTSIZE)
    ax.set_xlabel('Policy index (see legend)', fontsize=AXIS_LABEL_FONTSIZE - 1)
    ax.set_ylabel('Avg TTFT (ms)', fontsize=AXIS_LABEL_FONTSIZE, color='black')
    ax2.set_ylabel('P99 TTFT (ms)', fontsize=AXIS_LABEL_FONTSIZE, color='dimgray')
    ax2.tick_params(axis='y', labelcolor='dimgray')
    ax.set_title(f'TTFT  (RPS {rps})', fontsize=SUBTITLE_FONTSIZE)
    ax.tick_params(axis='y', labelsize=TICK_FONTSIZE)
    ax.grid(axis='y', alpha=0.3)

    # Mini legend inside the subplot distinguishing Avg vs P99 style
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_handles = [
        Patch(facecolor='#555555', edgecolor='black', label='Avg (left axis)'),
        Patch(facecolor='#aaaaaa', edgecolor='black', label='P99 — lighter shade (right axis)'),
        Line2D([], [], color='black', linewidth=2.5, linestyle='-',
               label=f'CB-random after {CB_RANDOM_WARMUP_REQUESTS} reqs'),
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


def _draw_best_policy_strip(ax, first_run_data, policy_colors, window_size):
    """Draw a colored strip along the bottom of *ax* showing the best policy per window.

    For each window position where at least two policies have data, the strip
    is colored by the policy with the lowest windowed-avg TTFT.  The strip is
    rendered in axes-transform y so it stays fixed at the bottom regardless of
    zoom / y-limits.

    Args:
        ax: matplotlib Axes that already has the time-series lines plotted.
        first_run_data: {policy: (centers, avgs)} — first-run windowed data.
        policy_colors: {policy: color}.
        window_size: used to compute the width of each rectangle.
    """
    from matplotlib.patches import Rectangle
    import matplotlib.transforms as mtransforms

    # Build a unified set of window centers across all policies
    all_centers = sorted({c for centers, _ in first_run_data.values() for c in centers})
    if not all_centers:
        return

    # For each center, find the policy with the lowest avg TTFT
    # Build lookup: {policy: {center: avg}}
    policy_lookup = {}
    for policy, (centers, avgs) in first_run_data.items():
        policy_lookup[policy] = dict(zip(centers, avgs))

    best_segments = []  # [(x_start, x_end, policy)]
    half_w = window_size / 2.0

    for center in all_centers:
        candidates = {}
        for policy, lookup in policy_lookup.items():
            if center in lookup:
                candidates[policy] = lookup[center]
        if not candidates:
            continue
        best_policy = min(candidates, key=candidates.get)
        x_start = center - half_w
        x_end = center + half_w
        best_segments.append((x_start, x_end, best_policy))

    if not best_segments:
        return

    # Merge adjacent segments with the same policy
    merged = [best_segments[0]]
    for x_start, x_end, policy in best_segments[1:]:
        prev_start, prev_end, prev_policy = merged[-1]
        if policy == prev_policy and abs(x_start - prev_end) < 1e-6:
            merged[-1] = (prev_start, x_end, policy)
        else:
            merged.append((x_start, x_end, policy))

    # Draw the strip using a blended transform: x in data coords, y in axes coords
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    strip_height = 0.03  # 3% of axes height
    for x_start, x_end, policy in merged:
        color = policy_colors.get(policy, '#7f7f7f')
        rect = Rectangle(
            (x_start, 0), x_end - x_start, strip_height,
            transform=trans, color=color, alpha=0.85,
            clip_on=True, zorder=5,
        )
        ax.add_patch(rect)

    # Add a thin label on the left edge
    ax.text(
        0.0, strip_height / 2, ' best policy ',
        transform=ax.transAxes, fontsize=SUBFIG_LEGEND_FONTSIZE - 2,
        va='center', ha='left', color='white',
        bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.6),
        zorder=6,
    )


def _plot_timeseries_windowed_ttft(
    ax, df_group, rps_workload_pair, policies, policy_colors, raw_data,
    window_size=1000,
):
    """Plot non-overlapping windowed average TTFT time series for one RPS point.

    Each policy is a separate line. Multiple runs of the same policy are
    plotted with progressively lighter shades.

    Args:
        ax: matplotlib Axes (full width for this RPS)
        df_group: aggregated DataFrame for this workload group
        rps_workload_pair: (rps_int, workload_str)
        policies: ordered list of policies to plot
        policy_colors: {policy: color}
        raw_data: {strategy_full_name: DataFrame} with per-request data
        window_size: non-overlapping window size in number of requests
    """
    rps, workload = rps_workload_pair

    policy_style_idx = {}
    # Collect mean windowed data per policy for the "best policy" strip
    # {policy: (centers, mean_avgs)}
    first_run_data = {}

    for pi, policy in enumerate(policies):
        color = policy_colors.get(policy, '#7f7f7f')
        if color not in policy_style_idx:
            policy_style_idx[color] = 0
        style_idx = policy_style_idx[color]
        policy_style_idx[color] += 1

        linestyle = LINE_STYLES[style_idx % len(LINE_STYLES)]

        # Find all strategy_full_names for this policy + workload
        rows = df_group[
            (df_group['workload'] == workload) & (df_group['routing_policy'] == policy)
        ].copy()
        rows['_sort_dt'] = rows['strategy_full_name'].str.extract(
            r'(\d{8}_\d{6})$', expand=False
        ).fillna('')
        rows = rows.sort_values('_sort_dt')

        sfn_list = rows['strategy_full_name'].tolist()

        # Collect windowed data from all runs
        all_run_data = []  # list of (centers, avgs) per run
        for sfn in sfn_list:
            if sfn not in raw_data:
                continue
            df_raw = raw_data[sfn]
            centers, avgs = compute_windowed_avg_ttft(df_raw, window_size)
            if centers:
                all_run_data.append((np.array(centers), np.array(avgs)))

        if not all_run_data:
            continue

        marker = MARKERS[pi % len(MARKERS)]

        if len(all_run_data) == 1:
            # Single run: just draw the line with markers
            centers, avgs = all_run_data[0]
            ax.plot(
                centers, avgs,
                color=color, linestyle=linestyle, linewidth=1.8,
                marker=marker, markersize=10, markerfacecolor='none',
                markeredgecolor=color, markeredgewidth=1.8,
                label=policy, alpha=0.9, zorder=3,
            )
            first_run_data[policy] = (centers.tolist(), avgs.tolist())
        else:
            # Multiple runs: compute mean + min/max envelope across runs.
            # Align runs on the shortest common length.
            min_len = min(len(a) for _, a in all_run_data)
            centers = all_run_data[0][0][:min_len]
            stacked = np.stack([a[:min_len] for _, a in all_run_data], axis=0)  # (n_runs, n_windows)

            mean_vals = np.mean(stacked, axis=0)
            min_vals = np.min(stacked, axis=0)
            max_vals = np.max(stacked, axis=0)

            # Shaded band: min–max envelope
            ax.fill_between(
                centers, min_vals, max_vals,
                color=color, alpha=0.18, zorder=1,
            )
            # Mean line with markers
            ax.plot(
                centers, mean_vals,
                color=color, linestyle=linestyle, linewidth=1.8,
                marker=marker, markersize=10, markerfacecolor='none',
                markeredgecolor=color, markeredgewidth=1.8,
                label=f'{policy} (mean, n={len(all_run_data)})', alpha=0.9, zorder=3,
            )
            first_run_data[policy] = (centers.tolist(), mean_vals.tolist())

    # ── "Best policy" colored strip along the bottom of the x-axis ────────
    if first_run_data:
        _draw_best_policy_strip(ax, first_run_data, policy_colors, window_size)

    ax.set_xlabel('Request Index', fontsize=AXIS_LABEL_FONTSIZE, labelpad=6)
    ax.set_ylabel('Avg TTFT (ms)', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(f'Windowed Avg TTFT (window={window_size})  —  RPS {rps}',
                 fontsize=SUBTITLE_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.grid(alpha=0.3)

    # ── Secondary x-axis on the bottom showing time (seconds) ─────────
    # Build index-to-time mapping from the longest raw data series
    best_index_to_time = {}
    for sfn_list_policy in [rows['strategy_full_name'].tolist()
                            for _, rows in df_group[df_group['workload'] == workload]
                            .groupby('routing_policy')]:
        for sfn in sfn_list_policy:
            if sfn not in raw_data:
                continue
            df_raw = raw_data[sfn]
            if 'request_start_time' not in df_raw.columns:
                continue
            times = df_raw['request_start_time'].values.astype(float)
            t0 = times[0]
            rel_times = times - t0
            n = len(rel_times)
            idx_to_time = {}
            for start in range(0, n, window_size):
                end = min(start + window_size, n)
                center = (start + end - 1) / 2.0
                idx_to_time[center] = np.mean(rel_times[start:end])
            if len(idx_to_time) > len(best_index_to_time):
                best_index_to_time = idx_to_time

    if best_index_to_time:
        ax2 = ax.secondary_xaxis(-0.30)
        primary_ticks = ax.get_xticks()
        sorted_indices = sorted(best_index_to_time.keys())
        time_ticks = []
        time_labels = []
        for idx in primary_ticks:
            if not sorted_indices:
                continue
            if idx <= sorted_indices[0]:
                t = best_index_to_time[sorted_indices[0]]
            elif idx >= sorted_indices[-1]:
                t = best_index_to_time[sorted_indices[-1]]
            else:
                for i in range(len(sorted_indices) - 1):
                    if sorted_indices[i] <= idx <= sorted_indices[i + 1]:
                        frac = (idx - sorted_indices[i]) / (sorted_indices[i + 1] - sorted_indices[i])
                        t = best_index_to_time[sorted_indices[i]] + frac * (
                            best_index_to_time[sorted_indices[i + 1]] - best_index_to_time[sorted_indices[i]])
                        break
            time_ticks.append(idx)
            time_labels.append(f'{t:.0f}s')
        ax2.set_xticks(time_ticks)
        ax2.set_xticklabels(time_labels)
        ax2.set_xlabel('Time (seconds)', fontsize=AXIS_LABEL_FONTSIZE)
        ax2.tick_params(axis='x', labelsize=TICK_FONTSIZE - 2)


def plot_trendlines(df, output_dir, exclude_patterns=None, raw_data=None, window_size=1000):
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

    has_raw = raw_data is not None and len(raw_data) > 0
    pdf_path = os.path.join(output_dir, 'trendline_from_gateway_log.pdf')

    with PdfPages(pdf_path) as pdf:
        for gk in sorted_group_keys:
            rps_workload_pairs = groups[gk]

            group_workloads = [w for _, w in rps_workload_pairs]
            df_group = df[df['workload'].isin(group_workloads)]

            if 'avg_ttft' not in df_group.columns and 'p99_ttft' not in df_group.columns:
                continue

            n_cols = len(rps_workload_pairs)
            group_policies = [p for p in policies if p in df_group['routing_policy'].values]
            n_policies = len(group_policies)

            # ── Layout (top → bottom, in inches) ────────────────────────────
            #   title_in   : suptitle
            #   gap_in     : breathing room
            #   legend_in  : legend box
            #   gap2_in    : gap between legend and bar row
            #   bar_in     : bar-chart row
            #   ts_in * N  : one time-series row per RPS point (full width)
            ncol_legend = 1
            n_legend_rows = n_policies
            title_in   = 0.55
            gap_in     = 0.35
            legend_in  = n_legend_rows * 0.38 + 0.45
            gap2_in    = 0.30
            bar_in     = 5.5
            ts_in      = 5.0   # height per time-series row (extra space for secondary x-axis)
            n_ts_rows  = n_cols if has_raw else 0
            fig_height = title_in + gap_in + legend_in + gap2_in + bar_in + n_ts_rows * ts_in

            fig_width = max(6, 6 * n_cols)

            # Use GridSpec: row 0 has n_cols columns (bar charts),
            # rows 1..n_cols each have 1 column spanning full width (time series).
            from matplotlib.gridspec import GridSpec
            n_grid_rows = 1 + n_ts_rows
            height_ratios = [bar_in] + [ts_in] * n_ts_rows

            # Reserve top space for title + legend outside the gridspec
            header_frac = (title_in + gap_in + legend_in + gap2_in) / fig_height
            subplot_top = 1.0 - header_frac

            gs = GridSpec(
                n_grid_rows, n_cols,
                figure=None,
                height_ratios=height_ratios,
                hspace=0.65,
            )

            fig = plt.figure(figsize=(fig_width, fig_height))

            # ── Bar chart row (row 0, one subplot per RPS) ──────────────────
            bar_axes = []
            for ci in range(n_cols):
                ax = fig.add_subplot(gs[0, ci])
                bar_axes.append(ax)

            title_y = 1.0 - (title_in / 2) / fig_height
            legend_y = 1.0 - (title_in + gap_in) / fig_height

            short_label = _short_group_label(gk)
            fig.suptitle(f'TTFT — {short_label}', fontsize=TITLE_FONTSIZE, y=title_y)

            for ci, rps_pair in enumerate(rps_workload_pairs):
                _plot_bars_twin_y(
                    bar_axes[ci], df_group, rps_pair, group_policies, policy_colors,
                    raw_data=raw_data,
                )

            # Shared policy-color legend
            legend_labels = [f'{i+1}. {p}' for i, p in enumerate(group_policies)]
            legend_colors = [policy_colors.get(p, '#7f7f7f') for p in group_policies]
            from matplotlib.patches import Patch
            legend_handles = [Patch(facecolor=c, edgecolor='black', label=lbl)
                              for c, lbl in zip(legend_colors, legend_labels)]
            if legend_handles:
                fig.legend(
                    handles=legend_handles,
                    loc='upper left',
                    ncol=ncol_legend,
                    fontsize=LEGEND_FONTSIZE - 1,
                    bbox_to_anchor=(0.01, legend_y),
                    framealpha=0.9,
                    title='Routing policies',
                    title_fontsize=LEGEND_FONTSIZE,
                )

            # ── Time-series rows (one per RPS, full width) ──────────────────
            if has_raw:
                for ri, rps_pair in enumerate(rps_workload_pairs):
                    ax_ts = fig.add_subplot(gs[1 + ri, :])  # span all columns
                    _plot_timeseries_windowed_ttft(
                        ax_ts, df_group, rps_pair, group_policies,
                        policy_colors, raw_data, window_size=window_size,
                    )

            gs.tight_layout(fig, rect=[0, 0, 1, subplot_top])
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

    csv_path = os.path.join(output_dir, 'performance_summary_gateway.csv')
    out_df.to_csv(csv_path, index=False)
    print(f"Saved performance summary CSV to {csv_path}  ({len(out_df)} rows)")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Plot trend-line graphs from gateway routing metrics '
                    '(routing_strategy_metrics_gateway.csv produced by compare_routing_strategies.py)'
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
        '--window-size', '-w', type=int, default=1000,
        help='Window size (number of requests) for the windowed avg TTFT time-series plot (default: 1000).',
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

    # Find and merge gateway CSV files
    files = find_gateway_metrics_files(base_dir, target_dirs)
    if not files:
        print(f"No {GATEWAY_CSV_NAME} files found")
        sys.exit(1)

    print(f"Found {len(files)} metrics files")
    df = merge_gateway_metrics_files(files)
    if df is None or len(df) == 0:
        print("No data to process")
        sys.exit(1)

    # Re-derive detailed routing_policy from strategy_full_name (same as client-log version)
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

    # Load per-request raw data for time series plots
    print("\nLoading per-request raw data for time series plots...")
    raw_data = load_raw_request_data(files, df)

    # Generate plots (with optional policy exclusions)
    exclude_patterns = args.exclude
    if exclude_patterns:
        print(f"\nExcluding from plots policies matching: {exclude_patterns}")
    plot_trendlines(df, output_dir, exclude_patterns=exclude_patterns, raw_data=raw_data, window_size=args.window_size)
    print("Done!")


if __name__ == "__main__":
    main()
