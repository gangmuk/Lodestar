#!/usr/bin/env python3
"""
Publication-quality analysis of KV cache hit ratio's nonlinear, conditional
effect on TTFT in LLM request routing.

Generates a 2x3 figure demonstrating:
  (a) Pearson vs Spearman discrepancy — evidence of nonlinearity
  (b) The confound: kv_hit_ratio is anti-correlated with input_tokens
  (c) Binned TTFT vs kv_hit_ratio — the nonlinear curve
  (d) Sign reversal: kv_hit_ratio helps long requests, hurts short ones
  (e) Load modulation: effect vanishes under high system load
  (f) 3-way interaction heatmap — the full picture a linear model cannot capture

Usage:
    python plot_kv_hit_ratio_analysis.py <path_to_data-processed.csv> [output.pdf]
"""

import sys
import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'lines.linewidth': 1.2,
    'lines.markersize': 4,
    'text.usetex': False,
})

# Color palette (colorblind-safe, inspired by Okabe-Ito)
C_BLUE   = '#0072B2'
C_ORANGE = '#E69F00'
C_GREEN  = '#009E73'
C_RED    = '#D55E00'
C_PURPLE = '#CC79A7'
C_GREY   = '#999999'
C_BLACK  = '#000000'

# ---------------------------------------------------------------------------
# Data loading and per-sample feature extraction
# ---------------------------------------------------------------------------
def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path)
    n = len(df)

    # Extract the SELECTED pod's features for each sample
    sel_kv = np.empty(n)
    sel_waiting = np.empty(n)
    sel_prefill = np.empty(n)
    sel_running = np.empty(n)
    sel_inflight_prefill = np.empty(n)
    sel_gpu_kv = np.empty(n)

    for i in range(n):
        pod = df.loc[i, 'selected_pod']
        sel_kv[i]               = df.loc[i, f'{pod}-kv_hit_ratio']
        sel_waiting[i]          = df.loc[i, f'{pod}-waiting_requests']
        sel_prefill[i]          = df.loc[i, f'{pod}-prefill_tokens']
        sel_running[i]          = df.loc[i, f'{pod}-running_requests']
        sel_inflight_prefill[i] = df.loc[i, f'{pod}-inflight_prefill_requests']
        sel_gpu_kv[i]           = df.loc[i, f'{pod}-gpu_kv_cache']

    ttft          = df['ttft'].values.astype(np.float64)
    input_tokens  = df['input_tokens'].values.astype(np.float64)

    return (df, ttft, input_tokens, sel_kv, sel_waiting, sel_prefill,
            sel_running, sel_inflight_prefill, sel_gpu_kv)


# ---------------------------------------------------------------------------
# Panel (a): Scatter + dual correlation annotation
# ---------------------------------------------------------------------------
def _draw_vertical_arrow(ax, x_frac=0.95, y_top=0.50, y_bot=0.05,
                         label='More KV cache\nhit benefit', width=0.14):
    """Draw a wide vertical arrow with text inside."""
    from matplotlib.patches import FancyArrow
    # Wide arrow body with arrowhead
    arrow = FancyArrow(
        x_frac, y_top, 0, -(y_top - y_bot),
        width=width, head_width=width * 1.8, head_length=0.04,
        transform=ax.transAxes, fc=C_GREY, ec='white',
        alpha=0.25, zorder=10, linewidth=0.5)
    ax.add_patch(arrow)
    # Text inside the arrow, rotated 90°
    ax.text(x_frac, (y_top + y_bot) / 2 + 0.02, label,
            transform=ax.transAxes, fontsize=11, color='#444444',
            ha='center', va='center', rotation=90, fontweight='bold')


def _fan_chart(ax, x_data, y_data, n_bins, color, percentiles_outer=(1, 99),
               percentiles_mid=(10, 90), percentiles_inner=(25, 75)):
    """Draw a fan chart: median line + shaded percentile bands."""
    bins = np.percentile(x_data, np.linspace(0, 100, n_bins + 1))
    centers, p50 = [], []
    bands = {k: [] for k in ['p_lo_outer', 'p_hi_outer', 'p_lo_mid', 'p_hi_mid',
                              'p_lo_inner', 'p_hi_inner']}

    for j in range(len(bins) - 1):
        mask = (x_data >= bins[j]) & (x_data < bins[j + 1])
        if j == len(bins) - 2:  # include right edge in last bin
            mask = (x_data >= bins[j]) & (x_data <= bins[j + 1])
        if mask.sum() > 10:
            y = y_data[mask]
            centers.append(np.median(x_data[mask]))
            p50.append(np.median(y))
            bands['p_lo_outer'].append(np.percentile(y, percentiles_outer[0]))
            bands['p_hi_outer'].append(np.percentile(y, percentiles_outer[1]))
            bands['p_lo_mid'].append(np.percentile(y, percentiles_mid[0]))
            bands['p_hi_mid'].append(np.percentile(y, percentiles_mid[1]))
            bands['p_lo_inner'].append(np.percentile(y, percentiles_inner[0]))
            bands['p_hi_inner'].append(np.percentile(y, percentiles_inner[1]))

    centers = np.array(centers)
    p50 = np.array(p50)
    for k in bands:
        bands[k] = np.array(bands[k])

    # Outer band (p1–p99)
    # ax.fill_between(centers, bands['p_lo_outer'], bands['p_hi_outer'],
    #                 alpha=0.05, color=color, label='p1–p99')
    # # Mid band (p10–p90)
    # ax.fill_between(centers, bands['p_lo_mid'], bands['p_hi_mid'],
    #                 alpha=0.10, color=color, label='p10–p90')
    # # Inner band (p25–p75)
    # ax.fill_between(centers, bands['p_lo_inner'], bands['p_hi_inner'],
    #                 alpha=0.18, color=color, label='p25–p75')
    # # Median line
    # ax.plot(centers, p50, color=color, linewidth=1, marker='o', markersize=2, zorder=5, label='Median')

    return centers, p50


def panel_a(ax, sel_kv, ttft):
    # Scatter samples
    rng = np.random.RandomState(42)
    idx = rng.choice(len(sel_kv), size=min(3000, len(sel_kv)), replace=False)
    ax.scatter(sel_kv[idx], ttft[idx], s=10, alpha=0.15, color=C_BLUE,
               rasterized=True, edgecolors='none', label='Samples')

    # Fan chart: percentile bands
    _fan_chart(ax, sel_kv, ttft, n_bins=20, color=C_BLUE)

    # Linear fit line for contrast
    slope, intercept = np.polyfit(sel_kv, ttft, 1)
    x_line = np.linspace(sel_kv.min(), sel_kv.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color=C_RED, linewidth=1.5,
            linestyle='--', label='Linear fit', zorder=4)

    r_p, _ = stats.pearsonr(sel_kv, ttft)
    r_s, _ = stats.spearmanr(sel_kv, ttft)

    ax.set_xlim(-0.5, None)
    ax.set_ylim(-0.5, None)
    ax.set_xlabel('KV cache hit ratio (selected pod)')
    ax.set_ylabel('TTFT (ms)')
    ax.set_title('(a)  KV hit vs TTFT')
    ax.legend(loc='upper left', framealpha=0.9, fontsize=10)


# ---------------------------------------------------------------------------
# Panel (b): System load vs TTFT (similar fan chart as panel a)
# ---------------------------------------------------------------------------
def panel_b_load(ax, sel_waiting, ttft):
    # Scatter samples with jitter (waiting_requests is integer)
    rng = np.random.RandomState(99)
    idx = rng.choice(len(sel_waiting), size=min(3000, len(sel_waiting)), replace=False)
    jitter_x = rng.normal(0, 0.2, size=len(idx))
    ax.scatter(sel_waiting[idx] + jitter_x, ttft[idx], s=10, alpha=0.15, color=C_GREEN,
               rasterized=True, edgecolors='none', label='Samples')

    # Fan chart: percentile bands
    _fan_chart(ax, sel_waiting, ttft, n_bins=15, color=C_ORANGE)

    # Linear fit line for contrast
    slope, intercept = np.polyfit(sel_waiting, ttft, 1)
    x_line = np.linspace(sel_waiting.min(), sel_waiting.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color=C_RED, linewidth=1.5,
            linestyle='--', label='Linear fit', zorder=4)

    r_p, _ = stats.pearsonr(sel_waiting, ttft)
    r_s, _ = stats.spearmanr(sel_waiting, ttft)

    ax.set_xlim(-0.05, 6)
    ax.set_ylim(-0.5, 15000)
    ax.set_aspect('auto')
    ax.set_xlabel('Waiting requests (selected pod)')
    ax.set_ylabel('TTFT (ms)')
    ax.set_title('(b)  System load vs TTFT')
    ax.legend(loc='upper left', framealpha=0.9, fontsize=10)


# ---------------------------------------------------------------------------
# Panel (c): Confound — kv vs input_tokens (was panel b)
# ---------------------------------------------------------------------------
def panel_b(ax, sel_kv, input_tokens):
    # Scatter samples with jitter (kv_hit_ratio is integer-valued, causes overlap)
    rng = np.random.RandomState(7)
    idx = rng.choice(len(sel_kv), size=min(3000, len(sel_kv)), replace=False)
    jitter_x = rng.normal(0, 150, size=len(idx))  # small relative to token range
    jitter_y = rng.normal(0, 0.4, size=len(idx))  # small relative to kv range 0-58
    ax.scatter(input_tokens[idx] + jitter_x, sel_kv[idx] + jitter_y,
               s=5, alpha=0.12, color=C_GREEN,
               rasterized=True, edgecolors='none', label='Samples')

    # Fan chart: percentile bands
    _fan_chart(ax, input_tokens, sel_kv, n_bins=25, color=C_GREEN)

    r_s, _ = stats.spearmanr(sel_kv, input_tokens)

    ax.set_xlabel('Input tokens')
    ax.set_ylabel('KV cache hit ratio')
    ax.set_title(f'(c)  Confound: shorter requests get higher cache hits\n(Spearman $\\rho$={r_s:+.3f})')
    ax.legend(loc='upper right', framealpha=0.9, fontsize=10)


# ---------------------------------------------------------------------------
# Panel (c): Binned TTFT curve (showing the nonlinear shape)
# ---------------------------------------------------------------------------
def panel_c(ax, sel_kv, ttft, input_tokens):
    kv_edges = [0, 1, 3, 5, 8, 12, 18, 25, 35, 58]
    centers, means, medians, ci_lo, ci_hi, counts = [], [], [], [], [], []
    mean_input_per_bin = []

    for j in range(len(kv_edges) - 1):
        mask = (sel_kv >= kv_edges[j]) & (sel_kv < kv_edges[j + 1])
        if mask.sum() > 30:
            t = ttft[mask]
            centers.append((kv_edges[j] + kv_edges[j + 1]) / 2)
            means.append(t.mean())
            medians.append(np.median(t))
            # Bootstrap 95% CI for median
            boot = [np.median(np.random.choice(t, len(t), replace=True))
                    for _ in range(500)]
            ci_lo.append(np.percentile(boot, 2.5))
            ci_hi.append(np.percentile(boot, 97.5))
            counts.append(mask.sum())
            mean_input_per_bin.append(input_tokens[mask].mean())

    centers = np.array(centers)
    medians = np.array(medians)
    ci_lo = np.array(ci_lo)
    ci_hi = np.array(ci_hi)

    ax.fill_between(centers, ci_lo, ci_hi, alpha=0.2, color=C_BLUE)
    ax.plot(centers, medians, color=C_BLUE, marker='o', markersize=4, linewidth=2, label='Median TTFT', zorder=5)

    # Annotate sample counts
    for x, y, n in zip(centers, medians, counts):
        ax.annotate(f'n={n}', xy=(x, y), fontsize=8, ha='center',
                    va='bottom', xytext=(0, 6), textcoords='offset points',
                    color=C_GREY)

    # Secondary axis: mean input tokens per bin (to show the confound)
    ax2 = ax.twinx()
    ax2.bar(centers, mean_input_per_bin, width=2.5, alpha=0.18,
            color=C_ORANGE, zorder=1, label='Mean input tokens')
    ax2.set_ylabel('Mean input tokens', color=C_ORANGE, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=C_ORANGE, labelsize=7)

    ax.set_xlabel('KV cache hit ratio (selected pod)')
    ax.set_ylabel('Median TTFT (ms)')
    ax.set_title('(d)  Non-monotonic curve driven\nby input-length confound')
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    # Combined legend
    lines_a, labels_a = ax.get_legend_handles_labels()
    lines_b, labels_b = ax2.get_legend_handles_labels()
    ax.legend(lines_a + lines_b, labels_a + labels_b, loc='upper right',
              framealpha=0.9, fontsize=10)


# ---------------------------------------------------------------------------
# Panel (d): Controlled sign reversal (grouped bar chart)
# ---------------------------------------------------------------------------
def panel_d(ax, sel_kv, ttft, input_tokens, sel_waiting):
    input_edges = [1, 1000, 5000, 30000]
    labels = ['Short\n(1–1000)', 'Medium\n(1001–5000)', 'Long\n(5001–30000)']
    diffs_pct = []
    hi_vals, lo_vals = [], []
    thresholds = []

    # Filter to mild load (wait < 2) to control for load confound
    mild_load = sel_waiting < 2

    for i in range(3):
        lo_q, hi_q = input_edges[i], input_edges[i + 1]
        if i < 2:
            inp_mask = (input_tokens >= lo_q) & (input_tokens < hi_q)
        else:
            inp_mask = (input_tokens >= lo_q) & (input_tokens <= hi_q)
        mask = inp_mask & mild_load
        bkv = sel_kv[mask]
        bttft = ttft[mask]
        med = np.median(bkv)
        thresholds.append(med)
        high_m = bttft[bkv > med].mean()
        low_m  = bttft[bkv <= med].mean()
        hi_vals.append(high_m)
        lo_vals.append(low_m)
        diffs_pct.append((low_m - high_m) / low_m * 100)

    thresh_str = '/'.join(f'{t:.0f}' for t in thresholds)
    x = np.arange(3)
    w = 0.32
    bars_lo = ax.bar(x - w / 2, lo_vals, w,
                     label=f'Low KV hit ($\\leq$ median)',
                     color=C_RED, alpha=0.85, edgecolor='white', linewidth=0.5)
    bars_hi = ax.bar(x + w / 2, hi_vals, w,
                     label=f'High KV hit (> median)',
                     color=C_BLUE, alpha=0.85, edgecolor='white', linewidth=0.5)

    # Annotate percentage difference
    for j, (xl, xh, d) in enumerate(zip(lo_vals, hi_vals, diffs_pct)):
        y_top = max(xl, xh)
        color = C_GREEN if d > 0 else C_RED
        sign = '+' if d > 0 else ''
        ax.annotate(f'{sign}{d:.1f}%', xy=(x[j], y_top),
                    xytext=(0, 8), textcoords='offset points',
                    ha='center', va='bottom', fontsize=11, fontweight='bold',
                    color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Mean TTFT (ms)')
    ax.set_title('(c)  KV hit effect by request length')
    ax.legend(loc='upper left', framealpha=0.9)

    # Add headroom so annotations don't clip
    ymax = max(max(lo_vals), max(hi_vals))
    ax.set_ylim(0, ymax * 1.18)

    # Add a subtle horizontal line at 0-diff reference
    ax.axhline(y=0, color='gray', linewidth=0.3)


# ---------------------------------------------------------------------------
# Panel (e): Load modulation (correlation by load bucket)
# ---------------------------------------------------------------------------
def panel_e(ax, sel_kv, ttft, sel_waiting):
    wait_bins = [0, 1, 5, 100]
    wait_labels = ['0-0', '1-4', '5+']
    bin_labels = []
    rho_vals = []
    n_vals = []

    for j in range(len(wait_bins) - 1):
        mask = (sel_waiting >= wait_bins[j]) & (sel_waiting < wait_bins[j + 1])
        if mask.sum() > 50:
            rho, _ = stats.spearmanr(sel_kv[mask], ttft[mask])
            rho_vals.append(rho)
            bin_labels.append(wait_labels[j])
            n_vals.append(mask.sum())

    x = np.arange(len(rho_vals))
    colors = [C_BLUE if r < -0.1 else (C_GREY if abs(r) <= 0.1 else C_RED)
              for r in rho_vals]

    bars = ax.bar(x, rho_vals, color=colors, alpha=0.85, edgecolor='white',
                  linewidth=0.5)


    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axhline(y=-0.1, color='gray', linewidth=0.3, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax.set_xlabel('Waiting requests (system load)')
    ax.set_ylabel('Spearman $\\rho$(KV hit, TTFT)')
    ax.set_title('(d)  KV hit effect by system load')
    ax.set_ylim(min(rho_vals) - 0.15, 0.15)

    _draw_vertical_arrow(ax)


# ---------------------------------------------------------------------------
# Panel (f): 3-way interaction heatmap
# ---------------------------------------------------------------------------
def _quartile_edges_and_labels(input_tokens):
    """Return quartile edges and labels matching panel (d) categories."""
    edges = np.percentile(input_tokens, [0, 25, 50, 75, 100])
    labels = [
        'Short\n({:.0f}–{:.0f})'.format(edges[0], edges[1]),
        'Medium\n({:.0f}–{:.0f})'.format(edges[1], edges[2]),
        'Long\n({:.0f}–{:.0f})'.format(edges[2], edges[3]),
        'Very long\n({:.0f}–{:.0f})'.format(edges[3], edges[4]),
    ]
    short_labels = [
        'Short ({:.0f}–{:.0f})'.format(edges[0], edges[1]),
        'Medium ({:.0f}–{:.0f})'.format(edges[1], edges[2]),
        'Long ({:.0f}–{:.0f})'.format(edges[2], edges[3]),
        'Very long ({:.0f}–{:.0f})'.format(edges[3], edges[4]),
    ]
    return edges, labels, short_labels


def panel_f(ax, sel_kv, ttft, input_tokens, sel_waiting):
    # Create 2D grid: input_length_bucket x load_bucket
    # Cell value: Spearman rho(kv_hit, ttft)
    input_edges, input_labels, _ = _quartile_edges_and_labels(input_tokens)
    n_inp = len(input_edges) - 1

    load_edges = [0, 1, 3, 16]  # low, medium, high
    load_labels = ['Low\n(wait 0)', 'Med\n(wait 1-2)', 'High\n(wait 3+)']

    rho_grid = np.full((len(load_edges) - 1, n_inp), np.nan)
    n_grid = np.full_like(rho_grid, np.nan)

    for li in range(len(load_edges) - 1):
        for ii in range(n_inp):
            load_mask = ((sel_waiting >= load_edges[li]) &
                         (sel_waiting < load_edges[li + 1]))
            if ii < n_inp - 1:
                inp_mask = ((input_tokens >= input_edges[ii]) &
                            (input_tokens < input_edges[ii + 1]))
            else:
                inp_mask = ((input_tokens >= input_edges[ii]) &
                            (input_tokens <= input_edges[ii + 1]))
            combined = load_mask & inp_mask
            n = combined.sum()
            n_grid[li, ii] = n
            if n > 30:
                rho, _ = stats.spearmanr(sel_kv[combined], ttft[combined])
                rho_grid[li, ii] = rho

    im = ax.imshow(rho_grid, cmap='RdBu_r', vmin=-0.5, vmax=0.3,
                   aspect='auto', origin='lower')

    # Annotate cells
    for li in range(rho_grid.shape[0]):
        for ii in range(rho_grid.shape[1]):
            rho_val = rho_grid[li, ii]
            n_val = int(n_grid[li, ii])
            if not np.isnan(rho_val):
                text_color = 'white' if abs(rho_val) > 0.25 else 'black'
                ax.text(ii, li, '$\\rho$={:+.2f}\nn={}'.format(rho_val, n_val),
                        ha='center', va='center', fontsize=7, color=text_color,
                        fontweight='bold')

    # X-axis: quartile labels (shorter version for heatmap)
    x_tick_labels = [
        'Short', 'Medium', 'Long', 'Very long'
    ]
    ax.set_xticks(range(n_inp))
    ax.set_xticklabels(x_tick_labels, fontsize=7)
    ax.set_yticks(range(len(load_labels)))
    ax.set_yticklabels(load_labels)
    ax.set_xlabel('Request length')
    ax.set_ylabel('System load')
    ax.set_title('(g)  3-way interaction: KV hit effect\ndepends on length AND load')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.08)
    cbar.set_label('Spearman $\\rho$(KV hit, TTFT)', fontsize=7)
    cbar.ax.tick_params(labelsize=6)


# ---------------------------------------------------------------------------
# Panel (g): Load modulation broken down by input length groups
# ---------------------------------------------------------------------------
def panel_g(ax, sel_kv, ttft, sel_waiting, input_tokens):
    # Three fixed input length buckets
    input_edges = [1, 1000, 5000, 30000]
    short_labels = ['Short (1–1000)', 'Medium (1001–5000)', 'Long (5001–30000)']
    input_colors = [C_BLUE, C_ORANGE, C_RED]

    # Load buckets (same as panel d)
    wait_bins = [0, 1, 5, 100]
    wait_labels = ['0-0', '1-4', '5+']
    load_labels = []
    valid_load_bins = []

    # First pass: find which load bins have enough data
    for j in range(len(wait_bins) - 1):
        mask = (sel_waiting >= wait_bins[j]) & (sel_waiting < wait_bins[j + 1])
        if mask.sum() > 50:
            load_labels.append(wait_labels[j])
            valid_load_bins.append((wait_bins[j], wait_bins[j + 1]))

    n_load = len(valid_load_bins)
    n_len = len(short_labels)
    x = np.arange(n_load)
    total_w = 0.8
    bar_w = total_w / n_len

    for li, (il_label, il_color) in enumerate(zip(short_labels, input_colors)):
        lo_inp = input_edges[li]
        hi_inp = input_edges[li + 1]
        if li < n_len - 1:
            inp_mask = (input_tokens >= lo_inp) & (input_tokens < hi_inp)
        else:
            inp_mask = (input_tokens >= lo_inp) & (input_tokens <= hi_inp)

        rho_vals = []
        n_vals = []
        for wlo, whi in valid_load_bins:
            load_mask = (sel_waiting >= wlo) & (sel_waiting < whi)
            combined = inp_mask & load_mask
            n = combined.sum()
            n_vals.append(n)
            if n > 30:
                rho, _ = stats.spearmanr(sel_kv[combined], ttft[combined])
                rho_vals.append(rho)
            else:
                rho_vals.append(0)

        offset = (li - (n_len - 1) / 2) * bar_w
        bars = ax.bar(x + offset, rho_vals, bar_w * 0.9, label=il_label,
                       color=il_color, alpha=0.8, edgecolor='white', linewidth=0.4)

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axhline(y=-0.1, color='gray', linewidth=0.3, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(load_labels, rotation=45, ha='right')
    ax.set_xlabel('Waiting requests (system load)')
    ax.set_ylabel('Spearman $\\rho$(KV hit, TTFT)')
    ax.set_ylim(None, 0.15)
    ax.set_title('(e)  KV hit effect by load AND request length')
    ax.legend(loc='upper left', framealpha=0.9, fontsize=10, ncol=1)

    _draw_vertical_arrow(ax)


# ---------------------------------------------------------------------------
# Panel (i): Same as (h) but transposed: x = input length, bars = load groups
# ---------------------------------------------------------------------------
def panel_i(ax, sel_kv, ttft, sel_waiting, input_tokens):
    # Input length quartiles on x-axis
    input_edges, _, short_labels = _quartile_edges_and_labels(input_tokens)
    n_len = len(short_labels)

    # Load buckets as grouped bars
    load_bins = [(0, 1, 'Wait 0'), (1, 3, 'Wait 1-2'), (3, 8, 'Wait 3-7'),
                 (8, 16, 'Wait 8+')]
    load_colors = [C_BLUE, C_GREEN, C_ORANGE, C_RED]

    # Filter to bins with enough data
    valid_loads = []
    for wlo, whi, wlabel in load_bins:
        if ((sel_waiting >= wlo) & (sel_waiting < whi)).sum() > 50:
            valid_loads.append((wlo, whi, wlabel))

    n_load = len(valid_loads)
    x = np.arange(n_len)
    total_w = 0.8
    bar_w = total_w / n_load

    for li, (wlo, whi, wlabel) in enumerate(valid_loads):
        load_mask = (sel_waiting >= wlo) & (sel_waiting < whi)

        rho_vals = []
        n_vals = []
        for ii in range(n_len):
            lo_inp = input_edges[ii]
            hi_inp = input_edges[ii + 1]
            if ii < n_len - 1:
                inp_mask = (input_tokens >= lo_inp) & (input_tokens < hi_inp)
            else:
                inp_mask = (input_tokens >= lo_inp) & (input_tokens <= hi_inp)

            combined = load_mask & inp_mask
            n = combined.sum()
            n_vals.append(n)
            if n > 30:
                rho, _ = stats.spearmanr(sel_kv[combined], ttft[combined])
                rho_vals.append(rho)
            else:
                rho_vals.append(0)

        offset = (li - (n_load - 1) / 2) * bar_w
        bars = ax.bar(x + offset, rho_vals, bar_w * 0.9, label=wlabel,
                       color=load_colors[li], alpha=0.8, edgecolor='white',
                       linewidth=0.4)

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axhline(y=-0.1, color='gray', linewidth=0.3, linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=45, ha='right')
    ax.set_xlabel('Input length group')
    ax.set_ylabel('Spearman $\\rho$(KV hit, TTFT)')
    ax.set_title('(f)  Cache effect by request length AND load')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.30), framealpha=0.9, fontsize=10, ncol=1)

    _draw_vertical_arrow(ax)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    csv_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else csv_path.replace('.csv', '') + '_kv_analysis.pdf'

    print(f'Loading data from {csv_path} ...')
    (df, ttft, input_tokens, sel_kv, sel_waiting, sel_prefill,
     sel_running, sel_inflight_prefill, sel_gpu_kv) = load_and_prepare(csv_path)
    print(f'  {len(df)} samples, {len(df["selected_pod"].unique())} pods')

    # Create figure — 3 rows x 2 cols (5 panels)
    fig = plt.figure(figsize=(9, 13))
    gs = gridspec.GridSpec(3, 2, hspace=0.65, wspace=0.42,
                           left=0.09, right=0.95, top=0.97, bottom=0.03)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, 0])

    print('  Plotting panel (a): KV hit vs TTFT ...')
    panel_a(ax_a, sel_kv, ttft)

    print('  Plotting panel (b): Load vs TTFT ...')
    panel_b_load(ax_b, sel_waiting, ttft)

    print('  Plotting panel (c): Controlled sign reversal ...')
    panel_d(ax_c, sel_kv, ttft, input_tokens, sel_waiting)

    print('  Plotting panel (d): Load modulation ...')
    panel_e(ax_d, sel_kv, ttft, sel_waiting)

    print('  Plotting panel (e): Load × length grouped bars ...')
    panel_g(ax_e, sel_kv, ttft, sel_waiting, input_tokens)

    # Save
    fig.savefig(output_path, bbox_inches='tight')
    print(f'\nSaved to {output_path}')

    # Also save PNG for quick viewing
    png_path = output_path.replace('.pdf', '.png')
    fig.savefig(png_path, bbox_inches='tight')
    print(f'Saved to {png_path}')

    # --- Separate PDFs for paper figures (no titles, larger fonts) ---
    base_dir = os.path.dirname(output_path)

    def _enlarge_ax(ax, label_fs=16, tick_fs=14, legend_fs=14):
        """Bump font sizes on a standalone axis for paper readability."""
        ax.xaxis.label.set_fontsize(label_fs)
        ax.yaxis.label.set_fontsize(label_fs)
        ax.tick_params(axis='both', labelsize=tick_fs)
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                t.set_fontsize(legend_fs)
        # Enlarge the "More KV cache hit benefit" arrow text if present
        for child in ax.get_children():
            if hasattr(child, 'get_text') and 'KV cache' in str(child.get_text()):
                child.set_fontsize(14)
        # Enlarge arrow patch (FancyArrow) if present
        from matplotlib.patches import FancyArrow
        for p in ax.patches:
            if isinstance(p, FancyArrow):
                # Remove old arrow, redraw bigger
                p.set_alpha(0.30)

    def _redraw_arrow_bigger(ax):
        """Replace the arrow drawn by _draw_vertical_arrow with a bigger version."""
        from matplotlib.patches import FancyArrow
        # Remove existing arrows and arrow text
        to_remove = []
        for child in ax.get_children():
            if isinstance(child, FancyArrow):
                to_remove.append(child)
            elif hasattr(child, 'get_text') and 'KV cache' in str(child.get_text()):
                to_remove.append(child)
        for c in to_remove:
            c.remove()
        # Redraw with bigger sizes — thicker arrow, 3-line text
        x_frac, y_top, y_bot = 0.95, 0.50, 0.05
        width = 0.22
        arrow = FancyArrow(
            x_frac, y_top, 0, -(y_top - y_bot),
            width=width, head_width=width * 1.8, head_length=0.04,
            transform=ax.transAxes, fc=C_GREY, ec='white',
            alpha=0.25, zorder=10, linewidth=0.5)
        ax.add_patch(arrow)
        ax.text(x_frac, (y_top + y_bot) / 2 + 0.02, 'More\nKV cache\nhit benefit',
                transform=ax.transAxes, fontsize=14, color='#444444',
                ha='center', va='center', rotation=90, fontweight='bold')

    # (a) KV hit vs TTFT
    fig_a, ax_a_sep = plt.subplots(figsize=(4.5, 3.5))
    panel_a(ax_a_sep, sel_kv, ttft)
    ax_a_sep.set_title('')
    _enlarge_ax(ax_a_sep)
    path_a = os.path.join(base_dir, 'fig_kv_hit_vs_ttft.pdf')
    fig_a.savefig(path_a, bbox_inches='tight')
    plt.close(fig_a)
    print(f'Saved separate: {path_a}')

    # (b) System load vs TTFT
    fig_b, ax_b_sep = plt.subplots(figsize=(4.5, 3.5))
    panel_b_load(ax_b_sep, sel_waiting, ttft)
    ax_b_sep.set_title('')
    _enlarge_ax(ax_b_sep)
    path_b = os.path.join(base_dir, 'fig_system_load_vs_ttft.pdf')
    fig_b.savefig(path_b, bbox_inches='tight')
    plt.close(fig_b)
    print(f'Saved separate: {path_b}')

    # (e) KV hit effect by load AND request length
    fig_e, ax_e_sep = plt.subplots(figsize=(4.5, 3.5))
    panel_g(ax_e_sep, sel_kv, ttft, sel_waiting, input_tokens)
    ax_e_sep.set_title('')
    _enlarge_ax(ax_e_sep)
    _redraw_arrow_bigger(ax_e_sep)
    path_e = os.path.join(base_dir, 'fig_kv_hit_by_load_and_length.pdf')
    fig_e.savefig(path_e, bbox_inches='tight')
    plt.close(fig_e)
    print(f'Saved separate: {path_e}')

    # Print summary statistics used in the paper paragraph
    print('\n' + '=' * 70)
    print('SUMMARY STATISTICS FOR PAPER')
    print('=' * 70)
    r_p, _ = stats.pearsonr(sel_kv, ttft)
    r_s, _ = stats.spearmanr(sel_kv, ttft)
    print(f'Pearson r(kv, ttft)  = {r_p:+.4f}')
    print(f'Spearman rho(kv, ttft) = {r_s:+.4f}')

    r_conf, _ = stats.spearmanr(sel_kv, input_tokens)
    print(f'Confound: Spearman rho(kv, input_tokens) = {r_conf:+.4f}')

    for name, mask in [('Low load (wait=0)', sel_waiting == 0),
                       ('High load (wait>=2)', sel_waiting >= 2)]:
        rho, _ = stats.spearmanr(sel_kv[mask], ttft[mask])
        print(f'{name}: Spearman rho = {rho:+.4f}, N = {mask.sum()}')

    # Partial correlation
    X = np.column_stack([
        input_tokens, df['total_tokens'].values,
        sel_prefill, sel_running, sel_waiting, sel_inflight_prefill, sel_gpu_kv])
    X_aug = np.hstack([X, np.ones((len(X), 1))])
    kv_resid = sel_kv - X_aug @ np.linalg.lstsq(X_aug, sel_kv, rcond=None)[0]
    ttft_resid = ttft - X_aug @ np.linalg.lstsq(X_aug, ttft, rcond=None)[0]
    partial_r, partial_p = stats.pearsonr(kv_resid, ttft_resid)
    print(f'Partial r (controlling 7 confounds) = {partial_r:+.4f} (p={partial_p:.2e})')


if __name__ == '__main__':
    main()
