#!/usr/bin/env python3
"""
Adaptation ablation: continuous learning vs frozen router under workload shift.

4-panel vertically-stacked figure (shared x-axis):
  (a) Sliding-window mean TTFT
  (b) KV Hit Ratio — routed inst. (solid) vs system avg. (dashed)
  (c) Prefill Tokens — routed inst. (solid) vs system avg. (dashed)
  (d) Waiting Requests — routed inst. (solid) vs system avg. (dashed)

Usage:
  cd /path/to/without_bitsandbytes
  python plot_adaptation_ablation.py .
  python plot_adaptation_ablation.py . --output my_figure.pdf
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

DEFAULT_CSV_NAME = "filtered-aibrix-gateway-plugins-processed.log.csv"
LOG_NAME = "all-routing-agent-service.log.txt"
GATEWAY_LOG_NAME = "all-aibrix-gateway-plugins.log.txt"

# ── Experiment identification ─────────────────────────────────
UNLIMITED_KEY = "maxnumtrains_unlimited"
LIMITED_KEY = "maxnumtrains_7"
PREFIX_CACHE_KEY = "prefix_cache_1"

LABELS = {
    UNLIMITED_KEY: "QS",
    LIMITED_KEY: "QS \n(mid-frozen)",
    PREFIX_CACHE_KEY: "Prefix Cache",
}

COLORS = {
    UNLIMITED_KEY: "#ff7f00",  # orange
    LIMITED_KEY: "#4a90d9",    # icy blue
    PREFIX_CACHE_KEY: "#2ca02c",  # green
}

SORT_ORDER = {LIMITED_KEY: 0, PREFIX_CACHE_KEY: 1, UNLIMITED_KEY: 2}

BAR_LABELS = {
    UNLIMITED_KEY: "QS",
    LIMITED_KEY: "QS\n(mid-frozen)",
    PREFIX_CACHE_KEY: "Pfx Cache",
}


# ── Discovery ─────────────────────────────────────────────────
def discover_experiments(
    base_dir: str, csv_name: str
) -> List[Tuple[str, str, str]]:
    experiments = []
    for name in sorted(os.listdir(base_dir)):
        subdir = os.path.join(base_dir, name)
        if not os.path.isdir(subdir):
            continue
        csv_path = os.path.join(subdir, csv_name)
        if not os.path.exists(csv_path):
            continue
        for key in [UNLIMITED_KEY, LIMITED_KEY, PREFIX_CACHE_KEY]:
            if key in name:
                experiments.append((name, key, LABELS[key]))
                break

    if len(experiments) < 2:
        print(f"Need at least 2 experiment dirs with {csv_name}")
        print(f"  found: {[e[0] for e in experiments]}")
        sys.exit(1)

    experiments.sort(key=lambda t: SORT_ORDER.get(t[1], 99))
    return experiments


# ── Data loading ──────────────────────────────────────────────
def load_experiment(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.sort_values("request_id").reset_index(drop=True)
    return df


def extract_selected_pod_metric(df: pd.DataFrame, suffix: str) -> np.ndarray:
    vals = np.empty(len(df), dtype=float)
    for i, (_, row) in enumerate(df.iterrows()):
        col = f'{row["selected_pod"]}-{suffix}'
        vals[i] = row[col]
    return vals


def extract_system_avg_metric(df: pd.DataFrame, suffix: str) -> np.ndarray:
    pod_cols = sorted([c for c in df.columns if c.endswith(f"-{suffix}")])
    return df[pod_cols].mean(axis=1).to_numpy(dtype=float)


def parse_predictions(log_path: str) -> dict:
    """Parse predicted rewards from routing agent log."""
    preds = {}
    pattern = re.compile(
        r"inference for (\d+): pod=(\d+), predicted_rewards=\[([^\]]+)\], "
        r"chosen_pod_predicted_reward=([0-9e.\-]+)"
    )
    with open(log_path) as f:
        buf = ""
        for line in f:
            if "Neural CB inference for" in line:
                if buf:
                    m = pattern.search(buf)
                    if m:
                        preds[int(m.group(1))] = float(m.group(4))
                buf = line.strip()
            elif buf and (line.startswith(" ") or line.startswith("\t")):
                buf += " " + line.strip()
            else:
                if buf:
                    m = pattern.search(buf)
                    if m:
                        preds[int(m.group(1))] = float(m.group(4))
                    buf = ""
        if buf:
            m = pattern.search(buf)
            if m:
                preds[int(m.group(1))] = float(m.group(4))
    return preds


def ttft_to_reward(ttft: np.ndarray) -> np.ndarray:
    return -ttft / 1000.0


def sliding_mae(pred, actual, window, step):
    """Sliding-window MAE between predicted and actual reward."""
    n = len(pred)
    if n < window:
        window = n
    centers, vals = [], []
    for start in range(0, n - window + 1, step):
        end = start + window
        vals.append(float(np.mean(np.abs(pred[start:end] - actual[start:end]))))
        centers.append(start + window // 2)
    return np.array(centers), np.array(vals)


def parse_preemptions(gateway_log_path: str, request_ids: list) -> dict:
    """Parse per-request total preemptions (sum across pods) from gateway log.

    vllmNumPreemptions is a cumulative counter per pod. We extract the
    cumulative sum across all pods per request, then diff consecutive
    values to get per-request new preemptions.
    Returns {request_id: new_preemptions} or empty dict if field not found.
    """
    import json as _json
    pattern = re.compile(
        r"requestID@(\d+).*?vllmNumPreemptions@(\{[^}]*\})"
    )
    cum_by_req = {}
    with open(gateway_log_path) as f:
        for line in f:
            if "vllmNumPreemptions" not in line:
                continue
            m = pattern.search(line)
            if not m:
                continue
            rid = int(m.group(1))
            try:
                pod_counts = _json.loads(m.group(2))
                cum_by_req[rid] = sum(pod_counts.values())
            except (_json.JSONDecodeError, AttributeError):
                continue
    if not cum_by_req:
        return {}
    # Convert cumulative to per-request delta
    sorted_rids = sorted(cum_by_req.keys())
    deltas = {}
    prev = 0
    for rid in sorted_rids:
        cur = cum_by_req[rid]
        deltas[rid] = max(0, cur - prev)
        prev = cur
    return deltas


def sliding_window(
    values: np.ndarray, window: int, step: int
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(values)
    if n < window:
        window = n
    centers = []
    means = []
    for start in range(0, n - window + 1, step):
        centers.append(start + window // 2)
        means.append(float(np.mean(values[start : start + window])))
    return np.array(centers), np.array(means)


def _interp_common(c1, m1, c2, m2, n=500):
    x_min = max(c1[0], c2[0])
    x_max = min(c1[-1], c2[-1])
    x = np.linspace(x_min, x_max, n)
    return x, np.interp(x, c1, m1), np.interp(x, c2, m2)


def _draw_phase_bg(ax, midpoint, total_requests):
    ax.axvspan(0, midpoint, color="#e8f0fe", alpha=0.45, zorder=0)
    ax.axvspan(midpoint, total_requests, color="#fce8e6", alpha=0.45, zorder=0)
    ax.axvline(midpoint, color="black", linestyle="--", linewidth=1.2,
               alpha=0.7, zorder=3)


def _plot_sel_vs_avg_panel(ax, data, metric_sel, metric_avg, midpoint,
                           total_requests, ylabel, title):
    """Plot a panel with routed-pod (solid) vs system-avg (dashed) for both methods."""
    _draw_phase_bg(ax, midpoint, total_requests)

    all_keys = [k for k in [LIMITED_KEY, PREFIX_CACHE_KEY, UNLIMITED_KEY] if k in data]
    for key in all_keys:
        d = data[key]
        color = d["color"]
        # Routed pod (solid)
        ax.plot(d[f"{metric_sel}_centers"], d[f"{metric_sel}_means"],
                linewidth=2.0, color=color, linestyle="-", zorder=4)
        # System average (dashed with x markers)
        avg_centers = d[f"{metric_avg}_centers"]
        avg_means = d[f"{metric_avg}_means"]
        marker_every = max(1, len(avg_centers) // 20)
        ax.plot(avg_centers, avg_means,
                linewidth=1.5, color=color, linestyle="--", alpha=0.7,
                marker="x", markersize=6, markevery=marker_every,
                markeredgewidth=1.5, zorder=4)


    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(alpha=0.2)


# ── Bar chart helper ──────────────────────────────────────────
def _plot_bar_panel(ax, all_keys, data, ttft_slice_fn, title):
    """Grouped bar chart: avg on left y-axis, p99 on right y-axis."""
    n_keys = len(all_keys)
    bar_width = 0.35
    x = np.arange(n_keys)

    avgs, p99s = [], []
    colors = []
    labels = []
    for key in all_keys:
        d = data[key]
        ttft = ttft_slice_fn(d)
        avgs.append(np.mean(ttft))
        p99s.append(np.percentile(ttft, 99))
        colors.append(d["color"])
        labels.append(BAR_LABELS[key])

    # Left axis: avg
    bars_avg = ax.bar(x - bar_width / 2, avgs, bar_width,
                      color=colors, edgecolor="black", linewidth=0.6, zorder=3)
    for bar in bars_avg:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h,
                f"{h:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Avg TTFT (ms)")

    # Right axis: p99
    ax2 = ax.twinx()
    bars_p99 = ax2.bar(x + bar_width / 2, p99s, bar_width,
                       color=colors, edgecolor="black", linewidth=0.6,
                       alpha=0.55, hatch="//", zorder=3)
    for bar in bars_p99:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h,
                 f"{h:.0f}", ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("P99 TTFT (ms)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=15)
    ax.grid(axis="y", alpha=0.2)
    return ax2


INPUT_LEN_BINS = [
    ("Short\n(0-1500)", 0, 1500),
    ("Med\n(1500-5000)", 1500, 5000),
    ("Long\n(5000+)", 5000, float("inf")),
]


def _plot_bar_by_input_len(ax, all_keys, data, row_slice_fn, title):
    """Grouped bar chart: avg (left y) + p99 (right y) per input-length bin."""
    n_bins = len(INPUT_LEN_BINS)
    n_keys = len(all_keys)
    bar_width = 0.8 / (n_keys * 2)
    x = np.arange(n_bins)

    ax2 = ax.twinx()

    for ki, key in enumerate(all_keys):
        d = data[key]
        ttft, inp = row_slice_fn(d)
        color = d["color"]
        for bi, (_, lo, hi) in enumerate(INPUT_LEN_BINS):
            mask = (inp >= lo) & (inp < hi)
            subset = ttft[mask]
            if len(subset) == 0:
                continue
            avg_val = np.mean(subset)
            p99_val = np.percentile(subset, 99)
            group_offset = (ki - (n_keys - 1) / 2) * (bar_width * 2.2)
            # Avg on left axis
            ax.bar(bi + group_offset - bar_width / 2, avg_val, bar_width,
                   color=color, edgecolor="black", linewidth=0.4, zorder=3)
            ax.text(bi + group_offset - bar_width / 2, avg_val,
                    f"{avg_val:.0f}", ha="center", va="bottom", fontsize=7)
            # P99 on right axis
            ax2.bar(bi + group_offset + bar_width / 2, p99_val, bar_width,
                    color=color, edgecolor="black", linewidth=0.4,
                    alpha=0.55, hatch="//", zorder=3)
            ax2.text(bi + group_offset + bar_width / 2, p99_val,
                     f"{p99_val:.0f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in INPUT_LEN_BINS], fontsize=10)
    ax.set_ylabel("Avg TTFT (ms)")
    ax2.set_ylabel("P99 TTFT (ms)")
    ax.set_title(title, loc="left", fontweight="bold", fontsize=13)
    ax.grid(axis="y", alpha=0.2)
    return ax2


def _sliding_routing_metrics(pods_arr: np.ndarray, window: int, step: int):
    """Sliding-window CV, Jain's FI, and Norm. Entropy over pod assignments."""
    n = len(pods_arr)
    if n < window:
        window = n
    unique_pods = np.unique(pods_arr)
    n_pods = len(unique_pods)
    centers, cvs, jains, entropies = [], [], [], []
    for start in range(0, n - window + 1, step):
        chunk = pods_arr[start:start + window]
        # Count per pod (include all pods even if 0)
        counts = np.array([np.sum(chunk == p) for p in unique_pods], dtype=float)
        centers.append(start + window // 2)
        # CV
        mean_c = np.mean(counts)
        cvs.append(float(np.std(counts) / mean_c * 100) if mean_c > 0 else 0.0)
        # Jain's FI
        sum_sq = np.sum(counts ** 2)
        jains.append(float(np.sum(counts) ** 2 / (n_pods * sum_sq)) if sum_sq > 0 else 0.0)
        # Normalized entropy
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        h = -np.sum(probs * np.log(probs))
        entropies.append(float(h / np.log(n_pods)) if n_pods > 1 else 1.0)
    return np.array(centers), np.array(cvs), np.array(jains), np.array(entropies)


def _routing_cv(selected_pods: pd.Series) -> float:
    """Coefficient of variation of per-pod request counts (0 = perfectly uniform)."""
    counts = selected_pods.value_counts().values.astype(float)
    return float(np.std(counts) / np.mean(counts)) if np.mean(counts) > 0 else 0.0


def _plot_routing_cv_panel(ax, all_keys, data):
    """Grouped bar: routing CV for Overall / Phase1 / Phase2."""
    phases = ["Overall", "Phase 1", "Phase 2"]
    n_phases = len(phases)
    n_keys = len(all_keys)
    bar_width = 0.8 / n_keys
    x = np.arange(n_phases)

    for i, key in enumerate(all_keys):
        d = data[key]
        pods = d["selected_pods"]
        half = d["half"]
        cvs = [
            _routing_cv(pods),
            _routing_cv(pods.iloc[:half]),
            _routing_cv(pods.iloc[half:]),
        ]
        offset = (i - (n_keys - 1) / 2) * bar_width
        bars = ax.bar(x + offset, [v * 100 for v in cvs], bar_width,
                      color=d["color"], edgecolor="black", linewidth=0.6,
                      zorder=3)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(phases, fontsize=11)
    ax.set_ylabel("CV (%)")
    ax.set_title("Routing Balance: CV\n(lower = more uniform)",
                 loc="left", fontweight="bold", fontsize=13)
    ax.grid(axis="y", alpha=0.2)


def _jain_fairness(selected_pods: pd.Series) -> float:
    """Jain's Fairness Index: (sum(x))^2 / (n * sum(x^2)). 1.0 = perfectly uniform."""
    counts = selected_pods.value_counts().values.astype(float)
    n = len(counts)
    if n == 0 or np.sum(counts ** 2) == 0:
        return 0.0
    return float(np.sum(counts) ** 2 / (n * np.sum(counts ** 2)))


def _norm_entropy(selected_pods: pd.Series) -> float:
    """Normalized entropy: H(counts) / log(n_pods). 1.0 = perfectly uniform."""
    counts = selected_pods.value_counts().values.astype(float)
    n = len(counts)
    if n <= 1:
        return 1.0
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    h = -np.sum(probs * np.log(probs))
    return float(h / np.log(n))


def _plot_routing_jain_panel(ax, all_keys, data):
    """Grouped bar: Jain's Fairness Index for Overall / Phase1 / Phase2."""
    phases = ["Overall", "Phase 1", "Phase 2"]
    n_phases = len(phases)
    n_keys = len(all_keys)
    bar_width = 0.8 / n_keys
    x = np.arange(n_phases)

    for i, key in enumerate(all_keys):
        d = data[key]
        pods = d["selected_pods"]
        half = d["half"]
        vals = [
            _jain_fairness(pods),
            _jain_fairness(pods.iloc[:half]),
            _jain_fairness(pods.iloc[half:]),
        ]
        offset = (i - (n_keys - 1) / 2) * bar_width
        bars = ax.bar(x + offset, vals, bar_width,
                      color=d["color"], edgecolor="black", linewidth=0.6,
                      zorder=3)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h,
                    f"{h:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(phases, fontsize=11)
    ax.set_ylabel("Jain's FI")
    ax.set_ylim(0.99, 1.001)
    ax.set_title("Routing Balance: Jain's FI\n(1.0 = perfectly uniform)",
                 loc="left", fontweight="bold", fontsize=13)
    ax.grid(axis="y", alpha=0.2)


def _plot_routing_entropy_panel(ax, all_keys, data):
    """Grouped bar: Normalized entropy for Overall / Phase1 / Phase2."""
    phases = ["Overall", "Phase 1", "Phase 2"]
    n_phases = len(phases)
    n_keys = len(all_keys)
    bar_width = 0.8 / n_keys
    x = np.arange(n_phases)

    for i, key in enumerate(all_keys):
        d = data[key]
        pods = d["selected_pods"]
        half = d["half"]
        vals = [
            _norm_entropy(pods),
            _norm_entropy(pods.iloc[:half]),
            _norm_entropy(pods.iloc[half:]),
        ]
        offset = (i - (n_keys - 1) / 2) * bar_width
        bars = ax.bar(x + offset, vals, bar_width,
                      color=d["color"], edgecolor="black", linewidth=0.6,
                      zorder=3)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h,
                    f"{h:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(phases, fontsize=11)
    ax.set_ylabel("Norm. Entropy")
    ax.set_ylim(0.999, 1.0005)
    ax.set_title("Routing Balance: Entropy\n(1.0 = perfectly uniform)",
                 loc="left", fontweight="bold", fontsize=13)
    ax.grid(axis="y", alpha=0.2)


# ── Convergence helpers ───────────────────────────────────────
def _find_crossover(cont_centers, cont_means, pfx_centers, pfx_means, midpoint):
    """Find first request index in Phase 2 where continuous <= prefix cache."""
    x, cont_i, pfx_i = _interp_common(cont_centers, cont_means,
                                        pfx_centers, pfx_means, n=2000)
    for i, xi in enumerate(x):
        if xi >= midpoint and cont_i[i] <= pfx_i[i]:
            return int(xi)
    return None


def _find_self_convergence(centers, means, midpoint, threshold_pct=10):
    """Find when continuous learner's TTFT settles to within threshold_pct%
    of its eventual steady-state (last 25% of Phase 2 data)."""
    # Compute steady-state from last 25% of data
    p2_mask = centers >= midpoint
    p2_centers = centers[p2_mask]
    p2_means = means[p2_mask]
    if len(p2_means) == 0:
        return None
    last_quarter = p2_means[len(p2_means) * 3 // 4:]
    if len(last_quarter) == 0:
        return None
    steady_state = float(np.mean(last_quarter))
    # Find first point in Phase 2 within threshold of steady state
    thresh = steady_state * (1 + threshold_pct / 100.0)
    for i, (c, m) in enumerate(zip(p2_centers, p2_means)):
        if m <= thresh:
            return int(c)
    return None


def _compute_gap_and_regret(data, midpoint, rps=7.8):
    """Compute instantaneous gap and cumulative regret (cont vs pfx) in Phase 2."""
    cont = data[UNLIMITED_KEY]
    pfx = data[PREFIX_CACHE_KEY]

    # Instantaneous gap: sliding-window TTFT_cont - TTFT_pfx on common x
    x, cont_i, pfx_i = _interp_common(
        cont["ttft_centers"], cont["ttft_means"],
        pfx["ttft_centers"], pfx["ttft_means"], n=2000,
    )
    gap = cont_i - pfx_i

    # Cumulative regret on raw per-request TTFT from Phase 2
    cont_raw_p2 = cont["ttft_raw"][cont["half"]:]
    pfx_raw_p2 = pfx["ttft_raw"][pfx["half"]:]
    n_common = min(len(cont_raw_p2), len(pfx_raw_p2))
    per_req_diff = cont_raw_p2[:n_common] - pfx_raw_p2[:n_common]
    cum_regret = np.cumsum(per_req_diff) / 1000.0  # convert ms -> seconds
    regret_x = np.arange(n_common) + midpoint  # absolute request index

    return x, gap, regret_x, cum_regret


# ── Plotting ──────────────────────────────────────────────────
def plot_figure(
    data: dict,
    midpoint: int,
    total_requests: int,
    window: int,
    step: int,
    out_path: str,
    rps: float = 7.8,
) -> None:
    plt.rcParams.update({
        "font.size": 16,
        "font.family": "sans-serif",
        "axes.titlesize": 19,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 15,
    })

    has_convergence = (UNLIMITED_KEY in data and PREFIX_CACHE_KEY in data)
    has_mae = any(d.get("mae_centers") is not None for d in data.values())
    has_preempt = any(d.get("preempt_centers") is not None for d in data.values())

    # Row layout: TTFT, KV, Wait, [Preempt], GPU, [MAE], Entropy_ts, TTFT Bars
    ts_rows = 4 + int(has_preempt) + int(has_mae) + 1  # +1 for sliding entropy
    n_rows = ts_rows + 1  # +1 for TTFT bars
    height_ratios = [1] * ts_rows + [1.2]

    fig = plt.figure(figsize=(15, 3.8 * n_rows))
    gs = fig.add_gridspec(
        n_rows, 1, height_ratios=height_ratios,
        hspace=0.55,
    )

    # Timeseries panels (full width, shared x)
    row = 0
    ax_ttft = fig.add_subplot(gs[row, 0]); row += 1
    ax_kv = fig.add_subplot(gs[row, 0], sharex=ax_ttft); row += 1
    ax_wait = fig.add_subplot(gs[row, 0], sharex=ax_ttft); row += 1

    ax_preempt = None
    if has_preempt:
        ax_preempt = fig.add_subplot(gs[row, 0], sharex=ax_ttft); row += 1

    ax_gpu = fig.add_subplot(gs[row, 0], sharex=ax_ttft); row += 1

    ax_mae = None
    if has_mae:
        ax_mae = fig.add_subplot(gs[row, 0], sharex=ax_ttft); row += 1

    # Sliding routing balance timeseries (entropy only)
    ax_entropy_ts = fig.add_subplot(gs[row, 0], sharex=ax_ttft); row += 1

    # Combined TTFT bar chart
    ax_bars = fig.add_subplot(gs[row, 0])

    all_keys = [k for k in [LIMITED_KEY, PREFIX_CACHE_KEY, UNLIMITED_KEY] if k in data]

    # ── Panel 1: TTFT timeseries ──
    _draw_phase_bg(ax_ttft, midpoint, total_requests)
    for key in all_keys:
        d = data[key]
        ax_ttft.plot(d["ttft_centers"], d["ttft_means"],
                     linewidth=2.0, color=d["color"], zorder=4)

    ax_ttft.set_ylabel("Mean TTFT (ms)")
    ax_ttft.set_title("Time-to-First-Token", loc="left", fontweight="bold")
    ax_ttft.grid(alpha=0.2)

    # Phase annotations
    ax_ttft.text(
        midpoint * 0.5, ax_ttft.get_ylim()[1] * 0.92,
        "0% prefix sharing", ha="center", fontsize=12,
        fontstyle="italic", color="#444444",
    )
    ax_ttft.text(
        midpoint + (total_requests - midpoint) * 0.5,
        ax_ttft.get_ylim()[1] * 0.92,
        "50% prefix sharing", ha="center", fontsize=12,
        fontstyle="italic", color="#444444",
    )
    ax_ttft.annotate(
        "Workload\nshift",
        xy=(midpoint, ax_ttft.get_ylim()[1] * 0.65),
        xytext=(midpoint + total_requests * 0.06, ax_ttft.get_ylim()[1] * 0.78),
        fontsize=11, ha="left",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
    )

    # ── Convergence annotation on TTFT panel ──
    if UNLIMITED_KEY in data:
        cont_d = data[UNLIMITED_KEY]
        crossover = None
        # Method 1: crossover vs prefix_cache if available
        if has_convergence:
            pfx_d = data[PREFIX_CACHE_KEY]
            crossover = _find_crossover(
                cont_d["ttft_centers"], cont_d["ttft_means"],
                pfx_d["ttft_centers"], pfx_d["ttft_means"],
                midpoint,
            )
        # Method 2: self-convergence (TTFT settles within 10% of steady-state)
        if crossover is None:
            crossover = _find_self_convergence(
                cont_d["ttft_centers"], cont_d["ttft_means"], midpoint,
            )
        if crossover is not None:
            adapt_reqs = crossover - midpoint
            adapt_time = adapt_reqs / rps
            # Shade adaptation window
            ax_ttft.axvspan(midpoint, crossover, color="#fff3cd", alpha=0.5,
                            zorder=1, label="_adapt")
            ax_ttft.axvline(crossover, color="#e67e22", linestyle="-",
                            linewidth=2.0, alpha=0.8, zorder=5)
            # Annotation
            ypos = ax_ttft.get_ylim()[1] * 0.50
            ax_ttft.annotate(
                f"Adapted\n~{adapt_reqs} reqs\n({adapt_time:.0f}s)",
                xy=(crossover, ypos),
                xytext=(crossover + total_requests * 0.05, ypos * 1.1),
                fontsize=11, ha="left", color="#e67e22", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#e67e22", lw=1.5),
                zorder=6,
            )

    # ── Panel 2: KV Hit Ratio — routed vs system avg. ──
    _plot_sel_vs_avg_panel(
        ax_kv, data, "kv_sel", "kv_avg", midpoint, total_requests,
        "KV Hit Ratio (%)", "KV Cache Hit: Routed Pod vs System Avg",
    )

    # ── Panel 3: Waiting Requests — routed vs system avg. ──
    _plot_sel_vs_avg_panel(
        ax_wait, data, "wait_sel", "wait_avg", midpoint, total_requests,
        "Waiting Requests", "Waiting Requests: Routed Pod vs System Avg",
    )

    # ── Panel: Preemptions (if available) ──
    if ax_preempt is not None:
        _draw_phase_bg(ax_preempt, midpoint, total_requests)
        for key in all_keys:
            d = data[key]
            if d.get("preempt_centers") is not None:
                ax_preempt.plot(d["preempt_centers"], d["preempt_means"],
                                linewidth=2.0, color=d["color"], zorder=4)
        ax_preempt.set_ylabel("New Preemptions\n(per window)")
        ax_preempt.set_title("vLLM Preemptions (sliding-window sum of new preemptions)",
                             loc="left", fontweight="bold")
        ax_preempt.grid(alpha=0.2)

    # ── Panel 4: GPU KV Cache Utilization — routed vs system avg. ──
    _plot_sel_vs_avg_panel(
        ax_gpu, data, "gpu_sel", "gpu_avg", midpoint, total_requests,
        "GPU KV Cache Util.", "GPU KV Cache Utilization: Routed Pod vs System Avg",
    )

    # ── Panel: Prediction Error (MAE) ──
    if ax_mae is not None:
        _draw_phase_bg(ax_mae, midpoint, total_requests)
        for key in all_keys:
            d = data[key]
            if d.get("mae_centers") is not None:
                ax_mae.plot(d["mae_centers"], d["mae_vals"], linewidth=2.0,
                            color=d["color"], zorder=4)
        ax_mae.set_ylabel("MAE (reward units)")
        ax_mae.set_title("Prediction Error (MAE: predicted vs actual reward)",
                         loc="left", fontweight="bold")
        ax_mae.grid(alpha=0.2)

    # ── Sliding routing balance timeseries ──
    for key in all_keys:
        d = data[key]
        centers, cvs, jains_v, ents = _sliding_routing_metrics(
            d["pods_arr"], window, step,
        )
        d["rt_centers"] = centers
        d["rt_cvs"] = cvs
        d["rt_jains"] = jains_v
        d["rt_ents"] = ents

    _draw_phase_bg(ax_entropy_ts, midpoint, total_requests)

    for key in all_keys:
        d = data[key]
        ax_entropy_ts.plot(d["rt_centers"], d["rt_ents"],
                           linewidth=2.0, color=d["color"], zorder=4)

    ax_entropy_ts.set_ylabel("Norm. Entropy")
    ax_entropy_ts.set_title("Sliding Routing Balance: Normalized Entropy (1.0 = uniform)",
                            loc="left", fontweight="bold")
    ax_entropy_ts.grid(alpha=0.2)

    # X label on last full-width timeseries panel
    ax_entropy_ts.set_xlabel("Request Index")
    ax_ttft.set_xlim(0, total_requests)

    # ── Combined TTFT bar chart: Overall | Phase 1 | Phase 2 ──
    phases = [
        ("Overall", lambda d: d["ttft_raw"]),
        ("Phase 1\n(5% sharing)", lambda d: d["ttft_raw"][:d["half"]]),
        ("Phase 2\n(50% sharing)", lambda d: d["ttft_raw"][d["half"]:]),
    ]
    n_policies = len(all_keys)
    n_phases = len(phases)
    bar_width = 0.8 / (n_policies * 2)
    positions = []
    group_width = n_policies * bar_width * 2.2
    gap = group_width * 0.6

    ax2 = ax_bars.twinx()
    for pi, (phase_label, slice_fn) in enumerate(phases):
        phase_center = pi * (group_width + gap)
        positions.append(phase_center)
        for ki, key in enumerate(all_keys):
            d = data[key]
            ttft = slice_fn(d)
            avg_val = np.mean(ttft)
            p99_val = np.percentile(ttft, 99)
            group_offset = (ki - (n_policies - 1) / 2) * (bar_width * 2.2)
            # Avg on left axis
            b = ax_bars.bar(phase_center + group_offset - bar_width / 2, avg_val,
                            bar_width, color=d["color"], edgecolor="black",
                            linewidth=0.5, zorder=3)
            ax_bars.text(b[0].get_x() + b[0].get_width() / 2, avg_val,
                         f"{avg_val:.0f}", ha="center", va="bottom", fontsize=9)
            # P99 on right axis
            b2 = ax2.bar(phase_center + group_offset + bar_width / 2, p99_val,
                         bar_width, color=d["color"], edgecolor="black",
                         linewidth=0.5, alpha=0.55, hatch="//", zorder=3)
            ax2.text(b2[0].get_x() + b2[0].get_width() / 2, p99_val,
                     f"{p99_val:.0f}", ha="center", va="bottom", fontsize=9)

    # Gray vertical separators
    for i in range(1, n_phases):
        sep_x = (positions[i - 1] + positions[i]) / 2
        ax_bars.axvline(sep_x, color="gray", linestyle="-", linewidth=1.5,
                        alpha=0.4, zorder=1)

    ax_bars.set_xticks(positions)
    ax_bars.set_xticklabels([p[0] for p in phases], fontsize=13)
    ax_bars.set_ylabel("Avg TTFT (ms)")
    ax2.set_ylabel("P99 TTFT (ms)")
    ax_bars.set_title("TTFT: Avg + P99 by Phase",
                      loc="left", fontweight="bold", fontsize=15)
    ax_bars.grid(axis="y", alpha=0.2)

    bar_legend = [Patch(facecolor=COLORS[k], edgecolor="black", linewidth=0.5,
                        label=BAR_LABELS[k]) for k in all_keys]
    bar_legend += [
        Patch(facecolor="gray", edgecolor="black", linewidth=0.5,
              label="Avg (left)"),
        Patch(facecolor="gray", edgecolor="black", linewidth=0.5,
              alpha=0.55, hatch="//", label="P99 (right)"),
    ]
    ax_bars.legend(handles=bar_legend, loc="upper right", fontsize=10)

    # ── Shared timeseries legend ──────────────────────────────
    legend_handles = [
        Line2D([0], [0], color=COLORS[LIMITED_KEY], lw=2,
               label="QS (mid-frozen): routed inst."),
        Line2D([0], [0], color=COLORS[LIMITED_KEY], lw=1.5, linestyle="--",
               marker="x", markersize=6, markeredgewidth=1.5,
               alpha=0.7, label="QS (mid-frozen): system avg."),
        Line2D([0], [0], color=COLORS[UNLIMITED_KEY], lw=2,
               label="QS: routed inst."),
        Line2D([0], [0], color=COLORS[UNLIMITED_KEY], lw=1.5, linestyle="--",
               marker="x", markersize=6, markeredgewidth=1.5,
               alpha=0.7, label="QS: system avg."),
    ]
    if PREFIX_CACHE_KEY in data:
        legend_handles += [
            Line2D([0], [0], color=COLORS[PREFIX_CACHE_KEY], lw=2,
                   label="Prefix Cache: routed inst."),
            Line2D([0], [0], color=COLORS[PREFIX_CACHE_KEY], lw=1.5, linestyle="--",
                   marker="x", markersize=6, markeredgewidth=1.5,
                   alpha=0.7, label="Prefix Cache: system avg."),
        ]
    legend_handles += [
        Patch(facecolor="#e8f0fe", edgecolor="gray", alpha=0.6,
              label="Phase 1: 5% sharing"),
        Patch(facecolor="#fce8e6", edgecolor="gray", alpha=0.6,
              label="Phase 2: 50% sharing"),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=2,
        framealpha=0.9,
        fontsize=14,
    )

    fig.subplots_adjust(bottom=0.04)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


# ── Dense paper figure ────────────────────────────────────────
def plot_figure_dense(
    data: dict,
    midpoint: int,
    total_requests: int,
    window: int,
    step: int,
    out_path: str,
    rps: float = 7.8,
) -> None:
    """Compact 2x2 figure for paper: (TTFT, KV hit) / (MAE, GPU KV) + TTFT bar."""
    plt.rcParams.update({
        "font.size": 14,
        "font.family": "sans-serif",
        "axes.titlesize": 15,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
    })

    has_convergence = (UNLIMITED_KEY in data and PREFIX_CACHE_KEY in data)
    has_mae = any(d.get("mae_centers") is not None for d in data.values())

    # Layout: Row 1 = TTFT (2cols) + KV hit (2cols)
    #         Row 2 = MAE (1col) + Bar (1col) + GPU KV (2cols)
    #         MAE and Bar sit under TTFT; GPU KV sits under KV hit
    fig = plt.figure(figsize=(14, 5.5))
    gs = fig.add_gridspec(
        2, 12, height_ratios=[1, 0.85],
        width_ratios=[1, 1, 1, 0.25, 1, 1, 0.3, 1, 1, 1, 1, 0.3],
        hspace=0.55, wspace=0.80,
    )

    ax_ttft = fig.add_subplot(gs[0, 0:6])
    ax_kv = fig.add_subplot(gs[0, 7:11])
    ax_mae = fig.add_subplot(gs[1, 0:3]) if has_mae else None
    ax_bars = fig.add_subplot(gs[1, 4:6])
    ax_gpu = fig.add_subplot(gs[1, 7:11])

    all_keys = [k for k in [LIMITED_KEY, PREFIX_CACHE_KEY, UNLIMITED_KEY] if k in data]

    # ── TTFT timeseries ──
    _draw_phase_bg(ax_ttft, midpoint, total_requests)
    for key in all_keys:
        d = data[key]
        ax_ttft.plot(d["ttft_centers"], d["ttft_means"],
                     linewidth=2.0, color=d["color"], zorder=4)
    ax_ttft.set_ylabel("Mean TTFT (ms)")
    ax_ttft.set_title("(a) TTFT Time Series", loc="left", fontweight="bold")
    ax_ttft.grid(alpha=0.2)
    # Extra top margin for text annotations
    ymin_t, ymax_t = ax_ttft.get_ylim()
    ax_ttft.set_ylim(ymin_t, ymax_t * 1.18)

    # Phase annotations with opaque background
    ax_ttft.text(midpoint * 0.30, ax_ttft.get_ylim()[1] * 0.85,
                 "5% sharing", ha="center", fontsize=13,
                 fontstyle="italic", color="#333333",
                 bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2),
                 zorder=6)
    ax_ttft.text(midpoint + (total_requests - midpoint) * 0.55,
                 ax_ttft.get_ylim()[1] * 0.85,
                 "50% sharing", ha="center", fontsize=13,
                 fontstyle="italic", color="#333333",
                 bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2),
                 zorder=6)
    ax_ttft.annotate(
        "Workload\nshift",
        xy=(midpoint, ax_ttft.get_ylim()[1] * 0.55),
        xytext=(midpoint - total_requests * 0.06, ax_ttft.get_ylim()[1] * 0.72),
        fontsize=12, ha="center", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2),
        zorder=6,
    )

    # Convergence annotation
    if UNLIMITED_KEY in data:
        cont_d = data[UNLIMITED_KEY]
        crossover = None
        if has_convergence:
            pfx_d = data[PREFIX_CACHE_KEY]
            crossover = _find_crossover(
                cont_d["ttft_centers"], cont_d["ttft_means"],
                pfx_d["ttft_centers"], pfx_d["ttft_means"], midpoint)
        if crossover is None:
            crossover = _find_self_convergence(
                cont_d["ttft_centers"], cont_d["ttft_means"], midpoint)
        if crossover is not None:
            adapt_reqs = crossover - midpoint
            adapt_time = adapt_reqs / rps
            ax_ttft.axvspan(midpoint, crossover, color="#fff3cd", alpha=0.5, zorder=1)
            ax_ttft.axvline(crossover, color="#e67e22", linestyle="-",
                            linewidth=2.0, alpha=0.8, zorder=5)
            ypos = ax_ttft.get_ylim()[1] * 0.35
            ax_ttft.annotate(
                f"Adapted\n~{adapt_reqs} reqs\n({adapt_time:.0f}s)",
                xy=(crossover, ypos),
                xytext=(crossover - total_requests * 0.25, ypos * 0.60),
                fontsize=12, ha="center", color="#e67e22", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#e67e22", lw=1.5),
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2),
                zorder=6)

    # ── KV Cache Hit (solid lines for both routed and system avg) ──
    _draw_phase_bg(ax_kv, midpoint, total_requests)
    for key in all_keys:
        d = data[key]
        color = d["color"]
        ax_kv.plot(d["kv_sel_centers"], d["kv_sel_means"],
                   linewidth=2.0, color=color, linestyle="-", zorder=4)
        ax_kv.plot(d["kv_avg_centers"], d["kv_avg_means"],
                   linewidth=1.5, color=color, linestyle="-", alpha=0.7,
                   marker="x", markersize=5,
                   markevery=max(1, len(d["kv_avg_centers"]) // 20),
                   markeredgewidth=1.5, zorder=4)
    ax_kv.set_ylabel("KV Hit Ratio (%)")
    ax_kv.set_title("(d) KV Cache Hit Ratio", loc="left", fontweight="bold")
    ax_kv.grid(alpha=0.2)

    # ── GPU KV Cache Utilization (system avg only) ──
    _draw_phase_bg(ax_gpu, midpoint, total_requests)
    for key in all_keys:
        d = data[key]
        ax_gpu.plot(d["gpu_avg_centers"], d["gpu_avg_means"],
                    linewidth=2.0, color=d["color"], zorder=4)
    ax_gpu.set_ylabel("GPU KV Cache Util.")
    ax_gpu.set_title("(e) GPU KV Cache Util.", loc="left", fontweight="bold")
    ax_gpu.grid(alpha=0.2)
    ax_gpu.set_ylim(0.2, None)

    # ── MAE ──
    if ax_mae is not None:
        _draw_phase_bg(ax_mae, midpoint, total_requests)
        for key in all_keys:
            d = data[key]
            if d.get("mae_centers") is not None:
                ax_mae.plot(d["mae_centers"], d["mae_vals"], linewidth=2.0,
                            color=d["color"], zorder=4)
        ax_mae.set_ylabel("MAE")
        ax_mae.set_title("(b) Prediction Error",
                         loc="left", fontweight="bold")
        ax_mae.grid(alpha=0.2)
        ax_mae.set_xlim(0, total_requests)

    # Convert x-axis from request index to time (seconds)
    total_time = total_requests / rps
    # Wider panels get 500s ticks, narrower panels get 1000s ticks
    for ax_ts, step_s in [(ax_ttft, 500), (ax_kv, 500),
                          (ax_gpu, 500), (ax_mae, 1000)]:
        if ax_ts is None:
            continue
        tt = np.arange(0, total_time + 1, step_s)
        ax_ts.set_xticks(tt * rps)
        ax_ts.set_xticklabels([f"{int(t)}" for t in tt])
        ax_ts.set_xlim(0, total_requests)

    # X labels on all timeseries
    ax_ttft.set_xlabel("Time (s)")
    for ax_bottom in [ax_kv, ax_mae, ax_gpu]:
        if ax_bottom is not None:
            ax_bottom.set_xlabel("Time (s)")

    # ── Combined TTFT bar ──
    n_policies = len(all_keys)
    bar_width = 0.35
    x = np.arange(n_policies)

    avgs, p99s, colors, labels = [], [], [], []
    for key in all_keys:
        d = data[key]
        avgs.append(np.mean(d["ttft_raw"]))
        p99s.append(np.percentile(d["ttft_raw"], 99))
        colors.append(d["color"])
        labels.append(BAR_LABELS[key])

    # Avg on left axis
    bars_avg = ax_bars.bar(x - bar_width / 2, avgs, bar_width,
                           color=colors, edgecolor="black", linewidth=0.5, zorder=3)
    for bar in bars_avg:
        h = bar.get_height()
        ax_bars.text(bar.get_x() + bar.get_width() * 0.3, h,
                     f"{h:.0f}", ha="left", va="bottom", fontsize=9,
                     rotation=45)

    # P99 on right axis
    ax2 = ax_bars.twinx()
    bars_p99 = ax2.bar(x + bar_width / 2, p99s, bar_width,
                       color=colors, edgecolor="black", linewidth=0.5,
                       alpha=0.55, hatch="//", zorder=3)
    for bar in bars_p99:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() * 0.3, h,
                 f"{h:.0f}", ha="left", va="bottom", fontsize=9,
                 rotation=45)

    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels(labels)
    ax_bars.set_ylabel("TTFT (ms)")
    ax2.set_yticklabels([])
    ax2.set_ylabel("")
    ax_bars.set_title("(c) TTFT",
                      loc="left", fontweight="bold")
    ax_bars.grid(axis="y", alpha=0.2)
    # Extra top margin for value labels
    ymin_b, ymax_b = ax_bars.get_ylim()
    ax_bars.set_ylim(ymin_b, ymax_b * 1.30)
    ymin_b2, ymax_b2 = ax2.get_ylim()
    ax2.set_ylim(ymin_b2, ymax_b2 * 1.30)
    # Bar legend
    bar_legend = [
        Patch(facecolor="gray", edgecolor="black", linewidth=0.5,
              label="Avg"),
        Patch(facecolor="gray", edgecolor="black", linewidth=0.5,
              alpha=0.55, hatch="//", label="P99"),
    ]
    ax_bars.legend(handles=bar_legend, loc="upper right", fontsize=10)

    # ── Shared legend ──
    legend_handles = [
        Line2D([0], [0], color=COLORS[LIMITED_KEY], lw=2,
               label="QS (mid-frozen): routed inst."),
        Line2D([0], [0], color=COLORS[LIMITED_KEY], lw=1.5, linestyle="--",
               marker="x", markersize=5, markeredgewidth=1.5,
               alpha=0.7, label="QS (mid-frozen): system avg."),
        Line2D([0], [0], color=COLORS[UNLIMITED_KEY], lw=2,
               label="QS: routed inst."),
        Line2D([0], [0], color=COLORS[UNLIMITED_KEY], lw=1.5, linestyle="--",
               marker="x", markersize=5, markeredgewidth=1.5,
               alpha=0.7, label="QS: system avg."),
    ]
    if PREFIX_CACHE_KEY in data:
        legend_handles += [
            Line2D([0], [0], color=COLORS[PREFIX_CACHE_KEY], lw=2,
                   label="Prefix Cache: routed inst."),
            Line2D([0], [0], color=COLORS[PREFIX_CACHE_KEY], lw=1.5, linestyle="--",
                   marker="x", markersize=5, markeredgewidth=1.5,
                   alpha=0.7, label="Prefix Cache: system avg."),
        ]
    legend_handles += [
        Patch(facecolor="#e8f0fe", edgecolor="gray", alpha=0.6,
              label="Phase 1: 5% sharing"),
        Patch(facecolor="#fce8e6", edgecolor="gray", alpha=0.6,
              label="Phase 2: 50% sharing"),
    ]
    fig.legend(handles=legend_handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.10), ncol=3, framealpha=0.9, fontsize=13)

    fig.subplots_adjust(top=0.88)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adaptation ablation: continuous vs frozen router."
    )
    parser.add_argument("base_dir", help="Directory containing the two experiment subdirs")
    parser.add_argument("--csv-name", default=DEFAULT_CSV_NAME)
    parser.add_argument("--window-size", type=int, default=500)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--output", default=None,
                        help="Output path (default: <base_dir>/adaptation_ablation.pdf)")
    args = parser.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    out_path = args.output or os.path.join(base_dir, "adaptation_ablation.pdf")
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    step = args.step if args.step else max(1, args.window_size // 5)

    experiments = discover_experiments(base_dir, args.csv_name)

    data = {}
    midpoint = None

    print("=== Adaptation Ablation ===")
    for dirname, key, label in experiments:
        csv_path = os.path.join(base_dir, dirname, args.csv_name)
        df = load_experiment(csv_path)
        n = len(df)
        half = n // 2
        if midpoint is None:
            midpoint = half

        ttft = df["ttft"].to_numpy(dtype=float)
        sel_kv = extract_selected_pod_metric(df, "kv_hit_ratio")
        avg_kv = extract_system_avg_metric(df, "kv_hit_ratio")
        sel_pf = extract_selected_pod_metric(df, "prefill_tokens")
        avg_pf = extract_system_avg_metric(df, "prefill_tokens")
        sel_wait = extract_selected_pod_metric(df, "waiting_requests")
        avg_wait = extract_system_avg_metric(df, "waiting_requests")
        sel_gpu = extract_selected_pod_metric(df, "gpu_kv_cache")
        avg_gpu = extract_system_avg_metric(df, "gpu_kv_cache")

        t_c, t_m = sliding_window(ttft, args.window_size, step)
        ks_c, ks_m = sliding_window(sel_kv, args.window_size, step)
        ka_c, ka_m = sliding_window(avg_kv, args.window_size, step)
        ps_c, ps_m = sliding_window(sel_pf, args.window_size, step)
        pa_c, pa_m = sliding_window(avg_pf, args.window_size, step)
        ws_c, ws_m = sliding_window(sel_wait, args.window_size, step)
        wa_c, wa_m = sliding_window(avg_wait, args.window_size, step)
        gs_c, gs_m = sliding_window(sel_gpu, args.window_size, step)
        ga_c, ga_m = sliding_window(avg_gpu, args.window_size, step)

        # Parse prediction log for MAE (CB experiments only)
        log_path = os.path.join(base_dir, dirname, LOG_NAME)
        mae_centers, mae_vals = None, None
        if os.path.exists(log_path) and key != PREFIX_CACHE_KEY:
            preds = parse_predictions(log_path)
            req_ids = sorted(set(preds.keys()) & set(df["request_id"]))
            if len(req_ids) > 0:
                pred_r = np.array([preds[r] for r in req_ids])
                actual_ttft = df.set_index("request_id").loc[req_ids, "ttft"].values
                actual_r = ttft_to_reward(actual_ttft)
                mae_centers, mae_vals = sliding_mae(
                    pred_r, actual_r, args.window_size, step,
                )
                print(f"    Parsed {len(req_ids)} predictions for MAE")

        # Parse preemptions from gateway log if available
        gw_log_path = os.path.join(base_dir, dirname, GATEWAY_LOG_NAME)
        preempt_centers, preempt_means = None, None
        if os.path.exists(gw_log_path):
            preempt_deltas = parse_preemptions(gw_log_path, list(df["request_id"]))
            if preempt_deltas:
                # Align to sorted request_ids
                req_ids_sorted = df["request_id"].tolist()
                preempt_arr = np.array([preempt_deltas.get(r, 0) for r in req_ids_sorted],
                                       dtype=float)
                preempt_centers, preempt_means = sliding_window(
                    preempt_arr, args.window_size, step,
                )
                total_preempt = int(np.sum(preempt_arr))
                print(f"    Preemptions: {total_preempt} total")

        data[key] = {
            "label": label,
            "color": COLORS[key],
            "ttft_raw": ttft,
            "input_tokens": df["input_tokens"].to_numpy(dtype=float),
            "half": half,
            "selected_pods": df["selected_pod"],
            "pods_arr": df["selected_pod"].to_numpy(),
            "ttft_centers": t_c, "ttft_means": t_m,
            "kv_sel_centers": ks_c, "kv_sel_means": ks_m,
            "kv_avg_centers": ka_c, "kv_avg_means": ka_m,
            "pf_sel_centers": ps_c, "pf_sel_means": ps_m,
            "pf_avg_centers": pa_c, "pf_avg_means": pa_m,
            "wait_sel_centers": ws_c, "wait_sel_means": ws_m,
            "wait_avg_centers": wa_c, "wait_avg_means": wa_m,
            "gpu_sel_centers": gs_c, "gpu_sel_means": gs_m,
            "gpu_avg_centers": ga_c, "gpu_avg_means": ga_m,
            "mae_centers": mae_centers, "mae_vals": mae_vals,
            "preempt_centers": preempt_centers, "preempt_means": preempt_means,
            "n": n,
        }

        p2_sel_kv = float(np.mean(sel_kv[half:]))
        p2_avg_kv = float(np.mean(avg_kv[half:]))
        p2_sel_pf = float(np.mean(sel_pf[half:]))
        p2_avg_pf = float(np.mean(avg_pf[half:]))
        p2_sel_w = float(np.mean(sel_wait[half:]))
        p2_avg_w = float(np.mean(avg_wait[half:]))
        p2_sel_g = float(np.mean(sel_gpu[half:]))
        p2_avg_g = float(np.mean(avg_gpu[half:]))

        print(f"\n  {label.replace(chr(10), ' ')}  [{dirname}]")
        print(f"    TTFT:      Phase1={np.mean(ttft[:half]):.0f}ms   Phase2={np.mean(ttft[half:]):.0f}ms")
        print(f"    KV hit:    sel={p2_sel_kv:.1f}%  sys={p2_avg_kv:.1f}%  adv={p2_sel_kv-p2_avg_kv:+.1f}%")
        print(f"    Prefill:   sel={p2_sel_pf:.0f}   sys={p2_avg_pf:.0f}   adv={p2_sel_pf-p2_avg_pf:+.0f}")
        print(f"    Waiting:   sel={p2_sel_w:.2f}   sys={p2_avg_w:.2f}   adv={p2_sel_w-p2_avg_w:+.2f}")
        print(f"    GPU KV$:   sel={p2_sel_g:.3f}   sys={p2_avg_g:.3f}   adv={p2_sel_g-p2_avg_g:+.3f}")

    total_requests = max(d["n"] for d in data.values())
    plot_figure(data, midpoint, total_requests, args.window_size, step, out_path)

    # Dense paper version
    dense_path = os.path.splitext(out_path)[0] + "_paper.pdf"
    plot_figure_dense(data, midpoint, total_requests, args.window_size, step, dense_path)

    # ── Export sliding-window data to CSV ──
    csv_out = os.path.splitext(out_path)[0] + ".csv"
    rows = []
    for key in [LIMITED_KEY, PREFIX_CACHE_KEY, UNLIMITED_KEY]:
        if key not in data:
            continue
        d = data[key]
        label_flat = d["label"].replace("\n", " ")
        for i in range(len(d["ttft_centers"])):
            rows.append({
                "method": label_flat,
                "request_index": int(d["ttft_centers"][i]),
                "phase": "Phase1_0pct_sharing" if d["ttft_centers"][i] < midpoint else "Phase2_50pct_sharing",
                "mean_ttft_ms": round(d["ttft_means"][i], 2),
                "mean_sel_kv_hit_ratio": round(d["kv_sel_means"][i], 2),
                "mean_sys_kv_hit_ratio": round(d["kv_avg_means"][i], 2),
                "mean_sel_prefill_tokens": round(d["pf_sel_means"][i], 2),
                "mean_sys_prefill_tokens": round(d["pf_avg_means"][i], 2),
                "mean_sel_waiting_requests": round(d["wait_sel_means"][i], 2),
                "mean_sys_waiting_requests": round(d["wait_avg_means"][i], 2),
                "mean_sel_gpu_kv_cache": round(d["gpu_sel_means"][i], 4),
                "mean_sys_gpu_kv_cache": round(d["gpu_avg_means"][i], 4),
            })
    pd.DataFrame(rows).to_csv(csv_out, index=False)
    print(f"\nSaved: {csv_out}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
