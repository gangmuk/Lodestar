#!/usr/bin/env python3
"""
Model evolution analysis for online learning checkpoints.

Analyzes what features the reward model learns to value as more online data
arrives, using gradient-based sensitivity analysis (not connection weights,
which is misleading for deep ReLU networks).

Methods:
  1. Gradient Sensitivity: avg |dOutput/dInput| over realistic samples
  2. Perturbation Sensitivity: output change when each feature moves ±1 std

All methods use actual feature distributions from feature_normalization_statistics.csv
and feature_distribution_statistics.csv so that inputs match what the model saw
during training (z-scored features with real correlations and skew).

Usage:
    python3 analyze_model_evolution.py <directory_with_reward_net_checkpoints>

Output:
    model_evolution_analysis/model_evolution_analysis.pdf
    model_evolution_analysis/*.csv
"""

import os
import sys
import glob
import re
import csv
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 12,
    'font.family': 'sans-serif',
    'figure.dpi': 150,
})


# Number of checkpoints to aggregate per plotted point.
# 1 = original per-iteration plotting.
PLOT_AGGREGATE_WINDOW = 3


# ── Auto-detect feature names from model + metadata ─────────────────────────

def detect_feature_names(directory, input_dim):
    """Read feature names from metadata.json. Crash if anything doesn't match.

    Model input tensor order (from _create_per_pod_contexts):
        [pod_features_with_staleness, kv_hit_ratios, request_features]

    metadata.json must provide model_input_feature_names (written by encoding.py)
    with exactly input_dim entries.  If that field is missing (old metadata), we
    reconstruct from pod_features_list + kv + request names and require the total
    to match exactly — no guessing, no fallback.
    """
    # ── Find metadata.json ───────────────────────────────────────────────
    meta_path = None
    for pattern in ['encoded_data/*/metadata.json', 'encoded_data/metadata.json']:
        matches = glob.glob(os.path.join(directory, pattern))
        if matches:
            meta_path = matches[0]
            break

    if meta_path is None:
        print(f"ERROR: No metadata.json found under {directory}/encoded_data/")
        print(f"  Cannot determine feature names for model input_dim={input_dim}.")
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)
    print(f"  Reading metadata from {meta_path}")

    # ── Prefer model_input_feature_names (new field) ─────────────────────
    model_names = meta.get('model_input_feature_names')
    if model_names is not None:
        if len(model_names) != input_dim:
            print(f"ERROR: metadata.model_input_feature_names has {len(model_names)} entries "
                  f"but model input_dim={input_dim}.")
            print(f"  model_input_feature_names: {model_names}")
            print(f"  metadata path: {meta_path}")
            sys.exit(1)
        print(f"  Feature names ({len(model_names)}): {model_names}")
        return model_names

    # ── Reconstruct from pod_features_list + kv + request ────────────────
    feat_dims = meta.get('feature_dimensions', {})
    pod_dim = feat_dims.get('pod_features_with_staleness')
    kv_dim = feat_dims.get('kv_hit_ratios')
    req_dim = feat_dims.get('request_features')
    pod_names = meta.get('pod_features_list', [])
    req_names = meta.get('numeric_request_features', [])

    # Validate everything is present
    for field, val in [('feature_dimensions.pod_features_with_staleness', pod_dim),
                       ('feature_dimensions.kv_hit_ratios', kv_dim),
                       ('feature_dimensions.request_features', req_dim)]:
        if val is None:
            print(f"ERROR: metadata.json missing '{field}'.")
            print(f"  metadata path: {meta_path}")
            sys.exit(1)

    # Check dimension sum matches model
    metadata_total = pod_dim + kv_dim + req_dim
    if metadata_total != input_dim:
        print(f"ERROR: metadata dimensions don't sum to model input_dim.")
        print(f"  pod_features_with_staleness({pod_dim}) + kv_hit_ratios({kv_dim}) + request_features({req_dim}) = {metadata_total}")
        print(f"  model input_dim = {input_dim}")
        print(f"  metadata path: {meta_path}")
        sys.exit(1)

    # Check pod feature names count matches pod dim exactly
    if len(pod_names) != pod_dim:
        print(f"ERROR: pod_features_list has {len(pod_names)} names but pod_features_with_staleness dim = {pod_dim}.")
        print(f"  pod_features_list: {pod_names}")
        print(f"  This means {pod_dim - len(pod_names)} feature(s) were appended at runtime but not recorded.")
        print(f"  Fix: add 'model_input_feature_names' field to metadata.json with all {input_dim} feature names.")
        print(f"  metadata path: {meta_path}")
        sys.exit(1)

    # Check request feature names count matches
    if len(req_names) != req_dim:
        print(f"ERROR: numeric_request_features has {len(req_names)} names but request_features dim = {req_dim}.")
        print(f"  numeric_request_features: {req_names}")
        print(f"  metadata path: {meta_path}")
        sys.exit(1)

    kv_names = ['kv_hit_ratio'] if kv_dim == 1 else [f'kv_hit_ratio_{i}' for i in range(kv_dim)]
    names = list(pod_names) + kv_names + list(req_names)

    print(f"  Feature names ({len(names)}): {names}")
    return names


# ── Load actual feature distributions ────────────────────────────────────────

def load_normalization_stats(directory, feature_names):
    """Load per-feature mean/std from feature_normalization_statistics.csv.
    Returns dict {feature_name: {'mean': float, 'std': float}}."""
    stats_path = os.path.join(directory, 'feature_normalization_statistics.csv')
    if not os.path.exists(stats_path):
        return None

    stats = {}
    with open(stats_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['feature_name']
            stype = row['stats_type']
            val = float(row['value'])
            if name not in stats:
                stats[name] = {}
            stats[name][stype] = val

    print(f"  Loaded normalization stats from {stats_path}")
    return stats


def _compute_dist_stats_from_serving_log(csv_path, feature_names):
    """Compute per-feature distribution stats directly from the processed serving log.

    For pod features (columns like pod_XXXX-feature), pools all pod columns together
    (same as how the model sees them — per-pod, not per-request).
    For request features (plain columns), uses the column directly.

    Returns dict {feature_name: {'mean', 'std', 'min', 'max', 'p05', 'p95', 'zero_ratio'}}.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    print(f"  Loaded serving log: {csv_path} ({len(df)} rows)")

    stats = {}
    for fname in feature_names:
        # Try pod columns first (pod_XXXX-feature)
        pod_cols = [c for c in df.columns if c.endswith(f'-{fname}')]
        if pod_cols:
            vals = df[pod_cols].values.flatten()
            vals = vals[~np.isnan(vals)]
        elif fname in df.columns:
            vals = df[fname].dropna().values
        else:
            print(f"  WARNING: Feature '{fname}' not found in serving log")
            continue

        if len(vals) == 0:
            continue

        n_zeros = int((vals == 0).sum())
        stats[fname] = {
            'mean': float(vals.mean()),
            'std': float(vals.std()),
            'min': float(vals.min()),
            'max': float(vals.max()),
            'p05': float(np.percentile(vals, 5)),
            'p95': float(np.percentile(vals, 95)),
            'zero_ratio': n_zeros / len(vals),
            'count': len(vals),
        }

    return stats


def load_feature_distribution(directory, feature_names):
    """Load per-feature distribution stats, preferring the actual serving log.

    Priority:
      1. ../../filtered-aibrix-gateway-plugins-processed.log.csv (actual serving data)
      2. feature_distribution_statistics.csv (offline snapshot — may not match serving)

    Returns dict {feature_name: {'mean', 'std', 'min', 'max', 'p05', 'p95', 'zero_ratio'}}.
    """
    # Search for the processed serving log relative to the final_model dir
    serving_log_candidates = [
        os.path.join(directory, '..', '..', 'filtered-aibrix-gateway-plugins-processed.log.csv'),
        os.path.join(directory, '..', 'filtered-aibrix-gateway-plugins-processed.log.csv'),
        os.path.join(directory, 'filtered-aibrix-gateway-plugins-processed.log.csv'),
    ]
    for candidate in serving_log_candidates:
        candidate = os.path.normpath(candidate)
        if os.path.exists(candidate):
            print(f"  Found serving log: {candidate}")
            stats = _compute_dist_stats_from_serving_log(candidate, feature_names)
            if stats:
                return stats

    # Fallback to feature_distribution_statistics.csv
    dist_path = os.path.join(directory, 'feature_distribution_statistics.csv')
    if not os.path.exists(dist_path):
        return None

    print(f"  WARNING: No serving log found, falling back to {dist_path} (offline snapshot)")
    stats = {}
    with open(dist_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['feature_name']
            entry = {}
            for k, v in row.items():
                if k not in ('feature_name', 'feature_type'):
                    try:
                        entry[k] = float(v)
                    except (ValueError, TypeError):
                        pass
            stats[name] = entry

    return stats


def generate_realistic_samples(norm_stats, dist_stats, feature_names, n_samples=1000):
    """Generate samples that approximate the actual normalized feature distribution.

    Strategy: for each feature, sample from a truncated distribution in RAW space
    using the actual serving data stats, then z-score normalize using the mean/std
    from feature_normalization_statistics.csv.
    This produces inputs in the same space the model was trained on.

    Falls back to N(0,1) for any feature missing stats.
    """
    input_dim = len(feature_names)
    samples = torch.zeros(n_samples, input_dim)

    for i, fname in enumerate(feature_names):
        n_stats = norm_stats.get(fname) if norm_stats else None
        d_stats = dist_stats.get(fname) if dist_stats else None

        if n_stats and d_stats and n_stats.get('std', 0) > 0:
            raw_mean = d_stats.get('mean', n_stats['mean'])
            raw_std = d_stats.get('std', n_stats['std'])
            raw_min = d_stats.get('min', raw_mean - 3 * raw_std)
            raw_max = d_stats.get('max', raw_mean + 3 * raw_std)
            p05 = d_stats.get('p05', raw_min)
            p95 = d_stats.get('p95', raw_max)
            zero_ratio = d_stats.get('zero_ratio', 0.0)

            raw = np.zeros(n_samples)
            n_zeros = int(n_samples * zero_ratio)
            n_bulk = n_samples - n_zeros
            if n_bulk > 0:
                raw[n_zeros:] = np.random.uniform(p05, max(p95, p05 + 1e-6), n_bulk)

            # Normalize to z-score space (what the model actually sees)
            normalized = (raw - n_stats['mean']) / n_stats['std']
            samples[:, i] = torch.from_numpy(normalized).float()
        else:
            samples[:, i] = torch.randn(n_samples)
            if n_stats is None:
                print(f"  WARNING: No normalization stats for '{fname}', using N(0,1) fallback")

    return samples


def get_per_feature_stds_in_normalized_space(norm_stats, dist_stats, feature_names):
    """Return the std of each feature IN NORMALIZED SPACE.

    Computed as actual_raw_std / normalization_std.  When the serving data std
    differs from the normalization stats std (e.g. due to distribution shift
    between offline and online), this captures the true spread the model sees.

    Returns np.array of shape (input_dim,).
    """
    input_dim = len(feature_names)
    stds = np.ones(input_dim)

    for i, fname in enumerate(feature_names):
        n_stats = norm_stats.get(fname) if norm_stats else None
        d_stats = dist_stats.get(fname) if dist_stats else None
        if n_stats and d_stats and n_stats.get('std', 0) > 0:
            actual_raw_std = d_stats.get('std', n_stats['std'])
            stds[i] = actual_raw_std / n_stats['std']

    return stds


# ── Model loading ────────────────────────────────────────────────────────────

class RewardNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.scorer(x)


def load_checkpoints(directory):
    pattern = os.path.join(directory, 'reward_net-*.pth')
    files = glob.glob(pattern)
    checkpoints = []
    for f in files:
        match = re.search(r'reward_net-(\d+)\.pth$', f)
        if match:
            idx = int(match.group(1))
            try:
                sd = torch.load(f, map_location='cpu', weights_only=True)
                checkpoints.append((idx, sd))
            except Exception as e:
                print(f"  Skipping corrupted {f}: {e}")
    # Also pick up a single reward_net.pth (no iteration number)
    single = os.path.join(directory, 'reward_net.pth')
    if not checkpoints and os.path.exists(single):
        try:
            sd = torch.load(single, map_location='cpu', weights_only=True)
            checkpoints.append((0, sd))
        except Exception as e:
            print(f"  Skipping corrupted {single}: {e}")
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def build_model(state_dict, input_dim, hidden_dim):
    model = RewardNetwork(input_dim, hidden_dim)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ── Analysis methods ─────────────────────────────────────────────────────────

def compute_gradient_sensitivity(model, input_dim, realistic_samples=None, n_samples=1000):
    """Average |dOutput/dInput| over samples. Returns raw (unnormalized) values.
    Uses realistic_samples from actual feature distributions when available."""
    if realistic_samples is not None:
        x = realistic_samples[:n_samples].clone().detach().requires_grad_(True)
    else:
        x = torch.randn(n_samples, input_dim, requires_grad=True)
    out = model(x)
    out.sum().backward()
    return x.grad.abs().mean(dim=0).detach().numpy()


def compute_integrated_gradients(model, input_dim, realistic_samples=None,
                                 n_samples=200, n_steps=50):
    """Integrated Gradients (Sundararajan et al., ICML 2017) with zero baseline.
    Uses realistic_samples from actual feature distributions when available.
    Returns normalized relative importance (sums to 1)."""
    baseline = torch.zeros(1, input_dim)
    if realistic_samples is not None:
        x = realistic_samples[:n_samples].clone().detach()
    else:
        x = torch.randn(n_samples, input_dim)
    attributions = torch.zeros(n_samples, input_dim)

    for step in range(n_steps):
        alpha = (step + 0.5) / n_steps  # midpoint rule for better numerical integration
        interpolated = baseline + alpha * (x - baseline)
        interpolated = interpolated.detach().requires_grad_(True)
        out = model(interpolated)
        out.sum().backward()
        attributions += interpolated.grad

    # IG = (x - baseline) * avg_gradients_along_path
    ig = (x - baseline) * (attributions / n_steps)
    raw = ig.abs().mean(dim=0).detach().numpy()
    total = raw.sum()
    if total > 0:
        return raw / total
    return raw


def compute_perturbation_sensitivity(model, input_dim, per_feature_stds=None,
                                     realistic_samples=None):
    """Output change when each feature moves ±1 actual std from a realistic baseline.

    When realistic_samples are provided, the baseline is the mean of those samples
    and the perturbation magnitude is the actual per-feature std in normalized space.
    This ensures features with different effective ranges are compared fairly.

    Returns (sensitivity, direction) where direction[i] = +1 if higher feature → higher reward.
    """
    if realistic_samples is not None:
        baseline = realistic_samples.mean(dim=0, keepdim=True)
    else:
        baseline = torch.zeros(1, input_dim)

    if per_feature_stds is None:
        per_feature_stds = np.ones(input_dim)

    sensitivity = np.zeros(input_dim)
    direction = np.zeros(input_dim)
    for i in range(input_dim):
        step = float(per_feature_stds[i])
        with torch.no_grad():
            p_plus = baseline.clone(); p_plus[0, i] += step
            p_minus = baseline.clone(); p_minus[0, i] -= step
            out_plus = model(p_plus).item()
            out_minus = model(p_minus).item()
        diff = out_plus - out_minus
        sensitivity[i] = abs(diff)
        direction[i] = 1.0 if diff > 0 else (-1.0 if diff < 0 else 0.0)
    return sensitivity, direction


# ── Plotting ─────────────────────────────────────────────────────────────────

# Readable display names
DISPLAY_NAMES = {
    'inflight_prefill_requests': 'Inflight Prefill',
    'inflight_decode_requests': 'Inflight Decode',
    'inflight_requests': 'Inflight Requests',
    'gpu_kv_cache': 'GPU KV Cache',
    'cpu_kv_cache': 'CPU KV Cache',
    'running_requests': 'Running Requests',
    'waiting_requests': 'Waiting Requests',
    'prefill_tokens': 'Prefill Tokens',
    'decode_tokens': 'Decode Tokens',
    'kv_hit_ratio': 'KV Hit Ratio',
    'kv_hit_ratio_fresh': 'KV Hit Ratio (Fresh)',
    'input_tokens': 'Input Tokens',
    'output_tokens': 'Output Tokens',
    'total_tokens': 'Total Tokens',
    'kv_differential': 'KV Differential',
    'kv_concentration': 'KV Concentration',
    'staleness': 'Staleness',
}

# Features to highlight (key features for prefix-aware routing)
HIGHLIGHT_FEATURES = {'kv_hit_ratio', 'kv_hit_ratio_fresh', 'gpu_kv_cache', 'prefill_tokens', 'waiting_requests'}

# Known expected directions (higher feature → higher/lower TTFT → lower/higher reward)
EXPECTED_DIRECTIONS = {
    'gpu_kv_cache': -1,        # higher → worse (more memory pressure)
    'prefill_tokens': -1,      # higher → worse (more prefill work)
    'waiting_requests': -1,    # higher → worse (more queued)
    'running_requests': -1,    # higher → worse (more concurrent)
    'kv_hit_ratio': +1,        # higher → better (more prefix reuse)
    'kv_hit_ratio_fresh': +1,  # higher → better (fresh prefix blocks likely still in vLLM cache)
    'input_tokens': -1,        # higher → worse (more to prefill)
    'decode_tokens': -1,       # higher → worse (more decode work)
    'kv_differential': +1,     # higher → better (this pod has more prefix affinity)
    'kv_concentration': +1,    # higher → more uneven cache, stronger routing signal
    'staleness': 0,            # always zeros in this experiment
}


def get_display_name(feature):
    return DISPLAY_NAMES.get(feature, feature.replace('_', ' ').title())


def get_feature_color(feature, feature_names):
    """Assign a unique color to each feature so all are distinguishable."""
    # Fixed palette: highlighted features keep their signature colors
    highlight_colors = {
        'kv_hit_ratio': '#e63946',      # Red
        'kv_hit_ratio_fresh': '#f4a261', # Orange
        'gpu_kv_cache': '#457b9d',      # Blue
        'prefill_tokens': '#2a9d8f',    # Teal
        'waiting_requests': '#e9c46a',  # Gold
        'kv_differential': '#f4a261',   # Orange
        'kv_concentration': '#264653',  # Dark teal
    }
    if feature in highlight_colors:
        return highlight_colors[feature]
    # Generate distinct colors for remaining features using a perceptually-spaced colormap
    non_highlight = [f for f in feature_names if f not in highlight_colors]
    if feature in non_highlight:
        idx = non_highlight.index(feature)
        cmap = plt.cm.get_cmap('tab20', max(len(non_highlight), 1))
        rgba = cmap(idx % cmap.N)
        return matplotlib.colors.rgb2hex(rgba[:3])
    return '#666666'


def _fixed_legend_order(feature_names):
    """Return feature indices in a fixed legend order. Features not in the
    preferred list are appended alphabetically at the end."""
    PREFERRED_ORDER = [
        'prefill_tokens', 'gpu_kv_cache', 'kv_hit_ratio', 'kv_hit_ratio_fresh',
        'waiting_requests', 'input_tokens', 'inflight_prefill_requests',
        'running_requests', 'decode_tokens', 'inflight_decode_requests',
        'inflight_requests', 'cpu_kv_cache', 'total_tokens', 'output_tokens',
        'kv_differential', 'kv_concentration', 'staleness',
    ]
    rank = {name: i for i, name in enumerate(PREFERRED_ORDER)}
    indices = list(range(len(feature_names)))
    indices.sort(key=lambda i: (rank.get(feature_names[i], len(PREFERRED_ORDER)), feature_names[i]))
    return indices


def plot_gradient_evolution(checkpoints, grad_matrix, feature_names, pdf, save_dir,
                           title_suffix='', ylabel_prefix=''):
    """Gradient sensitivity evolution — how much each feature affects the output."""
    indices = np.array([c[0] for c in checkpoints])

    fig, ax = plt.subplots(figsize=(16, 9))

    # Plot in fixed legend order
    order = _fixed_legend_order(feature_names)

    for fi in order:
        fname = feature_names[fi]
        vals = grad_matrix[:, fi]
        color = get_feature_color(fname, feature_names)
        lw = 2.0
        alpha = 0.9
        ms = 5
        zorder = 2
        ax.plot(indices, vals, 'o-', color=color, linewidth=lw, markersize=ms,
                alpha=alpha, label=get_display_name(fname), zorder=zorder)

    ax.set_xlabel('Online Training Iteration', fontsize=16, fontweight='bold')
    ax.set_ylabel(f'{ylabel_prefix}Gradient Sensitivity', fontsize=16, fontweight='bold')
    ax.set_title(f'Feature Sensitivity Evolution{title_suffix}',
                 fontsize=20, fontweight='bold', pad=15)
    ax.set_xticks(indices)
    ax.grid(True, alpha=0.2, linestyle='-')
    ax.set_xlim(indices[0] - 0.5, indices[-1] + 0.5)
    ax.set_ylim(bottom=0)

    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=13,
              frameon=True, fancybox=True, shadow=False, ncol=1,
              borderpad=1.0, handlelength=2.5)

    fig.tight_layout(rect=[0, 0, 0.78, 1])
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    print(f"  Plotted: Feature Sensitivity Evolution{title_suffix}")


def aggregate_for_plotting(checkpoints, *matrices, window=1):
    """Aggregate consecutive checkpoints into fixed-size windows for smoother plots.

    Returns:
      aggregated_checkpoints: list[(idx, None)] using rounded mean iteration index
      aggregated_matrices:    list[np.ndarray] with rows aligned to aggregated checkpoints
    """
    if window <= 1 or len(checkpoints) <= 1:
        passthrough = [np.asarray(m) for m in matrices]
        return checkpoints, passthrough

    n = len(checkpoints)
    aggregated_checkpoints = []
    aggregated_mats = [[] for _ in matrices]

    for start in range(0, n, window):
        end = min(start + window, n)
        block_indices = [checkpoints[i][0] for i in range(start, end)]
        agg_idx = int(round(float(np.mean(block_indices))))
        aggregated_checkpoints.append((agg_idx, None))

        for mi, mat in enumerate(matrices):
            block = np.asarray(mat[start:end])
            aggregated_mats[mi].append(block.mean(axis=0))

    aggregated_mats = [np.vstack(rows) for rows in aggregated_mats]
    return aggregated_checkpoints, aggregated_mats


def plot_perturbation_sensitivity(checkpoints, sens_matrix, dir_matrix, feature_names, pdf, save_dir):
    """Perturbation sensitivity as a signed heatmap: features × checkpoints.

    Color encodes direction correctness AND magnitude:
      Blue  = correct direction (learned matches domain knowledge), darker = stronger
      Red   = wrong direction (learned opposes expectation), darker = stronger
      Gray  = no expected direction defined

    Cell annotations show the signed sensitivity value (+/-).
    """
    input_dim = len(feature_names)
    n_ckpts = len(checkpoints)
    indices = [c[0] for c in checkpoints]

    # Build signed matrix: positive = correct direction, negative = wrong
    signed_matrix = np.zeros((input_dim, n_ckpts))
    for ci in range(n_ckpts):
        for fi in range(input_dim):
            fname = feature_names[fi]
            sens = sens_matrix[ci, fi]
            direction = dir_matrix[ci, fi]
            expected = EXPECTED_DIRECTIONS.get(fname, None)
            if expected is not None and expected != 0:
                correct = (direction > 0) == (expected > 0)
                signed_matrix[fi, ci] = sens if correct else -sens
            else:
                # No expected direction: show raw magnitude as positive
                signed_matrix[fi, ci] = sens

    # Row order
    row_order = _fixed_legend_order(feature_names)
    ordered_names = [feature_names[i] for i in row_order]
    ordered_matrix = signed_matrix[row_order, :]

    vmax = np.max(np.abs(ordered_matrix)) * 1.05

    fig, ax = plt.subplots(figsize=(max(n_ckpts * 0.9 + 3, 10), input_dim * 0.7 + 2.5))

    im = ax.imshow(ordered_matrix, aspect='auto', cmap='RdBu', vmin=-vmax, vmax=vmax,
                   interpolation='nearest')

    # Annotate cells
    for r in range(len(ordered_names)):
        fname = ordered_names[r]
        expected = EXPECTED_DIRECTIONS.get(fname, None)
        for c in range(n_ckpts):
            val = ordered_matrix[r, c]
            # Text color for readability
            text_color = 'white' if abs(val) > vmax * 0.55 else 'black'
            # Show direction arrow
            fi = row_order[r]
            d = dir_matrix[c, fi]
            arrow = '+' if d > 0 else '−'
            ax.text(c, r, f'{arrow}{abs(val):.2f}', ha='center', va='center',
                    fontsize=8, fontweight='bold', color=text_color)

    ax.set_xticks(range(n_ckpts))
    ax.set_xticklabels([str(i) for i in indices], fontsize=12)
    ax.set_xlabel('Online Training Iteration', fontsize=15, fontweight='bold')

    ax.set_yticks(range(len(ordered_names)))
    ax.set_yticklabels([get_display_name(f) for f in ordered_names], fontsize=13)

    ax.set_title('Perturbation Sensitivity: Direction Correctness × Magnitude',
                 fontsize=18, fontweight='bold', pad=15)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Signed Sensitivity  (blue = correct, red = wrong)', fontsize=12)

    # Add expected direction legend as text below the plot
    legend_parts = []
    for fname in ordered_names:
        expected = EXPECTED_DIRECTIONS.get(fname, None)
        if expected is not None and expected != 0:
            exp_str = '↑ reward' if expected > 0 else '↓ reward'
            legend_parts.append(f'{get_display_name(fname)}: higher → {exp_str}')
    if legend_parts:
        legend_text = 'Expected directions:  ' + '  |  '.join(legend_parts)
        fig.text(0.5, -0.02, legend_text, ha='center', fontsize=9, style='italic',
                 color='#555555', wrap=True)

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    print(f"  Plotted: Perturbation sensitivity heatmap (feature × checkpoint)")


def plot_first_layer_weights(checkpoints, feature_names, pdf, save_dir, norm_matrix):
    """Visualize first-layer weights as a compact feature × checkpoint heatmap.

    Rows = features (fixed order), columns = checkpoints (0, 1, 2, ...).
    Color = L2 norm of that feature's weight column in that checkpoint.
    Annotated with the numeric value in each cell.
    """
    input_dim = len(feature_names)
    n_ckpts = len(checkpoints)
    indices = [c[0] for c in checkpoints]

    # Use fixed legend order for row ordering
    row_order = _fixed_legend_order(feature_names)
    ordered_names = [get_display_name(feature_names[i]) for i in row_order]
    ordered_matrix = np.asarray(norm_matrix).T[row_order, :]

    fig, ax = plt.subplots(figsize=(max(n_ckpts * 0.9 + 3, 10), input_dim * 0.7 + 2))

    im = ax.imshow(ordered_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')

    # Annotate each cell with its value
    for r in range(input_dim):
        for c in range(n_ckpts):
            val = ordered_matrix[r, c]
            text_color = 'white' if val > ordered_matrix.max() * 0.65 else 'black'
            ax.text(c, r, f'{val:.2f}', ha='center', va='center',
                    fontsize=9, fontweight='bold', color=text_color)

    ax.set_xticks(range(n_ckpts))
    ax.set_xticklabels([str(i) for i in indices], fontsize=12)
    ax.set_xlabel('Online Training Iteration', fontsize=15, fontweight='bold')

    ax.set_yticks(range(input_dim))
    ax.set_yticklabels(ordered_names, fontsize=13)

    ax.set_title('First-Layer Weight Magnitude (L2 Norm) per Feature × Checkpoint',
                 fontsize=18, fontweight='bold', pad=15)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('L2 Norm', fontsize=13)

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    print(f"  Plotted: First-layer weight heatmap (feature × checkpoint)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) > 1:
        directory = os.path.abspath(sys.argv[1])
    else:
        directory = os.path.dirname(os.path.abspath(__file__))

    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a directory.")
        sys.exit(1)

    save_dir = os.path.join(directory, 'model_evolution_analysis')
    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading checkpoints from: {directory}")
    checkpoints = load_checkpoints(directory)

    if len(checkpoints) < 1:
        print("No checkpoints found.")
        sys.exit(1)

    input_dim = checkpoints[0][1]['scorer.0.weight'].shape[1]
    hidden_dim = checkpoints[0][1]['scorer.0.weight'].shape[0]

    print(f"Found {len(checkpoints)} checkpoints: {[c[0] for c in checkpoints]}")
    print(f"Architecture: {input_dim} -> {hidden_dim} -> {hidden_dim} -> {hidden_dim} -> 1")

    # Auto-detect feature names
    feature_names = detect_feature_names(directory, input_dim)
    print(f"Features ({len(feature_names)}): {feature_names}")

    if len(feature_names) != input_dim:
        print(f"ERROR: Feature count ({len(feature_names)}) != model input dim ({input_dim}). Using generic names.")
        feature_names = [f'feature_{i}' for i in range(input_dim)]

    # Load actual feature distributions for realistic sampling
    norm_stats = load_normalization_stats(directory, feature_names)
    dist_stats = load_feature_distribution(directory, feature_names)
    if norm_stats and dist_stats:
        realistic_samples = generate_realistic_samples(norm_stats, dist_stats, feature_names, n_samples=1000)
        per_feature_stds = get_per_feature_stds_in_normalized_space(norm_stats, dist_stats, feature_names)
        print(f"  Using realistic samples from actual feature distributions")
        print(f"  Per-feature stds in normalized space: {dict(zip(feature_names, [f'{s:.3f}' for s in per_feature_stds]))}")
    else:
        realistic_samples = None
        per_feature_stds = None
        print(f"  WARNING: No normalization/distribution stats found, falling back to N(0,1) random inputs")

    # Compute all sensitivity metrics for all checkpoints
    grad_matrix = np.zeros((len(checkpoints), input_dim))
    grad_norm_matrix = np.zeros((len(checkpoints), input_dim))
    ig_matrix = np.zeros((len(checkpoints), input_dim))
    sens_matrix = np.zeros((len(checkpoints), input_dim))
    dir_matrix = np.zeros((len(checkpoints), input_dim))
    first_layer_norm_matrix = np.zeros((len(checkpoints), input_dim))

    for ci, (idx, sd) in enumerate(checkpoints):
        model = build_model(sd, input_dim, hidden_dim)
        raw_grad = compute_gradient_sensitivity(model, input_dim, realistic_samples)
        grad_matrix[ci] = raw_grad
        total = raw_grad.sum()
        grad_norm_matrix[ci] = raw_grad / total if total > 0 else raw_grad
        ig_matrix[ci] = compute_integrated_gradients(model, input_dim, realistic_samples)
        sens_matrix[ci], dir_matrix[ci] = compute_perturbation_sensitivity(
            model, input_dim, per_feature_stds, realistic_samples)
        first_layer_norm_matrix[ci] = np.linalg.norm(sd['scorer.0.weight'].numpy(), axis=0)
        print(f"  Checkpoint {idx}: grad_sum={total:.3f}")

    # Generate PDF
    output_pdf = os.path.join(save_dir, 'model_evolution_analysis.pdf')
    print(f"\nGenerating analysis -> {output_pdf}")

    plot_checkpoints, aggregated = aggregate_for_plotting(
        checkpoints,
        grad_matrix,
        grad_norm_matrix,
        ig_matrix,
        sens_matrix,
        dir_matrix,
        first_layer_norm_matrix,
        window=PLOT_AGGREGATE_WINDOW
    )
    (grad_plot_matrix,
     grad_norm_plot_matrix,
     ig_plot_matrix,
     sens_plot_matrix,
     dir_plot_matrix,
     first_layer_plot_matrix) = aggregated
    if PLOT_AGGREGATE_WINDOW > 1 and len(checkpoints) > 1:
        print(f"  Plot smoothing: averaging every {PLOT_AGGREGATE_WINDOW} checkpoints "
              f"({len(checkpoints)} -> {len(plot_checkpoints)} plotted points)")

    with PdfPages(output_pdf) as pdf:
        if len(plot_checkpoints) >= 2:
            plot_gradient_evolution(plot_checkpoints, grad_plot_matrix, feature_names, pdf, save_dir,
                                   title_suffix=' (Raw)', ylabel_prefix='Raw ')
            plot_gradient_evolution(plot_checkpoints, grad_norm_plot_matrix, feature_names, pdf, save_dir,
                                   title_suffix=' (Normalized)', ylabel_prefix='Relative ')
            plot_gradient_evolution(plot_checkpoints, ig_plot_matrix, feature_names, pdf, save_dir,
                                   title_suffix=' (Integrated Gradients)', ylabel_prefix='Relative ')
        plot_perturbation_sensitivity(plot_checkpoints, sens_plot_matrix, dir_plot_matrix, feature_names, pdf, save_dir)
        plot_first_layer_weights(plot_checkpoints, feature_names, pdf, save_dir, first_layer_plot_matrix)

    # Save CSVs
    csv_path = os.path.join(save_dir, 'gradient_sensitivity.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['iteration']
                    + [f'{fn}_grad_raw' for fn in feature_names]
                    + [f'{fn}_grad_normalized' for fn in feature_names]
                    + [f'{fn}_integrated_gradients' for fn in feature_names])
        for ci, (idx, _) in enumerate(checkpoints):
            w.writerow([idx] + list(grad_matrix[ci]) + list(grad_norm_matrix[ci]) + list(ig_matrix[ci]))

    csv_path = os.path.join(save_dir, 'perturbation_sensitivity.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['iteration'] + [f'{fn}_sensitivity' for fn in feature_names] + [f'{fn}_direction' for fn in feature_names])
        for ci, (idx, _) in enumerate(checkpoints):
            w.writerow([idx] + list(sens_matrix[ci]) + list(dir_matrix[ci]))

    print(f"\nDone! CSVs: {save_dir}/gradient_sensitivity.csv, perturbation_sensitivity.csv")


if __name__ == '__main__':
    main()
