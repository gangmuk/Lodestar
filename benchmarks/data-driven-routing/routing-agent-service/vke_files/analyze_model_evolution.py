#!/usr/bin/env python3
"""
Model evolution analysis for online learning checkpoints.

Analyzes what features the reward model learns to value as more online data
arrives, using gradient-based sensitivity analysis (not connection weights,
which is misleading for deep ReLU networks).

Methods:
  1. Gradient Sensitivity: avg |dOutput/dInput| over random samples
  2. Perturbation Sensitivity: output change when each feature moves ±1 std

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

def compute_gradient_sensitivity(model, input_dim, n_samples=1000):
    """Average |dOutput/dInput| over random samples. Returns raw (unnormalized) values."""
    x = torch.randn(n_samples, input_dim, requires_grad=True)
    out = model(x)
    out.sum().backward()
    return x.grad.abs().mean(dim=0).detach().numpy()


def compute_integrated_gradients(model, input_dim, n_samples=200, n_steps=50):
    """Integrated Gradients (Sundararajan et al., ICML 2017) with zero baseline.
    Returns normalized relative importance (sums to 1)."""
    baseline = torch.zeros(1, input_dim)
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


def compute_perturbation_sensitivity(model, input_dim):
    """Output change when each feature moves ±1 std from zero baseline.
    Returns (sensitivity, direction) where direction[i] = +1 if higher feature → higher reward."""
    baseline = torch.zeros(1, input_dim)
    sensitivity = np.zeros(input_dim)
    direction = np.zeros(input_dim)
    for i in range(input_dim):
        with torch.no_grad():
            p_plus = baseline.clone(); p_plus[0, i] = 1.0
            p_minus = baseline.clone(); p_minus[0, i] = -1.0
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
        lw = 3.0 if fname in HIGHLIGHT_FEATURES else 1.5
        alpha = 1.0 if fname in HIGHLIGHT_FEATURES else 0.5
        ms = 7 if fname in HIGHLIGHT_FEATURES else 4
        zorder = 10 if fname in HIGHLIGHT_FEATURES else 1
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


def plot_perturbation_sensitivity(checkpoints, sens_matrix, dir_matrix, feature_names, pdf, save_dir):
    """Perturbation sensitivity bar chart. Shows first vs last when multiple checkpoints, single panel otherwise."""
    if len(checkpoints) == 1:
        panels = [(0, checkpoints[0][0])]
        fig, axes = plt.subplots(1, 1, figsize=(12, 9))
        axes = [axes]
        suptitle = 'Learned Feature Sensitivity (Single Checkpoint)'
    else:
        first_idx, last_idx = 0, len(checkpoints) - 1
        panels = [(first_idx, checkpoints[first_idx][0]), (last_idx, checkpoints[last_idx][0])]
        fig, axes = plt.subplots(1, 2, figsize=(20, 9), sharey=True)
        suptitle = 'Learned Feature Sensitivity: Early vs Late Training'

    for ax_idx, (ckpt_idx, iter_num) in enumerate(panels):
        ax = axes[ax_idx]
        sens = sens_matrix[ckpt_idx]
        dirs = dir_matrix[ckpt_idx]

        # Sort by sensitivity
        order = np.argsort(-sens)
        sorted_names = [feature_names[i] for i in order]
        sorted_sens = sens[order]
        sorted_dirs = dirs[order]

        colors = [get_feature_color(fname, feature_names) for fname in sorted_names]

        ax.barh(range(len(sorted_names)), sorted_sens, color=colors, edgecolor='white', linewidth=0.5, height=0.7)

        # Add direction arrows and values
        for i, (s, d, fname) in enumerate(zip(sorted_sens, sorted_dirs, sorted_names)):
            arrow = '  (+)' if d > 0 else '  (-)'
            expected = EXPECTED_DIRECTIONS.get(fname, None)
            if expected is not None and expected != 0:
                correct = (d > 0) == (expected > 0)
                marker = ' ✓' if correct else ' ✗'
            else:
                marker = ''
            ax.text(s + 0.02, i, f'{s:.3f}{arrow}{marker}', va='center', fontsize=12, fontweight='bold' if fname in HIGHLIGHT_FEATURES else 'normal')

        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels([get_display_name(n) for n in sorted_names], fontsize=13)
        # Bold the highlighted features in y-tick labels
        for tick_label in ax.get_yticklabels():
            text = tick_label.get_text()
            for hf in HIGHLIGHT_FEATURES:
                if get_display_name(hf) == text:
                    tick_label.set_fontweight('bold')
                    break

        ax.set_xlabel('Perturbation Sensitivity  (|output(+1σ) - output(-1σ)|)', fontsize=14, fontweight='bold')
        ax.set_title(f'Iteration {iter_num}', fontsize=18, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.2)
        ax.invert_yaxis()

    fig.suptitle(suptitle, fontsize=22, fontweight='bold', y=1.02)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    n_pages = 1 if len(checkpoints) == 1 else 2
    print(f"  [{n_pages}/{n_pages}] Perturbation sensitivity")


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

    # Compute all sensitivity metrics for all checkpoints
    grad_matrix = np.zeros((len(checkpoints), input_dim))
    grad_norm_matrix = np.zeros((len(checkpoints), input_dim))
    ig_matrix = np.zeros((len(checkpoints), input_dim))
    sens_matrix = np.zeros((len(checkpoints), input_dim))
    dir_matrix = np.zeros((len(checkpoints), input_dim))

    for ci, (idx, sd) in enumerate(checkpoints):
        model = build_model(sd, input_dim, hidden_dim)
        raw_grad = compute_gradient_sensitivity(model, input_dim)
        grad_matrix[ci] = raw_grad
        total = raw_grad.sum()
        grad_norm_matrix[ci] = raw_grad / total if total > 0 else raw_grad
        ig_matrix[ci] = compute_integrated_gradients(model, input_dim)
        sens_matrix[ci], dir_matrix[ci] = compute_perturbation_sensitivity(model, input_dim)
        print(f"  Checkpoint {idx}: grad_sum={total:.3f}")

    # Generate PDF
    output_pdf = os.path.join(save_dir, 'model_evolution_analysis.pdf')
    print(f"\nGenerating analysis -> {output_pdf}")

    with PdfPages(output_pdf) as pdf:
        if len(checkpoints) >= 2:
            plot_gradient_evolution(checkpoints, grad_matrix, feature_names, pdf, save_dir,
                                   title_suffix=' (Raw)', ylabel_prefix='Raw ')
            plot_gradient_evolution(checkpoints, grad_norm_matrix, feature_names, pdf, save_dir,
                                   title_suffix=' (Normalized)', ylabel_prefix='Relative ')
            plot_gradient_evolution(checkpoints, ig_matrix, feature_names, pdf, save_dir,
                                   title_suffix=' (Integrated Gradients)', ylabel_prefix='Relative ')
        plot_perturbation_sensitivity(checkpoints, sens_matrix, dir_matrix, feature_names, pdf, save_dir)

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
