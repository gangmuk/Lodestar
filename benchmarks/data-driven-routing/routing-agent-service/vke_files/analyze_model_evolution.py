#!/usr/bin/env python3
"""
Rigorous model evolution analysis for online learning checkpoints.

Each checkpoint is independently trained from scratch (Xavier init) on a
cumulative replay buffer. We analyze what features the reward model learns
to value as more online data arrives.

Methods & References:
  - Connection Weights importance (Olden et al., Ecological Modelling, 2004)
Output:
  model_evolution_analysis/model_evolution_analysis.pdf  (1 page)
  model_evolution_analysis/*.csv  (all numerical results)
"""

import os
import sys
import glob
import re
import csv
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

# Global font size settings
plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
})

# ── Constants ────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    'inflight_requests', 'inflight_prefill_reqs', 'inflight_decode_reqs',
    'gpu_kv_cache', 'cpu_kv_cache', 'running_requests', 'waiting_requests',
    'prefill_tokens', 'decode_tokens', 'staleness',
    'kv_hit_ratio',
    'input_tokens', 'total_tokens',
]

# Distinct color per feature (tab20 colormap)
_TAB20 = plt.cm.tab20(np.linspace(0, 1, len(FEATURE_NAMES)))
FEATURE_COLORS = [_TAB20[i] for i in range(len(FEATURE_NAMES))]


# ── Data Loading ─────────────────────────────────────────────────────────────

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
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


# ── Feature Importance Methods ───────────────────────────────────────────────

def compute_connection_weights(state_dict):
    """
    Connection Weights importance (Olden et al., 2004).
    Propagates through all 4 layers: importance_i = (|W8| @ |W6| @ |W3| @ |W0|)[0, i]
    """
    W0 = state_dict['scorer.0.weight'].abs().float()  # [128, 13]
    W3 = state_dict['scorer.3.weight'].abs().float()  # [128, 128]
    W6 = state_dict['scorer.6.weight'].abs().float()  # [128, 128]
    W8 = state_dict['scorer.8.weight'].abs().float()  # [1, 128]
    cw = (W8 @ W6 @ W3 @ W0).squeeze().numpy()        # [13]
    return cw



# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: Feature Importance Evolution with Direct Line Labels
# ══════════════════════════════════════════════════════════════════════════════

def plot_importance_evolution(checkpoints, cw_norm, pdf, save_dir):
    """Line plot of normalized importance over iterations, with direct labels."""
    indices = np.array([c[0] for c in checkpoints])
    k = len(FEATURE_NAMES)

    fig, ax = plt.subplots(figsize=(16, 9))

    mean_imp = cw_norm.mean(axis=0)
    order = np.argsort(-mean_imp)

    lines_for_legend = []
    for fi in order:
        vals = cw_norm[:, fi]
        line, = ax.plot(indices, vals, 'o-', color=FEATURE_COLORS[fi],
                        linewidth=2, markersize=5, alpha=0.85,
                        label=FEATURE_NAMES[fi])
        lines_for_legend.append(line)

    # Uniform baseline
    uniform = 1.0 / k
    baseline_line = ax.axhline(y=uniform, color='red', linestyle='--', linewidth=1.5, alpha=0.6,
                                label=f'Uniform = {uniform:.3f}')

    ax.set_xlabel('Online Training Iteration (growing replay buffer)', fontsize=14)
    ax.set_ylabel('Normalized Importance (Connection Weights)', fontsize=14)
    ax.set_title('Feature Importance Evolution Across Online Learning Iterations',
                 fontsize=16, fontweight='bold')
    ax.set_xticks(indices)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(indices[0] - 0.5, indices[-1] + 0.5)

    # Legend outside on the right
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=11,
              frameon=True, fancybox=True, shadow=False, ncol=1)

    fig.subplots_adjust(bottom=0.1, right=0.78)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()
    print("  [1/1] Feature importance evolution")

    # Save CSV
    path = os.path.join(save_dir, 'importance_values.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['iteration'] +
                    [f'{fn}_cw_norm' for fn in FEATURE_NAMES])
        for i, idx in enumerate([c[0] for c in checkpoints]):
            w.writerow([idx] + list(cw_norm[i]))



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

    if len(checkpoints) < 2:
        print("Need at least 2 checkpoints.")
        sys.exit(1)

    input_dim = checkpoints[0][1]['scorer.0.weight'].shape[1]
    hidden_dim = checkpoints[0][1]['scorer.0.weight'].shape[0]
    if input_dim != len(FEATURE_NAMES):
        print(f"WARNING: Expected {len(FEATURE_NAMES)} features but model has {input_dim}.")

    print(f"Found {len(checkpoints)} checkpoints: {[c[0] for c in checkpoints]}")
    print(f"Architecture: {input_dim} -> {hidden_dim} -> {hidden_dim} -> {hidden_dim} -> 1")

    # Compute CW importance for all checkpoints
    cw_matrix = np.array([compute_connection_weights(c[1]) for c in checkpoints])
    cw_norm = cw_matrix / cw_matrix.sum(axis=1, keepdims=True)

    output_pdf = os.path.join(save_dir, 'model_evolution_analysis.pdf')
    print(f"\nGenerating 1-page analysis -> {output_pdf}")

    with PdfPages(output_pdf) as pdf:
        plot_importance_evolution(checkpoints, cw_norm, pdf, save_dir)

    print(f"\nDone! {output_pdf}")


if __name__ == '__main__':
    main()
