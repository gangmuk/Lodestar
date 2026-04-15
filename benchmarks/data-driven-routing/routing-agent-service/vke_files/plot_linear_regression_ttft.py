#!/usr/bin/env python3
import argparse
"""
Linear regression AND neural network reward prediction comparison.

Reads model_config.json to determine:
  - Which pod/request features to use (via EXCLUDED_POD_FEATURES / EXCLUDED_REQUEST_FEATURES)
  - Which reward function to apply (REWARD_FUNCTION, LATENCY_METRIC, TTFT_REWARD_WEIGHT)

Loads the trained neural network (reward_net.pth) and uses the ACTUAL pipeline modules
(FeatureStats.load_from_csv / RunningStats.normalize from data_normalizer.py) to ensure
normalization is EXACTLY identical to training/inference:
  1. Load normalization stats via FeatureStats.load_from_csv()
  2. Normalize each feature via RunningStats.normalize() (pooled stats for pod features)
  3. Build per-pod context: [9 pod features | 1 staleness | 1 kv_hit_ratio | N request features]
  4. Forward pass through RewardNetwork

Plots ground truth vs predicted reward for both models side by side.

Usage:
    python plot_linear_regression_ttft.py <path_to_data-processed.csv> [output.png]

Expects in the same directory as the CSV:
  - model_config.json
  - feature_normalization_statistics.csv
  - reward_net.pth
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.size': 22,
    'axes.titlesize': 24,
    'axes.labelsize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
})

import torch
import torch.nn as nn

# Add agent_codes/ to sys.path so we can import the actual pipeline modules
_AGENT_CODES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'agent_codes')
if _AGENT_CODES_DIR not in sys.path:
    sys.path.insert(0, _AGENT_CODES_DIR)

# Suppress noisy per-sample warnings from data_normalizer (e.g., zero-std features)
import logging
logging.getLogger("llm_router").setLevel(logging.ERROR)

from data_normalizer import FeatureStats

# ---------------------------------------------------------------------------
# All possible features (matching neural_contextual_bandit_perpodmodel_checkpoint.py)
# ---------------------------------------------------------------------------
ALL_POD_FEATURES = [
    'inflight_requests', 'inflight_prefill_requests', 'inflight_decode_requests',
    'gpu_kv_cache', 'cpu_kv_cache', 'running_requests', 'waiting_requests',
    'prefill_tokens', 'decode_tokens', 'kv_hit_ratio'
]
ALL_REQUEST_FEATURES = ['input_tokens', 'output_tokens', 'total_tokens']


# ---------------------------------------------------------------------------
# RewardNetwork (copied from neural_contextual_bandit_perpodmodel_checkpoint.py)
# ---------------------------------------------------------------------------
class RewardNetwork(nn.Module):
    def __init__(self, per_pod_context_dim, hidden_dim=128):
        super().__init__()
        self.per_pod_context_dim = per_pod_context_dim
        self.scorer = nn.Sequential(
            nn.Linear(per_pod_context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, context):
        return self.scorer(context)


# ---------------------------------------------------------------------------
# Reward functions (from preprocess.py)
# ---------------------------------------------------------------------------
def compute_reward(df, config):
    """Compute reward column based on model_config.json settings."""
    reward_function = config['REWARD_FUNCTION']
    latency_metric = config.get('LATENCY_METRIC', 'ttft')
    ttft_reward_weight = config.get('TTFT_REWARD_WEIGHT', 1.0)

    ttft = df['ttft'].values.astype(np.float64)
    tpot = df['avg_tpot'].values.astype(np.float64)

    if latency_metric == 'e2e_latency':
        latency = df['e2e_latency'].values.astype(np.float64)
    else:
        latency = None

    def combine(ttft_r, tpot_r):
        return ttft_reward_weight * ttft_r + max(0, (1 - ttft_reward_weight)) * tpot_r

    if reward_function == 'negative_linear':
        if latency_metric == 'e2e_latency':
            return -latency / 1000.0
        return combine(-ttft / 1000.0, -tpot / 1000.0)

    elif reward_function == 'negative_squared':
        if latency_metric == 'e2e_latency':
            return -np.square(latency / 1000.0)
        return combine(-np.square(ttft / 1000.0), -np.square(tpot / 1000.0))

    elif reward_function == 'negative_reciprocal':
        if latency_metric == 'e2e_latency':
            return -1000.0 / np.maximum(latency, 1.0)
        return combine(-1000.0 / np.maximum(ttft, 1.0), -1000.0 / np.maximum(tpot, 1.0))

    elif reward_function == 'simple_latency_minimization':
        if latency_metric == 'e2e_latency':
            return -np.log(latency + 1.0)
        return combine(-np.log(ttft + 1.0), -np.log(tpot + 1.0))

    elif reward_function == 'linear_simple':
        ttft_slo = config.get('TTFT_SLO', 1000)
        avg_tpot_slo = config.get('AVG_TPOT_SLO', 50)
        ttft_r = np.where(ttft <= 0, 0.5,
                 np.where(ttft <= ttft_slo,
                          0.5 - (0.4 * ttft / ttft_slo),
                          -0.1 - (0.4 * np.minimum(1.0, (ttft - ttft_slo) / ttft_slo))))
        tpot_r = np.where(tpot <= 0, -0.5,
                 np.where(tpot <= avg_tpot_slo,
                          0.1 + (0.4 * (1 - tpot / avg_tpot_slo)),
                          -0.1 - (0.4 * np.minimum(1.0, (tpot - avg_tpot_slo) / avg_tpot_slo))))
        return combine(ttft_r, tpot_r)

    else:
        if 'reward' in df.columns:
            print(f"  Unknown reward function '{reward_function}', using 'reward' column from CSV")
            return df['reward'].values.astype(np.float64)
        print(f"  Unknown reward function '{reward_function}', falling back to -ttft/1000")
        return -ttft / 1000.0


def load_config(csv_path):
    """Load model_config.json from the same directory as the CSV."""
    config_path = os.path.join(os.path.dirname(csv_path), 'model_config.json')
    if not os.path.exists(config_path):
        print(f"Warning: {config_path} not found, using defaults")
        return {
            'REWARD_FUNCTION': 'negative_linear',
            'LATENCY_METRIC': 'ttft',
            'TTFT_REWARD_WEIGHT': 1.0,
            'EXCLUDED_POD_FEATURES': ['none'],
            'EXCLUDED_REQUEST_FEATURES': ['output_tokens', 'total_tokens'],
        }
    with open(config_path) as f:
        config = json.load(f)
    print(f"Loaded config from {config_path}")
    return config


def load_feature_stats(model_dir):
    """Load feature normalization statistics using the ACTUAL FeatureStats.load_from_csv().

    Returns a FeatureStats instance — the exact same object used during training/inference.
    """
    stats_path = os.path.join(model_dir, 'feature_normalization_statistics.csv')
    if not os.path.exists(stats_path):
        print(f"Warning: {stats_path} not found, skipping normalization")
        return None
    stats_instance = FeatureStats.load_from_csv(stats_path)
    if stats_instance is None:
        print(f"ERROR: Failed to load FeatureStats from {stats_path}")
        return None
    print(f"Loaded FeatureStats from {stats_path} with {len(stats_instance.feature_stats)} feature types")
    for feat_name, rs in stats_instance.feature_stats.items():
        print(f"  {feat_name}: mean={rs.mean[0]:.4f}, std={rs.std[0]:.4f}, count={rs.count}")
    return stats_instance


def normalize_feature_value(value, feat_name, feature_stats):
    """Normalize a single value using the ACTUAL RunningStats.normalize() from data_normalizer.

    This calls the exact same RunningStats.normalize() used during training:
      - If std == 0 or close to 0: returns 0.0 (np.zeros_like)
      - If NaN in std: returns 0.0
      - Otherwise: (value - mean) / std
    """
    if feature_stats is None or feat_name not in feature_stats.feature_stats:
        return value
    rs = feature_stats.feature_stats[feat_name]
    # Use the actual RunningStats.normalize() — exactly matching training pipeline
    data = np.array([[value]], dtype=np.float64)
    normalized = rs.normalize(data)
    return float(normalized[0, 0])


def get_feature_lists(config):
    """Determine pod and request features based on config exclusions."""
    excluded_pod = set(config.get('EXCLUDED_POD_FEATURES', []))
    if 'none' in excluded_pod or 'None' in excluded_pod:
        excluded_pod = set()

    excluded_req = set(config.get('EXCLUDED_REQUEST_FEATURES', []))
    if 'none' in excluded_req or 'None' in excluded_req:
        excluded_req = set()

    pod_features = [f for f in ALL_POD_FEATURES if f not in excluded_pod]
    request_features = [f for f in ALL_REQUEST_FEATURES if f not in excluded_req]
    return pod_features, request_features


def load_and_build_features(csv_path, config):
    """Load CSV and build per-sample feature vectors using the selected pod's features.

    Returns raw (unnormalized) features for linear regression,
    plus the dataframe and metadata needed for neural net inference.
    """
    df = pd.read_csv(csv_path)

    pod_features, request_features = get_feature_lists(config)
    print(f"Pod features ({len(pod_features)}): {pod_features}")
    print(f"Request features ({len(request_features)}): {request_features}")

    pod_ids = sorted(set(
        col.rsplit('-', 1)[0]
        for col in df.columns
        if col.startswith('pod_') and '-' in col
    ))
    print(f"Found {len(pod_ids)} pods: {pod_ids}")

    reward = compute_reward(df, config)
    reward_func = config.get('REWARD_FUNCTION', 'negative_linear')
    latency_metric = config.get('LATENCY_METRIC', 'ttft')
    print(f"Reward function: {reward_func} (latency_metric={latency_metric})")
    print(f"Reward range: [{reward.min():.4f}, {reward.max():.4f}]")

    feature_rows = []
    targets = []
    selected_pods_list = []
    valid_indices = []
    skipped = 0

    for idx, (_, row) in enumerate(df.iterrows()):
        selected_pod = row['selected_pod']
        feat = []
        skip = False
        for f in pod_features:
            col = f"{selected_pod}-{f}"
            if col not in df.columns:
                skip = True
                break
            feat.append(row[col])
        if skip:
            skipped += 1
            continue
        for f in request_features:
            feat.append(row[f])
        feature_rows.append(feat)
        targets.append(reward[idx])
        selected_pods_list.append(selected_pod)
        valid_indices.append(idx)

    if skipped:
        print(f"Skipped {skipped} rows due to missing pod columns")

    feature_names = [f"pod_{f}" for f in pod_features] + request_features
    X = np.array(feature_rows)
    y = np.array(targets)
    return X, y, feature_names, df, valid_indices, selected_pods_list


def build_nn_input(df, valid_indices, selected_pods_list, config, feature_stats):
    """Build normalized input tensors for the neural network.

    Uses the ACTUAL FeatureStats / RunningStats.normalize() from data_normalizer.py,
    ensuring EXACTLY the same normalization as training/inference.

    Encoding pipeline (matching encoding.py):
      1. For each sample, get the selected pod's features
      2. Normalize each feature using RunningStats.normalize() with pooled stats
      3. Concatenate: [pod_features_without_kv(9) | staleness(1) | kv_hit_ratio(1) | request_features(N)]
         Total = 9 + 1 + 1 + N = 13 (with N=2 request features)

    The staleness feature is always 0 (encoding.py line 895).
    """
    pod_features_all, request_features_all = get_feature_lists(config)

    # Separate kv_hit_ratio from pod features (encoding.py extracts it separately)
    pod_features_no_kv = [f for f in pod_features_all if f != 'kv_hit_ratio']
    has_kv = 'kv_hit_ratio' in pod_features_all

    nn_inputs = []
    for i, idx in enumerate(valid_indices):
        row = df.iloc[idx]
        selected_pod = selected_pods_list[i]

        # Pod features (without kv_hit_ratio), normalized via actual RunningStats.normalize()
        feat = []
        for f in pod_features_no_kv:
            col = f"{selected_pod}-{f}"
            raw_val = row[col]
            feat.append(normalize_feature_value(raw_val, f, feature_stats))

        # Staleness feature (always 0, appended in encoding.py line 896)
        feat.append(0.0)

        # KV hit ratio, normalized (separate channel in encoding)
        if has_kv:
            col = f"{selected_pod}-kv_hit_ratio"
            raw_val = row[col]
            feat.append(normalize_feature_value(raw_val, 'kv_hit_ratio', feature_stats))

        # Request features, normalized via actual RunningStats.normalize()
        for f in request_features_all:
            feat.append(normalize_feature_value(row[f], f, feature_stats))

        nn_inputs.append(feat)

    context_dim = len(pod_features_no_kv) + 1 + (1 if has_kv else 0) + len(request_features_all)
    nn_feature_names = [f"pod_{f}" for f in pod_features_no_kv]
    nn_feature_names.append("staleness")
    if has_kv:
        nn_feature_names.append("pod_kv_hit_ratio")
    nn_feature_names += request_features_all

    print(f"\nNeural net input: {context_dim} dims = "
          f"{len(pod_features_no_kv)} pod + 1 staleness + {1 if has_kv else 0} kv + {len(request_features_all)} request")
    print(f"NN feature order: {nn_feature_names}")

    return np.array(nn_inputs, dtype=np.float32), context_dim


def load_reward_net(model_dir, context_dim, hidden_dim):
    """Load trained RewardNetwork from reward_net.pth."""
    pth_path = os.path.join(model_dir, 'reward_net.pth')
    if not os.path.exists(pth_path):
        print(f"ERROR: {pth_path} not found")
        return None

    model = RewardNetwork(context_dim, hidden_dim)
    state_dict = torch.load(pth_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded RewardNetwork from {pth_path} (context_dim={context_dim}, hidden_dim={hidden_dim})")
    return model


def main():
    parser = argparse.ArgumentParser(description='Linear regression + NN reward prediction plot.')
    parser.add_argument('csv_path', help='Path to data-processed.csv')
    parser.add_argument('--axis-limit', type=float, default=-2.0, help='Axis limit (default: -2.0)')
    args = parser.parse_args()

    csv_path = args.csv_path
    axis_limit = args.axis_limit
    model_dir = os.path.dirname(csv_path)

    config = load_config(csv_path)
    feature_stats = load_feature_stats(model_dir)

    print(f"\nLoading data from {csv_path}")
    X, y, feature_names, df, valid_indices, selected_pods_list = load_and_build_features(csv_path, config)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Train linear regression on 60%, evaluate and plot on the 40% test set only.
    indices = np.arange(len(y))
    idx_train, idx_test = train_test_split(indices, test_size=0.4, random_state=42)
    X_train, y_train = X[idx_train], y[idx_train]
    X_test, y_test = X[idx_test], y[idx_test]
    print(f"Linear regression trained on {len(idx_train)} samples, evaluated on {len(idx_test)} test samples")

    # ---- Linear Regression ----
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)  # predict on test data only

    r2_lr = r2_score(y_test, y_pred_lr)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    print(f"\n--- Linear Regression (test data) ---")
    print(f"  R²:   {r2_lr:.4f}")
    print(f"  MAE:  {mae_lr:.4f}")
    print(f"  RMSE: {rmse_lr:.4f}")

    print(f"\n  Feature coefficients:")
    for name, coef in sorted(zip(feature_names, lr_model.coef_), key=lambda x: abs(x[1]), reverse=True):
        print(f"    {name:>30s}: {coef:+.6f}")
    print(f"    {'intercept':>30s}: {lr_model.intercept_:+.6f}")

    # ---- Neural Network (evaluate on test set only) ----
    nn_X, context_dim = build_nn_input(df, valid_indices, selected_pods_list, config, feature_stats)
    nn_X_test = nn_X[idx_test]
    hidden_dim = config.get('hidden_dim', 128)
    reward_net = load_reward_net(model_dir, context_dim, hidden_dim)

    y_pred_nn = None
    r2_nn = mae_nn = rmse_nn = None
    if reward_net is not None:
        with torch.no_grad():
            nn_input = torch.from_numpy(nn_X_test).float()  # test data only
            nn_output = reward_net(nn_input).squeeze(-1).numpy()
        y_pred_nn = nn_output

        r2_nn = r2_score(y_test, y_pred_nn)
        mae_nn = mean_absolute_error(y_test, y_pred_nn)
        rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred_nn))
        print(f"\n--- Neural Network (test data) ---")
        print(f"  R²:   {r2_nn:.4f}")
        print(f"  MAE:  {mae_nn:.4f}")
        print(f"  RMSE: {rmse_nn:.4f}")

    # ---- Side-by-side plot ----
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    for ax, y_pred, color, title, r2, rmse in [
        (ax1, y_pred_lr, '#4878CF', 'Linear Regression', r2_lr, rmse_lr),
        (ax2, y_pred_nn, '#E8801B', 'Neural Network', r2_nn, rmse_nn),
    ]:
        if y_pred is None:
            continue
        ax.scatter(y_test, y_pred, alpha=0.35, s=12, color=color, edgecolors='none', rasterized=True)
        ax.plot([axis_limit, 0], [axis_limit, 0], color='#333333', linestyle='--', linewidth=1.2,
                zorder=5)
        ax.set_xlabel('Ground Truth Reward')
        ax.set_ylabel('Predicted Reward')
        ax.set_title(title)
        ax.set_xlim(axis_limit, 0)
        ax.set_ylim(axis_limit, 0)
        ax.set_aspect('equal')
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.grid(True, alpha=0.2, linewidth=0.5)

    fig2.tight_layout()

    sidebyside_path = 'reward_prediction_scatter_plot.pdf'
    fig2.savefig(sidebyside_path, dpi=200)
    print(f"Saved side-by-side plot to {sidebyside_path}")
    plt.close()

    # ---- Side-by-side hexbin density plot (shared color scale) ----
    from matplotlib.colors import LogNorm
    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)

    # First pass: draw hexbins to find global count range
    hexbins = []
    axes_list = []
    for ax, y_pred, title, r2, rmse in [
        (ax1, y_pred_lr, 'Linear Regression', r2_lr, rmse_lr),
        (ax2, y_pred_nn, 'Neural Network', r2_nn, rmse_nn),
    ]:
        if y_pred is None:
            continue
        hb = ax.hexbin(y_test, y_pred, gridsize=40, cmap='YlOrRd', mincnt=1,
                        extent=[axis_limit, 0, axis_limit, 0],
                        rasterized=True)
        hexbins.append(hb)
        axes_list.append((ax, title))

    # Compute shared vmin/vmax across both panels
    global_vmin = min(hb.get_array().min() for hb in hexbins)
    global_vmax = max(hb.get_array().max() for hb in hexbins)
    shared_norm = LogNorm(vmin=global_vmin, vmax=global_vmax)

    # Apply shared norm to all hexbins
    for hb in hexbins:
        hb.set_norm(shared_norm)

    for ax, title in axes_list:
        ax.plot([axis_limit, 0], [axis_limit, 0], color='#333333', linestyle='--', linewidth=1.2,
                zorder=5)
        ax.set_xlabel('Ground Truth Reward')
        ax.set_ylabel('Predicted Reward')
        ax.set_title(title)
        ax.set_xlim(axis_limit, 0)
        ax.set_ylim(axis_limit, 0)
        ax.set_aspect('equal')
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.grid(True, alpha=0.2, linewidth=0.5)

    # Single shared colorbar
    cb = fig3.colorbar(hexbins[0], ax=[ax1, ax2], shrink=0.8)
    cb.set_label('Count')

    hexbin_path = 'reward_prediction_hexbin.pdf'
    fig3.savefig(hexbin_path, dpi=200)
    print(f"Saved hexbin plot to {hexbin_path}")
    plt.close()

    # Save per-sample predictions to CSV
    pred_csv_path = 'reward_predictions.csv'
    pred_df = pd.DataFrame({
        'ground_truth_reward': y_test,
        'linear_regression_pred': y_pred_lr,
        'neural_network_pred': y_pred_nn if y_pred_nn is not None else np.nan,
    })
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"Saved predictions to {pred_csv_path}")


if __name__ == '__main__':
    main()
