"""
Gradient-Based Replay Buffer for Online Training

Implements GCR (Gradient Coreset Replay) approach for continual learning:
- Gradient feature extraction using last hidden layer activations
- Facility location selection for diverse subset
- Old Transfer monitoring for catastrophic forgetting detection
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from logger import logger


def compute_gradient_features(model, tensor_data, batch_size=512):
    """
    Compute proxy gradient features (128-dim last hidden layer activations)
    weighted by loss residual magnitude for replay buffer selection.

    Args:
        model: NeuralContextualBandit instance with reward_net
        tensor_data: dict with 'pod_features_with_staleness', 'kv_hit_ratios',
                     'request_features', 'actions', 'rewards'
        batch_size: processing batch size to limit memory

    Returns:
        tuple: (weighted_activations [N, hidden_dim], diagnostics dict)
    """
    pod_features = tensor_data['pod_features_with_staleness']
    kv_hit_ratios = tensor_data['kv_hit_ratios']
    request_features = tensor_data['request_features']
    actions = tensor_data['actions'].long()  # BUG FIX: ensure int64 for gather()
    rewards = tensor_data['rewards']

    N = len(actions)
    pod_dim = pod_features.shape[2]
    kv_dim = kv_hit_ratios.shape[2]

    # Build per-selected-action contexts (not all pods)
    selected_pod_feats = pod_features.gather(
        1, actions.view(-1, 1, 1).expand(-1, 1, pod_dim)
    ).squeeze(1)  # [N, pod_dim]
    selected_kv = kv_hit_ratios.gather(
        1, actions.view(-1, 1, 1).expand(-1, 1, kv_dim)
    ).squeeze(1)  # [N, kv_dim]
    contexts = torch.cat([selected_pod_feats, selected_kv, request_features], dim=1)  # [N, per_pod_context_dim]

    # Use model's activation extraction method
    activations, predictions = model.extract_activations_and_predictions(contexts, batch_size=batch_size)

    # Weight activations by |predicted - actual| (loss residual magnitude)
    actual = rewards.cpu().numpy()
    residuals = np.abs(predictions - actual)
    # Expand residuals for broadcasting: [N, 1]
    weighted_activations = activations * residuals[:, np.newaxis]

    # Collect diagnostics for logging
    diagnostics = {
        'predictions': predictions,
        'actual_rewards': actual,
        'residuals': residuals,
        'selected_kv': selected_kv.cpu().numpy(),
        'activation_norms': np.linalg.norm(weighted_activations, axis=1),
    }

    return weighted_activations, diagnostics


def select_replay_samples(gradient_features, replay_size, seed=42):
    """
    Greedy k-center facility location for diverse subset selection.

    L2-normalizes features, then iteratively picks samples that maximize
    minimum distance to already-selected set.

    Args:
        gradient_features: [N, D] numpy array of gradient features
        replay_size: number of samples to select
        seed: random seed for tie-breaking

    Returns:
        np.ndarray of selected indices
    """
    N, D = gradient_features.shape
    if N <= replay_size:
        return np.arange(N)

    rng = np.random.RandomState(seed)

    # L2-normalize rows
    norms = np.linalg.norm(gradient_features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normalized = gradient_features / norms

    # Pick first sample: highest L2 norm (most informative)
    first_idx = np.argmax(norms.squeeze())
    selected = [int(first_idx)]

    # Maintain min distances to selected set (initialized to inf)
    min_distances = np.full(N, np.inf, dtype=np.float64)
    # Mark first selected as excluded
    min_distances[first_idx] = -1.0

    for _ in range(replay_size - 1):
        # Update min_distances with cosine distance to last selected
        last = normalized[selected[-1]]  # [D]
        cosine_sim = normalized @ last  # [N]
        cosine_dist = 1.0 - cosine_sim  # [N]
        np.minimum(min_distances, cosine_dist, out=min_distances)

        # BUG FIX: only need to re-mark last selected (previous ones stay at -1
        # because np.minimum never increases a value below 0, and cosine_dist >= 0)
        min_distances[selected[-1]] = -1.0

        # Pick argmax of min_distances
        next_idx = int(np.argmax(min_distances))
        selected.append(next_idx)

    return np.array(selected)


def _log_gradient_diagnostics(diagnostics, selected_indices, valid_indices, total_N):
    """Log gradient-based selection diagnostics for post-mortem analysis."""
    residuals = diagnostics['residuals']
    activation_norms = diagnostics['activation_norms']
    predictions = diagnostics['predictions']
    actual = diagnostics['actual_rewards']
    selected_kv = diagnostics['selected_kv']  # [N, kv_dim] normalized kv_hit_ratio of chosen action

    # --- Residual distribution: all vs selected ---
    sel_residuals = residuals[valid_indices]
    logger.info(f"REPLAY_DIAG residuals: all(mean={residuals.mean():.4f}, std={residuals.std():.4f}, "
                f"p50={np.median(residuals):.4f}, p90={np.percentile(residuals, 90):.4f}, "
                f"p99={np.percentile(residuals, 99):.4f}, max={residuals.max():.4f})")
    logger.info(f"REPLAY_DIAG residuals: selected(mean={sel_residuals.mean():.4f}, std={sel_residuals.std():.4f}, "
                f"p50={np.median(sel_residuals):.4f}, p90={np.percentile(sel_residuals, 90):.4f}, "
                f"max={sel_residuals.max():.4f})")

    # --- Activation norm distribution: all vs selected ---
    sel_norms = activation_norms[valid_indices]
    logger.info(f"REPLAY_DIAG activation_norms: all(mean={activation_norms.mean():.4f}, std={activation_norms.std():.4f}, "
                f"p50={np.median(activation_norms):.4f}, max={activation_norms.max():.4f})")
    logger.info(f"REPLAY_DIAG activation_norms: selected(mean={sel_norms.mean():.4f}, std={sel_norms.std():.4f}, "
                f"p50={np.median(sel_norms):.4f}, max={sel_norms.max():.4f})")

    # --- Prediction vs actual reward: all vs selected ---
    logger.info(f"REPLAY_DIAG predictions: all(mean={predictions.mean():.4f}, std={predictions.std():.4f}), "
                f"actual_rewards: all(mean={actual.mean():.4f}, std={actual.std():.4f})")
    sel_pred = predictions[valid_indices]
    sel_actual = actual[valid_indices]
    logger.info(f"REPLAY_DIAG predictions: selected(mean={sel_pred.mean():.4f}, std={sel_pred.std():.4f}), "
                f"actual_rewards: selected(mean={sel_actual.mean():.4f}, std={sel_actual.std():.4f})")

    # --- Selected-action kv_hit_ratio (normalized): all vs selected ---
    # selected_kv shape is [N, kv_dim], flatten to [N] if kv_dim=1
    kv_flat = selected_kv.squeeze(-1) if selected_kv.ndim > 1 else selected_kv
    sel_kv_flat = kv_flat[valid_indices]
    logger.info(f"REPLAY_DIAG kv_hit_ratio(normalized, chosen action): all(mean={kv_flat.mean():.4f}, std={kv_flat.std():.4f}, "
                f"min={kv_flat.min():.4f}, max={kv_flat.max():.4f})")
    logger.info(f"REPLAY_DIAG kv_hit_ratio(normalized, chosen action): selected(mean={sel_kv_flat.mean():.4f}, "
                f"std={sel_kv_flat.std():.4f}, min={sel_kv_flat.min():.4f}, max={sel_kv_flat.max():.4f})")

    # --- Zero-residual fraction (model predicts perfectly → these get zero gradient weight) ---
    zero_residual_count = int((residuals < 1e-6).sum())
    sel_zero_count = int((sel_residuals < 1e-6).sum())
    logger.info(f"REPLAY_DIAG zero_residual_fraction: all={zero_residual_count}/{total_N} "
                f"({100.0 * zero_residual_count / max(total_N, 1):.1f}%), "
                f"selected={sel_zero_count}/{len(valid_indices)} "
                f"({100.0 * sel_zero_count / max(len(valid_indices), 1):.1f}%)")


def _log_selection_diagnostics(source_df, selected_df, all_indices, selected_indices):
    """Log data-level diagnostics comparing selected replay samples vs full candidate pool."""
    # --- kv_hit_ratio columns (raw, pre-normalization) ---
    kv_cols = [c for c in source_df.columns if 'kv_hit_ratio' in c]
    if kv_cols:
        # Per-pod kv_hit_ratio
        for col in kv_cols:
            all_vals = source_df[col].values
            sel_vals = selected_df[col].values if col in selected_df.columns else np.array([])
            if len(sel_vals) > 0:
                logger.info(f"REPLAY_DIAG {col}(raw): all(mean={np.nanmean(all_vals):.2f}, std={np.nanstd(all_vals):.2f}), "
                            f"selected(mean={np.nanmean(sel_vals):.2f}, std={np.nanstd(sel_vals):.2f})")

        # Max kv_hit_ratio across pods per sample (proxy for "best prefix sharing available")
        all_max_kv = source_df[kv_cols].max(axis=1)
        sel_max_kv = selected_df[kv_cols].max(axis=1) if all(c in selected_df.columns for c in kv_cols) else pd.Series(dtype=float)
        if len(sel_max_kv) > 0:
            logger.info(f"REPLAY_DIAG max_kv_across_pods(raw): all(mean={all_max_kv.mean():.2f}, p50={all_max_kv.median():.2f}), "
                        f"selected(mean={sel_max_kv.mean():.2f}, p50={sel_max_kv.median():.2f})")

        # Fraction of samples with high prefix sharing (max kv > 50%)
        all_high_kv_frac = (all_max_kv > 50).mean()
        sel_high_kv_frac = (sel_max_kv > 50).mean() if len(sel_max_kv) > 0 else 0.0
        logger.info(f"REPLAY_DIAG high_prefix_sharing(max_kv>50%): all={all_high_kv_frac:.3f}, selected={sel_high_kv_frac:.3f}")

    # --- Reward (ttft) distribution ---
    if 'ttft' in source_df.columns:
        all_ttft = source_df['ttft'].values
        sel_ttft = selected_df['ttft'].values if 'ttft' in selected_df.columns else np.array([])
        if len(sel_ttft) > 0:
            logger.info(f"REPLAY_DIAG ttft(raw): all(mean={np.nanmean(all_ttft):.1f}, p50={np.nanmedian(all_ttft):.1f}, "
                        f"p90={np.nanpercentile(all_ttft, 90):.1f}), "
                        f"selected(mean={np.nanmean(sel_ttft):.1f}, p50={np.nanmedian(sel_ttft):.1f}, "
                        f"p90={np.nanpercentile(sel_ttft, 90):.1f})")

    # --- Temporal composition: _collection_round distribution ---
    if '_collection_round' in source_df.columns:
        all_rounds = source_df['_collection_round']
        sel_rounds = selected_df['_collection_round'] if '_collection_round' in selected_df.columns else pd.Series(dtype=float)

        # Count per round (NaN = offline data)
        all_round_counts = all_rounds.value_counts(dropna=False).sort_index()
        sel_round_counts = sel_rounds.value_counts(dropna=False).sort_index() if len(sel_rounds) > 0 else pd.Series(dtype=int)

        all_offline = int(all_rounds.isna().sum())
        sel_offline = int(sel_rounds.isna().sum()) if len(sel_rounds) > 0 else 0

        # Build compact round distribution strings
        all_parts = []
        sel_parts = []
        if all_offline > 0:
            all_parts.append(f"offline={all_offline}")
        if sel_offline > 0:
            sel_parts.append(f"offline={sel_offline}")

        for rnd in sorted(all_round_counts.index.dropna()):
            rnd_int = int(rnd)
            all_parts.append(f"r{rnd_int}={int(all_round_counts.get(rnd, 0))}")
            if rnd in sel_round_counts.index:
                sel_parts.append(f"r{rnd_int}={int(sel_round_counts.get(rnd, 0))}")
            else:
                sel_parts.append(f"r{rnd_int}=0")

        logger.info(f"REPLAY_DIAG temporal_composition: all({', '.join(all_parts)})")
        logger.info(f"REPLAY_DIAG temporal_composition: selected({', '.join(sel_parts)})")

    # --- Action (pod) distribution ---
    if 'selected_pod' in source_df.columns:
        all_pods = source_df['selected_pod'].value_counts().sort_index()
        sel_pods = selected_df['selected_pod'].value_counts().sort_index() if 'selected_pod' in selected_df.columns else pd.Series(dtype=int)
        all_str = ', '.join(f"{p}={c}" for p, c in all_pods.items())
        sel_str = ', '.join(f"{p}={c}" for p, c in sel_pods.items()) if len(sel_pods) > 0 else "N/A"
        logger.info(f"REPLAY_DIAG pod_distribution: all({all_str})")
        logger.info(f"REPLAY_DIAG pod_distribution: selected({sel_str})")

    # --- Input token distribution ---
    if 'input_tokens' in source_df.columns:
        all_it = source_df['input_tokens'].values
        sel_it = selected_df['input_tokens'].values if 'input_tokens' in selected_df.columns else np.array([])
        if len(sel_it) > 0:
            logger.info(f"REPLAY_DIAG input_tokens(raw): all(mean={np.nanmean(all_it):.0f}, std={np.nanstd(all_it):.0f}), "
                        f"selected(mean={np.nanmean(sel_it):.0f}, std={np.nanstd(sel_it):.0f})")


class GradientReplayBuffer:
    """
    Maintains a gradient-diverse subset of historical training data
    for replay during online training.

    BUG FIX: Stores actual selected DataFrame rows instead of positional indices.
    Positional indices would be invalid across rounds because the source DataFrame
    (training_df_copy of round N) differs from the target DataFrame
    (historical_df of round N+1) in size, ordering, and content.
    """

    def __init__(self, buffer_size, persist_dir=None):
        self.buffer_size = buffer_size
        self.selected_data = pd.DataFrame()  # stored selected rows
        self.persist_dir = persist_dir

    def update(self, model, tensor_data, source_df):
        """
        Recompute replay buffer selection using current model's gradient features.

        For model=None (no previous model trained yet), falls back to random selection.

        Args:
            model: NeuralContextualBandit instance (previous round's model)
            tensor_data: encoded tensor dataset dict
            source_df: the DataFrame that tensor_data was encoded from;
                       selected rows are stored directly for use in next round
        """
        N = len(tensor_data['actions'])
        source_len = len(source_df)
        # tensor_data and source_df should correspond row-for-row
        effective_N = min(N, source_len)
        target_size = min(self.buffer_size, effective_N)

        if target_size <= 0:
            self.selected_data = pd.DataFrame()
            return

        if model is None:
            # No previous model — random fallback
            rng = np.random.RandomState(42)
            chosen = rng.choice(effective_N, size=target_size, replace=False)
            self.selected_data = source_df.iloc[chosen].reset_index(drop=True)
            logger.info(f"Replay buffer: random fallback, selected {len(self.selected_data)} samples")
            _log_selection_diagnostics(source_df, self.selected_data, np.arange(effective_N), chosen)
            return

        try:
            gradient_features, diagnostics = compute_gradient_features(model, tensor_data)
            indices = select_replay_samples(gradient_features, target_size)
            # Clamp indices to source_df length
            valid = indices[indices < source_len]
            self.selected_data = source_df.iloc[valid].reset_index(drop=True)
            logger.info(f"Replay buffer: gradient selection, {len(self.selected_data)} samples "
                        f"from {N} candidates (features shape: {gradient_features.shape})")
            _log_gradient_diagnostics(diagnostics, indices, valid, N)
            _log_selection_diagnostics(source_df, self.selected_data, np.arange(effective_N), valid)
        except Exception as e:
            logger.warning(f"Replay buffer gradient selection failed, falling back to random: {e}")
            import traceback
            logger.warning(traceback.format_exc())
            rng = np.random.RandomState(42)
            chosen = rng.choice(effective_N, size=target_size, replace=False)
            self.selected_data = source_df.iloc[chosen].reset_index(drop=True)

    def get_selected_data(self):
        """Return the stored replay DataFrame rows."""
        return self.selected_data

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, 'replay_buffer.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump({
                'buffer_size': self.buffer_size,
                'selected_data': self.selected_data,
            }, f)

    def load(self, path):
        filepath = os.path.join(path, 'replay_buffer.pkl')
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            self.buffer_size = state.get('buffer_size', self.buffer_size)
            self.selected_data = state.get('selected_data', pd.DataFrame())
            logger.info(f"Replay buffer loaded: {len(self.selected_data)} rows from {filepath}")


class OldTransferMonitor:
    """
    Monitors catastrophic forgetting by evaluating the current model
    on frozen validation windows from previous collection rounds.
    """

    def __init__(self, samples_per_window=100, persist_dir=None):
        self.samples_per_window = samples_per_window
        self.persist_dir = persist_dir
        self.validation_windows = {}  # collection_round -> tensor_data subset
        self.evaluation_history = []  # [{round: loss, ...}, ...] per training round

    def register_window(self, collection_round, tensor_data):
        """
        Register a frozen validation window for a collection round.
        Randomly samples samples_per_window from tensor_data.
        Skips if window already registered for this round.
        """
        if collection_round in self.validation_windows:
            return

        N = len(tensor_data['actions'])
        if N == 0:
            return

        sample_size = min(self.samples_per_window, N)
        rng = np.random.RandomState(collection_round)
        indices = rng.choice(N, size=sample_size, replace=False)
        indices_t = torch.tensor(indices)

        window = {}
        for key in tensor_data:
            if isinstance(tensor_data[key], torch.Tensor):
                if tensor_data[key].shape[0] == N:
                    window[key] = tensor_data[key][indices_t]
                else:
                    window[key] = tensor_data[key]
            else:
                window[key] = tensor_data[key]

        self.validation_windows[collection_round] = window
        logger.info(f"OldTransfer: registered window for round {collection_round} "
                    f"({sample_size} samples from {N})")

    def evaluate_all_windows(self, model):
        """
        Evaluate model loss on all frozen validation windows.

        Args:
            model: NeuralContextualBandit instance

        Returns:
            dict of {collection_round: mse_loss}
        """
        results = {}
        for round_id, window in self.validation_windows.items():
            try:
                loss = model.evaluate_loss(
                    pod_features=window['pod_features_with_staleness'],
                    kv_hit_ratios=window['kv_hit_ratios'],
                    request_features=window['request_features'],
                    actions=window['actions'],
                    rewards=window['rewards']
                )
                results[round_id] = loss
            except Exception as e:
                logger.warning(f"OldTransfer: eval failed for window {round_id}: {e}")
                results[round_id] = float('nan')

        self.evaluation_history.append(results)
        return results

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, 'old_transfer_monitor.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump({
                'samples_per_window': self.samples_per_window,
                'validation_windows': self.validation_windows,
                'evaluation_history': self.evaluation_history,
            }, f)

    def load(self, path):
        filepath = os.path.join(path, 'old_transfer_monitor.pkl')
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            self.samples_per_window = state.get('samples_per_window', self.samples_per_window)
            self.validation_windows = state.get('validation_windows', {})
            self.evaluation_history = state.get('evaluation_history', [])
            logger.info(f"OldTransfer monitor loaded: {len(self.validation_windows)} windows, "
                        f"{len(self.evaluation_history)} eval rounds from {filepath}")
