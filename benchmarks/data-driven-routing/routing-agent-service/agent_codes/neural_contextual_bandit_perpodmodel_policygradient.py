#!/usr/bin/env python3
"""
Neural Contextual Bandit for LLM Routing - POLICY GRADIENT VERSION
=================================================================

IMPORTANT: This is the POLICY GRADIENT implementation using REINFORCE.

Key differences from regression-based version:
1. Action selection: Sample from softmax(scores/temperature) instead of argmax
2. Training loss: -log π(a|s) × (R - baseline) instead of MSE
3. Exploration: Built-in via softmax temperature (epsilon = temperature)
4. Gradient flow: Updates ALL action scores via softmax, not just chosen action

Theory: REINFORCE (Williams 1992)
- Policy: π(a|s) = softmax(NN(s)/T)
- Objective: Maximize E[R]
- Gradient: ∇θ J = E[∇θ log π(a|s) × (R - b)]
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import pickle
import time
import pandas as pd
from logger import logger
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from plot_utils import plot_neural_cb_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RewardNetwork(nn.Module):
    """
    Per-Pod Reward Network: Scores a single (pod, request) pair independently.
    This architecture is scalable to any number of pods.
    """
    def __init__(self, per_pod_context_dim, hidden_dim=128):
        super().__init__()
        
        self.per_pod_context_dim = per_pod_context_dim
        
        # Single scorer network that evaluates one pod at a time
        self.feature_extractor = nn.Sequential(
            nn.Linear(per_pod_context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # CRITICAL: Separate output layer with small weight initialization
        # This prevents scores from exploding during early training
        self.output_layer = nn.Linear(hidden_dim, 1)

        # Initialize output layer with small weights (0.01 std)
        # This keeps initial scores close to 0, preventing numerical overflow
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.output_layer.bias, 0.0)

        logger.info(f"RewardNetwork (Per-Pod Policy Gradient): per_pod_context_dim={per_pod_context_dim}, "
                   f"hidden_dim={hidden_dim}, output_init_std=0.01")
    
    def forward(self, context):
        """
        Args:
            context: [batch_size, per_pod_context_dim]
                     Each row is [single_pod_features + single_pod_kv + request_features]
        Returns:
            scores: [batch_size, 1] - one score per pod (bounded output)
        """
        features = self.feature_extractor(context)  # [batch, hidden_dim]
        scores = self.output_layer(features)  # [batch, 1]

        # BOUNDING STRATEGY: Use smaller scale (2.0) to prevent saturation while
        # keeping scores bounded. The original 10*tanh saturated too quickly.
        # With scale=2.0 and temperature=0.3, max logits = ±6.7, which is stable.
        scores = 2.0 * torch.tanh(scores)  # [batch, 1] in range [-2, 2]

        return scores


class NeuralContextualBandit:
    """
    Neural Contextual Bandit with proper online learning
    """
    def __init__(self, state_dim, action_dim, hyperparameters, final_model_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hyperparameters = hyperparameters
        self.final_model_dir = final_model_dir

        # NEW: Temporal feature tracking (routing history)
        self.routing_history = deque(maxlen=200)  # Track last 200 routing decisions
        self.history_window_sec = 2.0  # Consider last 2 seconds for temporal features
        
        # DIAGNOSTIC: Track inference count for periodic logging
        self.inference_count = 0
        self.log_features_every = 10000  # Log feature stats every N inferences

        # Calculate per-pod context dimension with NEW features
        # Each pod is evaluated with: [pod_features + pod_kv + cluster_features + temporal_features + request_features]
        cluster_feature_dim = 8  # NEW: Cluster-wide aggregate features
        temporal_feature_dim = 2  # NEW: Temporal routing history features

        self.per_pod_context_dim = (
            state_dim['pod_features'] +      # Single pod's features (e.g., 8)
            state_dim['kv_hit_ratios'] +     # Single pod's KV cache (e.g., 1)
            cluster_feature_dim +            # NEW: Cluster aggregate features (8)
            temporal_feature_dim +           # NEW: Temporal features (2)
            state_dim['request_features']    # Request features (e.g., 2)
        )  # Total: e.g., 21 dims per pod (was 11, now 21)

        logger.info(f"Per-Pod Context dimension: {self.per_pod_context_dim} "
                   f"(pod_features={state_dim['pod_features']}, "
                   f"kv={state_dim['kv_hit_ratios']}, "
                   f"cluster_features={cluster_feature_dim}, "
                   f"temporal_features={temporal_feature_dim}, "
                   f"request={state_dim['request_features']})")
        
        # Create per-pod reward prediction network
        self.reward_net = RewardNetwork(
            self.per_pod_context_dim,
            hidden_dim=hyperparameters.get('hidden_dim', 128)
        ).to(device)
        
        # Optimizer - cap learning rate at 1e-4 for policy gradient stability
        lr = min(hyperparameters.get('learning_rate', 3e-4), 1e-4)
        logger.info(f"NeuralContextualBandit: Using learning rate {lr} (capped from {hyperparameters.get('learning_rate', 3e-4)})")
        self.optimizer = torch.optim.Adam(
            self.reward_net.parameters(),
            lr=lr,
            weight_decay=hyperparameters.get('weight_decay', 1e-5)
        )
        
        # Experience replay buffer (keep last N experiences)
        self.buffer_size = hyperparameters.get('buffer_size', 10000)
        self.replay_buffer = deque(maxlen=self.buffer_size)
        
        # Exploration parameters (epsilon also serves as temperature for softmax)
        self.exploration_method = hyperparameters.get('exploration_method', 'epsilon_greedy')
        self.epsilon = hyperparameters.get('initial_epsilon', 0.3)
        self.initial_epsilon = self.epsilon  # NEW: Store for adaptive exploration
        self.epsilon_decay = hyperparameters.get('epsilon_decay', 0.995)
        # INCREASED epsilon_min from 0.05 to 0.1 to prevent temperature collapse
        # With unbounded scores, low temperature causes numerical instability
        self.epsilon_min = hyperparameters.get('epsilon_min', 0.1)
        
        # UCB parameters (if using UCB)
        self.ucb_confidence = hyperparameters.get('ucb_confidence', 2.0)
        self.action_counts = np.zeros(action_dim)
        self.total_steps = 0
        
        # Training parameters
        self.batch_size = hyperparameters.get('batch_size', 64)
        self.update_frequency = hyperparameters.get('update_frequency', 10)
        self.steps_since_update = 0
        
        # Metrics
        self.training_metrics = {
            'losses': [],
            'rewards': [],
            'epsilons': [],
            'update_steps': [],
            # Reward function quality tracking
            'reward_latency_pairs': [],  # [(reward, latency), ...] for function analysis
            'reward_latency_input_tuples': [],  # [(reward, latency, input_tokens), ...] for stratified analysis
            'ttft_values': [],  # Raw TTFT values
            'tpot_values': [],  # Raw TPOT values
            'action_distribution': np.zeros(action_dim),
            'predicted_rewards': [],  # For reward prediction accuracy analysis
            'actual_rewards': [],     # For reward prediction accuracy analysis
            'selected_actions': [],   # Track which actions were selected
            'exploration_count': 0,   # Count of exploratory actions
            'exploitation_count': 0,   # Count of exploitative actions
            # Off-policy evaluation metrics
            'all_predicted_rewards': [],  # All action predictions [batch, num_actions]
            'greedy_actions': [],         # What model would choose greedily
            'training_actions': [],       # What was actually chosen in training
            'counterfactual_gains': [],    # Estimated gain from model's choice
            'input_tokens_per_sample': [],  # Input tokens for each sample (for stratification)
            # Policy gradient specific metrics
            'scores': [],              # Raw scores from network (before temperature)
            'logits': [],              # Scores after temperature scaling (before softmax)
            'log_probs': [],           # Log probabilities of chosen actions
            'probs': [],               # Probabilities of chosen actions
            'all_log_probs': [],       # Mean log probabilities for all actions [num_pods] per update
            'all_probs': [],           # Mean probabilities for all actions [num_pods] per update
            'baseline': [],            # Baseline value (mean reward)
            'advantages': [],          # Advantage values (reward - baseline)
            'policy_loss': [],         # Policy gradient loss component
            'entropy': []              # Policy entropy
        }
        
        logger.info(f"NeuralContextualBandit initialized: exploration={self.exploration_method}")

    def update_epsilon_adaptive(self):
        """
        NEW: Adaptively adjust epsilon based on learning stability.
        Uses coefficient of variation (CV) of recent losses to determine exploration rate.
        - High CV (unstable learning) → increase exploration
        - Low CV (converged) → decrease exploration
        - Normal CV → standard decay
        """
        recent_losses = self.training_metrics['losses'][-100:]
        if len(recent_losses) < 10:
            return  # Not enough data yet

        # Compute coefficient of variation
        mean_loss = np.mean(recent_losses)
        std_loss = np.std(recent_losses)
        cv = std_loss / (mean_loss + 1e-6)

        # Adaptive epsilon adjustment based on CV
        if cv > 0.5:  # High instability → explore more
            self.epsilon = min(self.epsilon * 1.05, self.initial_epsilon)
            logger.debug(f"High loss CV ({cv:.3f}), increasing epsilon to {self.epsilon:.4f}")
        elif cv < 0.1:  # Converged → explore less
            self.epsilon = max(self.epsilon * 0.99, self.epsilon_min)
            logger.debug(f"Low loss CV ({cv:.3f}), decreasing epsilon to {self.epsilon:.4f}")
        else:  # Normal decay
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def _create_temporal_features(self, num_pods=None):
        """
        NEW: Create temporal features based on recent routing history.
        Returns features showing how recently each pod was selected.

        Args:
            num_pods: Optional number of pods. If None, uses self.action_dim.
                     This allows matching the number of pods in training data.

        Returns:
            temporal_features: [num_pods, 2] array with:
                - Column 0: recent_count (normalized, exponentially weighted)
                - Column 1: avg_recent_tokens (normalized)
        """
        if num_pods is None:
            num_pods = self.action_dim
        
        current_time = time.time()
        recent_counts = np.zeros(num_pods, dtype=np.float32)
        recent_tokens = np.zeros(num_pods, dtype=np.float32)

        # Iterate through recent routing decisions (newest first)
        for entry in reversed(self.routing_history):
            age = current_time - entry['timestamp']
            if age > self.history_window_sec:
                break  # Stop if outside window

            pod_idx = entry['pod']
            # Skip if pod_idx is out of bounds (can happen when training on historical data with different pod counts)
            if pod_idx >= num_pods:
                continue
                
            tokens = entry.get('input_tokens', 0)

            # Exponential decay: recent decisions weighted more
            # Half-life = 0.5 seconds (typical LLM prefill duration)
            weight = np.exp(-age / 0.5)

            recent_counts[pod_idx] += weight
            recent_tokens[pod_idx] += weight * tokens

        # Normalize counts to [0, 1] range
        max_count = recent_counts.max()
        if max_count > 0:
            normalized_counts = recent_counts / max_count
        else:
            normalized_counts = recent_counts

        # Average tokens per recent request
        # Use np.divide with where to avoid divide-by-zero warnings
        avg_tokens = np.divide(
            recent_tokens,
            recent_counts,
            out=np.zeros_like(recent_tokens, dtype=np.float32),
            where=(recent_counts > 0)
        )
        # Normalize assuming max 5000 tokens
        normalized_tokens = np.clip(avg_tokens / 5000.0, 0, 1)
        
        # DIAGNOSTIC: Log temporal feature statistics periodically
        temporal_features = np.stack([normalized_counts, normalized_tokens], axis=1)  # [num_pods, 2]
        
        if self.inference_count % self.log_features_every == 0:
            all_zero = np.all(temporal_features == 0)
            logger.debug(f"🔍 TEMPORAL FEATURES (inference #{self.inference_count}):")
            logger.debug(f"   routing_history size: {len(self.routing_history)}")
            logger.debug(f"   all_zero: {all_zero}")
            logger.debug(f"   normalized_counts: min={normalized_counts.min():.4f}, max={normalized_counts.max():.4f}, mean={normalized_counts.mean():.4f}")
            logger.debug(f"   normalized_tokens: min={normalized_tokens.min():.4f}, max={normalized_tokens.max():.4f}, mean={normalized_tokens.mean():.4f}")
            if all_zero:
                logger.warning(f"   ⚠️  Temporal features are ALL ZERO (cold start issue!)")

        return temporal_features

    def _create_cluster_features(self, pod_features, kv_hit_ratios):
        """
        NEW: Create cluster-wide aggregate features (scale-invariant).
        For each pod, compute its relationship to cluster statistics.

        Args:
            pod_features: [batch, num_pods, pod_feat_dim]
            kv_hit_ratios: [batch, num_pods, kv_dim]

        Returns:
            cluster_features: [batch, num_pods, 8] with scale-invariant features
        """
        batch_size, num_pods, pod_feat_dim = pod_features.shape

        # Convert to numpy for easier computation
        pod_feats_np = pod_features.cpu().numpy()  # [batch, num_pods, 8]
        kv_ratios_np = kv_hit_ratios.cpu().numpy()  # [batch, num_pods, 1]

        cluster_features_list = []

        for b in range(batch_size):
            # Extract key metrics (assuming standard pod feature ordering)
            # Order from encoding.py: [inflight_req, gpu_cache, cpu_cache, running_req, waiting_req, prefill_tok, decode_tok]
            inflight_reqs = pod_feats_np[b, :, 0] if pod_feat_dim > 0 else np.zeros(num_pods)
            gpu_cache = pod_feats_np[b, :, 1] if pod_feat_dim > 1 else np.zeros(num_pods)
            running_reqs = pod_feats_np[b, :, 3] if pod_feat_dim > 3 else np.zeros(num_pods)
            waiting_reqs = pod_feats_np[b, :, 4] if pod_feat_dim > 4 else np.zeros(num_pods)
            prefill_tokens = pod_feats_np[b, :, 5] if pod_feat_dim > 5 else np.zeros(num_pods)
            kv_hits = kv_ratios_np[b, :, 0]

            # Cluster statistics (mean and std)
            # Use meaningful minimum std to prevent divide-by-zero when pods are balanced
            min_std = 0.1

            # Z-score normalized features (mean-centered, std-scaled)
            # inflight_requests has strongest correlation with reward (r=-0.80)
            inflight_mean = inflight_reqs.mean()
            inflight_std = max(inflight_reqs.std(), min_std)
            z_inflight = (inflight_reqs - inflight_mean) / inflight_std

            running_mean = running_reqs.mean()
            running_std = max(running_reqs.std(), min_std)
            z_running = (running_reqs - running_mean) / running_std

            waiting_mean = waiting_reqs.mean()
            waiting_std = max(waiting_reqs.std(), min_std)
            z_waiting = (waiting_reqs - waiting_mean) / waiting_std

            prefill_mean = prefill_tokens.mean()
            prefill_std = max(prefill_tokens.std(), min_std)
            z_prefill = (prefill_tokens - prefill_mean) / prefill_std

            # Rank-based features (scale-invariant)
            # Use inflight for ranking as it's the strongest predictor
            rank_by_inflight = np.argsort(np.argsort(inflight_reqs)) / max(num_pods - 1, 1)  # [0, 1]
            rank_by_kv = np.argsort(np.argsort(-kv_hits)) / max(num_pods - 1, 1)  # [0, 1], higher kv = lower rank (better)

            # Cluster-wide context features
            cluster_utilization = np.full(num_pods, inflight_mean / 10.0)  # Normalized by theoretical max
            load_cv = inflight_std / (inflight_mean + 1e-6)  # Coefficient of variation
            cluster_load_variance = np.full(num_pods, load_cv)

            # Stack features: [num_pods, 8]
            cluster_feats = np.stack([
                z_inflight,             # 0: Z-score inflight requests (strongest predictor)
                z_running,              # 1: Z-score running requests
                z_waiting,              # 2: Z-score waiting requests
                z_prefill,              # 3: Z-score prefill tokens
                rank_by_inflight,       # 4: Rank by inflight [0,1], lower=better
                rank_by_kv,             # 5: Rank by KV cache [0,1], lower=better (higher kv)
                cluster_utilization,    # 6: Cluster-wide utilization
                cluster_load_variance   # 7: Cluster load imbalance
            ], axis=1)  # [num_pods, 8]

            cluster_features_list.append(cluster_feats)

        # Stack batches and convert back to tensor
        cluster_features_np = np.stack(cluster_features_list, axis=0)  # [batch, num_pods, 8]
        cluster_features_torch = torch.from_numpy(cluster_features_np).float().to(pod_features.device)

        return cluster_features_torch

    def _create_per_pod_contexts(self, pod_features, kv_hit_ratios, request_features):
        """
        Create per-pod contexts for independent evaluation.
        NEW: Now includes cluster features and temporal features!

        Args:
            pod_features: [batch, num_pods, pod_feat_dim]
            kv_hit_ratios: [batch, num_pods, kv_dim]
            request_features: [batch, req_feat_dim]

        Returns:
            contexts: [batch * num_pods, per_pod_context_dim]
                      Each row is [pod_features + kv + cluster_features + temporal_features + request_features]
        """
        batch_size, num_pods, pod_feat_dim = pod_features.shape

        # NEW: Get cluster-wide aggregate features [batch, num_pods, 8]
        cluster_features = self._create_cluster_features(pod_features, kv_hit_ratios)

        # NEW: Get temporal routing history features [num_pods, 2]
        # Pass num_pods to match the input data (important for training on historical data with different pod counts)
        temporal_features = self._create_temporal_features(num_pods=num_pods)
        temporal_features_torch = torch.from_numpy(temporal_features).float().to(pod_features.device)
        # Expand for batch: [1, num_pods, 2] → [batch, num_pods, 2]
        temporal_features_torch = temporal_features_torch.unsqueeze(0).expand(batch_size, -1, -1)

        # FIX: Add small deterministic noise to temporal features if all zeros (cold start issue)
        if torch.all(temporal_features_torch == 0):
            # Add tiny uniform noise [0, 0.01] to break symmetry at deployment start
            # Use deterministic seed for reproducibility (compatible with older PyTorch)
            torch.manual_seed(42)
            temporal_features_torch = temporal_features_torch + torch.rand_like(temporal_features_torch) * 0.01

        # Expand request features for each pod
        # [batch, req_feat] → [batch, num_pods, req_feat]
        request_repeated = request_features.unsqueeze(1).expand(-1, num_pods, -1)

        # Concatenate all features for each pod:
        # [pod_features, kv_ratio, cluster_features, temporal_features, request_features]
        # [batch, num_pods, 8 + 1 + 8 + 2 + 2] = [batch, num_pods, 21]
        per_pod_contexts = torch.cat([
            pod_features,              # [batch, num_pods, 8]
            kv_hit_ratios,            # [batch, num_pods, 1]
            cluster_features,          # [batch, num_pods, 8]  NEW
            temporal_features_torch,   # [batch, num_pods, 2]  NEW
            request_repeated          # [batch, num_pods, 2]
        ], dim=2)

        # Reshape to [batch * num_pods, per_pod_context_dim]
        per_pod_contexts = per_pod_contexts.reshape(batch_size * num_pods, -1)
        
        # DIAGNOSTIC: Log final concatenated feature statistics periodically
        if self.inference_count % self.log_features_every == 0:
            contexts_np = per_pod_contexts.cpu().numpy()
            logger.debug(f"🔍 FINAL PER-POD CONTEXTS (inference #{self.inference_count}):")
            logger.debug(f"   shape: {per_pod_contexts.shape}")
            logger.debug(f"   min: {contexts_np.min():.4f}, max: {contexts_np.max():.4f}, mean: {contexts_np.mean():.4f}, std: {contexts_np.std():.4f}")
            logger.debug(f"   Feature breakdown (first pod):")
            logger.debug(f"     pod_features[0:8]:      {contexts_np[0, 0:8]}")
            logger.debug(f"     kv_hit_ratio[8]:        {contexts_np[0, 8]:.4f}")
            logger.debug(f"     cluster_features[9:17]: min={contexts_np[0, 9:17].min():.4f}, max={contexts_np[0, 9:17].max():.4f}, mean={contexts_np[0, 9:17].mean():.4f}")
            logger.debug(f"     temporal_features[17:19]: {contexts_np[0, 17:19]}")
            logger.debug(f"     request_features[19:21]:  {contexts_np[0, 19:21]}")

        return per_pod_contexts
    
    def choose_action(self, pod_features, kv_hit_ratios, request_features, evaluate=False):
        """
        POLICY GRADIENT VERSION: Sample action from softmax policy

        Args:
            pod_features: [1, num_pods, pod_feat_dim]
            kv_hit_ratios: [1, num_pods, kv_dim]
            request_features: [1, req_feat_dim]
            evaluate: If True, use greedy selection (argmax)

        Returns:
            action: Selected pod index
            predicted_rewards: Scores for all pods (for logging)
            explored: Boolean indicating if sampled from policy (True) or greedy (False)
            probabilities: Softmax probabilities for all pods (None if evaluate=True)
        """
        # Create per-pod contexts: [num_pods, per_pod_context_dim]
        contexts = self._create_per_pod_contexts(pod_features, kv_hit_ratios, request_features)

        # Get scores for each pod independently: [num_pods, 1]
        # Scores are bounded to [-5, 5] by tanh activation in the network
        scores_batch = self.reward_net(contexts)

        # Reshape to [num_pods] for easier handling
        scores = scores_batch.squeeze(1)  # [num_pods]

        if evaluate:
            # Greedy selection for evaluation
            with torch.no_grad():
                action = int(torch.argmax(scores).item())
            explored = False
            probabilities = None  # No probabilities for greedy selection
        else:
            # POLICY GRADIENT: Sample from softmax policy
            # Apply temperature for exploration control (epsilon acts as temperature)
            # FIXED: Use minimum temperature 0.1 to match training
            temperature = max(self.epsilon, 0.1)  # Minimum temperature 0.1
            logits = scores / temperature

            # Compute softmax probabilities (LEARNED POLICY)
            probs = F.softmax(logits, dim=0)

            # Sample action from categorical distribution
            action = torch.multinomial(probs, num_samples=1).item()

            # FIXED: This is the learned stochastic policy, not random exploration
            explored = False  # Softmax sampling is the learned policy, not exploration
            probabilities = probs.detach().cpu().numpy()

        # Update counters
        self.action_counts[action] += 1
        self.total_steps += 1

        # FIX BUG #1: Update routing_history during inference (not just training)
        # Extract input_tokens from request_features (typically first feature)
        input_tokens = 0
        if request_features.shape[1] > 0:
            input_tokens = float(request_features[0, 0].item())  # Assuming input_tokens is first feature

        self.routing_history.append({
            'timestamp': time.time(),
            'pod': action,
            'input_tokens': input_tokens
        })

        # Adaptive epsilon (temperature) adjustment
        if self.total_steps % 100 == 0:
            self.update_epsilon_adaptive()

        return action, scores.detach().cpu().numpy(), explored, probabilities
    
    def remember(self, pod_features, kv_hit_ratios, request_features, action, reward, input_tokens=None):
        """
        POLICY GRADIENT VERSION: Store experience with log probability

        Args:
            pod_features: [1, num_pods, pod_feat_dim]
            kv_hit_ratios: [1, num_pods, kv_dim]
            request_features: [1, req_feat_dim]
            action: Scalar action (which pod was selected)
            reward: Scalar reward
            input_tokens: Optional scalar input token count (for stratified analysis)
        """
        # Create per-pod contexts for all pods
        per_pod_contexts = self._create_per_pod_contexts(pod_features, kv_hit_ratios, request_features)

        # POLICY GRADIENT: Compute log probability of chosen action
        # Scores are bounded to [-5, 5] by tanh activation
        with torch.no_grad():
            scores = self.reward_net(per_pod_contexts).squeeze(1)  # [num_pods] in [-5, 5]

            # FIXED: Increased min temperature from 0.01 to 0.1 to prevent policy collapse
            temperature = max(self.epsilon, 0.1)
            logits = scores / temperature

            log_probs = F.log_softmax(logits, dim=0)  # [num_pods]

            # Extract log prob for the chosen action
            action_idx = action if isinstance(action, int) else action.item()
            log_prob_action = log_probs[action_idx].item()

        experience = {
            'per_pod_contexts': per_pod_contexts.cpu(),  # [num_pods, per_pod_context_dim]
            'action': action_idx,
            'reward': reward.item() if torch.is_tensor(reward) else reward,
            'log_prob': log_prob_action,  # NEW: Store log probability for policy gradient
            'input_tokens': input_tokens.item() if torch.is_tensor(input_tokens) else (input_tokens if input_tokens is not None else -1)
        }

        self.replay_buffer.append(experience)
        self.steps_since_update += 1
        # Note: update_steps is tracked in learn() method, not here

        # Track routing history for temporal features
        self.routing_history.append({
            'timestamp': time.time(),
            'pod': action_idx,
            'input_tokens': input_tokens.item() if torch.is_tensor(input_tokens) else (input_tokens if input_tokens is not None else 0)
        })

        # Note: Automatic learning disabled for batch training
        # The train function handles learning explicitly
        # Uncomment below for online learning scenarios:
        # if self.steps_since_update >= self.update_frequency:
        #     self.learn()
        #     self.steps_since_update = 0
    
    def learn(self, epoch, batch_index, batch_size, update_in_epoch):
        """
        POLICY GRADIENT VERSION: Update policy using REINFORCE algorithm.
        Loss = -log π(a|s) × (R - baseline)
        """
        learn_start_time = time.time()
        if len(self.replay_buffer) < self.batch_size:
            logger.debug(f"Not enough experiences to learn: {len(self.replay_buffer)} < {self.batch_size}")
            return {'loss': 0.0, 'reward': 0.0, 'epoch': epoch, 'batch_index': batch_index}

        # Sample batch from replay buffer
        batch_size = min(self.batch_size, len(self.replay_buffer))
        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        batch = [self.replay_buffer[index] for index in indices]

        # Prepare batch tensors
        per_pod_contexts_batch = torch.stack([exp['per_pod_contexts'] for exp in batch], dim=0).to(device)
        actions = torch.tensor([exp['action'] for exp in batch], dtype=torch.long).to(device)
        rewards = torch.tensor([exp['reward'] for exp in batch], dtype=torch.float32).to(device)
        input_tokens_batch = [exp.get('input_tokens', -1) for exp in batch]

        # Reshape contexts: [batch_size * num_pods, per_pod_context_dim]
        batch_size_actual, num_pods, per_pod_dim = per_pod_contexts_batch.shape
        contexts_flat = per_pod_contexts_batch.reshape(batch_size_actual * num_pods, per_pod_dim)

        # Forward pass: get scores for each pod (bounded to [-5, 5] by tanh)
        scores_flat = self.reward_net(contexts_flat)  # [batch_size * num_pods, 1]
        scores = scores_flat.reshape(batch_size_actual, num_pods)  # [batch_size, num_pods]

        # DIAGNOSTIC: Check score differentiation across pods
        num_updates = len(self.training_metrics.get('losses', [])) + 1
        if num_updates % 50 == 1:
            score_spread = scores.max(dim=1).values - scores.min(dim=1).values  # [batch_size]
            logger.info(f"[DIAG] Update {num_updates}: score_spread (max-min per sample) mean={score_spread.mean().item():.4f}, min={score_spread.min().item():.4f}, max={score_spread.max().item():.4f}")
            logger.info(f"[DIAG] Update {num_updates}: scores[0] (first sample across pods) = {scores[0].detach().cpu().numpy()}")

            # Check if scores are all nearly identical (network not differentiating)
            if score_spread.mean().item() < 0.1:
                logger.warning(f"[DIAG] Update {num_updates}: ⚠️ VERY LOW SCORE SPREAD - network may not be differentiating between pods!")

        # Compute log probabilities using current policy
        # FIXED: Increased min temperature from 0.01 to 0.1 to prevent policy collapse
        # temperature = max(self.epsilon, 0.1)
        temperature = max(self.epsilon, 0.3)
        # temperature = 1 # testing with tanh removed in forward function
        logits = scores / temperature  # Normalized scores (max ~[-50, 50] at T=0.1)

        # Compute both probabilities and log probabilities
        log_probs = F.log_softmax(logits, dim=1)  # [batch_size, num_pods]
        probs = F.softmax(logits, dim=1)  # [batch_size, num_pods] - needed for entropy

        # Get log probability of chosen actions
        chosen_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)  # [batch_size]

        # BASELINE: Subtract mean reward to reduce variance
        baseline = rewards.mean()
        advantages_raw = rewards - baseline  # [batch_size]

        # Normalize advantages to further reduce variance
        advantages_std_before_norm = advantages_raw.std().item()
        if len(advantages_raw) > 1 and advantages_std_before_norm > 1e-6:
            advantages = (advantages_raw - advantages_raw.mean()) / (advantages_raw.std() + 1e-8)
        else:
            # If std is too small, don't normalize - use raw advantages
            advantages = advantages_raw

        # DIAGNOSTIC: Log advantage statistics every 50 updates
        num_updates = len(self.training_metrics.get('losses', [])) + 1
        if num_updates % 50 == 1:
            logger.info(f"[DIAG] Update {num_updates}: rewards min={rewards.min().item():.4f}, max={rewards.max().item():.4f}, std={rewards.std().item():.4f}")
            logger.info(f"[DIAG] Update {num_updates}: advantages_raw std={advantages_std_before_norm:.6f}")
            logger.info(f"[DIAG] Update {num_updates}: advantages (normalized) min={advantages.min().item():.4f}, max={advantages.max().item():.4f}, std={advantages.std().item():.4f}")
            logger.info(f"[DIAG] Update {num_updates}: chosen_log_probs min={chosen_log_probs.min().item():.4f}, max={chosen_log_probs.max().item():.4f}")

            # Check per-sample policy gradient: log_prob * advantage
            per_sample_grad = chosen_log_probs * advantages
            logger.info(f"[DIAG] Update {num_updates}: per_sample_grad (log_prob*adv) min={per_sample_grad.min().item():.4f}, max={per_sample_grad.max().item():.4f}, mean={per_sample_grad.mean().item():.4f}")

        # REINFORCE loss: -log π(a|s) × (R - baseline)
        # Negative because we want gradient ASCENT on expected reward
        policy_loss = -(chosen_log_probs * advantages).mean()

        # ENTROPY REGULARIZATION: Prevent premature convergence
        # Encourages policy to remain somewhat stochastic
        # H(π) = -Σ π(a|s) log π(a|s)
        entropy = -(probs * log_probs).sum(dim=1).mean()
        entropy_coeff = 0.01  # Small coefficient to balance exploration vs exploitation

        # Total loss: Policy gradient - entropy bonus
        loss = policy_loss - entropy_coeff * entropy

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # DIAGNOSTIC: Check gradient magnitudes before clipping
        if num_updates % 50 == 1:
            total_grad_norm = 0.0
            for p in self.reward_net.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.data.norm(2).item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            logger.info(f"[DIAG] Update {num_updates}: gradient norm (before clip) = {total_grad_norm:.6f}")

            # Check if gradients are vanishing
            if total_grad_norm < 1e-6:
                logger.warning(f"[DIAG] Update {num_updates}: ⚠️ VANISHING GRADIENTS! norm={total_grad_norm:.6f}")

        torch.nn.utils.clip_grad_norm_(self.reward_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update epsilon (temperature decay)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Track metrics
        self.training_metrics['losses'].append(loss.item())
        self.training_metrics['rewards'].append(rewards.mean().item())
        self.training_metrics['epsilons'].append(self.epsilon)
        # Track update step number (number of learning updates)
        self.training_metrics['update_steps'].append(len(self.training_metrics['losses']))

        # Track policy gradient specific metrics
        self.training_metrics['baseline'].append(baseline.item())
        self.training_metrics['policy_loss'].append(policy_loss.item())
        self.training_metrics['entropy'].append(entropy.item())

        # Track advantages - both mean (should be ~0) and std (the actual signal)
        self.training_metrics['advantages'].append(advantages.mean().item())

        # NEW: Track the raw advantage std BEFORE normalization - this is the key learning signal
        if 'advantages_std_raw' not in self.training_metrics:
            self.training_metrics['advantages_std_raw'] = []
        self.training_metrics['advantages_std_raw'].append(advantages_std_before_norm)
        
        # Track log_probs and probs for chosen actions (mean per batch)
        chosen_probs = probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        self.training_metrics['log_probs'].append(chosen_log_probs.mean().item())
        self.training_metrics['probs'].append(chosen_probs.mean().item())
        
        # Track mean probabilities for ALL actions (mean across batch for each action)
        # probs is [batch_size, num_pods], so mean(dim=0) gives [num_pods]
        all_probs_mean = probs.mean(dim=0).detach().cpu().numpy()  # [num_pods]
        all_log_probs_mean = log_probs.mean(dim=0).detach().cpu().numpy()  # [num_pods]
        self.training_metrics['all_probs'].append(all_probs_mean.tolist())
        self.training_metrics['all_log_probs'].append(all_log_probs_mean.tolist())

        # Sample detailed metrics every 10th update to avoid memory issues
        if len(self.training_metrics['losses']) % 10 == 0:
            # For policy gradient, "predicted reward" is the score (before softmax)
            # Get scores for chosen actions
            chosen_scores = scores.gather(1, actions.unsqueeze(1)).squeeze(1)
            self.training_metrics['predicted_rewards'].extend(chosen_scores.detach().cpu().numpy().tolist())
            self.training_metrics['actual_rewards'].extend(rewards.cpu().numpy().tolist())
            self.training_metrics['selected_actions'].extend(actions.cpu().numpy().tolist())

            # OFF-POLICY EVALUATION: What would model choose greedily?
            greedy_actions_batch = torch.argmax(scores, dim=1)  # Model's greedy choice
            self.training_metrics['greedy_actions'].extend(greedy_actions_batch.cpu().numpy().tolist())
            self.training_metrics['training_actions'].extend(actions.cpu().numpy().tolist())

            # Store all scores for counterfactual analysis
            # Only store a small sample to avoid memory issues
            if len(self.training_metrics['all_predicted_rewards']) < 500:  # Limit to 500 samples
                self.training_metrics['all_predicted_rewards'].append(scores.detach().cpu().numpy())
            
            # Track scores and logits arrays (sample to avoid memory issues)
            if len(self.training_metrics['scores']) < 500:  # Limit to 500 samples
                self.training_metrics['scores'].append(scores.detach().cpu().numpy())
            if len(self.training_metrics['logits']) < 500:  # Limit to 500 samples
                self.training_metrics['logits'].append(logits.detach().cpu().numpy())

            # Log score statistics every 100 updates for debugging
            # Use len(losses) as the update number since it tracks actual learning updates
            num_updates = len(self.training_metrics['losses'])
            if num_updates % 100 == 0:
                score_spreads = (scores.max(dim=1).values - scores.min(dim=1).values).detach().cpu().numpy()
                logger.info(f"Update {num_updates}: Score spread: mean={score_spreads.mean():.4f}, min={score_spreads.min():.4f}, max={score_spreads.max():.4f}")
                logger.info(f"Update {num_updates}: Scores sample: {scores[0].detach().cpu().numpy()}")

            # Calculate policy quality metric: difference between greedy and selected action scores
            # NOTE: This compares PREDICTED scores, not actual rewards (which are unknown for non-selected actions)
            # Positive values suggest the model would have chosen higher-scoring actions
            greedy_scores = scores.gather(1, greedy_actions_batch.unsqueeze(1)).squeeze(1)
            selected_scores = scores.gather(1, actions.unsqueeze(1)).squeeze(1)
            policy_quality = greedy_scores.detach().cpu() - selected_scores.detach().cpu()
            self.training_metrics['counterfactual_gains'].extend(policy_quality.numpy().tolist())

            # Store input_tokens for each sample (for stratified analysis in plots 13-15)
            self.training_metrics['input_tokens_per_sample'].extend(input_tokens_batch)

        progress_str = f"Update: {update_in_epoch}"
        logger.info(f"[PG Update] Epoch: {epoch}, {progress_str}, Loss: {loss.item():.4f}, Policy: {policy_loss.item():.4f}, "
                   f"Entropy: {entropy.item():.4f}, Reward: {rewards.mean().item():.4f}, "
                   f"Temp: {self.epsilon:.4f}, Buffer: {len(self.replay_buffer)}, learn_time: {time.time() - learn_start_time:.2f}s")
        
        return {
            'loss': loss.item(),
            'reward': rewards.mean().item(),
            'epsilon': self.epsilon,
            'epoch': epoch,
            'batch_index': batch_index
        }
    
    def save(self, final_model_dir):
        """Save model and metadata"""
        os.makedirs(final_model_dir, exist_ok=True)
        
        # Save network weights
        torch.save(self.reward_net.state_dict(), os.path.join(final_model_dir, 'reward_net.pth'))
        
        # Save optimizer state
        torch.save(self.optimizer.state_dict(), os.path.join(final_model_dir, 'optimizer.pth'))
        
        # Save metadata
        metadata = {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'per_pod_context_dim': self.per_pod_context_dim,
            'hyperparameters': self.hyperparameters,
            'epsilon': self.epsilon,
            'action_counts': self.action_counts.tolist(),
            'total_steps': self.total_steps,
            'training_metrics': self.training_metrics
        }
        
        with open(os.path.join(final_model_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Model saved to {final_model_dir}")
    
    def load_model(self, final_model_dir):
        """Load model and metadata"""
        # Load network weights
        reward_net_path = os.path.join(final_model_dir, 'reward_net.pth')
        if os.path.exists(reward_net_path):
            self.reward_net.load_state_dict(torch.load(reward_net_path, map_location=device))
            logger.info(f"Loaded reward network from {reward_net_path}")
        
        # Load optimizer state
        optimizer_path = os.path.join(final_model_dir, 'optimizer.pth')
        if os.path.exists(optimizer_path):
            try:
                self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
            except:
                logger.warning("Could not load optimizer state")
        
        # Load metadata
        metadata_path = os.path.join(final_model_dir, 'metadata.pkl')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            self.epsilon = metadata.get('epsilon', self.epsilon)
            # CRITICAL: Also update initial_epsilon to prevent adaptive epsilon from resetting to 0.3
            # Without this, update_epsilon_adaptive() can increase epsilon back to initial_epsilon (0.3)
            # even though we trained it down to 0.05
            self.initial_epsilon = self.epsilon
            self.action_counts = np.array(metadata.get('action_counts', self.action_counts))
            self.total_steps = metadata.get('total_steps', 0)

            # Merge loaded training_metrics with current template to ensure new fields exist
            loaded_metrics = metadata.get('training_metrics', {})
            for key in self.training_metrics:
                if key in loaded_metrics:
                    self.training_metrics[key] = loaded_metrics[key]
                # else: keep the freshly initialized default value
            
            # CRITICAL: Validate off-policy metrics have matching lengths
            # If loading old model with partial metrics, clear them to avoid dimension mismatches
            off_policy_keys = ['counterfactual_gains', 'greedy_actions', 'training_actions', 'input_tokens_per_sample']
            lengths = [len(self.training_metrics.get(k, [])) for k in off_policy_keys]
            if len(set(lengths)) > 1:  # Different lengths detected
                logger.warning(f"Detected mismatched off-policy metric lengths {dict(zip(off_policy_keys, lengths))}. "
                             f"Clearing off-policy metrics to avoid plotting errors.")
                for key in off_policy_keys:
                    self.training_metrics[key] = []
            
            logger.info(f"Loaded metadata: epsilon={self.epsilon:.4f}, total_steps={self.total_steps}")


# Inference function (compatible with existing code)
_cached_agent = None
_cached_metadata = None
_model_mtime = None  # Track model file modification time for cross-worker updates

def infer_from_tensor(tensor_data, request_id, model_updated, HYPERPARAMETERS, final_model_dir, sorted_all_pod_ids):
    """
    Inference function compatible with existing routing service
    """
    global _cached_agent, _cached_metadata, _model_mtime
    
    infer_start_time = time.time()
    overhead_summary = {}
    
    # Extract tensors
    tensor_transfer_start = time.time()
    pod_features = tensor_data['pod_features_with_staleness'].to(device)
    kv_hit_ratios = tensor_data['kv_hit_ratios'].to(device)
    request_features = tensor_data['request_features'].to(device)
    overhead_summary['tensor_transfer'] = time.time() - tensor_transfer_start
    
    # Ensure batch format
    if len(pod_features.shape) == 2:
        pod_features = pod_features.unsqueeze(0)
    if len(kv_hit_ratios.shape) == 2:
        kv_hit_ratios = kv_hit_ratios.unsqueeze(0)
    if len(request_features.shape) == 1:
        request_features = request_features.unsqueeze(0)
    
    # Get or create agent
    get_agent_start = time.time()
    current_config = {
        'pod_features': pod_features.shape[2],
        'kv_hit_ratios': kv_hit_ratios.shape[2],
        'request_features': request_features.shape[1],
        'num_pods': pod_features.shape[1]
    }
    
    # Check if model file has been updated (for cross-worker synchronization)
    model_file_updated = False
    model_path = os.path.join(final_model_dir, 'reward_net.pth')
    if os.path.exists(model_path):
        current_mtime = os.path.getmtime(model_path)
        if _model_mtime is None or current_mtime > _model_mtime:
            model_file_updated = True
            _model_mtime = current_mtime
    
    # Recreate if: dimensions changed, agent doesn't exist, model flag set, or file updated
    if _cached_agent is None or _cached_metadata != current_config or model_updated or model_file_updated:
        logger.info(f"Creating/reloading Neural Contextual Bandit agent (first_time={_cached_agent is None}, "
                   f"config_changed={_cached_metadata != current_config}, model_updated={model_updated}, "
                   f"file_updated={model_file_updated})")
        
        state_dim = {
            'pod_features': current_config['pod_features'],
            'kv_hit_ratios': current_config['kv_hit_ratios'],
            'request_features': current_config['request_features']
        }
        
        _cached_agent = NeuralContextualBandit(
            state_dim=state_dim,
            action_dim=current_config['num_pods'],
            hyperparameters=HYPERPARAMETERS,
            final_model_dir=final_model_dir
        )
        
        # Try to load existing model
        if os.path.exists(os.path.join(final_model_dir, 'reward_net.pth')):
            _cached_agent.load_model(final_model_dir)
        
        _cached_metadata = current_config
    else:
        # Agent reused - this is expected for most requests
        pass
    
    overhead_summary['get_agent'] = time.time() - get_agent_start
    
    # Inference
    inference_start = time.time()
    action, predicted_rewards, explored, probabilities = _cached_agent.choose_action(
        pod_features, kv_hit_ratios, request_features,
        evaluate=not HYPERPARAMETERS.get('explore', True)
    )
    overhead_summary['inference'] = time.time() - inference_start
    
    # DIAGNOSTIC: Increment inference count for periodic logging
    _cached_agent.inference_count += 1
    
    # DIAGNOSTIC: Log model scores periodically
    if _cached_agent.inference_count % _cached_agent.log_features_every == 0:
        logger.info(f"🔍 MODEL SCORES (inference #{_cached_agent.inference_count}):")
        logger.info(f"   predicted_rewards: {predicted_rewards}")
        logger.info(f"   min: {predicted_rewards.min():.4f}, max: {predicted_rewards.max():.4f}, mean: {predicted_rewards.mean():.4f}, std: {predicted_rewards.std():.4f}")
        logger.info(f"   probabilities: {probabilities}")
        logger.info(f"   selected action: {action}, score: {predicted_rewards[action]:.4f}, prob: {probabilities[action]:.4f}")

    logger.info(f"Neural CB request {request_id}: action={action}, explored={explored}, total_steps={_cached_agent.total_steps}, buffer_size={len(_cached_agent.replay_buffer)}, epsilon={_cached_agent.epsilon:.3f}")

    # Format predicted_rewards as dict (same format as predicted_latencies)
    predicted_rewards_formatted = {sorted_all_pod_ids[i]: float(predicted_rewards[i]) for i in range(len(sorted_all_pod_ids))}
    chosen_pod_predicted_reward = float(predicted_rewards[action])

    # Format probabilities (CRITICAL: These are the learned policy probabilities)
    if probabilities is not None:
        pod_probabilities = {sorted_all_pod_ids[i]: float(probabilities[i]) for i in range(len(sorted_all_pod_ids))}
        confidence = float(probabilities[action])  # Probability of chosen action
    else:
        pod_probabilities = None  # Will be set to uniform by routing service
        confidence = chosen_pod_predicted_reward

    # Prepare result
    result = {
        'selected_pod_index': int(action),
        'predicted_rewards': predicted_rewards_formatted,
        'chosen_pod_predicted_reward': chosen_pod_predicted_reward,
        'pod_probabilities': pod_probabilities,  # CRITICAL: Softmax probabilities from learned policy
        'confidence': confidence,  # Probability of chosen action (not score)
        'epsilon': _cached_agent.epsilon,
        'total_steps': _cached_agent.total_steps,
        'explored': explored  # False for policy gradient (softmax is the learned policy)
    }
    
    overhead_summary['total_inference'] = time.time() - infer_start_time
    
    return result, overhead_summary


def save_training_metrics_csv(agent, final_model_dir):
    """
    Save training metrics to CSV file incrementally.
    This can be called after each epoch to have incremental saves.
    """
    metrics = agent.training_metrics
    if not metrics['losses']:
        return
    
    num_updates = len(metrics['losses'])
    
    # Helper function to safely get metric list with same length
    def get_metric(key, default=None):
        if key in metrics and len(metrics[key]) == num_updates:
            return metrics[key]
        return [default] * num_updates
    
    # Build base metrics dataframe
    metrics_dict = {
        'update_step': list(range(num_updates)),
        'loss': metrics['losses'],
        'reward': get_metric('rewards'),
        'epsilon': get_metric('epsilons'),
        'policy_loss': get_metric('policy_loss'),
        'entropy': get_metric('entropy'),
        'baseline': get_metric('baseline'),
        'advantages': get_metric('advantages'),
        'log_probs': get_metric('log_probs'),  # Chosen action's log prob
        'probs': get_metric('probs')  # Chosen action's prob
    }
    
    # Add all actions' probabilities if available
    if 'all_probs' in metrics and metrics['all_probs'] and len(metrics['all_probs']) > 0:
        # Determine number of actions from first entry
        num_actions = len(metrics['all_probs'][0])
        
        # Add columns for each action's probability
        for action_idx in range(num_actions):
            metrics_dict[f'probs_action_{action_idx}'] = [
                metrics['all_probs'][i][action_idx] if i < len(metrics['all_probs']) and len(metrics['all_probs'][i]) > action_idx else None
                for i in range(num_updates)
            ]
    
    # Add all actions' log probabilities if available
    if 'all_log_probs' in metrics and metrics['all_log_probs'] and len(metrics['all_log_probs']) > 0:
        # Determine number of actions from first entry
        num_actions = len(metrics['all_log_probs'][0])
        
        # Add columns for each action's log probability
        for action_idx in range(num_actions):
            metrics_dict[f'log_probs_action_{action_idx}'] = [
                metrics['all_log_probs'][i][action_idx] if i < len(metrics['all_log_probs']) and len(metrics['all_log_probs'][i]) > action_idx else None
                for i in range(num_updates)
            ]
    
    metrics_df = pd.DataFrame(metrics_dict)
    csv_path = os.path.join(final_model_dir, 'training_metrics.csv')
    metrics_df.to_csv(csv_path, index=False)
    logger.debug(f"Saved training metrics CSV: {csv_path} ({num_updates} updates)")


def train(encoded_training_dir, final_model_dir, HYPERPARAMETERS, num_trains):
    """
    Train neural contextual bandit on batch of encoded experiences.
    Compatible with existing data pipeline (called from online_train_routine).
    
    Args:
        encoded_training_dir: Directory containing encoded .pt files
        final_model_dir: Directory to save model
        HYPERPARAMETERS: Model hyperparameters
    """
    global _cached_agent
    
    logger.info(f"Starting Neural CB batch training: num_trains={num_trains}, epochs={HYPERPARAMETERS.get('training_epochs', 10)}, dir={encoded_training_dir}")
    
    # Load encoded tensor files
    if not os.path.exists(encoded_training_dir):
        logger.error(f"Encoded data directory not found: {encoded_training_dir}")
        return
    
    # Look for tensor_dataset.pt files in batch subdirectories (batch_1, batch_2, etc.)
    # OR directly in the encoded_training_dir (for online training)
    tensor_files = []
    
    # First check for batch subdirectories (offline training pattern)
    for item in os.listdir(encoded_training_dir):
        item_path = os.path.join(encoded_training_dir, item)
        if os.path.isdir(item_path):
            tensor_file = os.path.join(item_path, 'tensor_dataset.pt')
            if os.path.exists(tensor_file):
                tensor_files.append(tensor_file)
    
    # If no batch subdirectories, check for direct file (online training pattern)
    if not tensor_files:
        direct_file = os.path.join(encoded_training_dir, 'tensor_dataset.pt')
        if os.path.exists(direct_file):
            tensor_files.append(direct_file)
    
    if not tensor_files:
        logger.error(f"No tensor_dataset.pt files found in {encoded_training_dir} (checked batch subdirectories and direct file)")
        return
    
    # Sort by batch number for consistent ordering
    tensor_files.sort()
    file_desc = [os.path.basename(os.path.dirname(f)) if os.path.dirname(f) != encoded_training_dir else 'direct' for f in tensor_files]
    logger.info(f"Found {len(tensor_files)} encoded tensor file(s): {file_desc}")
    
    # Load first file to get dimensions
    batch_data = torch.load(tensor_files[0])
    
    # Initialize agent if needed
    if _cached_agent is None:
        state_dim = {
            'pod_features': batch_data['pod_features_with_staleness'].shape[2],
            'kv_hit_ratios': batch_data['kv_hit_ratios'].shape[2],
            'request_features': batch_data['request_features'].shape[1]
        }
        action_dim = batch_data['pod_features_with_staleness'].shape[1]
        
        logger.info(f"Initializing Neural CB agent: state_dim={state_dim}, action_dim={action_dim}")
        
        _cached_agent = NeuralContextualBandit(
            state_dim=state_dim,
            action_dim=action_dim,
            hyperparameters=HYPERPARAMETERS,
            final_model_dir=final_model_dir
        )
        
        # Try to load existing model
        model_path = os.path.join(final_model_dir, 'reward_net.pth')
        if os.path.exists(model_path):
            try:
                _cached_agent.load_model(final_model_dir)
                logger.info(f"Loaded existing model from {final_model_dir}")
            except Exception as e:
                logger.warning(f"Failed to load existing model: {e}, starting fresh")
    
    # Training loop
    total_samples = 0
    for epoch in range(HYPERPARAMETERS.get('training_epochs', 10)):
        epoch_start = time.time()
        epoch_losses = []
        epoch_rewards = []
        
        for tensor_file in tensor_files:
            # tensor_file is already the full path
            batch_data = torch.load(tensor_file)
            
            # Extract tensors and move to device
            pod_features = batch_data['pod_features_with_staleness'].to(device)
            kv_hit_ratios = batch_data['kv_hit_ratios'].to(device)
            request_features = batch_data['request_features'].to(device)
            actions = batch_data['actions']
            rewards = batch_data['rewards']
            
            # Extract latency and context values for reward function analysis
            ttft = batch_data.get('ttft', None)
            avg_tpot = batch_data.get('avg_tpot', None)
            input_tokens = batch_data.get('input_tokens', None)
            
            batch_size = len(actions)
            
            # Add experiences to replay buffer
            for batch_index in range(batch_size):
                # Pass input_tokens if available for stratified analysis
                inp_tok = input_tokens[batch_index] if input_tokens is not None else None
                _cached_agent.remember(
                    pod_features[batch_index:batch_index+1],
                    kv_hit_ratios[batch_index:batch_index+1],
                    request_features[batch_index:batch_index+1],
                    actions[batch_index].item(),
                    rewards[batch_index].item(),
                    input_tokens=inp_tok
                )
                
                # Collect reward-latency-context tuples for stratified function analysis (sample 10% to save memory)
                if np.random.random() < 0.1 and ttft is not None:
                    _cached_agent.training_metrics['reward_latency_pairs'].append(
                        (rewards[batch_index].item(), ttft[batch_index].item())
                    )
                    _cached_agent.training_metrics['ttft_values'].append(ttft[batch_index].item())
                    if avg_tpot is not None:
                        _cached_agent.training_metrics['tpot_values'].append(avg_tpot[batch_index].item())
                    
                    # NEW: Store (reward, latency, input_tokens) for stratified analysis
                    if input_tokens is not None:
                        _cached_agent.training_metrics['reward_latency_input_tuples'].append(
                            (rewards[batch_index].item(), ttft[batch_index].item(), input_tokens[batch_index].item())
                        )
                
                total_samples += 1

                # Trigger learning periodically (use update_frequency hyperparameter)
                update_freq = HYPERPARAMETERS.get('update_frequency', 500)
                if total_samples % update_freq == 0 and len(_cached_agent.replay_buffer) >= _cached_agent.batch_size:
                    update_in_epoch = len(epoch_losses) + 1
                    metrics = _cached_agent.learn(epoch, batch_index, batch_size, update_in_epoch)
                    epoch_losses.append(metrics['loss'])
                    epoch_rewards.append(metrics['reward'])
        
        # Log epoch metrics
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0.0
        epoch_time = time.time() - epoch_start
        
        logger.info(f"Epoch {epoch+1}/{HYPERPARAMETERS.get('training_epochs', 10)}: loss={avg_loss:.4f}, avg_reward={avg_reward:.4f}, "
                   f"time={epoch_time:.2f}s, buffer_size={len(_cached_agent.replay_buffer)}")
        
        # Save training metrics CSV incrementally after each epoch
        save_training_metrics_csv(_cached_agent, final_model_dir)
    
    # Save trained model
    _cached_agent.save(final_model_dir)
    logger.info(f"Neural CB batch training complete: {total_samples} samples processed, model saved to {final_model_dir}")
    
    # Generate comprehensive training plots (use total_steps as num_trains)
    plot_path = plot_neural_cb_metrics(_cached_agent, final_model_dir, HYPERPARAMETERS.get('training_epochs', 10), total_samples, num_trains=num_trains)
    return plot_path


if __name__ == "__main__":
    # Test the neural contextual bandit
    logger.info("Testing Neural Contextual Bandit...")
    
    state_dim = {'pod_features': 8, 'kv_hit_ratios': 1, 'request_features': 3}
    action_dim = 7
    hyperparameters = {
        'hidden_dim': 128,
        'learning_rate': 3e-4,
        'buffer_size': 1000,
        'exploration_method': 'epsilon_greedy',
        'initial_epsilon': 0.3,
        'batch_size': 32,
        'update_frequency': 10
    }
    
    agent = NeuralContextualBandit(
        state_dim=state_dim,
        action_dim=action_dim,
        hyperparameters=hyperparameters,
        final_model_dir='/tmp/test_neural_cb'
    )
    logger.info("Neural Contextual Bandit initialized successfully!")
    

if __name__ == "__main__":
    # Test the neural contextual bandit
    logger.info("Testing Neural Contextual Bandit...")
    
    state_dim = {'pod_features': 8, 'kv_hit_ratios': 1, 'request_features': 3}
    action_dim = 7
    hyperparameters = {
        'hidden_dim': 128,
        'learning_rate': 3e-4,
        'buffer_size': 1000,
        'exploration_method': 'epsilon_greedy',
        'initial_epsilon': 0.3,
        'batch_size': 32,
        'update_frequency': 10
    }
    
    agent = NeuralContextualBandit(state_dim, action_dim, hyperparameters, "/tmp/test_bandit")
    
    # Simulate some experiences
    for i in range(100):
        pod_features = torch.randn(1, action_dim, state_dim['pod_features'])
        kv_hit_ratios = torch.rand(1, action_dim, state_dim['kv_hit_ratios'])
        request_features = torch.randn(1, state_dim['request_features'])
        
        action, _ = agent.choose_action(pod_features, kv_hit_ratios, request_features)
        
        # Simulate reward (inverse latency)
        simulated_latency = np.random.uniform(100, 500)  # ms
        reward = 1.0 / (simulated_latency / 100.0)  # Normalize
        
        agent.remember(pod_features, kv_hit_ratios, request_features, action, reward)
    
    logger.info("Test completed successfully!")

