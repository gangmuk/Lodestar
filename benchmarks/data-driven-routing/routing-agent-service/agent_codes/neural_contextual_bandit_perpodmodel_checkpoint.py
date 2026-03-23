#!/usr/bin/env python3
"""
Neural Contextual Bandit for LLM Routing
Correct formulation for latency-optimal pod selection
"""

import os
import shutil
import tempfile
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import time
from logger import logger
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inference/runtime tuning flags (env overrides)
CB_LOAD_METADATA_ON_INFER = int(os.getenv("CB_LOAD_METADATA_ON_INFER", "0"))


class RewardNetwork(nn.Module):
    """
    Per-Pod Reward Network: Scores a single (pod, request) pair independently.
    This architecture is scalable to any number of pods.
    """
    def __init__(self, per_pod_context_dim, hidden_dim=128, weight_initialization='xavier'):
        super().__init__()
        
        self.per_pod_context_dim = per_pod_context_dim
        
        # Single scorer network that evaluates one pod at a time
        self.scorer = nn.Sequential(
            nn.Linear(per_pod_context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output: single reward score
        )
        
        # Initialize weights based on specified method
        if weight_initialization == 'xavier':
            self._xavier_initialize_weights()
        elif weight_initialization == 'kaiming':
            self._kaiming_initialize_weights()
        elif weight_initialization == 'static':
            self._static_weight_initialization()
        else:
            logger.warning(f"Unknown weight initialization: {weight_initialization}, using Xavier")
            self._xavier_initialize_weights()
        
        logger.info(f"RewardNetwork (Per-Pod): per_pod_context_dim={per_pod_context_dim}, hidden_dim={hidden_dim}, weight_init={weight_initialization}")
    
    def _xavier_initialize_weights(self):
        """Xavier/Glorot initialization for better gradient flow"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.01)
    
    def _kaiming_initialize_weights(self):
        """He/Kaiming initialization for ReLU networks"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.01)
    
    def _static_weight_initialization(self):
        """Static initialization for testing determinism"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.constant_(module.weight, 0.1)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.01)
    
    def forward(self, context):
        """
        Args:
            context: [batch_size, per_pod_context_dim]
                     Each row is [single_pod_features + single_pod_kv + request_features]
        Returns:
            rewards: [batch_size, 1] - one reward per pod
        """
        return self.scorer(context)  # [batch, 1]


class NeuralContextualBandit:
    """
    Neural Contextual Bandit with proper online learning
    """
    def __init__(self, state_dim, action_dim, hyperparameters, final_model_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hyperparameters = hyperparameters
        self.final_model_dir = final_model_dir
        
        # Calculate per-pod context dimension (not global)
        # Each pod is evaluated independently with: [pod_features + pod_kv + request_features]
        self.per_pod_context_dim = (
            state_dim['pod_features'] +      # Single pod's features (e.g., 8)
            state_dim['kv_hit_ratios'] +     # Single pod's KV cache (e.g., 1)
            state_dim['request_features']    # Request features (e.g., 2)
        )  # Total: e.g., 11 dims per pod
        
        logger.info(f"Per-Pod Context dimension: {self.per_pod_context_dim} "
                   f"(pod_features={state_dim['pod_features']}, "
                   f"kv={state_dim['kv_hit_ratios']}, "
                   f"request={state_dim['request_features']})")
        
        # Create per-pod reward prediction network
        weight_init = hyperparameters.get('weight_initialization', 'xavier')
        self.reward_net = RewardNetwork(
            self.per_pod_context_dim,
            hidden_dim=hyperparameters.get('hidden_dim', 128),
            weight_initialization=weight_init
        ).to(device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.reward_net.parameters(),
            lr=hyperparameters.get('learning_rate', 0.0001),
            weight_decay=hyperparameters.get('weight_decay', 1e-5)
        )

        # Learning rate scheduler - multiple options based on hyperparameters
        self.scheduler_type = hyperparameters.get('lr_scheduler_type', 'constant')
        self.gradient_norms = []  # Track gradient norms for gradient-based scheduling
        self.min_grad_norm_threshold = 1e-4  # Minimum gradient norm threshold
        self.max_grad_norm_threshold = 10.0  # Maximum gradient norm threshold
        
        if self.scheduler_type == 'constant':
            # No scheduler - constant learning rate
            self.scheduler = None
            logger.info("🔧 Using constant learning rate (no scheduler)")
            
        elif self.scheduler_type == 'exponential':
            # Fixed exponential decay - predictable and stable
            gamma = hyperparameters.get('lr_scheduler_gamma', 0.95)
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
            logger.info(f"🔧 Using ExponentialLR scheduler (gamma={gamma})")
            
        elif self.scheduler_type == 'gradient_adaptive':
            # Gradient-based scheduling - reduce LR based on gradient norms
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.3, patience=2
            )
            logger.info("🔧 Using gradient-adaptive scheduler (gradient norm-based)")
            
        else:
            logger.warning(f"Unknown scheduler type '{self.scheduler_type}', defaulting to constant")
            self.scheduler = None
            self.scheduler_type = 'constant'

        # Exploration parameters
        self.exploration_method = hyperparameters.get('exploration_method', 'epsilon_greedy')
        self.epsilon = hyperparameters.get('initial_epsilon', 0.3)
        self.epsilon_decay = hyperparameters.get('epsilon_decay', 0.995)
        self.epsilon_min = hyperparameters.get('epsilon_min', 0.05)
        
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
            'eval_losses': [],  # Evaluation loss for validation
            'rewards': [],
            'epsilons': [],
            'learning_rates': [],  # Track learning rate over time
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
            'input_tokens_per_sample': []  # Input tokens for each sample (for stratification)
        }
        
        logger.info(f"NeuralContextualBandit initialized: exploration={self.exploration_method}")
    
    def _create_per_pod_contexts(self, pod_features, kv_hit_ratios, request_features):
        """
        Create per-pod contexts for independent evaluation.
        Each pod is evaluated with its own features + request features.
        
        Args:
            pod_features: [batch, num_pods, pod_feat_dim]
            kv_hit_ratios: [batch, num_pods, kv_dim]
            request_features: [batch, req_feat_dim]
        
        Returns:
            contexts: [batch * num_pods, per_pod_context_dim]
                      Each row is [single_pod_features + single_pod_kv + request_features]
        """
        batch_size, num_pods, pod_feat_dim = pod_features.shape
        _, _, kv_dim = kv_hit_ratios.shape
        
        # Expand request features for each pod (memory efficient - shares underlying data)
        # [batch, req_feat] → [batch, num_pods, req_feat]
        request_repeated = request_features.unsqueeze(1).expand(-1, num_pods, -1)
        
        # Concatenate for each pod: [pod_features, kv_ratio, request]
        # [batch, num_pods, pod_feat_dim + kv_dim + req_feat_dim]
        per_pod_contexts = torch.cat([pod_features, kv_hit_ratios, request_repeated], dim=2)
        
        # Reshape to [batch * num_pods, per_pod_context_dim]
        per_pod_contexts = per_pod_contexts.reshape(batch_size * num_pods, -1)
        
        return per_pod_contexts
    
    def choose_action(self, pod_features, kv_hit_ratios, request_features, evaluate=False):
        """
        Select action (pod) based on current policy
        
        Args:
            pod_features: [1, num_pods, pod_feat_dim]
            kv_hit_ratios: [1, num_pods, kv_dim]
            request_features: [1, req_feat_dim]
            evaluate: If True, use pure exploitation
        
        Returns:
            action: Selected pod index
            predicted_rewards: Expected rewards for all pods (for logging)
            explored: Boolean indicating if this action was from exploration (True) or exploitation (False)
        """
        with torch.no_grad():
            # Create per-pod contexts: [num_pods, per_pod_context_dim]
            contexts = self._create_per_pod_contexts(pod_features, kv_hit_ratios, request_features)
            
            # Get reward for each pod independently: [num_pods, 1]
            predicted_rewards_batch = self.reward_net(contexts)
            
            # Reshape to [num_pods] for easier handling
            predicted_rewards = predicted_rewards_batch.squeeze(1).cpu().numpy()  # [num_pods]
            
            explored = False  # Track if we explored or exploited
            
            if evaluate or self.exploration_method == 'greedy':
                # Pure exploitation
                action = int(np.argmax(predicted_rewards))
                explored = False
            
            elif self.exploration_method == 'epsilon_greedy':
                # Epsilon-greedy exploration with PC1-biased strategy
                # Instead of random pod, explore by selecting the pod with highest KV hit ratio.
                # This bootstraps prefix concentration signal for the differential KV feature.
                actual_num_pods = len(predicted_rewards)
                if np.random.random() < self.epsilon:
                    # PC1-biased exploration: pick pod with highest KV hit ratio
                    if kv_hit_ratios is not None and kv_hit_ratios.numel() > 0:
                        kv_values = kv_hit_ratios.squeeze(0).squeeze(-1).cpu().numpy()  # [num_pods]
                        action = int(np.argmax(kv_values))
                    else:
                        action = np.random.randint(0, actual_num_pods)
                    explored = True
                else:
                    action = int(np.argmax(predicted_rewards))
                    explored = False
            
            elif self.exploration_method == 'ucb':
                # Upper Confidence Bound
                # Use actual number of pods from input, not self.action_dim (which may differ if trained on different pod count)
                actual_num_pods = len(predicted_rewards)
                exploitation = predicted_rewards

                # Add exploration bonus: sqrt(2 * log(t) / n_a)
                # Only use action_counts for the actual number of pods
                action_counts_slice = self.action_counts[:actual_num_pods] if len(self.action_counts) >= actual_num_pods else np.zeros(actual_num_pods)
                exploration_bonus = np.sqrt(
                    self.ucb_confidence * np.log(self.total_steps + 1) / (action_counts_slice + 1)
                )

                ucb_values = exploitation + exploration_bonus
                action = int(np.argmax(ucb_values))
                # UCB inherently balances exploration/exploitation, mark as exploitation for simplicity
                explored = False
            
            elif self.exploration_method == 'thompson_sampling':
                # Thompson Sampling: Add Gaussian noise to predictions
                noise_std = self.epsilon  # Use epsilon as noise level
                noisy_rewards = predicted_rewards + np.random.randn(len(predicted_rewards)) * noise_std
                action = int(np.argmax(noisy_rewards))
                # Thompson sampling always adds noise, consider it exploration
                explored = True
            
            else:
                raise ValueError(f"Unknown exploration method: {self.exploration_method}")

            # Update counters (handle case where runtime pods differ from training)
            if action < len(self.action_counts):
                self.action_counts[action] += 1
            self.total_steps += 1

            return action, predicted_rewards, explored
    
    def learn(self, pod_features, kv_hit_ratios, request_features, actions, rewards, input_tokens=None, sample_weights=None):
        """
        Update the reward network using batch data directly.
        Per-pod architecture: trains on individual pod evaluations.

        Args:
            pod_features: [batch_size, num_pods, pod_feat_dim]
            kv_hit_ratios: [batch_size, num_pods, kv_dim]
            request_features: [batch_size, req_feat_dim]
            actions: [batch_size] - which pod was selected for each request
            rewards: [batch_size] - actual rewards received
            input_tokens: [batch_size] - optional input token counts for analysis
        """
        batch_size = len(actions)

        if batch_size == 0:
            logger.debug("Empty batch, skipping learning")
            return {'loss': 0.0, 'reward': 0.0}

        # Create per-pod contexts for all samples in a single vectorized call
        # _create_per_pod_contexts already handles full batch: [batch, num_pods, feat] -> [batch*num_pods, dim]
        contexts_flat = self._create_per_pod_contexts(pod_features, kv_hit_ratios, request_features).to(device)
        num_pods = pod_features.shape[1]
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)

        if input_tokens is not None:
            input_tokens_batch = input_tokens if isinstance(input_tokens, list) else input_tokens.tolist()
        else:
            input_tokens_batch = [-1] * batch_size
        
        # Forward pass: get reward for each pod independently
        # Output: [batch_size * num_pods, 1]
        predicted_rewards_flat = self.reward_net(contexts_flat)
        
        # Reshape back to [batch_size, num_pods]
        predicted_rewards = predicted_rewards_flat.reshape(batch_size, num_pods)
        
        # Get predicted rewards for the actions that were taken
        predicted_action_rewards = predicted_rewards.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute loss (MSE between predicted and actual rewards)
        if sample_weights is not None:
            sample_weights_tensor = sample_weights.to(predicted_action_rewards.device) if isinstance(sample_weights, torch.Tensor) else torch.tensor(sample_weights, dtype=torch.float32).to(predicted_action_rewards.device)
            per_sample_loss = (predicted_action_rewards - rewards) ** 2
            loss = (per_sample_loss * sample_weights_tensor).sum() / sample_weights_tensor.sum()
        else:
            loss = F.mse_loss(predicted_action_rewards, rewards)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Track gradient norm for gradient_adaptive scheduler
        if self.scheduler_type == 'gradient_adaptive':
            total_norm = 0.0
            for p in self.reward_net.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            self.gradient_norms.append(total_norm)
        
        torch.nn.utils.clip_grad_norm_(self.reward_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Note: exponential scheduler is stepped at epoch level (not batch level) for better stability
        
        # Update epsilon (decay exploration)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Get current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # Track metrics
        self.training_metrics['losses'].append(loss.item())
        self.training_metrics['rewards'].append(rewards.mean().item())
        self.training_metrics['epsilons'].append(self.epsilon)
        self.training_metrics['learning_rates'].append(current_lr)
        
        # Track predicted vs actual rewards (for reward prediction accuracy analysis)
        # Sample every 10th update to avoid memory issues
        if len(self.training_metrics['losses']) % 10 == 0:
            self.training_metrics['predicted_rewards'].extend(predicted_action_rewards.detach().cpu().numpy().tolist())
            self.training_metrics['actual_rewards'].extend(rewards.cpu().numpy().tolist())
            self.training_metrics['selected_actions'].extend(actions.cpu().numpy().tolist())
            
            # OFF-POLICY EVALUATION: What would model choose greedily?
            greedy_actions_batch = torch.argmax(predicted_rewards, dim=1)  # Model's greedy choice
            self.training_metrics['greedy_actions'].extend(greedy_actions_batch.cpu().numpy().tolist())
            self.training_metrics['training_actions'].extend(actions.cpu().numpy().tolist())
            
            # Store all predictions for counterfactual analysis
            # Only store a small sample to avoid memory issues
            if len(self.training_metrics['all_predicted_rewards']) < 500:  # Limit to 500 samples
                self.training_metrics['all_predicted_rewards'].append(predicted_rewards.detach().cpu().numpy())
            
            # Calculate counterfactual gain: predicted_reward[greedy] - actual_reward[selected]
            greedy_predicted_rewards = predicted_rewards.gather(1, greedy_actions_batch.unsqueeze(1)).squeeze(1)
            counterfactual_gain = greedy_predicted_rewards.detach().cpu() - rewards.cpu()
            self.training_metrics['counterfactual_gains'].extend(counterfactual_gain.numpy().tolist())
            
            # Store input_tokens for each sample (for stratified analysis in plots 13-15)
            self.training_metrics['input_tokens_per_sample'].extend(input_tokens_batch)
        
        logger.debug(f"[Update] Loss: {loss.item():.4f}, Avg Reward: {rewards.mean().item():.4f}, "
                    f"Epsilon: {self.epsilon:.4f}, Batch size: {batch_size}")
        
        return {
            'loss': loss.item(),
            'reward': rewards.mean().item(),
            'epsilon': self.epsilon
        }
    
    def evaluate_loss(self, pod_features, kv_hit_ratios, request_features, actions, rewards, sample_weights=None):
        """
        Compute loss on a held-out dataset without updating weights.
        Used for monitoring overfitting during training.
        """
        batch_size = len(actions)
        if batch_size == 0:
            return 0.0

        self.reward_net.eval()
        with torch.no_grad():
            contexts_flat = self._create_per_pod_contexts(pod_features, kv_hit_ratios, request_features).to(device)
            num_pods = pod_features.shape[1]
            actions_t = torch.tensor(actions, dtype=torch.long).to(device)
            rewards_t = torch.tensor(rewards, dtype=torch.float32).to(device)

            predicted_rewards_flat = self.reward_net(contexts_flat)
            predicted_rewards = predicted_rewards_flat.reshape(batch_size, num_pods)
            predicted_action_rewards = predicted_rewards.gather(1, actions_t.unsqueeze(1)).squeeze(1)

            if sample_weights is not None:
                sw = sample_weights.to(device) if isinstance(sample_weights, torch.Tensor) else torch.tensor(sample_weights, dtype=torch.float32).to(device)
                per_sample_loss = (predicted_action_rewards - rewards_t) ** 2
                loss = (per_sample_loss * sw).sum() / sw.sum()
            else:
                loss = F.mse_loss(predicted_action_rewards, rewards_t)

        self.reward_net.train()
        return loss.item()

    def extract_activations_and_predictions(self, contexts, batch_size=512):
        """
        Extract last hidden layer activations and predictions for replay buffer selection.

        Args:
            contexts: [N, per_pod_context_dim] tensor — per-selected-action contexts
        Returns:
            activations: [N, hidden_dim] numpy array (from scorer[6] ReLU output)
            predictions: [N] numpy array
        """
        self.reward_net.eval()

        all_activations = []
        all_predictions = []
        captured = {}

        def hook_fn(module, input, output):
            captured['activation'] = output.detach()

        # Register hook on scorer[6] (ReLU before final Linear)
        hook = self.reward_net.scorer[6].register_forward_hook(hook_fn)

        try:
            N = contexts.shape[0]
            with torch.no_grad():
                for start in range(0, N, batch_size):
                    end = min(start + batch_size, N)
                    batch = contexts[start:end].to(device)
                    preds = self.reward_net(batch)  # [batch, 1]
                    all_predictions.append(preds.squeeze(1).cpu().numpy())
                    all_activations.append(captured['activation'].cpu().numpy())
        finally:
            hook.remove()
            self.reward_net.train()

        activations = np.concatenate(all_activations, axis=0)
        predictions = np.concatenate(all_predictions, axis=0)
        return activations, predictions

    def save(self, final_model_dir, num_trains=None):
        """Save model and metadata"""
        os.makedirs(final_model_dir, exist_ok=True)

        # Build suffix for checkpoint files
        suffix = f"-{num_trains}" if num_trains is not None else ""

        # Save network weights
        torch.save(self.reward_net.state_dict(), os.path.join(final_model_dir, 'reward_net.pth'))
        if suffix:
            torch.save(self.reward_net.state_dict(), os.path.join(final_model_dir, f'reward_net{suffix}.pth'))

        # Save optimizer state
        torch.save(self.optimizer.state_dict(), os.path.join(final_model_dir, 'optimizer.pth'))
        if suffix:
            torch.save(self.optimizer.state_dict(), os.path.join(final_model_dir, f'optimizer{suffix}.pth'))

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
        if suffix:
            with open(os.path.join(final_model_dir, f'metadata{suffix}.pkl'), 'wb') as f:
                pickle.dump(metadata, f)

        logger.info(f"Model saved to {final_model_dir} (checkpoint: {num_trains})")
    
    def load_model(self, final_model_dir, load_metadata=True):
        """Load model and metadata (metadata optional for fast inference reloads)."""
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
        
        # Load metadata (optional - can be heavy for inference hot path)
        if load_metadata:
            metadata_path = os.path.join(final_model_dir, 'metadata.pkl')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                self.epsilon = metadata.get('epsilon', self.epsilon)
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

                # Ensure learning_rates aligns with losses length for CSV export/plots
                losses_len = len(self.training_metrics.get('losses', []))
                lr_list = self.training_metrics.get('learning_rates', [])
                if losses_len > 0:
                    if not isinstance(lr_list, list):
                        lr_list = list(lr_list)
                    if len(lr_list) == 0:
                        # Old models without learning rate history: backfill with current optimizer LR
                        current_lr = self.optimizer.param_groups[0]['lr']
                        self.training_metrics['learning_rates'] = [current_lr] * losses_len
                        logger.info(f"Backfilled learning_rates for {losses_len} steps with current LR={current_lr:.8f}")
                    elif len(lr_list) < losses_len:
                        # Partially missing history: pad the front with the earliest known LR
                        pad_len = losses_len - len(lr_list)
                        pad_value = lr_list[0]
                        self.training_metrics['learning_rates'] = [pad_value] * pad_len + lr_list
                        logger.info(f"Padded learning_rates from {len(lr_list)} to {losses_len} using initial LR={pad_value:.8f}")
                    elif len(lr_list) > losses_len:
                        # Extra entries (should not normally happen) – truncate to match
                        self.training_metrics['learning_rates'] = lr_list[:losses_len]
                        logger.warning(f"Truncated learning_rates from {len(lr_list)} to {losses_len} to match losses length")
                
                logger.info(f"Loaded metadata: epsilon={self.epsilon:.4f}, total_steps={self.total_steps}")
        else:
            logger.info("Skipped metadata.pkl load for faster inference reload")


# Inference function (compatible with existing code)
_cached_agent = None
_cached_metadata = None
_cached_config_key = None
_model_updated_consumed = False
_reload_in_progress = False
_agent_lock = threading.Lock()

def preload_agent_from_metadata(final_model_dir, HYPERPARAMETERS, num_pods=None):
    """Preload agent on startup using metadata.pkl for dimensions."""
    global _cached_agent, _cached_metadata, _cached_config_key, _model_updated_consumed, _reload_in_progress

    metadata_path = os.path.join(final_model_dir, 'metadata.pkl')
    if not os.path.exists(metadata_path):
        logger.error(f"Preload failed: metadata.pkl not found at {metadata_path}")
        return False

    try:
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
    except Exception as e:
        logger.error(f"Preload failed: error reading metadata.pkl: {e}")
        return False

    state_dim = metadata.get('state_dim')
    if not state_dim:
        logger.error("Preload failed: metadata.pkl missing state_dim")
        return False

    action_dim = int(num_pods) if num_pods is not None else int(metadata.get('action_dim', 0))
    if action_dim <= 0:
        logger.error(f"Preload failed: invalid action_dim={action_dim}")
        return False

    config_snapshot = {
        'pod_features': int(state_dim.get('pod_features', 0)),
        'kv_hit_ratios': int(state_dim.get('kv_hit_ratios', 0)),
        'request_features': int(state_dim.get('request_features', 0)),
        'num_pods': int(action_dim)
    }
    if config_snapshot['pod_features'] <= 0 or config_snapshot['kv_hit_ratios'] <= 0 or config_snapshot['request_features'] <= 0:
        logger.error(f"Preload failed: invalid state_dim={state_dim}")
        return False

    config_key = (
        config_snapshot['pod_features'],
        config_snapshot['kv_hit_ratios'],
        config_snapshot['request_features'],
        config_snapshot['num_pods']
    )

    with _agent_lock:
        if _cached_agent is not None and _cached_config_key == config_key:
            logger.info("Preload skipped: agent already initialized with matching config")
            return True

        new_agent = NeuralContextualBandit(
            state_dim={
                'pod_features': config_snapshot['pod_features'],
                'kv_hit_ratios': config_snapshot['kv_hit_ratios'],
                'request_features': config_snapshot['request_features']
            },
            action_dim=config_snapshot['num_pods'],
            hyperparameters=HYPERPARAMETERS,
            final_model_dir=final_model_dir
        )

        # Skip loading pretrained weights when LOAD_PRETRAINED_MODEL=0 or RETRAIN_AT_STARTUP=1
        # (dimensions may differ due to new features, or user wants to start from scratch)
        load_pretrained = HYPERPARAMETERS.get('LOAD_PRETRAINED_MODEL', 1)
        retrain_at_startup = HYPERPARAMETERS.get('RETRAIN_AT_STARTUP', False)
        skip_weights = (not load_pretrained) or retrain_at_startup
        if os.path.exists(os.path.join(final_model_dir, 'reward_net.pth')) and not skip_weights:
            new_agent.load_model(final_model_dir, load_metadata=True)
        elif skip_weights:
            logger.info(f"Preload: Skipping pretrained weights (LOAD_PRETRAINED_MODEL={load_pretrained}, RETRAIN_AT_STARTUP={retrain_at_startup}), using random weights")
        else:
            logger.warning(f"Preload: reward_net.pth not found in {final_model_dir}, using random weights")

        _cached_agent = new_agent
        _cached_metadata = config_snapshot
        _cached_config_key = config_key
        _model_updated_consumed = True
        _reload_in_progress = False

    logger.info(f"Preloaded Neural Contextual Bandit agent with config={config_snapshot}")
    return True

def infer_from_tensor(tensor_data, request_id, model_updated, HYPERPARAMETERS, final_model_dir, sorted_all_pod_ids):
    """
    Inference function compatible with existing routing service
    """
    global _cached_agent, _cached_metadata, _cached_config_key, _model_updated_consumed, _reload_in_progress
    
    infer_start_time = time.time()
    overhead_summary = {}
    
    # Extract tensors
    tensor_transfer_start = time.time()
    pod_features = tensor_data['pod_features_with_staleness'].to(device)
    kv_hit_ratios = tensor_data['kv_hit_ratios'].to(device)
    request_features = tensor_data['request_features'].to(device)
    overhead_summary['tensor_transfer'] = time.time() - tensor_transfer_start
    
    batch_format_start = time.time()
    # Ensure batch format
    if len(pod_features.shape) == 2:
        pod_features = pod_features.unsqueeze(0)
    if len(kv_hit_ratios.shape) == 2:
        kv_hit_ratios = kv_hit_ratios.unsqueeze(0)
    if len(request_features.shape) == 1:
        request_features = request_features.unsqueeze(0)
    overhead_summary['batch_format'] = time.time() - batch_format_start
    # Get or create agent
    get_agent_start = time.time()
    current_config = {
        'pod_features': int(pod_features.shape[2]),
        'kv_hit_ratios': int(kv_hit_ratios.shape[2]),
        'request_features': int(request_features.shape[1]),
        'num_pods': int(pod_features.shape[1])
    }
    current_config_key = (
        current_config['pod_features'],
        current_config['kv_hit_ratios'],
        current_config['request_features'],
        current_config['num_pods']
    )
    
    # Recreate if: dimensions changed, agent doesn't exist, or model_updated flag is set.
    # Fast path: check without lock; only block for missing agent or config changes.
    # For model_updated, start async reload and never block requests.
    # Consume model_updated once per True->False cycle to avoid repeated reloads under concurrency.
    if not model_updated:
        _model_updated_consumed = False
    model_updated_effective = bool(model_updated) and not _model_updated_consumed

    lock_wait_start = time.time()
    needs_sync_reload = (_cached_agent is None) or (_cached_config_key != current_config_key)
    if needs_sync_reload:
        lock_acquire_start = time.time()
        with _agent_lock:
            overhead_summary['get_agent_lock_wait'] = time.time() - lock_acquire_start
            # Double-check under lock
            needs_sync_reload = (_cached_agent is None) or (_cached_config_key != current_config_key)
            if needs_sync_reload:
                reload_start = time.time()
                reload_reasons = []
                if _cached_agent is None:
                    reload_reasons.append("missing_agent")
                if _cached_config_key != current_config_key:
                    reload_reasons.append("config_changed")
                logger.info(
                    "Creating/reloading Neural Contextual Bandit agent "
                    f"(reasons={reload_reasons}, prev_config={_cached_metadata}, new_config={current_config}, "
                    f"model_updated={model_updated}, request_id={request_id})"
                )
                
                state_dim = {
                    'pod_features': current_config['pod_features'],
                    'kv_hit_ratios': current_config['kv_hit_ratios'],
                    'request_features': current_config['request_features']
                }
                
                new_agent = NeuralContextualBandit(
                    state_dim=state_dim,
                    action_dim=current_config['num_pods'],
                    hyperparameters=HYPERPARAMETERS,
                    final_model_dir=final_model_dir
                )
                
                # Try to load existing model
                # Skip if LOAD_PRETRAINED_MODEL=0 or RETRAIN_AT_STARTUP=1 (dimensions may differ from new features)
                load_pretrained = HYPERPARAMETERS.get('LOAD_PRETRAINED_MODEL', 1)
                retrain_at_startup = HYPERPARAMETERS.get('RETRAIN_AT_STARTUP', False)
                skip_weights = (not load_pretrained) or retrain_at_startup
                if os.path.exists(os.path.join(final_model_dir, 'reward_net.pth')) and not skip_weights:
                    load_metadata_on_infer = bool(HYPERPARAMETERS.get('LOAD_METADATA_ON_INFER', False) or CB_LOAD_METADATA_ON_INFER)
                    new_agent.load_model(final_model_dir, load_metadata=load_metadata_on_infer)
                elif skip_weights:
                    logger.info(f"Skipping pretrained model load (LOAD_PRETRAINED_MODEL={load_pretrained}, RETRAIN_AT_STARTUP={retrain_at_startup}), using random weights")

                # Preserve epsilon and total_steps from previous agent to avoid
                # resetting exploration state on model reload (e.g., after online training)
                if _cached_agent is not None:
                    prev_epsilon = _cached_agent.epsilon
                    prev_total_steps = _cached_agent.total_steps
                    new_agent.epsilon = prev_epsilon
                    new_agent.total_steps = prev_total_steps
                    logger.info(f"Preserved exploration state from previous agent: epsilon={prev_epsilon:.4f}, total_steps={prev_total_steps}")

                _cached_agent = new_agent
                _cached_metadata = current_config
                _cached_config_key = current_config_key
                if model_updated_effective:
                    _model_updated_consumed = True
                overhead_summary['get_agent_reload'] = time.time() - reload_start
        overhead_summary.setdefault('get_agent_lock_wait', time.time() - lock_wait_start)
    else:
        overhead_summary['get_agent_lock_wait'] = 0.0

    # Async reload for model updates: never block requests.
    if model_updated_effective and _cached_agent is not None:
        # Capture current exploration state BEFORE async thread starts
        _prev_epsilon = _cached_agent.epsilon
        _prev_total_steps = _cached_agent.total_steps

        def _async_reload(state_dim, action_dim, config_snapshot):
            global _cached_agent, _cached_metadata, _cached_config_key, _reload_in_progress, _model_updated_consumed
            try:
                new_agent = NeuralContextualBandit(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    hyperparameters=HYPERPARAMETERS,
                    final_model_dir=final_model_dir
                )
                if os.path.exists(os.path.join(final_model_dir, 'reward_net.pth')):
                    load_metadata_on_infer = bool(HYPERPARAMETERS.get('LOAD_METADATA_ON_INFER', False) or CB_LOAD_METADATA_ON_INFER)
                    try:
                        new_agent.load_model(final_model_dir, load_metadata=load_metadata_on_infer)
                    except RuntimeError as e:
                        if "size mismatch" in str(e):
                            logger.warning(f"Async reload: dimension mismatch loading saved model, using random weights: {e}")
                        else:
                            raise

                # Preserve epsilon and total_steps from previous agent
                new_agent.epsilon = _prev_epsilon
                new_agent.total_steps = _prev_total_steps
                logger.info(f"Async reload: preserved exploration state: epsilon={_prev_epsilon:.4f}, total_steps={_prev_total_steps}")

                # Swap under lock (very short critical section)
                with _agent_lock:
                    _cached_agent = new_agent
                    _cached_metadata = config_snapshot
                    _cached_config_key = (
                        config_snapshot['pod_features'],
                        config_snapshot['kv_hit_ratios'],
                        config_snapshot['request_features'],
                        config_snapshot['num_pods']
                    )
            except Exception:
                logger.exception("Async reload failed")
                _model_updated_consumed = False
            finally:
                _reload_in_progress = False

        acquired = _agent_lock.acquire(blocking=False)
        if acquired:
            try:
                if not _reload_in_progress:
                    _reload_in_progress = True
                    _model_updated_consumed = True
                    overhead_summary['get_agent_async_reload_started'] = 1.0
                    state_dim = {
                        'pod_features': current_config['pod_features'],
                        'kv_hit_ratios': current_config['kv_hit_ratios'],
                        'request_features': current_config['request_features']
                    }
                    t = threading.Thread(
                        target=_async_reload,
                        args=(state_dim, current_config['num_pods'], current_config),
                        daemon=True
                    )
                    t.start()
                else:
                    overhead_summary['get_agent_async_reload_started'] = 0.0
            finally:
                _agent_lock.release()
        else:
            overhead_summary['get_agent_lock_skipped'] = 1.0
    
    overhead_summary['get_agent'] = time.time() - get_agent_start
    
    # Inference
    inference_start = time.time()
    action, predicted_rewards, explored = _cached_agent.choose_action(
        pod_features, kv_hit_ratios, request_features, 
        evaluate=not HYPERPARAMETERS.get('explore', True)
    )
    overhead_summary['inference'] = time.time() - inference_start
    
    result_formatting_start = time.time()
    # Always keep predicted_rewards as numpy array. JSON serialization happens in routing_agent_service.py.
    chosen_pod_predicted_reward = float(predicted_rewards[action])
    
    # Prepare probabilities (optional)
    pod_probabilities = None
    if HYPERPARAMETERS.get('RETURN_POD_PROBABILITIES', 0) >= 1:
        # Use numerically stable softmax; vectorized for speed
        rewards_array = np.asarray(predicted_rewards, dtype=np.float64)
        max_reward = np.max(rewards_array)
        exp_rewards = np.exp(rewards_array - max_reward)
        sum_exp_rewards = np.sum(exp_rewards)
        if sum_exp_rewards == 0.0 or not np.isfinite(sum_exp_rewards):
            pod_probabilities = {pod_id: 1.0 / len(sorted_all_pod_ids) for pod_id in sorted_all_pod_ids}
        else:
            probs = exp_rewards / sum_exp_rewards
            pod_probabilities = {sorted_all_pod_ids[i]: float(probs[i]) for i in range(len(sorted_all_pod_ids))}
    
    result = {
        'selected_pod_index': int(action),
        'predicted_rewards': predicted_rewards,
        'chosen_pod_predicted_reward': chosen_pod_predicted_reward,
        'pod_probabilities': pod_probabilities,
        'confidence': chosen_pod_predicted_reward,  # Keep for backward compatibility
        'epsilon': _cached_agent.epsilon,
        'total_steps': _cached_agent.total_steps,
        'explored': explored
    }
    overhead_summary['result_formatting'] = time.time() - result_formatting_start
    overhead_summary['total_inference'] = time.time() - infer_start_time
    
    return result, overhead_summary


def train(encoded_training_dir, final_model_dir, HYPERPARAMETERS, num_trains):
    """
    Train neural contextual bandit on batch of encoded experiences.
    Compatible with existing data pipeline (called from online_train_routine).
    
    Args:
        encoded_training_dir: Directory containing encoded .pt files
        final_model_dir: Directory to save model
        HYPERPARAMETERS: Model hyperparameters
            - ONLINE_TRAIN_FROM_SCRATCH (bool, default=False): If True, start training 
              from random initialization (scratch). If False, load and continue from 
              last trained weights (default behavior).
    """
    global _cached_agent
    
    logger.info(f"Starting Neural CB batch training: num_trains={num_trains}, epochs={HYPERPARAMETERS.get('training_epochs', 10)}, dir={encoded_training_dir}")
    logger.info(f"HYPERPARAMETERS: {HYPERPARAMETERS}")
    
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

    # ===== TRAIN / TEST SPLIT =====
    test_split_ratio = HYPERPARAMETERS.get('test_split_ratio', 0.2)
    split_seed = HYPERPARAMETERS.get('training_seed', 42)
    split_rng = np.random.RandomState(seed=split_seed)

    train_tensor_files = []
    test_data_list = []  # list of dicts, each a tensor_dataset
    total_train_samples_counted = 0

    if len(tensor_files) >= 5:
        # Enough files: split by file
        perm = split_rng.permutation(len(tensor_files))
        n_test = max(1, int(len(tensor_files) * test_split_ratio))
        test_indices = set(perm[:n_test].tolist())
        for idx, tf in enumerate(tensor_files):
            if idx in test_indices:
                test_data_list.append(torch.load(tf))
            else:
                train_tensor_files.append(tf)
    else:
        # Few files: split samples within each file
        for tf in tensor_files:
            data = torch.load(tf)
            n = len(data['actions'])
            indices = split_rng.permutation(n)
            n_test = max(1, int(n * test_split_ratio))
            test_idx = torch.tensor(indices[:n_test])
            train_idx = torch.tensor(indices[n_test:])

            # Build test split
            test_split = {}
            train_split = {}
            for key in data:
                if isinstance(data[key], torch.Tensor):
                    if data[key].shape[0] == n:
                        test_split[key] = data[key][test_idx]
                        train_split[key] = data[key][train_idx]
                    else:
                        test_split[key] = data[key]
                        train_split[key] = data[key]
                else:
                    test_split[key] = data[key]
                    train_split[key] = data[key]

            test_data_list.append(test_split)
            total_train_samples_counted += len(train_idx)
            # Save train split to a temp file so training loop can load it consistently
            tmp_dir = tempfile.mkdtemp(prefix='train_split_')
            tmp_path = os.path.join(tmp_dir, 'tensor_dataset.pt')
            torch.save(train_split, tmp_path)
            train_tensor_files.append(tmp_path)

    total_test_samples = sum(len(d['actions']) for d in test_data_list)
    logger.info(f"Train/Test split: {len(train_tensor_files)} train files, "
                f"{len(test_data_list)} test chunks ({total_test_samples} samples), "
                f"test_split_ratio={test_split_ratio}")

    # Load first file to get dimensions
    batch_data = torch.load(train_tensor_files[0]) if train_tensor_files else torch.load(tensor_files[0])
    
    # Check if we should start from scratch or load existing model
    # Default behavior: load from last trained weights (backward compatible)
    start_from_scratch = HYPERPARAMETERS.get('ONLINE_TRAIN_FROM_SCRATCH', False)
    
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
        
        if start_from_scratch:
            logger.info("🔄 ONLINE_TRAIN_FROM_SCRATCH=True: Starting training from scratch (random initialization)")
            # Agent is already initialized with random weights, no need to load
        else:
            # Try to load existing model (default behavior)
            model_path = os.path.join(final_model_dir, 'reward_net.pth')
            if os.path.exists(model_path):
                try:
                    _cached_agent.load_model(final_model_dir)
                    logger.info(f"✅ Loaded existing model from {final_model_dir} (continuing from last trained weights)")
                except Exception as e:
                    logger.warning(f"Failed to load existing model: {e}, starting fresh")
            else:
                logger.info(f"ℹ️  No existing model found at {model_path}, starting from scratch (random initialization)")
    else:
        # Agent already exists - check if we need to reset it for from-scratch training
        if start_from_scratch:
            logger.info("🔄 ONLINE_TRAIN_FROM_SCRATCH=True: Reinitializing agent from scratch (random weights)")
            # Reinitialize the agent with random weights
            state_dim = {
                'pod_features': batch_data['pod_features_with_staleness'].shape[2],
                'kv_hit_ratios': batch_data['kv_hit_ratios'].shape[2],
                'request_features': batch_data['request_features'].shape[1]
            }
            action_dim = batch_data['pod_features_with_staleness'].shape[1]
            
            _cached_agent = NeuralContextualBandit(
                state_dim=state_dim,
                action_dim=action_dim,
                hyperparameters=HYPERPARAMETERS,
                final_model_dir=final_model_dir
            )
            # Don't load any weights - use random initialization
        else:
            # Agent exists and we want to continue from last weights
            # Check if model was updated since agent was created
            model_path = os.path.join(final_model_dir, 'reward_net.pth')
            if os.path.exists(model_path):
                try:
                    # Reload to ensure we have the latest weights
                    _cached_agent.load_model(final_model_dir)
                    logger.info(f"✅ Reloaded existing model from {final_model_dir} (continuing from last trained weights)")
                except Exception as e:
                    logger.warning(f"Failed to reload existing model: {e}, using current agent state")
    
    # Reset LR at the start of each online training round so the model isn't frozen by decayed LR
    reset_lr = HYPERPARAMETERS.get('RESET_LR_PER_ROUND', bool(1))
    if reset_lr and _cached_agent.scheduler is not None and num_trains is not None:
        initial_lr = HYPERPARAMETERS.get('learning_rate', 0.0003)
        for param_group in _cached_agent.optimizer.param_groups:
            param_group['lr'] = initial_lr
        # Recreate scheduler so it decays fresh within this round
        if _cached_agent.scheduler_type == 'exponential':
            gamma = HYPERPARAMETERS.get('lr_scheduler_gamma', 0.95)
            _cached_agent.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                _cached_agent.optimizer, gamma=gamma)
        logger.info(f"Reset LR to {initial_lr} for training round {num_trains}")

    # Reset training_metrics so each round's CSV only contains that round's data
    action_dim = _cached_agent.action_dim
    _cached_agent.training_metrics = {
        'losses': [],
        'eval_losses': [],
        'rewards': [],
        'epsilons': [],
        'learning_rates': [],
        'reward_latency_pairs': [],
        'reward_latency_input_tuples': [],
        'ttft_values': [],
        'tpot_values': [],
        'action_distribution': np.zeros(action_dim),
        'predicted_rewards': [],
        'actual_rewards': [],
        'selected_actions': [],
        'exploration_count': 0,
        'exploitation_count': 0,
        'all_predicted_rewards': [],
        'greedy_actions': [],
        'training_actions': [],
        'counterfactual_gains': [],
        'input_tokens_per_sample': []
    }

    # Training loop
    total_samples = 0
    total_input_samples = 0
    for epoch in range(HYPERPARAMETERS.get('training_epochs', 10)):
        epoch_start = time.time()
        epoch_losses = []
        epoch_rewards = []

        # Shuffle tensor files at the start of each epoch to prevent periodic loss spikes
        # Use epoch-specific random state to ensure different order each epoch despite global seed
        shuffled_tensor_files = train_tensor_files.copy()
        # Use abs() and modulo to ensure seed is within valid range [0, 2^32-1]
        epoch_seed = (HYPERPARAMETERS.get('training_seed', 42) + epoch + abs(int(hash(time.time())))) % (2**32)
        epoch_rng = np.random.RandomState(seed=epoch_seed)
        epoch_rng.shuffle(shuffled_tensor_files)

        for tensor_file in shuffled_tensor_files:
            # tensor_file is already the full path
            batch_data = torch.load(tensor_file)

            # Extract tensors
            pod_features = batch_data['pod_features_with_staleness']
            kv_hit_ratios = batch_data['kv_hit_ratios']
            request_features = batch_data['request_features']
            actions = batch_data['actions']
            rewards = batch_data['rewards']

            # Extract latency and context values for reward function analysis
            ttft = batch_data.get('ttft', None)
            avg_tpot = batch_data.get('avg_tpot', None)
            input_tokens = batch_data.get('input_tokens', None)
            sample_weights = batch_data.get('sample_weights', None)

            dataset_size = len(actions)
            if epoch == 0:
                total_input_samples += dataset_size
            train_batch_size = HYPERPARAMETERS.get('batch_size', 256)

            # Split dataset into mini-batches for training
            num_batches = (dataset_size + train_batch_size - 1) // train_batch_size

            # Shuffle indices for this epoch
            indices = torch.randperm(dataset_size)

            for batch_idx in range(num_batches):
                start_idx = batch_idx * train_batch_size
                end_idx = min(start_idx + train_batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]

                # Extract mini-batch
                batch_pod_features = pod_features[batch_indices]
                batch_kv_ratios = kv_hit_ratios[batch_indices]
                batch_request_features = request_features[batch_indices]
                batch_actions = actions[batch_indices]
                batch_rewards = rewards[batch_indices]
                batch_input_tokens = input_tokens[batch_indices] if input_tokens is not None else None
                batch_weights = sample_weights[batch_indices] if sample_weights is not None else None

                batch_size = len(batch_actions)
                total_samples += batch_size

                # Collect reward-latency-context tuples for stratified function analysis (sample 10% to save memory)
                if ttft is not None:
                    batch_ttft = ttft[batch_indices]
                    batch_tpot = avg_tpot[batch_indices] if avg_tpot is not None else None

                    sample_indices = np.random.choice(batch_size, size=max(1, int(batch_size * 0.1)), replace=False)
                    for i in sample_indices:
                        _cached_agent.training_metrics['reward_latency_pairs'].append(
                            (batch_rewards[i].item(), batch_ttft[i].item())
                        )
                        _cached_agent.training_metrics['ttft_values'].append(batch_ttft[i].item())
                        if batch_tpot is not None:
                            _cached_agent.training_metrics['tpot_values'].append(batch_tpot[i].item())

                        # Store (reward, latency, input_tokens) for stratified analysis
                        if batch_input_tokens is not None:
                            _cached_agent.training_metrics['reward_latency_input_tuples'].append(
                                (batch_rewards[i].item(), batch_ttft[i].item(), batch_input_tokens[i].item())
                            )

                # Train on mini-batch
                metrics = _cached_agent.learn(
                    batch_pod_features,
                    batch_kv_ratios,
                    batch_request_features,
                    batch_actions,
                    batch_rewards,
                    input_tokens=batch_input_tokens,
                    sample_weights=batch_weights
                )
                epoch_losses.append(metrics['loss'])
                epoch_rewards.append(metrics['reward'])
        
        # Log epoch metrics
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0.0
        epoch_time = time.time() - epoch_start
        
        # Update learning rate scheduler at epoch level
        if _cached_agent.scheduler is not None:
            current_lr = _cached_agent.optimizer.param_groups[0]['lr']
            
            if _cached_agent.scheduler_type == 'exponential':
                # Exponential decay - step once per epoch
                _cached_agent.scheduler.step()
                new_lr = _cached_agent.optimizer.param_groups[0]['lr']
                logger.info(f"📉 LR Scheduler (exponential): LR={current_lr:.8f} → {new_lr:.8f}")
                
            elif _cached_agent.scheduler_type == 'gradient_adaptive':
                if len(_cached_agent.gradient_norms) > 0:
                    # Use recent gradient norms from this epoch
                    recent_grad_norms = _cached_agent.gradient_norms[-len(epoch_losses):] if len(_cached_agent.gradient_norms) >= len(epoch_losses) else _cached_agent.gradient_norms
                    avg_grad_norm = np.mean(recent_grad_norms) if len(recent_grad_norms) > 0 else 0.0
                    
                    # Check gradient norm conditions
                    if avg_grad_norm < _cached_agent.min_grad_norm_threshold:
                        # Gradients too small - might need LR reduction or training is done
                        _cached_agent.scheduler.step(0.1)  # Trigger reduction
                        new_lr = _cached_agent.optimizer.param_groups[0]['lr']
                        logger.info(f"🔍 Gradient norm too small ({avg_grad_norm:.8f} < {_cached_agent.min_grad_norm_threshold:.8f}) - reducing LR: {current_lr:.8f} → {new_lr:.8f}")
                    elif avg_grad_norm > _cached_agent.max_grad_norm_threshold:
                        # Gradients too large - might be unstable
                        _cached_agent.scheduler.step(0.1)  # Trigger reduction
                        new_lr = _cached_agent.optimizer.param_groups[0]['lr']
                        logger.info(f"⚡ Gradient norm too large ({avg_grad_norm:.8f} > {_cached_agent.max_grad_norm_threshold:.2f}) - reducing LR: {current_lr:.8f} → {new_lr:.8f}")
                    else:
                        # Gradients in good range - no change needed
                        logger.info(f"✅ Gradient norm healthy ({avg_grad_norm:.6f}) - keeping LR: {current_lr:.8f}")
                else:
                    logger.warning("No gradient norms collected - skipping gradient-based scheduling")
        
        # ===== TEST SET EVALUATION =====
        if test_data_list:
            test_losses = []
            for test_data in test_data_list:
                test_loss = _cached_agent.evaluate_loss(
                    test_data['pod_features_with_staleness'],
                    test_data['kv_hit_ratios'],
                    test_data['request_features'],
                    test_data['actions'],
                    test_data['rewards'],
                    sample_weights=test_data.get('sample_weights', None)
                )
                test_losses.append(test_loss)
            avg_test_loss = np.mean(test_losses)
            _cached_agent.training_metrics['eval_losses'].append(avg_test_loss)
            overfit_gap = avg_test_loss - avg_loss
            logger.info(f"Epoch {epoch+1}/{HYPERPARAMETERS.get('training_epochs', 10)}: "
                       f"train_loss={avg_loss:.4f}, test_loss={avg_test_loss:.4f}, "
                       f"gap={overfit_gap:+.4f}, avg_reward={avg_reward:.4f}, time={epoch_time:.2f}s")
        else:
            logger.info(f"Epoch {epoch+1}/{HYPERPARAMETERS.get('training_epochs', 10)}: loss={avg_loss:.4f}, avg_reward={avg_reward:.4f}, "
                       f"time={epoch_time:.2f}s")

    # Clean up temp split files
    for tf in train_tensor_files:
        if '/train_split_' in tf:
            tmp_dir = os.path.dirname(tf)
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)

    # Save trained model
    _cached_agent.save(final_model_dir, num_trains=num_trains)
    logger.info(f"Neural CB batch training complete: {total_samples} samples processed, model saved to {final_model_dir}")
    
    # Generate comprehensive training plots (use total_steps as num_trains)
    plot_path = plot_neural_cb_metrics(_cached_agent, final_model_dir, HYPERPARAMETERS.get('training_epochs', 10), total_samples, num_trains=num_trains, total_input_samples=total_input_samples)
    return plot_path


def plot_neural_cb_metrics(agent, final_model_dir, training_epochs, total_samples, num_trains, total_input_samples=None):
    """
    Create comprehensive training metrics visualization for Neural Contextual Bandit.
    
    Args:
        agent: Trained NeuralContextualBandit instance
        final_model_dir: Directory to save plots
        training_epochs: Number of training epochs
        total_samples: Total number of samples processed
        num_trains: Number of training iterations (for filename)
    
    Returns:
        Path to saved plot file
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
    # Set matplotlib style
    plt.style.use('default')
    sns.set_palette("husl")
    
    metrics = agent.training_metrics
    
    if len(metrics['losses']) == 0:
        logger.warning("No training metrics to plot")
        return None
    
    # Create comprehensive plot (3 rows x 4 cols = 12 plots)
    fig = plt.figure(figsize=(24, 16))
    display_samples = total_input_samples if total_input_samples is not None else total_samples
    fig.suptitle(f'Neural Contextual Bandit Training Results\n'
                 f'Epochs: {training_epochs} | Total Samples: {display_samples:,} | Updates: {len(metrics["losses"]):,}',
                 fontsize=16, fontweight='bold', y=0.98)
    
    # === ROW 1: TRAINING OVERVIEW ===
    # 1. Training & Evaluation Loss
    plt.subplot(3, 4, 1)
    if metrics['losses']:
        plt.plot(metrics['losses'], 'b-', linewidth=1.5, alpha=0.7, label='Training Loss')
        # Add moving average for training loss
        if len(metrics['losses']) > 10:
            window = min(50, len(metrics['losses']) // 10)
            moving_avg = pd.Series(metrics['losses']).rolling(window=window).mean()
            plt.plot(moving_avg, 'b--', linewidth=2, alpha=0.5, label=f'Train MA ({window})')
        
        # Plot evaluation loss if available — as a step-wise line spanning each epoch
        if metrics.get('eval_losses') and len(metrics['eval_losses']) > 0:
            eval_losses = metrics['eval_losses']
            n_updates = len(metrics['losses'])
            n_epochs = len(eval_losses)
            # Hold each epoch's eval loss constant across its update steps
            eval_line = np.empty(n_updates)
            for ei in range(n_epochs):
                start = int(ei * n_updates / n_epochs)
                end = int((ei + 1) * n_updates / n_epochs)
                eval_line[start:end] = eval_losses[ei]
            plt.plot(eval_line, 'r-', linewidth=2, alpha=0.8, label='Test Loss')
            # Draw vertical lines at epoch boundaries
            for ei in range(1, n_epochs):
                x = int(ei * n_updates / n_epochs)
                plt.axvline(x=x, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

        plt.legend(fontsize=8)
        plt.title('1. Training & Evaluation Loss')
        plt.xlabel('Update Step')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        
        # Add final loss annotation
        final_loss = metrics['losses'][-1]
        avg_loss = np.mean(metrics['losses'][-100:]) if len(metrics['losses']) >= 100 else np.mean(metrics['losses'])
        annotation_text = f'Train Final: {final_loss:.4f}\nTrain Avg (last 100): {avg_loss:.4f}'
        if metrics.get('eval_losses') and len(metrics['eval_losses']) > 0:
            final_eval_loss = metrics['eval_losses'][-1]
            annotation_text += f'\nEval Final: {final_eval_loss:.4f}'
        plt.text(0.02, 0.98, annotation_text,
                transform=plt.gca().transAxes, verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 2. Average Reward
    plt.subplot(3, 4, 2)
    if metrics['rewards']:
        plt.plot(metrics['rewards'], 'g-', linewidth=1.5, alpha=0.7)
        # Add moving average
        if len(metrics['rewards']) > 10:
            window = min(50, len(metrics['rewards']) // 10)
            moving_avg = pd.Series(metrics['rewards']).rolling(window=window).mean()
            plt.plot(moving_avg, 'darkgreen', linewidth=2, label=f'{window}-step MA')
            plt.legend()
        plt.title('2. Average Reward per Update')
        plt.xlabel('Update Step')
        plt.ylabel('Reward')
        plt.grid(True, alpha=0.3)
        
        # Add reward statistics
        final_reward = metrics['rewards'][-1]
        avg_reward = np.mean(metrics['rewards'][-100:]) if len(metrics['rewards']) >= 100 else np.mean(metrics['rewards'])
        plt.text(0.02, 0.98, f'Final: {final_reward:.4f}\nAvg (last 100): {avg_reward:.4f}',
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # 3. Exploration Rate (Epsilon) and Learning Rate (dual axis)
    plt.subplot(3, 4, 3)
    if metrics['epsilons'] and metrics.get('learning_rates'):
        ax1 = plt.gca()
        color1 = 'orange'
        ax1.set_xlabel('Update Step', fontsize=10)
        ax1.set_ylabel('Epsilon', color=color1, fontsize=10)
        line1 = ax1.plot(metrics['epsilons'], color=color1, linewidth=2, label='Epsilon')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, max(metrics['epsilons']) * 1.1)
        
        # Add learning rate on secondary y-axis
        ax2 = ax1.twinx()
        color2 = 'blue'
        ax2.set_ylabel('Learning Rate', color=color2, fontsize=10)
        line2 = ax2.plot(metrics['learning_rates'], color=color2, linewidth=2, linestyle='--', alpha=0.7, label='LR')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        plt.title('3. Epsilon & Learning Rate', fontsize=11, fontweight='bold')
        
        # Add statistics
        initial_eps = metrics['epsilons'][0]
        final_eps = metrics['epsilons'][-1]
        initial_lr = metrics['learning_rates'][0]
        final_lr = metrics['learning_rates'][-1]
        plt.text(0.02, 0.98, f'ε: {initial_eps:.4f}→{final_eps:.4f}\nLR: {initial_lr:.6f}→{final_lr:.6f}',
                transform=ax1.transAxes, verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    elif metrics['epsilons']:
        # Fallback if learning rates not available
        plt.plot(metrics['epsilons'], 'orange', linewidth=2)
        plt.title('3. Exploration Rate (Epsilon)')
        plt.xlabel('Update Step')
        plt.ylabel('Epsilon')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, max(metrics['epsilons']) * 1.1)
        
        initial_eps = metrics['epsilons'][0]
        final_eps = metrics['epsilons'][-1]
        plt.text(0.02, 0.98, f'Initial: {initial_eps:.4f}\nFinal: {final_eps:.4f}',
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # === ROW 2: OFF-POLICY ANALYSIS ===
    # 5. Reward Distribution (sample-level, not batch-level)
    plt.subplot(3, 4, 5)

    # Load actual sample rewards from encoded data
    import torch
    import glob
    sample_rewards = None
    encoded_dir = os.path.join(final_model_dir, 'encoded_data')
    if os.path.exists(encoded_dir):
        # Find first tensor file
        tensor_files = glob.glob(os.path.join(encoded_dir, 'batch_*', 'tensor_dataset.pt'))
        if tensor_files:
            try:
                data = torch.load(tensor_files[0])
                if 'rewards' in data:
                    sample_rewards = data['rewards'].numpy()
            except Exception as e:
                logger.warning(f"Could not load sample rewards: {e}")

    if sample_rewards is not None:
        # Plot actual sample reward distribution
        plt.hist(sample_rewards, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.axvline(np.mean(sample_rewards), color='r', linestyle='--', linewidth=2, label='Mean')
        plt.axvline(np.median(sample_rewards), color='g', linestyle='--', linewidth=2, label='Median')
        plt.title('5. Sample Reward Distribution')
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add statistics
        mean_reward = np.mean(sample_rewards)
        std_reward = np.std(sample_rewards)
        plt.text(0.98, 0.98, f'Samples: {len(sample_rewards)}\nMean: {mean_reward:.4f}\nStd: {std_reward:.4f}',
                transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    elif metrics['rewards']:
        # Fallback: plot batch mean rewards if sample rewards not available
        plt.hist(metrics['rewards'], bins=50, alpha=0.7, color='orange', edgecolor='black')
        plt.axvline(np.mean(metrics['rewards']), color='r', linestyle='--', linewidth=2, label='Mean')
        plt.axvline(np.median(metrics['rewards']), color='g', linestyle='--', linewidth=2, label='Median')
        plt.title('5. Batch Mean Reward Dist (fallback)')
        plt.xlabel('Batch Mean Reward')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)

        mean_reward = np.mean(metrics['rewards'])
        std_reward = np.std(metrics['rewards'])
        plt.text(0.98, 0.98, f'Batches: {len(metrics["rewards"])}\nMean: {mean_reward:.4f}\nStd: {std_reward:.4f}',
                transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='orange', alpha=0.8))
    
    # 4. Model Architecture Info
    plt.subplot(3, 4, 4)
    plt.axis('off')
    plt.title('4. Model Architecture', pad=10)
    
    arch_text = "NEURAL CONTEXTUAL BANDIT\n" + "="*25 + "\n"
    arch_text += f"Exploration: {getattr(agent, 'exploration_method', 'N/A')}\n"
    
    # Get initial epsilon from first metric or current epsilon
    initial_eps = agent.training_metrics['epsilons'][0] if agent.training_metrics['epsilons'] else getattr(agent, 'epsilon', 0.3)
    arch_text += f"Initial ε: {initial_eps:.3f}\n"
    arch_text += f"Final ε: {getattr(agent, 'epsilon', 0.0):.3f}\n"
    arch_text += f"Decay: {getattr(agent, 'epsilon_decay', 0.0):.4f}\n"
    arch_text += f"Min ε: {getattr(agent, 'epsilon_min', 0.0):.3f}\n\n"
    arch_text += f"Buffer Size: {getattr(agent, 'buffer_size', 0)}\n"
    arch_text += f"Batch Size: {getattr(agent, 'batch_size', 0)}\n"
    
    # Extract from hyperparameters since they're not stored as attributes
    learning_rate = agent.hyperparameters.get('learning_rate', 3e-4) if hasattr(agent, 'hyperparameters') else 0.0
    hidden_dim = agent.hyperparameters.get('hidden_dim', 128) if hasattr(agent, 'hyperparameters') else 0
    gamma = agent.hyperparameters.get('gamma', 0.99) if hasattr(agent, 'hyperparameters') else 0.0
    
    arch_text += f"Learning Rate: {learning_rate:.6f}\n"
    arch_text += f"Gamma (discount): {gamma:.3f}\n\n"
    arch_text += f"Context Dim: {getattr(agent, 'context_dim', 0)}\n"
    arch_text += f"Action Dim: {getattr(agent, 'action_dim', 0)}\n"
    arch_text += f"Hidden Dim: {hidden_dim}\n"
    
    total_params = sum(p.numel() for p in agent.reward_net.parameters())
    arch_text += f"Parameters: {total_params:,}\n"
    
    plt.text(0.1, 0.9, arch_text, transform=plt.gca().transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 6. Counterfactual Gain Distribution (Stratified by Input Length)
    plt.subplot(3, 4, 6)
    if metrics.get('counterfactual_gains') and metrics.get('input_tokens_per_sample'):
        gains = np.array(metrics['counterfactual_gains'])
        inp_tokens_metric = np.array(metrics['input_tokens_per_sample'])
        
        # Validate lengths match (defensive check)
        if len(gains) != len(inp_tokens_metric):
            min_len = min(len(gains), len(inp_tokens_metric))
            logger.warning(f"Plot 13: Length mismatch - gains={len(gains)}, input_tokens={len(inp_tokens_metric)}. Truncating to {min_len}")
            gains = gains[:min_len]
            inp_tokens_metric = inp_tokens_metric[:min_len]
        
        # Filter out samples without input_tokens
        valid_mask = inp_tokens_metric > 0
        if valid_mask.sum() > 10:
            gains_valid = gains[valid_mask]
            inp_tokens_valid = inp_tokens_metric[valid_mask]
            
            # Define input length buckets
            inp_quantiles_13 = np.percentile(inp_tokens_valid, [0, 33, 67, 100])
            bucket_colors_13 = ['green', 'orange', 'red']
            bucket_names_13 = [
                f'Short\n({inp_quantiles_13[0]:.0f}-{inp_quantiles_13[1]:.0f})',
                f'Med\n({inp_quantiles_13[1]:.0f}-{inp_quantiles_13[2]:.0f})',
                f'Long\n({inp_quantiles_13[2]:.0f}-{inp_quantiles_13[3]:.0f})'
            ]
            
            # Plot overall distribution first (all requests)
            plt.hist(gains_valid, bins=30, alpha=0.3, 
                    color='blue', histtype='stepfilled',
                    label='All requests', zorder=1)
            
            # Plot stratified histograms with step style for better visibility
            for i, (low, high) in enumerate([(inp_quantiles_13[0], inp_quantiles_13[1]), 
                                              (inp_quantiles_13[1], inp_quantiles_13[2]), 
                                              (inp_quantiles_13[2], inp_quantiles_13[3])]):
                mask = (inp_tokens_valid >= low) & (inp_tokens_valid < high) if i < 2 else (inp_tokens_valid >= low)
                if mask.sum() > 0:
                    # Use histtype='step' for better visibility when overlapping
                    plt.hist(gains_valid[mask], bins=30, alpha=1.0, color=bucket_colors_13[i], 
                            histtype='step', linewidth=2.5, label=bucket_names_13[i], zorder=2)
            
            plt.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.8, label='Zero Gain', zorder=3)
            plt.xlabel('Counterfactual Gain', fontsize=10)
            plt.ylabel('Frequency', fontsize=10)
            plt.title('6. Gain by Input Length', fontsize=11, fontweight='bold')
            plt.legend(fontsize=7, loc='upper right')
            plt.grid(True, alpha=0.3)
            
            mean_gain = np.mean(gains_valid)
            pct_better = (gains_valid > 0).sum() / len(gains_valid) * 100
            assessment = "✅ BETTER" if mean_gain > 0.05 else ("➡️ Modest" if mean_gain > 0.01 else "⚠️ Similar")
            
            plt.text(0.02, 0.98, f'Mean: {mean_gain:.4f}\nBetter: {pct_better:.1f}%\n{assessment}',
                    transform=plt.gca().transAxes, verticalalignment='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        else:
            # Fallback: non-stratified
            plt.hist(gains, bins=50, alpha=0.7, color='purple', edgecolor='black')
            plt.axvline(0, color='r', linestyle='--', linewidth=2)
            plt.axvline(np.mean(gains), color='g', linestyle='-', linewidth=2)
            plt.xlabel('Counterfactual Gain')
            plt.ylabel('Frequency')
            plt.title('6. Expected Gain (Aggregated)')
            plt.grid(True, alpha=0.3)
            mean_gain = np.mean(gains)
            plt.text(0.02, 0.98, f'Mean: {mean_gain:.4f}',
                    transform=plt.gca().transAxes, verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    else:
        plt.text(0.5, 0.5, 'No counterfactual\ndata', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.axis('off')
    
    # 7. Per-Context Action Differentiation (SNR by Input Length)
    plt.subplot(3, 4, 7)
    if metrics.get('all_predicted_rewards') and len(metrics['all_predicted_rewards']) > 0:
        # Stack all predictions [num_samples, num_actions]
        all_preds = np.concatenate(metrics['all_predicted_rewards'], axis=0)
        
        # CRITICAL METRIC: For each sample, what's the spread between best and worst pod?
        action_spreads = all_preds.max(axis=1) - all_preds.min(axis=1)
        
        # Get prediction uncertainty (RMSE)
        avg_loss = np.mean(metrics['losses'][-100:]) if len(metrics['losses']) >= 100 else np.mean(metrics['losses'])
        rmse = np.sqrt(avg_loss)
        
        # Try to stratify by input length (if available and matching length)
        if metrics.get('input_tokens_per_sample') and len(metrics['input_tokens_per_sample']) >= len(action_spreads):
            # Match samples (first N samples from input_tokens_per_sample that correspond to all_predicted_rewards)
            inp_tokens_for_spreads = np.array(metrics['input_tokens_per_sample'][:len(action_spreads)])
            valid_mask = inp_tokens_for_spreads > 0
            
            if valid_mask.sum() > 10:
                spreads_valid = action_spreads[valid_mask]
                inp_tokens_valid = inp_tokens_for_spreads[valid_mask]
                
                # Calculate overall SNR first
                overall_mean_spread = np.mean(spreads_valid)
                overall_snr = overall_mean_spread / rmse if rmse > 0 else 0
                
                # Define buckets and calculate SNR per bucket
                inp_quantiles_15 = np.percentile(inp_tokens_valid, [0, 33, 67, 100])
                snr_per_bucket = []
                bucket_labels_15 = []
                
                for i, (low, high) in enumerate([(inp_quantiles_15[0], inp_quantiles_15[1]), 
                                                  (inp_quantiles_15[1], inp_quantiles_15[2]), 
                                                  (inp_quantiles_15[2], inp_quantiles_15[3])]):
                    mask = (inp_tokens_valid >= low) & (inp_tokens_valid < high) if i < 2 else (inp_tokens_valid >= low)
                    if mask.sum() > 0:
                        mean_spread_bucket = np.mean(spreads_valid[mask])
                        snr_bucket = mean_spread_bucket / rmse if rmse > 0 else 0
                        snr_per_bucket.append(snr_bucket)
                        bucket_labels_15.append(f'Len{i+1}\n{int(low)}-{int(high)}')
                
                # Add overall bar
                snr_per_bucket.append(overall_snr)
                bucket_labels_15.append('All\nRequests')
                
                # Plot SNR per bucket
                colors_15 = ['green', 'orange', 'red', 'blue'][:len(snr_per_bucket)]
                x_pos = np.arange(len(snr_per_bucket))
                bars = plt.bar(x_pos, snr_per_bucket, color=colors_15, 
                              alpha=0.7, edgecolor='black')
                plt.axhline(y=1.0, color='blue', linestyle='--', linewidth=1.5, alpha=0.5, label='SNR=1.0')
                plt.ylabel('SNR (Spread/RMSE)', fontsize=10)
                plt.title('7. SNR by Input Length', fontsize=11, fontweight='bold')
                plt.xticks(x_pos, bucket_labels_15, fontsize=8)
                plt.legend(fontsize=7)
                plt.grid(True, alpha=0.3, axis='y')
                
                # Add values on bars
                for bar, snr_val in zip(bars, snr_per_bucket):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                            f'{snr_val:.2f}', ha='center', va='bottom', fontsize=8)
                
                avg_snr = np.mean(snr_per_bucket)
                assessment = "✅ STRONG" if avg_snr > 1.0 else ("⚠️ MODERATE" if avg_snr > 0.5 else "❌ WEAK")
                plt.text(0.02, 0.98, f'Avg SNR: {avg_snr:.2f}\n{assessment}',
                        transform=plt.gca().transAxes, verticalalignment='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            else:
                # Fallback: aggregated histogram
                plt.hist(action_spreads, bins=50, alpha=0.7, color='orange', edgecolor='black')
                plt.axvline(np.mean(action_spreads), color='r', linestyle='--', linewidth=2)
                plt.title('7. Action Spread (Aggregated)')
                plt.xlabel('Reward Spread')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                snr = np.mean(action_spreads) / rmse if rmse > 0 else 0
                plt.text(0.98, 0.98, f'SNR: {snr:.2f}',
                        transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right',
                        fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        else:
            # Fallback: aggregated histogram
            plt.hist(action_spreads, bins=50, alpha=0.7, color='orange', edgecolor='black')
            plt.axvline(np.mean(action_spreads), color='r', linestyle='--', linewidth=2)
            plt.title('7. Per-Context Action Spread')
            plt.xlabel('Reward Spread')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            mean_spread = np.mean(action_spreads)
            snr = mean_spread / rmse if rmse > 0 else 0
            plt.text(0.98, 0.98, f'Mean: {mean_spread:.3f}\nSNR: {snr:.2f}',
                    transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right',
                    fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    else:
        plt.text(0.5, 0.5, 'No all_predicted_rewards\ndata available', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.axis('off')
    
    # === ROW 3: REWARD FUNCTION DIAGNOSTICS ===
    # Use reward_latency_input_tuples if available for stratified analysis, otherwise fallback to reward_latency_pairs
    if metrics.get('reward_latency_input_tuples') and len(metrics['reward_latency_input_tuples']) > 10:
        # STRATIFIED ANALYSIS BY INPUT LENGTH
        rewards_array = np.array([r for r, l, inp in metrics['reward_latency_input_tuples']])
        latencies_array = np.array([l for r, l, inp in metrics['reward_latency_input_tuples']])
        input_tokens_array = np.array([inp for r, l, inp in metrics['reward_latency_input_tuples']])
        
        # Define input length buckets
        input_quantiles = np.percentile(input_tokens_array, [0, 33, 67, 100])
        bucket_names = [
            f'Short\n({input_quantiles[0]:.0f}-{input_quantiles[1]:.0f} tokens)',
            f'Medium\n({input_quantiles[1]:.0f}-{input_quantiles[2]:.0f} tokens)',
            f'Long\n({input_quantiles[2]:.0f}-{input_quantiles[3]:.0f} tokens)'
        ]
        bucket_colors = ['green', 'orange', 'red']
        
    elif metrics.get('reward_latency_pairs') and len(metrics['reward_latency_pairs']) > 10:
        # FALLBACK: Aggregated analysis (no input length stratification)
        rewards_array = np.array([r for r, l in metrics['reward_latency_pairs']])
        latencies_array = np.array([l for r, l in metrics['reward_latency_pairs']])
        input_tokens_array = None
    else:
        rewards_array = None
        latencies_array = None
        input_tokens_array = None
    
    if rewards_array is not None and latencies_array is not None:
        
        # 9. Latency Distribution Stratified by Input Length
        plt.subplot(3, 4, 9)
        if input_tokens_array is not None:
            # Plot overall distribution first (all requests)
            plt.hist(latencies_array, bins=30, alpha=0.3, 
                    color='blue', histtype='stepfilled',
                    label='All requests', zorder=1)
            
            # Stratified latency histograms on top
            for i, (low, high) in enumerate([(input_quantiles[0], input_quantiles[1]), 
                                              (input_quantiles[1], input_quantiles[2]), 
                                              (input_quantiles[2], input_quantiles[3])]):
                mask = (input_tokens_array >= low) & (input_tokens_array < high) if i < 2 else (input_tokens_array >= low)
                if mask.sum() > 0:
                    plt.hist(latencies_array[mask], bins=30, alpha=0.5, color=bucket_colors[i], 
                            edgecolor='black', label=bucket_names[i].replace('\n', ' '), zorder=2)
            
            plt.xlabel('TTFT (ms)', fontsize=10)
            plt.ylabel('Frequency', fontsize=10)
            plt.title('9. Latency Dist by Input Length', fontsize=11, fontweight='bold')
            plt.legend(fontsize=7, loc='best')
            plt.grid(True, alpha=0.3)
            
            plt.text(0.98, 0.98, f'All: P50={np.median(latencies_array):.0f}, P95={np.percentile(latencies_array, 95):.0f}',
                    transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        else:
            # Fallback: aggregated histogram
            plt.hist(latencies_array, bins=50, alpha=0.7, color='orange', edgecolor='black')
            plt.axvline(np.median(latencies_array), color='r', linestyle='--', linewidth=2)
            plt.axvline(np.percentile(latencies_array, 95), color='purple', linestyle='--', linewidth=2)
            plt.xlabel('TTFT (ms)', fontsize=10)
            plt.ylabel('Frequency', fontsize=10)
            plt.title('9. Latency Distribution', fontsize=11, fontweight='bold')
            plt.grid(True, alpha=0.3)
            
            plt.text(0.98, 0.98, f'Min: {latencies_array.min():.0f}\nMax: {latencies_array.max():.0f}\n'
                    f'P50: {np.median(latencies_array):.0f}\nP95: {np.percentile(latencies_array, 95):.0f}',
                    transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # 10. Reward Distribution Stratified by Input Length
        plt.subplot(3, 4, 10)
        if input_tokens_array is not None:
            # Plot overall distribution first (all requests)
            plt.hist(rewards_array, bins=30, alpha=0.3, 
                    color='blue', histtype='stepfilled',
                    label='All requests', zorder=1)
            
            # Stratified reward histograms on top
            for i, (low, high) in enumerate([(input_quantiles[0], input_quantiles[1]), 
                                              (input_quantiles[1], input_quantiles[2]), 
                                              (input_quantiles[2], input_quantiles[3])]):
                mask = (input_tokens_array >= low) & (input_tokens_array < high) if i < 2 else (input_tokens_array >= low)
                if mask.sum() > 0:
                    plt.hist(rewards_array[mask], bins=30, alpha=0.5, color=bucket_colors[i], 
                            edgecolor='black', label=bucket_names[i].replace('\n', ' '), zorder=2)
            
            plt.xlabel('Reward', fontsize=10)
            plt.ylabel('Frequency', fontsize=10)
            plt.title('10. Reward Dist by Input Length', fontsize=11, fontweight='bold')
            plt.legend(fontsize=7, loc='best')
            plt.grid(True, alpha=0.3)
            
            reward_range = rewards_array.max() - rewards_array.min()
            plt.text(0.98, 0.98, f'All: Range={reward_range:.3f}\nMean={rewards_array.mean():.3f}',
                    transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        else:
            # Fallback: aggregated histogram
            plt.hist(rewards_array, bins=50, alpha=0.7, color='green', edgecolor='black')
            plt.axvline(np.median(rewards_array), color='r', linestyle='--', linewidth=2)
            plt.xlabel('Reward', fontsize=10)
            plt.ylabel('Frequency', fontsize=10)
            plt.title('10. Reward Distribution', fontsize=11, fontweight='bold')
            plt.grid(True, alpha=0.3)
            
            reward_range = rewards_array.max() - rewards_array.min()
            plt.text(0.98, 0.98, f'Range: {reward_range:.3f}\nMean: {rewards_array.mean():.3f}\nStd: {rewards_array.std():.3f}',
                    transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # 8. Reward Discrimination by Input Length and Latency Category
        plt.subplot(3, 4, 8)
        if input_tokens_array is not None:
            # Calculate overall discrimination first (all requests)
            p50_overall = np.percentile(latencies_array, 50)
            p90_overall = np.percentile(latencies_array, 90)
            good_mask_overall = latencies_array < p50_overall
            bad_mask_overall = latencies_array >= p90_overall
            
            if good_mask_overall.sum() > 0 and bad_mask_overall.sum() > 0:
                avg_good_overall = rewards_array[good_mask_overall].mean()
                avg_bad_overall = rewards_array[bad_mask_overall].mean()
                overall_discrimination = avg_good_overall - avg_bad_overall
            else:
                overall_discrimination = 0
            
            # STRATIFIED: Show reward discrimination for each input length bucket
            discrimination_results = []
            
            for i, (low, high) in enumerate([(input_quantiles[0], input_quantiles[1]), 
                                              (input_quantiles[1], input_quantiles[2]), 
                                              (input_quantiles[2], input_quantiles[3])]):
                bucket_mask = (input_tokens_array >= low) & (input_tokens_array < high) if i < 2 else (input_tokens_array >= low)
                
                if bucket_mask.sum() > 0:
                    bucket_lats = latencies_array[bucket_mask]
                    bucket_rews = rewards_array[bucket_mask]
                    
                    # Calculate P50 and P90 within this bucket
                    p50_bucket = np.percentile(bucket_lats, 50)
                    p90_bucket = np.percentile(bucket_lats, 90)
                    
                    good_mask_bucket = bucket_lats < p50_bucket
                    bad_mask_bucket = bucket_lats >= p90_bucket
                    
                    if good_mask_bucket.sum() > 0 and bad_mask_bucket.sum() > 0:
                        avg_good = bucket_rews[good_mask_bucket].mean()
                        avg_bad = bucket_rews[bad_mask_bucket].mean()
                        discrimination_results.append(avg_good - avg_bad)
                    else:
                        discrimination_results.append(0)
            
            # Add overall bar
            discrimination_results.append(overall_discrimination)
            
            # Plot discrimination for each bucket
            colors_22 = bucket_colors[:3] + ['blue']
            bucket_labels_22 = bucket_names + ['All\nRequests']
            x_pos = np.arange(len(discrimination_results))
            bars = plt.bar(x_pos, discrimination_results, color=colors_22[:len(discrimination_results)], 
                          edgecolor='black', alpha=0.7)
            plt.axhline(y=0, color='r', linestyle='--', linewidth=1.5, alpha=0.5)
            plt.xlabel('Input Length Bucket', fontsize=10)
            plt.ylabel('Reward Spread\n(Good - Bad Latency)', fontsize=10)
            plt.title('8. Reward Spread (Good - Bad)', fontsize=11, fontweight='bold')
            plt.xticks(x_pos, [bn.replace('\n', ' ') for bn in bucket_labels_22], 
                      rotation=15, ha='right', fontsize=7)
            plt.grid(True, alpha=0.3, axis='y')
            
            avg_discrimination = np.mean(discrimination_results) if discrimination_results else 0
            
            if abs(avg_discrimination) < 0.05:
                assessment = "❌ POOR\nNo discrimination!"
                box_color = 'lightcoral'
            elif abs(avg_discrimination) < 0.2:
                assessment = "⚠️ WEAK"
                box_color = 'lightyellow'
            else:
                assessment = "✅ GOOD\nContext-aware!"
                box_color = 'lightgreen'
            
            plt.text(0.02, 0.98, f'Avg Spread: {avg_discrimination:.4f}\n{assessment}',
                    transform=plt.gca().transAxes, verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.9))
        else:
            # FALLBACK: Aggregated discrimination
            p50_lat = np.percentile(latencies_array, 50)
            p90_lat = np.percentile(latencies_array, 90)
            p99_lat = np.percentile(latencies_array, 99)
            
            good_mask = latencies_array < p50_lat
            medium_mask = (latencies_array >= p50_lat) & (latencies_array < p90_lat)
            bad_mask = (latencies_array >= p90_lat) & (latencies_array < p99_lat)
            catastrophic_mask = latencies_array >= p99_lat
            
            categories = []
            avg_rewards = []
            reward_stds = []
            
            if good_mask.sum() > 0:
                categories.append(f'Good\n<{p50_lat:.0f}')
                avg_rewards.append(rewards_array[good_mask].mean())
                reward_stds.append(rewards_array[good_mask].std())
            
            if medium_mask.sum() > 0:
                categories.append(f'Med\n{p50_lat:.0f}-{p90_lat:.0f}')
                avg_rewards.append(rewards_array[medium_mask].mean())
                reward_stds.append(rewards_array[medium_mask].std())
            
            if bad_mask.sum() > 0:
                categories.append(f'Bad\n{p90_lat:.0f}-{p99_lat:.0f}')
                avg_rewards.append(rewards_array[bad_mask].mean())
                reward_stds.append(rewards_array[bad_mask].std())
            
            if catastrophic_mask.sum() > 0:
                categories.append(f'Cata\n>{p99_lat:.0f}')
                avg_rewards.append(rewards_array[catastrophic_mask].mean())
                reward_stds.append(rewards_array[catastrophic_mask].std())
            
            x_pos = np.arange(len(categories))
            bars = plt.bar(x_pos, avg_rewards, yerr=reward_stds, capsize=5, 
                          color=['green', 'yellow', 'orange', 'red'][:len(categories)], 
                          edgecolor='black', alpha=0.7)
            plt.xlabel('Latency Category (ms)', fontsize=10)
            plt.ylabel('Avg Reward', fontsize=10)
            plt.title('8. Reward Spread (Good - Bad)', fontsize=11, fontweight='bold')
            plt.xticks(x_pos, categories, fontsize=9)
            plt.grid(True, alpha=0.3, axis='y')
            
            if len(avg_rewards) >= 2:
                discrimination = avg_rewards[0] - avg_rewards[-1]
                discrimination_pct = (discrimination / abs(avg_rewards[-1])) * 100 if avg_rewards[-1] != 0 else 0
                
                if abs(discrimination) < 0.05:
                    assessment = "❌ POOR\nCannot distinguish!"
                    box_color = 'lightcoral'
                elif abs(discrimination) < 0.2:
                    assessment = "⚠️ WEAK"
                    box_color = 'lightyellow'
                else:
                    assessment = "✅ GOOD"
                    box_color = 'lightgreen'
                
                plt.text(0.02, 0.98, f'Spread: {discrimination:.4f}\n({discrimination_pct:.0f}%)\n{assessment}',
                        transform=plt.gca().transAxes, verticalalignment='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.9))
    
    # ===== REWARD FUNCTION VALIDATION =====
    if metrics.get('reward_latency_input_tuples') and len(metrics['reward_latency_input_tuples']) > 10:
        rewards_array = np.array([r for r, l, inp in metrics['reward_latency_input_tuples']])
        latencies_array = np.array([l for r, l, inp in metrics['reward_latency_input_tuples']])
        input_tokens_array = np.array([inp for r, l, inp in metrics['reward_latency_input_tuples']])
        
        # Filter valid samples
        valid_mask = input_tokens_array > 10
        if valid_mask.sum() > 50:  # Need sufficient data for validation
            rewards_valid = rewards_array[valid_mask]
            latencies_valid = latencies_array[valid_mask]
            input_tokens_valid = input_tokens_array[valid_mask]
            
            # Define buckets
            inp_quantiles_val = np.percentile(input_tokens_valid, [0, 33, 67, 100])
            
            # 11. Reward Function Validation: Correlation & Spread
            plt.subplot(3, 4, 11)
            
            correlations = []
            spreads = []
            bucket_names_val = []
            
            for i, (low, high) in enumerate([(inp_quantiles_val[0], inp_quantiles_val[1]), 
                                              (inp_quantiles_val[1], inp_quantiles_val[2]), 
                                              (inp_quantiles_val[2], inp_quantiles_val[3])]):
                mask = (input_tokens_valid >= low) & (input_tokens_valid < high) if i < 2 else (input_tokens_valid >= low)
                
                if mask.sum() > 10:
                    bucket_rewards = rewards_valid[mask]
                    bucket_latencies = latencies_valid[mask]
                    
                    # Correlation (should be strongly negative)
                    corr = np.corrcoef(bucket_rewards, bucket_latencies)[0, 1]
                    correlations.append(corr)
                    
                    # Spread (should be > 0.5 for good discrimination)
                    spread = bucket_rewards.max() - bucket_rewards.min()
                    spreads.append(spread)
                    
                    bucket_names_val.append(f'{int(low)}-{int(high)}')
            
            # Plot correlation and spread side by side
            x = np.arange(len(bucket_names_val))
            width = 0.35
            
            ax1 = plt.gca()
            color = 'tab:blue'
            ax1.set_xlabel('Input Length (tokens)', fontsize=10)
            ax1.set_ylabel('Correlation (Reward vs Latency)', fontsize=10, color=color)
            bars1 = ax1.bar(x - width/2, correlations, width, label='Correlation', color=color, alpha=0.7)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.axhline(y=-0.5, color='blue', linestyle='--', linewidth=1, alpha=0.5, label='Target < -0.5')
            ax1.set_ylim(-1.0, 0.2)
            ax1.set_xticks(x)
            ax1.set_xticklabels(bucket_names_val, fontsize=8)
            
            ax2 = ax1.twinx()
            color = 'tab:orange'
            ax2.set_ylabel('Reward Spread (max-min)', fontsize=10, color=color)
            bars2 = ax2.bar(x + width/2, spreads, width, label='Spread', color=color, alpha=0.7)
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Target > 0.5')
            ax2.set_ylim(0, max(spreads) * 1.2)
            
            plt.title('11. Reward Validation\n(Correlation & Spread)', fontsize=11, fontweight='bold')
            
            # Add values on bars
            for i, (bar, val) in enumerate(zip(bars1, correlations)):
                ax1.text(bar.get_x() + bar.get_width()/2, val - 0.05, f'{val:.2f}',
                        ha='center', va='top', fontsize=7, color='blue')
            
            for i, (bar, val) in enumerate(zip(bars2, spreads)):
                ax2.text(bar.get_x() + bar.get_width()/2, val + 0.05, f'{val:.2f}',
                        ha='center', va='bottom', fontsize=7, color='orange')
            
            # Overall assessment
            avg_corr = np.mean(correlations)
            avg_spread = np.mean(spreads)
            
            if avg_corr < -0.5 and avg_spread > 0.5:
                status = "✅ EXCELLENT"
                status_color = 'lightgreen'
            elif avg_corr < -0.3 and avg_spread > 0.3:
                status = "⚠️ GOOD"
                status_color = 'lightyellow'
            else:
                status = "❌ POOR"
                status_color = 'lightcoral'
            
            ax1.text(0.02, 0.98, f'Avg Corr: {avg_corr:.3f}\nAvg Spread: {avg_spread:.3f}\n{status}',
                    transform=ax1.transAxes, verticalalignment='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.8))
            
            # 12. Reward Distribution Quality Check
            plt.subplot(3, 4, 12)
            
            # Plot overall reward distribution first (all requests)
            plt.hist(rewards_valid, bins=30, alpha=0.3, 
                    color='blue', histtype='stepfilled',
                    label='All requests', zorder=1)
            
            # Plot reward distribution per bucket with step style on top
            for i, (low, high) in enumerate([(inp_quantiles_val[0], inp_quantiles_val[1]), 
                                              (inp_quantiles_val[1], inp_quantiles_val[2]), 
                                              (inp_quantiles_val[2], inp_quantiles_val[3])]):
                mask = (input_tokens_valid >= low) & (input_tokens_valid < high) if i < 2 else (input_tokens_valid >= low)
                
                if mask.sum() > 10:
                    bucket_rewards = rewards_valid[mask]
                    
                    # Plot histogram with step style
                    plt.hist(bucket_rewards, bins=30, alpha=1.0, 
                            color=['green', 'orange', 'red'][i],
                            histtype='step', linewidth=2,
                            label=f'{int(low)}-{int(high)} tok', zorder=2)
            
            plt.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Zero', zorder=3)
            plt.xlabel('Reward', fontsize=10)
            plt.ylabel('Frequency', fontsize=10)
            plt.title('12. Reward Distribution Quality', fontsize=11, fontweight='bold')
            plt.legend(fontsize=7, loc='upper left')
            plt.grid(True, alpha=0.3)
            
            # Calculate percentiles
            p10 = np.percentile(rewards_valid, 10)
            p50 = np.percentile(rewards_valid, 50)
            p90 = np.percentile(rewards_valid, 90)
            reward_range = rewards_valid.max() - rewards_valid.min()
            
            # Assessment
            centered = abs(p50) < 0.5  # Median near zero
            good_range = reward_range > 2.0  # Sufficient spread
            no_extremes = (p10 > -4) and (p90 < 4)  # Not clipped
            
            if centered and good_range and no_extremes:
                assessment = "✅ HEALTHY\nWell-balanced"
                color_box = 'lightgreen'
            elif good_range:
                assessment = "⚠️ ACCEPTABLE\nSome imbalance"
                color_box = 'lightyellow'
            else:
                assessment = "❌ PROBLEMATIC\nPoor distribution"
                color_box = 'lightcoral'
            
            info_text = f'Range: {reward_range:.2f}\n'
            info_text += f'P50: {p50:.2f}\n'
            info_text += f'P10/P90: {p10:.2f}/{p90:.2f}\n'
            info_text += f'\n{assessment}'
            
            plt.text(0.98, 0.98, info_text,
                    transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right',
                    fontsize=8, bbox=dict(boxstyle='round', facecolor=color_box, alpha=0.9))
    
    plt.tight_layout(h_pad=2.5, w_pad=2.0)
    
    # Save plot with num_trains in filename
    plot_filename = f'comprehensive_neural_cb_metrics-{num_trains}.pdf'
    plot_path = os.path.join(final_model_dir, plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved comprehensive training plot: {plot_path}")
    
    # Also save CSV files for future analysis
    if metrics['losses']:
        # Prepare data with proper length matching
        num_steps = len(metrics['losses'])
        metrics_data = {
            'update_step': list(range(num_steps)),
            'loss': metrics['losses'],
            'reward': metrics['rewards'] if len(metrics['rewards']) == num_steps else [None] * num_steps,
            'epsilon': metrics['epsilons'] if len(metrics['epsilons']) == num_steps else [None] * num_steps,
            'learning_rate': metrics['learning_rates'] if len(metrics.get('learning_rates', [])) == num_steps else [None] * num_steps
        }
        metrics_df = pd.DataFrame(metrics_data)
        csv_path = os.path.join(final_model_dir, f'training_metrics-{num_trains}.csv')
        metrics_df.to_csv(csv_path, index=False)
        logger.info(f"Saved training metrics CSV: {csv_path}")
    
    return plot_path



# if __name__ == "__main__":
#     # Test the neural contextual bandit
#     logger.info("Testing Neural Contextual Bandit...")
    
#     state_dim = {'pod_features': 8, 'kv_hit_ratios': 1, 'request_features': 3}
#     action_dim = 7
#     hyperparameters = {
#         'hidden_dim': 128,
#         'learning_rate': 3e-4,
#         'buffer_size': 1000,
#         'exploration_method': 'epsilon_greedy',
#         'initial_epsilon': 0.3,
#         'batch_size': 32,
#         'update_frequency': 10
#     }
    
#     agent = NeuralContextualBandit(
#         state_dim=state_dim,
#         action_dim=action_dim,
#         hyperparameters=hyperparameters,
#         final_model_dir='/tmp/test_neural_cb'
#     )
#     logger.info("Neural Contextual Bandit initialized successfully!")
    