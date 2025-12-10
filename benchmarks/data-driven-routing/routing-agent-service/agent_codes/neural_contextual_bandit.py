#!/usr/bin/env python3
"""
Neural Contextual Bandit for LLM Routing
Correct formulation for latency-optimal pod selection
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import pickle
import time
from logger import logger
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RewardNetwork(nn.Module):
    """
    Network that predicts expected reward (inverse latency) for each pod
    """
    def __init__(self, context_dim, num_actions, hidden_dim=128):
        super().__init__()
        
        self.context_dim = context_dim
        self.num_actions = num_actions
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Separate head for each action (pod)
        self.action_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_actions)
        ])
        
        logger.info(f"RewardNetwork: context_dim={context_dim}, num_actions={num_actions}, hidden_dim={hidden_dim}")
    
    def forward(self, context):
        """
        Args:
            context: [batch_size, context_dim]
        Returns:
            expected_rewards: [batch_size, num_actions]
        """
        features = self.feature_extractor(context)
        
        # Get expected reward for each action
        rewards = []
        for head in self.action_heads:
            rewards.append(head(features))
        
        return torch.cat(rewards, dim=1)  # [batch, num_actions]


class NeuralContextualBandit:
    """
    Neural Contextual Bandit with proper online learning
    """
    def __init__(self, state_dim, action_dim, hyperparameters, final_model_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hyperparameters = hyperparameters
        self.final_model_dir = final_model_dir
        
        # Calculate total context dimension
        self.context_dim = (
            state_dim['pod_features'] * action_dim +  # All pod features
            state_dim['kv_hit_ratios'] * action_dim +  # All KV ratios
            state_dim['request_features']  # Request features
        )
        
        logger.info(f"Context dimension: {self.context_dim}")
        
        # Create reward prediction network
        self.reward_net = RewardNetwork(
            self.context_dim,
            action_dim,
            hidden_dim=hyperparameters.get('hidden_dim', 128)
        ).to(device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.reward_net.parameters(),
            lr=hyperparameters.get('learning_rate', 3e-4),
            weight_decay=hyperparameters.get('weight_decay', 1e-5)
        )
        
        # Experience replay buffer (keep last N experiences)
        self.buffer_size = hyperparameters.get('buffer_size', 10000)
        self.replay_buffer = deque(maxlen=self.buffer_size)
        
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
            'rewards': [],
            'epsilons': [],
            # Reward function quality tracking
            'reward_latency_pairs': [],  # [(reward, latency), ...] for function analysis
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
            'counterfactual_gains': []    # Estimated gain from model's choice
        }
        
        logger.info(f"NeuralContextualBandit initialized: exploration={self.exploration_method}")
    
    def _flatten_context(self, pod_features, kv_hit_ratios, request_features):
        """
        Flatten all inputs into a single context vector
        
        Args:
            pod_features: [batch, num_pods, pod_feat_dim]
            kv_hit_ratios: [batch, num_pods, kv_dim]
            request_features: [batch, req_feat_dim]
        
        Returns:
            context: [batch, context_dim]
        """
        batch_size = pod_features.shape[0]
        
        # Flatten pod-level features (use reshape to handle non-contiguous tensors)
        pod_flat = pod_features.reshape(batch_size, -1)  # [batch, num_pods * pod_feat_dim]
        kv_flat = kv_hit_ratios.reshape(batch_size, -1)  # [batch, num_pods * kv_dim]
        
        # Concatenate all context
        context = torch.cat([pod_flat, kv_flat, request_features], dim=1)
        
        return context
    
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
        """
        with torch.no_grad():
            context = self._flatten_context(pod_features, kv_hit_ratios, request_features)
            predicted_rewards = self.reward_net(context)  # [1, num_actions]
            
            if evaluate or self.exploration_method == 'greedy':
                # Pure exploitation
                action = torch.argmax(predicted_rewards, dim=1).item()
            
            elif self.exploration_method == 'epsilon_greedy':
                # Epsilon-greedy exploration
                if np.random.random() < self.epsilon:
                    action = np.random.randint(0, self.action_dim)
                else:
                    action = torch.argmax(predicted_rewards, dim=1).item()
            
            elif self.exploration_method == 'ucb':
                # Upper Confidence Bound
                exploitation = predicted_rewards[0].cpu().numpy()
                
                # Add exploration bonus: sqrt(2 * log(t) / n_a)
                exploration_bonus = np.sqrt(
                    self.ucb_confidence * np.log(self.total_steps + 1) / (self.action_counts + 1)
                )
                
                ucb_values = exploitation + exploration_bonus
                action = int(np.argmax(ucb_values))
            
            elif self.exploration_method == 'thompson_sampling':
                # Thompson Sampling: Add Gaussian noise to predictions
                noise_std = self.epsilon  # Use epsilon as noise level
                noisy_rewards = predicted_rewards + torch.randn_like(predicted_rewards) * noise_std
                action = torch.argmax(noisy_rewards, dim=1).item()
            
            else:
                raise ValueError(f"Unknown exploration method: {self.exploration_method}")
            
            # Update counters
            self.action_counts[action] += 1
            self.total_steps += 1
            
            return action, predicted_rewards[0].cpu().numpy()
    
    def remember(self, pod_features, kv_hit_ratios, request_features, action, reward):
        """
        Store experience in replay buffer
        
        Args:
            pod_features: [1, num_pods, pod_feat_dim]
            kv_hit_ratios: [1, num_pods, kv_dim]
            request_features: [1, req_feat_dim]
            action: Scalar action
            reward: Scalar reward
        """
        context = self._flatten_context(pod_features, kv_hit_ratios, request_features)
        
        experience = {
            'context': context.cpu(),
            'action': action if isinstance(action, int) else action.item(),
            'reward': reward.item() if torch.is_tensor(reward) else reward
        }
        
        self.replay_buffer.append(experience)
        self.steps_since_update += 1
        
        # Note: Automatic learning disabled for batch training
        # The train_batch() function handles learning explicitly
        # Uncomment below for online learning scenarios:
        # if self.steps_since_update >= self.update_frequency:
        #     self.learn()
        #     self.steps_since_update = 0
    
    def learn(self):
        """
        Update the reward network using experiences from replay buffer
        """
        if len(self.replay_buffer) < self.batch_size:
            logger.debug(f"Not enough experiences to learn: {len(self.replay_buffer)} < {self.batch_size}")
            return {'loss': 0.0, 'reward': 0.0}
        
        # Sample batch from replay buffer
        batch_size = min(self.batch_size, len(self.replay_buffer))
        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]
        
        # Prepare batch tensors
        contexts = torch.cat([exp['context'] for exp in batch], dim=0).to(device)
        actions = torch.tensor([exp['action'] for exp in batch], dtype=torch.long).to(device)
        rewards = torch.tensor([exp['reward'] for exp in batch], dtype=torch.float32).to(device)
        
        # Forward pass
        predicted_rewards = self.reward_net(contexts)  # [batch, num_actions]
        
        # Get predicted rewards for the actions that were taken
        predicted_action_rewards = predicted_rewards.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute loss (MSE between predicted and actual rewards)
        loss = F.mse_loss(predicted_action_rewards, rewards)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.reward_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update epsilon (decay exploration)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Track metrics
        self.training_metrics['losses'].append(loss.item())
        self.training_metrics['rewards'].append(rewards.mean().item())
        self.training_metrics['epsilons'].append(self.epsilon)
        
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
        
        logger.info(f"[Update] Loss: {loss.item():.4f}, Avg Reward: {rewards.mean().item():.4f}, "
                   f"Epsilon: {self.epsilon:.4f}, Buffer: {len(self.replay_buffer)}")
        
        return {
            'loss': loss.item(),
            'reward': rewards.mean().item(),
            'epsilon': self.epsilon
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
            'context_dim': self.context_dim,
            'hyperparameters': self.hyperparameters,
            'epsilon': self.epsilon,
            'action_counts': self.action_counts.tolist(),
            'total_steps': self.total_steps,
            'training_metrics': self.training_metrics
        }
        
        with open(os.path.join(final_model_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Model saved to {final_model_dir}")
    
    def load(self, directory):
        """Load model and metadata"""
        # Load network weights
        reward_net_path = os.path.join(directory, 'reward_net.pth')
        if os.path.exists(reward_net_path):
            self.reward_net.load_state_dict(torch.load(reward_net_path, map_location=device))
            logger.info(f"Loaded reward network from {reward_net_path}")
        
        # Load optimizer state
        optimizer_path = os.path.join(directory, 'optimizer.pth')
        if os.path.exists(optimizer_path):
            try:
                self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
            except:
                logger.warning("Could not load optimizer state")
        
        # Load metadata
        metadata_path = os.path.join(directory, 'metadata.pkl')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            self.epsilon = metadata.get('epsilon', self.epsilon)
            self.action_counts = np.array(metadata.get('action_counts', self.action_counts))
            self.total_steps = metadata.get('total_steps', 0)
            self.training_metrics = metadata.get('training_metrics', self.training_metrics)
            
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
            _cached_agent.load(final_model_dir)
        
        _cached_metadata = current_config
    else:
        # Agent reused - this is expected for most requests
        pass
    
    overhead_summary['get_agent'] = time.time() - get_agent_start
    
    # Inference
    inference_start = time.time()
    action, predicted_rewards = _cached_agent.choose_action(
        pod_features, kv_hit_ratios, request_features, 
        evaluate=not HYPERPARAMETERS.get('explore', True)
    )
    overhead_summary['inference'] = time.time() - inference_start
    
    logger.info(f"Neural CB request {request_id}: action={action}, total_steps={_cached_agent.total_steps}, buffer_size={len(_cached_agent.replay_buffer)}, epsilon={_cached_agent.epsilon:.3f}")
    
    # Format predicted_rewards as dict (same format as predicted_latencies)
    predicted_rewards_formatted = {sorted_all_pod_ids[i]: float(predicted_rewards[i]) for i in range(len(sorted_all_pod_ids))}
    chosen_pod_predicted_reward = float(predicted_rewards[action])
    
    # Prepare result
    result = {
        'selected_pod_index': int(action),
        'predicted_rewards': predicted_rewards_formatted,
        'chosen_pod_predicted_reward': chosen_pod_predicted_reward,
        'confidence': chosen_pod_predicted_reward,  # Keep for backward compatibility
        'epsilon': _cached_agent.epsilon,
        'total_steps': _cached_agent.total_steps
    }
    
    overhead_summary['total_inference'] = time.time() - infer_start_time
    
    return result, overhead_summary


def plot_neural_cb_metrics(agent, final_model_dir, num_epochs, total_samples):
    """
    Create comprehensive training metrics visualization for Neural Contextual Bandit.
    
    Args:
        agent: Trained NeuralContextualBandit instance
        final_model_dir: Directory to save plots
        num_epochs: Number of training epochs
        total_samples: Total number of samples processed
    
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
    
    # Create comprehensive plot - expanded grid for CB-specific plots
    fig = plt.figure(figsize=(30, 18))
    fig.suptitle(f'Neural Contextual Bandit Training Results\n'
                 f'Epochs: {num_epochs} | Total Samples: {total_samples:,} | Updates: {len(metrics["losses"]):,}',
                 fontsize=18, fontweight='bold', y=0.995)
    
    # 1. Training Loss
    plt.subplot(4, 6, 1)
    if metrics['losses']:
        plt.plot(metrics['losses'], 'b-', linewidth=1.5, alpha=0.7)
        # Add moving average
        if len(metrics['losses']) > 10:
            window = min(50, len(metrics['losses']) // 10)
            moving_avg = pd.Series(metrics['losses']).rolling(window=window).mean()
            plt.plot(moving_avg, 'r-', linewidth=2, label=f'{window}-step MA')
            plt.legend()
        plt.title('1. Training Loss')
        plt.xlabel('Update Step')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        
        # Add final loss annotation
        final_loss = metrics['losses'][-1]
        avg_loss = np.mean(metrics['losses'][-100:]) if len(metrics['losses']) >= 100 else np.mean(metrics['losses'])
        plt.text(0.02, 0.98, f'Final: {final_loss:.4f}\nAvg (last 100): {avg_loss:.4f}',
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 2. Average Reward
    plt.subplot(4, 6, 2)
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
    
    # 3. Exploration Rate (Epsilon)
    plt.subplot(4, 6, 3)
    if metrics['epsilons']:
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
    
    # 4. Replay Buffer Size
    plt.subplot(4, 6, 4)
    buffer_sizes = [min(agent.buffer_size, i+1) for i in range(total_samples)]
    plt.plot(buffer_sizes, 'purple', linewidth=2)
    plt.axhline(y=agent.buffer_size, color='r', linestyle='--', label=f'Max Size: {agent.buffer_size}')
    plt.title('4. Replay Buffer Size')
    plt.xlabel('Sample')
    plt.ylabel('Buffer Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Loss Distribution
    plt.subplot(4, 6, 5)
    if metrics['losses']:
        plt.hist(metrics['losses'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        plt.axvline(np.mean(metrics['losses']), color='r', linestyle='--', linewidth=2, label='Mean')
        plt.axvline(np.median(metrics['losses']), color='g', linestyle='--', linewidth=2, label='Median')
        plt.title('5. Loss Distribution')
        plt.xlabel('Loss')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_loss = np.mean(metrics['losses'])
        std_loss = np.std(metrics['losses'])
        plt.text(0.98, 0.98, f'Mean: {mean_loss:.4f}\nStd: {std_loss:.4f}',
                transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 6. Reward Distribution
    plt.subplot(4, 6, 6)
    if metrics['rewards']:
        plt.hist(metrics['rewards'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.axvline(np.mean(metrics['rewards']), color='r', linestyle='--', linewidth=2, label='Mean')
        plt.axvline(np.median(metrics['rewards']), color='g', linestyle='--', linewidth=2, label='Median')
        plt.title('6. Reward Distribution')
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_reward = np.mean(metrics['rewards'])
        std_reward = np.std(metrics['rewards'])
        plt.text(0.98, 0.98, f'Mean: {mean_reward:.4f}\nStd: {std_reward:.4f}',
                transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # 7. Learning Progress (Loss & Reward together, normalized)
    plt.subplot(4, 6, 7)
    if metrics['losses'] and metrics['rewards']:
        # Normalize to 0-1 range for comparison
        losses_norm = np.array(metrics['losses'])
        losses_norm = (losses_norm - losses_norm.min()) / (losses_norm.max() - losses_norm.min() + 1e-8)
        
        rewards_norm = np.array(metrics['rewards'])
        # Invert rewards (lower is better after normalization for visualization)
        rewards_norm = (rewards_norm - rewards_norm.min()) / (rewards_norm.max() - rewards_norm.min() + 1e-8)
        
        plt.plot(losses_norm, 'b-', alpha=0.6, linewidth=1.5, label='Loss (norm)')
        plt.plot(1 - rewards_norm, 'g-', alpha=0.6, linewidth=1.5, label='Inv. Reward (norm)')
        plt.title('7. Learning Progress (Normalized)')
        plt.xlabel('Update Step')
        plt.ylabel('Normalized Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 8. Loss Improvement Over Time
    plt.subplot(4, 6, 8)
    if metrics['losses'] and len(metrics['losses']) > 10:
        # Calculate improvement: difference from initial loss
        initial_loss = np.mean(metrics['losses'][:10])
        improvement = [(initial_loss - loss) / initial_loss * 100 for loss in metrics['losses']]
        plt.plot(improvement, 'darkblue', linewidth=2)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.title('8. Loss Improvement from Initial')
        plt.xlabel('Update Step')
        plt.ylabel('Improvement (%)')
        plt.grid(True, alpha=0.3)
        
        final_improvement = improvement[-1]
        plt.text(0.02, 0.98, f'Final: {final_improvement:.1f}%',
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    # 9. Reward Improvement Over Time
    plt.subplot(4, 6, 9)
    if metrics['rewards'] and len(metrics['rewards']) > 10:
        # Calculate improvement: difference from initial reward
        initial_reward = np.mean(metrics['rewards'][:10])
        improvement = [(reward - initial_reward) / abs(initial_reward) * 100 if initial_reward != 0 else 0 
                      for reward in metrics['rewards']]
        plt.plot(improvement, 'darkgreen', linewidth=2)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.title('9. Reward Improvement from Initial')
        plt.xlabel('Update Step')
        plt.ylabel('Improvement (%)')
        plt.grid(True, alpha=0.3)
        
        final_improvement = improvement[-1]
        plt.text(0.02, 0.98, f'Final: {final_improvement:.1f}%',
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # 10. Model Architecture Info
    plt.subplot(4, 6, 10)
    plt.axis('off')
    plt.title('10. Model Architecture Info', pad=10)
    
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
    
    # 11. Training Statistics
    plt.subplot(4, 6, 11)
    plt.axis('off')
    plt.title('11. Training Statistics', pad=10)
    
    stats_text = "TRAINING SUMMARY\n" + "="*20 + "\n"
    stats_text += f"Total Epochs: {num_epochs}\n"
    stats_text += f"Total Samples: {total_samples:,}\n"
    stats_text += f"Total Updates: {len(metrics['losses']):,}\n"
    stats_text += f"Updates/Epoch: {len(metrics['losses'])//num_epochs if num_epochs > 0 else 0}\n\n"
    
    if metrics['losses']:
        stats_text += f"LOSS:\n"
        stats_text += f"  Initial: {metrics['losses'][0]:.4f}\n"
        stats_text += f"  Final: {metrics['losses'][-1]:.4f}\n"
        stats_text += f"  Mean: {np.mean(metrics['losses']):.4f}\n"
        stats_text += f"  Min: {np.min(metrics['losses']):.4f}\n"
        stats_text += f"  Max: {np.max(metrics['losses']):.4f}\n\n"
    
    if metrics['rewards']:
        stats_text += f"REWARD:\n"
        stats_text += f"  Initial: {metrics['rewards'][0]:.4f}\n"
        stats_text += f"  Final: {metrics['rewards'][-1]:.4f}\n"
        stats_text += f"  Mean: {np.mean(metrics['rewards']):.4f}\n"
        stats_text += f"  Min: {np.min(metrics['rewards']):.4f}\n"
        stats_text += f"  Max: {np.max(metrics['rewards']):.4f}\n"
    
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    # 12. Performance Assessment
    plt.subplot(4, 6, 12)
    plt.axis('off')
    plt.title('12. Performance Assessment', pad=10)
    
    assessment_text = "PERFORMANCE ASSESSMENT\n" + "="*22 + "\n"
    
    if metrics['losses'] and len(metrics['losses']) > 10:
        initial_loss = np.mean(metrics['losses'][:10])
        final_loss = np.mean(metrics['losses'][-10:])
        loss_reduction = (initial_loss - final_loss) / initial_loss * 100
        
        assessment_text += f"Loss Reduction: {loss_reduction:.1f}%\n"
        
        if loss_reduction > 50:
            assessment_text += "✅ EXCELLENT Learning\n"
        elif loss_reduction > 30:
            assessment_text += "✅ GOOD Learning\n"
        elif loss_reduction > 10:
            assessment_text += "✅ MODERATE Learning\n"
        elif loss_reduction > 0:
            assessment_text += "⚠️  MODEST Learning\n"
    else:
            assessment_text += "❌ LIMITED Learning\n"
            assessment_text += "\n"
    
    if metrics['rewards'] and len(metrics['rewards']) > 10:
        initial_reward = np.mean(metrics['rewards'][:10])
        final_reward = np.mean(metrics['rewards'][-10:])
        reward_improvement = (final_reward - initial_reward) / abs(initial_reward) * 100 if initial_reward != 0 else 0
        
        assessment_text += f"Reward Improvement:\n  {reward_improvement:.1f}%\n\n"
    
    # Stability assessment
    if metrics['losses'] and len(metrics['losses']) > 100:
        recent_std = np.std(metrics['losses'][-100:])
        early_std = np.std(metrics['losses'][:100])
        
        assessment_text += f"Loss Stability:\n"
        assessment_text += f"  Early Std: {early_std:.4f}\n"
        assessment_text += f"  Recent Std: {recent_std:.4f}\n"
        
        if recent_std < early_std * 0.8:
            assessment_text += "✅ Converging\n"
        elif recent_std < early_std * 1.2:
            assessment_text += "➡️  Stable\n"
        else:
            assessment_text += "⚠️  Unstable\n"
    
    plt.text(0.1, 0.9, assessment_text, transform=plt.gca().transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # ==================================================================
    # CONTEXTUAL BANDIT-SPECIFIC PLOTS
    # ==================================================================
    
    # 13. OFF-POLICY: Counterfactual Gain Distribution
    plt.subplot(4, 6, 13)
    if metrics.get('counterfactual_gains'):
        gains = np.array(metrics['counterfactual_gains'])
        
        plt.hist(gains, bins=50, alpha=0.7, color='purple', edgecolor='black')
        plt.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero Gain')
        plt.axvline(np.mean(gains), color='g', linestyle='-', linewidth=2, label=f'Mean: {np.mean(gains):.4f}')
        
        plt.xlabel('Counterfactual Gain')
        plt.ylabel('Frequency')
        plt.title('13. Expected Gain from Model Policy')
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
        
        # Calculate statistics
        pct_better = (gains > 0).sum() / len(gains) * 100
        mean_gain = np.mean(gains)
        median_gain = np.median(gains)
        
        stats_text = f'Better: {pct_better:.1f}%\n'
        stats_text += f'Mean: {mean_gain:.4f}\n'
        stats_text += f'Median: {median_gain:.4f}'
        
        # Assessment
        if mean_gain > 0.05:
            stats_text += '\n✅ BETTER than data'
        elif mean_gain > 0.01:
            stats_text += '\n➡️  Slightly better'
        elif mean_gain > -0.01:
            stats_text += '\n⚠️  Similar to data'
        else:
            stats_text += '\n❌ WORSE than data'
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    else:
        plt.text(0.5, 0.5, 'No counterfactual\ndata', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.axis('off')
    
    # 14. OFF-POLICY: Action Agreement Analysis
    plt.subplot(4, 6, 14)
    if metrics.get('greedy_actions') and metrics.get('training_actions'):
        greedy = np.array(metrics['greedy_actions'])
        training = np.array(metrics['training_actions'])
        gains = np.array(metrics['counterfactual_gains'])
        
        # Calculate agreement
        agreement = greedy == training
        disagreement = ~agreement
        
        agree_pct = agreement.sum() / len(agreement) * 100
        disagree_pct = 100 - agree_pct
        
        # For disagreements, check if gains are positive
        if disagreement.sum() > 0:
            disagree_gains = gains[disagreement]
            better_disagree_pct = (disagree_gains > 0).sum() / len(disagree_gains) * 100 if len(disagree_gains) > 0 else 0
        else:
            better_disagree_pct = 0
        
        # Create bar chart
        categories = ['Agree', 'Disagree\n(Better)', 'Disagree\n(Worse)']
        if disagreement.sum() > 0:
            better_count = (disagree_gains > 0).sum()
            worse_count = (disagree_gains <= 0).sum()
        else:
            better_count = 0
            worse_count = 0
            
        counts = [agreement.sum(), better_count, worse_count]
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        
        bars = plt.bar(categories, counts, color=colors, alpha=0.8, edgecolor='black')
        plt.title('14. Model vs Training Policy')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        
        # Add percentages
        total = len(agreement)
        for bar, count in zip(bars, counts):
            if count > 0:
                pct = count / total * 100
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{int(count)}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
        
        # Add summary
        summary = f'Disagree: {disagree_pct:.1f}%\n'
        if disagree_pct > 0:
            summary += f'Of those, {better_disagree_pct:.1f}%\nare better'
        plt.text(0.98, 0.98, summary, transform=plt.gca().transAxes,
                verticalalignment='top', horizontalalignment='right', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    else:
        plt.text(0.5, 0.5, 'No policy\ncomparison data', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.axis('off')
    
    # 15. CRITICAL: Per-Context Action Differentiation
    plt.subplot(4, 6, 15)
    if metrics.get('all_predicted_rewards') and len(metrics['all_predicted_rewards']) > 0:
        # Stack all predictions [num_samples, num_actions]
        all_preds = np.concatenate(metrics['all_predicted_rewards'], axis=0)
        
        # CRITICAL METRIC: For each sample, what's the spread between best and worst pod?
        action_spreads = all_preds.max(axis=1) - all_preds.min(axis=1)
        
        plt.hist(action_spreads, bins=50, alpha=0.7, color='orange', edgecolor='black')
        plt.axvline(np.mean(action_spreads), color='r', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(action_spreads):.3f}')
        plt.axvline(np.median(action_spreads), color='g', linestyle='--', linewidth=2, 
                   label=f'Median: {np.median(action_spreads):.3f}')
        
        plt.title('15. Per-Context Action Spread (CRITICAL!)')
        plt.xlabel('Reward Spread: max(pred) - min(pred) for same context')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Assessment: Compare spread to prediction uncertainty
        mean_spread = np.mean(action_spreads)
        median_spread = np.median(action_spreads)
        
        # Get prediction uncertainty (RMSE)
        avg_loss = np.mean(metrics['losses'][-100:]) if len(metrics['losses']) >= 100 else np.mean(metrics['losses'])
        rmse = np.sqrt(avg_loss)
        
        # Signal-to-noise ratio
        snr = mean_spread / rmse if rmse > 0 else 0
        
        # Assessment
        if mean_spread > 0.5 and snr > 1.0:
            assessment = "✅ STRONG\nContext-dependent\nlearning!"
            color = 'lightgreen'
        elif mean_spread > 0.3 and snr > 0.5:
            assessment = "⚠️  MODERATE\nSome differentiation"
            color = 'lightyellow'
        else:
            assessment = "❌ WEAK\nTreats pods similarly\nacross contexts"
            color = 'lightcoral'
        
        info_text = f'Mean Spread: {mean_spread:.3f}\n'
        info_text += f'Prediction RMSE: {rmse:.3f}\n'
        info_text += f'SNR: {snr:.2f}\n\n'
        info_text += f'{assessment}'
        
        plt.text(0.98, 0.98, info_text,
                transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.9), fontsize=9)
    else:
        plt.text(0.5, 0.5, 'No all_predicted_rewards\ndata available', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.axis('off')
    
    # 16. COMPREHENSIVE OFF-POLICY EVALUATION
    plt.subplot(4, 6, 16)
    plt.axis('off')
    plt.title('16. Off-Policy Evaluation Summary', pad=10)
    
    eval_text = "OFF-POLICY EVALUATION\n" + "="*25 + "\n"
    
    # Initialize scores for final assessment
    snr_score = 0
    gain_score = 0
    disagree_score = 0
    coverage_score = 0
    
    # 1. SNR Analysis
    if metrics.get('all_predicted_rewards') and len(metrics['all_predicted_rewards']) > 0:
        all_preds = np.concatenate(metrics['all_predicted_rewards'], axis=0)
        action_spreads = all_preds.max(axis=1) - all_preds.min(axis=1)
        mean_spread = np.mean(action_spreads)
        
        avg_loss = np.mean(metrics['losses'][-100:]) if len(metrics['losses']) >= 100 else np.mean(metrics['losses'])
        rmse = np.sqrt(avg_loss)
        snr = mean_spread / rmse if rmse > 0 else 0
        
        eval_text += f"1. SNR (Signal-to-Noise):\n"
        eval_text += f"   Spread: {mean_spread:.3f}\n"
        eval_text += f"   RMSE: {rmse:.3f}\n"
        eval_text += f"   SNR: {snr:.2f}\n"
        
        if snr > 2.0:
            eval_text += "   ✅ STRONG differentiation\n"
            snr_score = 2
        elif snr > 1.0:
            eval_text += "   ⚠️  MODERATE differentiation\n"
            snr_score = 1
        else:
            eval_text += "   ❌ WEAK differentiation\n"
            snr_score = 0
        eval_text += "\n"
    
    # 2. Counterfactual Gain
    if metrics.get('counterfactual_gains'):
        gains = np.array(metrics['counterfactual_gains'])
        mean_gain = np.mean(gains)
        median_gain = np.median(gains)
        pct_better = (gains > 0).sum() / len(gains) * 100
        
        eval_text += f"2. Counterfactual Gain:\n"
        eval_text += f"   Mean: {mean_gain:.4f}\n"
        eval_text += f"   Median: {median_gain:.4f}\n"
        eval_text += f"   Better: {pct_better:.1f}%\n"
        
        if mean_gain > 0.05:
            eval_text += "   ✅ SIGNIFICANT improvement\n"
            gain_score = 2
        elif mean_gain > 0.01:
            eval_text += "   ➡️  MODEST improvement\n"
            gain_score = 1
        elif mean_gain > -0.01:
            eval_text += "   ⚠️  MARGINAL (data was good)\n"
            gain_score = 1  # Not bad, just saturated
        else:
            eval_text += "   ❌ NEGATIVE (review model)\n"
            gain_score = 0
        eval_text += "\n"
    
    # 3. Policy Disagreement
    if metrics.get('policy_agreements'):
        agreements = np.array(metrics['policy_agreements'])
        agree_pct = np.mean(agreements) * 100
        disagree_pct = 100 - agree_pct
        
        # Among disagreements, how many are better?
        if metrics.get('counterfactual_gains'):
            disagree_mask = ~agreements.astype(bool)
            if disagree_mask.sum() > 0:
                gains_when_disagree = np.array(metrics['counterfactual_gains'])[disagree_mask]
                better_disagree_pct = (gains_when_disagree > 0).sum() / len(gains_when_disagree) * 100
            else:
                better_disagree_pct = 0
        else:
            better_disagree_pct = 0
        
        eval_text += f"3. Policy Disagreement:\n"
        eval_text += f"   Disagree: {disagree_pct:.1f}%\n"
        eval_text += f"   Better when disagree: {better_disagree_pct:.1f}%\n"
        
        # High disagreement with small gain → model confident but saturated
        # High disagreement with large gain → model found improvements
        # Low disagreement → model imitating
        if disagree_pct > 50 and better_disagree_pct > 60:
            eval_text += "   ✅ CONFIDENT policy\n"
            disagree_score = 2
        elif disagree_pct > 30:
            eval_text += "   ➡️  MODERATE divergence\n"
            disagree_score = 1
        else:
            eval_text += "   ⚠️  LOW (may be imitating)\n"
            disagree_score = 0
        eval_text += "\n"
    
    # 4. Coverage (Action Diversity)
    if metrics.get('selected_actions'):
        action_counts = np.bincount(metrics['selected_actions'], minlength=agent.action_dim)
        total = action_counts.sum()
        non_zero = np.count_nonzero(action_counts)
        
        # Entropy as coverage proxy
        probs = action_counts / total if total > 0 else action_counts
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs)) if len(probs) > 0 else 0
        max_entropy = np.log(agent.action_dim)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        eval_text += f"4. Coverage (Training):\n"
        eval_text += f"   Actions used: {non_zero}/{agent.action_dim}\n"
        eval_text += f"   Entropy: {normalized_entropy:.3f}\n"
        
        if normalized_entropy > 0.7:
            eval_text += "   ✅ GOOD coverage\n"
            coverage_score = 2
        elif normalized_entropy > 0.4:
            eval_text += "   ⚠️  MODERATE coverage\n"
            coverage_score = 1
        else:
            eval_text += "   ❌ POOR coverage\n"
            coverage_score = 0
        eval_text += "\n"
    
    # Final Combined Assessment
    total_score = snr_score + gain_score + disagree_score + coverage_score
    max_score = 8
    
    eval_text += "="*25 + "\n"
    eval_text += "DEPLOYMENT READINESS:\n"
    eval_text += f"Score: {total_score}/{max_score}\n"
    
    # Interpretation of score combination
    if total_score >= 7:
        eval_text += "✅ READY for deployment\n"
        eval_text += "Policy learned well"
        box_color = 'lightgreen'
    elif total_score >= 5:
        if gain_score >= 1 and snr_score >= 1:
            eval_text += "⚠️  READY but modest gains\n"
            eval_text += "Training data was good"
            box_color = 'lightyellow'
        else:
            eval_text += "⚠️  CAUTIOUS deployment\n"
            eval_text += "Monitor closely"
            box_color = 'lightyellow'
    else:
        eval_text += "❌ REVIEW NEEDED\n"
        eval_text += "Check data/features"
        box_color = 'lightcoral'
    
    # Special case: High SNR + Low Gain = Training data was optimal
    if snr_score == 2 and gain_score == 1 and total_score >= 5:
        eval_text += "\n💡 Training policy\nalready near-optimal"
    
    plt.text(0.05, 0.95, eval_text, transform=plt.gca().transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.9))
    
    # ===== REWARD FUNCTION DIAGNOSTICS (Plots 17-22) =====
    if metrics.get('reward_latency_pairs') and len(metrics['reward_latency_pairs']) > 10:
        rewards_array = np.array([r for r, l in metrics['reward_latency_pairs']])
        latencies_array = np.array([l for r, l in metrics['reward_latency_pairs']])
        
        # 17. Reward vs Latency Scatter Plot
        plt.subplot(4, 6, 17)
        plt.scatter(latencies_array, rewards_array, alpha=0.3, s=10, c='blue', label='Training data')
        plt.xlabel('TTFT (ms)', fontsize=10)
        plt.ylabel('Reward', fontsize=10)
        plt.title('17. Reward Function Shape', fontsize=11, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Fit and plot trend line
        if len(latencies_array) > 2:
            z = np.polyfit(latencies_array, rewards_array, 2)
            p = np.poly1d(z)
            x_trend = np.linspace(latencies_array.min(), latencies_array.max(), 100)
            plt.plot(x_trend, p(x_trend), "r--", linewidth=2, alpha=0.7, label='Quadratic fit')
        
        plt.legend(loc='best', fontsize=8)
        
        corr = np.corrcoef(latencies_array, rewards_array)[0,1]
        plt.text(0.02, 0.98, f'Corr: {corr:.3f}\nN: {len(latencies_array):,}',
                transform=plt.gca().transAxes, verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 18. Reward Resolution (Spread per latency quantile)
        plt.subplot(4, 6, 18)
        n_quantiles = 10
        quantiles = np.percentile(latencies_array, np.linspace(0, 100, n_quantiles + 1))
        reward_spreads = []
        quantile_labels = []
        
        for i in range(n_quantiles):
            mask = (latencies_array >= quantiles[i]) & (latencies_array < quantiles[i+1])
            if mask.sum() > 0:
                reward_spread = rewards_array[mask].max() - rewards_array[mask].min()
                reward_spreads.append(reward_spread)
                quantile_labels.append(f'{quantiles[i]:.0f}-{quantiles[i+1]:.0f}')
        
        plt.bar(range(len(reward_spreads)), reward_spreads, color='steelblue', edgecolor='black')
        plt.xlabel('Latency Quantile', fontsize=10)
        plt.ylabel('Reward Spread', fontsize=10)
        plt.title('18. Reward Resolution', fontsize=11, fontweight='bold')
        plt.xticks(range(len(quantile_labels)), quantile_labels, rotation=45, ha='right', fontsize=7)
        plt.grid(True, alpha=0.3, axis='y')
        
        avg_spread = np.mean(reward_spreads) if reward_spreads else 0
        status = "✅ GOOD" if avg_spread > 0.1 else "⚠️ POOR"
        plt.text(0.02, 0.98, f'Avg: {avg_spread:.4f}\n{status}',
                transform=plt.gca().transAxes, verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='lightgreen' if avg_spread > 0.1 else 'lightcoral', alpha=0.8))
        
        # 19. Reward Sensitivity (gradient)
        plt.subplot(4, 6, 19)
        sorted_idx = np.argsort(latencies_array)
        lat_sorted = latencies_array[sorted_idx]
        rew_sorted = rewards_array[sorted_idx]
        
        window = max(10, len(lat_sorted) // 50)
        if len(lat_sorted) > window * 2:
            sensitivities = []
            sensitivity_lats = []
            for i in range(window, len(lat_sorted) - window):
                d_reward = rew_sorted[i+window] - rew_sorted[i-window]
                d_latency = lat_sorted[i+window] - lat_sorted[i-window]
                if abs(d_latency) > 1e-6:
                    sensitivities.append(d_reward / d_latency)
                    sensitivity_lats.append(lat_sorted[i])
            
            if sensitivities:
                plt.plot(sensitivity_lats, sensitivities, 'g-', linewidth=1.5, alpha=0.7)
                plt.axhline(y=0, color='r', linestyle='--', linewidth=1.5)
                plt.xlabel('Latency (ms)', fontsize=10)
                plt.ylabel('d(Reward)/d(Latency)', fontsize=10)
                plt.title('19. Reward Sensitivity', fontsize=11, fontweight='bold')
                plt.grid(True, alpha=0.3)
                
                avg_sensitivity = np.mean(np.abs(sensitivities))
                neg_slope = "✅" if np.mean(sensitivities) < 0 else "⚠️"
                plt.text(0.02, 0.98, f'Avg |Sens|: {avg_sensitivity:.6f}\nNeg: {neg_slope}',
                        transform=plt.gca().transAxes, verticalalignment='top', fontsize=8,
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # 20. Latency Distribution
        plt.subplot(4, 6, 20)
        plt.hist(latencies_array, bins=50, alpha=0.7, color='orange', edgecolor='black')
        plt.axvline(np.median(latencies_array), color='r', linestyle='--', linewidth=2)
        plt.axvline(np.percentile(latencies_array, 95), color='purple', linestyle='--', linewidth=2)
        plt.xlabel('TTFT (ms)', fontsize=10)
        plt.ylabel('Frequency', fontsize=10)
        plt.title('20. Latency Distribution', fontsize=11, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plt.text(0.98, 0.98, f'Min: {latencies_array.min():.0f}\nMax: {latencies_array.max():.0f}\n'
                f'P50: {np.median(latencies_array):.0f}\nP95: {np.percentile(latencies_array, 95):.0f}',
                transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # 21. Reward Distribution
        plt.subplot(4, 6, 21)
        plt.hist(rewards_array, bins=50, alpha=0.7, color='green', edgecolor='black')
        plt.axvline(np.median(rewards_array), color='r', linestyle='--', linewidth=2)
        plt.xlabel('Reward', fontsize=10)
        plt.ylabel('Frequency', fontsize=10)
        plt.title('21. Reward Distribution', fontsize=11, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        reward_range = rewards_array.max() - rewards_array.min()
        plt.text(0.98, 0.98, f'Range: {reward_range:.3f}\nMean: {rewards_array.mean():.3f}\nStd: {rewards_array.std():.3f}',
                transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # 22. 🚨 CRITICAL: Reward Discrimination by Latency Category
        plt.subplot(4, 6, 22)
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
        plt.title('22. 🚨 Reward Discrimination', fontsize=11, fontweight='bold')
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
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(final_model_dir, 'comprehensive_neural_cb_metrics.pdf')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved comprehensive training plot: {plot_path}")
    
    # Also save CSV files for future analysis
    if metrics['losses']:
        metrics_df = pd.DataFrame({
            'update_step': list(range(len(metrics['losses']))),
            'loss': metrics['losses'],
            'reward': metrics['rewards'] if len(metrics['rewards']) == len(metrics['losses']) else [None] * len(metrics['losses']),
            'epsilon': metrics['epsilons'] if len(metrics['epsilons']) == len(metrics['losses']) else [None] * len(metrics['losses'])
        })
        csv_path = os.path.join(final_model_dir, 'training_metrics.csv')
        metrics_df.to_csv(csv_path, index=False)
        logger.info(f"Saved training metrics CSV: {csv_path}")
    
    return plot_path


def train_batch(encoded_training_dir, final_model_dir, HYPERPARAMETERS, num_epochs=3):
    """
    Train neural contextual bandit on batch of encoded experiences.
    Compatible with existing data pipeline (called from online_train_routine).
    
    Args:
        encoded_training_dir: Directory containing encoded .pt files
        final_model_dir: Directory to save model
        HYPERPARAMETERS: Model hyperparameters
        num_epochs: Number of training epochs over the data
    """
    global _cached_agent
    
    logger.info(f"Starting Neural CB batch training: epochs={num_epochs}, dir={encoded_training_dir}")
    
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
                _cached_agent.load(final_model_dir)
                logger.info(f"Loaded existing model from {final_model_dir}")
            except Exception as e:
                logger.warning(f"Failed to load existing model: {e}, starting fresh")
    
    # Training loop
    total_samples = 0
    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_losses = []
        epoch_rewards = []
        
        for tensor_file in tensor_files:
            # tensor_file is already the full path
            batch_data = torch.load(tensor_file)
            
            # Extract tensors
            pod_features = batch_data['pod_features_with_staleness']
            kv_hit_ratios = batch_data['kv_hit_ratios']
            request_features = batch_data['request_features']
            actions = batch_data['actions']
            rewards = batch_data['rewards']
            
            # Extract latency values for reward function analysis
            ttft = batch_data.get('ttft', None)
            avg_tpot = batch_data.get('avg_tpot', None)
            
            batch_size = len(actions)
            
            # Add experiences to replay buffer
            for i in range(batch_size):
                _cached_agent.remember(
                    pod_features[i:i+1],
                    kv_hit_ratios[i:i+1],
                    request_features[i:i+1],
                    actions[i].item(),
                    rewards[i].item()
                )
                
                # Collect reward-latency pairs for function analysis (sample 10% to save memory)
                if np.random.random() < 0.1 and ttft is not None:
                    _cached_agent.training_metrics['reward_latency_pairs'].append(
                        (rewards[i].item(), ttft[i].item())
                    )
                    _cached_agent.training_metrics['ttft_values'].append(ttft[i].item())
                    if avg_tpot is not None:
                        _cached_agent.training_metrics['tpot_values'].append(avg_tpot[i].item())
                
                total_samples += 1
                
                # Trigger learning periodically (every 500 samples, not every sample!)
                if total_samples % 500 == 0 and len(_cached_agent.replay_buffer) >= _cached_agent.batch_size:
                    metrics = _cached_agent.learn()
                    epoch_losses.append(metrics['loss'])
                    epoch_rewards.append(metrics['reward'])
        
        # Log epoch metrics
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0.0
        epoch_time = time.time() - epoch_start
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.4f}, avg_reward={avg_reward:.4f}, "
                   f"time={epoch_time:.2f}s, buffer_size={len(_cached_agent.replay_buffer)}")
    
    # Save trained model
    _cached_agent.save(final_model_dir)
    logger.info(f"Neural CB batch training complete: {total_samples} samples processed, model saved to {final_model_dir}")
    
    # Generate comprehensive training plots
    plot_path = plot_neural_cb_metrics(_cached_agent, final_model_dir, num_epochs, total_samples)
    return plot_path


def train(encoded_data_dir, final_model_dir, HYPERPARAMETERS, is_online_learning):
    """
    Legacy training function for compatibility.
    Now redirects to train_batch().
    """
    logger.info("Legacy train() called, redirecting to train_batch()")
    return train_batch(
        encoded_training_dir=encoded_data_dir,
        final_model_dir=final_model_dir,
        HYPERPARAMETERS=HYPERPARAMETERS,
        num_epochs=HYPERPARAMETERS.get('num_epochs', 3)
    )


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

