#!/usr/bin/env python3

# simplified_contextual_bandit.py

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.distributions import Categorical
import pickle
import time
import matplotlib.pyplot as plt
from datetime import datetime
import glob
from logger import logger
import traceback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
training_results_dir = "training_results"
final_model_path = "final_model"


class SimplePolicyNetwork(nn.Module):
    """
    Simplified Policy Network - much more appropriate for small datasets
    Flattens all inputs and uses simple feedforward layers
    """
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        super().__init__()
        
        # Calculate total input size by flattening everything
        pod_features_size = state_dim['pod_features'] * state_dim['num_pods']
        kv_hit_size = state_dim['kv_hit_ratios'] * state_dim['num_pods'] 
        request_size = state_dim['request_features']
        
        total_input_size = pod_features_size + kv_hit_size + request_size
        
        logger.info(f"Simplified model architecture:")
        logger.info(f"  Input size: {total_input_size}")
        logger.info(f"  - Pod features: {pod_features_size} ({state_dim['num_pods']} pods × {state_dim['pod_features']} features)")
        logger.info(f"  - KV hit ratios: {kv_hit_size} ({state_dim['num_pods']} pods × {state_dim['kv_hit_ratios']} ratios)")
        logger.info(f"  - Request features: {request_size}")
        logger.info(f"  Hidden dimension: {hidden_dim}")
        logger.info(f"  Output size: {action_dim} (number of pods)")
        
        # Simple feedforward network with dropout for regularization
        self.network = nn.Sequential(
            nn.Linear(total_input_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Store dimensions for reshaping
        self.state_dim = state_dim
        
        # Calculate and log parameter count
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"  Total parameters: {total_params:,}")
        
        # Create dummy policy_head attribute for compatibility with existing code
        self.policy_head = nn.Linear(hidden_dim // 2, action_dim)
        self.policy_head.out_features = action_dim
        
    def forward(self, pod_features, kv_hit_ratios, request_features, return_attention=False):
        batch_size = pod_features.shape[0]
        
        # Flatten all inputs and concatenate
        pod_flat = pod_features.view(batch_size, -1)  # [batch, num_pods * pod_feature_dim]
        kv_flat = kv_hit_ratios.view(batch_size, -1)  # [batch, num_pods * kv_dim]
        req_flat = request_features.view(batch_size, -1)  # [batch, request_dim]
        
        # Concatenate all features
        combined_input = torch.cat([pod_flat, kv_flat, req_flat], dim=1)
        
        # Forward pass through simple network
        logits = self.network(combined_input)
        
        # Convert to probabilities
        # action_probs = F.softmax(logits, dim=1)
        action_probs = F.softmax(logits / 3.0, dim=1)  # Temperature = 3.0
        if return_attention:
            # No attention in simplified model, return dummy weights for compatibility
            num_pods = self.state_dim['num_pods']
            dummy_attention = torch.ones(batch_size, num_pods, num_pods, device=action_probs.device) / num_pods
            return action_probs, dummy_attention
        
        return action_probs
    
    def get_action(self, pod_features, kv_hit_ratios, request_features, explore=True, epsilon=0.1):
        action_probs = self.forward(pod_features, kv_hit_ratios, request_features)
        
        if not explore:
            # Purely exploit - select the pod with highest probability
            return torch.argmax(action_probs, dim=1)
        
        # Epsilon-greedy exploration
        batch_size = pod_features.shape[0]
        device = action_probs.device
        random_actions = torch.randint(0, action_probs.shape[1], (batch_size,), device=device)
        greedy_actions = torch.argmax(action_probs, dim=1)
        
        # Random mask for exploration
        explore_mask = (torch.rand(batch_size, device=device) < epsilon).long()
        
        # Choose either exploration or exploitation based on epsilon
        actions = (1 - explore_mask) * greedy_actions + explore_mask * random_actions
        
        # Calculate log probabilities for the chosen actions
        log_probs = torch.log(torch.gather(action_probs, 1, actions.unsqueeze(1)).squeeze(1) + 1e-10)
        
        return actions, log_probs


class SimplifiedContextualBandit:
    """
    Simplified Contextual Bandit optimized for small datasets
    """
    def __init__(self, state_dim, action_dim, hidden_dim=32, lr=1e-3, 
                 batch_size=32, exploration_rate=0.2):
        self.batch_size = batch_size
        self.exploration_rate = exploration_rate
        
        # Initialize simplified policy network
        self.policy = SimplePolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
        
        # Optimizer with weight decay for regularization
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), 
            lr=lr, 
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            verbose=True
        )
        
        # Initialize memory attributes
        self.pod_features = []
        self.kv_hit_ratios = []
        self.request_features = []
        self.actions = []
        self.rewards = []
        
        # Metrics for tracking
        self.loss_history = []
        self.reward_history = []
        self.entropy_history = []
        
    def remember(self, pod_features, kv_hit_ratios, request_features, action, reward):
        """Store context-action-reward tuple in memory"""
        self.pod_features.append(pod_features)
        self.kv_hit_ratios.append(kv_hit_ratios)
        self.request_features.append(request_features)
        self.actions.append(action)
        self.rewards.append(reward)
        
    def choose_action(self, pod_features, kv_hit_ratios, request_features, evaluate=False):
        """Select an action (pod) for the given context"""
        with torch.no_grad():
            if evaluate:
                # Pure exploitation during evaluation
                action_probs = self.policy(pod_features, kv_hit_ratios, request_features)
                action = torch.argmax(action_probs, dim=1)
                return action
            else:
                # Exploration-exploitation during training
                action, log_prob = self.policy.get_action(
                    pod_features, kv_hit_ratios, request_features, 
                    explore=True, epsilon=self.exploration_rate
                )
                return action, log_prob
    
    def learn(self):
        """Update policy using rewards with improved stability"""
        if len(self.pod_features) == 0:
            return {
                'loss': 0.0,
                'reward': 0.0,
                'entropy': 0.0
            }
        
        # Stack all tensors
        pod_features = torch.cat(self.pod_features, dim=0)
        kv_hit_ratios = torch.cat(self.kv_hit_ratios, dim=0)
        request_features = torch.cat(self.request_features, dim=0)
        actions = torch.cat(self.actions, dim=0)
        rewards = torch.cat(self.rewards, dim=0).view(-1, 1)
        
        # Improved reward normalization with clipping
        if rewards.std() > 1e-6:
            normalized_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            # Clip extreme values to prevent instability
            normalized_rewards = torch.clamp(normalized_rewards, -3.0, 3.0)
        else:
            # If rewards have no variance, use them as-is
            normalized_rewards = rewards
        
        # Create batches
        n_samples = len(self.pod_features)
        batch_size = min(self.batch_size, n_samples)
        batch_start = np.arange(0, n_samples, batch_size)
        indices = np.arange(n_samples, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + batch_size] for i in batch_start]
        
        logger.debug(f"Starting learning with {n_samples} experiences in memory")
        
        epoch_loss = 0
        epoch_entropy = 0
        num_updates = 0
        
        # Process each batch
        for batch_idx, batch_indices in enumerate(batches):
            # Get batch data
            batch_pod_features = pod_features[batch_indices]
            batch_kv_hit_ratios = kv_hit_ratios[batch_indices]
            batch_request_features = request_features[batch_indices]
            batch_actions = actions[batch_indices]
            batch_rewards = normalized_rewards[batch_indices]
            
            # Get current policy distributions
            action_probs = self.policy(batch_pod_features, batch_kv_hit_ratios, batch_request_features)
            
            # Add small epsilon to prevent numerical issues
            action_probs = action_probs + 1e-8
            action_probs = action_probs / action_probs.sum(dim=1, keepdim=True)
            
            dist = Categorical(action_probs)
            
            # Calculate log probabilities of the actions taken
            log_probs = dist.log_prob(batch_actions)
            
            # Calculate entropy for monitoring exploration
            entropy = dist.entropy().mean()
            
            # Calculate loss (policy gradient with baseline)
            loss = -(log_probs * batch_rewards.squeeze()).mean()
            
            # Add small entropy bonus to encourage exploration
            # entropy_bonus = 0.01 * entropy
            entropy_bonus = 0.2 * entropy  # 20x stronger!
            total_loss = loss - entropy_bonus
            
            # Update policy
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            epoch_loss += total_loss.item()
            epoch_entropy += entropy.item()
            num_updates += 1
        
        # Update learning rate based on loss
        avg_loss = epoch_loss / max(1, num_updates)
        self.scheduler.step(avg_loss)
        
        # Clear memory
        self.clear_memory()
        
        # Store metrics
        avg_reward = rewards.mean().item()
        avg_entropy = epoch_entropy / max(1, num_updates)
        
        self.loss_history.append(avg_loss)
        self.reward_history.append(avg_reward)
        self.entropy_history.append(avg_entropy)
        
        return {
            'loss': avg_loss,
            'reward': avg_reward,
            'entropy': avg_entropy
        }

    def clear_memory(self):
        """Clear memory buffers"""
        self.pod_features = []
        self.kv_hit_ratios = []
        self.request_features = []
        self.actions = []
        self.rewards = []

    def save(self, directory):
        """Save the agent's parameters to the specified directory"""
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Creating directory: {directory}")
        
        # Save policy network
        torch.save(self.policy.state_dict(), os.path.join(directory, 'policy.pth'))
        
        # Save optimizer state
        torch.save(self.optimizer.state_dict(), os.path.join(directory, 'optimizer.pth'))
        
        # Save training history
        history = {
            'loss': self.loss_history,
            'reward': self.reward_history,
            'entropy': self.entropy_history
        }
        
        with open(os.path.join(directory, 'history.pkl'), 'wb') as f:
            pickle.dump(history, f)
            
        # Copy to final model path
        os.makedirs(final_model_path, exist_ok=True)
        os.system(f"cp {directory}/* {final_model_path}")
        logger.info(f"Saved simplified agent to {directory}")
    
    def load(self, directory):
        """Load the agent's parameters from the specified directory"""
        # Load policy network
        policy_path = os.path.join(directory, 'policy.pth')
        if os.path.exists(policy_path):
            self.policy.load_state_dict(torch.load(policy_path, map_location=device))
        
        # Load optimizer state if available
        optimizer_path = os.path.join(directory, 'optimizer.pth')
        if os.path.exists(optimizer_path):
            try:
                self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
            except:
                logger.warning("Could not load optimizer state, continuing with fresh optimizer")
        
        # Load training history if available
        history_path = os.path.join(directory, 'history.pkl')
        if os.path.exists(history_path):
            with open(history_path, 'rb') as f:
                history = pickle.load(f)
                
                self.loss_history = history.get('loss', [])
                self.reward_history = history.get('reward', [])
                self.entropy_history = history.get('entropy', [])
                
        logger.info(f"Loaded simplified agent from {directory}")


class RoutingDataset(Dataset):
    """
    Dataset for LLM routing data, creates batches for contextual bandit training
    """
    def __init__(self, tensor_data):
        self.pod_features = tensor_data['pod_features_with_staleness']
        self.kv_hit_ratios = tensor_data['kv_hit_ratios']
        self.request_features = tensor_data['request_features']
        self.actions = tensor_data['actions']
        self.rewards = tensor_data['rewards']
        
    def __len__(self):
        return len(self.rewards)
    
    def __getitem__(self, idx):
        return {
            'pod_features': self.pod_features[idx],
            'kv_hit_ratios': self.kv_hit_ratios[idx],
            'request_features': self.request_features[idx],
            'action': self.actions[idx],
            'reward': self.rewards[idx]
        }


def analyze_dataset_detailed(combined_data):
    """
    Comprehensive analysis to understand dataset characteristics
    """
    logger.info("=" * 80)
    logger.info("DATASET ANALYSIS")
    logger.info("=" * 80)
    
    total_samples = len(combined_data['actions'])
    num_pods = combined_data['pod_features'].shape[1]
    
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Number of pods: {num_pods}")
    logger.info(f"Random baseline accuracy: {1/num_pods:.3f} ({100/num_pods:.1f}%)")
    
    # Action distribution analysis
    actions = combined_data['actions']
    action_counts = torch.bincount(actions, minlength=num_pods)
    action_percentages = action_counts.float() / total_samples * 100
    
    logger.info("\nAction distribution:")
    for i in range(num_pods):
        logger.info(f"  Pod {i}: {action_counts[i]} samples ({action_percentages[i]:.1f}%)")
    
    # Check for class imbalance
    max_action_pct = action_percentages.max().item()
    min_action_pct = action_percentages.min().item()
    imbalance_ratio = max_action_pct / max(min_action_pct, 0.1)
    
    logger.info(f"\nClass imbalance analysis:")
    logger.info(f"  Most frequent action: {max_action_pct:.1f}%")
    logger.info(f"  Least frequent action: {min_action_pct:.1f}%")
    logger.info(f"  Imbalance ratio: {imbalance_ratio:.1f}x")
    
    if imbalance_ratio > 3:
        logger.warning(f"  ⚠️  CLASS IMBALANCE DETECTED! ({imbalance_ratio:.1f}x)")
    
    # Reward analysis
    rewards = combined_data['rewards']
    logger.info(f"\nReward statistics:")
    logger.info(f"  Range: [{rewards.min():.4f}, {rewards.max():.4f}]")
    logger.info(f"  Mean: {rewards.mean():.4f}")
    logger.info(f"  Std: {rewards.std():.4f}")
    
    # Reward by action
    logger.info(f"\nReward by action:")
    action_reward_stats = {}
    for action_idx in range(num_pods):
        action_mask = actions == action_idx
        if action_mask.sum() > 0:
            action_rewards = rewards[action_mask]
            mean_reward = action_rewards.mean().item()
            action_reward_stats[action_idx] = mean_reward
            logger.info(f"  Pod {action_idx}: μ={mean_reward:.4f} (n={action_mask.sum()})")
    
    # Find reward differences
    if len(action_reward_stats) > 1:
        reward_values = list(action_reward_stats.values())
        reward_gap = max(reward_values) - min(reward_values)
        logger.info(f"\nReward signal strength:")
        logger.info(f"  Reward gap: {reward_gap:.4f}")
        
        if reward_gap < 0.01:
            logger.warning(f"  ⚠️  VERY WEAK REWARD SIGNAL! (gap: {reward_gap:.4f})")
    
    return {
        'total_samples': total_samples,
        'imbalance_ratio': imbalance_ratio,
        'reward_gap': reward_gap if 'reward_gap' in locals() else 0,
        'action_distribution': action_counts
    }


def evaluate_agent(agent, eval_data, num_samples=100):
    """Evaluate agent performance"""
    # Extract data
    pod_features = eval_data['pod_features_with_staleness'].to(device)
    kv_hit_ratios = eval_data['kv_hit_ratios'].to(device)
    request_features = eval_data['request_features'].to(device)
    true_actions = eval_data['actions'].to(device)
    rewards = eval_data['rewards'].to(device)
    
    # Limit to specified number of samples
    if len(rewards) > num_samples:
        indices = torch.randperm(len(rewards))[:num_samples]
        pod_features = pod_features[indices]
        kv_hit_ratios = kv_hit_ratios[indices]
        request_features = request_features[indices]
        true_actions = true_actions[indices]
        rewards = rewards[indices]
    
    # Evaluate agent
    agent.policy.eval()
    with torch.no_grad():
        # Get agent's actions
        pred_actions = agent.choose_action(pod_features, kv_hit_ratios, request_features, evaluate=True)
        
        # Get action probabilities
        action_probs = agent.policy(pod_features, kv_hit_ratios, request_features)
        
        # Calculate accuracy
        accuracy = (pred_actions == true_actions).float().mean().item()
        
        # Calculate average confidence
        max_probs = action_probs.max(dim=1)[0]
        avg_confidence = max_probs.mean().item()
        
        # Calculate reward for predicted actions
        true_reward = rewards.mean().item()
        
    # Additional metrics
    metrics = {
        'accuracy': accuracy,
        'avg_confidence': avg_confidence,
        'true_reward': true_reward,
        'probs': action_probs.cpu().numpy(),
        'pred_actions': pred_actions.cpu().numpy(),
        'true_actions': true_actions.cpu().numpy()
    }
    
    agent.policy.train()
    return metrics

def plot_training_metrics(agent, eval_metrics, output_dir, combined_data=None):
    """
    Enhanced plotting function with correct action distribution analysis
    """
    # Create larger figure with more subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Determine action dimension
    if hasattr(agent.policy, 'policy_head'):
        action_dim = agent.policy.policy_head.out_features
    else:
        action_dim = 7  # Default for your setup
    
    # 1. Policy Loss
    plt.subplot(3, 4, 1)
    if agent.loss_history:
        plt.plot(agent.loss_history)
        plt.title('Policy Loss')
        plt.xlabel('Updates')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
    
    # 2. Average Reward
    plt.subplot(3, 4, 2)
    if agent.reward_history:
        plt.plot(agent.reward_history)
        plt.title('Average Reward')
        plt.xlabel('Updates')
        plt.ylabel('Reward')
        plt.grid(True, alpha=0.3)
    
    # 3. Policy Entropy
    plt.subplot(3, 4, 3)
    if agent.entropy_history:
        plt.plot(agent.entropy_history)
        plt.title('Policy Entropy')
        plt.xlabel('Updates')
        plt.ylabel('Entropy')
        plt.grid(True, alpha=0.3)
    
    # 4. Evaluation Accuracy
    plt.subplot(3, 4, 4)
    if eval_metrics:
        accuracies = [m['accuracy'] for m in eval_metrics]
        plt.plot(accuracies, 'b-', linewidth=2)
        plt.title('Evaluation Accuracy')
        plt.xlabel('Evaluations')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)
        
        # Add random baseline
        random_baseline = 1.0 / action_dim
        plt.axhline(y=random_baseline, color='r', linestyle='--', 
                   label=f'Random Baseline ({random_baseline:.3f})')
        plt.legend()
        
        # Highlight final accuracy
        if accuracies:
            final_acc = accuracies[-1]
            plt.text(0.02, 0.98, f'Final: {final_acc:.3f}', 
                    transform=plt.gca().transAxes, 
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 5. Average Confidence Over Time
    plt.subplot(3, 4, 5)
    if eval_metrics:
        confidences = [m.get('avg_confidence', 0) for m in eval_metrics]
        plt.plot(confidences, 'g-', linewidth=2)
        plt.title('Average Confidence')
        plt.xlabel('Evaluations')
        plt.ylabel('Confidence')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Add interpretation
        if confidences:
            final_conf = confidences[-1]
            plt.text(0.02, 0.98, f'Final: {final_conf:.3f}', 
                    transform=plt.gca().transAxes, 
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # 6. Training Data Action Distribution (Ground Truth)
    plt.subplot(3, 4, 6)
    if combined_data is not None and 'actions' in combined_data:
        training_actions = combined_data['actions']
        training_action_counts = torch.bincount(training_actions, minlength=action_dim).numpy()
        
        bars = plt.bar(range(action_dim), training_action_counts, 
                      color='skyblue', alpha=0.7, edgecolor='navy')
        plt.title('Training Data Distribution')
        plt.xlabel('Pod ID')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        
        # Add percentages on bars
        total_samples = training_action_counts.sum()
        for i, (bar, count) in enumerate(zip(bars, training_action_counts)):
            pct = count / total_samples * 100
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{count}\n({pct:.1f}%)', 
                    ha='center', va='bottom', fontsize=9)
        
        # Check for imbalance
        max_pct = (training_action_counts.max() / total_samples * 100)
        min_pct = (training_action_counts.min() / total_samples * 100)
        imbalance_ratio = max_pct / max(min_pct, 0.1)
        
        plt.text(0.02, 0.98, f'Imbalance: {imbalance_ratio:.1f}x', 
                transform=plt.gca().transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', 
                         facecolor='red' if imbalance_ratio > 3 else 'lightblue', 
                         alpha=0.8))
    
    # 7. Model Predictions vs Ground Truth (Latest Evaluation)
    plt.subplot(3, 4, 7)
    if eval_metrics:
        last_eval = eval_metrics[-1]
        pred_actions = last_eval['pred_actions']
        true_actions = last_eval['true_actions']
        
        pred_counts = np.bincount(pred_actions, minlength=action_dim)
        true_counts = np.bincount(true_actions, minlength=action_dim)
        
        x = np.arange(action_dim)
        width = 0.35
        
        bars1 = plt.bar(x - width/2, true_counts, width, 
                       label='Ground Truth', alpha=0.7, color='lightcoral')
        bars2 = plt.bar(x + width/2, pred_counts, width, 
                       label='Predicted', alpha=0.7, color='lightblue')
        
        plt.title('Predictions vs Ground Truth\n(Latest Evaluation)')
        plt.xlabel('Pod ID')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add counts on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    plt.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                            f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    # 8. Action Probability Heatmap (Latest Evaluation)
    plt.subplot(3, 4, 8)
    if eval_metrics and 'probs' in eval_metrics[-1]:
        probs = eval_metrics[-1]['probs']
        
        # Show average probabilities
        avg_probs = np.mean(probs, axis=0)
        bars = plt.bar(range(action_dim), avg_probs, 
                      color='orange', alpha=0.7)
        plt.title('Average Action Probabilities')
        plt.xlabel('Pod ID')
        plt.ylabel('Probability')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Add values on bars
        for i, (bar, prob) in enumerate(zip(bars, avg_probs)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{prob:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Add uniform baseline
        uniform_prob = 1.0 / action_dim
        plt.axhline(y=uniform_prob, color='r', linestyle='--', 
                   label=f'Uniform ({uniform_prob:.3f})')
        plt.legend()
    
    # 9. Confidence Distribution
    plt.subplot(3, 4, 9)
    if eval_metrics and 'probs' in eval_metrics[-1]:
        probs = eval_metrics[-1]['probs']
        max_probs = np.max(probs, axis=1)  # Confidence for each prediction
        
        plt.hist(max_probs, bins=20, alpha=0.7, color='purple', edgecolor='black')
        plt.title('Confidence Distribution')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_conf = np.mean(max_probs)
        std_conf = np.std(max_probs)
        plt.axvline(mean_conf, color='red', linestyle='-', linewidth=2, 
                   label=f'Mean: {mean_conf:.3f}')
        plt.axvline(mean_conf - std_conf, color='red', linestyle='--', alpha=0.7)
        plt.axvline(mean_conf + std_conf, color='red', linestyle='--', alpha=0.7)
        plt.legend()
    
    # 10. Accuracy vs Confidence Scatter
    plt.subplot(3, 4, 10)
    if eval_metrics:
        accuracies = [m['accuracy'] for m in eval_metrics]
        confidences = [m.get('avg_confidence', 0) for m in eval_metrics]
        
        plt.scatter(confidences, accuracies, alpha=0.6, s=50)
        plt.title('Accuracy vs Confidence')
        plt.xlabel('Average Confidence')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)
        
        # Add ideal line (perfect calibration)
        min_val = min(min(confidences), min(accuracies))
        max_val = max(max(confidences), max(accuracies))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect Calibration')
        plt.legend()
    
    # 11. Learning Curves Summary
    plt.subplot(3, 4, 11)
    if agent.loss_history and agent.reward_history:
        # Normalize curves for comparison
        loss_norm = (np.array(agent.loss_history) - np.min(agent.loss_history)) / (np.max(agent.loss_history) - np.min(agent.loss_history) + 1e-8)
        reward_norm = (np.array(agent.reward_history) - np.min(agent.reward_history)) / (np.max(agent.reward_history) - np.min(agent.reward_history) + 1e-8)
        
        plt.plot(loss_norm, label='Loss (normalized)', alpha=0.7)
        plt.plot(reward_norm, label='Reward (normalized)', alpha=0.7)
        
        if eval_metrics:
            accuracies = [m['accuracy'] for m in eval_metrics]
            # Interpolate accuracy to match other curves
            acc_interp = np.interp(np.linspace(0, len(accuracies)-1, len(loss_norm)), 
                                  np.arange(len(accuracies)), accuracies)
            acc_norm = (acc_interp - np.min(acc_interp)) / (np.max(acc_interp) - np.min(acc_interp) + 1e-8)
            plt.plot(acc_norm, label='Accuracy (normalized)', alpha=0.7)
        
        plt.title('Learning Curves (Normalized)')
        plt.xlabel('Updates')
        plt.ylabel('Normalized Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 12. Model Summary Statistics
    plt.subplot(3, 4, 12)
    plt.axis('off')
    
    # Create summary text
    summary_text = "MODEL SUMMARY\n" + "="*15 + "\n"
    
    if combined_data is not None:
        total_samples = len(combined_data['actions'])
        summary_text += f"Dataset size: {total_samples:,} samples\n"
    
    if hasattr(agent.policy, 'parameters'):
        total_params = sum(p.numel() for p in agent.policy.parameters())
        summary_text += f"Model parameters: {total_params:,}\n"
    
    if eval_metrics:
        final_accuracy = eval_metrics[-1]['accuracy']
        final_confidence = eval_metrics[-1].get('avg_confidence', 0)
        random_baseline = 1.0 / action_dim
        
        summary_text += f"\nFINAL PERFORMANCE:\n"
        summary_text += f"Accuracy: {final_accuracy:.3f} ({final_accuracy*100:.1f}%)\n"
        summary_text += f"Confidence: {final_confidence:.3f} ({final_confidence*100:.1f}%)\n"
        summary_text += f"Random baseline: {random_baseline:.3f} ({random_baseline*100:.1f}%)\n"
        
        if final_accuracy > random_baseline * 1.5:
            summary_text += f"\n✅ LEARNING DETECTED\n"
        elif final_accuracy > random_baseline * 1.1:
            summary_text += f"\n⚠️  MODEST LEARNING\n"
        else:
            summary_text += f"\n❌ NO CLEAR LEARNING\n"
        
        # Calibration assessment
        if abs(final_confidence - final_accuracy) < 0.1:
            summary_text += "✅ Well calibrated\n"
        elif final_confidence > final_accuracy + 0.2:
            summary_text += "⚠️  Overconfident\n"
        else:
            summary_text += "⚠️  Underconfident\n"
    
    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    fn = f"{output_dir}/comprehensive_training_metrics.pdf"
    plt.savefig(fn, dpi=150, bbox_inches='tight')
    
    # Copy to final model path
    import os
    final_model_path = "final_model"  # Adjust if different
    if os.path.exists(final_model_path):
        os.system(f"cp {fn} {final_model_path}/")
    
    plt.close()
    
    logger.info(f"Saved comprehensive training metrics plots to {output_dir}")
    
    # Print summary to console
    if eval_metrics:
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        if combined_data is not None:
            print(f"Dataset size: {len(combined_data['actions']):,} samples")
        if hasattr(agent.policy, 'parameters'):
            total_params = sum(p.numel() for p in agent.policy.parameters())
            print(f"Model parameters: {total_params:,}")
        
        final_accuracy = eval_metrics[-1]['accuracy']
        final_confidence = eval_metrics[-1].get('avg_confidence', 0)
        random_baseline = 1.0 / action_dim
        
        print(f"Final accuracy: {final_accuracy:.3f} ({final_accuracy*100:.1f}%)")
        print(f"Final confidence: {final_confidence:.3f} ({final_confidence*100:.1f}%)")
        print(f"Random baseline: {random_baseline:.3f} ({random_baseline*100:.1f}%)")
        
        if final_accuracy > random_baseline * 1.5:
            print("✅ Model is learning significantly!")
        elif final_accuracy > random_baseline * 1.1:
            print("⚠️  Model shows modest learning")
        else:
            print("❌ Model performance close to random")
        print("="*60)


def load_all_encoded_data(encoded_data_dir):
    """Load and combine data from all batch directories"""
    logger.info(f"Loading data from {encoded_data_dir}")
    
    # Find all batch subdirectories
    batch_dirs = glob.glob(os.path.join(encoded_data_dir, "batch_*"))
    logger.info(f"Found {len(batch_dirs)} batch directories")
    
    combined_data = None
    total_samples = 0
    
    # Process each batch directory
    for batch_dir in batch_dirs:
        # Look for tensor_dataset.pt in the batch directory or its train subdirectory
        tensor_path = os.path.join(batch_dir, "tensor_dataset.pt")
        if not os.path.exists(tensor_path):
            train_dir = os.path.join(batch_dir, "train")
            if os.path.exists(train_dir):
                tensor_path = os.path.join(train_dir, "tensor_dataset.pt")
                
        if not os.path.exists(tensor_path):
            logger.warning(f"No tensor_dataset.pt found in {batch_dir}")
            continue
            
        try:
            # Load tensor data
            batch_data = torch.load(tensor_path, map_location='cpu')
            
            # If this is the first valid batch, use it as the base
            if combined_data is None:
                combined_data = batch_data
                total_samples = batch_data['rewards'].size(0)
                logger.info(f"First batch has {total_samples} samples")
            else:
                # Combine the data by concatenating along the batch dimension
                for key in combined_data:
                    if key in batch_data:
                        combined_data[key] = torch.cat([combined_data[key], batch_data[key]], dim=0)
                        
                batch_samples = batch_data['rewards'].size(0)
                total_samples += batch_samples
                logger.debug(f"Added {batch_samples} samples from {os.path.basename(batch_dir)}")
                    
        except Exception as e:
            logger.error(f"Error loading data from {batch_dir}: {e}")
            continue
    
    if combined_data is None:
        logger.error("No valid data could be loaded from any batch")
        raise ValueError("No valid data found in the encoded_data directory")
        
    logger.info(f"Successfully combined data from multiple batches, total samples: {total_samples}")
    
    return combined_data


def load_previous_model():
    """Load previous model if it exists"""
    global final_model_path
    if os.path.exists(final_model_path):
        logger.info(f"Found previous model at {final_model_path}")
        return final_model_path
    return None


def train(encoded_data_dir):
    """Main training function with optimized hyperparameters for small datasets"""
    global training_results_dir
    
    # Optimized hyperparameters for small datasets
    def read_hyperparameters_from_file(file_path):
        """Read hyperparameters from a file"""
        with open(file_path, 'r') as f:
            lines = f.readlines()
        params = {}
        for line in lines:
            if '=' in line:
                key, value = line.split('=')
                if value.strip().isdigit():
                    params[key.strip()] = float(value.strip())
                else:
                    params[key.strip()] = value.strip()
            logger.info(f"Read hyperparameter: {key.strip()} = {params[key.strip()]}")
        return params
    ''' example hyperparameters.txt content:
    hidden_dim=32
    batch_size=16
    lr=0.001
    exploration_rate=0.3
    training_epochs=50
    max_updates_per_epoch=50
    eval_interval=10
    '''
    # hyperparams = read_hyperparameters_from_file('hyperparameters.txt')
    hyperparams = {
        'hidden_dim': 32,  # Reduced hidden dimension for small dataset
        'batch_size': 16,  # Smaller batch size for small dataset
        'lr': 0.001,  # Learning rate
        'exploration_rate': 0.25,  # Exploration rate
        'training_epochs': 10,  # Number of training epochs
        'max_updates_per_epoch': 50,  # Maximum updates per epoch
        'eval_interval': 10  # Evaluation interval
    }
    seed = 42
    continue_training = False
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Set output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(training_results_dir, exist_ok=True)
    output_dir = os.path.join(training_results_dir, f"simple_cb_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and combine data from all batches
    combined_data = load_all_encoded_data(encoded_data_dir)
    
    # Analyze dataset
    dataset_analysis = analyze_dataset_detailed(combined_data)
    
    # Check if we should continue training from a previous model
    previous_model_path = None
    if continue_training:
        previous_model_path = load_previous_model()
    
    # Create configuration
    config = {
        'hidden_dim': hyperparams['hidden_dim'],
        'batch_size': hyperparams['batch_size'],
        'learning_rate': hyperparams['lr'],
        'exploration_rate': hyperparams['exploration_rate'],
        'num_training_epochs': hyperparams['training_epochs'],
        'max_updates_per_epoch': hyperparams['max_updates_per_epoch'],
        'eval_interval': hyperparams['eval_interval'],
        'seed': seed,
        'model_type': 'simplified',
        'dataset_analysis': dataset_analysis
    }
    
    # Save configuration
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4, default=str)
    
    # Determine state dimensions
    state_dim = {
        'pod_features': combined_data['pod_features_with_staleness'].shape[2],
        'kv_hit_ratios': combined_data['kv_hit_ratios'].shape[2],
        'request_features': combined_data['request_features'].shape[1],
        'num_pods': combined_data['pod_features'].shape[1]
    }
    
    # Determine action dimension (number of pods)
    action_dim = combined_data['pod_features'].shape[1]
    
    logger.info(f"State dimensions: {state_dim}")
    logger.info(f"Action dimension: {action_dim}")
    
    # Create Simplified Contextual Bandit agent
    agent = SimplifiedContextualBandit(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=config['hidden_dim'],
        lr=config['learning_rate'],
        batch_size=config['batch_size'],
        exploration_rate=config['exploration_rate']
    )
    
    # Load previous model if available
    if previous_model_path:
        try:
            agent.load(previous_model_path)
            logger.info(f"Successfully loaded previous model from {previous_model_path}")
        except Exception as e:
            logger.error(f"Error loading previous model: {e}")
    
    # Create dataset
    dataset = RoutingDataset(combined_data)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    number_of_batches = len(dataloader)
    
    logger.info(f"Loaded dataset with {len(dataset)} samples")
    logger.info(f"Batch size: {config['batch_size']}")
    logger.info(f"Number of batches in training data: {number_of_batches}")

    # Training loop
    logger.info("Starting training with simplified model...")
    total_updates = 0
    eval_metrics = []
    best_accuracy = 0.0
    
    for epoch in range(config['num_training_epochs']):
        epoch_start_time = time.time()
        epoch_loss = 0
        epoch_reward = 0
        epoch_entropy = 0
        epoch_updates = 0
        
        dataloader_iter = iter(dataloader)
        num_iter_per_data = 3  # Reduced iterations per data
        total_iter = number_of_batches * num_iter_per_data
        final_total_num_iteration = min(config['max_updates_per_epoch'], total_iter)
        
        logger.info(f"Epoch: {epoch+1}/{config['num_training_epochs']}, "
                   f"Total iterations: {final_total_num_iteration}")
        
        for batch_iter_idx in range(final_total_num_iteration):
            try:
                # Get next batch
                batch = next(dataloader_iter)
            except StopIteration:
                # Restart iterator if we've gone through all batches
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)
            
            # Process batch data
            pod_features = batch['pod_features'].to(device)
            kv_hit_ratios = batch['kv_hit_ratios'].to(device)
            request_features = batch['request_features'].to(device)
            actions = batch['action'].to(device)
            rewards = batch['reward'].to(device).unsqueeze(1)
            
            # Store all data of this batch in agent memory
            for j in range(len(rewards)):
                agent.remember(
                    pod_features[j:j+1], 
                    kv_hit_ratios[j:j+1], 
                    request_features[j:j+1], 
                    actions[j:j+1], 
                    rewards[j:j+1]
                )
            
            # Trigger learning every 3rd batch iteration
            trigger_learning = (batch_iter_idx+1) % 3 == 0 or batch_iter_idx == final_total_num_iteration - 1
            if trigger_learning:
                if len(agent.pod_features) > 0:  # Only learn if we have collected experiences
                    try:
                        update_metrics = agent.learn()
                        total_updates += 1
                        epoch_updates += 1
                        epoch_loss += update_metrics['loss']
                        epoch_reward += update_metrics['reward']
                        epoch_entropy += update_metrics['entropy']
                        
                        # Log progress less frequently
                        if batch_iter_idx % max(1, final_total_num_iteration//3) == 0:
                            logger.info(f"Batch: {batch_iter_idx+1}/{final_total_num_iteration}, "
                                       f"Loss: {update_metrics['loss']:.4f}, "
                                       f"Reward: {update_metrics['reward']:.4f}, "
                                       f"Entropy: {update_metrics['entropy']:.4f}")
                    except Exception as e:
                        logger.error(f"Error during learning: {e}")
            
            # Evaluate the agent periodically
            if (batch_iter_idx + 1) % max(1, final_total_num_iteration // config['eval_interval']) == 0 or batch_iter_idx == final_total_num_iteration - 1:
                logger.info(f"Evaluating agent at batch {batch_iter_idx+1}/{final_total_num_iteration}")
                
                try:
                    # Create a validation subset for evaluation
                    eval_indices = torch.randperm(len(dataset))[:min(200, len(dataset))]
                    eval_data = {
                        'pod_features_with_staleness': combined_data['pod_features_with_staleness'][eval_indices],
                        'kv_hit_ratios': combined_data['kv_hit_ratios'][eval_indices],
                        'request_features': combined_data['request_features'][eval_indices],
                        'actions': combined_data['actions'][eval_indices],
                        'rewards': combined_data['rewards'][eval_indices]
                    }
                    
                    metrics = evaluate_agent(agent, eval_data)
                    eval_metrics.append(metrics)
                    
                    logger.info(f"Evaluation metrics - Accuracy: {metrics['accuracy']:.4f}, "
                               f"Confidence: {metrics['avg_confidence']:.4f}, "
                               f"True Reward: {metrics['true_reward']:.4f}")
                    
                    # Save best model
                    if metrics['accuracy'] > best_accuracy:
                        best_accuracy = metrics['accuracy']
                        best_model_dir = os.path.join(output_dir, "best_model")
                        agent.save(best_model_dir)
                        logger.info(f"New best accuracy: {best_accuracy:.4f}, saved to {best_model_dir}")
                    
                except Exception as e:
                    logger.error(f"Error during evaluation: {e}")
        
        # End of epoch
        epoch_duration = time.time() - epoch_start_time
        
        if epoch_updates > 0:
            avg_loss = epoch_loss / epoch_updates
            avg_reward = epoch_reward / epoch_updates
            avg_entropy = epoch_entropy / epoch_updates
            
            logger.info(f"Epoch {epoch+1}/{config['num_training_epochs']} completed in {epoch_duration:.2f}s - "
                       f"Avg Loss: {avg_loss:.4f}, "
                       f"Avg Reward: {avg_reward:.4f}, "
                       f"Avg Entropy: {avg_entropy:.4f}")
        else:
            logger.warning(f"Epoch {epoch+1}/{config['num_training_epochs']} completed with no updates")
            
        # Early stopping if performance is very poor
        if len(eval_metrics) > 10:
            recent_accuracies = [m['accuracy'] for m in eval_metrics[-5:]]
            if all(acc < 0.2 for acc in recent_accuracies):
                logger.warning("Performance consistently poor, consider checking data quality")
    
    # End of training
    logger.info(f"Training completed with {total_updates} total updates")
    logger.info(f"Best accuracy achieved: {best_accuracy:.4f}")
    
    # Save final model
    agent.save(output_dir)
    logger.info(f"Saved final model to {output_dir}")
    
    # Plot training metrics
    try:
        plot_training_metrics(agent, eval_metrics, output_dir, combined_data)
    except Exception as e:
        logger.error(f"Error plotting training metrics: {e}")
        # Fallback to basic plotting
        try:
            plot_training_metrics(agent, eval_metrics, output_dir, None)
        except:
            logger.error("All plotting failed")
    
    # Print final summary
    if eval_metrics:
        final_accuracy = eval_metrics[-1]['accuracy']
        final_confidence = eval_metrics[-1]['avg_confidence']
        random_baseline = 1.0 / action_dim
        
        logger.info("=" * 60)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Dataset size: {len(dataset)} samples")
        logger.info(f"Model parameters: ~{sum(p.numel() for p in agent.policy.parameters()):,}")
        logger.info(f"Random baseline: {random_baseline:.3f} ({random_baseline*100:.1f}%)")
        logger.info(f"Final accuracy: {final_accuracy:.3f} ({final_accuracy*100:.1f}%)")
        logger.info(f"Best accuracy: {best_accuracy:.3f} ({best_accuracy*100:.1f}%)")
        logger.info(f"Final confidence: {final_confidence:.3f} ({final_confidence*100:.1f}%)")
        
        if final_accuracy > random_baseline * 1.5:
            logger.info("✅ Model is learning! Performance significantly above random.")
        elif final_accuracy > random_baseline * 1.1:
            logger.info("⚠️  Model shows modest learning, but room for improvement.")
        else:
            logger.warning("❌ Model performance is close to random. Check data quality.")
    
    return {
        'agent': agent,
        'model_dir': output_dir,
        'output_dir': output_dir,
        'config': config,
        'eval_metrics': eval_metrics,
        'best_accuracy': best_accuracy
    }


# Global cache for agent instance (for inference)
_cached_agent = None
_cached_agent_config = None
_cached_metadata = None


def infer_from_tensor(tensor_data, exploration_enabled=False, exploration_rate=0.1, model_updated=False):
    """
    Inference function optimized for the simplified model
    """
    global final_model_path, _cached_metadata, _cached_agent, _cached_agent_config
    
    infer_start_time = time.time()
    
    # Load feature metadata (cached)
    if _cached_metadata is None:
        logger.info("Loading feature metadata into cache...")
        _cached_metadata = {
            'metadata': None,
            'pod_features_list': None,
            'feature_indices_map': None
        }
        
        try:
            if os.path.exists("metadata.json"):
                with open("metadata.json", 'r') as f:
                    _cached_metadata['metadata'] = json.load(f)
            if os.path.exists("pod_features_list.pkl"):
                with open("pod_features_list.pkl", 'rb') as f:
                    _cached_metadata['pod_features_list'] = pickle.load(f)
            if os.path.exists("feature_indices_map.pkl"):
                with open("feature_indices_map.pkl", 'rb') as f:
                    _cached_metadata['feature_indices_map'] = pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading feature metadata: {e}")

    # Extract data from tensor dataset and move to device
    try:
        pod_features = tensor_data['pod_features_with_staleness'].to(device)
        kv_hit_ratios = tensor_data['kv_hit_ratios'].to(device)
        request_features = tensor_data['request_features'].to(device)
    except KeyError as e:
        logger.error(f"Missing key in tensor data: {e}")
        raise ValueError(f"Missing key in tensor data: {e}")
    
    # Ensure data is in batch format (add batch dimension if needed)
    if len(pod_features.shape) == 2:
        pod_features = pod_features.unsqueeze(0)
    if len(kv_hit_ratios.shape) == 2:
        kv_hit_ratios = kv_hit_ratios.unsqueeze(0)
    if len(request_features.shape) == 1:
        request_features = request_features.unsqueeze(0)

    # Cache agent instance
    current_config = {
        'pod_features': pod_features.shape[2],
        'kv_hit_ratios': kv_hit_ratios.shape[2], 
        'request_features': request_features.shape[1],
        'num_pods': pod_features.shape[1],
        'exploration_rate': exploration_rate,
        'final_model_path': final_model_path
    }
    
    # Check if we can reuse cached agent
    agent_cache_hit = False
    if (_cached_agent is not None and 
        _cached_agent_config is not None and
        _cached_agent_config == current_config):
        agent = _cached_agent
        agent_cache_hit = True
        logger.debug("Agent cache hit - reusing cached agent")
    else:
        # Create new agent
        logger.info("Creating new simplified agent for inference")
        
        state_dim = {
            'pod_features': current_config['pod_features'],
            'kv_hit_ratios': current_config['kv_hit_ratios'],
            'request_features': current_config['request_features'],
            'num_pods': current_config['num_pods']
        }
        
        action_dim = current_config['num_pods']
        
        agent = SimplifiedContextualBandit(
            state_dim=state_dim,
            action_dim=action_dim,
            exploration_rate=exploration_rate
        )
        
        _cached_agent = agent
        _cached_agent_config = current_config.copy()

    # Load model weights if needed
    if not agent_cache_hit or model_updated:
        agent.load(final_model_path)
        agent.policy.eval()
        logger.info("Loaded model weights from disk")

    # Run inference
    agent.policy.eval()
    with torch.no_grad():
        # Get action probabilities
        action_probs = agent.policy(pod_features, kv_hit_ratios, request_features)
        
        if exploration_enabled:
            # Use exploration strategy
            action, _ = agent.policy.get_action(
                pod_features, kv_hit_ratios, request_features, 
                explore=True, 
                epsilon=exploration_rate
            )
            selected_action = action.item()
            confidence = action_probs[0, selected_action].item()
        else:
            # Pure exploitation
            selected_action = torch.argmax(action_probs, dim=1).item()
            confidence = action_probs[0, selected_action].item()

    total_inference_time = time.time() - infer_start_time
    
    # Return inference results
    results = {
        'selected_pod_index': selected_action,
        'confidence': confidence,
        'pod_probabilities': action_probs[0].cpu().numpy().tolist(),
        'final_model_path': final_model_path,
        'exploration_enabled': exploration_enabled,
        'model_type': 'simplified'
    }
    
    timing_info = {
        'total_inference_time_ms': total_inference_time * 1000,
        'agent_cache_hit': agent_cache_hit,
        'model_updated': model_updated
    }
    
    return results, timing_info


def print_tensor_data_summary(tensor_data):
    """Print a comprehensive summary of tensor_data for debugging"""
    logger.info("=" * 60)
    logger.info("TENSOR_DATA SUMMARY")
    logger.info("=" * 60)
    
    logger.info(f"Total number of keys: {len(tensor_data)}")
    logger.info(f"Keys: {list(tensor_data.keys())}")
    
    for key, tensor in tensor_data.items():
        if hasattr(tensor, 'shape'):
            logger.info(f"\n{key}:")
            logger.info(f"  Shape: {tensor.shape}")
            logger.info(f"  Dtype: {tensor.dtype}")
            logger.info(f"  Min: {tensor.min().item() if tensor.numel() > 0 else 'Empty'}")
            logger.info(f"  Max: {tensor.max().item() if tensor.numel() > 0 else 'Empty'}")
            logger.info(f"  Mean: {tensor.float().mean().item() if tensor.numel() > 0 else 'Empty'}")
        else:
            logger.info(f"\n{key}: {type(tensor)} - {tensor}")
    
    logger.info("=" * 60)


if __name__ == "__main__":
    # Example usage
    if len(sys.argv) > 1:
        encoded_data_dir = sys.argv[1]
        logger.info(f"Starting simplified contextual bandit training with data from: {encoded_data_dir}")
        results = train(encoded_data_dir)
        logger.info("Training completed successfully!")
    else:
        logger.info("Usage: python simplified_contextual_bandit.py <encoded_data_dir>")
        logger.info("Example: python simplified_contextual_bandit.py encoded_data/")