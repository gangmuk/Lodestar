#!/usr/bin/env python3

# simplified_contextual_bandit.py

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Categorical
import pickle
import time
import matplotlib.pyplot as plt
import glob
import random
from logger import logger, INCLUDE_GPU_IN_FEATURE
# INCLUDE_GPU_IN_FEATURE = True

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)  # Python's random module
torch.cuda.manual_seed_all(seed)  # For CUDA operations
os.environ['PYTHONHASHSEED'] = str(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
final_model_dir = "final_model"

class FixedPolicyNetwork(nn.Module):
    """
    Fixed architecture that preserves pod structure
    """
    def __init__(self, state_dim, action_dim, hidden_dim, weight_initialization):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Calculate feature sizes
        pod_feature_size = state_dim['pod_features']  # Features per pod: pod feature + kv hit feature
        kv_feature_size = state_dim['kv_hit_ratios']  # KV features per pod (e.g., 1)
        request_feature_size = state_dim['request_features']  # Global request features (e.g., 3)
        
        # Per-pod feature size
        per_pod_features = pod_feature_size + kv_feature_size
        logger.info(f"pod_feature_size: {pod_feature_size}, kv_feature_size: {kv_feature_size}, request_feature_size: {request_feature_size}")
        logger.info(f"Per-pod features: {per_pod_features}")
        # APPROACH 1: Pod-aware scoring network
        # For each pod, combine pod features + request features → score
        combined_input_size = per_pod_features + request_feature_size
        
        self.pod_scorer = nn.Sequential(
            nn.Linear(combined_input_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)  # Single score per pod
        )
        
        logger.info(f"Fixed architecture:")
        logger.info(f"  Per-pod features: {per_pod_features}")
        logger.info(f"  Request features: {request_feature_size}")
        logger.info(f"  Combined input per pod: {combined_input_size}")
        logger.info(f"  Pod scorer outputs 1 score per pod")

        # Initialize weights. This is not necessary though.
        if weight_initialization == 'static':
            self._static_weight_initialization()
        elif weight_initialization == 'kaiming':
            self._kaiming_initialize_weights()
        elif weight_initialization == 'xavier':
            self._xavier_initialize_weights()

    def _static_weight_initialization(self):
        """Simple static weight initialization - just set a fixed seed before initialization"""
        
        # Save current random state
        current_torch_state = torch.get_rng_state()
        current_numpy_state = np.random.get_state()
        
        # Set fixed seed for weight initialization
        torch.manual_seed(12345)
        np.random.seed(12345)
        
        logger.info("🔧 Initializing weights with fixed seed 12345")
        
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Get layer position info
                is_output_layer = (module.out_features == 1)
                is_first_layer = ('0' in name)  # First layer in sequential
                
                if is_output_layer:
                    # Output layer: smaller weights for stability
                    torch.nn.init.xavier_uniform_(module.weight, gain=0.1)
                    torch.nn.init.constant_(module.bias, 0.0)
                elif is_first_layer:
                    # First layer: slightly smaller to prevent saturation
                    torch.nn.init.kaiming_uniform_(module.weight, 
                                                mode='fan_in', 
                                                nonlinearity='relu')
                    torch.nn.init.constant_(module.bias, 0.01)
                else:
                    # Hidden layers: standard He initialization
                    torch.nn.init.kaiming_uniform_(module.weight, 
                                                mode='fan_in', 
                                                nonlinearity='relu')
                    torch.nn.init.constant_(module.bias, 0.01)
        
        # Restore original random state (so training randomness is not affected)
        torch.set_rng_state(current_torch_state)
        np.random.set_state(current_numpy_state)
        
        logger.info("✅ Weight initialization complete, random state restored")
    
    def _kaiming_initialize_weights(self):
        """initialization with kaiming_uniform_, layer-specific strategies"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Get layer position info
                is_output_layer = (module.out_features == 1)
                is_first_layer = ('0' in name)  # First layer in sequential
                
                if is_output_layer:
                    # Output layer: smaller weights for stability
                    torch.nn.init.xavier_uniform_(module.weight, gain=0.1)
                    torch.nn.init.constant_(module.bias, 0.0)
                elif is_first_layer:
                    # First layer: slightly smaller to prevent saturation
                    torch.nn.init.kaiming_uniform_(module.weight, 
                                                mode='fan_in', 
                                                nonlinearity='relu')
                    torch.nn.init.constant_(module.bias, 0.01)
                else:
                    # Hidden layers: standard He initialization
                    torch.nn.init.kaiming_uniform_(module.weight, 
                                                mode='fan_in', 
                                                nonlinearity='relu')
                    torch.nn.init.constant_(module.bias, 0.01)

    def _xavier_initialize_weights(self):
        """Xavier/Glorot initialization with layer-specific strategies"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Get layer position info
                is_output_layer = (module.out_features == 1)
                is_first_layer = ('0' in name)  # First layer in sequential
                
                if is_output_layer:
                    # Output layer: smaller weights for stability
                    torch.nn.init.xavier_uniform_(module.weight, gain=0.1)
                    torch.nn.init.constant_(module.bias, 0.0)
                elif is_first_layer:
                    # First layer: slightly smaller to prevent saturation
                    torch.nn.init.xavier_uniform_(module.weight, gain=0.8)
                    torch.nn.init.constant_(module.bias, 0.01)
                else:
                    # Hidden layers: standard Xavier initialization
                    torch.nn.init.xavier_uniform_(module.weight, gain=1.0)
                    torch.nn.init.constant_(module.bias, 0.01)

    def forward(self, pod_features, kv_hit_ratios, request_features, return_attention=False):
        batch_size = pod_features.shape[0]
        num_pods = pod_features.shape[1]
        
        # Combine pod features and kv ratios for each pod
        # pod_features: [batch, num_pods, pod_feature_dim]
        # kv_hit_ratios: [batch, num_pods, kv_dim]
        combined_pod_features = torch.cat([pod_features, kv_hit_ratios], dim=2)
        # combined_pod_features: [batch, num_pods, pod_feature_dim + kv_dim]
        
        # Expand request features to match each pod
        # request_features: [batch, request_dim] → [batch, num_pods, request_dim]
        expanded_request = request_features.unsqueeze(1).expand(-1, num_pods, -1)
        
        # Combine pod-specific features with request context
        # full_features: [batch, num_pods, pod_features + kv_features + request_features]
        full_features = torch.cat([combined_pod_features, expanded_request], dim=2)
        
        # Reshape to process all pods in batch
        # [batch * num_pods, combined_feature_size]
        reshaped_features = full_features.view(batch_size * num_pods, -1)
        
        # Score each pod given its features + request context
        # print(f"type(reshaped_features): {type(reshaped_features)}, reshaped_features.shape: {reshaped_features.shape}")
        # print(f"reshaped_features: {reshaped_features}")
        # exit()
        pod_scores = self.pod_scorer(reshaped_features)  # [batch * num_pods, 1]
        
        # Reshape back to [batch, num_pods]
        pod_scores = pod_scores.view(batch_size, num_pods)
        
        # Convert to probabilities
        action_probs = F.softmax(pod_scores, dim=1)
        
        if return_attention:
            # No attention in this model, return dummy
            dummy_attention = torch.ones(batch_size, num_pods, num_pods, device=action_probs.device) / num_pods
            return action_probs, dummy_attention
        
        return action_probs
    
    def get_action(self, pod_features, kv_hit_ratios, request_features, explore, epsilon):
        """Get action with exploration"""
        action_probs = self.forward(pod_features, kv_hit_ratios, request_features)
        
        if not explore:
            return torch.argmax(action_probs, dim=1)
        
        # Epsilon-greedy exploration
        batch_size = pod_features.shape[0]
        device = action_probs.device
        
        random_actions = torch.randint(0, action_probs.shape[1], (batch_size,), device=device)
        greedy_actions = torch.argmax(action_probs, dim=1)
        explore_mask = (torch.rand(batch_size, device=device) < epsilon).long()
        actions = (1 - explore_mask) * greedy_actions + explore_mask * random_actions
        log_probs = torch.log(torch.gather(action_probs, 1, actions.unsqueeze(1)).squeeze(1) + 1e-10)
        
        logger.debug(f"explore_mask: {explore_mask.cpu().numpy()}")
        logger.debug(f"greedy_actions: {greedy_actions.cpu().numpy()}")
        logger.debug(f"random_actions: {random_actions.cpu().numpy()}")
        logger.debug(f"actions: {actions.cpu().numpy()}")
        logger.debug(f"action_probs: {action_probs.cpu().numpy()}")
        logger.debug(f"log_probs: {log_probs.cpu().numpy()}")
        
        return actions, log_probs


class SimplifiedContextualBandit:
    def __init__(self, state_dim, action_dim, HYPERPARAMETERS):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hyperparameters = HYPERPARAMETERS
        self.current_epoch = 0
        self.global_batch_counter = 0
        self.learn_call_counter = 0
        
        # Initialize simplified policy network
        # self.policy = SimplePolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.policy = FixedPolicyNetwork(state_dim, action_dim,  self.hyperparameters['hidden_dim'],  self.hyperparameters['weight_initialization']).to(device)

        if HYPERPARAMETERS.get('deterministic_training', False):
            optim_seed = HYPERPARAMETERS['training_seed'] + 1000  # Different from model seed
            torch.manual_seed(optim_seed)
            logger.info(f"🔧 Creating optimizer with deterministic seed: {optim_seed}")
        

        # Optimizer with weight decay for regularization
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr= self.hyperparameters['learning_rate'], weight_decay= self.hyperparameters['weight_decay'])
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        
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
                    explore=self.hyperparameters['explore'], epsilon= self.hyperparameters['exploration_rate']
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

        self.learn_call_counter += 1
        
        if self.hyperparameters['deterministic_training']:
            batch_seed = self.hyperparameters['training_seed'] + self.learn_call_counter
            torch.manual_seed(batch_seed)
            np.random.seed(batch_seed)
            
        # Stack all tensors
        pod_features = torch.cat(self.pod_features, dim=0)
        kv_hit_ratios = torch.cat(self.kv_hit_ratios, dim=0)
        request_features = torch.cat(self.request_features, dim=0)
        actions = torch.cat(self.actions, dim=0)
        rewards = torch.cat(self.rewards, dim=0).view(-1, 1)
        if  self.hyperparameters['per_learn_reward_normalization']:
            # Reward Normalization!
            # NOTE: There is another normalization in train() function. Choose only one.
            # Improved reward normalization with clipping
            if rewards.std() > 1e-6:
                normalized_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
                # Clip extreme values to prevent instability
                normalized_rewards = torch.clamp(normalized_rewards, -3.0, 3.0)
                logger.info(f"NOTE: Rewards normalized: mean={normalized_rewards.mean():.3f}, std={normalized_rewards.std():.3f}")
            else:
                # If rewards have no variance, use them as-is
                normalized_rewards = rewards
        else:
            normalized_rewards = rewards
        
        # Create batches
        n_samples = len(self.pod_features)
        batch_size = min( self.hyperparameters['batch_size'], n_samples)
        batch_start = np.arange(0, n_samples,  self.hyperparameters['batch_size'])
        indices = np.arange(n_samples, dtype=np.int64)

        # MODIFY THIS: Make batch shuffling deterministic
        if self.hyperparameters.get('deterministic_training', False):
            # Don't shuffle for deterministic training, or use deterministic shuffle
            # Option 1: No shuffling (most deterministic)
            pass  # Keep indices in order
            
            # Option 2: Deterministic shuffling (if you want some randomness but reproducible)
            # shuffle_seed = self.hyperparameters.get('training_seed', 54321) + self.learn_call_counter
            # np.random.seed(shuffle_seed)
            # np.random.shuffle(indices)
        else:
            np.random.shuffle(indices)  # Original random shuffling

        batches = [indices[i:i +  self.hyperparameters['batch_size']] for i in batch_start]
        
        # if len(batches) > 1:
        #     batches = batches[:1]  # Only process first batch
        #     logger.info(f"Limiting to 1 batch update (was {len(batches)})")
            
        logger.debug(f"Learn call #{self.learn_call_counter} (Epoch {self.current_epoch}): Processing {n_samples} samples in {len(batches)} batches")
        
        epoch_loss = 0
        epoch_entropy = 0
        num_updates = 0
        
        logger.info(f"="*60)
        logger.info(f"LEARN CALL #{self.learn_call_counter}: {n_samples} samples → {len(batches)} batches")
        
        # ADD THIS: Track parameter changes
        initial_param_snapshot = {name: param.clone().detach() for name, param in self.policy.named_parameters()}
        
        # Process each batch
        for local_batch_idx, batch_indices in enumerate(batches):
            # Get batch data
            batch_pod_features = pod_features[batch_indices]
            batch_kv_hit_ratios = kv_hit_ratios[batch_indices]
            batch_request_features = request_features[batch_indices]
            batch_actions = actions[batch_indices]
            batch_rewards = normalized_rewards[batch_indices]
            # Per-batch Reward Normalization!
            # # NOTE: per-batch reward normalization. There is another reward normalization in learn function. Choose only one of them.
            # if batch_rewards.std() > 1e-6:
            #     batch_rewards = (batch_rewards - batch_rewards.mean()) / (batch_rewards.std() + 1e-8)
            self.global_batch_counter += 1
            logger.debug(f"Epoch {self.current_epoch}, Learn call {self.learn_call_counter}, "
                        f"Local batch {local_batch_idx}, Global batch {self.global_batch_counter}: "
                        f"rewards range [{batch_rewards.min():.3f}, {batch_rewards.max():.3f}]")
            logger.debug(f"Epoch {self.current_epoch}, Global batch {self.global_batch_counter}: "
                        f"actions {batch_actions.cpu().numpy()}")
            logger.debug(f"Epoch {self.current_epoch}, Global batch {self.global_batch_counter}: "
                        f"rewards {batch_rewards.squeeze().cpu().numpy()}")
            

            # Get current policy distributions
            action_probs = self.policy(batch_pod_features, batch_kv_hit_ratios, batch_request_features)
            
            # # ADD THIS:
            # pred_actions = torch.argmax(action_probs, dim=1)
            # logger.info(f"Batch {local_batch_idx}: model predictions {pred_actions[:10].cpu().numpy()}")
            # logger.info(f"Batch {local_batch_idx}: actual actions    {batch_actions[:10].cpu().numpy()}")

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
            
            ## Higher entropy means action probabilites are more uniform, and model is uncertain about its actions.
            ## entropy_bonus_factor encourages exploration and prevent the policy from converging too early to a suboptimal solution.
            # entropy_bonus = 0.01 * entropy
            entropy_bonus =  self.hyperparameters['entropy_bonus_factor'] * entropy  # 10x stronger!
            total_loss = loss - entropy_bonus

            logger.info(f"[Epoch {self.current_epoch:2d}] [Global Batch Idx {self.global_batch_counter:4d}] "
                   f"[Learn #{self.learn_call_counter:2d}.{local_batch_idx}] "
                   f"Loss={loss.item():.4f}, Entropy={entropy.item():.4f}, Total={total_loss.item():.4f}")

            # Update policy
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            

            max_param_change = 0
            for name, param in self.policy.named_parameters():
                param_change = (param - initial_param_snapshot[name]).abs().max().item()
                max_param_change = max(max_param_change, param_change)
            logger.info(f"    → Max param change: {max_param_change:.6f}")
            for name, param in self.policy.named_parameters():
                initial_param_snapshot[name] = param.clone().detach()


            # Track metrics
            epoch_loss += total_loss.item()
            epoch_entropy += entropy.item()
            num_updates += 1

        logger.info(f"="*60)

        # Update learning rate based on loss
        avg_loss = epoch_loss / max(1, num_updates)
        self.scheduler.step(avg_loss)
        
        # Clear memory.
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

    def save(self, final_model_dir):
        """Save the agent's parameters to the specified directory"""
        os.makedirs(final_model_dir, exist_ok=True)
        logger.info(f"Creating final_model_dir: {final_model_dir}")
        
        # Save policy network
        torch.save(self.policy.state_dict(), os.path.join(final_model_dir, 'policy.pth'))
        
        # Save optimizer state
        torch.save(self.optimizer.state_dict(), os.path.join(final_model_dir, 'optimizer.pth'))
        
        # Save training history
        history = {
            'loss': self.loss_history,
            'reward': self.reward_history,
            'entropy': self.entropy_history
        }
        
        with open(os.path.join(final_model_dir, 'history.pkl'), 'wb') as f:
            pickle.dump(history, f)
            
        # Copy to final model path
        os.makedirs(final_model_dir, exist_ok=True)
        logger.info(f"Saved simplified agent to {final_model_dir}")
    
    def load(self, directory):
        """Load the agent's parameters from the specified directory"""
        # Load policy network
        policy_path = os.path.join(directory, 'policy.pth')
        if os.path.exists(policy_path):
            logger.info(f"🔍 DIMENSION CHECK: Loading model from {policy_path}")
            
            # Load the state dict
            saved_state_dict = torch.load(policy_path, map_location=device)
            
            # Check dimensions of the first layer (most likely to have mismatch)
            current_model_dict = self.policy.state_dict()
            
            logger.info("🔍 COMPARING MODEL DIMENSIONS:")
            for key in saved_state_dict.keys():
                if key in current_model_dict:
                    saved_shape = saved_state_dict[key].shape
                    current_shape = current_model_dict[key].shape
                    logger.info(f"  Layer {key}:")
                    logger.info(f"    Saved model:   {saved_shape}")
                    logger.info(f"    Current model: {current_shape}")
                    if saved_shape != current_shape:
                        logger.error(f"❌ Model architecture DIMENSION MISMATCH in layer {key}!")
                        logger.error(f"   Saved: {saved_shape}, Current: {current_shape}")
                        logger.error(f"   This will cause silent failures and wrong behavior!")
                        assert False
            
            # If we get here, dimensions match
            logger.info("✅ All layer dimensions match - safe to load")
            self.policy.load_state_dict(saved_state_dict)
        else:
            logger.error(f"No policy file found at {policy_path}")
            assert False
            
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
        reward_distribution = {}
        reward_distribution['mean'] = np.mean(reward_values)
        reward_distribution['std'] = np.std(reward_values)
        reward_distribution['min'] = np.min(reward_values)
        reward_distribution['p10'] = np.percentile(reward_values, 10)
        reward_distribution['p20'] = np.percentile(reward_values, 20)
        reward_distribution['p30'] = np.percentile(reward_values, 30)
        reward_distribution['p40'] = np.percentile(reward_values, 40)
        reward_distribution['p50'] = np.percentile(reward_values, 50)
        reward_distribution['p60'] = np.percentile(reward_values, 60)
        reward_distribution['p70'] = np.percentile(reward_values, 70)
        reward_distribution['p80'] = np.percentile(reward_values, 80)
        reward_distribution['p90'] = np.percentile(reward_values, 90)
        reward_distribution['p99'] = np.percentile(reward_values, 99)
        reward_distribution['max'] = np.max(reward_values)
        logger.info(f"\nReward signal strength:")
        logger.info(f"  Reward gap: {reward_gap:.4f}")
        
        if reward_gap < 0.01:
            logger.warning(f"  ⚠️  VERY WEAK REWARD SIGNAL! (gap: {reward_gap:.4f})")
    
    return {
        'total_samples': total_samples,
        'imbalance_ratio': imbalance_ratio,
        'reward_gap': reward_gap if 'reward_gap' in locals() else 0,
        'reward_distribution': reward_distribution,
        'action_distribution': action_counts,
    }

def analyze_reward_signal_strength(combined_data, agent=None):
    """
    Systematic analysis of whether reward range is sufficient for learning
    """
    logger.info("🔍 REWARD SIGNAL STRENGTH ANALYSIS")
    logger.info("=" * 60)
    
    actions = combined_data['actions']
    rewards = combined_data['rewards']
    num_pods = len(torch.unique(actions))
    
    # 1. BASIC REWARD STATISTICS
    logger.info("1️⃣ BASIC REWARD STATISTICS:")
    logger.info(f"  Range: [{rewards.min():.4f}, {rewards.max():.4f}]")
    logger.info(f"  Spread: {rewards.max() - rewards.min():.4f}")
    logger.info(f"  Mean: {rewards.mean():.4f}")
    logger.info(f"  Std: {rewards.std():.4f}")
    
    # 2. REWARD BY ACTION ANALYSIS
    logger.info("\n2️⃣ REWARD BY ACTION:")
    reward_stats = {}
    for pod_idx in range(num_pods):
        mask = actions == pod_idx
        if mask.sum() > 0:
            pod_rewards = rewards[mask]
            reward_stats[pod_idx] = {
                'mean': pod_rewards.mean().item(),
                'std': pod_rewards.std().item(),
                'count': mask.sum().item(),
                'min': pod_rewards.min().item(),
                'max': pod_rewards.max().item()
            }
            logger.info(f"  Pod {pod_idx}: μ={reward_stats[pod_idx]['mean']:.4f}, "
                       f"σ={reward_stats[pod_idx]['std']:.4f}, n={reward_stats[pod_idx]['count']}")
    
    # 3. SIGNAL STRENGTH METRICS
    logger.info("\n3️⃣ SIGNAL STRENGTH ANALYSIS:")
    
    # 3a. Between-action variance vs within-action variance
    reward_means = [stats['mean'] for stats in reward_stats.values()]
    reward_stds = [stats['std'] for stats in reward_stats.values()]
    
    between_action_var = np.var(reward_means)
    avg_within_action_var = np.mean([std**2 for std in reward_stds])
    
    signal_to_noise = between_action_var / (avg_within_action_var + 1e-8)
    
    logger.info(f"  Between-action variance: {between_action_var:.6f}")
    logger.info(f"  Average within-action variance: {avg_within_action_var:.6f}")
    logger.info(f"  Signal-to-Noise Ratio: {signal_to_noise:.4f}")
    
    # Signal strength interpretation
    if signal_to_noise > 1.0:
        logger.info("  ✅ STRONG SIGNAL: Differences between actions > noise within actions")
    elif signal_to_noise > 0.1:
        logger.info("  ⚠️  MODERATE SIGNAL: Some signal present but noisy")
    else:
        logger.info("  ❌ WEAK SIGNAL: Noise dominates, hard to distinguish actions")
    
    # 3b. Effect Size (Cohen's d) between best and worst actions
    best_pod = max(reward_stats.keys(), key=lambda k: reward_stats[k]['mean'])
    worst_pod = min(reward_stats.keys(), key=lambda k: reward_stats[k]['mean'])
    
    best_mean = reward_stats[best_pod]['mean']
    worst_mean = reward_stats[worst_pod]['mean']
    pooled_std = np.sqrt((reward_stats[best_pod]['std']**2 + reward_stats[worst_pod]['std']**2) / 2)
    
    cohens_d = (best_mean - worst_mean) / (pooled_std + 1e-8)
    
    logger.info(f"\n  Effect Size Analysis (Best vs Worst):")
    logger.info(f"    Best pod {best_pod}: {best_mean:.4f}")
    logger.info(f"    Worst pod {worst_pod}: {worst_mean:.4f}")
    logger.info(f"    Cohen's d: {cohens_d:.4f}")
    
    # Effect size interpretation
    if abs(cohens_d) > 0.8:
        logger.info("    ✅ LARGE EFFECT: Strong difference between best/worst")
    elif abs(cohens_d) > 0.5:
        logger.info("    📊 MEDIUM EFFECT: Moderate difference")
    elif abs(cohens_d) > 0.2:
        logger.info("    ⚠️  SMALL EFFECT: Weak difference")
    else:
        logger.info("    ❌ NEGLIGIBLE EFFECT: Almost no difference")
    
    # 4. LEARNING DIFFICULTY ESTIMATION
    logger.info("\n4️⃣ LEARNING DIFFICULTY ESTIMATION:")
    
    # 4a. Minimum detectable difference
    reward_gap = max(reward_means) - min(reward_means)
    random_baseline_prob = 1.0 / num_pods
    uniform_entropy = -np.log(random_baseline_prob)
    
    logger.info(f"  Reward gap: {reward_gap:.4f}")
    logger.info(f"  Random baseline entropy: {uniform_entropy:.4f}")
    logger.info(f"  Reward gap / entropy: {reward_gap / uniform_entropy:.4f}")
    
    # Rule of thumb: need reward_gap > entropy for clear learning
    if reward_gap > uniform_entropy:
        logger.info("  ✅ SUFFICIENT: Reward gap > baseline entropy")
    elif reward_gap > uniform_entropy * 0.5:
        logger.info("  ⚠️  BORDERLINE: Reward gap ~0.5x entropy")
    else:
        logger.info("  ❌ INSUFFICIENT: Reward gap < 0.5x entropy")
    
    # 4b. Relative reward differences
    logger.info(f"\n  Relative Analysis:")
    for i, (pod_i, stats_i) in enumerate(reward_stats.items()):
        for j, (pod_j, stats_j) in enumerate(reward_stats.items()):
            if i < j:  # Only compare each pair once
                diff = abs(stats_i['mean'] - stats_j['mean'])
                pooled_std = np.sqrt((stats_i['std']**2 + stats_j['std']**2) / 2)
                relative_diff = diff / (pooled_std + 1e-8)
                
                if relative_diff > 2.0:
                    significance = "✅ VERY CLEAR"
                elif relative_diff > 1.0:
                    significance = "📊 CLEAR"
                elif relative_diff > 0.5:
                    significance = "⚠️  MARGINAL"
                else:
                    significance = "❌ UNCLEAR"
                
                logger.info(f"    Pod {pod_i} vs {pod_j}: diff={diff:.4f}, "
                           f"relative={relative_diff:.2f}, {significance}")
    
    # 5. EXPLORATION IMPACT ANALYSIS
    logger.info("\n5️⃣ EXPLORATION IMPACT:")
    
    if agent is not None:
        exploration_rate = getattr(agent, 'exploration_rate', 0.3)
        
        # Calculate how much exploration washes out the signal
        exploration_noise = exploration_rate * uniform_entropy
        signal_after_exploration = reward_gap - exploration_noise
        
        logger.info(f"  Exploration rate: {exploration_rate:.1%}")
        logger.info(f"  Exploration noise: {exploration_noise:.4f}")
        logger.info(f"  Effective signal: {signal_after_exploration:.4f}")
        
        if signal_after_exploration > 0:
            logger.info("  ✅ Signal survives exploration noise")
        else:
            logger.info("  ❌ Exploration noise dominates signal!")
            logger.info(f"  💡 Reduce exploration to < {reward_gap/uniform_entropy:.1%}")
    
    # 6. RECOMMENDATIONS
    logger.info("\n6️⃣ RECOMMENDATIONS:")
    
    recommendations = []
    
    if signal_to_noise < 0.1:
        recommendations.append("🔴 CRITICAL: Amplify reward differences by 5-10x")
    elif signal_to_noise < 1.0:
        recommendations.append("🟡 Amplify reward differences by 2-3x")
    
    if abs(cohens_d) < 0.2:
        recommendations.append("🔴 CRITICAL: Reward differences too small to learn")
    elif abs(cohens_d) < 0.5:
        recommendations.append("🟡 Consider increasing reward scale")
    
    if reward_gap < uniform_entropy * 0.5:
        recommendations.append("🔴 CRITICAL: Reward gap smaller than random baseline")
    
    if agent and hasattr(agent, 'exploration_rate'):
        if agent.exploration_rate > reward_gap / uniform_entropy:
            recommendations.append(f"🟡 Reduce exploration rate to < {reward_gap/uniform_entropy:.1%}")
    
    if not recommendations:
        recommendations.append("✅ Reward signal appears adequate")
    
    for rec in recommendations:
        logger.info(f"  {rec}")
    
    return {
        'signal_to_noise_ratio': signal_to_noise,
        'cohens_d': cohens_d,
        'reward_gap': reward_gap,
        'baseline_entropy': uniform_entropy,
        'reward_stats': reward_stats,
        'recommendations': recommendations
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
        # pred_actions = agent.choose_action(pod_features, kv_hit_ratios, request_features, evaluate=False)

        # Get action probabilities
        action_probs = agent.policy(pod_features, kv_hit_ratios, request_features)
        
        # Calculate accuracy
        pred_actions = pred_actions.cpu()
        true_actions = true_actions.cpu()
        if pred_actions.shape != true_actions.shape:
            logger.error(f"Shape mismatch: pred_actions {pred_actions.shape} vs true_actions {true_actions.shape}")
            accuracy = 0.0
        else:
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
    Enhanced plotting function with proper train/eval data separation
    """
    # Create larger figure with more subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Determine action dimension
    if hasattr(agent.policy, 'policy_head'):
        action_dim = agent.policy.policy_head.out_features
    else:
        action_dim = 7  # Default for your setup
    
    # Create proper train/eval split for analysis if combined_data is available
    eval_data_for_analysis = None
    train_data_for_analysis = None
    
    if combined_data is not None:
        # Create a consistent train/eval split (80/20)
        total_samples = len(combined_data['actions'])
        eval_size = min(500, int(0.2 * total_samples))  # Use 20% for eval, max 500 samples
        
        # Use fixed seed for reproducible split
        torch.manual_seed(42)
        indices = torch.randperm(total_samples)
        eval_indices = indices[:eval_size]
        train_indices = indices[eval_size:]
        
        # Create eval data subset
        eval_data_for_analysis = {
            'pod_features_with_staleness': combined_data['pod_features_with_staleness'][eval_indices],
            'kv_hit_ratios': combined_data['kv_hit_ratios'][eval_indices],
            'request_features': combined_data['request_features'][eval_indices],
            'actions': combined_data['actions'][eval_indices],
            'rewards': combined_data['rewards'][eval_indices]
        }
        
        # Create train data subset for analysis
        train_data_for_analysis = {
            'actions': combined_data['actions'][train_indices],
            'rewards': combined_data['rewards'][train_indices]
        }
        
        logger.info(f"Created consistent train/eval split: {len(train_indices)} train, {len(eval_indices)} eval samples")
    
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
    
    # 6. Training Data Action Distribution (Ground Truth from TRAINING data)
    plt.subplot(3, 4, 6)
    if train_data_for_analysis is not None:
        training_actions = train_data_for_analysis['actions']
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
    elif combined_data is not None and 'actions' in combined_data:
        # Fallback to original behavior if split failed
        training_actions = combined_data['actions']
        training_action_counts = torch.bincount(training_actions, minlength=action_dim).numpy()
        
        bars = plt.bar(range(action_dim), training_action_counts, 
                      color='skyblue', alpha=0.7, edgecolor='navy')
        plt.title('All Data Distribution')
        plt.xlabel('Pod ID')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
    
    # 7. Model Predictions vs Ground Truth (Latest Evaluation on EVAL data)
    plt.subplot(3, 4, 7)
    if eval_data_for_analysis is not None:
        # Generate fresh predictions on the consistent eval set
        agent.policy.eval()
        with torch.no_grad():
            pod_features = eval_data_for_analysis['pod_features_with_staleness'].to(device)
            kv_hit_ratios = eval_data_for_analysis['kv_hit_ratios'].to(device)
            request_features = eval_data_for_analysis['request_features'].to(device)
            true_actions = eval_data_for_analysis['actions']
            
            pred_actions = agent.choose_action(pod_features, kv_hit_ratios, request_features, evaluate=True)
            pred_actions = pred_actions.cpu().numpy()
            true_actions = true_actions.numpy()
        
        pred_counts = np.bincount(pred_actions, minlength=action_dim)
        true_counts = np.bincount(true_actions, minlength=action_dim)
        
        x = np.arange(action_dim)
        width = 0.35
        
        bars1 = plt.bar(x - width/2, true_counts, width, 
                       label='Ground Truth (Eval)', alpha=0.7, color='lightcoral')
        bars2 = plt.bar(x + width/2, pred_counts, width, 
                       label='Predicted (Eval)', alpha=0.7, color='lightblue')
        
        plt.title('Predictions vs Ground Truth\n(Evaluation Data)')
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
                            
        # Add accuracy annotation
        accuracy = (pred_actions == true_actions).mean()
        plt.text(0.98, 0.98, f'Accuracy: {accuracy:.3f}', 
                transform=plt.gca().transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
    elif eval_metrics and 'pred_actions' in eval_metrics[-1]:
        # Fallback to original behavior
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
    
    # 8. Action Probability Heatmap (Latest Evaluation on EVAL data)
    plt.subplot(3, 4, 8)
    if eval_data_for_analysis is not None:
        # Generate fresh probabilities on the consistent eval set
        agent.policy.eval()
        with torch.no_grad():
            pod_features = eval_data_for_analysis['pod_features_with_staleness'].to(device)
            kv_hit_ratios = eval_data_for_analysis['kv_hit_ratios'].to(device)
            request_features = eval_data_for_analysis['request_features'].to(device)
            
            action_probs = agent.policy(pod_features, kv_hit_ratios, request_features)
            probs = action_probs.cpu().numpy()
        
        # Show average probabilities
        avg_probs = np.mean(probs, axis=0)
        bars = plt.bar(range(action_dim), avg_probs, 
                      color='orange', alpha=0.7)
        plt.title('Average Action Probabilities\n(Evaluation Data)')
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
        
    elif eval_metrics and 'probs' in eval_metrics[-1]:
        # Fallback to original behavior
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
    
    # 9. Confidence Distribution (on EVAL data)
    plt.subplot(3, 4, 9)
    if eval_data_for_analysis is not None:
        # Use the probabilities generated above
        agent.policy.eval()
        with torch.no_grad():
            pod_features = eval_data_for_analysis['pod_features_with_staleness'].to(device)
            kv_hit_ratios = eval_data_for_analysis['kv_hit_ratios'].to(device)
            request_features = eval_data_for_analysis['request_features'].to(device)
            
            action_probs = agent.policy(pod_features, kv_hit_ratios, request_features)
            probs = action_probs.cpu().numpy()
        
        max_probs = np.max(probs, axis=1)  # Confidence for each prediction
        
        plt.hist(max_probs, bins=20, alpha=0.7, color='purple', edgecolor='black')
        plt.title('Confidence Distribution\n(Evaluation Data)')
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
        
    elif eval_metrics and 'probs' in eval_metrics[-1]:
        # Fallback to original behavior
        probs = eval_metrics[-1]['probs']
        max_probs = np.max(probs, axis=1)  # Confidence for each prediction
        
        plt.hist(max_probs, bins=20, alpha=0.7, color='purple', edgecolor='black')
        plt.title('Confidence Distribution')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
    
    # 10. Accuracy vs Confidence Scatter
    plt.subplot(3, 4, 10)
    if eval_metrics:
        accuracies = [m['accuracy'] for m in eval_metrics]
        confidences = [m.get('avg_confidence', 0) for m in eval_metrics]
        
        plt.scatter(confidences, accuracies, alpha=0.6, s=50)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
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
        summary_text += f"Total samples: {total_samples:,}\n"
        if train_data_for_analysis is not None:
            summary_text += f"Train samples: {len(train_data_for_analysis['actions']):,}\n"
            summary_text += f"Eval samples: {len(eval_data_for_analysis['actions']):,}\n"
    
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
            summary_text += f"\nLEARNING DETECTED\n"
        elif final_accuracy > random_baseline * 1.1:
            summary_text += f"\nMODEST LEARNING\n"
        else:
            summary_text += f"\nNO CLEAR LEARNING\n"
        
        # Calibration assessment
        if abs(final_confidence - final_accuracy) < 0.1:
            summary_text += "Well calibrated\n"
        elif final_confidence > final_accuracy + 0.2:
            summary_text += "Overconfident\n"
        else:
            summary_text += "Underconfident\n"
    
    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    pdf_fn = f"{output_dir}/comprehensive_training_metrics.pdf"
    plt.savefig(pdf_fn, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"* Saved training plots: {pdf_fn}")
    
    # Print summary to console
    if eval_metrics:
        logger.info("\n" + "="*60)
        logger.info("TRAINING SUMMARY")
        logger.info("="*60)
        if combined_data is not None:
            logger.info(f"Total samples: {len(combined_data['actions']):,}")
            if train_data_for_analysis is not None:
                logger.info(f"Train/Eval split: {len(train_data_for_analysis['actions']):,}/{len(eval_data_for_analysis['actions']):,}")
        if hasattr(agent.policy, 'parameters'):
            total_params = sum(p.numel() for p in agent.policy.parameters())
            logger.info(f"Model parameters: {total_params:,}")
        
        final_accuracy = eval_metrics[-1]['accuracy']
        final_confidence = eval_metrics[-1].get('avg_confidence', 0)
        random_baseline = 1.0 / action_dim
        
        logger.info(f"Final accuracy: {final_accuracy:.3f} ({final_accuracy*100:.1f}%)")
        logger.info(f"Final confidence: {final_confidence:.3f} ({final_confidence*100:.1f}%)")
        logger.info(f"Random baseline: {random_baseline:.3f} ({random_baseline*100:.1f}%)")
        
        if final_accuracy > random_baseline * 1.5:
            logger.info("✅ Model is learning significantly!")
        elif final_accuracy > random_baseline * 1.1:
            logger.info("⚠️  Model shows modest learning")
        else:
            logger.info("❌ Model performance close to random")
        logger.info("="*60)


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


def read_hyperparameters_from_file(file_path):
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

def analyze_llm_routing_performance(combined_data, agent=None):
    """
    LLM-specific routing performance analysis that accounts for inherent variability
    """
    logger.info("🔍 LLM ROUTING PERFORMANCE ANALYSIS")
    logger.info("=" * 60)
    
    actions = combined_data['actions']
    rewards = combined_data['rewards']
    num_pods = len(torch.unique(actions))
    
    # Extract additional context if available
    kv_hit_ratios = combined_data.get('kv_hit_ratios', None)
    pod_features = combined_data.get('pod_features_with_staleness', None)
    request_features = combined_data.get('request_features', None)
    
    logger.info("1️⃣ CONTEXT-AWARE PERFORMANCE ANALYSIS:")
    
    # 1. PERFORMANCE BY REQUEST CONTEXT
    if request_features is not None:
        # Analyze by request size (token count)
        input_tokens = request_features[:, 0] if request_features.shape[1] > 0 else None
        output_tokens = request_features[:, 1] if request_features.shape[1] > 1 else None
        
        if input_tokens is not None:
            # Categorize requests by size
            small_requests = input_tokens < torch.quantile(input_tokens, 0.33)
            medium_requests = (input_tokens >= torch.quantile(input_tokens, 0.33)) & (input_tokens < torch.quantile(input_tokens, 0.67))
            large_requests = input_tokens >= torch.quantile(input_tokens, 0.67)
            
            logger.info("\nPerformance by Request Size:")
            for mask, name in [(small_requests, "Small"), (medium_requests, "Medium"), (large_requests, "Large")]:
                if mask.sum() > 0:
                    category_rewards = rewards[mask]
                    category_actions = actions[mask]
                    
                    logger.info(f"\n  {name} Requests ({mask.sum()} samples):")
                    for pod_idx in range(num_pods):
                        pod_mask = category_actions == pod_idx
                        if pod_mask.sum() > 0:
                            pod_rewards = category_rewards[pod_mask]
                            logger.info(f"    Pod {pod_idx}: μ={pod_rewards.mean():.4f}, n={pod_mask.sum()}")
    
    # 2. CACHE-AWARE ANALYSIS
    if kv_hit_ratios is not None:
        logger.info("\n2️⃣ CACHE-AWARE ANALYSIS:")
        
        # Average KV hit ratio per pod per sample
        avg_kv_per_sample = kv_hit_ratios.mean(dim=2).squeeze() if len(kv_hit_ratios.shape) > 2 else kv_hit_ratios.squeeze()
        
        # For each sample, get the KV hit ratio of the selected pod
        selected_kv_ratios = []
        for i, action in enumerate(actions):
            if i < len(avg_kv_per_sample) and action < avg_kv_per_sample.shape[1]:
                selected_kv_ratios.append(avg_kv_per_sample[i, action].item())
        
        if selected_kv_ratios:
            selected_kv_ratios = torch.tensor(selected_kv_ratios)
            
            # Categorize by cache hit ratio
            high_cache = selected_kv_ratios > torch.quantile(selected_kv_ratios, 0.75)
            low_cache = selected_kv_ratios < torch.quantile(selected_kv_ratios, 0.25)
            
            logger.info(f"High Cache Requests: μ={rewards[high_cache].mean():.4f} (n={high_cache.sum()})")
            logger.info(f"Low Cache Requests: μ={rewards[low_cache].mean():.4f} (n={low_cache.sum()})")
            
            cache_performance_diff = rewards[high_cache].mean() - rewards[low_cache].mean()
            logger.info(f"Cache Impact: {cache_performance_diff:.4f}")
    
    # 3. TAIL LATENCY ANALYSIS (Most Important for User Experience)
    logger.info("\n3️⃣ TAIL LATENCY ANALYSIS:")
    
    # Convert rewards back to approximate latency categories
    # Assuming negative rewards = bad latency, positive = good latency
    excellent_threshold = torch.quantile(rewards, 0.9)  # Top 10%
    poor_threshold = torch.quantile(rewards, 0.1)       # Bottom 10%
    
    logger.info("Pod Performance by Latency Categories:")
    logger.info(f"Excellent (>{excellent_threshold:.3f}): Top 10% of requests")
    logger.info(f"Poor (<{poor_threshold:.3f}): Bottom 10% of requests")
    
    pod_tail_performance = {}
    for pod_idx in range(num_pods):
        pod_mask = actions == pod_idx
        if pod_mask.sum() > 0:
            pod_rewards = rewards[pod_mask]
            
            excellent_rate = (pod_rewards > excellent_threshold).float().mean().item()
            poor_rate = (pod_rewards < poor_threshold).float().mean().item()
            p50 = torch.quantile(pod_rewards, 0.5).item()
            p90 = torch.quantile(pod_rewards, 0.9).item()
            p99 = torch.quantile(pod_rewards, 0.99).item()
            
            pod_tail_performance[pod_idx] = {
                'excellent_rate': excellent_rate,
                'poor_rate': poor_rate,
                'p50': p50,
                'p90': p90,
                'p99': p99,
                'total_requests': pod_mask.sum().item()
            }
            
            logger.info(f"  Pod {pod_idx}: Excellent={excellent_rate:.1%}, Poor={poor_rate:.1%}, "
                       f"P90={p90:.3f}, P99={p99:.3f}")
    
    # 4. CONSISTENCY ANALYSIS
    logger.info("\n4️⃣ CONSISTENCY ANALYSIS:")
    
    consistency_metrics = {}
    for pod_idx in range(num_pods):
        pod_mask = actions == pod_idx
        if pod_mask.sum() > 0:
            pod_rewards = rewards[pod_mask]
            
            # Coefficient of variation (std/mean) - lower is more consistent
            cv = (pod_rewards.std() / (abs(pod_rewards.mean()) + 1e-8)).item()
            
            # Interquartile range - smaller means more consistent
            iqr = (torch.quantile(pod_rewards, 0.75) - torch.quantile(pod_rewards, 0.25)).item()
            
            # Percentage of requests within 1 std of mean
            within_1_std = ((pod_rewards - pod_rewards.mean()).abs() < pod_rewards.std()).float().mean().item()
            
            consistency_metrics[pod_idx] = {
                'coefficient_of_variation': cv,
                'iqr': iqr,
                'within_1_std': within_1_std
            }
            
            logger.info(f"  Pod {pod_idx}: CV={cv:.3f}, IQR={iqr:.3f}, Within1σ={within_1_std:.1%}")
    
    # 5. BUSINESS IMPACT ANALYSIS
    logger.info("\n5️⃣ BUSINESS IMPACT ANALYSIS:")
    
    # Calculate SLA violation rates (assuming negative rewards = SLA violations)
    sla_threshold = 0.0  # Adjust based on your SLA definition
    
    business_metrics = {}
    for pod_idx in range(num_pods):
        pod_mask = actions == pod_idx
        if pod_mask.sum() > 0:
            pod_rewards = rewards[pod_mask]
            
            sla_violation_rate = (pod_rewards < sla_threshold).float().mean().item()
            avg_user_experience = pod_rewards.mean().item()
            
            # Estimated user satisfaction (higher rewards = better experience)
            user_satisfaction = (pod_rewards > 0.1).float().mean().item()  # Adjust threshold
            
            business_metrics[pod_idx] = {
                'sla_violation_rate': sla_violation_rate,
                'avg_user_experience': avg_user_experience,
                'user_satisfaction_rate': user_satisfaction
            }
            
            logger.info(f"  Pod {pod_idx}: SLA violations={sla_violation_rate:.1%}, "
                       f"User satisfaction={user_satisfaction:.1%}")
    
    # 6. ROUTING EFFECTIVENESS ANALYSIS
    logger.info("\n6️⃣ ROUTING EFFECTIVENESS:")
    
    if agent is not None:
        # Simulate routing decisions vs random routing
        random_baseline_reward = rewards.mean().item()
        
        # Get current model's expected performance
        model_expected_reward = 0
        for pod_idx in range(num_pods):
            pod_mask = actions == pod_idx
            if pod_mask.sum() > 0:
                pod_prob = pod_mask.float().mean().item()  # Frequency in data
                pod_reward = rewards[pod_mask].mean().item()
                model_expected_reward += pod_prob * pod_reward
        
        improvement_over_random = model_expected_reward - random_baseline_reward
        
        logger.info(f"Random routing expected reward: {random_baseline_reward:.4f}")
        logger.info(f"Current model expected reward: {model_expected_reward:.4f}")
        logger.info(f"Improvement over random: {improvement_over_random:.4f}")
        
        # Calculate potential improvement if always choosing best pod
        best_pod_idx = max(pod_tail_performance.keys(), 
                          key=lambda x: pod_tail_performance[x]['p50'])
        best_pod_reward = pod_tail_performance[best_pod_idx]['p50']
        
        potential_improvement = best_pod_reward - random_baseline_reward
        
        logger.info(f"Best pod (Pod {best_pod_idx}) P50 reward: {best_pod_reward:.4f}")
        logger.info(f"Potential improvement: {potential_improvement:.4f}")
        
        if potential_improvement > 0:
            efficiency = improvement_over_random / potential_improvement
            logger.info(f"Routing efficiency: {efficiency:.1%}")
    
    # 7. RECOMMENDATIONS FOR LLM ROUTING
    logger.info("\n7️⃣ LLM ROUTING RECOMMENDATIONS:")
    
    recommendations = []
    
    # Find best performing pod for each metric
    best_consistency_pod = min(consistency_metrics.keys(), 
                              key=lambda x: consistency_metrics[x]['coefficient_of_variation'])
    best_tail_pod = max(pod_tail_performance.keys(), 
                       key=lambda x: pod_tail_performance[x]['p90'])
    best_satisfaction_pod = max(business_metrics.keys(), 
                               key=lambda x: business_metrics[x]['user_satisfaction_rate'])
    
    logger.info(f"Most consistent pod: {best_consistency_pod}")
    logger.info(f"Best tail latency pod: {best_tail_pod}")
    logger.info(f"Best user satisfaction pod: {best_satisfaction_pod}")
    
    # Check if there's a clear winner
    if best_consistency_pod == best_tail_pod == best_satisfaction_pod:
        recommendations.append(f"✅ Pod {best_consistency_pod} is clearly superior across all metrics")
    else:
        recommendations.append("📊 Different pods excel in different metrics - context-aware routing recommended")
    
    # Check for problematic pods
    for pod_idx, metrics in business_metrics.items():
        if metrics['sla_violation_rate'] > 0.2:  # >20% violations
            recommendations.append(f"🔴 Pod {pod_idx} has high SLA violation rate ({metrics['sla_violation_rate']:.1%})")
    
    for rec in recommendations:
        logger.info(f"  {rec}")
    
    return {
        'pod_tail_performance': pod_tail_performance,
        'consistency_metrics': consistency_metrics,
        'business_metrics': business_metrics,
        'recommendations': recommendations,
        'best_pods': {
            'consistency': best_consistency_pod,
            'tail_latency': best_tail_pod,
            'user_satisfaction': best_satisfaction_pod
        }
    }


def analyze_routing_context_sensitivity(combined_data):
    """
    Analyze how routing performance varies by request context
    """
    logger.info("\n🔍 CONTEXT SENSITIVITY ANALYSIS")
    logger.info("=" * 60)
    
    actions = combined_data['actions']
    rewards = combined_data['rewards']
    request_features = combined_data.get('request_features', None)
    pod_features = combined_data.get('pod_features_with_staleness', None)
    
    if request_features is None:
        logger.warning("No request features available for context analysis")
        return {}
    
    num_pods = len(torch.unique(actions))
    
    # Analyze by request characteristics
    input_tokens = request_features[:, 0] if request_features.shape[1] > 0 else None
    output_tokens = request_features[:, 1] if request_features.shape[1] > 1 else None
    
    context_analysis = {}
    
    if input_tokens is not None:
        # Categorize by input length
        contexts = {
            "short": input_tokens < torch.quantile(input_tokens, 0.33),
            "medium": (input_tokens >= torch.quantile(input_tokens, 0.33)) & 
                     (input_tokens < torch.quantile(input_tokens, 0.67)),
            "long": input_tokens >= torch.quantile(input_tokens, 0.67)
        }
        
        logger.info("Performance by Input Length:")
        
        for context_name, context_mask in contexts.items():
            if context_mask.sum() == 0:
                continue
                
            context_actions = actions[context_mask]
            context_rewards = rewards[context_mask]
            
            logger.info(f"\n{context_name.upper()} inputs ({context_mask.sum()} samples):")
            
            best_pod_for_context = -1
            best_reward_for_context = float('-inf')
            
            for pod_idx in range(num_pods):
                pod_mask = context_actions == pod_idx
                if pod_mask.sum() > 5:  # Require minimum samples
                    pod_rewards = context_rewards[pod_mask]
                    mean_reward = pod_rewards.mean().item()
                    
                    logger.info(f"  Pod {pod_idx}: μ={mean_reward:.4f}, "
                               f"p90={torch.quantile(pod_rewards, 0.9):.4f}, n={pod_mask.sum()}")
                    
                    if mean_reward > best_reward_for_context:
                        best_reward_for_context = mean_reward
                        best_pod_for_context = pod_idx
            
            context_analysis[context_name] = {
                'best_pod': best_pod_for_context,
                'best_reward': best_reward_for_context,
                'total_requests': context_mask.sum().item()
            }
            
            if best_pod_for_context >= 0:
                logger.info(f"  → Best pod for {context_name} requests: Pod {best_pod_for_context}")
    
    return context_analysis


# Integration function
def comprehensive_llm_routing_analysis(combined_data, agent=None):
    """
    Complete LLM routing analysis replacing generic signal analysis
    """
    # LLM-specific performance analysis
    llm_performance = analyze_llm_routing_performance(combined_data, agent)
    
    # Context sensitivity analysis
    context_sensitivity = analyze_routing_context_sensitivity(combined_data)
    
    return {
        'llm_performance': llm_performance,
        'context_sensitivity': context_sensitivity
    }


# main entry point
def train(encoded_data_dir, model_output_dir, HYPERPARAMETERS, is_online_learning):
    global final_model_dir
    final_model_dir = model_output_dir
    os.makedirs(final_model_dir, exist_ok=True)
    logger.info(f"Starting training process. final_model_dir: {final_model_dir}")
    
    if HYPERPARAMETERS.get('deterministic_training', False):
        training_seed = HYPERPARAMETERS['training_seed']
        logger.info(f"🔒 DETERMINISTIC TRAINING MODE - Setting training seed: {training_seed}")
        
        # Set all seeds for training
        torch.manual_seed(training_seed)
        np.random.seed(training_seed)
        random.seed(training_seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(training_seed)
            torch.cuda.manual_seed_all(training_seed)
        
        # Make operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        logger.info("✅ Training determinism enabled")


    # Load and combine data from all batches
    combined_data = load_all_encoded_data(encoded_data_dir)
    dataset_analysis = analyze_dataset_detailed(combined_data)
    HYPERPARAMETERS['dataset_analysis'] = dataset_analysis
    state_dim = {
        'pod_features': combined_data['pod_features_with_staleness'].shape[2],
        'kv_hit_ratios': combined_data['kv_hit_ratios'].shape[2],
        'request_features': combined_data['request_features'].shape[1],
        'num_pods': combined_data['pod_features'].shape[1]
    }
    action_dim = combined_data['pod_features'].shape[1]    
    logger.info(f"State dimensions: {state_dim}")
    logger.info(f"Action dimension: {action_dim}")

    logger.info(f"State dimensions (should show GPU moved from request to pod features):")
    logger.info(f"  pod_features: {state_dim['pod_features']}")
    logger.info(f"  request_features: {state_dim['request_features']}")
    
    # Create Simplified Contextual Bandit agent
    agent = SimplifiedContextualBandit(
        state_dim=state_dim,
        action_dim=action_dim,
        HYPERPARAMETERS=HYPERPARAMETERS,
    )

    ##############################################
    # Load pretrained model
    if final_model_dir and os.path.exists(final_model_dir):
        try:
            agent.load(final_model_dir)
            logger.info(f"Successfully loaded pretrained model from {final_model_dir} for online learning")
            ################################################
            # Adjust learning rate for online learning (typically lower)
            if is_online_learning:
                original_learning_rate = HYPERPARAMETERS['learning_rate']
                HYPERPARAMETERS['learning_rate'] *= 0.1  # 10x lower learning rate
                logger.info(f"Adjusted learning rate from {original_learning_rate} to {HYPERPARAMETERS['learning_rate']} for online learning")
            for param_group in agent.optimizer.param_groups:
                param_group['lr'] = HYPERPARAMETERS['learning_rate']
        except Exception as e:
            logger.error(f"Error loading pretrained model: {e}")
            logger.info("Starting training from scratch")

    HYPERPARAMETERS['training_epochs'] = HYPERPARAMETERS['training_epochs']
    ## Use fewer epochs for online learning
    if is_online_learning:
        HYPERPARAMETERS['training_epochs'] = max(5, HYPERPARAMETERS['training_epochs'] // 4)
    logger.info(f"Online learning mode: reduced epochs to {HYPERPARAMETERS['training_epochs']}, exploration to {HYPERPARAMETERS['exploration_rate']}")
    
    # Update agent's exploration rate
    agent.exploration_rate = HYPERPARAMETERS['exploration_rate']
    
    # Create dataset
    dataset = RoutingDataset(combined_data)
    
    # WITH this:
    if HYPERPARAMETERS.get('deterministic_training', False):
        # Create deterministic DataLoader
        generator = torch.Generator()
        generator.manual_seed(HYPERPARAMETERS['training_seed'])
        dataloader = DataLoader(
            dataset, 
            batch_size=HYPERPARAMETERS['batch_size'], 
            shuffle=True,  # We want shuffling, but deterministic
            generator=generator,  # This makes shuffling deterministic
            worker_init_fn=lambda worker_id: np.random.seed(HYPERPARAMETERS.get('training_seed', 54321) + worker_id)
        )
        logger.info("✅ Created deterministic DataLoader")
    else:
        dataloader = DataLoader(
            dataset, 
            batch_size=HYPERPARAMETERS['batch_size'], 
            shuffle=True
        )
    
    
    number_of_batches = len(dataloader)
    
    logger.info(f"Loaded dataset with {len(dataset)} samples")
    # Training loop (rest remains the same but with adjusted epochs)
    logger.info("Starting training...")
    total_updates = 0
    eval_metrics = []
    best_accuracy = 0.0
    
    for epoch in range(HYPERPARAMETERS['training_epochs']):
        logger.info(f"=" * 80)
        logger.info(f"EPOCH {epoch+1}/{HYPERPARAMETERS['training_epochs']}")
        logger.info(f"=" * 80)
        agent.current_epoch = epoch
        epoch_start_time = time.time()
        epoch_loss = 0
        epoch_reward = 0
        epoch_entropy = 0
        epoch_updates = 0
        
        dataloader_iter = iter(dataloader)
        num_iter_per_data = 1
        total_iter = number_of_batches * num_iter_per_data
        # final_total_num_iteration_per_epoch = min(HYPERPARAMETERS['max_updates_per_epoch'], total_iter)
        final_total_num_iteration_per_epoch = len(dataloader)
        logger.info(f"=" * 80)
        logger.info(f"EPOCH {epoch+1}/{HYPERPARAMETERS['training_epochs']} - Processing final_total_num_iteration_per_epoch: {final_total_num_iteration_per_epoch} iterations")
        logger.info(f"=" * 80)        

        for batch_iter_idx in range(final_total_num_iteration_per_epoch):
            # try:
            batch = next(dataloader_iter)
            # except StopIteration:
            #     dataloader_iter = iter(dataloader)
            #     batch = next(dataloader_iter)
            
            pod_features = batch['pod_features'].to(device)
            kv_hit_ratios = batch['kv_hit_ratios'].to(device)
            request_features = batch['request_features'].to(device)
            actions = batch['action'].to(device)
            rewards = batch['reward'].to(device).unsqueeze(1)
            
            for j in range(len(rewards)):
                agent.remember(
                    pod_features[j:j+1], 
                    kv_hit_ratios[j:j+1], 
                    request_features[j:j+1], 
                    actions[j:j+1], 
                    rewards[j:j+1]
                )
            
            trigger_learning = (batch_iter_idx+1) % HYPERPARAMETERS['learning_every_x_iter'] == 0 or batch_iter_idx == final_total_num_iteration_per_epoch - 1
            if trigger_learning:
                if len(agent.pod_features) > 0:
                    try:
                        update_metrics = agent.learn()
                        total_updates += 1
                        epoch_updates += 1
                        epoch_loss += update_metrics['loss']
                        epoch_reward += update_metrics['reward']
                        epoch_entropy += update_metrics['entropy']
                        if batch_iter_idx % max(1, final_total_num_iteration_per_epoch//3) == 0:
                            logger.info(f"Batch: {batch_iter_idx+1}/{final_total_num_iteration_per_epoch}, Loss: {update_metrics['loss']:.4f}")
                    except Exception as e:
                        logger.error(f"Error during learning: {e}")
        
            if (batch_iter_idx + 1) % max(1, final_total_num_iteration_per_epoch // HYPERPARAMETERS['eval_interval']) == 0:
                logger.info(f"Evaluating agent at batch {batch_iter_idx+1}")
                try:
                    eval_indices = torch.randperm(len(dataset))[:min(200, len(dataset))]
                    eval_data = {
                        'pod_features_with_staleness': combined_data['pod_features_with_staleness'][eval_indices],
                        'kv_hit_ratios': combined_data['kv_hit_ratios'][eval_indices],
                        'request_features': combined_data['request_features'][eval_indices],
                        'actions': combined_data['actions'][eval_indices],
                        'rewards': combined_data['rewards'][eval_indices]
                    }
                    
                    # Use the standard evaluate_agent function for plotting compatibility
                    metrics = evaluate_agent(agent, eval_data, num_samples=200)
                    eval_metrics.append(metrics)
                    logger.info(f"Evaluation - Accuracy: {metrics['accuracy']:.4f}")
                    logger.info(f"Evaluation - Confidence: {metrics.get('avg_confidence', 0):.4f}")
                    
                    # Do identity analysis separately (every few evaluations to avoid spam)
                    if len(eval_metrics) % 3 == 0:  # Every 3rd evaluation
                        try:
                            identity_analysis = analyze_pod_identity_learning(agent, eval_data, final_model_dir)
                            most_favored = identity_analysis.get('most_favored_pod', -1)
                            max_concentration = identity_analysis.get('max_concentration', 0)
                            correlation = identity_analysis.get('correlation', 0)
                            logger.info(f"Identity Analysis - Most favored pod: {most_favored}, "
                                    f"Concentration: {max_concentration:.1%}, "
                                    f"Correlation: {correlation:.3f}")
                            if max_concentration > 0.6:
                                logger.warning(f"⚠️  Pod {most_favored} is getting {max_concentration:.1%} of traffic!")
                        except Exception as e:
                            logger.error(f"Error in identity analysis: {e}")
                            
                except Exception as e:
                    logger.error(f"Error during evaluation: {e}")
                    import traceback
                    traceback.print_exc()
        
        # End of epoch logging
        epoch_duration = time.time() - epoch_start_time
        if epoch_updates > 0:
            avg_loss = epoch_loss / epoch_updates
            avg_reward = epoch_reward / epoch_updates
            avg_entropy = epoch_entropy / epoch_updates
            
            logger.info(f"=" * 80)
            logger.info(f"EPOCH {epoch+1} SUMMARY:")
            logger.info(f"  Duration: {epoch_duration:.1f}s")
            logger.info(f"  Updates: {epoch_updates}")
            logger.info(f"  Global batches processed: {agent.global_batch_counter}")
            logger.info(f"  Avg Loss: {avg_loss:.4f}")
            logger.info(f"  Avg Reward: {avg_reward:.4f}")
            logger.info(f"  Avg Entropy: {avg_entropy:.4f}")
            logger.info(f"=" * 80)

    # Save final model
    agent.save(final_model_dir)
    try:
        plot_training_metrics(agent, eval_metrics, final_model_dir, combined_data)
    except Exception as e:
        logger.error(f"Error plotting training metrics: {e}")
    # os.system(f"cp -r {final_model_dir} final_model")

    # Save configuration
    with open(os.path.join(final_model_dir, 'model_config.json'), 'w') as f:
        json.dump(HYPERPARAMETERS, f, indent=4, default=str)
    
    return {
        'agent': agent,
        'model_dir': final_model_dir,
        'eval_metrics': eval_metrics,
        'best_accuracy': best_accuracy,
        'HYPERPARAMETERS': HYPERPARAMETERS,
    }

# Global cache for agent instance (for inference)
_cached_agent = None
_cached_agent_config = None
_cached_metadata = None

def infer_from_tensor(tensor_data, request_id, model_updated, HYPERPARAMETERS):
    global final_model_dir, _cached_agent, _cached_agent_config
    try:
        pod_features = tensor_data['pod_features_with_staleness'].to(device)
        kv_hit_ratios = tensor_data['kv_hit_ratios'].to(device)
        request_features = tensor_data['request_features'].to(device)
    except KeyError as e:
        logger.error(f"Missing key in tensor_data: {e}")
        raise ValueError(f"Missing key in tensor_data: {e}")
    
    # Ensure batch format
    if len(pod_features.shape) == 2:
        pod_features = pod_features.unsqueeze(0)
    if len(kv_hit_ratios.shape) == 2:
        kv_hit_ratios = kv_hit_ratios.unsqueeze(0)
    if len(request_features.shape) == 1:
        request_features = request_features.unsqueeze(0)

    # Get or create agent
    agent = _get_or_create_agent(pod_features, kv_hit_ratios, request_features, model_updated, HYPERPARAMETERS)
    
    # Optional detailed logging
    _log_inference_details(pod_features, kv_hit_ratios, request_features, agent, request_id)
    
    # Core inference
    infer_start_time = time.time()
    agent.policy.eval()
    with torch.no_grad():
        action_probs = agent.policy(pod_features, kv_hit_ratios, request_features)
        
        # ADD DEBUGGING
        logger.info(f"🔍 Request {request_id}:")
        logger.info(f"  Raw probabilities: {action_probs[0].cpu().numpy()}")
        logger.info(f"  Exploration rate: {HYPERPARAMETERS['exploration_rate']}")
        logger.info(f"  Explore flag: {HYPERPARAMETERS['explore']}")
        
        # Check for uniform probabilities (model collapse)
        prob_std = action_probs[0].std().item()
        if prob_std < 0.01:  # Very small standard deviation = uniform
            logger.warning(f"⚠️  MODEL COLLAPSE: All probabilities nearly uniform (std={prob_std:.6f})")
        
        
        if HYPERPARAMETERS['exploration_rate'] > 0:
            action, _ = agent.policy.get_action(pod_features, kv_hit_ratios, request_features, explore=HYPERPARAMETERS['explore'], epsilon=HYPERPARAMETERS['exploration_rate'])
            selected_action = action.item()
        else:
            selected_action = torch.argmax(action_probs, dim=1).item()
        
        confidence = action_probs[0, selected_action].item()

    total_inference_time = time.time() - infer_start_time
    
    # Return result
    result = {
        'selected_pod_index': selected_action,
        'confidence': confidence,
        'pod_probabilities': action_probs[0].cpu().numpy().tolist(),
        'final_model_dir': final_model_dir,
        'exploration_enabled': HYPERPARAMETERS['exploration_rate'] > 0,
        'model_type': 'simplified'
    }
    
    timing_info = {
        'total_inference_time_ms': total_inference_time * 1000,
        'agent_cache_hit': hasattr(_get_or_create_agent, '_last_cache_hit'),
        'model_updated': model_updated
    }
    
    return result, timing_info

def _get_or_create_agent(pod_features, kv_hit_ratios, request_features, 
                        model_updated, HYPERPARAMETERS):
    """
    Get cached agent or create new one with dimension validation
    """
    global _cached_agent, _cached_agent_config
    
    current_config = {
        'pod_features': pod_features.shape[2],
        'kv_hit_ratios': kv_hit_ratios.shape[2], 
        'request_features': request_features.shape[1],
        'num_pods': pod_features.shape[1],
        'exploration_rate': HYPERPARAMETERS['exploration_rate'],
        'final_model_dir': final_model_dir
    }
    
    # Check if we can reuse cached agent
    if (_cached_agent is not None and 
        _cached_agent_config is not None and
        _cached_agent_config == current_config):
        agent = _cached_agent
        _get_or_create_agent._last_cache_hit = True
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
        
        agent = SimplifiedContextualBandit(
            state_dim=state_dim,
            action_dim=current_config['num_pods'],
            HYPERPARAMETERS=HYPERPARAMETERS,
        )
        
        _cached_agent = agent
        _cached_agent_config = current_config.copy()
        _get_or_create_agent._last_cache_hit = False

    # Load model weights if needed
    if not getattr(_get_or_create_agent, '_last_cache_hit', False) or model_updated:
        _load_and_validate_model(agent, current_config)
    
    return agent


def _load_and_validate_model(agent, current_config):
    """
    Load model with dimension validation
    """
    try:
        agent.load(final_model_dir)
        agent.policy.eval()
        
        # Validate dimensions
        expected_input_dim = (current_config['pod_features'] + 
                            current_config['kv_hit_ratios'] + 
                            current_config['request_features'])
        actual_input_dim = agent.policy.pod_scorer[0].in_features
        
        if expected_input_dim != actual_input_dim:
            raise ValueError(
                f"Model dimension mismatch: expected {expected_input_dim}, "
                f"got {actual_input_dim}"
            )
        
        logger.info("✅ Model loaded and validated successfully")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
    
def _log_inference_details(pod_features, kv_hit_ratios, request_features, 
                          agent, request_id):
    """
    Optional detailed logging for debugging (only when DEBUG level enabled)
    """
    logger.debug("=" * 60)
    logger.debug(f"INFERENCE DETAILS - Request ID: {request_id}")
    logger.debug("=" * 60)
    
    # Input dimension analysis
    logger.debug(f"Input shapes:")
    logger.debug(f"  pod_features: {pod_features.shape}")
    logger.debug(f"  kv_hit_ratios: {kv_hit_ratios.shape}")  
    logger.debug(f"  request_features: {request_features.shape}")
    
    # Feature statistics
    logger.debug(f"Feature statistics:")
    logger.debug(f"  Pod features: min={pod_features.min():.4f}, max={pod_features.max():.4f}")
    logger.debug(f"  KV ratios: min={kv_hit_ratios.min():.4f}, max={kv_hit_ratios.max():.4f}")
    logger.debug(f"  Request features: min={request_features.min():.4f}, max={request_features.max():.4f}")
    
    # Model internals (if needed for debugging)
    _analyze_model_behavior(pod_features, kv_hit_ratios, request_features, agent, request_id)


def _analyze_model_behavior(pod_features, kv_hit_ratios, request_features, 
                           agent, request_id):
    """
    Detailed model behavior analysis (only for debugging)
    """
    with torch.no_grad():
        # Get raw scores
        batch_size = pod_features.shape[0]
        num_pods = pod_features.shape[1]
        
        combined_pod_features = torch.cat([pod_features, kv_hit_ratios], dim=2)
        expanded_request = request_features.unsqueeze(1).expand(-1, num_pods, -1)
        full_features = torch.cat([combined_pod_features, expanded_request], dim=2)
        reshaped_features = full_features.view(batch_size * num_pods, -1)
        
        raw_scores = agent.policy.pod_scorer(reshaped_features)
        raw_scores = raw_scores.view(batch_size, num_pods)
        
        logger.debug(f"Raw scores: {raw_scores[0].cpu().numpy()}")
        logger.debug(f"Score range: [{raw_scores.min():.4f}, {raw_scores.max():.4f}]")
        
        # Final probabilities
        action_probs = agent.policy(pod_features, kv_hit_ratios, request_features)
        logger.debug(f"Final probabilities: {action_probs[0].cpu().numpy()}")
        
def analyze_feature_impact(tensor_data, agent, feature_names=None):
    """
    Analyze impact of individual features on model decisions
    This can be called separately for detailed analysis
    """
    pod_features = tensor_data['pod_features_with_staleness'].to(device)
    kv_hit_ratios = tensor_data['kv_hit_ratios'].to(device)
    request_features = tensor_data['request_features'].to(device)
    
    # Ensure batch format
    if len(pod_features.shape) == 2:
        pod_features = pod_features.unsqueeze(0)
    if len(kv_hit_ratios.shape) == 2:
        kv_hit_ratios = kv_hit_ratios.unsqueeze(0)
    if len(request_features.shape) == 1:
        request_features = request_features.unsqueeze(0)
    
    agent.policy.eval()
    
    # Baseline prediction
    with torch.no_grad():
        baseline_probs = agent.policy(pod_features, kv_hit_ratios, request_features)
        baseline_action = torch.argmax(baseline_probs, dim=1).item()
    
    feature_impacts = {}
    
    # Test impact of zeroing out each feature type
    test_cases = [
        ('pod_features', lambda pf, kv, rf: (torch.zeros_like(pf), kv, rf)),
        ('kv_hit_ratios', lambda pf, kv, rf: (pf, torch.zeros_like(kv), rf)),
        ('request_features', lambda pf, kv, rf: (pf, kv, torch.zeros_like(rf)))
    ]
    
    for feature_type, modifier in test_cases:
        modified_pod, modified_kv, modified_req = modifier(pod_features, kv_hit_ratios, request_features)
        
        with torch.no_grad():
            modified_probs = agent.policy(modified_pod, modified_kv, modified_req)
            modified_action = torch.argmax(modified_probs, dim=1).item()
            
            prob_diff = (baseline_probs - modified_probs).abs().max().item()
            action_changed = (baseline_action != modified_action)
            
            feature_impacts[feature_type] = {
                'prob_difference': prob_diff,
                'action_changed': action_changed,
                'new_action': modified_action
            }
    
    return {
        'baseline_action': baseline_action,
        'baseline_probs': baseline_probs[0].cpu().numpy().tolist(),
        'feature_impacts': feature_impacts
    }


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



def comprehensive_root_cause_analysis(agent, combined_data, eval_data, HYPERPARAMETERS):
    logger.info("=" * 80)
    logger.info("🔍 COMPREHENSIVE ROOT CAUSE ANALYSIS")
    logger.info("=" * 80)

    for key, value in HYPERPARAMETERS.items():
        logger.info(f"Hyperparameter: {key} = {value}")
    
    # ==========================================================================
    # 1. DATA QUALITY ANALYSIS
    # ==========================================================================
    
    logger.info("="*50)
    logger.info("1️⃣  DATA QUALITY ANALYSIS")
    logger.info("="*50)
    
    def analyze_data_quality():
        actions = combined_data['actions']
        rewards = combined_data['rewards']
        
        logger.info(f"Dataset size: {len(actions)} samples")
        logger.info(f"Number of pods: {actions.max().item() + 1}")
        
        # Action distribution analysis
        action_counts = torch.bincount(actions)
        action_percentages = action_counts.float() / len(actions) * 100
        
        logger.info("\nAction distribution:")
        for i, (count, pct) in enumerate(zip(action_counts, action_percentages)):
            logger.info(f"  Pod {i}: {count} samples ({pct:.1f}%)")
        
        # Calculate imbalance
        max_pct = action_percentages.max().item()
        min_pct = action_percentages.min().item()
        imbalance_ratio = max_pct / max(min_pct, 0.1)
        logger.info(f"\nClass imbalance ratio: {imbalance_ratio:.2f}x")
        
        # Reward analysis by action
        logger.info(f"\nReward analysis:")
        logger.info(f"Overall reward: mean={rewards.mean():.4f}, std={rewards.std():.4f}")
        
        reward_by_action = {}
        for action_id in range(action_counts.shape[0]):
            mask = actions == action_id
            if mask.sum() > 0:
                action_rewards = rewards[mask]
                mean_reward = action_rewards.mean().item()
                std_reward = action_rewards.std().item()
                reward_by_action[action_id] = {
                    'mean': mean_reward, 
                    'std': std_reward,
                    'count': mask.sum().item()
                }
                logger.info(f"  Pod {action_id}: μ={mean_reward:.4f}, σ={std_reward:.4f}, n={mask.sum()}")
        
        # Calculate reward signal strength
        reward_means = [stats['mean'] for stats in reward_by_action.values()]
        reward_gap = max(reward_means) - min(reward_means)
        logger.info(f"\nReward signal strength:")
        logger.info(f"  Reward gap (max-min): {reward_gap:.4f}")
        logger.info(f"  Reward range: [{min(reward_means):.4f}, {max(reward_means):.4f}]")
        
        # Determine if reward signal is learnable
        if reward_gap < 0.01:
            logger.info("  ⚠️  VERY WEAK REWARD SIGNAL - might explain poor learning")
        elif reward_gap < 0.05:
            logger.info("  ⚠️  WEAK REWARD SIGNAL - challenging to learn")
        else:
            logger.info("  ✅ REASONABLE REWARD SIGNAL")
        
        return {
            'imbalance_ratio': imbalance_ratio,
            'reward_gap': reward_gap,
            'reward_by_action': reward_by_action,
            'data_size': len(actions)
        }
    
    data_analysis = analyze_data_quality()
    
    # ==========================================================================
    # 2. FEATURE ANALYSIS
    # ==========================================================================
    
    logger.info("\n" + "="*50)
    logger.info("2️⃣  FEATURE ANALYSIS")
    logger.info("="*50)
    
    def analyze_features():
        pod_features = combined_data['pod_features_with_staleness']
        kv_hit_ratios = combined_data['kv_hit_ratios'] 
        request_features = combined_data['request_features']
        
        logger.info(f"Pod features shape: {pod_features.shape}")
        logger.info(f"KV hit ratios shape: {kv_hit_ratios.shape}")
        logger.info(f"Request features shape: {request_features.shape}")
        
        # Analyze feature statistics
        def feature_stats(tensor, name):
            logger.info(f"\n{name}:")
            logger.info(f"  Range: [{tensor.min():.4f}, {tensor.max():.4f}]")
            logger.info(f"  Mean: {tensor.mean():.4f}")
            logger.info(f"  Std: {tensor.std():.4f}")
            
            # Check for constant features
            if len(tensor.shape) == 3:  # [samples, pods, features]
                feature_vars = tensor.var(dim=(0,1))  # Variance across samples and pods
            else:  # [samples, features]
                feature_vars = tensor.var(dim=0)  # Variance across samples
            
            low_var_features = (feature_vars < 1e-6).sum().item()
            total_features = feature_vars.numel()
            logger.info(f"  Low variance features: {low_var_features}/{total_features}")
            
            if low_var_features > total_features * 0.5:
                logger.info(f"  ⚠️  TOO MANY CONSTANT FEATURES!")
            
            return {
                'range': (tensor.min().item(), tensor.max().item()),
                'mean': tensor.mean().item(),
                'std': tensor.std().item(),
                'low_var_ratio': low_var_features / total_features
            }
        
        pod_stats = feature_stats(pod_features, "Pod features")
        kv_stats = feature_stats(kv_hit_ratios, "KV hit ratios") 
        req_stats = feature_stats(request_features, "Request features")
        
        return {
            'pod_stats': pod_stats,
            'kv_stats': kv_stats, 
            'req_stats': req_stats
        }
    
    feature_analysis = analyze_features()
    
    # ==========================================================================
    # 3. MODEL ANALYSIS
    # ==========================================================================
    
    logger.info("\n" + "="*50)
    logger.info("3️⃣  MODEL ANALYSIS")
    logger.info("="*50)
    
    def analyze_model():
        # Get model statistics
        total_params = sum(p.numel() for p in agent.policy.parameters())
        trainable_params = sum(p.numel() for p in agent.policy.parameters() if p.requires_grad)
        
        logger.info(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")
        logger.info(f"Dataset size: {data_analysis['data_size']}")
        logger.info(f"Samples per parameter: {data_analysis['data_size'] / total_params:.2f}")
        
        # Rule of thumb: need 10-100 samples per parameter
        if data_analysis['data_size'] / total_params < 10:
            logger.info("  ⚠️  SEVERE OVERFITTING RISK - too few samples per parameter")
        elif data_analysis['data_size'] / total_params < 50:
            logger.info("  ⚠️  MODERATE OVERFITTING RISK")
        else:
            logger.info("  ✅ REASONABLE SAMPLE-TO-PARAMETER RATIO")
        
        # Analyze model weights and gradients
        logger.info(f"\nLayer analysis:")
        for name, param in agent.policy.named_parameters():
            if param.grad is not None:
                weight_norm = param.data.norm().item()
                grad_norm = param.grad.norm().item()
                logger.info(f"  {name}: weight_norm={weight_norm:.4f}, grad_norm={grad_norm:.4f}")
            else:
                weight_norm = param.data.norm().item()
                logger.info(f"  {name}: weight_norm={weight_norm:.4f}, grad_norm=None")
        
        return {
            'total_params': total_params,
            'samples_per_param': data_analysis['data_size'] / total_params
        }
    
    model_analysis = analyze_model()
    
    # ==========================================================================
    # 4. PREDICTION ANALYSIS
    # ==========================================================================
    
    logger.info("\n" + "="*50)
    logger.info("4️⃣  PREDICTION ANALYSIS")
    logger.info("="*50)
    
    def analyze_predictions():
        agent.policy.eval()
        with torch.no_grad():
            # Get predictions on evaluation data
            pod_features = eval_data['pod_features_with_staleness'][:100].to(device)
            kv_hit_ratios = eval_data['kv_hit_ratios'][:100].to(device)
            request_features = eval_data['request_features'][:100].to(device)
            true_actions = eval_data['actions'][:100].to(device)
            
            # ADD THESE PRINTS HERE:
            logger.info(f"Context diversity check:")
            logger.info(f"Request features: min={request_features.min():.3f}, max={request_features.max():.3f}, std={request_features.std():.3f}")
            logger.info(f"Pod features: min={pod_features.min():.3f}, max={pod_features.max():.3f}, std={pod_features.std():.3f}")
            logger.info(f"First 5 request contexts:\n{request_features[:5]}")

            # logger.info(f"All 100 test samples:")
            # logger.info(f"Sample | Request Features (3) | Pod Features (7x8) | KV Ratios (7x1) | True Action")
            # for i in range(100):
            #     req = request_features[i].cpu().numpy()
            #     pod = pod_features[i].cpu().numpy().flatten()  # Flatten 7x8 to 56 values
            #     kv = kv_hit_ratios[i].cpu().numpy().flatten()  # Flatten 7x1 to 7 values
            #     true_act = true_actions[i].item()
                
            #     logger.info(f"{i:3} | {req} | {pod} | {kv} | {true_act}")

            # # Get raw logits and probabilities
            # logits = agent.policy.network(torch.cat([
            #     pod_features.view(pod_features.shape[0], -1),
            #     kv_hit_ratios.view(kv_hit_ratios.shape[0], -1),
            #     request_features
            # ], dim=1))

            # Get logits from the new network architecture
            pod_scores = []
            batch_size = pod_features.shape[0]
            num_pods = pod_features.shape[1]

            # Combine pod features and kv ratios
            combined_pod_features = torch.cat([pod_features, kv_hit_ratios], dim=2)
            expanded_request = request_features.unsqueeze(1).expand(-1, num_pods, -1)
            full_features = torch.cat([combined_pod_features, expanded_request], dim=2)
            reshaped_features = full_features.view(batch_size * num_pods, -1)

            # Get scores from pod_scorer
            raw_scores = agent.policy.pod_scorer(reshaped_features)
            logits = raw_scores.view(batch_size, num_pods)
            
            action_probs = agent.policy(pod_features, kv_hit_ratios, request_features)
            pred_actions = torch.argmax(action_probs, dim=1)
            
            logger.info(f"Logits analysis (100 samples):")
            logger.info(f"  Range: [{logits.min():.4f}, {logits.max():.4f}]")
            logger.info(f"  Mean: {logits.mean():.4f}")
            logger.info(f"  Std: {logits.std():.4f}")
            
            # Check if logits are extreme
            logit_range = logits.max().item() - logits.min().item()
            if logit_range > 20:
                logger.info(f"  ⚠️  EXTREME LOGIT RANGE ({logit_range:.2f}) - explains overconfidence!")
            elif logit_range > 10:
                logger.info(f"  ⚠️  LARGE LOGIT RANGE ({logit_range:.2f}) - may cause overconfidence")
            else:
                logger.info(f"  ✅ REASONABLE LOGIT RANGE ({logit_range:.2f})")
            
            logger.info(f"\nProbability analysis:")
            logger.info(f"  Min prob: {action_probs.min():.6f}")
            logger.info(f"  Max prob: {action_probs.max():.6f}")
            logger.info(f"  Mean max prob (confidence): {action_probs.max(dim=1)[0].mean():.4f}")
            
            # Entropy analysis
            entropy = torch.distributions.Categorical(action_probs).entropy()
            logger.info(f"  Mean entropy: {entropy.mean():.4f}")
            logger.info(f"  Max possible entropy: {np.log(action_probs.shape[1]):.4f}")
            logger.info(f"  Relative entropy: {entropy.mean().item() / np.log(action_probs.shape[1]):.4f}")
            
            if entropy.mean().item() < 0.1:
                logger.info(f"  ⚠️  EXTREMELY LOW ENTROPY - model is overconfident!")
            
            # Action distribution
            pred_counts = torch.bincount(pred_actions, minlength=7).cpu()
            true_counts = torch.bincount(true_actions, minlength=7).cpu()
            
            logger.info(f"\nAction distribution (100 samples):")
            logger.info(f"  Predicted: {pred_counts.numpy()}")
            logger.info(f"  True:      {true_counts.numpy()}")
            
            # Check for action collapse
            most_frequent_pred = pred_counts.max().item()
            if most_frequent_pred > 80:  # >80% of predictions are same action
                logger.info(f"  ⚠️  ACTION COLLAPSE - model predicts same action {most_frequent_pred}% of time!")
            
            return {
                'logit_range': logit_range,
                'mean_confidence': action_probs.max(dim=1)[0].mean().item(),
                'mean_entropy': entropy.mean().item(),
                'action_collapse': most_frequent_pred > 80
            }
    
    prediction_analysis = analyze_predictions()
    
    # ==========================================================================
    # 5. BASELINE COMPARISON
    # ==========================================================================
    
    logger.info("\n" + "="*50)
    logger.info("5️⃣  BASELINE COMPARISON")
    logger.info("="*50)
    
    # ==========================================================================
    # 6. ROOT CAUSE DIAGNOSIS
    # ==========================================================================
    
    logger.info("\n" + "="*50)
    logger.info("DIAGNOSIS")
    logger.info("="*50)
    
    def diagnose_root_cause(data_analysis, feature_analysis, model_analysis, prediction_analysis):
        issues = []
        severity_score = 0
        
        # Check data issues
        if data_analysis['reward_gap'] < 0.01:
            issues.append("❌ CRITICAL: Extremely weak reward signal - model can't learn meaningful differences")
            severity_score += 3
        elif data_analysis['reward_gap'] < 0.05:
            issues.append("⚠️  WARNING: Weak reward signal - makes learning difficult")
            severity_score += 1
        
        if data_analysis['imbalance_ratio'] > 5:
            issues.append("⚠️  WARNING: Severe class imbalance - model biased toward majority class")
            severity_score += 1
        
        # Check feature issues
        total_low_var_ratio = (feature_analysis['pod_stats']['low_var_ratio'] + 
                              feature_analysis['kv_stats']['low_var_ratio'] + 
                              feature_analysis['req_stats']['low_var_ratio']) / 3
        
        if total_low_var_ratio > 0.5:
            issues.append("⚠️  WARNING: Many constant/low-variance features - limited discriminative power")
            severity_score += 1
        
        # Check model issues
        if model_analysis['samples_per_param'] < 10:
            issues.append("❌ CRITICAL: Severe overfitting risk - too few samples per parameter")
            severity_score += 3
        elif model_analysis['samples_per_param'] < 50:
            issues.append("⚠️  WARNING: Moderate overfitting risk")
            severity_score += 1
        
        # Check prediction issues
        if prediction_analysis['logit_range'] > 20:
            issues.append("❌ CRITICAL: Extreme logit range - causing overconfidence")
            severity_score += 3
        elif prediction_analysis['logit_range'] > 10:
            issues.append("⚠️  WARNING: Large logit range - may cause overconfidence")
            severity_score += 1
        
        if prediction_analysis['mean_entropy'] < 0.1:
            issues.append("❌ CRITICAL: Extremely low entropy - model is overconfident")
            severity_score += 2
        
        if prediction_analysis['action_collapse']:
            issues.append("❌ CRITICAL: Action collapse - model only predicts one action")
            severity_score += 2
        
        # Print diagnosis
        if not issues:
            logger.info("✅ No major issues detected")
        else:
            logger.info("Issues found:")
            for issue in issues:
                logger.info(f"  {issue}")
        
        logger.info(f"\nSeverity score: {severity_score}/15")
        
        # Provide recommendations
        logger.info(f"\n📋 RECOMMENDED ACTIONS:")
        
        if severity_score >= 8:
            logger.info("🚨 CRITICAL ISSUES - Major changes needed:")
            if data_analysis['reward_gap'] < 0.01:
                logger.info("  1. Check reward calculation - differences too small to learn")
                logger.info("  2. Consider different reward metrics or data collection")
            if model_analysis['samples_per_param'] < 10:
                logger.info("  3. Collect much more data OR drastically simplify model")
                logger.info("  4. Try linear/logistic regression first")
        
        elif severity_score >= 4:
            logger.info("⚠️  MODERATE ISSUES - Focused fixes needed:")
            if prediction_analysis['logit_range'] > 10:
                logger.info("  1. Add strong regularization (weight decay, dropout)")
                logger.info("  2. Use temperature scaling (T=10-20)")
                logger.info("  3. Add gradient clipping (max_norm=0.1)")
            if prediction_analysis['mean_entropy'] < 0.1:
                logger.info("  4. Increase entropy regularization significantly")
                logger.info("  5. Use label smoothing")
        
        else:
            logger.info("✅ MINOR ISSUES - Fine-tuning needed:")
            logger.info("  1. Adjust hyperparameters")
            logger.info("  2. Try temperature scaling (T=3-5)")
        
        return {
            'issues': issues,
            'severity_score': severity_score
        }
    
    # def diagnose_root_cause(data_analysis, feature_analysis, model_analysis, prediction_analysis):
    diagnosis = diagnose_root_cause(data_analysis, feature_analysis, model_analysis, prediction_analysis)
    
    # ==========================================================================
    # 7. SUMMARY REPORT
    # ==========================================================================
    
    logger.info("\n" + "="*80)
    logger.info("📊 SUMMARY REPORT")
    logger.info("="*80)
    
    logger.info(f"Dataset: {data_analysis['data_size']} samples, {data_analysis['imbalance_ratio']:.1f}x imbalance")
    logger.info(f"Reward signal: {data_analysis['reward_gap']:.4f} gap")
    logger.info(f"Model: {model_analysis['total_params']:,} params, {model_analysis['samples_per_param']:.1f} samples/param")
    logger.info(f"Predictions: {prediction_analysis['logit_range']:.1f} logit range, {prediction_analysis['mean_confidence']:.1%} confidence")
    
    return {
        'data_analysis': data_analysis,
        'feature_analysis': feature_analysis,
        'model_analysis': model_analysis,
        'prediction_analysis': prediction_analysis,
        'diagnosis': diagnosis
    }

def analyze_pod_identity_learning(agent, eval_data, output_dir):
    """
    Analyze how the model learns pod identity and preferences
    """
    logger.info("🔍 ANALYZING POD IDENTITY LEARNING")
    logger.info("=" * 60)
    
    agent.policy.eval()
    with torch.no_grad():
        # Use evaluation data
        sample_size = min(200, len(eval_data['actions']))
        indices = torch.randperm(len(eval_data['actions']))[:sample_size]
        
        pod_features = eval_data['pod_features_with_staleness'][indices].to(device)
        kv_hit_ratios = eval_data['kv_hit_ratios'][indices].to(device)
        request_features = eval_data['request_features'][indices].to(device)
        true_actions = eval_data['actions'][indices]
        rewards = eval_data['rewards'][indices]
        
        batch_size = pod_features.shape[0]
        num_pods = pod_features.shape[1]
        feature_dim = pod_features.shape[2]
        
        logger.info(f"Analyzing {sample_size} samples with {num_pods} pods, {feature_dim} features per pod")
        
        # 1. ANALYZE POD FEATURE PATTERNS
        logger.info("\n1️⃣ POD FEATURE PATTERNS:")
        
        # Calculate average features per pod across all samples
        avg_features_per_pod = pod_features.mean(dim=0)  # [num_pods, feature_dim]
        
        for pod_idx in range(num_pods):
            pod_avg_features = avg_features_per_pod[pod_idx].cpu().numpy()
            logger.info(f"  Pod {pod_idx} avg features: {pod_avg_features[:5]}...")  # First 5 features
            
            # Check for distinctive features
            feature_range = pod_avg_features.max() - pod_avg_features.min()
            logger.info(f"    Feature range: {feature_range:.4f}")
        
        # 2. ANALYZE MODEL'S RAW SCORING
        logger.info("\n2️⃣ MODEL RAW SCORING ANALYSIS:")
        
        # Get raw scores before softmax
        combined_pod_features = torch.cat([pod_features, kv_hit_ratios], dim=2)
        expanded_request = request_features.unsqueeze(1).expand(-1, num_pods, -1)
        full_features = torch.cat([combined_pod_features, expanded_request], dim=2)
        reshaped_features = full_features.view(batch_size * num_pods, -1)
        
        raw_scores = agent.policy.pod_scorer(reshaped_features)
        raw_scores = raw_scores.view(batch_size, num_pods)
        
        # Analyze score patterns
        avg_scores_per_pod = raw_scores.mean(dim=0)
        std_scores_per_pod = raw_scores.std(dim=0)
        
        logger.info("Average raw scores per pod:")
        for pod_idx in range(num_pods):
            logger.info(f"  Pod {pod_idx}: μ={avg_scores_per_pod[pod_idx]:.4f}, σ={std_scores_per_pod[pod_idx]:.4f}")
        
        score_spread = avg_scores_per_pod.max() - avg_scores_per_pod.min()
        logger.info(f"\nScore spread across pods: {score_spread:.4f}")
        
        # 3. IDENTIFY MOST/LEAST FAVORED PODS
        logger.info("\n3️⃣ POD PREFERENCES:")
        
        most_favored_idx = torch.argmax(avg_scores_per_pod).item()
        least_favored_idx = torch.argmin(avg_scores_per_pod).item()
        
        logger.info(f"Most favored pod: {most_favored_idx} (score: {avg_scores_per_pod[most_favored_idx]:.4f})")
        logger.info(f"Least favored pod: {least_favored_idx} (score: {avg_scores_per_pod[least_favored_idx]:.4f})")
        
        # Get final probabilities
        action_probs = agent.policy(pod_features, kv_hit_ratios, request_features)
        avg_probs = action_probs.mean(dim=0)
        
        logger.info("\nAverage selection probabilities:")
        for pod_idx in range(num_pods):
            logger.info(f"  Pod {pod_idx}: {avg_probs[pod_idx]:.3f} ({avg_probs[pod_idx]*100:.1f}%)")
        
        # 4. CORRELATION WITH HISTORICAL PERFORMANCE
        logger.info("\n4️⃣ CORRELATION WITH HISTORICAL REWARDS:")
        
        # Calculate historical performance per pod
        pod_performance = {}
        for pod_idx in range(num_pods):
            pod_mask = true_actions == pod_idx
            if pod_mask.sum() > 0:
                pod_rewards = rewards[pod_mask]
                pod_performance[pod_idx] = {
                    'count': pod_mask.sum().item(),
                    'avg_reward': pod_rewards.mean().item(),
                    'std_reward': pod_rewards.std().item() if len(pod_rewards) > 1 else 0.0
                }
            else:
                pod_performance[pod_idx] = {'count': 0, 'avg_reward': 0.0, 'std_reward': 0.0}
        
        logger.info("Historical pod performance:")
        for pod_idx in range(num_pods):
            perf = pod_performance[pod_idx]
            current_prob = avg_probs[pod_idx].item()
            logger.info(f"  Pod {pod_idx}: {perf['count']} samples, reward μ={perf['avg_reward']:.4f}, current prob={current_prob:.3f}")
        
        # Calculate correlation between historical rewards and current preferences
        hist_rewards = [pod_performance[i]['avg_reward'] for i in range(num_pods)]
        current_probs = avg_probs.cpu().numpy()
        
        if len(set(hist_rewards)) > 1:  # Only if there's variance
            correlation = np.corrcoef(hist_rewards, current_probs)[0, 1]
            logger.info(f"Correlation calculation:")
            logger.info(f"  hist_rewards: {hist_rewards}")  
            logger.info(f"  current_probs: {current_probs}")
            logger.info(f"  correlation: {correlation}")

        # ADD THE CALIBRATION CHECK HERE:
        logger.info("\n" + "="*50)
        logger.info("CALIBRATION CHECK:")
        logger.info("="*50)
        for pod_idx in range(num_pods):
            historical_reward = pod_performance[pod_idx]['avg_reward']
            current_prob = avg_probs[pod_idx].item()
            logger.info(f"Pod {pod_idx}: reward={historical_reward:.3f}, prob={current_prob:.3f}")

        # Print sorted by reward (best to worst)
        sorted_pods = sorted(range(num_pods), key=lambda x: pod_performance[x]['avg_reward'], reverse=True)
        logger.info("\nPods sorted by performance (best to worst):")
        for pod_idx in sorted_pods:
            reward = pod_performance[pod_idx]['avg_reward']
            prob = avg_probs[pod_idx].item()
            logger.info(f"  Pod {pod_idx}: {reward:.3f} → {prob:.1%}")
        logger.info("="*50)

        # 5. FEATURE IMPORTANCE ANALYSIS
        logger.info("\n5️⃣ FEATURE IMPORTANCE ANALYSIS:")
        
        # Analyze which features differ most between high/low performing pods
        high_perf_pods = [i for i in range(num_pods) if pod_performance[i]['avg_reward'] > np.mean(hist_rewards)]
        low_perf_pods = [i for i in range(num_pods) if pod_performance[i]['avg_reward'] < np.mean(hist_rewards)]
        
        if high_perf_pods and low_perf_pods:
            high_features = avg_features_per_pod[high_perf_pods].mean(dim=0)
            low_features = avg_features_per_pod[low_perf_pods].mean(dim=0)
            feature_diff = torch.abs(high_features - low_features)
            
            # Find most discriminative features
            top_features = torch.topk(feature_diff, min(5, len(feature_diff)))
            
            logger.info("Most discriminative features (high vs low performers):")
            for i, (diff, idx) in enumerate(zip(top_features.values, top_features.indices)):
                logger.info(f"  Feature {idx.item()}: difference = {diff.item():.4f}")
                logger.info(f"    High performers avg: {high_features[idx].item():.4f}")
                logger.info(f"    Low performers avg: {low_features[idx].item():.4f}")
        
        # 6. TEMPORAL ANALYSIS
        logger.info("\n6️⃣ PREDICTION CONSISTENCY:")
        
        # Check how consistently the model picks the same pods
        predictions = torch.argmax(action_probs, dim=1)
        pred_counts = torch.bincount(predictions, minlength=num_pods)
        
        logger.info("Prediction distribution:")
        for pod_idx in range(num_pods):
            count = pred_counts[pod_idx].item()
            percentage = count / sample_size * 100
            logger.info(f"  Pod {pod_idx}: {count}/{sample_size} ({percentage:.1f}%)")
        
        # Check for over-concentration
        max_concentration = pred_counts.max().item() / sample_size
        if max_concentration > 0.6:
            dominant_pod = torch.argmax(pred_counts).item()
            logger.info(f"  ⚠️  OVER-CONCENTRATION: Pod {dominant_pod} gets {max_concentration*100:.1f}% of predictions")
        
        # 7. SAVE DETAILED ANALYSIS
        analysis_results = {
            'avg_scores_per_pod': avg_scores_per_pod.cpu().numpy().tolist(),
            'avg_probs_per_pod': avg_probs.cpu().numpy().tolist(),
            'pod_performance': pod_performance,
            'correlation': correlation if 'correlation' in locals() else 0.0,
            'prediction_distribution': pred_counts.cpu().numpy().tolist(),
            'score_spread': score_spread.item(),
            'max_concentration': max_concentration,
            'most_favored_pod': most_favored_idx,
            'least_favored_pod': least_favored_idx
        }
        
        with open(f"{output_dir}/pod_identity_analysis.json", 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        logger.info(f"\n💾 Detailed analysis saved to {output_dir}/pod_identity_analysis.json")
        
        return analysis_results


# Enhanced version of your existing evaluation function
def enhanced_evaluation_with_identity_analysis(agent, eval_data, output_dir):
    """Enhanced evaluation that includes pod identity analysis"""
    
    # Regular evaluation
    metrics = evaluate_agent(agent, eval_data)
    
    # Pod identity analysis
    identity_analysis = analyze_pod_identity_learning(agent, eval_data, output_dir)
    
    # Combine results
    enhanced_metrics = {**metrics, 'identity_analysis': identity_analysis}
    
    return enhanced_metrics



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