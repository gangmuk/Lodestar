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
    def __init__(self, state_dim, action_dim, hidden_dim, custom_weight_initialization):
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
        if custom_weight_initialization:
            # self._initialize_weights()
            self._static_weight_initialization()
            
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
    
    def _initialize_weights(self):
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

    # def _initialize_weights(self):
    #     """Xavier/Glorot initialization with layer-specific strategies"""
    #     for name, module in self.named_modules():
    #         if isinstance(module, nn.Linear):
    #             # Get layer position info
    #             is_output_layer = (module.out_features == 1)
    #             is_first_layer = ('0' in name)  # First layer in sequential
                
    #             if is_output_layer:
    #                 # Output layer: smaller weights for stability
    #                 torch.nn.init.xavier_uniform_(module.weight, gain=0.1)
    #                 torch.nn.init.constant_(module.bias, 0.0)
    #             elif is_first_layer:
    #                 # First layer: slightly smaller to prevent saturation
    #                 torch.nn.init.xavier_uniform_(module.weight, gain=0.8)
    #                 torch.nn.init.constant_(module.bias, 0.01)
    #             else:
    #                 # Hidden layers: standard Xavier initialization
    #                 torch.nn.init.xavier_uniform_(module.weight, gain=1.0)
    #                 torch.nn.init.constant_(module.bias, 0.01)

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
        
        logger.info(f"explore_mask: {explore_mask.cpu().numpy()}")
        logger.info(f"greedy_actions: {greedy_actions.cpu().numpy()}")
        logger.info(f"random_actions: {random_actions.cpu().numpy()}")
        logger.info(f"actions: {actions.cpu().numpy()}")
        logger.info(f"action_probs: {action_probs.cpu().numpy()}")
        logger.info(f"log_probs: {log_probs.cpu().numpy()}")
        
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
        self.policy = FixedPolicyNetwork(state_dim, action_dim,  self.hyperparameters['hidden_dim'],  self.hyperparameters['custom_weight_initialization']).to(device)

        if HYPERPARAMETERS.get('deterministic_training', False):
            optim_seed = HYPERPARAMETERS['training_seed'] + 1000  # Different from model seed
            torch.manual_seed(optim_seed)
            logger.info(f"🔧 Creating optimizer with deterministic seed: {optim_seed}")
        

        # Optimizer with weight decay for regularization
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr= self.hyperparameters['lr'], weight_decay= self.hyperparameters['weight_decay'])
        
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
def train(encoded_data_dir, model_output_dir, HYPERPARAMETERS):
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
            # Adjust learning rate for online learning (typically lower)
            online_lr = HYPERPARAMETERS['lr'] * 0.1  # 10x lower learning rate
            for param_group in agent.optimizer.param_groups:
                param_group['lr'] = online_lr
            logger.info(f"Adjusted learning rate to {online_lr} for online learning")
        except Exception as e:
            logger.error(f"Error loading pretrained model: {e}")
            logger.info("Starting training from scratch")

    # Use fewer epochs for online learning
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
        logger.info("Example: python simplified_contextual_bandit.py encoded_data/")# feature_normalization.py
"""
Shared feature normalization logic for both offline training and online inference.
This module centralizes all normalization logic to ensure consistency.
"""

import pandas as pd
import numpy as np
import pickle
from logger import logger
from typing import Tuple
import os
import csv
import json

class RunningStats:
    """Maintains running mean and standard deviation for feature normalization"""
    def __init__(self, feature_names):
        self.count = 0
        self.mean = None
        self.sum_sq_diff = None
        self.std = None
        self.min = None
        self.max = None
        if feature_names == None:
            logger.error("RunningStats initialized with None feature_names, setting to empty list")
            assert False
        self.feature_names = feature_names
        self.values = []
        
    def update_stats_incrementally(self, new_data):
        if new_data is None or len(new_data) == 0:
            logger.error("Received empty data for RunningStats.update, skipping")
            return
        new_data = np.array(new_data, dtype=np.float64)
        new_count = len(new_data)
        old_mean = self.mean
        old_var = self.sum_sq_diff
        old_std = self.std
        old_min = self.min
        old_max = self.max
        if self.count == 0: # The very first update
            self.mean = np.mean(new_data, axis=0)
            self.sum_sq_diff = np.var(new_data, axis=0) * new_count
            self.count = new_count
            self.std = np.sqrt(self.sum_sq_diff / new_count)
            self.min = np.min(new_data, axis=0) # Initialize min
            self.max = np.max(new_data, axis=0) # Initialize max
            logger.info(f"The very first RunningStats.update call for {self.feature_names}. Initialized running stats with {new_count} samples")
            logger.info(f"qwer, {self.feature_names}, new_count={new_count}, total_count={self.count} samples, old_mean={old_mean}, old_std={old_std}, old_var={old_var}, old_min={old_min}, old_max={old_max}, new_mean={self.mean}, new_std={self.std}, new_var={self.sum_sq_diff}, new_min={self.min}, new_max={self.max}")
            return
        batch_mean = np.mean(new_data, axis=0)
        batch_var = np.var(new_data, axis=0) * new_count
        new_count = len(new_data)
        new_total = self.count + new_count
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * new_count / new_total
        self.sum_sq_diff = self.sum_sq_diff + batch_var + delta**2 * self.count * new_count / new_total
        self.std = np.sqrt(self.sum_sq_diff / new_total)
        self.count = new_total
        logger.info(f"asdf, {self.feature_names}, new_count={new_count}, total_count={self.count} samples, old_mean={old_mean}, old_std={old_std}, old_var={old_var}, new_mean={self.mean}, new_std={self.std}, new_var={self.sum_sq_diff}")
        
    def normalize(self, data):
        if self.count == 0:
            logger.error(f"{self.feature_names}: No statistics available. normalization cannot be performed.")
            assert False
        mean = self.mean
        std = self.std
        
        # Handle zero standard deviation case (constant features)
        if np.any(std == 0) or np.any(np.isclose(std, 0, atol=1e-10)):
            logger.warning(f"{self.feature_names}: Zero standard deviation detected (std={std}). Returning zero-centered data.")
            return np.zeros_like(data)
        
        # Check for NaN in std
        if np.any(np.isnan(std)):
            logger.error(f"{self.feature_names}: NaN detected in standard deviation: {std}")
            return np.zeros_like(data)
        
        # Perform normalization
        normalized = (data - mean) / std
        
        # Verify result doesn't contain NaN
        if np.any(np.isnan(normalized)):
            logger.error(f"{self.feature_names}: Normalization produced NaN values. mean={mean}, std={std}")
            return np.zeros_like(data)
        
        return normalized
    
class PerFeatureRunningStats:
    def __init__(self):
        self.feature_stats = {}  # Dict[feature_name, RunningStats]
        self.CONFIG = None

    def write_stats_to_file(self, feature_normalization_stats_file):
        # csv_filename = feature_normalization_stats_file.replace('.pkl', '.csv')
        with open(feature_normalization_stats_file, 'w') as f:
            f.write('feature_name,stats_type,value\n')
            for feature_name, stats in self.feature_stats.items():
                f.write(f'{feature_name},count,{stats.count}\n')
                mean_val = stats.mean.item() if hasattr(stats.mean, 'item') else stats.mean
                std_val = stats.std.item() if hasattr(stats.std, 'item') else stats.std
                f.write(f'{feature_name},mean,{mean_val}\n')
                f.write(f'{feature_name},std,{std_val}\n')
        logger.info(f"Saved per-feature statistics for {len(self.feature_stats)} features to {feature_normalization_stats_file}")
        
    @classmethod
    def create_new_empty_instance(cls):
        return cls()

    @classmethod
    def create_new_instance_with_stats_file(cls, feature_normalization_stats_file):
        if not os.path.exists(feature_normalization_stats_file):
            logger.error(f"Feature normalization stats file {feature_normalization_stats_file} does not exist.")
            assert False
        instance = cls()
        try:
            with open(feature_normalization_stats_file, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                temp_feature_data = {}
                for row in reader:
                    feature_name = row['feature_name']
                    stats_type = row['stats_type']
                    value_str = row['value']
                    logger.debug(f"Loading normalization stats, feature_name={feature_name}, stats_type={stats_type}, value={value_str}")
                    if feature_name not in temp_feature_data:
                        temp_feature_data[feature_name] = {'count': 0, 'mean': None, 'std': None, 'feature_names': feature_name}
                    
                    if stats_type == 'count':
                        temp_feature_data[feature_name]['count'] = int(value_str)
                    elif stats_type == 'mean':
                        try:
                            # Attempt to load as JSON array, otherwise treat as scalar
                            temp_feature_data[feature_name]['mean'] = np.array(json.loads(value_str))
                        except json.JSONDecodeError:
                            temp_feature_data[feature_name]['mean'] = float(value_str)
                    elif stats_type == 'std':
                        try:
                            # Attempt to load as JSON array, otherwise treat as scalar
                            temp_feature_data[feature_name]['std'] = np.array(json.loads(value_str))
                        except json.JSONDecodeError:
                            temp_feature_data[feature_name]['std'] = float(value_str)
                    else:
                        logger.error(f"Unknown stats_type {stats_type} for feature {feature_name} in {feature_normalization_stats_file}")
                        assert False
            for feature_name, stats_data in temp_feature_data.items():
                stats = RunningStats(feature_names=stats_data['feature_names'])
                stats.count = stats_data['count']
                stats.mean = stats_data['mean']
                stats.std = stats_data['std']
                stats.sum_sq_diff = stats_data['std'] ** 2 * stats_data['count'] if stats_data['std'] is not None else None
                instance.feature_stats[feature_name] = stats
            logger.info(f"Loaded per-feature statistics for {len(instance.feature_stats)} features from {feature_normalization_stats_file}")
        except Exception as e:
            logger.error(f"Error loading statistics file {feature_normalization_stats_file}: {e}. Expected per-feature CSV format.")
            try:
                with open(feature_normalization_stats_file, 'r') as f:
                    content = f.read()
                    logger.error(f"Content of {feature_normalization_stats_file}:\n{content}")
                logger.error("Please ensure the file is in the correct per-feature CSV format.")
            except FileNotFoundError:
                logger.error(f"Error: The file '{feature_normalization_stats_file}' was not found after initial check.")
            except Exception as e_inner:
                logger.error(f"An unexpected error occurred trying to read file content: {e_inner}")

            assert False
        logger.debug("Per-feature statistics loaded:")
        for feature_name, stats in instance.feature_stats.items():
            logger.debug(f"{feature_name}: count={stats.count}, mean={stats.mean}, std={stats.std}")
        return instance
    
    @property
    def count(self):
        """Return total count across all features (for compatibility)"""
        if not self.feature_stats:
            return 0
        return max(stats.count for stats in self.feature_stats.values())
    
    def get_feature_names(self):
        """Get list of all feature names with statistics"""
        return list(self.feature_stats.keys())

def _get_normalizable_features(processed_df):
    request_features = ['input_tokens', 'output_tokens', 'total_tokens']
    pod_features = [
        col for col in processed_df.columns 
        if col.startswith('pod_') 
        and processed_df[col].dtype in ['float64', 'int64'] 
        and 'gpu_model' not in col
    ]
    return request_features + pod_features

def _normalize_single_feature(processed_df, feature, stats_instance, is_training, request_id=None):
    log_prefix = f"request_id,{request_id}," if request_id else ""
    if feature not in processed_df.columns:
        logger.error(f"{log_prefix}Feature {feature} not found in DataFrame")
        assert False
        
    if is_training:
        feature_std = processed_df[feature].values.std()
        
        # Initialize stats if needed (for both constant and non-constant features)
        if feature not in stats_instance.feature_stats:
            stats_instance.feature_stats[feature] = RunningStats(feature_names=feature)
        
        if feature_std == 0 or np.isclose(feature_std, 0, atol=1e-10):
            # Handle constant feature
            feature_data = processed_df[feature].values.reshape(-1, 1)
            
            # Set stats manually for constant feature
            stats_instance.feature_stats[feature].count = len(feature_data)
            stats_instance.feature_stats[feature].mean = np.mean(feature_data, axis=0)
            stats_instance.feature_stats[feature].std = np.array([0.0])  # Mark as constant
            stats_instance.feature_stats[feature].sum_sq_diff = 0.0  # No variance
            
            logger.info(f"⚪ {feature}, Saved as constant feature (std={feature_std:.6f}, value={stats_instance.feature_stats[feature].mean})")
            
            # Add to config tracking
            stats_instance.CONFIG.setdefault("CONSTANT_FEATURES", set()).add(feature)
            stats_instance.CONFIG["NUM_CONSTANT_FEATURES"] = len(stats_instance.CONFIG.get("CONSTANT_FEATURES", set()))
            return  # Skip normalization but stats are saved
            
        # Check for NaN values in the feature
        if np.any(np.isnan(processed_df[feature].values)):
            logger.error(f"❌ {feature}: Contains NaN values before normalization")
            assert False
    
        # Normal feature processing (non-constant) - stats already exist
        logger.info(f"🔍 {feature}, Normalizing. Variance is high (std: {processed_df[feature].values.std():.3f})")
        stats_instance.CONFIG.setdefault("FEATURES_NORMALIZED", set()).add(feature)
        stats_instance.CONFIG["NUM_FEATURES_NORMALIZED"] = len(stats_instance.CONFIG["FEATURES_NORMALIZED"])
        
        # Rest of your existing training code...
        feature_data = processed_df[feature].values.reshape(-1, 1)
        prev_std = processed_df[feature].values.std()
        prev_min = processed_df[feature].values.min()
        prev_max = processed_df[feature].values.max()
        prev_mean = processed_df[feature].values.mean()
        
        stats_instance.feature_stats[feature].update_stats_incrementally(feature_data)
        print(f"Updated stats for {feature}: count={stats_instance.feature_stats[feature].count}, mean={stats_instance.feature_stats[feature].mean}, std={stats_instance.feature_stats[feature].std}, var={stats_instance.feature_stats[feature].sum_sq_diff}")
        
        
        # Verify computed std is valid
        computed_std = stats_instance.feature_stats[feature].std
        if np.any(computed_std == 0) or np.any(np.isnan(computed_std)):
            logger.warning(f"⚠️  {feature}: Invalid computed std ({computed_std}), skipping normalization")
            return
        
        # Apply normalization
        normalized_feature = stats_instance.feature_stats[feature].normalize(feature_data)
        
        # Verify normalized data doesn't contain NaN
        if np.any(np.isnan(normalized_feature)):
            logger.error(f"❌ {feature}: Normalization produced NaN values, skipping")
            return
        
        processed_df[feature] = normalized_feature.flatten()
        
        new_std = processed_df[feature].std()
        if new_std <= 0.5:
            logger.warning(f"⚠️  Post-normalization variance too low for {feature} (std: {new_std:.3f})")
        logger.info(f"✅ {feature}, Normalize. prev std: {prev_std:.3f} new std: {new_std:.3f}")
        logger.info(f"✅ {feature}, Normalize. prev min: {prev_min:.3f} new min: {normalized_feature.min():.3f}")
        logger.info(f"✅ {feature}, Normalize. prev max: {prev_max:.3f} new max: {normalized_feature.max():.3f}")
        logger.info(f"✅ {feature}, Normalize. prev mean: {prev_mean:.3f} new mean: {normalized_feature.mean():.3f}")
        
    else:  # Inference
        if feature not in stats_instance.feature_stats:
            logger.error(f"{log_prefix}Feature {feature} not found in normalization stats")
            logger.error(f"Available features: {list(stats_instance.feature_stats.keys())}")
            logger.error(f"This indicates a training/inference feature mismatch - not a constant feature issue")
            assert False
            
        # Check if this was a constant feature during training
        if hasattr(stats_instance.feature_stats[feature], 'std') and np.allclose(stats_instance.feature_stats[feature].std, 0):
            logger.warning(f"{log_prefix}{feature} was constant during training (value={stats_instance.feature_stats[feature].mean}) - skipping normalization")
            return  # Don't normalize constant features
        
        # Add this line:
        feature_data = processed_df[feature].values.reshape(-1, 1)
        
        # Apply normalization using pre-computed stats
        normalized_feature = stats_instance.feature_stats[feature].normalize(feature_data)
        processed_df[feature] = normalized_feature.flatten()

def normalize_features_for_training(processed_df, stats_instance: PerFeatureRunningStats) -> pd.DataFrame:
    target_features = _get_normalizable_features(processed_df)
    logger.info(f"🔍 Normalizing features: {target_features}")
    
    logger.info("🔍 DEBUGGING FEATURES BEFORE NORMALIZATION:")
    for feature in target_features:
        if feature in processed_df.columns:
            values = processed_df[feature].values
            logger.info(f"{feature}: min={values.min()}, max={values.max()}, std={values.std():.6f}, has_nan={np.any(np.isnan(values))}")
            unique_vals = np.unique(values)
            if len(unique_vals) <= 5:
                logger.info(f"{feature}: unique values = {unique_vals}")
            elif len(unique_vals) <= 20:
                logger.info(f"{feature}: {len(unique_vals)} unique values, range = [{unique_vals.min()}, {unique_vals.max()}]")
        else:
            logger.warning(f"{feature}: NOT FOUND in DataFrame")
            
    
    # Check all features exist (matches original)
    for feature in target_features:
        assert feature in processed_df.columns
    
    # Normalize each feature and update stats
    for feature in target_features:
        _normalize_single_feature(processed_df, feature, stats_instance, is_training=True)
    
    # Apply feature amplification (batch approach like original)
    amplified_count = 0
    if stats_instance.CONFIG.get("FEATURE_AMPLIFICATION", False) and stats_instance.CONFIG.get("ENABLE_POD_NORMALIZATION", False) and stats_instance.CONFIG.get("SIGNAL_AMPLIFICATION_DEGREE", 1.0) > 1.0:
        critical_features = ['running_requests', 'waiting_requests', 'decode_tokens', 'prefill_tokens']
        pod_features = [col for col in processed_df.columns if col.startswith('pod_')]
        for feature in pod_features:
            if any(critical in feature for critical in critical_features):
                if feature in processed_df.columns:
                    processed_df[feature] = processed_df[feature] * stats_instance.CONFIG["SIGNAL_AMPLIFICATION_DEGREE"]
                    stats_instance.CONFIG.setdefault("FEATURES_AMPLIFIED", set()).add(feature)
                    stats_instance.CONFIG["NUM_FEATURES_AMPLIFIED"] = len(stats_instance.CONFIG["FEATURES_AMPLIFIED"])
                    amplified_count += 1
                    logger.info(f"📈 Amplified critical feature: {feature} by {stats_instance.CONFIG['SIGNAL_AMPLIFICATION_DEGREE']}%, min: {processed_df[feature].min()}, max: {processed_df[feature].max()}, mean: {processed_df[feature].mean()}")
    
    # Apply reward amplification
    processed_df = try_reward_amplification(processed_df, stats_instance.CONFIG)
    
    logger.info(f"✅ FEATURE PROCESSING COMPLETE:")
    return processed_df

def normalize_features_for_inference(processed_df: pd.DataFrame, stats_instance: PerFeatureRunningStats, request_id: str) -> pd.DataFrame:
    ## Not sure we really need to copy....
    # df_copy = processed_df
    df_copy = processed_df.copy()
    target_features = _get_normalizable_features(df_copy)
    if stats_instance.count == 0:
        logger.error(f"request_id,{request_id},No normalization statistics available for inference")
        assert False
    for feature in target_features:
        _normalize_single_feature(df_copy, feature, stats_instance, is_training=False, request_id=request_id)
        if feature in stats_instance.CONFIG.get("FEATURES_AMPLIFIED", set()):
            if feature in df_copy.columns:
                df_copy[feature] = df_copy[feature] * stats_instance.CONFIG['SIGNAL_AMPLIFICATION_DEGREE']
                logger.info(f"request_id,{request_id},Amplified critical feature {feature} after normalization")
            else:
                logger.error(f"request_id,{request_id},Feature {feature} not found in DataFrame for amplification")
                assert False
    return df_copy


def try_reward_amplification(df: pd.DataFrame, CONFIG) -> pd.DataFrame:
    if 'reward' in df.columns:
        rewards = df['reward'].values
        logger.info("\n🎯 REWARD ENGINEERING")
        logger.info("=" * 30)
        logger.info(f"Original rewards: range=[{rewards.min():.3f}, {rewards.max():.3f}], std={rewards.std():.3f}")
        reward_gap = rewards.max() - rewards.min()
        if reward_gap < CONFIG["REWARD_AMPLIFICATION_THRESHOLD"]:
            logger.info(f"Reward gap is too small: {reward_gap:.2f}, 📈 Applying reward amplification ({CONFIG['REWARD_AMPLIFICATION_THRESHOLD']})")
            reward_mean = rewards.mean()
            df['reward'] = reward_mean + (rewards - reward_mean) * CONFIG["REWARD_AMPLIFICATION_DEGREE"]
            new_rewards = df['reward'].values
            logger.info(f"Amplified rewards: range=[{new_rewards.min():.3f}, {new_rewards.max():.3f}], std={new_rewards.std():.3f}")
        else:
            logger.info("✅ Reward signal already strong enough")
    return df


def create_new_instance_with_stats_file(feature_normalization_stats_file: str) -> PerFeatureRunningStats:
    return PerFeatureRunningStats.create_new_instance_with_stats_file(feature_normalization_stats_file)

def create_new_empty_instance() -> PerFeatureRunningStats:
    return PerFeatureRunningStats.create_new_empty_instance()

def get_stats_instance(CONFIG, feature_normalization_stats_file=None):
    if feature_normalization_stats_file is not None and not os.path.exists(feature_normalization_stats_file):
        logger.error(f"Feature normalization stats file {feature_normalization_stats_file} does not exist.")
        assert False
    if feature_normalization_stats_file is not None:
        if not os.path.exists(feature_normalization_stats_file):
            logger.error(f"Feature normalization stats file {feature_normalization_stats_file} does not exist. Creating new empty instance.")
            assert False
        logger.info(f"Creating stats instance from {feature_normalization_stats_file}")
        stats_instance = create_new_instance_with_stats_file(feature_normalization_stats_file)
    else:
        ## offline training path
        logger.info(f"{feature_normalization_stats_file} does not exist. Creating stats instance EMPTY one.")
        stats_instance =  create_new_empty_instance()
        
    stats_instance.CONFIG = CONFIG
    return stats_instanceimport logging
import os

INCLUDE_GPU_IN_FEATURE = False


# Configure logging
def setup_logging():
    logging_level = os.environ.get("LOG_LEVEL", "INFO")
    
    # Add %(filename)s to the format string to include the file name
    # print function name as well
    logging.basicConfig(
        level=logging_level,
        # format="%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s:%(lineno)d - %(message)s",
        format="%(filename)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("llm_router.log")
        ]
    )

    # logging.basicConfig(level=getattr(logging, logging_level), format='%(asctime)s - %(levelname)s - %(filename)s - %(message)s')

    
    
    logger = logging.getLogger("llm_router")
    return logger

# Create and export a common logger instance
logger = setup_logging()# offline_routing_agent.py

from flask import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import os
import logging
import time
import sys
import encoding
import simpler_contextual_bandit
import preprocess
import pickle
import threading
import argparse
import random_forest
import torch
import feature_normalization
import model_and_data_analysis_helper
from logger import logger, INCLUDE_GPU_IN_FEATURE
from kubernetes import client, config
import shutil
import re
import csv
import utils.utils as utils
import random
import hashlib


def set_all_seeds(seed=42):
    """Set seeds for all sources of randomness to ensure reproducible results."""
    # Python's random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch CPU operations
    torch.manual_seed(seed)
    
    # PyTorch GPU operations (if using CUDA)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Python hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Make PyTorch operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set PyTorch to use deterministic algorithms where possible
    if hasattr(torch, 'use_deterministic_algorithms'):
        torch.use_deterministic_algorithms(True, warn_only=True)
    
    print(f"All seeds set to {seed} for reproducible results")

set_all_seeds(42)

RL_MODEL_HYPERPARAMETERS = {
    'model_type': 'simplified',
    'hidden_dim': 32, # 256,
    'batch_size': 32,
    'lr': 0.01, # 0.001
    'weight_decay': 0.0001,
    
    'exploration_rate': 0.0,
    'explore': False,
    
    'training_epochs': 10, # 5,
    'max_updates_per_epoch': 100, # 1000000000
    'eval_interval': 10,
    'custom_weight_initialization': True,
    'entropy_bonus_factor': 0.01,
    'learning_every_x_iter': 5,
    'per_learn_reward_normalization': False,
    'normalization': {
        "SIGNAL_AMPLIFICATION_DEGREE": 1.0,  # 1.5
        "REWARD_AMPLIFICATION_DEGREE": 2.0,
        "REWARD_AMPLIFICATION_THRESHOLD": 0.5,
        "STD_THRESHOLD_FOR_REQ_FEAT_NORMALIZATION": 0.1,
        "STD_THRESHOLD_FOR_POD_FEAT_NORMALIZATION": 0.1,
        "FEATURES_NORMALIZED": set(),
        "NUM_FEATURES_NORMALIZED": 0,
        "FEATURE_AMPLIFICATION": False,
        "FEATURES_AMPLIFIED": set(),
        "NUM_FEATURES_AMPLIFIED": 0,
    },
    'dataset_analysis': None,
    'deterministic_training': True,
    'training_seed': 42,
}

# Global variables (simplified for offline use)
NUM_TRAINS = 0
MODEL_UPDATED = False
TRAINING_DATA_UPDATED = False

TOTAL_NUM_DATA = 0
MIN_NUM_TRAINING_DATA = 500
LOCK_TRAINING_DATA = threading.Lock()
stats_instance = None
request_features_train = ['input_tokens', 'output_tokens', 'total_tokens']
# request_features_reward = ['ttft', 'avg_tpot', 'e2e_latency']

def static_hash(value: str) -> str:
    hash_object = hashlib.sha256(value.encode())
    return hash_object.hexdigest()[:8]

def write_to_file(log_data, raw_data):
    with open(raw_data, "w") as log_file:
        for request_id, log_message in log_data.items():
            log_file.write(f"{log_message}\n")
    logger.info(f"Successfully wrote {len(log_data)} entries to {raw_data}")

# def read_csv_data(csv_file):
#     logger.info(f"Reading data from {csv_file}")
#     try:
#         df = pd.read_csv(csv_file)
#         if 'log_message' in df.columns:
#             log_messages = df['log_message'].tolist()
#         elif len(df.columns) == 1:
#             # Single column, assume it's log messages
#             log_messages = df.iloc[:, 0].tolist()
#         else:
#             logger.error(f"CSV file must have a 'log_message' column or be a single column file")
#             return None
#     except:
#         try:
#             with open(csv_file, 'r') as f:
#                 log_messages = [line.strip() for line in f if line.strip()]
#         except Exception as e:
#             logger.error(f"Error reading file {csv_file}: {e}")
#             return None
#     cleaned_messages = []
#     for i, log_message in enumerate(log_messages):
#         if log_message and log_message.strip():
#             clean_message = log_message.strip()
#             if i < 3:
#                 logger.info(f"Original message {i}: {clean_message[:150]}...")
#             bracket_pos = clean_message.rfind('] ')
#             if bracket_pos != -1:
#                 clean_message = clean_message[bracket_pos + 2:]
#             if not clean_message.startswith('**@latency_metrics@'):
#                 metrics_pos = clean_message.find('**@latency_metrics@')
#                 if metrics_pos != -1:
#                     clean_message = clean_message[metrics_pos:]
#             if clean_message.startswith('**@latency_metrics@'):
#                 cleaned_messages.append(clean_message)
#                 # Debug: show cleaned message for first few entries
#                 if i < 3:
#                     logger.info(f"Cleaned message {i}: {clean_message[:150]}...")
#             else:
#                 logger.warning(f"Skipping malformed log message {i}: {log_message[:100]}...")
#     log_data = {}
#     for i, log_message in enumerate(cleaned_messages):
#         log_data[f"request_{i}"] = log_message
#     logger.info(f"Successfully read {len(log_data)} log messages from {csv_file} (cleaned from {len(log_messages)} raw entries)")
#     if log_data:
#         first_key = list(log_data.keys())[0]
#         sample_message = log_data[first_key]
#         logger.info(f"Sample cleaned message: {sample_message[:200]}...")
#     return log_data

def read_csv_data(log_file):
    """Simple function to read log entries from a text file in deterministic order"""
    from collections import OrderedDict
    
    log_data = OrderedDict()
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        request_count = 0
        for line_num, line in enumerate(lines):
            line = line.strip()
            if line and '**@latency_metrics@' in line:
                # Remove the log prefix (everything up to and including '] ')
                bracket_pos = line.rfind('] ')
                if bracket_pos != -1:
                    clean_line = line[bracket_pos + 2:]
                else:
                    clean_line = line
                
                # Use line number to ensure deterministic ordering
                log_data[f"request_{request_count}"] = clean_line
                request_count += 1
        
        print(f"Successfully read {len(log_data)} log entries from {log_file}")
        print(f"Entries are in the exact order they appeared in the file (lines processed: {len(lines)})")
        return log_data
        
    except Exception as e:
        print(f"Error reading file {log_file}: {e}")
        return None

def train_model(args, ENCODED_DATA_DIR):
    global NUM_TRAINS, MODEL_UPDATED, TRAINING_DATA_UPDATED, TOTAL_NUM_DATA 
    if TRAINING_DATA_UPDATED and TOTAL_NUM_DATA > MIN_NUM_TRAINING_DATA:
        training_start_time = time.time()
        logger.info(f"Starting {NUM_TRAINS}th training of routing agent")
        try:
            if args.model == "random_forest":
                random_forest.train(ENCODED_DATA_DIR)
            elif args.model == "simpler_contextual_bandit":
                # set_all_seeds(RL_MODEL_HYPERPARAMETERS['training_seed'])
                # if not verify_training_determinism(
                #     ENCODED_DATA_DIR, 
                #     f"{args.model_dir}_test", 
                #     RL_MODEL_HYPERPARAMETERS
                # ):
                #     print("❌ Training is not deterministic - fixing required!")
                #     return
                # else:
                #     print("✅ Training determinism verified!")
                
                set_all_seeds(RL_MODEL_HYPERPARAMETERS['training_seed'])
                simpler_contextual_bandit.train(ENCODED_DATA_DIR, args.model_dir, RL_MODEL_HYPERPARAMETERS)
            else:
                logger.error(f"Unknown model type: {args.model}")
                assert False
            MODEL_UPDATED = True
            TRAINING_DATA_UPDATED = False
            NUM_TRAINS += 1
            logger.info(f"Successfully completed {NUM_TRAINS-1}th training of routing agent, took {time.time() - training_start_time} seconds")
            return True
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            logger.error(f"Error training model: {str(e)}")
            logger.error(f"Traceback: {error_traceback}")
            assert False
    else:
        logger.info(f"Not enough training data available (TOTAL_NUM_DATA: {TOTAL_NUM_DATA}), skipping training")
        assert False


def test_inference(args, log_message, request_id):
    global NUM_TRAINS, MODEL_UPDATED, stats_instance
    set_all_seeds(42)
    if NUM_TRAINS == 0:
        logger.warning("No trained model available, please train first")
        return None
    handle_infer_start_time = time.time()
    processed_df, _, sorted_all_pod_ids, _ = preprocess.main(None, log_message, args.ttft_slo, args.avg_tpot_slo, RL_MODEL_HYPERPARAMETERS)
    preprocess_overhead = time.time() - handle_infer_start_time
    original_pod_choice = processed_df['selected_pod'].iloc[0] if len(processed_df) > 0 else None
    normalized_df = feature_normalization.normalize_features_for_inference(processed_df, stats_instance, request_id)
    encode_start_time = time.time()
    tensor_dataset, _ = encoding.encode_for_inference(sorted_all_pod_ids, normalized_df, request_features_train, RL_MODEL_HYPERPARAMETERS)
    handle_infer_total_total_encoding_overhead = time.time() - encode_start_time
    infer_from_tensor_start_time = time.time()
    if args.model == "random_forest":
        result, _ = random_forest.infer_from_tensor(
            tensor_data=tensor_dataset, 
            exploration_enabled=True, 
            exploration_rate=RL_MODEL_HYPERPARAMETERS['exploration_rate'], 
            model_updated=MODEL_UPDATED
    )
    elif args.model == "simpler_contextual_bandit":
        result, _ = simpler_contextual_bandit.infer_from_tensor(
            tensor_data=tensor_dataset, 
            request_id=request_id,
            model_updated=MODEL_UPDATED,
            HYPERPARAMETERS=RL_MODEL_HYPERPARAMETERS,
        )
    if MODEL_UPDATED:
        logger.info("Model updated flag consumed, resetting to False")
        MODEL_UPDATED = False
    handle_infer_total_total_infer_from_tensor_overhead = time.time() - infer_from_tensor_start_time
    selected_pod_index = result['selected_pod_index']
    if selected_pod_index >= len(sorted_all_pod_ids):
        logger.warning(f"Selected pod index {selected_pod_index} out of range, defaulting to first pod")
        selected_pod_index = 0
    selected_pod = sorted_all_pod_ids[selected_pod_index]
    handle_infer_total_overhead = time.time() - handle_infer_start_time
    prediction_matches = (selected_pod == original_pod_choice) if original_pod_choice else None
    result_summary = {
        "selected_pod": selected_pod,
        "original_pod_choice": original_pod_choice,
        "pod_probabilities": result['pod_probabilities'],
        "prediction_matches": prediction_matches,
        "confidence": result['confidence'],
        "total_inference_time_ms": handle_infer_total_overhead * 1000,
        "preprocess_time_ms": preprocess_overhead * 1000,
        "encoding_time_ms": handle_infer_total_total_encoding_overhead * 1000,
        "inference_time_ms": handle_infer_total_total_infer_from_tensor_overhead * 1000,
    }
    
    # Enhanced logging with match/mismatch status
    if original_pod_choice:
        match_status = "✅ MATCH" if prediction_matches else "❌ MISMATCH"
        logger.info(f"Inference result: predicted={selected_pod}, original={original_pod_choice}, {match_status}, confidence={result['confidence']:.4f}")
    else:
        logger.info(f"Inference result: predicted={selected_pod}, original=UNKNOWN, confidence={result['confidence']:.4f}")

    return result_summary

def process_training_data(args, train_data, stats_instance, ENCODED_DATA_DIR):
    global NUM_TRAINS, MODEL_UPDATED, TRAINING_DATA_UPDATED, TOTAL_NUM_DATA
    flush_start_time = time.time()
    logger.info(f"Processing training data with {len(train_data)} entries")
    if not os.path.exists("temp_training_data"):
        os.mkdir("temp_training_data")
    raw_data = "temp_training_data/offline_batch.csv"
    write_to_file(train_data, raw_data)
    ts_preprocess = time.time()
    processed_df, _, sorted_all_pod_ids, _ = preprocess.main(raw_data, "", args.ttft_slo, args.avg_tpot_slo, RL_MODEL_HYPERPARAMETERS)
    processed_df.to_csv(f"{args.data_dir}/processed_data.csv", index=False)
    logger.info(f"Successfully parsed data, took {time.time() - ts_preprocess} seconds")
    
    # update_stats_incrementally is called inside normalize_features_for_training
    processed_df = feature_normalization.normalize_features_for_training(processed_df, stats_instance)
    # processed_df = feature_normalization.try_reward_amplification(processed_df)
    processed_df.to_csv(f"{args.data_dir}/normalized_data.csv", index=False)
    
    # encoding
    ts_encode = time.time()
    encoded_data_output_dir = f"{ENCODED_DATA_DIR}/batch_1"
    encoding.encode_for_train(sorted_all_pod_ids, processed_df, encoded_data_output_dir, request_features_train, RL_MODEL_HYPERPARAMETERS)
    logger.info(f"Successfully encoded data to {encoded_data_output_dir}, took {time.time() - ts_encode} seconds")

    # Verify encoded data
    expected_tensor_path = f"{encoded_data_output_dir}/tensor_dataset.pt"
    train_tensor_path = f"{encoded_data_output_dir}/train/tensor_dataset.pt"
    if os.path.exists(expected_tensor_path):
        logger.info(f"✓ Found tensor dataset at: {expected_tensor_path}")
    elif os.path.exists(train_tensor_path):
        logger.info(f"✓ Found tensor dataset at: {train_tensor_path}")
    TRAINING_DATA_UPDATED = True
    TOTAL_NUM_DATA += len(train_data)
    logger.info(f"Successfully processed {len(train_data)} log messages, took {time.time() - flush_start_time} seconds")
    return True


def ensure_deterministic_data_split(all_data, split_ratio=0.8, seed=42):
    """Ensure consistent train/test split across runs."""
    # Sort by keys to ensure consistent ordering
    sorted_items = sorted(all_data.items())
    all_messages = [msg for _, msg in sorted_items]
    
    # Use seed for any randomization if needed
    random.seed(seed)
    
    split_point = int(len(all_messages) * split_ratio)
    train_messages = all_messages[:split_point]
    test_messages = all_messages[split_point:]
    
    print(f"Deterministic split: {len(train_messages)} train, {len(test_messages)} test")
    print(f"First test message hash: {static_hash(test_messages[0]) if test_messages else 'None'}")
    
    return train_messages, test_messages


# Fixed verification function - remove unused variables
def verify_training_determinism(encoded_data_dir, model_output_dir, HYPERPARAMETERS):
    """Verify that training produces identical results across runs"""
    logger.info("🔍 VERIFYING TRAINING DETERMINISM")
    
    # Train model twice with same settings
    logger.info("Training model #1...")
    set_all_seeds(HYPERPARAMETERS['training_seed'])
    simpler_contextual_bandit.train(encoded_data_dir, f"{model_output_dir}_test1", HYPERPARAMETERS)
    
    logger.info("Training model #2...")
    set_all_seeds(HYPERPARAMETERS['training_seed'])
    simpler_contextual_bandit.train(encoded_data_dir, f"{model_output_dir}_test2", HYPERPARAMETERS)
    
    # Compare final model weights
    model1_path = f"{model_output_dir}_test1/policy.pth"
    model2_path = f"{model_output_dir}_test2/policy.pth"
    
    if os.path.exists(model1_path) and os.path.exists(model2_path):
        weights1 = torch.load(model1_path, map_location='cpu')
        weights2 = torch.load(model2_path, map_location='cpu')
        
        weights_identical = True
        total_diff = 0.0
        
        for key in weights1.keys():
            if not torch.equal(weights1[key], weights2[key]):
                diff = (weights1[key] - weights2[key]).abs().max().item()
                total_diff += diff
                logger.error(f"❌ Weight mismatch in layer: {key}, max_diff: {diff:.8f}")
                weights_identical = False
            else:
                logger.debug(f"✅ Weights identical in layer: {key}")
                logger.info(f"Layer {key} weights are identical. weights1[{key}]: {weights1[key]}, weights2[{key}]: {weights2[key]}")
        
        if weights_identical:
            logger.info("✅ TRAINING DETERMINISM VERIFIED - Identical weights across runs")
        else:
            logger.error(f"❌ TRAINING DETERMINISM FAILED - Total weight difference: {total_diff:.8f}")
        
        # Clean up test models
        import shutil
        try:
            shutil.rmtree(f"{model_output_dir}_test1")
            shutil.rmtree(f"{model_output_dir}_test2")
            logger.info("🧹 Cleaned up test model directories")
        except:
            pass
        
        return weights_identical
    else:
        logger.error("❌ Could not find model files for comparison")
        return False


def main():
    global stats_instance
    parser = argparse.ArgumentParser(description='Offline Routing Agent Training and Testing')
    parser.add_argument('data_file', help='CSV file containing log messages for training')
    parser.add_argument('--skip_training', action='store_true', help='Skip training and only do inference')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='Train/test split ratio')
    parser.add_argument('--model', choices=['random_forest', 'simpler_contextual_bandit'], default='simpler_contextual_bandit', help='Model type to use for training (default: simpler_contextual_bandit)')
    parser.add_argument('--ttft_slo', type=float, help='TTFT SLO threshold for preprocessing', default=1000)
    parser.add_argument('--avg_tpot_slo', type=float, help='Average TPOT SLO threshold for preprocessing', default=50)
    parser.add_argument('--analyze_behavior', action='store_true', help='Analyze what the model has learned through feature sensitivity tests')
    args = parser.parse_args()
    if not os.path.exists(args.data_file):
        logger.error(f"Data file {args.data_file} not found")
        assert False
    
    def replace_pod_ip_with_generalpodid(data_file):
        all_pod_ips_from_training_data = sorted(utils.get_all_pod_ips_from_data_file(data_file))
        if not all_pod_ips_from_training_data:
            logger.error(f"No pod IPs found in data file {data_file}")
            assert False
            
        logger.info(f"🔍 Deterministic pod IP order: {all_pod_ips_from_training_data}")

        pod_ip_to_generalpodid = utils.create_pod_ip_to_generalpodid_mapping(all_pod_ips_from_training_data)
        
        logger.info(f"🔍 Deterministic mapping: {pod_ip_to_generalpodid}")

        
        
        with open(data_file, 'r') as f:
            content = f.read()
        for pod_ip, generalpodid in pod_ip_to_generalpodid.items():
            content = content.replace(pod_ip, generalpodid)
        replaced_data_file = data_file.replace('.csv', '_replaced.csv')
        with open(replaced_data_file, 'w') as f:
            f.write(content)
        logger.info(f"File write {replaced_data_file} with replaced generalpodids")
        return replaced_data_file
    
    replaced_data_file = replace_pod_ip_with_generalpodid(args.data_file)
    all_data = {}
    if os.path.isfile(replaced_data_file):
        data_dir = os.path.dirname(replaced_data_file)
        logger.info(f"data_file is a file: {replaced_data_file}")
        all_data = read_csv_data(replaced_data_file)
    # elif os.path.isdir(args.data_file):
    #     data_dir = args.data_file
    #     logger.info(f"data_file is a directory: {args.data_file}")
    #     for root, dirs, files in os.walk(args.data_file):
    #         for file in files:
    #             if file == "data.csv":
    #                 file_path = os.path.join(root, file)
    #                 logger.info(f"Found data.csv at: {file_path}")
    #                 data = read_csv_data(file_path)
    #                 if data:
    #                     # Merge data dictionaries with unique keys
    #                     for key, value in data.items():
    #                         new_key = f"{os.path.basename(root)}_{key}"
    #                         all_data[new_key] = value
    else:
        logger.error(f"args.data_file must be a file It is a directory or the path does not exist. args.data_file: {replaced_data_file}")
        assert False

    if all_data is None or len(all_data) == 0:
        logger.error("Failed to read data or no valid log messages found")
        return
    
    train_messages, test_messages = ensure_deterministic_data_split(all_data, args.split_ratio)
    test_messages = test_messages[:10]
    
    args.data_dir = data_dir
    args.model_dir = f"{data_dir}/final_model"
    os.makedirs(args.model_dir, exist_ok=True)
    train_data = {f"request_{i}": msg for i, msg in enumerate(train_messages)}
    def extract_request_id(log_message):
        match = re.search(r'@requestID@([^@]+)@', log_message)
        return match.group(1) if match else None
    # test_data = {f"request_{i}": msg for i, msg in enumerate(test_messages)}
    test_data = []
    for msg in test_messages:
        test_data.append({"request_id": extract_request_id(msg), "message": msg})

    ENCODED_DATA_DIR = "encoded_data"
    if not os.path.exists(ENCODED_DATA_DIR):
        os.makedirs(ENCODED_DATA_DIR)
    if os.path.exists(ENCODED_DATA_DIR):
        shutil.rmtree(ENCODED_DATA_DIR)
        os.makedirs(ENCODED_DATA_DIR)
        logger.info(f"Cleaned and recreated {ENCODED_DATA_DIR} for fresh offline training")
    feature_normalization_stats_file = f"{args.model_dir}/feature_normalization_statistics.csv"
    
    if stats_instance is not None:
        logger.error("Using existing stats instance for normalization")
        assert False
    stats_instance = feature_normalization.get_stats_instance(RL_MODEL_HYPERPARAMETERS['normalization'], None)
    process_training_data(args, train_data, stats_instance, ENCODED_DATA_DIR)
    stats_instance.write_stats_to_file(feature_normalization_stats_file)
    model_and_data_analysis_helper.diagnose_training_data_issues(ENCODED_DATA_DIR)
    train_model(args, ENCODED_DATA_DIR)

    # NEW: Behavior Analysis (before regular testing)
    if args.analyze_behavior and test_data and len(test_data) > 0:
        logger.info("=== STARTING BEHAVIOR ANALYSIS ===")
        # model_and_data_analysis_helper.analyze_model_behavior(args, test_data, feature_normalization_stats_file)
        _ = model_and_data_analysis_helper.analyze_detailed_feature_sensitivity(args, test_data, feature_normalization_stats_file)
        logger.info("=== BEHAVIOR ANALYSIS COMPLETED ===")
    
    
    # Test inference
    if test_data and len(test_data) > 0:
        logger.info("=== STARTING TESTING PHASE ===")
        success_count = 0
        match_count = 0
        mismatch_count = 0
        unknown_original_count = 0
        test_count = 10
        selected_pod_list = []
        pod_probabilities_list = []
        message_list = []
        for td in test_data:
            log_message = td['message']
            request_id = td['request_id']
            result = test_inference(args, log_message, request_id)
            selected_pod_list.append(result['selected_pod'])
            message_list.append(log_message)
            
            print()
            print(f"Request_id: {request_id}, Selected Pod: {result['selected_pod']}")
            # print(f"Message: {log_message}")
            print(f"pod_probabilities_list: ", end="")
            for prob in result['pod_probabilities']:
                print(f"{prob:.2f}", end=", ")
            print()
            print()
            
            if result:
                success_count += 1
                if result['prediction_matches'] is True:
                    match_count += 1
                elif result['prediction_matches'] is False:
                    mismatch_count += 1
                else:
                    unknown_original_count += 1
                if result['original_pod_choice']:
                    match_status = "MATCH" if result['prediction_matches'] else "MISMATCH"
                    logger.info(f"  → Predicted: {result['selected_pod']}, Original: {result['original_pod_choice']}, Status: {match_status}, Confidence: {result['confidence']:.3f}")
                else:
                    logger.info(f"  → Predicted: {result['selected_pod']}, Original: UNKNOWN, Confidence: {result['confidence']:.3f}")
            else:
                logger.error(f"✗ Failed inference for {request_id}")
                
        # logger.info("=" * 60)
        # logger.info("=== TESTING SUMMARY ===")
        # logger.info(f"Total tests: {test_count}")
        # logger.info(f"Successful inferences: {success_count}/{test_count} ({success_count/test_count*100:.1f}%)")
        # if match_count + mismatch_count > 0:
        #     accuracy = match_count / (match_count + mismatch_count) * 100
        #     logger.info(f"Prediction accuracy: {match_count}/{match_count + mismatch_count} ({accuracy:.1f}%)")
        #     logger.info(f"  - Matches: {match_count}")
        #     logger.info(f"  - Mismatches: {mismatch_count}")
        # if unknown_original_count > 0:
        #     logger.info(f"  - Unknown original: {unknown_original_count}")
        # logger.info("=" * 60)

if __name__ == "__main__":
    main()#!/usr/bin/env python3

# preprocess.py

import pandas as pd
import numpy as np
import json
import ast
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
import argparse
import sys
import time
from logger import logger, INCLUDE_GPU_IN_FEATURE
import utils.utils as utils
# INCLUDE_GPU_IN_FEATURE = True

def parse_json_columns(df, json_columns):
        for col in json_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        return df

def parse_log_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Check if this is a metrics line
            if "latency_metrics" not in line:
                logger.error(f"Invalid line. {line}")
                assert False
            if "**@" in line:
                line = line.split("**@latency_metrics@")[1]
            parts = line.split('@')
            row = {}
            json_columns = list()
            column_names = list()
            for i in range(0, len(parts), 2):
                if i + 1 >= len(parts):
                    break
                column_name = parts[i]
                column_names.append(column_name)
                value = parts[i+1]
                if value.startswith('{') and value.endswith('}'):
                    try:
                        # NEW: Fix escaped quotes issue - replace \" with " before parsing
                        fixed_value = value.replace('\\"', '"')
                        json_columns.append(column_name)
                        row[column_name] = json.loads(fixed_value)
                    except Exception as e:
                        logger.error(f"Error decoding JSON, column: {column_name}, value: {value}")
                        logger.error(f"Error: {e}")
                        # Since we can't parse it, store as string to avoid losing data
                        row[column_name] = value
                else:
                    try:
                        row[column_name] = int(value)
                    except ValueError:
                        try:
                            row[column_name] = float(value)
                        except ValueError:
                            row[column_name] = value
            data.append(row)
    parsed_df = pd.DataFrame(data, columns=column_names)
    if len(parsed_df) == 0:
        logger.error("No data found in the log file.")
        assert False
    return parsed_df, json_columns

# def normalize_time(df):
#     first_request_start_time = df['request_start_time'].min()
#     df['normalized_start_time'] = df['request_start_time'] - first_request_start_time
#     df['normalized_end_time'] = df['request_end_time'] - first_request_start_time
#     df['normalized_start_time'] /= 1_000_000
#     df['normalized_end_time'] /= 1_000_000
    
    
#     if 'log_window_start_time' in df.columns:
#         df['log_window_start_time'] = df['log_window_start_time'] - first_request_start_time
#         df['log_window_start_time'] /= 1_000_000
#     if 'log_window_end_time' in df.columns:
#         df['log_window_end_time'] = df['log_window_end_time'] - first_request_start_time
#         df['log_window_end_time'] /= 1_000_000

#     df.loc[:, 'normalized_start_time'] = df['normalized_start_time'] - df['normalized_start_time'].min()
#     df.loc[:, 'normalized_end_time'] = df['normalized_end_time'] - df['normalized_start_time'].min()
#     df = df.sort_values(by='normalized_start_time', ascending=True)
#     df['time_bucket'] = df['normalized_start_time'].astype(int)
#     df = df[['normalized_start_time', 'time_bucket', 'normalized_end_time'] + [col for col in df.columns if col != 'normalized_start_time' and col != 'normalized_end_time' and col != 'time_bucket']]
#     df.reset_index(drop=True, inplace=True)
#     return df

def safe_parse_json(json_str):
    """Safely parse Python dictionary-like strings or JSON strings"""
    # If already a dictionary, return as is
    if isinstance(json_str, dict):
        return json_str
    if pd.isna(json_str) or not json_str:
        logger.error(f"ERROR: Empty or NaN JSON string: {str(json_str)}...")
        assert False
    try:
        # Try standard JSON parsing
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        try:
            # Try replacing single quotes with double quotes
            if isinstance(json_str, str):
                return json.loads(json_str.replace("'", '"'))
            else:
                logger.error(f"ERROR: Invalid JSON string: {str(json_str)}...")
                assert False
        except (json.JSONDecodeError, TypeError):
            try:
                # Try using ast.literal_eval for Python dict literals
                if isinstance(json_str, str):
                    return ast.literal_eval(json_str)
                else:
                    logger.error(f"ERROR: Invalid JSON string: {str(json_str)}...")
                    assert False
            except (SyntaxError, ValueError, TypeError):
                logger.error(f"ERROR: Could not parse JSON: {str(json_str)}...")
                assert False

def calculate_ttft_reward(row, ttft_slo):
    try:
        ttft = float(row['ttft'])
        
        if ttft <= 0:
            return 0.5  # Maximum reward for perfect performance
        elif ttft <= ttft_slo:
            # Linear scaling from 0.5 (best) to 0.1 (at threshold)
            return 0.5 - (0.4 * ttft / ttft_slo)
        else:
            # Negative reward scaling with how much it exceeds threshold
            excess_factor = min(1.0, (ttft - ttft_slo) / ttft_slo)
            return -0.1 - (0.4 * excess_factor)
    except (ValueError, TypeError, ZeroDivisionError):
        return -0.5  # Default penalty for invalid data

def calculate_tpot_reward(row, avg_tpot_slo):
    try:
        avg_tpot = float(row['avg_tpot'])
        
        if avg_tpot <= 0:
            return -0.5  # Penalize invalid values
        elif avg_tpot <= avg_tpot_slo:
            # Linear scaling from 0.5 (best) to 0.1 (at threshold)
            return 0.1 + (0.4 * (1 - avg_tpot / avg_tpot_slo))
        else:
            # Negative reward scaling with how much it exceeds threshold
            excess_factor = min(1.0, (avg_tpot - avg_tpot_slo) / avg_tpot_slo)
            return -0.1 - (0.4 * excess_factor)
    except (ValueError, TypeError, ZeroDivisionError):
        return -0.5  # Default penalty for invalid data

def extract_key_pod_metrics(pod_metrics, pod_id):
    """Extract the most relevant metrics for a pod from the pod metrics"""
    if pod_id not in pod_metrics:
        logger.error(f"Error: Pod ID {pod_id} not found in pod metrics.")
        assert False
    return {
        'last_second_avg_ttft_ms': pod_metrics[pod_id]['last_second_avg_ttft_ms'],
        'last_second_avg_tpot_ms': pod_metrics[pod_id]['last_second_avg_tpot_ms'],
        'last_second_p99_ttft_ms': pod_metrics[pod_id]['last_second_p99_ttft_ms'],
        'last_second_p99_tpot_ms': pod_metrics[pod_id]['last_second_p99_tpot_ms'],
        'last_second_total_requests': pod_metrics[pod_id]['last_second_total_requests'],
        'last_second_total_tokens': pod_metrics[pod_id]['last_second_total_tokens'],
        'last_second_total_decode_tokens': pod_metrics[pod_id]['last_second_total_decode_tokens'],
        'last_second_total_prefill_tokens': pod_metrics[pod_id]['last_second_total_prefill_tokens'],
    }


## new
def preprocess_dataset(parsed_df, ttft_slo, avg_tpot_slo, RL_MODEL_HYPERPARAMETERS):
    # Pre-parse all JSON columns once to avoid repeated parsing
    logger.info("Pre-parsing JSON columns...")
    json_columns = [
        'allPodsKvCacheHitRatios', 
        'numInflightRequestsAllPods', 
        'vllmGPUKVCacheUsage', 
        'vllmCPUKVCacheUsage', 
        'vllmNumRequestsRunning', 
        'vllmNumRequestsWaiting', 
        'podMetricsLastSecond', 
        'numPrefillTokensForAllPods', 
        'numDecodeTokensForAllPods',
    ]
    
    json_parse_start_time = time.time()
    for col in json_columns:
        if col in parsed_df.columns:
            sample_val = parsed_df[col].iloc[0]
            if isinstance(sample_val, str):
                parsed_df[col] = parsed_df[col].apply(safe_parse_json)
    json_parse_overhead = time.time() - json_parse_start_time

    # Collect all unique pod IDs in a single pass
    all_pods_set = set()
    logger.info("Collecting all unique pod IDs across the dataset...")
    
    # Vectorized approach - get all unique pods from all relevant columns at once
    for col in ['allPodsKvCacheHitRatios', 'numInflightRequestsAllPods', 'podMetricsLastSecond']:
        if col in parsed_df.columns:
            for data in parsed_df[col]:
                if data:
                    all_pods_set.update(data.keys())
    
    # all_pods = list(all_pods_set) # BUG: NON-DETERMINISTIC ORDERING.  set-to-list conversion can produce different orderings. It affects the computation in _optimized_process_pod_features in encoding.py. This all_pods is returned in this function -> preprocess.main ->  encoding.encode_for_inference/encode_for_train -> encoding.prepare_for_encoding -> LLMRoutingDataProcessor.pod_ids = all_pods -> encoding._optimized_process_pod_features, for pod_idx, pod_id in enumerate(self.pod_ids): feature arrangement depends on the order of all_pods
    # NOTE: Always sort the pod list... maybe it should have never used the set or dictionary when maintaining pods
    all_pods = list(all_pods_set)
    sorted_all_pod_ids = sorted(list(all_pods_set))
    logger.info(f"Identified {len(sorted_all_pod_ids)} pods: {sorted_all_pod_ids}")

    logger.debug(f"Original dataset shape: {parsed_df.shape}")
    logger.debug(f"Columns: {parsed_df.columns.tolist()}")
    
    expected_columns = [
        'requestID', 
        'selectedpod', 
        'ttft', 
        'avg_tpot', 
        'total_decode_time', 
        'e2e',
        'numInputTokens', 
        'numOutputTokens', 
        'numTotalTokens',
        'allPodsKvCacheHitRatios', 
        'numInflightRequestsAllPods',
        'vllmGPUKVCacheUsage', 
        'vllmCPUKVCacheUsage',
        'vllmNumRequestsRunning', 
        'vllmNumRequestsWaiting',
        'podMetricsLastSecond', 
        'numPrefillTokensForAllPods',
        'numDecodeTokensForAllPods',
        # 'GPU_model',
    ]

    expected_last_second_pod_metrics_keys = [
        'last_second_avg_ttft_ms', 
        'last_second_min_ttft_ms', 
        'last_second_max_ttft_ms', 
        'last_second_p50_ttft_ms', 
        'last_second_p90_ttft_ms', 
        'last_second_p95_ttft_ms', 
        'last_second_p99_ttft_ms', 
        'last_second_ttft_samples', 
        'last_second_avg_tpot_ms', 
        'last_second_min_tpot_ms', 
        'last_second_max_tpot_ms', 
        'last_second_p50_tpot_ms', 
        'last_second_p90_tpot_ms', 
        'last_second_p95_tpot_ms', 
        'last_second_p99_tpot_ms', 
        'last_second_tpot_samples', 
        'last_second_total_requests', 
        'last_second_total_decode_tokens', 
        'last_second_total_prefill_tokens', 
        'last_second_total_tokens',
    ]

    column_check_start_time = time.time()

    if INCLUDE_GPU_IN_FEATURE:
        parsed_df['gpu_model_encoded'] = parsed_df['selectedpod'].map(RL_MODEL_HYPERPARAMETERS['pod_ip_to_gpu_model_encoded'])
        unmapped_pods = parsed_df[parsed_df['gpu_model_encoded'].isna()]['selectedpod'].unique()
        if len(unmapped_pods) > 0:
            logger.error(f"CRITICAL: Found unmapped GPU models for pods: {unmapped_pods}")
            unmapped_pods_gpu_models = parsed_df[parsed_df['selectedpod'].isin(unmapped_pods)]['gpu_model_encoded'].unique()
            logger.error(f"Unmapped GPU models: {unmapped_pods_gpu_models}")
            assert False
        parsed_df['gpu_model_encoded'] = parsed_df['gpu_model_encoded'].astype(int)
    
    # Check for missing expected columns
    missing_columns = [col for col in expected_columns if col not in parsed_df.columns]
    if missing_columns:
        logger.error(f"Error: Missing expected columns: {missing_columns}")
        assert False
    
    # Check for unknown columns
    unknown_columns = [col for col in parsed_df.columns if col not in expected_columns]
    if unknown_columns:
        logger.warning(f"Warning: Unused columns: {unknown_columns}")

    # Filter out rows with empty 'podMetricsLastSecond' - vectorized approach
    valid_mask = parsed_df['podMetricsLastSecond'].notna()
    
    # Additional filtering for empty dictionaries - vectorized
    # non_empty_mask = parsed_df['podMetricsLastSecond'].apply(lambda x: x and len(x) > 0 if isinstance(x, dict) else False)

    non_empty_mask = parsed_df['podMetricsLastSecond'].apply(lambda x: isinstance(x, dict) and len(x) > 0)
    
    num_filter = len(parsed_df) - non_empty_mask.sum()
    logger.info(f"Filtered out {num_filter} rows with empty podMetricsLastSecond.")
    
    parsed_df = parsed_df[valid_mask & non_empty_mask].copy()
    column_check_overhead = time.time() - column_check_start_time # 0-4ms

    podmetrics_parse_start_time = time.time()
    # Process first row to check podMetricsLastSecond structure (same as before)
    if 'podMetricsLastSecond' in parsed_df.columns and len(parsed_df) > 0:
        first_row = parsed_df.iloc[0]
        logger.warning(f"WARNING: We are using the first row only to check podMetricsLastSecond structure")
        pod_metrics = first_row['podMetricsLastSecond']  # Already parsed
        logger.debug(f"features in pod_metrics: {pod_metrics.keys()}")
        try:
            logger.debug(f"features in pod_metrics: {pod_metrics[list(pod_metrics.keys())[0]].keys()}")
        except Exception as e:
            logger.error(f"Error: {e}")
            logger.error(f"first_row['podMetricsLastSecond']: {first_row['podMetricsLastSecond']}")
            logger.error(f"pod_metrics: {pod_metrics}")
            logger.error(f"first_row: {first_row}")
            assert False
        if pod_metrics:
            # Check structure for each pod
            for pod_id, metrics in pod_metrics.items():
                logger.debug(f"metrics: {metrics}")
                # Check for missing expected keys
                missing_keys = [key for key in expected_last_second_pod_metrics_keys if key not in metrics]
                if missing_keys:
                    logger.error(f"Error: Missing expected keys in podMetricsLastSecond for pod {pod_id}: {missing_keys}")
                    assert False
                
                # Check for unknown keys
                unknown_keys = [key for key in metrics.keys() if key not in expected_last_second_pod_metrics_keys]
                if unknown_keys:
                    logger.error(f"Error: Found unknown keys in podMetricsLastSecond for pod {pod_id}: {unknown_keys}")
                    assert False
    else:
        logger.error("Error: podMetricsLastSecond column not found in the DataFrame.")
        assert False
    podmetrics_parse_overhead = time.time() - podmetrics_parse_start_time # 0-1ms


    numeric_conversion_start_time = time.time()
    # Convert string columns to appropriate types - vectorized
    numeric_columns = [
        'ttft', 
        'avg_tpot', 
        'total_decode_time', 
        'e2e', 
        'numInputTokens', 
        'numOutputTokens', 
        'numTotalTokens',
    ]
    
    for col in numeric_columns:
        if col in parsed_df.columns:
            parsed_df[col] = pd.to_numeric(parsed_df[col], errors='coerce')
    numeric_conversion_overhead = time.time() - numeric_conversion_start_time # 0-1ms
    

    # Pre-create pod_gpu_models and pod features structure
    if INCLUDE_GPU_IN_FEATURE:
        pod_gpu_models = {pod_id: "NVIDIA-L20" for pod_id in sorted_all_pod_ids}
    
    # Vectorized processing using pandas operations
    logger.info("Processing records in vectorized manner...")
    
    get_value_start_time = time.time()
    # Extract base features
    base_data = {
        'request_id': parsed_df['requestID'].values,
        'selected_pod': parsed_df['selectedpod'].values,
        'input_tokens': parsed_df['numInputTokens'].values,
        'output_tokens': parsed_df['numOutputTokens'].values,
        'total_tokens': parsed_df['numTotalTokens'].values,
        'ttft': parsed_df['ttft'].values,
        'avg_tpot': parsed_df['avg_tpot'].values,
        'e2e_latency': parsed_df['e2e'].values,
    }
    if INCLUDE_GPU_IN_FEATURE:
       base_data['gpu_model_encoded'] = parsed_df['gpu_model_encoded'].values
    
    # Pre-extract all JSON data to avoid repeated parsing
    all_kv_cache = parsed_df['allPodsKvCacheHitRatios'].values
    all_inflight = parsed_df['numInflightRequestsAllPods'].values  
    all_gpu_cache = parsed_df['vllmGPUKVCacheUsage'].values
    all_cpu_cache = parsed_df['vllmCPUKVCacheUsage'].values
    all_running = parsed_df['vllmNumRequestsRunning'].values
    all_waiting = parsed_df['vllmNumRequestsWaiting'].values
    all_prefill = parsed_df['numPrefillTokensForAllPods'].values
    all_decode = parsed_df['numDecodeTokensForAllPods'].values
    all_pod_metrics = parsed_df['podMetricsLastSecond'].values
    
    # Process pod features for all rows at once
    for pod_id in sorted_all_pod_ids:
        # Vectorized extraction for each pod across all rows
        base_data[f"{pod_id}-kv_hit_ratio"] = [data.get(pod_id, 0) for data in all_kv_cache]
        base_data[f"{pod_id}-inflight_requests"] = [data.get(pod_id, 0) for data in all_inflight]
        base_data[f"{pod_id}-gpu_kv_cache"] = [data.get(pod_id, 0) for data in all_gpu_cache]
        base_data[f"{pod_id}-cpu_kv_cache"] = [data.get(pod_id, 0) for data in all_cpu_cache]
        base_data[f"{pod_id}-running_requests"] = [data.get(pod_id, 0) for data in all_running]
        base_data[f"{pod_id}-waiting_requests"] = [data.get(pod_id, 0) for data in all_waiting]
        base_data[f"{pod_id}-prefill_tokens"] = [data.get(pod_id, 0) for data in all_prefill]
        base_data[f"{pod_id}-decode_tokens"] = [data.get(pod_id, 0) for data in all_decode]
        if INCLUDE_GPU_IN_FEATURE:
            base_data[f"{pod_id}-gpu_model"] = ["NVIDIA-L20"] * len(parsed_df)
        
        # Extract key metrics for this pod across all rows
        for metric_key in ['last_second_avg_ttft_ms', 'last_second_avg_tpot_ms', 'last_second_p99_ttft_ms', 
                          'last_second_p99_tpot_ms', 'last_second_total_requests', 'last_second_total_tokens',
                          'last_second_total_decode_tokens', 'last_second_total_prefill_tokens']:
            base_data[f"{pod_id}-{metric_key}"] = [
                metrics.get(pod_id, {}).get(metric_key, 0) for metrics in all_pod_metrics
            ]
    get_value_overhead = time.time() - get_value_start_time # 0ms

    # Pre-calculate all derived values before DataFrame creation
    num_rows = len(base_data['request_id'])

    pod_index_start_time = time.time()
    # Map pod IDs to integer indices for the action space - do this early with numpy arrays
    unique_pods = np.unique(base_data['selected_pod'])
    pod_to_index = {str(pod): idx for idx, pod in enumerate(unique_pods)}
    index_to_pod = {int(idx): str(pod) for pod, idx in pod_to_index.items()}

    # Pre-calculate all derived columns as numpy arrays (much faster than pandas operations)
    selected_pods_array = np.array(base_data['selected_pod'])
    action_values = np.array([pod_to_index[str(pod)] for pod in selected_pods_array])

    ttft_values = np.array(base_data['ttft'], dtype=np.float64)
    tpot_values = np.array(base_data['avg_tpot'], dtype=np.float64)
    pod_index_overhead = time.time() - pod_index_start_time

    # Vectorized reward calculations using numpy (faster than pandas)
    reward_calc_start_time = time.time()
    ttft_rewards = np.where(
        ttft_values <= 0, 
        0.5,  # Maximum reward for perfect performance
        np.where(
            ttft_values <= ttft_slo,
            0.5 - (0.4 * ttft_values / ttft_slo),  # Linear scaling
            -0.1 - (0.4 * np.minimum(1.0, (ttft_values - ttft_slo) / ttft_slo))  # Negative reward
        )
    )

    tpot_rewards = np.where(
        tpot_values <= 0,
        -0.5,  # Penalize invalid values
        np.where(
            tpot_values <= avg_tpot_slo,
            0.1 + (0.4 * (1 - tpot_values / avg_tpot_slo)),  # Linear scaling
            -0.1 - (0.4 * np.minimum(1.0, (tpot_values - avg_tpot_slo) / avg_tpot_slo))  # Negative reward
        )
    )
    reward_calc_overhead = time.time() - reward_calc_start_time

    slo_update_start_time = time.time()
    # Add all the computed columns to base_data before DataFrame creation
    base_data.update({
        'action': action_values,
        'avg_tpot_slo_satisfied': tpot_values <= avg_tpot_slo,
        'avg_ttft_slo_satisfied': ttft_values <= ttft_slo,
        'ttft_reward': ttft_rewards,
        'tpot_reward': tpot_rewards,
        'reward': ttft_rewards + tpot_rewards,
    })
    slo_update_overhead = time.time() - slo_update_start_time

    # Create DataFrame only once with all data
    create_df_start_time = time.time()
    processed_df = pd.DataFrame(base_data)
    create_df_overhead = time.time() - create_df_start_time


    # Replace fillna(0) with a more targeted approach since most values should already be handled
    # Only fill NaN values in specific columns that might have them
    nan_columns = processed_df.columns[processed_df.isnull().any()].tolist()
    if nan_columns:
        processed_df[nan_columns] = processed_df[nan_columns].fillna(0)

    # Save mapping information
    mapping_info = {
        'pod_to_index': pod_to_index,
        'index_to_pod': index_to_pod,
    }
    if INCLUDE_GPU_IN_FEATURE:
        mapping_info['pod_gpu_models'] = pod_gpu_models

    logger.debug(f"Processed dataset shape: {processed_df.shape}")
    logger.debug(f"Processed columns: {processed_df.columns[:10].tolist()}...")

    if INCLUDE_GPU_IN_FEATURE:
        logger.debug("\nPod GPU model mapping:")
        for pod_id, gpu_model in pod_gpu_models.items():
            logger.debug(f"  Pod {pod_id} -> GPU model {gpu_model}")

    ##################################################################

    preprocess_dataset_overhead_summary = {
        'preprocess.preprocess_dataset_json_parse_overhead': json_parse_overhead*1000,
        'preprocess.preprocess_dataset_column_check_overhead': column_check_overhead*1000,
        'preprocess.preprocess_dataset_podmetrics_parse_overhead': podmetrics_parse_overhead*1000,
        'preprocess.preprocess_dataset_numeric_conversion_overhead': numeric_conversion_overhead*1000,
        'preprocess.preprocess_dataset_get_value_overhead': get_value_overhead*1000,
        'preprocess.preprocess_dataset_create_df_overhead': create_df_overhead*1000,
        'preprocess.preprocess_dataset_pod_index_overhead': pod_index_overhead*1000,
        'preprocess.preprocess_dataset_reward_calc_overhead': reward_calc_overhead*1000,
        'preprocess.preprocess_dataset_slo_update_overhead': slo_update_overhead*1000,
    }
    
    return processed_df, mapping_info, sorted_all_pod_ids, preprocess_dataset_overhead_summary


# Optimized version - just replace your existing parse_log_message function with this
def parse_log_message(log_message):
    # Fast check without string operations
    if "latency_metrics" not in log_message:
        logger.error(f"Invalid line. {log_message}")
        return pd.DataFrame(), []
    # Find start position more efficiently
    start_idx = log_message.find("latency_metrics@") + 16
    if start_idx == 15:  # find returned -1
        return pd.DataFrame(), []
    # Split only the relevant part
    parts = log_message[start_idx:].split('@')
    row = {}
    json_columns = []
    # Process pairs directly
    i = 0
    while i < len(parts) - 1:
        key = parts[i]
        if key == "numInputTokens":
            logger.info(f"")
        value = parts[i + 1]
        # Fast JSON detection and parsing
        if value and value[0] == '{' and value[-1] == '}':
            try:
                # Only fix quotes if needed
                if '\\"' in value:
                    value = value.replace('\\"', '"')
                row[key] = json.loads(value)
                json_columns.append(key)
            except Exception as e:
                logger.error(f"Error decoding JSON, column: {key}, value: {value}")
                logger.error(f"Error: {e}")
                row[key] = value
        else:
            # Fast type conversion with better float detection
            if value.isdigit():
                row[key] = int(value)
            elif value.replace('.', '').replace('-', '').isdigit() and value.count('.') == 1:
                # Only convert to float if there's exactly one decimal point
                row[key] = float(value)
            else:
                row[key] = value
        i += 2
    # Create DataFrame only if we have data
    if row:
        pd_df_start_time = time.time()
        df = pd.DataFrame([row])
        pd_df_overhead = time.time() - pd_df_start_time
        logger.debug(f"pd_df_overhead: {pd_df_overhead*1000}ms")

        return df, json_columns
    else:
        return pd.DataFrame(), []

def preprocess_single_row_fast(parsed_df, RL_MODEL_HYPERPARAMETERS):
    row = parsed_df.iloc[0].to_dict()
    if not row.get('podMetricsLastSecond'):
        logger.error("Error: podMetricsLastSecond is missing or empty in the row data.")
        logger.error(f"Row data: {row}")
        assert False
    
    base_features = {
        'request_id': row['requestID'],
        'selected_pod': row['selectedpod'],
        'input_tokens': row['numInputTokens'],
        'output_tokens': row['numOutputTokens'],
        'total_tokens': row['numTotalTokens'],
        'ttft': row['ttft'],
        'avg_tpot': row['avg_tpot'],
        'e2e_latency': row['e2e'],
    }
    if INCLUDE_GPU_IN_FEATURE:
        
        selected_pod_generalpodid = RL_MODEL_HYPERPARAMETERS['pod_ip_to_generalpodid'][row['selectedpod']]
        
        gpu_model_encoded = RL_MODEL_HYPERPARAMETERS['pod_ip_to_gpu_model_encoded'][selected_pod_generalpodid]
        
        base_features['gpu_model_encoded'] = gpu_model_encoded
    
    # Extract metrics directly without repeated dictionary lookups
    kv_cache = row['allPodsKvCacheHitRatios']
    inflight = row['numInflightRequestsAllPods']
    gpu_cache = row['vllmGPUKVCacheUsage']
    cpu_cache = row['vllmCPUKVCacheUsage']
    running = row['vllmNumRequestsRunning']
    waiting = row['vllmNumRequestsWaiting']
    prefill = row['numPrefillTokensForAllPods']
    decode = row['numDecodeTokensForAllPods']
    pod_metrics = row['podMetricsLastSecond']
    ## I am weirdly using podMetricsLastSecond to get pod ids...
    sorted_all_pod_ids = sorted(list(pod_metrics.keys()))
    temp_sorted_all_pod_ids = sorted(list(set(list(kv_cache.keys()) +
                                    list(inflight.keys()) +
                                    list(gpu_cache.keys()) +
                                    list(cpu_cache.keys()) +
                                    list(running.keys()) +
                                    list(waiting.keys()))))
    assert len(temp_sorted_all_pod_ids) == len(sorted_all_pod_ids)
    for i in range(len(temp_sorted_all_pod_ids)):
        assert temp_sorted_all_pod_ids[i] == sorted_all_pod_ids[i], f"Mismatch at index {i}: {temp_sorted_all_pod_ids[i]} != {sorted_all_pod_ids[i]}"
    for pod_id in sorted_all_pod_ids:
        pod_prefix = f"{pod_id}"
        if INCLUDE_GPU_IN_FEATURE:
            if RL_MODEL_HYPERPARAMETERS and 'pod_gpu_mapping' in RL_MODEL_HYPERPARAMETERS:
                if pod_id not in RL_MODEL_HYPERPARAMETERS['pod_gpu_mapping']:
                    logger.error(f"Error: Pod ID {pod_id} not found in RL_MODEL_HYPERPARAMETERS['pod_gpu_mapping']:{RL_MODEL_HYPERPARAMETERS['pod_gpu_mapping']}")
                    assert False
                gpu_model = RL_MODEL_HYPERPARAMETERS['pod_gpu_mapping'][pod_id]
            else:
                assert False
                
        # Direct assignment without intermediate dictionaries
        base_features[f"{pod_prefix}-kv_hit_ratio"] = kv_cache.get(pod_id, 0)
        base_features[f"{pod_prefix}-inflight_requests"] = inflight.get(pod_id, 0)
        base_features[f"{pod_prefix}-gpu_kv_cache"] = gpu_cache.get(pod_id, 0)
        base_features[f"{pod_prefix}-cpu_kv_cache"] = cpu_cache.get(pod_id, 0)
        base_features[f"{pod_prefix}-running_requests"] = running.get(pod_id, 0)
        base_features[f"{pod_prefix}-waiting_requests"] = waiting.get(pod_id, 0)
        base_features[f"{pod_prefix}-prefill_tokens"] = prefill.get(pod_id, 0)
        base_features[f"{pod_prefix}-decode_tokens"] = decode.get(pod_id, 0)
        if INCLUDE_GPU_IN_FEATURE:
            base_features[f"{pod_prefix}-gpu_model"] = gpu_model
        
        # Pod metrics
        pod_metrics_for_pod = pod_metrics.get(pod_id, {})
        base_features[f"{pod_prefix}-last_second_avg_ttft_ms"] = pod_metrics_for_pod.get('last_second_avg_ttft_ms', 0)
        base_features[f"{pod_prefix}-last_second_avg_tpot_ms"] = pod_metrics_for_pod.get('last_second_avg_tpot_ms', 0)
        base_features[f"{pod_prefix}-last_second_p99_ttft_ms"] = pod_metrics_for_pod.get('last_second_p99_ttft_ms', 0)
        base_features[f"{pod_prefix}-last_second_p99_tpot_ms"] = pod_metrics_for_pod.get('last_second_p99_tpot_ms', 0)
        base_features[f"{pod_prefix}-last_second_total_requests"] = pod_metrics_for_pod.get('last_second_total_requests', 0)
        base_features[f"{pod_prefix}-last_second_total_tokens"] = pod_metrics_for_pod.get('last_second_total_tokens', 0)
        base_features[f"{pod_prefix}-last_second_total_decode_tokens"] = pod_metrics_for_pod.get('last_second_total_decode_tokens', 0)
        base_features[f"{pod_prefix}-last_second_total_prefill_tokens"] = pod_metrics_for_pod.get('last_second_total_prefill_tokens', 0)
    
    processed_df = pd.DataFrame([base_features])
    preprocess_dataset_overhead_summary = {}
    
    return processed_df, sorted_all_pod_ids, preprocess_dataset_overhead_summary


def main(input_file, log_message, TTFT_SLO, AVG_TPOT_SLO, RL_MODEL_HYPERPARAMETERS):
    if input_file == None and (log_message == "" or log_message is None):
        logger.error("Error: Both input_file and log_message are empty or None.")
        assert False
    if input_file is not None and log_message != "":
        logger.error("Error: Both input_file and log_message are provided. Please provide only one.")
        assert False
    if input_file is not None:  # Training path
        parsed_df, json_columns = parse_log_file(input_file)
    else:  # Inference path
        ################################################
        label_selector = "model.aibrix.ai/name=llama-3-8b-instruct"
        if 'running_pods' not in RL_MODEL_HYPERPARAMETERS:
            RL_MODEL_HYPERPARAMETERS['running_pods'] = utils.get_running_pods_by_label(label_selector)

        if 'all_pod_ips_from_running_pods' not in RL_MODEL_HYPERPARAMETERS:
            RL_MODEL_HYPERPARAMETERS['all_pod_ips_from_running_pods'] = utils.fetch_running_pod_ips(RL_MODEL_HYPERPARAMETERS['running_pods'])
            print(f"all_pod_ips_from_running_pods: {RL_MODEL_HYPERPARAMETERS['all_pod_ips_from_running_pods']}")

        unique_pod_ips = sorted(RL_MODEL_HYPERPARAMETERS['all_pod_ips_from_running_pods'])

        if 'pod_ip_to_generalpodid' not in RL_MODEL_HYPERPARAMETERS:
            pod_ip_to_generalpodid = utils.create_pod_ip_to_generalpodid_mapping(unique_pod_ips)
            RL_MODEL_HYPERPARAMETERS['pod_ip_to_generalpodid'] = pod_ip_to_generalpodid

        if 'generalpodid_to_gpu_model' not in RL_MODEL_HYPERPARAMETERS:
            generalpodid_to_gpu_model = utils.fetch_generalpodid_to_gpu_model(RL_MODEL_HYPERPARAMETERS['running_pods'], pod_ip_to_generalpodid)
            RL_MODEL_HYPERPARAMETERS['generalpodid_to_gpu_model'] = generalpodid_to_gpu_model

        if 'pod_ip_to_gpu_model' not in RL_MODEL_HYPERPARAMETERS or 'pod_ip_to_gpu_model_encoded' not in RL_MODEL_HYPERPARAMETERS:
            pod_ip_to_gpu_model, pod_ip_to_gpu_model_encoded = utils.create_pod_ip_to_gpu_model_mapping(generalpodid_to_gpu_model, pod_ip_to_generalpodid)
            RL_MODEL_HYPERPARAMETERS['pod_ip_to_gpu_model'] = pod_ip_to_gpu_model
            RL_MODEL_HYPERPARAMETERS['pod_ip_to_gpu_model_encoded'] = pod_ip_to_gpu_model_encoded
        
        logger.info(f"pod_ip_to_generalpodid: {RL_MODEL_HYPERPARAMETERS['pod_ip_to_generalpodid']}")
        logger.info(f"generalpodid_to_gpu_model: {RL_MODEL_HYPERPARAMETERS['generalpodid_to_gpu_model']}")
        logger.info(f"pod_ip_to_gpu_model: {RL_MODEL_HYPERPARAMETERS['pod_ip_to_gpu_model']}")
        logger.info(f"pod_ip_to_gpu_model_encoded: {RL_MODEL_HYPERPARAMETERS['pod_ip_to_gpu_model_encoded']}")
        ################################################
        
        parsed_df, json_columns = parse_log_message(log_message)
    if len(parsed_df) == 0:
        logger.error("No data found after parsing JSON columns.")
        logger.error(f"Log message: {log_message}")
        assert False
    if len(parsed_df) == 1 and input_file is None:
        processed_df, sorted_all_pod_ids, _ = preprocess_single_row_fast(parsed_df, RL_MODEL_HYPERPARAMETERS)
    else:
        # Existing batch processing for training
        # REMOVED: No need for parse_json_columns since JSON is already parsed
        # df = parse_json_columns(df, json_columns)
        processed_df, mapping_info, sorted_all_pod_ids, _ = preprocess_dataset(parsed_df, TTFT_SLO, AVG_TPOT_SLO, RL_MODEL_HYPERPARAMETERS)
    mapping_info_write_start_time = time.time()
    output_file = None
    if input_file is not None:
        try:
            input_dir = os.path.dirname(input_file)
            # Save mapping information
            output_file = os.path.join(input_dir, "processed_dataset.csv")
            mapping_file = output_file.replace('.csv', '_mapping.json')
            if not os.path.exists(mapping_file):
                with open(mapping_file, 'w') as f:
                    try:
                        json.dump(mapping_info, f, indent=2)
                        logger.info("JSON serialization successful")
                    except TypeError as e:
                        logger.error(f"JSON serialization failed: {e}")
                        # Try to identify the problematic part by serializing each part separately
                        logger.info("Trying to identify the problematic part:")
                        try:
                            json.dumps(mapping_info['pod_to_index'])
                            logger.info("pod_to_index serialization: OK")
                        except TypeError as e:
                            logger.error(f"pod_to_index serialization failed: {e}")
                        
                        try:
                            json.dumps(mapping_info['index_to_pod'])
                            logger.info("index_to_pod serialization: OK")
                        except TypeError as e:
                            logger.error(f"index_to_pod serialization failed: {e}")
                        if INCLUDE_GPU_IN_FEATURE:
                            try:
                                json.dumps(mapping_info['pod_gpu_models'])
                                logger.info("pod_gpu_models serialization: OK")
                            except TypeError as e:
                                logger.error(f"pod_gpu_models serialization failed: {e}")
                        raise
                
            logger.info(f"Mapping information saved to {mapping_file}")
            logger.info("\nPod mapping (for action space):")
            for pod, idx in mapping_info['pod_to_index'].items():
                logger.info(f"  Pod {pod} -> Action {idx}")
        except Exception as e:
            logger.error(f"Error processing dataset: {e}")
            assert False

    preprocess_dataset_overhead_summary = {}
    return processed_df, output_file, sorted_all_pod_ids, preprocess_dataset_overhead_summary
#!/usr/bin/env python3

# encoding.py

"""
LLM Request Router - Enhanced Data Preprocessing
-----------------------------------------------
Transforms raw request routing data into structured tensors for transformer-based RL model.
Implements:
- Pod state extraction and normalization
- Expected KV hit ratio isolation for cross-attention
- Request feature extraction
- Metrics-based positional encoding
- Temporal feature handling with staleness indicators
- Request-pod interaction features
- One-hot encoding for categorical features
"""

import sys
import os
import pandas as pd
import numpy as np
from collections import defaultdict
import torch
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle
import logging
import re
import argparse
from datetime import datetime
import time
from logger import logger, INCLUDE_GPU_IN_FEATURE
import json
# INCLUDE_GPU_IN_FEATURE = True


random_seed = 42
np.random.seed(random_seed)
class LLMRoutingDataProcessor:
    """Processes raw LLM request routing data into formatted tensors for RL training.
    
    Implements advanced encoding techniques:
    1. Metrics-based positional encoding for transformer
    2. Cross-attention preparation for KV hit ratio
    3. Temporal feature handling with staleness indicators
    4. Request-pod interaction features
    """
    
    def __init__(self, output_dir):
        """Initialize the data processor.
        
        Args:
            output_dir: Directory to save processed data and statistics
        """
        self.output_dir = output_dir
        
        # Initialize scalers
        self.pod_feature_scaler = StandardScaler()
        self.request_feature_scaler = StandardScaler()
        self.kv_hit_scaler = StandardScaler()
        
        # Track feature metadata
        self.pod_features = []
        self.numeric_request_features = []
        self.categorical_request_features = []
        self.sorted_all_pod_ids = []
        
        # Key metrics for positional encoding
        self.key_metric_names = [
            'running_requests', 'gpu_kv_cache', 'cpu_kv_cache', 'waiting_requests', 'prefill_tokens', 'decode_tokens', 'kv_hit_ratio', 
        ]
        
        # Statistics tracking
        self.feature_stats = {
            'pod_feature_means': None,
            'pod_feature_stds': None,
            'request_feature_means': None,
            'request_feature_stds': None,
            'kv_hit_means': None,
            'kv_hit_stds': None
        }
        
        # Encoders
        self.pod_encoder = None
        self.selected_pod_encoder = None
        
        # Used for _validate_tensor_compatibility
        self._reference_tensor_data = None

        if INCLUDE_GPU_IN_FEATURE:
            self.gpu_models = set()
            self.num_gpu_types = 0


    def analyze_request_features(self, df, request_features_train, request_features_reward):
        """Analyze request features - OPTIMIZED."""
        # Columns to exclude from features
        exclude_cols = set([
            'request_id', 'selected_pod', 'action', 'reward', 
            'ttft_reward', 'tpot_reward', 'ttft_normalized', 'tpot_normalized',
        ] + request_features_reward)
        
        exclude_patterns = ['reward', 'action', 'slo_satisfied', 'normalized']
        
        # OPTIMIZATION: Use set operations for faster filtering
        pod_prefixes = set(f"pod_{pod_id}" for pod_id in self.sorted_all_pod_ids)
        
        candidate_request_features = [
            col for col in df.columns 
            if not any(col.startswith(prefix) for prefix in pod_prefixes)
            and not any(pat in col for pat in exclude_patterns)
            and col not in exclude_cols
        ]
        
        logger.info(f"Request features - Training features: {request_features_train}")
        logger.info(f"Request features - Reward features (excluded from training): {request_features_reward}")
        logger.info(f"Request features - Found {len(candidate_request_features)} candidate columns: {candidate_request_features}")

        # OPTIMIZATION: Vectorized numeric/categorical classification
        numeric_cols = []
        categorical_cols = []
        
        for col in candidate_request_features:
            # Skip columns with too many NaN values
            if df[col].isna().mean() > 0:
                logger.error(f"Request features - {col} has NaN values.")
                assert False
            
            # OPTIMIZATION: Direct dtype check first, then conversion check
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)
            else:
                try:
                    pd.to_numeric(df[col])
                    numeric_cols.append(col)
                except:
                    categorical_cols.append(col)
        
        self.numeric_request_features = numeric_cols
        self.categorical_request_features = categorical_cols
        
        logger.info(f"Request features - number of numeric columns: {len(numeric_cols)}")
        logger.info(f"Request features - number of categorical columns {len(categorical_cols)}")
        if len(numeric_cols) > 0:
            logger.info(f"Request features - numeric features: {numeric_cols}")
        if len(categorical_cols) > 0:
            logger.info(f"Request features - categorical features: {categorical_cols}")

    def encode_pod_ids(self, df):
        """Create encoders for pod IDs - OPTIMIZED."""
        if self.sorted_all_pod_ids:
            # OPTIMIZATION: Pre-convert to numpy array
            sorted_all_pod_ids_np_array = np.array(self.sorted_all_pod_ids).reshape(-1, 1)
            self.pod_encoder = OneHotEncoder(sparse_output=False)
            self.pod_encoder.fit(sorted_all_pod_ids_np_array)

            if 'selected_pod' in df.columns:
                # OPTIMIZATION: Use unique() only once
                selected_pods = df['selected_pod'].dropna().unique()
                selected_pods_array = np.array(selected_pods).reshape(-1, 1)
                self.selected_pod_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                self.selected_pod_encoder.fit(selected_pods_array)
                
                logger.info(f"Encoded {len(selected_pods)} unique selected pods")
        else:
            logger.warning("No pod IDs found, skipping pod encoding")

    def classify_feature_timing(self):
        """Classify feature timing - OPTIMIZED."""
        # OPTIMIZATION: Vectorized classification
        feature_timing = {
            feature: 'historical' if 'last_second' in feature else 'current'
            for feature in self.pod_features
        }
        
        current_features = [f for f, timing in feature_timing.items() if timing == 'current']
        historical_features = [f for f, timing in feature_timing.items() if timing == 'historical']
        
        logger.info(f"Current-time features: {current_features}")
        logger.info(f"historical features: {historical_features}")
        
        # Validation (kept same logic)
        for historical_feat in historical_features:
            if 'last_second' not in historical_feat:
                logger.error(f"Feature {historical_feat} is classified as historical but does not contain 'last_second'")
                assert False
        for current_feat in current_features:
            if 'last_second' in current_feat:
                logger.error(f"Feature {current_feat} is classified as current but contains 'last_second'")
                assert False
                
        return feature_timing

    def prepare_metrics_based_positional_encoding(self, pod_features, feature_indices_map):
        # Find indices of key metrics for positional encoding
        key_metrics_indices = []
        max_feature_dim = pod_features.shape[2]
        
        for metric in self.key_metric_names:
            matching_features = [
                idx for feature, idx in feature_indices_map.items() 
                if metric in feature and idx < max_feature_dim
            ]
            key_metrics_indices.extend(matching_features)
        
        # Filter out any indices that are still out of bounds
        key_metrics_indices = [idx for idx in key_metrics_indices if idx < max_feature_dim]
        
        # If no key metrics found, use a subset of available features
        if not key_metrics_indices and pod_features.shape[2] > 0:
            # Use first few numeric features (excluding one-hot encoded)
            key_metrics_indices = list(range(min(3, pod_features.shape[2])))
        
        # Extract key metrics for positional encoding
        if key_metrics_indices:
            logger.info(f"Using {len(key_metrics_indices)} metrics for positional encoding, indices: {key_metrics_indices}")
            pos_encoding_features = pod_features[:, :, key_metrics_indices]
        else:
            # Fallback if no suitable metrics found
            pos_encoding_features = np.zeros((pod_features.shape[0], pod_features.shape[1], 1))
            logger.warning("No suitable metrics for positional encoding, using zeros")
        
        return pos_encoding_features


    def add_staleness_features(self, pod_features, timestamps, feature_timing, feature_indices_map):
        """Add staleness indicators for historical features - OPTIMIZED."""
        # OPTIMIZATION: Pre-compute historical feature indices
        historical_features = [f for f, timing in feature_timing.items() if timing == 'historical']
        historical_indices = [
            idx for feature, idx in feature_indices_map.items() 
            if feature in historical_features
        ]
        
        if not historical_indices or len(timestamps) == 0 or np.all(timestamps == 0):
            logger.info("No historical features or valid timestamps, skipping staleness")
            staleness_features = np.zeros((pod_features.shape[0], pod_features.shape[1], 1))
            return np.concatenate([pod_features, staleness_features], axis=2)
        
        # OPTIMIZATION: Vectorized staleness calculation
        max_staleness = 60.0
        sorted_indices = np.argsort(timestamps)
        sorted_timestamps = timestamps[sorted_indices]
        time_diffs = np.diff(sorted_timestamps, prepend=sorted_timestamps[0])
        time_diffs = np.maximum(time_diffs, 0)
        
        # OPTIMIZATION: Use advanced indexing for reordering
        staleness = np.zeros_like(timestamps)
        staleness[sorted_indices] = time_diffs
        staleness = np.clip(staleness / max_staleness, 0, 1)
        
        # OPTIMIZATION: Broadcasting instead of loop
        staleness_features = np.broadcast_to(
            staleness[:, np.newaxis, np.newaxis], 
            (pod_features.shape[0], pod_features.shape[1], 1)
        ).copy()
        
        logger.info(f"Added staleness indicator for {len(historical_indices)} historical features")
        return np.concatenate([pod_features, staleness_features], axis=2)


    def prepare_cross_attention_inputs(self, pod_features, kv_hit_ratios):
        """Format inputs for cross-attention between pod features and KV hit ratios.
        
        This separates pod state from KV hit ratios to enable cross-attention
        in the transformer model.
        
        Args:
            pod_features: Normalized pod features [batch, n_pods, feature_dim]
            kv_hit_ratios: Normalized KV hit ratios [batch, n_pods, 1]
            
        Returns:
            Dictionary with query and key/value tensors
        """
        # Ensure kv_hit_ratios has the right shape
        if kv_hit_ratios.shape[2] != 1:
            logger.warning(f"Expected KV hit ratios to have shape [batch, n_pods, 1], got {kv_hit_ratios.shape}")
        
        return {
            'query': pod_features,  # Pod features as query
            'key_value': kv_hit_ratios  # KV hit ratios as key/value
        }


    def create_request_pod_interaction_features(self, request_features, pod_features):
        """Create request-pod interaction features - OPTIMIZED."""
        if request_features.shape[1] == 0:
            logger.warning("No request features available for interaction")
            return None
            
        batch_size, n_pods, _ = pod_features.shape
        
        # OPTIMIZATION: Use numpy broadcasting instead of repeat
        expanded_request = np.broadcast_to(
            request_features[:, np.newaxis, :], 
            (batch_size, n_pods, request_features.shape[1])
        ).copy()
        
        logger.info(f"Created request-pod interaction features with shape {expanded_request.shape}")
        return expanded_request


    def _filter_identity_features(self, pod_features_array, feature_names):
        """
        Remove features that enable pod identity learning.
        Keep only real-time, routing-relevant features.
        """
        # Define which features to KEEP (current state, routing-relevant)
        CURRENT_STATE_FEATURES = [
            'inflight_requests',     # Current load
            'kv_hit_ratio',         # Current cache performance  
            'gpu_kv_cache',         # Current GPU memory usage
            'cpu_kv_cache',         # Current CPU cache usage
            'running_requests',     # Currently processing
            'waiting_requests',     # Currently queued
            'prefill_tokens',       # Current prefill load
            'decode_tokens'         # Current decode load
        ]
        
        # Find indices of features to keep
        keep_indices = []
        kept_features = []
        
        for i, feature_name in enumerate(feature_names):
            if feature_name in CURRENT_STATE_FEATURES:
                keep_indices.append(i)
                kept_features.append(feature_name)
        
        if not keep_indices:
            logger.warning("No current-state features found, keeping all features")
            return pod_features_array, feature_names
        
        # Filter the feature array
        filtered_features = pod_features_array[:, :, keep_indices]
        if len(kept_features) != len(feature_names):
            logger.info(f"Feature masking applied:")
            logger.info(f"  Original features: {len(feature_names)} -> Kept features: {len(kept_features)}")
            logger.info(f"  Kept features: {kept_features}")
            logger.info(f"  Original shape: {pod_features_array.shape} -> New shape: {filtered_features.shape}")
        else:
            logger.debug("No feature masking applied, all features kept")
        
        return filtered_features, kept_features

    def randomize_pod_positions(self, pod_features, kv_hit_ratios):
        """
        Randomize which pod appears in which tensor position for each sample.
        This prevents the model from learning pod identity based on tensor positions.
        
        Args:
            pod_features: [batch_size, num_pods, feature_dim] 
            kv_hit_ratios: [batch_size, num_pods, 1]
        
        Returns:
            Tuple of (shuffled_pod_features, shuffled_kv_hit_ratios)
        """
        batch_size, num_pods = pod_features.shape[:2]
        
        # Create shuffled tensors
        shuffled_pod_features = pod_features.clone()
        shuffled_kv_hit_ratios = kv_hit_ratios.clone()
        
        # Randomize pod order for each sample independently
        for sample_idx in range(batch_size):
            # Generate random permutation for this sample
            perm = torch.randperm(num_pods)
            
            # Apply the same permutation to both tensors
            shuffled_pod_features[sample_idx] = pod_features[sample_idx][perm]
            shuffled_kv_hit_ratios[sample_idx] = kv_hit_ratios[sample_idx][perm]
        
        return shuffled_pod_features, shuffled_kv_hit_ratios


    def _optimized_process_pod_features(self, pod_data, n_samples, overhead_summary, HYPERPARAMETERS):
        if not pod_data:
            logger.error("No pod data in expected format")
            assert False
        
        vectorized_extraction_start_time = time.time()
        
        # Include ALL features we want to potentially keep
        ALL_NUMERIC_FEATURES = [
            'inflight_requests', # index 0
            'gpu_kv_cache', # index 1
            'cpu_kv_cache', # index 2
            'running_requests', # index 3
            'waiting_requests', # index 4
            'prefill_tokens', # index 5
            'decode_tokens', # index 6 ← This is your 5.4486e+04 value
            'kv_hit_ratio' # index 7
        ]
        n_pods = len(self.sorted_all_pod_ids)
        n_numeric = len(ALL_NUMERIC_FEATURES)
        
        # Calculate total feature dimensions including GPU one-hot encoding
        if INCLUDE_GPU_IN_FEATURE:
            gpu_onehot_dim = self.num_gpu_types
            total_feature_dim = n_numeric + gpu_onehot_dim
        else:
            total_feature_dim = n_numeric
        
        all_features_array = np.zeros((n_samples, n_pods, total_feature_dim), dtype=np.float32)

        if INCLUDE_GPU_IN_FEATURE:
            gpu_encoded_per_pod = {}
            for pod_id in self.sorted_all_pod_ids:
                if pod_id not in HYPERPARAMETERS['pod_gpu_id_mapping']:
                    logger.error(f"CRITICAL: Pod {pod_id} not found in pod_gpu_id_mapping!")
                    logger.error(f"Available pods: {list(HYPERPARAMETERS['pod_gpu_id_mapping'].keys())}")
                    assert False, f"Unknown GPU model for pod {pod_id}"
                gpu_model_id = HYPERPARAMETERS['pod_gpu_id_mapping'][pod_id]
                if gpu_model_id < 0 or gpu_model_id >= self.num_gpu_types:
                    logger.error(f"CRITICAL: Invalid GPU model ID {gpu_model_id} for pod {pod_id}")
                    logger.error(f"Expected GPU model ID in range [0, {self.num_gpu_types-1}]")
                    assert False, f"Invalid GPU model ID {gpu_model_id}"
                gpu_encoded_per_pod[pod_id] = gpu_model_id


        ## THIS IS WHERE THE BUG MANIFESTS:
        ## The all_pods order determines how features are arranged in tensors
        # Extract all features into single array
        for pod_idx, pod_id in enumerate(self.sorted_all_pod_ids):
            if pod_id in pod_data:
                pod_features = pod_data[pod_id]
                
                # Hardcoded feature assignments (matching ALL_NUMERIC_FEATURES order)
                if 'inflight_requests' in pod_features:
                    all_features_array[:, pod_idx, 0] = pod_features['inflight_requests'].fillna(0)
                if 'gpu_kv_cache' in pod_features:
                    all_features_array[:, pod_idx, 1] = pod_features['gpu_kv_cache'].fillna(0)
                if 'cpu_kv_cache' in pod_features:
                    all_features_array[:, pod_idx, 2] = pod_features['cpu_kv_cache'].fillna(0)
                if 'running_requests' in pod_features:
                    all_features_array[:, pod_idx, 3] = pod_features['running_requests'].fillna(0)
                if 'waiting_requests' in pod_features:
                    all_features_array[:, pod_idx, 4] = pod_features['waiting_requests'].fillna(0)
                if 'prefill_tokens' in pod_features:
                    all_features_array[:, pod_idx, 5] = pod_features['prefill_tokens'].fillna(0)
                if 'decode_tokens' in pod_features:
                    all_features_array[:, pod_idx, 6] = pod_features['decode_tokens'].fillna(0)
                # KV ratio in main array
                if 'kv_hit_ratio' in pod_features:
                    all_features_array[:, pod_idx, 7] = pod_features['kv_hit_ratio'].fillna(0)
                if INCLUDE_GPU_IN_FEATURE:
                    gpu_model_id = gpu_encoded_per_pod[pod_id]
                    gpu_onehot = np.zeros(gpu_onehot_dim)
                    gpu_onehot[gpu_model_id] = 1
                    all_features_array[:, pod_idx, n_numeric:] = gpu_onehot

        vectorized_extraction_overhead = time.time() - vectorized_extraction_start_time
        
        build_feature_start_time = time.time()
        
        # Separate numeric and GPU features before masking
        numeric_features_only = all_features_array[:, :, :n_numeric]  # First 8 features
        if INCLUDE_GPU_IN_FEATURE:
            gpu_features_only = all_features_array[:, :, n_numeric:]      # Last gpu_onehot_dim features
        
        # Apply masking to numeric features only
        original_features = ALL_NUMERIC_FEATURES.copy()
        filtered_numeric_features, kept_numeric_features = self._filter_identity_features(
            numeric_features_only, original_features
        )
        
        # Combine filtered numeric + all GPU features (GPU features are never masked)
        if INCLUDE_GPU_IN_FEATURE:
            filtered_features_array = np.concatenate([filtered_numeric_features, gpu_features_only], axis=2)
            gpu_feature_names = [f'gpu_model_{i}' for i in range(self.num_gpu_types)]
            kept_features = kept_numeric_features + gpu_feature_names
        else:
            filtered_features_array = filtered_numeric_features
            kept_features = kept_numeric_features
        
        # SOLUTION 1: Always ensure kv_hit_ratio is available separately
        kv_extraction_start_time = time.time()
        
        if 'kv_hit_ratio' in kept_features:
            # Extract KV ratios from filtered array
            kv_index = kept_features.index('kv_hit_ratio')
            kv_hit_norm = filtered_features_array[:, :, kv_index:kv_index+1]  # Keep as [batch, pods, 1]
            pod_kv_hit_array = kv_hit_norm.copy()
            
            # Remove KV from pod features to avoid duplication in model input
            other_indices = [i for i in range(len(kept_features)) if i != kv_index]
            if other_indices:  # Only if there are other features besides kv_hit_ratio
                pod_features_array = filtered_features_array[:, :, other_indices]
                kept_pod_features = [feat for i, feat in enumerate(kept_features) if i != kv_index]
            else:
                # Edge case: only kv_hit_ratio was kept - create minimal pod features
                logger.warning("Only kv_hit_ratio was kept after masking, creating minimal pod features")
                pod_features_array = np.ones((n_samples, n_pods, 1), dtype=np.float32) * 0.5  # Neutral values
                kept_pod_features = ['minimal_feature']
            
            logger.info(f"Extracted KV ratios separately: {kv_hit_norm.shape}")
            logger.info(f"Remaining pod features: {len(kept_pod_features)} features")
            
        else:
            # KV hit ratio was filtered out - this shouldn't happen with our CURRENT_STATE_FEATURES
            logger.error("kv_hit_ratio was filtered out by masking - this should not happen!")
            logger.error("Check your CURRENT_STATE_FEATURES list in _filter_identity_features")
            
            # Create fallback KV tensor and use all filtered features as pod features
            kv_hit_norm = np.zeros((n_samples, n_pods, 1), dtype=np.float32)
            pod_kv_hit_array = kv_hit_norm.copy()
            pod_features_array = filtered_features_array
            kept_pod_features = kept_features
            
            logger.warning("Using fallback: zero KV ratios and all filtered features as pod features")
        
        kv_extraction_overhead = time.time() - kv_extraction_start_time
        
        # APPLY POD RANDOMIZATION HERE
        randomization_start_time = time.time()
        
        # Convert to tensors for randomization
        pod_features_tensor = torch.from_numpy(pod_features_array).float()
        kv_hit_tensor = torch.from_numpy(kv_hit_norm).float()
        
        # # Apply randomization
        # logger.info("Applying pod position randomization...")
        # randomized_pod_features, randomized_kv_hit = self.randomize_pod_positions(
        #     pod_features_tensor, kv_hit_tensor
        # )

        randomized_pod_features = pod_features_tensor
        randomized_kv_hit = kv_hit_tensor
        
        # Convert back to numpy
        pod_features_array = randomized_pod_features.numpy()
        kv_hit_norm = randomized_kv_hit.numpy()
        pod_kv_hit_array = kv_hit_norm.copy()
        
        randomization_overhead = time.time() - randomization_start_time
        
        # logger.info(f"✅ Pod randomization applied - each sample has different pod order")
        # logger.info(f"   This prevents model from learning pod identity based on tensor positions")
        
        # Update feature list to reflect final pod features (without kv_hit_ratio)
        self.pod_features = kept_pod_features
        
        build_feature_overhead = time.time() - build_feature_start_time
        
        logger.info(f"FINAL TENSOR ANALYSIS:")
        logger.info(f"pod_features_array shape: {pod_features_array.shape}")
        logger.info(f"First pod features: {pod_features_array[0, 0, :]}")
        logger.info(f"Second pod features: {pod_features_array[0, 1, :]}")
        if INCLUDE_GPU_IN_FEATURE:
            logger.info(f"Are all pod GPU features identical?")
            for i in range(min(3, pod_features_array.shape[1])):
                gpu_start_idx = len(kept_pod_features) - self.num_gpu_types
                gpu_features = pod_features_array[0, i, gpu_start_idx:]
                logger.info(f"  Pod {i} GPU features: {gpu_features}")
        logger.info(f"Final feature composition:")
        logger.info(f"  pod_features_array shape: {pod_features_array.shape}")
        logger.info(f"  kept_pod_features: {kept_pod_features}")
        logger.info(f"  kv_hit_norm shape: {kv_hit_norm.shape}")
        if INCLUDE_GPU_IN_FEATURE:
            logger.info(f"  GPU features included in pod_features: {[f for f in kept_pod_features if 'gpu_model' in f]}")
        
        return pod_features_array, pod_kv_hit_array, kv_hit_norm, {}
        
    def prepare_for_encoding(self, processed_df, sorted_all_pod_ids, request_features_train, overhead_summary, HYPERPARAMETERS):
        
        self.sorted_all_pod_ids = sorted_all_pod_ids
        pod_data = self._ultra_fast_extract_pod_columns(processed_df, sorted_all_pod_ids)
        self.numeric_request_features = request_features_train  # Assume all numeric
        self.categorical_request_features = []
        
        # STEP 3: SKIP encode_pod_ids for inference
        encode_pod_ids_start = time.time()
        # Set minimal required attributes without building encoders
        self.pod_encoder = None
        self.selected_pod_encoder = None
        encode_pod_ids_overhead = time.time() - encode_pod_ids_start
        
        # STEP 4: MINIMAL feature timing
        classify_feature_timing_start = time.time()
        # Build feature list fast
        pod_feature_columns = [col for col in processed_df.columns if col.startswith('pod_')]
        unique_features = list(set(col.split('-')[1] for col in pod_feature_columns if '-' in col))
        self.pod_features = sorted(unique_features)
        feature_timing = {f: 'historical' if 'last_second' in f else 'current' for f in self.pod_features}
        classify_feature_timing_overhead = time.time() - classify_feature_timing_start
        
        # STEP 5: FAST request feature
        n_samples = len(processed_df)
        request_features, request_numeric_features_overhead = self.extract_request_features(processed_df, request_features_train, n_samples)

        # STEP 7: ULTRA-OPTIMIZED pod processing
        pod_features_array, pod_kv_hit_array, kv_hit_norm, per_pod_feature_indices = self._optimized_process_pod_features(pod_data, n_samples, overhead_summary, HYPERPARAMETERS)

        # STEP 8: actions/rewards (continues as normal)
        actions, rewards, ttft_rewards, tpot_rewards = self.extract_actions_rewards(processed_df, n_samples)

        # STEP 9: SKIP combining
        
        # STEP 10: MINIMAL positional encoding
        positional_encodings = np.zeros((pod_features_array.shape[0], pod_features_array.shape[1], 1), dtype=np.float32)
        
        # STEP 11: MINIMAL staleness
        staleness_features = np.zeros((pod_features_array.shape[0], pod_features_array.shape[1], 1), dtype=np.float32)
        pod_features_with_staleness = np.concatenate([pod_features_array, staleness_features], axis=2)
        logger.info(f"pod_features_array.shape: {pod_features_array.shape}")
        logger.info(f"staleness_features.shape: {staleness_features.shape}")
        logger.info(f"pod_features_with_staleness.shape: {pod_features_with_staleness.shape}")
        
        # STEP 12: MINIMAL cross attention
        cross_attention_inputs = {'query': pod_features_with_staleness, 'key_value': kv_hit_norm}
        
        # STEP 13: FAST interaction features
        interaction_features = np.broadcast_to(request_features[:, np.newaxis, :], (n_samples, pod_features_array.shape[1], request_features.shape[1])).copy()

        processed_data = {
            'pod_features': pod_features_array,
            'pod_raw_features': pod_features_array,
            'kv_hit_ratios': kv_hit_norm,
            'kv_hit_raw': pod_kv_hit_array,
            'positional_encodings': positional_encodings,
            'pod_features_with_staleness': pod_features_with_staleness,
            'cross_attention_inputs': cross_attention_inputs,
            'request_features': request_features,
            'request_numeric_features': request_features,
            'request_categorical_features': np.zeros((n_samples, 0)),
            'interaction_features': interaction_features,
            'timestamps': np.zeros(n_samples),
            'feature_timing': feature_timing,
            'pod_ids': self.sorted_all_pod_ids,
            'actions': actions,
            'rewards': rewards,
            'ttft_rewards': ttft_rewards,
            'tpot_rewards': tpot_rewards,
            'feature_stats': getattr(self, 'feature_stats', {}),
            'pod_features_list': self.pod_features,
            'feature_indices_map': per_pod_feature_indices[self.sorted_all_pod_ids[0]] if per_pod_feature_indices and self.sorted_all_pod_ids else {},
            'numeric_request_features': self.numeric_request_features,
            'categorical_request_features': self.categorical_request_features,
            'encoders': {'pod_encoder': None, 'selected_pod_encoder': None, 'categorical_encoders': {}}
        }
        
        return processed_data


    def _ultra_fast_extract_pod_columns(self, processed_df, sorted_all_pod_ids):
        pod_data = {}
        for col in processed_df.columns:
            if col.startswith('pod_') and '-' in col:
                logger.info(f"Processing column: {col}")
                pod_id, feature = col.split('-', 1)
                # pod_id = pod_id.replace('pod_', '')
                if pod_id in sorted_all_pod_ids:
                    if pod_id not in pod_data:
                        pod_data[pod_id] = {}
                    pod_data[pod_id][feature] = processed_df[col]
                else:
                    logger.error(f"Pod ID {pod_id} not found in sorted_all_pod_ids: {sorted_all_pod_ids}, column: {col}")
                    exit()
        if not pod_data:
            logger.error("No pod data found in the DataFrame")
            logger.error(f"processed_df: {processed_df}")
            logger.error(f"Expected pod IDs: {sorted_all_pod_ids}")
            logger.error(f"Extracted pod data: {pod_data}")
            processed_df.to_csv('debug_processed_df.csv', index=False)
            exit(1)
        return pod_data

    
    def extract_request_features(self, processed_df, request_features_train, n_samples):
        request_features_start_time = time.time()
        
        if request_features_train:
            # Use hardcoded indices instead of column name lookup
            if len(request_features_train) == 3:  # input_tokens, output_tokens, total_tokens
                # Hardcode indices [2, 3, 4] for maximum speed
                request_features = processed_df.values[:, 2:5].astype(np.float32, copy=False)
            else:
                # # Fallback for different feature counts
                # request_features = processed_df[request_features_train].values.astype(np.float32, copy=False)
                logger.error(f"Unexpected request features count: {len(request_features_train)}, expected 3")
                assert False
        else:
            logger.error("No request features provided for inference, using empty array")
            # request_features = np.zeros((n_samples, 0), dtype=np.float32)
            assert False
        
        request_features_overhead = time.time() - request_features_start_time
        return request_features, request_features_overhead


    def _ultra_fast_process_pod_features(self, pod_data, n_samples):
        n_pods = len(self.sorted_all_pod_ids)
        if not pod_data:
            # Return minimal defaults
            default_shape = (n_samples, n_pods, 1)
            zeros = np.zeros(default_shape, dtype=np.float32)
            return zeros, zeros, zeros, zeros, {pod_id: {} for pod_id in self.sorted_all_pod_ids}
        
        # OPTIMIZATION 1: Pre-filter features (avoid repeated checks)
        numeric_features = [f for f in self.pod_features if f not in ['kv_hit_ratio', 'gpu_model']]
        n_numeric = len(numeric_features)
        
        if n_numeric == 0:
            # Handle edge case fast
            kv_arrays = np.zeros((n_samples, n_pods, 1), dtype=np.float32)
            for pod_idx, pod_id in enumerate(self.sorted_all_pod_ids):
                if 'kv_hit_ratio' in pod_data.get(pod_id, {}):
                    kv_arrays[:, pod_idx, 0] = pod_data[pod_id]['kv_hit_ratio'].fillna(0).values
            return kv_arrays, kv_arrays, kv_arrays, kv_arrays, {}
        
        # OPTIMIZATION 2: Single allocation with pre-determined size
        numeric_arrays = np.zeros((n_samples, n_pods, n_numeric), dtype=np.float32)
        kv_arrays = np.zeros((n_samples, n_pods, 1), dtype=np.float32)
        
        # OPTIMIZATION 3: Vectorized extraction using pre-built pod mapping
        pod_indices = {pod_id: idx for idx, pod_id in enumerate(self.sorted_all_pod_ids)}
        
        # OPTIMIZATION 4: Process all features in single pass per pod
        for pod_id, pod_idx in pod_indices.items():
            pod_features_data = pod_data.get(pod_id, {})
            
            # Extract all numeric features for this pod at once
            for feat_idx, feature in enumerate(numeric_features):
                feature_data = pod_features_data.get(feature)
                if feature_data is not None:
                    # OPTIMIZATION: Use .values only once and handle fillna efficiently
                    values = feature_data.values
                    if pd.isna(values).any():
                        values = np.nan_to_num(values, nan=0.0)
                    numeric_arrays[:, pod_idx, feat_idx] = values
            
            # Extract KV ratio
            kv_data = pod_features_data.get('kv_hit_ratio')
            if kv_data is not None:
                values = kv_data.values
                if pd.isna(values).any():
                    values = np.nan_to_num(values, nan=0.0)
                kv_arrays[:, pod_idx, 0] = values
        
        pod_features_array = numeric_arrays
        
        # OPTIMIZATION 5: Minimal normalization with pre-check
        if not hasattr(self, 'pod_feature_scaler'):
            from sklearn.preprocessing import StandardScaler
            self.pod_feature_scaler = StandardScaler()
            self.kv_hit_scaler = StandardScaler()
        
        # OPTIMIZATION 6: Fast reshaping and normalization
        pod_shape = pod_features_array.shape
        kv_shape = kv_arrays.shape
        
        if pod_shape[2] > 0:  # Only normalize if we have features
            pod_flat = pod_features_array.reshape(-1, pod_shape[2])
            
            # OPTIMIZATION 7: Skip normalization if data is constant (saves time)
            if np.std(pod_flat) > 1e-8:  # Only normalize if there's variance
                self.pod_feature_scaler.fit(pod_flat)
                pod_features_norm = self.pod_feature_scaler.transform(pod_flat).reshape(pod_shape)
            else:
                pod_features_norm = pod_features_array  # Skip normalization for constant data
        else:
            pod_features_norm = pod_features_array
        
        # KV normalization
        kv_flat = kv_arrays.reshape(-1, 1)
        if np.std(kv_flat) > 1e-8:
            self.kv_hit_scaler.fit(kv_flat)
            kv_hit_norm = self.kv_hit_scaler.transform(kv_flat).reshape(kv_shape)
        else:
            kv_hit_norm = kv_arrays
        
        # OPTIMIZATION 8: Fast feature indices building
        reference_indices = {feature: i for i, feature in enumerate(numeric_features)}
        per_pod_indices = {pod_id: reference_indices for pod_id in self.sorted_all_pod_ids}
        
        return pod_features_array, kv_arrays, pod_features_norm, kv_hit_norm, per_pod_indices




    def extract_actions_rewards(self, df, n_samples):
        """Fast action/reward extraction - minimal validation."""
        actions = np.zeros(n_samples, dtype=np.int64)
        rewards = np.zeros(n_samples, dtype=np.float32)
        ttft_rewards = np.zeros(n_samples, dtype=np.float32)
        tpot_rewards = np.zeros(n_samples, dtype=np.float32)
        
        # Direct extraction without validation
        if 'selected_pod' in df.columns:
            pod_to_idx = {pod_id: i for i, pod_id in enumerate(self.sorted_all_pod_ids)}
            selected_pods = df['selected_pod'].values
            for i, pod in enumerate(selected_pods):
                if pd.notna(pod):
                    idx = pod_to_idx.get(str(pod))
                    if idx is not None:
                        actions[i] = idx
        
        # Direct column extraction
        for col, target in [('reward', rewards), ('ttft_reward', ttft_rewards), ('tpot_reward', tpot_rewards)]:
            if col in df.columns:
                target[:] = df[col].fillna(0).values.astype(np.float32)
        
        return actions, rewards, ttft_rewards, tpot_rewards

    def save_processed_data(self, processed_data):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{self.output_dir}"
        os.makedirs(output_dir, exist_ok=True)
           
        # Create a PyTorch tensor dataset
        tensor_data = {
            # Basic tensors
            'pod_features': torch.FloatTensor(processed_data['pod_features']),
            'kv_hit_ratios': torch.FloatTensor(processed_data['kv_hit_ratios']),
            'request_features': torch.FloatTensor(processed_data['request_features']),
            'actions': torch.LongTensor(processed_data['actions']),
            'rewards': torch.FloatTensor(processed_data['rewards']),
            
            # Enhanced features for transformer
            'positional_encodings': torch.FloatTensor(processed_data['positional_encodings']),
            'pod_features_with_staleness': torch.FloatTensor(processed_data['pod_features_with_staleness']),
            
            # Cross-attention components
            'query': torch.FloatTensor(processed_data['cross_attention_inputs']['query']),
            'key_value': torch.FloatTensor(processed_data['cross_attention_inputs']['key_value']),
        }
        
        # Add interaction features if available
        if processed_data['interaction_features'] is not None:
            tensor_data['interaction_features'] = torch.FloatTensor(processed_data['interaction_features'])
            
        # Add additional reward components if available
        if 'ttft_rewards' in processed_data and processed_data['ttft_rewards'] is not None:
            tensor_data['ttft_rewards'] = torch.FloatTensor(processed_data['ttft_rewards'])
        if 'tpot_rewards' in processed_data and processed_data['tpot_rewards'] is not None:
            tensor_data['tpot_rewards'] = torch.FloatTensor(processed_data['tpot_rewards'])
            
        # global_tensor_path = "global_tensor_dataset.pt"
        # self._append_to_global_tensor_dataset(tensor_data, global_tensor_path)
        torch.save(tensor_data, os.path.join(output_dir, "tensor_dataset.pt"))
        
        if hasattr(self, '_reference_tensor_data') and self._reference_tensor_data is not None:
            if self._validate_tensor_compatibility(self._reference_tensor_data, tensor_data):
                logger.debug("✅ Tensor data compatible with reference batch")
            else:
                logger.warning("⚠️ Tensor data incompatible with reference batch")
        else:
            # Store first batch as reference for future validations
            self._reference_tensor_data = {k: v.clone() if isinstance(v, torch.Tensor) else v 
                                        for k, v in tensor_data.items()}
            logger.debug("📝 Stored reference tensor data for future validation")


        metadata = {
            'dataset_size': len(processed_data['actions']),
            'num_pods': len(processed_data['pod_ids']),
            'feature_dimensions': {
                'pod_features': processed_data['pod_features'].shape[2],
                'pod_features_with_staleness': processed_data['pod_features_with_staleness'].shape[2],
                'kv_hit_ratios': processed_data['kv_hit_ratios'].shape[2],
                'request_features': processed_data['request_features'].shape[1],
                'positional_encodings': processed_data['positional_encodings'].shape[2],
            },
            'reward_statistics': {
                'mean': float(np.mean(processed_data['rewards'])),
                'std': float(np.std(processed_data['rewards'])),
                'min': float(np.min(processed_data['rewards'])),
                'max': float(np.max(processed_data['rewards'])),
            },
            'action_distribution': {
                str(i): int(np.sum(processed_data['actions'] == i)) 
                for i in range(len(processed_data['pod_ids']))
            },
            'timestamp': timestamp,
            'processing_info': {
                'historical_features': len([f for f, t in processed_data['feature_timing'].items() if t == 'historical']),
                'current_features': len([f for f, t in processed_data['feature_timing'].items() if t == 'current'])
            }
        }
        
        with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        # with open("metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved processed data to {output_dir}")
        return output_dir
    
    # def _append_to_global_tensor_dataset(self, new_tensor_data, global_tensor_path):
    #     try:
    #         # Check if global dataset already exists
    #         if os.path.exists(global_tensor_path):
    #             logger.info(f"Loading existing global tensor dataset from {global_tensor_path}")
                
    #             # Load existing data
    #             existing_data = torch.load(global_tensor_path, map_location='cpu')
                
    #             # # Validate compatibility
    #             if not self._validate_tensor_compatibility(existing_data, new_tensor_data):
    #                 logger.error("New tensor data incompatible with existing global dataset")
    #                 return
                
    #             # Concatenate tensors
    #             merged_data = {}
    #             for key in existing_data.keys():
    #                 if key in new_tensor_data:
    #                     if isinstance(existing_data[key], torch.Tensor) and isinstance(new_tensor_data[key], torch.Tensor):
    #                         # Concatenate along batch dimension (dim=0)
    #                         merged_data[key] = torch.cat([existing_data[key], new_tensor_data[key]], dim=0)
    #                         logger.debug(f"Concatenated {key}: {existing_data[key].shape[0]} + {new_tensor_data[key].shape[0]} = {merged_data[key].shape[0]}")
    #                     else:
    #                         # For non-tensors, keep the existing value or update if needed
    #                         merged_data[key] = existing_data[key]
    #                 else:
    #                     # Keep existing data for keys not in new data
    #                     merged_data[key] = existing_data[key]
                
    #             # Add any new keys from new_tensor_data that weren't in existing_data
    #             for key in new_tensor_data.keys():
    #                 if key not in merged_data:
    #                     logger.warning(f"New key {key} found in new data, adding to global dataset")
    #                     merged_data[key] = new_tensor_data[key]
                
    #         else:
    #             logger.info(f"Creating new global tensor dataset at {global_tensor_path}")
    #             merged_data = new_tensor_data.copy()
            
    #         # Save the merged dataset
    #         torch.save(merged_data, global_tensor_path)
            
    #         # Log the final sizes
    #         total_samples = merged_data['actions'].shape[0] if 'actions' in merged_data else 0
    #         new_samples = new_tensor_data['actions'].shape[0] if 'actions' in new_tensor_data else 0
    #         logger.info(f"Successfully appended {new_samples} samples to global dataset. Total samples: {total_samples}")
            
    #     except Exception as e:
    #         logger.error(f"Failed to append to global tensor dataset: {e}")
    #         # Don't raise the exception to avoid breaking the main processing

    def _validate_tensor_compatibility(self, existing_data, new_data):
        """Validate that new tensor data is compatible with existing data for concatenation.
        
        Args:
            existing_data: Existing tensor dataset
            new_data: New tensor data to append
            
        Returns:
            True if compatible, False otherwise
        """
        # Check if both datasets have the same keys (for tensors)
        existing_tensor_keys = {k for k, v in existing_data.items() if isinstance(v, torch.Tensor)}
        new_tensor_keys = {k for k, v in new_data.items() if isinstance(v, torch.Tensor)}
        
        missing_keys = existing_tensor_keys - new_tensor_keys
        extra_keys = new_tensor_keys - existing_tensor_keys
        
        if missing_keys:
            logger.error(f"New data missing tensor keys: {missing_keys}")
            return False
        
        if extra_keys:
            logger.warning(f"New data has extra tensor keys: {extra_keys}")
            # We can still proceed, just add the new keys
        
        # Check tensor shape compatibility (all dimensions except batch should match)
        for key in existing_tensor_keys.intersection(new_tensor_keys):
            existing_shape = existing_data[key].shape
            new_shape = new_data[key].shape
            
            if len(existing_shape) != len(new_shape):
                logger.error(f"Tensor {key}: dimension mismatch - existing: {existing_shape}, new: {new_shape}")
                return False
            
            if len(existing_shape) > 1 and existing_shape[1:] != new_shape[1:]:
                logger.error(f"Tensor {key}: shape mismatch - existing: {existing_shape}, new: {new_shape}")
                return False
        
        return True

    def create_dataset_loaders(self, processed_data, batch_size=32, val_split=0.1):
        """Create PyTorch DataLoader objects for training and validation.

        Args:
            processed_data: Dictionary with preprocessed data
            batch_size: Batch size for training
            val_split: Fraction of data to use for validation

        Returns:
            train_loader, val_loader: DataLoader objects
        """
        try:
            import torch
            from torch.utils.data import TensorDataset, DataLoader, random_split

            # Create tensor dataset
            tensor_data = [
                torch.FloatTensor(processed_data['pod_features_with_staleness']),
                torch.FloatTensor(processed_data['kv_hit_ratios']),
                torch.FloatTensor(processed_data['request_features']),
                torch.LongTensor(processed_data['actions']),
                torch.FloatTensor(processed_data['rewards'])
            ]

            # Add positional encodings
            if 'positional_encodings' in processed_data:
                tensor_data.append(torch.FloatTensor(processed_data['positional_encodings']))

            # Create dataset
            dataset = TensorDataset(*tensor_data)

            # Split into train and validation
            val_size = int(len(dataset) * val_split)
            train_size = len(dataset) - val_size

            train_dataset, val_dataset = random_split(
                dataset, [train_size, val_size]
            )

            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=torch.cuda.is_available()
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=torch.cuda.is_available()
            )

            logger.info(f"Created data loaders with {train_size} training and {val_size} validation samples")

            return train_loader, val_loader

        except ImportError:
            logger.warning("PyTorch not available, skipping data loader creation")
            return None, None


def encode_for_train(sorted_all_pod_ids, processed_df, output_dir, request_features_train, HYPERPARAMETERS):
    if len(processed_df) > 0:
        logger.info("First row selected_pod value: " + str(processed_df.iloc[0].get('selected_pod', 'N/A')))
    # Check if data contains the expected column pattern
    pod_cols = [c for c in processed_df.columns if 'pod_' in c or '-pod' in c]
    if not pod_cols:
        logger.warning("No columns with 'pod_' prefix or '-pod' pattern found")

    assert processed_df['selected_pod'].iloc[0] in sorted_all_pod_ids

    # Basic data quality checks
    logger.info("Performing data quality checks...")
    missing_col_pct = processed_df.isnull().mean() * 100
    high_missing = missing_col_pct[missing_col_pct > 20].index.tolist()
    if high_missing:
        logger.error(f"Columns with >20% missing values: {len(high_missing)} columns")
        assert False
        
    logger.info("Processing training data...")
    data_processor = LLMRoutingDataProcessor(output_dir=output_dir)
    overhead_summary = {}
    train_processed = data_processor.prepare_for_encoding(processed_df, sorted_all_pod_ids, request_features_train, overhead_summary, HYPERPARAMETERS)
    train_path = data_processor.save_processed_data(train_processed)
    logger.info("Data processing complete!")
    logger.info(f"Training data: {train_path}")
    logger.info(f"Dataset shapes:")
    logger.info(f"  pod_features: {train_processed['pod_features'].shape}")
    logger.info(f"  pod_features_with_staleness: {train_processed['pod_features_with_staleness'].shape}")
    logger.info(f"  kv_hit_ratios: {train_processed['kv_hit_ratios'].shape}")
    logger.info(f"  request_features: {train_processed['request_features'].shape}")
    logger.info(f"  positional_encodings: {train_processed['positional_encodings'].shape}")
    logger.info(f"  actions: {train_processed['actions'].shape}")
    logger.info(f"  rewards: {train_processed['rewards'].shape}")
    return train_path


def encode_for_inference(sorted_all_pod_ids, processed_df, request_features_train, HYPERPARAMETERS):
    prepare_for_encoding_start = time.time()
    processor = LLMRoutingDataProcessor(output_dir="temp_inference")
    overhead_summary = {}
    processed_data = processor.prepare_for_encoding(processed_df, sorted_all_pod_ids, request_features_train, overhead_summary, HYPERPARAMETERS)
    prepare_for_encoding_overhead = time.time() - prepare_for_encoding_start
    post_process_start_time = time.time()
    tensor_data = {}
    tensor_data['pod_features'] = torch.from_numpy(processed_data['pod_features']).float()
    tensor_data['kv_hit_ratios'] = torch.from_numpy(processed_data['kv_hit_ratios']).float()
    tensor_data['request_features'] = torch.from_numpy(processed_data['request_features']).float()
    tensor_data['actions'] = torch.from_numpy(processed_data['actions']).long()
    tensor_data['rewards'] = torch.from_numpy(processed_data['rewards']).float()
    tensor_data['positional_encodings'] = torch.from_numpy(processed_data['positional_encodings']).float()
    tensor_data['pod_features_with_staleness'] = torch.from_numpy(processed_data['pod_features_with_staleness']).float()
    tensor_data['query'] = torch.from_numpy(processed_data['cross_attention_inputs']['query']).float()
    tensor_data['key_value'] = torch.from_numpy(processed_data['cross_attention_inputs']['key_value']).float()
    ttft_rewards = processed_data.get('ttft_rewards')
    if ttft_rewards is not None:
        tensor_data['ttft_rewards'] = torch.from_numpy(ttft_rewards).float()
    tpot_rewards = processed_data.get('tpot_rewards')
    if tpot_rewards is not None:
        tensor_data['tpot_rewards'] = torch.from_numpy(tpot_rewards).float()
    post_process_overhead = time.time() - post_process_start_time
    # # Optional tensors
    # if processed_data['interaction_features'] is not None:
    #     tensor_data['interaction_features'] = torch.from_numpy(processed_data['interaction_features']).float()
        
    overhead_summary['encoding.encode_for_inference.mask_overhead'] = 0
    overhead_summary['encoding.encode_for_inference.prepare_for_encoding_overhead'] = prepare_for_encoding_overhead * 1000
    overhead_summary['encoding.encode_for_inference.post_process_overhead'] = post_process_overhead * 1000

    return tensor_data, overhead_summary