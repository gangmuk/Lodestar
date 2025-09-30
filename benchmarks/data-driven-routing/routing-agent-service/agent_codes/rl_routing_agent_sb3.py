#!/usr/bin/env python3

"""
Proper SB3 Integration for RL Routing Agent
This implementation uses Stable Baselines3 infrastructure properly
while maintaining our custom reward logic.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from typing import Dict, List, Tuple, Any
import pickle

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import obs_as_tensor

from logger import logger

class RoutingEnvironment(gym.Env):
    """
    Proper Gymnasium environment for request routing
    """
    def __init__(self, state_dim: Dict[str, int], action_dim: int):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Define observation space - flattened features
        pod_features_size = state_dim['pod_features'] * action_dim
        kv_features_size = state_dim['kv_hit_ratios'] * action_dim  
        request_features_size = state_dim['request_features']
        
        total_obs_size = pod_features_size + kv_features_size + request_features_size
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(total_obs_size,), 
            dtype=np.float32
        )
        
        # Define action space
        self.action_space = spaces.Discrete(action_dim)
        
        # Current state for external control
        self.current_obs = None
        self.pending_reward = None
        self.step_count = 0
        
        logger.info(f"RoutingEnvironment initialized:")
        logger.info(f"  Observation space: {self.observation_space.shape}")
        logger.info(f"  Action space: {self.action_space.n}")
        
    def _flatten_state(self, pod_features, kv_hit_ratios, request_features):
        """Convert structured state to flat observation"""
        pod_flat = pod_features.flatten()
        kv_flat = kv_hit_ratios.flatten()
        obs = np.concatenate([pod_flat, kv_flat, request_features])
        return obs.astype(np.float32)
        
    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)
        self.step_count = 0
        
        # Return dummy observation - will be set externally
        dummy_obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        self.current_obs = dummy_obs
        
        return dummy_obs, {}
        
    def step(self, action: int):
        """Step function - controlled externally"""
        self.step_count += 1
        
        # Return current state - reward and next_obs set externally
        reward = self.pending_reward if self.pending_reward is not None else 0.0
        self.pending_reward = None
        
        # Episode management (for SB3)
        terminated = False
        truncated = self.step_count >= 1000  # Prevent infinite episodes
        
        return self.current_obs, reward, terminated, truncated, {}
    
    def set_state_and_reward(self, obs, reward):
        """External interface to set state and reward"""
        self.current_obs = obs
        self.pending_reward = reward


class RoutingPolicyNetwork(BaseFeaturesExtractor):
    """
    Custom feature extractor that implements our routing policy architecture
    This integrates with SB3's ActorCriticPolicy
    """
    def __init__(self, observation_space: gym.Space, state_dim: Dict[str, int], 
                 hidden_dim: int = 64, **kwargs):
        # Output dimension - this will be the input to the policy head
        features_dim = hidden_dim // 2
        super().__init__(observation_space, features_dim)
        
        self.state_dim = state_dim
        self.action_dim = None  # Will be set when we know the number of pods
        
        # Calculate feature sizes
        pod_feature_size = state_dim['pod_features']
        kv_feature_size = state_dim['kv_hit_ratios'] 
        request_feature_size = state_dim['request_features']
        
        per_pod_features = pod_feature_size + kv_feature_size
        combined_input_size = per_pod_features + request_feature_size
        
        # Pod scoring network (same as original)
        self.pod_scorer = nn.Sequential(
            nn.Linear(combined_input_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(), 
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Output layer to convert score statistics to features
        # We'll have 3 statistics: max, mean, std
        self.features_net = nn.Sequential(
            nn.Linear(3, features_dim),
            nn.ReLU()
        )
        
        logger.info(f"RoutingPolicyNetwork initialized:")
        logger.info(f"  Per-pod features: {per_pod_features}")
        logger.info(f"  Request features: {request_feature_size}")
        logger.info(f"  Combined input per pod: {combined_input_size}")
        logger.info(f"  Features output dim: {features_dim}")
        
    def _unflatten_obs(self, obs):
        """Convert flattened observation back to structured format"""
        batch_size = obs.shape[0]
        
        # Infer action_dim from observation shape if not set
        if self.action_dim is None:
            pod_feature_size = self.state_dim['pod_features']
            kv_feature_size = self.state_dim['kv_hit_ratios']
            request_feature_size = self.state_dim['request_features']
            
            remaining_size = obs.shape[1] - request_feature_size
            per_pod_size = pod_feature_size + kv_feature_size
            self.action_dim = remaining_size // per_pod_size
            
            logger.info(f"Inferred action_dim: {self.action_dim}")
        
        # Split observation
        pod_feature_size = self.state_dim['pod_features']
        kv_feature_size = self.state_dim['kv_hit_ratios']
        request_feature_size = self.state_dim['request_features']
        
        pod_features_size = pod_feature_size * self.action_dim
        kv_features_size = kv_feature_size * self.action_dim
        
        pod_features = obs[:, :pod_features_size].reshape(batch_size, self.action_dim, pod_feature_size)
        kv_hit_ratios = obs[:, pod_features_size:pod_features_size + kv_features_size].reshape(batch_size, self.action_dim, kv_feature_size)
        request_features = obs[:, pod_features_size + kv_features_size:]
        
        return pod_features, kv_hit_ratios, request_features
        
    def forward(self, observations):
        """Forward pass through feature extractor"""
        batch_size = observations.shape[0]
        
        # Convert to structured format
        pod_features, kv_hit_ratios, request_features = self._unflatten_obs(observations)
        num_pods = pod_features.shape[1]
        
        # Combine pod features and kv ratios
        combined_pod_features = torch.cat([pod_features, kv_hit_ratios], dim=2)
        
        # Expand request features to match each pod
        expanded_request = request_features.unsqueeze(1).expand(-1, num_pods, -1)
        
        # Combine pod-specific features with request context
        full_features = torch.cat([combined_pod_features, expanded_request], dim=2)
        
        # Reshape to process all pods in batch
        reshaped_features = full_features.view(batch_size * num_pods, -1)
        pod_scores = self.pod_scorer(reshaped_features)
        pod_scores = pod_scores.view(batch_size, num_pods)
        
        # Convert pod scores to features for the policy head
        # Use max pooling to get a single feature vector
        max_score, _ = torch.max(pod_scores, dim=1, keepdim=True)
        mean_score = torch.mean(pod_scores, dim=1, keepdim=True)
        std_score = torch.std(pod_scores, dim=1, keepdim=True)
        
        # Combine statistics as features
        score_features = torch.cat([max_score, mean_score, std_score], dim=1)
        features = self.features_net(score_features)
        
        return features


class CustomRoutingPolicy(ActorCriticPolicy):
    """
    Custom routing policy that uses our domain-specific architecture
    """
    def __init__(self, observation_space, action_space, lr_schedule, 
                 state_dim: Dict[str, int], hidden_dim: int = 64, **kwargs):
        
        # Store parameters for feature extractor
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        super().__init__(
            observation_space, 
            action_space, 
            lr_schedule,
            features_extractor_class=RoutingPolicyNetwork,
            features_extractor_kwargs={
                'state_dim': state_dim,
                'hidden_dim': hidden_dim
            },
            **kwargs
        )


class CustomRewardCallback(BaseCallback):
    """
    Callback to implement our custom reward calculation
    """
    def __init__(self, reward_multiplier_func=None, verbose=0):
        super().__init__(verbose)
        self.reward_multiplier_func = reward_multiplier_func
        
    def _on_step(self) -> bool:
        """Apply custom reward calculation"""
        if self.reward_multiplier_func is not None:
            # Get the action probabilities for the actions that were taken
            obs = self.locals['obs_tensor']
            actions = self.locals['actions']
            
            with torch.no_grad():
                # Get action probabilities from current policy
                action_probs = self.model.policy.get_distribution(obs).probs
                selected_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Apply custom reward: π(a|s) * original_reward
            original_rewards = self.locals['rewards']
            custom_rewards = selected_probs.cpu().numpy() * original_rewards
            
            # Update the rewards in the rollout buffer
            self.locals['rewards'] = custom_rewards
            
        return True


class RLRoutingAgentSB3:
    """
    Clean RL routing agent using proper SB3 integration
    """
    def __init__(self, state_dim: Dict[str, int], action_dim: int, **hyperparameters):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hyperparameters = hyperparameters
        
        # Create environment
        self.env = RoutingEnvironment(state_dim, action_dim)
        
        # Extract hyperparameters
        learning_rate = hyperparameters.get('learning_rate', 3e-4)
        hidden_dim = hyperparameters.get('hidden_dim', 64)
        custom_reward = hyperparameters.get('use_custom_reward', True)
        
        # Create SB3 model with custom policy
        self.model = PPO(
            CustomRoutingPolicy,
            self.env,
            learning_rate=learning_rate,
            n_steps=hyperparameters.get('n_steps', 2048),
            batch_size=hyperparameters.get('batch_size', 64),
            n_epochs=hyperparameters.get('n_epochs', 10),
            gamma=hyperparameters.get('reward_decay_factor', 0.95),
            gae_lambda=hyperparameters.get('gae_lambda', 0.95),
            clip_range=hyperparameters.get('clip_range', 0.2),
            ent_coef=hyperparameters.get('entropy_coeff', 0.01),
            vf_coef=hyperparameters.get('vf_coef', 0.5),
            max_grad_norm=hyperparameters.get('max_grad_norm', 0.5),
            policy_kwargs={
                'state_dim': state_dim,
                'hidden_dim': hidden_dim
            },
            verbose=1
        )
        
        # Setup custom reward callback if requested
        self.custom_reward_callback = None
        if custom_reward:
            self.custom_reward_callback = CustomRewardCallback()
            
        # Experience tracking for external interface
        self.experience_buffer = deque(maxlen=1000)
        self.total_steps = 0
        
        logger.info(f"RLRoutingAgentSB3 initialized with PPO")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Gamma: {hyperparameters.get('reward_decay_factor', 0.95)}")
        logger.info(f"  Custom reward: {custom_reward}")
        
    def predict(self, pod_features, kv_hit_ratios, request_features, deterministic=False):
        """Predict action for given state"""
        # Flatten state
        obs = self.env._flatten_state(pod_features, kv_hit_ratios, request_features)
        
        # Use SB3 model for prediction
        action, _ = self.model.predict(obs, deterministic=deterministic)
        
        # Get action probabilities for compatibility
        obs_tensor = obs_as_tensor(obs, self.model.device).unsqueeze(0)
        with torch.no_grad():
            distribution = self.model.policy.get_distribution(obs_tensor)
            # For CategoricalDistribution, use .distribution.probs
            action_probs = distribution.distribution.probs.cpu().numpy()[0]
        
        return int(action), action_probs
        
    def remember_experience(self, pod_features, kv_hit_ratios, request_features, 
                          action: int, point_reward: float, lock=None):
        """
        Store experience for potential online learning.
        
        Args:
            lock: Optional RWLock for thread-safe buffer access (uses write lock)
        """
        # Flatten state
        obs = self.env._flatten_state(pod_features, kv_hit_ratios, request_features)
        
        # Calculate custom reward if using our formulation
        if self.custom_reward_callback is not None:
            obs_tensor = obs_as_tensor(obs, self.model.device).unsqueeze(0)
            with torch.no_grad():
                distribution = self.model.policy.get_distribution(obs_tensor)
                action_prob = distribution.distribution.probs[0, action].item()
            custom_reward = action_prob * point_reward
        else:
            custom_reward = point_reward
        
        experience = {
            'obs': obs,
            'action': action,
            'reward': custom_reward,
            'point_reward': point_reward,
            'timestamp': time.time()
        }
        
        # Thread-safe buffer append (deque.append is NOT thread-safe!)
        # Use write lock for exclusive access during buffer modification
        if lock is not None:
            with lock.write():
                self.experience_buffer.append(experience)
                self.total_steps += 1
        else:
            self.experience_buffer.append(experience)
            self.total_steps += 1
        
        # Set state and reward in environment for potential learning
        self.env.set_state_and_reward(obs, custom_reward)
        
        logger.debug(f"Stored experience: action={action}, point_reward={point_reward:.4f}, "
                    f"custom_reward={custom_reward:.4f}")
    
    def update_online(self, n_steps: int = None):
        """Perform online learning update"""
        if len(self.experience_buffer) < 10:  # Minimum experiences needed
            return
            
        if n_steps is None:
            n_steps = min(len(self.experience_buffer), 64)
        
        # Use SB3's learning mechanism
        callbacks = [self.custom_reward_callback] if self.custom_reward_callback else None
        
        try:
            self.model.learn(
                total_timesteps=n_steps,
                callback=callbacks,
                reset_num_timesteps=False,
                progress_bar=False
            )
            logger.info(f"Online learning update completed: {n_steps} steps")
        except Exception as e:
            logger.error(f"Error in online learning: {e}")
    
    def learn(self, total_timesteps: int, **kwargs):
        """Standard SB3 learning interface"""
        callbacks = [self.custom_reward_callback] if self.custom_reward_callback else None
        
        return self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            **kwargs
        )
    
    def save(self, path: str):
        """Save model using SB3"""
        self.model.save(path)
        
        # Save additional state
        additional_state = {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hyperparameters': self.hyperparameters,
            'total_steps': self.total_steps
        }
        
        with open(f"{path}_additional.pkl", 'wb') as f:
            pickle.dump(additional_state, f)
            
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model using SB3 with fallback to contextual bandit transfer"""
        try:
            # First try to load PPO checkpoint
            self.model = PPO.load(path, env=self.env)
            logger.info(f"Loaded PPO model from {path}")
            
            # Load additional state
            try:
                with open(f"{path}_additional.pkl", 'rb') as f:
                    additional_state = pickle.load(f)
                    self.total_steps = additional_state.get('total_steps', 0)
            except FileNotFoundError:
                logger.warning("Additional state file not found, using defaults")
                
        except Exception as e:
            logger.warning(f"Failed to load PPO checkpoint: {e}")
            
            # Fallback: try to load contextual bandit weights for transfer learning
            cb_policy_path = os.path.join(path, 'policy.pth') if os.path.isdir(path) else f"{path}/policy.pth"
            if os.path.exists(cb_policy_path):
                logger.info(f"🔄 Attempting transfer learning from contextual bandit: {cb_policy_path}")
                self._load_contextual_bandit_weights(cb_policy_path)
            else:
                logger.info("No contextual bandit weights found, starting with random weights")
    
    def _load_contextual_bandit_weights(self, cb_policy_path):
        """Load pod_scorer weights from contextual bandit policy.pth"""
        try:
            logger.info(f"🔄 Loading contextual bandit weights for transfer learning...")
            
            # Load contextual bandit state dict
            cb_state_dict = torch.load(cb_policy_path, map_location='cpu')
            logger.info(f"Loaded contextual bandit with {len(cb_state_dict)} parameters")
            
            # Extract pod_scorer weights (they should have identical layer names)
            pod_scorer_weights = {
                k: v for k, v in cb_state_dict.items() 
                if k.startswith('pod_scorer.')
            }
            
            if not pod_scorer_weights:
                logger.warning("No pod_scorer weights found in contextual bandit")
                return
            
            logger.info(f"Found {len(pod_scorer_weights)} pod_scorer layers to transfer:")
            for layer_name in pod_scorer_weights.keys():
                logger.info(f"  - {layer_name}: {pod_scorer_weights[layer_name].shape}")
            
            # Get the RL agent's feature extractor pod_scorer
            rl_pod_scorer = self.model.policy.features_extractor.pod_scorer
            rl_state_dict = rl_pod_scorer.state_dict()
            
            # Validate dimensions match
            dimension_mismatch = False
            for layer_name, cb_weights in pod_scorer_weights.items():
                if layer_name in rl_state_dict:
                    rl_shape = rl_state_dict[layer_name].shape
                    cb_shape = cb_weights.shape
                    if rl_shape != cb_shape:
                        logger.error(f"Dimension mismatch for {layer_name}: RL={rl_shape}, CB={cb_shape}")
                        dimension_mismatch = True
                    else:
                        logger.debug(f"✅ {layer_name}: shapes match {rl_shape}")
                else:
                    logger.warning(f"Layer {layer_name} not found in RL agent")
            
            if dimension_mismatch:
                logger.error("❌ Cannot transfer weights due to dimension mismatches")
                return
            
            # Transfer the weights
            rl_pod_scorer.load_state_dict(pod_scorer_weights, strict=False)
            
            # Optional: freeze the transferred weights to preserve learned representations
            freeze_transferred = self.hyperparameters.get('freeze_transferred_weights', False)
            if freeze_transferred:
                for param in rl_pod_scorer.parameters():
                    param.requires_grad = False
                logger.info("🔒 Froze transferred pod_scorer weights")
            else:
                logger.info("🔓 Transferred weights remain trainable")
            
            logger.info(f"✅ Successfully transferred {len(pod_scorer_weights)} layers from contextual bandit!")
            logger.info("🚀 RL agent will start with pre-trained pod scoring function")
            
        except Exception as e:
            logger.error(f"Failed to load contextual bandit weights: {e}")
            logger.info("Continuing with random initialization")
            import traceback
            traceback.print_exc()
    
    def get_metrics(self):
        """Get current training metrics"""
        return {
            'total_steps': self.total_steps,
            'buffer_size': len(self.experience_buffer),
            'model_num_timesteps': self.model.num_timesteps,
        }


def create_rl_routing_agent_sb3(state_dim: Dict[str, int], action_dim: int, **hyperparameters):
    """Factory function to create SB3-based RL routing agent"""
    return RLRoutingAgentSB3(state_dim, action_dim, **hyperparameters)


def infer_rl_agent(tensor_data, request_id, sorted_all_pod_ids, processed_df, 
                   rl_agent, hyperparameters, agent_lock=None):
    """
    Complete RL agent inference workflow matching other subalgorithm interfaces.
    
    This function is designed for high concurrency on the request critical path:
    - Prediction uses PyTorch's thread-safe read-only inference (no locking needed)
    - Only locks for: experience buffer writes (fast, ~0.1ms)
    - Enables 1000s of concurrent requests vs ~50 req/s with coarse-grained locking
    
    IMPORTANT: rl_agent must be non-None and properly initialized by caller
    (initialization is handled in routing_agent_service.py under lock)
    
    Args:
        tensor_data: Dict with 'pod_features', 'kv_hit_ratios', 'request_features' tensors
        request_id: Request identifier for logging
        sorted_all_pod_ids: List of pod IDs in order
        processed_df: DataFrame with request data including selected_pod, ttft, avg_tpot
        rl_agent: RLRoutingAgentSB3 instance (must be initialized, not None)
        hyperparameters: Dict with RL_MODEL_HYPERPARAMETERS
        agent_lock: Optional threading.Lock for thread-safe buffer writes
        
    Returns:
        Tuple of (agent, result_dict, overhead_summary_dict)
        - agent: Same agent instance (unchanged)
        - result_dict: Inference results with selected_pod_index, probabilities, etc.
        - overhead_summary_dict: Timing breakdown
    """
    import preprocess  # Import here to avoid circular dependency
    
    if rl_agent is None:
        raise ValueError("rl_agent must be initialized by caller (routing_agent_service.py)")
    
    overhead_summary = {}
    rl_infer_start = time.time()
    
    # Extract numpy arrays from tensors
    extract_start = time.time()
    pod_features_t = tensor_data['pod_features']
    kv_hit_t = tensor_data['kv_hit_ratios']
    req_features_t = tensor_data['request_features']
    pod_features_np = pod_features_t.cpu().numpy()
    kv_hit_np = kv_hit_t.cpu().numpy()
    req_features_np = req_features_t.cpu().numpy()
    overhead_summary['extract_tensors'] = time.time() - extract_start
    
    # Agent is already initialized by caller - no init logic needed here
    
    # Predict action for current state (use first/only sample)
    # Uses READ lock to allow concurrent predictions while preventing model updates
    # Background worker uses WRITE lock during update_online() - mutually exclusive
    # This enables 100s of concurrent predictions with ~10ms latency each
    predict_start = time.time()
    if agent_lock is not None:
        # RWLock: read() allows many concurrent predictions
        with agent_lock.read():
            action_idx, action_probs = rl_agent.predict(
                pod_features_np[0], kv_hit_np[0], req_features_np[0]
            )
    else:
        # No lock provided - unsafe if online learning enabled
        action_idx, action_probs = rl_agent.predict(
            pod_features_np[0], kv_hit_np[0], req_features_np[0]
        )
    overhead_summary['predict'] = time.time() - predict_start
    
    # Compute reward for the last taken action and remember experience
    remember_start = time.time()
    
    # Get the previous action (pod that was actually selected for this request)
    try:
        selected_pod_prev = processed_df['selected_pod'].iloc[0]
        prev_action_idx = sorted_all_pod_ids.index(str(selected_pod_prev))
    except Exception as e:
        logger.warning(f"Could not find previous action from selected_pod: {e}, using current action")
        prev_action_idx = action_idx
    
    # Extract latency metrics
    ttft_val = float(processed_df['ttft'].iloc[0])
    tpot_val = float(processed_df['avg_tpot'].iloc[0])
    ttft_slo = hyperparameters['TTFT_SLO']
    avg_tpot_slo = hyperparameters['AVG_TPOT_SLO']
    ttft_reward_weight = hyperparameters['TTFT_REWARD_WEIGHT']
    
    # Calculate reward using configured reward function
    reward_fn = hyperparameters.get('REWARD_FUNCTION', 'linear_simple')
    if reward_fn == 'linear_simple':
        reward_res = preprocess.calculate_rewards_simple(
            np.array([ttft_val]), np.array([tpot_val]), 
            ttft_slo, avg_tpot_slo, ttft_reward_weight
        )
    elif reward_fn == 'linear_simple_extended':
        reward_res = preprocess.calculate_rewards_simple_extended(
            np.array([ttft_val]), np.array([tpot_val]), 
            ttft_slo, avg_tpot_slo, ttft_reward_weight
        )
    elif reward_fn == 'piecewise_linear_steeper_gradient':
        reward_res = preprocess.calculate_rewards_piecewise_linear_steeper_gradient(
            np.array([ttft_val]), np.array([tpot_val]), 
            ttft_slo, avg_tpot_slo, ttft_reward_weight
        )
    elif reward_fn == 'latency_optimized':
        reward_res = preprocess.calculate_rewards_latency_optimization(
            np.array([ttft_val]), np.array([tpot_val]), 
            ttft_slo, avg_tpot_slo, ttft_reward_weight
        )
    else:
        logger.error(f"Unknown reward function: {reward_fn}")
        raise ValueError(f"Unknown reward function: {reward_fn}")
    
    point_reward = float(reward_res['combined_rewards'][0])
    
    # Store experience in agent's buffer (uses fine-grained lock for thread safety)
    rl_agent.remember_experience(
        pod_features_np[0], kv_hit_np[0], req_features_np[0], 
        prev_action_idx, point_reward, lock=agent_lock
    )
    overhead_summary['remember_experience'] = time.time() - remember_start
    
    # Build result dict matching other subalgorithm interfaces
    result_start = time.time()
    pod_prob_map = {pid: float(action_probs[i]) for i, pid in enumerate(sorted_all_pod_ids)}
    result = {
        'selected_pod_index': int(action_idx),
        'pod_probabilities': pod_prob_map,
        'confidence': float(np.max(action_probs)),
        'explore_mask': 0,
        'predicted_latencies': {pod_id: -1 for pod_id in sorted_all_pod_ids},
        'chosen_pod_predicted_latency': -1,
    }
    overhead_summary['build_result'] = time.time() - result_start
    
    overhead_summary['rl_total'] = time.time() - rl_infer_start
    
    logger.info(f"RL inference complete: action={action_idx}, reward={point_reward:.4f}, "
                f"confidence={result['confidence']:.4f}")
    
    return rl_agent, result, overhead_summary


# Example usage and testing
if __name__ == "__main__":
    # Test configuration
    state_dim = {
        'pod_features': 10,
        'kv_hit_ratios': 1, 
        'request_features': 3
    }
    action_dim = 4  # 4 pods
    
    # Create agent
    agent = create_rl_routing_agent_sb3(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=3e-4,
        reward_decay_factor=0.95,
        n_steps=64,
        batch_size=32
    )
    
    # Test prediction
    logger.info("Testing SB3 RL agent...")
    
    for i in range(10):
        # Generate random state
        pod_features = np.random.randn(action_dim, state_dim['pod_features'])
        kv_hit_ratios = np.random.rand(action_dim, state_dim['kv_hit_ratios'])
        request_features = np.random.randn(state_dim['request_features'])
        
        # Get prediction
        action, probs = agent.predict(pod_features, kv_hit_ratios, request_features)
        
        # Simulate reward
        point_reward = 1.0 if action == 0 else 0.1 + np.random.random() * 0.3
        
        # Store experience
        agent.remember_experience(pod_features, kv_hit_ratios, request_features, action, point_reward)
        
        logger.info(f"Step {i}: action={action}, point_reward={point_reward:.3f}")
    
    # Test online learning
    agent.update_online(n_steps=32)
    
    metrics = agent.get_metrics()
    logger.info(f"Final metrics: {metrics}")
    logger.info("SB3 RL agent test completed successfully!")