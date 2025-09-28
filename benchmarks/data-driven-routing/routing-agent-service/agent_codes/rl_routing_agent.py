#!/usr/bin/env python3

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

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy  
from stable_baselines3.common.utils import get_schedule_fn, discount_cumulative_rewards
from stable_baselines3.common.type_aliases import GymEnv, Schedule

from logger import logger

class RoutingEnvironment(gym.Env):
    """
    Custom Gymnasium environment for request routing
    """
    def __init__(self, state_dim: Dict[str, int], action_dim: int):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Define observation space
        # Flatten all features into a single vector
        pod_features_size = state_dim['pod_features'] * action_dim
        kv_features_size = state_dim['kv_hit_ratios'] * action_dim  
        request_features_size = state_dim['request_features']
        
        total_obs_size = pod_features_size + kv_features_size + request_features_size
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(total_obs_size,), 
            dtype=np.float32
        )
        
        # Define action space (discrete: choose one pod)
        self.action_space = spaces.Discrete(action_dim)
        
        # Episode management
        self.episode_start_time = None
        self.episode_length_minutes = 3  # Default episode length
        self.max_steps_per_episode = 1000  # Fallback limit
        self.current_step = 0
        
        logger.info(f"RoutingEnvironment initialized:")
        logger.info(f"  Observation space: {self.observation_space.shape}")
        logger.info(f"  Action space: {self.action_space.n}")
        logger.info(f"  Episode length: {self.episode_length_minutes} minutes")
        
    def _flatten_state(self, pod_features, kv_hit_ratios, request_features):
        """Flatten state components into a single observation vector"""
        # pod_features: [num_pods, pod_feature_dim]
        # kv_hit_ratios: [num_pods, kv_dim] 
        # request_features: [request_dim]
        
        pod_flat = pod_features.flatten()
        kv_flat = kv_hit_ratios.flatten()
        
        # Concatenate all features
        obs = np.concatenate([pod_flat, kv_flat, request_features])
        return obs.astype(np.float32)
        
    def reset(self, seed=None, options=None):
        """Reset environment for new episode"""
        super().reset(seed=seed)
        
        self.episode_start_time = time.time()
        self.current_step = 0
        
        # Return dummy observation (will be replaced by real data)
        dummy_obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        return dummy_obs, {}
        
    def step(self, action: int):
        """Execute one step - placeholder, actual rewards come from external system"""
        self.current_step += 1
        
        # Check if episode should end
        episode_time = time.time() - self.episode_start_time
        episode_timeout = episode_time > (self.episode_length_minutes * 60)
        step_limit_reached = self.current_step >= self.max_steps_per_episode
        
        terminated = False  # Tasks don't naturally terminate
        truncated = episode_timeout or step_limit_reached
        
        # Dummy values - real rewards and observations come from external system
        dummy_obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        dummy_reward = 0.0
        
        return dummy_obs, dummy_reward, terminated, truncated, {}


class RoutingPolicy(BasePolicy):
    """
    Custom policy network for routing decisions
    Reuses the architecture from existing FixedPolicyNetwork
    """
    def __init__(self, observation_space, action_space, lr_schedule: Schedule,
                 state_dim: Dict[str, int], hidden_dim: int = 64, **kwargs):
        super().__init__(observation_space, action_space, **kwargs)
        
        self.state_dim = state_dim
        self.action_dim = action_space.n
        
        # Calculate feature sizes (same as original implementation)
        pod_feature_size = state_dim['pod_features']
        kv_feature_size = state_dim['kv_hit_ratios'] 
        request_feature_size = state_dim['request_features']
        
        per_pod_features = pod_feature_size + kv_feature_size
        combined_input_size = per_pod_features + request_feature_size
        
        # Pod-aware scoring network (same architecture as original)
        self.pod_scorer = nn.Sequential(
            nn.Linear(combined_input_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(), 
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        logger.info(f"RoutingPolicy initialized:")
        logger.info(f"  Per-pod features: {per_pod_features}")
        logger.info(f"  Request features: {request_feature_size}")
        logger.info(f"  Combined input per pod: {combined_input_size}")
        
    def _unflatten_state(self, obs):
        """Convert flattened observation back to structured format"""
        # Calculate sizes
        pod_feature_size = self.state_dim['pod_features']
        kv_feature_size = self.state_dim['kv_hit_ratios']
        request_feature_size = self.state_dim['request_features']
        
        # Split observation
        pod_features_size = pod_feature_size * self.action_dim
        kv_features_size = kv_feature_size * self.action_dim
        
        pod_features = obs[:pod_features_size].reshape(self.action_dim, pod_feature_size)
        kv_hit_ratios = obs[pod_features_size:pod_features_size + kv_features_size].reshape(self.action_dim, kv_feature_size)
        request_features = obs[pod_features_size + kv_features_size:]
        
        return pod_features, kv_hit_ratios, request_features
        
    def forward(self, obs, deterministic=False):
        """Forward pass through policy network"""
        batch_size = obs.shape[0] if len(obs.shape) > 1 else 1
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
            
        # Convert to structured format
        pod_features, kv_hit_ratios, request_features = self._unflatten_state(obs[0])
        
        # Convert to tensors and add batch dimension
        pod_features = torch.tensor(pod_features, dtype=torch.float32).unsqueeze(0)
        kv_hit_ratios = torch.tensor(kv_hit_ratios, dtype=torch.float32).unsqueeze(0)
        request_features = torch.tensor(request_features, dtype=torch.float32).unsqueeze(0)
        
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
        
        # Get action probabilities
        action_probs = F.softmax(pod_scores, dim=1)
        
        return action_probs
        
    def _predict(self, observation, deterministic=False):
        """Predict action given observation"""
        action_probs = self.forward(observation, deterministic)
        
        if deterministic:
            actions = torch.argmax(action_probs, dim=1)
        else:
            dist = torch.distributions.Categorical(action_probs)
            actions = dist.sample()
            
        return actions.cpu().numpy()
        
    def predict_values(self, obs):
        """Placeholder for value function (not used in our policy gradient)"""
        return torch.zeros(obs.shape[0], 1)
        
    def _predict(self, observation, deterministic=False):
        """Required by SB3 BasePolicy"""
        action_probs = self.forward(observation, deterministic)
        
        if deterministic:
            actions = torch.argmax(action_probs, dim=1)
        else:
            dist = torch.distributions.Categorical(action_probs)
            actions = dist.sample()
            
        return actions.cpu().numpy()


class RLRoutingAgent(BaseAlgorithm):
    """
    Custom RL algorithm for request routing using policy gradients
    Implements the custom reward formulation: reward_t = π(a_t|s_t) * point_reward_t
    """
    
    def __init__(self, policy, env, state_dim: Dict[str, int], 
                 learning_rate: float = 3e-4,
                 reward_decay_factor: float = 0.95,
                 baseline_decay: float = 0.95,
                 mini_batch_size: int = 30,
                 episode_length_minutes: int = 3,
                 update_frequency_seconds: int = 60,
                 entropy_coeff: float = 0.01,
                 hidden_dim: int = 64,
                 **kwargs):
        
        super().__init__(policy, env, learning_rate=learning_rate, **kwargs)
        
        # RL-specific hyperparameters
        self.reward_decay_factor = reward_decay_factor
        self.baseline_decay = baseline_decay
        self.mini_batch_size = mini_batch_size
        self.episode_length_minutes = episode_length_minutes  
        self.update_frequency_seconds = update_frequency_seconds
        self.entropy_coeff = entropy_coeff
        
        # Initialize policy (will be set by SB3)
        self.state_dim = state_dim
        self._setup_model()
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Experience buffer for mini-batch updates
        self.experience_buffer = deque(maxlen=1000)
        
        # Baseline for variance reduction
        self.baseline = 0.0
        
        # Episode and timing management
        self.episode_start_time = time.time()
        self.last_update_time = time.time()
        
        # Metrics tracking
        self.total_steps = 0
        self.total_episodes = 0
        self.recent_rewards = deque(maxlen=100)
        
        logger.info(f"RLRoutingAgent initialized:")
        logger.info(f"  Reward decay factor: {reward_decay_factor}")
        logger.info(f"  Baseline decay: {baseline_decay}")
        logger.info(f"  Mini-batch size: {mini_batch_size}")
        logger.info(f"  Episode length: {episode_length_minutes} minutes")
        logger.info(f"  Update frequency: {update_frequency_seconds} seconds")
        
        
    def remember_experience(self, pod_features, kv_hit_ratios, request_features, 
                          action: int, point_reward: float):
        """Store experience tuple for later learning"""
        # Convert to tensors (for future use if needed)
        # pod_features_tensor = torch.tensor(pod_features, dtype=torch.float32)
        # kv_hit_ratios_tensor = torch.tensor(kv_hit_ratios, dtype=torch.float32)
        # request_features_tensor = torch.tensor(request_features, dtype=torch.float32)
        
        # Flatten state for environment compatibility
        obs = self.env._flatten_state(pod_features, kv_hit_ratios, request_features)
        
        # Get action probability for taken action
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_probs = self.policy.forward(obs_tensor)
            action_prob = action_probs[0, action].item()
        
        # Calculate custom reward: π(a|s) * point_reward
        custom_reward = action_prob * point_reward
        
        # Store experience
        experience = {
            'obs': obs,
            'action': action,
            'point_reward': point_reward,
            'action_prob': action_prob,
            'custom_reward': custom_reward,
            'timestamp': time.time()
        }
        
        self.experience_buffer.append(experience)
        self.total_steps += 1
        
        logger.debug(f"Stored experience: action={action}, point_reward={point_reward:.4f}, "
                    f"action_prob={action_prob:.4f}, custom_reward={custom_reward:.4f}")
        
        # Check if should update
        self._maybe_update()
        
    def _maybe_update(self):
        """Check if it's time to update the policy"""
        current_time = time.time()
        
        # Check update conditions
        batch_size_reached = len(self.experience_buffer) >= self.mini_batch_size
        time_for_update = (current_time - self.last_update_time) >= self.update_frequency_seconds
        
        if batch_size_reached and time_for_update:
            self._update_policy()
            self.last_update_time = current_time
            
    def _update_policy(self):
        """Update policy using accumulated experiences"""
        if len(self.experience_buffer) < self.mini_batch_size:
            return
            
        logger.info(f"Updating policy with {len(self.experience_buffer)} experiences")
        
        # Extract experiences
        experiences = list(self.experience_buffer)
        
        # Calculate discounted rewards
        discounted_rewards = self._calculate_discounted_rewards(experiences)
        
        # Prepare batch data
        observations = torch.stack([torch.tensor(exp['obs'], dtype=torch.float32) for exp in experiences])
        actions = torch.tensor([exp['action'] for exp in experiences], dtype=torch.long)
        
        # Get current policy probabilities
        action_probs = self.policy.forward(observations)
        selected_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Calculate advantages using baseline
        advantages = []
        for reward in discounted_rewards:
            self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * reward
            advantage = reward - self.baseline
            advantages.append(advantage)
        
        advantages = torch.tensor(advantages, dtype=torch.float32)
        
        # Policy gradient loss (our custom formulation)
        policy_loss = -(selected_probs * advantages).mean()
        
        # Add entropy bonus for exploration
        dist = torch.distributions.Categorical(action_probs)
        entropy = dist.entropy().mean()
        entropy_loss = -self.entropy_coeff * entropy
        
        total_loss = policy_loss + entropy_loss
        
        # Update policy
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Track metrics
        avg_reward = np.mean([exp['custom_reward'] for exp in experiences])
        self.recent_rewards.append(avg_reward)
        
        logger.info(f"Policy updated: loss={total_loss.item():.4f}, "
                   f"avg_reward={avg_reward:.4f}, entropy={entropy.item():.4f}, "
                   f"baseline={self.baseline:.4f}")
        
        # Clear processed experiences (keep some for stability)
        for _ in range(len(experiences) // 2):
            self.experience_buffer.popleft()
            
    def _calculate_discounted_rewards(self, experiences):
        """Calculate discounted cumulative rewards"""
        rewards = [exp['custom_reward'] for exp in experiences]
        discounted = []
        
        cumulative = 0
        for reward in reversed(rewards):
            cumulative = reward + self.reward_decay_factor * cumulative
            discounted.append(cumulative)
            
        return list(reversed(discounted))
        
    def predict(self, pod_features, kv_hit_ratios, request_features, deterministic=False):
        """Predict action for given state"""
        # Flatten state
        obs = self.env._flatten_state(pod_features, kv_hit_ratios, request_features)
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            action_probs = self.policy.forward(obs_tensor)
            
            if deterministic:
                action = torch.argmax(action_probs, dim=1)
            else:
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                
        return action.item(), action_probs[0].cpu().numpy()
        
    def learn(self, total_timesteps: int = None):
        """Placeholder for compatibility - actual learning happens online"""
        logger.info(f"RL agent ready for online learning")
        return self
        
    def save(self, path: str):
        """Save agent state"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'baseline': self.baseline,
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
        }, path)
        logger.info(f"Agent saved to {path}")
        
    def load(self, path: str):
        """Load agent state"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.baseline = checkpoint['baseline']
        self.total_steps = checkpoint['total_steps']
        self.total_episodes = checkpoint['total_episodes']
        logger.info(f"Agent loaded from {path}")
        
    def get_metrics(self):
        """Get current training metrics"""
        return {
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'baseline': self.baseline,
            'recent_avg_reward': np.mean(self.recent_rewards) if self.recent_rewards else 0.0,
            'buffer_size': len(self.experience_buffer)
        }


def create_rl_routing_agent(state_dim: Dict[str, int], action_dim: int, **hyperparameters):
    """Factory function to create RL routing agent"""
    
    # Create environment
    env = RoutingEnvironment(state_dim, action_dim)
    
    # Create agent with hyperparameters
    agent = RLRoutingAgent(
        env=env,
        state_dim=state_dim,
        **hyperparameters
    )
    
    return agent


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
    agent = create_rl_routing_agent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=3e-4,
        reward_decay_factor=0.95,
        mini_batch_size=20,
        episode_length_minutes=2
    )
    
    # Simulate some experiences
    logger.info("Testing RL agent with simulated experiences...")
    
    for i in range(50):
        # Generate random state
        pod_features = np.random.randn(action_dim, state_dim['pod_features'])
        kv_hit_ratios = np.random.rand(action_dim, state_dim['kv_hit_ratios'])
        request_features = np.random.randn(state_dim['request_features'])
        
        # Get prediction
        action, probs = agent.predict(pod_features, kv_hit_ratios, request_features)
        
        # Simulate point reward (higher for action 0, lower for others)
        point_reward = 1.0 if action == 0 else 0.1 + np.random.random() * 0.3
        
        # Store experience
        agent.remember_experience(pod_features, kv_hit_ratios, request_features, action, point_reward)
        
        if i % 10 == 0:
            metrics = agent.get_metrics()
            logger.info(f"Step {i}: action={action}, point_reward={point_reward:.3f}, "
                       f"baseline={metrics['baseline']:.3f}")
    
    final_metrics = agent.get_metrics()
    logger.info(f"Final metrics: {final_metrics}")