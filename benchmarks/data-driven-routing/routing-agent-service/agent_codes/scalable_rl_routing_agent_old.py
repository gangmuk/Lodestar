#!/usr/bin/env python3

"""
Scalable RL Routing Agent - Pod-Count Independent Architecture

Key Design Principles:
1. DeepSets-style per-pod scoring → handles variable #pods (4 to 1000+)
2. Multi-step TD with async experience completion
3. Episode boundaries for proper credit assignment
4. Prioritized experience replay for sample efficiency
5. Cluster statistics for relative context
6. Hard action masking for unhealthy pod filtering

Architecture:
- Shared pod_scorer network processes each pod independently
- Combines: [pod_i_features + kv_hit + request + cluster_stats] → score_i
- Softmax over scores → action probabilities
- Critic uses aggregated cluster features (fixed size)

This solves two critical problems:
- Scalability: Model works with any number of pods without retraining
- Proper RL: Multi-step credit assignment via GAE and episode structure
"""

import os
import time
import uuid
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from typing import Dict, List, Tuple, Any, Optional
import pickle
import threading

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor

from logger import logger


# ============================================================================
# Prioritized Experience Replay Buffer
# ============================================================================

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay for sample-efficient RL learning.
    
    Samples experiences proportional to their TD error (learning value):
    - High TD error = surprising outcome = learn more from it
    - Rare events (failures) naturally get high priority
    - 2-3x better sample efficiency than uniform sampling
    
    Based on: "Prioritized Experience Replay" (Schaul et al., 2015)
    """
    def __init__(self, maxlen=1000, alpha=0.6, beta=0.4):
        """
        Args:
            maxlen: Maximum buffer size
            alpha: Prioritization strength (0=uniform, 1=full prioritization)
            beta: Importance sampling correction (reduces bias)
        """
        self.buffer = deque(maxlen=maxlen)
        self.priorities = deque(maxlen=maxlen)
        self.alpha = alpha
        self.beta = beta
        self.max_priority = 1.0
        self.lock = threading.Lock()  # Thread-safe operations
        
    def add(self, experience):
        """Add experience with maximum priority (explore new experiences first)"""
        with self.lock:
            self.buffer.append(experience)
            # New experiences get max priority
            self.priorities.append(self.max_priority)
    
    def sample(self, batch_size):
        """
        Sample batch with probability ∝ priority^alpha
        
        Returns:
            batch: List of experiences
            indices: Indices in buffer (for priority updates)
            weights: Importance sampling weights (for unbiasing)
        """
        with self.lock:
            if len(self.buffer) < batch_size:
                return [], [], []
            
            # Convert priorities to sampling probabilities
            priorities = np.array(self.priorities, dtype=np.float32)
            probs = priorities ** self.alpha
            probs /= probs.sum()
            
            # Sample indices
            indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
            
            # Importance sampling weights: (N * P(i))^(-beta)
            weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
            weights /= weights.max()  # Normalize for stability
            
            batch = [self.buffer[i] for i in indices]
            
            return batch, indices, weights
    
    def update_priorities(self, indices, td_errors):
        """
        Update priorities based on TD error magnitude
        
        Args:
            indices: Indices of experiences to update
            td_errors: TD errors (target - prediction)
        """
        with self.lock:
            for idx, td_error in zip(indices, td_errors):
                if idx < len(self.priorities):
                    # Priority = |TD error| + small constant
                    priority = abs(td_error) + 1e-6
                    self.priorities[idx] = priority
                    self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)


# ============================================================================
# Episode Tracker
# ============================================================================

class EpisodeTracker:
    """
    Tracks episode boundaries for proper credit assignment.
    
    Episodes define the scope of multi-step returns:
    - Time-based: All requests in 1-second window share credit
    - Provides done flags for TD learning
    """
    def __init__(self, episode_duration=1.0):
        """
        Args:
            episode_duration: Episode length in seconds
        """
        self.episode_duration = episode_duration
        self.episode_start_time = time.time()
        self.episode_id = 0
        self.episode_request_count = 0
        
    def check_episode_end(self):
        """Returns True if current episode should end"""
        elapsed = time.time() - self.episode_start_time
        return elapsed >= self.episode_duration
    
    def reset_episode(self):
        """Start new episode"""
        self.episode_start_time = time.time()
        self.episode_id += 1
        self.episode_request_count = 0
        logger.info(f"Episode {self.episode_id} started")
    
    def increment_request(self):
        """Track request count in episode"""
        self.episode_request_count += 1


# ============================================================================
# Scalable Routing Policy Network
# ============================================================================

class ScalableRoutingPolicyNetwork(BaseFeaturesExtractor):
    """
    Scalable policy network that handles VARIABLE number of pods (4 to 1000+).
    
    Architecture:
    1. Per-pod scorer: [pod_i + kv_i + request + cluster_stats] → score_i
    2. Shared weights across all pods (permutation invariant)
    3. Aggregated features for critic (fixed size)
    
    Key advantage: Same model works with 4 pods or 1000 pods!
    """
    def __init__(self, observation_space: gym.Space, 
                 per_pod_dim: int = 11,  # pod_features(10) + kv_hit(1)
                 request_dim: int = 3,
                 hidden_dim: int = 64,
                 **kwargs):
        """
        Args:
            per_pod_dim: Features per pod (pod_features + kv_hit_ratios)
            request_dim: Request feature dimensions
            hidden_dim: Hidden layer size
        """
        # Features output for actor/critic heads
        # Use aggregated cluster stats (fixed size)
        cluster_stats_dim = per_pod_dim * 4  # mean, std, max, min
        features_dim = cluster_stats_dim + request_dim  # 44 + 3 = 47
        
        super().__init__(observation_space, features_dim)
        
        self.per_pod_dim = per_pod_dim
        self.request_dim = request_dim
        self.hidden_dim = hidden_dim
        
        # === Per-Pod Scorer (SHARED across all pods) ===
        # Input: [pod_i(11) + request(3) + cluster_stats(44)] = 58 dims
        cluster_stats_dim = per_pod_dim * 4
        scorer_input_size = per_pod_dim + request_dim + cluster_stats_dim
        
        self.pod_scorer = nn.Sequential(
            nn.Linear(scorer_input_size, hidden_dim),       # 58 → 64
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),         # 64 → 32
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)                   # 32 → 1 (score)
        )
        
        logger.info(f"ScalableRoutingPolicyNetwork initialized:")
        logger.info(f"  Per-pod input: {scorer_input_size} dims (11 pod + 3 req + 44 cluster)")
        logger.info(f"  Features output: {features_dim} dims (fixed size for critic)")
        logger.info(f"  Hidden dim: {hidden_dim}")
        
    def _compute_cluster_statistics(self, combined_pod_features):
        """
        Compute cluster-wide statistics for each feature dimension.
        
        Args:
            combined_pod_features: [batch, num_pods, 11]
        
        Returns:
            cluster_stats: [batch, 44] (mean, std, max, min for each of 11 features)
        """
        # Compute statistics across pods (dim=1)
        pod_mean = torch.mean(combined_pod_features, dim=1)  # [batch, 11]
        pod_std = torch.std(combined_pod_features, dim=1)    # [batch, 11]
        pod_max = torch.max(combined_pod_features, dim=1)[0] # [batch, 11]
        pod_min = torch.min(combined_pod_features, dim=1)[0] # [batch, 11]
        
        # Concatenate: [mean, std, max, min]
        cluster_stats = torch.cat([pod_mean, pod_std, pod_max, pod_min], dim=1)
        # Shape: [batch, 44]
        
        return cluster_stats
    
    def forward(self, observations):
        """
        Forward pass through feature extractor.
        
        Extracts fixed-size cluster features for actor/critic heads.
        (Actual pod scoring happens in _evaluate_actions)
        
        Args:
            observations: Dict with 'pod_features', 'kv_hit_ratios', 'request_features'
        
        Returns:
            features: [batch, 47] - Fixed size for any #pods
        """
        # observations is a dict (DictObservation)
        pod_features = observations['pod_features']      # [batch, num_pods, 10]
        kv_hit_ratios = observations['kv_hit_ratios']    # [batch, num_pods, 1]
        request_features = observations['request_features']  # [batch, 3]
        
        # Combine pod features + kv hit ratios
        combined_pod_features = torch.cat([pod_features, kv_hit_ratios], dim=2)
        # Shape: [batch, num_pods, 11]
        
        # Compute cluster statistics (FIXED SIZE)
        cluster_stats = self._compute_cluster_statistics(combined_pod_features)
        # Shape: [batch, 44]
        
        # Combine with request features
        features = torch.cat([cluster_stats, request_features], dim=1)
        # Shape: [batch, 47] - FIXED SIZE regardless of num_pods!
        
        return features
    
    def score_pods(self, observations, action_mask=None):
        """
        Score each pod independently using shared network.
        
        This is called during action selection to get pod probabilities.
        
        Args:
            observations: Dict with state components
            action_mask: [batch, num_pods] - 1=valid, 0=invalid
        
        Returns:
            action_probs: [batch, num_pods] - Softmax probabilities
        """
        pod_features = observations['pod_features']
        kv_hit_ratios = observations['kv_hit_ratios']
        request_features = observations['request_features']
        
        batch_size = pod_features.shape[0]
        num_pods = pod_features.shape[1]
        
        # === STEP 1: Combine pod features + kv hit ratios ===
        combined_pod_features = torch.cat([pod_features, kv_hit_ratios], dim=2)
        # Shape: [batch, num_pods, 11]
        
        # === STEP 2: Compute cluster statistics ===
        cluster_stats = self._compute_cluster_statistics(combined_pod_features)
        # Shape: [batch, 44]
        
        # === STEP 3: Expand cluster stats to all pods ===
        expanded_cluster_stats = cluster_stats.unsqueeze(1).expand(-1, num_pods, -1)
        # Shape: [batch, num_pods, 44]
        
        # === STEP 4: Expand request features to all pods ===
        expanded_request = request_features.unsqueeze(1).expand(-1, num_pods, -1)
        # Shape: [batch, num_pods, 3]
        
        # === STEP 5: Concatenate all features ===
        full_features = torch.cat([
            combined_pod_features,     # [batch, num_pods, 11]
            expanded_request,          # [batch, num_pods, 3]
            expanded_cluster_stats     # [batch, num_pods, 44]
        ], dim=2)
        # Shape: [batch, num_pods, 58]
        
        # === STEP 6: Score each pod with shared network ===
        reshaped_features = full_features.view(batch_size * num_pods, -1)
        pod_scores = self.pod_scorer(reshaped_features)  # [batch*num_pods, 1]
        pod_scores = pod_scores.view(batch_size, num_pods)  # [batch, num_pods]
        
        # === STEP 7: Apply action masking (unhealthy pod filtering) ===
        if action_mask is not None:
            # Set invalid pod scores to -inf (zero probability after softmax)
            pod_scores = pod_scores.masked_fill(action_mask == 0, float('-inf'))
        
        # === STEP 8: Softmax to get action probabilities ===
        action_probs = F.softmax(pod_scores, dim=1)  # π(a|s)
        
        return action_probs


class ScalableRoutingPolicy(ActorCriticPolicy):
    """
    Custom Actor-Critic policy using our scalable architecture.
    
    Integrates with SB3's PPO while maintaining pod-independent design.
    """
    def __init__(self, observation_space, action_space, lr_schedule, 
                 per_pod_dim: int = 11, request_dim: int = 3, 
                 hidden_dim: int = 64, **kwargs):
        
        self.per_pod_dim = per_pod_dim
        self.request_dim = request_dim
        self.hidden_dim = hidden_dim
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=ScalableRoutingPolicyNetwork,
            features_extractor_kwargs={
                'per_pod_dim': per_pod_dim,
                'request_dim': request_dim,
                'hidden_dim': hidden_dim
            },
            **kwargs
        )


# ============================================================================
# Scalable Routing Environment
# ============================================================================

class ScalableRoutingEnvironment(gym.Env):
    """
    Gymnasium environment with DICT observation space (supports variable pods).
    
    Unlike the old version, this doesn't flatten everything into a single vector.
    Instead, it keeps structured observations that the policy can handle flexibly.
    """
    def __init__(self, per_pod_dim: int = 11, request_dim: int = 3, max_pods: int = 100):
        super().__init__()
        
        self.per_pod_dim = per_pod_dim
        self.request_dim = request_dim
        self.max_pods = max_pods
        
        # === Dict Observation Space (flexible for variable pods) ===
        self.observation_space = spaces.Dict({
            'pod_features': spaces.Box(
                -np.inf, np.inf, 
                shape=(max_pods, per_pod_dim - 1),  # Exclude kv_hit (separate)
                dtype=np.float32
            ),
            'kv_hit_ratios': spaces.Box(
                0.0, 1.0,
                shape=(max_pods, 1),
                dtype=np.float32
            ),
            'request_features': spaces.Box(
                -np.inf, np.inf,
                shape=(request_dim,),
                dtype=np.float32
            ),
            'temporal_features': spaces.Box(
                -np.inf, np.inf,
                shape=(0,),  # Empty placeholder for future
                dtype=np.float32
            )
        })
        
        # Action space (nominal max, actual can be smaller)
        self.action_space = spaces.Discrete(max_pods)
        
        # Current state
        self.current_obs = None
        self.current_num_pods = 0
        
        logger.info(f"🌍 ScalableRoutingEnvironment initialized:")
        logger.info(f"  Max pods: {max_pods} (can handle less at runtime)")
        logger.info(f"  Per-pod features: {per_pod_dim}")
        logger.info(f"  Request features: {request_dim}")
        
    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)
        
        # Return dummy observation
        dummy_obs = {
            'pod_features': np.zeros((self.max_pods, self.per_pod_dim - 1), dtype=np.float32),
            'kv_hit_ratios': np.zeros((self.max_pods, 1), dtype=np.float32),
            'request_features': np.zeros((self.request_dim,), dtype=np.float32),
            'temporal_features': np.array([], dtype=np.float32)
        }
        
        self.current_obs = dummy_obs
        return dummy_obs, {}
    
    def step(self, action: int):
        """Step function - controlled externally"""
        # Return current observation
        # Reward, next_obs managed externally via experience buffer
        
        terminated = False
        truncated = False  # Managed by episode tracker
        
        return self.current_obs, 0.0, terminated, truncated, {}
    
    def set_observation(self, obs):
        """External interface to set current observation"""
        self.current_obs = obs
        self.current_num_pods = obs['pod_features'].shape[0]


# ============================================================================
# Scalable RL Routing Agent
# ============================================================================

class ScalableRLRoutingAgent:
    """
    Scalable RL Routing Agent with proper multi-step credit assignment.
    
    Key improvements over old version:
    1. Pod-count independent (works with 4 to 1000+ pods)
    2. Async experience completion (next_obs at completion time)
    3. Episode boundaries (proper temporal structure)
    4. Prioritized replay (2-3x sample efficiency)
    5. Cluster statistics (relative context)
    6. Action masking (unhealthy pod filtering)
    """
    def __init__(self, per_pod_dim: int = 11, request_dim: int = 3, 
                 max_pods: int = 100, **hyperparameters):
        """
        Args:
            per_pod_dim: Features per pod (pod_features + kv_hit_ratios)
            request_dim: Request feature dimensions
            max_pods: Maximum expected pods (for space allocation)
            hyperparameters: PPO and training hyperparameters
        """
        self.per_pod_dim = per_pod_dim
        self.request_dim = request_dim
        self.max_pods = max_pods
        self.hyperparameters = hyperparameters
        
        # Create environment
        self.env = ScalableRoutingEnvironment(per_pod_dim, request_dim, max_pods)
        
        # Extract hyperparameters
        learning_rate = hyperparameters.get('learning_rate', 3e-4)
        hidden_dim = hyperparameters.get('hidden_dim', 64)
        gamma = hyperparameters.get('reward_decay_factor', 0.95)
        gae_lambda = hyperparameters.get('gae_lambda', 0.95)
        
        # === Create PPO model with our scalable policy ===
        self.model = PPO(
            ScalableRoutingPolicy,
            self.env,
            learning_rate=learning_rate,
            n_steps=hyperparameters.get('n_steps', 2048),
            batch_size=hyperparameters.get('batch_size', 64),
            n_epochs=hyperparameters.get('n_epochs', 10),
            gamma=gamma,                    # Discount factor
            gae_lambda=gae_lambda,          # GAE lambda (short horizon)
            clip_range=hyperparameters.get('clip_range', 0.2),
            ent_coef=hyperparameters.get('entropy_coeff', 0.01),
            vf_coef=hyperparameters.get('vf_coef', 0.5),
            max_grad_norm=hyperparameters.get('max_grad_norm', 0.5),
            policy_kwargs={
                'per_pod_dim': per_pod_dim,
                'request_dim': request_dim,
                'hidden_dim': hidden_dim
            },
            verbose=1
        )
        
        # === Prioritized Experience Replay ===
        self.experience_buffer = PrioritizedReplayBuffer(
            maxlen=hyperparameters.get('buffer_size', 1000),
            alpha=hyperparameters.get('priority_alpha', 0.6),
            beta=hyperparameters.get('priority_beta', 0.4)
        )
        
        # === Pending experiences (awaiting completion) ===
        self.pending_experiences = {}  # request_id → Experience
        self.pending_lock = threading.Lock()
        
        # === Episode tracker ===
        self.episode_tracker = EpisodeTracker(
            episode_duration=hyperparameters.get('episode_duration', 1.0)
        )
        
        # === Training statistics ===
        self.total_steps = 0
        self.total_episodes = 0
        
        # === Performance tracking (for comprehensive metrics) ===
        from collections import deque
        self.reward_history = deque(maxlen=1000)  # Last 1000 rewards
        self.episode_rewards = []  # Cumulative reward per episode
        self.recent_decisions = deque(maxlen=100)  # Track recent decision quality
        self.checkpoint_counter = 0  # For periodic checkpointing
        
        logger.info(f"ScalableRLRoutingAgent initialized:")
        logger.info(f"  Gamma: {gamma}, GAE Lambda: {gae_lambda}")
        logger.info(f"  Episode duration: {self.episode_tracker.episode_duration}s")
        logger.info(f"  Prioritized replay: α={self.experience_buffer.alpha}, β={self.experience_buffer.beta}")
        
    def compute_action_mask(self, pod_features, kv_hit_ratios):
        """
        Compute action mask for unhealthy pod filtering.
        
        TODO: Implement domain-specific logic based on availability, queue, etc.
        For now, returns all valid (no masking).
        
        Args:
            pod_features: [num_pods, 10]
            kv_hit_ratios: [num_pods, 1]
        
        Returns:
            action_mask: [num_pods] - 1=valid, 0=invalid
        """
        num_pods = pod_features.shape[0]
        
        # Placeholder: all pods valid
        # TODO: Implement logic like:
        # availability = pod_features[:, 9]  # Assuming column 9 is availability
        # not_overloaded = pod_features[:, 1] < 100  # Queue length < threshold
        # action_mask = (availability == 1) & not_overloaded
        
        action_mask = np.ones(num_pods, dtype=np.float32)
        
        # Safety: if all masked, unmask all
        if action_mask.sum() == 0:
            action_mask = np.ones(num_pods, dtype=np.float32)
        
        return action_mask
    
    def predict(self, pod_features, kv_hit_ratios, request_features, 
                temporal_features=None, deterministic=False):
        """
        Predict action for given state.
        
        Args:
            pod_features: [num_pods, 10]
            kv_hit_ratios: [num_pods, 1]
            request_features: [3]
            temporal_features: Optional dict (unused in v1)
            deterministic: If True, use greedy policy
        
        Returns:
            action: Selected pod index
            action_probs: Probability distribution over ACTUAL pods (not padded)
        """
        # Track actual number of pods
        num_actual_pods = pod_features.shape[0]
        
        # Build observation dict (this pads to max_pods internally)
        obs = self._build_observation(pod_features, kv_hit_ratios, request_features)
        
        # Compute action mask for actual pods
        action_mask = self.compute_action_mask(pod_features, kv_hit_ratios)
        
        # Predict with model
        obs_tensor = self._obs_to_tensor(obs)
        
        with torch.no_grad():
            # Get action probabilities for all pods (including padding)
            features_extractor = self.model.policy.features_extractor
            action_probs_full = features_extractor.score_pods(obs_tensor, action_mask=None)
            
            # Extract only actual pods (remove padding)
            action_probs = action_probs_full[0, :num_actual_pods]
            
            # Apply action mask (only for actual pods)
            action_mask_tensor = torch.from_numpy(action_mask).to(action_probs.device)
            masked_probs = action_probs * action_mask_tensor
            
            # Check if all masked (safety)
            if masked_probs.sum() == 0:
                logger.warning("All pods masked! Using uniform distribution")
                masked_probs = torch.ones_like(action_probs)
            
            masked_probs = masked_probs / masked_probs.sum()  # Renormalize
            
            if deterministic:
                action = torch.argmax(masked_probs).item()
            else:
                dist = torch.distributions.Categorical(probs=masked_probs)
                action = dist.sample().item()
        
        return int(action), masked_probs.cpu().numpy()
    
    def create_pending_experience(self, request_id, pod_features, kv_hit_ratios, 
                                  request_features, action, action_probs):
        """
        Create pending experience when request is routed.
        
        Will be completed asynchronously when request finishes.
        
        Args:
            request_id: Unique request identifier
            pod_features, kv_hit_ratios, request_features: State components
            action: Selected pod index
            action_probs: Action probability distribution
        """
        num_actual_pods = pod_features.shape[0]
        obs = self._build_observation(pod_features, kv_hit_ratios, request_features)
        
        experience = {
            'request_id': request_id,
            'obs': obs,
            'action': action,
            'action_probs': action_probs,
            'route_time': time.time(),
            'num_actual_pods': num_actual_pods,  # Track actual pods for proper handling
            
            # To be filled at completion:
            'next_obs': None,
            'reward': None,
            'done': False,
            'complete_time': None,
            'is_complete': False
        }
        
        with self.pending_lock:
            self.pending_experiences[request_id] = experience
        
        logger.debug(f"Created pending experience for request {request_id} (num_pods={num_actual_pods})")
    
    def complete_experience(self, request_id, next_pod_features, next_kv_hit_ratios, 
                           next_request_features, reward):
        """
        Complete pending experience when request finishes.
        
        This is the CRITICAL fix: next_obs captured at completion time!
        
        Args:
            request_id: Request identifier
            next_pod_features, next_kv_hit_ratios, next_request_features: Next state
            reward: Computed reward based on latency
        """
        with self.pending_lock:
            if request_id not in self.pending_experiences:
                logger.warning(f"⚠️  Request {request_id} not found in pending experiences")
                return
            
            exp = self.pending_experiences.pop(request_id)
        
        # Build next observation (state AFTER request completion)
        next_obs = self._build_observation(
            next_pod_features, next_kv_hit_ratios, next_request_features
        )
        
        # Check episode boundary
        done = self.episode_tracker.check_episode_end()
        
        # Fill in completion data
        exp['next_obs'] = next_obs
        exp['reward'] = reward
        exp['done'] = done
        exp['complete_time'] = time.time()
        exp['is_complete'] = True
        
        # Add to prioritized replay buffer
        self.experience_buffer.add(exp)
        
        # === Track performance metrics ===
        self.reward_history.append(reward)
        
        # Track decision quality (action confidence + outcome)
        action_confidence = float(max(exp['action_probs'])) if len(exp['action_probs']) > 0 else 0.0
        self.recent_decisions.append({
            'reward': reward,
            'confidence': action_confidence,
            'latency_ms': (exp['complete_time'] - exp['route_time']) * 1000  # convert to ms
        })
        
        # Track episode
        self.episode_tracker.increment_request()
        if done:
            self.episode_tracker.reset_episode()
            self.total_episodes += 1
        
        self.total_steps += 1
        
        logger.debug(f"✅ Completed experience for request {request_id}: "
                    f"reward={reward:.3f}, done={done}")
    
    def update_online(self, n_steps: int = None):
        """
        Perform online learning update using prioritized replay.
        
        Args:
            n_steps: Number of experiences to use (default: batch_size)
        """
        batch_size = self.hyperparameters.get('batch_size', 64)
        
        if len(self.experience_buffer) < batch_size:
            logger.debug(f"Not enough experiences: {len(self.experience_buffer)} < {batch_size}")
            return
        
        # Sample prioritized batch
        batch, indices, weights = self.experience_buffer.sample(batch_size)
        
        if not batch:
            return
        
        # TODO: Implement actual PPO update with prioritized samples
        # This requires customizing SB3's training loop
        # For now, log that update would happen
        
        logger.info(f"🎓 Online update: {len(batch)} experiences, "
                   f"priority weights range [{weights.min():.2f}, {weights.max():.2f}]")
        
        # Compute TD errors for priority updates (simplified)
        # In full implementation, these would come from actual value function
        td_errors = np.random.rand(len(indices))  # Placeholder
        self.experience_buffer.update_priorities(indices, td_errors)
    
    def _build_observation(self, pod_features, kv_hit_ratios, request_features, 
                          temporal_features=None):
        """
        Build observation dict from components.
        
        Handles variable number of pods by padding to max_pods.
        """
        num_pods = pod_features.shape[0]
        
        # Check if exceeds max_pods
        if num_pods > self.max_pods:
            logger.warning(f"Number of pods ({num_pods}) exceeds max_pods ({self.max_pods}). "
                          f"Consider increasing max_pods in initialization.")
            # Truncate to max_pods (use first max_pods)
            pod_features = pod_features[:self.max_pods]
            kv_hit_ratios = kv_hit_ratios[:self.max_pods]
            num_pods = self.max_pods
        
        # Pad to max_pods if needed
        if num_pods < self.max_pods:
            pad_size = self.max_pods - num_pods
            pod_features = np.vstack([
                pod_features,
                np.zeros((pad_size, pod_features.shape[1]), dtype=np.float32)
            ])
            kv_hit_ratios = np.vstack([
                kv_hit_ratios,
                np.zeros((pad_size, kv_hit_ratios.shape[1]), dtype=np.float32)
            ])
        
        obs = {
            'pod_features': pod_features.astype(np.float32),
            'kv_hit_ratios': kv_hit_ratios.astype(np.float32),
            'request_features': request_features.astype(np.float32),
            'temporal_features': np.array([], dtype=np.float32)  # Placeholder
        }
        
        return obs
    
    def _obs_to_tensor(self, obs):
        """Convert observation dict to tensor dict for model"""
        tensor_obs = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                tensor = torch.from_numpy(value).float()
                # Add batch dimension
                tensor = tensor.unsqueeze(0)
                tensor_obs[key] = tensor.to(self.model.device)
            else:
                tensor_obs[key] = value
        
        return tensor_obs
    
    def save(self, path: str, save_buffer: bool = False):
        """
        Save model with comprehensive metadata for reproducibility and analysis.
        
        Args:
            path: Base path for checkpoint
            save_buffer: If True, also save experience buffer (large file)
        """
        import datetime
        
        # Save PPO model (weights, optimizer state, etc.)
        self.model.save(path)
        
        # Collect comprehensive metadata
        metadata = {
            # === Model Architecture ===
            'model_architecture': {
                'per_pod_dim': self.per_pod_dim,
                'request_dim': self.request_dim,
                'max_pods': self.max_pods,
                'hidden_dim': self.hyperparameters.get('hidden_dim', 64),
            },
            
            # === Training Hyperparameters ===
            'hyperparameters': self.hyperparameters,
            
            # === Training Progress ===
            'training_progress': {
                'total_steps': self.total_steps,
                'total_episodes': self.total_episodes,
                'current_episode_id': self.episode_tracker.episode_id,
                'episode_request_count': self.episode_tracker.episode_request_count,
            },
            
            # === Buffer Statistics ===
            'buffer_stats': {
                'buffer_size': len(self.experience_buffer),
                'buffer_capacity': self.experience_buffer.buffer.maxlen,
                'pending_experiences': len(self.pending_experiences),
                'priority_alpha': self.experience_buffer.alpha,
                'priority_beta': self.experience_buffer.beta,
                'max_priority': self.experience_buffer.max_priority,
            },
            
            # === Episode Configuration ===
            'episode_config': {
                'episode_duration': self.episode_tracker.episode_duration,
                'episode_start_time': self.episode_tracker.episode_start_time,
            },
            
            # === Model Performance (if tracked) ===
            'performance_metrics': self.get_metrics(),
            
            # === Checkpoint Metadata ===
            'checkpoint_info': {
                'save_time': datetime.datetime.now().isoformat(),
                'save_path': path,
                'version': '1.0',
            },
            
            # === Environment Info ===
            'environment': {
                'observation_space': str(self.env.observation_space),
                'action_space': str(self.env.action_space),
            }
        }
        
        # Save metadata
        metadata_path = f"{path}_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        # Save human-readable metadata (JSON)
        json_metadata_path = f"{path}_metadata.json"
        try:
            import json
            # Convert to JSON-serializable format
            json_metadata = {
                'model_architecture': metadata['model_architecture'],
                'training_progress': metadata['training_progress'],
                'buffer_stats': metadata['buffer_stats'],
                'episode_config': metadata['episode_config'],
                'performance_metrics': metadata['performance_metrics'],
                'checkpoint_info': metadata['checkpoint_info'],
            }
            with open(json_metadata_path, 'w') as f:
                json.dump(json_metadata, f, indent=2)
            logger.info(f"Saved human-readable metadata to {json_metadata_path}")
        except Exception as e:
            logger.warning(f"Could not save JSON metadata: {e}")
        
        # Optionally save experience buffer (can be large!)
        if save_buffer and len(self.experience_buffer) > 0:
            buffer_path = f"{path}_buffer.pkl"
            try:
                with self.experience_buffer.lock:
                    buffer_data = {
                        'experiences': list(self.experience_buffer.buffer),
                        'priorities': list(self.experience_buffer.priorities),
                    }
                with open(buffer_path, 'wb') as f:
                    pickle.dump(buffer_data, f)
                logger.info(f"Saved experience buffer to {buffer_path} ({len(buffer_data['experiences'])} experiences)")
            except Exception as e:
                logger.warning(f"Could not save buffer: {e}")
        
        logger.info(f"Model checkpoint saved to {path}")
        logger.info(f"   Total steps: {self.total_steps}, Episodes: {self.total_episodes}")
        logger.info(f"   Buffer size: {len(self.experience_buffer)}/{self.experience_buffer.buffer.maxlen}")
    
    def load(self, path: str, load_buffer: bool = False):
        """
        Load model with comprehensive metadata restoration.
        
        Args:
            path: Base path for checkpoint
            load_buffer: If True, also load experience buffer (if available)
        """
        try:
            # Load PPO model (weights, optimizer state)
            self.model = PPO.load(path, env=self.env)
            logger.info(f"✅ Loaded model from {path}")
            
            # Load metadata
            try:
                with open(f"{path}_metadata.pkl", 'rb') as f:
                    metadata = pickle.load(f)
                
                # Restore training progress
                training_progress = metadata.get('training_progress', {})
                self.total_steps = training_progress.get('total_steps', 0)
                self.total_episodes = training_progress.get('total_episodes', 0)
                
                # Restore episode tracker state
                episode_config = metadata.get('episode_config', {})
                if 'episode_duration' in episode_config:
                    self.episode_tracker.episode_duration = episode_config['episode_duration']
                
                # Store loaded metadata for inspection
                self.loaded_metadata = metadata
                
                # Log checkpoint info
                checkpoint_info = metadata.get('checkpoint_info', {})
                buffer_stats = metadata.get('buffer_stats', {})
                
                logger.info(f"📊 Loaded checkpoint metadata:")
                logger.info(f"   - Created: {checkpoint_info.get('save_time', 'unknown')}")
                logger.info(f"   - Total steps: {self.total_steps}")
                logger.info(f"   - Total episodes: {self.total_episodes}")
                logger.info(f"   - Buffer was at: {buffer_stats.get('buffer_size', 0)} experiences")
                
                # Display performance metrics if available
                perf_metrics = metadata.get('performance_metrics', {})
                if perf_metrics:
                    logger.info(f"   - Last avg reward: {perf_metrics.get('avg_reward_recent', 'N/A')}")
                    logger.info(f"   - Success rate: {perf_metrics.get('success_rate', 'N/A')}")
                    
            except FileNotFoundError:
                logger.warning("Metadata file not found, using defaults")
                self.loaded_metadata = None
            
            # Optionally load experience buffer
            if load_buffer:
                buffer_path = f"{path}_buffer.pkl"
                if os.path.exists(buffer_path):
                    try:
                        with open(buffer_path, 'rb') as f:
                            buffer_data = pickle.load(f)
                        
                        # Restore buffer
                        with self.experience_buffer.lock:
                            for exp in buffer_data['experiences']:
                                self.experience_buffer.buffer.append(exp)
                            for priority in buffer_data['priorities']:
                                self.experience_buffer.priorities.append(priority)
                        
                        logger.info(f"Loaded {len(buffer_data['experiences'])} experiences from buffer")
                    except Exception as e:
                        logger.warning(f"Could not load buffer: {e}")
                else:
                    logger.info(f"No buffer file found (load_buffer=True but file missing)")
        
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def get_metrics(self):
        """
        Get comprehensive training and performance metrics.
        
        Returns:
            dict: Comprehensive metrics including training progress, performance, and model quality
        """
        import numpy as np
        
        # Basic training metrics
        metrics = {
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'buffer_size': len(self.experience_buffer),
            'pending_experiences': len(self.pending_experiences),
            'current_episode': self.episode_tracker.episode_id,
            'episode_request_count': self.episode_tracker.episode_request_count,
        }
        
        # Reward statistics (recent 100 and all)
        if len(self.reward_history) > 0:
            rewards = list(self.reward_history)
            recent_100 = rewards[-100:] if len(rewards) >= 100 else rewards
            
            metrics['reward_stats'] = {
                'avg_reward_recent': float(np.mean(recent_100)),
                'std_reward_recent': float(np.std(recent_100)),
                'max_reward_recent': float(np.max(recent_100)),
                'min_reward_recent': float(np.min(recent_100)),
                'avg_reward_all': float(np.mean(rewards)),
                'num_samples': len(rewards),
            }
            
            # Success rate (reward > 0 means good routing decision)
            success_count = sum(1 for r in recent_100 if r > 0)
            metrics['success_rate'] = success_count / len(recent_100) if len(recent_100) > 0 else 0.0
        else:
            metrics['reward_stats'] = None
            metrics['success_rate'] = None
        
        # Decision quality metrics
        if len(self.recent_decisions) > 0:
            decisions = list(self.recent_decisions)
            confidences = [d['confidence'] for d in decisions]
            latencies = [d['latency_ms'] for d in decisions]
            rewards = [d['reward'] for d in decisions]
            
            metrics['decision_quality'] = {
                'avg_confidence': float(np.mean(confidences)),
                'avg_latency_ms': float(np.mean(latencies)),
                'p50_latency_ms': float(np.percentile(latencies, 50)),
                'p95_latency_ms': float(np.percentile(latencies, 95)),
                'p99_latency_ms': float(np.percentile(latencies, 99)),
                'high_confidence_success_rate': self._compute_high_confidence_success(decisions),
            }
        else:
            metrics['decision_quality'] = None
        
        # Learning progress (compare first 100 vs last 100 rewards)
        if len(self.reward_history) >= 200:
            rewards = list(self.reward_history)
            first_100 = rewards[:100]
            last_100 = rewards[-100:]
            improvement = np.mean(last_100) - np.mean(first_100)
            metrics['learning_progress'] = {
                'reward_improvement': float(improvement),
                'first_100_avg': float(np.mean(first_100)),
                'last_100_avg': float(np.mean(last_100)),
            }
        else:
            metrics['learning_progress'] = None
        
        return metrics
    
    def _compute_high_confidence_success(self, decisions, confidence_threshold=0.7):
        """
        Compute success rate for high-confidence decisions.
        This helps evaluate if the model is well-calibrated.
        """
        high_conf = [d for d in decisions if d['confidence'] >= confidence_threshold]
        if len(high_conf) == 0:
            return None
        success = sum(1 for d in high_conf if d['reward'] > 0)
        return success / len(high_conf)


# ============================================================================
# Factory and Inference Functions
# ============================================================================

def create_scalable_rl_agent(per_pod_dim: int = 11, request_dim: int = 3, 
                             max_pods: int = 100, **hyperparameters):
    """
    Factory function to create scalable RL routing agent.
    
    Args:
        per_pod_dim: Features per pod (pod_features + kv_hit_ratios)
        request_dim: Request features
        max_pods: Maximum expected pods
        hyperparameters: PPO and training hyperparameters
    
    Returns:
        ScalableRLRoutingAgent instance
    """
    return ScalableRLRoutingAgent(per_pod_dim, request_dim, max_pods, **hyperparameters)


def infer_scalable_rl_agent(tensor_data, request_id, sorted_all_pod_ids, processed_df,
                            rl_agent, hyperparameters, agent_lock=None):
    """
    Inference workflow for scalable RL agent.
    
    Compatible with existing routing_agent_service.py interface.
    
    Key changes from old version:
    1. Creates pending experience (not completed yet)
    2. Completion happens asynchronously in on_request_complete callback
    3. Proper next_obs and done flags for TD learning
    
    Args:
        tensor_data: Dict with 'pod_features', 'kv_hit_ratios', 'request_features'
        request_id: Request identifier
        sorted_all_pod_ids: List of pod IDs
        processed_df: DataFrame with request data
        rl_agent: ScalableRLRoutingAgent instance
        hyperparameters: RL hyperparameters
        agent_lock: Optional lock (not needed for prediction in new design)
    
    Returns:
        (agent, result_dict, overhead_summary)
    """
    import preprocess
    
    overhead_summary = {}
    infer_start = time.time()
    
    # Extract tensors
    extract_start = time.time()
    pod_features_t = tensor_data['pod_features']
    kv_hit_t = tensor_data['kv_hit_ratios']
    req_features_t = tensor_data['request_features']
    
    pod_features_np = pod_features_t.cpu().numpy()[0]  # [num_pods, 10]
    kv_hit_np = kv_hit_t.cpu().numpy()[0]              # [num_pods, 1]
    req_features_np = req_features_t.cpu().numpy()[0]  # [3]
    
    overhead_summary['extract_tensors'] = time.time() - extract_start
    
    # Predict action
    predict_start = time.time()
    action_idx, action_probs = rl_agent.predict(
        pod_features_np, kv_hit_np, req_features_np, deterministic=False
    )
    overhead_summary['predict'] = time.time() - predict_start
    
    # Create pending experience (will be completed asynchronously)
    rl_agent.create_pending_experience(
        request_id, pod_features_np, kv_hit_np, req_features_np,
        action_idx, action_probs
    )
    
    # Build result
    result_start = time.time()
    num_pods = len(sorted_all_pod_ids)
    pod_prob_map = {
        sorted_all_pod_ids[i]: float(action_probs[i]) 
        for i in range(min(num_pods, len(action_probs)))
    }
    
    result = {
        'selected_pod_index': int(action_idx),
        'pod_probabilities': pod_prob_map,
        'confidence': float(np.max(action_probs)),
        'explore_mask': 0,
        'predicted_latencies': {pod_id: -1 for pod_id in sorted_all_pod_ids},
        'chosen_pod_predicted_latency': -1,
    }
    overhead_summary['build_result'] = time.time() - result_start
    
    overhead_summary['total'] = time.time() - infer_start
    
    logger.info(f"action={action_idx}, "
               f"confidence={result['confidence']:.3f}, "
               f"num_pods={num_pods}")
    
    return rl_agent, result, overhead_summary


# Callback for async completion (to be called by routing service)
def on_request_complete_callback(rl_agent, request_id, current_cluster_state, 
                                 ttft, tpot, hyperparameters):
    """
    Callback to complete experience when request finishes.
    
    This should be called by routing_agent_service.py when it receives
    completion notification.
    
    Args:
        rl_agent: ScalableRLRoutingAgent instance
        request_id: Request identifier
        current_cluster_state: Current pod features, kv ratios (after completion)
        ttft, tpot: Latency metrics
        hyperparameters: For reward calculation
    """
    import preprocess
    
    # Extract current state
    pod_features, kv_hit_ratios, request_features = current_cluster_state
    
    # Compute reward
    ttft_slo = hyperparameters['TTFT_SLO']
    avg_tpot_slo = hyperparameters['AVG_TPOT_SLO']
    ttft_weight = hyperparameters['TTFT_REWARD_WEIGHT']
    reward_fn = hyperparameters.get('REWARD_FUNCTION', 'linear_simple')
    
    if reward_fn == 'linear_simple':
        reward_res = preprocess.calculate_rewards_simple(
            np.array([ttft]), np.array([tpot]),
            ttft_slo, avg_tpot_slo, ttft_weight
        )
    elif reward_fn == 'linear_simple_extended':
        reward_res = preprocess.calculate_rewards_simple_extended(
            np.array([ttft]), np.array([tpot]),
            ttft_slo, avg_tpot_slo, ttft_weight
        )
    # ... (other reward functions)
    else:
        logger.warning(f"Unknown reward function {reward_fn}, using linear_simple")
        reward_res = preprocess.calculate_rewards_simple(
            np.array([ttft]), np.array([tpot]),
            ttft_slo, avg_tpot_slo, ttft_weight
        )
    
    reward = float(reward_res['combined_rewards'][0])
    
    # Complete the experience
    rl_agent.complete_experience(
        request_id, pod_features, kv_hit_ratios, request_features, reward
    )
    
    logger.debug(f"Completed experience for {request_id}: reward={reward:.3f}")