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

import time
import threading
import queue
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn


from tqdm.auto import tqdm
from gymnasium import spaces
from torchinfo import summary
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.logger import configure


from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, Callable, Union


from logger import logger


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
        hyperparameters: training hyperparameters
    
    Returns:
        ScalableRLRoutingAgent instance
    """


    return ScalableRLRoutingAgent(per_pod_dim, request_dim, max_pods, **hyperparameters)


def infer(request_id: str, prev_reward: float, pod_features: np.ndarray, kv_hit_ratios: np.ndarray, request_features: np.ndarray, temporal_features: np.ndarray, BROKER, timeout_in_seconds: float):
    # Generate random state
    infer_from_tensor_overhead_summary = {}
    infer_start_time = time.time()
    state = {
        'obs': {
            'pod_features': pod_features,
            'kv_hit_ratios': kv_hit_ratios,
            'request_features': request_features,
            'temporal_features': temporal_features,
        }
    }

    # obs = state['obs']
    # with torch.no_grad():
    #     obs_tensor = agent.model.policy.obs_to_tensor(obs)[0]
    #     distribution = agent.model.policy.get_distribution(obs_tensor)
    #     action = distribution.get_actions(deterministic=False)
    #     action_probs = distribution.distribution.probs.cpu().numpy()[0]  # [num_pods]
    # pod_idx = int(action.item())
    
    # logger.debug(f"Direct inference for {request_id}: action={pod_idx}, confidence={action_probs[pod_idx]:.3f}")
    # return pod_idx, action_probs
    
    pending = BROKER.submit(request_id=request_id, state=state, prev_reward=prev_reward)
    # timeout_in_seconds = 5 # TODO: inference should be made within less than 100ms
    decision_result = BROKER.wait_for_decision(request_id, timeout=timeout_in_seconds)
    
    if decision_result is None:
        logger.error(f"Decision timed out (timeout={timeout_in_seconds}), requestID, {request_id}")
        assert False
    pod_idx, _ = decision_result
    # if pod_idx is None:
    #     # fallback policy (your existing logic)
    #     pod_idx = 0
    BROKER.set_decision(request_id, pod_idx)
    
    BROKER.pop(request_id)
    infer_from_tensor_overhead_summary['total'] = time.time() - infer_start_time
    logger.info(f"scalable_rl_routing_agent, infer, request_id={request_id}, action={pod_idx}, took={infer_from_tensor_overhead_summary['total']:.3f}s")
    return pod_idx, infer_from_tensor_overhead_summary



# ============================================================================
# Broker
# ============================================================================

@dataclass
class PendingReq:
    request_id: str
    state: Dict[str, Any]                      # must include "obs"
    prev_reward: Optional[float] = None        # reward for the *previous* request
    decision_event: threading.Event = field(default_factory=threading.Event)
    decision_action: Optional[int] = None
    decision_probs: Optional[Any] = None       # action probabilities from policy

class RequestBroker:
    def __init__(self, maxsize: int = 10000):
        '''
        _by_id access is protected by _lock.
        This is for sb3 SubprocVecEnv, where multiple workers run independently (XXX)
        '''
        self._queue = queue.Queue(maxsize=maxsize)
        self._by_id: Dict[str, PendingReq] = {}
        self._lock = threading.Lock()

    def submit(self, request_id: str, state: Dict[str, Any],
               prev_reward: Optional[float]) -> PendingReq:
        pr = PendingReq(request_id=request_id, state=state, prev_reward=prev_reward)
        with self._lock:
            self._by_id[request_id] = pr
        self._queue.put(pr)
        return pr

    def get_next(self, timeout: Optional[float] = None) -> PendingReq:
        return self._queue.get(timeout=timeout)

    def set_decision(self, request_id: str, action: int, probs: Optional[Any] = None):
        with self._lock:
            pr = self._by_id.get(request_id)
        if pr:
            RED_COLOR = "\033[91m"
            RESET_COLOR = "\033[0m" 
            print(f"{RED_COLOR}set_decision{RESET_COLOR}", action)

            pr.decision_action = int(action)
            pr.decision_probs = probs # TODO: actison probabilities for debugging
            pr.decision_event.set()
        else:
            logger.error(f"Request {request_id} sets decision but not found in broker")
            assert False

    def wait_for_decision(self, request_id: str, timeout: Optional[float]):
        """
        Wait for decision and return (action, probs) tuple.
        Returns None if timeout or not found.
        """
        with self._lock:
            pr = self._by_id.get(request_id)
        if pr is None:
            logger.error(f"Request {request_id} waits for decision but not found in broker")
            assert False
        ok = pr.decision_event.wait(timeout)
        if not ok:
            logger.error(f"pr.decision_event timed out (timeout={timeout})...  Request {request_id}")
            return None
        return (pr.decision_action, pr.decision_probs) if ok else None

    def pop(self, request_id: str):
        with self._lock:
            self._by_id.pop(request_id, None)


# ============================================================================
# Envs
# ============================================================================

class GatewayRequestSource:
    def __init__(self, broker: RequestBroker):
        self.broker = broker

    def get_next(self, timeout: Optional[float] = None) -> PendingReq:
        return self.broker.get_next(timeout=timeout)

@dataclass
class Request:
    pending: PendingReq
    broker: RequestBroker

    @property
    def state(self) -> Dict[str, Any]:
        return self.pending.state

    def get_obs(self) -> Dict[str, Any]:
        return self.pending.state["obs"]

    def route(self, pod_idx: int):
        # TODO: action probabilities for debugging
        self.broker.set_decision(self.pending.request_id, pod_idx)


class EpisodeLengthWrapper(gym.Wrapper):
    def __init__(self, env, horizon: int):
        super().__init__(env)
        self.horizon = horizon
        self._ts = 0

    def reset(self, **kwargs):
        self._ts = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        self._ts += 1
        if self._ts >= self.horizon:
            term = True  # end episode every `horizon` steps
        return obs, rew, term, trunc, info


class EpisodeCounterWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._episode_count = 0
    
    def reset(self, **kwargs):
        self._episode_count += 1
        obs, info = self.env.reset(**kwargs)
        info = dict(info or {})
        info['episode_count'] = self._episode_count
        return obs, info
    
    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        info = dict(info or {})
        info['episode_count'] = self._episode_count
        return obs, rew, term, trunc, info


class RealTimeWrapper(gym.Wrapper):
    def __init__(self, env, period_s: float):
        super().__init__(env)
        self.period_s = float(period_s)
        self._next_tick = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        now = time.perf_counter()
        self._next_tick = now + self.period_s
        info = dict(info or {})
        info["period_s"] = self.period_s
        return obs, info

    def step(self, action):
        start = time.perf_counter()
        obs, r, term, trunc, info = self.env.step(action)
        now = time.perf_counter()
        sleep_for = self._next_tick - now
        if sleep_for > 0:
            time.sleep(sleep_for)
            jitter = 0.0
            self._next_tick += self.period_s
        else:
            jitter = -sleep_for
            missed = int((-sleep_for) // self.period_s) + 1
            self._next_tick += missed * self.period_s
        info = dict(info or {})
        info.update({"step_time_s": now - start, "neg_slack_s": jitter})
        return obs, r, term, trunc, info


class ScalableRoutingEnvironment(gym.Env):
    """
    Gymnasium environment with DICT observation space (supports variable pods).
    
    Unlike the old version, this doesn't flatten everything into a single vector.
    Instead, it keeps structured observations that the policy can handle flexibly.

    Wanyu 10/14/2025
    1. Change max_pods to num_pods
    2. Use Request class to manage requests (interface to Gateway)
    """
    def __init__(self, num_requests: int, per_pod_dim: int = 11, request_dim: int = 3, source: GatewayRequestSource=None):
        super().__init__()

        self.num_requests = num_requests
        self.per_pod_dim = per_pod_dim
        self.request_dim = request_dim
        self.source = source

        # Just a placeholder, first dimension = 1 for initialization, dynamically set for each step
        self.observation_space = spaces.Dict({
            'pod_features': spaces.Box(-np.inf, np.inf, shape=(1, per_pod_dim - 1), dtype=np.float32),
            'kv_hit_ratios': spaces.Box(0.0, 1.0, shape=(1, 1), dtype=np.float32),
            'request_features': spaces.Box(-np.inf, np.inf, shape=(request_dim,), dtype=np.float32),
            'temporal_features': spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
        })
   
        self.action_space = spaces.Discrete(1)

        self.request_count = 0
        self._request: Optional[Request] = None
        self._first_reward: float = 0.0
        
        logger.info(f"🌍 ScalableRoutingEnvironment initialized:")
        logger.info(f"  Number of requests: {num_requests}")
        logger.info(f"  Per-pod features: {per_pod_dim}")
        logger.info(f"  Request features: {request_dim}")

    def _pull(self) -> Request:
        assert self.source is not None, "GatewayRequestSource required"
        pending: PendingReq = self.source.get_next(timeout=None)  # blocks until /infer submits
        return Request(pending=pending, broker=self.source.broker)

    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)

        GREEN = '\033[92m'
        RESET = '\033[0m'
        RED = '\033[91m'
        logger.info(f"{GREEN}Resetting environment...{RESET}")

        self.request_count = 0
        print(f"{RED}pulling request{RESET}")
        self._request = self._pull()
        print(f"{RED}request pulled{RESET}")
        self._first_reward = float(self._request.pending.prev_reward or 0.0)
        observation = self._request.get_obs()
        self.update_space(observation['pod_features'].shape[0])
        info = self._request.state # dict()

        logger.info(f"ScalableRoutingEnvironment reset with request {self._request.pending.request_id}, \
            prev_reward {self._first_reward}")
        
        return observation, info
    
    # TODO: action probabilities for debugging
    # this is the entry point. I think it is gym's internal function... how can we get the action probabilities?
    def step(self, action: int):
        self._request.route(action)

        next_req = self._pull()

        observation = next_req.get_obs()

        self.update_space(observation['pod_features'].shape[0])
        reward = float(next_req.pending.prev_reward) if self.request_count > 0 else self._first_reward
        info = next_req.state # dict()
        self.request_count += 1

        terminated = (self.request_count == self.num_requests)
        truncated = False
        
        self._request = next_req
        
        return observation, reward, terminated, truncated, info

    def update_space(self, num_pods: int):
        self.observation_space['pod_features'] = spaces.Box(-np.inf, np.inf, shape=(num_pods, self.per_pod_dim - 1), dtype=np.float32)
        self.observation_space['kv_hit_ratios'] = spaces.Box(0.0, 1.0, shape=(num_pods, 1), dtype=np.float32)
        self.action_space = spaces.Discrete(num_pods)


# ============================================================================
# Policies
# ============================================================================



class PodFeatExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, 
                 per_pod_dim: int = 11,  # pod_features(10) + kv_hit(1)
                 request_dim: int = 3,
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
        self.num_pods = None
        
        super().__init__(observation_space, features_dim)
        
        logger.info(f"🏗️ PodFeatExtractor initialized with feature dimension {features_dim}")
        

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
        
        batch_size = pod_features.shape[0]
        num_pods = pod_features.shape[1]
        assert num_pods == kv_hit_ratios.shape[1], "Number of pods in pod_features and kv_hit_ratios must match"
        self.num_pods = num_pods
        
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
        features = torch.cat([
            combined_pod_features,     # [batch, num_pods, 11]
            expanded_request,          # [batch, num_pods, 3]
            expanded_cluster_stats     # [batch, num_pods, 44]
        ], dim=2).view(batch_size * num_pods, -1)
        # Shape: [batch * num_pods, 58]
        
        return features


class PodScorer(nn.Module):
    """
    Custom network for policy and value function (mlp_extractor).
    It receives as input per pod features extracted by the feature extractor.
    Input size is independent of the number of pods.

    SB3: obs ──> features_extractor ──>  mlp_extractor  ──>  heads (action_net / value_net)
                  (produces features_dim)     

    :param feature_dim: dimension of the features extracted with the features_extractor
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,  
        feature_dim: int, 
        hidden_dim: int = 64, 
        last_layer_dim_pi: int = 1, 
        last_layer_dim_vf: int = 1, 
        ):

        super(PodScorer, self).__init__()
        
        self.pod_scorer_pi = None
        self.pod_scorer_vf = None

        if last_layer_dim_pi > 0:
            self.pod_scorer_pi = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),       # 58 → 64
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim // 2),         # 64 → 32
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, last_layer_dim_pi)                   # 32 → last_layer_dim_pi (score)
            )

        if last_layer_dim_vf > 0:
            self.pod_scorer_vf = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),       # 58 → 64
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim // 2),         # 64 → 32
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, last_layer_dim_vf)                   # 32 → last_layer_dim_vf (score)
            )

        self.latent_dim_pi = last_layer_dim_pi # for easy rewrite of _build()
        self.latent_dim_vf = last_layer_dim_vf # for easy rewrite of _build()
        self.batch_size = None
        self.num_pods = None


    def get_num_pods(self):
        assert self.num_pods is not None, "num_pods is not set"
        return self.num_pods
    
    def set_num_pods(self, num_pods: int):
        self.num_pods = num_pods
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num_pods = self.get_num_pods()
        self.batch_size = features.shape[0] // num_pods
        if self.pod_scorer_vf is None:
            return self.forward_actor(features)
        elif self.pod_scorer_pi is None: 
            return self.forward_critic(features)
        else:
            return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        if self.pod_scorer_pi is None:
            raise ValueError("pod_scorer_pi is not initialized")

        policy_pod_scores = self.pod_scorer_pi(features)  # [batch*num_pods, 1]
        policy_pod_scores = policy_pod_scores.view(self.batch_size, -1)  # [batch, num_pods*last_layer_dim_pi]
        
        return policy_pod_scores

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        if self.pod_scorer_vf is None:
            raise ValueError("pod_scorer_vf is not initialized")

        value_pod_scores = self.pod_scorer_vf(features)  # [batch*num_pods, 1]
        value_pod_scores = value_pod_scores.view(self.batch_size, -1)  # [batch, num_pods*last_layer_dim_vf]
        value_pod_scores = value_pod_scores.mean(dim=1, keepdim=True)  # [batch, 1]
        
        return value_pod_scores



class ActorCriticRoutingPolicy(ActorCriticPolicy):
    """
    Custom Actor-Critic policy using our scalable architecture.
    
    Integrates with SB3's PPO while maintaining pod-independent design.
    """
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space, 
        lr_schedule: Callable[[float], float],
        per_pod_dim: int = 11, 
        request_dim: int = 3, 
        hidden_dim: int = 64, 
        last_layer_dim_pi: int = 1,
        last_layer_dim_vf: int = 1,
        **kwargs):
        
        self.feature_dim = per_pod_dim * 5 + request_dim # 5: mean, std, max, min and raw
        self.hidden_dim = hidden_dim
        self.last_layer_dim_pi = last_layer_dim_pi
        self.last_layer_dim_vf = last_layer_dim_vf
        
        super(ActorCriticRoutingPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=PodFeatExtractor,
            features_extractor_kwargs={
                'per_pod_dim': per_pod_dim,
                'request_dim': request_dim,
            },
            **kwargs
        )

    def _build_mlp_extractor(self) -> None:
        """
        https://github.com/DLR-RM/stable-baselines3/blob/d487f2d2355a6cf81ea26a0bbbdf1a727ca2a886/stable_baselines3/common/policies.py#L570
        
        forward: https://github.com/DLR-RM/stable-baselines3/blob/d487f2d2355a6cf81ea26a0bbbdf1a727ca2a886/stable_baselines3/common/policies.py#L636
        """
        
        self.mlp_extractor = PodScorer(self.feature_dim, \
            self.hidden_dim, self.last_layer_dim_pi, self.last_layer_dim_vf)
        
        self.mlp_extractor.set_num_pods(1)
        logger.info(f"🧠 MLP Extractor Architecture: \n")
        model_stats = summary(self.mlp_extractor, input_size=(1, self.feature_dim))
        self.mlp_extractor.set_num_pods(None)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        super()._build(lr_schedule)
        assert isinstance(self.action_dist, (CategoricalDistribution))
        self.action_net = nn.Identity() # Original sb3 implementation is action_logits = nn.Linear(latent_dim, self.action_dim)
        # self.value_net = nn.Identity()


    def extract_features(self, obs, features_extractor) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:

        pi_features, vf_features = super().extract_features(obs, features_extractor)
        num_pods = self.features_extractor.num_pods
        self.features_extractor.set_num_pods(num_pods)

        return pi_features, vf_features



# ============================================================================
# Agent
# ============================================================================

BROKER = RequestBroker()
SOURCE = GatewayRequestSource(BROKER)

# ============================================================================
# Scalable RL Routing Agent
# ============================================================================

class ScalableRLRoutingAgent:

    def __init__(
        self, 
        per_pod_dim: int = 11, 
        request_dim: int = 3, 
        max_pods: int = 100, 
        inference_mode: bool = False,
        rl: str = 'PPO',
        use_prioritized_replay: bool = False, 
        **hyperparameters
        ):
        """
        Args:
            per_pod_dim: Features per pod (pod_features + kv_hit_ratios)
            request_dim: Request feature dimensions
            max_pods: Maximum expected pods (for space allocation)
            hyperparameters: PPO and training hyperparameters
        """

        RED_COLOR = "\033[91m"
        RESET_COLOR = "\033[0m" 
        logger.info(f"{RED_COLOR}🤖 ScalableRLRoutingAgent initializing... \
            per_pod_dim={per_pod_dim}, request_dim={request_dim}, max_pods={max_pods}, \
            hyperparameters={hyperparameters}{RESET_COLOR}")

        self.per_pod_dim = per_pod_dim
        self.request_dim = request_dim
        self.max_pods = max_pods # XXX: useless
        self.hyperparameters = hyperparameters
        
        # Create environment
        self.static_num_pods = hyperparameters.get('static_num_pods', False)
        self.env = self.make_env(hyperparameters.get('horizon', 1024))
        self.setup_model(rl, per_pod_dim, request_dim, hyperparameters)
        
        # === Prioritized Experience Replay ===
        if use_prioritized_replay:
            self.experience_buffer = PrioritizedReplayBuffer(
                maxlen=hyperparameters.get('buffer_size', 1000),
                alpha=hyperparameters.get('priority_alpha', 0.6),
                beta=hyperparameters.get('priority_beta', 0.4)
            )
        
        
        # === Training statistics ===
        self.total_steps = 0
        self.total_episodes = 0
    
        logger.info(f"ScalableRLRoutingAgent initialization complete")


    def make_env(self, horizon: int):

        RED_COLOR = "\033[91m"
        RESET_COLOR = "\033[0m" 
        logger.info(f"{RED_COLOR}Making environment...{RESET_COLOR}")


        env = ScalableRoutingEnvironment(
            num_requests=10_000_000_000_000, # effectively infinite; EpisodeLengthWrapper handles resets
            per_pod_dim=self.per_pod_dim,
            request_dim=self.request_dim,
            source = SOURCE,
        )

        env = Monitor(env)
        env = EpisodeLengthWrapper(env, horizon=horizon)
        env = EpisodeCounterWrapper(env)
        if self.static_num_pods:
            env = DummyVecEnv([lambda: env])

        logger.info(f"Environment created with horizon {horizon}, \
            this should be the number of requests per workload.")

        return env
        

    def setup_model(self, rl: str, per_pod_dim: int, request_dim: int, hyperparameters: dict):
        # Extract hyperparameters
        learning_rate = hyperparameters.get('learning_rate', 3e-4)
        hidden_dim = hyperparameters.get('hidden_dim', 64)
        gamma = hyperparameters.get('reward_decay_factor', 1)
        gae_lambda = hyperparameters.get('gae_lambda', 0.95)

        if rl == 'PG':
            pass
        elif rl == 'PPO':
            # === Create PPO model with our scalable policy ===
            # TODO: revisit hyperparameters
            self.model = PPO(
                ActorCriticRoutingPolicy,
                self.env,
                learning_rate=learning_rate,
                n_steps=hyperparameters.get('n_steps', 256), # number of env steps per policy update, must less that horizon
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
                    'hidden_dim': hidden_dim,
                    'last_layer_dim_pi': hyperparameters.get('last_layer_dim_pi', 1),
                    'last_layer_dim_vf': hyperparameters.get('last_layer_dim_vf', 0),
                },
                verbose=1
            )
        else:
            raise ValueError(f"{rl} not supported")
    

    def train(self, total_timesteps: int, save_path: str):
        if self.static_num_pods:
            self.model.learn(total_timesteps=total_timesteps, progress_bar=True)
        else:
            trainer = Trainer(self.model, self.env, log_dir="./logs")
            trainer.train(total_timesteps)
        self.total_steps = total_timesteps
        self.save(save_path) ## TODO: match


    def predict_sb3(self, pod_features, kv_hit_ratios, request_features):
        assert self.num_pods == pod_features.shape[0]
        
        # Build observation dict (this pads to max_pods internally)
        obs = build_observation(self.num_pods, pod_features, kv_hit_ratios, request_features)
        action, _ = self.model.predict(obs)

        return int(action)
    
    
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
            
            # # === Training Progress ===
            # 'training_progress': {
            #     'total_steps': self.total_steps,
            #     'total_episodes': self.total_episodes,
            #     'current_episode_id': self.episode_tracker.episode_id,
            #     'episode_request_count': self.episode_tracker.episode_request_count,
            # },
            
            # # === Buffer Statistics ===
            # 'buffer_stats': {
            #     'buffer_size': len(self.experience_buffer),
            #     'buffer_capacity': self.experience_buffer.buffer.maxlen,
            #     'pending_experiences': len(self.pending_experiences),
            #     'priority_alpha': self.experience_buffer.alpha,
            #     'priority_beta': self.experience_buffer.beta,
            #     'max_priority': self.experience_buffer.max_priority,
            # },
            
            # # === Episode Configuration ===
            # 'episode_config': {
            #     'episode_duration': self.episode_tracker.episode_duration,
            #     'episode_start_time': self.episode_tracker.episode_start_time,
            # },
            
            # # === Model Performance (if tracked) ===
            # 'performance_metrics': self.get_metrics(),
            
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
                # 'training_progress': metadata['training_progress'],
                # 'buffer_stats': metadata['buffer_stats'],
                # 'episode_config': metadata['episode_config'],
                # 'performance_metrics': metadata['performance_metrics'],
                'checkpoint_info': metadata['checkpoint_info'],
            }
            with open(json_metadata_path, 'w') as f:
                json.dump(json_metadata, f, indent=2)
            logger.info(f"Saved human-readable metadata to {json_metadata_path}")
        except Exception as e:
            logger.warning(f"Could not save JSON metadata: {e}")
        
        # # Optionally save experience buffer (can be large!)
        # if save_buffer and len(self.experience_buffer) > 0:
        #     buffer_path = f"{path}_buffer.pkl"
        #     try:
        #         with self.experience_buffer.lock:
        #             buffer_data = {
        #                 'experiences': list(self.experience_buffer.buffer),
        #                 'priorities': list(self.experience_buffer.priorities),
        #             }
        #         with open(buffer_path, 'wb') as f:
        #             pickle.dump(buffer_data, f)
        #         logger.info(f"Saved experience buffer to {buffer_path} ({len(buffer_data['experiences'])} experiences)")
        #     except Exception as e:
        #         logger.warning(f"Could not save buffer: {e}")
        
        logger.info(f"Model checkpoint saved to {path}")
        # logger.info(f"   Total steps: {self.total_steps}, Episodes: {self.total_episodes}")
        # logger.info(f"   Buffer size: {len(self.experience_buffer)}/{self.experience_buffer.buffer.maxlen}")
   
    
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
                
                # # Restore training progress
                # training_progress = metadata.get('training_progress', {})
                # self.total_steps = training_progress.get('total_steps', 0)
                # self.total_episodes = training_progress.get('total_episodes', 0)
                
                # # Restore episode tracker state
                # episode_config = metadata.get('episode_config', {})
                # if 'episode_duration' in episode_config:
                #     self.episode_tracker.episode_duration = episode_config['episode_duration']
                
                # Store loaded metadata for inspection
                self.loaded_metadata = metadata
                
                # Log checkpoint info
                checkpoint_info = metadata.get('checkpoint_info', {})
                # buffer_stats = metadata.get('buffer_stats', {})
                
                logger.info(f"📊 Loaded checkpoint metadata:")
                logger.info(f"   - Created: {checkpoint_info.get('save_time', 'unknown')}")
                # logger.info(f"   - Total steps: {self.total_steps}")
                # logger.info(f"   - Total episodes: {self.total_episodes}")
                # logger.info(f"   - Buffer was at: {buffer_stats.get('buffer_size', 0)} experiences")
                
                # # Display performance metrics if available
                # perf_metrics = metadata.get('performance_metrics', {})
                # if perf_metrics:
                #     logger.info(f"   - Last avg reward: {perf_metrics.get('avg_reward_recent', 'N/A')}")
                #     logger.info(f"   - Success rate: {perf_metrics.get('success_rate', 'N/A')}")
                    
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
    
    # def get_metrics(self):
    #     """
    #     Get comprehensive training and performance metrics.
        
    #     Returns:
    #         dict: Comprehensive metrics including training progress, performance, and model quality
    #     """
    #     import numpy as np
        
    #     # Basic training metrics
    #     metrics = {
    #         'total_steps': self.total_steps,
    #         'total_episodes': self.total_episodes,
    #         'buffer_size': len(self.experience_buffer),
    #         'pending_experiences': len(self.pending_experiences),
    #         'current_episode': self.episode_tracker.episode_id,
    #         'episode_request_count': self.episode_tracker.episode_request_count,
    #     }
        
    #     # Reward statistics (recent 100 and all)
    #     if len(self.reward_history) > 0:
    #         rewards = list(self.reward_history)
    #         recent_100 = rewards[-100:] if len(rewards) >= 100 else rewards
            
    #         metrics['reward_stats'] = {
    #             'avg_reward_recent': float(np.mean(recent_100)),
    #             'std_reward_recent': float(np.std(recent_100)),
    #             'max_reward_recent': float(np.max(recent_100)),
    #             'min_reward_recent': float(np.min(recent_100)),
    #             'avg_reward_all': float(np.mean(rewards)),
    #             'num_samples': len(rewards),
    #         }
            
    #         # Success rate (reward > 0 means good routing decision)
    #         success_count = sum(1 for r in recent_100 if r > 0)
    #         metrics['success_rate'] = success_count / len(recent_100) if len(recent_100) > 0 else 0.0
    #     else:
    #         metrics['reward_stats'] = None
    #         metrics['success_rate'] = None
        
    #     # Decision quality metrics
    #     if len(self.recent_decisions) > 0:
    #         decisions = list(self.recent_decisions)
    #         confidences = [d['confidence'] for d in decisions]
    #         latencies = [d['latency_ms'] for d in decisions]
    #         rewards = [d['reward'] for d in decisions]
            
    #         metrics['decision_quality'] = {
    #             'avg_confidence': float(np.mean(confidences)),
    #             'avg_latency_ms': float(np.mean(latencies)),
    #             'p50_latency_ms': float(np.percentile(latencies, 50)),
    #             'p95_latency_ms': float(np.percentile(latencies, 95)),
    #             'p99_latency_ms': float(np.percentile(latencies, 99)),
    #             'high_confidence_success_rate': self._compute_high_confidence_success(decisions),
    #         }
    #     else:
    #         metrics['decision_quality'] = None
        
    #     # Learning progress (compare first 100 vs last 100 rewards)
    #     if len(self.reward_history) >= 200:
    #         rewards = list(self.reward_history)
    #         first_100 = rewards[:100]
    #         last_100 = rewards[-100:]
    #         improvement = np.mean(last_100) - np.mean(first_100)
    #         metrics['learning_progress'] = {
    #             'reward_improvement': float(improvement),
    #             'first_100_avg': float(np.mean(first_100)),
    #             'last_100_avg': float(np.mean(last_100)),
    #         }
    #     else:
    #         metrics['learning_progress'] = None
        
    #     return metrics
    
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



class Trainer:
    """
    A drop-in manual training loop for dynamic Gym environments.
    Keeps SB3 logging, callbacks, and progress bar.
    """

    def __init__(self, model, env, log_dir="./logs", eval_env=None):
        self.model = model
        self.env = env
        self.eval_env = eval_env
        self.logger = configure(log_dir, ["stdout", "tensorboard"])
        self.model.set_logger(self.logger)
        self.total_steps = 0

        # === Setup callbacks (use any you like) ===
        checkpoint_callback = CheckpointCallback(
            save_freq=10_000,
            save_path=f"{log_dir}/checkpoints",
            name_prefix="rl_model",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )

        callbacks = [checkpoint_callback]

        if eval_env is not None:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=f"{log_dir}/best_model",
                log_path=f"{log_dir}/results",
                eval_freq=5_000,
                deterministic=True,
                render=False,
            )
            callbacks.append(eval_callback)

        self.callback = CallbackList(callbacks)
        self.callback.init_callback(self.model)  # must initialize manually

    def train(self, total_timesteps: int):
        obs, _ = self.env.reset()  # <-- FIXED here
        done = False
        episode_reward = 0.0

        pbar = tqdm(total=total_timesteps, desc="Training", unit="steps")

        for step in range(total_timesteps):
            # === 1. Predict ===
            action, _ = self.model.predict(obs, deterministic=False)
            obs, reward, done, truncated, info = self.env.step(action)

            # === 2. Record reward ===
            episode_reward += reward
            self.model.logger.record("rollout/reward", reward)

            # === 3. Handle episode end ===
            if done:
                self.model.logger.record("rollout/episode_reward", episode_reward)
                obs, _ = self.env.reset()
                episode_reward = 0.0

            # === 4. Callbacks & progress ===
            continue_training = self.callback.on_step()
            if not continue_training:
                print("Training stopped early by callback.")
                break

            if step % 1000 == 0:
                self.model.logger.dump(step=step)

            pbar.update(1)

        pbar.close()
        self.callback.on_training_end()
        print("Training complete ✅")


    def save(self, path):
        self.model.save(path)
        print(f"Model saved to {path}")



# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":

    logger.info("🧪 Testing ScalableRLRoutingAgent...")
    
    num_pods = 4
    # Create agent
    agent = create_scalable_rl_agent(
        num_pods=num_pods,
        per_pod_dim=11,
        request_dim=3,
        max_pods=10,
        learning_rate=3e-4,
        reward_decay_factor=1,
        gae_lambda=0.95,
        # episode_duration=1.0,
        n_steps=5,
        horizon=10,
        batch_size=2,
        last_layer_dim_vf=1,
    )
    
    # Simulate routing workflow
    train_thread = threading.Thread(
        target=agent.train,
        kwargs={"total_timesteps": 200_000, "save_path": "scalable_rl_agent.pth"},
        daemon=True,   # dies with the main process
    )
    train_thread.start()
    logger.info("🏋️ Training RL agent...")
    import random
    for i in range(20):
        pod_features = np.random.randn(num_pods, 10
        ).astype(np.float32)
        kv_hit_ratios = np.random.rand(num_pods, 1).astype(np.float32)
        request_features = np.random.randn(3).astype(np.float32)
        temporal_features = np.array([1], dtype=np.float32)
        request_id = f"req_{i}"
        prev_reward = random.random()
        # pod_idx, action_probs = infer(request_id, prev_reward, pod_features, kv_hit_ratios, request_features, temporal_features, BROKER, agent)
        if i % 10 == 0:
            timeout_in_seconds = 30
        else:
            timeout_in_seconds = 0.1
        pod_idx = infer(request_id, prev_reward, pod_features, kv_hit_ratios, request_features, temporal_features, BROKER, timeout_in_seconds)
        
        ##### END
        
        # Simulate completion (in real system, this happens asynchronously)
        time.sleep(0.01)
        
        # confidence = action_probs[pod_idx] if action_probs is not None else 0.0
        confidence = 0.0
        logger.info(f"Step {i}: action={pod_idx}, prev_reward={prev_reward:.2f}, "
                   f"confidence={confidence:.3f}, num_pods={num_pods}")
    
    # Check metrics
    # metrics = agent.get_metrics()
    # logger.info(f"📊 Final metrics: {metrics}")
    logger.info("✅ Test completed successfully!")


