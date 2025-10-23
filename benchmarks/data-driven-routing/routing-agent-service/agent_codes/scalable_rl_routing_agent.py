from __future__ import annotations

from routing_agent_service import BLUE_COLOR, RED_COLOR, RESET_COLOR, GREEN_COLOR
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
from telnetlib import EC
import time
import threading
import queue
import pickle
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn


from tqdm.auto import tqdm
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule, PyTorchObs
from stable_baselines3.common.distributions import CategoricalDistribution, Distribution
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import get_linear_fn


from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, Callable, Union


from logger import logger

NUM_PODS = 7




def infer(request_id: str, prev_reward: float, pod_features: np.ndarray, kv_hit_ratios: np.ndarray, request_features: np.ndarray, temporal_features: np.ndarray, broker: RequestBroker, timeout_in_seconds: float):
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
    
    pending = broker.submit(request_id=request_id, state=state, prev_reward=prev_reward)
    # timeout_in_seconds = 5 # TODO: inference should be made within less than 100ms
    decision_result = broker.wait_for_decision(request_id, timeout=timeout_in_seconds)
    
    if decision_result is None:
        logger.error(f"{RED_COLOR}Decision timed out (timeout={timeout_in_seconds}), requestID, {request_id}{RESET_COLOR}")
        # assert False
        pod_idx = 0
    else:
        pod_idx, _ = decision_result
    # if pod_idx is None:
    #     # fallback policy (your existing logic)
    #     pod_idx = 0
    # broker.set_decision(request_id, pod_idx)
    
    # BROKER.pop(request_id)
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
        # logger.info(f"{GREEN_COLOR}Submitting request {request_id} to broker{RESET_COLOR}")
        # print(f"{GREEN_COLOR}BROKER id in submit:{RESET_COLOR}", id(self))
        pr = PendingReq(request_id=request_id, state=state, prev_reward=prev_reward)
        with self._lock:
            self._by_id[request_id] = pr
        self._queue.put(pr)
        return pr

    def get_next(self, timeout: Optional[float] = None, reset=False) -> PendingReq:
        # logger.info(f"{BLUE_COLOR}Getting next request from broker{RESET_COLOR}")
        # print(f"{BLUE_COLOR}BROKER id in get_next:{RESET_COLOR}", id(self))

        return self._queue.get(timeout=timeout)

    def set_decision(self, request_id: str, action: int, probs: Optional[Any] = None):
        # logger.info(f"{RED_COLOR}Setting decision for request {request_id} to action {action}{RESET_COLOR}")
        # print(f"{RED_COLOR}BROKER id in set_decision:{RESET_COLOR}", id(self))
        with self._lock:
            pr = self._by_id.get(request_id)
        if pr:
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
        self.broker.set_decision(self.pending.request_id, pod_idx)



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


# class 


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

    """
    def __init__(self, num_requests_per_episode: int, per_pod_dim: int = 11, request_dim: int = 3, source: GatewayRequestSource=None):
        super().__init__()

        self.num_requests_per_episode = num_requests_per_episode
        self.per_pod_dim = per_pod_dim
        self.request_dim = request_dim
        self.source = source
        global NUM_PODS

        # Just a placeholder, first dimension = 1 for initialization, dynamically set for each step
        self.observation_space = spaces.Dict({
            'pod_features': spaces.Box(-np.inf, np.inf, shape=(NUM_PODS, per_pod_dim - 1), dtype=np.float32),
            'kv_hit_ratios': spaces.Box(0.0, 1.0, shape=(NUM_PODS, 1), dtype=np.float32),
            'request_features': spaces.Box(-np.inf, np.inf, shape=(request_dim,), dtype=np.float32),
            'temporal_features': spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
        })
   
        self.action_space = spaces.Discrete(1)

        self.request_count = 0
        self._request: Optional[Request] = None
        self._first_reward: float = 0.0
        
        logger.info(f"🌍 ScalableRoutingEnvironment initialized:")
        logger.info(f"  Number of requests per episode: {num_requests_per_episode}")
        logger.info(f"  Per-pod features: {per_pod_dim}")
        logger.info(f"  Request features: {request_dim}")

    def _pull(self) -> Request:
        assert self.source is not None, "GatewayRequestSource required"
        pending: PendingReq = self.source.get_next(timeout=None)  # blocks until /infer submits
        return Request(pending=pending, broker=self.source.broker)

    def reset(self, seed=None, options=None):
        """Reset environment"""
        start_time = time.time()
        super().reset(seed=seed)

        logger.info(f"{GREEN_COLOR}Resetting environment...{RESET_COLOR}")

        self.request_count = 0
        self._request = None

        logger.info(f"{GREEN_COLOR}ScalableRoutingEnvironment reset: request_count={self.request_count}, _request={self._request}{RESET_COLOR}")

        # print(f"{GREEN}pulling request{RESET}")
        # self._request = self._pull()
        # print(f"{GREEN}request pulled{RESET}")
        # self._first_reward = float(self._request.pending.prev_reward or 0.0)
        # observation = self._request.get_obs()
        # self.update_space(observation['pod_features'].shape[0])
        # info = self._request.state # dict()

        logger.info(f"ScalableRoutingEnvironment reset took {time.time() - start_time} seconds")
        return self.make_dummy_observation(), self.make_dummy_info()
    
    # TODO: action probabilities for debugging
    # this is the entry point. I think it is gym's internal function... how can we get the action probabilities?
    def step_simple(self, action: int):
        if self._request is None:
            self._request = self._pull()
        self._request.route(action)
        self.request_count += 1
        observation = self._request.get_obs()
        self.update_space(observation['pod_features'].shape[0])
        info = self._request.state # dict()

        terminated = (self.request_count == self.num_requests_per_episode)
        if not terminated:
            next_req = self._pull()
            reward = - float(next_req.pending.prev_reward)
            self._request = next_req
        else:
            reward = 0.0
            self._request = None
            logger.info(f"Episode terminated at request count {self.request_count}")

        truncated = False
        
        return observation, reward, terminated, truncated, info

    def step(self, action: int):
        if self._request is None: # transit to the first request in the episode
            self._request = self._pull()
            logger.info(f"{GREEN_COLOR}First request of the episode pulled{RESET_COLOR}")
            observation = self._request.get_obs()
            self.update_space(observation['pod_features'].shape[0])
            reward = 0 # first request in the episode, no reward
            terminated = (self.request_count == self.num_requests_per_episode)
            
        else: 
            self._request.route(action) 
            self.request_count += 1

            terminated = (self.request_count == self.num_requests_per_episode)
            if not terminated:
                next_req = self._pull()
                observation = next_req.get_obs()
                self.update_space(observation['pod_features'].shape[0])
                reward = - float(next_req.pending.prev_reward or 0.0)
                self._request = next_req
            else:
                observation = self.make_dummy_observation() # dummy obs for the last request in the episode
                reward = 0 ## TODO: inaccurate
                logger.info(f"{GREEN_COLOR}Episode terminated at request count {self.request_count}{RESET_COLOR}")

        info = self._request.state # dict()
        truncated = False
        return observation, reward, terminated, truncated, info

    def update_space(self, num_pods: int):
        self.observation_space['pod_features'] = spaces.Box(-np.inf, np.inf, shape=(num_pods, self.per_pod_dim - 1), dtype=np.float32)
        self.observation_space['kv_hit_ratios'] = spaces.Box(0.0, 1.0, shape=(num_pods, 1), dtype=np.float32)
        self.action_space = spaces.Discrete(num_pods)

    def make_dummy_observation(self):
        return {
            'pod_features': np.zeros((NUM_PODS, self.per_pod_dim - 1), dtype=np.float32),
            'kv_hit_ratios': np.zeros((NUM_PODS, 1), dtype=np.float32),
            'request_features': np.zeros((self.request_dim,), dtype=np.float32),
            'temporal_features': np.zeros([1], dtype=np.float32)
        }
    
    def make_dummy_info(self):
        return {
            'observation': self.make_dummy_observation()
        }


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
        assert num_pods == kv_hit_ratios.shape[1], f"Number of pods in pod_features and kv_hit_ratios must match {num_pods} != {kv_hit_ratios.shape[1]}"
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
        num_pods = self.get_num_pods()
        self.batch_size = features.shape[0] // num_pods
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
            share_features_extractor=False,
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

        # self.mlp_extractor.set_num_pods(1)
        # from torchinfo import summary
        # logger.info(f"🧠 MLP Extractor Architecture: \n")
        # summary(self.mlp_extractor, input_size=(1, self.feature_dim))
        # self.mlp_extractor.set_num_pods(None)

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


    def extract_features(self, obs, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        pi_features, vf_features = super().extract_features(obs, features_extractor)
        num_pods = self.features_extractor.num_pods  # base feature extractor

        if features_extractor is not None:
            features_extractor.set_num_pods(num_pods) # mlp feature extractor
        elif self.mlp_extractor is not None:
            self.mlp_extractor.set_num_pods(num_pods)

        return pi_features, vf_features


    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        features = super(ActorCriticPolicy, self).extract_features(obs, self.pi_features_extractor)
        num_pods = self.features_extractor.num_pods  # base feature extractor
        self.mlp_extractor.set_num_pods(num_pods)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi)

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> Distribution:
        action_dist = super()._get_action_dist_from_latent(latent_pi)

        PURPLE_COLOR = "\033[95m"
        RESET_COLOR = "\033[0m" 
        ## TODO: log distribution

        if isinstance(action_dist, CategoricalDistribution):
            probs = action_dist.distribution.probs.detach().cpu().numpy()
            mean_probs = probs.mean(axis=0)

            logger.debug(f"{PURPLE_COLOR}action_probs: {mean_probs}{RESET_COLOR}")

        return action_dist


# ============================================================================
# Agent
# ============================================================================
BROKER = RequestBroker()
TRAIN_BROKER = BROKER
EVAL_BROKER = BROKER

# TRAIN_BROKER = RequestBroker()
# EVAL_BROKER = RequestBroker()

TRAIN_SOURCE = GatewayRequestSource(TRAIN_BROKER)
EVAL_SOURCE = GatewayRequestSource(EVAL_BROKER)

# ============================================================================
# Scalable RL Routing Agent
# ============================================================================

class ScalableRLRoutingAgent:

    def __init__(
        self, 
        per_pod_dim: int,
        request_dim: int,
        max_pods: int,
        num_requests_per_episode: int,      # this is horizon in RL term
        num_episodes_per_iteration: int,     # Number of episodes per iteration
        num_iterations: int,             # Number of iterations
        rl: str,
        static_num_pods: bool,
        learning_rate: float,
        hidden_dim: int,
        gamma: float,
        gae_lambda: float,
        tb_log_dir: str,
        batch_size: int,
        n_epochs: int,
        clip_range: float,
        entropy_coeff: float,
        vf_coef: float,
        max_grad_norm: float,
        last_layer_dim_pi: int,
        last_layer_dim_vf: int,
        use_prioritized_replay: bool,
        buffer_size: int,
        priority_alpha: float,
        priority_beta: float,
        lr_scheduler_type: str,
        load_tb_best: str = '/app/final_model/init_model/best_model.zip',
        ):
        """
        Args:
            per_pod_dim: Features per pod (pod_features + kv_hit_ratios)
            request_dim: Request feature dimensions
            max_pods: Maximum expected pods (for space allocation)
            num_requests_per_episode: Number of requests per episode
            num_episodes_per_iteration: Number of episodes per iteration
            num_iterations: Number of iterations
        """

        # Store episode/iteration configuration
        self.num_requests_per_episode = num_requests_per_episode
        self.num_episodes_per_iteration = num_episodes_per_iteration
        self.num_iterations = num_iterations
        self.num_requests_per_iteration = num_requests_per_episode * num_episodes_per_iteration
        
        RED_COLOR = "\033[91m"
        RESET_COLOR = "\033[0m" 
        logger.info(f"{RED_COLOR}🤖 ScalableRLRoutingAgent initializing...{RESET_COLOR}")
        logger.info(f"  Training Configuration:")
        logger.info(f"    - Requests per episode: {num_requests_per_episode}")
        logger.info(f"    - Episodes per iteration: {num_episodes_per_iteration}")
        logger.info(f"    - Number of iterations: {num_iterations}")
        logger.info(f"    - Total timesteps: {num_requests_per_episode * num_episodes_per_iteration * num_iterations}")
        logger.info(f"  Model Configuration:")
        logger.info(f"    - per_pod_dim={per_pod_dim}, request_dim={request_dim}, max_pods={max_pods}")
        logger.info(f"    - rl={rl}, static_num_pods={static_num_pods}")
        logger.info(f"    - learning_rate={learning_rate}, hidden_dim={hidden_dim}")
        logger.info(f"    - gamma={gamma}, gae_lambda={gae_lambda}")
        logger.info(f"    - num_requests_per_episode={num_requests_per_episode}, batch_size={batch_size}, n_epochs={n_epochs}")
        logger.info(f"    - clip_range={clip_range}, entropy_coeff={entropy_coeff}, vf_coef={vf_coef}")
        logger.info(f"    - max_grad_norm={max_grad_norm}")
        logger.info(f"    - last_layer_dim_pi={last_layer_dim_pi}, last_layer_dim_vf={last_layer_dim_vf}")
        logger.info(f"    - use_prioritized_replay={use_prioritized_replay}, buffer_size={buffer_size}")
        logger.info(f"    - priority_alpha={priority_alpha}, priority_beta={priority_beta}")

        self.per_pod_dim = per_pod_dim
        self.request_dim = request_dim
        self.max_pods = max_pods
        
        # Store all parameters needed for save() metadata
        self.num_requests_per_episode = num_requests_per_episode
        self.rl = rl
        self.hidden_dim = hidden_dim
        self.last_layer_dim_pi = last_layer_dim_pi
        self.last_layer_dim_vf = last_layer_dim_vf
        self.use_prioritized_replay = use_prioritized_replay
        self.buffer_size = buffer_size
        self.priority_alpha = priority_alpha
        self.priority_beta = priority_beta
        
        # Create environment
        self.static_num_pods = static_num_pods
        self.train_env = self.make_env(num_requests_per_episode, TRAIN_SOURCE)
        self.eval_env = self.make_env(num_requests_per_episode, EVAL_SOURCE)
        self.env = self.train_env  # Alias for compatibility with save/load methods
        self.tb_log_dir = os.path.abspath(tb_log_dir)
        self.setup_model(rl, per_pod_dim, request_dim, learning_rate, hidden_dim, gamma, gae_lambda, batch_size, n_epochs, clip_range, entropy_coeff, vf_coef, max_grad_norm, last_layer_dim_pi, last_layer_dim_vf, lr_scheduler_type)

        # CYAN_COLOR = "\033[96m"
        # RESET_COLOR = "\033[0m" 
        # logger.info(f"{CYAN_COLOR}Tensorboard log directory: {os.path.abspath(self.tb_log_dir)}{RESET_COLOR}")
        # if hasattr(self.model, "logger"):
        #    logger.info(f"Logger type: {type(self.model.logger)}")
        #    logger.info(f"Logger outputs: {self.model.logger.output_formats}") 
        
        ## NOTE (gangmuk): I commented out since we are not using prioritized experience replay for now
        # # === Prioritized Experience Replay ===
        # if use_prioritized_replay:
        #     self.experience_buffer = PrioritizedReplayBuffer(
        #         maxlen=buffer_size,
        #         alpha=priority_alpha,
        #         beta=priority_beta
        #     )
        
        # === Training statistics ===
        self.total_steps = 0
        self.total_episodes = 0

        if load_tb_best is not None:
            self.load_tb_best(load_tb_best)
    
        logger.info(f"ScalableRLRoutingAgent initialization complete")


    def make_env(self, num_requests_per_episode: int, source: GatewayRequestSource):

        GREEN_COLOR = "\033[92m"
        RESET_COLOR = "\033[0m" 
        logger.info(f"{GREEN_COLOR}Making environment...{RESET_COLOR}")

        env = ScalableRoutingEnvironment(
            num_requests_per_episode=num_requests_per_episode,
            per_pod_dim=self.per_pod_dim,
            request_dim=self.request_dim,
            source = source,
        )

        env = Monitor(env)
        env = EpisodeCounterWrapper(env) # track episode count
        if self.static_num_pods:
            env = DummyVecEnv([lambda: env])
            env = VecNormalize(env, norm_obs=True, norm_reward=True)

        return env
        

    def setup_model(self, \
        rl: str, \
        per_pod_dim: int, \
        request_dim: int, \
        learning_rate: float, \
        hidden_dim: int, \
        gamma: float, \
        gae_lambda: float, \
        batch_size: int, \
        n_epochs: int, \
        clip_range: float, \
        entropy_coeff: float, \
        vf_coef: float, \
        max_grad_norm: float, \
        last_layer_dim_pi: int, \
        last_layer_dim_vf: int, \
        lr_scheduler_type: str):
        

        if rl == 'PG':
            pass
        elif rl == 'PPO':
            # === Learning Rate Configuration ===
            # Option 1: Constant LR (good for short training runs)
            # Option 2: Linear decay with minimum bound (standard PPO)
            min_lr = 0.0003  # Minimum LR floor (prevents learning from stopping)
            
            if lr_scheduler_type == 'linear':
                # Linear LR scheduler: decays over training with minimum bound
                initial_lr = learning_rate
                # Apply minimum LR bound to prevent decay to near-zero
                final_lr = max(learning_rate * 0.1, min_lr)
                lr_schedule = get_linear_fn(initial_lr, final_lr, end_fraction=1.0)
                logger.info(f"📉 Using linear LR schedule: {initial_lr} → {final_lr} (min_lr={min_lr})")
            else:
                # Constant LR (simpler, good for exploration)
                lr_schedule = learning_rate
                logger.info(f"📊 Using constant LR: {learning_rate}")
            
            # === Create PPO model with our scalable policy ===
            self.model = PPO(
                ActorCriticRoutingPolicy,
                self.train_env,
                learning_rate=lr_schedule,  # Can be constant or schedule
                n_steps=self.num_requests_per_iteration, # model update every iteration
                # n_steps=self.num_requests_per_episode, # model update every episode
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,                    # Discount factor
                gae_lambda=gae_lambda,          # GAE lambda (short horizon)
                clip_range=clip_range,
                ent_coef=entropy_coeff,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                policy_kwargs={
                    'per_pod_dim': per_pod_dim,
                    'request_dim': request_dim,
                    'hidden_dim': hidden_dim,
                    'last_layer_dim_pi': last_layer_dim_pi,
                    'last_layer_dim_vf': last_layer_dim_vf,
                },
                verbose=1,
                tensorboard_log=self.tb_log_dir,
            )
            CYAN_COLOR = "\033[96m"
            RESET_COLOR = "\033[0m" 
            logger.info(f"{CYAN_COLOR}Tensorboard log directory: {os.path.abspath(self.tb_log_dir)}{RESET_COLOR}")
            if hasattr(self.model, "logger"):
                logger.info(f"Logger type: {type(self.model.logger)}")
                logger.info(f"Logger outputs: {self.model.logger.output_formats}")             
        else:
            raise ValueError(f"{rl} not supported")

    

    def train(self, save_path: str, eval_freq: int, n_eval_episodes: int):
        # Calculate total timesteps from episode/iteration configuration
        total_timesteps = ((self.num_requests_per_episode + 1) * 
                          self.num_episodes_per_iteration * 
                          self.num_iterations)
        
        logger.info(f"Starting training for {total_timesteps} timesteps "
                   f"({self.num_iterations} iterations × "
                   f"{self.num_episodes_per_iteration} episodes × "
                   f"{self.num_requests_per_episode} requests)")
        
        if self.static_num_pods:
            eval_callback = EvalCallback(self.eval_env, best_model_save_path=f"{self.tb_log_dir}/best_model", \
                log_path=f"{self.tb_log_dir}/results", eval_freq=eval_freq, n_eval_episodes=n_eval_episodes, deterministic=True, render=False)
            
            # Continue training from loaded model's step count (don't reset to 0)
            # This ensures continuous x-axis in TensorBoard across restarts
            self.model.learn(
                total_timesteps=total_timesteps, 
                progress_bar=True, 
                callback=eval_callback,
                reset_num_timesteps=False  # ← KEY: Continue from loaded model's step count
            )
            

        else:
            trainer = Trainer(self.model, self.train_env, log_dir=self.tb_log_dir, eval_env=self.eval_env)
            trainer.train(total_timesteps, eval_freq=eval_freq, n_eval_episodes=n_eval_episodes)
        self.total_steps = total_timesteps
        self.save(save_path)
    
    
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
                'hidden_dim': self.hidden_dim,
                'last_layer_dim_pi': self.last_layer_dim_pi,
                'last_layer_dim_vf': self.last_layer_dim_vf,
                'use_prioritized_replay': self.use_prioritized_replay,
                'buffer_size': self.buffer_size,
                'priority_alpha': self.priority_alpha,
                'priority_beta': self.priority_beta,
                'num_requests_per_episode': self.num_requests_per_episode,
                'rl': self.rl,
            },
            
            # === Training Configuration ===
            'training_config': {
                'num_requests_per_episode': self.num_requests_per_episode,
                'num_episodes_per_iteration': self.num_episodes_per_iteration,
                'num_iterations': self.num_iterations,
                'total_timesteps': self.num_requests_per_episode * self.num_episodes_per_iteration * self.num_iterations,
            },
            
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
                'training_config': metadata['training_config'],
                'checkpoint_info': metadata['checkpoint_info'],
            }
            with open(json_metadata_path, 'w') as f:
                json.dump(json_metadata, f, indent=2)
            logger.info(f"Saved human-readable metadata to {json_metadata_path}")
        except Exception as e:
            logger.warning(f"Could not save JSON metadata: {e}")
        
        logger.info(f"Model checkpoint saved to {path}")
        # logger.info(f"   Total steps: {self.total_steps}, Episodes: {self.total_episodes}")
        # logger.info(f"   Buffer size: {len(self.experience_buffer)}/{self.experience_buffer.buffer.maxlen}")

    def load_tb_best(self, path: str):
        try:
            # === Load PPO model ===
            logger.info(f"🔄 Loading model from {path} ...")
            if ".zip" in path:
                path = path.replace(".zip", "")
                logger.info(f"Removed .zip from path since PPO.load() expects without .zip extension in name. Now path: {path}.")
            self.model = PPO.load(path, env=self.train_env, device="auto")
            logger.info(f"✅ Model successfully loaded from {path}")

            return True
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False


    
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

    def __init__(self, model, env, log_dir="./tb_logs", eval_env=None):
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

            # === 1.1 Get action probability distribution ===
            # obs_tensor = self.model.policy.obs_to_tensor(obs)
            obs_tensor = {key: torch.as_tensor(_obs, device=self.model.device).unsqueeze(0) for (key, _obs) in obs.items()}
            dist = self.model.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.detach().cpu().numpy()[0]

            self.model.logger.record("policy/action_probs", probs)
            logger.info(f"probs: {probs}")

            # Also log entropy and max-prob for diagnostics
            entropy = dist.distribution.entropy().mean().item()
            max_prob = float(probs.max())
            self.model.logger.record("policy/entropy", entropy)
            self.model.logger.record("policy/max_action_prob", max_prob)

            # === 1. Predict ===
            action, _ = self.model.predict(obs, deterministic=False)
            logger.info(f"action: {action}")
            obs, reward, done, truncated, info = self.env.step(action)

            for i in range(len(probs)):
                self.model.logger.record(f"policy/action_taken_{i}", 1.0 if i == action else 0.0)

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

def main():
    logger.info("🧪 Testing ScalableRLRoutingAgent...")

    global NUM_PODS

    NUM_REQUESTS_PER_EPISODE = 10
    NUM_EPISODES_PER_ITERATION_TRAIN = 1
    NUM_EPISODES_PER_ITERATION_EVAL = 1
    NUM_ITERATIONS_TRAIN = 2
    EVAL_FREQ = 2
    TOTAL_STEPS_EVAL = (NUM_REQUESTS_PER_EPISODE + 1) * NUM_EPISODES_PER_ITERATION_TRAIN * EVAL_FREQ
    
    # Create agent
    agent = ScalableRLRoutingAgent(
        per_pod_dim=11,
        request_dim=3,
        max_pods=10,
        # Training configuration (simple and clear!)
        num_requests_per_episode=NUM_REQUESTS_PER_EPISODE,        # 1 episode = 5 requests (for testing)
        num_episodes_per_iteration=NUM_EPISODES_PER_ITERATION_TRAIN,      # 1 iteration = 8 episodes
        num_iterations=NUM_ITERATIONS_TRAIN,              # Total = 5 iterations (5*8*5 = 200 timesteps)
        rl='PPO',
        static_num_pods=True,
        learning_rate=0.0005,
        hidden_dim=64,
        gamma=1.0,
        gae_lambda=0.95,
        tb_log_dir='./tb_logs',
        batch_size=256,
        n_epochs=4,
        clip_range=0.2,
        entropy_coeff=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        last_layer_dim_pi=1,
        last_layer_dim_vf=1,
        use_prioritized_replay=False,
        buffer_size=1000,
        priority_alpha=0.6,
        priority_beta=0.4,
        lr_scheduler_type='linear',
        load_tb_best="./tb_logs/best_model/best_model.zip",
    )
    
    # Simulate routing workflow
    train_thread = threading.Thread(
        target=agent.train,
        kwargs={"save_path": "scalable_rl_agent.pth", "eval_freq": TOTAL_STEPS_EVAL, "n_eval_episodes": NUM_EPISODES_PER_ITERATION_EVAL},
        daemon=True,   # dies with the main process
    )
    train_thread.start()
    logger.info("🏋️ Training RL agent...")
    import random

    for i in range(0, NUM_ITERATIONS_TRAIN):
        print(f"{BLUE_COLOR}Training iteration {i} of {NUM_ITERATIONS_TRAIN}{RESET_COLOR}")

        for j in range(0, NUM_EPISODES_PER_ITERATION_TRAIN):

            print(f"{BLUE_COLOR}Training episode {j} of {NUM_EPISODES_PER_ITERATION_TRAIN}{RESET_COLOR}")
            
            for k in range(0, NUM_REQUESTS_PER_EPISODE):
                # NUM_PODS = random.randint(4, 10) # comment this to use fixed NUM_PODS
                request_id = f"req_train_{i}_{j}_{k}"
                prev_reward = random.random()
                pod_features = np.random.randn(NUM_PODS, 10).astype(np.float32)
                kv_hit_ratios = np.random.rand(NUM_PODS, 1).astype(np.float32)
                request_features = np.random.randn(3).astype(np.float32)
                temporal_features = np.array([1], dtype=np.float32)
                
                pod_idx = infer(request_id, prev_reward, pod_features, kv_hit_ratios, request_features, temporal_features, TRAIN_BROKER, 2)
                
                # Simulate completion (in real system, this happens asynchronously)
                time.sleep(0.01)

                # confidence = action_probs[pod_idx] if action_probs is not None else 0.0
                confidence = 0.0
                logger.info(f"Step {i}_{j}_{k}: action={pod_idx}, prev_reward={prev_reward:.2f}, "
                f"confidence={confidence:.3f}, num_pods={NUM_PODS}")
        
        time.sleep(3)

        if EVAL_FREQ > 0 and (i + 1) % EVAL_FREQ == 0:
            for j in range(0, NUM_EPISODES_PER_ITERATION_EVAL):
                print(f"{BLUE_COLOR}Evaluating episode {j} of {NUM_EPISODES_PER_ITERATION_EVAL}{RESET_COLOR}")
                for k in range(0, NUM_REQUESTS_PER_EPISODE):
                    request_id = f"req_eval_{i}_{j}_{k}"
                    prev_reward = random.random()
                    pod_features = np.random.randn(NUM_PODS, 10).astype(np.float32)
                    kv_hit_ratios = np.random.rand(NUM_PODS, 1).astype(np.float32)
                    request_features = np.random.randn(3).astype(np.float32)
                    temporal_features = np.array([1], dtype=np.float32)

                    pod_idx = infer(request_id, prev_reward, pod_features, kv_hit_ratios, request_features, temporal_features, EVAL_BROKER, 2)

                    # Simulate completion (in real system, this happens asynchronously)
                    time.sleep(0.01)

                    # confidence = action_probs[pod_idx] if action_probs is not None else 0.0
                    confidence = 0.0
                    logger.info(f"Step {i}_{j}_{k}: action={pod_idx}, prev_reward={prev_reward:.2f}, "
                    f"confidence={confidence:.3f}, num_pods={NUM_PODS}")
            
            ##### END
            
    # Check metrics
    # metrics = agent.get_metrics()
    # logger.info(f"📊 Final metrics: {metrics}")
    logger.info("✅ Test completed successfully!")




if __name__ == "__main__":
    main()