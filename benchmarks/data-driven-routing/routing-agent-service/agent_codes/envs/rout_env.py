import gymnasium as gym
from gymnasium import spaces
import numpy as np

from logger import logger
from typing import Optional

from .request import Request
from broker import PendingReq
from .request_source_gateway import GatewayRequestSource

class ScalableRoutingEnvironment(gym.Env):
    """
    Gymnasium environment with DICT observation space (supports variable pods).
    
    Unlike the old version, this doesn't flatten everything into a single vector.
    Instead, it keeps structured observations that the policy can handle flexibly.

    Wanyu 10/14/2025
    1. Change max_pods to num_pods
    2. Use Request class to manage requests (interface to Gateway)
    """
    def __init__(self, num_pods: int, num_requests: int, per_pod_dim: int = 11, request_dim: int = 3, source: GatewayRequestSource=None):
        super().__init__()
        
        self.num_requests = num_requests
        self.per_pod_dim = per_pod_dim
        self.request_dim = request_dim
        self.source = source

        # None shape to support variable number of pods, will break if we call space.sample()
        self.observation_space = spaces.Dict({
            'pod_features': spaces.Box(-np.inf, np.inf, shape=(None, per_pod_dim - 1), dtype=np.float32),
            'kv_hit_ratios': spaces.Box(0.0, 1.0, shape=(None, 1), dtype=np.float32),
            'request_features': spaces.Box(-np.inf, np.inf, shape=(request_dim,), dtype=np.float32),
            'temporal_features': spaces.Box(-np.inf, np.inf, shape=(0,), dtype=np.float32)
        })
   
        self.action_space = spaces.Box(0, np.inf, shape=(1,), dtype=np.int32)

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

        logger.info(f"Resetting environment...")

        self.request_count = 0
        self._request = self._pull()
        self._first_reward = float(self._request.pending.prev_reward or 0.0)
        observation = self._request.get_obs()
        info = self._request.state # dict()

        logger.info(f"ScalableRoutingEnvironment reset with request {self._request.pending.request_id}, \
            prev_reward {self._first_reward}")
        
        return observation, info
    

    def step(self, action: int):
        self._request.route(action)

        next_req = self._pull()
        observation = next_req.get_obs()
        reward = float(next_req.pending.prev_reward) if self.request_count > 0 else self._first_reward
        info = next_req.state # dict()
        self.request_count += 1

        terminated = (self.request_count == self.num_requests)
        truncated = False
        
        self._request = next_req
        
        return observation, reward, terminated, truncated, info