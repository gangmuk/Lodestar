import gymnasium as gym
from gymnasium import spaces
import numpy as np
from logger import logger

from envs import Request

class ScalableRoutingEnvironment(gym.Env):
    """
    Gymnasium environment with DICT observation space (supports variable pods).
    
    Unlike the old version, this doesn't flatten everything into a single vector.
    Instead, it keeps structured observations that the policy can handle flexibly.

    Wanyu 10/14/2025
    1. Change max_pods to num_pods
    2. Use Request class to manage requests (interface to Gateway)
    """
    def __init__(self, num_pods: int, per_pod_dim: int = 11, request_dim: int = 3, max_pods: int = 100):
        super().__init__()
        
        self.num_pods = num_pods
        self.per_pod_dim = per_pod_dim
        self.request_dim = request_dim
        self.max_pods = max_pods

        # Use num_pods instead of max_pods
        self.observation_space = spaces.Dict({
            'pod_features': spaces.Box(
                -np.inf, np.inf,
                shape=(num_pods, per_pod_dim - 1),
                dtype=np.float32
            ),
            'kv_hit_ratios': spaces.Box(
                0.0, 1.0,
                shape=(num_pods, 1),
                dtype=np.float32
            ),
            'request_features': spaces.Box(
                -np.inf, np.inf,
                shape=(request_dim,),
                dtype=np.float32
            ),
            'temporal_features': spaces.Box(
                -np.inf, np.inf,
                shape=(0,),
                dtype=np.float32
            )
        })
        
        self.action_space = spaces.Discrete(num_pods)
        
        # Current state
        self.current_obs = None
        self.current_req = Request()
        
        logger.info(f"🌍 ScalableRoutingEnvironment initialized:")
        logger.info(f"  Max pods: {max_pods} (can handle less at runtime)")
        logger.info(f"  Per-pod features: {per_pod_dim}")
        logger.info(f"  Request features: {request_dim}")

    
    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)
        
        # Return dummy observation
        # dummy_obs = {
        #     'pod_features': np.zeros((self.max_pods, self.per_pod_dim - 1), dtype=np.float32),
        #     'kv_hit_ratios': np.zeros((self.max_pods, 1), dtype=np.float32),
        #     'request_features': np.zeros((self.request_dim,), dtype=np.float32),
        #     'temporal_features': np.array([], dtype=np.float32)
        # }
        # Use num_pods instead of max_pods
        dummy_obs = {
            'pod_features': np.zeros((self.num_pods, self.per_pod_dim - 1), dtype=np.float32),
            'kv_hit_ratios': np.zeros((self.num_pods, 1), dtype=np.float32),
            'request_features': np.zeros((self.request_dim,), dtype=np.float32),
            'temporal_features': np.array([], dtype=np.float32)
        }

        self.current_obs = dummy_obs
        return dummy_obs, {} # Double check
    
    def step(self, action: int):
        self.current_req.route(action)
        next_req = self.current_req.wait_for_request()  
        obs = next_req.get_obs()
        reward = next_req.get_reward()
        terminated = False # TODO: double check
        truncated = False  # Managed by episode tracker
        
        return obs, reward, terminated, truncated, {}
    
    # def set_observation(self, obs):
    #     """External interface to set current observation"""
    #     self.current_obs = obs
    #     self.current_num_pods = obs['pod_features'].shape[0]