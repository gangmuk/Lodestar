import gymnasium as gym
from gymnasium import spaces
import numpy as np
from logger import logger

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

    ## TODO: Implement reward function
    def compute_reward(self, state):
        """Compute reward for the action"""
        return 0.0

    def make_obs(self, state):
        """Make observation from state"""
        return self.current_obs
    
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

        ## TODO: could managed externally but easier to set here through read_state() API
        ## read_state() should read all relevant states, obs is subset of state

        ## TODO: state = self.xxx.read_state()
        state = None          
        reward = self.compute_reward(state)
        obs = self.make_obs(state)
        terminated = False
        truncated = False  # Managed by episode tracker
        
        return obs, reward, terminated, truncated, {}
    
    def set_observation(self, obs):
        """External interface to set current observation"""
        self.current_obs = obs
        self.current_num_pods = obs['pod_features'].shape[0]