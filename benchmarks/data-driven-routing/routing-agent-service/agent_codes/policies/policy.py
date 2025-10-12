import gymnasium as gym

from typing import Callable
from stable_baselines3.common.policies import ActorCriticPolicy

from nets import PodFeatExtractor, PodScorer


class ActorCriticRoutingPolicy(ActorCriticPolicy):
    """
    Custom Actor-Critic policy using our scalable architecture.
    
    Integrates with SB3's PPO while maintaining pod-independent design.
    """
    def __init__(
        self,
        num_pods: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space, 
        lr_schedule: Callable[[float], float],
        per_pod_dim: int = 11, 
        request_dim: int = 3, 
        hidden_dim: int = 64, 
        last_layer_dim_pi: int = 1,
        last_layer_dim_vf: int = 1,
        *args, 
        **kwargs):
        
        self.num_pods = num_pods
        self.per_pod_dim = per_pod_dim
        self.request_dim = request_dim
        # self.hidden_dim = hidden_dim
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
                # 'hidden_dim': hidden_dim
            },
            *args,
            **kwargs
        )

    def _build_mlp_extractor(self) -> None:
        """
        forward: https://github.com/DLR-RM/stable-baselines3/blob/d487f2d2355a6cf81ea26a0bbbdf1a727ca2a886/stable_baselines3/common/policies.py#L636
        """
        
        self.mlp_extractor = PodScorer(self.num_pods, self.feature_dim, self.hidden_dim, self.last_layer_dim_pi, self.last_layer_dim_vf)