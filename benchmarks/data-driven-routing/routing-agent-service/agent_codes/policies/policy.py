import gymnasium as gym
import torch
import torch.nn as nn

from typing import Callable, Union
from logger import logger
from torchinfo import summary
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.type_aliases import Schedule

from stable_baselines3.common.policies import ActorCriticPolicy

from .nets import PodFeatExtractor, PodScorer


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
