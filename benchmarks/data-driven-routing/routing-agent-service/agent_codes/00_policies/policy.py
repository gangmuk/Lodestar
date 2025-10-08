from stable_baselines3.common.policies import ActorCriticPolicy
from nets import PodFeatExtractor

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
            features_extractor_class=PodFeatExtractor,
            features_extractor_kwargs={
                'per_pod_dim': per_pod_dim,
                'request_dim': request_dim,
                'hidden_dim': hidden_dim
            },
            **kwargs
        )