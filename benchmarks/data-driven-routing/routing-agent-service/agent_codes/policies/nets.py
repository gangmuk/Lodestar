import torch
import torch.nn as nn
import gymnasium as gym

from typing import Tuple
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from logger import logger


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
        
        super().__init__(observation_space, features_dim)
        
        
        logger.info(f"🏗️  ScalableRoutingPolicyNetwork initialized:")
        # logger.info(f"  Per-pod input: {scorer_input_size} dims (11 pod + 3 req + 44 cluster)")
        logger.info(f"  Features output: {features_dim} dims (fixed size for critic)")
        # logger.info(f"  Hidden dim: {hidden_dim}")
        

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

## TODO: Support fixed sized value network, this needs to change PodFeatExtractor forward 
# to remove concat of pod features.
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
        num_pods: int, 
        feature_dim: int, 
        hidden_dim: int = 64, 
        last_layer_dim_pi: int = 1, 
        last_layer_dim_vf: int = 0, 
        ):

        super(PodScorer, self).__init__()
        
        self.num_pods = num_pods

        self.pod_scorer_pi = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),       # 58 → 64
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),         # 64 → 32
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, last_layer_dim_pi)                   # 32 → last_layer_dim_pi (score)
        )

        self.pod_scorer_vf = None
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

        self.action_mask = None ## TODO: current implementation doesn't support action_mask

    
    def forward(self, features: torch.Tensore) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pod_scorer_vf is None:
            return self.forward_policy(features)
        else:
            return self.forward_policy(features), self.forward_value(features)


    def forward_policy(self, features: torch.Tensor) -> torch.Tensor:
        policy_pod_scores = self.pod_scorer(features)  # [batch*num_pods, 1]
        policy_pod_scores = policy_pod_scores.view(-1, self.num_pods * self.last_layer_dim_pi)  # [batch, num_pods*last_layer_dim_pi]
        
        # === STEP 7: Apply action masking (unhealthy pod filtering) ===
        if self.action_mask is not None:
            # Set invalid pod scores to -inf (zero probability after softmax)
            policy_pod_scores = policy_pod_scores.masked_fill(self.action_mask == 0, float('-inf'))
        
        return policy_pod_scores

    def forward_value(self, features: torch.Tensor) -> torch.Tensor:
        if self.pod_scorer_vf is None:
            raise ValueError("pod_scorer_vf is not initialized")

        value_pod_scores = self.pod_scorer_vf(features)  # [batch*num_pods, 1]
        value_pod_scores = value_pod_scores.view(-1, self.num_pods * self.last_layer_dim_vf)  # [batch, num_pods*last_layer_dim_vf]
        
        # === STEP 7: Apply action masking (unhealthy pod filtering) ===
        if self.action_mask is not None:
            # Set invalid pod scores to -inf (zero probability after softmax)
            value_pod_scores = value_pod_scores.masked_fill(self.action_mask == 0, float('-inf'))
        
        return value_pod_scores

