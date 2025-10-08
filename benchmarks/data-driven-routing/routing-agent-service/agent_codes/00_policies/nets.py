import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from logger import logger


class PodFeatExtractor(BaseFeaturesExtractor):
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
        
        logger.info(f"🏗️  ScalableRoutingPolicyNetwork initialized:")
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