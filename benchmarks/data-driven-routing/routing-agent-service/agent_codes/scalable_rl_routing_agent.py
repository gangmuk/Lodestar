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
import numpy as np

from logger import logger

from agents import ScalableRLRoutingAgent

# WANYU 10/07/2025
# 1. Restructured RL modules 
# 2. TODO: Rewrite RoutingAgent
#   2.1 Seperate deployment and training
#   2.2 Change control flow to use RL's step() function.
#       2.2.1 See 00_envs/wrappers.py for example of triggering each step by specifying a time interval
#       2.2.2 Actively read states/obs and set reward


# WANYU 10/09/2025
# 1. Move PrioritizedReplayBuffer, EpisodeTracker, ScalableRLRoutingAgent to 00_agents
# 2. ScalableRLRoutingAgent:
#    2.1 By default, PrioritizedReplayBuffer is not used
#    2.2 Add train() function for offline RL training


# WANYU 10/11/2025
# 1. Rewrite RL agent
#    1.1 Problems of orginal implementation: Score NNs are defined in feature extractor, although it doesn't execute them, but this complicates the logic.
#    1.2 New implementation:
#        1.2.1 Feature extractor only concatenates cluster statistics and request features
#        1.2.2 Add a new NN class PodScorer, which replace sb3's mlp_extractor

# WANYU 10/12/2025
# 1. Add rl parameter to control the RL algorithm, support PG and PPO
# 
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


## TODO: implement offline RL training
# Check interface
def train_scalable_rl_agent(rl_agent, hyperparameters):
    rl_agent.train(hyperparameters)

def infer_scalable_rl_agent(tensor_data, request_id, sorted_all_pod_ids, processed_df,
                            rl_agent, hyperparameters, agent_lock=None):
    """
    Inference workflow for scalable RL agent.
    
    Compatible with existing routing_agent_service.py interface.
    
    Key changes from old version:
    1. Creates pending experience (not completed yet)
    2. Completion happens asynchronously in on_request_complete callback
    3. Proper next_obs and done flags for TD learning
    
    Args:
        tensor_data: Dict with 'pod_features', 'kv_hit_ratios', 'request_features'
        request_id: Request identifier
        sorted_all_pod_ids: List of pod IDs
        processed_df: DataFrame with request data
        rl_agent: ScalableRLRoutingAgent instance
        hyperparameters: RL hyperparameters
        agent_lock: Optional lock (not needed for prediction in new design)
    
    Returns:
        (agent, result_dict, overhead_summary)
    """
    import preprocess
    
    overhead_summary = {} # logging time for extracting features, predicting, and creating pending experience
    infer_start = time.time()
    
    # Extract tensors
    extract_start = time.time()
    pod_features_t = tensor_data['pod_features']
    kv_hit_t = tensor_data['kv_hit_ratios']
    req_features_t = tensor_data['request_features']
    
    pod_features_np = pod_features_t.cpu().numpy()[0]  # [num_pods, 10]
    kv_hit_np = kv_hit_t.cpu().numpy()[0]              # [num_pods, 1]
    req_features_np = req_features_t.cpu().numpy()[0]  # [3]
    
    overhead_summary['extract_tensors'] = time.time() - extract_start
    
    # Predict action
    num_pods = len(sorted_all_pod_ids)
    assert num_pods <= len(action_probs)

    predict_start = time.time()
    action_idx, action_probs = rl_agent.predict(
        pod_features_np, kv_hit_np, req_features_np, deterministic=False
    )
    overhead_summary['predict'] = time.time() - predict_start
    
    # Create pending experience (will be completed asynchronously)
    rl_agent.create_pending_experience(
        request_id, pod_features_np, kv_hit_np, req_features_np,
        action_idx, action_probs
    )
    
    # Build result
    result_start = time.time()
    pod_prob_map = {
        sorted_all_pod_ids[i]: float(action_probs[i]) 
        for i in range(min(num_pods, len(action_probs)))
    }
    
    result = {
        'selected_pod_index': int(action_idx),
        'pod_probabilities': pod_prob_map,
        'confidence': float(np.max(action_probs)), # XXX: double check if this is correct
        'explore_mask': 0,
        'predicted_latencies': {pod_id: -1 for pod_id in sorted_all_pod_ids},
        'chosen_pod_predicted_latency': -1,
    }
    overhead_summary['build_result'] = time.time() - result_start
    
    overhead_summary['total'] = time.time() - infer_start
    
    logger.info(f"🎯 Scalable RL inference: action={action_idx}, "
               f"confidence={result['confidence']:.3f}, "
               f"num_pods={num_pods}")
    
    return rl_agent, result, overhead_summary


# Callback for async completion (to be called by routing service)
def on_request_complete_callback(rl_agent, request_id, current_cluster_state, 
                                 ttft, tpot, hyperparameters):
    """
    Callback to complete experience when request finishes.
    
    This should be called by routing_agent_service.py when it receives
    completion notification.
    
    Args:
        rl_agent: ScalableRLRoutingAgent instance
        request_id: Request identifier
        current_cluster_state: Current pod features, kv ratios (after completion)
        ttft, tpot: Latency metrics
        hyperparameters: For reward calculation
    """
    import preprocess
    
    # Extract current state
    pod_features, kv_hit_ratios, request_features = current_cluster_state
    
    # Compute reward
    ttft_slo = hyperparameters['TTFT_SLO']
    avg_tpot_slo = hyperparameters['AVG_TPOT_SLO']
    ttft_weight = hyperparameters['TTFT_REWARD_WEIGHT']
    reward_fn = hyperparameters.get('REWARD_FUNCTION', 'linear_simple')
    
    if reward_fn == 'linear_simple':
        reward_res = preprocess.calculate_rewards_simple(
            np.array([ttft]), np.array([tpot]),
            ttft_slo, avg_tpot_slo, ttft_weight
        )
    elif reward_fn == 'linear_simple_extended':
        reward_res = preprocess.calculate_rewards_simple_extended(
            np.array([ttft]), np.array([tpot]),
            ttft_slo, avg_tpot_slo, ttft_weight
        )
    # ... (other reward functions)
    else:
        logger.warning(f"Unknown reward function {reward_fn}, using linear_simple")
        reward_res = preprocess.calculate_rewards_simple(
            np.array([ttft]), np.array([tpot]),
            ttft_slo, avg_tpot_slo, ttft_weight
        )
    
    reward = float(reward_res['combined_rewards'][0])
    
    # Complete the experience
    rl_agent.complete_experience(
        request_id, pod_features, kv_hit_ratios, request_features, reward
    )
    
    logger.debug(f"✅ Completed experience for {request_id}: reward={reward:.3f}")




# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    logger.info("🧪 Testing ScalableRLRoutingAgent...")
    
    # Create agent
    agent = create_scalable_rl_agent(
        per_pod_dim=11,
        request_dim=3,
        max_pods=10,
        learning_rate=3e-4,
        reward_decay_factor=0.95,
        gae_lambda=0.95,
        episode_duration=1.0
    )
    
    # Simulate routing workflow
    for i in range(20):
        # Generate random state
        num_pods = np.random.randint(4, 9)  # Variable pods!
        pod_features = np.random.randn(num_pods, 10).astype(np.float32)
        kv_hit_ratios = np.random.rand(num_pods, 1).astype(np.float32)
        request_features = np.random.randn(3).astype(np.float32)
        
        # Predict
        request_id = f"req_{i}"
        action, probs = agent.predict(pod_features, kv_hit_ratios, request_features)
        
        # Create pending experience
        agent.create_pending_experience(
            request_id, pod_features, kv_hit_ratios, request_features,
            action, probs
        )
        
        # Simulate completion (in real system, this happens asynchronously)
        time.sleep(0.01)
        
        # Get next state (after request completes)
        next_pod_features = np.random.randn(num_pods, 10).astype(np.float32)
        next_kv_hit = np.random.rand(num_pods, 1).astype(np.float32)
        next_request = np.random.randn(3).astype(np.float32)
        
        # Random reward
        reward = np.random.randn() * 2.0
        
        # Complete experience
        agent.complete_experience(
            request_id, next_pod_features, next_kv_hit, next_request, reward
        )
        
        logger.info(f"Step {i}: action={action}, reward={reward:.2f}, "
                   f"num_pods={num_pods}, done={agent.episode_tracker.check_episode_end()}")
    
    # Check metrics
    metrics = agent.get_metrics()
    logger.info(f"📊 Final metrics: {metrics}")
    logger.info("✅ Test completed successfully!")