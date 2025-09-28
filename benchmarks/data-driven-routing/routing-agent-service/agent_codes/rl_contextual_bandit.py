#!/usr/bin/env python3

"""
RL Contextual Bandit Implementation
This module provides a compatible interface with the existing routing_agent_service.py
while implementing reinforcement learning instead of standard contextual bandit.
"""

import os
import time
import numpy as np
import torch
import pickle
from typing import Dict, List, Tuple, Any
from logger import logger
from rl_routing_agent import create_rl_routing_agent, RLRoutingAgent, RoutingEnvironment

# Global agent instance
global_rl_agent = None
global_state_dim = None
global_action_dim = None

def create_rl_agent(state_dim: Dict[str, int], action_dim: int, hyperparameters: Dict[str, Any]):
    """Create and initialize the global RL agent"""
    global global_rl_agent, global_state_dim, global_action_dim
    
    logger.info(f"Creating RL agent with state_dim={state_dim}, action_dim={action_dim}")
    
    # Extract RL-specific hyperparameters
    rl_hyperparams = {
        'learning_rate': hyperparameters.get('learning_rate', 3e-4),
        'reward_decay_factor': hyperparameters.get('reward_decay_factor', 0.95),
        'baseline_decay': hyperparameters.get('baseline_decay', 0.95),
        'mini_batch_size': hyperparameters.get('mini_batch_size', 30),
        'episode_length_minutes': hyperparameters.get('episode_length_minutes', 3),
        'update_frequency_seconds': hyperparameters.get('update_frequency_seconds', 60),
        'entropy_coeff': hyperparameters.get('entropy_coeff', 0.01),
        'hidden_dim': hyperparameters.get('hidden_dim', 64),
    }
    
    global_rl_agent = create_rl_routing_agent(
        state_dim=state_dim,
        action_dim=action_dim,
        **rl_hyperparams
    )
    
    global_state_dim = state_dim
    global_action_dim = action_dim
    
    logger.info("RL agent created successfully")
    return global_rl_agent


def infer_from_tensor(tensor_data, request_id, model_updated, hyperparameters, final_model_dir):
    """
    Compatible interface with existing routing service
    This function replaces simpler_contextual_bandit.infer_from_tensor()
    """
    global global_rl_agent
    
    infer_start_time = time.time()
    infer_overhead_summary = {}
    
    logger.info(f"RL inference for request {request_id}")
    
    try:
        # Extract data from tensor format (same as contextual bandit)
        pod_features = tensor_data['pod_features_with_staleness'].squeeze(0).cpu().numpy()  # [num_pods, feature_dim]
        kv_hit_ratios = tensor_data['kv_hit_ratios'].squeeze(0).cpu().numpy()  # [num_pods, 1]
        request_features = tensor_data['request_features'].squeeze(0).cpu().numpy()  # [feature_dim]
        
        num_pods = pod_features.shape[0]
        
        logger.debug(f"Tensor data shapes: pod_features={pod_features.shape}, "
                    f"kv_hit_ratios={kv_hit_ratios.shape}, request_features={request_features.shape}")
        
        # Initialize agent if needed
        if global_rl_agent is None:
            logger.info("Initializing RL agent from saved model or creating new one")
            
            # Determine state dimensions from data
            state_dim = {
                'pod_features': pod_features.shape[1],
                'kv_hit_ratios': kv_hit_ratios.shape[1],
                'request_features': request_features.shape[0]
            }
            
            # Load or create agent
            agent_path = os.path.join(final_model_dir, 'rl_agent.pth')
            if os.path.exists(agent_path):
                logger.info(f"Loading RL agent from {agent_path}")
                create_rl_agent(state_dim, num_pods, hyperparameters)
                global_rl_agent.load(agent_path)
            else:
                logger.info("Creating new RL agent")
                create_rl_agent(state_dim, num_pods, hyperparameters)
        
        # Get action prediction
        predict_start = time.time()
        action, action_probs = global_rl_agent.predict(
            pod_features, kv_hit_ratios, request_features, 
            deterministic=False  # Use stochastic policy for exploration
        )
        infer_overhead_summary['predict'] = time.time() - predict_start
        
        # Prepare result in expected format
        result = {
            'chosen_pod_index': int(action),
            'action_probs': action_probs.tolist(),
            'confidence': float(action_probs[action]),
            'model_type': 'rl_contextual_bandit',
            'agent_metrics': global_rl_agent.get_metrics()
        }
        
        logger.info(f"RL inference result for {request_id}: chosen_pod={action}, "
                   f"confidence={result['confidence']:.3f}")
        
        infer_overhead_summary['total'] = time.time() - infer_start_time
        
        return result, infer_overhead_summary
        
    except Exception as e:
        logger.error(f"Error in RL inference: {e}")
        # Fallback to random selection
        random_action = np.random.randint(0, num_pods if 'num_pods' in locals() else 2)
        uniform_probs = np.ones(num_pods if 'num_pods' in locals() else 2) / (num_pods if 'num_pods' in locals() else 2)
        
        result = {
            'chosen_pod_index': int(random_action),
            'action_probs': uniform_probs.tolist(),
            'confidence': float(uniform_probs[random_action]),
            'model_type': 'rl_contextual_bandit_fallback',
            'error': str(e)
        }
        
        infer_overhead_summary['total'] = time.time() - infer_start_time
        return result, infer_overhead_summary


def store_experience(tensor_data, action, reward_info, request_id=None):
    """
    Store experience for online learning
    This should be called after getting feedback about the action taken
    """
    global global_rl_agent
    
    if global_rl_agent is None:
        logger.warning("No RL agent available for experience storage")
        return
    
    try:
        # Extract data from tensor format
        pod_features = tensor_data['pod_features_with_staleness'].squeeze(0).cpu().numpy()
        kv_hit_ratios = tensor_data['kv_hit_ratios'].squeeze(0).cpu().numpy()
        request_features = tensor_data['request_features'].squeeze(0).cpu().numpy()
        
        # Calculate point reward from reward_info
        # reward_info should contain latency metrics, SLO violations, etc.
        point_reward = calculate_point_reward(reward_info)
        
        # Store experience for online learning
        global_rl_agent.remember_experience(
            pod_features, kv_hit_ratios, request_features,
            action, point_reward
        )
        
        logger.debug(f"Stored experience: action={action}, point_reward={point_reward:.4f}")
        
    except Exception as e:
        logger.error(f"Error storing experience: {e}")


def calculate_point_reward(reward_info):
    """
    Calculate point reward from feedback information
    This should be customized based on your reward function
    """
    try:
        # Example reward calculation - customize this!
        latency = reward_info.get('latency', 1000)  # ms
        slo_met = reward_info.get('slo_met', True)
        
        # Simple reward: higher reward for lower latency and meeting SLO
        if slo_met:
            # Reward decreases with latency
            reward = max(0.1, 1.0 - (latency / 1000.0))  
        else:
            # Penalty for SLO violation
            reward = -0.5
            
        return float(reward)
        
    except Exception as e:
        logger.error(f"Error calculating reward: {e}")
        return 0.0


def train(encoded_data_dir, final_model_dir, hyperparameters, is_online_learning=True):
    """
    Training function compatible with existing training pipeline
    For RL, this mainly sets up the agent for online learning
    """
    global global_rl_agent
    
    logger.info(f"RL training setup: encoded_data_dir={encoded_data_dir}, "
               f"final_model_dir={final_model_dir}, online_learning={is_online_learning}")
    
    try:
        # Create output directory
        os.makedirs(final_model_dir, exist_ok=True)
        
        if is_online_learning:
            logger.info("RL agent configured for online learning")
            
            # The agent will be created during first inference
            # Save hyperparameters for later use
            hyperparams_path = os.path.join(final_model_dir, 'rl_hyperparameters.pkl')
            with open(hyperparams_path, 'wb') as f:
                pickle.dump(hyperparameters, f)
                
            logger.info(f"RL hyperparameters saved to {hyperparams_path}")
            
        else:
            # For offline training, we would load historical data
            # and train the agent before deployment
            logger.info("Offline RL training not yet implemented")
            # TODO: Implement offline training if needed
            
        return global_rl_agent
        
    except Exception as e:
        logger.error(f"Error in RL training setup: {e}")
        raise


def save_agent(final_model_dir):
    """Save the RL agent state"""
    global global_rl_agent
    
    if global_rl_agent is None:
        logger.warning("No RL agent to save")
        return
        
    try:
        agent_path = os.path.join(final_model_dir, 'rl_agent.pth')
        global_rl_agent.save(agent_path)
        logger.info(f"RL agent saved to {agent_path}")
        
    except Exception as e:
        logger.error(f"Error saving RL agent: {e}")


def load_agent(final_model_dir, state_dim=None, action_dim=None, hyperparameters=None):
    """Load the RL agent state"""
    global global_rl_agent
    
    try:
        agent_path = os.path.join(final_model_dir, 'rl_agent.pth')
        
        if not os.path.exists(agent_path):
            logger.warning(f"No saved RL agent found at {agent_path}")
            return None
            
        # Load hyperparameters if not provided
        if hyperparameters is None:
            hyperparams_path = os.path.join(final_model_dir, 'rl_hyperparameters.pkl')
            if os.path.exists(hyperparams_path):
                with open(hyperparams_path, 'rb') as f:
                    hyperparameters = pickle.load(f)
            else:
                logger.warning("No hyperparameters found, using defaults")
                hyperparameters = {}
        
        # Create agent if needed
        if global_rl_agent is None and state_dim and action_dim:
            create_rl_agent(state_dim, action_dim, hyperparameters)
            
        if global_rl_agent:
            global_rl_agent.load(agent_path)
            logger.info(f"RL agent loaded from {agent_path}")
            
        return global_rl_agent
        
    except Exception as e:
        logger.error(f"Error loading RL agent: {e}")
        return None


# Compatibility functions to mimic simpler_contextual_bandit interface
class RLSimplifiedContextualBandit:
    """
    Wrapper class to maintain compatibility with existing training code
    """
    def __init__(self, state_dim, action_dim, hyperparameters, final_model_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hyperparameters = hyperparameters
        self.final_model_dir = final_model_dir
        
        # Create RL agent
        self.agent = create_rl_agent(state_dim, action_dim, hyperparameters)
        
    def learn(self):
        """Compatibility method - actual learning happens online"""
        return {
            'loss': 0.0,
            'reward': np.mean(self.agent.recent_rewards) if self.agent.recent_rewards else 0.0,
            'entropy': 0.0
        }
    
    def save(self, final_model_dir):
        """Save agent"""
        save_agent(final_model_dir)
        
    def remember(self, pod_features, kv_hit_ratios, request_features, action, reward):
        """Store experience"""
        # Convert torch tensors to numpy if needed
        if hasattr(pod_features, 'cpu'):
            pod_features = pod_features.cpu().numpy()
        if hasattr(kv_hit_ratios, 'cpu'):
            kv_hit_ratios = kv_hit_ratios.cpu().numpy()
        if hasattr(request_features, 'cpu'):
            request_features = request_features.cpu().numpy()
            
        self.agent.remember_experience(pod_features, kv_hit_ratios, request_features, action, reward)


# Example usage and testing
if __name__ == "__main__":
    # Test the RL contextual bandit interface
    logger.info("Testing RL contextual bandit interface...")
    
    # Create mock tensor data
    state_dim = {'pod_features': 10, 'kv_hit_ratios': 1, 'request_features': 3}
    action_dim = 4
    hyperparameters = {
        'learning_rate': 3e-4,
        'reward_decay_factor': 0.95,
        'mini_batch_size': 20
    }
    
    # Create agent
    agent = create_rl_agent(state_dim, action_dim, hyperparameters)
    
    # Mock tensor data
    tensor_data = {
        'pod_features_with_staleness': torch.randn(1, action_dim, state_dim['pod_features']),
        'kv_hit_ratios': torch.rand(1, action_dim, state_dim['kv_hit_ratios']),
        'request_features': torch.randn(1, state_dim['request_features'])
    }
    
    # Test inference
    result, overhead = infer_from_tensor(tensor_data, "test_request", False, hyperparameters, "/tmp")
    logger.info(f"Test inference result: {result}")
    
    # Test experience storage
    reward_info = {'latency': 500, 'slo_met': True}
    store_experience(tensor_data, result['chosen_pod_index'], reward_info, "test_request")
    
    logger.info("RL contextual bandit test completed successfully")