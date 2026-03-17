#!/usr/bin/env python3

"""
SB3-based RL Contextual Bandit Integration
Drop-in replacement for simpler_contextual_bandit using proper SB3 infrastructure
"""

import os
import time
import numpy as np
import torch
import pickle
from typing import Dict, List, Tuple, Any
from logger import logger
from rl_routing_agent_sb3 import create_rl_routing_agent_sb3, RLRoutingAgentSB3

# Global agent instance
global_rl_agent = None
global_state_dim = None
global_action_dim = None

def create_rl_agent(state_dim: Dict[str, int], action_dim: int, hyperparameters: Dict[str, Any]):
    """Create and initialize the global SB3-based RL agent"""
    global global_rl_agent, global_state_dim, global_action_dim
    
    logger.info(f"Creating SB3 RL agent with state_dim={state_dim}, action_dim={action_dim}")
    
    # Extract RL-specific hyperparameters and map to SB3 parameters
    rl_hyperparams = {
        'learning_rate': hyperparameters.get('learning_rate', 3e-4),
        'reward_decay_factor': hyperparameters.get('reward_decay_factor', 0.95),  # maps to gamma
        'hidden_dim': hyperparameters.get('hidden_dim', 64),
        'use_custom_reward': hyperparameters.get('use_custom_reward', True),
        
        # SB3-specific parameters
        'n_steps': hyperparameters.get('n_steps', 2048),
        'batch_size': hyperparameters.get('batch_size', 64),
        'n_epochs': hyperparameters.get('n_epochs', 10),
        'gae_lambda': hyperparameters.get('gae_lambda', 0.95),
        'clip_range': hyperparameters.get('clip_range', 0.2),
        'entropy_coeff': hyperparameters.get('entropy_coeff', 0.01),
        'vf_coef': hyperparameters.get('vf_coef', 0.5),
        'max_grad_norm': hyperparameters.get('max_grad_norm', 0.5),
    }
    
    global_rl_agent = create_rl_routing_agent_sb3(
        state_dim=state_dim,
        action_dim=action_dim,
        **rl_hyperparams
    )
    
    global_state_dim = state_dim
    global_action_dim = action_dim
    
    logger.info("SB3 RL agent created successfully")
    return global_rl_agent


def infer_from_tensor(tensor_data, request_id, model_updated, hyperparameters, final_model_dir):
    """
    Compatible interface with existing routing service using SB3
    """
    global global_rl_agent
    
    infer_start_time = time.time()
    infer_overhead_summary = {}
    
    logger.info(f"SB3 RL inference for request {request_id}")
    
    try:
        # Extract data from tensor format (same as before)
        pod_features = tensor_data['pod_features_with_staleness'].squeeze(0).cpu().numpy()
        kv_hit_ratios = tensor_data['kv_hit_ratios'].squeeze(0).cpu().numpy()
        request_features = tensor_data['request_features'].squeeze(0).cpu().numpy()
        
        num_pods = pod_features.shape[0]
        
        logger.debug(f"Tensor data shapes: pod_features={pod_features.shape}, "
                    f"kv_hit_ratios={kv_hit_ratios.shape}, request_features={request_features.shape}")
        
        # Initialize agent if needed
        if global_rl_agent is None:
            logger.info("Initializing SB3 RL agent from saved model or creating new one")
            
            # Determine state dimensions from data
            state_dim = {
                'pod_features': pod_features.shape[1],
                'kv_hit_ratios': kv_hit_ratios.shape[1],
                'request_features': request_features.shape[0]
            }
            
            # Load or create agent
            agent_path = os.path.join(final_model_dir, 'rl_agent_sb3.zip')
            if os.path.exists(agent_path):
                logger.info(f"Loading SB3 RL agent from {agent_path}")
                create_rl_agent(state_dim, num_pods, hyperparameters)
                global_rl_agent.load(agent_path)
            else:
                logger.info("Creating new SB3 RL agent")
                create_rl_agent(state_dim, num_pods, hyperparameters)
        
        # Get action prediction using SB3
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
            'model_type': 'rl_contextual_bandit_sb3',
            'agent_metrics': global_rl_agent.get_metrics()
        }
        
        logger.info(f"SB3 RL inference result for {request_id}: chosen_pod={action}, "
                   f"confidence={result['confidence']:.3f}")
        
        infer_overhead_summary['total'] = time.time() - infer_start_time
        
        return result, infer_overhead_summary
        
    except Exception as e:
        logger.error(f"Error in SB3 RL inference: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to random selection
        random_action = np.random.randint(0, num_pods if 'num_pods' in locals() else 2)
        uniform_probs = np.ones(num_pods if 'num_pods' in locals() else 2) / (num_pods if 'num_pods' in locals() else 2)
        
        result = {
            'chosen_pod_index': int(random_action),
            'action_probs': uniform_probs.tolist(),
            'confidence': float(uniform_probs[random_action]),
            'model_type': 'rl_contextual_bandit_sb3_fallback',
            'error': str(e)
        }
        
        infer_overhead_summary['total'] = time.time() - infer_start_time
        return result, infer_overhead_summary


def store_experience(tensor_data, action, reward_info, request_id=None):
    """
    Store experience for online learning using SB3
    """
    global global_rl_agent
    
    if global_rl_agent is None:
        logger.warning("No SB3 RL agent available for experience storage")
        return
    
    try:
        # Extract data from tensor format
        pod_features = tensor_data['pod_features_with_staleness'].squeeze(0).cpu().numpy()
        kv_hit_ratios = tensor_data['kv_hit_ratios'].squeeze(0).cpu().numpy()
        request_features = tensor_data['request_features'].squeeze(0).cpu().numpy()
        
        # Calculate point reward from reward_info
        point_reward = calculate_point_reward(reward_info)
        
        # Store experience using SB3 agent
        global_rl_agent.remember_experience(
            pod_features, kv_hit_ratios, request_features,
            action, point_reward
        )
        
        logger.debug(f"Stored SB3 experience: action={action}, point_reward={point_reward:.4f}")
        
        # Trigger online learning if enough experiences accumulated
        if len(global_rl_agent.experience_buffer) >= global_rl_agent.hyperparameters.get('online_update_threshold', 50):
            logger.info("Triggering SB3 online learning update")
            global_rl_agent.update_online()
        
    except Exception as e:
        logger.error(f"Error storing SB3 experience: {e}")


def calculate_point_reward(reward_info):
    """
    Calculate point reward from feedback information
    Enhanced version with more sophisticated reward calculation
    """
    try:
        # Enhanced reward calculation
        latency = reward_info.get('latency', 1000)  # ms
        slo_met = reward_info.get('slo_met', True)
        target_latency = reward_info.get('target_latency', 500)  # ms
        
        # Multi-objective reward
        if slo_met:
            # Latency component: reward decreases exponentially with latency
            latency_reward = np.exp(-(latency / target_latency))
            
            # SLO compliance bonus
            slo_bonus = 0.5
            
            reward = latency_reward + slo_bonus
        else:
            # Severe penalty for SLO violation
            penalty = -1.0
            
            # Additional penalty based on how much we exceeded SLO
            overshoot = max(0, latency - target_latency) / target_latency
            overshoot_penalty = -0.5 * overshoot
            
            reward = penalty + overshoot_penalty
            
        # Clip reward to reasonable range
        reward = np.clip(reward, -2.0, 2.0)
        
        return float(reward)
        
    except Exception as e:
        logger.error(f"Error calculating reward: {e}")
        return 0.0


def train(encoded_data_dir, final_model_dir, hyperparameters, is_online_learning=True):
    """
    Training function using SB3 infrastructure
    """
    global global_rl_agent
    
    logger.info(f"SB3 RL training setup: encoded_data_dir={encoded_data_dir}, "
               f"final_model_dir={final_model_dir}, online_learning={is_online_learning}")
    
    try:
        # Create output directory
        os.makedirs(final_model_dir, exist_ok=True)
        
        if is_online_learning:
            logger.info("SB3 RL agent configured for online learning")
            
            # Save hyperparameters for later use
            hyperparams_path = os.path.join(final_model_dir, 'rl_sb3_hyperparameters.pkl')
            with open(hyperparams_path, 'wb') as f:
                pickle.dump(hyperparameters, f)
                
            logger.info(f"SB3 RL hyperparameters saved to {hyperparams_path}")
            
        else:
            # For offline training with SB3
            logger.info("Offline SB3 RL training - would need to load historical data")
            # TODO: Implement offline training using SB3 if needed
            # This would involve creating a replay environment and training on historical episodes
            
        return global_rl_agent
        
    except Exception as e:
        logger.error(f"Error in SB3 RL training setup: {e}")
        raise


def save_agent(final_model_dir):
    """Save the SB3 RL agent"""
    global global_rl_agent
    
    if global_rl_agent is None:
        logger.warning("No SB3 RL agent to save")
        return
        
    try:
        agent_path = os.path.join(final_model_dir, 'rl_agent_sb3.zip')
        global_rl_agent.save(agent_path)
        logger.info(f"SB3 RL agent saved to {agent_path}")
        
    except Exception as e:
        logger.error(f"Error saving SB3 RL agent: {e}")


def load_agent(final_model_dir, state_dim=None, action_dim=None, hyperparameters=None):
    """Load the SB3 RL agent"""
    global global_rl_agent
    
    try:
        agent_path = os.path.join(final_model_dir, 'rl_agent_sb3.zip')
        
        if not os.path.exists(agent_path):
            logger.warning(f"No saved SB3 RL agent found at {agent_path}")
            return None
            
        # Load hyperparameters if not provided
        if hyperparameters is None:
            hyperparams_path = os.path.join(final_model_dir, 'rl_sb3_hyperparameters.pkl')
            if os.path.exists(hyperparams_path):
                with open(hyperparams_path, 'rb') as f:
                    hyperparameters = pickle.load(f)
            else:
                logger.warning("No SB3 hyperparameters found, using defaults")
                hyperparameters = {}
        
        # Create agent if needed
        if global_rl_agent is None and state_dim and action_dim:
            create_rl_agent(state_dim, action_dim, hyperparameters)
            
        if global_rl_agent:
            global_rl_agent.load(agent_path)
            logger.info(f"SB3 RL agent loaded from {agent_path}")
            
        return global_rl_agent
        
    except Exception as e:
        logger.error(f"Error loading SB3 RL agent: {e}")
        return None


# Compatibility wrapper for existing training code
class RLSimplifiedContextualBanditSB3:
    """
    SB3-based wrapper to maintain compatibility with existing training code
    """
    def __init__(self, state_dim, action_dim, hyperparameters, final_model_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hyperparameters = hyperparameters
        self.final_model_dir = final_model_dir
        
        # Create SB3 RL agent
        self.agent = create_rl_agent(state_dim, action_dim, hyperparameters)
        
        logger.info("RLSimplifiedContextualBanditSB3 initialized")
        
    def learn(self):
        """Compatibility method - SB3 handles learning automatically"""
        # Trigger online learning if we have experiences
        if hasattr(self.agent, 'experience_buffer') and len(self.agent.experience_buffer) > 0:
            self.agent.update_online()
        
        metrics = self.agent.get_metrics()
        return {
            'loss': 0.0,  # SB3 handles this internally
            'reward': 0.0,  # Would need to track separately if needed
            'entropy': 0.0  # SB3 handles this internally
        }
    
    def save(self, final_model_dir):
        """Save SB3 agent"""
        save_agent(final_model_dir)
        
    def remember(self, pod_features, kv_hit_ratios, request_features, action, reward):
        """Store experience using SB3"""
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
    # Test the SB3 RL contextual bandit interface
    logger.info("Testing SB3 RL contextual bandit interface...")
    
    # Create mock tensor data
    state_dim = {'pod_features': 10, 'kv_hit_ratios': 1, 'request_features': 3}
    action_dim = 4
    hyperparameters = {
        'learning_rate': 3e-4,
        'reward_decay_factor': 0.95,
        'n_steps': 64,
        'batch_size': 32,
        'use_custom_reward': True
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
    reward_info = {'latency': 300, 'slo_met': True, 'target_latency': 500}
    store_experience(tensor_data, result['chosen_pod_index'], reward_info, "test_request")
    
    logger.info("SB3 RL contextual bandit test completed successfully")