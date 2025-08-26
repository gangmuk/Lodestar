#!/usr/bin/env python3

# random_forest.py

import os
import sys
import json
import numpy as np
import torch
import pickle
import time
import matplotlib.pyplot as plt
from datetime import datetime
import glob
from logger import logger
import traceback
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import joblib

# Global variables to match the expected API
device = "cpu"  # Random Forest doesn't use GPU
training_results_dir = "training_results"
final_model_path = "final_model"

class RandomForestAgent:
    """
    Random Forest implementation with the same API as SimplifiedContextualBandit
    """
    def __init__(self, state_dim=None, action_dim=7, **kwargs):
        self.action_dim = action_dim
        self.state_dim = state_dim
        
        # Random Forest parameters
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )
        
        # Track if model is trained
        self.is_trained = False
        
        # Storage for training history (for compatibility)
        self.loss_history = []
        self.reward_history = []
        self.entropy_history = []
        
        logger.info(f"Random Forest model initialized:")
        logger.info(f"  n_estimators: 100")
        logger.info(f"  max_depth: 10")
        logger.info(f"  Action dimension: {action_dim}")
        
    def _flatten_features(self, pod_features, kv_hit_ratios, request_features):
        """
        Flatten all input features into a single feature vector
        Same as the neural network preprocessing
        """
        batch_size = pod_features.shape[0]
        
        # Flatten pod features and KV ratios
        pod_flat = pod_features.view(batch_size, -1).cpu().numpy()
        kv_flat = kv_hit_ratios.view(batch_size, -1).cpu().numpy()
        req_flat = request_features.view(batch_size, -1).cpu().numpy()
        
        # Concatenate all features
        combined_features = np.concatenate([pod_flat, kv_flat, req_flat], axis=1)
        
        return combined_features
        
    def remember(self, pod_features, kv_hit_ratios, request_features, action, reward):
        """Store training data (not used in Random Forest, but kept for API compatibility)"""
        pass
        
    def choose_action(self, pod_features, kv_hit_ratios, request_features, evaluate=False):
        """Select an action using the trained Random Forest"""
        if not self.is_trained:
            # Random action if not trained
            batch_size = pod_features.shape[0]
            random_actions = torch.randint(0, self.action_dim, (batch_size,))
            return random_actions
            
        # Flatten features
        features = self._flatten_features(pod_features, kv_hit_ratios, request_features)
        
        # Predict action
        predicted_actions = self.model.predict(features)
        
        return torch.tensor(predicted_actions, dtype=torch.long)
    
    def get_action_probabilities(self, pod_features, kv_hit_ratios, request_features):
        """Get action probabilities for confidence calculation"""
        if not self.is_trained:
            # Uniform probabilities if not trained
            batch_size = pod_features.shape[0]
            uniform_probs = torch.ones(batch_size, self.action_dim) / self.action_dim
            return uniform_probs
            
        # Flatten features
        features = self._flatten_features(pod_features, kv_hit_ratios, request_features)
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(features)
        
        return torch.tensor(probabilities, dtype=torch.float32)
    
    def learn(self):
        """Random Forest doesn't need iterative learning, return dummy metrics"""
        return {
            'loss': 0.0,
            'reward': 0.0,
            'entropy': 0.0
        }
    
    def clear_memory(self):
        """Clear memory (not needed for Random Forest)"""
        pass
    
    def save(self, directory):
        """Save the Random Forest model"""
        os.makedirs(directory, exist_ok=True)
        
        # Save the sklearn model
        model_path = os.path.join(directory, 'random_forest_model.pkl')
        joblib.dump(self.model, model_path)
        
        # Save metadata
        metadata = {
            'action_dim': self.action_dim,
            'state_dim': self.state_dim,
            'is_trained': self.is_trained,
            'model_type': 'random_forest'
        }
        
        metadata_path = os.path.join(directory, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        # Copy to final model path
        os.makedirs(final_model_path, exist_ok=True)
        os.system(f"cp {model_path} {final_model_path}/")
        os.system(f"cp {metadata_path} {final_model_path}/")
        
        logger.info(f"Saved Random Forest model to {directory}")
    
    def load(self, directory):
        """Load the Random Forest model"""
        model_path = os.path.join(directory, 'random_forest_model.pkl')
        metadata_path = os.path.join(directory, 'metadata.json')
        
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            self.is_trained = True
            logger.info(f"Loaded Random Forest model from {directory}")
            
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.action_dim = metadata.get('action_dim', self.action_dim)
                self.state_dim = metadata.get('state_dim', self.state_dim)
                self.is_trained = metadata.get('is_trained', False)


def load_all_encoded_data(encoded_data_dir):
    """Load and combine data from all batch directories (same as original)"""
    logger.info(f"Loading data from {encoded_data_dir}")
    
    # Find all batch subdirectories
    batch_dirs = glob.glob(os.path.join(encoded_data_dir, "batch_*"))
    logger.info(f"Found {len(batch_dirs)} batch directories")
    
    combined_data = None
    total_samples = 0
    
    # Process each batch directory
    for batch_dir in batch_dirs:
        # Look for tensor_dataset.pt in the batch directory or its train subdirectory
        tensor_path = os.path.join(batch_dir, "tensor_dataset.pt")
        if not os.path.exists(tensor_path):
            train_dir = os.path.join(batch_dir, "train")
            if os.path.exists(train_dir):
                tensor_path = os.path.join(train_dir, "tensor_dataset.pt")
                
        if not os.path.exists(tensor_path):
            logger.warning(f"No tensor_dataset.pt found in {batch_dir}")
            continue
            
        try:
            # Load tensor data
            batch_data = torch.load(tensor_path, map_location='cpu')
            
            # If this is the first valid batch, use it as the base
            if combined_data is None:
                combined_data = batch_data
                total_samples = batch_data['rewards'].size(0)
                logger.info(f"First batch has {total_samples} samples")
            else:
                # Combine the data by concatenating along the batch dimension
                for key in combined_data:
                    if key in batch_data:
                        combined_data[key] = torch.cat([combined_data[key], batch_data[key]], dim=0)
                        
                batch_samples = batch_data['rewards'].size(0)
                total_samples += batch_samples
                logger.debug(f"Added {batch_samples} samples from {os.path.basename(batch_dir)}")
                    
        except Exception as e:
            logger.error(f"Error loading data from {batch_dir}: {e}")
            continue
    
    if combined_data is None:
        logger.error("No valid data could be loaded from any batch")
        raise ValueError("No valid data found in the encoded_data directory")
        
    logger.info(f"Successfully combined data from multiple batches, total samples: {total_samples}")
    
    return combined_data


def train(encoded_data_dir):
    """Main training function for Random Forest"""
    
    # Set output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(training_results_dir, exist_ok=True)
    output_dir = os.path.join(training_results_dir, f"random_forest_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Starting Random Forest training...")
    
    # Load and combine data from all batches
    combined_data = load_all_encoded_data(encoded_data_dir)
    
    # Extract features and labels
    pod_features = combined_data['pod_features_with_staleness']
    kv_hit_ratios = combined_data['kv_hit_ratios']
    request_features = combined_data['request_features']
    actions = combined_data['actions']
    rewards = combined_data['rewards']
    
    logger.info(f"Dataset loaded: {len(actions)} samples")
    logger.info(f"Features shape: pod_features={pod_features.shape}, kv_ratios={kv_hit_ratios.shape}, request={request_features.shape}")
    
    # Determine dimensions
    action_dim = len(torch.unique(actions))
    state_dim = {
        'pod_features': pod_features.shape[2],
        'kv_hit_ratios': kv_hit_ratios.shape[2],
        'request_features': request_features.shape[1],
        'num_pods': pod_features.shape[1]
    }
    
    # Create Random Forest agent
    agent = RandomForestAgent(state_dim=state_dim, action_dim=action_dim)
    
    # Flatten features for sklearn
    features = agent._flatten_features(pod_features, kv_hit_ratios, request_features)
    labels = actions.cpu().numpy()
    
    logger.info(f"Flattened features shape: {features.shape}")
    logger.info(f"Labels shape: {labels.shape}")
    logger.info(f"Number of unique actions: {len(np.unique(labels))}")
    
    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    # Train the model
    training_start_time = time.time()
    logger.info("Training Random Forest...")
    
    agent.model.fit(X_train, y_train)
    agent.is_trained = True
    
    training_time = time.time() - training_start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate the model
    train_predictions = agent.model.predict(X_train)
    test_predictions = agent.model.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)
    
    logger.info(f"Training accuracy: {train_accuracy:.4f} ({train_accuracy*100:.1f}%)")
    logger.info(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")
    
    # Random baseline
    random_baseline = 1.0 / action_dim
    logger.info(f"Random baseline: {random_baseline:.4f} ({random_baseline*100:.1f}%)")
    
    # Feature importance
    feature_importance = agent.model.feature_importances_
    logger.info(f"Top 10 most important features:")
    top_indices = np.argsort(feature_importance)[-10:][::-1]
    for i, idx in enumerate(top_indices):
        logger.info(f"  {i+1}. Feature {idx}: {feature_importance[idx]:.4f}")
    
    # Save configuration and results
    config = {
        'model_type': 'random_forest',
        'n_estimators': 100,
        'max_depth': 10,
        'training_time': training_time,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'random_baseline': random_baseline,
        'dataset_size': len(actions),
        'feature_dim': features.shape[1],
        'action_dim': action_dim
    }
    
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Save the model
    agent.save(output_dir)
    
    # Create simple training plot
    try:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.bar(['Train', 'Test', 'Random'], [train_accuracy, test_accuracy, random_baseline])
        plt.title('Model Performance')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        
        plt.subplot(1, 3, 2)
        action_counts = np.bincount(labels, minlength=action_dim)
        plt.bar(range(action_dim), action_counts)
        plt.title('Action Distribution')
        plt.xlabel('Pod ID')
        plt.ylabel('Count')
        
        plt.subplot(1, 3, 3)
        plt.plot(top_indices[:5], feature_importance[top_indices[:5]], 'o-')
        plt.title('Top 5 Feature Importances')
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_results.png'), dpi=150, bbox_inches='tight')
        
        # Copy to final model path
        if os.path.exists(final_model_path):
            os.system(f"cp {os.path.join(output_dir, 'training_results.png')} {final_model_path}/")
        
        plt.close()
        logger.info(f"Saved training plots to {output_dir}")
        
    except Exception as e:
        logger.warning(f"Could not create plots: {e}")
    
    # Print final summary
    logger.info("=" * 60)
    logger.info("RANDOM FOREST TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Dataset size: {len(actions)} samples")
    logger.info(f"Training time: {training_time:.2f} seconds")
    logger.info(f"Test accuracy: {test_accuracy:.3f} ({test_accuracy*100:.1f}%)")
    logger.info(f"Random baseline: {random_baseline:.3f} ({random_baseline*100:.1f}%)")
    
    if test_accuracy > random_baseline * 1.5:
        logger.info("✅ Model is learning significantly!")
    elif test_accuracy > random_baseline * 1.1:
        logger.info("⚠️  Model shows modest learning")
    else:
        logger.info("❌ Model performance close to random")
    
    logger.info("=" * 60)
    
    return {
        'agent': agent,
        'model_dir': output_dir,
        'output_dir': output_dir,
        'config': config,
        'test_accuracy': test_accuracy,
        'train_accuracy': train_accuracy
    }


# Global cache for agent instance (for inference)
_cached_agent = None
_cached_agent_config = None


def infer_from_tensor(tensor_data, exploration_enabled=False, exploration_rate=0.1, model_updated=False):
    """
    Inference function for Random Forest model
    """
    global final_model_path, _cached_agent, _cached_agent_config
    
    infer_start_time = time.time()
    
    # Extract data from tensor dataset
    try:
        pod_features = tensor_data['pod_features_with_staleness']
        kv_hit_ratios = tensor_data['kv_hit_ratios']
        request_features = tensor_data['request_features']
    except KeyError as e:
        logger.error(f"Missing key in tensor data: {e}")
        raise ValueError(f"Missing key in tensor data: {e}")
    
    # Ensure data is in batch format (add batch dimension if needed)
    if len(pod_features.shape) == 2:
        pod_features = pod_features.unsqueeze(0)
    if len(kv_hit_ratios.shape) == 2:
        kv_hit_ratios = kv_hit_ratios.unsqueeze(0)
    if len(request_features.shape) == 1:
        request_features = request_features.unsqueeze(0)

    # Cache agent instance
    current_config = {
        'pod_features': pod_features.shape[2],
        'kv_hit_ratios': kv_hit_ratios.shape[2], 
        'request_features': request_features.shape[1],
        'num_pods': pod_features.shape[1],
        'final_model_path': final_model_path
    }
    
    # Check if we can reuse cached agent
    agent_cache_hit = False
    if (_cached_agent is not None and 
        _cached_agent_config is not None and
        _cached_agent_config == current_config):
        agent = _cached_agent
        agent_cache_hit = True
        logger.debug("Agent cache hit - reusing cached Random Forest agent")
    else:
        # Create new agent
        logger.info("Creating new Random Forest agent for inference")
        
        state_dim = {
            'pod_features': current_config['pod_features'],
            'kv_hit_ratios': current_config['kv_hit_ratios'],
            'request_features': current_config['request_features'],
            'num_pods': current_config['num_pods']
        }
        
        action_dim = current_config['num_pods']
        
        agent = RandomForestAgent(state_dim=state_dim, action_dim=action_dim)
        
        _cached_agent = agent
        _cached_agent_config = current_config.copy()

    # Load model weights if needed
    if not agent_cache_hit or model_updated:
        agent.load(final_model_path)
        logger.info("Loaded Random Forest model from disk")

    # Run inference
    if not agent.is_trained:
        logger.warning("Random Forest model not trained, using random action")
        selected_action = np.random.randint(0, agent.action_dim)
        confidence = 1.0 / agent.action_dim
    else:
        # Get action probabilities for confidence
        action_probs = agent.get_action_probabilities(pod_features, kv_hit_ratios, request_features)
        
        if exploration_enabled and np.random.random() < exploration_rate:
            # Random exploration
            selected_action = np.random.randint(0, agent.action_dim)
            confidence = action_probs[0, selected_action].item()
        else:
            # Use model prediction
            predicted_actions = agent.choose_action(pod_features, kv_hit_ratios, request_features, evaluate=True)
            selected_action = predicted_actions[0].item()
            confidence = action_probs[0, selected_action].item()

    total_inference_time = time.time() - infer_start_time
    
    # Return inference results
    results = {
        'selected_pod_index': selected_action,
        'confidence': confidence,
        'pod_probabilities': action_probs[0].numpy().tolist() if agent.is_trained else [1.0/agent.action_dim] * agent.action_dim,
        'final_model_path': final_model_path,
        'exploration_enabled': exploration_enabled,
        'model_type': 'random_forest'
    }
    
    timing_info = {
        'total_inference_time_ms': total_inference_time * 1000,
        'agent_cache_hit': agent_cache_hit,
        'model_updated': model_updated
    }
    
    return results, timing_info


if __name__ == "__main__":
    # Example usage
    if len(sys.argv) > 1:
        encoded_data_dir = sys.argv[1]
        logger.info(f"Starting Random Forest training with data from: {encoded_data_dir}")
        results = train(encoded_data_dir)
        logger.info("Random Forest training completed successfully!")
    else:
        logger.info("Usage: python random_forest.py <encoded_data_dir>")
        logger.info("Example: python random_forest.py encoded_data/")