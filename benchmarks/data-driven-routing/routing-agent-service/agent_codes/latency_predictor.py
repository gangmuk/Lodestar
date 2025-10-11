#!/usr/bin/env python3

"""
Latency Predictor Model for LLM Request Routing

This module implements a neural network that predicts latency for each pod
and selects the pod with the lowest predicted latency for routing decisions.

Unlike the contextual bandit approach which learns from rewards, this model
directly predicts latency metrics (TTFT, avg_TPOT, e2e_latency) and chooses
the pod with the best predicted performance.

Key features:
- Neural network for latency prediction
- Configurable latency metric (ttft, avg_tpot, e2e_latency)
- MSE loss for training
- Direct latency-based routing decisions
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import pickle
from logger import logger
import utils
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Use same device detection as contextual bandit
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


class LatencyPredictionNetwork(nn.Module):
    """
    Neural network that predicts latency for a given pod-request pair.
    
    Architecture similar to FixedPolicyNetwork but outputs a single latency value
    instead of action probabilities.
    """
    
    def __init__(self, pod_feature_size, kv_feature_size, request_feature_size, hidden_dim, weight_initialization='xavier'):
        super(LatencyPredictionNetwork, self).__init__()
        
        self.pod_feature_size = pod_feature_size
        self.kv_feature_size = kv_feature_size  
        self.request_feature_size = request_feature_size
        self.hidden_dim = hidden_dim
        
        # Combined input size: pod features + kv features + request features
        combined_input_size = pod_feature_size + kv_feature_size + request_feature_size
        
        logger.info(f"LatencyPredictionNetwork architecture:")
        logger.info(f"  Pod features: {pod_feature_size}")
        logger.info(f"  KV features: {kv_feature_size}")
        logger.info(f"  Request features: {request_feature_size}")
        logger.info(f"  Combined input: {combined_input_size}")
        logger.info(f"  Hidden dim: {hidden_dim}")
        
        # Network architecture: similar to pod_scorer in contextual bandit
        self.latency_predictor = nn.Sequential(
            nn.Linear(combined_input_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)  # Output single latency value
        )
        
        # Initialize weights
        if weight_initialization == 'xavier':
            self._xavier_initialize_weights()
        elif weight_initialization == 'kaiming':
            self._kaiming_initialize_weights()
        elif weight_initialization == 'static':
            self._static_weight_initialization()
        else:
            logger.warning(f"Unknown weight initialization: {weight_initialization}, using Xavier")
            self._xavier_initialize_weights()
    
    def _xavier_initialize_weights(self):
        """Xavier/Glorot initialization for better gradient flow"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.01)
    
    def _kaiming_initialize_weights(self):
        """He/Kaiming initialization for ReLU networks"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.01)
    
    def _static_weight_initialization(self):
        """Static initialization for testing determinism"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.constant_(module.weight, 0.1)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.01)
    
    def forward(self, pod_features, kv_hit_ratios, request_features):
        """
        Forward pass to predict latency for each pod-request combination.
        
        Args:
            pod_features: [batch_size, num_pods, pod_feature_dim]
            kv_hit_ratios: [batch_size, num_pods, kv_dim]  
            request_features: [batch_size, request_dim]
            
        Returns:
            predicted_latencies: [batch_size, num_pods] - predicted latency for each pod
        """
        batch_size = pod_features.shape[0]
        num_pods = pod_features.shape[1]
        
        # Combine pod features and kv ratios for each pod
        combined_pod_features = torch.cat([pod_features, kv_hit_ratios], dim=2)
        # combined_pod_features: [batch, num_pods, pod_feature_dim + kv_dim]
        
        # Expand request features to match each pod
        # request_features: [batch, request_dim] → [batch, num_pods, request_dim]
        expanded_request = request_features.unsqueeze(1).expand(-1, num_pods, -1)
        
        # Combine all features for each pod-request pair
        # full_features: [batch, num_pods, pod_features + kv_features + request_features]
        full_features = torch.cat([combined_pod_features, expanded_request], dim=2)
        
        # Reshape to process all pod-request pairs in batch
        # [batch * num_pods, combined_feature_size]
        reshaped_features = full_features.view(batch_size * num_pods, -1)
        
        # Predict latency for each pod-request pair
        predicted_latencies = self.latency_predictor(reshaped_features)  # [batch * num_pods, 1]
        
        # Reshape back to [batch_size, num_pods]
        predicted_latencies = predicted_latencies.view(batch_size, num_pods)
        
        return predicted_latencies
    
    def predict_and_select_pod(self, pod_features, kv_hit_ratios, request_features):
        """
        Predict latencies and select the pod with lowest predicted latency.
        
        Returns:
            dict with selected_pod_index, predicted_latencies, confidence
        """
        with torch.no_grad():
            predicted_latencies = self.forward(pod_features, kv_hit_ratios, request_features)
            
            # Select pod with minimum predicted latency
            selected_pod_indices = torch.argmin(predicted_latencies, dim=1)
            min_latencies = torch.min(predicted_latencies, dim=1)[0]
            
            # Calculate confidence as inverse of minimum latency (normalized)
            # Higher confidence for lower predicted latency
            max_latency = torch.max(predicted_latencies, dim=1)[0]
            confidence = 1.0 - (min_latencies / (max_latency + 1e-8))

            return {
                'selected_pod_index': selected_pod_indices,
                'predicted_latencies': predicted_latencies,
                'min_latencies': min_latencies,
                'confidence': confidence
            }


class LatencyDataset(Dataset):
    """
    Dataset for latency prediction training.
    
    For each sample, we extract the features of the selected pod and use the 
    actual latency as the target for supervised learning.
    """
    
    def __init__(self, tensor_data, latency_metric='ttft'):
        """
        Args:
            tensor_data: Same tensor data from encoding pipeline
            latency_metric: Which latency to predict ('ttft', 'avg_tpot', 'e2e_latency')
        """
        self.pod_features = tensor_data['pod_features_with_staleness']
        self.kv_hit_ratios = tensor_data['kv_hit_ratios']
        self.request_features = tensor_data['request_features']
        self.actions = tensor_data['actions']  # Selected pod indices
        
        # Select target latency metric
        if latency_metric == 'ttft':
            self.target_latencies = tensor_data['ttft']
        elif latency_metric == 'avg_tpot':
            self.target_latencies = tensor_data['avg_tpot']
        elif latency_metric == 'e2e_latency':
            self.target_latencies = tensor_data['e2e_latency']
        else:
            raise ValueError(f"Unknown latency metric: {latency_metric}")
        
        self.latency_metric = latency_metric
        logger.info(f"Created LatencyDataset with {len(self)} samples, targeting {latency_metric}")
        
    def __len__(self):
        return len(self.target_latencies)
    
    def __getitem__(self, idx):
        """
        Return features and target latency for supervised learning.
        
        For latency prediction, we use the selected pod's features and the actual latency
        as ground truth for that pod-request combination.
        """
        return {
            'pod_features': self.pod_features[idx],
            'kv_hit_ratios': self.kv_hit_ratios[idx], 
            'request_features': self.request_features[idx],
            'selected_pod_idx': self.actions[idx],
            'target_latency': self.target_latencies[idx]
        }


class LatencyPredictor:
    """
    Main class for latency prediction model training and inference.
    """
    
    def __init__(self, state_dims, HYPERPARAMETERS, final_model_dir):
        """
        Args:
            state_dims: Dict with pod_features, kv_hit_ratios, request_features dimensions
            HYPERPARAMETERS: Model hyperparameters
            final_model_dir: Directory to save/load model
        """
        self.state_dims = state_dims
        self.HYPERPARAMETERS = HYPERPARAMETERS
        self.final_model_dir = final_model_dir
        
        # Model configuration
        self.latency_metric = HYPERPARAMETERS.get('LATENCY_METRIC', 'ttft')
        self.hidden_dim = HYPERPARAMETERS.get('hidden_dim', 64)
        self.learning_rate = HYPERPARAMETERS.get('offline_learning_rate', 0.001)
        self.weight_initialization = HYPERPARAMETERS.get('weight_initialization', 'xavier')
        
        # Create network
        self.network = LatencyPredictionNetwork(
            pod_feature_size=state_dims['pod_features'],
            kv_feature_size=state_dims['kv_hit_ratios'],
            request_feature_size=state_dims['request_features'],
            hidden_dim=self.hidden_dim,
            weight_initialization=self.weight_initialization
        ).to(device)
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        # Training tracking - Enhanced for comprehensive plotting
        self.current_epoch = 0
        self.training_losses = []
        self.validation_losses = []
        self.validation_mae = []  # Mean Absolute Error
        self.validation_r2 = []   # R-squared scores
        self.routing_accuracies = []  # How often we pick the actual best pod
        self.epoch_times = []  # Training time per epoch
        
        # Prediction tracking for analysis
        self.latest_predictions = None
        self.latest_targets = None
        self.latest_pod_selections = None
        self.latest_true_selections = None
        
        logger.info(f"Created LatencyPredictor for {self.latency_metric} prediction")
        logger.info(f"Network parameters: {sum(p.numel() for p in self.network.parameters())}")
    
    def train_epoch(self, dataloader):
        """Train for one epoch using MSE loss."""
        self.network.train()
        epoch_losses = []
        
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            pod_features = batch['pod_features'].to(device)
            kv_hit_ratios = batch['kv_hit_ratios'].to(device)
            request_features = batch['request_features'].to(device)
            selected_pod_indices = batch['selected_pod_idx'].to(device)
            target_latencies = batch['target_latency'].to(device).float()
            
            # Forward pass: predict latencies for all pods
            predicted_latencies = self.network(pod_features, kv_hit_ratios, request_features)
            
            # Extract predictions for the selected pods only
            batch_size = predicted_latencies.shape[0]
            selected_predictions = predicted_latencies[torch.arange(batch_size), selected_pod_indices]
            
            # Compute MSE loss between predicted and actual latency for selected pods
            loss = self.criterion(selected_predictions, target_latencies)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            epoch_losses.append(loss.item())
            
            if batch_idx % 10 == 0:
                logger.debug(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = np.mean(epoch_losses)
        self.training_losses.append(avg_loss)
        logger.info(f"Epoch {self.current_epoch}, Avg Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def evaluate(self, dataloader):
        """Enhanced evaluation with comprehensive metrics for plotting."""
        self.network.eval()
        total_loss = 0
        correct_selections = 0
        total_samples = 0
        
        all_predictions = []
        all_targets = []
        all_model_selections = []
        all_true_selections = []
        
        with torch.no_grad():
            for batch in dataloader:
                pod_features = batch['pod_features'].to(device)
                kv_hit_ratios = batch['kv_hit_ratios'].to(device)
                request_features = batch['request_features'].to(device)
                selected_pod_indices = batch['selected_pod_idx'].to(device)
                target_latencies = batch['target_latency'].to(device).float()
                
                # Predict latencies for all pods
                predicted_latencies = self.network(pod_features, kv_hit_ratios, request_features)
                
                # Loss calculation
                batch_size = predicted_latencies.shape[0]
                selected_predictions = predicted_latencies[torch.arange(batch_size), selected_pod_indices]
                loss = self.criterion(selected_predictions, target_latencies)
                total_loss += loss.item()
                
                # Collect predictions for comprehensive metrics
                all_predictions.extend(selected_predictions.cpu().numpy())
                all_targets.extend(target_latencies.cpu().numpy())
                
                # Accuracy: check if model selects the same pod as original
                model_selections = torch.argmin(predicted_latencies, dim=1)
                correct_selections += (model_selections == selected_pod_indices).sum().item()
                total_samples += batch_size
                
                # Collect selections for analysis
                all_model_selections.extend(model_selections.cpu().numpy())
                all_true_selections.extend(selected_pod_indices.cpu().numpy())
        
        # Calculate comprehensive metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_selections / total_samples
        
        # Regression metrics
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        mae = mean_absolute_error(all_targets, all_predictions)
        r2 = r2_score(all_targets, all_predictions)
        
        # Store latest predictions for plotting
        self.latest_predictions = all_predictions
        self.latest_targets = all_targets
        self.latest_pod_selections = np.array(all_model_selections)
        self.latest_true_selections = np.array(all_true_selections)
        
        # Track metrics for plotting
        self.validation_losses.append(avg_loss)
        self.validation_mae.append(mae)
        self.validation_r2.append(r2)
        self.routing_accuracies.append(accuracy)
        
        logger.info(f"Validation - Loss: {avg_loss:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, Routing Accuracy: {accuracy:.4f}")
        
        return {
            'loss': avg_loss,
            'mae': mae,
            'r2': r2,
            'accuracy': accuracy,
            'predictions': all_predictions,
            'targets': all_targets,
            'model_selections': all_model_selections,
            'true_selections': all_true_selections,
            'correct_selections': correct_selections,
            'total_samples': total_samples
        }
    
    def predict(self, pod_features, kv_hit_ratios, request_features):
        """Make prediction for inference."""
        self.network.eval()
        result = self.network.predict_and_select_pod(pod_features, kv_hit_ratios, request_features)
        return result
    
    def save(self, final_model_dir):
        """Save model and configuration."""
        os.makedirs(final_model_dir, exist_ok=True)
        
        # Save model state
        model_path = os.path.join(final_model_dir, 'latency_predictor.pth')
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'training_losses': self.training_losses,
            'latency_metric': self.latency_metric,
            'state_dims': self.state_dims
        }, model_path)
        
        # Save hyperparameters
        config_path = os.path.join(final_model_dir, 'newly_saved_model_config.json')
        config = self.HYPERPARAMETERS.copy()
        config['latency_metric'] = self.latency_metric
        config['state_dims'] = self.state_dims

        # Convert non-JSON-serializable types (sets, etc.) to lists
        def make_json_serializable(obj):
            if isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_json_serializable(item) for item in obj]
            return obj

        config = make_json_serializable(config)

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved LatencyPredictor model to {final_model_dir}")
    
    def load(self, final_model_dir):
        """Load model and configuration."""
        model_path = os.path.join(final_model_dir, 'latency_predictor.pth')
        
        if not os.path.exists(model_path):
            logger.error(f"No model file found at {model_path}")
            return False
        
        # Load model - the file is trusted, so we can use weights_only=False
        # The numpy._core issue is unavoidable with this model file
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Only load model weights for inference
        self.network.load_state_dict(checkpoint['model_state_dict'])
        
        # For inference, we don't need optimizer state, epoch, or training losses
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # Skip this
        self.current_epoch = 0  # Not needed for inference
        self.training_losses = []  # Not needed for inference
        
        logger.info(f"Loaded LatencyPredictor model from {final_model_dir}")
        logger.info(f"Model trained for {self.current_epoch} epochs")
        return True


def load_encoded_data(encoded_data_dir):
    """Load encoded tensor data (reusing function from simpler_contextual_bandit.py)."""
    tensor_files = []
    
    # Find all tensor files
    for root, dirs, files in os.walk(encoded_data_dir):
        for file in files:
            if file == 'tensor_dataset.pt':
                tensor_files.append(os.path.join(root, file))
    
    if not tensor_files:
        logger.error(f"No tensor_dataset.pt files found in {encoded_data_dir}")
        return None
    
    logger.info(f"Found {len(tensor_files)} tensor files")
    
    # Load and combine all tensor data
    combined_data = None
    total_samples = 0
    
    for tensor_file in tensor_files:
        logger.info(f"Loading {tensor_file}")
        batch_data = torch.load(tensor_file, map_location='cpu', weights_only=False)
        
        if combined_data is None:
            combined_data = batch_data
            total_samples = len(batch_data['actions'])
        else:
            # Concatenate tensors
            for key in batch_data.keys():
                if isinstance(batch_data[key], torch.Tensor):
                    combined_data[key] = torch.cat([combined_data[key], batch_data[key]], dim=0)
            total_samples += len(batch_data['actions'])
    
    logger.info(f"Successfully loaded {total_samples} samples")
    return combined_data


def train_latency_predictor(encoded_data_dir, final_model_dir, HYPERPARAMETERS):
    """
    Main training function for latency predictor.
    
    Args:
        encoded_data_dir: Directory containing encoded tensor data
        final_model_dir: Directory to save trained model
        HYPERPARAMETERS: Model hyperparameters
    """
    logger.info("Starting latency predictor training...")
    
    # Load encoded data
    combined_data = load_encoded_data(encoded_data_dir)
    if combined_data is None:
        logger.error("Failed to load encoded data")
        return None
    
    # Get state dimensions
    state_dims = {
        'pod_features': combined_data['pod_features_with_staleness'].shape[2],
        'kv_hit_ratios': combined_data['kv_hit_ratios'].shape[2],
        'request_features': combined_data['request_features'].shape[1],
        'num_pods': combined_data['pod_features_with_staleness'].shape[1]
    }
    
    logger.info(f"State dimensions: {state_dims}")
    
    # Create model
    latency_metric = HYPERPARAMETERS.get('LATENCY_METRIC', 'ttft')
    predictor = LatencyPredictor(state_dims, HYPERPARAMETERS, final_model_dir)
    
    # Create dataset and dataloader
    dataset = LatencyDataset(combined_data, latency_metric=latency_metric)
    
    # Train/validation split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    batch_size = HYPERPARAMETERS.get('batch_size', 64)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Train samples: {train_size}, Validation samples: {val_size}")
    
    # Training loop
    num_epochs = HYPERPARAMETERS.get('training_epochs', 5)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        predictor.current_epoch = epoch
        epoch_start_time = time.time()
        
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss = predictor.train_epoch(train_loader)
        
        # Validate
        val_metrics = predictor.evaluate(val_loader)
        
        # Track epoch time
        epoch_time = time.time() - epoch_start_time
        predictor.epoch_times.append(epoch_time)
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            predictor.save(final_model_dir)
            logger.info(f"Saved best model with validation loss: {best_val_loss:.4f}")
        
        logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.1f}s")
    
    logger.info("Training completed!")
    
    # Generate comprehensive training plots
    try:
        plot_path = plot_latency_predictor_metrics(predictor, train_dataset, val_dataset, final_model_dir)
        logger.info(f"Generated training plots: {plot_path}")
    except Exception as e:
        logger.error(f"Error generating plots: {e}")
        import traceback
        traceback.print_exc()
    
    return plot_path


def infer_latency_predictor_with_model(predictor, tensor_data, request_id, sorted_all_pod_ids):
    """
    Inference function using a cached latency predictor model.

    Args:
        predictor: Cached LatencyPredictor instance
        tensor_data: Input tensor data
        request_id: Request identifier
        sorted_all_pod_ids: List of pod IDs in same order as predictions

    Returns:
        Dict with prediction results. predicted_latencies will be a dict {pod_id: latency}
    """
    overhead_summary = {}

    # Prepare input tensors
    prepare_start = time.time()
    pod_features = tensor_data['pod_features_with_staleness'].to(device)
    kv_hit_ratios = tensor_data['kv_hit_ratios'].to(device)
    request_features = tensor_data['request_features'].to(device)

    # Ensure batch format
    if len(pod_features.shape) == 2:
        pod_features = pod_features.unsqueeze(0)
    if len(kv_hit_ratios.shape) == 2:
        kv_hit_ratios = kv_hit_ratios.unsqueeze(0)
    if len(request_features.shape) == 1:
        request_features = request_features.unsqueeze(0)
    overhead_summary['prepare_tensors'] = time.time() - prepare_start

    # Make prediction
    predict_start = time.time()
    result = predictor.predict(pod_features, kv_hit_ratios, request_features)
    overhead_summary['model_inference'] = time.time() - predict_start

    # Format result for compatibility
    format_start = time.time()
    selected_pod_index = result['selected_pod_index'][0].item()
    predicted_latencies_tensor = result['predicted_latencies'][0]  # Keep as tensor
    predicted_latencies = predicted_latencies_tensor.cpu().numpy()
    confidence = float(result['confidence'].item() if result['confidence'].numel() == 1 else result['confidence'][0].item())

    # Apply softmax to convert latencies to probabilities (lower latency = higher probability)
    softmax_probs = torch.softmax(-predicted_latencies_tensor, dim=0).cpu().numpy()

    # Format predicted_latencies as dict
    predicted_latencies_formatted = {sorted_all_pod_ids[i]: float(latency.item()) for i, latency in enumerate(predicted_latencies)}
    chosen_pod_predicted_latency = float(predicted_latencies_formatted[sorted_all_pod_ids[selected_pod_index]])
    overhead_summary['format_results'] = time.time() - format_start

    return {
        'selected_pod_index': selected_pod_index,
        'predicted_latencies': predicted_latencies_formatted,
        'chosen_pod_predicted_latency': chosen_pod_predicted_latency,
        'confidence': confidence,
        'pod_probabilities': [float(p) for p in softmax_probs.tolist()],
        'latency_metric': predictor.latency_metric,
        'explore_mask': 0  # Latency predictor always uses exploitation (no exploration)
    }, overhead_summary


def infer_latency_predictor(tensor_data, request_id, model_updated, HYPERPARAMETERS, final_model_dir, sorted_all_pod_ids):
    """
    Legacy inference function for latency predictor (creates new model each time).

    DEPRECATED: Use infer_latency_predictor_with_model() with a cached model instead.
    This function is kept for backward compatibility but is inefficient.

    Args:
        tensor_data: Input tensor data
        request_id: Request identifier
        model_updated: Whether model was updated (for compatibility)
        HYPERPARAMETERS: Model hyperparameters
        final_model_dir: Model directory
        sorted_all_pod_ids: Optional list of pod IDs in same order as predictions

    Returns:
        Dict with prediction results. predicted_latencies will be a dict {pod_id: latency}
        if sorted_all_pod_ids is provided, otherwise a list.
    """
    # Load model if needed
    state_dims = {
        'pod_features': tensor_data['pod_features_with_staleness'].shape[2],
        'kv_hit_ratios': tensor_data['kv_hit_ratios'].shape[2],
        'request_features': tensor_data['request_features'].shape[1],
        'num_pods': tensor_data['pod_features_with_staleness'].shape[1]
    }

    predictor = LatencyPredictor(state_dims, HYPERPARAMETERS, final_model_dir)

    # FAIL FAST - Let any loading errors propagate up
    predictor.load(final_model_dir)

    # Use the new cached inference function
    return infer_latency_predictor_with_model(predictor, tensor_data, request_id, sorted_all_pod_ids)


def plot_latency_predictor_metrics(predictor, train_data, val_data, final_model_dir):
    """
    Create comprehensive training metrics visualization for latency predictor.
    
    Args:
        predictor: Trained LatencyPredictor instance
        train_data: Training dataset for analysis
        val_data: Validation dataset for analysis  
        final_model_dir: Directory to save plots
    
    Returns:
        Path to saved plot file
    """
    # Set matplotlib style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Determine number of pods
    num_pods = predictor.state_dims['num_pods']
    latency_metric = predictor.latency_metric.upper()
    
    # 1. Training Loss
    plt.subplot(3, 4, 1)
    if predictor.training_losses:
        plt.plot(predictor.training_losses, 'b-', linewidth=2, label='Training Loss')
        if predictor.validation_losses:
            plt.plot(predictor.validation_losses, 'r-', linewidth=2, label='Validation Loss')
        plt.title(f'{latency_metric} Prediction Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add final loss annotation
        if predictor.training_losses:
            final_train_loss = predictor.training_losses[-1]
            final_val_loss = predictor.validation_losses[-1] if predictor.validation_losses else None
            loss_text = f'Train: {final_train_loss:.3f}'
            if final_val_loss:
                loss_text += f'\nVal: {final_val_loss:.3f}'
            plt.text(0.02, 0.98, loss_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 2. Mean Absolute Error
    plt.subplot(3, 4, 2)
    if predictor.validation_mae:
        plt.plot(predictor.validation_mae, 'g-', linewidth=2)
        plt.title(f'Mean Absolute Error ({latency_metric})')
        plt.xlabel('Epoch')
        plt.ylabel(f'MAE ({latency_metric.lower()} units)')
        plt.grid(True, alpha=0.3)
        
        # Add final MAE
        final_mae = predictor.validation_mae[-1]
        plt.text(0.02, 0.98, f'Final MAE: {final_mae:.3f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # 3. R-squared Score
    plt.subplot(3, 4, 3)
    if predictor.validation_r2:
        plt.plot(predictor.validation_r2, 'purple', linewidth=2)
        plt.title('R² Score (Goodness of Fit)')
        plt.xlabel('Epoch')
        plt.ylabel('R² Score')
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.1, 1.0)
        
        # Add interpretation
        final_r2 = predictor.validation_r2[-1]
        if final_r2 > 0.8:
            interpretation = "Excellent"
        elif final_r2 > 0.6:
            interpretation = "Good"
        elif final_r2 > 0.3:
            interpretation = "Moderate"
        else:
            interpretation = "Poor"
            
        plt.text(0.02, 0.98, f'R²: {final_r2:.3f}\n{interpretation}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # 4. Routing Accuracy
    plt.subplot(3, 4, 4)
    if predictor.routing_accuracies:
        plt.plot(predictor.routing_accuracies, 'orange', linewidth=2)
        plt.title('Routing Decision Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Add random baseline
        random_baseline = 1.0 / num_pods
        plt.axhline(y=random_baseline, color='r', linestyle='--', 
                   label=f'Random ({random_baseline:.3f})')
        plt.legend()
        
        # Add final accuracy
        final_acc = predictor.routing_accuracies[-1]
        plt.text(0.02, 0.98, f'Final: {final_acc:.3f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    # 5. Predicted vs Actual Latency Scatter
    plt.subplot(3, 4, 5)
    if predictor.latest_predictions is not None and predictor.latest_targets is not None:
        predictions = predictor.latest_predictions
        targets = predictor.latest_targets
        
        plt.scatter(targets, predictions, alpha=0.6, s=20)
        
        # Perfect prediction line
        min_val = min(min(targets), min(predictions))
        max_val = max(max(targets), max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, 
                label='Perfect Prediction')
        
        plt.xlabel(f'Actual {latency_metric}')
        plt.ylabel(f'Predicted {latency_metric}')
        plt.title(f'Prediction Accuracy Scatter')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr = np.corrcoef(targets, predictions)[0, 1]
        plt.text(0.02, 0.98, f'Correlation: {corr:.3f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 6. Prediction Error Distribution
    plt.subplot(3, 4, 6)
    if predictor.latest_predictions is not None and predictor.latest_targets is not None:
        errors = predictor.latest_predictions - predictor.latest_targets
        
        plt.hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        plt.xlabel(f'Prediction Error ({latency_metric})')
        plt.ylabel('Frequency')
        plt.title('Prediction Error Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        plt.text(0.02, 0.98, f'Mean: {mean_error:.3f}\nStd: {std_error:.3f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # 7. Pod Selection Distribution (Model vs Truth)
    plt.subplot(3, 4, 7)
    if predictor.latest_pod_selections is not None and predictor.latest_true_selections is not None:
        model_counts = np.bincount(predictor.latest_pod_selections, minlength=num_pods)
        true_counts = np.bincount(predictor.latest_true_selections, minlength=num_pods)
        
        x = np.arange(num_pods)
        width = 0.35
        
        bars1 = plt.bar(x - width/2, true_counts, width, 
                       label='Ground Truth', alpha=0.7, color='lightcoral')
        bars2 = plt.bar(x + width/2, model_counts, width, 
                       label='Model Prediction', alpha=0.7, color='lightblue')
        
        plt.title('Pod Selection Distribution')
        plt.xlabel('Pod ID')
        plt.ylabel('Selection Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add counts on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    plt.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                            f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    # 8. Average Predicted Latency by Pod
    plt.subplot(3, 4, 8)
    if val_data is not None:
        # Generate predictions for all pods on validation data
        predictor.network.eval()
        with torch.no_grad():
            # Sample some validation data
            sample_size = min(200, len(val_data))
            indices = torch.randperm(len(val_data))[:sample_size]
            
            sample_data = torch.utils.data.Subset(val_data, indices)
            sample_loader = DataLoader(sample_data, batch_size=64, shuffle=False)
            
            all_pod_predictions = []
            for batch in sample_loader:
                pod_features = batch['pod_features'].to(device)
                kv_hit_ratios = batch['kv_hit_ratios'].to(device)
                request_features = batch['request_features'].to(device)
                
                predicted_latencies = predictor.network(pod_features, kv_hit_ratios, request_features)
                all_pod_predictions.append(predicted_latencies.cpu().numpy())
            
            if all_pod_predictions:
                all_predictions = np.concatenate(all_pod_predictions, axis=0)
                avg_latency_per_pod = np.mean(all_predictions, axis=0)
                
                bars = plt.bar(range(num_pods), avg_latency_per_pod, 
                              color='orange', alpha=0.7)
                plt.title(f'Average Predicted {latency_metric} by Pod')
                plt.xlabel('Pod ID')
                plt.ylabel(f'Avg Predicted {latency_metric}')
                plt.grid(True, alpha=0.3)
                
                # Add values on bars
                for i, (bar, latency) in enumerate(zip(bars, avg_latency_per_pod)):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avg_latency_per_pod)*0.01,
                            f'{latency:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 9. Training Time per Epoch
    plt.subplot(3, 4, 9)
    if predictor.epoch_times:
        plt.plot(predictor.epoch_times, 'brown', linewidth=2, marker='o')
        plt.title('Training Time per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Time (seconds)')
        plt.grid(True, alpha=0.3)
        
        # Add average time
        avg_time = np.mean(predictor.epoch_times)
        plt.text(0.02, 0.98, f'Avg: {avg_time:.1f}s', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 10. Learning Curves Summary  
    plt.subplot(3, 4, 10)
    if predictor.training_losses and predictor.validation_losses:
        # Normalize curves for comparison
        train_loss_norm = np.array(predictor.training_losses)
        val_loss_norm = np.array(predictor.validation_losses)
        
        # Normalize to 0-1 range
        train_loss_norm = (train_loss_norm - train_loss_norm.min()) / (train_loss_norm.max() - train_loss_norm.min() + 1e-8)
        val_loss_norm = (val_loss_norm - val_loss_norm.min()) / (val_loss_norm.max() - val_loss_norm.min() + 1e-8)
        
        plt.plot(train_loss_norm, label='Training Loss', alpha=0.7)
        plt.plot(val_loss_norm, label='Validation Loss', alpha=0.7)
        
        if predictor.routing_accuracies:
            acc_norm = np.array(predictor.routing_accuracies)
            plt.plot(acc_norm, label='Routing Accuracy', alpha=0.7)
        
        plt.title('Learning Curves (Normalized)')
        plt.xlabel('Epoch')
        plt.ylabel('Normalized Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 11. Model Architecture Info
    plt.subplot(3, 4, 11)
    plt.axis('off')
    
    # Create architecture summary
    arch_text = f"LATENCY PREDICTOR\n{'='*20}\n"
    arch_text += f"Target Metric: {latency_metric}\n"
    arch_text += f"Pod Features: {predictor.state_dims['pod_features']}\n"
    arch_text += f"KV Features: {predictor.state_dims['kv_hit_ratios']}\n"
    arch_text += f"Request Features: {predictor.state_dims['request_features']}\n"
    arch_text += f"Hidden Dim: {predictor.hidden_dim}\n"
    arch_text += f"Num Pods: {num_pods}\n"
    
    total_params = sum(p.numel() for p in predictor.network.parameters())
    arch_text += f"Parameters: {total_params:,}\n"
    arch_text += f"Learning Rate: {predictor.learning_rate}\n"
    
    plt.text(0.1, 0.9, arch_text, transform=plt.gca().transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 12. Performance Summary
    plt.subplot(3, 4, 12)
    plt.axis('off')
    
    summary_text = "PERFORMANCE SUMMARY\n" + "="*18 + "\n"
    
    if predictor.training_losses:
        final_train_loss = predictor.training_losses[-1]
        summary_text += f"Final Train Loss: {final_train_loss:.4f}\n"
    
    if predictor.validation_losses:
        final_val_loss = predictor.validation_losses[-1]
        summary_text += f"Final Val Loss: {final_val_loss:.4f}\n"
    
    if predictor.validation_mae:
        final_mae = predictor.validation_mae[-1]
        summary_text += f"Final MAE: {final_mae:.4f}\n"
    
    if predictor.validation_r2:
        final_r2 = predictor.validation_r2[-1]
        summary_text += f"Final R²: {final_r2:.4f}\n"
    
    if predictor.routing_accuracies:
        final_acc = predictor.routing_accuracies[-1]
        random_baseline = 1.0 / num_pods
        summary_text += f"Routing Accuracy: {final_acc:.4f}\n"
        summary_text += f"Random Baseline: {random_baseline:.4f}\n"
        
        if final_acc > random_baseline * 1.5:
            summary_text += "\n✅ STRONG LEARNING\n"
        elif final_acc > random_baseline * 1.2:
            summary_text += "\n✅ GOOD LEARNING\n"
        elif final_acc > random_baseline * 1.1:
            summary_text += "\n⚠️ MODEST LEARNING\n"
        else:
            summary_text += "\n❌ LIMITED LEARNING\n"
    
    # Model quality assessment
    if predictor.validation_r2:
        if final_r2 > 0.8:
            summary_text += "Excellent Fit\n"
        elif final_r2 > 0.6:
            summary_text += "Good Fit\n"
        elif final_r2 > 0.3:
            summary_text += "Moderate Fit\n"
        else:
            summary_text += "Poor Fit\n"
    
    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    pdf_fn = f"{final_model_dir}/comprehensive_latency_predictor_metrics.pdf"
    plt.savefig(pdf_fn, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"* Saved latency predictor training plots: {pdf_fn}")
    
    # Print summary to console
    logger.info("\n" + "="*60)
    logger.info("LATENCY PREDICTOR TRAINING SUMMARY")
    logger.info("="*60)
    logger.info(f"Target metric: {latency_metric}")
    logger.info(f"Model parameters: {total_params:,}")
    
    if predictor.validation_mae:
        logger.info(f"Final MAE: {predictor.validation_mae[-1]:.4f}")
    if predictor.validation_r2:
        logger.info(f"Final R²: {predictor.validation_r2[-1]:.4f}")
    if predictor.routing_accuracies:
        final_acc = predictor.routing_accuracies[-1]
        random_baseline = 1.0 / num_pods
        logger.info(f"Routing accuracy: {final_acc:.4f} (vs random {random_baseline:.4f})")
        
        if final_acc > random_baseline * 1.5:
            logger.info("✅ Model shows strong latency-based routing!")
        elif final_acc > random_baseline * 1.2:
            logger.info("✅ Model shows good latency prediction!")
        elif final_acc > random_baseline * 1.1:
            logger.info("⚠️  Model shows modest improvement")
        else:
            logger.info("❌ Model performance close to random")
    logger.info("="*60)
    
    return pdf_fn


if __name__ == "__main__":
    # Test the latency predictor
    print("LatencyPredictor module loaded successfully!")
