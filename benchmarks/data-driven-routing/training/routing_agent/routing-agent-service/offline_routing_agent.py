# offline_routing_agent.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import os
import logging
import time
import sys
import encoding
import simpler_contextual_bandit
from logger import logger
import preprocess
import pickle
import threading
import argparse
import random_forest
import torch


# Global variables (simplified for offline use)
ENCODED_DATA_DIR = "encoded_data"
STATS_FILE = "request_feature_stats.pkl"
NUM_TRAINS = 0
MODEL_UPDATED = False
TRAINING_DATA_UPDATED = False
TOTAL_NUM_DATA = 0
MIN_NUM_TRAINING_DATA = 500
LOCK_TRAINING_DATA = threading.Lock()
signal_amplification_degree = 1.0 # 1.5
reward_amplification_degree = 2.0
reward_amplification_threshold = 0.5

# Copy all the classes from original file
class RunningStats:
    """Maintains running mean and standard deviation for feature normalization"""
    def __init__(self, feature_names=None):
        self.count = 0
        self.mean = None
        self.var = None  # Variance
        self.feature_names = feature_names
        
    def update(self, new_data):
        """Update statistics with new batch of data"""
        if new_data is None or len(new_data) == 0:
            return
        
        # Convert to numpy array
        new_data = np.array(new_data, dtype=np.float64)
        new_count = len(new_data)
        
        # First update
        if self.count == 0:
            self.mean = np.mean(new_data, axis=0)
            self.var = np.var(new_data, axis=0) * new_count
            self.count = new_count
            logger.info(f"Initialized running stats with {new_count} samples")
            return
        
        # Compute batch statistics
        batch_mean = np.mean(new_data, axis=0)
        batch_var = np.var(new_data, axis=0) * new_count
        
        # Update running statistics using Welford's algorithm
        new_count = len(new_data)
        new_total = self.count + new_count
        
        # Update mean
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * new_count / new_total
        
        # Update variance
        self.var = self.var + batch_var + delta**2 * self.count * new_count / new_total
        
        # Update count
        self.count = new_total
        
        logger.debug(f"Updated running stats, now have {self.count} samples")
        
    def get_mean(self):
        """Get current mean"""
        return self.mean if self.mean is not None else 0
        
    def get_std(self):
        """Get current standard deviation"""
        if self.count <= 1 or self.var is None:
            return np.ones_like(self.mean) if self.mean is not None else 1.0
        std = np.sqrt(self.var / self.count)
        # Ensure no zeros to prevent division by zero during normalization
        if isinstance(std, np.ndarray):
            std[std < 1.0] = 1.0
        return std
        
    def normalize(self, data):
        """Normalize data using current statistics"""
        if self.count == 0:
            logger.warning("No statistics available, returning original data")
            return data
        
        mean = self.get_mean()
        std = self.get_std()
        
        return (data - mean) / std
        
    def save(self, filename):
        """Save statistics to file"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'count': self.count,
                'mean': self.mean,
                'var': self.var,
                'feature_names': self.feature_names
            }, f)
        logger.info(f"Saved running statistics to {filename}")
        
    @classmethod
    def load(cls, filename):
        """Load statistics from file"""
        if not os.path.exists(filename):
            logger.info(f"Statistics file {filename} not found, initializing new stats")
            return cls()
        
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            
        stats = cls(feature_names=data.get('feature_names'))
        stats.count = data.get('count', 0)
        stats.mean = data.get('mean')
        stats.var = data.get('var')
        
        logger.info(f"Loaded running statistics from {filename} with {stats.count} samples")
        return stats

class PerFeatureRunningStats:
    """Maintains separate running statistics for each feature"""
    def __init__(self):
        self.feature_stats = {}  # Dict[feature_name, RunningStats]
        
    def update(self, data, feature_names):
        """Update statistics for each feature separately"""
        if data is None or len(data) == 0:
            return
            
        data = np.array(data, dtype=np.float64)
        
        for i, feature_name in enumerate(feature_names):
            if feature_name not in self.feature_stats:
                self.feature_stats[feature_name] = RunningStats()
            
            # Extract single feature column
            feature_data = data[:, i:i+1]  # Keep 2D shape
            self.feature_stats[feature_name].update(feature_data)
    
    def normalize(self, data, feature_names):
        """Normalize each feature separately using its own statistics"""
        if data is None or len(data) == 0:
            return data
            
        data = np.array(data, dtype=np.float64)
        normalized_data = np.zeros_like(data)
        
        for i, feature_name in enumerate(feature_names):
            if feature_name in self.feature_stats and self.feature_stats[feature_name].count > 0:
                # Normalize using feature-specific stats
                feature_column = data[:, i:i+1]  # Keep 2D shape
                normalized_column = self.feature_stats[feature_name].normalize(feature_column)
                normalized_data[:, i] = normalized_column.flatten()
            else:
                # No normalization if no stats available
                normalized_data[:, i] = data[:, i]
                logger.warning(f"No statistics available for feature '{feature_name}', using original values")
        
        return normalized_data
    
    def save(self, filename):
        """Save all feature statistics to file"""
        save_data = {}
        for feature_name, stats in self.feature_stats.items():
            save_data[feature_name] = {
                'count': stats.count,
                'mean': stats.mean,
                'var': stats.var,
                'feature_names': stats.feature_names
            }
        
        with open(filename, 'wb') as f:
            pickle.dump(save_data, f)
        logger.info(f"Saved per-feature statistics for {len(self.feature_stats)} features to {filename}")
    
    @classmethod
    def load(cls, filename):
        """Load feature statistics from file"""
        if not os.path.exists(filename):
            logger.info(f"Statistics file {filename} not found, initializing new per-feature stats")
            return cls()
        
        instance = cls()
        
        try:
            with open(filename, 'rb') as f:
                save_data = pickle.load(f)
            
            # Handle both old format (single RunningStats) and new format (per-feature)
            if isinstance(save_data, dict) and 'count' in save_data:
                # Old format - single RunningStats for all features
                logger.info("Found old format statistics file, initializing new per-feature stats")
                return cls()
            
            # New format - per-feature statistics
            for feature_name, stats_data in save_data.items():
                stats = RunningStats(feature_names=stats_data.get('feature_names'))
                stats.count = stats_data.get('count', 0)
                stats.mean = stats_data.get('mean')
                stats.var = stats_data.get('var')
                instance.feature_stats[feature_name] = stats
            
            logger.info(f"Loaded per-feature statistics for {len(instance.feature_stats)} features from {filename}")
            
        except Exception as e:
            logger.warning(f"Error loading statistics file {filename}: {e}, initializing new stats")
            return cls()
        
        return instance
    
    @property
    def count(self):
        """Return total count across all features (for compatibility)"""
        if not self.feature_stats:
            return 0
        return max(stats.count for stats in self.feature_stats.values())
    
    def get_feature_names(self):
        """Get list of all feature names with statistics"""
        return list(self.feature_stats.keys())

request_stats = None

def get_request_stats():
    """Get or initialize request feature statistics"""
    global request_stats
    if request_stats is None:
        request_stats = PerFeatureRunningStats.load(STATS_FILE)
    return request_stats

def write_to_file(log_data, raw_data):
    with open(raw_data, "w") as log_file:
        for request_id, log_message in log_data.items():
            log_file.write(f"{log_message}\n")
    logger.info(f"Successfully wrote {len(log_data)} entries to {raw_data}")

request_features_train = ['input_tokens', 'output_tokens', 'total_tokens']
request_features_reward = ['ttft', 'avg_tpot', 'e2e_latency']

def read_csv_data(csv_file):
    """Read log messages from CSV file"""
    logger.info(f"Reading data from {csv_file}")
    
    # Try to read as CSV first
    try:
        df = pd.read_csv(csv_file)
        if 'log_message' in df.columns:
            log_messages = df['log_message'].tolist()
        elif len(df.columns) == 1:
            # Single column, assume it's log messages
            log_messages = df.iloc[:, 0].tolist()
        else:
            logger.error(f"CSV file must have a 'log_message' column or be a single column file")
            return None
    except:
        # If CSV reading fails, try reading as plain text file
        try:
            with open(csv_file, 'r') as f:
                log_messages = [line.strip() for line in f if line.strip()]
        except Exception as e:
            logger.error(f"Error reading file {csv_file}: {e}")
            return None
    
    # Clean log messages - remove Go log prefixes
    cleaned_messages = []
    for i, log_message in enumerate(log_messages):
        if log_message and log_message.strip():
            # Remove Go log prefix: "I0527 05:00:00.334792       1 gateway_rsp_body.go:647] "
            # Look for the pattern and extract everything after the "] "
            clean_message = log_message.strip()
            
            # Debug: show original message for first few entries
            if i < 3:
                logger.info(f"Original message {i}: {clean_message[:150]}...")
            
            # Find the last occurrence of "] " which should mark the end of the log prefix
            bracket_pos = clean_message.rfind('] ')
            if bracket_pos != -1:
                # Extract everything after "] "
                clean_message = clean_message[bracket_pos + 2:]
            
            # Additional check: if message starts with "**@latency_metrics@", keep it as is
            # Otherwise, try to find that pattern in the message
            if not clean_message.startswith('**@latency_metrics@'):
                metrics_pos = clean_message.find('**@latency_metrics@')
                if metrics_pos != -1:
                    clean_message = clean_message[metrics_pos:]
            
            if clean_message.startswith('**@latency_metrics@'):
                cleaned_messages.append(clean_message)
                # Debug: show cleaned message for first few entries
                if i < 3:
                    logger.info(f"Cleaned message {i}: {clean_message[:150]}...")
            else:
                logger.warning(f"Skipping malformed log message {i}: {log_message[:100]}...")
    
    # Convert to dictionary format expected by the training pipeline
    log_data = {}
    for i, log_message in enumerate(cleaned_messages):
        log_data[f"request_{i}"] = log_message
    
    logger.info(f"Successfully read {len(log_data)} log messages from {csv_file} (cleaned from {len(log_messages)} raw entries)")
    
    # Show a sample of the final cleaned data
    if log_data:
        first_key = list(log_data.keys())[0]
        sample_message = log_data[first_key]
        logger.info(f"Sample cleaned message: {sample_message[:200]}...")
    
    return log_data

# def process_training_data(args, log_data):
#     """Process training data - create single batch from CSV input"""
#     global ENCODED_DATA_DIR, NUM_TRAINS, MODEL_UPDATED, TRAINING_DATA_UPDATED, TOTAL_NUM_DATA
    
#     flush_start_time = time.time()
    
#     try:
#         logger.info(f"Processing training data with {len(log_data)} entries")
        
#         # Create raw data file (simplified - just one batch)
#         if not os.path.exists("temp_training_data"):
#             os.mkdir("temp_training_data")
#         raw_data = "temp_training_data/offline_batch.csv"
        
#         # Write raw data to file
#         ts_write_raw_data = time.time()
#         write_to_file(log_data, raw_data)
#         logger.info(f"Wrote {len(log_data)} entries to {raw_data}, took {time.time() - ts_write_raw_data} seconds")

#         # Preprocess raw data
#         ts_preprocess = time.time()
#         df, _, all_pods, _ = preprocess.main(raw_data, "", args.ttft_slo, args.avg_tpot_slo)
#         logger.info(f"Successfully parsed data, took {time.time() - ts_preprocess} seconds")
        
#         # Re-enable PROPER per-feature normalization 
#         request_features = ['input_tokens', 'output_tokens', 'total_tokens']
#         pod_features_cols = [col for col in df.columns if col.startswith('pod_') and df[col].dtype in ['float64', 'int64']]
#         all_features = request_features + pod_features_cols
#         logger.info(f"feature print. Request features: {request_features}, Pod features: {pod_features_cols}")
#         logger.info(f"Found {len(all_features)} features to normalize: {len(request_features)} request + {len(pod_features_cols)} pod features")
        
#         # Get or create stats object
#         stats = get_request_stats()
        
#         # Normalize each feature INDIVIDUALLY (not as a group)
#         for feature in all_features:
#             if feature in df.columns:
#                 # Extract single feature column as 2D array for consistency
#                 feature_data = df[feature].values.reshape(-1, 1)
                
#                 # Update stats for this specific feature
#                 if feature not in stats.feature_stats:
#                     stats.feature_stats[feature] = RunningStats()
                
#                 stats.feature_stats[feature].update(feature_data)
                
#                 # Normalize this feature using its own stats
#                 normalized_feature = stats.feature_stats[feature].normalize(feature_data)
#                 df[feature] = normalized_feature.flatten()
                
#                 logger.debug(f"Normalized {feature}: range [{df[feature].min():.3f}, {df[feature].max():.3f}], mean={df[feature].mean():.3f}, std={df[feature].std():.3f}")
        
#         # Save stats
#         stats.save(STATS_FILE)
#         logger.info(f"Applied individual feature normalization to {len(all_features)} features")

#         # Encode preprocessed data - create single batch directory
#         ts_encode = time.time()
#         encoded_data_subdir = f"{ENCODED_DATA_DIR}/batch_1"  # Simple single batch
#         encoding.encode_for_train(all_pods, df, encoded_data_subdir, stats, request_features_train, request_features_reward)
#         logger.info(f"Successfully encoded data to {encoded_data_subdir}, took {time.time() - ts_encode} seconds")
        
#         # Verify the encoded data was created
#         expected_tensor_path = f"{encoded_data_subdir}/tensor_dataset.pt"
#         train_tensor_path = f"{encoded_data_subdir}/train/tensor_dataset.pt"
        
#         if os.path.exists(expected_tensor_path):
#             logger.info(f"✓ Found tensor dataset at: {expected_tensor_path}")
#         elif os.path.exists(train_tensor_path):
#             logger.info(f"✓ Found tensor dataset at: {train_tensor_path}")
#         else:
#             logger.warning(f"⚠️  Tensor dataset not found at expected locations:")
#             logger.warning(f"   - {expected_tensor_path}")
#             logger.warning(f"   - {train_tensor_path}")
#             # List what was actually created
#             if os.path.exists(encoded_data_subdir):
#                 files = os.listdir(encoded_data_subdir)
#                 logger.info(f"   Files in {encoded_data_subdir}: {files}")
                
#                 # Check if there's a train subdirectory
#                 train_dir = f"{encoded_data_subdir}/train"
#                 if os.path.exists(train_dir):
#                     train_files = os.listdir(train_dir)
#                     logger.info(f"   Files in {train_dir}: {train_files}")
        
#         TRAINING_DATA_UPDATED = True
#         TOTAL_NUM_DATA += len(log_data)
        
#         logger.info(f"Successfully processed {len(log_data)} log messages in single batch, took {time.time() - flush_start_time} seconds")
#         return True
        
#     except Exception as e:
#         import traceback
#         error_traceback = traceback.format_exc()
#         logger.error(f"Error processing training data: {str(e)}")
#         logger.error(f"Traceback: {error_traceback}")
#         return False

def train_model(args):
    """Train the model"""
    global NUM_TRAINS, MODEL_UPDATED, TRAINING_DATA_UPDATED, TOTAL_NUM_DATA
    
    if TRAINING_DATA_UPDATED and TOTAL_NUM_DATA > MIN_NUM_TRAINING_DATA:
        training_start_time = time.time()
        logger.info(f"Starting {NUM_TRAINS}th training of routing agent")
        
        try:
            if args.model == "random_forest":
                random_forest.train(ENCODED_DATA_DIR)
            elif args.model == "simpler_contextual_bandit":
                simpler_contextual_bandit.train(ENCODED_DATA_DIR)
            else:
                logger.error(f"Unknown model type: {args.model}")
                return False
            MODEL_UPDATED = True
            TRAINING_DATA_UPDATED = False
            NUM_TRAINS += 1
            
            logger.info(f"Successfully completed {NUM_TRAINS-1}th training of routing agent, took {time.time() - training_start_time} seconds")
            return True
            
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            logger.error(f"Error training model: {str(e)}")
            logger.error(f"Traceback: {error_traceback}")
            return False
    else:
        logger.info(f"Not enough training data available (TOTAL_NUM_DATA: {TOTAL_NUM_DATA}), skipping training")
        return False


# def test_inference(args, log_message):
#     """Test inference on a single log message with original vs predicted comparison"""
#     global NUM_TRAINS, MODEL_UPDATED
    
#     if NUM_TRAINS == 0:
#         logger.warning("No trained model available, please train first")
#         return None
        
#     handle_infer_start_time = time.time()
    
#     try:
#         logger.debug(f"Testing inference on log message: {log_message[:200]}...")
        
#         # Preprocess the log message
#         preprocess_start_time = time.time()
        
#         # Debug: show what we're passing to preprocess
#         logger.debug(f"Calling preprocess.main(None, log_message, {args.ttft_slo}, {args.avg_tpot_slo})")
        
#         processed_df, _, all_pods, preprocess_dataset_overhead_summary = preprocess.main(None, log_message, args.ttft_slo, args.avg_tpot_slo)
#         logger.debug(f"Successfully parsed data for inference")
#         handle_infer_total_total_preprocess_overhead = time.time() - preprocess_start_time

#         # EXTRACT ORIGINAL POD CHOICE FROM PREPROCESSED DATA
#         # The preprocess.main should have extracted this information
#         original_pod_choice = None
#         if 'selected_pod' in processed_df.columns:
#             original_pod_choice = processed_df['selected_pod'].iloc[0] if len(processed_df) > 0 else None
#         elif hasattr(processed_df, 'original_pod') or 'original_pod' in processed_df.columns:
#             original_pod_choice = processed_df['original_pod'].iloc[0] if len(processed_df) > 0 else None
        
#         # If not found in DataFrame, try to extract from the raw log message
#         if original_pod_choice is None:
#             import re
#             pattern = r'@selectedpod@([^@]+)@'
#             match = re.search(pattern, log_message)
#             if match:
#                 original_pod_choice = match.group(1)
        
#         logger.debug(f"Original pod choice: {original_pod_choice}")

#         # Apply SAME individual feature normalization as training
#         request_features = ['input_tokens', 'output_tokens', 'total_tokens']
#         pod_features_cols = [col for col in processed_df.columns if col.startswith('pod_') and processed_df[col].dtype in ['float64', 'int64']]
        
#         all_features = request_features + pod_features_cols
#         stats = get_request_stats()
        
#         # Normalize each feature individually (same as training)
#         if stats.count > 0:
#             for feature in all_features:
#                 if feature in processed_df.columns and feature in stats.feature_stats:
#                     feature_data = processed_df[feature].values.reshape(-1, 1)
#                     normalized_feature = stats.feature_stats[feature].normalize(feature_data)
#                     processed_df[feature] = normalized_feature.flatten()
#                     logger.debug(f"Normalized {feature} for inference: range [{processed_df[feature].min():.3f}, {processed_df[feature].max():.3f}]")
            
#             logger.debug(f"Applied individual feature normalization to {len(all_features)} features for inference")
#         else:
#             logger.warning(f"No normalization stats available for inference")
        
#         # Encode data
#         encode_start_time = time.time()
#         tensor_dataset, encode_for_inference_overhead_summary = encoding.encode_for_inference(all_pods, processed_df, stats, request_features_train, request_features_reward)
#         logger.debug(f"Successfully encoded data in memory for inference")
#         handle_infer_total_total_encoding_overhead = time.time() - encode_start_time

#         # Perform inference
#         infer_from_tensor_start_time = time.time()
#         if args.model == "random_forest":
#             result, infer_from_tensor_overhead_summary = random_forest.infer_from_tensor(
#                 tensor_data=tensor_dataset, 
#                 exploration_enabled=True, 
#                 exploration_rate=0.2, 
#                 model_updated=MODEL_UPDATED
#         )
#         elif args.model == "simpler_contextual_bandit":
#             result, infer_from_tensor_overhead_summary = simpler_contextual_bandit.infer_from_tensor(
#                 tensor_data=tensor_dataset, 
#                 model_updated=MODEL_UPDATED
#             )
#         if MODEL_UPDATED:
#             logger.info("Model updated flag consumed, resetting to False")
#             MODEL_UPDATED = False
#         handle_infer_total_total_infer_from_tensor_overhead = time.time() - infer_from_tensor_start_time

#         # Map the pod index back to the actual pod ID
#         selected_pod_index = result.get('selected_pod_index', 0)
#         if selected_pod_index >= len(all_pods):
#             logger.warning(f"Selected pod index {selected_pod_index} out of range, defaulting to first pod")
#             selected_pod_index = 0
            
#         selected_pod = all_pods[selected_pod_index]
#         confidence = result['confidence']
#         handle_infer_total_overhead = time.time() - handle_infer_start_time
        
#         # Compare with original choice
#         prediction_matches = (selected_pod == original_pod_choice) if original_pod_choice else None
        
#         # Return enhanced result with comparison
#         result_summary = {
#             "selected_pod": selected_pod,
#             "original_pod_choice": original_pod_choice,
#             "prediction_matches": prediction_matches,
#             "confidence": confidence,
#             "total_inference_time_ms": handle_infer_total_overhead * 1000,
#             "preprocess_time_ms": handle_infer_total_total_preprocess_overhead * 1000,
#             "encoding_time_ms": handle_infer_total_total_encoding_overhead * 1000,
#             "inference_time_ms": handle_infer_total_total_infer_from_tensor_overhead * 1000,
#         }
        
#         # Enhanced logging with match/mismatch status
#         if original_pod_choice:
#             match_status = "✅ MATCH" if prediction_matches else "❌ MISMATCH"
#             logger.info(f"Inference result: predicted={selected_pod}, original={original_pod_choice}, {match_status}, confidence={confidence:.4f}")
#         else:
#             logger.info(f"Inference result: predicted={selected_pod}, original=UNKNOWN, confidence={confidence:.4f}")
        
#         return result_summary
        
#     except AssertionError as e:
#         logger.error(f"AssertionError in preprocessing - this suggests the log message format is unexpected")
#         logger.error(f"Log message being processed: {log_message}")
#         logger.error(f"Error: {str(e)}")
#         return None
        
#     except Exception as e:
#         import traceback
#         error_traceback = traceback.format_exc()
#         logger.error(f"Error in test_inference: {str(e)}")
#         logger.error(f"Traceback: {error_traceback}")
#         logger.error(f"Log message being processed: {log_message}")
#         return None


def test_inference(args, log_message):
    """Test inference on a single log message with original vs predicted comparison"""
    global NUM_TRAINS, MODEL_UPDATED
    
    if NUM_TRAINS == 0:
        logger.warning("No trained model available, please train first")
        return None
        
    handle_infer_start_time = time.time()
    
    try:
        logger.debug(f"Testing inference on log message: {log_message[:200]}...")
        
        # Preprocess the log message
        preprocess_start_time = time.time()
        
        # Debug: show what we're passing to preprocess
        logger.debug(f"Calling preprocess.main(None, log_message, {args.ttft_slo}, {args.avg_tpot_slo})")
        
        processed_df, _, all_pods, preprocess_dataset_overhead_summary = preprocess.main(None, log_message, args.ttft_slo, args.avg_tpot_slo)
        logger.debug(f"Successfully parsed data for inference")
        handle_infer_total_total_preprocess_overhead = time.time() - preprocess_start_time

        # EXTRACT ORIGINAL POD CHOICE FROM PREPROCESSED DATA
        original_pod_choice = None
        if 'selected_pod' in processed_df.columns:
            original_pod_choice = processed_df['selected_pod'].iloc[0] if len(processed_df) > 0 else None
        elif hasattr(processed_df, 'original_pod') or 'original_pod' in processed_df.columns:
            original_pod_choice = processed_df['original_pod'].iloc[0] if len(processed_df) > 0 else None
        
        # If not found in DataFrame, try to extract from the raw log message
        if original_pod_choice is None:
            import re
            pattern = r'@selectedpod@([^@]+)@'
            match = re.search(pattern, log_message)
            if match:
                original_pod_choice = match.group(1)
        
        logger.debug(f"Original pod choice: {original_pod_choice}")

        # ===== UPDATED NORMALIZATION LOGIC (MATCHES TRAINING) =====
        
        # Apply SAME pod-centric normalization as training
        request_features = ['input_tokens', 'output_tokens', 'total_tokens']
        pod_features_cols = [col for col in processed_df.columns if col.startswith('pod_') and 
                            processed_df[col].dtype in ['float64', 'int64']]
        
        stats = get_request_stats()
        
        if stats.count > 0:
            logger.debug("Applying pod-centric normalization for inference")
            
            # 1. Request features - only normalize if they were normalized in training
            # (most likely they weren't due to low variance)
            for feature in request_features:
                if feature in processed_df.columns and feature in stats.feature_stats:
                    # This feature was normalized in training, so normalize for inference
                    feature_data = processed_df[feature].values.reshape(-1, 1)
                    normalized_feature = stats.feature_stats[feature].normalize(feature_data)
                    processed_df[feature] = normalized_feature.flatten()
                    logger.debug(f"Normalized request feature {feature} for inference")
                else:
                    # This feature was NOT normalized in training, keep raw values
                    logger.debug(f"Kept raw values for request feature {feature}")
            
            # 2. Pod features - normalize those that were normalized in training
            pod_normalized_count = 0
            for feature in pod_features_cols:
                if 'kv_hit_ratio' in feature:
                    continue  # Skip normalization
                if feature in processed_df.columns and feature in stats.feature_stats:
                    feature_data = processed_df[feature].values.reshape(-1, 1)
                    normalized_feature = stats.feature_stats[feature].normalize(feature_data)
                    processed_df[feature] = normalized_feature.flatten()
                    pod_normalized_count += 1
                    logger.debug(f"Normalized pod feature {feature} for inference")
            
            # 3. Apply same critical feature amplification as training
            critical_features = ['running_requests', 'waiting_requests', 'decode_tokens', 'prefill_tokens']
            amplified_count = 0
            for feature in pod_features_cols:
                if any(critical in feature for critical in critical_features):
                    if feature in processed_df.columns:
                        processed_df[feature] = processed_df[feature] * signal_amplification_degree
                        amplified_count += 1
                        logger.debug(f"Amplified critical feature {feature} for inference")
            
            logger.debug(f"Applied pod-centric normalization: {pod_normalized_count} pod features normalized, {amplified_count} amplified")
            
        else:
            logger.warning(f"No normalization stats available for inference")
        
        # ===== END UPDATED NORMALIZATION =====
        
        # Encode data
        encode_start_time = time.time()
        tensor_dataset, encode_for_inference_overhead_summary = encoding.encode_for_inference(all_pods, processed_df, stats, request_features_train, request_features_reward)
        logger.debug(f"Successfully encoded data in memory for inference")
        handle_infer_total_total_encoding_overhead = time.time() - encode_start_time

        # Perform inference
        infer_from_tensor_start_time = time.time()
        if args.model == "random_forest":
            result, infer_from_tensor_overhead_summary = random_forest.infer_from_tensor(
                tensor_data=tensor_dataset, 
                exploration_enabled=True, 
                exploration_rate=0.2, 
                model_updated=MODEL_UPDATED
        )
        elif args.model == "simpler_contextual_bandit":
            result, infer_from_tensor_overhead_summary = simpler_contextual_bandit.infer_from_tensor(
                tensor_data=tensor_dataset, 
                model_updated=MODEL_UPDATED
            )
        if MODEL_UPDATED:
            logger.info("Model updated flag consumed, resetting to False")
            MODEL_UPDATED = False
        handle_infer_total_total_infer_from_tensor_overhead = time.time() - infer_from_tensor_start_time

        # Map the pod index back to the actual pod ID
        selected_pod_index = result.get('selected_pod_index', 0)
        if selected_pod_index >= len(all_pods):
            logger.warning(f"Selected pod index {selected_pod_index} out of range, defaulting to first pod")
            selected_pod_index = 0
            
        selected_pod = all_pods[selected_pod_index]
        confidence = result['confidence']
        handle_infer_total_overhead = time.time() - handle_infer_start_time
        
        # Compare with original choice
        prediction_matches = (selected_pod == original_pod_choice) if original_pod_choice else None
        
        # Return enhanced result with comparison
        result_summary = {
            "selected_pod": selected_pod,
            "original_pod_choice": original_pod_choice,
            "prediction_matches": prediction_matches,
            "confidence": confidence,
            "total_inference_time_ms": handle_infer_total_overhead * 1000,
            "preprocess_time_ms": handle_infer_total_total_preprocess_overhead * 1000,
            "encoding_time_ms": handle_infer_total_total_encoding_overhead * 1000,
            "inference_time_ms": handle_infer_total_total_infer_from_tensor_overhead * 1000,
        }
        
        # Enhanced logging with match/mismatch status
        if original_pod_choice:
            match_status = "✅ MATCH" if prediction_matches else "❌ MISMATCH"
            logger.info(f"Inference result: predicted={selected_pod}, original={original_pod_choice}, {match_status}, confidence={confidence:.4f}")
        else:
            logger.info(f"Inference result: predicted={selected_pod}, original=UNKNOWN, confidence={confidence:.4f}")
        
        return result_summary
        
    except AssertionError as e:
        logger.error(f"AssertionError in preprocessing - this suggests the log message format is unexpected")
        logger.error(f"Log message being processed: {log_message}")
        logger.error(f"Error: {str(e)}")
        return None
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Error in test_inference: {str(e)}")
        logger.error(f"Traceback: {error_traceback}")
        logger.error(f"Log message being processed: {log_message}")
        return None

# SOLUTION: Modify process_training_data() to focus on pod feature learning
def process_training_data(args, log_data):
    """Process training data with pod-centric feature handling"""
    global ENCODED_DATA_DIR, NUM_TRAINS, MODEL_UPDATED, TRAINING_DATA_UPDATED, TOTAL_NUM_DATA
    
    flush_start_time = time.time()
    
    try:
        logger.info(f"Processing training data with {len(log_data)} entries")
        
        # Create raw data file
        if not os.path.exists("temp_training_data"):
            os.mkdir("temp_training_data")
        raw_data = "temp_training_data/offline_batch.csv"
        
        # Write raw data to file
        ts_write_raw_data = time.time()
        write_to_file(log_data, raw_data)
        logger.info(f"Wrote {len(log_data)} entries to {raw_data}, took {time.time() - ts_write_raw_data} seconds")

        # Preprocess raw data
        ts_preprocess = time.time()
        df, _, all_pods, _ = preprocess.main(raw_data, "", args.ttft_slo, args.avg_tpot_slo)
        logger.info(f"Successfully parsed data, took {time.time() - ts_preprocess} seconds")
        
        # ===== POD-CENTRIC FEATURE ENGINEERING =====
        request_features = ['input_tokens', 'output_tokens', 'total_tokens']
        pod_features_cols = [col for col in df.columns if col.startswith('pod_') and df[col].dtype in ['float64', 'int64']]
        logger.debug(f"Print features. Request features: {request_features}, Pod features: {pod_features_cols}")
        
        logger.info("🔧 POD-CENTRIC FEATURE PROCESSING")
        logger.info("=" * 50)
        
        # Analyze raw feature ranges
        logger.info("Raw feature analysis:")
        high_variance_pod_features = []
        
        for feature in pod_features_cols:
            if feature in df.columns:
                values = df[feature].values
                std_val = values.std()
                logger.info(f"  {feature}: std={std_val:.3f}, range=[{values.min():.2f}, {values.max():.2f}]")
                
                # Only normalize features with reasonable variance
                if std_val > 0.1:  # Threshold for meaningful variance
                    high_variance_pod_features.append(feature)
                else:
                    logger.warning(f"    ⚠️  Low variance - will skip normalization")
        
        logger.info(f"High variance pod features: {len(high_variance_pod_features)}")
        
        # For request features - use raw values if variance is low
        logger.info("\nRequest feature handling:")
        for feature in request_features:
            if feature in df.columns:
                values = df[feature].values
                std_val = values.std()
                logger.info(f"  {feature}: std={std_val:.3f}")
                if std_val < 10:  # Very low variance threshold
                    logger.info(f"    → Using RAW values (no normalization)")
                else:
                    logger.info(f"    → Will normalize")
        
        # ===== SELECTIVE NORMALIZATION STRATEGY =====
        stats = get_request_stats()
        
        # 1. DON'T normalize request features with low variance
        request_normalized_count = 0
        for feature in request_features:
            if feature in df.columns:
                values = df[feature].values
                if values.std() > 10:  # Only normalize if reasonable variance
                    feature_data = values.reshape(-1, 1)
                    if feature not in stats.feature_stats:
                        stats.feature_stats[feature] = RunningStats()
                    stats.feature_stats[feature].update(feature_data)
                    normalized_feature = stats.feature_stats[feature].normalize(feature_data)
                    df[feature] = normalized_feature.flatten()
                    request_normalized_count += 1
                    logger.info(f"✅ Normalized request feature: {feature}")
                else:
                    logger.info(f"⚪ Kept raw values for: {feature}")
        
        # 2. CAREFULLY normalize high-variance pod features
        pod_normalized_count = 0
        for feature in high_variance_pod_features:
            if feature in df.columns:
                if 'kv_hit_ratio' in feature:
                    logger.info(f"⚪ Skipping normalization for {feature} (already 0-100 scale)")
                    continue
                feature_data = df[feature].values.reshape(-1, 1)
                
                # Create fresh stats for this feature to avoid over-normalization
                if feature not in stats.feature_stats:
                    stats.feature_stats[feature] = RunningStats()
                
                # Update and normalize
                stats.feature_stats[feature].update(feature_data)
                normalized_feature = stats.feature_stats[feature].normalize(feature_data)
                
                # Verify normalization didn't destroy variance
                original_std = df[feature].std()
                df[feature] = normalized_feature.flatten()
                new_std = df[feature].std()
                
                if new_std > 0.5:  # Ensure reasonable post-normalization variance
                    pod_normalized_count += 1
                    logger.info(f"✅ Normalized pod feature: {feature} (std: {original_std:.3f} → {new_std:.3f})")
                else:
                    logger.warning(f"⚠️  Post-normalization variance too low for {feature}")
        
        # 3. FEATURE IMPORTANCE AMPLIFICATION
        # For pod features that are critical for routing, apply gentle scaling
        critical_features = ['running_requests', 'waiting_requests', 'decode_tokens', 'prefill_tokens']
        
        for feature in pod_features_cols:
            if any(critical in feature for critical in critical_features):
                if feature in df.columns:
                    # Apply gentle amplification to ensure these features have impact
                    df[feature] = df[feature] * signal_amplification_degree  # 50% amplification
                    logger.info(f"📈 Amplified critical feature: {feature}")
        
        # Save stats
        stats.save(STATS_FILE)
        
        logger.info(f"✅ FEATURE PROCESSING COMPLETE:")
        logger.info(f"  - Request features normalized: {request_normalized_count}/{len(request_features)}")
        logger.info(f"  - Pod features normalized: {pod_normalized_count}/{len(pod_features_cols)}")
        logger.info(f"  - Critical features amplified: {len([f for f in pod_features_cols if any(c in f for c in critical_features)])}")

        # ===== ENHANCED REWARD ENGINEERING =====
        logger.info("\n🎯 REWARD ENGINEERING")
        logger.info("=" * 30)
        
        # Analyze current reward distribution
        if 'reward' in df.columns:
            rewards = df['reward'].values
            logger.info(f"Original rewards: range=[{rewards.min():.3f}, {rewards.max():.3f}], std={rewards.std():.3f}")
            
            # Apply reward shaping to amplify learning signal
            reward_gap = rewards.max() - rewards.min()
            if reward_gap < reward_amplification_threshold:  # If reward differences are small
                logger.info("📈 Applying reward amplification")
                # Amplify reward differences
                reward_mean = rewards.mean()
                df['reward'] = reward_mean + (rewards - reward_mean) * reward_amplification_degree  # amplification
                new_rewards = df['reward'].values
                logger.info(f"Amplified rewards: range=[{new_rewards.min():.3f}, {new_rewards.max():.3f}], std={new_rewards.std():.3f}")
            else:
                logger.info("✅ Reward signal already strong enough")

        # Continue with encoding...
        ts_encode = time.time()
        encoded_data_subdir = f"{ENCODED_DATA_DIR}/batch_1"
        encoding.encode_for_train(all_pods, df, encoded_data_subdir, stats, request_features_train, request_features_reward)
        logger.info(f"Successfully encoded data to {encoded_data_subdir}, took {time.time() - ts_encode} seconds")
        
        # Verify encoded data
        expected_tensor_path = f"{encoded_data_subdir}/tensor_dataset.pt"
        train_tensor_path = f"{encoded_data_subdir}/train/tensor_dataset.pt"
        
        if os.path.exists(expected_tensor_path):
            logger.info(f"✓ Found tensor dataset at: {expected_tensor_path}")
        elif os.path.exists(train_tensor_path):
            logger.info(f"✓ Found tensor dataset at: {train_tensor_path}")
        
        TRAINING_DATA_UPDATED = True
        TOTAL_NUM_DATA += len(log_data)
        
        logger.info(f"Successfully processed {len(log_data)} log messages, took {time.time() - flush_start_time} seconds")
        return True
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Error processing training data: {str(e)}")
        logger.error(f"Traceback: {error_traceback}")
        return False

def analyze_detailed_feature_sensitivity(args, test_data_subset=None):
    """
    Detailed feature-specific sensitivity analysis.
    Tests each type of pod feature individually to understand model behavior.
    """
    global NUM_TRAINS
    
    if NUM_TRAINS == 0:
        logger.warning("No trained model available for detailed feature analysis")
        return None
    
    logger.info("🔬 DETAILED FEATURE-SPECIFIC SENSITIVITY ANALYSIS")
    logger.info("=" * 70)
    
    if test_data_subset is None:
        return None
    
    # Define specific pod feature types to test
    feature_types = {
        'kv_hit_ratio': 'KV Cache Hit Ratio',
        'running_requests': 'Running Requests',
        'waiting_requests': 'Waiting Requests', 
        'decode_tokens': 'Decode Tokens',
        'prefill_tokens': 'Prefill Tokens',
        'inflight_requests': 'Inflight Requests',
        'last_second_avg_ttft_ms': 'Average TTFT',
        'last_second_avg_tpot_ms': 'Average TPOT',
        'last_second_p99_ttft_ms': 'P99 TTFT',
        'last_second_total_requests': 'Total Requests/sec'
    }
    
    feature_sensitivity_results = {}
    
    # Test first 3 samples for detailed analysis
    test_items = list(test_data_subset.items())[:3]
    
    for sample_idx, (request_id, log_message) in enumerate(test_items):
        logger.info(f"\n--- ANALYZING SAMPLE {sample_idx + 1}/3 ({request_id}) ---")
        
        try:
            # Preprocess to get baseline data
            processed_df, _, all_pods, _ = preprocess.main(None, log_message, args.ttft_slo, args.avg_tpot_slo)
            
            # Apply same normalization as training
            request_features = ['input_tokens', 'output_tokens', 'total_tokens']
            pod_features_cols = [col for col in processed_df.columns if col.startswith('pod_') and 
                               processed_df[col].dtype in ['float64', 'int64']]
            
            stats = get_request_stats()
            if stats.count > 0:
                # Apply same normalization as training
                for feature in pod_features_cols:
                    if feature in processed_df.columns and feature in stats.feature_stats:
                        feature_data = processed_df[feature].values.reshape(-1, 1)
                        normalized_feature = stats.feature_stats[feature].normalize(feature_data)
                        processed_df[feature] = normalized_feature.flatten()
                
                # Apply same amplification as training
                critical_features = ['running_requests', 'waiting_requests', 'decode_tokens', 'prefill_tokens']
                for feature in pod_features_cols:
                    if any(critical in feature for critical in critical_features):
                        if feature in processed_df.columns:
                            processed_df[feature] = processed_df[feature] * signal_amplification_degree
            
            # Encode baseline data
            tensor_dataset, _ = encoding.encode_for_inference(all_pods, processed_df, stats, 
                                                            request_features_train, request_features_reward)
            
            # Get baseline prediction
            if args.model == "simpler_contextual_bandit":
                baseline_result, _ = simpler_contextual_bandit.infer_from_tensor(tensor_data=tensor_dataset, model_updated=False)
            else:
                baseline_result, _ = random_forest.infer_from_tensor(tensor_data=tensor_dataset, model_updated=False)
            
            baseline_pod = baseline_result['selected_pod_index']
            baseline_confidence = baseline_result['confidence']
            baseline_probs = baseline_result.get('pod_probabilities', [])
            
            logger.info(f"Baseline prediction: Pod {baseline_pod} (confidence: {baseline_confidence:.3f})")
            
            # Test each feature type individually
            for feature_key, feature_name in feature_types.items():
                logger.info(f"\n🧪 TESTING {feature_name.upper()} SENSITIVITY")
                logger.info("-" * 50)
                
                # Find columns for this feature type across all pods
                feature_cols = [col for col in processed_df.columns if feature_key in col and col.startswith('pod_')]
                
                if not feature_cols:
                    logger.info(f"  No {feature_name} features found")
                    continue
                
                logger.info(f"  Found {len(feature_cols)} {feature_name} features across pods")
                
                # Test different modification levels
                feature_changes = 0
                modification_levels = [-20.0, -10.0, +10.0, +20.0]  # Different intensity levels
                
                for mod_level in modification_levels:
                    # Create modified tensor
                    modified_tensor = {k: v.clone() if hasattr(v, 'clone') else v for k, v in tensor_dataset.items()}
                    
                    # Find the feature in the tensor structure
                    # We need to modify the pod_features_with_staleness tensor
                    
                    # Get pod feature names to find the right index
                    pod_feature_names = ['inflight_requests', 'gpu_kv_cache', 'cpu_kv_cache', 'running_requests', 
                                       'waiting_requests', 'prefill_tokens', 'decode_tokens', 'kv_hit_ratio']
                    
                    if feature_key == 'kv_hit_ratio':
                        # KV hit ratio is in kv_hit_ratios tensor
                        preferred_pod_idx = baseline_pod
                        if preferred_pod_idx < modified_tensor['kv_hit_ratios'].shape[1]:
                            original_value = modified_tensor['kv_hit_ratios'][0, preferred_pod_idx, 0].item()
                            modified_tensor['kv_hit_ratios'][0, preferred_pod_idx, 0] = original_value + mod_level * 0.3
                            
                    else:
                        # Other features are in pod_features_with_staleness
                        feature_idx = None
                        for i, name in enumerate(pod_feature_names):
                            if feature_key in name:
                                feature_idx = i
                                break
                        
                        if feature_idx is not None:
                            preferred_pod_idx = baseline_pod
                            if (preferred_pod_idx < modified_tensor['pod_features_with_staleness'].shape[1] and 
                                feature_idx < modified_tensor['pod_features_with_staleness'].shape[2]):
                                
                                original_value = modified_tensor['pod_features_with_staleness'][0, preferred_pod_idx, feature_idx].item()
                                modified_tensor['pod_features_with_staleness'][0, preferred_pod_idx, feature_idx] = original_value + mod_level
                    
                    # Get modified prediction
                    if args.model == "simpler_contextual_bandit":
                        modified_result, _ = simpler_contextual_bandit.infer_from_tensor(tensor_data=modified_tensor, model_updated=False)
                    else:
                        modified_result, _ = random_forest.infer_from_tensor(tensor_data=modified_tensor, model_updated=False)
                    
                    modified_pod = modified_result['selected_pod_index']
                    modified_confidence = modified_result['confidence']
                    
                    if modified_pod != baseline_pod:
                        feature_changes += 1
                        logger.info(f"    {feature_name} Δ{mod_level:+.1f}: Pod {baseline_pod}→{modified_pod} "
                                   f"(conf: {baseline_confidence:.3f}→{modified_confidence:.3f}) ✓")
                    else:
                        prob_change = abs(modified_confidence - baseline_confidence)
                        logger.info(f"    {feature_name} Δ{mod_level:+.1f}: Pod {baseline_pod} (no change) "
                                   f"(conf: {modified_confidence:.3f}, Δ{prob_change:.3f})")
                
                # Calculate sensitivity for this feature type
                feature_sensitivity = feature_changes / len(modification_levels)
                
                if feature_key not in feature_sensitivity_results:
                    feature_sensitivity_results[feature_key] = []
                feature_sensitivity_results[feature_key].append(feature_sensitivity)
                
                logger.info(f"  {feature_name} sensitivity: {feature_sensitivity:.1%} ({feature_changes}/{len(modification_levels)} tests changed prediction)")
        
        except Exception as e:
            logger.error(f"Error analyzing sample {sample_idx + 1}: {str(e)}")
            continue
    
    # --- DETAILED SENSITIVITY SUMMARY ---
    logger.info(f"\n" + "=" * 70)
    logger.info("🎯 DETAILED FEATURE SENSITIVITY SUMMARY")
    logger.info("=" * 70)
    
    # Calculate average sensitivity for each feature type
    feature_avg_sensitivity = {}
    for feature_key, sensitivities in feature_sensitivity_results.items():
        avg_sens = sum(sensitivities) / len(sensitivities) if sensitivities else 0
        feature_avg_sensitivity[feature_key] = avg_sens
    
    # Sort by sensitivity level
    sorted_features = sorted(feature_avg_sensitivity.items(), key=lambda x: x[1], reverse=True)
    
    logger.info("\nFeature sensitivity ranking (highest to lowest):")
    logger.info("-" * 50)
    
    for i, (feature_key, avg_sensitivity) in enumerate(sorted_features, 1):
        feature_name = feature_types.get(feature_key, feature_key)
        
        if avg_sensitivity > 0.5:
            status = "🔥 HIGH"
        elif avg_sensitivity > 0.25:
            status = "📊 MODERATE"
        elif avg_sensitivity > 0.1:
            status = "⚠️  LOW"
        else:
            status = "❌ MINIMAL"
        
        logger.info(f"{i:2d}. {feature_name:<20} {avg_sensitivity:6.1%} {status}")
    
    # Insights and recommendations
    logger.info(f"\n🔍 KEY INSIGHTS:")
    logger.info("-" * 20)
    
    high_sensitivity_features = [k for k, v in feature_avg_sensitivity.items() if v > 0.5]
    low_sensitivity_features = [k for k, v in feature_avg_sensitivity.items() if v < 0.1]
    
    if high_sensitivity_features:
        feature_names = [feature_types.get(k, k) for k in high_sensitivity_features]
        logger.info(f"✅ Model strongly responds to: {', '.join(feature_names)}")
    
    if low_sensitivity_features:
        feature_names = [feature_types.get(k, k) for k in low_sensitivity_features]
        logger.info(f"⚠️  Model largely ignores: {', '.join(feature_names)}")
    
    # Overall assessment
    overall_pod_sensitivity = sum(feature_avg_sensitivity.values()) / len(feature_avg_sensitivity) if feature_avg_sensitivity else 0
    logger.info(f"\n📊 Overall Pod Feature Sensitivity: {overall_pod_sensitivity:.1%}")
    
    if overall_pod_sensitivity > 0.4:
        logger.info("🎉 EXCELLENT: Model demonstrates strong pod-aware routing!")
    elif overall_pod_sensitivity > 0.25:
        logger.info("✅ GOOD: Model shows meaningful pod state awareness")
    elif overall_pod_sensitivity > 0.15:
        logger.info("📊 MODERATE: Some pod feature learning evident")
    else:
        logger.info("⚠️  LIMITED: Model shows weak pod feature utilization")
    
    logger.info("=" * 70)
    
    return {
        'feature_sensitivity_results': feature_sensitivity_results,
        'feature_avg_sensitivity': feature_avg_sensitivity,
        'sorted_features': sorted_features,
        'overall_pod_sensitivity': overall_pod_sensitivity
    }

def analyze_model_behavior(args, test_data_subset=None):
    """
    Analyze what the model has actually learned by systematically modifying features
    and observing prediction changes. This reveals if the model is truly contextual.
    """
    global NUM_TRAINS
    
    if NUM_TRAINS == 0:
        logger.warning("No trained model available for behavior analysis")
        return None
    
    logger.info("🔬 ANALYZING MODEL BEHAVIOR - What has the model learned?")
    logger.info("=" * 70)
    
    # Get a few test samples for analysis
    if test_data_subset is None:
        return None
    
    analysis_results = {
        'cache_sensitivity': [],
        'request_size_sensitivity': [],
        'pod_feature_sensitivity': [],
        'summary': {}
    }
    
    # Take first 5 test samples for detailed analysis
    test_items = list(test_data_subset.items())[:5]
    
    for sample_idx, (request_id, log_message) in enumerate(test_items):
        logger.info(f"\n--- ANALYZING SAMPLE {sample_idx + 1}/5 ({request_id}) ---")
        
        try:
            # Preprocess to get baseline data
            processed_df, _, all_pods, _ = preprocess.main(None, log_message, args.ttft_slo, args.avg_tpot_slo)
            
            # Apply normalization (same as training)
            request_features = ['input_tokens', 'output_tokens', 'total_tokens']
            pod_features_cols = [col for col in processed_df.columns if col.startswith('pod_') and 
                               processed_df[col].dtype in ['float64', 'int64']]
            all_features = request_features + pod_features_cols
            stats = get_request_stats()
            
            if stats.count > 0:
                for feature in all_features:
                    if feature in processed_df.columns and feature in stats.feature_stats:
                        feature_data = processed_df[feature].values.reshape(-1, 1)
                        normalized_feature = stats.feature_stats[feature].normalize(feature_data)
                        processed_df[feature] = normalized_feature.flatten()
            
            # Encode baseline data
            tensor_dataset, _ = encoding.encode_for_inference(all_pods, processed_df, stats, 
                                                            request_features_train, request_features_reward)
            
            # Get baseline prediction
            if args.model == "simpler_contextual_bandit":
                baseline_result, _ = simpler_contextual_bandit.infer_from_tensor(tensor_data=tensor_dataset, model_updated=False)
            else:
                baseline_result, _ = random_forest.infer_from_tensor(tensor_data=tensor_dataset, model_updated=False)
            
            baseline_pod = baseline_result['selected_pod_index']
            baseline_confidence = baseline_result['confidence']
            baseline_probs = baseline_result.get('pod_probabilities', [])
            
            logger.info(f"Baseline prediction: Pod {baseline_pod} (confidence: {baseline_confidence:.3f})")
            logger.info(f"Baseline probabilities: {[f'{p:.3f}' for p in baseline_probs]}")
            
            # --- TEST 1: CACHE SENSITIVITY ---
            logger.info(f"\n🧪 TEST 1: Cache Hit Ratio Sensitivity")
            logger.info("-" * 40)
            
            cache_changes = 0
            for cache_delta in [-0.6, -0.3, +0.3, +0.6]:  # Try different cache changes
                # Create modified tensor with cache changes
                modified_tensor = {k: v.clone() if hasattr(v, 'clone') else v for k, v in tensor_dataset.items()}
                
                # Modify cache ratios: lower preferred pod, raise alternative pod
                preferred_pod_idx = baseline_pod
                alternative_pod_idx = (baseline_pod + 1) % len(all_pods)
                
                # Apply cache modifications
                original_preferred_cache = modified_tensor['kv_hit_ratios'][0, preferred_pod_idx, 0].item()
                original_alternative_cache = modified_tensor['kv_hit_ratios'][0, alternative_pod_idx, 0].item()
                
                modified_tensor['kv_hit_ratios'][0, preferred_pod_idx, 0] = max(0.0, original_preferred_cache + cache_delta)
                modified_tensor['kv_hit_ratios'][0, alternative_pod_idx, 0] = min(1.0, original_alternative_cache - cache_delta)
                
                # Get modified prediction
                if args.model == "simpler_contextual_bandit":
                    modified_result, _ = simpler_contextual_bandit.infer_from_tensor(tensor_data=modified_tensor, model_updated=False)
                else:
                    modified_result, _ = random_forest.infer_from_tensor(tensor_data=modified_tensor, model_updated=False)
                
                modified_pod = modified_result['selected_pod_index']
                modified_confidence = modified_result['confidence']
                
                if modified_pod != baseline_pod:
                    cache_changes += 1
                    logger.info(f"  Cache Δ{cache_delta:+.1f}: Pod {baseline_pod}→{modified_pod} (conf: {baseline_confidence:.3f}→{modified_confidence:.3f}) ✓")
                else:
                    logger.info(f"  Cache Δ{cache_delta:+.1f}: Pod {baseline_pod} (no change) (conf: {modified_confidence:.3f})")
            
            cache_sensitivity = cache_changes / 4.0  # 4 tests
            analysis_results['cache_sensitivity'].append(cache_sensitivity)
            logger.info(f"Cache sensitivity: {cache_sensitivity:.1%} ({cache_changes}/4 tests changed prediction)")
            
            # --- TEST 2: REQUEST SIZE SENSITIVITY ---
            logger.info(f"\n🧪 TEST 2: Request Size Sensitivity") 
            logger.info("-" * 40)
            
            size_changes = 0
            original_input_tokens = tensor_dataset['request_features'][0, 0].item() if tensor_dataset['request_features'].shape[1] > 0 else 0
            
            for size_multiplier in [0.3, 0.6, 1.5, 3.0]:  # Different request sizes
                modified_tensor = {k: v.clone() if hasattr(v, 'clone') else v for k, v in tensor_dataset.items()}
                
                # Modify request size (first feature assumed to be input tokens)
                if modified_tensor['request_features'].shape[1] > 0:
                    modified_tensor['request_features'][0, 0] = original_input_tokens * size_multiplier
                
                # Get modified prediction
                if args.model == "simpler_contextual_bandit":
                    modified_result, _ = simpler_contextual_bandit.infer_from_tensor(tensor_data=modified_tensor, model_updated=False)
                else:
                    modified_result, _ = random_forest.infer_from_tensor(tensor_data=modified_tensor, model_updated=False)
                
                modified_pod = modified_result['selected_pod_index']
                modified_confidence = modified_result['confidence']
                
                if modified_pod != baseline_pod:
                    size_changes += 1
                    logger.info(f"  Size ×{size_multiplier}: Pod {baseline_pod}→{modified_pod} (conf: {baseline_confidence:.3f}→{modified_confidence:.3f}) ✓")
                else:
                    logger.info(f"  Size ×{size_multiplier}: Pod {baseline_pod} (no change) (conf: {modified_confidence:.3f})")
            
            size_sensitivity = size_changes / 4.0
            analysis_results['request_size_sensitivity'].append(size_sensitivity)
            logger.info(f"Request size sensitivity: {size_sensitivity:.1%} ({size_changes}/4 tests changed prediction)")
            
            # --- TEST 3: POD FEATURE SENSITIVITY ---
            logger.info(f"\n🧪 TEST 3: Pod Feature Sensitivity")
            logger.info("-" * 40)
            
            pod_feature_changes = 0
            pod_features_tested = 0
            
            # Test modifying individual pod features
            for feature_idx in range(min(3, tensor_dataset['pod_features_with_staleness'].shape[2])):  # Test first 3 pod features
                for delta in [-1.0, +1.0]:  # Try increasing/decreasing each feature
                    modified_tensor = {k: v.clone() if hasattr(v, 'clone') else v for k, v in tensor_dataset.items()}
                    
                    # Modify specific pod feature for preferred pod
                    preferred_pod_idx = baseline_pod
                    original_value = modified_tensor['pod_features_with_staleness'][0, preferred_pod_idx, feature_idx].item()
                    modified_tensor['pod_features_with_staleness'][0, preferred_pod_idx, feature_idx] = original_value + delta
                    
                    # Get modified prediction
                    if args.model == "simpler_contextual_bandit":
                        modified_result, _ = simpler_contextual_bandit.infer_from_tensor(tensor_data=modified_tensor, model_updated=False)
                    else:
                        modified_result, _ = random_forest.infer_from_tensor(tensor_data=modified_tensor, model_updated=False)
                    
                    modified_pod = modified_result['selected_pod_index']
                    modified_confidence = modified_result['confidence']
                    
                    pod_features_tested += 1
                    if modified_pod != baseline_pod:
                        pod_feature_changes += 1
                        logger.info(f"  Feature[{feature_idx}] Δ{delta:+.1f}: Pod {baseline_pod}→{modified_pod} (conf: {baseline_confidence:.3f}→{modified_confidence:.3f}) ✓")
                    else:
                        logger.info(f"  Feature[{feature_idx}] Δ{delta:+.1f}: Pod {baseline_pod} (no change) (conf: {modified_confidence:.3f})")
            
            pod_sensitivity = pod_feature_changes / max(1, pod_features_tested)
            analysis_results['pod_feature_sensitivity'].append(pod_sensitivity)
            logger.info(f"Pod feature sensitivity: {pod_sensitivity:.1%} ({pod_feature_changes}/{pod_features_tested} tests changed prediction)")
            
        except Exception as e:
            logger.error(f"Error analyzing sample {sample_idx + 1}: {str(e)}")
            continue
    
    # --- SUMMARY ANALYSIS ---
    logger.info(f"\n" + "=" * 70)
    logger.info("🎯 BEHAVIOR ANALYSIS SUMMARY")
    logger.info("=" * 70)
    
    if analysis_results['cache_sensitivity']:
        avg_cache_sensitivity = sum(analysis_results['cache_sensitivity']) / len(analysis_results['cache_sensitivity'])
        logger.info(f"Average Cache Sensitivity: {avg_cache_sensitivity:.1%}")
        
        if avg_cache_sensitivity > 0.5:
            logger.info("✅ Model strongly considers cache hit ratios")
        elif avg_cache_sensitivity > 0.25:
            logger.info("📊 Model moderately considers cache hit ratios")
        else:
            logger.info("❌ Model largely ignores cache hit ratios")
    
    if analysis_results['request_size_sensitivity']:
        avg_size_sensitivity = sum(analysis_results['request_size_sensitivity']) / len(analysis_results['request_size_sensitivity'])
        logger.info(f"Average Request Size Sensitivity: {avg_size_sensitivity:.1%}")
        
        if avg_size_sensitivity > 0.5:
            logger.info("✅ Model strongly adapts to request size")
        elif avg_size_sensitivity > 0.25:
            logger.info("📊 Model moderately adapts to request size")
        else:
            logger.info("❌ Model largely ignores request size")
    
    if analysis_results['pod_feature_sensitivity']:
        avg_pod_sensitivity = sum(analysis_results['pod_feature_sensitivity']) / len(analysis_results['pod_feature_sensitivity'])
        logger.info(f"Average Pod Feature Sensitivity: {avg_pod_sensitivity:.1%}")
        
        if avg_pod_sensitivity > 0.5:
            logger.info("✅ Model strongly considers pod characteristics")
        elif avg_pod_sensitivity > 0.25:
            logger.info("📊 Model moderately considers pod characteristics")
        else:
            logger.info("❌ Model largely ignores pod characteristics")
    
    # Overall contextual learning assessment
    sensitivities = []
    if analysis_results['cache_sensitivity']:
        sensitivities.append(sum(analysis_results['cache_sensitivity']) / len(analysis_results['cache_sensitivity']))
    if analysis_results['request_size_sensitivity']:
        sensitivities.append(sum(analysis_results['request_size_sensitivity']) / len(analysis_results['request_size_sensitivity']))
    if analysis_results['pod_feature_sensitivity']:
        sensitivities.append(sum(analysis_results['pod_feature_sensitivity']) / len(analysis_results['pod_feature_sensitivity']))
    
    if sensitivities:
        overall_contextual_score = sum(sensitivities) / len(sensitivities)
        logger.info(f"\nOverall Contextual Learning Score: {overall_contextual_score:.1%}")
        
        if overall_contextual_score > 0.6:
            logger.info("🎉 EXCELLENT: Model demonstrates strong contextual learning!")
        elif overall_contextual_score > 0.4:
            logger.info("✅ GOOD: Model shows contextual behavior")
        elif overall_contextual_score > 0.2:
            logger.info("⚠️  MODERATE: Some contextual learning, but room for improvement")
        else:
            logger.info("❌ POOR: Model appears to learn static preferences, not contextual routing")
        
        # Store summary
        analysis_results['summary'] = {
            'overall_score': overall_contextual_score,
            'avg_cache_sensitivity': sum(analysis_results['cache_sensitivity']) / len(analysis_results['cache_sensitivity']) if analysis_results['cache_sensitivity'] else 0,
            'avg_size_sensitivity': sum(analysis_results['request_size_sensitivity']) / len(analysis_results['request_size_sensitivity']) if analysis_results['request_size_sensitivity'] else 0,
            'avg_pod_sensitivity': sum(analysis_results['pod_feature_sensitivity']) / len(analysis_results['pod_feature_sensitivity']) if analysis_results['pod_feature_sensitivity'] else 0
        }
    
    logger.info("=" * 70)
    return analysis_results

def diagnose_training_data_issues(args, train_data_sample):
    """
    Diagnose why the model is learning static preferences instead of contextual routing.
    Call this right after process_training_data() in main().
    """
    logger.info("🔬 DIAGNOSING TRAINING DATA ISSUES")
    logger.info("=" * 60)
    
    # Check encoded data
    encoded_data_subdir = f"{ENCODED_DATA_DIR}/batch_1"
    tensor_path = f"{encoded_data_subdir}/tensor_dataset.pt"
    train_tensor_path = f"{encoded_data_subdir}/train/tensor_dataset.pt"
    
    if os.path.exists(tensor_path):
        tensor_data = torch.load(tensor_path)
    elif os.path.exists(train_tensor_path):
        tensor_data = torch.load(train_tensor_path)
    else:
        logger.error("No tensor data found for diagnosis")
        return
    
    # 1. ACTION DISTRIBUTION ANALYSIS
    logger.info("\n1️⃣ ACTION DISTRIBUTION ANALYSIS:")
    logger.info("-" * 40)
    
    actions = tensor_data['actions']
    action_counts = torch.bincount(actions, minlength=7)
    total_samples = len(actions)
    
    logger.info(f"Total training samples: {total_samples}")
    for pod_id in range(7):
        count = action_counts[pod_id].item()
        percentage = count / total_samples * 100
        logger.info(f"Pod {pod_id}: {count} samples ({percentage:.1f}%)")
    
    # Check for severe imbalance
    max_count = action_counts.max().item()
    min_count = action_counts[action_counts > 0].min().item() if (action_counts > 0).sum() > 0 else 1
    imbalance_ratio = max_count / min_count
    
    logger.info(f"Imbalance ratio: {imbalance_ratio:.1f}x")
    if imbalance_ratio > 5:
        logger.warning(f"⚠️  SEVERE IMBALANCE: Pod {torch.argmax(action_counts).item()} dominates training data!")
    
    # 2. REWARD SIGNAL ANALYSIS
    logger.info("\n2️⃣ REWARD SIGNAL ANALYSIS:")
    logger.info("-" * 40)
    
    rewards = tensor_data['rewards']
    logger.info(f"Reward range: [{rewards.min().item():.4f}, {rewards.max().item():.4f}]")
    logger.info(f"Reward std: {rewards.std().item():.4f}")
    
    # Reward by action
    reward_by_pod = {}
    for pod_id in range(7):
        pod_mask = actions == pod_id
        if pod_mask.sum() > 0:
            pod_rewards = rewards[pod_mask]
            reward_by_pod[pod_id] = {
                'mean': pod_rewards.mean().item(),
                'std': pod_rewards.std().item(),
                'count': pod_mask.sum().item()
            }
            logger.info(f"Pod {pod_id}: μ={reward_by_pod[pod_id]['mean']:.4f}, "
                       f"σ={reward_by_pod[pod_id]['std']:.4f}, n={reward_by_pod[pod_id]['count']}")
    
    # Check reward differentiation
    if len(reward_by_pod) > 1:
        pod_means = [stats['mean'] for stats in reward_by_pod.values()]
        reward_gap = max(pod_means) - min(pod_means)
        logger.info(f"Reward gap between best/worst pods: {reward_gap:.4f}")
        
        if reward_gap < 0.01:
            logger.warning("⚠️  VERY WEAK REWARD SIGNAL: Pods have nearly identical rewards!")
        elif reward_gap < 0.05:
            logger.warning("⚠️  WEAK REWARD SIGNAL: Small differences between pods")
    
    # 3. FEATURE VARIANCE ANALYSIS
    logger.info("\n3️⃣ FEATURE VARIANCE ANALYSIS:")
    logger.info("-" * 40)
    
    # Pod features variance
    pod_features = tensor_data['pod_features_with_staleness']
    logger.info(f"Pod features shape: {pod_features.shape}")
    
    # Calculate variance across samples for each feature
    pod_feature_vars = pod_features.var(dim=0).mean(dim=0)  # Average variance across pods
    logger.info("Pod feature variances:")
    feature_names = ['inflight_requests', 'gpu_kv_cache', 'cpu_kv_cache', 'running_requests', 
                    'waiting_requests', 'prefill_tokens', 'decode_tokens']
    
    low_variance_features = 0
    for i, var in enumerate(pod_feature_vars):
        feature_name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
        logger.info(f"  {feature_name}: {var.item():.6f}")
        if var.item() < 1e-3:
            low_variance_features += 1
            logger.warning(f"    ⚠️  Very low variance - feature may be static!")
    
    if low_variance_features > len(pod_feature_vars) * 0.5:
        logger.warning(f"⚠️  {low_variance_features}/{len(pod_feature_vars)} pod features have very low variance!")
    
    # KV hit ratios variance
    kv_ratios = tensor_data['kv_hit_ratios']
    kv_var = kv_ratios.var(dim=0).mean()
    logger.info(f"KV hit ratios variance: {kv_var.item():.6f}")
    if kv_var.item() < 1e-3:
        logger.warning("⚠️  KV hit ratios have very low variance!")
    
    # Request features variance
    request_features = tensor_data['request_features']
    request_vars = request_features.var(dim=0)
    logger.info("Request feature variances:")
    request_names = ['input_tokens', 'output_tokens', 'total_tokens']
    
    for i, var in enumerate(request_vars):
        feature_name = request_names[i] if i < len(request_names) else f"request_feature_{i}"
        logger.info(f"  {feature_name}: {var.item():.6f}")
        if var.item() < 1e-3:
            logger.warning(f"    ⚠️  Very low variance - feature may be static!")
    
    # 4. SAMPLE DATA INSPECTION
    logger.info("\n4️⃣ SAMPLE DATA INSPECTION:")
    logger.info("-" * 40)
    
    # Show first few samples to understand data characteristics
    logger.info("First 3 training samples:")
    for i in range(min(3, len(actions))):
        logger.info(f"\nSample {i}:")
        logger.info(f"  Action (selected pod): {actions[i].item()}")
        logger.info(f"  Reward: {rewards[i].item():.4f}")
        logger.info(f"  Request features: {request_features[i].numpy()}")
        logger.info(f"  KV ratios: {kv_ratios[i].numpy().flatten()}")
        logger.info(f"  Pod features (first 3): {pod_features[i, :, :3].numpy()}")
    
    # 5. RECOMMENDATIONS
    logger.info("\n5️⃣ RECOMMENDATIONS:")
    logger.info("-" * 40)
    
    recommendations = []
    
    if imbalance_ratio > 5:
        recommendations.append("🔴 CRITICAL: Balance training data - consider data augmentation or stratified sampling")
    
    if reward_gap < 0.01:
        recommendations.append("🔴 CRITICAL: Amplify reward differences or use different reward calculation")
    
    if low_variance_features > 3:
        recommendations.append("🟡 Add more dynamic pod state features (current load, temperature, etc.)")
    
    if kv_var.item() < 1e-3:
        recommendations.append("🟡 KV hit ratios may be too static - ensure they vary meaningfully")
    
    # Check if request features vary
    request_var_count = (request_vars < 1e-3).sum().item()
    if request_var_count > 0:
        recommendations.append("🟡 Some request features are static - add more request diversity")
    
    if not recommendations:
        recommendations.append("✅ No obvious data issues detected")
    
    logger.info("Action items:")
    for rec in recommendations:
        logger.info(f"  {rec}")
    
    logger.info("\n" + "=" * 60)
    
    return {
        'action_distribution': action_counts,
        'imbalance_ratio': imbalance_ratio,
        'reward_gap': reward_gap if 'reward_gap' in locals() else 0,
        'low_variance_features': low_variance_features,
        'recommendations': recommendations
    }


def main():
    parser = argparse.ArgumentParser(description='Offline Routing Agent Training and Testing')
    parser.add_argument('data_file', help='CSV file containing log messages for training')
    parser.add_argument('--test_file', help='Optional CSV file containing log messages for testing')
    parser.add_argument('--test_single', help='Single log message string for testing')
    parser.add_argument('--skip_training', action='store_true', help='Skip training and only do inference')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='Train/test split ratio (default: 0.8 for 80%% train, 20%% test)')
    parser.add_argument('--auto_split', action='store_true', help='Automatically split data_file into train/test')
    parser.add_argument('--model', choices=['random_forest', 'simpler_contextual_bandit'], default='random_forest', help='Model type to use for training (default: random_forest)')
    parser.add_argument('--ttft_slo', type=float, help='TTFT SLO threshold for preprocessing', default=500)
    parser.add_argument('--avg_tpot_slo', type=float, help='Average TPOT SLO threshold for preprocessing', default=40)
    parser.add_argument('--analyze_behavior', action='store_true', help='Analyze what the model has learned through feature sensitivity tests')

    args = parser.parse_args()
    
    # Check if data file exists
    if not os.path.exists(args.data_file):
        logger.error(f"Data file {args.data_file} not found")
        return
    
    # Check if test file exists (if specified)
    if args.test_file and not os.path.exists(args.test_file):
        logger.error(f"Test file {args.test_file} not found")
        return
    
    # Create necessary directories
    if not os.path.exists(ENCODED_DATA_DIR):
        os.makedirs(ENCODED_DATA_DIR)
    
    # Clean up any existing encoded data for fresh start
    if os.path.exists(ENCODED_DATA_DIR):
        import shutil
        shutil.rmtree(ENCODED_DATA_DIR)
        os.makedirs(ENCODED_DATA_DIR)
        logger.info(f"Cleaned and recreated {ENCODED_DATA_DIR} for fresh offline training")

    # Handle data splitting
    if args.auto_split or not args.test_file:
        logger.info("=== SPLITTING DATA ===")
        all_data = read_csv_data(args.data_file)
        if all_data is None or len(all_data) == 0:
            logger.error("Failed to read data or no valid log messages found")
            return
        
        # Split data
        all_messages = list(all_data.values())
        split_point = int(len(all_messages) * args.split_ratio)
        
        train_messages = all_messages[:split_point]
        test_messages = all_messages[split_point:]
        
        # Convert back to dict format
        train_data = {f"request_{i}": msg for i, msg in enumerate(train_messages)}
        test_data = {f"request_{i}": msg for i, msg in enumerate(test_messages)}
        
        logger.info(f"Split {len(all_data)} messages into {len(train_data)} train + {len(test_data)} test")
        
    else:
        # Use separate files
        train_data = read_csv_data(args.data_file)
        test_data = read_csv_data(args.test_file) if args.test_file else None
        
        if train_data is None or len(train_data) == 0:
            logger.error("Failed to read training data or no valid log messages found")
            return
    
    # Read and process training data
    if not args.skip_training:
        logger.info("=== STARTING TRAINING PHASE ===")
        
        # Process training data
        if not process_training_data(args, train_data):
            logger.error("Failed to process training data")
            return
        
        # ADD THIS:
        diagnose_training_data_issues(args, train_data)

        # Train model
        if not train_model(args):
            logger.error("Failed to train model")
            return
        
        logger.info("=== TRAINING COMPLETED ===")
    else:
        logger.info("=== SKIPPING TRAINING (using existing model) ===")

    # NEW: Behavior Analysis (before regular testing)
    if args.analyze_behavior and test_data and len(test_data) > 0:
        logger.info("=== STARTING BEHAVIOR ANALYSIS ===")
        analyze_model_behavior(args, test_data)
        analyze_detailed_feature_sensitivity(args, test_data)
        logger.info("=== BEHAVIOR ANALYSIS COMPLETED ===")
    
    
    # Test inference
    if test_data and len(test_data) > 0:
        logger.info("=== STARTING TESTING PHASE ===")
        
        # Test on multiple examples with match tracking
        success_count = 0
        match_count = 0
        mismatch_count = 0
        unknown_original_count = 0
        total_count = min(len(test_data), 10)  # Limit to 100 tests for faster iteration
        
        logger.info(f"Testing on {total_count} samples (out of {len(test_data)} available)")
        
        test_items = list(test_data.items())[:total_count]
        for i, (request_id, log_message) in enumerate(test_items):
            logger.info(f"Testing inference {i+1}/{total_count} (request_id: {request_id})")
            result = test_inference(args, log_message)
            if result:
                success_count += 1
                
                # Track prediction accuracy
                if result['prediction_matches'] is True:
                    match_count += 1
                elif result['prediction_matches'] is False:
                    mismatch_count += 1
                else:
                    unknown_original_count += 1
                
                # Show detailed result with comparison
                if result['original_pod_choice']:
                    match_status = "MATCH" if result['prediction_matches'] else "MISMATCH"
                    logger.info(f"  → Predicted: {result['selected_pod']}, Original: {result['original_pod_choice']}, Status: {match_status}, Confidence: {result['confidence']:.3f}")
                else:
                    logger.info(f"  → Predicted: {result['selected_pod']}, Original: UNKNOWN, Confidence: {result['confidence']:.3f}")
            else:
                logger.error(f"✗ Failed inference for {request_id}")
        
        # Print enhanced summary with accuracy metrics
        logger.info("=" * 60)
        logger.info("=== TESTING SUMMARY ===")
        logger.info(f"Total tests: {total_count}")
        logger.info(f"Successful inferences: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
        
        if match_count + mismatch_count > 0:
            accuracy = match_count / (match_count + mismatch_count) * 100
            logger.info(f"Prediction accuracy: {match_count}/{match_count + mismatch_count} ({accuracy:.1f}%)")
            logger.info(f"  - Matches: {match_count}")
            logger.info(f"  - Mismatches: {mismatch_count}")
        
        if unknown_original_count > 0:
            logger.info(f"  - Unknown original: {unknown_original_count}")
        
        logger.info("=" * 60)

if __name__ == "__main__":
    main()