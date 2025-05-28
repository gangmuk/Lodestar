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

# Global variables (simplified for offline use)
ENCODED_DATA_DIR = "encoded_data"
STATS_FILE = "request_feature_stats.pkl"
NUM_TRAINS = 0
MODEL_UPDATED = False
TRAINING_DATA_UPDATED = False
TOTAL_NUM_DATA = 0
MIN_NUM_TRAINING_DATA = 500
LOCK_TRAINING_DATA = threading.Lock()

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

def process_training_data(args, log_data):
    """Process training data - create single batch from CSV input"""
    global ENCODED_DATA_DIR, NUM_TRAINS, MODEL_UPDATED, TRAINING_DATA_UPDATED, TOTAL_NUM_DATA
    
    flush_start_time = time.time()
    
    try:
        logger.info(f"Processing training data with {len(log_data)} entries")
        
        # Create raw data file (simplified - just one batch)
        if not os.path.exists("raw_training_data"):
            os.mkdir("raw_training_data")
        raw_data = "raw_training_data/offline_batch.csv"
        
        # Write raw data to file
        ts_write_raw_data = time.time()
        write_to_file(log_data, raw_data)
        logger.info(f"Wrote {len(log_data)} entries to {raw_data}, took {time.time() - ts_write_raw_data} seconds")

        # Preprocess raw data
        ts_preprocess = time.time()
        df, _, all_pods, _ = preprocess.main(raw_data, "", args.ttft_slo, args.avg_tpot_slo)
        logger.info(f"Successfully parsed data, took {time.time() - ts_preprocess} seconds")
        
        # Re-enable PROPER per-feature normalization 
        request_features = ['input_tokens', 'output_tokens', 'total_tokens']
        pod_features_cols = [col for col in df.columns if col.startswith('pod_') and df[col].dtype in ['float64', 'int64']]
        all_features = request_features + pod_features_cols
        
        logger.info(f"Found {len(all_features)} features to normalize: {len(request_features)} request + {len(pod_features_cols)} pod features")
        
        # Get or create stats object
        stats = get_request_stats()
        
        # Normalize each feature INDIVIDUALLY (not as a group)
        for feature in all_features:
            if feature in df.columns:
                # Extract single feature column as 2D array for consistency
                feature_data = df[feature].values.reshape(-1, 1)
                
                # Update stats for this specific feature
                if feature not in stats.feature_stats:
                    stats.feature_stats[feature] = RunningStats()
                
                stats.feature_stats[feature].update(feature_data)
                
                # Normalize this feature using its own stats
                normalized_feature = stats.feature_stats[feature].normalize(feature_data)
                df[feature] = normalized_feature.flatten()
                
                logger.debug(f"Normalized {feature}: range [{df[feature].min():.3f}, {df[feature].max():.3f}], mean={df[feature].mean():.3f}, std={df[feature].std():.3f}")
        
        # Save stats
        stats.save(STATS_FILE)
        logger.info(f"Applied individual feature normalization to {len(all_features)} features")

        # Encode preprocessed data - create single batch directory
        ts_encode = time.time()
        encoded_data_subdir = f"{ENCODED_DATA_DIR}/batch_1"  # Simple single batch
        encoding.encode_for_train(all_pods, df, encoded_data_subdir, stats, request_features_train, request_features_reward)
        logger.info(f"Successfully encoded data to {encoded_data_subdir}, took {time.time() - ts_encode} seconds")
        
        # Verify the encoded data was created
        expected_tensor_path = f"{encoded_data_subdir}/tensor_dataset.pt"
        train_tensor_path = f"{encoded_data_subdir}/train/tensor_dataset.pt"
        
        if os.path.exists(expected_tensor_path):
            logger.info(f"✓ Found tensor dataset at: {expected_tensor_path}")
        elif os.path.exists(train_tensor_path):
            logger.info(f"✓ Found tensor dataset at: {train_tensor_path}")
        else:
            logger.warning(f"⚠️  Tensor dataset not found at expected locations:")
            logger.warning(f"   - {expected_tensor_path}")
            logger.warning(f"   - {train_tensor_path}")
            # List what was actually created
            if os.path.exists(encoded_data_subdir):
                files = os.listdir(encoded_data_subdir)
                logger.info(f"   Files in {encoded_data_subdir}: {files}")
                
                # Check if there's a train subdirectory
                train_dir = f"{encoded_data_subdir}/train"
                if os.path.exists(train_dir):
                    train_files = os.listdir(train_dir)
                    logger.info(f"   Files in {train_dir}: {train_files}")
        
        TRAINING_DATA_UPDATED = True
        TOTAL_NUM_DATA += len(log_data)
        
        logger.info(f"Successfully processed {len(log_data)} log messages in single batch, took {time.time() - flush_start_time} seconds")
        return True
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Error processing training data: {str(e)}")
        logger.error(f"Traceback: {error_traceback}")
        return False

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

def test_inference(args, log_message):
    """Test inference on a single log message"""
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

        # Apply SAME individual feature normalization as training
        request_features = ['input_tokens', 'output_tokens', 'total_tokens']
        pod_features_cols = [col for col in processed_df.columns if col.startswith('pod_') and 
                            processed_df[col].dtype in ['float64', 'int64']]
        
        all_features = request_features + pod_features_cols
        stats = get_request_stats()
        
        # Normalize each feature individually (same as training)
        if stats.count > 0:
            for feature in all_features:
                if feature in processed_df.columns and feature in stats.feature_stats:
                    feature_data = processed_df[feature].values.reshape(-1, 1)
                    normalized_feature = stats.feature_stats[feature].normalize(feature_data)
                    processed_df[feature] = normalized_feature.flatten()
                    logger.debug(f"Normalized {feature} for inference: range [{processed_df[feature].min():.3f}, {processed_df[feature].max():.3f}]")
            
            logger.debug(f"Applied individual feature normalization to {len(all_features)} features for inference")
        else:
            logger.warning(f"No normalization stats available for inference")
        
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
                # exploration_enabled=EXPLORE_ENABLED, 
                # exploration_rate=EXPLORATION_RATE, 
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
        
        # Return the result
        result_summary = {
            "selected_pod": selected_pod,
            "confidence": confidence,
            "total_inference_time_ms": handle_infer_total_overhead * 1000,
            "preprocess_time_ms": handle_infer_total_total_preprocess_overhead * 1000,
            "encoding_time_ms": handle_infer_total_total_encoding_overhead * 1000,
            "inference_time_ms": handle_infer_total_total_infer_from_tensor_overhead * 1000,
        }
        
        logger.debug(f"Inference result: selected_pod={selected_pod}, confidence={confidence:.4f}")
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
        
        # Train model
        if not train_model(args):
            logger.error("Failed to train model")
            return
        
        logger.info("=== TRAINING COMPLETED ===")
    else:
        logger.info("=== SKIPPING TRAINING (using existing model) ===")
    
    # Test inference
    if test_data and len(test_data) > 0:
        logger.info("=== STARTING TESTING PHASE ===")
        
        # Test on multiple examples
        success_count = 0
        total_count = min(len(test_data), 50)  # Limit to 10 tests for faster iteration
        
        logger.info(f"Testing on {total_count} samples (out of {len(test_data)} available)")
        
        test_items = list(test_data.items())[:total_count]
        for i, (request_id, log_message) in enumerate(test_items):
            logger.info(f"Testing inference {i+1}/{total_count} (request_id: {request_id})")
            result = test_inference(args, log_message)
            if result:
                success_count += 1
                logger.info(f"✓ Success: selected_pod={result['selected_pod']}, confidence={result['confidence']:.3f}")
            else:
                logger.error(f"✗ Failed inference for {request_id}")
        
        logger.info(f"=== TESTING COMPLETED: {success_count}/{total_count} successful ===")
    
    elif args.test_single:
        logger.info("=== TESTING SINGLE INFERENCE ===")
        result = test_inference(args, args.test_single)
        if result:
            logger.info(f"✓ Single inference result: {result}")
        else:
            logger.error("✗ Single inference failed")
    
    elif not args.skip_training:
        logger.info("=== NO TESTING DATA PROVIDED ===")
        logger.info("Use --test-file, --test-single, or --auto-split to enable testing")
    
    logger.info("=== OFFLINE TESTING COMPLETED ===")

if __name__ == "__main__":
    main()