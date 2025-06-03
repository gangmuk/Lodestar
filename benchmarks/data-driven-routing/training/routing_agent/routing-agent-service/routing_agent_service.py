# routing_agent_service.py

# import threading
# import joblib
import pandas as pd
import numpy as np
# import uvicorn
# from pydantic import BaseModel
from typing import Dict, List, Optional, Any, Union
import os
import logging
import time
# import asyncio
# from concurrent.futures import ThreadPoolExecutor
import sys
# import concurrent.futures
import encoding
# import sac
import ppo
import contextual_bandit
import simpler_contextual_bandit
from flask import Flask, request, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
from logger import logger
import preprocess
import pickle
import threading

app = Flask(__name__)

BATCH_ID = 0
ENCODED_DATA_DIR = "encoded_data"
STATS_FILE = "request_feature_stats.pkl"  # Add this near the top with your other constants
NUM_TRAINS = 0
MODEL_UPDATED = False
TRAINING_DATA_UPDATED = False
TOTAL_NUM_DATA = 0
NUM_NEW_DATA = 0
MIN_NUM_TRAINING_DATA = 1000  # Minimum number of training data required to trigger training
LOCK_TRAINING_DATA = threading.Lock()
LOAD_PRETRAINED_MODEL = os.getenv("LOAD_PRETRAINED_MODEL", "true").lower() == "true"
PRETRAINED_MODEL_PATH = os.getenv("PRETRAINED_MODEL_PATH", "final_model")
ENABLE_ONLINE_LEARNING = os.getenv("ENABLE_ONLINE_LEARNING", "true").lower() == "true"
MODEL = os.getenv("MODEL", "simpler_contextual_bandit")
final_model_path = "final_model"
CONTINUE_FROM_PRETRAINED = os.getenv("CONTINUE_FROM_PRETRAINED", "true").lower() == "true"
TTFT_SLO = int(os.getenv("TTFT_SLO", 500))
AVG_TPOT_SLO = int(os.getenv("AVG_TPOT_SLO", 40))
first_request_starting_time = None
signal_amplification_degree = 1.0 # 1.5
reward_amplification_degree = 2.0
reward_amplification_threshold = 0.5

logger.info(f"TTFT_SLO: {TTFT_SLO}")
logger.info(f"AVG_TPOT_SLO: {AVG_TPOT_SLO}")

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
            logger.debug(f"Initialized running stats with {new_count} samples")
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
        # request_stats = RunningStats.load(STATS_FILE)
        request_stats = PerFeatureRunningStats.load(STATS_FILE)
    return request_stats

def write_to_file(log_data, raw_data):
    with open(raw_data, "w") as log_file:
        for request_id, log_message in log_data.items():
            log_file.write(f"{log_message}\n")
    logger.info(f"Successfully wrote {len(log_data)} entries to {raw_data}")

request_features_train = ['input_tokens', 'output_tokens', 'total_tokens']
request_features_reward = ['ttft', 'avg_tpot', 'e2e_latency']


# Fixed handle_flush function
@app.route("/flush", methods=["POST"])
def handle_flush():
    global BATCH_ID, ENCODED_DATA_DIR, NUM_TRAINS, MODEL_UPDATED, TRAINING_DATA_UPDATED, TOTAL_NUM_DATA, NUM_NEW_DATA
    flush_start_time = time.time()
    log_data = request.json
    try:
        logger.info(f"Received log data with {len(log_data) if log_data else 0} entries")
        if not os.path.exists("raw_training_data"):
            os.mkdir("raw_training_data")
        raw_data = f"raw_training_data/batch_{BATCH_ID}.csv"
        BATCH_ID += 1
        
        # Write raw data to file
        ts_write_raw_data = time.time()
        write_to_file(log_data, raw_data)
        logger.info(f"wrote {len(log_data)} entries to {raw_data}, took {time.time() - ts_write_raw_data} seconds")

        # Preprocess raw data
        ts_preprocess = time.time()
        df, _, all_pods, _ = preprocess.main(raw_data, "", TTFT_SLO, AVG_TPOT_SLO)
        logger.info(f"Successfully parsed data, took {time.time() - ts_preprocess} seconds")
        
        
        ## old
        # request_features = ['input_tokens', 'output_tokens', 'total_tokens']
        # pod_features_cols = [col for col in df.columns if col.startswith('pod_') and df[col].dtype in ['float64', 'int64']]
        # all_features = request_features + pod_features_cols
        
        # # Update running stats for each feature separately
        # stats = get_request_stats()
        # stats.update(df[all_features].values, all_features)  # Pass feature names
        # stats.save(STATS_FILE)
        
        # # Apply per-feature normalization
        # normalized_values = stats.normalize(df[all_features].values, all_features)  # Pass feature names
        # for i, feature in enumerate(all_features):
        #     df[feature] = normalized_values[:, i]

        # ===== POD-CENTRIC FEATURE ENGINEERING =====
        request_features = ['input_tokens', 'output_tokens', 'total_tokens']
        pod_features_cols = [col for col in df.columns if col.startswith('pod_') and df[col].dtype in ['float64', 'int64']]
        logger.debug(f"Found features. Request: {request_features}, Pod: {len(pod_features_cols)} features")

        # ===== SELECTIVE NORMALIZATION STRATEGY =====
        stats = get_request_stats()

        # 1. Handle request features - only normalize if they have reasonable variance
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
                    logger.debug(f"Normalized request feature: {feature}")
                else:
                    logger.debug(f"Kept raw values for: {feature}")

        # 2. Handle pod features - normalize high-variance features only
        pod_normalized_count = 0
        for feature in pod_features_cols:
            if feature in df.columns:
                if 'kv_hit_ratio' in feature:
                    logger.debug(f"Skipping normalization for {feature} (already 0-100 scale)")
                    continue
                
                values = df[feature].values
                if values.std() > 0.1:  # Only normalize features with meaningful variance
                    feature_data = values.reshape(-1, 1)
                    
                    if feature not in stats.feature_stats:
                        stats.feature_stats[feature] = RunningStats()
                    
                    stats.feature_stats[feature].update(feature_data)
                    normalized_feature = stats.feature_stats[feature].normalize(feature_data)
                    df[feature] = normalized_feature.flatten()
                    pod_normalized_count += 1
                    logger.debug(f"Normalized pod feature: {feature}")

        # 3. FEATURE IMPORTANCE AMPLIFICATION
        critical_features = ['running_requests', 'waiting_requests', 'decode_tokens', 'prefill_tokens']
        amplified_count = 0
        for feature in pod_features_cols:
            if any(critical in feature for critical in critical_features):
                if feature in df.columns:
                    df[feature] = df[feature] * signal_amplification_degree
                    amplified_count += 1
                    logger.debug(f"Amplified critical feature: {feature}")

        # Save updated stats
        stats.save(STATS_FILE)

        logger.debug(f"Feature processing: {request_normalized_count} request features normalized, "
                f"{pod_normalized_count} pod features normalized, {amplified_count} features amplified")

        # ===== REWARD ENGINEERING =====
        if 'reward' in df.columns:
            rewards = df['reward'].values
            reward_gap = rewards.max() - rewards.min()
            if reward_gap < reward_amplification_threshold:
                logger.debug("Applying reward amplification")
                reward_mean = rewards.mean()
                df['reward'] = reward_mean + (rewards - reward_mean) * reward_amplification_degree
                logger.debug(f"Amplified rewards: gap {reward_gap:.3f} -> {df['reward'].max() - df['reward'].min():.3f}")
        
        # Encode preprocessed data
        ts_encode = time.time()
        encoded_data_subdir = f"{ENCODED_DATA_DIR}/batch_{BATCH_ID}"
        # encoding.encode_for_train(all_pods, df, encoded_data_subdir, stats, request_features_train, request_features_reward)
        encoding.encode_for_train(all_pods, df, encoded_data_subdir, None, request_features_train, request_features_reward)
        logger.info(f"Successfully encoded data to {encoded_data_subdir}, took {time.time() - ts_encode} seconds")
        logger.info(f"Successfully flushed {len(log_data)} log messages, took {time.time() - flush_start_time} seconds")
        TRAINING_DATA_UPDATED = True
        TOTAL_NUM_DATA += len(log_data)
        NUM_NEW_DATA += len(log_data)
            
        return jsonify({"status": "success", "message": f"Successfully processed {len(log_data)} log messages"}), 200
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Unhandled exception: {str(e)}")
        logger.error(f"Traceback: {error_traceback}")
        return jsonify({"status": "error", "message": str(e), "traceback": error_traceback}), 500


# Fixed handle_infer function
@app.route("/infer", methods=["POST"])
def handle_infer():
    global NUM_TRAINS, MODEL_UPDATED, first_request_starting_time
    if first_request_starting_time == None:
        first_request_starting_time = time.time()
        logger.info(f"First request starting time set to {first_request_starting_time}")
    if NUM_TRAINS == 0:
        logger.warning("No trained model available, please call /flush to train first")
        return jsonify({"error": "No trained model available, please call /flush to train first"}), 503
    handle_infer_start_time = time.time()
    try:
        # Get the log message as a string from the request body
        prep_start_time = time.time()
        log_data = request.json
        # Handle string input directly
        if isinstance(log_data, str):
            log_message = log_data
            request_id = "default"  # or extract from log_message
        else:
            # Handle dict input (original logic)
            if len(list(log_data.keys())) != 1:
                logger.error(f"There must be only one request for inference, but got {len(list(log_data.keys()))} requests")
                return jsonify({"error": "Invalid request format"}), 400
            
            first_key = list(log_data.keys())[0]
            log_message = log_data[first_key]
        logger.info(f"Received inference request:\n{log_message}")

        # Extract request ID for logging purposes
        parts = log_message.split("requestID@")
        if len(parts) > 1:
            request_id_parts = parts[1].split("@")
            if request_id_parts:
                request_id = request_id_parts[0]
        handle_infer_total_prep_overhead = time.time() - prep_start_time

        # Use the existing preprocessing function to parse the log
        preprocess_start_time = time.time()
        processed_df, _, all_pods, preprocess_dataset_overhead_summary = preprocess.main(None, log_message, TTFT_SLO, AVG_TPOT_SLO)
        logger.debug(f"Successfully parsed data for request_{request_id}")
        handle_infer_total_total_preprocess_overhead = time.time() - preprocess_start_time

        # Get running statistics and apply normalization (SAME AS TRAINING)
        get_stat_start_time = time.time()
        stats = get_request_stats()
        if stats is None or stats.count == 0:
            # logger.warning(f"No running statistics available, stats: {stats}, stats.count: {stats.count}, stats.mean: {stats.mean}, stats.var: {stats.var}")
            logger.warning(f"No running statistics available, stats: {stats}, stats.count: {stats.count}")

            
        # Apply SAME normalization as training
        request_features = ['input_tokens', 'output_tokens', 'total_tokens']
        pod_features_cols = [col for col in processed_df.columns if col.startswith('pod_') and 
                            processed_df[col].dtype in ['float64', 'int64']]
        
        all_features = request_features + pod_features_cols
        

        # if all(feature in processed_df.columns for feature in all_features) and stats.count > 0:
        #     normalized_values = stats.normalize(processed_df[all_features].values, all_features)  # Pass feature names
        #     for i, feature in enumerate(all_features):
        #         processed_df[feature] = normalized_values[:, i]
        #     logger.debug(f"Applied per-feature normalization to {len(all_features)} features for inference: {len(request_features)} request + {len(pod_features_cols)} pod features")
        # else:
        #     logger.warning(f"Could not apply normalization - missing features or no stats available")

        # Apply SAME pod-centric normalization as training
        if stats.count > 0:
            logger.debug("Applying pod-centric normalization for inference")
            
            # 1. Request features - only normalize if they were normalized in training
            for feature in request_features:
                if feature in processed_df.columns and feature in stats.feature_stats:
                    feature_data = processed_df[feature].values.reshape(-1, 1)
                    normalized_feature = stats.feature_stats[feature].normalize(feature_data)
                    processed_df[feature] = normalized_feature.flatten()
                    logger.debug(f"Normalized request feature {feature} for inference")
                else:
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
        
        handle_infer_total_get_stat_overhead = time.time() - get_stat_start_time

        
        # Encode data (normalization already done)
        encode_start_time = time.time()
        tensor_dataset, encode_for_inference_overhead_summary = encoding.encode_for_inference(all_pods, processed_df, stats, request_features_train, request_features_reward)
        logger.debug(f"Successfully encoded data in memory for inference")
        handle_infer_total_total_encoding_overhead = time.time() - encode_start_time

        infer_from_tensor_start_time = time.time()
        if MODEL == "simpler_contextual_bandit":
            result, infer_from_tensor_overhead_summary = simpler_contextual_bandit.infer_from_tensor(tensor_data=tensor_dataset, model_updated=MODEL_UPDATED)
        elif MODEL == "contextual_bandit":
            result, infer_from_tensor_overhead_summary = contextual_bandit.infer_from_tensor(tensor_data=tensor_dataset, model_updated=MODEL_UPDATED)
        else:
            logger.error(f"Unknown model {MODEL}, please set MODEL environment variable to 'simpler_contextual_bandit' or 'contextual_bandit'")
            return jsonify({"error": f"Unknown model {MODEL}, please set MODEL environment variable to 'simpler_contextual_bandit' or 'contextual_bandit'"}), 500
        if MODEL_UPDATED:
            logger.info("Model updated flag consumed, resetting to False")
            MODEL_UPDATED = False
        handle_infer_total_total_infer_from_tensor_overhead = time.time() - infer_from_tensor_start_time
        result["requestID"] = request_id
        result["num_trains"] = NUM_TRAINS
        result["request_timestamp"] = time.time() - first_request_starting_time
        logger.info(f"Inference result: {result}")
        
        handle_infer_total_wrapup_start_time = time.time()
        # Map the pod index back to the actual pod ID
        selected_pod_index = result.get('selected_pod_index', 0)
        if selected_pod_index >= len(all_pods):
            logger.warning(f"Selected pod index {selected_pod_index} out of range, defaulting to first pod")
            selected_pod_index = 0
            
        selected_pod = all_pods[selected_pod_index]
        handle_infer_total_wrapup_overhead = time.time() - handle_infer_total_wrapup_start_time
        handle_infer_total_overhead = time.time() - handle_infer_start_time
        
        # Return the result
        response = {
            "num_trains": NUM_TRAINS,
            "selected_pod": selected_pod,
            "confidence": result['confidence'],
            "request_id": request_id,
            "* handle_infer_total_prep_overhead": handle_infer_total_prep_overhead*1000,
            "* handle_infer_total_total_preprocess_overhead": handle_infer_total_total_preprocess_overhead*1000,
            # "* handle_infer_total_get_stat_overhead": handle_infer_total_get_stat_overhead*1000,
            "* handle_infer_total_total_encoding_overhead": handle_infer_total_total_encoding_overhead*1000,
            "* handle_infer_total_wrapup_overhead": handle_infer_total_wrapup_overhead*1000,
            "* handle_infer_total_total_infer_from_tensor_overhead": handle_infer_total_total_infer_from_tensor_overhead*1000,
            "* handle_infer_total_overhead": handle_infer_total_overhead*1000,
        }

        for key, value in encode_for_inference_overhead_summary.items():
            response[key] = value
        for key, value in preprocess_dataset_overhead_summary.items():
            response[key] = value
        for key, value in infer_from_tensor_overhead_summary.items():
            response[key] = value
        
        logger.debug(f"Selected pod {selected_pod} with confidence {result['confidence']}")
        return jsonify(response), 200
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Error in handle_infer: {str(e)}")
        logger.error(f"Traceback: {error_traceback}")
        return jsonify({"error": str(e), "traceback": error_traceback}), 500


def train_routine():
    global NUM_TRAINS, MODEL_UPDATED, TRAINING_DATA_UPDATED, TOTAL_NUM_DATA, final_model_path, CONTINUE_FROM_PRETRAINED, NUM_NEW_DATA
    # Load pretrained model on first training if available
    if NUM_TRAINS == 0 and LOAD_PRETRAINED_MODEL:
        if not os.path.exists(PRETRAINED_MODEL_PATH):
            logger.error(f"Pretrained model path {PRETRAINED_MODEL_PATH} does not exist, cannot load pretrained model")
            assert False
        else:
            logger.info(f"Loading pretrained model from {PRETRAINED_MODEL_PATH} for online learning")
            try:
                # Copy pretrained model to final_model_path for inference
                os.makedirs(final_model_path, exist_ok=True)
                os.system(f"cp {PRETRAINED_MODEL_PATH}/* {final_model_path}/")
                
                # Mark model as updated so inference will load it
                MODEL_UPDATED = True
                NUM_TRAINS = 1  # Set to 1 to indicate we have a model
                logger.info("Successfully loaded pretrained model for online learning")
                
                # If online learning is disabled, just use the pretrained model
                if not ENABLE_ONLINE_LEARNING:
                    logger.info("Online learning disabled - using pretrained model only")
                    return
                    
            except Exception as e:
                logger.error(f"Failed to load pretrained model: {e}")
                NUM_TRAINS = 0  # Reset to train from scratch
    
    # Continue with existing training logic only if online learning is enabled
    if ENABLE_ONLINE_LEARNING and TRAINING_DATA_UPDATED and NUM_NEW_DATA > MIN_NUM_TRAINING_DATA:
        training_start_time = time.time()
        logger.info(f"train_routine, Starting {NUM_TRAINS}th online training iteration")
        try:
            if MODEL == "simpler_contextual_bandit":
                simpler_contextual_bandit.train(ENCODED_DATA_DIR, secondary_final_model_path_=None, continue_from_pretrained=CONTINUE_FROM_PRETRAINED)
            elif MODEL == "contextual_bandit":
                contextual_bandit.train(ENCODED_DATA_DIR, continue_from_pretrained=CONTINUE_FROM_PRETRAINED)
            else:
                logger.error(f"Unknown model {MODEL}")
                return
        except Exception as e:
            logger.error(f"Error during training: {e}")
            return
        MODEL_UPDATED = True
        TRAINING_DATA_UPDATED = False
        logger.info(f"train_routine, Successfully completed {NUM_TRAINS}th online training, took {time.time() - training_start_time} seconds")
        NUM_TRAINS += 1
        NUM_NEW_DATA = 0
    else:
        if not ENABLE_ONLINE_LEARNING:
            logger.info("Online learning disabled - skipping training")
        else:
            logger.info(f"train_routine, not enough training data available (TOTAL_NUM_DATA: {TOTAL_NUM_DATA}), skipping training")




if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=train_routine, trigger="interval", seconds=1)
    scheduler.start()
    atexit.register(lambda: scheduler.shutdown())

    app.run(host="0.0.0.0", port=port, debug=False)