#!/usr/bin/env python3

# encoding.py

"""
LLM Request Router - Enhanced Data Preprocessing
-----------------------------------------------
Transforms raw request routing data into structured tensors for transformer-based RL model.
Implements:
- Pod state extraction and normalization
- Expected KV hit ratio isolation for cross-attention
- Request feature extraction
- Metrics-based positional encoding
- Temporal feature handling with staleness indicators
- Request-pod interaction features
- One-hot encoding for categorical features
"""

import sys
import os
import pandas as pd
import numpy as np
from collections import defaultdict
import torch
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle
import logging
import re
import argparse
from datetime import datetime
import time
from logger import logger
import json
from functools import lru_cache
import utils

# GPU features can now be toggled via INCLUDE_GPU_FEATURES in HYPERPARAMETERS
# Default kept for backward compatibility, but will be overridden by HYPERPARAMETERS
INCLUDE_GPU_IN_FEATURE = True  # Legacy default, overridden by HYPERPARAMETERS['INCLUDE_GPU_FEATURES']

random_seed = 42
np.random.seed(random_seed)
class DataEncoder:
    """Processes raw LLM request routing data into formatted tensors for RL training.
    
    Implements advanced encoding techniques:
    1. Metrics-based positional encoding for transformer
    2. Cross-attention preparation for KV hit ratio
    3. Temporal feature handling with staleness indicators
    4. Request-pod interaction features
    """
    
    def __init__(self, output_dir):
        """Initialize the data processor.
        
        Args:
            output_dir: Directory to save processed data and statistics
        """
        self.output_dir = output_dir
        
        # Initialize scalers
        self.pod_feature_scaler = StandardScaler()
        self.request_feature_scaler = StandardScaler()
        self.kv_hit_scaler = StandardScaler()
        
        # Track feature metadata
        self.pod_features = []
        self.numeric_request_features = []
        self.categorical_request_features = []
        self.sorted_all_pod_ids = []
        
        # Key metrics for positional encoding
        self.key_metric_names = [
            'running_requests', 'gpu_kv_cache', 'cpu_kv_cache', 'waiting_requests', 'prefill_tokens', 'decode_tokens', 'kv_hit_ratio', 
        ]
        
        # Statistics tracking
        self.feature_stats = {
            'pod_feature_means': None,
            'pod_feature_stds': None,
            'request_feature_means': None,
            'request_feature_stds': None,
            'kv_hit_means': None,
            'kv_hit_stds': None
        }
        
        # Encoders
        self.pod_encoder = None
        self.selected_pod_encoder = None
        
        # Used for _validate_tensor_compatibility
        self._reference_tensor_data = None

        # GPU feature handling (always enabled)
        self.gpu_models = set()
        self.num_gpu_types = 0


    def analyze_request_features(self, df, request_features_train, request_features_reward):
        """Analyze request features - OPTIMIZED."""
        # Columns to exclude from features
        exclude_cols = set([
            'request_id', 'selected_pod', 'action', 'reward', 
            'ttft_reward', 'tpot_reward', 'ttft_normalized', 'tpot_normalized',
        ] + request_features_reward)
        
        exclude_patterns = ['reward', 'action', 'slo_satisfied', 'normalized']
        
        # OPTIMIZATION: Use set operations for faster filtering
        pod_prefixes = set(f"pod_{pod_id}" for pod_id in self.sorted_all_pod_ids)
        
        candidate_request_features = [
            col for col in df.columns 
            if not any(col.startswith(prefix) for prefix in pod_prefixes)
            and not any(pat in col for pat in exclude_patterns)
            and col not in exclude_cols
        ]
        
        logger.info(f"Request features - Training features: {request_features_train}")
        logger.info(f"Request features - Reward features (excluded from training): {request_features_reward}")
        logger.info(f"Request features - Found {len(candidate_request_features)} candidate columns: {candidate_request_features}")

        # OPTIMIZATION: Vectorized numeric/categorical classification
        numeric_cols = []
        categorical_cols = []
        
        for col in candidate_request_features:
            # Skip columns with too many NaN values
            if df[col].isna().mean() > 0:
                logger.error(f"Request features - {col} has NaN values.")
                assert False
            
            # OPTIMIZATION: Direct dtype check first, then conversion check
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)
            else:
                try:
                    pd.to_numeric(df[col])
                    numeric_cols.append(col)
                except:
                    categorical_cols.append(col)
        
        self.numeric_request_features = numeric_cols
        self.categorical_request_features = categorical_cols
        
        logger.info(f"Request features - number of numeric columns: {len(numeric_cols)}")
        logger.info(f"Request features - number of categorical columns {len(categorical_cols)}")
        if len(numeric_cols) > 0:
            logger.info(f"Request features - numeric features: {numeric_cols}")
        if len(categorical_cols) > 0:
            logger.info(f"Request features - categorical features: {categorical_cols}")

    def encode_pod_ids(self, df):
        """Create encoders for pod IDs - OPTIMIZED."""
        if self.sorted_all_pod_ids:
            # OPTIMIZATION: Pre-convert to numpy array
            sorted_all_pod_ids_np_array = np.array(self.sorted_all_pod_ids).reshape(-1, 1)
            self.pod_encoder = OneHotEncoder(sparse_output=False)
            self.pod_encoder.fit(sorted_all_pod_ids_np_array)

            if 'selected_pod' in df.columns:
                # OPTIMIZATION: Use unique() only once
                selected_pods = df['selected_pod'].dropna().unique()
                selected_pods_array = np.array(selected_pods).reshape(-1, 1)
                self.selected_pod_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                self.selected_pod_encoder.fit(selected_pods_array)
                
                logger.info(f"Encoded {len(selected_pods)} unique selected pods")
        else:
            logger.warning("No pod IDs found, skipping pod encoding")

    def classify_feature_timing(self):
        """Classify feature timing - OPTIMIZED."""
        # OPTIMIZATION: Vectorized classification
        feature_timing = {
            feature: 'historical' if 'last_second' in feature else 'current'
            for feature in self.pod_features
        }
        
        current_features = [f for f, timing in feature_timing.items() if timing == 'current']
        historical_features = [f for f, timing in feature_timing.items() if timing == 'historical']
        
        logger.info(f"Current-time features: {current_features}")
        logger.info(f"historical features: {historical_features}")
        
        # Validation (kept same logic)
        for historical_feat in historical_features:
            if 'last_second' not in historical_feat:
                logger.error(f"Feature {historical_feat} is classified as historical but does not contain 'last_second'")
                assert False
        for current_feat in current_features:
            if 'last_second' in current_feat:
                logger.error(f"Feature {current_feat} is classified as current but contains 'last_second'")
                assert False
                
        return feature_timing


    ## not used currently
    # def prepare_metrics_based_positional_encoding(self, pod_features, feature_indices_map):
    #     # Find indices of key metrics for positional encoding
    #     key_metrics_indices = []
    #     max_feature_dim = pod_features.shape[2]
    #     for metric in self.key_metric_names:
    #         matching_features = [
    #             idx for feature, idx in feature_indices_map.items() 
    #             if metric in feature and idx < max_feature_dim
    #         ]
    #         key_metrics_indices.extend(matching_features)
    #     # Filter out any indices that are still out of bounds
    #     key_metrics_indices = [idx for idx in key_metrics_indices if idx < max_feature_dim]
    #     # If no key metrics found, use a subset of available features
    #     if not key_metrics_indices and pod_features.shape[2] > 0:
    #         # Use first few numeric features (excluding one-hot encoded)
    #         key_metrics_indices = list(range(min(3, pod_features.shape[2])))
    #     # Extract key metrics for positional encoding
    #     if key_metrics_indices:
    #         logger.info(f"Using {len(key_metrics_indices)} metrics for positional encoding, indices: {key_metrics_indices}")
    #         pos_encoding_features = pod_features[:, :, key_metrics_indices]
    #     else:
    #         # Fallback if no suitable metrics found
    #         pos_encoding_features = np.zeros((pod_features.shape[0], pod_features.shape[1], 1))
    #         logger.warning("No suitable metrics for positional encoding, using zeros")
    #     return pos_encoding_features


    ## not used currently
    # def add_staleness_features(self, pod_features, timestamps, feature_timing, feature_indices_map):
    #     """Add staleness indicators for historical features - OPTIMIZED."""
    #     # OPTIMIZATION: Pre-compute historical feature indices
    #     historical_features = [f for f, timing in feature_timing.items() if timing == 'historical']
    #     historical_indices = [
    #         idx for feature, idx in feature_indices_map.items() 
    #         if feature in historical_features
    #     ]
    #     if not historical_indices or len(timestamps) == 0 or np.all(timestamps == 0):
    #         logger.info("No historical features or valid timestamps, skipping staleness")
    #         staleness_features = np.zeros((pod_features.shape[0], pod_features.shape[1], 1))
    #         return np.concatenate([pod_features, staleness_features], axis=2)
    #     # OPTIMIZATION: Vectorized staleness calculation
    #     max_staleness = 60.0
    #     sorted_indices = np.argsort(timestamps)
    #     sorted_timestamps = timestamps[sorted_indices]
    #     time_diffs = np.diff(sorted_timestamps, prepend=sorted_timestamps[0])
    #     time_diffs = np.maximum(time_diffs, 0)
    #     # OPTIMIZATION: Use advanced indexing for reordering
    #     staleness = np.zeros_like(timestamps)
    #     staleness[sorted_indices] = time_diffs
    #     staleness = np.clip(staleness / max_staleness, 0, 1)
    #     # OPTIMIZATION: Broadcasting instead of loop
    #     staleness_features = np.broadcast_to(
    #         staleness[:, np.newaxis, np.newaxis], 
    #         (pod_features.shape[0], pod_features.shape[1], 1)
    #     ).copy()
    #     logger.info(f"Added staleness indicator for {len(historical_indices)} historical features")
    #     return np.concatenate([pod_features, staleness_features], axis=2)


    
    ## not used currently
    # def prepare_cross_attention_inputs(self, pod_features, kv_hit_ratios):
    #     """Format inputs for cross-attention between pod features and KV hit ratios.
    #     This separates pod state from KV hit ratios to enable cross-attention
    #     in the transformer model.
    #     Args:
    #         pod_features: Normalized pod features [batch, n_pods, feature_dim]
    #         kv_hit_ratios: Normalized KV hit ratios [batch, n_pods, 1]
    #     Returns:
    #         Dictionary with query and key/value tensors
    #     """
    #     # Ensure kv_hit_ratios has the right shape
    #     if kv_hit_ratios.shape[2] != 1:
    #         logger.warning(f"Expected KV hit ratios to have shape [batch, n_pods, 1], got {kv_hit_ratios.shape}")
    #     return {
    #         'query': pod_features,  # Pod features as query
    #         'key_value': kv_hit_ratios  # KV hit ratios as key/value
    #     }


    # def create_request_pod_interaction_features(self, request_features, pod_features):
    #     """Create request-pod interaction features - OPTIMIZED."""
    #     if request_features.shape[1] == 0:
    #         logger.warning("No request features available for interaction")
    #         return None
    #     batch_size, n_pods, _ = pod_features.shape
    #     # OPTIMIZATION: Use numpy broadcasting instead of repeat
    #     expanded_request = np.broadcast_to(
    #         request_features[:, np.newaxis, :], 
    #         (batch_size, n_pods, request_features.shape[1])
    #     ).copy()
    #     logger.info(f"Created request-pod interaction features with shape {expanded_request.shape}")
    #     return expanded_request


    def _filter_identity_features(self, pod_features_array, feature_names):
        """
        Remove features that enable pod identity learning.
        Keep only real-time, routing-relevant features.
        """
        # Define which features to KEEP (current state, routing-relevant)
        CURRENT_STATE_FEATURES = [
            'inflight_requests',         # Current total inflight requests
            'inflight_prefill_requests', # Current inflight prefill requests (NEW)
            'inflight_decode_requests',  # Current inflight decode requests (NEW)
            'kv_hit_ratio',              # Current cache performance
            'gpu_kv_cache',              # Current GPU memory usage
            'cpu_kv_cache',              # Current CPU cache usage
            'running_requests',          # Currently processing
            'waiting_requests',          # Currently queued
            'prefill_tokens',            # Current prefill load
            'decode_tokens'              # Current decode load
        ]
        
        # Find indices of features to keep
        keep_indices = []
        kept_features = []
        
        for i, feature_name in enumerate(feature_names):
            if feature_name in CURRENT_STATE_FEATURES:
                keep_indices.append(i)
                kept_features.append(feature_name)
        
        if not keep_indices:
            logger.warning("No current-state features found, keeping all features")
            return pod_features_array, feature_names
        
        # Filter the feature array
        filtered_features = pod_features_array[:, :, keep_indices]
        if len(kept_features) != len(feature_names):
            logger.info(f"Feature masking applied:")
            logger.info(f"  Original features: {len(feature_names)} -> Kept features: {len(kept_features)}")
            logger.info(f"  Kept features: {kept_features}")
            logger.info(f"  Original shape: {pod_features_array.shape} -> New shape: {filtered_features.shape}")
        else:
            logger.debug("No feature masking applied, all features kept")
        
        return filtered_features, kept_features

    def randomize_pod_positions(self, pod_features, kv_hit_ratios):
        """
        Randomize which pod appears in which tensor position for each sample.
        This prevents the model from learning pod identity based on tensor positions.
        
        Args:
            pod_features: [batch_size, num_pods, feature_dim] 
            kv_hit_ratios: [batch_size, num_pods, 1]
        
        Returns:
            Tuple of (shuffled_pod_features, shuffled_kv_hit_ratios)
        """
        batch_size, num_pods = pod_features.shape[:2]
        
        # Create shuffled tensors
        shuffled_pod_features = pod_features.clone()
        shuffled_kv_hit_ratios = kv_hit_ratios.clone()
        
        # Randomize pod order for each sample independently
        for sample_idx in range(batch_size):
            # Generate random permutation for this sample
            perm = torch.randperm(num_pods)
            
            # Apply the same permutation to both tensors
            shuffled_pod_features[sample_idx] = pod_features[sample_idx][perm]
            shuffled_kv_hit_ratios[sample_idx] = kv_hit_ratios[sample_idx][perm]
        
        return shuffled_pod_features, shuffled_kv_hit_ratios

    def _single_row_process_pod_features(self, pod_data, overhead_summary, HYPERPARAMETERS):
        """Optimized processing for single-row inference (n_samples=1)."""
        vectorized_extraction_start_time = time.time()
        
        # Feature definitions (same as batch version) with proper exclusion
        base_features_list = [
            'inflight_requests', 'inflight_prefill_requests', 'inflight_decode_requests',
            'gpu_kv_cache', 'cpu_kv_cache',
            'running_requests', 'waiting_requests', 'prefill_tokens',
            'decode_tokens', 'kv_hit_ratio'
        ]
        excluded = set(HYPERPARAMETERS.get('EXCLUDED_POD_FEATURES', []))
        # Apply exclusions and remove kv_hit_ratio from pod features (handled separately)
        filtered_numeric = [f for f in base_features_list if f not in excluded]
        POD_NUMERIC_FEATURES = [f for f in filtered_numeric if f != 'kv_hit_ratio']
        n_pods = len(self.sorted_all_pod_ids)
        n_pod_numeric = len(POD_NUMERIC_FEATURES)
        
        # Calculate feature dimensions (for pod features only, not including kv_hit_ratio)
        # GPU one-hot encoding is conditional based on INCLUDE_GPU_FEATURES
        gpu_onehot_dim = self.num_gpu_types
        total_feature_dim = n_pod_numeric + gpu_onehot_dim
        
        # OPTIMIZATION: Use 2D array instead of 3D for single row
        pod_features_2d = np.zeros((n_pods, total_feature_dim), dtype=np.float32)
        kv_hit_ratios_1d = np.zeros(n_pods, dtype=np.float32)
        
        # GPU encoding setup (conditional)
        # Extract GPU info from training data (CSV) or runtime state
        gpu_encoded_per_pod = {}
        
        # Only process GPU features if enabled
        if gpu_onehot_dim == 0:
            logger.debug("Skipping GPU encoding (INCLUDE_GPU_FEATURES=0)")
        else:
            for pod_id in self.sorted_all_pod_ids:
                gpu_model_id = None
                
                # First, try to get GPU from the data itself (training CSV or inference data)
                if pod_id in pod_data and 'GPU' in pod_data[pod_id]:
                    # Extract GPU model name from data (e.g., "GPU-L3c", "NVIDIA-A30")
                    gpu_column = pod_data[pod_id]['GPU']
                    # Handle both Series (batch) and single value (inference)
                    if hasattr(gpu_column, 'iloc'):
                        gpu_model_name = gpu_column.iloc[0]
                    else:
                        gpu_model_name = gpu_column
                    
                    # Check if GPU value is NaN (missing data in CSV)
                    if pd.isna(gpu_model_name):
                        logger.warning(f"GPU value is NaN for pod {pod_id} in CSV, trying runtime mapping fallback")
                        gpu_model_id = None  # Signal to use fallback
                    # Look up in static mapping
                    elif gpu_model_name in utils.GPU_MODEL_TO_ENCODE:
                        gpu_model_id = utils.GPU_MODEL_TO_ENCODE[gpu_model_name]
                    else:
                        logger.error(f"Unknown GPU model name: {gpu_model_name} for pod {pod_id}")
                        logger.error(f"Available GPU models: {list(utils.GPU_MODEL_TO_ENCODE.keys())}")
                        assert False
                
                # If still None, try runtime mapping fallback (for inference or when CSV has NaN)
                if gpu_model_id is None and pod_id in HYPERPARAMETERS.get('pod_gpu_id_mapping', {}):
                    gpu_model_id = HYPERPARAMETERS['pod_gpu_id_mapping'][pod_id]
                    logger.debug(f"Using runtime mapping for pod {pod_id}: GPU ID {gpu_model_id}")
                
                # No GPU info available anywhere
                if gpu_model_id is None:
                    logger.error(f"No GPU info found for pod {pod_id}")
                    logger.error(f"Pod not in pod_data GPU column and not in HYPERPARAMETERS['pod_gpu_id_mapping']")
                    logger.error(f"Available pods in pod_data: {list(pod_data.keys())}")
                    if pod_id in pod_data:
                        logger.error(f"Available features for {pod_id}: {list(pod_data[pod_id].keys())}")
                    logger.error(f"Available pods in HYPERPARAMETERS: {list(HYPERPARAMETERS.get('pod_gpu_id_mapping', {}).keys())}")
                    assert False
                
                # Validate GPU model ID
                if gpu_model_id < 0 or gpu_model_id >= self.num_gpu_types:
                    logger.error(f"Invalid GPU model ID {gpu_model_id} for pod {pod_id}")
                    logger.error(f"Expected GPU model ID in range [0, {self.num_gpu_types-1}]")
                    assert False
                
                gpu_encoded_per_pod[pod_id] = gpu_model_id
        
        # Direct value extraction for single row (no array indexing)
        for pod_idx, pod_id in enumerate(self.sorted_all_pod_ids):
            if pod_id in pod_data:
                pod_features = pod_data[pod_id]
                
                # Extract pod numeric features (excluding kv_hit_ratio) 
                for feat_idx, feature_name in enumerate(POD_NUMERIC_FEATURES):
                    if feature_name in pod_features:
                        value = pod_features[feature_name].iloc[0] if hasattr(pod_features[feature_name], 'iloc') else pod_features[feature_name]
                        pod_features_2d[pod_idx, feat_idx] = value
                
                # Extract kv_hit_ratio separately (not included in pod features)
                if 'kv_hit_ratio' in pod_features:
                    kv_value = pod_features['kv_hit_ratio'].iloc[0] if hasattr(pod_features['kv_hit_ratio'], 'iloc') else pod_features['kv_hit_ratio']
                    kv_hit_ratios_1d[pod_idx] = kv_value
                
                # GPU one-hot encoding (conditional)
                if gpu_onehot_dim > 0:
                    gpu_model_id = gpu_encoded_per_pod[pod_id]
                    gpu_onehot = np.zeros(gpu_onehot_dim)
                    gpu_onehot[gpu_model_id] = 1
                    pod_features_2d[pod_idx, n_pod_numeric:] = gpu_onehot
        
        vectorized_extraction_overhead = time.time() - vectorized_extraction_start_time
        overhead_summary['vectorized_extraction'] = vectorized_extraction_overhead
        
        # Convert back to 3D format expected by caller (reshape 2D -> 3D with batch dimension 1)
        pod_features_3d = pod_features_2d.reshape(1, n_pods, total_feature_dim)
        # CRITICAL FIX: KV hit ratios must be [1, n_pods, 1] not [1, n_pods]
        kv_hit_ratios_3d = kv_hit_ratios_1d.reshape(1, n_pods, 1)
        
        # Create per-pod feature indices (using pod features only, excluding kv_hit_ratio)
        per_pod_feature_indices = {}
        for pod_idx, pod_id in enumerate(self.sorted_all_pod_ids):
            per_pod_feature_indices[pod_id] = {feature: idx for idx, feature in enumerate(POD_NUMERIC_FEATURES)}
        

        
        return pod_features_3d, kv_hit_ratios_3d, kv_hit_ratios_3d, per_pod_feature_indices

    def _process_pod_features(self, pod_data, n_samples, overhead_summary, HYPERPARAMETERS):
        if not pod_data:
            logger.error("No pod data in expected format")
            assert False
        
        # OPTIMIZATION 3: Single-row fast path for inference
        if n_samples == 1:
            return self._single_row_process_pod_features(pod_data, overhead_summary, HYPERPARAMETERS)
        
        vectorized_extraction_start_time = time.time()
        
        # Include ALL features we want to potentially keep
        base_features_list = [
            'inflight_requests',
            'inflight_prefill_requests',  # NEW: Per-pod inflight prefill requests
            'inflight_decode_requests',   # NEW: Per-pod inflight decode requests
            'gpu_kv_cache',
            'cpu_kv_cache',
            'running_requests',
            'waiting_requests',
            'prefill_tokens',
            'decode_tokens',
            'kv_hit_ratio'
        ]
        excluded = set(HYPERPARAMETERS.get('EXCLUDED_POD_FEATURES', []))
        ALL_NUMERIC_FEATURES = [f for f in base_features_list if f not in excluded]
        n_pods = len(self.sorted_all_pod_ids)
        n_numeric = len(ALL_NUMERIC_FEATURES)
        
        # Calculate total feature dimensions including GPU one-hot encoding (conditional)
        gpu_onehot_dim = self.num_gpu_types
        total_feature_dim = n_numeric + gpu_onehot_dim
        
        all_features_array = np.zeros((n_samples, n_pods, total_feature_dim), dtype=np.float32)

        # GPU encoding setup (conditional)
        # Extract GPU info from training data (CSV) or runtime state
        gpu_encoded_per_pod = {}
        
        # Only process GPU features if enabled
        if gpu_onehot_dim == 0:
            logger.debug("Skipping GPU encoding in batch processing (INCLUDE_GPU_FEATURES=0)")
        else:
            for pod_id in self.sorted_all_pod_ids:
                gpu_model_id = None
                
                # First, try to get GPU from the data itself (training CSV or inference data)
                if pod_id in pod_data and 'GPU' in pod_data[pod_id]:
                    # Extract GPU model name from data (e.g., "GPU-L3c", "NVIDIA-A30")
                    gpu_column = pod_data[pod_id]['GPU']
                    # Handle both Series (batch) and single value
                    if hasattr(gpu_column, 'iloc'):
                        gpu_model_name = gpu_column.iloc[0]
                    else:
                        gpu_model_name = gpu_column
                    
                    # Check if GPU value is NaN (missing data in CSV)
                    if pd.isna(gpu_model_name):
                        logger.warning(f"GPU value is NaN for pod {pod_id} in CSV, trying runtime mapping fallback")
                        gpu_model_id = None  # Signal to use fallback
                    # Look up in static mapping
                    elif gpu_model_name in utils.GPU_MODEL_TO_ENCODE:
                        gpu_model_id = utils.GPU_MODEL_TO_ENCODE[gpu_model_name]
                    else:
                        logger.error(f"Unknown GPU model name: {gpu_model_name} for pod {pod_id}")
                        logger.error(f"Available GPU models: {list(utils.GPU_MODEL_TO_ENCODE.keys())}")
                        assert False
                
                # If still None, try runtime mapping fallback (for inference or when CSV has NaN)
                if gpu_model_id is None and pod_id in HYPERPARAMETERS.get('pod_gpu_id_mapping', {}):
                    gpu_model_id = HYPERPARAMETERS['pod_gpu_id_mapping'][pod_id]
                    logger.debug(f"Using runtime mapping for pod {pod_id}: GPU ID {gpu_model_id}")
                
                # No GPU info available anywhere
                if gpu_model_id is None:
                    logger.error(f"No GPU info found for pod {pod_id}")
                    logger.error(f"Pod not in pod_data GPU column and not in HYPERPARAMETERS['pod_gpu_id_mapping']")
                    logger.error(f"Available pods in pod_data: {list(pod_data.keys())}")
                    if pod_id in pod_data:
                        logger.error(f"Available features for {pod_id}: {list(pod_data[pod_id].keys())}")
                    logger.error(f"Available pods in HYPERPARAMETERS: {list(HYPERPARAMETERS.get('pod_gpu_id_mapping', {}).keys())}")
                    assert False
                
                # Validate GPU model ID
                if gpu_model_id < 0 or gpu_model_id >= self.num_gpu_types:
                    logger.error(f"Invalid GPU model ID {gpu_model_id} for pod {pod_id}")
                    logger.error(f"Expected GPU model ID in range [0, {self.num_gpu_types-1}]")
                    assert False
                
                gpu_encoded_per_pod[pod_id] = gpu_model_id

        ## THIS IS WHERE THE BUG MANIFESTS:
        ## The all_pods order determines how features are arranged in tensors
        # Extract all features into single array
        for pod_idx, pod_id in enumerate(self.sorted_all_pod_ids):
            if pod_id in pod_data:
                pod_features = pod_data[pod_id]
                
                # Assign features dynamically based on ALL_NUMERIC_FEATURES order
                # This correctly handles feature exclusion by adjusting indices
                for feat_index, feat_name in enumerate(ALL_NUMERIC_FEATURES):
                    if feat_name in pod_features:
                        all_features_array[:, pod_idx, feat_index] = pod_features[feat_name].fillna(0)
                
                # GPU one-hot encoding (conditional)
                if gpu_onehot_dim > 0:
                    gpu_model_id = gpu_encoded_per_pod[pod_id]
                    gpu_onehot = np.zeros(gpu_onehot_dim)
                    gpu_onehot[gpu_model_id] = 1
                    all_features_array[:, pod_idx, n_numeric:] = gpu_onehot

        vectorized_extraction_overhead = time.time() - vectorized_extraction_start_time
        
        build_feature_start_time = time.time()
        
        # Separate numeric and GPU features before masking
        numeric_features_only = all_features_array[:, :, :n_numeric]  # First n_numeric features
        gpu_features_only = all_features_array[:, :, n_numeric:]      # Last gpu_onehot_dim features
        
        # Apply masking to numeric features only
        original_features = ALL_NUMERIC_FEATURES.copy()
        filtered_numeric_features, kept_numeric_features = self._filter_identity_features(
            numeric_features_only, original_features
        )
        
        # Combine filtered numeric + all GPU features (GPU features are never masked, always included)
        filtered_features_array = np.concatenate([filtered_numeric_features, gpu_features_only], axis=2)
        gpu_feature_names = [f'gpu_model_{i}' for i in range(self.num_gpu_types)]
        kept_features = kept_numeric_features + gpu_feature_names
        
        # SOLUTION 1: Always ensure kv_hit_ratio is available separately
        kv_extraction_start_time = time.time()
        
        if 'kv_hit_ratio' in kept_features:
            # Extract KV ratios from filtered array
            kv_index = kept_features.index('kv_hit_ratio')
            kv_hit_norm = filtered_features_array[:, :, kv_index:kv_index+1]  # Keep as [batch, pods, 1]
            pod_kv_hit_array = kv_hit_norm.copy()
            
            # Remove KV from pod features to avoid duplication in model input
            other_indices = [i for i in range(len(kept_features)) if i != kv_index]
            if other_indices:  # Only if there are other features besides kv_hit_ratio
                pod_features_array = filtered_features_array[:, :, other_indices]
                kept_pod_features = [feat for i, feat in enumerate(kept_features) if i != kv_index]
            else:
                # Edge case: only kv_hit_ratio was kept - create minimal pod features
                logger.warning("Only kv_hit_ratio was kept after masking, creating minimal pod features")
                pod_features_array = np.ones((n_samples, n_pods, 1), dtype=np.float32) * 0.5  # Neutral values
                kept_pod_features = ['minimal_feature']
            
            logger.info(f"Extracted KV ratios separately: {kv_hit_norm.shape}")
            logger.info(f"Remaining pod features: {len(kept_pod_features)} features")
            
        else:
            # KV hit ratio was filtered out - this shouldn't happen with our CURRENT_STATE_FEATURES
            logger.error("kv_hit_ratio was filtered out by masking - this should not happen!")
            logger.error("Check your CURRENT_STATE_FEATURES list in _filter_identity_features")
            
            # Create fallback KV tensor and use all filtered features as pod features
            kv_hit_norm = np.zeros((n_samples, n_pods, 1), dtype=np.float32)
            pod_kv_hit_array = kv_hit_norm.copy()
            pod_features_array = filtered_features_array
            kept_pod_features = kept_features
            
            logger.warning("Using fallback: zero KV ratios and all filtered features as pod features")
        
        kv_extraction_overhead = time.time() - kv_extraction_start_time
        
        # APPLY POD RANDOMIZATION HERE
        randomization_start_time = time.time()
        
        # Convert to tensors for randomization
        pod_features_tensor = torch.from_numpy(pod_features_array).float()
        kv_hit_tensor = torch.from_numpy(kv_hit_norm).float()
        
        # # Apply randomization
        # logger.info("Applying pod position randomization...")
        # randomized_pod_features, randomized_kv_hit = self.randomize_pod_positions(
        #     pod_features_tensor, kv_hit_tensor
        # )

        randomized_pod_features = pod_features_tensor
        randomized_kv_hit = kv_hit_tensor
        
        # Convert back to numpy
        pod_features_array = randomized_pod_features.numpy()
        kv_hit_norm = randomized_kv_hit.numpy()
        pod_kv_hit_array = kv_hit_norm.copy()
        
        randomization_overhead = time.time() - randomization_start_time
        
        # logger.info(f"✅ Pod randomization applied - each sample has different pod order")
        # logger.info(f"   This prevents model from learning pod identity based on tensor positions")
        
        # Update feature list to reflect final pod features (without kv_hit_ratio)
        self.pod_features = kept_pod_features
        
        build_feature_overhead = time.time() - build_feature_start_time
        
        logger.debug(f"FINAL TENSOR ANALYSIS:")
        logger.debug(f"pod_features_array shape: {pod_features_array.shape}")
        logger.debug(f"First pod features: {pod_features_array[0, 0, :]}")
        logger.debug(f"Second pod features: {pod_features_array[0, 1, :]}")
        logger.debug(f"GPU features check (always included):")
        for i in range(min(3, pod_features_array.shape[1])):
            gpu_start_idx = len(kept_pod_features) - self.num_gpu_types
            gpu_features = pod_features_array[0, i, gpu_start_idx:]
            logger.debug(f"  Pod {i} GPU one-hot: {gpu_features}")
        logger.debug(f"Final feature composition:")
        logger.debug(f"  pod_features_array shape: {pod_features_array.shape}")
        logger.debug(f"  kept_pod_features: {kept_pod_features}")
        logger.debug(f"  kv_hit_norm shape: {kv_hit_norm.shape}")
        logger.debug(f"  GPU features included: {[f for f in kept_pod_features if 'gpu_model' in f]}")
        


        return pod_features_array, pod_kv_hit_array, kv_hit_norm, {}
    
    def prepare_for_encoding(self, processed_df, sorted_all_pod_ids, request_features_train, HYPERPARAMETERS):
        logger.info(f"prepare_for_encoding received request_features_train: {request_features_train}")
        overhead_summary = {}
        self.sorted_all_pod_ids = sorted_all_pod_ids
        
        # CRITICAL: Extract RAW input_tokens BEFORE any normalization for plotting
        raw_input_tokens = None
        if 'input_tokens' in processed_df.columns:
            raw_input_tokens = processed_df['input_tokens'].fillna(0).values.astype(np.float32)
            logger.debug(f"Extracted raw input_tokens: min={raw_input_tokens.min():.0f}, max={raw_input_tokens.max():.0f}, mean={raw_input_tokens.mean():.0f}")
        
        # Initialize GPU one-hot dimension based on INCLUDE_GPU_FEATURES flag
        # Default to True for backward compatibility if not specified
        include_gpu_features = HYPERPARAMETERS['INCLUDE_GPU_FEATURES']
        if include_gpu_features:
            # Use fixed size from utils.GPU_MODEL_TO_ENCODE to ensure consistency across training/inference
            self.num_gpu_types = len(utils.GPU_MODEL_TO_ENCODE)
            logger.debug(f"✅ GPU one-hot encoding ENABLED: {self.num_gpu_types} GPU types")
        else:
            self.num_gpu_types = 0
            logger.debug(f"⚠️  GPU one-hot encoding DISABLED (INCLUDE_GPU_FEATURES=0)")
        extract_pod_columns_start = time.time()
        pod_data = self._extract_pod_columns(processed_df, sorted_all_pod_ids)
        overhead_summary['extract_pod_columns'] = time.time() - extract_pod_columns_start
        self.numeric_request_features = request_features_train  # Assume all numeric
        self.categorical_request_features = []
        self.pod_encoder = None
        self.selected_pod_encoder = None
        
        classify_feature_timing_start = time.time()
        pod_feature_columns = [col for col in processed_df.columns if col.startswith('pod_')]
        unique_features = list(set(col.split('-')[1] for col in pod_feature_columns if '-' in col))
        self.pod_features = sorted(unique_features)
        feature_timing = {f: 'historical' if 'last_second' in f else 'current' for f in self.pod_features}
        overhead_summary['classify_feature_timing'] = time.time() - classify_feature_timing_start
        
        # STEP 5: FAST request feature
        n_samples = len(processed_df)
        extract_request_feature_start = time.time()
        request_features, _ = self.extract_request_features(processed_df, request_features_train, n_samples)
        overhead_summary['extract_request_feature'] = time.time() - extract_request_feature_start

        # STEP 7: ULTRA-OPTIMIZED pod processing
        process_pod_feature_start = time.time()
        pod_features_array, pod_kv_hit_array, kv_hit_norm, per_pod_feature_indices = self._process_pod_features(pod_data, n_samples, overhead_summary, HYPERPARAMETERS)
        overhead_summary['process_pod_feature'] = time.time() - process_pod_feature_start

        # STEP 8: actions/rewards (continues as normal)
        extract_actions_start = time.time()
        actions, rewards, ttft_rewards, tpot_rewards, ttft, avg_tpot, e2e_latency, input_tokens_normalized = self.extract_actions_rewards(processed_df, n_samples)
        overhead_summary['extract_actions'] = time.time() - extract_actions_start
        
        # Use raw_input_tokens if available (extracted before normalization), otherwise use normalized version
        if raw_input_tokens is not None:
            input_tokens_for_plotting = raw_input_tokens
        else:
            input_tokens_for_plotting = input_tokens_normalized
            logger.warning("raw_input_tokens not available, using potentially normalized values for plotting")

        # STEP 10: MINIMAL positional encoding
        positional_encoding_start = time.time()
        positional_encodings = np.zeros((pod_features_array.shape[0], pod_features_array.shape[1], 1), dtype=np.float32)
        staleness_features = np.zeros((pod_features_array.shape[0], pod_features_array.shape[1], 1), dtype=np.float32)
        pod_features_with_staleness = np.concatenate([pod_features_array, staleness_features], axis=2)
        logger.debug(f"pod_features_array.shape: {pod_features_array.shape}")
        logger.debug(f"staleness_features.shape: {staleness_features.shape}")
        logger.debug(f"pod_features_with_staleness.shape: {pod_features_with_staleness.shape}")
        cross_attention_inputs = {'query': pod_features_with_staleness, 'key_value': kv_hit_norm}
        overhead_summary['positional_encoding'] = time.time() - positional_encoding_start
        
        # STEP 13: FAST interaction features
        interaction_features_start = time.time()
        # OPTIMIZATION: Simplified interaction features for single row
        if n_samples == 1:
            # Direct creation without broadcast for single row (more efficient)
            n_pods = pod_features_array.shape[1]
            n_request_features = request_features.shape[1]
            interaction_features = np.tile(request_features[0], (n_pods, 1)).reshape(1, n_pods, n_request_features)
        else:
            # Original broadcast approach for batch processing
            interaction_features = np.broadcast_to(request_features[:, np.newaxis, :], (n_samples, pod_features_array.shape[1], request_features.shape[1])).copy()
        overhead_summary['interaction_features'] = time.time() - interaction_features_start

        processed_data = {
            'pod_features': pod_features_array,
            'pod_raw_features': pod_features_array,
            'kv_hit_ratios': kv_hit_norm,
            'kv_hit_raw': pod_kv_hit_array,
            'positional_encodings': positional_encodings,
            'pod_features_with_staleness': pod_features_with_staleness,
            'cross_attention_inputs': cross_attention_inputs,
            'request_features': request_features,
            'request_numeric_features': request_features,
            'request_categorical_features': np.zeros((n_samples, 0)),
            'interaction_features': interaction_features,
            'timestamps': np.zeros(n_samples),
            'feature_timing': feature_timing,
            'pod_ids': self.sorted_all_pod_ids,
            'actions': actions,
            'rewards': rewards,
            'ttft_rewards': ttft_rewards,
            'tpot_rewards': tpot_rewards,
            'ttft': ttft,
            'avg_tpot': avg_tpot,
            'e2e_latency': e2e_latency,
            'input_tokens': input_tokens_for_plotting,  # RAW values for stratified reward analysis
            'feature_stats': getattr(self, 'feature_stats', {}),
            'pod_features_list': self.pod_features,
            'feature_indices_map': per_pod_feature_indices[self.sorted_all_pod_ids[0]] if per_pod_feature_indices and self.sorted_all_pod_ids else {},
            'numeric_request_features': self.numeric_request_features,
            'categorical_request_features': self.categorical_request_features,
            'encoders': {'pod_encoder': None, 'selected_pod_encoder': None, 'categorical_encoders': {}}
        }
        
        return processed_data, overhead_summary


    def _extract_pod_columns(self, processed_df, sorted_all_pod_ids):
        pod_data = {}
        for col in processed_df.columns:
            if col.startswith('pod_') and '-' in col:
                pod_id, feature = col.split('-', 1)
                # pod_id = pod_id.replace('pod_', '')
                if pod_id in sorted_all_pod_ids:
                    if pod_id not in pod_data:
                        pod_data[pod_id] = {}
                    pod_data[pod_id][feature] = processed_df[col]
                else:
                    logger.error(f"Pod ID {pod_id} not found in sorted_all_pod_ids: {sorted_all_pod_ids}, column: {col}")
                    exit()
        if not pod_data:
            logger.error("No pod data found in the DataFrame")
            logger.error(f"processed_df: {processed_df}")
            logger.error(f"Expected pod IDs: {sorted_all_pod_ids}")
            logger.error(f"Extracted pod data: {pod_data}")
            processed_df.to_csv('debug_processed_df.csv', index=False)
            exit(1)
        return pod_data

    
    def extract_request_features(self, processed_df, request_features_train, n_samples):
        request_features_start_time = time.time()

        if request_features_train:
            # Extract request features by column names (supports variable length)
            try:
                request_features = processed_df[request_features_train].values.astype(np.float32, copy=False)
                logger.debug(f"Extracted {len(request_features_train)} request features: {request_features_train}")
            except KeyError as e:
                logger.error(f"Missing request feature column: {e}")
                logger.error(f"Available columns: {list(processed_df.columns)}")
                logger.error(f"Requested features: {request_features_train}")
                assert False
        else:
            logger.warning("No request features provided for inference, using empty array")
            request_features = np.zeros((n_samples, 0), dtype=np.float32)

        request_features_overhead = time.time() - request_features_start_time
        return request_features, request_features_overhead


    def extract_actions_rewards(self, df, n_samples):
        """Fast action/reward extraction - minimal validation."""
        actions = np.zeros(n_samples, dtype=np.int64)
        rewards = np.zeros(n_samples, dtype=np.float32)
        ttft_rewards = np.zeros(n_samples, dtype=np.float32)
        tpot_rewards = np.zeros(n_samples, dtype=np.float32)
        ttft = np.zeros(n_samples, dtype=np.float32)
        avg_tpot = np.zeros(n_samples, dtype=np.float32)
        e2e_latency = np.zeros(n_samples, dtype=np.float32)
        input_tokens = np.zeros(n_samples, dtype=np.float32)  # NEW: Add input_tokens
        
        # Direct extraction without validation
        if 'selected_pod' in df.columns:
            pod_to_idx = {pod_id: i for i, pod_id in enumerate(self.sorted_all_pod_ids)}
            selected_pods = df['selected_pod'].values
            for i, pod in enumerate(selected_pods):
                if pd.notna(pod):
                    idx = pod_to_idx.get(str(pod))
                    if idx is not None:
                        actions[i] = idx
        
        # Direct column extraction (added input_tokens)
        for col, target in [('reward', rewards), ('ttft_reward', ttft_rewards), ('tpot_reward', tpot_rewards), 
                           ('ttft', ttft), ('avg_tpot', avg_tpot), ('e2e_latency', e2e_latency), 
                           ('input_tokens', input_tokens)]:
            if col in df.columns:
                try:
                    target[:] = df[col].fillna(0).values.astype(np.float32)
                except Exception as e:
                    the_first_row = df.iloc[0]
                    logger.error(f"df.columns: {df.columns}")
                    logger.error(f"the_first_row: {the_first_row}")
                    logger.error(f"the_first_row[{col}]: {the_first_row[col]}")
                    logger.error(f"Error processing column {col}: {e}")
                    exit(1)

        return actions, rewards, ttft_rewards, tpot_rewards, ttft, avg_tpot, e2e_latency, input_tokens

    def save_processed_data(self, processed_data, max_samples_for_reference=10000000):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.output_dir, exist_ok=True)

        # Randomly sample max_samples_for_reference samples for explainability reference
        n_samples = len(processed_data['actions'])
        if n_samples > max_samples_for_reference:
            # Randomly sample indices
            np.random.seed(42)  # For reproducibility
            sample_indices = np.random.choice(n_samples, max_samples_for_reference, replace=False)
            sample_indices = np.sort(sample_indices)  # Sort for better memory access

            logger.info(f"Sampling {max_samples_for_reference} out of {n_samples} samples for reference dataset (explainability)")

            def sample_array(arr, indices):
                if arr.ndim == 1:
                    return arr[indices]
                elif arr.ndim == 2:
                    return arr[indices, :]
                elif arr.ndim == 3:
                    return arr[indices, :, :]
                else:
                    return arr[indices]

        else:
            sample_indices = None
            logger.info(f"Saving all {n_samples} samples (fewer than max_samples_for_reference)")

        # Create a PyTorch tensor dataset
        tensor_data = {
            # Basic tensors
            'pod_features': torch.FloatTensor(
                processed_data['pod_features'][sample_indices] if sample_indices is not None else processed_data['pod_features']
            ),
            'kv_hit_ratios': torch.FloatTensor(
                processed_data['kv_hit_ratios'][sample_indices] if sample_indices is not None else processed_data['kv_hit_ratios']
            ),
            'request_features': torch.FloatTensor(
                processed_data['request_features'][sample_indices] if sample_indices is not None else processed_data['request_features']
            ),
            'actions': torch.LongTensor(
                processed_data['actions'][sample_indices] if sample_indices is not None else processed_data['actions']
            ),
            'rewards': torch.FloatTensor(
                processed_data['rewards'][sample_indices] if sample_indices is not None else processed_data['rewards']
            ),
            'ttft': torch.FloatTensor(
                processed_data['ttft'][sample_indices] if sample_indices is not None else processed_data['ttft']
            ),
            'avg_tpot': torch.FloatTensor(
                processed_data['avg_tpot'][sample_indices] if sample_indices is not None else processed_data['avg_tpot']
            ),
            'e2e_latency': torch.FloatTensor(
                processed_data['e2e_latency'][sample_indices] if sample_indices is not None else processed_data['e2e_latency']
            ),
            'input_tokens': torch.FloatTensor(
                processed_data['input_tokens'][sample_indices] if sample_indices is not None else processed_data['input_tokens']
            ),
            
            'pod_features_with_staleness': torch.FloatTensor(
                processed_data['pod_features_with_staleness'][sample_indices] if sample_indices is not None else processed_data['pod_features_with_staleness']
            ),
            
            # Enhanced features for transformer
            # 'positional_encodings': torch.FloatTensor(processed_data['positional_encodings']),
            # Cross-attention components
            # 'query': torch.FloatTensor(processed_data['cross_attention_inputs']['query']),
            # 'key_value': torch.FloatTensor(processed_data['cross_attention_inputs']['key_value']),
        }
        
        # Add interaction features if available
        if processed_data['interaction_features'] is not None:
            tensor_data['interaction_features'] = torch.FloatTensor(
                processed_data['interaction_features'][sample_indices] if sample_indices is not None else processed_data['interaction_features']
            )

        # Add additional reward components if available
        if 'ttft_rewards' in processed_data and processed_data['ttft_rewards'] is not None:
            tensor_data['ttft_rewards'] = torch.FloatTensor(
                processed_data['ttft_rewards'][sample_indices] if sample_indices is not None else processed_data['ttft_rewards']
            )
        if 'tpot_rewards' in processed_data and processed_data['tpot_rewards'] is not None:
            tensor_data['tpot_rewards'] = torch.FloatTensor(
                processed_data['tpot_rewards'][sample_indices] if sample_indices is not None else processed_data['tpot_rewards']
            )
        # global_tensor_path = "global_tensor_dataset.pt"
        # self._append_to_global_tensor_dataset(tensor_data, global_tensor_path)
        torch.save(tensor_data, os.path.join(self.output_dir, "tensor_dataset.pt"))
        
        if hasattr(self, '_reference_tensor_data') and self._reference_tensor_data is not None:
            if self._validate_tensor_compatibility(self._reference_tensor_data, tensor_data):
                logger.debug("✅ Tensor data compatible with reference batch")
            else:
                logger.warning("⚠️ Tensor data incompatible with reference batch")
        else:
            # Store first batch as reference for future validations
            self._reference_tensor_data = {k: v.clone() if isinstance(v, torch.Tensor) else v 
                                        for k, v in tensor_data.items()}
            logger.debug("📝 Stored reference tensor data for future validation")


        # Update metadata to reflect sampled dataset size
        sampled_dataset_size = max_samples_for_reference if sample_indices is not None else n_samples

        metadata = {
            'dataset_size': sampled_dataset_size,
            'original_dataset_size': n_samples,
            'sampling_applied': sample_indices is not None,
            'num_pods': len(processed_data['pod_ids']),
            # Names for correct XAI labeling
            'pod_features_list': processed_data.get('pod_features_list', []),
            'numeric_request_features': processed_data.get('numeric_request_features', []),
            'categorical_request_features': processed_data.get('categorical_request_features', []),
            'pod_ids': processed_data.get('pod_ids', []),
            'feature_dimensions': {
                'pod_features': processed_data['pod_features'].shape[2],
                'pod_features_with_staleness': processed_data['pod_features_with_staleness'].shape[2],
                'kv_hit_ratios': processed_data['kv_hit_ratios'].shape[2],
                'request_features': processed_data['request_features'].shape[1],
                'positional_encodings': processed_data['positional_encodings'].shape[2],
            },
            'reward_statistics': {
                'mean': float(np.mean(processed_data['rewards'][sample_indices] if sample_indices is not None else processed_data['rewards'])),
                'std': float(np.std(processed_data['rewards'][sample_indices] if sample_indices is not None else processed_data['rewards'])),
                'min': float(np.min(processed_data['rewards'][sample_indices] if sample_indices is not None else processed_data['rewards'])),
                'max': float(np.max(processed_data['rewards'][sample_indices] if sample_indices is not None else processed_data['rewards'])),
            },
            'action_distribution': {
                str(i): int(np.sum((processed_data['actions'][sample_indices] if sample_indices is not None else processed_data['actions']) == i))
                for i in range(len(processed_data['pod_ids']))
            },
            'timestamp': timestamp,
            'processing_info': {
                'historical_features': len([f for f, t in processed_data['feature_timing'].items() if t == 'historical']),
                'current_features': len([f for f, t in processed_data['feature_timing'].items() if t == 'current'])
            }
        }
        
        with open(os.path.join(self.output_dir, "metadata.json"), 'w') as f:
        # with open("metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved processed data to {self.output_dir}")
    

    def _validate_tensor_compatibility(self, existing_data, new_data):
        """Validate that new tensor data is compatible with existing data for concatenation.
        
        Args:
            existing_data: Existing tensor dataset
            new_data: New tensor data to append
            
        Returns:
            True if compatible, False otherwise
        """
        # Check if both datasets have the same keys (for tensors)
        existing_tensor_keys = {k for k, v in existing_data.items() if isinstance(v, torch.Tensor)}
        new_tensor_keys = {k for k, v in new_data.items() if isinstance(v, torch.Tensor)}
        
        missing_keys = existing_tensor_keys - new_tensor_keys
        extra_keys = new_tensor_keys - existing_tensor_keys
        
        if missing_keys:
            logger.error(f"New data missing tensor keys: {missing_keys}")
            return False
        
        if extra_keys:
            logger.warning(f"New data has extra tensor keys: {extra_keys}")
            # We can still proceed, just add the new keys
        
        # Check tensor shape compatibility (all dimensions except batch should match)
        for key in existing_tensor_keys.intersection(new_tensor_keys):
            existing_shape = existing_data[key].shape
            new_shape = new_data[key].shape
            
            if len(existing_shape) != len(new_shape):
                logger.error(f"Tensor {key}: dimension mismatch - existing: {existing_shape}, new: {new_shape}")
                return False
            
            if len(existing_shape) > 1 and existing_shape[1:] != new_shape[1:]:
                logger.error(f"Tensor {key}: shape mismatch - existing: {existing_shape}, new: {new_shape}")
                return False
        
        return True

    def create_dataset_loaders(self, processed_data, batch_size=32, val_split=0.1):
        """Create PyTorch DataLoader objects for training and validation.

        Args:
            processed_data: Dictionary with preprocessed data
            batch_size: Batch size for training
            val_split: Fraction of data to use for validation

        Returns:
            train_loader, val_loader: DataLoader objects
        """
        try:
            import torch
            from torch.utils.data import TensorDataset, DataLoader, random_split

            # Create tensor dataset
            tensor_data = [
                torch.FloatTensor(processed_data['pod_features_with_staleness']),
                torch.FloatTensor(processed_data['kv_hit_ratios']),
                torch.FloatTensor(processed_data['request_features']),
                torch.LongTensor(processed_data['actions']),
                torch.FloatTensor(processed_data['rewards'])
            ]

            # Add positional encodings
            if 'positional_encodings' in processed_data:
                tensor_data.append(torch.FloatTensor(processed_data['positional_encodings']))

            # Create dataset
            dataset = TensorDataset(*tensor_data)

            # Split into train and validation
            val_size = int(len(dataset) * val_split)
            train_size = len(dataset) - val_size

            train_dataset, val_dataset = random_split(
                dataset, [train_size, val_size]
            )

            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=torch.cuda.is_available()
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=torch.cuda.is_available()
            )

            logger.info(f"Created data loaders with {train_size} training and {val_size} validation samples")

            return train_loader, val_loader

        except ImportError:
            logger.warning("PyTorch not available, skipping data loader creation")
            return None, None


def encode_for_train(sorted_all_pod_ids, processed_df, output_dir, request_features_train, HYPERPARAMETERS):
    logger.info(f"encode_for_train received request_features_train: {request_features_train}")
    if len(processed_df) > 0:
        logger.info("First row selected_pod value: " + str(processed_df.iloc[0].get('selected_pod', 'N/A')))
    # Check if data contains the expected column pattern
    pod_cols = [c for c in processed_df.columns if 'pod_' in c or '-pod' in c]
    if not pod_cols:
        logger.warning("No columns with 'pod_' prefix or '-pod' pattern found")

    assert processed_df['selected_pod'].iloc[0] in sorted_all_pod_ids, f"Selected pod {processed_df['selected_pod'].iloc[0]} not in sorted_all_pod_ids {sorted_all_pod_ids}"

    # Basic data quality checks
    logger.info("Performing data quality checks...")
    # Only consider numeric columns for missing-value thresholding to avoid metadata columns
    numeric_df = processed_df.select_dtypes(include=[np.number])
    missing_col_pct = numeric_df.isnull().mean() * 100
    high_missing = missing_col_pct[missing_col_pct > 20]
    if len(high_missing) > 0:
        # Log detailed diagnostics before failing
        sorted_high_missing = high_missing.sort_values(ascending=False)
        top_list = [(col, round(pct, 1)) for col, pct in list(sorted_high_missing.items())[:20]]
        logger.error(f"Columns with >20% missing values: {len(sorted_high_missing)} columns. Top offenders: {top_list}")
        # Additional hint for dynamic pod columns
        pod_cols = [c for c in sorted_high_missing.index if c.startswith('pod_')]
        if pod_cols:
            logger.error(f"High-missing pod columns detected (sample): {pod_cols[:10]}")
        assert False
        
    logger.info("Processing training data...")
    
    
    data_encoder = DataEncoder(output_dir=output_dir)
    train_processed, prepare_for_encoding_overhead_summary = data_encoder.prepare_for_encoding(processed_df, sorted_all_pod_ids, request_features_train, HYPERPARAMETERS)
    data_encoder.save_processed_data(train_processed)
    
    logger.info("Data processing complete!")
    logger.info(f"Training data: {data_encoder.output_dir}")
    logger.info(f"Dataset shapes:")
    logger.info(f"  pod_features: {train_processed['pod_features'].shape}")
    logger.info(f"  pod_features_with_staleness: {train_processed['pod_features_with_staleness'].shape}")
    logger.info(f"  kv_hit_ratios: {train_processed['kv_hit_ratios'].shape}")
    logger.info(f"  request_features: {train_processed['request_features'].shape}")
    logger.info(f"  positional_encodings: {train_processed['positional_encodings'].shape}")
    logger.info(f"  actions: {train_processed['actions'].shape}")
    logger.info(f"  rewards: {train_processed['rewards'].shape}")
    assert output_dir == data_encoder.output_dir


# Global cached processor instance for inference performance
_cached_encoder = None

def get_cached_data_encoder():
    """Get or create a cached processor instance for inference."""
    global _cached_encoder
    if _cached_encoder is None:
        _cached_encoder = DataEncoder(output_dir="temp_inference")
    return _cached_encoder

def encode_for_inference(sorted_all_pod_ids, processed_df, request_features_train, HYPERPARAMETERS):
    prepare_for_encoding_start = time.time()
    data_encoder = get_cached_data_encoder()
    processed_data, prepare_for_encoding_overhead_summary = data_encoder.prepare_for_encoding(processed_df, sorted_all_pod_ids, request_features_train, HYPERPARAMETERS)
    
    overhead_summary = {}
    overhead_summary['prepare_for_encoding'] = time.time() - prepare_for_encoding_start
    for key, val in prepare_for_encoding_overhead_summary.items():
        overhead_summary[f"prepare_for_encoding.{key}"] = val
    
    post_process_start_time = time.time()
    
    # OPTIMIZATION 2: Batch tensor conversion for better performance
    tensor_data = {}
    
    # Define conversion mappings for batch processing
    float_conversions = {
        'pod_features': processed_data['pod_features'],
        'kv_hit_ratios': processed_data['kv_hit_ratios'],
        'request_features': processed_data['request_features'],
        'rewards': processed_data['rewards'],
        'positional_encodings': processed_data['positional_encodings'],
        'pod_features_with_staleness': processed_data['pod_features_with_staleness'],
        'query': processed_data['cross_attention_inputs']['query'],
        'key_value': processed_data['cross_attention_inputs']['key_value']
    }
    
    # Optional float tensors
    ttft_rewards = processed_data.get('ttft_rewards')
    if ttft_rewards is not None:
        float_conversions['ttft_rewards'] = ttft_rewards
    tpot_rewards = processed_data.get('tpot_rewards')
    if tpot_rewards is not None:
        float_conversions['tpot_rewards'] = tpot_rewards
    
    # Batch convert float tensors
    for key, array in float_conversions.items():
        tensor_data[key] = torch.from_numpy(array).float()
    
    # Convert long tensors separately
    tensor_data['actions'] = torch.from_numpy(processed_data['actions']).long()
    
    overhead_summary['post_process'] = time.time() - post_process_start_time
    
    # # Optional tensors
    # if processed_data['interaction_features'] is not None:
    #     tensor_data['interaction_features'] = torch.from_numpy(processed_data['interaction_features']).float()
        
    overhead_summary['end_to_end'] = time.time() - prepare_for_encoding_start
    return tensor_data, overhead_summary