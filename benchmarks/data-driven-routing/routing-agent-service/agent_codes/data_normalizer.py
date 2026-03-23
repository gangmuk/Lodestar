#!/usr/bin/env python3
"""
Comprehensive Data Normalization Module for LLM Routing System

This module provides all normalization functionality in one place:
- High-level interface for normalizing processed CSVs
- Low-level normalization logic and statistics management
- Automatic feature detection and normalization
- Reward calculation and SLO handling

Key features:
- Standalone normalization function taking processed CSV as input
- Configurable reward function calculation
- Preserves original data while adding normalized columns
- Automatic detection of normalizable features
- Comprehensive statistics management
"""

import pandas as pd
import numpy as np
import os
import time
import argparse
import pickle
import preprocess
from logger import logger
from typing import Tuple, Dict, Any, Set, List, FrozenSet
import json

# ============================================================================
# MODULE-LEVEL CACHES FOR PERFORMANCE OPTIMIZATION
# ============================================================================
# These caches avoid recomputing the same values on every request.
# They are keyed by immutable signatures (frozensets) of DataFrame columns.

# Cache for _get_normalizable_features: (columns_key, no_normalize_key) -> result
_NORMALIZABLE_FEATURES_CACHE: Dict[tuple, tuple] = {}

# Cache for matching_columns per feature_type: (columns_key, feature_type) -> list
_MATCHING_COLUMNS_CACHE: Dict[tuple, List[str]] = {}

# Cache for pod feature type extraction: column_name -> feature_type (or None)
_POD_FEATURE_TYPE_CACHE: Dict[str, str] = {}

def _get_columns_cache_key(df_columns) -> FrozenSet[str]:
    """Generate a cache key from DataFrame columns."""
    return frozenset(df_columns)

def clear_normalizer_caches():
    """Clear all module-level caches. Call when schema changes."""
    global _NORMALIZABLE_FEATURES_CACHE, _MATCHING_COLUMNS_CACHE, _POD_FEATURE_TYPE_CACHE
    _NORMALIZABLE_FEATURES_CACHE.clear()
    _MATCHING_COLUMNS_CACHE.clear()
    _POD_FEATURE_TYPE_CACHE.clear()
    logger.info("Cleared all normalizer caches")


class RunningStats:
    """Maintains running mean and standard deviation for feature normalization"""
    def __init__(self, feature_names):
        self.count = 0
        self.mean = None
        self.sum_sq_diff = None
        self.std = None
        self.min = None
        self.max = None
        if feature_names == None:
            logger.error("RunningStats initialized with None feature_names, setting to empty list")
            assert False
        self.feature_names = feature_names
        self.values = []
        
    def replace_stats_from_data(self, new_data):
        """Recompute stats from scratch using ONLY the provided data.

        Unlike update_stats_incrementally which accumulates, this replaces
        all statistics so they reflect exactly the given data.
        Use this when ONLINE_TRAIN_FROM_SCRATCH=1 to ensure normalization
        stats match only the current training window.
        """
        if new_data is None or len(new_data) == 0:
            logger.error("Received empty data for RunningStats.replace_stats_from_data, skipping")
            return
        new_data = np.array(new_data, dtype=np.float64)
        old_count = self.count
        old_mean = self.mean
        self.count = len(new_data)
        self.mean = np.mean(new_data, axis=0)
        self.sum_sq_diff = np.var(new_data, axis=0) * self.count
        self.std = np.sqrt(self.sum_sq_diff / self.count)
        self.min = np.min(new_data, axis=0)
        self.max = np.max(new_data, axis=0)
        logger.info(f"RunningStats.replace_stats_from_data for {self.feature_names}: "
                   f"replaced stats (old_count={old_count}, old_mean={old_mean}) "
                   f"with {self.count} samples, mean={self.mean}, std={self.std}")

    def update_stats_incrementally(self, new_data):
        if new_data is None or len(new_data) == 0:
            logger.error("Received empty data for RunningStats.update, skipping")
            return
        new_data = np.array(new_data, dtype=np.float64)
        new_count = len(new_data)
        old_mean = self.mean
        old_sum_sq_diff = self.sum_sq_diff
        old_std = self.std
        old_min = self.min
        old_max = self.max
        if self.count == 0: # The very first update
            self.mean = np.mean(new_data, axis=0)
            self.sum_sq_diff = np.var(new_data, axis=0) * new_count
            self.count = new_count
            self.std = np.sqrt(self.sum_sq_diff / new_count)
            if self.min is None or self.max is None:
                logger.warning(f"min/max were None for {self.feature_names} despite count={self.count}. Initializing...")
                self.min = np.min(new_data, axis=0)
                self.max = np.max(new_data, axis=0)
            else:
                self.min = np.minimum(self.min, np.min(new_data, axis=0))
                self.max = np.maximum(self.max, np.max(new_data, axis=0))
            logger.info(f"The very first RunningStats.update call for {self.feature_names}. Initialized running stats with {new_count} samples")
            return
        batch_mean = np.mean(new_data, axis=0)
        batch_var = np.var(new_data, axis=0) * new_count
        new_count = len(new_data)
        new_total = self.count + new_count
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * new_count / new_total
        self.sum_sq_diff = self.sum_sq_diff + batch_var + delta**2 * self.count * new_count / new_total
        self.std = np.sqrt(self.sum_sq_diff / new_total)
        self.count = new_total
        if self.min is None or self.max is None:
            logger.warning(f"min/max were None for {self.feature_names} despite count={self.count}. Initializing...")
            self.min = np.min(new_data, axis=0)
            self.max = np.max(new_data, axis=0)
        else:
            self.min = np.minimum(self.min, np.min(new_data, axis=0))
            self.max = np.maximum(self.max, np.max(new_data, axis=0))
        
    def normalize(self, data):
        if self.count == 0:
            logger.error(f"{self.feature_names}: No statistics available. normalization cannot be performed.")
            assert False
        mean = self.mean
        std = self.std
        
        # Handle zero standard deviation case (constant features)
        if np.any(std == 0) or np.any(np.isclose(std, 0, atol=1e-10)):
            logger.warning(f"{self.feature_names}: Zero standard deviation detected (std={std}). Returning zero-centered data.")
            return np.zeros_like(data)
        
        # Check for NaN in std
        if np.any(np.isnan(std)):
            logger.error(f"{self.feature_names}: NaN detected in standard deviation: {std}")
            return np.zeros_like(data)
        
        # Perform normalization
        normalized = (data - mean) / std
        
        # Verify result doesn't contain NaN
        if np.any(np.isnan(normalized)):
            logger.error(f"{self.feature_names}: Normalization produced NaN values. mean={mean}, std={std}")
            return np.zeros_like(data)
        
        return normalized


class FeatureStats:
    """Manages statistics for all features and provides normalization interface"""
    
    def __init__(self, feature_names=None):
        self.feature_stats = {}
        self.CONFIG = {
            "FEATURES_NORMALIZED": set(),
            "NUM_FEATURES_NORMALIZED": 0,
            "TOTAL_FEATURES": 0
        }
        if feature_names:
            self._initialize_stats(feature_names)
    
    def _initialize_stats(self, feature_names):
        """Initialize statistics for given feature names"""
        for feature in feature_names:
            if feature not in self.feature_stats:
                self.feature_stats[feature] = RunningStats(feature_names=feature)
        self.CONFIG["TOTAL_FEATURES"] = len(feature_names)
    
    def get_max_count(self):
        """Get maximum count across all features"""
        if not self.feature_stats:
            return 0
        return max(stats.count for stats in self.feature_stats.values())
    
    def get_feature_names(self):
        """Get list of all feature names with statistics"""
        return list(self.feature_stats.keys())
    
    def write_stats_to_file(self, filename):
        """
        Write feature statistics to a CSV file in long format (feature_name, stats_type, value).
        
        Only writes pooled statistics (e.g., 'kv_hit_ratio') without pod-specific variations
        (e.g., skip 'pod_0000-kv_hit_ratio', 'pod_0001-kv_hit_ratio').
        """
        
        stats_data = []
        seen_feature_names = set()  # Track what we've written to avoid duplicates
        
        for feature_name, stats in self.feature_stats.items():
            # Check if this is an individual pod feature (e.g., "pod_0000-kv_hit_ratio")
            feature_type = _extract_pod_feature_type(feature_name)
            
            if feature_type:
                # This is an individual pod feature - SKIP IT completely
                # We only want pooled stats (which are stored under the feature_type directly)
                logger.debug(f"Skipping individual pod feature {feature_name} (pooled as '{feature_type}')")
                continue
            
            # This is either a non-pod feature OR a pooled feature type
            save_name = feature_name
            
            # Skip if we've already written this feature
            if save_name in seen_feature_names:
                logger.warning(f"Duplicate feature name detected: {save_name}, skipping")
                continue
            seen_feature_names.add(save_name)
            
            # Add each stat type as a separate row
            stats_data.append({
                'feature_name': save_name,
                'stats_type': 'count',
                'value': stats.count
            })
            stats_data.append({
                'feature_name': save_name,
                'stats_type': 'mean',
                'value': stats.mean.item() if hasattr(stats.mean, 'item') else stats.mean
            })
            stats_data.append({
                'feature_name': save_name,
                'stats_type': 'std',
                'value': stats.std.item() if hasattr(stats.std, 'item') else stats.std
            })
            stats_data.append({
                'feature_name': save_name,
                'stats_type': 'min',
                'value': stats.min.item() if hasattr(stats.min, 'item') else stats.min
            })
            stats_data.append({
                'feature_name': save_name,
                'stats_type': 'max',
                'value': stats.max.item() if hasattr(stats.max, 'item') else stats.max
            })
            stats_data.append({
                'feature_name': save_name,
                'stats_type': 'sum_sq_diff',
                'value': stats.sum_sq_diff.item() if hasattr(stats.sum_sq_diff, 'item') else stats.sum_sq_diff
            })
        
        df = pd.DataFrame(stats_data)
        df.to_csv(filename, index=False)
        
        # Count saved features
        num_pod_types = len([f for f in seen_feature_names if not f.startswith('input_tokens') and not f.startswith('output_tokens') and not f.startswith('total_tokens')])
        num_non_pod = len(seen_feature_names) - num_pod_types
        logger.info(f"✅ Saved POOLED feature statistics to {filename}: {num_pod_types} pooled pod types + {num_non_pod} non-pod features = {len(seen_feature_names)} total")
    
    @classmethod
    def load_from_csv(cls, filename):
        """
        Load feature statistics from a CSV file in long format (feature_name, stats_type, value).
        
        Handles both old format (per-pod stats) and new format (pooled stats).
        For pooled stats, replicates them for all pods (pod_0000 through pod_0050).
        """
        
        if not os.path.exists(filename):
            logger.error(f"Statistics file not found: {filename}")
            return None
        
        try:
            df = pd.read_csv(filename)
            stats_instance = cls()
            
            # Separate pod features from non-pod features
            non_pod_features = []
            pod_feature_types = {}  # feature_type -> stats_dict (for pooled format)
            old_format_pod_features = {}  # feature_type -> list of (pod_id, stats_dict) (for old format)
            
            for feature_name in df['feature_name'].unique():
                if feature_name.startswith('pod_') and '-' in feature_name:
                    # OLD FORMAT: per-pod statistics (e.g., "pod_0000-kv_hit_ratio")
                    parts = feature_name.split('-', 1)
                    pod_id = parts[0]  # e.g., 'pod_0000'
                    feature_type = parts[1]  # e.g., 'kv_hit_ratio'
                    
                    if feature_type not in old_format_pod_features:
                        old_format_pod_features[feature_type] = []
                    
                    # Extract stats for this pod feature
                    feature_df = df[df['feature_name'] == feature_name]
                    stats_dict = {}
                    for _, row in feature_df.iterrows():
                        stats_dict[row['stats_type']] = row['value']
                    
                    old_format_pod_features[feature_type].append((pod_id, stats_dict))
                elif not feature_name.startswith('pod_') and any(kw in feature_name for kw in ['kv_hit_ratio', 'inflight_requests', 'gpu_kv_cache', 'cpu_kv_cache', 'running_requests', 'waiting_requests', 'prefill_tokens', 'decode_tokens', 'inflight_prefill_requests', 'inflight_decode_requests']):
                    # NEW FORMAT: pooled statistics (e.g., "kv_hit_ratio")
                    feature_df = df[df['feature_name'] == feature_name]
                    stats_dict = {}
                    for _, row in feature_df.iterrows():
                        stats_dict[row['stats_type']] = row['value']
                    pod_feature_types[feature_name] = stats_dict
                else:
                    # Non-pod feature (input_tokens, output_tokens, total_tokens)
                    non_pod_features.append(feature_name)
            
            # Load non-pod features as-is
            for feature_name in non_pod_features:
                feature_df = df[df['feature_name'] == feature_name]
                stats = RunningStats(feature_names=feature_name)
                
                for _, row in feature_df.iterrows():
                    stat_type = row['stats_type']
                    value = row['value']
                    
                    if stat_type == 'count':
                        stats.count = int(value)
                    elif stat_type == 'mean':
                        stats.mean = np.array([float(value)])
                    elif stat_type == 'std':
                        stats.std = np.array([float(value)])
                    elif stat_type == 'min':
                        stats.min = np.array([float(value)])
                    elif stat_type == 'max':
                        stats.max = np.array([float(value)])
                    elif stat_type == 'sum_sq_diff':
                        stats.sum_sq_diff = float(value)
                
                stats_instance.feature_stats[feature_name] = stats
            
            # Handle pod features based on format detected
            if pod_feature_types:
                # NEW FORMAT: Use pooled statistics directly
                logger.info(f"✅ Loading NEW FORMAT with {len(pod_feature_types)} pooled pod feature types")
                
                # Store the pooled stats with the base feature type name
                for feature_type, stats_dict in pod_feature_types.items():
                    stats = RunningStats(feature_names=feature_type)
                    
                    for stat_type, value in stats_dict.items():
                        if stat_type == 'count':
                            stats.count = int(value)
                        elif stat_type == 'mean':
                            stats.mean = np.array([float(value)])
                        elif stat_type == 'std':
                            stats.std = np.array([float(value)])
                        elif stat_type == 'min':
                            # Handle empty/NaN min
                            if pd.isna(value) or value == '':
                                stats.min = np.array([0.0])
                            else:
                                stats.min = np.array([float(value)])
                        elif stat_type == 'max':
                            # Handle empty/NaN max
                            if pd.isna(value) or value == '':
                                stats.max = np.array([0.0])
                            else:
                                stats.max = np.array([float(value)])
                        elif stat_type == 'sum_sq_diff':
                            stats.sum_sq_diff = float(value)
                    
                    # Store with base feature type name (used during normalization)
                    stats_instance.feature_stats[feature_type] = stats
                    
                    logger.info(f"   Pooled '{feature_type}': mean={stats.mean[0]:.3f}, std={stats.std[0]:.3f}")
                
            elif old_format_pod_features:
                # OLD FORMAT: Need to create unified statistics
                logger.info(f"⚠️  Loading OLD FORMAT with per-pod statistics for {len(old_format_pod_features)} feature types")
                logger.info(f"   Converting to pooled format...")
                
                for feature_type, pod_stats_list in old_format_pod_features.items():
                    # Aggregate statistics across all pods
                    counts, means, stds, mins, maxs, sum_sq_diffs = [], [], [], [], [], []
                    
                    for pod_id, stats_dict in pod_stats_list:
                        try:
                            counts.append(float(stats_dict['count']))
                            means.append(float(stats_dict['mean']))
                            stds.append(float(stats_dict['std']))
                            
                            min_val = stats_dict.get('min', 0.0)
                            max_val = stats_dict.get('max', 0.0)
                            if pd.isna(min_val) or min_val == '':
                                min_val = 0.0
                            if pd.isna(max_val) or max_val == '':
                                max_val = 0.0
                            mins.append(float(min_val))
                            maxs.append(float(max_val))
                            
                            sum_sq_diffs.append(float(stats_dict['sum_sq_diff']))
                        except (ValueError, KeyError) as e:
                            logger.warning(f"Skipping invalid stats for {pod_id}-{feature_type}: {e}")
                            continue
                    
                    if not counts:
                        logger.warning(f"No valid stats found for feature type: {feature_type}")
                        continue
                    
                    # Create pooled statistics
                    stats = RunningStats(feature_names=feature_type)
                    stats.count = int(np.mean(counts))
                    stats.mean = np.array([np.mean(means)])
                    stats.std = np.array([np.mean(stds)])
                    stats.min = np.array([np.min(mins)])
                    stats.max = np.array([np.max(maxs)])
                    stats.sum_sq_diff = np.mean(sum_sq_diffs)
                    
                    # Store with base feature type name
                    stats_instance.feature_stats[feature_type] = stats
                    
                    logger.info(f"   Pooled '{feature_type}': mean={stats.mean[0]:.3f}, std={stats.std[0]:.3f}")
            
            stats_instance.CONFIG["TOTAL_FEATURES"] = len(stats_instance.feature_stats)
            stats_instance.CONFIG["NUM_FEATURES_NORMALIZED"] = len(stats_instance.feature_stats)
            stats_instance.CONFIG["FEATURES_NORMALIZED"] = set(stats_instance.feature_stats.keys())
            
            logger.info(f"✅ Loaded feature statistics from {filename}")
            logger.info(f"   - Non-pod features: {len(non_pod_features)}")
            logger.info(f"   - Pod feature types (pooled): {len(pod_feature_types) or len(old_format_pod_features)}")
            logger.info(f"   - Total features in stats_instance: {len(stats_instance.feature_stats)}")
            
            return stats_instance
            
        except Exception as e:
            logger.error(f"Failed to load statistics from {filename}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None


def _extract_pod_feature_type(column_name: str) -> str:
    """
    Extract feature type from pod column name (cached).

    Args:
        column_name: e.g., "pod_0000-kv_hit_ratio"

    Returns:
        str: Feature type (e.g., "kv_hit_ratio") or None if not a pod feature
    """
    # Check cache first
    if column_name in _POD_FEATURE_TYPE_CACHE:
        return _POD_FEATURE_TYPE_CACHE[column_name]

    # Compute and cache
    result = None
    if column_name.startswith('pod_') and '-' in column_name:
        parts = column_name.split('-', 1)
        if len(parts) == 2:
            result = parts[1]  # Return the feature type

    _POD_FEATURE_TYPE_CACHE[column_name] = result
    return result


def _get_normalizable_features(processed_df, no_normalize_features: list):
    """
    Automatically detect which features can be normalized (cached).

    Returns:
        tuple: (normalizable_features, non_normalizable_features, pod_feature_types)

    NOTE: normalizable_features now includes POOLED feature types (e.g., "kv_hit_ratio")
          instead of individual pod features (e.g., "pod_0000-kv_hit_ratio") to avoid
          creating duplicate statistics entries.

    OPTIMIZATION: Results are cached based on column signature + no_normalize_features.
    """
    # OPTIMIZATION: Handle both dict (single-row inference) and DataFrame inputs
    is_dict_input = isinstance(processed_df, dict)
    columns = list(processed_df.keys()) if is_dict_input else processed_df.columns

    # Create cache key from columns and exclusion list
    columns_key = _get_columns_cache_key(columns)
    no_normalize_key = frozenset(no_normalize_features) if no_normalize_features else frozenset()
    cache_key = (columns_key, no_normalize_key)

    # Check cache
    if cache_key in _NORMALIZABLE_FEATURES_CACHE:
        return _NORMALIZABLE_FEATURES_CACHE[cache_key]

    # Compute (same logic as before)
    normalizable_features = ['input_tokens', 'output_tokens', 'total_tokens', 'kv_concentration']
    pod_feature_types = set()  # Track unique pod feature types (e.g., "kv_hit_ratio")
    all_pod_columns = []  # Track all pod columns for validation

    for col in columns:
        if col.startswith('pod_') and 'gpu_model' not in col and 'GPU' not in col:
            feature_type = _extract_pod_feature_type(col)
            if feature_type and feature_type not in no_normalize_features:
                # Don't add individual pod columns to normalizable_features
                # Instead, just track the feature type
                pod_feature_types.add(feature_type)
                all_pod_columns.append(col)
            else:
                logger.debug(f"Excluding {col} from normalization. Feature type: {feature_type}, no_normalize_features: {no_normalize_features}")

    # Add pooled feature types to normalizable_features (instead of individual pod features)
    normalizable_features.extend(sorted(pod_feature_types))

    # Find non-normalizable features
    all_pod_columns_set = set(all_pod_columns)
    normalizable_set = set(normalizable_features)
    non_normalizable_features = [
        col for col in columns
        if col not in all_pod_columns_set and col not in normalizable_set
    ]

    logger.debug(f"Found {len(pod_feature_types)} unique pod feature types: {sorted(pod_feature_types)}")
    logger.debug(f"Normalizable features: {len(normalizable_features)} (including {len(pod_feature_types)} pooled pod types)")

    # Cache the result (convert pod_feature_types to frozenset for immutability)
    result = (normalizable_features, non_normalizable_features, pod_feature_types)
    _NORMALIZABLE_FEATURES_CACHE[cache_key] = result

    return result


def _get_matching_columns_cached(df_columns, feature_type: str) -> List[str]:
    """
    Get all pod columns matching a feature type (cached).

    Args:
        df_columns: DataFrame columns (used for cache key)
        feature_type: e.g., "kv_hit_ratio"

    Returns:
        List of column names matching the feature type
    """
    columns_key = _get_columns_cache_key(df_columns)
    cache_key = (columns_key, feature_type)

    if cache_key in _MATCHING_COLUMNS_CACHE:
        return _MATCHING_COLUMNS_CACHE[cache_key]

    # Compute matching columns
    matching = [col for col in df_columns if _extract_pod_feature_type(col) == feature_type]
    _MATCHING_COLUMNS_CACHE[cache_key] = matching

    return matching


def _compute_pooled_pod_statistics(processed_df, pod_feature_types, stats_instance, update_statistics=True, recompute_from_scratch=False):
    """
    Compute pooled statistics for pod features by aggregating across all pods.

    Args:
        processed_df: DataFrame with pod features
        pod_feature_types: Set of unique pod feature types (e.g., {"kv_hit_ratio", "inflight_requests"})
        stats_instance: FeatureStats instance to store the pooled statistics
        update_statistics: If False, skip updating stats (used during online training to prevent distribution shift)
        recompute_from_scratch: If True, replace stats entirely from the current data
            (use when ONLINE_TRAIN_FROM_SCRATCH=1 so stats reflect only the current training window).
            If False, accumulate incrementally (use when ONLINE_TRAIN_FROM_SCRATCH=0).

    Returns:
        dict: Mapping from feature_type to pooled stats
    """
    pooled_stats = {}

    for feature_type in pod_feature_types:
        # OPTIMIZATION: Use cached matching columns lookup
        matching_columns = _get_matching_columns_cached(processed_df.columns, feature_type)

        if not matching_columns:
            continue

        # Pool all values from all pods for this feature type
        all_values = []
        for col in matching_columns:
            values = processed_df[col].values
            all_values.extend(values)

        all_values = np.array(all_values, dtype=np.float64).reshape(-1, 1)

        logger.info(f"🔄 Pooling {len(matching_columns)} pod columns for feature type '{feature_type}' "
                   f"({len(all_values)} total samples)")

        # Create or update pooled statistics for this feature type
        if feature_type not in stats_instance.feature_stats:
            # First time seeing this feature - initialize it even if update_statistics=False
            stats_instance.feature_stats[feature_type] = RunningStats(feature_names=feature_type)
            # Always update stats for new features
            stats_instance.feature_stats[feature_type].update_stats_incrementally(all_values)
            logger.info(f"🆕 Initialized NEW pooled stats for '{feature_type}'")
        elif update_statistics:
            if recompute_from_scratch:
                # Replace stats entirely from current data (no accumulation from previous rounds)
                stats_instance.feature_stats[feature_type].replace_stats_from_data(all_values)
                logger.info(f"🔄 REPLACED pooled stats for '{feature_type}' from scratch (current window only)")
            else:
                # Incremental update (accumulate on top of existing stats)
                stats_instance.feature_stats[feature_type].update_stats_incrementally(all_values)
                logger.info(f"🔄 Updated pooled stats for '{feature_type}' (incremental)")
        else:
            # Feature exists but we're NOT allowed to update (online training freeze)
            logger.info(f"❄️  FROZE pooled stats for '{feature_type}' (online training mode)")

        pooled_stats[feature_type] = stats_instance.feature_stats[feature_type]

        logger.info(f"✅ Pooled stats for '{feature_type}': mean={pooled_stats[feature_type].mean[0]:.3f}, "
                   f"std={pooled_stats[feature_type].std[0]:.3f}, "
                   f"min={pooled_stats[feature_type].min[0]:.3f}, "
                   f"max={pooled_stats[feature_type].max[0]:.3f}")

    return pooled_stats


def _normalize_single_feature(processed_df, feature, stats_instance, update_statistics, request_id=None, recompute_from_scratch=False):
    """
    Normalize a single feature using the provided statistics.

    For pod features (e.g., "pod_0000-kv_hit_ratio"), this uses POOLED statistics
    from the base feature type (e.g., "kv_hit_ratio") to ensure all pods use
    the same normalization parameters.

    Args:
        processed_df: DataFrame containing the feature
        feature: Feature name to normalize
        stats_instance: FeatureStats instance containing normalization statistics
        update_statistics: Whether this is training (True) or inference (False)
        request_id: Optional request ID for logging
        recompute_from_scratch: If True, replace stats entirely from the current data
            (use when ONLINE_TRAIN_FROM_SCRATCH=1). If False, accumulate incrementally.
    """
    log_prefix = f"request_id,{request_id}," if request_id else ""
    
    if feature not in processed_df.columns:
        logger.error(f"{log_prefix}Feature {feature} not found in DataFrame")
        assert False
    
    # Determine which stats to use (pooled for pod features, individual for others)
    feature_type = _extract_pod_feature_type(feature)
    if feature_type:
        # This is a pod feature - use pooled statistics
        stats_key = feature_type
        logger.debug(f"{log_prefix}Using pooled stats '{stats_key}' for pod feature '{feature}'")
    else:
        # Regular feature - use individual statistics
        stats_key = feature
        
    if update_statistics:
        # NOTE: For pod features, statistics are computed in _compute_pooled_pod_statistics
        # This function only normalizes using the pre-computed pooled stats
        if feature_type:
            # Pod features: Use pooled statistics (already computed)
            if stats_key not in stats_instance.feature_stats:
                logger.error(f"{log_prefix}Pooled stats for '{stats_key}' not found. "
                           f"Call _compute_pooled_pod_statistics first!")
                assert False
        else:
            # Non-pod features: Compute individual statistics
            feature_std = processed_df[feature].values.std()
            
            # Initialize stats if needed
            if stats_key not in stats_instance.feature_stats:
                stats_instance.feature_stats[stats_key] = RunningStats(feature_names=stats_key)
            
            if feature_std == 0 or np.isclose(feature_std, 0, atol=1e-10):
                # Handle constant feature
                feature_data = processed_df[feature].values.reshape(-1, 1)
                
                # Set stats manually for constant feature
                stats_instance.feature_stats[stats_key].count = len(feature_data)
                stats_instance.feature_stats[stats_key].mean = np.mean(feature_data, axis=0)
                stats_instance.feature_stats[stats_key].std = np.array([0.0])  # Mark as constant
                stats_instance.feature_stats[stats_key].sum_sq_diff = 0.0  # No variance
                
                logger.info(f"⚪ {feature}, Saved as constant feature (std={feature_std:.6f}, value={stats_instance.feature_stats[stats_key].mean})")
                
                return  # Skip normalization but stats are saved
                
            # Check for NaN values in the feature
            if np.any(np.isnan(processed_df[feature].values)):
                logger.error(f"❌ {feature}: Contains NaN values before normalization")
                assert False
        
            # Normal feature processing (non-constant) - update stats
            logger.info(f"🔍 {feature}, Normalizing. Variance is high (std: {processed_df[feature].values.std():.3f})")
            stats_instance.CONFIG.setdefault("FEATURES_NORMALIZED", set()).add(stats_key)
            stats_instance.CONFIG["NUM_FEATURES_NORMALIZED"] = len(stats_instance.CONFIG["FEATURES_NORMALIZED"])
            
            # Update statistics for non-pod features
            feature_data = processed_df[feature].values.reshape(-1, 1)
            prev_std = processed_df[feature].values.std()
            prev_min = processed_df[feature].values.min()
            prev_max = processed_df[feature].values.max()
            prev_mean = processed_df[feature].values.mean()

            ##############################################
            if recompute_from_scratch:
                stats_instance.feature_stats[stats_key].replace_stats_from_data(feature_data)
            else:
                stats_instance.feature_stats[stats_key].update_stats_incrementally(feature_data)
            ##############################################
            
            # Verify computed std is valid
            computed_std = stats_instance.feature_stats[stats_key].std
            if np.any(computed_std == 0) or np.any(np.isnan(computed_std)):
                logger.warning(f"⚠️  {feature}: Invalid computed std ({computed_std}), skipping normalization")
                return
        
        # Apply normalization (for both pod and non-pod features)
        feature_data = processed_df[feature].values.reshape(-1, 1)
        prev_mean = processed_df[feature].values.mean()
        prev_std = processed_df[feature].values.std()
        
        ##############################################
        # Use pooled stats for pod features, individual stats for others
        normalized_feature = stats_instance.feature_stats[stats_key].normalize(feature_data)
        ##############################################
        
        # Verify normalized data doesn't contain NaN
        if np.any(np.isnan(normalized_feature)):
            logger.error(f"❌ {feature}: Normalization produced NaN values, skipping")
            return
        
        processed_df[feature] = normalized_feature.flatten()
        
        new_std = processed_df[feature].std()
        if new_std <= 0.5 and not feature_type:  # Only warn for non-pod features
            logger.warning(f"⚠️  Post-normalization variance too low for {feature} (std: {new_std:.3f})")
        
        if feature_type:
            logger.info(f"✅ {feature} → POOLED '{stats_key}', prev mean: {prev_mean:.3f}, new mean: {normalized_feature.mean():.3f}")
        else:
            logger.info(f"✅ {feature}, prev std: {prev_std:.3f}, new std: {new_std:.3f}")
        
    else:  # Inference
        if stats_key not in stats_instance.feature_stats:
            logger.error(f"{log_prefix}Stats key '{stats_key}' not found for feature '{feature}'")
            logger.error(f"Available features: {list(stats_instance.feature_stats.keys())[:20]}...")
            logger.error(f"This indicates a training/inference feature mismatch")
            assert False
            
        # Check if this was a constant feature during training
        if hasattr(stats_instance.feature_stats[stats_key], 'std') and np.allclose(stats_instance.feature_stats[stats_key].std, 0):
            logger.debug(f"{log_prefix}{feature} was constant during training (value={stats_instance.feature_stats[stats_key].mean}) - skipping normalization")
            return  # Don't normalize constant features
        
        # Apply normalization using stored statistics (pooled for pod features)
        feature_data = processed_df[feature].values.reshape(-1, 1)
        normalized_feature = stats_instance.feature_stats[stats_key].normalize(feature_data)
        
        # Verify normalized data doesn't contain NaN
        if np.any(np.isnan(normalized_feature)):
            logger.error(f"{log_prefix}❌ {feature}: Normalization produced NaN values, skipping")
            return
        
        processed_df[feature] = normalized_feature.flatten()
        if feature_type:
            logger.debug(f"{log_prefix}✅ {feature} → POOLED '{stats_key}'")
        else:
            logger.debug(f"{log_prefix}✅ {feature}, Normalized using stored stats")


def normalize_features_batch_inference(processed_df, stats_instance, normalizable_features, pod_feature_types, request_id=None):
    """
    Batch normalize all features for inference (optimized).

    This function avoids the overhead of calling _normalize_single_feature in a loop by:
    1. Pre-computing all stats lookups once
    2. Using vectorized operations where possible
    3. Leveraging cached matching columns

    Args:
        processed_df: DataFrame or dict to normalize (modified in place)
        stats_instance: FeatureStats instance with normalization statistics
        normalizable_features: List of features to normalize (from _get_normalizable_features)
        pod_feature_types: Set of pod feature types (from _get_normalizable_features)
        request_id: Optional request ID for logging

    Returns:
        None (modifies processed_df in place)
    """
    log_prefix = f"request_id,{request_id}," if request_id else ""

    # OPTIMIZATION: Handle both dict (single-row inference) and DataFrame inputs
    is_dict_input = isinstance(processed_df, dict)
    columns = list(processed_df.keys()) if is_dict_input else processed_df.columns

    # Pre-fetch all needed stats objects to avoid repeated dict lookups
    stats_cache = {}
    for feature_type in pod_feature_types:
        if feature_type in stats_instance.feature_stats:
            stats_cache[feature_type] = stats_instance.feature_stats[feature_type]
    for feature in normalizable_features:
        if feature not in pod_feature_types and feature in stats_instance.feature_stats:
            stats_cache[feature] = stats_instance.feature_stats[feature]

    # Process pod features by type (batch all columns of same type together)
    for feature_type in pod_feature_types:
        if feature_type not in stats_cache:
            logger.error(f"{log_prefix}Stats for pod feature type '{feature_type}' not found")
            logger.error(f"{log_prefix}This indicates a training/inference feature mismatch")
            assert False  # Fail fast - same as original _normalize_single_feature

        stats = stats_cache[feature_type]

        # Skip constant features (same check as original)
        if hasattr(stats, 'std') and np.allclose(stats.std, 0):
            logger.debug(f"{log_prefix}Skipping constant pod feature type '{feature_type}'")
            continue

        # Get all columns for this feature type (cached)
        matching_columns = _get_matching_columns_cached(columns, feature_type)

        if not matching_columns:
            continue

        # Pre-fetch stats values for vectorized normalization
        mean_val = stats.mean[0] if hasattr(stats.mean, '__len__') else stats.mean
        std_val = stats.std[0] if hasattr(stats.std, '__len__') else stats.std

        if std_val == 0 or np.isnan(std_val):
            logger.debug(f"{log_prefix}Skipping pod feature type '{feature_type}' (std=0 or NaN)")
            continue

        # Normalize all matching columns
        for col in matching_columns:
            if col in columns:
                if is_dict_input:
                    raw_val = processed_df[col]
                    normalized_val = (raw_val - mean_val) / std_val
                    if np.isnan(normalized_val):
                        logger.error(f"{log_prefix}❌ {col}: Normalization produced NaN values, skipping")
                        continue
                    processed_df[col] = normalized_val
                else:
                    raw_val = processed_df[col].values
                    normalized_val = (raw_val - mean_val) / std_val
                    # NaN check - same as original _normalize_single_feature
                    if np.any(np.isnan(normalized_val)):
                        logger.error(f"{log_prefix}❌ {col}: Normalization produced NaN values, skipping")
                        continue
                    processed_df[col] = normalized_val

    # Process non-pod features
    for feature in normalizable_features:
        if feature in pod_feature_types:
            continue  # Already processed above

        if feature not in stats_cache:
            logger.error(f"{log_prefix}Stats for feature '{feature}' not found")
            logger.error(f"{log_prefix}This indicates a training/inference feature mismatch")
            assert False  # Fail fast - same as original _normalize_single_feature

        if feature not in columns:
            logger.warning(f"{log_prefix}Feature '{feature}' not in DataFrame/dict")
            continue

        stats = stats_cache[feature]

        # Skip constant features (same check as original)
        if hasattr(stats, 'std') and np.allclose(stats.std, 0):
            logger.debug(f"{log_prefix}Skipping constant feature '{feature}'")
            continue

        # Pre-fetch stats values
        mean_val = stats.mean[0] if hasattr(stats.mean, '__len__') else stats.mean
        std_val = stats.std[0] if hasattr(stats.std, '__len__') else stats.std

        if std_val == 0 or np.isnan(std_val):
            logger.debug(f"{log_prefix}Skipping feature '{feature}' (std=0 or NaN)")
            continue

        # Normalize
        if is_dict_input:
            raw_val = processed_df[feature]
            normalized_val = (raw_val - mean_val) / std_val
            if np.isnan(normalized_val):
                logger.error(f"{log_prefix}❌ {feature}: Normalization produced NaN values, skipping")
                continue
            processed_df[feature] = normalized_val
        else:
            raw_val = processed_df[feature].values
            normalized_val = (raw_val - mean_val) / std_val
            # NaN check - same as original _normalize_single_feature
            if np.any(np.isnan(normalized_val)):
                logger.error(f"{log_prefix}❌ {feature}: Normalization produced NaN values, skipping")
                continue
            processed_df[feature] = normalized_val

    logger.debug(f"{log_prefix}Batch normalized {len(normalizable_features)} features + {len(pod_feature_types)} pod types")


def normalize_processed_data(processed_csv_file, output_csv_file=None,
                           reward_function='linear_simple', stats_file=None, hyperparameters=None,
                           fixed_range_bounds=None):
    """
    Normalize processed CSV data and calculate rewards using specified function.
    
    Args:
        processed_csv_file: Path to processed CSV file with raw values
        output_csv_file: Path for output normalized CSV (optional)
        reward_function: Reward function to use ('linear_simple', 'linear_simple_extended', 'piecewise_linear_steeper_gradient')
        stats_file: Path to save/load normalization statistics
        hyperparameters: Model hyperparameters dict (should contain TTFT_SLO and AVG_TPOT_SLO)
        
    Returns:
        tuple: (normalized_df, stats_instance, summary)
    """
    start_time = time.time()
    logger.info(f"Normalizing processed data: {processed_csv_file}")
    
    # Step 1: Load processed data
    if not os.path.exists(processed_csv_file):
        raise FileNotFoundError(f"Processed CSV file not found: {processed_csv_file}")
    
    df = pd.read_csv(processed_csv_file)
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    
    # Step 2: Extract SLO values from hyperparameters or metadata
    ttft_slo = hyperparameters['TTFT_SLO']
    avg_tpot_slo = hyperparameters['AVG_TPOT_SLO']
    ttft_reward_weight = hyperparameters['TTFT_REWARD_WEIGHT']
    
    # Step 3: Calculate rewards using specified function
    logger.info(f"Calculating rewards using function: {reward_function}")
    ttft_values = df['ttft'].values
    tpot_values = df['avg_tpot'].values
    
    if reward_function == 'linear_simple':
        reward_result = preprocess.calculate_rewards_simple(ttft_values, tpot_values, ttft_slo, avg_tpot_slo, ttft_reward_weight)
    elif reward_function == 'linear_simple_extended':
        reward_result = preprocess.calculate_rewards_simple_extended(ttft_values, tpot_values, ttft_slo, avg_tpot_slo, ttft_reward_weight)
    elif reward_function == 'piecewise_linear_steeper_gradient':
        reward_result = preprocess.calculate_rewards_piecewise_linear_steeper_gradient(ttft_values, tpot_values, ttft_slo, avg_tpot_slo, ttft_reward_weight)
    elif reward_function == 'gradual_within_slo':
        reward_result = preprocess.calculate_rewards_gradual_within_slo(ttft_values, tpot_values, ttft_slo, avg_tpot_slo, ttft_reward_weight)
    elif reward_function == 'inverse_latency':
        reward_result = preprocess.calculate_rewards_inverse_latency(ttft_values, tpot_values, ttft_slo, avg_tpot_slo, ttft_reward_weight)
    elif reward_function == 'latency_optimized':
        reward_result = preprocess.calculate_rewards_latency_optimization(ttft_values, tpot_values, ttft_slo, avg_tpot_slo, ttft_reward_weight)
    elif reward_function == 'simple_latency_minimization':
        reward_result = preprocess.calculate_rewards_simple_latency_minimization(
            ttft_values, tpot_values, ttft_reward_weight
        )
    elif reward_function == 'negative_reciprocal':
        reward_result = preprocess.calculate_rewards_negative_reciprocal(
            ttft_values, tpot_values, ttft_reward_weight
        )
    elif reward_function == 'negative_linear':
        reward_result = preprocess.calculate_rewards_negative_linear(
            ttft_values, tpot_values, ttft_reward_weight
        )
    elif reward_function == 'negative_squared':
        reward_result = preprocess.calculate_rewards_negative_squared(
            ttft_values, tpot_values, ttft_reward_weight
        )
    elif reward_function == 'quantile_based':
        # Check if input_tokens and output_tokens are available
        if 'input_tokens' in df.columns and 'output_tokens' in df.columns:
            input_tokens = df['input_tokens'].values
            output_tokens = df['output_tokens'].values
            reward_result = preprocess.calculate_rewards_quantile_based(
                ttft_values, tpot_values, input_tokens, output_tokens, ttft_reward_weight
            )
        else:
            logger.error("quantile_based reward function requires input_tokens and output_tokens columns")
            logger.error("Falling back to latency_optimized for post-processing")
            reward_result = preprocess.calculate_rewards_latency_optimization(ttft_values, tpot_values, ttft_slo, avg_tpot_slo, ttft_reward_weight)
    elif reward_function == 'absolute_latency':
        logger.info(f"Calculating absolute latency rewards (transferable across distributions)")
        reward_result = preprocess.calculate_rewards_absolute_latency(
            ttft_values, tpot_values,
            ttft_slo=hyperparameters.get('TTFT_SLO', 15000),
            tpot_slo=hyperparameters.get('AVG_TPOT_SLO', 100),
            ttft_reward_weight=ttft_reward_weight
        )
    elif reward_function == 'throughput_based':
        logger.info(f"Calculating throughput-based rewards (context-aware, input-length agnostic)")
        if 'input_tokens' in df.columns:
            input_tokens = df['input_tokens'].values
            reward_result = preprocess.calculate_rewards_throughput_based(
                ttft_values, tpot_values, input_tokens, ttft_reward_weight
            )
        else:
            logger.error("throughput_based reward function requires input_tokens column")
            logger.error("Falling back to simple_latency_minimization for post-processing")
            reward_result = preprocess.calculate_rewards_simple_latency_minimization(ttft_values, tpot_values, ttft_reward_weight)
    elif reward_function == 'log_normalized':
        logger.info(f"Calculating variance-normalized log rewards")
        if 'TTFT_P99' in hyperparameters and 'TPOT_P99' in hyperparameters:
            ttft_p99 = hyperparameters['TTFT_P99']
            tpot_p99 = hyperparameters['TPOT_P99']
            logger.info(f"Using TTFT_P99={ttft_p99:.2f}ms and TPOT_P99={tpot_p99:.2f}ms from hyperparameters")
        else:
            # Compute P99 values from the data itself if not in hyperparameters
            ttft_p99 = float(df['ttft'].quantile(0.99))
            tpot_p99 = float(df['avg_tpot'].quantile(0.99))
            logger.info(f"TTFT_P99 and TPOT_P99 not in hyperparameters, computing from data: TTFT_P99={ttft_p99:.2f}ms, TPOT_P99={tpot_p99:.2f}ms")
            # Store in hyperparameters for later use
            hyperparameters['TTFT_P99'] = ttft_p99
            hyperparameters['TPOT_P99'] = tpot_p99
        
        # Validate and fix P99 values
        # If TPOT_P99 is 0 (all TPOT values are 0), use a minimum value to avoid division by zero
        if tpot_p99 <= 0:
            logger.warning(f"TPOT_P99 is {tpot_p99} (all TPOT values are 0). Using minimum value of 1.0 for log calculation.")
            tpot_p99 = 1.0
            hyperparameters['TPOT_P99'] = tpot_p99
        
        if ttft_p99 <= 0:
            logger.error(f"Invalid TTFT_P99 value: {ttft_p99}. Cannot compute log_normalized rewards.")
            raise ValueError(f"TTFT_P99 must be positive for log_normalized reward function. Got TTFT_P99={ttft_p99}")
        
        reward_result = preprocess.calculate_rewards_log_normalized(
            ttft_values, tpot_values,
            ttft_p99=ttft_p99,
            tpot_p99=tpot_p99,
            ttft_reward_weight=ttft_reward_weight
        )
        
        # Validate reward result
        if reward_result is None:
            raise ValueError("Reward calculation returned None")
        if 'ttft_rewards' not in reward_result or 'tpot_rewards' not in reward_result or 'combined_rewards' not in reward_result:
            raise ValueError(f"Reward calculation returned invalid structure: {reward_result.keys()}")
        
        # Check for NaN values
        ttft_rewards = reward_result['ttft_rewards']
        tpot_rewards = reward_result['tpot_rewards']
        combined_rewards = reward_result['combined_rewards']
        
        if np.isnan(ttft_rewards).any() or np.isnan(tpot_rewards).any() or np.isnan(combined_rewards).any():
            nan_count_ttft = np.isnan(ttft_rewards).sum()
            nan_count_tpot = np.isnan(tpot_rewards).sum()
            nan_count_combined = np.isnan(combined_rewards).sum()
            logger.error(f"Reward calculation produced NaN values: ttft_rewards={nan_count_ttft}/{len(ttft_rewards)}, tpot_rewards={nan_count_tpot}/{len(tpot_rewards)}, combined_rewards={nan_count_combined}/{len(combined_rewards)}")
            logger.error(f"TTFT stats: min={np.nanmin(ttft_values):.2f}, max={np.nanmax(ttft_values):.2f}, p99={ttft_p99:.2f}")
            logger.error(f"TPOT stats: min={np.nanmin(tpot_values):.2f}, max={np.nanmax(tpot_values):.2f}, p99={tpot_p99:.2f}")
            raise ValueError("Reward calculation produced NaN values. Check input data and P99 values.")
        
        logger.info(f"Reward calculation successful: ttft_rewards range=[{np.min(ttft_rewards):.4f}, {np.max(ttft_rewards):.4f}], tpot_rewards range=[{np.min(tpot_rewards):.4f}, {np.max(tpot_rewards):.4f}], combined_rewards range=[{np.min(combined_rewards):.4f}, {np.max(combined_rewards):.4f}]")
    elif reward_function == 'quantile_advantage':
        if 'input_tokens' in df.columns:
            input_tokens = df['input_tokens'].values
            reward_result = preprocess.calculate_rewards_quantile_advantage(
                ttft_values, input_tokens, num_buckets=5
            )
        else:
            logger.error("quantile_advantage reward function requires input_tokens column")
            logger.error("Falling back to negative_linear for post-processing")
            reward_result = preprocess.calculate_rewards_negative_linear(ttft_values, tpot_values, ttft_reward_weight)
    elif reward_function == 'context_aware':
        logger.error("context_aware reward function requires detailed context data (input_tokens, kv_cache_hit_ratios) not available in this tool")
        logger.error("Falling back to latency_optimized for post-processing")
        reward_result = preprocess.calculate_rewards_latency_optimization(ttft_values, tpot_values, ttft_slo, avg_tpot_slo, ttft_reward_weight)
    else:
        logger.error(f"Unknown reward function: {reward_function}")
        raise ValueError(f"Unknown reward function: {reward_function}")
    
    # Add reward columns to dataframe
    # Validate array lengths match dataframe length
    expected_length = len(df)
    ttft_rewards = reward_result['ttft_rewards']
    tpot_rewards = reward_result['tpot_rewards']
    combined_rewards = reward_result['combined_rewards']
    
    if len(ttft_rewards) != expected_length:
        raise ValueError(f"ttft_rewards length mismatch: expected {expected_length}, got {len(ttft_rewards)}")
    if len(tpot_rewards) != expected_length:
        raise ValueError(f"tpot_rewards length mismatch: expected {expected_length}, got {len(tpot_rewards)}")
    if len(combined_rewards) != expected_length:
        raise ValueError(f"combined_rewards length mismatch: expected {expected_length}, got {len(combined_rewards)}")
    
    df['ttft_reward'] = ttft_rewards
    df['tpot_reward'] = tpot_rewards
    df['reward'] = combined_rewards
    
    # Final validation - check that columns were added correctly
    if 'ttft_reward' not in df.columns or 'tpot_reward' not in df.columns or 'reward' not in df.columns:
        raise ValueError(f"Failed to add reward columns. DataFrame columns: {df.columns.tolist()}")
    
    # Check for NaN values after assignment
    nan_ttft = df['ttft_reward'].isna().sum()
    nan_tpot = df['tpot_reward'].isna().sum()
    nan_combined = df['reward'].isna().sum()
    
    if nan_ttft > 0 or nan_tpot > 0 or nan_combined > 0:
        logger.warning(f"Reward columns contain NaN values: ttft_reward={nan_ttft}/{len(df)}, tpot_reward={nan_tpot}/{len(df)}, reward={nan_combined}/{len(df)}")
        # Fill NaN with 0 as fallback (though this shouldn't happen)
        df['ttft_reward'] = df['ttft_reward'].fillna(0)
        df['tpot_reward'] = df['tpot_reward'].fillna(0)
        df['reward'] = df['reward'].fillna(0)
        logger.warning("Filled NaN values in reward columns with 0")
    
    # Add SLO satisfaction columns
    df['avg_tpot_slo_satisfied'] = tpot_values <= avg_tpot_slo
    df['avg_ttft_slo_satisfied'] = ttft_values <= ttft_slo
    
    logger.info(f"Reward statistics: min={df['reward'].min():.4f}, max={df['reward'].max():.4f}, mean={df['reward'].mean():.4f}")
    
    # Step 4: Create action mapping if needed
    if 'action' not in df.columns:
        unique_pods = df['selected_pod'].unique()
        pod_to_action = {pod: idx for idx, pod in enumerate(unique_pods)}
        df['action'] = df['selected_pod'].map(pod_to_action)
        logger.info(f"Created action mapping: {pod_to_action}")
    
    # Step 5: Detect normalizable features automatically
    normalizable_features, non_normalizable_features, pod_feature_types = _get_normalizable_features(df, hyperparameters.get('NO_NORMALIZE_FEATURES', []))
    logger.info(f"Detected {len(normalizable_features)} normalizable features and {len(non_normalizable_features)} non-normalizable features")
    logger.debug(f"Normalizable features: {normalizable_features[:5]}...")
    logger.debug(f"Non-normalizable features: {non_normalizable_features[:5]}...")
    logger.info(f"Pod feature types: {sorted(pod_feature_types)}")
    
    # Step 6: Initialize or load statistics
    if stats_file and os.path.exists(stats_file):
        logger.info(f"Loading existing statistics from: {stats_file}")
        with open(stats_file, 'rb') as f:
            stats_instance = pickle.load(f)
        logger.info(f"Loaded stats for {len(stats_instance.feature_stats)} features")
    else:
        logger.info("Creating new statistics instance")
        stats_instance = FeatureStats(normalizable_features)
    
    # Step 6.5: Compute POOLED statistics for pod features
    if pod_feature_types:
        logger.info(f"🔄 Computing pooled statistics for {len(pod_feature_types)} pod feature types")
        _compute_pooled_pod_statistics(df, pod_feature_types, stats_instance)
        logger.info(f"✅ Pooled statistics computed successfully")

    # Step 6.6: If fixed_range_bounds provided, override all stats with fixed values.
    # This implements domain-knowledge normalization: x / practical_max
    # by setting mean=0, std=practical_max so the z-score formula (x - mean) / std = x / max.
    use_fixed_range = fixed_range_bounds is not None
    if use_fixed_range:
        logger.info(f"📐 FIXED-RANGE normalization: overriding stats with domain-knowledge bounds")
        for feature_name, practical_max in fixed_range_bounds.items():
            rs = RunningStats(feature_names=feature_name)
            rs.count = 1  # Mark as initialized
            rs.mean = np.array([0.0])
            rs.std = np.array([float(practical_max)])
            rs.min = np.array([0.0])
            rs.max = np.array([float(practical_max)])
            rs.sum_sq_diff = np.array([0.0])
            stats_instance.feature_stats[feature_name] = rs
            # Track as normalized feature
            stats_instance.CONFIG.setdefault("FEATURES_NORMALIZED", set()).add(feature_name)
            logger.info(f"  {feature_name}: [0, {practical_max}] → normalized = x / {practical_max}")
        stats_instance.CONFIG["NUM_FEATURES_NORMALIZED"] = len(stats_instance.CONFIG.get("FEATURES_NORMALIZED", set()))

    # Step 7: Normalize features (using pooled stats for pod features)
    # When using fixed-range bounds, set update_statistics=False to prevent
    # the normalizer from recomputing stats from data (which would overwrite
    # our fixed-range values).
    update_stats_during_norm = not use_fixed_range
    logger.info(f"Starting feature normalization... (update_statistics={update_stats_during_norm})")
    for feature in normalizable_features:
        try:
            # Check if this is a pooled pod feature type (not a column name)
            if feature in pod_feature_types:
                # OPTIMIZATION: Use cached matching columns lookup
                matching_columns = _get_matching_columns_cached(df.columns, feature)
                for col in matching_columns:
                    _normalize_single_feature(df, col, stats_instance, update_statistics=update_stats_during_norm)
            else:
                # Regular feature - normalize directly
                _normalize_single_feature(df, feature, stats_instance, update_statistics=update_stats_during_norm)
        except Exception as e:
            logger.error(f"Failed to normalize feature {feature}: {e}")
            # Continue with other features instead of crashing
    
    # Step 8: Save statistics if requested
    if stats_file:
        logger.info(f"Saving statistics to: {stats_file}")
        with open(stats_file, 'wb') as f:
            pickle.dump(stats_instance, f)
    
    # Step 9: Save normalized data if requested
    if output_csv_file:
        logger.info(f"Saving normalized data to: {output_csv_file}")
        df.to_csv(output_csv_file, index=False)
    
    # Step 10: Prepare summary
    summary = {
        'input_file': processed_csv_file,
        'output_file': output_csv_file,
        'num_samples': len(df),
        'num_features_normalized': stats_instance.CONFIG["NUM_FEATURES_NORMALIZED"],
        'total_features': stats_instance.CONFIG["TOTAL_FEATURES"],
        'reward_function': reward_function,
        'ttft_slo': ttft_slo,
        'avg_tpot_slo': avg_tpot_slo,
        'processing_time': time.time() - start_time
    }
    
    logger.info(f"Normalization completed in {summary['processing_time']:.2f} seconds")
    logger.info(f"Summary: {summary}")
    
    return df, stats_instance, summary


def analyze_normalization_impact(input_csv_file, output_csv_file=None, 
                               reward_function='linear_simple', stats_file=None, hyperparameters=None):
    """
    Analyze the impact of normalization on the dataset.
    
    Args:
        input_csv_file: Path to input CSV file
        output_csv_file: Path for output analysis CSV (optional)
        reward_function: Reward function to use
        stats_file: Path to save/load normalization statistics
        hyperparameters: Model hyperparameters dict
        
    Returns:
        tuple: (analysis_df, stats_instance, summary)
    """
    logger.info(f"Analyzing normalization impact for: {input_csv_file}")
    
    # Normalize the data
    normalized_df, stats_instance, summary = normalize_processed_data(
        input_csv_file, output_csv_file, reward_function, stats_file, hyperparameters
    )
    
    # Perform additional analysis
    logger.info("Performing normalization impact analysis...")
    
    # Calculate feature statistics before and after normalization
    analysis_results = []
    for feature in stats_instance.get_feature_names():
        if feature in normalized_df.columns:
            stats = stats_instance.feature_stats[feature]
            analysis_results.append({
                'feature': feature,
                'count': stats.count,
                'mean': stats.mean.item() if hasattr(stats.mean, 'item') else stats.mean,
                'std': stats.std.item() if hasattr(stats.std, 'item') else stats.std,
                'min': stats.min.item() if hasattr(stats.min, 'item') else stats.min,
                'max': stats.max.item() if hasattr(stats.max, 'item') else stats.max,
                'was_normalized': feature in stats_instance.CONFIG["FEATURES_NORMALIZED"]
            })
    
    analysis_df = pd.DataFrame(analysis_results)
    
    if output_csv_file:
        analysis_file = output_csv_file.replace('.csv', '_analysis.csv')
        analysis_df.to_csv(analysis_file, index=False)
        logger.info(f"Analysis saved to: {analysis_file}")
    
    return analysis_df, stats_instance, summary


def main():
    """Command-line interface for data normalization"""
    parser = argparse.ArgumentParser(description='Normalize processed CSV data for LLM routing')
    parser.add_argument('input_csv', help='Input processed CSV file')
    parser.add_argument('--output', '-o', help='Output normalized CSV file (auto-generated if not specified)')
    parser.add_argument('--reward-function', '-r', default='simple_latency_minimization', 
                       choices=['linear_simple', 'linear_simple_extended', 'piecewise_linear_steeper_gradient', 'inverse_latency', 'latency_optimized', 'context_aware', 'quantile_based', 'simple_latency_minimization', 'negative_reciprocal', 'negative_linear', 'negative_squared'],
                       help='Reward function to use')
    parser.add_argument('--stats-file', '-s', help='Statistics file for saving/loading normalization stats')
    parser.add_argument('--hyperparameters', '-H', help='JSON file containing model hyperparameters')
    parser.add_argument('--analyze', '-a', action='store_true', help='Perform detailed analysis')
    
    args = parser.parse_args()
    
    # Auto-generate output files if not specified
    if not args.output:
        input_name = os.path.splitext(os.path.basename(args.input_csv))[0]
        args.output = f"{input_name}-normalized.csv"
    
    if not args.stats_file:
        input_name = os.path.splitext(os.path.basename(args.input_csv))[0]
        args.stats_file = f"normalization_statistics.csv"
    
    # Load hyperparameters if provided
    hyperparameters = None
    if args.hyperparameters and os.path.exists(args.hyperparameters):
        with open(args.hyperparameters, 'r') as f:
            hyperparameters = json.load(f)
    
    try:
        if args.analyze:
            analysis_df, stats_instance, summary = analyze_normalization_impact(
                args.input_csv, args.output, args.reward_function, args.stats_file, hyperparameters
            )
            print(f"Analysis completed successfully!")
            print(f"Features analyzed: {len(analysis_df)}")
            print(f"Features normalized: {summary['num_features_normalized']}")
        else:
            normalized_df, stats_instance, summary = normalize_processed_data(
                args.input_csv, args.output, args.reward_function, args.stats_file, hyperparameters
            )
            print(f"Normalization completed successfully!")
            print(f"Samples processed: {summary['num_samples']}")
            print(f"Features normalized: {summary['num_features_normalized']}")
            
    except Exception as e:
        logger.error(f"Failed to process {args.input_csv}: {e}")
        raise


if __name__ == "__main__":
    main()
