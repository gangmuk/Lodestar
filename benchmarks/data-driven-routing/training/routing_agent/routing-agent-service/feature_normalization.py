# feature_normalization.py
"""
Shared feature normalization logic for both offline training and online inference.
This module centralizes all normalization logic to ensure consistency.
"""

import pandas as pd
import numpy as np
import pickle
from logger import logger
from typing import Tuple
import os
import json

class RunningStats:
    """Maintains running mean and standard deviation for feature normalization"""
    def __init__(self, feature_names):
        self.count = 0
        self.mean = None
        self.var = None  # Variance
        if feature_names == None:
            logger.error("RunningStats initialized with None feature_names, setting to empty list")
            assert False
        self.feature_names = feature_names
        
    def update_stats_incrementally(self, new_data):
        """Incrementally update statistics with new batch of data using Welford's algorithm"""
        if new_data is None or len(new_data) == 0:
            logger.error("Received empty data for RunningStats.update, skipping")
            return
        
        # Convert to numpy array
        new_data = np.array(new_data, dtype=np.float64)
        new_count = len(new_data)
        
        # First update
        if self.count == 0:
            self.mean = np.mean(new_data, axis=0)
            self.var = np.var(new_data, axis=0) * new_count
            self.count = new_count
            logger.info(f"The very first RunningStats.update call for {self.feature_names}. Initialized running stats with {new_count} samples")
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
        
        logger.info(f"{self.feature_names}, Updated running stats, now have {self.count} samples")
        
    def get_mean(self):
        return self.mean if self.mean is not None else 0
        
    def get_std(self):
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
            logger.error(f"{self.feature_names}: No statistics available. normalization cannot be performed.")
            assert False
        mean = self.get_mean()
        std = self.get_std()
        return (data - mean) / std
    
    ## not used
    # def save(self, filename):
    #     """Save statistics to file"""
    #     with open(filename, 'wb') as f:
    #         pickle.dump({
    #             'count': self.count,
    #             'mean': self.mean,
    #             'var': self.var,
    #             'feature_names': self.feature_names
    #         }, f)
    #     logger.info(f"Saved running statistics to {filename}")
        
class PerFeatureRunningStats:
    """Maintains separate running statistics for each feature"""
    def __init__(self):
        self.feature_stats = {}  # Dict[feature_name, RunningStats]
        self.CONFIG = None
    
    def write_stats_to_file(self, feature_normalization_stats_file):
        """Save all feature statistics to file"""
        save_data = {}
        for feature_name, stats in self.feature_stats.items():
            save_data[feature_name] = {
                'count': stats.count,
                'mean': stats.mean,
                'var': stats.var,
                'feature_names': stats.feature_names
            }
        
        with open(feature_normalization_stats_file, 'wb') as f:
            pickle.dump(save_data, f)
        logger.info(f"Saved per-feature statistics for {len(self.feature_stats)} features to {feature_normalization_stats_file}")

        # also write them in text file
        stat_in_text_file = feature_normalization_stats_file.replace('.pkl', '.txt')
        with open(stat_in_text_file, 'w') as f:
            f.write(f"Per-feature statistics for {len(self.feature_stats)} features:\n")
            for feature_name, stats in self.feature_stats.items():
                f.write(f"{feature_name}: count={stats.count}, mean={stats.mean}, var={stats.var}\n")
        
    @classmethod
    def create_new_empty_instance(cls):
        return cls()

    @classmethod
    def create_new_instance_with_stats_file(cls, feature_normalization_stats_file):
        if not os.path.exists(feature_normalization_stats_file):
            logger.error(f"Feature normalization stats file {feature_normalization_stats_file} does not exist.")
            assert False
        instance = cls()
        try:
            with open(feature_normalization_stats_file, 'rb') as pkl_file:
                save_data = pickle.load(pkl_file)
            
            # Validate it's the expected per-feature format
            if not isinstance(save_data, dict):
                logger.error(f"Invalid file format in {feature_normalization_stats_file}. Expected dictionary format.")
                assert False

            # Check if it looks like per-feature format (each value should be a dict with stats)
            for feature_name, stats_data in save_data.items():
                if not isinstance(stats_data, dict) or 'count' not in stats_data:
                    logger.error(f"Invalid per-feature format in {feature_normalization_stats_file}. Feature '{feature_name}' missing required fields.")
                    assert False
            
            # New format - per-feature statistics
            for feature_name, stats_data in save_data.items():
                stats = RunningStats(feature_names=stats_data['feature_names'])
                stats.count = stats_data.get('count', 0)
                stats.mean = stats_data.get('mean')
                stats.var = stats_data.get('var')
                instance.feature_stats[feature_name] = stats
            
            logger.info(f"Loaded per-feature statistics for {len(instance.feature_stats)} features from {feature_normalization_stats_file}")
            
        except Exception as e:
            logger.error(f"Error loading statistics file {feature_normalization_stats_file}: {e}. Expected per-feature format.")
            # print pickle file
            try:
                with open(feature_normalization_stats_file, 'rb') as f:
                    content = pickle.load(f) # Use pickle.load() to deserialize the content
                    logger.error(f"Content of {feature_normalization_stats_file}:\n{content}")
                logger.error("Please ensure the file is in the correct per-feature format.")
            except FileNotFoundError:
                logger.error(f"Error: The file '{feature_normalization_stats_file}' was not found.")
            except pickle.UnpicklingError as e:
                logger.error(f"Error: Could not unpickle the file '{feature_normalization_stats_file}'. It might be corrupted or not a valid pickle file. Details: {e}")
            except Exception as e:
                logger.error(f"An unexpected error occurred: {e}")

            assert False
        
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


def normalize_features_for_training(df: pd.DataFrame, stats_instance: PerFeatureRunningStats) -> Tuple[pd.DataFrame, PerFeatureRunningStats, dict]:
    # Feature categorization
    request_features = ['input_tokens', 'output_tokens', 'total_tokens']
    # pod_features_cols = [col for col in df.columns if col.startswith('pod_') and df[col].dtype in ['float64', 'int64']]
    pod_features_cols = [col for col in df.columns if col.startswith('pod_') and df[col].dtype in ['float64', 'int64'] and 'gpu_model' not in col]

    # Analyze raw feature ranges (offline only)
    logger.info("Raw feature analysis:")
    high_variance_pod_features = []
    
    logger.info(f"High variance pod features: {len(high_variance_pod_features)}")
    
    # ===== SELECTIVE NORMALIZATION STRATEGY =====
    # 1. Handle request features - only normalize if they have reasonable variance
    request_normalized_count = 0
    if stats_instance.CONFIG["ENABLE_REQUEST_NORMALIZATION"]:
        for feature in request_features:
            if feature in df.columns:
                values = df[feature].values
                if values.std() > stats_instance.CONFIG["STD_THRESHOLD_FOR_REQ_FEAT_NORMALIZATION"]:
                    logger.info(f"🔍 {feature}, Normalizing. Variance is high (std: {values.std():.3f})")
                    stats_instance.CONFIG["FEATURES_NORMALIZED"].add(feature)
                    stats_instance.CONFIG["NUM_FEATURES_NORMALIZED"] = len(stats_instance.CONFIG["FEATURES_NORMALIZED"])
                    feature_data = values.reshape(-1, 1)
                    if feature not in stats_instance.feature_stats:
                        stats_instance.feature_stats[feature] = RunningStats(feature_names=feature)
                    stats_instance.feature_stats[feature].update_stats_incrementally(feature_data)
                    normalized_feature = stats_instance.feature_stats[feature].normalize(feature_data)
                    df[feature] = normalized_feature.flatten()
                    request_normalized_count += 1
                    new_std = df[feature].std()
                    logger.info(f"✅ {feature}, Normalize. prev std: {values.std()} -> new std: {new_std:.3f}")
                else:
                    logger.info(f"⚪ {feature}, Kept raw values. Variance is already low (std: {values.std():.3f})")
    else:
        logger.info("🚫 Request feature normalization DISABLED - using raw values")
        
    # 2. Handle pod features - normalize high-variance features only
    pod_normalized_count = 0
    if stats_instance.CONFIG["ENABLE_POD_NORMALIZATION"]:
        for feature in pod_features_cols:
            if feature in df.columns:
                if 'kv_hit_ratio' in feature:
                    logger.info(f"⚪ {feature}, Skip normalization. (already 0-100 scale)")
                    continue
                values = df[feature].values
                if values.std() > stats_instance.CONFIG["STD_THRESHOLD_FOR_POD_FEAT_NORMALIZATION"]:
                    stats_instance.CONFIG["FEATURES_NORMALIZED"].add(feature)
                    stats_instance.CONFIG["NUM_FEATURES_NORMALIZED"] = len(stats_instance.CONFIG["FEATURES_NORMALIZED"])
                    logger.info(f"🔍 Normalizing pod feature: {feature} (std: {values.std():.3f})")
                    feature_data = values.reshape(-1, 1)
                    if feature not in stats_instance.feature_stats:
                        stats_instance.feature_stats[feature] = RunningStats(feature_names=feature)
                    original_std = df[feature].std()
                    stats_instance.feature_stats[feature].update_stats_incrementally(feature_data)
                    normalized_feature = stats_instance.feature_stats[feature].normalize(feature_data)
                    df[feature] = normalized_feature.flatten()
                    pod_normalized_count += 1
                    new_std = df[feature].std()
                    logger.info(f"✅ Normalized pod feature: {feature}, prev std: {original_std:.3f} → new std: {new_std:.3f}")
                    if new_std <= 0.5:
                        logger.warning(f"⚠️  Post-normalization variance too low for {feature} (std: {original_std:.3f} → {new_std:.3f})")
                else:
                    logger.info(f"⚪ Variance is already low (std: {values.std():.3f}). Kept raw values for pod feature: {feature}")
    else:
        logger.info("🚫 Pod feature normalization DISABLED - using raw values")

    # 3. FEATURE IMPORTANCE AMPLIFICATION
    amplified_count = 0
    if stats_instance.CONFIG["FEATURE_AMPLIFICATION"] and stats_instance.CONFIG["ENABLE_POD_NORMALIZATION"] and stats_instance.CONFIG["SIGNAL_AMPLIFICATION_DEGREE"] > 1.0:
        critical_features = ['running_requests', 'waiting_requests', 'decode_tokens', 'prefill_tokens']
        for feature in pod_features_cols:
            if any(critical in feature for critical in critical_features):
                if feature in df.columns:
                    df[feature] = df[feature] * stats_instance.CONFIG["SIGNAL_AMPLIFICATION_DEGREE"]
                    stats_instance.CONFIG["FEATURES_AMPLIFIED"].add(feature)
                    stats_instance.CONFIG["NUM_FEATURES_AMPLIFIED"] = len(stats_instance.CONFIG["FEATURES_AMPLIFIED"])
                    amplified_count += 1
                    logger.info(f"📈 Amplified critical feature: {feature} by {stats_instance.CONFIG['SIGNAL_AMPLIFICATION_DEGREE']}%, min: {df[feature].min()}, max: {df[feature].max()}, mean: {df[feature].mean()}")

    # Summary logging
    logger.info(f"✅ FEATURE PROCESSING COMPLETE:")
    logger.info(f"  - Request features normalized: {request_normalized_count}/{len(request_features)}")
    logger.info(f"  - Pod features normalized: {pod_normalized_count}/{len(pod_features_cols)}")
    normalization_summary = {
        'request_normalized_count': request_normalized_count,
        'pod_normalized_count': pod_normalized_count,
        'amplified_count': amplified_count,
        'total_request_features': len(request_features),
        'total_pod_features': len(pod_features_cols)
    }

    df = try_reward_amplification(df, stats_instance.CONFIG)
    
    return df, stats_instance, normalization_summary


def normalize_features_for_inference(df: pd.DataFrame, stats_instance: PerFeatureRunningStats) -> pd.DataFrame:
    request_features = ['input_tokens', 'output_tokens', 'total_tokens']
    # pod_features_cols = [col for col in df.columns if col.startswith('pod_') and df[col].dtype in ['float64', 'int64']]
    pod_features_cols = [col for col in df.columns if col.startswith('pod_') and df[col].dtype in ['float64', 'int64'] and 'gpu_model' not in col]
    if stats_instance.count > 0:
        logger.debug("Applying normalize_features_for_inference")
        if stats_instance.CONFIG["ENABLE_REQUEST_NORMALIZATION"]:    
            # 1. Request features - only normalize if they were normalized in training
            for feature in request_features:
                if feature in df.columns and feature in stats_instance.feature_stats:
                    feature_data = df[feature].values.reshape(-1, 1)
                    normalized_feature = stats_instance.feature_stats[feature].normalize(feature_data)
                    df[feature] = normalized_feature.flatten()
                    logger.debug(f"✅ Normalized request feature {feature} for inference")
                else:
                    logger.debug(f"Kept raw values for request feature {feature}")
        else:
            logger.debug("Request feature normalization DISABLED for inference")

        # 2. Pod features - normalize those that were normalized in training
        pod_normalized_count = 0
        if stats_instance.CONFIG["ENABLE_POD_NORMALIZATION"]:
            for feature in pod_features_cols:
                if 'kv_hit_ratio' in feature:
                    continue  # Skip normalization
                if feature in df.columns and feature in stats_instance.feature_stats:
                    feature_data = df[feature].values.reshape(-1, 1)
                    normalized_feature = stats_instance.feature_stats[feature].normalize(feature_data)
                    df[feature] = normalized_feature.flatten()
                    pod_normalized_count += 1
                    logger.debug(f"Normalized pod feature {feature} for inference")
            
            amplified_count = 0
            # 3. Apply same critical feature amplification as training
            critical_features = ['running_requests', 'waiting_requests', 'decode_tokens', 'prefill_tokens']
            for feature in pod_features_cols:
                if feature in stats_instance.CONFIG["FEATURES_AMPLIFIED"]:
                    if feature in df.columns:
                        df[feature] = df[feature] * stats_instance.CONFIG['SIGNAL_AMPLIFICATION_DEGREE']
                        amplified_count += 1
                        logger.debug(f"Amplified critical feature {feature} for inference")
            logger.debug(f"Applied normalization: {pod_normalized_count} pod features normalized, {amplified_count} amplified")
        else:
            logger.debug("Pod feature normalization DISABLED for inference")
    else:
        logger.warning(f"No normalization stats_instance available for inference")
    
    return df


def try_reward_amplification(df: pd.DataFrame, CONFIG) -> pd.DataFrame:
    if 'reward' in df.columns:
        rewards = df['reward'].values
        logger.info("\n🎯 REWARD ENGINEERING")
        logger.info("=" * 30)
        logger.info(f"Original rewards: range=[{rewards.min():.3f}, {rewards.max():.3f}], std={rewards.std():.3f}")
        reward_gap = rewards.max() - rewards.min()
        if reward_gap < CONFIG["REWARD_AMPLIFICATION_THRESHOLD"]:
            logger.info(f"Reward gap is too small: {reward_gap:.2f}, 📈 Applying reward amplification ({CONFIG['REWARD_AMPLIFICATION_THRESHOLD']})")
            reward_mean = rewards.mean()
            df['reward'] = reward_mean + (rewards - reward_mean) * CONFIG["REWARD_AMPLIFICATION_DEGREE"]
            new_rewards = df['reward'].values
            logger.info(f"Amplified rewards: range=[{new_rewards.min():.3f}, {new_rewards.max():.3f}], std={new_rewards.std():.3f}")
        else:
            logger.info("✅ Reward signal already strong enough")
    return df


def create_new_instance_with_stats_file(feature_normalization_stats_file: str) -> PerFeatureRunningStats:
    return PerFeatureRunningStats.create_new_instance_with_stats_file(feature_normalization_stats_file)

def create_new_empty_instance() -> PerFeatureRunningStats:
    return PerFeatureRunningStats.create_new_empty_instance()

def get_stats_instance(feature_normalization_stats_file, CONFIG):
    if os.path.exists(feature_normalization_stats_file):
        logger.info(f"Creating new stats instance from {feature_normalization_stats_file}")
        stats_instance = create_new_instance_with_stats_file(feature_normalization_stats_file)
    else:
        logger.info(f"{feature_normalization_stats_file} does not exist. Creating new EMPTY stats instance")
        stats_instance =  create_new_empty_instance()
    stats_instance.CONFIG = CONFIG
    return stats_instance