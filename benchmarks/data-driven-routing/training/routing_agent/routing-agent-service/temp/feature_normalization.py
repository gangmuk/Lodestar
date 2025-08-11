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
import csv
import json

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
        
    def update_stats_incrementally(self, new_data):
        if new_data is None or len(new_data) == 0:
            logger.error("Received empty data for RunningStats.update, skipping")
            return
        new_data = np.array(new_data, dtype=np.float64)
        new_count = len(new_data)
        old_mean = self.mean
        old_var = self.sum_sq_diff
        old_std = self.std
        old_min = self.min
        old_max = self.max
        if self.count == 0: # The very first update
            self.mean = np.mean(new_data, axis=0)
            self.sum_sq_diff = np.var(new_data, axis=0) * new_count
            self.count = new_count
            self.std = np.sqrt(self.sum_sq_diff / new_count)
            self.min = np.min(new_data, axis=0) # Initialize min
            self.max = np.max(new_data, axis=0) # Initialize max
            logger.info(f"The very first RunningStats.update call for {self.feature_names}. Initialized running stats with {new_count} samples")
            logger.info(f"qwer, {self.feature_names}, new_count={new_count}, total_count={self.count} samples, old_mean={old_mean}, old_std={old_std}, old_var={old_var}, old_min={old_min}, old_max={old_max}, new_mean={self.mean}, new_std={self.std}, new_var={self.sum_sq_diff}, new_min={self.min}, new_max={self.max}")
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
        logger.info(f"asdf, {self.feature_names}, new_count={new_count}, total_count={self.count} samples, old_mean={old_mean}, old_std={old_std}, old_var={old_var}, new_mean={self.mean}, new_std={self.std}, new_var={self.sum_sq_diff}")
        
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
    
class PerFeatureRunningStats:
    def __init__(self):
        self.feature_stats = {}  # Dict[feature_name, RunningStats]
        self.CONFIG = None

    def write_stats_to_file(self, feature_normalization_stats_file):
        # csv_filename = feature_normalization_stats_file.replace('.pkl', '.csv')
        with open(feature_normalization_stats_file, 'w') as f:
            f.write('feature_name,stats_type,value\n')
            for feature_name, stats in self.feature_stats.items():
                f.write(f'{feature_name},count,{stats.count}\n')
                mean_val = stats.mean.item() if hasattr(stats.mean, 'item') else stats.mean
                std_val = stats.std.item() if hasattr(stats.std, 'item') else stats.std
                f.write(f'{feature_name},mean,{mean_val}\n')
                f.write(f'{feature_name},std,{std_val}\n')
        logger.info(f"Saved per-feature statistics for {len(self.feature_stats)} features to {feature_normalization_stats_file}")
        
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
            with open(feature_normalization_stats_file, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                temp_feature_data = {}
                for row in reader:
                    feature_name = row['feature_name']
                    stats_type = row['stats_type']
                    value_str = row['value']
                    logger.debug(f"Loading normalization stats, feature_name={feature_name}, stats_type={stats_type}, value={value_str}")
                    if feature_name not in temp_feature_data:
                        temp_feature_data[feature_name] = {'count': 0, 'mean': None, 'std': None, 'feature_names': feature_name}
                    
                    if stats_type == 'count':
                        temp_feature_data[feature_name]['count'] = int(value_str)
                    elif stats_type == 'mean':
                        try:
                            # Attempt to load as JSON array, otherwise treat as scalar
                            temp_feature_data[feature_name]['mean'] = np.array(json.loads(value_str))
                        except json.JSONDecodeError:
                            temp_feature_data[feature_name]['mean'] = float(value_str)
                    elif stats_type == 'std':
                        try:
                            # Attempt to load as JSON array, otherwise treat as scalar
                            temp_feature_data[feature_name]['std'] = np.array(json.loads(value_str))
                        except json.JSONDecodeError:
                            temp_feature_data[feature_name]['std'] = float(value_str)
                    else:
                        logger.error(f"Unknown stats_type {stats_type} for feature {feature_name} in {feature_normalization_stats_file}")
                        assert False
            for feature_name, stats_data in temp_feature_data.items():
                stats = RunningStats(feature_names=stats_data['feature_names'])
                stats.count = stats_data['count']
                stats.mean = stats_data['mean']
                stats.std = stats_data['std']
                stats.sum_sq_diff = stats_data['std'] ** 2 * stats_data['count'] if stats_data['std'] is not None else None
                instance.feature_stats[feature_name] = stats
            logger.info(f"Loaded per-feature statistics for {len(instance.feature_stats)} features from {feature_normalization_stats_file}")
        except Exception as e:
            logger.error(f"Error loading statistics file {feature_normalization_stats_file}: {e}. Expected per-feature CSV format.")
            try:
                with open(feature_normalization_stats_file, 'r') as f:
                    content = f.read()
                    logger.error(f"Content of {feature_normalization_stats_file}:\n{content}")
                logger.error("Please ensure the file is in the correct per-feature CSV format.")
            except FileNotFoundError:
                logger.error(f"Error: The file '{feature_normalization_stats_file}' was not found after initial check.")
            except Exception as e_inner:
                logger.error(f"An unexpected error occurred trying to read file content: {e_inner}")

            assert False
        logger.debug("Per-feature statistics loaded:")
        for feature_name, stats in instance.feature_stats.items():
            logger.debug(f"{feature_name}: count={stats.count}, mean={stats.mean}, std={stats.std}")
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

def _get_normalizable_features(processed_df):
    request_features = ['input_tokens', 'output_tokens', 'total_tokens']
    pod_features = [
        col for col in processed_df.columns 
        if col.startswith('pod_') 
        and processed_df[col].dtype in ['float64', 'int64'] 
        and 'gpu_model' not in col
    ]
    return request_features + pod_features

def _normalize_single_feature(processed_df, feature, stats_instance, is_training, request_id=None):
    log_prefix = f"request_id,{request_id}," if request_id else ""
    if feature not in processed_df.columns:
        logger.error(f"{log_prefix}Feature {feature} not found in DataFrame")
        assert False
        
    if is_training:
        feature_std = processed_df[feature].values.std()
        
        # Initialize stats if needed (for both constant and non-constant features)
        if feature not in stats_instance.feature_stats:
            stats_instance.feature_stats[feature] = RunningStats(feature_names=feature)
        
        if feature_std == 0 or np.isclose(feature_std, 0, atol=1e-10):
            # Handle constant feature
            feature_data = processed_df[feature].values.reshape(-1, 1)
            
            # Set stats manually for constant feature
            stats_instance.feature_stats[feature].count = len(feature_data)
            stats_instance.feature_stats[feature].mean = np.mean(feature_data, axis=0)
            stats_instance.feature_stats[feature].std = np.array([0.0])  # Mark as constant
            stats_instance.feature_stats[feature].sum_sq_diff = 0.0  # No variance
            
            logger.info(f"⚪ {feature}, Saved as constant feature (std={feature_std:.6f}, value={stats_instance.feature_stats[feature].mean})")
            
            # Add to config tracking
            stats_instance.CONFIG.setdefault("CONSTANT_FEATURES", set()).add(feature)
            stats_instance.CONFIG["NUM_CONSTANT_FEATURES"] = len(stats_instance.CONFIG.get("CONSTANT_FEATURES", set()))
            return  # Skip normalization but stats are saved
            
        # Check for NaN values in the feature
        if np.any(np.isnan(processed_df[feature].values)):
            logger.error(f"❌ {feature}: Contains NaN values before normalization")
            assert False
    
        # Normal feature processing (non-constant) - stats already exist
        logger.info(f"🔍 {feature}, Normalizing. Variance is high (std: {processed_df[feature].values.std():.3f})")
        stats_instance.CONFIG.setdefault("FEATURES_NORMALIZED", set()).add(feature)
        stats_instance.CONFIG["NUM_FEATURES_NORMALIZED"] = len(stats_instance.CONFIG["FEATURES_NORMALIZED"])
        
        # Rest of your existing training code...
        feature_data = processed_df[feature].values.reshape(-1, 1)
        prev_std = processed_df[feature].values.std()
        prev_min = processed_df[feature].values.min()
        prev_max = processed_df[feature].values.max()
        prev_mean = processed_df[feature].values.mean()
        
        stats_instance.feature_stats[feature].update_stats_incrementally(feature_data)
        print(f"Updated stats for {feature}: count={stats_instance.feature_stats[feature].count}, mean={stats_instance.feature_stats[feature].mean}, std={stats_instance.feature_stats[feature].std}, var={stats_instance.feature_stats[feature].sum_sq_diff}")
        
        
        # Verify computed std is valid
        computed_std = stats_instance.feature_stats[feature].std
        if np.any(computed_std == 0) or np.any(np.isnan(computed_std)):
            logger.warning(f"⚠️  {feature}: Invalid computed std ({computed_std}), skipping normalization")
            return
        
        # Apply normalization
        normalized_feature = stats_instance.feature_stats[feature].normalize(feature_data)
        
        # Verify normalized data doesn't contain NaN
        if np.any(np.isnan(normalized_feature)):
            logger.error(f"❌ {feature}: Normalization produced NaN values, skipping")
            return
        
        processed_df[feature] = normalized_feature.flatten()
        
        new_std = processed_df[feature].std()
        if new_std <= 0.5:
            logger.warning(f"⚠️  Post-normalization variance too low for {feature} (std: {new_std:.3f})")
        logger.info(f"✅ {feature}, Normalize. prev std: {prev_std:.3f} new std: {new_std:.3f}")
        logger.info(f"✅ {feature}, Normalize. prev min: {prev_min:.3f} new min: {normalized_feature.min():.3f}")
        logger.info(f"✅ {feature}, Normalize. prev max: {prev_max:.3f} new max: {normalized_feature.max():.3f}")
        logger.info(f"✅ {feature}, Normalize. prev mean: {prev_mean:.3f} new mean: {normalized_feature.mean():.3f}")
        
    else:  # Inference
        if feature not in stats_instance.feature_stats:
            logger.error(f"{log_prefix}Feature {feature} not found in normalization stats")
            logger.error(f"Available features: {list(stats_instance.feature_stats.keys())}")
            logger.error(f"This indicates a training/inference feature mismatch - not a constant feature issue")
            assert False
            
        # Check if this was a constant feature during training
        if hasattr(stats_instance.feature_stats[feature], 'std') and np.allclose(stats_instance.feature_stats[feature].std, 0):
            logger.warning(f"{log_prefix}{feature} was constant during training (value={stats_instance.feature_stats[feature].mean}) - skipping normalization")
            return  # Don't normalize constant features
        
        # Add this line:
        feature_data = processed_df[feature].values.reshape(-1, 1)
        
        # Apply normalization using pre-computed stats
        normalized_feature = stats_instance.feature_stats[feature].normalize(feature_data)
        processed_df[feature] = normalized_feature.flatten()

def normalize_features_for_training(processed_df, stats_instance: PerFeatureRunningStats) -> pd.DataFrame:
    target_features = _get_normalizable_features(processed_df)
    logger.info(f"🔍 Normalizing features: {target_features}")
    
    logger.info("🔍 DEBUGGING FEATURES BEFORE NORMALIZATION:")
    for feature in target_features:
        if feature in processed_df.columns:
            values = processed_df[feature].values
            logger.info(f"{feature}: min={values.min()}, max={values.max()}, std={values.std():.6f}, has_nan={np.any(np.isnan(values))}")
            unique_vals = np.unique(values)
            if len(unique_vals) <= 5:
                logger.info(f"{feature}: unique values = {unique_vals}")
            elif len(unique_vals) <= 20:
                logger.info(f"{feature}: {len(unique_vals)} unique values, range = [{unique_vals.min()}, {unique_vals.max()}]")
        else:
            logger.warning(f"{feature}: NOT FOUND in DataFrame")
            
    
    # Check all features exist (matches original)
    for feature in target_features:
        assert feature in processed_df.columns
    
    # Normalize each feature and update stats
    for feature in target_features:
        _normalize_single_feature(processed_df, feature, stats_instance, is_training=True)
    
    # Apply feature amplification (batch approach like original)
    amplified_count = 0
    if stats_instance.CONFIG.get("FEATURE_AMPLIFICATION", False) and stats_instance.CONFIG.get("ENABLE_POD_NORMALIZATION", False) and stats_instance.CONFIG.get("SIGNAL_AMPLIFICATION_DEGREE", 1.0) > 1.0:
        critical_features = ['running_requests', 'waiting_requests', 'decode_tokens', 'prefill_tokens']
        pod_features = [col for col in processed_df.columns if col.startswith('pod_')]
        for feature in pod_features:
            if any(critical in feature for critical in critical_features):
                if feature in processed_df.columns:
                    processed_df[feature] = processed_df[feature] * stats_instance.CONFIG["SIGNAL_AMPLIFICATION_DEGREE"]
                    stats_instance.CONFIG.setdefault("FEATURES_AMPLIFIED", set()).add(feature)
                    stats_instance.CONFIG["NUM_FEATURES_AMPLIFIED"] = len(stats_instance.CONFIG["FEATURES_AMPLIFIED"])
                    amplified_count += 1
                    logger.info(f"📈 Amplified critical feature: {feature} by {stats_instance.CONFIG['SIGNAL_AMPLIFICATION_DEGREE']}%, min: {processed_df[feature].min()}, max: {processed_df[feature].max()}, mean: {processed_df[feature].mean()}")
    
    # Apply reward amplification
    processed_df = try_reward_amplification(processed_df, stats_instance.CONFIG)
    
    logger.info(f"✅ FEATURE PROCESSING COMPLETE:")
    return processed_df

def normalize_features_for_inference(processed_df: pd.DataFrame, stats_instance: PerFeatureRunningStats, request_id: str) -> pd.DataFrame:
    ## Not sure we really need to copy....
    # df_copy = processed_df.copy()
    df_copy = processed_df # It seems fine and logically shoudul be fine I think..
    
    target_features = _get_normalizable_features(df_copy)
    if stats_instance.count == 0:
        logger.error(f"request_id,{request_id},No normalization statistics available for inference")
        assert False
    for feature in target_features:
        _normalize_single_feature(df_copy, feature, stats_instance, is_training=False, request_id=request_id)
        if feature in stats_instance.CONFIG.get("FEATURES_AMPLIFIED", set()):
            if feature in df_copy.columns:
                df_copy[feature] = df_copy[feature] * stats_instance.CONFIG['SIGNAL_AMPLIFICATION_DEGREE']
                logger.info(f"request_id,{request_id},Amplified critical feature {feature} after normalization")
            else:
                logger.error(f"request_id,{request_id},Feature {feature} not found in DataFrame for amplification")
                assert False
    return df_copy


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

def get_stats_instance(CONFIG, feature_normalization_stats_file=None):
    if feature_normalization_stats_file is not None and not os.path.exists(feature_normalization_stats_file):
        logger.error(f"Feature normalization stats file {feature_normalization_stats_file} does not exist.")
        assert False
    if feature_normalization_stats_file is not None:
        if not os.path.exists(feature_normalization_stats_file):
            logger.error(f"Feature normalization stats file {feature_normalization_stats_file} does not exist. Creating new empty instance.")
            assert False
        logger.info(f"Creating stats instance from {feature_normalization_stats_file}")
        stats_instance = create_new_instance_with_stats_file(feature_normalization_stats_file)
    else:
        ## offline training path
        logger.info(f"{feature_normalization_stats_file} does not exist. Creating stats instance EMPTY one.")
        stats_instance =  create_new_empty_instance()
        
    stats_instance.CONFIG = CONFIG
    return stats_instance