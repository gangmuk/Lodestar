# feature_normalization.py
"""
Shared feature normalization logic for both offline training and online inference.
This module centralizes all normalization logic to ensure consistency.
"""

import pandas as pd
import numpy as np
import pickle
from logger import logger
from typing import Dict, List, Optional, Any, Union, Tuple
import os


# Global normalization parameters
SIGNAL_AMPLIFICATION_DEGREE = 1.0  # 1.5
REWARD_AMPLIFICATION_DEGREE = 2.0
REWARD_AMPLIFICATION_THRESHOLD = 0.5
STD_THRESHOLD_FOR_NORMALIZATION = 0.1


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


def normalize_features_for_training(df: pd.DataFrame, stats: PerFeatureRunningStats) -> Tuple[pd.DataFrame, PerFeatureRunningStats, dict]:
    # Feature categorization
    request_features = ['input_tokens', 'output_tokens', 'total_tokens']
    pod_features_cols = [col for col in df.columns if col.startswith('pod_') and 
                        df[col].dtype in ['float64', 'int64']]
    
    logger.info("🔧 POD-CENTRIC FEATURE PROCESSING")
    logger.info("=" * 50)
    
    # Analyze raw feature ranges (offline only)
    logger.info("Raw feature analysis:")
    high_variance_pod_features = []
    
    for feature in pod_features_cols:
        if feature in df.columns:
            values = df[feature].values
            std_val = values.std()
            logger.info(f"  {feature}: std={std_val:.3f}, range=[{values.min():.2f}, {values.max():.2f}]")
            
            if std_val > STD_THRESHOLD_FOR_NORMALIZATION:
                high_variance_pod_features.append(feature)
                logger.warning(f"Normalize since it has high variance ({feature}, std:{std_val})")
            else:
                logger.warning(f"Skip normalize since it has low variance ({feature}, std:{std_val})")
    
    logger.info(f"High variance pod features: {len(high_variance_pod_features)}")
    
    # Request feature analysis (offline only)
    logger.info("\nRequest feature handling:")
    for feature in request_features:
        if feature in df.columns:
            values = df[feature].values
            std_val = values.std()
            logger.info(f"  {feature}: std={std_val:.3f}")
            if std_val > STD_THRESHOLD_FOR_NORMALIZATION:
                logger.info(f"    → Will normalize")
            else:
                logger.info(f"    → Using RAW values (no normalization)")
    
    # ===== SELECTIVE NORMALIZATION STRATEGY =====
    
    # 1. Handle request features - only normalize if they have reasonable variance
    request_normalized_count = 0
    for feature in request_features:
        if feature in df.columns:
            values = df[feature].values
            if values.std() > STD_THRESHOLD_FOR_NORMALIZATION:
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

    # 2. Handle pod features - normalize high-variance features only
    pod_normalized_count = 0
    for feature in pod_features_cols:
        if feature in df.columns:
            if 'kv_hit_ratio' in feature:
                logger.info(f"⚪ Skipping normalization for {feature} (already 0-100 scale)")
                continue
            
            values = df[feature].values
            if values.std() > STD_THRESHOLD_FOR_NORMALIZATION:
                feature_data = values.reshape(-1, 1)
                
                if feature not in stats.feature_stats:
                    stats.feature_stats[feature] = RunningStats()
                
                # Store original std for verification
                original_std = df[feature].std()
                
                stats.feature_stats[feature].update(feature_data)
                normalized_feature = stats.feature_stats[feature].normalize(feature_data)
                df[feature] = normalized_feature.flatten()
                
                # Verify normalization didn't destroy variance
                new_std = df[feature].std()
                
                if new_std > 0.5:  # Ensure reasonable post-normalization variance
                    pod_normalized_count += 1
                    logger.info(f"✅ Normalized pod feature: {feature} (std: {original_std:.3f} → {new_std:.3f})")
                else:
                    logger.warning(f"⚠️  Post-normalization variance too low for {feature} (std: {original_std:.3f} → {new_std:.3f})")

    # 3. FEATURE IMPORTANCE AMPLIFICATION
    amplified_count = 0
    if SIGNAL_AMPLIFICATION_DEGREE > 1.0:
        critical_features = ['running_requests', 'waiting_requests', 'decode_tokens', 'prefill_tokens']
        for feature in pod_features_cols:
            if any(critical in feature for critical in critical_features):
                if feature in df.columns:
                    df[feature] = df[feature] * SIGNAL_AMPLIFICATION_DEGREE
                    amplified_count += 1
                    logger.info(f"📈 Amplified critical feature: {feature} by {SIGNAL_AMPLIFICATION_DEGREE}%, min: {df[feature].min()}, max: {df[feature].max()}, mean: {df[feature].mean()}")

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
    
    return df, stats, normalization_summary


def normalize_features_for_inference(df: pd.DataFrame, stats: PerFeatureRunningStats) -> pd.DataFrame:
    """
    Normalize features for inference using training statistics.
    
    Args:
        df: DataFrame with preprocessed features
        stats: PerFeatureRunningStats object with training statistics
    
    Returns:
        pd.DataFrame: Normalized DataFrame
    """
    request_features = ['input_tokens', 'output_tokens', 'total_tokens']
    pod_features_cols = [col for col in df.columns if col.startswith('pod_') and 
                        df[col].dtype in ['float64', 'int64']]
    
    if stats.count > 0:
        logger.debug("Applying pod-centric normalization for inference")
        
        # 1. Request features - only normalize if they were normalized in training
        for feature in request_features:
            if feature in df.columns and feature in stats.feature_stats:
                feature_data = df[feature].values.reshape(-1, 1)
                normalized_feature = stats.feature_stats[feature].normalize(feature_data)
                df[feature] = normalized_feature.flatten()
                logger.debug(f"Normalized request feature {feature} for inference")
            else:
                logger.debug(f"Kept raw values for request feature {feature}")
        
        # 2. Pod features - normalize those that were normalized in training
        pod_normalized_count = 0
        for feature in pod_features_cols:
            if 'kv_hit_ratio' in feature:
                continue  # Skip normalization
            if feature in df.columns and feature in stats.feature_stats:
                feature_data = df[feature].values.reshape(-1, 1)
                normalized_feature = stats.feature_stats[feature].normalize(feature_data)
                df[feature] = normalized_feature.flatten()
                pod_normalized_count += 1
                logger.debug(f"Normalized pod feature {feature} for inference")
        
        # 3. Apply same critical feature amplification as training
        critical_features = ['running_requests', 'waiting_requests', 'decode_tokens', 'prefill_tokens']
        amplified_count = 0
        for feature in pod_features_cols:
            if any(critical in feature for critical in critical_features):
                if feature in df.columns:
                    df[feature] = df[feature] * SIGNAL_AMPLIFICATION_DEGREE
                    amplified_count += 1
                    logger.debug(f"Amplified critical feature {feature} for inference")
        
        logger.debug(f"Applied pod-centric normalization: {pod_normalized_count} pod features normalized, {amplified_count} amplified")
        
    else:
        logger.warning(f"No normalization stats available for inference")
    
    return df


def apply_reward_engineering(df: pd.DataFrame) -> pd.DataFrame:
    if 'reward' in df.columns:
        rewards = df['reward'].values
        logger.info("\n🎯 REWARD ENGINEERING")
        logger.info("=" * 30)
        logger.info(f"Original rewards: range=[{rewards.min():.3f}, {rewards.max():.3f}], std={rewards.std():.3f}")
        reward_gap = rewards.max() - rewards.min()
        if reward_gap < REWARD_AMPLIFICATION_THRESHOLD:
            logger.info(f"Reward gap is too small: {reward_gap:.2f}, 📈 Applying reward amplification ({REWARD_AMPLIFICATION_THRESHOLD})")
            reward_mean = rewards.mean()
            df['reward'] = reward_mean + (rewards - reward_mean) * REWARD_AMPLIFICATION_DEGREE
            new_rewards = df['reward'].values
            logger.info(f"Amplified rewards: range=[{new_rewards.min():.3f}, {new_rewards.max():.3f}], std={new_rewards.std():.3f}")
        else:
            logger.info("✅ Reward signal already strong enough")
    return df


def load_stats(stats_file: str) -> PerFeatureRunningStats:
    """Load statistics from file"""
    return PerFeatureRunningStats.load(stats_file)


def save_stats(stats: PerFeatureRunningStats, stats_file: str) -> None:
    """Save statistics to file"""
    stats.save(stats_file)