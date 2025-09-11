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
from typing import Tuple, Dict, Any
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
        """Write feature statistics to a CSV file in long format (feature_name, stats_type, value)"""
        
        stats_data = []
        for feature_name, stats in self.feature_stats.items():
            # Add each stat type as a separate row
            stats_data.append({
                'feature_name': feature_name,
                'stats_type': 'count',
                'value': stats.count
            })
            stats_data.append({
                'feature_name': feature_name,
                'stats_type': 'mean',
                'value': stats.mean.item() if hasattr(stats.mean, 'item') else stats.mean
            })
            stats_data.append({
                'feature_name': feature_name,
                'stats_type': 'std',
                'value': stats.std.item() if hasattr(stats.std, 'item') else stats.std
            })
            stats_data.append({
                'feature_name': feature_name,
                'stats_type': 'min',
                'value': stats.min.item() if hasattr(stats.min, 'item') else stats.min
            })
            stats_data.append({
                'feature_name': feature_name,
                'stats_type': 'max',
                'value': stats.max.item() if hasattr(stats.max, 'item') else stats.max
            })
            stats_data.append({
                'feature_name': feature_name,
                'stats_type': 'sum_sq_diff',
                'value': stats.sum_sq_diff.item() if hasattr(stats.sum_sq_diff, 'item') else stats.sum_sq_diff
            })
        
        df = pd.DataFrame(stats_data)
        df.to_csv(filename, index=False)
        logger.info(f"Saved feature statistics to {filename} for {len(self.feature_stats)} features")
    
    @classmethod
    def load_from_csv(cls, filename):
        """Load feature statistics from a CSV file in long format (feature_name, stats_type, value)"""
        
        if not os.path.exists(filename):
            logger.error(f"Statistics file not found: {filename}")
            return None
        
        try:
            df = pd.read_csv(filename)
            stats_instance = cls()
            
            # Group by feature_name and extract stats
            for feature_name in df['feature_name'].unique():
                feature_df = df[df['feature_name'] == feature_name]
                
                # Create RunningStats stats_instance for this feature
                stats = RunningStats(feature_names=feature_name)
                
                # Extract values for each stat type
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
            
            stats_instance.CONFIG["TOTAL_FEATURES"] = len(stats_instance.feature_stats)
            stats_instance.CONFIG["NUM_FEATURES_NORMALIZED"] = len(stats_instance.feature_stats)
            stats_instance.CONFIG["FEATURES_NORMALIZED"] = set(stats_instance.feature_stats.keys())
            
            logger.info(f"Loaded feature statistics from {filename} for {len(stats_instance.feature_stats)} features")
            return stats_instance
            
        except Exception as e:
            logger.error(f"Failed to load statistics from {filename}: {e}")
            logger.error(f"stats_instance.feature_stats: {stats_instance.feature_stats}")
            logger.error(f"feature_statistics: {df.to_csv()}")
            return None


def _get_normalizable_features(processed_df):
    """
    Automatically detect which features can be normalized.
    
    Returns:
        tuple: (normalizable_features, non_normalizable_features)
    """
    normalizable_features = ['input_tokens', 'output_tokens', 'total_tokens']
    
    # Add all pod-specific features except gpu_model (which is categorical)
    for col in processed_df.columns:
        if col.startswith('pod_') and 'gpu_model' not in col:
            normalizable_features.append(col)
    
    # Find non-normalizable features
    non_normalizable_features = []
    for col in processed_df.columns:
        if col not in normalizable_features:
            non_normalizable_features.append(col)
    
    return normalizable_features, non_normalizable_features


def _normalize_single_feature(processed_df, feature, stats_instance, is_training, request_id=None):
    """
    Normalize a single feature using the provided statistics.
    
    Args:
        processed_df: DataFrame containing the feature
        feature: Feature name to normalize
        stats_instance: FeatureStats instance containing normalization statistics
        is_training: Whether this is training (True) or inference (False)
        request_id: Optional request ID for logging
    """
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
            
            return  # Skip normalization but stats are saved
            
        # Check for NaN values in the feature
        if np.any(np.isnan(processed_df[feature].values)):
            logger.error(f"❌ {feature}: Contains NaN values before normalization")
            assert False
    
        # Normal feature processing (non-constant) - stats already exist
        logger.info(f"🔍 {feature}, Normalizing. Variance is high (std: {processed_df[feature].values.std():.3f})")
        stats_instance.CONFIG.setdefault("FEATURES_NORMALIZED", set()).add(feature)
        stats_instance.CONFIG["NUM_FEATURES_NORMALIZED"] = len(stats_instance.CONFIG["FEATURES_NORMALIZED"])
        
        # Update statistics
        feature_data = processed_df[feature].values.reshape(-1, 1)
        prev_std = processed_df[feature].values.std()
        prev_min = processed_df[feature].values.min()
        prev_max = processed_df[feature].values.max()
        prev_mean = processed_df[feature].values.mean()
        
        ##############################################
        stats_instance.feature_stats[feature].update_stats_incrementally(feature_data)
        ##############################################
        
        # Verify computed std is valid
        computed_std = stats_instance.feature_stats[feature].std
        if np.any(computed_std == 0) or np.any(np.isnan(computed_std)):
            logger.warning(f"⚠️  {feature}: Invalid computed std ({computed_std}), skipping normalization")
            return
        
        ##############################################
        # Apply normalization
        normalized_feature = stats_instance.feature_stats[feature].normalize(feature_data)
        ##############################################
        
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
            logger.debug(f"{log_prefix}{feature} was constant during training (value={stats_instance.feature_stats[feature].mean}) - skipping normalization")
            return  # Don't normalize constant features
        
        # Apply normalization using stored statistics
        feature_data = processed_df[feature].values.reshape(-1, 1)
        normalized_feature = stats_instance.feature_stats[feature].normalize(feature_data)
        
        # Verify normalized data doesn't contain NaN
        if np.any(np.isnan(normalized_feature)):
            logger.error(f"{log_prefix}❌ {feature}: Normalization produced NaN values, skipping")
            return
        
        processed_df[feature] = normalized_feature.flatten()
        logger.debug(f"{log_prefix}✅ {feature}, Normalized using stored stats")


def normalize_processed_data(processed_csv_file, output_csv_file=None, 
                           reward_function='linear_simple', stats_file=None, hyperparameters=None):
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
    else:
        logger.error(f"Unknown reward function: {reward_function}")
        raise ValueError(f"Unknown reward function: {reward_function}")
    
    # Add reward columns to dataframe
    df['ttft_reward'] = reward_result['ttft_rewards']
    df['tpot_reward'] = reward_result['tpot_rewards']
    df['reward'] = reward_result['combined_rewards']
    
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
    normalizable_features, non_normalizable_features = _get_normalizable_features(df)
    logger.info(f"Detected {len(normalizable_features)} normalizable features and {len(non_normalizable_features)} non-normalizable features")
    logger.debug(f"Normalizable features: {normalizable_features[:5]}...")
    logger.debug(f"Non-normalizable features: {non_normalizable_features[:5]}...")
    
    # Step 6: Initialize or load statistics
    if stats_file and os.path.exists(stats_file):
        logger.info(f"Loading existing statistics from: {stats_file}")
        with open(stats_file, 'rb') as f:
            stats_instance = pickle.load(f)
        logger.info(f"Loaded stats for {len(stats_instance.feature_stats)} features")
    else:
        logger.info("Creating new statistics instance")
        stats_instance = FeatureStats(normalizable_features)
    
    # Step 7: Normalize features
    logger.info("Starting feature normalization...")
    for feature in normalizable_features:
        try:
            _normalize_single_feature(df, feature, stats_instance, is_training=True)
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
    parser.add_argument('--reward-function', '-r', default='linear_simple', 
                       choices=['linear_simple', 'linear_simple_extended', 'piecewise_linear_steeper_gradient'],
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
