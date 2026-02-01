#!/usr/bin/env python3

# preprocess.py

import pandas as pd
import numpy as np
import json
import ast
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
import argparse
import sys
import time
from logger import logger
import utils as utils

# ============================================================================
# MODULE-LEVEL CONSTANTS FOR PERFORMANCE OPTIMIZATION
# ============================================================================
# Pre-computed constants to avoid repeated string operations in hot paths

# Constants for parse_log_message
_LATENCY_METRICS_PREFIX = "latency_metrics@"
_LATENCY_METRICS_PREFIX_LEN = 16  # len("latency_metrics@")

# NOTE: GPU feature inclusion is now controlled by INCLUDE_GPU_FEATURES 
# environment variable in routing_agent_service.py, not here.
# GPU column extraction here just parses raw data - actual one-hot 
# encoding happens in encoding.py based on hyperparameters['INCLUDE_GPU_FEATURES']

def parse_json_columns(df, json_columns):
    for col in json_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    return df

def parse_log_file(file_path, pod_ip_mapping=None):
    """
    Parse log file and optionally replace pod IPs with general pod IDs on the fly.

    Args:
        file_path: Path to the log file
        pod_ip_mapping: Optional dict mapping pod IPs to general pod IDs.
                       If provided, replacements happen during parsing (faster than pre-processing).
    """
    import re

    # OPTIMIZED: Read entire file at once for faster I/O
    with open(file_path, 'r') as file:
        content = file.read()

    # OPTIMIZED: Single-pass pod IP replacement on entire content
    if pod_ip_mapping:
        sorted_ips = sorted(pod_ip_mapping.keys(), key=len, reverse=True)
        ip_pattern = re.compile('|'.join(re.escape(ip) for ip in sorted_ips))
        content = ip_pattern.sub(lambda m: pod_ip_mapping[m.group(0)], content)

    lines = content.split('\n')
    del content  # Free memory

    data = []
    column_names = None
    json_columns = []

    for line in lines:
        if not line:  # Skip empty lines
            continue

        # Check if this is a metrics line
        if "latency_metrics" not in line:
            logger.error(f"Invalid line. {line}")
            assert False
        if "**@" in line:
            line = line.split("**@latency_metrics@")[1]
        parts = line.split('@')
        row = {}

        # Get column names from first row only
        if column_names is None:
            column_names = []
            for i in range(0, len(parts), 2):
                if i + 1 >= len(parts):
                    break
                column_names.append(parts[i])

        for i in range(0, len(parts), 2):
            if i + 1 >= len(parts):
                break
            column_name = parts[i]
            value = parts[i+1]
            # Handle JSON objects
            if value.startswith('{') and value.endswith('}'):
                try:
                    # NEW: Fix escaped quotes issue - replace \" with " before parsing
                    fixed_value = value.replace('\\"', '"')
                    if column_name not in json_columns:
                        json_columns.append(column_name)
                    row[column_name] = json.loads(fixed_value)
                except Exception as e:
                    logger.error(f"Error decoding JSON, column: {column_name}, value: {value}")
                    logger.error(f"Error: {e}")
                    # Since we can't parse it, store as string to avoid losing data
                    row[column_name] = value
            # Handle 'null' as empty dict for JSON columns (gateway sometimes logs null instead of {})
            elif value == 'null':
                if column_name not in json_columns:
                    json_columns.append(column_name)
                row[column_name] = {}
            else:
                try:
                    row[column_name] = int(value)
                except ValueError:
                    try:
                        row[column_name] = float(value)
                    except ValueError:
                        row[column_name] = value
        data.append(row)

    parsed_df = pd.DataFrame(data, columns=column_names)
    if len(parsed_df) == 0:
        logger.error("No data found in the log file.")
        assert False
    return parsed_df, json_columns

def safe_parse_json(json_str):
    """Safely parse Python dictionary-like strings or JSON strings"""
    # If already a dictionary, return as is
    if isinstance(json_str, dict):
        return json_str
    if pd.isna(json_str) or not json_str:
        logger.error(f"ERROR: Empty or NaN JSON string: {str(json_str)}...")
        assert False
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        if isinstance(json_str, str):
            return json.loads(json_str.replace("'", '"'))
        else:
            logger.error(f"ERROR: Invalid JSON string: {str(json_str)}...")
            assert False

def extract_key_pod_metrics(pod_metrics, pod_id):
    """Extract the most relevant metrics for a pod from the pod metrics"""
    if pod_id not in pod_metrics:
        logger.error(f"Error: Pod ID {pod_id} not found in pod metrics.")
        assert False
    return {
        'last_second_avg_ttft_ms': pod_metrics[pod_id]['last_second_avg_ttft_ms'],
        'last_second_avg_tpot_ms': pod_metrics[pod_id]['last_second_avg_tpot_ms'],
        'last_second_p99_ttft_ms': pod_metrics[pod_id]['last_second_p99_ttft_ms'],
        'last_second_p99_tpot_ms': pod_metrics[pod_id]['last_second_p99_tpot_ms'],
        'last_second_total_requests': pod_metrics[pod_id]['last_second_total_requests'],
        'last_second_total_tokens': pod_metrics[pod_id]['last_second_total_tokens'],
        'last_second_total_decode_tokens': pod_metrics[pod_id]['last_second_total_decode_tokens'],
        'last_second_total_prefill_tokens': pod_metrics[pod_id]['last_second_total_prefill_tokens'],
    }
    
def calculate_combined_rewards(ttft_rewards, tpot_rewards, ttft_reward_weight):
    return ttft_reward_weight*ttft_rewards + max(0, (1-ttft_reward_weight))*tpot_rewards

def calculate_rewards_simple(ttft_values, tpot_values, ttft_slo, avg_tpot_slo, ttft_reward_weight):
        ttft_rewards = np.where(
            ttft_values <= 0, 
            0.5,  # Maximum reward for perfect performance
            np.where(
                ttft_values <= ttft_slo,
                0.5 - (0.4 * ttft_values / ttft_slo),  # Linear scaling
                -0.1 - (0.4 * np.minimum(1.0, (ttft_values - ttft_slo) / ttft_slo))  # Negative reward
            )
        )

        tpot_rewards = np.where(
            tpot_values <= 0,
            -0.5,  # Penalize invalid values
            np.where(
                tpot_values <= avg_tpot_slo,
                0.1 + (0.4 * (1 - tpot_values / avg_tpot_slo)),  # Linear scaling
                -0.1 - (0.4 * np.minimum(1.0, (tpot_values - avg_tpot_slo) / avg_tpot_slo))  # Negative reward
            )
        )
        return {
            'ttft_rewards': ttft_rewards,
            'tpot_rewards': tpot_rewards,
            'combined_rewards': calculate_combined_rewards(ttft_rewards, tpot_rewards, ttft_reward_weight),
        }


def calculate_rewards_simple_extended(ttft_values, tpot_values, ttft_slo, avg_tpot_slo, ttft_reward_weight):
    """
    expressible TTFT range: 0-5000ms, 
    expressible TPOT: 0-200ms, 
    ttft reward range: -1.7 ~ 0.5
    ttft reward range: -1.3 ~ 0.5
    combined reward range: -3.0 ~ 1.0
    """
    ttft_rewards = np.where(
        ttft_values <= 0, 
        0.5,  # Maximum reward for perfect performance
        np.where(
            ttft_values <= ttft_slo,
            0.5 - (0.4 * ttft_values / ttft_slo),  # Linear scaling
            # CHANGE: 1.0 -> 4.0 for TTFT to cover 0-5000ms (5x SLO)
            -0.1 - (0.4 * np.minimum(4.0, (ttft_values - ttft_slo) / ttft_slo))  # Extended penalty
        )
    )

    tpot_rewards = np.where(
        tpot_values <= 0,
        -0.5,  # Penalize invalid values
        np.where(
            tpot_values <= avg_tpot_slo,
            0.1 + (0.4 * (1 - tpot_values / avg_tpot_slo)),  # Linear scaling
            # CHANGE: 1.0 -> 3.0 for TPOT to cover 0-200ms (4x SLO)  
            -0.1 - (0.4 * np.minimum(3.0, (tpot_values - avg_tpot_slo) / avg_tpot_slo))  # Extended penalty
        )
    )
    return {
        'ttft_rewards': ttft_rewards,
        'tpot_rewards': tpot_rewards, 
        'combined_rewards': calculate_combined_rewards(ttft_rewards, tpot_rewards, ttft_reward_weight),
    }
    
def calculate_rewards_piecewise_linear_steeper_gradient(ttft_values, tpot_values, ttft_slo, avg_tpot_slo, ttft_reward_weight):
    """
    Maximum performance reward function with steeper optimization gradient within the good performance region
    
    Expressible ranges:
    - TTFT: 0-5000ms 
    - TPOT: 0-200ms
    
    Reward ranges:
    - TTFT: -1.7 to +2.1 (range: 3.8)
    - TPOT: -1.3 to +1.7 (range: 3.0)  
    - Combined: -3.0 to +3.8
    
    This creates strong incentives for optimal performance while maintaining
    harsh penalties for SLO violations.
    """
    
    ttft_rewards = np.where(
        ttft_values <= 0, 
        2.1,  # Maximum reward for perfect TTFT
        np.where(
            ttft_values <= ttft_slo,
            # Balanced positive scaling: 0.5 to 2.1 (matches penalty range of 1.6)
            0.5 + (1.6 * (1 - ttft_values / ttft_slo)),  
            # Extended harsh penalty: -0.1 to -1.7
            -0.1 - (0.4 * np.minimum(4.0, (ttft_values - ttft_slo) / ttft_slo))
        )
    )

    tpot_rewards = np.where(
        tpot_values <= 0,
        -0.5,  # Still penalize invalid values
        np.where(
            tpot_values <= avg_tpot_slo,
            # Balanced positive scaling: 0.5 to 1.7 (matches penalty range of 1.2)  
            0.5 + (1.2 * (1 - tpot_values / avg_tpot_slo)),
            # Extended harsh penalty: -0.1 to -1.3
            -0.1 - (0.4 * np.minimum(3.0, (tpot_values - avg_tpot_slo) / avg_tpot_slo))
        )
    )
    
    return {
        'ttft_rewards': ttft_rewards,
        'tpot_rewards': tpot_rewards,
        'combined_rewards': calculate_combined_rewards(ttft_rewards, tpot_rewards, ttft_reward_weight),
    }



def calculate_rewards_inverse_latency(ttft_values, tpot_values, ttft_slo, avg_tpot_slo, ttft_reward_weight):
    """
    Inverse latency reward function: reward = k / (latency + offset)
    
    This is a smooth, monotonic, convex function that:
    - Heavily rewards low latencies (hyperbolic curve)
    - Naturally penalizes high latencies (approaches 0)
    - No discontinuities or thresholds
    - Exhibits risk-seeking behavior (Jensen's inequality)
    
    Scaling:
    - TTFT: reward = 1000 / (ttft + 100) → range [0.1, 10.0]
    - TPOT: reward = 100 / (tpot + 10) → range [0.1, 10.0]
    
    Combined reward range (with weight=2.0): [0.2, 20.0]
    Normalized to [-1, +1]: (raw - 10.1) / 9.9
    """
    
    # Avoid division by zero with small offset
    # Scale factor chosen so typical latencies give rewards in [0, 10] range
    ttft_raw = 1000.0 / (ttft_values + 100.0)  # Range: [10.0, 0.1] for [0ms, 10000ms]
    tpot_raw = 100.0 / (tpot_values + 10.0)    # Range: [10.0, 0.1] for [0ms, 1000ms]
    
    # Normalize to similar range as other reward functions [-1, +1]
    # Map [0.1, 10.0] → [-1, +1]
    ttft_rewards = (ttft_raw - 5.05) / 4.95  # Center around 0, range ≈ [-1, +1]
    tpot_rewards = (tpot_raw - 5.05) / 4.95
    
    return {
        'ttft_rewards': ttft_rewards,
        'tpot_rewards': tpot_rewards,
        'combined_rewards': calculate_combined_rewards(ttft_rewards, tpot_rewards, ttft_reward_weight),
    }


def calculate_rewards_latency_optimization(ttft_values, tpot_values, ttft_slo, avg_tpot_slo, ttft_reward_weight):
    """
    Latency optimization reward function that continuously rewards lower latency.
    
    This function provides meaningful reward differences across the entire range,
    encouraging the system to minimize latency even within SLO boundaries.
    
    Expressible ranges:
    - TTFT: 0-10000ms 
    - TPOT: 0-200ms
    
    Reward ranges:
    - TTFT: -2.0 to +2.0 (range: 4.0)
    - TPOT: -1.5 to +1.5 (range: 3.0)  
    - Combined: -3.5 to +3.5
    
    Key features:
    - Linear scaling across entire range for meaningful differences
    - Strong incentives for lower latency (0ms = max reward)
    - Harsh penalties for SLO violations
    - Continuous optimization signal
    """
    
    # TTFT rewards with linear scaling across entire range for latency optimization
    ttft_rewards = np.where(
        ttft_values <= 0, 
        2.0,  # Maximum reward for perfect TTFT
        np.where(
            ttft_values <= ttft_slo,
            # Linear scaling within SLO: 2.0 to 0.0 (meaningful differences)
            2.0 - (2.0 * ttft_values / ttft_slo),  
            # Extended harsh penalty for SLO violations: -0.5 to -2.0
            -0.5 - (0.4 * np.minimum(3.75, (ttft_values - ttft_slo) / ttft_slo))
        )
    )

    # TPOT rewards with linear scaling across entire range for latency optimization
    tpot_rewards = np.where(
        tpot_values <= 0,
        -0.5,  # Penalize invalid values
        np.where(
            tpot_values <= avg_tpot_slo,
            # Linear scaling within SLO: 1.5 to 0.0 (meaningful differences)
            1.5 - (1.5 * tpot_values / avg_tpot_slo),
            # Extended harsh penalty for SLO violations: -0.5 to -1.5
            -0.5 - (0.4 * np.minimum(2.5, (tpot_values - avg_tpot_slo) / avg_tpot_slo))
        )
    )
    
    return {
        'ttft_rewards': ttft_rewards,
        'tpot_rewards': tpot_rewards, 
        'combined_rewards': calculate_combined_rewards(ttft_rewards, tpot_rewards, ttft_reward_weight), }


def calculate_rewards_simple_latency_minimization(ttft_values, tpot_values, ttft_reward_weight):
    """
    Simple global latency minimization - NO normalization tricks!
    
    Philosophy: The reward function should directly reflect the optimization objective.
    
    Goal: Minimize latency globally
    → Reward: Higher for lower latency (context-blind, absolute performance)
    
    Formula: reward = -log(latency_ms + 1)
    
    Why log transform?
    1. Compresses extreme values (numerical stability for skewed distributions)
    2. Diminishing returns: 100ms→50ms improvement > 5000ms→4950ms improvement
    3. Always monotonic: lower latency = higher reward
    4. Bounded gradients: prevents extreme outliers from dominating training
    
    Context-awareness comes from FEATURES, not reward normalization:
    - Model sees input_tokens, output_tokens, kv_hit_ratio, inflight_requests
    - Learns: E[latency | input=long, instance=A] vs E[latency | input=long, instance=B]
    - Routes to whichever instance has lower EXPECTED latency for THIS specific request
    
    Properties:
    - Reward distribution will mirror (inverse of) latency distribution
    - Skewed latency → skewed rewards (this is CORRECT, not a bug!)
    - No artificial centering at 0
    - Absolute performance metric: -log(100) is ALWAYS better than -log(5000)
    
    Args:
        ttft_values: Actual TTFT in ms
        tpot_values: Actual TPOT in ms
        ttft_reward_weight: Weight for TTFT vs TPOT (0-1 range)
    
    Returns:
        Dict with ttft_rewards, tpot_rewards, combined_rewards
    """
    
    # Negative log-latency: lower latency → higher (less negative) reward
    # Add 1 to avoid log(0)
    ttft_rewards = -np.log(ttft_values + 1.0)
    tpot_rewards = -np.log(tpot_values + 1.0)
    
    # Examples:
    # TTFT = 100ms → reward = -log(101) = -4.615
    # TTFT = 500ms → reward = -log(501) = -6.216
    # TTFT = 5000ms → reward = -log(5001) = -8.517
    
    return {
        'ttft_rewards': ttft_rewards,
        'tpot_rewards': tpot_rewards,
        'combined_rewards': calculate_combined_rewards(ttft_rewards, tpot_rewards, ttft_reward_weight), }


def calculate_rewards_negative_reciprocal(ttft_values, tpot_values, ttft_reward_weight):
    """
    Negative reciprocal reward: reward = -K / latency
    
    Philosophy: More sensitive to improvements at low latencies.
    
    Why better than log?
    - Larger reward differences across the typical latency range
    - 500ms → 1000ms gives bigger penalty than 5000ms → 5500ms (desirable!)
    - More intuitive: reward is inversely proportional to latency
    
    Examples:
    - TTFT = 100ms  → reward = -1000/100  = -10.0
    - TTFT = 500ms  → reward = -1000/500  = -2.0  (8.0 difference)
    - TTFT = 1000ms → reward = -1000/1000 = -1.0  (1.0 difference)
    - TTFT = 5000ms → reward = -1000/5000 = -0.2  (0.8 difference)
    
    Range for typical latencies (100-10000ms): -10.0 to -0.1
    
    Pros:
    - Good differentiation in low-latency range (where it matters most)
    - Natural diminishing returns for high latency
    - Simple, interpretable
    
    Args:
        ttft_values: Actual TTFT in ms
        tpot_values: Actual TPOT in ms
        ttft_reward_weight: Weight for TTFT vs TPOT
    """
    
    # Negative reciprocal with scaling factor
    # Use 1000 as numerator to get reasonable reward magnitudes
    ttft_rewards = -1000.0 / np.maximum(ttft_values, 1.0)  # Avoid division by zero
    tpot_rewards = -1000.0 / np.maximum(tpot_values, 1.0)
    
    return {
        'ttft_rewards': ttft_rewards,
        'tpot_rewards': tpot_rewards,
        'combined_rewards': calculate_combined_rewards(ttft_rewards, tpot_rewards, ttft_reward_weight),
    }


def calculate_rewards_negative_linear(ttft_values, tpot_values, ttft_reward_weight):
    """
    Negative linear reward: reward = -latency / K
    
    Philosophy: Simplest possible - directly penalize latency linearly.
    
    Why this might work:
    - Most direct representation of "minimize latency"
    - Equal reward difference for equal latency difference
    - No compression or amplification
    - Lets the model learn the true latency landscape
    
    Examples:
    - TTFT = 100ms  → reward = -100/1000  = -0.1
    - TTFT = 500ms  → reward = -500/1000  = -0.5  (0.4 difference)
    - TTFT = 1000ms → reward = -1000/1000 = -1.0  (0.5 difference)
    - TTFT = 5000ms → reward = -5000/1000 = -5.0  (4.0 difference)
    
    Range for typical latencies (100-10000ms): -0.1 to -10.0
    
    Pros:
    - Extremely simple and interpretable
    - Linear relationship preserved
    - Large reward spread across full range
    
    Cons:
    - May be sensitive to outliers (very high latency gets very negative reward)
    - But that's arguably correct! We REALLY want to avoid 10s latencies
    
    Args:
        ttft_values: Actual TTFT in ms
        tpot_values: Actual TPOT in ms
        ttft_reward_weight: Weight for TTFT vs TPOT
    """
    
    # Simple linear scaling
    ttft_rewards = -ttft_values / 1000.0
    tpot_rewards = -tpot_values / 1000.0
    
    return {
        'ttft_rewards': ttft_rewards,
        'tpot_rewards': tpot_rewards,
        'combined_rewards': calculate_combined_rewards(ttft_rewards, tpot_rewards, ttft_reward_weight), }


def calculate_rewards_negative_squared(ttft_values, tpot_values, ttft_reward_weight):
    """
    Negative squared reward: reward = -(latency / K)^2
    
    Philosophy: Heavily penalize high latencies (tail latencies are REALLY bad).
    
    Why this might work:
    - Quadratic penalty → much worse reward for tail latencies
    - Aligns with user experience: 5s latency feels >2x worse than 2.5s
    - Incentivizes avoiding worst-case scenarios
    
    Examples:
    - TTFT = 100ms  → reward = -(100/1000)^2  = -0.01
    - TTFT = 500ms  → reward = -(500/1000)^2  = -0.25  (0.24 difference)
    - TTFT = 1000ms → reward = -(1000/1000)^2 = -1.0   (0.75 difference)
    - TTFT = 5000ms → reward = -(5000/1000)^2 = -25.0  (24.0 difference!)
    
    Range for typical latencies (100-10000ms): -0.01 to -100
    
    Pros:
    - Very strong signal to avoid tail latencies
    - Good differentiation across entire range
    - Reflects non-linear user dissatisfaction
    
    Cons:
    - Large reward spread might destabilize training if outliers exist
    - May need reward clipping
    
    Args:
        ttft_values: Actual TTFT in ms
        tpot_values: Actual TPOT in ms
        ttft_reward_weight: Weight for TTFT vs TPOT
    """
    
    # Squared penalty with scaling
    ttft_rewards = -np.square(ttft_values / 1000.0)
    tpot_rewards = -np.square(tpot_values / 1000.0)
    
    # Optional: Clip extreme values to prevent training instability
    # Uncomment if needed
    # ttft_rewards = np.clip(ttft_rewards, -50.0, 0.0)
    # tpot_rewards = np.clip(tpot_rewards, -50.0, 0.0)
    
    return {
        'ttft_rewards': ttft_rewards,
        'tpot_rewards': tpot_rewards,
        'combined_rewards': calculate_combined_rewards(ttft_rewards, tpot_rewards, ttft_reward_weight),
    }


def calculate_rewards_quantile_based(ttft_values,ot_values, input_tokens, output_tokens,
                                      ttft_reward_weight):
    """
    Data-driven Z-score normalized reward function - NO hard-coded SLOs!
    
    Key Idea: Reward = NEGATIVE Z-SCORE within same input length bucket.
    - Compares apples-to-apples (requests with similar complexity)
    - PRESERVES natural latency distribution shape (skewed → skewed rewards)
    - Automatically adapts to actual system performance
    - No hyperparameter tuning needed
    
    Methodology:
    1. Group requests by input length (3 buckets: short, medium, long)
    2. Within each bucket, calculate mean and std of latencies
    3. Z-score normalize: z = -(latency - mean) / std
       - Negative sign: lower latency → higher reward
    4. Clip extreme outliers to [-3, +3] range
    
    Properties:
    - Fast requests (below mean) → positive rewards
    - Slow requests (above mean) → negative rewards
    - Distribution shape matches latency distribution (natural!)
    - Standard deviations from mean = intuitive interpretation
    
    Args:
        ttft_values: Actual TTFT in ms
        tpot_values: Actual TPOT in ms
        input_tokens: Number of input tokens (for context grouping)
        output_tokens: Number of output tokens
        ttft_reward_weight: Weight for TTFT vs TPOT (default 0.7)
    
    Returns:
        Dict with ttft_rewards, tpot_rewards, combined_rewards
    """
    
    # Bucket by input length (context-aware grouping)
    input_quantiles = np.percentile(input_tokens, [0, 33, 67, 100])
    
    ttft_rewards = np.zeros_like(ttft_values, dtype=np.float64)
    tpot_rewards = np.zeros_like(tpot_values, dtype=np.float64)
    
    # Process each input length bucket separately
    for i in range(3):
        low = input_quantiles[i]
        high = input_quantiles[i+1]
        mask = (input_tokens >= low) & (input_tokens < high) if i < 2 else (input_tokens >= low)
        
        if mask.sum() == 0:
            continue
        
        # === TTFT Rewards: Z-score normalization ===
        bucket_ttft = ttft_values[mask]
        
        # Calculate mean and std within this bucket
        mean_ttft = np.mean(bucket_ttft)
        std_ttft = np.std(bucket_ttft)
        
        # Z-score: negative sign so lower latency → higher reward
        # Add small epsilon to prevent division by zero
        if std_ttft > 1e-6:
            bucket_ttft_rewards = -(bucket_ttft - mean_ttft) / std_ttft
        else:
            # If all latencies are identical, give neutral reward
            bucket_ttft_rewards = np.zeros_like(bucket_ttft)
        
        # Clip extreme outliers to [-3, +3] range (3 standard deviations)
        bucket_ttft_rewards = np.clip(bucket_ttft_rewards, -3.0, 3.0)
        
        ttft_rewards[mask] = bucket_ttft_rewards
        
        # === TPOT Rewards: Same Z-score normalization ===
        bucket_tpot = tpot_values[mask]
        
        mean_tpot = np.mean(bucket_tpot)
        std_tpot = np.std(bucket_tpot)
        
        if std_tpot > 1e-6:
            bucket_tpot_rewards = -(bucket_tpot - mean_tpot) / std_tpot
        else:
            bucket_tpot_rewards = np.zeros_like(bucket_tpot)
        
        bucket_tpot_rewards = np.clip(bucket_tpot_rewards, -3.0, 3.0)
        
        tpot_rewards[mask] = bucket_tpot_rewards
    
    return {
        'ttft_rewards': ttft_rewards,
        'tpot_rewards': tpot_rewards,
        'combined_rewards': calculate_combined_rewards(ttft_rewards, tpot_rewards, ttft_reward_weight),
    }


def calculate_rewards_absolute_latency(ttft_values, tpot_values, 
                                       ttft_slo, tpot_slo,
                                       ttft_reward_weight):
    """
    Absolute latency-based reward that TRANSFERS across distributions.
    
    Unlike quantile-based (relative rankings), this uses absolute thresholds.
    
    Formula:
    - Below SLO: reward = 1.0 - (latency / SLO)     → [0, 1]
    - Above SLO: reward = -((latency / SLO) - 1.0)  → [-5, 0]
    
    Args:
        ttft_values: TTFT in ms
        tpot_values: TPOT in ms
        ttft_slo: TTFT SLO in ms (default: 15000)
        tpot_slo: TPOT SLO in ms (default: 100)
        ttft_reward_weight: Weight for TTFT vs TPOT
    
    Returns:
        Dict with ttft_rewards, tpot_rewards, combined_rewards
    """
    import numpy as np
    
    ttft_rewards = np.where(
        ttft_values <= ttft_slo,
        1.0 - (ttft_values / ttft_slo),
        -np.clip((ttft_values / ttft_slo) - 1.0, 0, 5)
    )
    
    tpot_rewards = np.where(
        tpot_values <= tpot_slo,
        1.0 - (tpot_values / tpot_slo),
        -np.clip((tpot_values / tpot_slo) - 1.0, 0, 5)
    )
    
    return {
        'ttft_rewards': ttft_rewards,
        'tpot_rewards': tpot_rewards,
        'combined_rewards': calculate_combined_rewards(ttft_rewards, tpot_rewards, ttft_reward_weight), }


def calculate_rewards_throughput_based(ttft_values, tpot_values, input_tokens, ttft_reward_weight):
    """
    OPTION A: Throughput-based reward for LLM inference (context-aware, transferable).
    
    Theory: Reward = log(throughput) where throughput = tokens/second.
    This AUTOMATICALLY accounts for input length differences!
    
    Key advantages:
    1. Input-length agnostic: 500ms for 100 tokens vs 2500ms for 5000 tokens compared fairly
    2. Hardware-specific: Different GPUs have different throughput (transferable metric)
    3. Absolute performance: 2000 tok/s is ALWAYS better than 200 tok/s
    4. No hyperparameters: No SLO, no normalization constants needed
    5. Theoretically sound: Bounded variance for policy gradient
    
    Examples:
        Short prompt: 100 tokens @ 500ms = 200 tok/s → reward = log(201) ≈ 5.3
        Long prompt:  5000 tokens @ 2500ms = 2000 tok/s → reward = log(2001) ≈ 7.6
        Model learns: "Long prompt is 10x more efficient! Route to that pod!"
    
    Reward ranges (for typical LLM inference):
        Prefill throughput: 100-5000 tokens/sec → log reward: [4.6, 8.5]
        Decode throughput: 10-100 tokens/sec → log reward: [2.4, 4.6]
        Combined: [4.0, 7.5] (compressed, bounded variance)
    
    Args:
        ttft_values: TTFT in ms (time to first token - prefill latency)
        tpot_values: TPOT in ms (time per output token - decode latency)
        input_tokens: Number of input tokens (for throughput calculation)
        ttft_reward_weight: Weight for TTFT vs TPOT (default 0.7)
    
    Returns:
        Dict with ttft_rewards, tpot_rewards, combined_rewards
    """
    # Convert ms to seconds for throughput calculation
    ttft_seconds = np.maximum(ttft_values / 1000.0, 0.001)  # Avoid division by zero
    tpot_seconds = np.maximum(tpot_values / 1000.0, 0.001)
    
    # Prefill throughput: tokens/second during prefill phase
    prefill_throughput = input_tokens / ttft_seconds
    
    # Decode throughput: tokens/second during decode phase
    # TPOT is already per-token, so inverse gives throughput
    decode_throughput = 1.0 / tpot_seconds
    
    # Log-transform for variance reduction (REINFORCE requirement)
    # Add 1 to avoid log(0) edge case
    ttft_rewards = np.log(prefill_throughput + 1)
    tpot_rewards = np.log(decode_throughput + 1)
    
    return {
        'ttft_rewards': ttft_rewards,
        'tpot_rewards': tpot_rewards,
        'combined_rewards': calculate_combined_rewards(ttft_rewards, tpot_rewards, ttft_reward_weight), }


def calculate_rewards_log_normalized(ttft_values, tpot_values, ttft_p99, tpot_p99, ttft_reward_weight):
    """
    OPTION B: Variance-normalized log reward for policy gradient.
    
    Theory: REINFORCE variance ∝ range(R)². By normalizing both metrics to [0,1] scale,
    we minimize gradient variance while preserving interpretability of weights.
    
    This approach:
    1. Uses training data statistics (p99)normalization
    2. Gives proper meaning to weights: ttft_weight means "70% of reward from TTFT"
    3. Ensures equal gradient contribution from TTFT and TPOT
    4. Context-awareness comes from model features (input_tokens), not reward function
    
    Normalization constants (ttft_p99, tpot_p99) are computed from training data
    and saved in model config for consistent inference.
    
    Reward range: [-1, 0] (bounded, normalized)
    
    Args:
        ttft_values: TTFT in ms
        tpot_values: TPOT in ms
        ttft_p99: 99th percentile of TTFT from training data (normalization constant)
        tpot_p99: 99th percentile of TPOT from training data (normalization constant)
        ttft_reward_weight: Weight for TTFT vs TPOT (default 0.7)
    
    Returns:
        Dict with ttft_rewards, tpot_rewards, combined_rewards
    """
    # Normalize to [0, 1] range using training data p99
    # log(value + 1) / log(p99 + 1) maps [0, p99] → [0, 1]
    ttft_normalized = np.log(ttft_values + 1) / np.log(ttft_p99 + 1)
    tpot_normalized = np.log(tpot_values + 1) / np.log(tpot_p99 + 1)
    
    # Negative sign: lower latency → higher (less negative) reward
    ttft_rewards = -ttft_normalized
    tpot_rewards = -tpot_normalized
    
    return {
        'ttft_rewards': ttft_rewards,
        'tpot_rewards': tpot_rewards,
        'combined_rewards': -(ttft_reward_weight * ttft_normalized + (1 - ttft_reward_weight) * tpot_normalized),
    }


def calculate_rewards_context_aware(ttft_values, tpot_values, input_tokens, output_tokens, 
                                      kv_cache_hit_ratios, base_ttft_slo, avg_tpot_slo, 
                                      ttft_reward_weight):
    """
    LLM-inference-aware reward function that adjusts expectations based on request context.
    
    Domain Knowledge Applied:
    1. TTFT (Prefill Phase):
       - Base overhead: ~50-100ms (scheduling, model loading)
       - Per-token cost: ~0.3-0.5ms/token for typical GPUs (A10, A100)
       - KV cache hits dramatically reduce computation (cached tokens are "free")
       - Formula: Expected_TTFT = base + (effective_tokens * per_token_latency)
       - effective_tokens = input_tokens * (1 - kv_cache_hit_ratio)
    
    2. TPOT (Decode Phase):
       - Should be roughly constant (~10-50ms per generated token)
       - Can degrade slightly with very long contexts (>8K tokens) due to KV cache memory pressure
       - Less sensitive to input length, more to total context length
    
    3. Efficiency Metrics:
       - Throughput: tokens/second during prefill
       - Cache efficiency: reduction in effective work due to KV cache
    
    Args:
        ttft_values: Actual TTFT in ms (time to first token - prefill latency)
        tpot_values: Actual TPOT in ms (time per output token - decode latency)
        input_tokens: Number of input/prefill tokens
        output_tokens: Number of generated/output tokens
        kv_cache_hit_ratios: KV cache hit ratio for selected pod (0.0-1.0)
        base_ttft_slo: Base TTFT SLO in ms (for minimal-length requests)
        avg_tpot_slo: Expected TPOT in ms (target per-token decode latency)
        ttft_reward_weight: Weight for TTFT vs TPOT rewards
    
    Returns:
        Dict with ttft_rewards, tpot_rewards, combined_rewards, and diagnostic info
    
    Example:
        Request A: 50 tokens, 0% cache hit, 200ms TTFT
        - Expected: 100 + 50*0.5 = 125ms
        - Actual: 200ms (60% slower than expected) → Lower reward
        
        Request B: 5000 tokens, 80% cache hit, 800ms TTFT  
        - Effective tokens: 5000 * (1-0.8) = 1000 tokens
        - Expected: 100 + 1000*0.5 = 600ms
        - Actual: 800ms (33% slower) → Better than naive comparison
        
        Request C: 5000 tokens, 0% cache hit, 800ms TTFT
        - Expected: 100 + 5000*0.5 = 2600ms
        - Actual: 800ms (3.25x faster!) → Very high reward
    """
    
    # === TTFT (Prefill) Reward Calculation ===
    # Constants based on LLM inference characteristics
    BASE_OVERHEAD_MS = 50  # Base latency (scheduling, kernel launch, etc.)
    PER_TOKEN_LATENCY_MS = 0.4  # ms per token for typical GPU (A10: ~0.3-0.5ms/token)
    
    # Calculate effective tokens after KV cache benefit
    # KV cache hit means those tokens don't need recomputation
    effective_input_tokens = input_tokens * (1.0 - kv_cache_hit_ratios)
    
    # Calculate context-aware expected TTFT
    # This is what we SHOULD expect given the request characteristics
    expected_ttft = BASE_OVERHEAD_MS + (effective_input_tokens * PER_TOKEN_LATENCY_MS)
    
    # Calculate adaptive SLO based on context
    # Allow more time for longer contexts, but still maintain standards
    adaptive_ttft_slo = np.maximum(
        base_ttft_slo,  # Never go below base SLO
        BASE_OVERHEAD_MS + (input_tokens * PER_TOKEN_LATENCY_MS * 1.5)  # 1.5x for safety margin
    )
    
    # Calculate efficiency: how well did we perform vs. expectations?
    # efficiency > 1.0 means faster than expected (good!)
    # efficiency < 1.0 means slower than expected (bad)
    ttft_efficiency = np.where(
        ttft_values > 0,
        expected_ttft / np.maximum(ttft_values, 1.0),  # Avoid division by zero
        1.0  # Perfect efficiency if instant
    )
    
    # Reward based on both absolute performance AND efficiency
    ttft_rewards = np.where(
        ttft_values <= 0,
        2.0,  # Perfect performance
        np.where(
            ttft_values <= adaptive_ttft_slo,
            # Within SLO: Reward based on efficiency (how much better than expected?)
            # efficiency 1.0 → reward 0.5
            # efficiency 2.0 → reward 2.0 (twice as fast!)
            # efficiency 0.5 → reward -0.5 (twice as slow)
            0.5 + (1.5 * (ttft_efficiency - 1.0)),
            # SLO violation: Harsh penalty scaled by how much we exceeded
            -0.5 - (1.0 * np.minimum(3.0, (ttft_values - adaptive_ttft_slo) / adaptive_ttft_slo))
        )
    )
    
    # === TPOT (Decode) Reward Calculation ===
    # TPOT should be fairly constant, but can degrade with long contexts
    # Long KV cache (input + output tokens) can cause memory bandwidth issues
    total_context_length = input_tokens + output_tokens
    
    # Adjust TPOT expectations for very long contexts
    # Context < 4K: no adjustment
    # Context > 8K: allow 20% more time
    context_length_factor = np.where(
        total_context_length < 4000,
        1.0,
        np.minimum(1.2, 1.0 + (total_context_length - 4000) / 20000)  # Linear increase
    )
    
    adaptive_tpot_slo = avg_tpot_slo * context_length_factor
    
    # TPOT rewards with context-aware expectations
    tpot_rewards = np.where(
        tpot_values <= 0,
        -0.5,  # Penalize invalid values
        np.where(
            tpot_values <= adaptive_tpot_slo,
            # Within SLO: Linear scaling with bonus for excellence
            1.5 - (1.5 * tpot_values / adaptive_tpot_slo),
            # SLO violation: Penalty
            -0.5 - (0.5 * np.minimum(2.5, (tpot_values - adaptive_tpot_slo) / adaptive_tpot_slo))
        )
    )
    
    return {
        'ttft_rewards': ttft_rewards,
        'tpot_rewards': tpot_rewards,
        'combined_rewards': calculate_combined_rewards(ttft_rewards, tpot_rewards, ttft_reward_weight),
        # Diagnostic info (useful for analysis)
        'effective_input_tokens': effective_input_tokens,
        'expected_ttft': expected_ttft,
        'adaptive_ttft_slo': adaptive_ttft_slo,
        'ttft_efficiency': ttft_efficiency,
        'adaptive_tpot_slo': adaptive_tpot_slo,
    }


## new - unified preprocessing function
def preprocess_data_unified(parsed_df, hyperparameters, sorted_all_pod_ids, is_training):
    num_rows = len(parsed_df)
    processing_type = "batch" if num_rows > 1 else "single row"
    logger.debug(f"Processing {num_rows} rows ({processing_type}) with is_training={is_training}")
    
    # Pre-parse all JSON columns once to avoid repeated parsing
    json_columns = [
        'allPodsKvCacheHitRatios',
        'numInflightRequestsAllPods',
        'numInflightPrefillRequestsAllPods',  # Per-pod inflight prefill requests
        'numInflightDecodeRequestsAllPods',   # Per-pod inflight decode requests
        'vllmGPUKVCacheUsage',
        'vllmCPUKVCacheUsage',
        'vllmNumRequestsRunning',
        'vllmNumRequestsWaiting',
        # 'podMetricsLastSecond',  # Made optional - will be handled separately
        'numPrefillTokensForAllPods',
        'numDecodeTokensForAllPods',
        'GPU',
    ]
    
    json_parse_start_time = time.time()
    for col in json_columns:
        if col in parsed_df.columns:
            sample_val = parsed_df[col].iloc[0]
            if isinstance(sample_val, str):
                parsed_df[col] = parsed_df[col].apply(safe_parse_json)
        else:
            logger.warning(f"Column '{col}' not found in parsed DataFrame. Available columns: {list(parsed_df.columns)}")
    
    # # Handle podMetricsLastSecond separately (optional column)
    # if 'podMetricsLastSecond' in parsed_df.columns:
    #     sample_val = parsed_df['podMetricsLastSecond'].iloc[0]
    #     if isinstance(sample_val, str):
    #         parsed_df['podMetricsLastSecond'] = parsed_df['podMetricsLastSecond'].apply(safe_parse_json)
    #     logger.debug("Found podMetricsLastSecond column - will be ignored for feature extraction")
    # else:
    #     logger.debug("podMetricsLastSecond column not found - this is fine, features from this column are not used")
    
    json_parse_overhead = time.time() - json_parse_start_time

    # Collect all unique pod IDs in a single pass
    logger.debug("Collecting all unique pod IDs across the dataset...")
    logger.debug(f"Original dataset shape: {parsed_df.shape}")
    logger.debug(f"Columns: {parsed_df.columns.tolist()}")
    expected_columns = [
        'requestID',
        'selectedpod',
        'ttft',
        'avg_tpot',
        'total_decode_time',
        'e2e',
        'numInputTokens',
        # 'expectedNumOutputTokens',
        'numOutputTokens',
        'numTotalTokens',
        'request_start_time',  # NEW: Add request timing columns
        'request_end_time',    # NEW: Add request timing columns
        'allPodsKvCacheHitRatios',
        'numInflightRequestsAllPods',
        'vllmGPUKVCacheUsage',
        'vllmCPUKVCacheUsage',
        'vllmNumRequestsRunning',
        'vllmNumRequestsWaiting',
        # 'podMetricsLastSecond',  # Optional column - may be empty or missing
        'numPrefillTokensForAllPods',
        'numDecodeTokensForAllPods',
        'GPU',  # GPU model mapping per pod
        'subAlgorithm', # old training data does not have it... so...
        # 'prev_reward', ## uncomment it for scalable RL agent training
    ]
    
    ###########################################
    ## HARDCODE TEMPORARY FIX FOR OLD TRAINING DATA
    if 'subAlgorithm' not in parsed_df.columns:
        parsed_df['subAlgorithm'] = None
    if 'GPU' not in parsed_df.columns:
        # For old data without GPU info, create empty dict
        parsed_df['GPU'] = [{}] * len(parsed_df)
        logger.warning("GPU column not found in parsed data - adding empty GPU mapping for old training data compatibility")
    ###########################################

    # GPU columns (pod_xxxx-GPU) are already parsed above from podMetrics
    # They contain GPU model names like "NVIDIA-A30", "NVIDIA-L40S", etc.
    # No need to do anything here - the GPU info is already in the dataframe
    
    # Check for missing expected columns
    missing_columns = [col for col in expected_columns if col not in parsed_df.columns]
    if missing_columns:
        logger.error(f"Error: Missing expected columns: {missing_columns}")
        logger.error(f"parsed_df.columns: {parsed_df.columns}")
        logger.error(f"expected_columns: {expected_columns}")
        assert False
    
    # Check for unknown columns
    unknown_columns = [col for col in parsed_df.columns if col not in expected_columns]
    if unknown_columns:
        logger.debug(f"Warning: Unused columns: {unknown_columns}")

    numeric_conversion_start_time = time.time()
    # Convert string columns to appropriate types - vectorized
    numeric_columns = [
        'ttft',
        'avg_tpot',
        'total_decode_time',
        'e2e',
        'numInputTokens',
        # 'expectedNumOutputTokens',
        'numOutputTokens',
        'numTotalTokens',
        'request_start_time',
        'request_end_time',
        'prev_reward',
    ]
    
    for col in numeric_columns:
        if col in parsed_df.columns:
            parsed_df[col] = pd.to_numeric(parsed_df[col], errors='coerce')
    numeric_conversion_overhead = time.time() - numeric_conversion_start_time # 0-1ms
    
    # Vectorized processing using pandas operations
    logger.debug("Processing records in vectorized manner...")
    
    get_value_start_time = time.time()
    # Extract base features
    base_data = {
        'request_id': parsed_df['requestID'].values,
        'selected_pod': parsed_df['selectedpod'].values,
        'input_tokens': parsed_df['numInputTokens'].values,
        # 'output_tokens': parsed_df['expectedNumOutputTokens'].values,
        'output_tokens': parsed_df['numOutputTokens'].values,
        'total_tokens': parsed_df['numTotalTokens'].values,
        'ttft': parsed_df['ttft'].values,
        'avg_tpot': parsed_df['avg_tpot'].values,
        'e2e_latency': parsed_df['e2e'].values,
        'request_start_time': parsed_df['request_start_time'].values,
        'request_end_time': parsed_df['request_end_time'].values, 
        'subAlgorithm': parsed_df['subAlgorithm'].values,
        # 'prev_reward': parsed_df['prev_reward'].values,
    }
    
    # GPU info is in pod_xxxx-GPU columns, will be extracted later in encoding phase
    
    # Helper function to safely get column values with default
    def safe_get_column(df, col_name, default_value=None):
        """Safely get column values, returning default if column doesn't exist."""
        if col_name in df.columns:
            return df[col_name].values
        else:
            logger.warning(f"Column '{col_name}' not found, using default value: {default_value}")
            if default_value is None:
                # For dict/list columns, return array of empty dicts
                return np.array([{}] * len(df))
            elif isinstance(default_value, (dict, list)):
                return np.array([default_value] * len(df))
            else:
                return np.full(len(df), default_value)
    
    # Pre-extract all JSON data to avoid repeated parsing
    all_kv_cache = safe_get_column(parsed_df, 'allPodsKvCacheHitRatios', {})
    all_inflight = safe_get_column(parsed_df, 'numInflightRequestsAllPods', {})
    all_inflight_prefill = safe_get_column(parsed_df, 'numInflightPrefillRequestsAllPods', {})  # NEW: Per-pod inflight prefill requests
    all_inflight_decode = safe_get_column(parsed_df, 'numInflightDecodeRequestsAllPods', {})   # NEW: Per-pod inflight decode requests
    all_gpu_cache = safe_get_column(parsed_df, 'vllmGPUKVCacheUsage', {})
    all_cpu_cache = safe_get_column(parsed_df, 'vllmCPUKVCacheUsage', {})
    all_running = safe_get_column(parsed_df, 'vllmNumRequestsRunning', {})
    all_waiting = safe_get_column(parsed_df, 'vllmNumRequestsWaiting', {})
    all_prefill = safe_get_column(parsed_df, 'numPrefillTokensForAllPods', {})
    all_decode = safe_get_column(parsed_df, 'numDecodeTokensForAllPods', {})
    all_gpu_models = safe_get_column(parsed_df, 'GPU', {})  # Extract GPU model mapping
    # NOTE: podMetricsLastSecond features are not used in training anymore
    # all_pod_metrics = parsed_df['podMetricsLastSecond'].values

    # Process pod features for all rows at once
    logger.debug(f"** hyperparameters: {hyperparameters}")
    # If hyperparameters is None or doesn't have EXCLUDED_POD_FEATURES, don't exclude any features
    if hyperparameters is None or 'EXCLUDED_POD_FEATURES' not in hyperparameters:
        logger.debug("No feature exclusion applied (hyperparameters not provided or EXCLUDED_POD_FEATURES not specified)")
        excluded_pod_features = set()
    else:
        excluded_pod_features = set(hyperparameters['EXCLUDED_POD_FEATURES'])
        if 'none' in excluded_pod_features or 'None' in excluded_pod_features:
            excluded_pod_features = set()

    # OPTIMIZED: Extract all pod features in a single pass per feature type using pandas
    # Convert list of dicts to DataFrame for vectorized extraction
    num_rows = len(parsed_df)

    # Helper function to extract pod values from list of dicts efficiently
    def extract_pod_features_fast(data_array, pod_ids, default_val=0):
        """Extract features for all pods from array of dicts in one pass."""
        # Fast path for single-row inference to avoid DataFrame creation
        if num_rows == 1:
            entry = data_array[0] if len(data_array) > 0 else {}
            if isinstance(entry, dict):
                pod_map = entry
            elif entry is None or (isinstance(entry, float) and np.isnan(entry)):
                pod_map = {}
            elif isinstance(entry, str):
                stripped = entry.strip()
                if not stripped or stripped.lower() in ("null", "none"):
                    pod_map = {}
                else:
                    try:
                        pod_map = json.loads(stripped)
                    except Exception:
                        try:
                            pod_map = ast.literal_eval(stripped)
                        except Exception:
                            pod_map = {}
            else:
                pod_map = {}

            result = {}
            for pod_id in pod_ids:
                value = pod_map.get(pod_id, default_val)
                # Preserve non-numeric values (e.g., GPU model strings)
                if isinstance(value, (str, bytes)):
                    result[pod_id] = np.array([value], dtype=object)
                else:
                    result[pod_id] = np.array([value], dtype=np.float32)
            return result

        parse_failures = 0
        normalized_rows = []
        for entry in data_array:
            if isinstance(entry, dict):
                normalized_rows.append(entry)
                continue
            if entry is None or (isinstance(entry, float) and np.isnan(entry)):
                normalized_rows.append({})
                continue
            if isinstance(entry, str):
                stripped = entry.strip()
                if not stripped or stripped.lower() in ("null", "none"):
                    normalized_rows.append({})
                    continue
                try:
                    normalized_rows.append(json.loads(stripped))
                    continue
                except Exception:
                    try:
                        normalized_rows.append(ast.literal_eval(stripped))
                        continue
                    except Exception:
                        parse_failures += 1
                        normalized_rows.append({})
                        continue
            parse_failures += 1
            normalized_rows.append({})
        if parse_failures > 0:
            logger.warning(f"extract_pod_features_fast: {parse_failures} non-dict entries replaced with empty dicts")
        # Convert to DataFrame - this is O(n) but done once per feature type
        df = pd.DataFrame(normalized_rows)
        result = {}
        for pod_id in pod_ids:
            if pod_id in df.columns:
                result[pod_id] = df[pod_id].fillna(default_val).values
            else:
                result[pod_id] = np.full(num_rows, default_val)
        return result

    # Extract all features at once per feature type (much faster than per-pod list comprehensions)
    sorted_pods = sorted_all_pod_ids
    if 'kv_hit_ratio' not in excluded_pod_features:
        kv_cache_features = extract_pod_features_fast(all_kv_cache, sorted_pods, 0)
        base_data.update({f"{pod_id}-kv_hit_ratio": kv_cache_features[pod_id] for pod_id in sorted_pods})

    if 'inflight_requests' not in excluded_pod_features:
        inflight_features = extract_pod_features_fast(all_inflight, sorted_pods, 0)
        base_data.update({f"{pod_id}-inflight_requests": inflight_features[pod_id] for pod_id in sorted_pods})

    # NEW: Extract per-pod inflight prefill requests
    if 'inflight_prefill_requests' not in excluded_pod_features:
        inflight_prefill_features = extract_pod_features_fast(all_inflight_prefill, sorted_pods, 0)
        base_data.update({f"{pod_id}-inflight_prefill_requests": inflight_prefill_features[pod_id] for pod_id in sorted_pods})

    # NEW: Extract per-pod inflight decode requests
    if 'inflight_decode_requests' not in excluded_pod_features:
        inflight_decode_features = extract_pod_features_fast(all_inflight_decode, sorted_pods, 0)
        base_data.update({f"{pod_id}-inflight_decode_requests": inflight_decode_features[pod_id] for pod_id in sorted_pods})

    if 'gpu_kv_cache' not in excluded_pod_features:
        gpu_cache_features = extract_pod_features_fast(all_gpu_cache, sorted_pods, 0)
        base_data.update({f"{pod_id}-gpu_kv_cache": gpu_cache_features[pod_id] for pod_id in sorted_pods})

    if 'cpu_kv_cache' not in excluded_pod_features:
        cpu_cache_features = extract_pod_features_fast(all_cpu_cache, sorted_pods, 0)
        base_data.update({f"{pod_id}-cpu_kv_cache": cpu_cache_features[pod_id] for pod_id in sorted_pods})

    if 'running_requests' not in excluded_pod_features:
        running_features = extract_pod_features_fast(all_running, sorted_pods, 0)
        base_data.update({f"{pod_id}-running_requests": running_features[pod_id] for pod_id in sorted_pods})

    if 'waiting_requests' not in excluded_pod_features:
        waiting_features = extract_pod_features_fast(all_waiting, sorted_pods, 0)
        base_data.update({f"{pod_id}-waiting_requests": waiting_features[pod_id] for pod_id in sorted_pods})

    if 'prefill_tokens' not in excluded_pod_features:
        prefill_features = extract_pod_features_fast(all_prefill, sorted_pods, 0)
        base_data.update({f"{pod_id}-prefill_tokens": prefill_features[pod_id] for pod_id in sorted_pods})

    if 'decode_tokens' not in excluded_pod_features:
        decode_features = extract_pod_features_fast(all_decode, sorted_pods, 0)
        base_data.update({f"{pod_id}-decode_tokens": decode_features[pod_id] for pod_id in sorted_pods})

    # GPU model is a string, use different default
    if 'GPU' not in excluded_pod_features:
        gpu_model_features = extract_pod_features_fast(all_gpu_models, sorted_pods, 'GPU-L3c')
        base_data.update({f"{pod_id}-GPU": gpu_model_features[pod_id] for pod_id in sorted_pods})
    get_value_overhead = time.time() - get_value_start_time # 0ms
    num_rows = len(base_data['request_id'])
    pod_index_start_time = time.time()
    if is_training:
        unique_pods = np.unique(base_data['selected_pod'])
        pod_to_index = {str(pod): idx for idx, pod in enumerate(unique_pods)}
        index_to_pod = {int(idx): str(pod) for pod, idx in pod_to_index.items()}
        selected_pods_array = np.array(base_data['selected_pod'])
        action_values = np.array([pod_to_index[str(pod)] for pod in selected_pods_array])
    else:
        pod_to_index = {}
        index_to_pod = {}
        action_values = None

    ttft_values = np.array(base_data['ttft'], dtype=np.float64)
    tpot_values = np.array(base_data['avg_tpot'], dtype=np.float64)
    pod_index_overhead = time.time() - pod_index_start_time

    reward_calc_overhead = -1

    # Training-specific calculations (rewards and action mapping)
    if is_training:
        base_data.update({
            'action': action_values,
        })

        reward_keys = {'TTFT_SLO', 'AVG_TPOT_SLO', 'TTFT_REWARD_WEIGHT', 'REWARD_FUNCTION'}
        # Skip reward calculation if hyperparameters is None or doesn't have required keys
        if hyperparameters is not None and reward_keys.issubset(hyperparameters.keys()):
            reward_calc_start_time = time.time()
            ttft_slo = hyperparameters['TTFT_SLO']
            avg_tpot_slo = hyperparameters['AVG_TPOT_SLO']
            ttft_reward_weight = hyperparameters['TTFT_REWARD_WEIGHT']
            reward_function = hyperparameters['REWARD_FUNCTION']
            logger.info(
                "Online training reward config: REWARD_FUNCTION=%s, TTFT_SLO=%s, AVG_TPOT_SLO=%s, TTFT_REWARD_WEIGHT=%s",
                reward_function,
                ttft_slo,
                avg_tpot_slo,
                ttft_reward_weight,
            )

            if reward_function == "linear_simple":
                reward = calculate_rewards_simple(ttft_values, tpot_values, ttft_slo, avg_tpot_slo, ttft_reward_weight)
            elif reward_function == "linear_simple_extended":
                reward = calculate_rewards_simple_extended(ttft_values, tpot_values, ttft_slo, avg_tpot_slo, ttft_reward_weight)
            elif reward_function == "piecewise_linear_steeper_gradient":
                reward = calculate_rewards_piecewise_linear_steeper_gradient(ttft_values, tpot_values, ttft_slo, avg_tpot_slo, ttft_reward_weight)
            elif reward_function == "latency_optimized":
                reward = calculate_rewards_latency_optimization(ttft_values, tpot_values, ttft_slo, avg_tpot_slo, ttft_reward_weight)
            elif reward_function == "inverse_latency":
                reward = calculate_rewards_inverse_latency(ttft_values, tpot_values, ttft_slo, avg_tpot_slo, ttft_reward_weight)
            elif reward_function == "simple_latency_minimization":
                reward = calculate_rewards_simple_latency_minimization(
                    ttft_values, tpot_values, ttft_reward_weight
                )
            elif reward_function == "negative_reciprocal":
                reward = calculate_rewards_negative_reciprocal(
                    ttft_values, tpot_values, ttft_reward_weight
                )
            elif reward_function == "negative_linear":
                reward = calculate_rewards_negative_linear(
                    ttft_values, tpot_values, ttft_reward_weight
                )
            elif reward_function == "negative_squared":
                reward = calculate_rewards_negative_squared(
                    ttft_values, tpot_values, ttft_reward_weight
                )
            elif reward_function == "quantile_based":
                reward = calculate_rewards_quantile_based(
                    ttft_values, tpot_values,
                    base_data['input_tokens'],
                    base_data['output_tokens'],
                    ttft_reward_weight
                )
            elif reward_function == "absolute_latency":
                reward = calculate_rewards_absolute_latency(
                    ttft_values, tpot_values,
                    ttft_slo=ttft_slo,
                    tpot_slo=avg_tpot_slo,
                    ttft_reward_weight=ttft_reward_weight
                )
            elif reward_function == "throughput_based":
                # OPTION A: Context-aware throughput-based reward
                input_tokens = np.array(base_data['input_tokens'], dtype=np.float64)
                reward = calculate_rewards_throughput_based(
                    ttft_values, tpot_values, input_tokens, ttft_reward_weight
                )
            elif reward_function == "log_normalized":
                # OPTION B: Variance-normalized log reward
                # Compute normalization constants from data if not in hyperparameters
                if 'TTFT_P99' in hyperparameters and 'TPOT_P99' in hyperparameters:
                    ttft_p99 = hyperparameters['TTFT_P99']
                    tpot_p99 = hyperparameters['TPOT_P99']
                    logger.info(f"Using TTFT_P99={ttft_p99:.2f}ms and TPOT_P99={tpot_p99:.2f}ms from hyperparameters")
                else:
                    # Compute P99 values from the data itself if not in hyperparameters
                    ttft_p99 = float(np.percentile(ttft_values, 99))
                    tpot_p99 = float(np.percentile(tpot_values, 99))
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
                
                reward = calculate_rewards_log_normalized(
                    ttft_values, tpot_values,
                    ttft_p99=ttft_p99,
                    tpot_p99=tpot_p99,
                    ttft_reward_weight=ttft_reward_weight
                )
            elif reward_function == "context_aware":
                # Extract KV cache hit ratios for selected pods
                # For each row, get the KV cache hit ratio of the selected pod
                selected_pods = base_data['selected_pod']
                kv_cache_hit_ratios = np.array([
                    base_data.get(f"{pod}-kv_hit_ratio", [0] * len(selected_pods))[i] / 100.0  # Convert percentage to ratio
                    for i, pod in enumerate(selected_pods)
                ])
                
                input_tokens = np.array(base_data['input_tokens'], dtype=np.float64)
                output_tokens = np.array(base_data['output_tokens'], dtype=np.float64)
                
                reward = calculate_rewards_context_aware(
                    ttft_values, tpot_values, input_tokens, output_tokens,
                    kv_cache_hit_ratios, ttft_slo, avg_tpot_slo, ttft_reward_weight
                )
            else:
                logger.error(f"Unknown reward function: {reward_function}")
                assert False

            base_data.update({
                'avg_tpot_slo_satisfied': tpot_values <= avg_tpot_slo,
                'avg_ttft_slo_satisfied': ttft_values <= ttft_slo,
                'ttft_reward': reward['ttft_rewards'],
                'tpot_reward': reward['tpot_rewards'],
                'reward': reward['combined_rewards'],
            })
            reward_calc_overhead = time.time() - reward_calc_start_time
        else:
            # Reward calculation skipped because hyperparameters is None or missing required keys
            if hyperparameters is None:
                logger.info("Reward calculation skipped: hyperparameters not provided")
            else:
                missing_keys = reward_keys - set(hyperparameters.keys())
                logger.info(f"Reward calculation skipped: missing hyperparameter keys: {missing_keys}")
    create_df_start_time = time.time()
    if num_rows == 1 and not is_training:
        # OPTIMIZATION: Skip DataFrame creation for single-row inference
        # Return dict directly - encoding.py will handle it
        single_row_data = {}
        for key, value in base_data.items():
            if isinstance(value, np.ndarray) and value.shape[0] == 1:
                single_row_data[key] = value[0]
            else:
                single_row_data[key] = value
        processed_df = single_row_data  # Return dict, not DataFrame
        create_df_overhead = time.time() - create_df_start_time
    elif num_rows == 1:
        # Single-row training still needs DataFrame
        single_row_data = {}
        for key, value in base_data.items():
            if isinstance(value, np.ndarray) and value.shape[0] == 1:
                single_row_data[key] = value[0]
            else:
                single_row_data[key] = value
        processed_df = pd.DataFrame(single_row_data, index=[0])
        create_df_overhead = time.time() - create_df_start_time
    else:
        processed_df = pd.DataFrame(base_data)
        create_df_overhead = time.time() - create_df_start_time

    # Replace fillna(0) with a more targeted approach since most values should already be handled
    # Only fill NaN values in specific columns that might have them
    if isinstance(processed_df, dict):
        # Dict input (single-row inference): fill NaN values directly
        for key, value in processed_df.items():
            if pd.isna(value) if not isinstance(value, (list, np.ndarray)) else False:
                processed_df[key] = 0
        logger.debug(f"Processed dict with {len(processed_df)} keys")
    else:
        nan_columns = processed_df.columns[processed_df.isnull().any()].tolist()
        if nan_columns:
            processed_df[nan_columns] = processed_df[nan_columns].fillna(0)
        logger.debug(f"Processed dataset shape: {processed_df.shape}")
        logger.debug(f"Processed columns: {list(processed_df.columns)[:10]}...")

    # Prepare overhead summary
    preprocess_overhead_summary = {
        'json_parse_overhead': json_parse_overhead,
        'column_check_overhead': -1,
        'podmetrics_parse_overhead': -1,
        'numeric_conversion_overhead': numeric_conversion_overhead,
        'get_value_overhead': get_value_overhead,
        'create_df_overhead': create_df_overhead,
        'pod_index_overhead': pod_index_overhead,
        'reward_calc_overhead': reward_calc_overhead,
        'slo_update_overhead': -1,
    }
    
    if is_training:
        # Training mode: return mapping info for action space creation
        # GPU models are now included as per-pod columns (pod_xxxx-gpu_model) in the processed_df
        return processed_df, sorted_all_pod_ids, preprocess_overhead_summary
    else:
        # Inference mode: simplified return for speed
        return processed_df, sorted_all_pod_ids, preprocess_overhead_summary


def _fast_parse_value(value: str):
    """
    Fast type conversion for log message values.
    Matches original behavior exactly:
    - Positive integers: "123" → int
    - Floats (including negative): "-1.5", "1.5" → float
    - Everything else: string

    NOTE: Original did NOT convert negative integers like "-123" to int.
    This is preserved for backward compatibility.
    """
    # Empty value
    if not value:
        return value

    # Fast path for positive integers (most common case)
    # isdigit() only returns True for strings of digits (no sign, no decimal)
    if value.isdigit():
        return int(value)

    # Float detection: matches original logic exactly
    # Original: value.replace('.', '').replace('-', '').isdigit() and value.count('.') == 1
    # This handles: "1.5", "-1.5", "0.123", etc.
    if value.count('.') == 1:
        stripped = value.replace('.', '').replace('-', '')
        if stripped.isdigit():
            try:
                return float(value)
            except ValueError:
                return value

    return value


def parse_log_message(log_message):
    """
    Parse a single log message into a DataFrame (optimized for inference).

    Optimizations:
    - Uses module-level constants for prefix detection
    - Fast type conversion with try/except (faster for valid numbers)
    - Minimal string operations
    """
    # Fast prefix check using 'in' (optimized by Python)
    if "latency_metrics" not in log_message:
        logger.error(f"Invalid line. {log_message}")
        return pd.DataFrame(), []

    # Find start position using pre-computed constant
    start_idx = log_message.find(_LATENCY_METRICS_PREFIX)
    if start_idx == -1:
        return pd.DataFrame(), []
    start_idx += _LATENCY_METRICS_PREFIX_LEN

    # Split only the relevant part
    parts = log_message[start_idx:].split('@')
    row = {}
    json_columns = []

    # Process pairs directly using range (slightly faster than while)
    num_parts = len(parts)
    for i in range(0, num_parts - 1, 2):
        key = parts[i]
        value = parts[i + 1]

        # Fast JSON detection and parsing
        if value and value[0] == '{' and value[-1] == '}':
            try:
                # Only fix quotes if needed
                if '\\"' in value:
                    value = value.replace('\\"', '"')
                row[key] = json.loads(value)
                json_columns.append(key)
            except Exception as e:
                logger.error(f"Error decoding JSON, column: {key}, value: {value}")
                logger.error(f"Error: {e}")
                row[key] = value
        else:
            # Fast type conversion
            row[key] = _fast_parse_value(value)

    # Create DataFrame only if we have data
    if row:
        df = pd.DataFrame([row])
        return df, json_columns
    else:
        return pd.DataFrame(), []


def main(input_file, log_message, hyperparameters, pod_ip_mapping=None):
    """
    Main preprocessing function.

    Args:
        input_file: Path to input file (training mode)
        log_message: Log message string (inference mode)
        hyperparameters: Dict of hyperparameters
        pod_ip_mapping: Optional dict mapping pod IPs to general pod IDs.
                       If provided, replacements happen during parsing (faster than pre-processing).
    """
    preprocess_dataset_overhead_summary = {}
    if input_file == None and (log_message == "" or log_message is None):
        logger.error("Error: Both input_file and log_message are empty or None.")
        assert False
    if input_file is not None and log_message != "":
        logger.error("Error: Both input_file and log_message are provided. Please provide only one.")
        assert False
    if input_file is not None:  # Training path
        parsed_df, json_columns = parse_log_file(input_file, pod_ip_mapping=pod_ip_mapping)
    else:  # Inference path
        parse_start_time = time.time()
        parsed_df, _ = parse_log_message(log_message)
        preprocess_dataset_overhead_summary["parse_log_message"] = time.time() - parse_start_time
    if len(parsed_df) == 0:
        logger.error("No data found after parsing JSON columns.")
        logger.error(f"Log message: {log_message}")
        assert False
    
    # Unified preprocessing for both single row (inference) and batch (training)
    sorted_all_pod_ids = utils.get_sorted_all_pod_ids('batch_dataframe', parsed_df)
    if len(parsed_df) == 1 and input_file is None:
        # Inference mode: single row, no training-specific features needed
        preprocess_start_time = time.time()
        is_training = False
        processed_df, sorted_all_pod_ids, preprocess_dataset_overhead_summary = preprocess_data_unified(parsed_df, hyperparameters, sorted_all_pod_ids, is_training)
        preprocess_dataset_overhead_summary["preprocess_unified_inference"] = time.time() - preprocess_start_time
        mapping_info = None  # No mapping info needed for inference
    else:
        # Training mode: batch processing with full features
        preprocess_start_time = time.time()
        is_training = True
        processed_df, sorted_all_pod_ids, preprocess_dataset_overhead_summary = preprocess_data_unified(parsed_df, hyperparameters, sorted_all_pod_ids, is_training)
        preprocess_dataset_overhead_summary["preprocess_unified_training"] = time.time() - preprocess_start_time
    return processed_df, sorted_all_pod_ids, preprocess_dataset_overhead_summary
