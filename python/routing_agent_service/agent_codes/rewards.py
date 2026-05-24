#!/usr/bin/env python3

# rewards.py
#
# Reward-labeling functions used by the data-prep pipeline (preprocess + data_normalizer + dataset_analyzer)
# to convert raw TTFT/TPOT/E2E latency measurements into reward labels for supervised training.
#
# This module is unrelated to reward_predictor.py — that file contains the NN that LEARNS to predict
# these labels from features. The functions here only know about latencies, not about the model.

import numpy as np
from logger import logger

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


def calculate_rewards_negative_linear_and_prefix_locality(ttft_values, tpot_values, ttft_reward_weight,
                                                           selected_pods, base_data, hyperparameters):
    """
    Negative linear reward + prefix locality bonus.

    reward = -TTFT/1000 + α * (selected_pod_kv_differential / 100)

    The prefix locality bonus teaches the model to MAINTAIN prefix concentration
    that exploration/fallback creates. It rewards routing to pods that already have
    high prefix affinity (high kv_differential) relative to other pods.

    Examples (α=0.5):
    - TTFT=500ms, kv_diff=+40% → reward = -0.5 + 0.5*0.4 = -0.3 (bonus for concentrated pod)
    - TTFT=500ms, kv_diff=0%   → reward = -0.5 + 0.0     = -0.5 (neutral)
    - TTFT=5000ms, kv_diff=+40% → reward = -5.0 + 0.2    = -4.8 (TTFT dominates when high)

    Args:
        ttft_values: Actual TTFT in ms
        tpot_values: Actual TPOT in ms
        ttft_reward_weight: Weight for TTFT vs TPOT
        selected_pods: array of selected pod IDs per sample
        base_data: dict containing pod_XXXX-kv_differential columns
        hyperparameters: dict with PREFIX_LOCALITY_WEIGHT
    """
    ttft_rewards = -ttft_values / 1000.0
    tpot_rewards = -tpot_values / 1000.0
    combined_rewards = calculate_combined_rewards(ttft_rewards, tpot_rewards, ttft_reward_weight)

    prefix_locality_weight = float(hyperparameters.get('PREFIX_LOCALITY_WEIGHT', 0.5))
    locality_bonus = np.zeros(len(selected_pods), dtype=np.float64)
    for i, pod in enumerate(selected_pods):
        diff_key = f"{pod}-kv_differential"
        if diff_key in base_data:
            locality_bonus[i] = base_data[diff_key][i] / 100.0

    combined_with_locality = combined_rewards + prefix_locality_weight * locality_bonus

    logger.info(f"negative_linear_and_prefix_locality: weight={prefix_locality_weight}, "
                f"locality_bonus(mean={locality_bonus.mean():.4f}, std={locality_bonus.std():.4f}), "
                f"reward_before={combined_rewards.mean():.4f}, reward_after={combined_with_locality.mean():.4f}")

    return {
        'ttft_rewards': ttft_rewards,
        'tpot_rewards': tpot_rewards,
        'combined_rewards': combined_with_locality,
    }


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


def calculate_rewards_e2e(e2e_values, reward_function, **kwargs):
    """
    Calculate rewards using end-to-end latency as a single metric.

    Applies the same reward shaping as the corresponding ttft/tpot reward
    functions but operates on e2e_latency directly.

    Args:
        e2e_values: End-to-end latency values in ms
        reward_function: Name of the reward function to apply
        **kwargs: Additional arguments (e2e_slo, e2e_p99, input_tokens, output_tokens, etc.)

    Returns:
        Dict with ttft_rewards (zeros), tpot_rewards (zeros), combined_rewards
    """
    zeros = np.zeros_like(e2e_values, dtype=np.float64)

    if reward_function == "negative_linear":
        rewards = -e2e_values / 1000.0

    elif reward_function == "negative_squared":
        rewards = -np.square(e2e_values / 1000.0)

    elif reward_function == "negative_reciprocal":
        rewards = -1000.0 / np.maximum(e2e_values, 1.0)

    elif reward_function == "simple_latency_minimization":
        rewards = -np.log(e2e_values + 1.0)

    elif reward_function == "log_normalized":
        e2e_p99 = kwargs.get('e2e_p99', None)
        if e2e_p99 is None or e2e_p99 <= 0:
            e2e_p99 = float(np.percentile(e2e_values, 99))
            if e2e_p99 <= 0:
                e2e_p99 = 1.0
        rewards = -np.log(e2e_values + 1) / np.log(e2e_p99 + 1)

    elif reward_function == "inverse_latency":
        e2e_raw = 1000.0 / (e2e_values + 100.0)
        rewards = (e2e_raw - 5.05) / 4.95

    elif reward_function in ("linear_simple", "linear_simple_extended",
                             "piecewise_linear_steeper_gradient", "latency_optimized"):
        # SLO-based: use e2e_slo if provided, otherwise fall back to ttft_slo
        e2e_slo = kwargs.get('e2e_slo', kwargs.get('ttft_slo', 1000))
        rewards = np.where(
            e2e_values <= 0,
            2.0,
            np.where(
                e2e_values <= e2e_slo,
                2.0 - (2.0 * e2e_values / e2e_slo),
                -0.5 - (0.4 * np.minimum(4.0, (e2e_values - e2e_slo) / e2e_slo))
            )
        )

    elif reward_function == "throughput_based":
        input_tokens = kwargs.get('input_tokens')
        output_tokens = kwargs.get('output_tokens')
        if input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
        elif input_tokens is not None:
            total_tokens = input_tokens
        else:
            total_tokens = np.ones_like(e2e_values)
        e2e_seconds = np.maximum(e2e_values / 1000.0, 0.001)
        throughput = total_tokens / e2e_seconds
        rewards = np.log(throughput + 1)

    elif reward_function == "quantile_based":
        input_tokens = kwargs.get('input_tokens')
        if input_tokens is not None:
            input_quantiles = np.percentile(input_tokens, [0, 33, 67, 100])
            rewards = np.zeros_like(e2e_values, dtype=np.float64)
            for i in range(3):
                low = input_quantiles[i]
                high = input_quantiles[i+1]
                mask = (input_tokens >= low) & (input_tokens < high) if i < 2 else (input_tokens >= low)
                if mask.sum() == 0:
                    continue
                bucket_e2e = e2e_values[mask]
                mean_e2e = np.mean(bucket_e2e)
                std_e2e = np.std(bucket_e2e)
                if std_e2e > 1e-6:
                    bucket_rewards = -(bucket_e2e - mean_e2e) / std_e2e
                else:
                    bucket_rewards = np.zeros_like(bucket_e2e)
                rewards[mask] = np.clip(bucket_rewards, -3.0, 3.0)
        else:
            # Fallback to negative linear if no input_tokens
            rewards = -e2e_values / 1000.0

    elif reward_function == "quantile_advantage":
        input_tokens = kwargs.get('input_tokens')
        if input_tokens is not None:
            num_buckets = kwargs.get('num_buckets', 5)
            result = calculate_rewards_quantile_advantage(e2e_values, input_tokens, num_buckets)
            rewards = result['combined_rewards']
        else:
            rewards = -e2e_values / 1000.0

    else:
        logger.error(f"Unsupported reward function for e2e_latency: {reward_function}, falling back to negative_linear")
        rewards = -e2e_values / 1000.0

    return {
        'ttft_rewards': zeros,
        'tpot_rewards': zeros,
        'combined_rewards': rewards,
    }


def calculate_rewards_quantile_based(ttft_values, tpot_values, input_tokens, output_tokens,
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


def calculate_rewards_quantile_advantage(latency_values, input_tokens, num_buckets=5):
    """
    Advantage-based reward for LLM inference routing.

    Groups requests by input length into percentile-based buckets,
    computes per-bucket mean latency as a baseline, and rewards based
    on how much better/worse the observed latency was compared to that
    baseline.

    The baseline cancels out at inference time (it is constant across
    pods for a given request), so even an imperfect baseline cannot
    bias the routing policy.  It only affects training convergence
    speed by centering targets and equalizing gradient magnitudes
    across request difficulties.

    No std division — preserves the natural scaling where long-request
    routing produces larger reward magnitudes (reflecting greater
    system impact).

    Works with any single latency metric (TTFT, E2E, etc.) controlled
    by the caller via LATENCY_METRIC hyperparameter.

    Args:
        latency_values: Latency in ms (np array) — TTFT or E2E
        input_tokens:   Number of input tokens per request (np array)
        num_buckets:    Number of input-length buckets (default 20)

    Returns:
        Dict with ttft_rewards (zeros), tpot_rewards (zeros),
        combined_rewards (advantage per sample).
    """
    zeros = np.zeros_like(latency_values, dtype=np.float64)

    # Bucket by input token count (percentile-based, equal sample count)
    effective_buckets = max(1, min(num_buckets, len(input_tokens)))
    percentile_edges = np.percentile(
        input_tokens, np.linspace(0, 100, effective_buckets + 1)
    )
    bucket_idx = np.clip(
        np.digitize(input_tokens, percentile_edges[1:-1], right=True),
        0, effective_buckets - 1,
    )

    # Per-bucket mean latency (the baseline)
    bucket_means = np.zeros(effective_buckets, dtype=np.float64)
    for b in range(effective_buckets):
        mask = bucket_idx == b
        if mask.sum() > 0:
            bucket_means[b] = latency_values[mask].mean()

    # Advantage: negative (latency - baseline)
    # Positive when pod was faster than average for similar requests
    baseline = bucket_means[bucket_idx]
    rewards = -(latency_values - baseline) / 1000.0

    return {
        'ttft_rewards': zeros,
        'tpot_rewards': zeros,
        'combined_rewards': rewards,
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


# ============================================================================
# Dispatch — single entry point for choosing a reward function by name
# ============================================================================
#
# The dispatcher lives in rewards.py (alongside the functions it dispatches to)
# so that adding a new reward function is a one-file change: write the function,
# register it in the appropriate dict below. This replaces what used to be a
# 100+ line elif chain in data_normalizer.py — that chain was where the
# `gradual_within_slo` typo lived for an unknown amount of time, because a
# string-keyed elif branch can reference a function that doesn't exist and
# only blow up at runtime.
#
# Three categories:
#   * SLO-based: same signature (ttft, tpot, ttft_slo, avg_tpot_slo, ttft_weight)
#   * Latency-only: (ttft, tpot, ttft_weight)
#   * Special: each has its own kwargs and/or fallback behavior
#
# Adding a function: add a key+function in the appropriate dict, OR add a new
# branch in compute_rewards() if it has special kwargs/fallback.


_SLO_REWARDS = {
    'linear_simple':                       calculate_rewards_simple,
    'linear_simple_extended':              calculate_rewards_simple_extended,
    'piecewise_linear_steeper_gradient':   calculate_rewards_piecewise_linear_steeper_gradient,
    'inverse_latency':                     calculate_rewards_inverse_latency,
    'latency_optimized':                   calculate_rewards_latency_optimization,
}

_LATENCY_REWARDS = {
    'simple_latency_minimization':         calculate_rewards_simple_latency_minimization,
    'negative_reciprocal':                 calculate_rewards_negative_reciprocal,
    'negative_linear':                     calculate_rewards_negative_linear,
    'negative_squared':                    calculate_rewards_negative_squared,
}

_SPECIAL_REWARDS = (
    'quantile_based',
    'throughput_based',
    'absolute_latency',
    'log_normalized',
    'quantile_advantage',
    'context_aware',
)

KNOWN_REWARD_FUNCTIONS = (
    set(_SLO_REWARDS) | set(_LATENCY_REWARDS) | set(_SPECIAL_REWARDS)
)


def compute_rewards(reward_function, df, hyperparameters):
    """Dispatch to the named reward function.

    This is the single entry point for reward computation from a processed
    DataFrame (the one consumed by data_normalizer.normalize_processed_data).

    For reward functions that need optional columns (input_tokens / output_tokens),
    falls back to a simpler function if the column is missing — matching the
    behavior of the original elif chain. The fallback choice is preserved
    verbatim from that chain so this is a strict refactor.

    For `log_normalized`, this also mutates hyperparameters in-place to persist
    computed TTFT_P99 / TPOT_P99 values back to the caller (preserved from the
    original elif chain — downstream code expects those keys to be set).

    Args:
        reward_function: name string from `KNOWN_REWARD_FUNCTIONS`.
        df: DataFrame with at minimum `ttft` and `avg_tpot` columns. Some reward
            functions additionally read `input_tokens` and/or `output_tokens`.
        hyperparameters: dict with TTFT_SLO, AVG_TPOT_SLO, TTFT_REWARD_WEIGHT.
            For `log_normalized`, also reads/writes TTFT_P99 and TPOT_P99.

    Returns:
        Dict with keys `ttft_rewards`, `tpot_rewards`, `combined_rewards`
        (numpy arrays).
    """
    ttft = df['ttft'].values
    tpot = df['avg_tpot'].values
    ttft_slo = hyperparameters['TTFT_SLO']
    avg_tpot_slo = hyperparameters['AVG_TPOT_SLO']
    ttft_weight = hyperparameters['TTFT_REWARD_WEIGHT']

    if reward_function in _SLO_REWARDS:
        return _SLO_REWARDS[reward_function](ttft, tpot, ttft_slo, avg_tpot_slo, ttft_weight)

    if reward_function in _LATENCY_REWARDS:
        return _LATENCY_REWARDS[reward_function](ttft, tpot, ttft_weight)

    if reward_function == 'quantile_based':
        if 'input_tokens' in df.columns and 'output_tokens' in df.columns:
            return calculate_rewards_quantile_based(
                ttft, tpot, df['input_tokens'].values, df['output_tokens'].values, ttft_weight)
        logger.error("quantile_based reward function requires input_tokens and output_tokens columns")
        logger.error("Falling back to latency_optimized for post-processing")
        return calculate_rewards_latency_optimization(ttft, tpot, ttft_slo, avg_tpot_slo, ttft_weight)

    if reward_function == 'throughput_based':
        logger.info("Calculating throughput-based rewards (context-aware, input-length agnostic)")
        if 'input_tokens' in df.columns:
            return calculate_rewards_throughput_based(ttft, tpot, df['input_tokens'].values, ttft_weight)
        logger.error("throughput_based reward function requires input_tokens column")
        logger.error("Falling back to simple_latency_minimization for post-processing")
        return calculate_rewards_simple_latency_minimization(ttft, tpot, ttft_weight)

    if reward_function == 'absolute_latency':
        logger.info("Calculating absolute latency rewards (transferable across distributions)")
        return calculate_rewards_absolute_latency(
            ttft, tpot,
            ttft_slo=hyperparameters.get('TTFT_SLO', 15000),
            tpot_slo=hyperparameters.get('AVG_TPOT_SLO', 100),
            ttft_reward_weight=ttft_weight,
        )

    if reward_function == 'quantile_advantage':
        if 'input_tokens' in df.columns:
            return calculate_rewards_quantile_advantage(ttft, df['input_tokens'].values, num_buckets=5)
        logger.error("quantile_advantage reward function requires input_tokens column")
        logger.error("Falling back to negative_linear for post-processing")
        return calculate_rewards_negative_linear(ttft, tpot, ttft_weight)

    if reward_function == 'context_aware':
        logger.error("context_aware reward function requires detailed context data (input_tokens, kv_cache_hit_ratios) not available in this tool")
        logger.error("Falling back to latency_optimized for post-processing")
        return calculate_rewards_latency_optimization(ttft, tpot, ttft_slo, avg_tpot_slo, ttft_weight)

    if reward_function == 'log_normalized':
        return _compute_log_normalized(df, ttft, tpot, ttft_weight, hyperparameters)

    raise ValueError(
        f"Unknown reward function: {reward_function!r}. "
        f"Known: {sorted(KNOWN_REWARD_FUNCTIONS)}"
    )


def _compute_log_normalized(df, ttft, tpot, ttft_weight, hyperparameters):
    """log_normalized branch: P99 resolution + NaN validation.

    Pulled out into a helper because it has more behavior than the others —
    it (a) lazily computes P99 from data when missing from hyperparameters,
    (b) persists P99 back to hyperparameters as a side effect, (c) raises on
    invalid P99 values, and (d) raises on NaN-producing outputs.
    """
    logger.info("Calculating variance-normalized log rewards")
    if 'TTFT_P99' in hyperparameters and 'TPOT_P99' in hyperparameters:
        ttft_p99 = hyperparameters['TTFT_P99']
        tpot_p99 = hyperparameters['TPOT_P99']
        logger.info(f"Using TTFT_P99={ttft_p99:.2f}ms and TPOT_P99={tpot_p99:.2f}ms from hyperparameters")
    else:
        ttft_p99 = float(df['ttft'].quantile(0.99))
        tpot_p99 = float(df['avg_tpot'].quantile(0.99))
        logger.info(f"TTFT_P99 and TPOT_P99 not in hyperparameters, computing from data: TTFT_P99={ttft_p99:.2f}ms, TPOT_P99={tpot_p99:.2f}ms")
        hyperparameters['TTFT_P99'] = ttft_p99
        hyperparameters['TPOT_P99'] = tpot_p99

    if tpot_p99 <= 0:
        logger.warning(f"TPOT_P99 is {tpot_p99} (all TPOT values are 0). Using minimum value of 1.0 for log calculation.")
        tpot_p99 = 1.0
        hyperparameters['TPOT_P99'] = tpot_p99

    if ttft_p99 <= 0:
        logger.error(f"Invalid TTFT_P99 value: {ttft_p99}. Cannot compute log_normalized rewards.")
        raise ValueError(f"TTFT_P99 must be positive for log_normalized reward function. Got TTFT_P99={ttft_p99}")

    result = calculate_rewards_log_normalized(
        ttft, tpot,
        ttft_p99=ttft_p99,
        tpot_p99=tpot_p99,
        ttft_reward_weight=ttft_weight,
    )

    if result is None:
        raise ValueError("Reward calculation returned None")
    if 'ttft_rewards' not in result or 'tpot_rewards' not in result or 'combined_rewards' not in result:
        raise ValueError(f"Reward calculation returned invalid structure: {list(result.keys())}")

    ttft_r = result['ttft_rewards']
    tpot_r = result['tpot_rewards']
    combined = result['combined_rewards']
    if np.isnan(ttft_r).any() or np.isnan(tpot_r).any() or np.isnan(combined).any():
        n_t = int(np.isnan(ttft_r).sum())
        n_p = int(np.isnan(tpot_r).sum())
        n_c = int(np.isnan(combined).sum())
        logger.error(f"Reward calculation produced NaN values: ttft_rewards={n_t}/{len(ttft_r)}, tpot_rewards={n_p}/{len(tpot_r)}, combined_rewards={n_c}/{len(combined)}")
        logger.error(f"TTFT stats: min={np.nanmin(ttft):.2f}, max={np.nanmax(ttft):.2f}, p99={ttft_p99:.2f}")
        logger.error(f"TPOT stats: min={np.nanmin(tpot):.2f}, max={np.nanmax(tpot):.2f}, p99={tpot_p99:.2f}")
        raise ValueError("Reward calculation produced NaN values. Check input data and P99 values.")

    logger.info(f"Reward calculation successful: ttft_rewards range=[{np.min(ttft_r):.4f}, {np.max(ttft_r):.4f}], tpot_rewards range=[{np.min(tpot_r):.4f}, {np.max(tpot_r):.4f}], combined_rewards range=[{np.min(combined):.4f}, {np.max(combined):.4f}]")
    return result
