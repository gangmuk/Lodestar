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

# Reward-labeling functions used by the training-path elif chain inside
# preprocess_data_unified. These previously lived in this module; they were
# extracted to rewards.py during the pre-release cleanup. The training path
# still calls them by their bare names (unqualified), so we re-export the
# names into preprocess.py's namespace via this `from ... import` block.
#
# NOTE: data_normalizer.py uses rewards.compute_rewards (the dispatcher);
# this file uses the individual functions directly because its elif chain
# has cases that don't fit the dispatcher's shape (e.g. LATENCY_METRIC=
# 'e2e_latency' invokes calculate_rewards_e2e, negative_linear_and_prefix_locality
# needs base_data, context_aware uses real per-row kv_hit_ratio).
# Consolidating both call sites is tracked as a follow-up.
from rewards import (
    calculate_rewards_e2e,
    calculate_rewards_simple,
    calculate_rewards_simple_extended,
    calculate_rewards_piecewise_linear_steeper_gradient,
    calculate_rewards_latency_optimization,
    calculate_rewards_inverse_latency,
    calculate_rewards_simple_latency_minimization,
    calculate_rewards_negative_reciprocal,
    calculate_rewards_negative_linear,
    calculate_rewards_negative_linear_and_prefix_locality,
    calculate_rewards_negative_squared,
    calculate_rewards_quantile_based,
    calculate_rewards_absolute_latency,
    calculate_rewards_throughput_based,
    calculate_rewards_log_normalized,
    calculate_rewards_quantile_advantage,
    calculate_rewards_context_aware,
)

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
    else:  # Inference path — fast dict-based pipeline, no pandas
        parse_start_time = time.time()
        row_dict, _ = _parse_log_to_dict(log_message)
        preprocess_dataset_overhead_summary["parse_log_message"] = time.time() - parse_start_time
        if not row_dict:
            logger.error("No data found after parsing log message.")
            logger.error(f"Log message: {log_message}")
            assert False
        sorted_all_pod_ids = utils.get_sorted_all_pod_ids('single_row', row_dict)
        preprocess_start_time = time.time()
        processed_df, sorted_all_pod_ids, preprocess_dataset_overhead_summary = preprocess_inference_fast(
            row_dict, hyperparameters, sorted_all_pod_ids)
        preprocess_dataset_overhead_summary["preprocess_unified_inference"] = time.time() - preprocess_start_time
        return processed_df, sorted_all_pod_ids, preprocess_dataset_overhead_summary

    if len(parsed_df) == 0:
        logger.error("No data found after parsing JSON columns.")
        logger.error(f"Log message: {log_message}")
        assert False

    # Training mode: batch processing with full features
    sorted_all_pod_ids = utils.get_sorted_all_pod_ids('batch_dataframe', parsed_df)
    preprocess_start_time = time.time()
    is_training = True
    processed_df, sorted_all_pod_ids, preprocess_dataset_overhead_summary = preprocess_data_unified(parsed_df, hyperparameters, sorted_all_pod_ids, is_training)
    preprocess_dataset_overhead_summary["preprocess_unified_training"] = time.time() - preprocess_start_time
    return processed_df, sorted_all_pod_ids, preprocess_dataset_overhead_summary


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

def preprocess_data_unified(parsed_df, hyperparameters, sorted_all_pod_ids, is_training):
    num_rows = len(parsed_df)
    processing_type = "batch" if num_rows > 1 else "single row"
    logger.debug(f"Processing {num_rows} rows ({processing_type}) with is_training={is_training}")
    
    # Pre-parse all JSON columns once to avoid repeated parsing
    json_columns = [
        'allPodsKvCacheHitRatios',
        'allPodsKvCacheLastAccess',  # Per-pod prefix last-access timestamps (unix millis)
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
    all_kv_cache_last_access = safe_get_column(parsed_df, 'allPodsKvCacheLastAccess', {})
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

    # Time-weighted KV hit ratio: kv_hit_ratio * exp(-ln2 * age / half_life)
    # Reflects actual vLLM cache freshness — recent blocks are likely still cached,
    # old blocks are likely evicted. Gives meaningful per-pod variance even under uniform routing.
    if 'kv_hit_ratio_fresh' not in excluded_pod_features and 'kv_hit_ratio' not in excluded_pod_features:
        kv_freshness_half_life = float(hyperparameters.get('KV_FRESHNESS_HALF_LIFE', 15.0)) if hyperparameters else 15.0
        last_access_features = extract_pod_features_fast(all_kv_cache_last_access, sorted_pods, 0)
        request_times = np.array(base_data.get('request_start_time', np.zeros(num_rows)), dtype=np.float64)
        for pod_id in sorted_pods:
            raw_hit = base_data[f"{pod_id}-kv_hit_ratio"]
            # Both request_start_time and last_access are in microseconds since
            # first request (gateway normalizes last_access to the same epoch).
            # age = (request_time - last_access) converted to seconds.
            last_access_us = np.array(last_access_features[pod_id], dtype=np.float64)
            if num_rows == 1 and not is_training:
                # Infer path: request_times may not be set yet; use 0 as "now"
                # relative to FirstRequestStartTime (the gateway just computed
                # last_access moments ago, so age ≈ 0 is a safe fallback).
                import time as _time
                # Approximate: current request's relative timestamp
                current_us = request_times[0] if len(request_times) > 0 else 0
                age_seconds = np.where(last_access_us > 0,
                                       np.maximum(0, (current_us - last_access_us) / 1e6),
                                       999.0)
            else:
                # Training/batch path: both are in microseconds since first request
                age_seconds = np.where(last_access_us > 0,
                                       np.maximum(0, (request_times - last_access_us) / 1e6),
                                       999.0)
            weight = np.exp(-0.693147 * np.clip(age_seconds, 0, 999) / max(kv_freshness_half_life, 0.1))
            base_data[f"{pod_id}-kv_hit_ratio_fresh"] = raw_hit * weight

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
            latency_metric = hyperparameters.get('LATENCY_METRIC', 'ttft')
            logger.info(
                "Online training reward config: REWARD_FUNCTION=%s, LATENCY_METRIC=%s, TTFT_SLO=%s, AVG_TPOT_SLO=%s, TTFT_REWARD_WEIGHT=%s",
                reward_function,
                latency_metric,
                ttft_slo,
                avg_tpot_slo,
                ttft_reward_weight,
            )

            if latency_metric == 'e2e_latency':
                # Use end-to-end latency directly as the reward signal
                e2e_values = np.array(base_data['e2e_latency'], dtype=np.float64)
                e2e_kwargs = {
                    'ttft_slo': ttft_slo,
                    'avg_tpot_slo': avg_tpot_slo,
                    'input_tokens': np.array(base_data['input_tokens'], dtype=np.float64),
                    'output_tokens': np.array(base_data['output_tokens'], dtype=np.float64),
                    'num_buckets': hyperparameters.get('REWARD_NUM_BUCKETS', 20),
                }
                if 'E2E_P99' in hyperparameters:
                    e2e_kwargs['e2e_p99'] = hyperparameters['E2E_P99']
                else:
                    e2e_p99 = float(np.percentile(e2e_values, 99))
                    e2e_kwargs['e2e_p99'] = e2e_p99
                    hyperparameters['E2E_P99'] = e2e_p99
                    logger.info(f"Computed E2E_P99={e2e_p99:.2f}ms from data")
                if 'E2E_SLO' in hyperparameters:
                    e2e_kwargs['e2e_slo'] = hyperparameters['E2E_SLO']
                logger.info(f"Using e2e_latency as reward signal with {reward_function} (e2e range: {e2e_values.min():.1f}-{e2e_values.max():.1f}ms)")
                reward = calculate_rewards_e2e(e2e_values, reward_function, **e2e_kwargs)
            elif reward_function == "linear_simple":
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
            elif reward_function == "negative_linear_and_prefix_locality":
                selected_pods = np.array(base_data['selected_pod'])
                reward = calculate_rewards_negative_linear_and_prefix_locality(
                    ttft_values, tpot_values, ttft_reward_weight,
                    selected_pods, base_data, hyperparameters
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
            elif reward_function == "quantile_advantage":
                # Uses TTFT as the latency metric (for E2E, use LATENCY_METRIC=e2e_latency)
                input_tokens = np.array(base_data['input_tokens'], dtype=np.float64)
                num_buckets = hyperparameters.get('REWARD_NUM_BUCKETS', 20)
                reward = calculate_rewards_quantile_advantage(
                    ttft_values, input_tokens, num_buckets
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


def _parse_log_to_dict(log_message):
    """
    Core log-message parser. Returns (row_dict, json_columns).
    No pandas involved — just a plain Python dict.
    """
    if "latency_metrics" not in log_message:
        logger.error(f"Invalid line. {log_message}")
        return {}, []

    start_idx = log_message.find(_LATENCY_METRICS_PREFIX)
    if start_idx == -1:
        return {}, []
    start_idx += _LATENCY_METRICS_PREFIX_LEN

    parts = log_message[start_idx:].split('@')
    row = {}
    json_columns = []

    num_parts = len(parts)
    for i in range(0, num_parts - 1, 2):
        key = parts[i]
        value = parts[i + 1]

        if value and value[0] == '{' and value[-1] == '}':
            try:
                if '\\"' in value:
                    value = value.replace('\\"', '"')
                row[key] = json.loads(value)
                json_columns.append(key)
            except Exception as e:
                logger.error(f"Error decoding JSON, column: {key}, value: {value}")
                logger.error(f"Error: {e}")
                row[key] = value
        else:
            row[key] = _fast_parse_value(value)

    return row, json_columns


def parse_log_message(log_message):
    """
    Parse a single log message into a DataFrame.
    Used by training path and any callers expecting DataFrame output.
    """
    row, json_columns = _parse_log_to_dict(log_message)
    if row:
        return pd.DataFrame([row]), json_columns
    else:
        return pd.DataFrame(), []


def preprocess_inference_fast(row_dict, hyperparameters, sorted_all_pod_ids):
    """
    Fast single-row inference preprocessing — no pandas, no numpy array wrapping.
    Equivalent to preprocess_data_unified(DataFrame([row]), ..., is_training=False)
    but ~4x faster by avoiding DataFrame construction and pandas operations.
    """
    import math
    overhead = {}

    # --- JSON column parsing (should already be dicts from _parse_log_to_dict) ---
    json_parse_start = time.time()
    json_columns = [
        'allPodsKvCacheHitRatios', 'allPodsKvCacheLastAccess',
        'numInflightRequestsAllPods', 'numInflightPrefillRequestsAllPods',
        'numInflightDecodeRequestsAllPods', 'vllmGPUKVCacheUsage',
        'vllmCPUKVCacheUsage', 'vllmNumRequestsRunning',
        'vllmNumRequestsWaiting', 'numPrefillTokensForAllPods',
        'numDecodeTokensForAllPods', 'GPU',
    ]
    for col in json_columns:
        val = row_dict.get(col)
        if isinstance(val, str):
            row_dict[col] = safe_parse_json(val)
        elif val is None:
            row_dict[col] = {}
    overhead['json_parse_overhead'] = time.time() - json_parse_start

    # Hardcode fix for missing columns (same as preprocess_data_unified)
    if 'subAlgorithm' not in row_dict:
        row_dict['subAlgorithm'] = None
    if 'GPU' not in row_dict:
        row_dict['GPU'] = {}

    # --- Numeric conversion: direct float() instead of pd.to_numeric ---
    numeric_start = time.time()
    numeric_columns = [
        'ttft', 'avg_tpot', 'total_decode_time', 'e2e',
        'numInputTokens', 'numOutputTokens', 'numTotalTokens',
        'request_start_time', 'request_end_time', 'prev_reward',
    ]
    for col in numeric_columns:
        if col in row_dict:
            v = row_dict[col]
            if not isinstance(v, (int, float)):
                try:
                    row_dict[col] = float(v)
                except (ValueError, TypeError):
                    row_dict[col] = float('nan')
    overhead['numeric_conversion_overhead'] = time.time() - numeric_start

    # --- Extract base features: direct dict access ---
    get_value_start = time.time()
    result = {
        'request_id': row_dict.get('requestID'),
        'selected_pod': row_dict.get('selectedpod'),
        'input_tokens': row_dict.get('numInputTokens', 0),
        'output_tokens': row_dict.get('numOutputTokens', 0),
        'total_tokens': row_dict.get('numTotalTokens', 0),
        'ttft': row_dict.get('ttft', 0),
        'avg_tpot': row_dict.get('avg_tpot', 0),
        'e2e_latency': row_dict.get('e2e', 0),
        'request_start_time': row_dict.get('request_start_time', 0),
        'request_end_time': row_dict.get('request_end_time', 0),
        'subAlgorithm': row_dict.get('subAlgorithm'),
    }

    # Feature exclusion
    excluded = set()
    if hyperparameters and 'EXCLUDED_POD_FEATURES' in hyperparameters:
        excluded = set(hyperparameters['EXCLUDED_POD_FEATURES'])
        if 'none' in excluded or 'None' in excluded:
            excluded = set()

    sorted_pods = sorted_all_pod_ids

    # --- Pod feature extraction: direct dict lookups, plain floats ---
    feature_configs = [
        ('kv_hit_ratio', 'allPodsKvCacheHitRatios', 0),
        ('inflight_requests', 'numInflightRequestsAllPods', 0),
        ('inflight_prefill_requests', 'numInflightPrefillRequestsAllPods', 0),
        ('inflight_decode_requests', 'numInflightDecodeRequestsAllPods', 0),
        ('gpu_kv_cache', 'vllmGPUKVCacheUsage', 0),
        ('cpu_kv_cache', 'vllmCPUKVCacheUsage', 0),
        ('running_requests', 'vllmNumRequestsRunning', 0),
        ('waiting_requests', 'vllmNumRequestsWaiting', 0),
        ('prefill_tokens', 'numPrefillTokensForAllPods', 0),
        ('decode_tokens', 'numDecodeTokensForAllPods', 0),
    ]

    for feature_name, col_name, default_val in feature_configs:
        if feature_name not in excluded:
            pod_map = row_dict.get(col_name, {})
            if not isinstance(pod_map, dict):
                pod_map = {}
            for pod_id in sorted_pods:
                val = pod_map.get(pod_id, default_val)
                try:
                    result[f"{pod_id}-{feature_name}"] = float(val)
                except (ValueError, TypeError):
                    result[f"{pod_id}-{feature_name}"] = float(default_val)

    # --- kv_hit_ratio_fresh: time-weighted freshness ---
    if 'kv_hit_ratio_fresh' not in excluded and 'kv_hit_ratio' not in excluded:
        half_life = float(hyperparameters.get('KV_FRESHNESS_HALF_LIFE', 15.0)) if hyperparameters else 15.0
        last_access_map = row_dict.get('allPodsKvCacheLastAccess', {})
        if not isinstance(last_access_map, dict):
            last_access_map = {}
        current_us = float(result.get('request_start_time', 0))
        for pod_id in sorted_pods:
            raw_hit = result.get(f"{pod_id}-kv_hit_ratio", 0.0)
            la = float(last_access_map.get(pod_id, 0))
            age_s = max(0, (current_us - la) / 1e6) if la > 0 else 999.0
            age_s = min(age_s, 999.0)
            weight = math.exp(-0.693147 * age_s / max(half_life, 0.1))
            result[f"{pod_id}-kv_hit_ratio_fresh"] = raw_hit * weight

    # --- GPU model (string, not numeric) ---
    if 'GPU' not in excluded:
        gpu_map = row_dict.get('GPU', {})
        if not isinstance(gpu_map, dict):
            gpu_map = {}
        for pod_id in sorted_pods:
            result[f"{pod_id}-GPU"] = gpu_map.get(pod_id, 'GPU-L3c')

    overhead['get_value_overhead'] = time.time() - get_value_start
    overhead['create_df_overhead'] = 0
    overhead['pod_index_overhead'] = 0
    overhead['reward_calc_overhead'] = -1
    overhead['slo_update_overhead'] = -1
    overhead['column_check_overhead'] = -1
    overhead['podmetrics_parse_overhead'] = -1

    # NaN fill
    for key, value in result.items():
        if isinstance(value, float) and math.isnan(value):
            result[key] = 0

    return result, sorted_all_pod_ids, overhead
