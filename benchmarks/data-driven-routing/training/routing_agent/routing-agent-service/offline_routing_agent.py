# offline_routing_agent.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import os
import logging
import time
import sys
import encoding
import simpler_contextual_bandit
from logger import logger
import preprocess
import pickle
import threading
import argparse
import random_forest
import torch
import feature_normalization


# Global variables (simplified for offline use)
ENCODED_DATA_DIR = "encoded_data"
STATS_FILE = "request_feature_stats.pkl"
NUM_TRAINS = 0
MODEL_UPDATED = False
TRAINING_DATA_UPDATED = False

TOTAL_NUM_DATA = 0
MIN_NUM_TRAINING_DATA = 500
LOCK_TRAINING_DATA = threading.Lock()
request_stats = None
request_features_train = ['input_tokens', 'output_tokens', 'total_tokens']
request_features_reward = ['ttft', 'avg_tpot', 'e2e_latency']

def get_request_stats():
    """Get or initialize request feature statistics"""
    global request_stats
    if request_stats is None:
        request_stats = feature_normalization.load_stats(STATS_FILE)
    return request_stats

def write_to_file(log_data, raw_data):
    with open(raw_data, "w") as log_file:
        for request_id, log_message in log_data.items():
            log_file.write(f"{log_message}\n")
    logger.info(f"Successfully wrote {len(log_data)} entries to {raw_data}")

def read_csv_data(csv_file):
    """Read log messages from CSV file"""
    logger.info(f"Reading data from {csv_file}")
    
    # Try to read as CSV first
    try:
        df = pd.read_csv(csv_file)
        if 'log_message' in df.columns:
            log_messages = df['log_message'].tolist()
        elif len(df.columns) == 1:
            # Single column, assume it's log messages
            log_messages = df.iloc[:, 0].tolist()
        else:
            logger.error(f"CSV file must have a 'log_message' column or be a single column file")
            return None
    except:
        # If CSV reading fails, try reading as plain text file
        try:
            with open(csv_file, 'r') as f:
                log_messages = [line.strip() for line in f if line.strip()]
        except Exception as e:
            logger.error(f"Error reading file {csv_file}: {e}")
            return None
    
    # Clean log messages - remove Go log prefixes
    cleaned_messages = []
    for i, log_message in enumerate(log_messages):
        if log_message and log_message.strip():
            # Remove Go log prefix: "I0527 05:00:00.334792       1 gateway_rsp_body.go:647] "
            # Look for the pattern and extract everything after the "] "
            clean_message = log_message.strip()
            
            # Debug: show original message for first few entries
            if i < 3:
                logger.info(f"Original message {i}: {clean_message[:150]}...")
            
            # Find the last occurrence of "] " which should mark the end of the log prefix
            bracket_pos = clean_message.rfind('] ')
            if bracket_pos != -1:
                # Extract everything after "] "
                clean_message = clean_message[bracket_pos + 2:]
            
            # Additional check: if message starts with "**@latency_metrics@", keep it as is
            # Otherwise, try to find that pattern in the message
            if not clean_message.startswith('**@latency_metrics@'):
                metrics_pos = clean_message.find('**@latency_metrics@')
                if metrics_pos != -1:
                    clean_message = clean_message[metrics_pos:]
            
            if clean_message.startswith('**@latency_metrics@'):
                cleaned_messages.append(clean_message)
                # Debug: show cleaned message for first few entries
                if i < 3:
                    logger.info(f"Cleaned message {i}: {clean_message[:150]}...")
            else:
                logger.warning(f"Skipping malformed log message {i}: {log_message[:100]}...")
    
    # Convert to dictionary format expected by the training pipeline
    log_data = {}
    for i, log_message in enumerate(cleaned_messages):
        log_data[f"request_{i}"] = log_message
    
    logger.info(f"Successfully read {len(log_data)} log messages from {csv_file} (cleaned from {len(log_messages)} raw entries)")
    
    # Show a sample of the final cleaned data
    if log_data:
        first_key = list(log_data.keys())[0]
        sample_message = log_data[first_key]
        logger.info(f"Sample cleaned message: {sample_message[:200]}...")
    
    return log_data

def train_model(args):
    """Train the model"""
    global NUM_TRAINS, MODEL_UPDATED, TRAINING_DATA_UPDATED, TOTAL_NUM_DATA
    
    if TRAINING_DATA_UPDATED and TOTAL_NUM_DATA > MIN_NUM_TRAINING_DATA:
        training_start_time = time.time()
        logger.info(f"Starting {NUM_TRAINS}th training of routing agent")
        
        try:
            if args.model == "random_forest":
                random_forest.train(ENCODED_DATA_DIR)
            elif args.model == "simpler_contextual_bandit":
                simpler_contextual_bandit.train(ENCODED_DATA_DIR, f"{args.data_dir}/final_model")
            else:
                logger.error(f"Unknown model type: {args.model}")
                return False
            MODEL_UPDATED = True
            TRAINING_DATA_UPDATED = False
            NUM_TRAINS += 1
            
            logger.info(f"Successfully completed {NUM_TRAINS-1}th training of routing agent, took {time.time() - training_start_time} seconds")
            return True
            
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            logger.error(f"Error training model: {str(e)}")
            logger.error(f"Traceback: {error_traceback}")
            return False
    else:
        logger.info(f"Not enough training data available (TOTAL_NUM_DATA: {TOTAL_NUM_DATA}), skipping training")
        return False


def test_inference(args, log_message):
    """Test inference on a single log message with original vs predicted comparison"""
    global NUM_TRAINS, MODEL_UPDATED
    
    if NUM_TRAINS == 0:
        logger.warning("No trained model available, please train first")
        return None
        
    handle_infer_start_time = time.time()
    
    try:
        logger.debug(f"Testing inference on log message: {log_message[:200]}...")
        
        # Preprocess the log message
        preprocess_start_time = time.time()
        
        # Debug: show what we're passing to preprocess
        logger.debug(f"Calling preprocess.main(None, log_message, {args.ttft_slo}, {args.avg_tpot_slo})")
        
        processed_df, _, all_pods, preprocess_dataset_overhead_summary = preprocess.main(None, log_message, args.ttft_slo, args.avg_tpot_slo)
        logger.debug(f"Successfully parsed data for inference")
        handle_infer_total_total_preprocess_overhead = time.time() - preprocess_start_time

        # EXTRACT ORIGINAL POD CHOICE FROM PREPROCESSED DATA
        original_pod_choice = None
        if 'selected_pod' in processed_df.columns:
            original_pod_choice = processed_df['selected_pod'].iloc[0] if len(processed_df) > 0 else None
        elif hasattr(processed_df, 'original_pod') or 'original_pod' in processed_df.columns:
            original_pod_choice = processed_df['original_pod'].iloc[0] if len(processed_df) > 0 else None
        
        # If not found in DataFrame, try to extract from the raw log message
        if original_pod_choice is None:
            import re
            pattern = r'@selectedpod@([^@]+)@'
            match = re.search(pattern, log_message)
            if match:
                original_pod_choice = match.group(1)
        
        logger.debug(f"Original pod choice: {original_pod_choice}")

        # ===== SHARED NORMALIZATION LOGIC =====
        stats = get_request_stats()
        processed_df = feature_normalization.normalize_features_for_inference(processed_df, stats)
        
        # Encode data
        encode_start_time = time.time()
        tensor_dataset, _ = encoding.encode_for_inference(all_pods, processed_df, stats, request_features_train, request_features_reward)
        logger.debug(f"Successfully encoded data in memory for inference")
        handle_infer_total_total_encoding_overhead = time.time() - encode_start_time

        # Perform inference
        infer_from_tensor_start_time = time.time()
        if args.model == "random_forest":
            result, _ = random_forest.infer_from_tensor(
                tensor_data=tensor_dataset, 
                exploration_enabled=True, 
                exploration_rate=0.2, 
                model_updated=MODEL_UPDATED
        )
        elif args.model == "simpler_contextual_bandit":
            result, _ = simpler_contextual_bandit.infer_from_tensor(
                tensor_data=tensor_dataset, 
                model_updated=MODEL_UPDATED
            )
        if MODEL_UPDATED:
            logger.info("Model updated flag consumed, resetting to False")
            MODEL_UPDATED = False
        handle_infer_total_total_infer_from_tensor_overhead = time.time() - infer_from_tensor_start_time

        # Map the pod index back to the actual pod ID
        selected_pod_index = result.get('selected_pod_index', 0)
        if selected_pod_index >= len(all_pods):
            logger.warning(f"Selected pod index {selected_pod_index} out of range, defaulting to first pod")
            selected_pod_index = 0
            
        selected_pod = all_pods[selected_pod_index]
        confidence = result['confidence']
        handle_infer_total_overhead = time.time() - handle_infer_start_time
        
        # Compare with original choice
        prediction_matches = (selected_pod == original_pod_choice) if original_pod_choice else None
        
        # Return enhanced result with comparison
        result_summary = {
            "selected_pod": selected_pod,
            "original_pod_choice": original_pod_choice,
            "prediction_matches": prediction_matches,
            "confidence": confidence,
            "total_inference_time_ms": handle_infer_total_overhead * 1000,
            "preprocess_time_ms": handle_infer_total_total_preprocess_overhead * 1000,
            "encoding_time_ms": handle_infer_total_total_encoding_overhead * 1000,
            "inference_time_ms": handle_infer_total_total_infer_from_tensor_overhead * 1000,
        }
        
        # Enhanced logging with match/mismatch status
        if original_pod_choice:
            match_status = "✅ MATCH" if prediction_matches else "❌ MISMATCH"
            logger.info(f"Inference result: predicted={selected_pod}, original={original_pod_choice}, {match_status}, confidence={confidence:.4f}")
        else:
            logger.info(f"Inference result: predicted={selected_pod}, original=UNKNOWN, confidence={confidence:.4f}")
        
        return result_summary
        
    except AssertionError as e:
        logger.error(f"AssertionError in preprocessing - this suggests the log message format is unexpected")
        logger.error(f"Log message being processed: {log_message}")
        logger.error(f"Error: {str(e)}")
        return None
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Error in test_inference: {str(e)}")
        logger.error(f"Traceback: {error_traceback}")
        logger.error(f"Log message being processed: {log_message}")
        return None

def process_training_data(args, log_data):
    """Process training data with shared normalization logic"""
    global ENCODED_DATA_DIR, NUM_TRAINS, MODEL_UPDATED, TRAINING_DATA_UPDATED, TOTAL_NUM_DATA
    
    flush_start_time = time.time()
    
    try:
        logger.info(f"Processing training data with {len(log_data)} entries")
        
        # Create raw data file
        if not os.path.exists("temp_training_data"):
            os.mkdir("temp_training_data")
        raw_data = "temp_training_data/offline_batch.csv"
        
        # Write raw data to file
        ts_write_raw_data = time.time()
        write_to_file(log_data, raw_data)
        logger.info(f"Wrote {len(log_data)} entries to {raw_data}, took {time.time() - ts_write_raw_data} seconds")

        # Preprocess raw data
        ts_preprocess = time.time()
        df, _, all_pods, _ = preprocess.main(raw_data, "", args.ttft_slo, args.avg_tpot_slo)
        logger.info(f"Successfully parsed data, took {time.time() - ts_preprocess} seconds")
        
        # ===== SHARED NORMALIZATION LOGIC =====
        stats = get_request_stats()
        df, stats, _ = feature_normalization.normalize_features_for_training(df, stats)
        feature_normalization.save_stats(stats, STATS_FILE)

        # ===== SHARED REWARD ENGINEERING =====
        df = feature_normalization.apply_reward_engineering(df)

        # Continue with encoding...
        ts_encode = time.time()
        encoded_data_subdir = f"{ENCODED_DATA_DIR}/batch_1"
        encoding.encode_for_train(all_pods, df, encoded_data_subdir, request_features_train, request_features_reward)
        logger.info(f"Successfully encoded data to {encoded_data_subdir}, took {time.time() - ts_encode} seconds")
        
        # Verify encoded data
        expected_tensor_path = f"{encoded_data_subdir}/tensor_dataset.pt"
        train_tensor_path = f"{encoded_data_subdir}/train/tensor_dataset.pt"
        
        if os.path.exists(expected_tensor_path):
            logger.info(f"✓ Found tensor dataset at: {expected_tensor_path}")
        elif os.path.exists(train_tensor_path):
            logger.info(f"✓ Found tensor dataset at: {train_tensor_path}")
        
        TRAINING_DATA_UPDATED = True
        TOTAL_NUM_DATA += len(log_data)
        
        logger.info(f"Successfully processed {len(log_data)} log messages, took {time.time() - flush_start_time} seconds")
        return True
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Error processing training data: {str(e)}")
        logger.error(f"Traceback: {error_traceback}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Offline Routing Agent Training and Testing')
    parser.add_argument('data_file', help='CSV file containing log messages for training')
    parser.add_argument('--skip_training', action='store_true', help='Skip training and only do inference')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='Train/test split ratio (default: 0.8 for 80%% train, 20%% test)')
    parser.add_argument('--model', choices=['random_forest', 'simpler_contextual_bandit'], default='simpler_contextual_bandit', help='Model type to use for training (default: simpler_contextual_bandit)')
    parser.add_argument('--ttft_slo', type=float, help='TTFT SLO threshold for preprocessing', default=1000)
    parser.add_argument('--avg_tpot_slo', type=float, help='Average TPOT SLO threshold for preprocessing', default=50)
    parser.add_argument('--analyze_behavior', action='store_true', help='Analyze what the model has learned through feature sensitivity tests')

    args = parser.parse_args()

    # Check if data file exists
    if not os.path.exists(args.data_file):
        logger.error(f"Data file {args.data_file} not found")
        return

    # Handle data splitting
    logger.info("=== SPLITTING DATA ===")
    all_data = {}
    # Check if data_file is file or directory
    if os.path.isfile(args.data_file):
        data_dir = os.path.dirname(args.data_file)
        logger.info(f"data_file specified: {args.data_file}")
        all_data = read_csv_data(args.data_file)

    elif os.path.isdir(args.data_file):
        data_dir = args.data_file
        logger.info(f"data_file is a directory: {args.data_file}")
        for root, dirs, files in os.walk(args.data_file):
            for file in files:
                if file == "data.csv":
                    file_path = os.path.join(root, file)
                    logger.info(f"Found data.csv at: {file_path}")
                    data = read_csv_data(file_path)
                    if data:
                        # Merge data dictionaries with unique keys
                        for key, value in data.items():
                            new_key = f"{os.path.basename(root)}_{key}"
                            all_data[new_key] = value

    args.data_dir = data_dir

    if all_data is None or len(all_data) == 0:
        logger.error("Failed to read data or no valid log messages found")
        return
    
    # Split data
    all_messages = list(all_data.values())
    split_point = int(len(all_messages) * args.split_ratio)
    
    train_messages = all_messages[:split_point]
    test_messages = all_messages[split_point:]
    
    # Convert back to dict format
    train_data = {f"request_{i}": msg for i, msg in enumerate(train_messages)}
    test_data = {f"request_{i}": msg for i, msg in enumerate(test_messages)}
    
    logger.info(f"Split {len(all_data)} messages into {len(train_data)} train + {len(test_data)} test")
        
    # Create necessary directories
    if not os.path.exists(ENCODED_DATA_DIR):
        os.makedirs(ENCODED_DATA_DIR)
    
    # Clean up any existing encoded data for fresh start
    if os.path.exists(ENCODED_DATA_DIR):
        import shutil
        shutil.rmtree(ENCODED_DATA_DIR)
        os.makedirs(ENCODED_DATA_DIR)
        logger.info(f"Cleaned and recreated {ENCODED_DATA_DIR} for fresh offline training")
    
    # Read and process training data
    if not args.skip_training:
        logger.info("=== STARTING TRAINING PHASE ===")
        
        # Process training data
        if not process_training_data(args, train_data):
            logger.error("Failed to process training data")
            return

        diagnose_training_data_issues(args, train_data)
        
        # Train model
        if not train_model(args):
            logger.error("Failed to train model")
            return
        
        logger.info("=== TRAINING COMPLETED ===")
    else:
        logger.info("=== SKIPPING TRAINING (using existing model) ===")

    # NEW: Behavior Analysis (before regular testing)
    if args.analyze_behavior and test_data and len(test_data) > 0:
        logger.info("=== STARTING BEHAVIOR ANALYSIS ===")
        # analyze_model_behavior(args, test_data)
        _ =analyze_detailed_feature_sensitivity(args, test_data)
        logger.info("=== BEHAVIOR ANALYSIS COMPLETED ===")
    
    
    # Test inference
    if test_data and len(test_data) > 0:
        logger.info("=== STARTING TESTING PHASE ===")
        
        # Test on multiple examples with match tracking
        success_count = 0
        match_count = 0
        mismatch_count = 0
        unknown_original_count = 0
        total_count = min(len(test_data), 10)  # Limit to 10 tests for faster iteration
        
        logger.info(f"Testing on {total_count} samples (out of {len(test_data)} available)")
        
        test_items = list(test_data.items())[:total_count]
        for i, (request_id, log_message) in enumerate(test_items):
            logger.info(f"Testing inference {i+1}/{total_count} (request_id: {request_id})")
            result = test_inference(args, log_message)
            if result:
                success_count += 1
                
                # Track prediction accuracy
                if result['prediction_matches'] is True:
                    match_count += 1
                elif result['prediction_matches'] is False:
                    mismatch_count += 1
                else:
                    unknown_original_count += 1
                
                # Show detailed result with comparison
                if result['original_pod_choice']:
                    match_status = "MATCH" if result['prediction_matches'] else "MISMATCH"
                    logger.info(f"  → Predicted: {result['selected_pod']}, Original: {result['original_pod_choice']}, Status: {match_status}, Confidence: {result['confidence']:.3f}")
                else:
                    logger.info(f"  → Predicted: {result['selected_pod']}, Original: UNKNOWN, Confidence: {result['confidence']:.3f}")
            else:
                logger.error(f"✗ Failed inference for {request_id}")
        
        # Print enhanced summary with accuracy metrics
        logger.info("=" * 60)
        logger.info("=== TESTING SUMMARY ===")
        logger.info(f"Total tests: {total_count}")
        logger.info(f"Successful inferences: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
        
        if match_count + mismatch_count > 0:
            accuracy = match_count / (match_count + mismatch_count) * 100
            logger.info(f"Prediction accuracy: {match_count}/{match_count + mismatch_count} ({accuracy:.1f}%)")
            logger.info(f"  - Matches: {match_count}")
            logger.info(f"  - Mismatches: {mismatch_count}")
        
        if unknown_original_count > 0:
            logger.info(f"  - Unknown original: {unknown_original_count}")
        
        logger.info("=" * 60)


###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
##################################### anaylsis helper functions ###################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################

def diagnose_training_data_issues(args, train_data_sample):
    """
    Diagnose why the model is learning static preferences instead of contextual routing.
    Call this right after process_training_data() in main().
    """
    logger.info("🔬 DIAGNOSING TRAINING DATA ISSUES")
    logger.info("=" * 60)
    
    # Check encoded data
    encoded_data_subdir = f"{ENCODED_DATA_DIR}/batch_1"
    tensor_path = f"{encoded_data_subdir}/tensor_dataset.pt"
    train_tensor_path = f"{encoded_data_subdir}/train/tensor_dataset.pt"
    
    if os.path.exists(tensor_path):
        tensor_data = torch.load(tensor_path)
    elif os.path.exists(train_tensor_path):
        tensor_data = torch.load(train_tensor_path)
    else:
        logger.error("No tensor data found for diagnosis")
        return
    
    # 1. ACTION DISTRIBUTION ANALYSIS
    logger.info("\n1️⃣ ACTION DISTRIBUTION ANALYSIS:")
    logger.info("-" * 40)
    
    actions = tensor_data['actions']
    action_counts = torch.bincount(actions, minlength=7)
    total_samples = len(actions)
    
    logger.info(f"Total training samples: {total_samples}")
    for pod_id in range(7):
        count = action_counts[pod_id].item()
        percentage = count / total_samples * 100
        logger.info(f"Pod {pod_id}: {count} samples ({percentage:.1f}%)")
    
    # Check for severe imbalance
    max_count = action_counts.max().item()
    min_count = action_counts[action_counts > 0].min().item() if (action_counts > 0).sum() > 0 else 1
    imbalance_ratio = max_count / min_count
    
    logger.info(f"Imbalance ratio: {imbalance_ratio:.1f}x")
    if imbalance_ratio > 5:
        logger.warning(f"⚠️  SEVERE IMBALANCE: Pod {torch.argmax(action_counts).item()} dominates training data!")
    
    # 2. REWARD SIGNAL ANALYSIS
    logger.info("\n2️⃣ REWARD SIGNAL ANALYSIS:")
    logger.info("-" * 40)
    
    rewards = tensor_data['rewards']
    logger.info(f"Reward range: [{rewards.min().item():.4f}, {rewards.max().item():.4f}]")
    logger.info(f"Reward std: {rewards.std().item():.4f}")
    
    # Reward by action
    reward_by_pod = {}
    for pod_id in range(7):
        pod_mask = actions == pod_id
        if pod_mask.sum() > 0:
            pod_rewards = rewards[pod_mask]
            reward_by_pod[pod_id] = {
                'mean': pod_rewards.mean().item(),
                'std': pod_rewards.std().item(),
                'count': pod_mask.sum().item()
            }
            logger.info(f"Pod {pod_id}: μ={reward_by_pod[pod_id]['mean']:.4f}, "
                       f"σ={reward_by_pod[pod_id]['std']:.4f}, n={reward_by_pod[pod_id]['count']}")
    
    # Check reward differentiation
    if len(reward_by_pod) > 1:
        pod_means = [stats['mean'] for stats in reward_by_pod.values()]
        reward_gap = max(pod_means) - min(pod_means)
        logger.info(f"Reward gap between best/worst pods: {reward_gap:.4f}")
        
        if reward_gap < 0.01:
            logger.warning("⚠️  VERY WEAK REWARD SIGNAL: Pods have nearly identical rewards!")
        elif reward_gap < 0.05:
            logger.warning("⚠️  WEAK REWARD SIGNAL: Small differences between pods")
    
    # 3. FEATURE VARIANCE ANALYSIS
    logger.info("\n3️⃣ FEATURE VARIANCE ANALYSIS:")
    logger.info("-" * 40)
    
    # Pod features variance
    pod_features = tensor_data['pod_features_with_staleness']
    logger.info(f"Pod features shape: {pod_features.shape}")
    
    # Calculate variance across samples for each feature
    pod_feature_vars = pod_features.var(dim=0).mean(dim=0)  # Average variance across pods
    logger.info("Pod feature variances:")
    feature_names = ['inflight_requests', 'gpu_kv_cache', 'cpu_kv_cache', 'running_requests', 
                    'waiting_requests', 'prefill_tokens', 'decode_tokens']
    
    low_variance_features = 0
    for i, var in enumerate(pod_feature_vars):
        feature_name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
        logger.info(f"  {feature_name}: {var.item():.6f}")
        if var.item() < 1e-3:
            low_variance_features += 1
            logger.warning(f"    ⚠️  Very low variance - feature may be static!")
    
    if low_variance_features > len(pod_feature_vars) * 0.5:
        logger.warning(f"⚠️  {low_variance_features}/{len(pod_feature_vars)} pod features have very low variance!")
    
    # KV hit ratios variance
    kv_ratios = tensor_data['kv_hit_ratios']
    kv_var = kv_ratios.var(dim=0).mean()
    logger.info(f"KV hit ratios variance: {kv_var.item():.6f}")
    if kv_var.item() < 1e-3:
        logger.warning("⚠️  KV hit ratios have very low variance!")
    
    # Request features variance
    request_features = tensor_data['request_features']
    request_vars = request_features.var(dim=0)
    logger.info("Request feature variances:")
    request_names = ['input_tokens', 'output_tokens', 'total_tokens']
    
    for i, var in enumerate(request_vars):
        feature_name = request_names[i] if i < len(request_names) else f"request_feature_{i}"
        logger.info(f"  {feature_name}: {var.item():.6f}")
        if var.item() < 1e-3:
            logger.warning(f"    ⚠️  Very low variance - feature may be static!")
    
    # 4. SAMPLE DATA INSPECTION
    logger.info("\n4️⃣ SAMPLE DATA INSPECTION:")
    logger.info("-" * 40)
    
    # Show first few samples to understand data characteristics
    logger.info("First 3 training samples:")
    for i in range(min(3, len(actions))):
        logger.info(f"\nSample {i}:")
        logger.info(f"  Action (selected pod): {actions[i].item()}")
        logger.info(f"  Reward: {rewards[i].item():.4f}")
        logger.info(f"  Request features: {request_features[i].numpy()}")
        logger.info(f"  KV ratios: {kv_ratios[i].numpy().flatten()}")
        logger.info(f"  Pod features (first 3): {pod_features[i, :, :3].numpy()}")
    
    # 5. RECOMMENDATIONS
    logger.info("\n5️⃣ RECOMMENDATIONS:")
    logger.info("-" * 40)
    
    recommendations = []
    
    if imbalance_ratio > 5:
        recommendations.append("🔴 CRITICAL: Balance training data - consider data augmentation or stratified sampling")
    
    if reward_gap < 0.01:
        recommendations.append("🔴 CRITICAL: Amplify reward differences or use different reward calculation")
    
    if low_variance_features > 3:
        recommendations.append("🟡 Add more dynamic pod state features (current load, temperature, etc.)")
    
    if kv_var.item() < 1e-3:
        recommendations.append("🟡 KV hit ratios may be too static - ensure they vary meaningfully")
    
    # Check if request features vary
    request_var_count = (request_vars < 1e-3).sum().item()
    if request_var_count > 0:
        recommendations.append("🟡 Some request features are static - add more request diversity")
    
    if not recommendations:
        recommendations.append("✅ No obvious data issues detected")
    
    logger.info("Action items:")
    for rec in recommendations:
        logger.info(f"  {rec}")
    
    logger.info("\n" + "=" * 60)
    
    return {
        'action_distribution': action_counts,
        'imbalance_ratio': imbalance_ratio,
        'reward_gap': reward_gap if 'reward_gap' in locals() else 0,
        'low_variance_features': low_variance_features,
        'recommendations': recommendations
    }

def analyze_detailed_feature_sensitivity(args, test_data_subset=None):
    """
    Improved feature-specific sensitivity analysis that uses statistically meaningful perturbations
    """
    global NUM_TRAINS
    
    if NUM_TRAINS == 0:
        logger.warning("No trained model available for detailed feature analysis")
        return None
    
    print("🔬 IMPROVED FEATURE-SPECIFIC SENSITIVITY ANALYSIS")
    print("=" * 70)
    
    if test_data_subset is None:
        return None
    
    # Define feature types and their expected characteristics
    feature_types = {
        'kv_hit_ratio': {
            'name': 'KV Cache Hit Ratio',
            'range': (0.0, 1.0),  # Natural range
            'perturbation_type': 'relative',  # Use relative changes
            'direction': 'both'  # Test both increases and decreases
        },
        'running_requests': {
            'name': 'Running Requests',
            'range': (0, 100),  # Typical range for request counts
            'perturbation_type': 'absolute',
            'direction': 'both'
        },
        'waiting_requests': {
            'name': 'Waiting Requests', 
            'range': (0, 100),
            'perturbation_type': 'absolute',
            'direction': 'both'
        },
        'decode_tokens': {
            'name': 'Decode Tokens',
            'range': (0, 10000),  # Typical token ranges
            'perturbation_type': 'relative',
            'direction': 'both'
        },
        'prefill_tokens': {
            'name': 'Prefill Tokens',
            'range': (0, 10000),
            'perturbation_type': 'relative', 
            'direction': 'both'
        },
        'inflight_requests': {
            'name': 'Inflight Requests',
            'range': (0, 100),
            'perturbation_type': 'absolute',
            'direction': 'both'
        },
        'last_second_avg_ttft_ms': {
            'name': 'Average TTFT',
            'range': (0, 5000),  # Milliseconds
            'perturbation_type': 'relative',
            'direction': 'increase_only'  # Higher latency = worse
        },
        'last_second_avg_tpot_ms': {
            'name': 'Average TPOT',
            'range': (0, 1000),
            'perturbation_type': 'relative',
            'direction': 'increase_only'
        },
        'last_second_p99_ttft_ms': {
            'name': 'P99 TTFT',
            'range': (0, 10000),
            'perturbation_type': 'relative',
            'direction': 'increase_only'
        },
        'last_second_total_requests': {
            'name': 'Total Requests/sec',
            'range': (0, 1000),
            'perturbation_type': 'relative',
            'direction': 'both'
        }
    }
    
    feature_sensitivity_results = {}
    
    # Test first 3 samples for detailed analysis
    test_items = list(test_data_subset.items())[:3]
    all_pods = None
    baseline_confidence = 0
    conf_mean = conf_std = conf_min = conf_max = 0
    prediction_diversity = 0
    for sample_idx, (request_id, log_message) in enumerate(test_items):
        print(f"\n--- ANALYZING SAMPLE {sample_idx + 1}/3 ({request_id}) ---")
        
        try:
            # Preprocess to get baseline data
            processed_df, _, all_pods, _ = preprocess.main(None, log_message, args.ttft_slo, args.avg_tpot_slo)
            
            # Apply same normalization as training
            request_features = ['input_tokens', 'output_tokens', 'total_tokens']
            pod_features_cols = [col for col in processed_df.columns if col.startswith('pod_') and processed_df[col].dtype in ['float64', 'int64']]
            
            stats = get_request_stats()
            if stats.count > 0:
                # Apply normalization and amplification as in training
                for feature in pod_features_cols:
                    if feature in processed_df.columns and feature in stats.feature_stats:
                        feature_data = processed_df[feature].values.reshape(-1, 1)
                        normalized_feature = stats.feature_stats[feature].normalize(feature_data)
                        processed_df[feature] = normalized_feature.flatten()
                
                # Apply same amplification as training
                critical_features = ['running_requests', 'waiting_requests', 'decode_tokens', 'prefill_tokens']
                for feature in pod_features_cols:
                    if any(critical in feature for critical in critical_features):
                        if feature in processed_df.columns:
                            processed_df[feature] = processed_df[feature] * feature_normalization.SIGNAL_AMPLIFICATION_DEGREE
            
            # Encode baseline data
            tensor_dataset, _ = encoding.encode_for_inference(all_pods, processed_df, stats, 
                                                            request_features_train, request_features_reward)
            
            # Get baseline prediction
            if args.model == "simpler_contextual_bandit":
                baseline_result, _ = simpler_contextual_bandit.infer_from_tensor(tensor_data=tensor_dataset, model_updated=False)
            else:
                baseline_result, _ = random_forest.infer_from_tensor(tensor_data=tensor_dataset, model_updated=False)
            
            baseline_pod = baseline_result['selected_pod_index']
            baseline_confidence = baseline_result['confidence']
            baseline_probs = baseline_result.get('pod_probabilities', [])
            
            print(f"Baseline prediction: Pod {baseline_pod} (confidence: {baseline_confidence:.3f})")
            print(f"Baseline probabilities: {[f'{p:.3f}' for p in baseline_probs[:7]]}")
            
            # Test each feature type individually with improved methodology
            for feature_key, feature_config in feature_types.items():
                print(f"\n🧪 TESTING {feature_config['name'].upper()} SENSITIVITY")
                print("-" * 50)
                
                # Find columns for this feature type across all pods
                feature_cols = [col for col in processed_df.columns if feature_key in col and col.startswith('pod_')]
                
                if not feature_cols:
                    print(f"  No {feature_config['name']} features found")
                    continue
                
                print(f"  Found {len(feature_cols)} {feature_config['name']} features across pods")
                
                # Calculate meaningful perturbations based on current data
                current_values = []
                for col in feature_cols:
                    if col in processed_df.columns:
                        current_values.extend(processed_df[col].values.tolist())
                
                if not current_values:
                    continue
                
                # Calculate data-driven perturbation magnitudes
                current_mean = np.mean(current_values)
                current_std = np.std(current_values)
                current_range = max(current_values) - min(current_values)
                
                print(f"  Current values: μ={current_mean:.3f}, σ={current_std:.3f}, range={current_range:.3f}")
                
                # Define perturbation levels based on feature characteristics
                if feature_config['perturbation_type'] == 'relative':
                    # Use percentage changes: ±10%, ±25%, ±50%, ±100%
                    if feature_config['direction'] == 'both':
                        perturbation_factors = [-0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 1.0]
                    else:  # increase_only for latency metrics
                        perturbation_factors = [0, 0.1, 0.25, 0.5, 1.0, 2.0]
                else:  # absolute changes
                    # Use multiples of standard deviation: ±0.5σ, ±1σ, ±2σ
                    if current_std > 0:
                        if feature_config['direction'] == 'both':
                            perturbation_deltas = [-2*current_std, -current_std, -0.5*current_std, 
                                                 0, 0.5*current_std, current_std, 2*current_std]
                        else:
                            perturbation_deltas = [0, 0.5*current_std, current_std, 2*current_std]
                    else:
                        # Fallback if no variance
                        perturbation_deltas = [0, 1, 5, 10]
                
                # Test systematic perturbations
                feature_changes = 0
                total_tests = 0
                sensitivity_details = []
                
                # Test perturbations on all pods, not just the preferred one
                for pod_idx in range(len(all_pods)):
                    pod_name = all_pods[pod_idx]
                    pod_feature_col = None
                    
                    # Find the feature column for this pod
                    for col in feature_cols:
                        if f"pod_{pod_name}" in col and feature_key in col:
                            pod_feature_col = col
                            break
                    
                    if pod_feature_col is None or pod_feature_col not in processed_df.columns:
                        continue
                    
                    original_value = processed_df[pod_feature_col].iloc[0]
                    
                    # Apply perturbations
                    if feature_config['perturbation_type'] == 'relative':
                        test_values = [original_value * (1 + factor) for factor in perturbation_factors]
                    else:
                        test_values = [original_value + delta for delta in perturbation_deltas]
                    
                    for test_idx, test_value in enumerate(test_values):
                        if test_value < 0 and feature_key not in ['running_requests', 'waiting_requests']:
                            continue  # Skip negative values for most features
                        
                        # Clamp values to reasonable ranges
                        if feature_key == 'kv_hit_ratio':
                            test_value = max(0.0, min(1.0, test_value))
                        elif 'requests' in feature_key:
                            test_value = max(0, test_value)
                        
                        # Create modified tensor
                        modified_tensor = {k: v.clone() if hasattr(v, 'clone') else v for k, v in tensor_dataset.items()}
                        
                        # Apply modification based on tensor structure
                        if feature_key == 'kv_hit_ratio':
                            # KV hit ratio is in kv_hit_ratios tensor
                            if pod_idx < modified_tensor['kv_hit_ratios'].shape[1]:
                                modified_tensor['kv_hit_ratios'][0, pod_idx, 0] = test_value
                        else:
                            # Other features are in pod_features_with_staleness
                            pod_feature_names = ['inflight_requests', 'gpu_kv_cache', 'cpu_kv_cache', 'running_requests', 
                                               'waiting_requests', 'prefill_tokens', 'decode_tokens']
                            
                            feature_idx = None
                            for i, name in enumerate(pod_feature_names):
                                if feature_key in name:
                                    feature_idx = i
                                    break
                            
                            if (feature_idx is not None and pod_idx < modified_tensor['pod_features_with_staleness'].shape[1] 
                                and feature_idx < modified_tensor['pod_features_with_staleness'].shape[2]):
                                modified_tensor['pod_features_with_staleness'][0, pod_idx, feature_idx] = test_value
                        
                        # Get modified prediction
                        try:
                            if args.model == "simpler_contextual_bandit":
                                modified_result, _ = simpler_contextual_bandit.infer_from_tensor(tensor_data=modified_tensor, model_updated=False)
                            else:
                                modified_result, _ = random_forest.infer_from_tensor(tensor_data=modified_tensor, model_updated=False)
                            
                            modified_pod = modified_result['selected_pod_index']
                            modified_confidence = modified_result['confidence']
                            
                            total_tests += 1
                            
                            # Calculate change magnitude
                            prediction_changed = modified_pod != baseline_pod
                            confidence_change = abs(modified_confidence - baseline_confidence)
                            relative_change = (test_value - original_value) / (abs(original_value) + 1e-8)
                            
                            if prediction_changed:
                                feature_changes += 1
                                print(f"    Pod {pod_idx} {feature_config['name']} "
                                           f"Δ{relative_change:+.1%}: Pod {baseline_pod}→{modified_pod} "
                                           f"(conf: {baseline_confidence:.3f}→{modified_confidence:.3f}) ✓")
                            else:
                                logger.debug(f"    Pod {pod_idx} {feature_config['name']} "
                                            f"Δ{relative_change:+.1%}: Pod {baseline_pod} (no change) "
                                            f"(conf: {modified_confidence:.3f}, Δ{confidence_change:.3f})")
                            
                            sensitivity_details.append({
                                'pod_idx': pod_idx,
                                'original_value': original_value,
                                'test_value': test_value,
                                'relative_change': relative_change,
                                'prediction_changed': prediction_changed,
                                'confidence_change': confidence_change,
                                'baseline_pod': baseline_pod,
                                'modified_pod': modified_pod,
                                'modified_confidence': modified_confidence,
                            })
                            
                        except Exception as e:
                            logger.warning(f"Error testing pod {pod_idx} modification: {e}")
                            continue
                
                # Calculate feature sensitivity metrics
                if total_tests > 0:
                    feature_sensitivity = feature_changes / total_tests
                    
                    # Additional metrics
                    avg_confidence_change = np.mean([d['confidence_change'] for d in sensitivity_details])
                    max_confidence_change = max([d['confidence_change'] for d in sensitivity_details])
                    
                    # Identify most sensitive pod
                    pod_changes = {}
                    for detail in sensitivity_details:
                        pod_idx = detail['pod_idx']
                        if pod_idx not in pod_changes:
                            pod_changes[pod_idx] = 0
                        if detail['prediction_changed']:
                            pod_changes[pod_idx] += 1
                    
                    most_sensitive_pod = max(pod_changes.keys(), key=lambda k: pod_changes[k]) if pod_changes else None
                    
                    if feature_key not in feature_sensitivity_results:
                        feature_sensitivity_results[feature_key] = []
                    
                    feature_sensitivity_results[feature_key].append({
                        'sensitivity': feature_sensitivity,
                        'total_tests': total_tests,
                        'changes': feature_changes,
                        'avg_confidence_change': avg_confidence_change,
                        'max_confidence_change': max_confidence_change,
                        'most_sensitive_pod': most_sensitive_pod,
                        'details': sensitivity_details
                    })
                    
                    print(f"  {feature_config['name']} sensitivity: {feature_sensitivity:.1%} "
                               f"({feature_changes}/{total_tests} tests changed prediction)")
                    print(f"  Average confidence change: {avg_confidence_change:.3f}")
                    print(f"  Most sensitive pod: {most_sensitive_pod}")
                else:
                    logger.warning(f"  No valid tests for {feature_config['name']}")
        
        except Exception as e:
            logger.error(f"Error analyzing sample {sample_idx + 1}: {str(e)}")
            continue
    
    # --- ENHANCED SENSITIVITY SUMMARY ---
    print(f"\n" + "=" * 70)
    print("🎯 IMPROVED FEATURE SENSITIVITY SUMMARY")
    print("=" * 70)
    
    # Calculate average sensitivity for each feature type
    feature_avg_sensitivity = {}
    feature_stats = {}
    
    for feature_key, results_list in feature_sensitivity_results.items():
        if results_list:
            sensitivities = [r['sensitivity'] for r in results_list]
            confidence_changes = [r['avg_confidence_change'] for r in results_list]
            
            feature_avg_sensitivity[feature_key] = np.mean(sensitivities)
            feature_stats[feature_key] = {
                'mean_sensitivity': np.mean(sensitivities),
                'std_sensitivity': np.std(sensitivities),
                'mean_confidence_change': np.mean(confidence_changes),
                'total_samples': len(results_list)
            }
    
    # Sort by sensitivity level
    sorted_features = sorted(feature_avg_sensitivity.items(), key=lambda x: x[1], reverse=True)
    
    print("\nFeature sensitivity ranking (highest to lowest):")
    print("-" * 50)
    
    for i, (feature_key, avg_sensitivity) in enumerate(sorted_features, 1):
        feature_name = feature_types.get(feature_key, {}).get('name', feature_key)
        stats = feature_stats[feature_key]
        
        if avg_sensitivity > 0.5:
            status = "🔥 HIGH"
        elif avg_sensitivity > 0.25:
            status = "📊 MODERATE"
        elif avg_sensitivity > 0.1:
            status = "⚠️  LOW"
        else:
            status = "❌ MINIMAL"
        
        print(f"{i:2d}. {feature_name:<20} {avg_sensitivity:6.1%} {status}")
        print(f"    Confidence change: {stats['mean_confidence_change']:.3f}±{stats['std_sensitivity']:.3f}")
    
    # Enhanced insights and recommendations
    print(f"\n🔍 ENHANCED INSIGHTS:")
    print("-" * 20)
    
    high_sensitivity_features = [k for k, v in feature_avg_sensitivity.items() if v > 0.3]
    moderate_sensitivity_features = [k for k, v in feature_avg_sensitivity.items() if 0.1 <= v <= 0.3]
    low_sensitivity_features = [k for k, v in feature_avg_sensitivity.items() if v < 0.1]
    
    if high_sensitivity_features:
        feature_names = [feature_types.get(k, {}).get('name', k) for k in high_sensitivity_features]
        print(f"✅ Model strongly responds to: {', '.join(feature_names)}")
    
    if moderate_sensitivity_features:
        feature_names = [feature_types.get(k, {}).get('name', k) for k in moderate_sensitivity_features]
        print(f"📊 Model moderately responds to: {', '.join(feature_names)}")
    
    if low_sensitivity_features:
        feature_names = [feature_types.get(k, {}).get('name', k) for k in low_sensitivity_features]
        print(f"⚠️  Model largely ignores: {', '.join(feature_names)}")
    
    # Overall assessment with more nuanced scoring
    overall_sensitivity_score = 0
    total_weight = 0
    
    for feature_key, sensitivity in feature_avg_sensitivity.items():
        # Weight more important features higher
        if feature_key in ['running_requests', 'waiting_requests', 'kv_hit_ratio']:
            weight = 2.0  # High importance
        elif feature_key in ['prefill_tokens', 'decode_tokens', 'inflight_requests']:
            weight = 1.5  # Medium importance
        else:
            weight = 1.0  # Standard importance
        
        overall_sensitivity_score += sensitivity * weight
        total_weight += weight
    
    if total_weight > 0:
        overall_sensitivity_score /= total_weight
        
        print(f"\n📊 Overall Weighted Pod Feature Sensitivity: {overall_sensitivity_score:.1%}")
        
        if overall_sensitivity_score > 0.4:
            print("🎉 EXCELLENT: Model demonstrates strong context-aware routing!")
        elif overall_sensitivity_score > 0.25:
            print("✅ GOOD: Model shows meaningful pod state awareness")
        elif overall_sensitivity_score > 0.15:
            print("📊 MODERATE: Some pod feature learning evident")
        else:
            print("⚠️  LIMITED: Model shows weak pod feature utilization")
    
    # Actionable recommendations
    print(f"\n💡 ACTIONABLE RECOMMENDATIONS:")
    print("-" * 25)
    
    if overall_sensitivity_score < 0.2:
        print("🔧 IMPROVE MODEL SENSITIVITY:")
        print("  - Increase feature amplification for critical features")
        print("  - Check feature normalization - might be over-normalizing")
        print("  - Verify reward signal differentiates based on pod performance")
    elif low_sensitivity_features:
        print("🔧 OPTIMIZE FEATURE UTILIZATION:")
        print(f"  - Consider removing or re-engineering: {', '.join(low_sensitivity_features)}")
        print("  - These features may be noise or poorly scaled")
    
    if high_sensitivity_features:
        print("✅ LEVERAGE HIGH-IMPACT FEATURES:")
        print(f"  - Focus monitoring on: {', '.join(high_sensitivity_features)}")
        print("  - These drive routing decisions most effectively")
    
    print("=" * 70)
    
    # 1. CONFIDENCE DISTRIBUTION ANALYSIS
    print(f"\n📊 CONFIDENCE DISTRIBUTION ANALYSIS:")
    print("-" * 40)
    
    all_confidences = []
    all_predictions = []
    
    for feature_key, results_list in feature_sensitivity_results.items():
        for result in results_list:
            for detail in result['details']:
                all_confidences.append(detail.get('modified_confidence', baseline_confidence))
                all_predictions.append(detail['modified_pod'])
    
    if all_confidences:
        conf_mean = np.mean(all_confidences)
        conf_std = np.std(all_confidences)
        conf_min = np.min(all_confidences)
        conf_max = np.max(all_confidences)
        
        print(f"Confidence across all perturbations:")
        print(f"  Mean: {conf_mean:.3f} ± {conf_std:.3f}")
        print(f"  Range: [{conf_min:.3f}, {conf_max:.3f}]")
        print(f"  Baseline: {baseline_confidence:.3f}")
        
        # Check if model is overconfident or underconfident
        if conf_max - conf_min < 0.1:
            print("  ⚠️  LOW CONFIDENCE SPREAD - Model may be uncertain or features poorly differentiated")
        elif conf_max > 0.8:
            print("  ⚠️  HIGH CONFIDENCE DETECTED - Check for overconfidence")
        
        # Analyze prediction diversity
        unique_predictions = len(set(all_predictions))
        total_pods = len(all_pods)
        prediction_diversity = unique_predictions / total_pods
        
        print(f"Prediction diversity: {unique_predictions}/{total_pods} pods used ({prediction_diversity:.1%})")
        
        if prediction_diversity < 0.5:
            print("  ⚠️  LOW PREDICTION DIVERSITY - Model may have strong biases")
    
    # 2. FEATURE INTERACTION ANALYSIS
    print(f"\n🔄 FEATURE INTERACTION HINTS:")
    print("-" * 30)
    
    # Look for patterns where certain pods are consistently sensitive
    pod_sensitivity_counts = {}
    for feature_key, results_list in feature_sensitivity_results.items():
        for result in results_list:
            most_sensitive_pod = result.get('most_sensitive_pod')
            if most_sensitive_pod is not None:
                if most_sensitive_pod not in pod_sensitivity_counts:
                    pod_sensitivity_counts[most_sensitive_pod] = []
                pod_sensitivity_counts[most_sensitive_pod].append(feature_key)
    
    print("Pods with high sensitivity to multiple features:")
    for pod_idx, features in pod_sensitivity_counts.items():
        if len(features) >= 3:  # Pod sensitive to 3+ features
            feature_names = [feature_types.get(f, {}).get('name', f) for f in features]
            print(f"  Pod {pod_idx}: {len(features)} features - {', '.join(feature_names)}")
            print("    💡 This pod may be a key decision boundary")
    
    # 3. ROUTING STRATEGY ANALYSIS
    print(f"\n🎯 INFERRED ROUTING STRATEGY:")
    print("-" * 30)
    
    # Analyze which direction of changes cause routing shifts
    capacity_features = ['running_requests', 'waiting_requests', 'inflight_requests']
    latency_features = ['last_second_avg_ttft_ms', 'last_second_avg_tpot_ms', 'last_second_p99_ttft_ms']
    cache_features = ['kv_hit_ratio']
    
    strategy_insights = []
    
    # Check capacity-based routing
    capacity_sensitivity = np.mean([feature_avg_sensitivity.get(f, 0) for f in capacity_features])
    if capacity_sensitivity > 0.15:
        strategy_insights.append("✅ Load-aware routing: Avoids overloaded pods")
    
    # Check latency-based routing  
    latency_sensitivity = np.mean([feature_avg_sensitivity.get(f, 0) for f in latency_features])
    if latency_sensitivity > 0.15:
        strategy_insights.append("✅ Performance-aware routing: Considers latency metrics")
    
    # Check cache-based routing
    cache_sensitivity = feature_avg_sensitivity.get('kv_hit_ratio', 0)
    if cache_sensitivity > 0.15:
        strategy_insights.append("✅ Cache-aware routing: Prefers high cache hit rates")
    
    if not strategy_insights:
        strategy_insights.append("⚠️  Unclear routing strategy - may be learning static preferences")
    
    for insight in strategy_insights:
        print(f"  {insight}")
    
    # 4. ACTIONABLE RECOMMENDATIONS
    print(f"\n💡 SPECIFIC RECOMMENDATIONS:")
    print("-" * 25)
    
    recommendations = []
    
    # Based on confidence analysis
    if conf_max - conf_min < 0.05:
        recommendations.append("🔧 Increase feature discrimination:")
        recommendations.append("   - Check if features are over-normalized")
        recommendations.append("   - Verify reward signal correlates with pod differences")
    
    # Based on sensitivity patterns
    if overall_sensitivity_score < 0.2:
        recommendations.append("🔧 Improve model responsiveness:")
        recommendations.append("   - Increase learning rate for pod-specific features")
        recommendations.append("   - Add regularization to prevent feature collapse")
    
    # Based on prediction diversity
    if 'prediction_diversity' in locals() and prediction_diversity < 0.6:
        dominant_pods = [k for k, v in pod_sensitivity_counts.items() if len(v) > 3]
        if dominant_pods:
            recommendations.append(f"🔧 Address pod bias toward: {dominant_pods}")
            recommendations.append("   - Check training data balance")
            recommendations.append("   - Verify reward calculation fairness")
    
    # Feature-specific recommendations
    if cache_sensitivity < 0.1:
        recommendations.append("🔧 Improve cache utilization:")
        recommendations.append("   - Verify KV hit ratio calculation accuracy")
        recommendations.append("   - Check if cache features are properly scaled")
    
    if latency_sensitivity < 0.1:
        recommendations.append("🔧 Enhance latency awareness:")
        recommendations.append("   - Ensure latency metrics are updated frequently")
        recommendations.append("   - Verify latency features aren't stale")
    
    if not recommendations:
        recommendations.append("✅ Model shows good feature utilization patterns")
    
    for rec in recommendations:
        print(f"  {rec}")
    
    # 5. COMPARATIVE BASELINE
    print(f"\n📏 BASELINE COMPARISON:")
    print("-" * 20)
    
    # Compare against random routing
    random_sensitivity = 1.0 / len(all_pods)  # Random chance of changing prediction
    print(f"Random baseline sensitivity: {random_sensitivity:.1%}")
    print(f"Model average sensitivity: {overall_sensitivity_score:.1%}")
    
    if overall_sensitivity_score > random_sensitivity * 3:
        print("✅ Model significantly outperforms random routing")
    elif overall_sensitivity_score > random_sensitivity * 1.5:
        print("📊 Model moderately better than random")
    else:
        print("❌ Model barely better than random - check training")
    
    # Compare against simple heuristics
    if capacity_sensitivity > 0.2:
        print("✅ Model implements reasonable load balancing")
    if latency_sensitivity > 0.15:
        print("✅ Model considers user experience metrics")
    if cache_sensitivity > 0.15:
        print("✅ Model optimizes for cache efficiency")
    
    print("=" * 70)
    
    return {
        'feature_sensitivity_results': feature_sensitivity_results,
        'feature_avg_sensitivity': feature_avg_sensitivity,
        'feature_stats': feature_stats,
        'sorted_features': sorted_features,
        'overall_sensitivity_score': overall_sensitivity_score,
        'confidence_analysis': {
            'mean': conf_mean if 'conf_mean' in locals() else 0,
            'std': conf_std if 'conf_std' in locals() else 0,
            'range': (conf_min, conf_max) if 'conf_min' in locals() else (0, 0),
            'prediction_diversity': prediction_diversity if 'prediction_diversity' in locals() else 0
        },
        'routing_strategy': {
            'capacity_sensitivity': capacity_sensitivity,
            'latency_sensitivity': latency_sensitivity, 
            'cache_sensitivity': cache_sensitivity,
            'insights': strategy_insights
        },
        'recommendations': recommendations,
        'pod_sensitivity_patterns': pod_sensitivity_counts
    }

def analyze_model_behavior(args, test_data_subset=None):
    """
    Analyze what the model has actually learned by systematically modifying features
    and observing prediction changes. This reveals if the model is truly contextual.
    """
    global NUM_TRAINS
    
    if NUM_TRAINS == 0:
        logger.warning("No trained model available for behavior analysis")
        return None
    
    logger.info("🔬 ANALYZING MODEL BEHAVIOR - What has the model learned?")
    logger.info("=" * 70)
    
    # Get a few test samples for analysis
    if test_data_subset is None:
        return None
    
    analysis_results = {
        'cache_sensitivity': [],
        'request_size_sensitivity': [],
        'pod_feature_sensitivity': [],
        'summary': {}
    }
    
    # Take first 5 test samples for detailed analysis
    test_items = list(test_data_subset.items())[:5]
    
    for sample_idx, (request_id, log_message) in enumerate(test_items):
        logger.info(f"\n--- ANALYZING SAMPLE {sample_idx + 1}/5 ({request_id}) ---")
        
        try:
            # Preprocess to get baseline data
            processed_df, _, all_pods, _ = preprocess.main(None, log_message, args.ttft_slo, args.avg_tpot_slo)
            
            # Apply normalization (same as training)
            request_features = ['input_tokens', 'output_tokens', 'total_tokens']
            pod_features_cols = [col for col in processed_df.columns if col.startswith('pod_') and processed_df[col].dtype in ['float64', 'int64']]
            all_features = request_features + pod_features_cols
            stats = get_request_stats()
            
            if stats.count > 0:
                for feature in all_features:
                    if feature in processed_df.columns and feature in stats.feature_stats:
                        feature_data = processed_df[feature].values.reshape(-1, 1)
                        normalized_feature = stats.feature_stats[feature].normalize(feature_data)
                        processed_df[feature] = normalized_feature.flatten()
            
            # Encode baseline data
            tensor_dataset, _ = encoding.encode_for_inference(all_pods, processed_df, stats, 
                                                            request_features_train, request_features_reward)
            
            # Get baseline prediction
            if args.model == "simpler_contextual_bandit":
                baseline_result, _ = simpler_contextual_bandit.infer_from_tensor(tensor_data=tensor_dataset, model_updated=False)
            else:
                baseline_result, _ = random_forest.infer_from_tensor(tensor_data=tensor_dataset, model_updated=False)
            
            baseline_pod = baseline_result['selected_pod_index']
            baseline_confidence = baseline_result['confidence']
            baseline_probs = baseline_result.get('pod_probabilities', [])
            
            logger.info(f"Baseline prediction: Pod {baseline_pod} (confidence: {baseline_confidence:.3f})")
            logger.info(f"Baseline probabilities: {[f'{p:.3f}' for p in baseline_probs]}")
            
            # --- TEST 1: CACHE SENSITIVITY ---
            logger.info(f"\n🧪 TEST 1: Cache Hit Ratio Sensitivity")
            logger.info("-" * 40)
            
            cache_changes = 0
            for cache_delta in [-0.6, -0.3, +0.3, +0.6]:  # Try different cache changes
                # Create modified tensor with cache changes
                modified_tensor = {k: v.clone() if hasattr(v, 'clone') else v for k, v in tensor_dataset.items()}
                
                # Modify cache ratios: lower preferred pod, raise alternative pod
                preferred_pod_idx = baseline_pod
                alternative_pod_idx = (baseline_pod + 1) % len(all_pods)
                
                # Apply cache modifications
                original_preferred_cache = modified_tensor['kv_hit_ratios'][0, preferred_pod_idx, 0].item()
                original_alternative_cache = modified_tensor['kv_hit_ratios'][0, alternative_pod_idx, 0].item()
                
                modified_tensor['kv_hit_ratios'][0, preferred_pod_idx, 0] = max(0.0, original_preferred_cache + cache_delta)
                modified_tensor['kv_hit_ratios'][0, alternative_pod_idx, 0] = min(1.0, original_alternative_cache - cache_delta)
                
                # Get modified prediction
                if args.model == "simpler_contextual_bandit":
                    modified_result, _ = simpler_contextual_bandit.infer_from_tensor(tensor_data=modified_tensor, model_updated=False)
                else:
                    modified_result, _ = random_forest.infer_from_tensor(tensor_data=modified_tensor, model_updated=False)
                
                modified_pod = modified_result['selected_pod_index']
                modified_confidence = modified_result['confidence']
                
                if modified_pod != baseline_pod:
                    cache_changes += 1
                    logger.info(f"  Cache Δ{cache_delta:+.1f}: Pod {baseline_pod}→{modified_pod} (conf: {baseline_confidence:.3f}→{modified_confidence:.3f}) ✓")
                else:
                    logger.info(f"  Cache Δ{cache_delta:+.1f}: Pod {baseline_pod} (no change) (conf: {modified_confidence:.3f})")
            
            cache_sensitivity = cache_changes / 4.0  # 4 tests
            analysis_results['cache_sensitivity'].append(cache_sensitivity)
            logger.info(f"Cache sensitivity: {cache_sensitivity:.1%} ({cache_changes}/4 tests changed prediction)")
            
            # --- TEST 2: REQUEST SIZE SENSITIVITY ---
            logger.info(f"\n🧪 TEST 2: Request Size Sensitivity") 
            logger.info("-" * 40)
            
            size_changes = 0
            original_input_tokens = tensor_dataset['request_features'][0, 0].item() if tensor_dataset['request_features'].shape[1] > 0 else 0
            
            for size_multiplier in [0.3, 0.6, 1.5, 3.0]:  # Different request sizes
                modified_tensor = {k: v.clone() if hasattr(v, 'clone') else v for k, v in tensor_dataset.items()}
                
                # Modify request size (first feature assumed to be input tokens)
                if modified_tensor['request_features'].shape[1] > 0:
                    modified_tensor['request_features'][0, 0] = original_input_tokens * size_multiplier
                
                # Get modified prediction
                if args.model == "simpler_contextual_bandit":
                    modified_result, _ = simpler_contextual_bandit.infer_from_tensor(tensor_data=modified_tensor, model_updated=False)
                else:
                    modified_result, _ = random_forest.infer_from_tensor(tensor_data=modified_tensor, model_updated=False)
                
                modified_pod = modified_result['selected_pod_index']
                modified_confidence = modified_result['confidence']
                
                if modified_pod != baseline_pod:
                    size_changes += 1
                    logger.info(f"  Size ×{size_multiplier}: Pod {baseline_pod}→{modified_pod} (conf: {baseline_confidence:.3f}→{modified_confidence:.3f}) ✓")
                else:
                    logger.info(f"  Size ×{size_multiplier}: Pod {baseline_pod} (no change) (conf: {modified_confidence:.3f})")
            
            size_sensitivity = size_changes / 4.0
            analysis_results['request_size_sensitivity'].append(size_sensitivity)
            logger.info(f"Request size sensitivity: {size_sensitivity:.1%} ({size_changes}/4 tests changed prediction)")
            
            # --- TEST 3: POD FEATURE SENSITIVITY ---
            logger.info(f"\n🧪 TEST 3: Pod Feature Sensitivity")
            logger.info("-" * 40)
            
            pod_feature_changes = 0
            pod_features_tested = 0
            
            # Test modifying individual pod features
            for feature_idx in range(min(3, tensor_dataset['pod_features_with_staleness'].shape[2])):  # Test first 3 pod features
                for delta in [-1.0, +1.0]:  # Try increasing/decreasing each feature
                    modified_tensor = {k: v.clone() if hasattr(v, 'clone') else v for k, v in tensor_dataset.items()}
                    
                    # Modify specific pod feature for preferred pod
                    preferred_pod_idx = baseline_pod
                    original_value = modified_tensor['pod_features_with_staleness'][0, preferred_pod_idx, feature_idx].item()
                    modified_tensor['pod_features_with_staleness'][0, preferred_pod_idx, feature_idx] = original_value + delta
                    
                    # Get modified prediction
                    if args.model == "simpler_contextual_bandit":
                        modified_result, _ = simpler_contextual_bandit.infer_from_tensor(tensor_data=modified_tensor, model_updated=False)
                    else:
                        modified_result, _ = random_forest.infer_from_tensor(tensor_data=modified_tensor, model_updated=False)
                    
                    modified_pod = modified_result['selected_pod_index']
                    modified_confidence = modified_result['confidence']
                    
                    pod_features_tested += 1
                    if modified_pod != baseline_pod:
                        pod_feature_changes += 1
                        logger.info(f"  Feature[{feature_idx}] Δ{delta:+.1f}: Pod {baseline_pod}→{modified_pod} (conf: {baseline_confidence:.3f}→{modified_confidence:.3f}) ✓")
                    else:
                        logger.info(f"  Feature[{feature_idx}] Δ{delta:+.1f}: Pod {baseline_pod} (no change) (conf: {modified_confidence:.3f})")
            
            pod_sensitivity = pod_feature_changes / max(1, pod_features_tested)
            analysis_results['pod_feature_sensitivity'].append(pod_sensitivity)
            logger.info(f"Pod feature sensitivity: {pod_sensitivity:.1%} ({pod_feature_changes}/{pod_features_tested} tests changed prediction)")
            
        except Exception as e:
            logger.error(f"Error analyzing sample {sample_idx + 1}: {str(e)}")
            continue
    
    # --- SUMMARY ANALYSIS ---
    logger.info(f"\n" + "=" * 70)
    logger.info("🎯 BEHAVIOR ANALYSIS SUMMARY")
    logger.info("=" * 70)
    
    if analysis_results['cache_sensitivity']:
        avg_cache_sensitivity = sum(analysis_results['cache_sensitivity']) / len(analysis_results['cache_sensitivity'])
        logger.info(f"Average Cache Sensitivity: {avg_cache_sensitivity:.1%}")
        
        if avg_cache_sensitivity > 0.5:
            logger.info("✅ Model strongly considers cache hit ratios")
        elif avg_cache_sensitivity > 0.25:
            logger.info("📊 Model moderately considers cache hit ratios")
        else:
            logger.info("❌ Model largely ignores cache hit ratios")
    
    if analysis_results['request_size_sensitivity']:
        avg_size_sensitivity = sum(analysis_results['request_size_sensitivity']) / len(analysis_results['request_size_sensitivity'])
        logger.info(f"Average Request Size Sensitivity: {avg_size_sensitivity:.1%}")
        
        if avg_size_sensitivity > 0.5:
            logger.info("✅ Model strongly adapts to request size")
        elif avg_size_sensitivity > 0.25:
            logger.info("📊 Model moderately adapts to request size")
        else:
            logger.info("❌ Model largely ignores request size")
    
    if analysis_results['pod_feature_sensitivity']:
        avg_pod_sensitivity = sum(analysis_results['pod_feature_sensitivity']) / len(analysis_results['pod_feature_sensitivity'])
        logger.info(f"Average Pod Feature Sensitivity: {avg_pod_sensitivity:.1%}")
        
        if avg_pod_sensitivity > 0.5:
            logger.info("✅ Model strongly considers pod characteristics")
        elif avg_pod_sensitivity > 0.25:
            logger.info("📊 Model moderately considers pod characteristics")
        else:
            logger.info("❌ Model largely ignores pod characteristics")
    
    # Overall contextual learning assessment
    sensitivities = []
    if analysis_results['cache_sensitivity']:
        sensitivities.append(sum(analysis_results['cache_sensitivity']) / len(analysis_results['cache_sensitivity']))
    if analysis_results['request_size_sensitivity']:
        sensitivities.append(sum(analysis_results['request_size_sensitivity']) / len(analysis_results['request_size_sensitivity']))
    if analysis_results['pod_feature_sensitivity']:
        sensitivities.append(sum(analysis_results['pod_feature_sensitivity']) / len(analysis_results['pod_feature_sensitivity']))
    
    if sensitivities:
        overall_contextual_score = sum(sensitivities) / len(sensitivities)
        logger.info(f"\nOverall Contextual Learning Score: {overall_contextual_score:.1%}")
        
        if overall_contextual_score > 0.6:
            logger.info("🎉 EXCELLENT: Model demonstrates strong contextual learning!")
        elif overall_contextual_score > 0.4:
            logger.info("✅ GOOD: Model shows contextual behavior")
        elif overall_contextual_score > 0.2:
            logger.info("⚠️  MODERATE: Some contextual learning, but room for improvement")
        else:
            logger.info("❌ POOR: Model appears to learn static preferences, not contextual routing")
        
        # Store summary
        analysis_results['summary'] = {
            'overall_score': overall_contextual_score,
            'avg_cache_sensitivity': sum(analysis_results['cache_sensitivity']) / len(analysis_results['cache_sensitivity']) if analysis_results['cache_sensitivity'] else 0,
            'avg_size_sensitivity': sum(analysis_results['request_size_sensitivity']) / len(analysis_results['request_size_sensitivity']) if analysis_results['request_size_sensitivity'] else 0,
            'avg_pod_sensitivity': sum(analysis_results['pod_feature_sensitivity']) / len(analysis_results['pod_feature_sensitivity']) if analysis_results['pod_feature_sensitivity'] else 0
        }
    
    logger.info("=" * 70)
    return analysis_results


if __name__ == "__main__":
    main()