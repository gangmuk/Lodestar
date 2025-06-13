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

# hyperparameters for simpler_contextual_bandit model
RL_MODEL_HYPERPARAMETERS = {
    'model_type': 'simplified',
    'hidden_dim': 32, # 256,
    'batch_size': 32,
    'lr': 0.01, # 0.001
    'weight_decay': 0.0001,
    'exploration_rate': 0.1,
    'training_epochs': 10, # 5,
    'max_updates_per_epoch': 100, # 1000000000
    'eval_interval': 10,
    'custom_weight_initialization': True,
    'entropy_bonus_factor': 0.01,
    'learning_every_x_iter': 5,
    'per_learn_reward_normalization': False,
    'normalization': {
        "SIGNAL_AMPLIFICATION_DEGREE": 1.0,  # 1.5
        "REWARD_AMPLIFICATION_DEGREE": 2.0,
        "REWARD_AMPLIFICATION_THRESHOLD": 0.5,
        "STD_THRESHOLD_FOR_REQ_FEAT_NORMALIZATION": 0.1,
        "STD_THRESHOLD_FOR_POD_FEAT_NORMALIZATION": 0.1,
        "ENABLE_POD_NORMALIZATION": True,
        "ENABLE_REQUEST_NORMALIZATION": True,
        "FEATURES_NORMALIZED": set(),
        "NUM_FEATURES_NORMALIZED": 0,
        "FEATURE_AMPLIFICATION": False,
        "FEATURES_AMPLIFIED": set(),
        "NUM_FEATURES_AMPLIFIED": 0,
    },
    'dataset_analysis': None,
    'GPU_MAP': {
        'NVIDIA-L20': 0,
        'NVIDIA-L40': 1,
        'NVIDIA-A10': 2,
        'NVIDIA-A100': 3,
        'NVIDIA-H100': 4,
    }
}


# Global variables (simplified for offline use)
ENCODED_DATA_DIR = "encoded_data"
NUM_TRAINS = 0
MODEL_UPDATED = False
TRAINING_DATA_UPDATED = False

TOTAL_NUM_DATA = 0
MIN_NUM_TRAINING_DATA = 500
LOCK_TRAINING_DATA = threading.Lock()
stats_instance = None
request_features_train = ['input_tokens', 'output_tokens', 'total_tokens']
# request_features_reward = ['ttft', 'avg_tpot', 'e2e_latency']

# this is supposed to be here which means after global variables
import model_and_data_analysis_helper

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
                simpler_contextual_bandit.train(ENCODED_DATA_DIR, args.model_dir, RL_MODEL_HYPERPARAMETERS)
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


def test_inference(args, log_message, feature_normalization_stats_file):
    """Test inference on a single log message with original vs predicted comparison"""
    global NUM_TRAINS, MODEL_UPDATED, stats_instance
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
        
        processed_df, _, all_pods, _ = preprocess.main(None, log_message, args.ttft_slo, args.avg_tpot_slo, RL_MODEL_HYPERPARAMETERS)
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
        if stats_instance is None:
            stats_instance = feature_normalization.get_stats_instance(feature_normalization_stats_file, RL_MODEL_HYPERPARAMETERS['normalization'])
        processed_df = feature_normalization.normalize_features_for_inference(processed_df, stats_instance)
        
        # Encode data
        encode_start_time = time.time()
        encoding.GPU_MAP = RL_MODEL_HYPERPARAMETERS['GPU_MAP']
        encoding.NUM_GPU_TYPES = len(RL_MODEL_HYPERPARAMETERS['GPU_MAP'])
        tensor_dataset, _ = encoding.encode_for_inference(all_pods, processed_df, request_features_train, RL_MODEL_HYPERPARAMETERS)
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
                model_updated=MODEL_UPDATED,
                HYPERPARAMETERS=RL_MODEL_HYPERPARAMETERS,
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
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Error in test_inference: {str(e)}")
        logger.error(f"Traceback: {error_traceback}")
        assert False

def process_training_data(args, log_data, feature_normalization_stats_file):
    """Process training data with shared normalization logic"""
    global ENCODED_DATA_DIR, NUM_TRAINS, MODEL_UPDATED, TRAINING_DATA_UPDATED, TOTAL_NUM_DATA, stats_instance
    flush_start_time = time.time()
    try:
        logger.info(f"Processing training data with {len(log_data)} entries")
        if not os.path.exists("temp_training_data"):
            os.mkdir("temp_training_data")
        raw_data = "temp_training_data/offline_batch.csv"
        
        # Write raw data to file
        ts_write_raw_data = time.time()
        write_to_file(log_data, raw_data)
        logger.info(f"Wrote {len(log_data)} entries to {raw_data}, took {time.time() - ts_write_raw_data} seconds")

        # Preprocess raw data
        ts_preprocess = time.time()
        processed_df, _, all_pods, _ = preprocess.main(raw_data, "", args.ttft_slo, args.avg_tpot_slo, RL_MODEL_HYPERPARAMETERS)
        logger.info(f"Successfully parsed data, took {time.time() - ts_preprocess} seconds")
        if stats_instance is None:
            stats_instance = feature_normalization.get_stats_instance(feature_normalization_stats_file, RL_MODEL_HYPERPARAMETERS['normalization'])
        processed_df, stats_instance, _ = feature_normalization.normalize_features_for_training(processed_df, stats_instance)
        stats_instance.write_stats_to_file(feature_normalization_stats_file)
        # processed_df = feature_normalization.try_reward_amplification(processed_df)
        
        # encoding
        ts_encode = time.time()
        encoded_data_output_dir = f"{ENCODED_DATA_DIR}/batch_1"
        encoding.GPU_MAP = RL_MODEL_HYPERPARAMETERS['GPU_MAP']
        encoding.NUM_GPU_TYPES = len(RL_MODEL_HYPERPARAMETERS['GPU_MAP'])
        encoding.encode_for_train(all_pods, processed_df, encoded_data_output_dir, request_features_train, RL_MODEL_HYPERPARAMETERS)
        logger.info(f"Successfully encoded data to {encoded_data_output_dir}, took {time.time() - ts_encode} seconds")

        # Verify encoded data
        expected_tensor_path = f"{encoded_data_output_dir}/tensor_dataset.pt"
        train_tensor_path = f"{encoded_data_output_dir}/train/tensor_dataset.pt"
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

def fetch_pod_gpu_mapping():
    """
    Fetch GPU model for each pod in the llama-3-8b-instruct deployment
    Returns a dictionary mapping pod_ip -> gpu_model
    """
    try:
        from kubernetes import client, config
        
        # Try local config first (for running outside cluster)
        try:
            config.load_kube_config()
            logger.info("Loaded local kubeconfig for Kubernetes access")
        except:
            # Fallback to in-cluster config (for running inside cluster)
            config.load_incluster_config()
            logger.info("Loaded in-cluster config for Kubernetes access")
        
        v1 = client.CoreV1Api()
        
        # Get all pods with the label selector for llama-3-8b-instruct
        label_selector = "model.aibrix.ai/name=llama-3-8b-instruct"
        pods = v1.list_pod_for_all_namespaces(label_selector=label_selector)
        
        pod_gpu_mapping = {}
        
        for pod in pods.items:
            if pod.status.phase == "Running" and pod.status.pod_ip:
                pod_ip = pod.status.pod_ip
                
                # Get the node name where this pod is running
                node_name = pod.spec.node_name
                
                if node_name:
                    # Get node details to extract GPU model
                    try:
                        node = v1.read_node(name=node_name)
                        node_labels = node.metadata.labels or {}
                        
                        # Extract GPU model from node label
                        gpu_model = node_labels.get('machine.cluster.vke.volcengine.com/gpu-name', 'unknown')
                        pod_gpu_mapping[pod_ip] = gpu_model
                        
                        logger.info(f"Pod {pod.metadata.name} (IP: {pod_ip}) -> GPU: {gpu_model}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to get node info for {node_name}: {e}")
                        pod_gpu_mapping[pod_ip] = 'unknown'
                else:
                    logger.warning(f"Pod {pod.metadata.name} has no node assignment")
                    pod_gpu_mapping[pod_ip] = 'unknown'
        
        logger.info(f"Successfully fetched GPU mapping for {len(pod_gpu_mapping)} pods")
        return pod_gpu_mapping
        
    except ImportError:
        logger.error("kubernetes package not installed. Install with: pip install kubernetes")
        assert False
    except Exception as e:
        logger.error(f"Failed to fetch pod GPU mapping: {e}")
        assert False

def main():
    parser = argparse.ArgumentParser(description='Offline Routing Agent Training and Testing')
    parser.add_argument('data_file', help='CSV file containing log messages for training')
    parser.add_argument('--skip_training', action='store_true', help='Skip training and only do inference')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='Train/test split ratio')
    parser.add_argument('--model', choices=['random_forest', 'simpler_contextual_bandit'], default='simpler_contextual_bandit', help='Model type to use for training (default: simpler_contextual_bandit)')
    parser.add_argument('--ttft_slo', type=float, help='TTFT SLO threshold for preprocessing', default=1000)
    parser.add_argument('--avg_tpot_slo', type=float, help='Average TPOT SLO threshold for preprocessing', default=50)
    parser.add_argument('--analyze_behavior', action='store_true', help='Analyze what the model has learned through feature sensitivity tests')

    args = parser.parse_args()

    # Check if data file exists
    if not os.path.exists(args.data_file):
        logger.error(f"Data file {args.data_file} not found")
        assert False


    logger.info("=== INITIALIZING GPU MAPPING ===")
    try:
        pod_gpu_mapping = fetch_pod_gpu_mapping()
        if pod_gpu_mapping:
            # Create GPU model to ID mapping
            unique_gpus = list(set(pod_gpu_mapping.values()))
            gpu_name_to_id = {gpu_name: idx for idx, gpu_name in enumerate(unique_gpus)}
            
            # Create direct pod_ip -> gpu_model_id mapping
            pod_gpu_id_mapping = {pod_ip: gpu_name_to_id[gpu_name] 
                                for pod_ip, gpu_name in pod_gpu_mapping.items()}
            
            # Update RL_MODEL_HYPERPARAMETERS with actual GPU mapping
            RL_MODEL_HYPERPARAMETERS['pod_gpu_mapping'] = pod_gpu_mapping
            RL_MODEL_HYPERPARAMETERS['pod_gpu_id_mapping'] = pod_gpu_id_mapping
            RL_MODEL_HYPERPARAMETERS['GPU_MAP'] = gpu_name_to_id
            
            logger.info(f"GPU name to ID mapping: {gpu_name_to_id}")
            logger.info(f"Created direct pod IP to GPU ID mapping for {len(pod_gpu_id_mapping)} pods")
        else:
            logger.warning("No pod GPU mapping available, using default GPU_MAP from hyperparameters")
            RL_MODEL_HYPERPARAMETERS['pod_gpu_mapping'] = {}
            RL_MODEL_HYPERPARAMETERS['pod_gpu_id_mapping'] = {}
            # Keep existing GPU_MAP from hyperparameters
            
    except Exception as e:
        logger.error(f"Failed to initialize GPU mapping: {e}")
        assert False

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
    args.model_dir = f"{data_dir}/final_model"
    os.makedirs(args.model_dir, exist_ok=True)
    feature_normalization_stats_file = f"{args.model_dir}/feature_normalization_statistics.pkl"
    logger.info(f"feature_normalization_stats_file: {feature_normalization_stats_file}")
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
    if not os.path.exists(ENCODED_DATA_DIR):
        os.makedirs(ENCODED_DATA_DIR)
    if os.path.exists(ENCODED_DATA_DIR):
        import shutil
        shutil.rmtree(ENCODED_DATA_DIR)
        os.makedirs(ENCODED_DATA_DIR)
        logger.info(f"Cleaned and recreated {ENCODED_DATA_DIR} for fresh offline training")
    
    # Read and process training data
    if not args.skip_training:
        logger.info("=== STARTING TRAINING PHASE ===")
        # Process training data
        if not process_training_data(args, train_data, feature_normalization_stats_file):
            logger.error("Failed to process training data")
            return
        model_and_data_analysis_helper.diagnose_training_data_issues(args, train_data)
        
        # Train model
        ret = train_model(args)
        if not ret:
            logger.error("Failed to train model")
            return
        logger.info("=== TRAINING COMPLETED ===")
    else:
        logger.info("=== SKIPPING TRAINING (using existing model) ===")

    # NEW: Behavior Analysis (before regular testing)
    if args.analyze_behavior and test_data and len(test_data) > 0:
        logger.info("=== STARTING BEHAVIOR ANALYSIS ===")
        # model_and_data_analysis_helper.analyze_model_behavior(args, test_data, feature_normalization_stats_file)
        _ = model_and_data_analysis_helper.analyze_detailed_feature_sensitivity(args, test_data, feature_normalization_stats_file)
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
            result = test_inference(args, log_message, feature_normalization_stats_file)
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


if __name__ == "__main__":
    main()