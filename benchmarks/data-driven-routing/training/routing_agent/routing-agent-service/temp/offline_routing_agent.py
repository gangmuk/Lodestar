# offline_routing_agent.py

from flask import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import os
import logging
import time
import sys
import encoding
import simpler_contextual_bandit
import preprocess
import pickle
import threading
import argparse
import random_forest
import torch
import feature_normalization
import model_and_data_analysis_helper
from logger import logger, INCLUDE_GPU_IN_FEATURE
from kubernetes import client, config
import shutil
import re
import csv
import utils.utils as utils
import random
import hashlib


def set_all_seeds(seed=42):
    """Set seeds for all sources of randomness to ensure reproducible results."""
    # Python's random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch CPU operations
    torch.manual_seed(seed)
    
    # PyTorch GPU operations (if using CUDA)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Python hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Make PyTorch operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set PyTorch to use deterministic algorithms where possible
    if hasattr(torch, 'use_deterministic_algorithms'):
        torch.use_deterministic_algorithms(True, warn_only=True)
    
    print(f"All seeds set to {seed} for reproducible results")

set_all_seeds(42)

RL_MODEL_HYPERPARAMETERS = {
    'model_type': 'simplified',
    'hidden_dim': 32, # 256,
    'batch_size': 32,
    'lr': 0.01, # 0.001
    'weight_decay': 0.0001,
    
    'exploration_rate': 0.0,
    'explore': False,
    
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
        "FEATURES_NORMALIZED": set(),
        "NUM_FEATURES_NORMALIZED": 0,
        "FEATURE_AMPLIFICATION": False,
        "FEATURES_AMPLIFIED": set(),
        "NUM_FEATURES_AMPLIFIED": 0,
    },
    'dataset_analysis': None,
    'deterministic_training': True,
    'training_seed': 42,
}

# Global variables (simplified for offline use)
NUM_TRAINS = 0
MODEL_UPDATED = False
TRAINING_DATA_UPDATED = False

TOTAL_NUM_DATA = 0
MIN_NUM_TRAINING_DATA = 500
LOCK_TRAINING_DATA = threading.Lock()
stats_instance = None
request_features_train = ['input_tokens', 'output_tokens', 'total_tokens']
# request_features_reward = ['ttft', 'avg_tpot', 'e2e_latency']

def static_hash(value: str) -> str:
    hash_object = hashlib.sha256(value.encode())
    return hash_object.hexdigest()[:8]

def write_to_file(log_data, raw_data):
    with open(raw_data, "w") as log_file:
        for request_id, log_message in log_data.items():
            log_file.write(f"{log_message}\n")
    logger.info(f"Successfully wrote {len(log_data)} entries to {raw_data}")

# def read_csv_data(csv_file):
#     logger.info(f"Reading data from {csv_file}")
#     try:
#         df = pd.read_csv(csv_file)
#         if 'log_message' in df.columns:
#             log_messages = df['log_message'].tolist()
#         elif len(df.columns) == 1:
#             # Single column, assume it's log messages
#             log_messages = df.iloc[:, 0].tolist()
#         else:
#             logger.error(f"CSV file must have a 'log_message' column or be a single column file")
#             return None
#     except:
#         try:
#             with open(csv_file, 'r') as f:
#                 log_messages = [line.strip() for line in f if line.strip()]
#         except Exception as e:
#             logger.error(f"Error reading file {csv_file}: {e}")
#             return None
#     cleaned_messages = []
#     for i, log_message in enumerate(log_messages):
#         if log_message and log_message.strip():
#             clean_message = log_message.strip()
#             if i < 3:
#                 logger.info(f"Original message {i}: {clean_message[:150]}...")
#             bracket_pos = clean_message.rfind('] ')
#             if bracket_pos != -1:
#                 clean_message = clean_message[bracket_pos + 2:]
#             if not clean_message.startswith('**@latency_metrics@'):
#                 metrics_pos = clean_message.find('**@latency_metrics@')
#                 if metrics_pos != -1:
#                     clean_message = clean_message[metrics_pos:]
#             if clean_message.startswith('**@latency_metrics@'):
#                 cleaned_messages.append(clean_message)
#                 # Debug: show cleaned message for first few entries
#                 if i < 3:
#                     logger.info(f"Cleaned message {i}: {clean_message[:150]}...")
#             else:
#                 logger.warning(f"Skipping malformed log message {i}: {log_message[:100]}...")
#     log_data = {}
#     for i, log_message in enumerate(cleaned_messages):
#         log_data[f"request_{i}"] = log_message
#     logger.info(f"Successfully read {len(log_data)} log messages from {csv_file} (cleaned from {len(log_messages)} raw entries)")
#     if log_data:
#         first_key = list(log_data.keys())[0]
#         sample_message = log_data[first_key]
#         logger.info(f"Sample cleaned message: {sample_message[:200]}...")
#     return log_data

def read_csv_data(log_file):
    """Simple function to read log entries from a text file in deterministic order"""
    from collections import OrderedDict
    
    log_data = OrderedDict()
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        request_count = 0
        for line_num, line in enumerate(lines):
            line = line.strip()
            if line and '**@latency_metrics@' in line:
                # Remove the log prefix (everything up to and including '] ')
                bracket_pos = line.rfind('] ')
                if bracket_pos != -1:
                    clean_line = line[bracket_pos + 2:]
                else:
                    clean_line = line
                
                # Use line number to ensure deterministic ordering
                log_data[f"request_{request_count}"] = clean_line
                request_count += 1
        
        print(f"Successfully read {len(log_data)} log entries from {log_file}")
        print(f"Entries are in the exact order they appeared in the file (lines processed: {len(lines)})")
        return log_data
        
    except Exception as e:
        print(f"Error reading file {log_file}: {e}")
        return None

def train_model(args, ENCODED_DATA_DIR):
    global NUM_TRAINS, MODEL_UPDATED, TRAINING_DATA_UPDATED, TOTAL_NUM_DATA 
    if TRAINING_DATA_UPDATED and TOTAL_NUM_DATA > MIN_NUM_TRAINING_DATA:
        training_start_time = time.time()
        logger.info(f"Starting {NUM_TRAINS}th training of routing agent")
        try:
            if args.model == "random_forest":
                random_forest.train(ENCODED_DATA_DIR)
            elif args.model == "simpler_contextual_bandit":
                # set_all_seeds(RL_MODEL_HYPERPARAMETERS['training_seed'])
                # if not verify_training_determinism(
                #     ENCODED_DATA_DIR, 
                #     f"{args.model_dir}_test", 
                #     RL_MODEL_HYPERPARAMETERS
                # ):
                #     print("❌ Training is not deterministic - fixing required!")
                #     return
                # else:
                #     print("✅ Training determinism verified!")
                
                set_all_seeds(RL_MODEL_HYPERPARAMETERS['training_seed'])
                simpler_contextual_bandit.train(ENCODED_DATA_DIR, args.model_dir, RL_MODEL_HYPERPARAMETERS)
            else:
                logger.error(f"Unknown model type: {args.model}")
                assert False
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
            assert False
    else:
        logger.info(f"Not enough training data available (TOTAL_NUM_DATA: {TOTAL_NUM_DATA}), skipping training")
        assert False


def test_inference(args, log_message, request_id):
    global NUM_TRAINS, MODEL_UPDATED, stats_instance
    set_all_seeds(42)
    if NUM_TRAINS == 0:
        logger.warning("No trained model available, please train first")
        return None
    handle_infer_start_time = time.time()
    processed_df, _, sorted_all_pod_ids, _ = preprocess.main(None, log_message, args.ttft_slo, args.avg_tpot_slo, RL_MODEL_HYPERPARAMETERS)
    preprocess_overhead = time.time() - handle_infer_start_time
    original_pod_choice = processed_df['selected_pod'].iloc[0] if len(processed_df) > 0 else None
    normalized_df = feature_normalization.normalize_features_for_inference(processed_df, stats_instance, request_id)
    encode_start_time = time.time()
    tensor_dataset, _ = encoding.encode_for_inference(sorted_all_pod_ids, normalized_df, request_features_train, RL_MODEL_HYPERPARAMETERS)
    handle_infer_total_total_encoding_overhead = time.time() - encode_start_time
    infer_from_tensor_start_time = time.time()
    if args.model == "random_forest":
        result, _ = random_forest.infer_from_tensor(
            tensor_data=tensor_dataset, 
            exploration_enabled=True, 
            exploration_rate=RL_MODEL_HYPERPARAMETERS['exploration_rate'], 
            model_updated=MODEL_UPDATED
    )
    elif args.model == "simpler_contextual_bandit":
        result, _ = simpler_contextual_bandit.infer_from_tensor(
            tensor_data=tensor_dataset, 
            request_id=request_id,
            model_updated=MODEL_UPDATED,
            HYPERPARAMETERS=RL_MODEL_HYPERPARAMETERS,
        )
    if MODEL_UPDATED:
        logger.info("Model updated flag consumed, resetting to False")
        MODEL_UPDATED = False
    handle_infer_total_total_infer_from_tensor_overhead = time.time() - infer_from_tensor_start_time
    selected_pod_index = result['selected_pod_index']
    if selected_pod_index >= len(sorted_all_pod_ids):
        logger.warning(f"Selected pod index {selected_pod_index} out of range, defaulting to first pod")
        selected_pod_index = 0
    selected_pod = sorted_all_pod_ids[selected_pod_index]
    handle_infer_total_overhead = time.time() - handle_infer_start_time
    prediction_matches = (selected_pod == original_pod_choice) if original_pod_choice else None
    result_summary = {
        "selected_pod": selected_pod,
        "original_pod_choice": original_pod_choice,
        "pod_probabilities": result['pod_probabilities'],
        "prediction_matches": prediction_matches,
        "confidence": result['confidence'],
        "total_inference_time_ms": handle_infer_total_overhead * 1000,
        "preprocess_time_ms": preprocess_overhead * 1000,
        "encoding_time_ms": handle_infer_total_total_encoding_overhead * 1000,
        "inference_time_ms": handle_infer_total_total_infer_from_tensor_overhead * 1000,
    }
    
    # Enhanced logging with match/mismatch status
    if original_pod_choice:
        match_status = "✅ MATCH" if prediction_matches else "❌ MISMATCH"
        logger.info(f"Inference result: predicted={selected_pod}, original={original_pod_choice}, {match_status}, confidence={result['confidence']:.4f}")
    else:
        logger.info(f"Inference result: predicted={selected_pod}, original=UNKNOWN, confidence={result['confidence']:.4f}")

    return result_summary

def process_training_data(args, train_data, stats_instance, ENCODED_DATA_DIR):
    global NUM_TRAINS, MODEL_UPDATED, TRAINING_DATA_UPDATED, TOTAL_NUM_DATA
    flush_start_time = time.time()
    logger.info(f"Processing training data with {len(train_data)} entries")
    if not os.path.exists("temp_training_data"):
        os.mkdir("temp_training_data")
    raw_data = "temp_training_data/offline_batch.csv"
    write_to_file(train_data, raw_data)
    ts_preprocess = time.time()
    processed_df, _, sorted_all_pod_ids, _ = preprocess.main(raw_data, "", args.ttft_slo, args.avg_tpot_slo, RL_MODEL_HYPERPARAMETERS)
    processed_df.to_csv(f"{args.data_dir}/processed_data.csv", index=False)
    logger.info(f"Successfully parsed data, took {time.time() - ts_preprocess} seconds")
    
    # update_stats_incrementally is called inside normalize_features_for_training
    processed_df = feature_normalization.normalize_features_for_training(processed_df, stats_instance)
    # processed_df = feature_normalization.try_reward_amplification(processed_df)
    processed_df.to_csv(f"{args.data_dir}/normalized_data.csv", index=False)
    
    # encoding
    ts_encode = time.time()
    encoded_data_output_dir = f"{ENCODED_DATA_DIR}/batch_1"
    encoding.encode_for_train(sorted_all_pod_ids, processed_df, encoded_data_output_dir, request_features_train, RL_MODEL_HYPERPARAMETERS)
    logger.info(f"Successfully encoded data to {encoded_data_output_dir}, took {time.time() - ts_encode} seconds")

    # Verify encoded data
    expected_tensor_path = f"{encoded_data_output_dir}/tensor_dataset.pt"
    train_tensor_path = f"{encoded_data_output_dir}/train/tensor_dataset.pt"
    if os.path.exists(expected_tensor_path):
        logger.info(f"✓ Found tensor dataset at: {expected_tensor_path}")
    elif os.path.exists(train_tensor_path):
        logger.info(f"✓ Found tensor dataset at: {train_tensor_path}")
    TRAINING_DATA_UPDATED = True
    TOTAL_NUM_DATA += len(train_data)
    logger.info(f"Successfully processed {len(train_data)} log messages, took {time.time() - flush_start_time} seconds")
    return True


def ensure_deterministic_data_split(all_data, split_ratio=0.8, seed=42):
    """Ensure consistent train/test split across runs."""
    # Sort by keys to ensure consistent ordering
    sorted_items = sorted(all_data.items())
    all_messages = [msg for _, msg in sorted_items]
    
    # Use seed for any randomization if needed
    random.seed(seed)
    
    split_point = int(len(all_messages) * split_ratio)
    train_messages = all_messages[:split_point]
    test_messages = all_messages[split_point:]
    
    print(f"Deterministic split: {len(train_messages)} train, {len(test_messages)} test")
    print(f"First test message hash: {static_hash(test_messages[0]) if test_messages else 'None'}")
    
    return train_messages, test_messages


# Fixed verification function - remove unused variables
def verify_training_determinism(encoded_data_dir, model_output_dir, HYPERPARAMETERS):
    """Verify that training produces identical results across runs"""
    logger.info("🔍 VERIFYING TRAINING DETERMINISM")
    
    # Train model twice with same settings
    logger.info("Training model #1...")
    set_all_seeds(HYPERPARAMETERS['training_seed'])
    simpler_contextual_bandit.train(encoded_data_dir, f"{model_output_dir}_test1", HYPERPARAMETERS)
    
    logger.info("Training model #2...")
    set_all_seeds(HYPERPARAMETERS['training_seed'])
    simpler_contextual_bandit.train(encoded_data_dir, f"{model_output_dir}_test2", HYPERPARAMETERS)
    
    # Compare final model weights
    model1_path = f"{model_output_dir}_test1/policy.pth"
    model2_path = f"{model_output_dir}_test2/policy.pth"
    
    if os.path.exists(model1_path) and os.path.exists(model2_path):
        weights1 = torch.load(model1_path, map_location='cpu')
        weights2 = torch.load(model2_path, map_location='cpu')
        
        weights_identical = True
        total_diff = 0.0
        
        for key in weights1.keys():
            if not torch.equal(weights1[key], weights2[key]):
                diff = (weights1[key] - weights2[key]).abs().max().item()
                total_diff += diff
                logger.error(f"❌ Weight mismatch in layer: {key}, max_diff: {diff:.8f}")
                weights_identical = False
            else:
                logger.debug(f"✅ Weights identical in layer: {key}")
                logger.info(f"Layer {key} weights are identical. weights1[{key}]: {weights1[key]}, weights2[{key}]: {weights2[key]}")
        
        if weights_identical:
            logger.info("✅ TRAINING DETERMINISM VERIFIED - Identical weights across runs")
        else:
            logger.error(f"❌ TRAINING DETERMINISM FAILED - Total weight difference: {total_diff:.8f}")
        
        # Clean up test models
        import shutil
        try:
            shutil.rmtree(f"{model_output_dir}_test1")
            shutil.rmtree(f"{model_output_dir}_test2")
            logger.info("🧹 Cleaned up test model directories")
        except:
            pass
        
        return weights_identical
    else:
        logger.error("❌ Could not find model files for comparison")
        return False


def main():
    global stats_instance
    parser = argparse.ArgumentParser(description='Offline Routing Agent Training and Testing')
    parser.add_argument('data_file', help='CSV file containing log messages for training')
    parser.add_argument('--skip_training', action='store_true', help='Skip training and only do inference')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='Train/test split ratio')
    parser.add_argument('--model', choices=['random_forest', 'simpler_contextual_bandit'], default='simpler_contextual_bandit', help='Model type to use for training (default: simpler_contextual_bandit)')
    parser.add_argument('--ttft_slo', type=float, help='TTFT SLO threshold for preprocessing', default=1000)
    parser.add_argument('--avg_tpot_slo', type=float, help='Average TPOT SLO threshold for preprocessing', default=50)
    parser.add_argument('--analyze_behavior', action='store_true', help='Analyze what the model has learned through feature sensitivity tests')
    args = parser.parse_args()
    if not os.path.exists(args.data_file):
        logger.error(f"Data file {args.data_file} not found")
        assert False
    
    def replace_pod_ip_with_generalpodid(data_file):
        all_pod_ips_from_training_data = sorted(utils.get_all_pod_ips_from_data_file(data_file))
        if not all_pod_ips_from_training_data:
            logger.error(f"No pod IPs found in data file {data_file}")
            assert False
            
        logger.info(f"🔍 Deterministic pod IP order: {all_pod_ips_from_training_data}")

        pod_ip_to_generalpodid = utils.create_pod_ip_to_generalpodid_mapping(all_pod_ips_from_training_data)
        
        logger.info(f"🔍 Deterministic mapping: {pod_ip_to_generalpodid}")

        
        
        with open(data_file, 'r') as f:
            content = f.read()
        for pod_ip, generalpodid in pod_ip_to_generalpodid.items():
            content = content.replace(pod_ip, generalpodid)
        replaced_data_file = data_file.replace('.csv', '_replaced.csv')
        with open(replaced_data_file, 'w') as f:
            f.write(content)
        logger.info(f"File write {replaced_data_file} with replaced generalpodids")
        return replaced_data_file
    
    replaced_data_file = replace_pod_ip_with_generalpodid(args.data_file)
    all_data = {}
    if os.path.isfile(replaced_data_file):
        data_dir = os.path.dirname(replaced_data_file)
        logger.info(f"data_file is a file: {replaced_data_file}")
        all_data = read_csv_data(replaced_data_file)
    # elif os.path.isdir(args.data_file):
    #     data_dir = args.data_file
    #     logger.info(f"data_file is a directory: {args.data_file}")
    #     for root, dirs, files in os.walk(args.data_file):
    #         for file in files:
    #             if file == "data.csv":
    #                 file_path = os.path.join(root, file)
    #                 logger.info(f"Found data.csv at: {file_path}")
    #                 data = read_csv_data(file_path)
    #                 if data:
    #                     # Merge data dictionaries with unique keys
    #                     for key, value in data.items():
    #                         new_key = f"{os.path.basename(root)}_{key}"
    #                         all_data[new_key] = value
    else:
        logger.error(f"args.data_file must be a file It is a directory or the path does not exist. args.data_file: {replaced_data_file}")
        assert False

    if all_data is None or len(all_data) == 0:
        logger.error("Failed to read data or no valid log messages found")
        return
    
    train_messages, test_messages = ensure_deterministic_data_split(all_data, args.split_ratio)
    test_messages = test_messages[:10]
    
    args.data_dir = data_dir
    args.model_dir = f"{data_dir}/final_model"
    os.makedirs(args.model_dir, exist_ok=True)
    train_data = {f"request_{i}": msg for i, msg in enumerate(train_messages)}
    def extract_request_id(log_message):
        match = re.search(r'@requestID@([^@]+)@', log_message)
        return match.group(1) if match else None
    # test_data = {f"request_{i}": msg for i, msg in enumerate(test_messages)}
    test_data = []
    for msg in test_messages:
        test_data.append({"request_id": extract_request_id(msg), "message": msg})

    ENCODED_DATA_DIR = "encoded_data"
    if not os.path.exists(ENCODED_DATA_DIR):
        os.makedirs(ENCODED_DATA_DIR)
    if os.path.exists(ENCODED_DATA_DIR):
        shutil.rmtree(ENCODED_DATA_DIR)
        os.makedirs(ENCODED_DATA_DIR)
        logger.info(f"Cleaned and recreated {ENCODED_DATA_DIR} for fresh offline training")
    feature_normalization_stats_file = f"{args.model_dir}/feature_normalization_statistics.csv"
    
    if stats_instance is not None:
        logger.error("Using existing stats instance for normalization")
        assert False
    stats_instance = feature_normalization.get_stats_instance(RL_MODEL_HYPERPARAMETERS['normalization'], None)
    process_training_data(args, train_data, stats_instance, ENCODED_DATA_DIR)
    stats_instance.write_stats_to_file(feature_normalization_stats_file)
    model_and_data_analysis_helper.diagnose_training_data_issues(ENCODED_DATA_DIR)
    train_model(args, ENCODED_DATA_DIR)

    # NEW: Behavior Analysis (before regular testing)
    if args.analyze_behavior and test_data and len(test_data) > 0:
        logger.info("=== STARTING BEHAVIOR ANALYSIS ===")
        # model_and_data_analysis_helper.analyze_model_behavior(args, test_data, feature_normalization_stats_file)
        _ = model_and_data_analysis_helper.analyze_detailed_feature_sensitivity(args, test_data, feature_normalization_stats_file)
        logger.info("=== BEHAVIOR ANALYSIS COMPLETED ===")
    
    
    # Test inference
    if test_data and len(test_data) > 0:
        logger.info("=== STARTING TESTING PHASE ===")
        success_count = 0
        match_count = 0
        mismatch_count = 0
        unknown_original_count = 0
        test_count = 10
        selected_pod_list = []
        pod_probabilities_list = []
        message_list = []
        for td in test_data:
            log_message = td['message']
            request_id = td['request_id']
            result = test_inference(args, log_message, request_id)
            selected_pod_list.append(result['selected_pod'])
            message_list.append(log_message)
            
            print()
            print(f"Request_id: {request_id}, Selected Pod: {result['selected_pod']}")
            # print(f"Message: {log_message}")
            print(f"pod_probabilities_list: ", end="")
            for prob in result['pod_probabilities']:
                print(f"{prob:.2f}", end=", ")
            print()
            print()
            
            if result:
                success_count += 1
                if result['prediction_matches'] is True:
                    match_count += 1
                elif result['prediction_matches'] is False:
                    mismatch_count += 1
                else:
                    unknown_original_count += 1
                if result['original_pod_choice']:
                    match_status = "MATCH" if result['prediction_matches'] else "MISMATCH"
                    logger.info(f"  → Predicted: {result['selected_pod']}, Original: {result['original_pod_choice']}, Status: {match_status}, Confidence: {result['confidence']:.3f}")
                else:
                    logger.info(f"  → Predicted: {result['selected_pod']}, Original: UNKNOWN, Confidence: {result['confidence']:.3f}")
            else:
                logger.error(f"✗ Failed inference for {request_id}")
                
        # logger.info("=" * 60)
        # logger.info("=== TESTING SUMMARY ===")
        # logger.info(f"Total tests: {test_count}")
        # logger.info(f"Successful inferences: {success_count}/{test_count} ({success_count/test_count*100:.1f}%)")
        # if match_count + mismatch_count > 0:
        #     accuracy = match_count / (match_count + mismatch_count) * 100
        #     logger.info(f"Prediction accuracy: {match_count}/{match_count + mismatch_count} ({accuracy:.1f}%)")
        #     logger.info(f"  - Matches: {match_count}")
        #     logger.info(f"  - Mismatches: {mismatch_count}")
        # if unknown_original_count > 0:
        #     logger.info(f"  - Unknown original: {unknown_original_count}")
        # logger.info("=" * 60)

if __name__ == "__main__":
    main()