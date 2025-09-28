# offline_routing_agent.py

import os
import time
import encoding
import simpler_contextual_bandit
import latency_predictor
import preprocess
import threading
import argparse
import json
import random_forest
import torch
import data_normalizer
import model_and_data_analysis_helper
import shutil
import re
import utils as utils
import random
from logger import logger, INCLUDE_GPU_IN_FEATURE
import pandas as pd
import data_normalizer

utils.set_all_seeds(42)


# Hyperparameters are loaded later inside main() from the single source JSON
RL_MODEL_HYPERPARAMETERS = {}

# RL_MODEL_HYPERPARAMETERS = {
#     'hidden_dim': 64, # 256,
#     'batch_size': 64,
#     # 'offline_learning_rate': 0.001,
#     'ONLINE_LEARNING_RATE': 0.0005,
#     'training_epochs': 5, # 5,
#     'learning_every_x_iter': 5,
#     'weight_decay': 0.0001,
#     'max_updates_per_epoch': 1000, # 1000000000
#     'exploration_rate': 0.1, # 0.1
#     'explore': True,
#     'weight_initialization': 'xavier', # 'kaiming', 'xavier', 'static'
    
#     'eval_interval': 10,
#     'entropy_bonus_factor': 0.02,
#     'per_learn_reward_normalization': False,
#     'normalization': {
#         "SIGNAL_AMPLIFICATION_DEGREE": 1.0,  # 1.5
#         "REWARD_AMPLIFICATION_DEGREE": 1.0,
#         "REWARD_AMPLIFICATION_THRESHOLD": 0.5,
#         "STD_THRESHOLD_FOR_REQ_FEAT_NORMALIZATION": 0.1,
#         "STD_THRESHOLD_FOR_POD_FEAT_NORMALIZATION": 0.1,
#         "FEATURES_NORMALIZED": set(),
#         "NUM_FEATURES_NORMALIZED": 0,
#         "FEATURE_AMPLIFICATION": False,
#         "FEATURES_AMPLIFIED": set(),
#         "NUM_FEATURES_AMPLIFIED": 0,
#     },
#     'dataset_analysis': None,
#     'deterministic_training': True,
#     'training_seed': 42,
#     # 'REWARD_FUNCTION': 'linear_simple',
#     # 'TTFT_SLO': 1000,  # Default TTFT SLO threshold (ms)
#     # 'AVG_TPOT_SLO': 50,  # Default average TPOT SLO threshold (ms)
# }

# Global variables (simplified for offline use)
# excluded_pod_feature = ["kv_hit_ratio"]
# excluded_pod_feature = []
NUM_TRAINS = 0
MODEL_UPDATED = False
TRAINING_DATA_UPDATED = False
POD_LABEL_SELECTOR="model.aibrix.ai/name=llama-3-8b-instruct"
TOTAL_NUM_DATA = 0
MIN_NUM_TRAINING_DATA = 50  # Reduced for testing with smaller datasets
LOCK_TRAINING_DATA = threading.Lock()
stats_instance = None
request_features_train = ['input_tokens', 'output_tokens', 'total_tokens']
# request_features_reward = ['ttft', 'avg_tpot', 'e2e_latency']

# NOTE: read_csv_data function removed - no longer needed with new pipeline
# Raw data processing is now handled by data_processor.py

def train_model(ENCODED_DATA_DIR, is_online_learning, final_model_dir):
    global NUM_TRAINS, MODEL_UPDATED, TRAINING_DATA_UPDATED, TOTAL_NUM_DATA 
    if TRAINING_DATA_UPDATED and TOTAL_NUM_DATA > MIN_NUM_TRAINING_DATA:
        training_start_time = time.time()
        logger.info(f"Starting {NUM_TRAINS}th training of routing agent")
        try:
            utils.set_all_seeds(RL_MODEL_HYPERPARAMETERS['training_seed'])
            
            # Select model type based on hyperparameters
            model_type = RL_MODEL_HYPERPARAMETERS.get('MODEL_TYPE', 'contextual_bandit')
            
            if model_type == 'latency_predictor':
                logger.info("Training with latency predictor model")
                saved_plot_path = latency_predictor.train_latency_predictor(ENCODED_DATA_DIR, final_model_dir, RL_MODEL_HYPERPARAMETERS)
            elif model_type == 'rl_contextual_bandit_sb3':
                logger.info("Training with SB3 RL contextual bandit model")
                import rl_contextual_bandit_sb3
                saved_plot_path = rl_contextual_bandit_sb3.train(ENCODED_DATA_DIR, final_model_dir, RL_MODEL_HYPERPARAMETERS, is_online_learning)
            else:
                logger.info("Training with contextual bandit model")
                saved_plot_path = simpler_contextual_bandit.train(ENCODED_DATA_DIR, final_model_dir, RL_MODEL_HYPERPARAMETERS, is_online_learning)
            
            MODEL_UPDATED = True
            TRAINING_DATA_UPDATED = False
            NUM_TRAINS += 1
            logger.info(f"Successfully completed {NUM_TRAINS-1}th training of routing agent, took {time.time() - training_start_time} seconds")
            return saved_plot_path
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            logger.error(f"Error training model: {str(e)}")
            logger.error(f"Traceback: {error_traceback}")
            assert False
    else:
        logger.info(f"Not enough training data available (TOTAL_NUM_DATA: {TOTAL_NUM_DATA}), skipping training")
        assert False


def create_test_data_from_processed_csv(processed_csv_file):
    """
    Create test data by generating mock log messages from processed CSV samples.
    
    Args:
        processed_csv_file: Path to processed CSV file
        
    Returns:
        list: Test data samples with mock log messages
    """
    import pandas as pd
    import json
    
    test_data = []
    
    # Load processed data to create test set
    df = pd.read_csv(processed_csv_file)
    
    # Use last 10% or 10 samples as test data
    test_size = min(10, max(1, int(len(df) * 0.1)))
    test_df = df.tail(test_size)
    
    # Extract actual pod IDs from the processed data
    all_pod_ids = sorted(df['selected_pod'].unique())
    logger.info(f"Using pod IDs from processed data: {all_pod_ids}")
    
    for _, row in test_df.iterrows():
        # Reconstruct a proper log message for testing (minimal but valid format)
        request_id = row.get('request_id', f"test_{row.name}")
        
        # Mock pod metrics with realistic values for all detected pods
        kv_cache_ratios = {pod: 0.1 for pod in all_pod_ids}
        inflight_requests = {pod: 2 for pod in all_pod_ids}
        gpu_cache_usage = {pod: 0.15 for pod in all_pod_ids}
        cpu_cache_usage = {pod: 0.0 for pod in all_pod_ids}
        running_requests = {pod: 2 for pod in all_pod_ids}
        waiting_requests = {pod: 0 for pod in all_pod_ids}
        prefill_tokens = {pod: 5000 for pod in all_pod_ids}
        decode_tokens = {pod: 50000 for pod in all_pod_ids}
        
        # Convert to JSON strings
        kv_cache_json = json.dumps(kv_cache_ratios)
        inflight_json = json.dumps(inflight_requests)
        gpu_cache_json = json.dumps(gpu_cache_usage)
        cpu_cache_json = json.dumps(cpu_cache_usage)
        running_json = json.dumps(running_requests)
        waiting_json = json.dumps(waiting_requests)
        prefill_json = json.dumps(prefill_tokens)
        decode_json = json.dumps(decode_tokens)
        
        mock_log_message = (
            f"**@latency_metrics@requestID@{request_id}@"
            f"request_start_time@1748656367682891@request_end_time@1748656374420081@"
            f"selectedpod@{row['selected_pod']}@ttft@{row['ttft']}@avg_tpot@{row['avg_tpot']}@"
            f"total_decode_time@{row['ttft'] + row['avg_tpot']}@e2e@{row.get('e2e_latency', row['ttft'] + row['avg_tpot'])}@"
            f"numInputTokens@{row.get('input_tokens', 4000)}@numOutputTokens@{row.get('output_tokens', 100)}@"
            f"numTotalTokens@{row.get('total_tokens', 4100)}@"
            f"allPodsKvCacheHitRatios@{kv_cache_json}@numInflightRequestsAllPods@{inflight_json}@"
            f"vllmGPUKVCacheUsage@{gpu_cache_json}@vllmCPUKVCacheUsage@{cpu_cache_json}@"
            f"vllmNumRequestsRunning@{running_json}@vllmNumRequestsWaiting@{waiting_json}@"
            f"podMetricsLastSecond@{{}}@numPrefillTokensForAllPods@{prefill_json}@"
            f"numDecodeTokensForAllPods@{decode_json}@numTrains@1"
        )
        test_data.append({
            "request_id": request_id,
            "message": mock_log_message
        })
    
    logger.info(f"Created {len(test_data)} test samples from processed data")
    return test_data


def run_test_inference_phase(args, test_data):
    """
    Run the test inference phase with the provided test data.
    
    Args:
        args: Command line arguments
        test_data: List of test samples with mock log messages
    """
    logger.info("=== STARTING TESTING PHASE ===")
    success_count = 0
    match_count = 0
    mismatch_count = 0
    unknown_original_count = 0
    test_count = 10
    
    for td in test_data:
        log_message = td['message']
        request_id = td['request_id']
        result = test_inference(args, log_message, request_id, args.final_model_dir)
        
        print()
        print(f"Request_id: {request_id}, Selected Pod: {result['selected_pod']}")
        print(f"pod_probabilities_list: ", end="")
        for prob in result['pod_probabilities']:
            print(f"{prob:.2f}", end=", ")
        print()
        
        # Print latency predictions if available
        if 'predicted_latencies' in result and result['predicted_latencies']:
            if isinstance(result['predicted_latencies'], dict):
                print(f"predicted_latencies: {result['predicted_latencies']}")
            print(f"chosen_pod_predicted_latency: {result.get('chosen_pod_predicted_latency', 'N/A')}")
        print()
        
        if result:
            success_count += 1
            if result['prediction_matches'] is True:
                match_count += 1
            elif result['prediction_matches'] is False:
                mismatch_count += 1
            else:
                unknown_original_count += 1
                
            # Log detailed results
            if result['original_pod_choice']:
                match_status = "MATCH" if result['prediction_matches'] else "MISMATCH"
                latency_info = f", Predicted Latency: {result.get('chosen_pod_predicted_latency', 'N/A')}" if result.get('chosen_pod_predicted_latency', -1) != -1 else ""
                logger.info(f"  → Predicted: {result['selected_pod']}, Original: {result['original_pod_choice']}, Status: {match_status}, Confidence: {result['confidence']:.3f}{latency_info}")
            else:
                latency_info = f", Predicted Latency: {result.get('chosen_pod_predicted_latency', 'N/A')}" if result.get('chosen_pod_predicted_latency', -1) != -1 else ""
                logger.info(f"  → Predicted: {result['selected_pod']}, Original: UNKNOWN, Confidence: {result['confidence']:.3f}{latency_info}")
        else:
            logger.error(f"✗ Failed inference for {request_id}")
                
        test_count -= 1
        if test_count <= 0:
            break
            
    logger.info(f"Testing complete: {success_count} successful inferences")
    logger.info(f"Prediction matches: {match_count}, mismatches: {mismatch_count}, unknown: {unknown_original_count}")


def test_inference(args, log_message, request_id, final_model_dir):
    global NUM_TRAINS, MODEL_UPDATED, stats_instance, RL_MODEL_HYPERPARAMETERS
    utils.set_all_seeds(42)
    if NUM_TRAINS == 0:
        logger.warning("No trained model available, please train first")
        return None
    
    # CRITICAL FIX: Load hyperparameters from trained model to match training configuration
    # This ensures inference uses the EXACT same hyperparameters as training
    config_path = f"{final_model_dir}/model_config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            import json
            model_config = json.load(f)
        # Update RL_MODEL_HYPERPARAMETERS with values from the trained model
        RL_MODEL_HYPERPARAMETERS.update(model_config)
        logger.info(f"Updated hyperparameters from trained model: {config_path}")
    else:
        logger.warning(f"Model config not found at {config_path}, using static hyperparameters")
    
    handle_infer_start_time = time.time()
    
    # CRITICAL FIX: Use exactly the same preprocessing as routing_agent_service.py for consistency
    # Step 1: Replace pod IPs with generalpodid format (pod_0000, pod_0001, etc.) if needed
    # This converts IP-based columns to pod_xxxx- format that encoding expects
    pod_ips = utils.extract_pod_ips_from_content(log_message)
    if pod_ips:
        log_message = utils.replace_pod_ip_with_generalpodid(log_message)
        logger.info(f"Replaced {len(pod_ips)} pod IPs with generalpodid format")
    else:
        logger.info("No pod IPs found in log message - likely already in pod_xxxx format")
    
    # Step 2: Preprocess using the same path as production
    processed_df, sorted_all_pod_ids, _ = preprocess.main(None, log_message, RL_MODEL_HYPERPARAMETERS)
    
    preprocess_overhead = time.time() - handle_infer_start_time
    original_pod_choice = processed_df['selected_pod'].iloc[0] if len(processed_df) > 0 else None
    
    ## new way
    normalizable_features, non_normalizable_features = data_normalizer._get_normalizable_features(processed_df, RL_MODEL_HYPERPARAMETERS.get('NO_NORMALIZE_FEATURES', []))
    logger.info(f"normalizable_features: {normalizable_features}")
    logger.info(f"non_normalizable_features: {non_normalizable_features}")
    if not stats_instance.feature_stats:
        logger.error(f"request_id,{request_id},No normalization statistics available for inference")
        assert False
    
    # CRITICAL FIX: Add the same feature validation as routing_agent_service.py
    # This excludes last_second_* features that cause dimension mismatch
    non_interest = ['request_id', 'requestID', 'ttft', 'avg_tpot', 'e2e_latency', 'selected_pod', 'request_start_time', 'request_end_time']
    non_interest += non_normalizable_features
    features_must_exist_in_stats_instance = []
    for feature in processed_df.columns:
        # NOTE: ignoring last_second_* features (same as routing_agent_service.py)
        if "last_second_" not in feature and feature not in non_interest and feature in normalizable_features:
            features_must_exist_in_stats_instance.append(feature)
    for feature in features_must_exist_in_stats_instance:
        if feature not in stats_instance.feature_stats:
            logger.error(f"Feature {feature} not found in stats_instance")
            logger.error(f"features_must_exist_in_stats_instance: {features_must_exist_in_stats_instance}")
            # logger.error(f"processed_df.columns: {list(processed_df.columns)}")
            # logger.error(f"Available stats features: {list(stats_instance.feature_stats.keys())}")
            assert False
            
    for feature in normalizable_features:
        stats = stats_instance.feature_stats[feature]
        
        # Check data type and convert if necessary
        feature_value = processed_df[feature].iloc[0]
        if not pd.api.types.is_numeric_dtype(processed_df[feature]):
            logger.warning(f"request_id,{request_id},Feature {feature} is not numeric (type: {type(feature_value)}), attempting conversion")
            try:
                processed_df[feature] = pd.to_numeric(processed_df[feature], errors='coerce')
                feature_value = processed_df[feature].iloc[0]
                if pd.isna(feature_value):
                    logger.error(f"request_id,{request_id},Feature {feature} could not be converted to numeric")
                    continue
            except Exception as e:
                logger.error(f"request_id,{request_id},Failed to convert feature {feature} to numeric: {e}")
                continue
        
        try:
            logger.info(f"before normalize, {feature}: value={feature_value:.2f} → Has stats: count={stats.count}, min={stats.min}, max={stats.max}, mean={stats.mean.item():.2f}, std={stats.std.item():.2f}")
        except Exception as e:
            logger.error(f"request_id,{request_id},Feature {feature} formatting error: {e}")
            continue
            
        data_normalizer._normalize_single_feature(processed_df, feature, stats_instance, is_training=False, request_id=request_id)
        
        try:
            normalized_value = processed_df[feature].iloc[0]
            logger.info(f"after normalize, {feature}: value={normalized_value:.2f} → Has stats: count={stats.count}, min={stats.min}, max={stats.max}, mean={stats.mean.item():.2f}, std={stats.std.item():.2f}")
        except Exception as e:
            logger.error(f"request_id,{request_id},Feature {feature} post-normalization formatting error: {e}")
            continue
        if feature in stats_instance.CONFIG.get("FEATURES_AMPLIFIED", set()):
            if feature in processed_df.columns:
                processed_df[feature] = processed_df[feature] * stats_instance.CONFIG['SIGNAL_AMPLIFICATION_DEGREE']
                logger.info(f"request_id,{request_id},Amplified critical feature {feature} after normalization")
            else:
                logger.error(f"request_id,{request_id},Feature {feature} not found in DataFrame for amplification")
                assert False
    ## old way
    # normalized_df = feature_normalization.normalize_features_for_inference(processed_df, stats_instance, request_id)
    
    # CRITICAL FIX: Add missing hyperparameters that routing_agent_service.py includes
    # These affect how encode_for_inference processes the data and may fix the dimension mismatch
    if 'pod_ip_to_gpu_model_encoded' not in RL_MODEL_HYPERPARAMETERS:
        # Create dummy mappings for offline inference (same as production would have)
        pod_ip_to_generalpodid = {pod_id: pod_id for pod_id in sorted_all_pod_ids}  # Identity mapping for offline
        generalpodid_to_pod_ip = {pod_id: pod_id for pod_id in sorted_all_pod_ids}
        
        # Create simple GPU model mappings (alternating between two model types)
        pod_ip_to_gpu_model = {pod_id: f"model_type_{i % 2}" for i, pod_id in enumerate(sorted_all_pod_ids)}
        pod_ip_to_gpu_model_encoded = {pod_id: i % 2 for i, pod_id in enumerate(sorted_all_pod_ids)}
        generalpodid_to_gpu_model = pod_ip_to_gpu_model  # Same for offline
        
        # Add all missing hyperparameters that routing_agent_service.py adds
        RL_MODEL_HYPERPARAMETERS['pod_ip_to_generalpodid'] = pod_ip_to_generalpodid
        RL_MODEL_HYPERPARAMETERS['generalpodid_to_pod_ip'] = generalpodid_to_pod_ip
        RL_MODEL_HYPERPARAMETERS['sorted_running_pod_ips'] = sorted_all_pod_ids
        RL_MODEL_HYPERPARAMETERS['pod_ip_to_gpu_model'] = pod_ip_to_gpu_model
        RL_MODEL_HYPERPARAMETERS['pod_ip_to_gpu_model_encoded'] = pod_ip_to_gpu_model_encoded
        RL_MODEL_HYPERPARAMETERS['generalpodid_to_gpu_model'] = generalpodid_to_gpu_model
        
        logger.info(f"Added missing hyperparameters for offline inference: {len(sorted_all_pod_ids)} pods")
    
    encode_start_time = time.time()
    tensor_dataset, _ = encoding.encode_for_inference(sorted_all_pod_ids, processed_df, request_features_train, RL_MODEL_HYPERPARAMETERS)
    handle_infer_total_total_encoding_overhead = time.time() - encode_start_time
    infer_from_tensor_start_time = time.time()
    
    # Select model type for inference
    model_type = RL_MODEL_HYPERPARAMETERS.get('MODEL_TYPE', 'contextual_bandit')
    
    if model_type == 'latency_predictor':
        result, _ = latency_predictor.infer_latency_predictor(
            tensor_data=tensor_dataset, 
            request_id=request_id,
            model_updated=MODEL_UPDATED,
            HYPERPARAMETERS=RL_MODEL_HYPERPARAMETERS,
            final_model_dir=args.final_model_dir,
            sorted_all_pod_ids=sorted_all_pod_ids,
        )
    else:
        result, _ = simpler_contextual_bandit.infer_from_tensor(
            tensor_data=tensor_dataset, 
            request_id=request_id,
            model_updated=MODEL_UPDATED,
            HYPERPARAMETERS=RL_MODEL_HYPERPARAMETERS,
            final_model_dir=args.final_model_dir,
        )
        result['predicted_latencies'] = {pod_id: -1 for pod_id in sorted_all_pod_ids}
        result['chosen_pod_predicted_latency'] = -1
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
        "predicted_latencies": result.get('predicted_latencies', {pod_id: -1 for pod_id in sorted_all_pod_ids}),
        "chosen_pod_predicted_latency": result.get('chosen_pod_predicted_latency', -1),
        "total_inference_time_ms": handle_infer_total_overhead * 1000,
        "preprocess_time_ms": preprocess_overhead * 1000,
        "encoding_time_ms": handle_infer_total_total_encoding_overhead * 1000,
        "inference_time_ms": handle_infer_total_total_infer_from_tensor_overhead * 1000,
    }
    
    # Enhanced logging with match/mismatch status
    if prediction_matches:
        match_status = "original routing == model routing"
    else:
        match_status = "original routing != model routing"
    
    latency_info = ""
    if result.get('chosen_pod_predicted_latency', -1) != -1:
        latency_info = f", predicted_latency={result['chosen_pod_predicted_latency']:.2f}"
    
    logger.info(f"Inference result: predicted={selected_pod}, original={original_pod_choice}, {match_status}, confidence={result['confidence']:.4f}{latency_info}")

    return result_summary

def normalize_and_encode_training_data(args, processed_csv_file, stats_instance, ENCODED_DATA_DIR):
    global NUM_TRAINS, MODEL_UPDATED, TRAINING_DATA_UPDATED, TOTAL_NUM_DATA
    flush_start_time = time.time()
    
    if not os.path.exists(processed_csv_file):
        logger.error(f"Processed CSV file not found: {processed_csv_file}")
        assert False
    
    logger.info(f"Loading processed training data from: {processed_csv_file}")
    
    # Load processed data (contains raw values)
    processed_df = pd.read_csv(processed_csv_file)
    logger.info(f"Loaded {len(processed_df)} samples with {len(processed_df.columns)} columns")
    
    # Extract sorted_all_pod_ids from the processed dataframe column names
    sorted_all_pod_ids = utils.get_sorted_all_pod_ids('processed_csv_columns', processed_df.columns.tolist())
    logger.info(f"Extracted {len(sorted_all_pod_ids)} pod IDs: {sorted_all_pod_ids}")
    
    # Apply normalization using the new data_normalizer module
    normalized_df, updated_stats_instance, summary = data_normalizer.normalize_processed_data(
        processed_csv_file,
        output_csv_file=None,  # Don't save, just return normalized data
        reward_function=RL_MODEL_HYPERPARAMETERS['REWARD_FUNCTION'],
        stats_file=None,  # Don't save stats yet
        hyperparameters=RL_MODEL_HYPERPARAMETERS
    )
    
    # Update the stats instance
    stats_instance.feature_stats = updated_stats_instance.feature_stats
    stats_instance.CONFIG = updated_stats_instance.CONFIG
    
    logger.info(f"Normalization complete: {summary['num_features_normalized']} features normalized")
    logger.info(f"Reward function used: {summary['reward_function']}")
    logger.info(f"Processing time: {summary['processing_time']:.2f} seconds")
    # encoding (use normalized data for training)
    ts_encode = time.time()
    encoded_data_output_dir = f"{ENCODED_DATA_DIR}/batch_1"
    encoding.encode_for_train(sorted_all_pod_ids, normalized_df, encoded_data_output_dir, request_features_train, RL_MODEL_HYPERPARAMETERS)
    logger.info(f"Successfully encoded data to {encoded_data_output_dir}, took {time.time() - ts_encode} seconds")

    # Verify encoded data
    expected_tensor_path = f"{encoded_data_output_dir}/tensor_dataset.pt"
    train_tensor_path = f"{encoded_data_output_dir}/train/tensor_dataset.pt"
    if os.path.exists(expected_tensor_path):
        logger.info(f"✓ Found tensor dataset at: {expected_tensor_path}")
    elif os.path.exists(train_tensor_path):
        logger.info(f"✓ Found tensor dataset at: {train_tensor_path}")
    TRAINING_DATA_UPDATED = True
    data_count = len(normalized_df)
    TOTAL_NUM_DATA += data_count
    logger.info(f"Successfully processed {data_count} log messages, took {time.time() - flush_start_time} seconds")
    return True


# NOTE: ensure_deterministic_data_split function removed - no longer needed with new pipeline
# Data splitting is now handled by working directly with processed CSV


# Fixed verification function - remove unused variables
def verify_training_determinism(encoded_data_dir, model_output_dir, HYPERPARAMETERS):
    """Verify that training produces identical results across runs"""
    logger.info("🔍 VERIFYING TRAINING DETERMINISM")
    
    # Train model twice with same settings
    model_type = HYPERPARAMETERS.get('MODEL_TYPE', 'contextual_bandit')
    
    logger.info("Training model #1...")
    utils.set_all_seeds(HYPERPARAMETERS['training_seed'])
    if model_type == 'rl_contextual_bandit_sb3':
        import rl_contextual_bandit_sb3
        saved_plot_path = rl_contextual_bandit_sb3.train(encoded_data_dir, f"{model_output_dir}_test1", HYPERPARAMETERS)
    else:
        saved_plot_path = simpler_contextual_bandit.train(encoded_data_dir, f"{model_output_dir}_test1", HYPERPARAMETERS)
    
    logger.info("Training model #2...")
    utils.set_all_seeds(HYPERPARAMETERS['training_seed'])
    if model_type == 'rl_contextual_bandit_sb3':
        import rl_contextual_bandit_sb3
        saved_plot_path = rl_contextual_bandit_sb3.train(encoded_data_dir, f"{model_output_dir}_test2", HYPERPARAMETERS)
    else:
        saved_plot_path = simpler_contextual_bandit.train(encoded_data_dir, f"{model_output_dir}_test2", HYPERPARAMETERS)
    
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
    parser.add_argument('processed_csv', help='Processed CSV file containing training data with raw values')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='Train/test split ratio')
    parser.add_argument('--analyze_behavior', action='store_true', help='Analyze what the model has learned through feature sensitivity tests')
    parser.add_argument('--final_model_dir', type=str, default=None, help='Final model directory')
    
    # Only this argument is required; JSON is the source of truth. Other flags are optional overrides.
    parser.add_argument('--hyperparameters', type=str, required=True, help='Path to JSON hyperparameter file (single source of truth)')
    args = parser.parse_args()
    
    # 1) Load JSON hyperparameters (required)
    if not os.path.exists(args.hyperparameters):
        logger.error(f"Hyperparameters JSON not found: {args.hyperparameters}")
        return
    try:
        with open(args.hyperparameters, 'r') as f:
            hp = json.load(f)
        if not isinstance(hp, dict):
            logger.error(f"Hyperparameters file is not a JSON object: {args.hyperparameters}")
            assert False
        RL_MODEL_HYPERPARAMETERS.clear()
        RL_MODEL_HYPERPARAMETERS.update(hp)
        logger.info(f"Loaded hyperparameters from {args.hyperparameters}")
    except Exception as e:
        logger.error(f"Failed to read hyperparameters from {args.hyperparameters}: {e}")
        assert False

    logger.info(f"EXCLUDED_POD_FEATURES: {RL_MODEL_HYPERPARAMETERS.get('EXCLUDED_POD_FEATURES', [])}")
    
    # Validate processed CSV file
    if not os.path.exists(args.processed_csv):
        logger.error(f"Processed CSV file not found: {args.processed_csv}")
        return
    
    # Validate the processed CSV format
    import data_processor
    validation = data_processor.validate_processed_csv(args.processed_csv)
    if not validation['valid']:
        logger.error(f"Invalid processed CSV: {validation['error']}")
        return
    
    logger.info(f"✓ Using processed CSV: {args.processed_csv}")
    logger.info(f"  Samples: {validation['num_samples']}, Pods: {validation['num_pods']}")
    logger.info(f"  TTFT range: {validation['ttft_range']}, TPOT range: {validation['avg_tpot_range']}")
    
    data_dir = os.path.dirname(args.processed_csv)

    feature_normalization_stats_file = f"{args.final_model_dir}/feature_normalization_statistics.csv"
    
    if stats_instance is not None:
        logger.error("Using existing stats instance for normalization")
        assert False
    
    # Load processed CSV to get feature information
    processed_df = pd.read_csv(args.processed_csv)
    
    # Get normalizable features and create stats instance
    normalizable_features, non_normalizable_features = data_normalizer._get_normalizable_features(processed_df, RL_MODEL_HYPERPARAMETERS.get('NO_NORMALIZE_FEATURES', []))
    stats_instance = data_normalizer.FeatureStats(normalizable_features)

    ENCODED_DATA_DIR = f"{args.final_model_dir}/encoded_data"
    if not os.path.exists(ENCODED_DATA_DIR):
        os.makedirs(ENCODED_DATA_DIR)
    if os.path.exists(ENCODED_DATA_DIR):
        shutil.rmtree(ENCODED_DATA_DIR)
        os.makedirs(ENCODED_DATA_DIR)
        logger.info(f"Cleaned and recreated {ENCODED_DATA_DIR} for fresh offline training")
    
    # Process training data using the new simplified approach
    normalize_and_encode_training_data(args, args.processed_csv, stats_instance, ENCODED_DATA_DIR)

    # Save stats using CSV format
    stats_file = f"{args.final_model_dir}/feature_normalization_statistics.csv"
    stats_instance.write_stats_to_file(stats_file)
    logger.info(f"Saved normalization statistics to: {stats_file}")
    
    model_and_data_analysis_helper.diagnose_training_data_issues(ENCODED_DATA_DIR)
    is_online_learning = False
    saved_plot_path = train_model(ENCODED_DATA_DIR, is_online_learning, args.final_model_dir)

    # NEW: Behavior Analysis (before regular testing)
    test_data = create_test_data_from_processed_csv(args.processed_csv)
    # if args.analyze_behavior and test_data and len(test_data) > 0:
    logger.info("=== STARTING BEHAVIOR ANALYSIS ===")
    # model_and_data_analysis_helper.analyze_model_behavior(args, test_data, feature_normalization_stats_file)
    _ = model_and_data_analysis_helper.analyze_detailed_feature_sensitivity(args, test_data, feature_normalization_stats_file)
    logger.info("=== BEHAVIOR ANALYSIS COMPLETED ===")
    
    
    # Run test inference if we have test data
    # if test_data and len(test_data) > 0:
    run_test_inference_phase(args, test_data)
        
        
    print(f"** saved_plot_path: {saved_plot_path}")
    print(f"** final_model_dir: {args.final_model_dir}")
    
if __name__ == "__main__":
    main()