# offline_routing_agent.py

import os
import time
import encoding
import simpler_contextual_bandit
import preprocess
import threading
import argparse
import random_forest
import torch
import feature_normalization
import model_and_data_analysis_helper
import shutil
import re
import utils as utils
import random
from logger import logger, INCLUDE_GPU_IN_FEATURE
import pandas as pd

utils.set_all_seeds(42)

RL_MODEL_HYPERPARAMETERS = {
    'model_type': 'simplified',
    'hidden_dim': 32, # 256,
    'batch_size': 32,
    'learning_rate': 0.01, # 0.01, 0.005
    'training_epochs': 5, # 5,
    'learning_every_x_iter': 5,
    'weight_decay': 0.0001,
    'max_updates_per_epoch': 1000, # 1000000000
    'exploration_rate': 0.1, # 0.1
    'explore': True,
    'weight_initialization': 'xavier', # 'kaiming', 'xavier', 'static'
    
    
    'eval_interval': 10,
    'entropy_bonus_factor': 0.01,
    'per_learn_reward_normalization': False,
    'normalization': {
        "SIGNAL_AMPLIFICATION_DEGREE": 1.0,  # 1.5
        "REWARD_AMPLIFICATION_DEGREE": 1.0,
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
    'REWARD_FUNCTION': 'linear_simple',
}

# Global variables (simplified for offline use)
NUM_TRAINS = 0
MODEL_UPDATED = False
TRAINING_DATA_UPDATED = False
POD_LABEL_SELECTOR="model.aibrix.ai/name=llama-3-8b-instruct"
TOTAL_NUM_DATA = 0
MIN_NUM_TRAINING_DATA = 500
LOCK_TRAINING_DATA = threading.Lock()
stats_instance = None
request_features_train = ['input_tokens', 'output_tokens', 'total_tokens']
# request_features_reward = ['ttft', 'avg_tpot', 'e2e_latency']

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

def train_model(ENCODED_DATA_DIR, is_online_learning, final_model_dir, training_data_filename=None):
    global NUM_TRAINS, MODEL_UPDATED, TRAINING_DATA_UPDATED, TOTAL_NUM_DATA 
    if TRAINING_DATA_UPDATED and TOTAL_NUM_DATA > MIN_NUM_TRAINING_DATA:
        training_start_time = time.time()
        logger.info(f"Starting {NUM_TRAINS}th training of routing agent")
        try:
            utils.set_all_seeds(RL_MODEL_HYPERPARAMETERS['training_seed'])
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
    processed_df, _, sorted_all_pod_ids, _ = preprocess.main(None, log_message, args.ttft_slo, args.avg_tpot_slo, RL_MODEL_HYPERPARAMETERS, POD_LABEL_SELECTOR)
    
    preprocess_overhead = time.time() - handle_infer_start_time
    original_pod_choice = processed_df['selected_pod'].iloc[0] if len(processed_df) > 0 else None
    
    # CRITICAL FIX: Add the same feature validation as routing_agent_service.py
    # This excludes last_second_* features that cause dimension mismatch
    non_interest = ['request_id', 'requestID', 'ttft', 'avg_tpot', 'e2e_latency', 'selected_pod']
    features_must_exist_in_stats_instance = []
    for feature in processed_df.columns:
        # NOTE: ignoring last_second_* features (same as routing_agent_service.py)
        if "last_second_" not in feature and feature not in non_interest:
            features_must_exist_in_stats_instance.append(feature)
    for feature in features_must_exist_in_stats_instance:
        if feature not in stats_instance.feature_stats:
            logger.error(f"Feature {feature} not found in stats_instance")
            logger.error(f"processed_df.columns: {list(processed_df.columns)}")
            logger.error(f"features_must_exist_in_stats_instance: {features_must_exist_in_stats_instance}")
            logger.error(f"Available stats features: {list(stats_instance.feature_stats.keys())}")
            assert False
    
    ## new way
    normalizable_features, non_normalizable_features = feature_normalization._get_normalizable_features(processed_df)
    logger.info(f"normalizable_features: {normalizable_features}")
    logger.info(f"non_normalizable_features: {non_normalizable_features}")
    if stats_instance.count == 0:
        logger.error(f"request_id,{request_id},No normalization statistics available for inference")
        assert False
    for feature in normalizable_features:
        stats = stats_instance.feature_stats[feature]
        logger.info(f"before normalize, {feature}: value={processed_df[feature].iloc[0]:.2f} → Has stats: count={stats.count}, min={stats.min}, max={stats.max}, mean={stats.mean.item():.2f}, std={stats.std.item():.2f}")
        feature_normalization._normalize_single_feature(processed_df, feature, stats_instance, is_training=False, request_id=request_id)
        logger.info(f"after normalize, {feature}: value={processed_df[feature].iloc[0]:.2f} → Has stats: count={stats.count}, min={stats.min}, max={stats.max}, mean={stats.mean.item():.2f}, std={stats.std.item():.2f}")
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
    if args.model == "random_forest":
        result, _ = random_forest.infer_from_tensor(
            tensor_data=tensor_dataset, 
            exploration_enabled=True, 
            exploration_rate=RL_MODEL_HYPERPARAMETERS['exploration_rate'], 
            model_updated=MODEL_UPDATED,
    )
    elif args.model == "simpler_contextual_bandit":
        result, _ = simpler_contextual_bandit.infer_from_tensor(
            tensor_data=tensor_dataset, 
            request_id=request_id,
            model_updated=MODEL_UPDATED,
            HYPERPARAMETERS=RL_MODEL_HYPERPARAMETERS,
            final_model_dir=args.final_model_dir,
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
    if prediction_matches:
        match_status = "original routing == model routing"
    else:
        match_status = "original routing != model routing"
    logger.info(f"Inference result: predicted={selected_pod}, original={original_pod_choice}, {match_status}, confidence={result['confidence']:.4f}")

    return result_summary

def process_training_data(args, data_dir, train_data, stats_instance, ENCODED_DATA_DIR, already_processed_df=None):
    global NUM_TRAINS, MODEL_UPDATED, TRAINING_DATA_UPDATED, TOTAL_NUM_DATA
    flush_start_time = time.time()
    if not train_data and already_processed_df is None:
        logger.error("None of training data provided. We need either train_data or already_processed_df.")
        assert False
    if train_data is not None:
        logger.info(f"Processing training data with {len(train_data)} entries")
        if not os.path.exists("temp_training_data"):
            os.mkdir("temp_training_data")
        raw_data = "temp_training_data/offline_batch.csv"
        utils.write_to_file(train_data, raw_data)
        
        ts_preprocess = time.time()
        processed_df, _, sorted_all_pod_ids, _ = preprocess.main(raw_data, "", args.ttft_slo, args.avg_tpot_slo, RL_MODEL_HYPERPARAMETERS, POD_LABEL_SELECTOR)
        processed_df.to_csv(f"{data_dir}/processed_data.csv", index=False)
        logger.info(f"Successfully parsed data, took {time.time() - ts_preprocess} seconds")
        
        # update_stats_incrementally is called inside normalize_features_for_training
        processed_df = feature_normalization.normalize_features_for_training(processed_df, stats_instance)
        # processed_df = feature_normalization.try_reward_amplification(processed_df)
        processed_df.to_csv(f"{data_dir}/processed_and_normalized_data.csv", index=False)
    else:
        if len(already_processed_df) != 0:
            processed_df = already_processed_df
            # Extract sorted_all_pod_ids from the already processed dataframe column names
            sorted_all_pod_ids = utils.get_sorted_all_pod_ids('processed_csv_columns', processed_df.columns.tolist())
            
            # BUGFIX: When using already processed data, we need to compute stats for inference
            # The data is already normalized, so we compute stats from the normalized data
            logger.info("Computing feature statistics from already processed/normalized data for inference")
            feature_normalization.compute_stats_from_normalized_data(processed_df, stats_instance)
        else:
            logger.error("We are using already_processed_df but already_processed_df is empty. Check already_processed_csv file.")
            assert False
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
    data_count = len(train_data) if train_data is not None else len(processed_df)
    TOTAL_NUM_DATA += data_count
    logger.info(f"Successfully processed {data_count} log messages, took {time.time() - flush_start_time} seconds")
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
    print(f"First test message hash: {utils.static_hash(test_messages[0]) if test_messages else 'None'}")
    
    return train_messages, test_messages


# Fixed verification function - remove unused variables
def verify_training_determinism(encoded_data_dir, model_output_dir, HYPERPARAMETERS):
    """Verify that training produces identical results across runs"""
    logger.info("🔍 VERIFYING TRAINING DETERMINISM")
    
    # Train model twice with same settings
    logger.info("Training model #1...")
    utils.set_all_seeds(HYPERPARAMETERS['training_seed'])
    saved_plot_path = simpler_contextual_bandit.train(encoded_data_dir, f"{model_output_dir}_test1", HYPERPARAMETERS)
    
    logger.info("Training model #2...")
    utils.set_all_seeds(HYPERPARAMETERS['training_seed'])
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
    parser.add_argument('data_file', help='CSV file containing log messages for training')
    parser.add_argument('--already_processed_csv', type=str, default=None, help='CSV file containing processed log messages for training')
    parser.add_argument('--skip_training', action='store_true', help='Skip training and only do inference')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='Train/test split ratio')
    parser.add_argument('--model', choices=['random_forest', 'simpler_contextual_bandit'], default='simpler_contextual_bandit', help='Model type to use for training (default: simpler_contextual_bandit)')
    parser.add_argument('--ttft_slo', type=float, help='TTFT SLO threshold for preprocessing', default=1000)
    parser.add_argument('--avg_tpot_slo', type=float, help='Average TPOT SLO threshold for preprocessing', default=50)
    parser.add_argument('--analyze_behavior', action='store_true', help='Analyze what the model has learned through feature sensitivity tests')
    
    parser.add_argument('--final_model_dir', type=str, default=None, help='Final model directory')
    
    args = parser.parse_args()
    
    if args.already_processed_csv and args.already_processed_csv != "none":
        if not os.path.exists(args.already_processed_csv):
            logger.error(f"Already processed CSV file {args.already_processed_csv} not found")
            assert False
        data_dir = os.path.dirname(args.already_processed_csv)
        train_data = None
        test_data = None
    else:
        if not os.path.exists(args.data_file):
            logger.error(f"Data file {args.data_file} not found")
            assert False
        data_dir = os.path.dirname(args.data_file)
        if 'replaced' not in args.data_file:
            logger.info(f"Data file {args.data_file} contains raw pod IPs, replacing with GeneralPodID")
            args.data_file = utils.replace_pod_ip_with_generalpodid(args.data_file)
            
        all_data = {}
        if os.path.isfile(args.data_file):
            data_dir = os.path.dirname(args.data_file)
            logger.info(f"data_file is a file: {args.data_file}")
            all_data = read_csv_data(args.data_file)

        if all_data is None or len(all_data) == 0:
            logger.error("Failed to read data or no valid log messages found")
            return
        
        train_messages, test_messages = ensure_deterministic_data_split(all_data, args.split_ratio)
        test_messages = test_messages[:10]
        train_data = {f"request_{i}": msg for i, msg in enumerate(train_messages)}
        def extract_request_id(log_message):
            match = re.search(r'@requestID@([^@]+)@', log_message)
            return match.group(1) if match else None
        
        test_data = []
        for msg in test_messages:
            test_data.append({"request_id": extract_request_id(msg), "message": msg})

    feature_normalization_stats_file = f"{args.final_model_dir}/feature_normalization_statistics.csv"
    
    if stats_instance is not None:
        logger.error("Using existing stats instance for normalization")
        assert False
        
    stats_instance = feature_normalization.get_stats_instance(RL_MODEL_HYPERPARAMETERS['normalization'], None)

    ENCODED_DATA_DIR = "encoded_data"
    if not os.path.exists(ENCODED_DATA_DIR):
        os.makedirs(ENCODED_DATA_DIR)
    if os.path.exists(ENCODED_DATA_DIR):
        shutil.rmtree(ENCODED_DATA_DIR)
        os.makedirs(ENCODED_DATA_DIR)
        logger.info(f"Cleaned and recreated {ENCODED_DATA_DIR} for fresh offline training")
    already_processed_df = pd.read_csv(args.already_processed_csv) if args.already_processed_csv and args.already_processed_csv != "none" else None
    process_training_data(args, data_dir, train_data, stats_instance, ENCODED_DATA_DIR, already_processed_df)

    stats_instance.write_stats_to_file(feature_normalization_stats_file)
    
    model_and_data_analysis_helper.diagnose_training_data_issues(ENCODED_DATA_DIR)
    
    # Pass training data filename for model directory naming without modifying hyperparameters
    training_data_filename = None
    if args.data_file:
        training_data_filename = os.path.basename(args.data_file).replace('.csv', '')
    elif args.already_processed_csv:
        training_data_filename = os.path.basename(args.already_processed_csv).replace('.csv', '')
    
    is_online_learning = False
        
    saved_plot_path = train_model(ENCODED_DATA_DIR, is_online_learning, args.final_model_dir, training_data_filename)

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
            result = test_inference(args, log_message, request_id, args.final_model_dir)
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
        
        
    print(f"** saved_plot_path: {saved_plot_path}")
    print(f"** final_model_dir: {args.final_model_dir}")
    
if __name__ == "__main__":
    main()