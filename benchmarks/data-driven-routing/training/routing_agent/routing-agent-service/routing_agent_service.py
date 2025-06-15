## routing_agent_service.py

# import threading
# import joblib
import pandas as pd
import numpy as np
# import uvicorn
# from pydantic import BaseModel
import os
import time
# import asyncio
# from concurrent.futures import ThreadPoolExecutor
import sys
# import concurrent.futures
import encoding
# import sac
import ppo
import contextual_bandit
import simpler_contextual_bandit
from flask import Flask, request, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
from logger import logger
import preprocess
import pickle
import threading
from kubernetes import client, config
import ast
import json
import feature_normalization
import signal
import sys

app = Flask(__name__)

RL_MODEL_HYPERPARAMETERS = None
hyperparameter_file_path = '/app/final_model/model_config.json'

NUM_FLUSH = 0
ENCODED_DATA_DIR = "encoded_data"
final_model_path = "final_model"
feature_normalization_stats_file = f"{final_model_path}/feature_normalization_statistics.csv"  # Add this near the top with your other constants;
NUM_TRAINS = 0
MODEL_UPDATED = True
TRAINING_DATA_UPDATED = False
LOCK_TRAINING_DATA = threading.Lock()
ENABLE_ONLINE_LEARNING = os.getenv("ENABLE_ONLINE_LEARNING", "true").lower() == "true"
MODEL = os.getenv("MODEL", "simpler_contextual_bandit")
TTFT_SLO = int(os.getenv("TTFT_SLO", 1000))
AVG_TPOT_SLO = int(os.getenv("AVG_TPOT_SLO", 50))
first_request_starting_time = None
stats_instance = None 
TOTAL_NUM_DATA = 0
NUM_NEW_DATA = 0
MIN_NUM_TRAINING_DATA = 500  # Minimum number of training data required to trigger training

logger.info(f"TTFT_SLO: {TTFT_SLO}")
logger.info(f"AVG_TPOT_SLO: {AVG_TPOT_SLO}")

def write_to_file(log_data, raw_data):
    with open(raw_data, "w") as log_file:
        for request_id, log_message in log_data.items():
            log_file.write(f"{log_message}\n")
    logger.info(f"Successfully wrote {len(log_data)} entries to {raw_data}")

request_features_train = ['input_tokens', 'output_tokens', 'total_tokens']
# request_features_reward = ['ttft', 'avg_tpot', 'e2e_latency']


# Fixed handle_flush function
@app.route("/flush", methods=["POST"])
def handle_flush():
    global NUM_FLUSH, ENCODED_DATA_DIR, TRAINING_DATA_UPDATED, TOTAL_NUM_DATA, NUM_NEW_DATA, feature_normalization_stats_file, stats_instance
    NUM_FLUSH += 1
    flush_start_time = time.time()
    log_data = request.json
    try:
        logger.info(f"Received log data with {len(log_data) if log_data else 0} entries")
        if not os.path.exists("raw_training_data"):
            os.mkdir("raw_training_data")
        raw_data_path = f"raw_training_data/batch_{NUM_FLUSH}.csv"
        
        # Write raw data to file
        ts_write_raw_data = time.time()
        write_to_file(log_data, raw_data_path)
        logger.info(f"wrote {len(log_data)} entries to {raw_data_path}, took {time.time() - ts_write_raw_data} seconds")

        # Preprocess raw data
        ts_preprocess = time.time()
        processed_df, _, all_pods, _ = preprocess.main(raw_data_path, "", TTFT_SLO, AVG_TPOT_SLO, RL_MODEL_HYPERPARAMETERS)
        logger.info(f"Successfully parsed data, took {time.time() - ts_preprocess} seconds")
        # try:
        #     if stats_instance is None:
        #         stats_instance = feature_normalization.get_stats_instance(RL_MODEL_HYPERPARAMETERS['normalization'], feature_normalization_stats_file)
        # except Exception as e:
        #     logger.error(f"Could not load feature normalization stats: {e}")
        #     assert False
            
        # if ENABLE_ONLINE_LEARNING:
        processed_df = feature_normalization.normalize_features_for_training(processed_df, stats_instance)
        stats_instance.write_stats_to_file(feature_normalization_stats_file)

        # Encode preprocessed data
        ts_encode = time.time()
        encoded_data_subdir = f"{ENCODED_DATA_DIR}/batch_{NUM_FLUSH}"
        encoding.GPU_MAP = RL_MODEL_HYPERPARAMETERS['GPU_MAP']
        encoding.NUM_GPU_TYPES = len(RL_MODEL_HYPERPARAMETERS['GPU_MAP'])
        encoding.encode_for_train(all_pods, processed_df, encoded_data_subdir, request_features_train, RL_MODEL_HYPERPARAMETERS)
        logger.info(f"Successfully encoded data to {encoded_data_subdir}, took {time.time() - ts_encode} seconds")
        logger.info(f"Successfully flushed {len(log_data)} log messages, took {time.time() - flush_start_time} seconds")
        TRAINING_DATA_UPDATED = True
        TOTAL_NUM_DATA += len(log_data)
        NUM_NEW_DATA += len(log_data)
            
        return jsonify({"status": "success", "message": f"Successfully processed {len(log_data)} log messages"}), 200
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Unhandled exception: {str(e)}")
        logger.error(f"Traceback: {error_traceback}")
        return jsonify({"status": "error", "message": str(e), "traceback": error_traceback}), 500


@app.route("/infer", methods=["POST"])
def handle_infer():
    global NUM_TRAINS, MODEL_UPDATED, first_request_starting_time, stats_instance, RL_MODEL_HYPERPARAMETERS
    if first_request_starting_time == None:
        first_request_starting_time = time.time()
        logger.info(f"First request starting time set to {first_request_starting_time}")
    # if NUM_TRAINS == 0:
    #     logger.warning("No trained model available, please call /flush to train first")
    #     return jsonify({"error": "No trained model available, please call /flush to train first"}), 503
    handle_infer_start_time = time.time()
    try:
        # Get the log message as a string from the request body
        prep_start_time = time.time()
        log_data = request.json
        # Handle string input directly
        if isinstance(log_data, str):
            log_message = log_data
            request_id = "default"  # or extract from log_message
        else:
            # Handle dict input (original logic)
            if len(list(log_data.keys())) != 1:
                logger.error(f"There must be only one request for inference, but got {len(list(log_data.keys()))} requests")
                return jsonify({"error": "Invalid request format"}), 400
            
            first_key = list(log_data.keys())[0]
            log_message = log_data[first_key]
        logger.debug(f"Received inference request in handle_infer:\n{log_message}")

        # Extract request ID for logging purposes
        parts = log_message.split("requestID@")
        if len(parts) > 1:
            request_id_parts = parts[1].split("@")
            if request_id_parts:
                request_id = request_id_parts[0]
        handle_infer_total_prep_overhead = time.time() - prep_start_time

        # Use the existing preprocessing function to parse the log
        preprocess_start_time = time.time()
        processed_df, _, all_pods, preprocess_dataset_overhead_summary = preprocess.main(None, log_message, TTFT_SLO, AVG_TPOT_SLO, RL_MODEL_HYPERPARAMETERS)
        logger.info(f"input_tokens: {processed_df['input_tokens'][0]}")
        logger.debug(f"Successfully parsed data for request_{request_id}")
        handle_infer_total_total_preprocess_overhead = time.time() - preprocess_start_time

        # Get running statistics and apply normalization (SAME AS TRAINING)
        get_stat_start_time = time.time()
        # if stats_instance is None:
        #     stats_instance = feature_normalization.get_stats_instance(RL_MODEL_HYPERPARAMETERS['normalization'], feature_normalization_stats_file)
        #     logger.info("🔍 STATS FILE CONTENT DEBUG:")
        #     logger.info(f"Total features in stats file: {len(stats_instance.feature_stats)}")
        #     for feature_name, stats in stats_instance.feature_stats.items():
        #         logger.info(f"  {feature_name}: count={stats.count}")
        if stats_instance is None:
            logger.error(f"No running statistics available, stats_instance: {stats_instance}")
            assert False
        if stats_instance.count == 0:
            logger.warning(f"Stats instance count is 0, no data available for normalization")

        #==================================================================================
        logger.info("🔍 PRE-NORMALIZATION DEBUG:")
        logger.info(f"Stats instance feature count: {len(stats_instance.feature_stats)}")
        logger.debug(f"Available stats features: {list(stats_instance.feature_stats.keys())}")
        for feature in ['input_tokens', 'output_tokens', 'total_tokens']:
            if feature in processed_df.columns:
                logger.info(f"before normalize, {feature}: min={processed_df[feature].min():.4f}, max={processed_df[feature].max():.4f}")
                if feature in stats_instance.feature_stats:
                    stats = stats_instance.feature_stats[feature]
                    mean_val = stats.get_mean()
                    std_val = stats.get_std()
                    # Handle numpy arrays safely
                    if hasattr(mean_val, 'item'):
                        mean_val = mean_val.item()
                    if hasattr(std_val, 'item'):
                        std_val = std_val.item()
                    logger.info(f"before normalize, {feature}: value={processed_df[feature].iloc[0]:.2f} → Has stats: count={stats.count}, mean={mean_val:.4f}, std={std_val:.4f}")
                else:
                    logger.warning(f"before normalize, {feature}: value={processed_df[feature].iloc[0]:.2f} → ❌ NO STATS FOUND for {feature}!")
        #==================================================================================
        processed_df = feature_normalization.normalize_features_for_inference(processed_df, stats_instance, request_id)
        #==================================================================================
        logger.info("🔍 POST-NORMALIZATION DEBUG:")
        logger.info(f"Stats instance feature count: {len(stats_instance.feature_stats)}")
        logger.debug(f"Available stats features: {list(stats_instance.feature_stats.keys())}")
        for feature in ['input_tokens', 'output_tokens', 'total_tokens']:
            if feature in processed_df.columns:
                if feature in stats_instance.feature_stats:
                    stats = stats_instance.feature_stats[feature]
                    mean_val = stats.get_mean()
                    std_val = stats.get_std()
                    # Handle numpy arrays safely
                    if hasattr(mean_val, 'item'):
                        mean_val = mean_val.item()
                    if hasattr(std_val, 'item'):
                        std_val = std_val.item()
                    logger.info(f"after normalize, {feature}: value={processed_df[feature].iloc[0]:.2f} → Has stats: count={stats.count}, mean={mean_val:.4f}, std={std_val:.4f}")
                else:
                    logger.warning(f"after normalize, {feature}: value={processed_df[feature].iloc[0]:.2f} → ❌ NO STATS FOUND for {feature}!")
        #==================================================================================
        handle_infer_total_get_stat_overhead = time.time() - get_stat_start_time

        # Encode data (normalization already done)
        encode_start_time = time.time()
        encoding.GPU_MAP = RL_MODEL_HYPERPARAMETERS['GPU_MAP']
        encoding.NUM_GPU_TYPES = len(RL_MODEL_HYPERPARAMETERS['GPU_MAP'])
        tensor_data, encode_for_inference_overhead_summary = encoding.encode_for_inference(all_pods, processed_df, request_features_train, RL_MODEL_HYPERPARAMETERS)
        logger.debug(f"Successfully encoded data in memory for inference")
        handle_infer_total_total_encoding_overhead = time.time() - encode_start_time

        infer_from_tensor_start_time = time.time()
        if MODEL == "simpler_contextual_bandit":
            result, infer_from_tensor_overhead_summary = simpler_contextual_bandit.infer_from_tensor(tensor_data=tensor_data, model_updated=MODEL_UPDATED, HYPERPARAMETERS=RL_MODEL_HYPERPARAMETERS)
        elif MODEL == "contextual_bandit":
            result, infer_from_tensor_overhead_summary = contextual_bandit.infer_from_tensor(tensor_data=tensor_data, model_updated=MODEL_UPDATED)
        else:
            logger.error(f"Unknown model {MODEL}, please set MODEL environment variable to 'simpler_contextual_bandit' or 'contextual_bandit'")
            return jsonify({"error": f"Unknown model {MODEL}, please set MODEL environment variable to 'simpler_contextual_bandit' or 'contextual_bandit'"}), 500
        if MODEL_UPDATED:
            logger.info("Model updated flag consumed, resetting to False")
            MODEL_UPDATED = False
        handle_infer_total_total_infer_from_tensor_overhead = time.time() - infer_from_tensor_start_time
        result["requestID"] = request_id
        result["num_trains"] = NUM_TRAINS
        result["request_timestamp"] = time.time() - first_request_starting_time
        logger.info(f"Inference result: {result}")
        
        handle_infer_total_wrapup_start_time = time.time()
        # Map the pod index back to the actual pod ID
        selected_pod_index = result.get('selected_pod_index', 0)
        if selected_pod_index >= len(all_pods):
            logger.warning(f"Selected pod index {selected_pod_index} out of range, defaulting to first pod")
            selected_pod_index = 0
            
        selected_pod = all_pods[selected_pod_index]
        handle_infer_total_wrapup_overhead = time.time() - handle_infer_total_wrapup_start_time
        handle_infer_total_overhead = time.time() - handle_infer_start_time
        
        # Return the result
        response = {
            "num_trains": NUM_TRAINS,
            "num_flush": NUM_FLUSH,
            "request_timestamp": time.time() - first_request_starting_time,
            "selected_pod": selected_pod,
            "confidence": result['confidence'],
            "request_id": request_id,
            "* handle_infer_total_prep_overhead": handle_infer_total_prep_overhead*1000,
            "* handle_infer_total_total_preprocess_overhead": handle_infer_total_total_preprocess_overhead*1000,
            "* handle_infer_total_get_stat_overhead": handle_infer_total_get_stat_overhead*1000,
            "* handle_infer_total_total_encoding_overhead": handle_infer_total_total_encoding_overhead*1000,
            "* handle_infer_total_wrapup_overhead": handle_infer_total_wrapup_overhead*1000,
            "* handle_infer_total_total_infer_from_tensor_overhead": handle_infer_total_total_infer_from_tensor_overhead*1000,
            "* handle_infer_total_overhead": handle_infer_total_overhead*1000,
        }

        for key, value in encode_for_inference_overhead_summary.items():
            response[key] = value
        for key, value in preprocess_dataset_overhead_summary.items():
            response[key] = value
        for key, value in infer_from_tensor_overhead_summary.items():
            response[key] = value
        
        logger.debug(f"Selected pod {selected_pod} with confidence {result['confidence']}")
        return jsonify(response), 200
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Error in handle_infer: {str(e)}")
        logger.error(f"Traceback: {error_traceback}")
        return jsonify({"error": str(e), "traceback": error_traceback}), 500


def online_train_routine():
    global NUM_TRAINS, MODEL_UPDATED, TRAINING_DATA_UPDATED, TOTAL_NUM_DATA, final_model_path, NUM_NEW_DATA, RL_MODEL_HYPERPARAMETERS, hyperparameter_file_path

    if  TRAINING_DATA_UPDATED and NUM_NEW_DATA > MIN_NUM_TRAINING_DATA:
        training_start_time = time.time()
        logger.info(f"online_train_routine, train! Starting {NUM_TRAINS}th online training iteration")
        try:
            if MODEL == "simpler_contextual_bandit":
                simpler_contextual_bandit.train(ENCODED_DATA_DIR, final_model_path, HYPERPARAMETERS=RL_MODEL_HYPERPARAMETERS)
            elif MODEL == "contextual_bandit":
                contextual_bandit.train(ENCODED_DATA_DIR)
            else:
                logger.error(f"Unknown model {MODEL}")
                return
        except Exception as e:
            logger.error(f"Error during training: {e}")
            return
        MODEL_UPDATED = True
        TRAINING_DATA_UPDATED = False
        logger.info(f"online_train_routine, Successfully completed {NUM_TRAINS}th online training, took {time.time() - training_start_time} seconds")
        NUM_TRAINS += 1
        NUM_NEW_DATA = 0
    else:
        logger.info(f"online_train_routine, not enough training data available (TOTAL_NUM_DATA: {TOTAL_NUM_DATA}), skipping training")


def load_rl_hyperparameters(file_path):
    if not os.path.exists(file_path):
        logger.error(f"Hyperparameter file {file_path} does not exist")
        assert False
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Convert FEATURES_NORMALIZED from string representation of set to actual set
    features_normalized_str = data['normalization']['FEATURES_NORMALIZED']
    if features_normalized_str and features_normalized_str != "set()":
        # Parse the string representation of the set
        features_normalized = ast.literal_eval(features_normalized_str)
    else:
        features_normalized = set()
    
    # Convert FEATURES_AMPLIFIED from string representation of set to actual set
    features_amplified_str = data['normalization']['FEATURES_AMPLIFIED']
    if features_amplified_str and features_amplified_str != "set()":
        features_amplified = ast.literal_eval(features_amplified_str)
    else:
        features_amplified = set()
    
    rl_model_hyperparameters = {
        'model_type': data['model_type'],
        'hidden_dim': data['hidden_dim'],
        'batch_size': data['batch_size'],
        'lr': data['lr'],
        'weight_decay': data['weight_decay'],
        'exploration_rate': data['exploration_rate'],
        'training_epochs': data['training_epochs'],
        'max_updates_per_epoch': data['max_updates_per_epoch'],
        'eval_interval': data['eval_interval'],
        'custom_weight_initialization': data['custom_weight_initialization'],
        'entropy_bonus_factor': data['entropy_bonus_factor'],
        'learning_every_x_iter': data['learning_every_x_iter'],
        'per_learn_reward_normalization': data['per_learn_reward_normalization'],
        'normalization': {
            "SIGNAL_AMPLIFICATION_DEGREE": data['normalization']["SIGNAL_AMPLIFICATION_DEGREE"],
            "REWARD_AMPLIFICATION_DEGREE": data['normalization']["REWARD_AMPLIFICATION_DEGREE"],
            "REWARD_AMPLIFICATION_THRESHOLD": data['normalization']["REWARD_AMPLIFICATION_THRESHOLD"],
            "STD_THRESHOLD_FOR_REQ_FEAT_NORMALIZATION": data['normalization']["STD_THRESHOLD_FOR_REQ_FEAT_NORMALIZATION"],
            "STD_THRESHOLD_FOR_POD_FEAT_NORMALIZATION": data['normalization']["STD_THRESHOLD_FOR_POD_FEAT_NORMALIZATION"],
            "ENABLE_POD_NORMALIZATION": data['normalization']["ENABLE_POD_NORMALIZATION"],
            "ENABLE_REQUEST_NORMALIZATION": data['normalization']["ENABLE_REQUEST_NORMALIZATION"],
            "FEATURES_NORMALIZED": features_normalized,
            "NUM_FEATURES_NORMALIZED": data['normalization']["NUM_FEATURES_NORMALIZED"],
            "FEATURE_AMPLIFICATION": data['normalization']["FEATURE_AMPLIFICATION"],
            "FEATURES_AMPLIFIED": features_amplified,
            "NUM_FEATURES_AMPLIFIED": data['normalization']["NUM_FEATURES_AMPLIFIED"],
        },
        'dataset_analysis': data['dataset_analysis'],
    }

    for key, value in rl_model_hyperparameters.items():
        if key == 'normalization':
            for sub_key, sub_value in value.items():
                logger.info(f"{key}.{sub_key}: {sub_value}")
        else:
            logger.info(f"{key}: {value}")

    return rl_model_hyperparameters


def fetch_pod_gpu_mapping():
    """
    Fetch GPU model for each pod in the llama-3-8b-instruct deployment
    Returns a dictionary mapping pod_ip -> gpu_model
    """
    try:
        from kubernetes import client, config
        
        # Try in-cluster config first (for running inside cluster)
        try:
            config.load_incluster_config()
            logger.info("Loaded in-cluster config for Kubernetes access")
        except:
            # Fallback to local config (for development/testing)
            config.load_kube_config()
            logger.info("Loaded local kubeconfig for Kubernetes access")
        
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
            else:
                logger.debug(f"Skipping pod {pod.metadata.name} - Status: {pod.status.phase}, IP: {pod.status.pod_ip}")
        
        logger.info(f"Successfully fetched GPU mapping for {len(pod_gpu_mapping)} pods")
        return pod_gpu_mapping
        
    except ImportError:
        logger.error("kubernetes package not installed. Install with: pip install kubernetes")
        return {}
    except Exception as e:
        logger.error(f"Failed to fetch pod GPU mapping: {e}")
        return {}

def test_kubernetes_permissions():
    """Test if we have the required Kubernetes permissions"""
    try:
        config.load_incluster_config()
        
        v1 = client.CoreV1Api()
        
        # Test 1: Can we list pods?
        try:
            pods = v1.list_pod_for_all_namespaces(label_selector="model.aibrix.ai/name=llama-3-8b-instruct", limit=1)
            logger.info("✅ Successfully listed pods - pod permissions OK")
        except Exception as e:
            logger.error(f"❌ Cannot list pods: {e}")
            return False
            
        # Test 2: Can we read nodes?
        try:
            nodes = v1.list_node(limit=1)
            logger.info("✅ Successfully listed nodes - node permissions OK")
        except Exception as e:
            logger.error(f"❌ Cannot list nodes: {e}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Kubernetes API access failed: {e}")
        return False

def graceful_shutdown(sig=None, frame=None):
    """Handle graceful shutdown when receiving SIGTERM or SIGINT"""
    logger.info(f"Received signal {sig if sig else 'shutdown'}, shutting down gracefully...")
    
    # Shutdown the scheduler if it exists
    if 'scheduler' in globals() and scheduler:
        try:
            scheduler.shutdown(wait=False)
            logger.info("Background scheduler shut down successfully")
        except Exception as e:
            logger.error(f"Error shutting down scheduler: {e}")
    
    # Any other cleanup you need can go here
    logger.info("Graceful shutdown completed")
    sys.exit(0)


def init():
    global RL_MODEL_HYPERPARAMETERS, stats_instance
    if RL_MODEL_HYPERPARAMETERS is None:
        logger.info("Loading RL hyperparameters from model_config.json")
        RL_MODEL_HYPERPARAMETERS = load_rl_hyperparameters(hyperparameter_file_path)
        
        # Test permissions first
        logger.info("Testing Kubernetes API permissions...")
        if not test_kubernetes_permissions():
            logger.error("Insufficient Kubernetes permissions - using fallback GPU mapping")
            assert False
        
        # Fetch pod GPU mapping and add to hyperparameters
        logger.info("Fetching pod GPU mapping from Kubernetes cluster")
        pod_gpu_mapping = fetch_pod_gpu_mapping()  # pod_ip -> gpu_model_name
        
        if pod_gpu_mapping:
            # Create GPU model to ID mapping
            unique_gpus = list(set(pod_gpu_mapping.values()))
            gpu_name_to_id = {gpu_name: idx for idx, gpu_name in enumerate(unique_gpus)}
            
            # Create direct pod_ip -> gpu_model_id mapping
            pod_gpu_id_mapping = {pod_ip: gpu_name_to_id[gpu_name] for pod_ip, gpu_name in pod_gpu_mapping.items()}
            
            RL_MODEL_HYPERPARAMETERS['pod_gpu_mapping'] = pod_gpu_mapping  # Keep original for logging
            RL_MODEL_HYPERPARAMETERS['pod_gpu_id_mapping'] = pod_gpu_id_mapping  # Direct mapping
            RL_MODEL_HYPERPARAMETERS['GPU_MAP'] = gpu_name_to_id  # Keep for compatibility
            
            logger.info(f"GPU name to ID mapping: {gpu_name_to_id}")
            logger.info(f"Created direct pod IP to GPU ID mapping for {len(pod_gpu_id_mapping)} pods")
        else:
            logger.error("No pod GPU mapping available, using default mappings")
            assert False

    stats_instance = feature_normalization.get_stats_instance(RL_MODEL_HYPERPARAMETERS['normalization'], feature_normalization_stats_file)

if __name__ == "__main__":
    signal.signal(signal.SIGTERM, graceful_shutdown)
    signal.signal(signal.SIGINT, graceful_shutdown)
    atexit.register(graceful_shutdown)
    
    init()

    port = int(os.environ.get("PORT", 8080))
    scheduler = BackgroundScheduler()
    # If online learning is disabled, just use the pretrained model
    if ENABLE_ONLINE_LEARNING:
        scheduler.add_job(func=online_train_routine, trigger="interval", seconds=1)
    else:
        logger.info("Online learning disabled. online_train_routine will not be invoked at all - using pretrained model only in inference")
    scheduler.start()
    atexit.register(lambda: scheduler.shutdown())
    app.run(host="0.0.0.0", port=port, debug=False)