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
# import ppo
# import contextual_bandit
import simpler_contextual_bandit
from flask import Flask, request, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
import preprocess
import pickle
import threading
from kubernetes import client, config
import ast
import json
import feature_normalization
import signal
import sys
import socket
import utils as utils
from kubernetes import client, config
from logger import logger, INCLUDE_GPU_IN_FEATURE

# INCLUDE_GPU_IN_FEATURE = True

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
        utils.write_to_file(log_data, raw_data_path)
        logger.info(f"wrote {len(log_data)} entries to {raw_data_path}, took {time.time() - ts_write_raw_data} seconds")

        
        replaced_data_path = utils.replace_pod_ip_with_generalpodid(raw_data_path)
        ts_preprocess = time.time()
        processed_df, _, sorted_all_pod_ids, _ = preprocess.main(replaced_data_path, "", TTFT_SLO, AVG_TPOT_SLO, RL_MODEL_HYPERPARAMETERS)
        logger.info(f"Successfully parsed data, took {time.time() - ts_preprocess} seconds")
        
        # if ENABLE_ONLINE_LEARNING:
        processed_df = feature_normalization.normalize_features_for_training(processed_df, stats_instance)
        stats_instance.write_stats_to_file(feature_normalization_stats_file)

        # Encode preprocessed data
        ts_encode = time.time()
        encoded_data_subdir = f"{ENCODED_DATA_DIR}/batch_{NUM_FLUSH}"
        encoding.encode_for_train(sorted_all_pod_ids, processed_df, encoded_data_subdir, request_features_train, RL_MODEL_HYPERPARAMETERS)
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
        if isinstance(log_data, str):
            log_message = log_data
        else:
            # Handle dict input (original logic)
            if len(list(log_data.keys())) != 1:
                logger.error(f"There must be only one request for inference, but got {len(list(log_data.keys()))} requests")
                return jsonify({"error": "Invalid request format"}), 400
            
            first_key = list(log_data.keys())[0]
            log_message = log_data[first_key]
        logger.debug(f"Received inference request in handle_infer, log_message:\n{log_message}")

        # Extract request ID for logging purposes
        request_id = "default"  # or extract from log_message
        log_message = utils.replace_pod_ip_with_generalpodid(log_message)
        # logger.info(f"log_message_with_replaced_pod_id: {log_message}")
        
        parts = log_message.split("requestID@")
        if len(parts) > 1:
            request_id_parts = parts[1].split("@")
            if request_id_parts:
                request_id = request_id_parts[0]
        else:
            logger.warning("No request ID found in log message, using default 'default'")
        handle_infer_total_prep_overhead = time.time() - prep_start_time

        # Use the existing preprocessing function to parse the log
        preprocess_start_time = time.time()
        processed_df, _, sorted_all_pod_ids, preprocess_dataset_overhead_summary = preprocess.main(None, log_message, TTFT_SLO, AVG_TPOT_SLO, RL_MODEL_HYPERPARAMETERS)
        logger.info(f"sorted_all_pod_ids: {sorted_all_pod_ids}")
        
        handle_infer_total_total_preprocess_overhead = time.time() - preprocess_start_time

        get_stat_start_time = time.time()
        if stats_instance is None:
            logger.error(f"No running statistics available, stats_instance: {stats_instance}")
            assert False
        if stats_instance.count == 0:
            logger.warning(f"Stats instance count is 0, no data available for normalization")

        #====================================================
        non_interest = ['request_id', 'requestID', 'ttft', 'avg_tpot', 'e2e_latency', 'selected_pod']
        features_must_exist_in_stats_instance = []
        for feature in processed_df.columns:
            # NOTE: ignoring last_second_* features
            if "last_second_" not in feature and feature not in non_interest:
                features_must_exist_in_stats_instance.append(feature)
        for feature in features_must_exist_in_stats_instance:
            if feature not in stats_instance.feature_stats:
                logger.error(f"Feature {feature} not found in stats_instance")
                logger.error(f"processed_df.columns: {list(processed_df.columns)}")
                logger.error(f"features_must_exist_in_stats_instance: {features_must_exist_in_stats_instance}")
                logger.error(f"Available stats features: {list(stats_instance.feature_stats.keys())}")
                assert False
        
        # stats = stats_instance.feature_stats[feature]
        # mean_val = stats.mean.item()
        # std_val = stats.std.item()
        # logger.info(f"before normalize, {feature}: value={processed_df[feature].iloc[0]:.2f} → Has stats: count={stats.count}, min={stats.min}, max={stats.max}, mean={mean_val:.2f}, std={std_val:.2f}")



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
        # processed_df = feature_normalization.normalize_features_for_inference(processed_df, stats_instance, request_id)
        
        #===================================================
        handle_infer_total_get_stat_overhead = time.time() - get_stat_start_time

        # Encode data (normalization already done)
        encode_start_time = time.time()
        tensor_data, encode_for_inference_overhead_summary = encoding.encode_for_inference(sorted_all_pod_ids, processed_df, request_features_train, RL_MODEL_HYPERPARAMETERS)
        handle_infer_total_total_encoding_overhead = time.time() - encode_start_time

        infer_from_tensor_start_time = time.time()
        if MODEL == "simpler_contextual_bandit":
            result, infer_from_tensor_overhead_summary = simpler_contextual_bandit.infer_from_tensor(tensor_data=tensor_data, request_id=request_id, model_updated=MODEL_UPDATED, HYPERPARAMETERS=RL_MODEL_HYPERPARAMETERS)
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
        selected_pod_index = result['selected_pod_index']
        if selected_pod_index >= len(sorted_all_pod_ids):
            logger.error(f"Selected pod index {selected_pod_index} out of range, defaulting to first pod")
            assert False
            
        selected_pod_generalpodid = sorted_all_pod_ids[selected_pod_index]
        selected_pod_ip = RL_MODEL_HYPERPARAMETERS['generalpodid_to_pod_ip'][selected_pod_generalpodid]
        handle_infer_total_wrapup_overhead = time.time() - handle_infer_total_wrapup_start_time
        handle_infer_total_overhead = time.time() - handle_infer_start_time
        
        # Return the result
        response = {
            "num_trains": NUM_TRAINS,
            "num_flush": NUM_FLUSH,
            "request_timestamp": time.time() - first_request_starting_time,
            "selected_pod": selected_pod_ip,
            "selected_pod_generalpodid": selected_pod_generalpodid,
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

        logger.info(f"selected_pod_generalpodid: {selected_pod_generalpodid}, selected_pod_ip: {selected_pod_ip}, pod_probability: {result['pod_probabilities']}")
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
                is_online_learning = True
                simpler_contextual_bandit.train(ENCODED_DATA_DIR, final_model_path, RL_MODEL_HYPERPARAMETERS, is_online_learning)
            # elif MODEL == "contextual_bandit":
            #     contextual_bandit.train(ENCODED_DATA_DIR)
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
    
    rl_model_hyperparameters = {}
    for key, value in data.items():
        if key == 'normalization':
            # Handle nested normalization parameters
            rl_model_hyperparameters[key] = {}
            for sub_key, sub_value in value.items():
                if sub_key == 'FEATURES_NORMALIZED':
                    # Convert string representation of set to actual set
                    if sub_value and sub_value != "set()":
                        rl_model_hyperparameters[key][sub_key] = ast.literal_eval(sub_value)
                    else:
                        rl_model_hyperparameters[key][sub_key] = set()
                elif sub_key == 'FEATURES_AMPLIFIED':
                    # Convert string representation of set to actual set
                    if sub_value and sub_value != "set()":
                        rl_model_hyperparameters[key][sub_key] = ast.literal_eval(sub_value)
                    else:
                        rl_model_hyperparameters[key][sub_key] = set()
                else:
                    rl_model_hyperparameters[key][sub_key] = sub_value
        else:
            rl_model_hyperparameters[key] = value

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
        try:
            config.load_incluster_config()
            logger.info("Loaded in-cluster config for Kubernetes access")
        except:
            config.load_kube_config()
            logger.info("Loaded local kubeconfig for Kubernetes access")
        v1 = client.CoreV1Api()
        label_selector = "model.aibrix.ai/name=llama-3-8b-instruct"
        pods = v1.list_pod_for_all_namespaces(label_selector=label_selector)
        pod_gpu_mapping = {}
        for pod in pods.items:
            if pod.status.phase == "Running" and pod.status.pod_ip:
                pod_ip = pod.status.pod_ip
                node_name = pod.spec.node_name
                if node_name:
                    try:
                        node = v1.read_node(name=node_name)
                        node_labels = node.metadata.labels or {}
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
        
        label_selector = "model.aibrix.ai/name=llama-3-8b-instruct"
        
        running_pods = utils.get_running_pods_by_label(label_selector)
        
        sorted_running_pod_ips = utils.fetch_running_pod_ips(running_pods)
        
        pod_ip_to_generalpodid = utils.create_pod_ip_to_generalpodid_mapping(sorted_running_pod_ips)
        generalpodid_to_pod_ip = {}
        for pod_ip, generalpodid in pod_ip_to_generalpodid.items():
            generalpodid_to_pod_ip[generalpodid] = pod_ip

        generalpodid_to_gpu_model = utils.fetch_generalpodid_to_gpu_model(running_pods, pod_ip_to_generalpodid)

        pod_ip_to_gpu_model, pod_ip_to_gpu_model_encoded = utils.create_pod_ip_to_gpu_model_mapping(generalpodid_to_gpu_model, pod_ip_to_generalpodid)
        
        logger.info(f"sorted_running_pod_ips: {sorted_running_pod_ips}")
        logger.info(f"pod_ip_to_generalpodid: {pod_ip_to_generalpodid}")
        logger.info(f"generalpodid_to_gpu_model: {generalpodid_to_gpu_model}")
        logger.info(f"pod_ip_to_gpu_model: {pod_ip_to_gpu_model}")
        logger.info(f"pod_ip_to_gpu_model_encoded: {pod_ip_to_gpu_model_encoded}")

        RL_MODEL_HYPERPARAMETERS['pod_ip_to_generalpodid'] = pod_ip_to_generalpodid
        RL_MODEL_HYPERPARAMETERS['generalpodid_to_pod_ip'] = generalpodid_to_pod_ip
        RL_MODEL_HYPERPARAMETERS['sorted_running_pod_ips'] = sorted_running_pod_ips
        RL_MODEL_HYPERPARAMETERS['pod_ip_to_gpu_model'] = pod_ip_to_gpu_model
        RL_MODEL_HYPERPARAMETERS['pod_ip_to_gpu_model_encoded'] = pod_ip_to_gpu_model_encoded
        RL_MODEL_HYPERPARAMETERS['generalpodid_to_gpu_model'] = generalpodid_to_gpu_model
        
        
    stats_instance = feature_normalization.get_stats_instance(RL_MODEL_HYPERPARAMETERS['normalization'], feature_normalization_stats_file)
    
    ## print all features and std, mean, and count
    logger.info("Per-feature statistics loaded:")
    for feature_name, stats in stats_instance.feature_stats.items():
        logger.info(f"stats_instance, {feature_name}: count={stats.count}, mean={stats.mean}, std={stats.std}")
        

if __name__ == "__main__":
    signal.signal(signal.SIGTERM, graceful_shutdown)
    signal.signal(signal.SIGINT, graceful_shutdown)
    atexit.register(graceful_shutdown)
    
    
    port = int(os.environ.get("PORT", 8080))
    if not utils.wait_for_port_available(port, max_wait=60):
        logger.error(f"Cannot start Flask app - port {port} is not available")
        sys.exit(1)
    
    init()

    scheduler = BackgroundScheduler()
    # If online learning is disabled, just use the pretrained model
    if ENABLE_ONLINE_LEARNING:
        scheduler.add_job(func=online_train_routine, trigger="interval", seconds=1)
    else:
        logger.info("Online learning disabled. online_train_routine will not be invoked at all - using pretrained model only in inference")
    scheduler.start()
    atexit.register(lambda: scheduler.shutdown())
    
    
    
    # NEW CODE: Add error handling around app.run()
    try:
        logger.info(f"Starting Flask app on port {port}")
        app.run(host="0.0.0.0", port=port, debug=False)
    except OSError as e:
        if "Address already in use" in str(e):
            logger.error(f"Port {port} is still in use. Trying to wait and retry...")
            time.sleep(5)
            if utils.wait_for_port_available(port, max_wait=30):
                app.run(host="0.0.0.0", port=port, debug=False)
            else:
                logger.error("Failed to start Flask app - port conflict persists")
                sys.exit(1)
        else:
            raise