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
import data_normalizer
import signal
import sys
import socket
import utils as utils
from kubernetes import client, config
from logger import logger, INCLUDE_GPU_IN_FEATURE

# INCLUDE_GPU_IN_FEATURE = True

app = Flask(__name__)


hyperparameter_file_path = '/app/final_model/model_config.json'

NUM_FLUSH = 0
ENCODED_DATA_DIR = "encoded_data"
final_model_dir = "/app/final_model"
feature_normalization_stats_file = f"{final_model_dir}/feature_normalization_statistics.csv"  # Add this near the top with your other constants;
NUM_TRAINS = 0
MODEL_UPDATED = True
LOCK_TRAINING_DATA = threading.Lock()
first_request_starting_time = None
stats_instance = None 
TOTAL_NUM_DATA = 0
NUM_NEW_DATA = 0
TRAINING_RIGHT_NOW = False

MIN_NUM_TRAINING_DATA = int(os.getenv("MIN_NUM_TRAINING_DATA", 1000))
POD_LABEL_SELECTOR = os.getenv("POD_LABEL_SELECTOR", "model.aibrix.ai/name=llama-3-8b-instruct")
ENABLE_ONLINE_LEARNING = os.getenv("ENABLE_ONLINE_LEARNING", "false").lower() == "true"
EXPLORATION_ENABLED = int(os.getenv("EXPLORATION_ENABLED", 0))
TTFT_REWARD_WEIGHT = float(os.getenv("TTFT_REWARD_WEIGHT", 0.5))
RL_MODEL_HYPERPARAMETERS = None

request_features_train = ['input_tokens', 'output_tokens', 'total_tokens']
# request_features_reward = ['ttft', 'avg_tpot', 'e2e_latency']

# Fixed handle_flush function
@app.route("/flush", methods=["POST"])
def handle_flush():
    global NUM_FLUSH, ENCODED_DATA_DIR, TOTAL_NUM_DATA, NUM_NEW_DATA, feature_normalization_stats_file, stats_instance
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
        ##################################################
        ## Write raw data to file
        utils.write_to_file(log_data, raw_data_path)
        ##################################################
        logger.info(f"wrote {len(log_data)} entries to {raw_data_path}, took {time.time() - ts_write_raw_data} seconds")

        replaced_data_path = utils.replace_pod_ip_with_generalpodid(raw_data_path)
        ts_preprocess = time.time()
        ##################################################
        ## Preprocess
        processed_df, sorted_all_pod_ids, _ = preprocess.main(replaced_data_path, "", RL_MODEL_HYPERPARAMETERS)
        ##################################################
        logger.info(f"Successfully parsed data, took {time.time() - ts_preprocess} seconds")
        
        normalizable_features, non_normalizable_features = data_normalizer._get_normalizable_features(processed_df)
        if stats_instance.get_max_count() == 0:
            logger.error(f"No normalization statistics available for training")
            assert False
        for feature in normalizable_features:
            ##################################################
            ## Normalize
            data_normalizer._normalize_single_feature(processed_df, feature, stats_instance, is_training=True)
            ##################################################
        if stats_instance is not None:
            stats_instance.write_stats_to_file(feature_normalization_stats_file)
        else:
            logger.error("stats_instance is None. Cannot save stats")
            assert False

        ##################################################
        ## Encode
        encoded_data_subdir = f"{ENCODED_DATA_DIR}/batch_{NUM_FLUSH}"
        encoding.encode_for_train(sorted_all_pod_ids, processed_df, encoded_data_subdir, request_features_train, RL_MODEL_HYPERPARAMETERS)
        ##################################################
        
        logger.info(f"Successfully flushed {len(log_data)} log messages, took {time.time() - flush_start_time} seconds")
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
    handle_infer_overhead_summary = {}
    if first_request_starting_time == None:
        first_request_starting_time = time.time()
        logger.info(f"First request starting time set to {first_request_starting_time}")
    # if NUM_TRAINS == 0:
    #     logger.warning("No trained model available, please call /flush to train first")
    #     return jsonify({"error": "No trained model available, please call /flush to train first"}), 503
    
    handle_infer_start_time = time.time()
    try:
        # Get the log message as a string from the request body
        request_prepare_start_time = time.time()
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
        replace_podid_start_time = time.time()
        log_message = utils.replace_pod_ip_with_generalpodid(log_message)
        # logger.info(f"log_message_with_replaced_pod_id: {log_message}")
        handle_infer_overhead_summary["replace_podid_overhead"] = time.time() - replace_podid_start_time
        parts = log_message.split("requestID@")
        if len(parts) > 1:
            request_id_parts = parts[1].split("@")
            if request_id_parts:
                request_id = request_id_parts[0]
        else:
            logger.warning("No request ID found in log message, using default 'default'")
        handle_infer_overhead_summary["request_prepare"] = time.time() - request_prepare_start_time
        
        # Use the existing preprocessing function to parse the log
        preprocess_start_time = time.time()
        processed_df, sorted_all_pod_ids, preprocess_dataset_overhead_summary = preprocess.main(None, log_message, RL_MODEL_HYPERPARAMETERS)
        logger.info(f"sorted_all_pod_ids: {sorted_all_pod_ids}")
        handle_infer_overhead_summary["preprocess_overhead"] = time.time() - preprocess_start_time

        normalize_start = time.time()
        if stats_instance is None:
            logger.error(f"No running statistics available, stats_instance: {stats_instance}")
            logger.error("Cannot perform inference without normalization statistics")
            return jsonify({"error": "No normalization statistics available"}), 500
            assert False
        if stats_instance.get_max_count() == 0:
            logger.error(f"Stats instance count is 0, no data available for normalization")
            assert False
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

        normalizable_features, non_normalizable_features = data_normalizer._get_normalizable_features(processed_df)
        if stats_instance.get_max_count() == 0:
            logger.error(f"request_id,{request_id},No normalization statistics available for inference")
            assert False
        for feature in normalizable_features:
            ##################################################
            data_normalizer._normalize_single_feature(processed_df, feature, stats_instance, is_training=False, request_id=request_id)
            ##################################################
        handle_infer_overhead_summary["normalize"] = time.time() - normalize_start

        ## Encode data (normalization already done)
        encode_start_time = time.time()
        tensor_data, encode_for_inference_overhead_summary = encoding.encode_for_inference(sorted_all_pod_ids, processed_df, request_features_train, RL_MODEL_HYPERPARAMETERS)
        handle_infer_overhead_summary["encode"] = time.time() - encode_start_time

        infer_from_tensor_start_time = time.time()
        result, infer_from_tensor_overhead_summary = simpler_contextual_bandit.infer_from_tensor(tensor_data, request_id, MODEL_UPDATED, RL_MODEL_HYPERPARAMETERS, final_model_dir)
        if MODEL_UPDATED:
            logger.info("Model updated flag consumed, resetting to False")
            MODEL_UPDATED = False
        handle_infer_overhead_summary["calling_infer_from_tensor"] = time.time() - infer_from_tensor_start_time
        
        
        remaining_work_start = time.time()
        result["requestID"] = request_id
        result["num_trains"] = NUM_TRAINS
        result["request_timestamp"] = time.time() - first_request_starting_time
        logger.info(f"Inference result: {result}")
        
        # Map the pod index back to the actual pod ID
        selected_pod_index = result['selected_pod_index']
        if selected_pod_index >= len(sorted_all_pod_ids):
            logger.error(f"Selected pod index {selected_pod_index} out of range, defaulting to first pod")
            assert False
        selected_pod_generalpodid = sorted_all_pod_ids[selected_pod_index]
        selected_pod_ip = RL_MODEL_HYPERPARAMETERS['generalpodid_to_pod_ip'][selected_pod_generalpodid]
            
        logger.info(f"selected_pod_generalpodid: {selected_pod_generalpodid}, selected_pod_ip: {selected_pod_ip}, pod_probability: {result['pod_probabilities']}")
        
        handle_infer_overhead_summary['remaining_work'] = time.time() - remaining_work_start
        handle_infer_overhead_summary["end_to_end"] = time.time() - handle_infer_start_time
        
        overhead_log = "oh"
        for key, value in handle_infer_overhead_summary.items():
            overhead_log += f", handle_infer_{key}: {value*1000:.0f}ms"
        for key, value in encode_for_inference_overhead_summary.items():
            overhead_log += f", encode_{key}: {value*1000:.0f}ms"
        for key, value in preprocess_dataset_overhead_summary.items():
            overhead_log += f", preprocess_{key}: {value*1000:.0f}ms"
        for key, value in infer_from_tensor_overhead_summary.items():
            overhead_log += f", infer_from_tensor_{key}: {value*1000:.0f}ms"
        response = {
            "num_trains": NUM_TRAINS,
            "num_flush": NUM_FLUSH,
            "request_timestamp": time.time() - first_request_starting_time,
            "selected_pod": selected_pod_ip,
            "selected_pod_generalpodid": selected_pod_generalpodid,
            "confidence": result['confidence'],
            "exploration": result['explore_mask'],
            "exploration_enabled": EXPLORATION_ENABLED,
            "request_id": request_id,
            "overhead_log": overhead_log,
        }
        return jsonify(response), 200
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Error in handle_infer: {str(e)}")
        logger.error(f"Traceback: {error_traceback}")
        return jsonify({"error": str(e), "traceback": error_traceback}), 500


def online_train_routine():
    global NUM_TRAINS, MODEL_UPDATED, TOTAL_NUM_DATA, final_model_dir, NUM_NEW_DATA, RL_MODEL_HYPERPARAMETERS, TRAINING_RIGHT_NOW
    if TRAINING_RIGHT_NOW:
        logger.info(f"Previous training still in progress, skipping training")
        return
    if NUM_NEW_DATA < MIN_NUM_TRAINING_DATA:
        logger.info(f"Not enough training data available (NUM_NEW_DATA: {NUM_NEW_DATA} < {MIN_NUM_TRAINING_DATA}). Wait until more data comes in.")
        return
    TRAINING_RIGHT_NOW = True
    training_start_time = time.time()
    logger.info(f"Start {NUM_TRAINS}th online training with {NUM_NEW_DATA} new training data")
    try:
        simpler_contextual_bandit.train(ENCODED_DATA_DIR, final_model_dir, RL_MODEL_HYPERPARAMETERS, ENABLE_ONLINE_LEARNING)
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return
    MODEL_UPDATED = True
    NUM_TRAINS += 1
    NUM_NEW_DATA = 0
    TRAINING_RIGHT_NOW = False
    logger.info(f"Successfully completed {NUM_TRAINS}th online training with {NUM_NEW_DATA} new training data, took {time.time() - training_start_time} seconds")


def test_kubernetes_permissions():
    """Test if we have the required Kubernetes permissions"""
    try:
        config.load_incluster_config()
        
        v1 = client.CoreV1Api()
        
        # Test 1: Can we list pods?
        try:
            pods = v1.list_pod_for_all_namespaces(label_selector=POD_LABEL_SELECTOR, limit=1)
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
        RL_MODEL_HYPERPARAMETERS = {}
        RL_MODEL_HYPERPARAMETERS['TTFT_REWARD_WEIGHT'] = TTFT_REWARD_WEIGHT
        RL_MODEL_HYPERPARAMETERS['EXPLORATION_ENABLED'] = EXPLORATION_ENABLED
        RL_MODEL_HYPERPARAMETERS['ENABLE_ONLINE_LEARNING'] = ENABLE_ONLINE_LEARNING
        logger.info("Loading RL hyperparameters from model_config.json")
        utils.load_rl_hyperparameters(hyperparameter_file_path, RL_MODEL_HYPERPARAMETERS)
        # Test permissions first
        logger.info("Testing Kubernetes API permissions...")
        if not test_kubernetes_permissions():
            logger.error("Insufficient Kubernetes permissions - using fallback GPU mapping")
            assert False

        running_pods = utils.get_running_pods_by_label(POD_LABEL_SELECTOR)

        sorted_running_pod_ips = utils.fetch_running_pod_ips(running_pods)
        
        pod_ip_to_generalpodid = utils.create_pod_ip_to_generalpodid_mapping(sorted_running_pod_ips)
        generalpodid_to_pod_ip = {}
        for pod_ip, generalpodid in pod_ip_to_generalpodid.items():
            generalpodid_to_pod_ip[generalpodid] = pod_ip

        generalpodid_to_gpu_model = utils.fetch_generalpodid_to_gpu_model(running_pods, pod_ip_to_generalpodid)

        pod_ip_to_gpu_model, pod_ip_to_gpu_model_encoded = utils.create_pod_ip_to_gpu_model_mapping(generalpodid_to_gpu_model, pod_ip_to_generalpodid)
        
        logger.debug(f"sorted_running_pod_ips: {sorted_running_pod_ips}")
        logger.debug(f"pod_ip_to_generalpodid: {pod_ip_to_generalpodid}")
        logger.debug(f"generalpodid_to_gpu_model: {generalpodid_to_gpu_model}")
        logger.debug(f"pod_ip_to_gpu_model: {pod_ip_to_gpu_model}")
        logger.debug(f"pod_ip_to_gpu_model_encoded: {pod_ip_to_gpu_model_encoded}")

        RL_MODEL_HYPERPARAMETERS['pod_ip_to_generalpodid'] = pod_ip_to_generalpodid
        RL_MODEL_HYPERPARAMETERS['generalpodid_to_pod_ip'] = generalpodid_to_pod_ip
        RL_MODEL_HYPERPARAMETERS['sorted_running_pod_ips'] = sorted_running_pod_ips
        RL_MODEL_HYPERPARAMETERS['pod_ip_to_gpu_model'] = pod_ip_to_gpu_model
        RL_MODEL_HYPERPARAMETERS['pod_ip_to_gpu_model_encoded'] = pod_ip_to_gpu_model_encoded
        RL_MODEL_HYPERPARAMETERS['generalpodid_to_gpu_model'] = generalpodid_to_gpu_model
        
        # Load normalization statistics from CSV file
        if os.path.exists(feature_normalization_stats_file):
            logger.info(f"Loading normalization statistics from: {feature_normalization_stats_file}")
            try:
                stats_instance = data_normalizer.FeatureStats.load_from_csv(feature_normalization_stats_file)
                if stats_instance is not None:
                    logger.info(f"Successfully loaded stats for {len(stats_instance.feature_stats)} features")
                else:
                    logger.error("Failed to load normalization statistics")
                    assert False
            except Exception as e:
                logger.error(f"Failed to load normalization statistics: {e}")
                assert False
        else:
            logger.error(f"Normalization statistics file not found: {feature_normalization_stats_file}")
            assert False
    
    # Print feature statistics if available
    if stats_instance is not None:
        logger.info("Per-feature statistics loaded:")
        for feature_name, stats in stats_instance.feature_stats.items():
            logger.info(f"stats_instance, {feature_name}: count={stats.count}, mean={stats.mean}, std={stats.std}")
    else:
        logger.warning("No normalization statistics available - inference will fail")
        assert False

if __name__ == "__main__":
    signal.signal(signal.SIGTERM, graceful_shutdown)
    signal.signal(signal.SIGINT, graceful_shutdown)
    atexit.register(graceful_shutdown)
    
    
    port = int(os.environ.get("PORT", 8080))
    if not utils.wait_for_port_available(port, max_wait=5):
        logger.error(f"Cannot start Flask app - port {port} is not available")
        sys.exit(1)
        
    logger.info(f"Port {port} is available, starting Flask app properly!")
    
    init()

    scheduler = BackgroundScheduler()
    # If online learning is disabled, just use the pretrained model
    if ENABLE_ONLINE_LEARNING:
        scheduler.add_job(func=online_train_routine, trigger="interval", seconds=30)
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