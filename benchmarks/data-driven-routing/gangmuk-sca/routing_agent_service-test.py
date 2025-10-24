## routing_agent_service-test.py
##
## SCALABILITY TEST MODE CONFIGURATION:
## - Uses REAL latency predictor model for accurate overhead measurement
## - Model path: /mnt/data/projects/aibrix-gangmuk/benchmarks/data-driven-routing/routing-agent-service/scalability_test/final_model-latency_predictor_ttft/
## - Loads real model_config.json, feature_normalization_statistics.csv, and latency_predictor.pth
## - Calls actual handle_infer endpoint via Flask test client
## - Measures complete end-to-end overhead: preprocessing, normalization, encoding, and ML inference
## - Mock pod configurations for different cluster sizes
##
## Run with: SCALABILITY_TEST=1 python routing_agent_service-test.py

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
# import simpler_contextual_bandit
import latency_predictor
# from rl_routing_agent_sb3 import create_rl_routing_agent_sb3, infer_rl_agent
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
import queue
from collections import deque
from rwlock import RWLock


## colors for logging
BLUE_COLOR = "\033[94m"
RED_COLOR = "\033[91m"
GREEN_COLOR = "\033[92m"
YELLOW_COLOR = "\033[93m"
PURPLE_COLOR = "\033[95m"
CYAN_COLOR = "\033[96m"
MAGENTA_COLOR = "\033[95m"
RESET_COLOR = "\033[0m" 

# INCLUDE_GPU_IN_FEATURE = True

app = Flask(__name__)
# Path for deployed service (inside container)
# hyperparameter_file_path = '/app/final_model/model_config.json'
# final_model_dir = "/app/final_model"

# Path for scalability testing (local machine)
# if currently on node1, use the following path
if 'node0' in socket.gethostname():
    hyperparameter_file_path = '/mnt/data/projects/aibrix-gangmuk/benchmarks/data-driven-routing/routing-agent-service/scalability_test/final_model-latency_predictor_ttft/model_config.json'
    final_model_dir = "/mnt/data/projects/aibrix-gangmuk/benchmarks/data-driven-routing/routing-agent-service/scalability_test/final_model-latency_predictor_ttft"
else:
    hyperparameter_file_path = './final_model-latency_predictor_ttft/model_config.json'
    final_model_dir = "./final_model-latency_predictor_ttft"

NUM_FLUSH = 0
ENCODED_DATA_DIR = "encoded_data"
feature_normalization_stats_file = f"{final_model_dir}/feature_normalization_statistics.csv"  # Add this near the top with your other constants;
NUM_TRAINS = 0
MODEL_UPDATED = True
LOCK_TRAINING_DATA = threading.Lock()
first_request_starting_time = None
stats_instance = None
TOTAL_NUM_DATA = 0
NUM_NEW_DATA = 0
TRAINING_RIGHT_NOW = False

# Training data accumulation (offline + online)
TRAINING_DF = None  # Holds all training data (offline CSV + online appended data)
TRAINING_DF_LOCK = threading.Lock()  # Thread safety for concurrent flush/train
PRINT_ONCE_AT_THE_FIRST_REQUEST = True
# RL agent globals
RL_AGENT = None  # Old RL agent (entire cluster as input) - for 'rl_agent' subAlgorithm
SCALABLE_RL_AGENT = None  # New scalable RL agent (pod-independent) - for 'scalable_rl_agent' subAlgorithm
LATENCY_PREDICTOR = None  # Latency predictor model - for 'latency_predictor' subAlgorithm
# RWLock enables concurrent predictions (readers) with exclusive updates (writer)
# - Predictions: use rwlock.read() for high concurrency
# - Updates: use rwlock.write() for exclusive access
# - Initialization: use rwlock.write() for exclusive access
RL_AGENT_LOCK = RWLock()
SCALABLE_RL_AGENT_LOCK = RWLock()
LATENCY_PREDICTOR_LOCK = RWLock()

# Scalable RL agent training thread
SCALABLE_RL_TRAINING_THREAD = None
SCALABLE_RL_TRAINING_SHUTDOWN = threading.Event()


# RL agent async update queue
RL_UPDATE_QUEUE = queue.Queue(maxsize=1000)  # Bounded queue to prevent memory issues
RL_UPDATE_THREAD = None
RL_UPDATE_SHUTDOWN = threading.Event()

# Request completion tracking (for scalable RL async completion)
PENDING_REQUESTS = {}  # request_id → (route_time, selected_pod_idx)
PENDING_REQUESTS_LOCK = threading.Lock()

MIN_NUM_TRAINING_DATA = int(os.getenv("MIN_NUM_TRAINING_DATA", 1000))
# POD_LABEL_SELECTOR = os.getenv("POD_LABEL_SELECTOR", "model.aibrix.ai/name=llama3-1-8b")
POD_LABEL_SELECTOR = "model.aibrix.ai/name=llama-3-8b-instruct" ## Wanyu's new code
logger.info(f"POD_LABEL_SELECTOR: {POD_LABEL_SELECTOR}")
if POD_LABEL_SELECTOR == "":
    logger.error(f"POD_LABEL_SELECTOR is empty")
    assert False
ENABLE_ONLINE_LEARNING = int(os.getenv("ENABLE_ONLINE_LEARNING", 1))
ENABLE_ONLINE_LEARNING = 1 ## Wanyu's new code
EXPLORATION_ENABLED = int(os.getenv("EXPLORATION_ENABLED", 0))
TTFT_REWARD_WEIGHT = float(os.getenv("TTFT_REWARD_WEIGHT", 0.5))
RL_MODEL_HYPERPARAMETERS = None

BROKER_LOCK = RWLock()

request_features_train = ['input_tokens', 'output_tokens', 'total_tokens']
# request_features_reward = ['ttft', 'avg_tpot', 'e2e_latency']

# @app.route("/request_complete", methods=["POST"])
# def handle_request_complete():
#     """
#     Endpoint for async request completion notifications (for scalable_rl_agent).
    
#     Expected payload:
#     {
#         "request_id": "req_12345",
#         "ttft": 45.6,           # milliseconds
#         "tpot": 12.3,           # milliseconds  
#         "selected_pod": "10.0.1.30"
#     }
#     """
#     global SCALABLE_RL_AGENT, RL_MODEL_HYPERPARAMETERS
    
#     try:
#         data = request.json
#         request_id = data.get('request_id')
#         ttft = data.get('ttft')
#         tpot = data.get('tpot')
#         selected_pod = data.get('selected_pod')
        
#         if not request_id or ttft is None or tpot is None:
#             logger.error(f"Missing required fields in request completion: {data}")
#             return jsonify({"error": "Missing required fields"}), 400
        
#         if SCALABLE_RL_AGENT is None:
#             logger.debug(f"Scalable RL agent not initialized, ignoring completion for {request_id}")
#             return jsonify({"status": "ok", "message": "agent not initialized"}), 200
        
#         # Get current cluster state (after completion)
#         try:
#             pod_features, kv_hit_ratios, request_features = get_current_cluster_features()
#             current_state = (pod_features, kv_hit_ratios, request_features)
            
#             # Complete the experience
#             on_request_complete_callback(
#                 rl_agent=SCALABLE_RL_AGENT,
#                 request_id=request_id,
#                 current_cluster_state=current_state,
#                 ttft=ttft,
#                 tpot=tpot,
#                 hyperparameters=RL_MODEL_HYPERPARAMETERS
#             )
            
#             logger.debug(f"✅ Completed experience for request {request_id} (ttft={ttft}ms, tpot={tpot}ms)")
#             return jsonify({"status": "ok"}), 200
            
#         except NotImplementedError:
#             logger.debug(f"⚠️  get_current_cluster_features() not implemented, skipping completion for {request_id}")
#             return jsonify({"status": "ok", "message": "cluster state fetch not implemented"}), 200
            
#     except Exception as e:
#         logger.error(f"Error in request completion handler: {e}")
#         import traceback
#         logger.error(traceback.format_exc())
#         return jsonify({"error": str(e)}), 500


# Fixed handle_flush function
@app.route("/flush", methods=["POST"])
def handle_flush():
    global NUM_FLUSH, ENCODED_DATA_DIR, TOTAL_NUM_DATA, NUM_NEW_DATA, feature_normalization_stats_file, stats_instance, TRAINING_DF
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

        podip_replaced_data_path = utils.replace_pod_ip_with_generalpodid(raw_data_path)
        ts_preprocess = time.time()
        ##################################################
        ## Preprocess
        processed_df, sorted_all_pod_ids, _ = preprocess.main(podip_replaced_data_path, "", RL_MODEL_HYPERPARAMETERS)
        ##################################################
        logger.info(f"Successfully parsed data, took {time.time() - ts_preprocess} seconds")

        # Append preprocessed data to TRAINING_DF for online learning
        if ENABLE_ONLINE_LEARNING:
            with TRAINING_DF_LOCK:
                if TRAINING_DF is None:
                    TRAINING_DF = processed_df.copy()
                    logger.info(f"Initialized TRAINING_DF with {len(processed_df)} samples")
                else:
                    old_size = len(TRAINING_DF)
                    TRAINING_DF = pd.concat([TRAINING_DF, processed_df], ignore_index=True)
                    logger.info(f"Appended {len(processed_df)} samples to TRAINING_DF (total: {old_size} → {len(TRAINING_DF)})")

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
    global NUM_TRAINS, MODEL_UPDATED, first_request_starting_time, stats_instance, RL_MODEL_HYPERPARAMETERS, PRINT_ONCE_AT_THE_FIRST_REQUEST
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
        if PRINT_ONCE_AT_THE_FIRST_REQUEST:
            logger.info(f"processed_df.columns: {list(processed_df.columns)}")
            logger.info(f"sorted_all_pod_ids: {sorted_all_pod_ids}")
            PRINT_ONCE_AT_THE_FIRST_REQUEST = False
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

        normalizable_features, non_normalizable_features = data_normalizer._get_normalizable_features(processed_df, RL_MODEL_HYPERPARAMETERS.get('NO_NORMALIZE_FEATURES', []))
        if stats_instance.get_max_count() == 0:
            logger.error(f"request_id,{request_id},No normalization statistics available for inference")
            assert False
            
        non_interest = ['request_id', 'requestID', 'ttft', 'avg_tpot', 'e2e_latency', 'selected_pod', 'request_start_time', 'request_end_time']
        features_must_exist_in_stats_instance = []
        for feature in processed_df.columns:
            # NOTE: ignoring last_second_* features
            if "last_second_" not in feature and feature not in non_interest and feature in normalizable_features:
                features_must_exist_in_stats_instance.append(feature)
        for feature in features_must_exist_in_stats_instance:
            if feature not in stats_instance.feature_stats:
                logger.error(f"Feature {feature} not found in stats_instance")
                # logger.error(f"processed_df.columns: {list(processed_df.columns)}")
                # logger.error(f"features_must_exist_in_stats_instance: {features_must_exist_in_stats_instance}")
                # logger.error(f"Available stats features: {list(stats_instance.feature_stats.keys())}")
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
        
        # Route to appropriate model based on model type
        # model_type = RL_MODEL_HYPERPARAMETERS.get('MODEL_TYPE', 'contextual_bandit')
        subAlgorithm = processed_df['subAlgorithm'].iloc[0]
        # subAlgorithm = 'latency_predictor' ## Wanyu's new code
        if subAlgorithm == 'latency_predictor':
            logger.info(f"requestID: {request_id}, subAlgorithm: {subAlgorithm}")

            global LATENCY_PREDICTOR
            
            # Check if initialization needed without blocking
            if LATENCY_PREDICTOR is None:
                with LATENCY_PREDICTOR_LOCK.write():
                    # Double-check after acquiring lock
                    if LATENCY_PREDICTOR is None:
                        state_dims = {
                            'pod_features': tensor_data['pod_features_with_staleness'].shape[2],
                            'kv_hit_ratios': tensor_data['kv_hit_ratios'].shape[2],
                            'request_features': tensor_data['request_features'].shape[1],
                            'num_pods': tensor_data['pod_features_with_staleness'].shape[1]
                        }
                        
                        # Use real predictor
                        logger.info(f"Initializing latency predictor with state_dims={state_dims}")
                        LATENCY_PREDICTOR = latency_predictor.LatencyPredictor(state_dims, RL_MODEL_HYPERPARAMETERS, final_model_dir)

                        # Load pretrained model
                        model_path = os.path.join(final_model_dir, 'latency_predictor.pth')
                        if os.path.exists(model_path):
                            try:
                                LATENCY_PREDICTOR.load(final_model_dir)
                                logger.info(f"Loaded latency predictor from {final_model_dir}")
                            except Exception as e:
                                logger.error(f"Failed to load latency predictor: {e}")
                        else:
                            logger.warning(f"No pretrained latency predictor found at {model_path}, using untrained model")

            # Inference with read lock (allows concurrent requests)
            with LATENCY_PREDICTOR_LOCK.read():
                # Real inference
                result, infer_from_tensor_overhead_summary = latency_predictor.infer_latency_predictor_with_model(
                    predictor=LATENCY_PREDICTOR,
                    tensor_data=tensor_data,
                    request_id=request_id,
                    sorted_all_pod_ids=sorted_all_pod_ids
                )
        elif subAlgorithm == 'contextual_bandit' or subAlgorithm == 'rl_naive':
            logger.info(f"subAlgorithm: {subAlgorithm}, Using contextual bandit model for inference (request_id: {request_id})")
            result, infer_from_tensor_overhead_summary = simpler_contextual_bandit.infer_from_tensor(tensor_data, request_id, MODEL_UPDATED, RL_MODEL_HYPERPARAMETERS, final_model_dir)
            result['predicted_latencies'] = {pod_id: -1 for pod_id in sorted_all_pod_ids}
            result['chosen_pod_predicted_latency'] = -1
        elif subAlgorithm == 'rl_agent':
            # === OLD RL AGENT (entire cluster as input state) ===
            logger.info(f"requestID: {request_id}, subAlgorithm: {subAlgorithm}, Using OLD RL agent (entire cluster) for inference")
            
            global RL_AGENT
            
            with RL_AGENT_LOCK.write():
                # Check if initialization needed
                pod_features_t = tensor_data['pod_features']
                n_pods = int(pod_features_t.shape[1])
                per_pod_dim = int(pod_features_t.shape[2])
                
                if (RL_AGENT is None or 
                    RL_AGENT.action_dim != n_pods or
                    RL_AGENT.state_dim.get('pod_features') != per_pod_dim):
                    # Initialize new agent
                    kv_hit_t = tensor_data['kv_hit_ratios']
                    req_features_t = tensor_data['request_features']
                    state_dim = {
                        'pod_features': per_pod_dim,
                        'kv_hit_ratios': int(kv_hit_t.shape[2]),
                        'request_features': int(req_features_t.shape[1]),
                    }
                    RL_AGENT = create_rl_routing_agent_sb3(
                        state_dim=state_dim,
                        action_dim=n_pods,
                        **RL_MODEL_HYPERPARAMETERS
                    )
                    ckpt_path = RL_MODEL_HYPERPARAMETERS.get('RL_CHECKPOINT_PATH')
                    if ckpt_path and os.path.exists(ckpt_path):
                        try:
                            RL_AGENT.load(ckpt_path)
                            logger.info(f"Loaded RL checkpoint from {ckpt_path}")
                        except Exception as e:
                            logger.error(f"Failed to load RL checkpoint {ckpt_path}: {e}")
                    logger.info(f"Initialized OLD RL agent with state_dim={state_dim}, action_dim={n_pods}")
                
                # Get agent reference under write lock
                current_agent = RL_AGENT
            
            # Inference uses read lock for predictions (allows concurrency)
            current_agent, result, infer_from_tensor_overhead_summary = infer_rl_agent(
                tensor_data=tensor_data,
                request_id=request_id,
                sorted_all_pod_ids=sorted_all_pod_ids,
                processed_df=processed_df,
                rl_agent=current_agent,
                hyperparameters=RL_MODEL_HYPERPARAMETERS,
                agent_lock=RL_AGENT_LOCK  # RWLock for read (predict) and write (buffer)
            )
            
            # Queue async update if online learning enabled
            update_overhead = 0.0
            if ENABLE_ONLINE_LEARNING:
                update_start = time.time()
                with RL_AGENT_LOCK.read():
                    if RL_AGENT is not None:
                        buffer_size = len(RL_AGENT.experience_buffer)
                        batch_size = RL_AGENT.hyperparameters.get('batch_size', 64)
                        
                        if buffer_size >= batch_size:
                            queue_rl_update(n_steps=batch_size)
                            logger.debug(f"Queued RL update: buffer_size={buffer_size}, batch_size={batch_size}")
                update_overhead = time.time() - update_start
            
            infer_from_tensor_overhead_summary['online_update'] = update_overhead
        ####################################################################################
        ####################################################################################
        elif subAlgorithm == 'scalable_rl_agent':
            from scalable_rl_routing_agent import BROKER, infer
            
            # === NEW SCALABLE RL AGENT (pod-count independent) ===
            logger.info(f"scalable_rl_routing_agent, requestID: {request_id}, subAlgorithm: {subAlgorithm}, Using SCALABLE RL agent (pod-independent) for inference")
            
            # Extract features from tensor_data
            pod_features = tensor_data['pod_features'].cpu().numpy()[0]  # [num_pods, 10]
            kv_hit_ratios = tensor_data['kv_hit_ratios'].cpu().numpy()[0]  # [num_pods, 1]
            request_features = tensor_data['request_features'].cpu().numpy()[0]  # [3]
            temporal_features = np.array([1], dtype=np.float32)  # Empty for now
            
            # Get previous reward from processed_df (gateway provides this)
            if 'prev_reward' in processed_df.columns:
                prev_reward = float(processed_df['prev_reward'].iloc[0])
            else:
                logger.error(f"scalable_rl_routing_agent, prev_reward not found in processed_df for requestID: {request_id}")
                assert False
            
            # Call infer function from scalable_rl_routing_agent
            infer_start = time.time()
            timeout_in_seconds = 5.0  # 5 second timeout for inference
            pod_idx, infer_from_tensor_overhead_summary = infer(request_id, prev_reward, pod_features, kv_hit_ratios, request_features, temporal_features, BROKER, timeout_in_seconds)
            infer_from_tensor_overhead_summary['scalable_rl_infer'] = time.time() - infer_start
            
            # Build result with actual probabilities
            num_pods = len(sorted_all_pod_ids)
            
            ## TODO: we need action probabilities for debugging
            # if action_probs is not None:
            #     # Use actual probabilities from policy
            #     pod_probabilities = {sorted_all_pod_ids[i]: float(action_probs[i]) for i in range(min(num_pods, len(action_probs)))}
            #     confidence = float(action_probs[pod_idx])
            # else:
            #     # Fallback to uniform
            #     pod_probabilities = {sorted_all_pod_ids[i]: 1.0/num_pods for i in range(num_pods)}
            #     confidence = 1.0/num_pods
            
            # TODO: these are placeholder. we need actual probabilities from the model.
            pod_probabilities = {sorted_all_pod_ids[i]: 1.0/num_pods for i in range(num_pods)}
            confidence = 1.0/num_pods
            result = {
                'selected_pod_index': int(pod_idx),
                'pod_probabilities': pod_probabilities,
                'confidence': confidence,
                'explore_mask': 1,  # RL always explores
                'predicted_latencies': {pod_id: -1 for pod_id in sorted_all_pod_ids},
                'chosen_pod_predicted_latency': -1,
            }
            
            logger.info(f"scalable_rl_routing_agent, requestID: {request_id}, action={pod_idx}, prev_reward={prev_reward:.2f}, confidence={confidence:.3f}, num_pods={num_pods}")
            
        ####################################################################################
        ####################################################################################
        # elif subAlgorithm == 'scalable_rl_agent_old':
        #     # === NEW SCALABLE RL AGENT (pod-count independent) ===
        #     logger.info(f"requestID: {request_id}, subAlgorithm: {subAlgorithm}, Using SCALABLE RL agent (pod-independent) for inference")
            
        #     global SCALABLE_RL_AGENT
            
        #     with SCALABLE_RL_AGENT_LOCK.write():
        #         if SCALABLE_RL_AGENT is None:
        #             # Initialize ONCE - works for any number of pods!
        #             pod_features_t = tensor_data['pod_features']
        #             per_pod_dim = int(pod_features_t.shape[2])  # e.g., 10
        #             kv_hit_t = tensor_data['kv_hit_ratios']
        #             kv_dim = int(kv_hit_t.shape[2])  # e.g., 1
        #             req_features_t = tensor_data['request_features']
        #             req_dim = int(req_features_t.shape[1])  # e.g., 3
                    
        #             # Per-pod dimension = pod_features + kv_hit_ratios
        #             total_per_pod_dim = per_pod_dim + kv_dim
                    
        #             SCALABLE_RL_AGENT = create_scalable_rl_agent(
        #                 per_pod_dim=total_per_pod_dim,  # 11 (10 pod + 1 kv)
        #                 request_dim=req_dim,             # 3
        #                 max_pods=100,                    # Max expected pods
        #                 **RL_MODEL_HYPERPARAMETERS
        #             )
                    
        #             # Load checkpoint if available
        #             ckpt_path = RL_MODEL_HYPERPARAMETERS.get('RL_CHECKPOINT_PATH')
        #             if ckpt_path and os.path.exists(ckpt_path):
        #                 try:
        #                     SCALABLE_RL_AGENT.load(ckpt_path)
        #                     logger.info(f"✅ Loaded scalable RL checkpoint from {ckpt_path}")
        #                 except Exception as e:
        #                     logger.warning(f"⚠️  Failed to load RL checkpoint {ckpt_path}: {e}")
                    
        #             logger.info(f"🚀 Initialized SCALABLE RL agent: per_pod_dim={total_per_pod_dim}, "
        #                       f"request_dim={req_dim}, max_pods=100 (works with ANY #pods!)")
                
        #         current_agent = SCALABLE_RL_AGENT
            
        #     # Inference (no lock needed - thread-safe in new design)
        #     current_agent, result, infer_from_tensor_overhead_summary = infer_scalable_rl_agent(
        #         tensor_data=tensor_data,
        #         request_id=request_id,
        #         sorted_all_pod_ids=sorted_all_pod_ids,
        #         processed_df=processed_df,
        #         rl_agent=current_agent,
        #         hyperparameters=RL_MODEL_HYPERPARAMETERS,
        #         agent_lock=None  # New agent doesn't need lock for prediction
        #     )
        ####################################################################################
        else:
            logger.info(f"requestID: {request_id}, contextual bandit model for inference")
            result, infer_from_tensor_overhead_summary = simpler_contextual_bandit.infer_from_tensor(tensor_data, request_id, MODEL_UPDATED, RL_MODEL_HYPERPARAMETERS, final_model_dir)
            result['predicted_latencies'] = {pod_id: -1 for pod_id in sorted_all_pod_ids}
            result['chosen_pod_predicted_latency'] = -1
        handle_infer_overhead_summary["calling_infer_from_tensor"] = time.time() - infer_from_tensor_start_time
        
        
        remaining_work_start = time.time()
        result["requestID"] = request_id
        result["num_trains"] = NUM_TRAINS
        result["request_timestamp"] = time.time() - first_request_starting_time
        logger.info(f"requestID: {request_id}, inference result: {result}")
        
        # Map the pod index back to the actual pod ID
        selected_pod_index = result['selected_pod_index']
        if selected_pod_index >= len(sorted_all_pod_ids):
            logger.error(f"Selected pod index {selected_pod_index} out of range, defaulting to first pod")
            assert False
        selected_pod_generalpodid = sorted_all_pod_ids[selected_pod_index]
        if selected_pod_generalpodid not in RL_MODEL_HYPERPARAMETERS['generalpodid_to_pod_ip']:
            logger.error(f"selected_pod_generalpodid: {selected_pod_generalpodid} not found in RL_MODEL_HYPERPARAMETERS['generalpodid_to_pod_ip']")
            logger.error(f"RL_MODEL_HYPERPARAMETERS['generalpodid_to_pod_ip']: {RL_MODEL_HYPERPARAMETERS['generalpodid_to_pod_ip']}")
            assert False
        selected_pod_ip = RL_MODEL_HYPERPARAMETERS['generalpodid_to_pod_ip'][selected_pod_generalpodid]
            
        logger.debug(f"selected_pod_generalpodid: {selected_pod_generalpodid}, selected_pod_ip: {selected_pod_ip}, pod_probability: {result['pod_probabilities']}")
        
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
            "predicted_latencies": result['predicted_latencies'],
            "chosen_pod_predicted_latency": result['chosen_pod_predicted_latency'],
        }
        return jsonify(response), 200
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Error in handle_infer: {str(e)}")
        logger.error(f"Traceback: {error_traceback}")
        return jsonify({"error": str(e), "traceback": error_traceback}), 500


def online_train_routine():
    global NUM_TRAINS, MODEL_UPDATED, TOTAL_NUM_DATA, final_model_dir, NUM_NEW_DATA, RL_MODEL_HYPERPARAMETERS, TRAINING_RIGHT_NOW, LATENCY_PREDICTOR, TRAINING_DF, stats_instance
    if TRAINING_RIGHT_NOW:
        logger.info(f"Previous training still in progress, skipping training")
        return
    if NUM_NEW_DATA < MIN_NUM_TRAINING_DATA:
        logger.info(f"Not enough training data available, NUM_NEW_DATA: {NUM_NEW_DATA} < {MIN_NUM_TRAINING_DATA}, wait until enough data are added. NUM_TRAINS: {NUM_TRAINS}, TOTAL_NUM_DATA: {TOTAL_NUM_DATA}")
        return
    TRAINING_RIGHT_NOW = True
    training_start_time = time.time()
    logger.info(f"online_train_routine start, {NUM_TRAINS}th online training with {NUM_NEW_DATA} new training data")
    try:
        # Route to appropriate training function based on model type
        model_type = RL_MODEL_HYPERPARAMETERS['MODEL_TYPE']
        if model_type == 'latency_predictor':
            logger.info(f"Training with latency predictor model on entire dataset (offline + online)")

            # Get a copy of TRAINING_DF for training (thread-safe)
            with TRAINING_DF_LOCK:
                if TRAINING_DF is None or len(TRAINING_DF) == 0:
                    logger.error("TRAINING_DF is empty, cannot train")
                    TRAINING_RIGHT_NOW = False
                    return
                training_df_copy = TRAINING_DF.copy()
                total_samples = len(training_df_copy)

            # logger.info(f"Training on (offline data: {ENCODED_DATA_DIR}, online data: {ENCODED_DATA_DIR}, total data: {total_samples}")
            logger.info(f"Training on total data: {total_samples}")

            # Drop non-numeric metadata columns from offline CSV that are absent online
            metadata_cols_to_drop = ['source_file', 'reward_function_used']
            cols_present_to_drop = [c for c in metadata_cols_to_drop if c in training_df_copy.columns]
            if cols_present_to_drop:
                logger.info(f"Dropping metadata columns not used for training: {cols_present_to_drop}")
                training_df_copy = training_df_copy.drop(columns=cols_present_to_drop)

            ############################################################################
            # Handle missing values proactively to avoid intermittent failures due to dynamic pod columns
            try:
                import numpy as np
                numeric_columns = training_df_copy.select_dtypes(include=[np.number]).columns
                total_missing_numeric = int(training_df_copy[numeric_columns].isna().sum().sum()) if len(numeric_columns) > 0 else 0
                if total_missing_numeric > 0:
                    logger.warning(f"Filling {total_missing_numeric} missing numeric values with 0 for training consistency")
                    training_df_copy[numeric_columns] = training_df_copy[numeric_columns].fillna(0)
                # Optional: warn about numeric columns with high missing rates (diagnostics only)
                missing_pct = training_df_copy[numeric_columns].isnull().mean() * 100 if len(numeric_columns) > 0 else pd.Series()
                high_missing = missing_pct[missing_pct > 5].sort_values(ascending=False)
                if len(high_missing) > 0:
                    topk = list(high_missing.head(10).items())
                    logger.error(f"High-missing columns (>5%) detected (top 10): {[(k, round(v,1)) for k,v in topk]}")
            except Exception as e:
                logger.warning(f"NaN handling diagnostics failed: {e}")
            ############################################################################
            
            # Normalize the entire dataset
            normalizable_features, non_normalizable_features = data_normalizer._get_normalizable_features(
                training_df_copy, RL_MODEL_HYPERPARAMETERS.get('NO_NORMALIZE_FEATURES', []))

            for feature in normalizable_features:
                data_normalizer._normalize_single_feature(training_df_copy, feature, stats_instance, is_training=False)

            # Get sorted pod IDs from the training data (preprocessed CSV format)
            sorted_all_pod_ids = utils.get_sorted_all_pod_ids('processed_csv_columns', training_df_copy.columns.tolist())
            logger.info(f"Training with pods: {sorted_all_pod_ids}")

            # Encode the entire dataset
            encode_start_time = time.time()
            os.makedirs(ENCODED_DATA_DIR, exist_ok=True)
            encoded_training_dir = os.path.join(ENCODED_DATA_DIR, "full_training_data")
            encoding.encode_for_train(sorted_all_pod_ids, training_df_copy, encoded_training_dir, request_features_train, RL_MODEL_HYPERPARAMETERS)
            logger.info(f"Encoded {total_samples} samples to {encoded_training_dir}, encode time: {time.time() - encode_start_time} seconds")

            # Train on the encoded dataset
            train_start_time = time.time()
            latency_predictor.train_latency_predictor(encoded_training_dir, final_model_dir, RL_MODEL_HYPERPARAMETERS)
            logger.info(f"train_latency_predictor done, train time: {time.time() - train_start_time} seconds")

            # Reload model in training thread (non-blocking for inference)
            with LATENCY_PREDICTOR_LOCK.write():
                if LATENCY_PREDICTOR is not None:
                    load_start_time = time.time()
                    LATENCY_PREDICTOR.load(final_model_dir)
                    logger.info(f"Reloaded latency predictor after training, load time: {time.time() - load_start_time} seconds")
        else:
            logger.info(f"Training with contextual bandit model")
            simpler_contextual_bandit.train(ENCODED_DATA_DIR, final_model_dir, RL_MODEL_HYPERPARAMETERS, ENABLE_ONLINE_LEARNING)
            logger.info(f"train_contextual_bandit done")
    except Exception as e:
        import traceback
        logger.error(f"Error during training: {e}")
        logger.error(traceback.format_exc())
        TRAINING_RIGHT_NOW = False
        return
    logger.info(f"Successfully completed online_train_routine done, {NUM_TRAINS}th online training with {NUM_NEW_DATA} new training data, took {time.time() - training_start_time} seconds")
    MODEL_UPDATED = True
    TRAINING_RIGHT_NOW = False
    NUM_TRAINS += 1
    NUM_NEW_DATA = 0


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

def rl_update_worker():
    """Background worker thread for RL agent updates"""
    global RL_AGENT, RL_AGENT_LOCK
    logger.info("RL update worker thread started")
    
    while not RL_UPDATE_SHUTDOWN.is_set():
        try:
            # Wait for update request with timeout
            update_request = RL_UPDATE_QUEUE.get(timeout=1.0)
            
            if update_request is None:  # Shutdown signal
                break
                
            # Perform the update with WRITE lock (exclusive access)
            # This blocks concurrent predictions to prevent PyTorch read+write races
            with RL_AGENT_LOCK.write():
                if RL_AGENT is not None:
                    try:
                        n_steps = update_request.get('n_steps', 32)
                        RL_AGENT.update_online(n_steps=n_steps)
                        logger.debug(f"RL agent updated with {n_steps} steps")
                    except Exception as e:
                        logger.error(f"Error updating RL agent: {e}")
            
            RL_UPDATE_QUEUE.task_done()
            
        except queue.Empty:
            continue  # Timeout, check shutdown flag
        except Exception as e:
            logger.error(f"Error in RL update worker: {e}")
    
    logger.info("RL update worker thread stopped")


def start_rl_update_worker():
    """Start the RL update worker thread"""
    global RL_UPDATE_THREAD
    if RL_UPDATE_THREAD is None or not RL_UPDATE_THREAD.is_alive():
        RL_UPDATE_THREAD = threading.Thread(target=rl_update_worker, daemon=True)
        RL_UPDATE_THREAD.start()
        logger.info("Started RL update worker thread")


def stop_rl_update_worker():
    """Stop the RL update worker thread"""
    global RL_UPDATE_THREAD
    if RL_UPDATE_THREAD and RL_UPDATE_THREAD.is_alive():
        logger.info("Stopping RL update worker thread...")
        RL_UPDATE_SHUTDOWN.set()
        
        # Send shutdown signal
        try:
            RL_UPDATE_QUEUE.put(None, timeout=1.0)
        except queue.Full:
            pass
        
        # Wait for thread to finish
        RL_UPDATE_THREAD.join(timeout=5.0)
        if RL_UPDATE_THREAD.is_alive():
            logger.warning("RL update worker thread did not stop gracefully")
        else:
            logger.info("RL update worker thread stopped successfully")


def scalable_rl_training_worker():
    """
    Background worker thread for scalable RL agent training.
    
    This continuously runs the RL training loop, which:
    1. Pulls requests from the environment (blocks until request available from BROKER)
    2. Predicts action (pod selection)
    3. Routes request (sets decision in BROKER, unblocking /infer)
    4. Collects experience and updates policy
    """
    global SCALABLE_RL_AGENT, SCALABLE_RL_TRAINING_SHUTDOWN
    logger.info("🏋️  Scalable RL training worker thread started")
    
    try:
        # Training loop - runs until shutdown
        while not SCALABLE_RL_TRAINING_SHUTDOWN.is_set():
            if SCALABLE_RL_AGENT is not None:
                try:
                    # Run training for a batch of steps
                    # The agent's learn() will internally call env.step() which pulls from BROKER
                    logger.info(f"{PURPLE_COLOR}Training scalable RL agent...{RESET_COLOR}")
                    # Create proper checkpoint file path (not just directory)
                    checkpoint_file = os.path.join(RL_MODEL_HYPERPARAMETERS['CHECKPOINT_DIR'], 'scalable_rl_agent')
                    
                    eval_freq = (RL_MODEL_HYPERPARAMETERS['num_requests_per_episode'] + 1) * RL_MODEL_HYPERPARAMETERS['num_episodes_per_iteration'] # how often to evaluate the model, the eval will be triggered every eval_freq steps. Hence, eval_freq should be the number of requests per iteration
                    SCALABLE_RL_AGENT.train(
                        save_path=checkpoint_file,
                        eval_freq=eval_freq,
                        n_eval_episodes=RL_MODEL_HYPERPARAMETERS['n_eval_episodes'],
                        )
                except Exception as e:
                    if not SCALABLE_RL_TRAINING_SHUTDOWN.is_set():
                        logger.error(f"Error in scalable RL training loop: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        time.sleep(1)  # Avoid tight error loop
            else:
                time.sleep(1)  # Wait for agent initialization
    except Exception as e:
        logger.error(f"Fatal error in scalable RL training worker: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        logger.info("Scalable RL training worker thread stopped")


def start_scalable_rl_training_worker():
    """Start the scalable RL training worker thread"""
    global SCALABLE_RL_TRAINING_THREAD
    if SCALABLE_RL_TRAINING_THREAD is None or not SCALABLE_RL_TRAINING_THREAD.is_alive():
        SCALABLE_RL_TRAINING_THREAD = threading.Thread(
            target=scalable_rl_training_worker,
            daemon=True,
            name="ScalableRLTraining"
        )
        SCALABLE_RL_TRAINING_THREAD.start()
        logger.info("✅ Started scalable RL training worker thread")


def stop_scalable_rl_training_worker():
    """Stop the scalable RL training worker thread"""
    global SCALABLE_RL_TRAINING_THREAD
    if SCALABLE_RL_TRAINING_THREAD and SCALABLE_RL_TRAINING_THREAD.is_alive():
        logger.info("Stopping scalable RL training worker thread...")
        SCALABLE_RL_TRAINING_SHUTDOWN.set()
        
        # Wait for thread to finish
        SCALABLE_RL_TRAINING_THREAD.join(timeout=5.0)
        if SCALABLE_RL_TRAINING_THREAD.is_alive():
            logger.warning("Scalable RL training worker thread did not stop gracefully")
        else:
            logger.info("Scalable RL training worker thread stopped successfully")


def queue_rl_update(n_steps=32):
    """Queue an RL agent update request (non-blocking)"""
    if ENABLE_ONLINE_LEARNING and RL_AGENT is not None:
        try:
            update_request = {'n_steps': n_steps}
            RL_UPDATE_QUEUE.put_nowait(update_request)
            logger.debug(f"Queued RL update request with {n_steps} steps")
        except queue.Full:
            logger.warning("RL update queue is full, skipping update request")


def get_current_cluster_features():
    """
    Fetch real-time cluster state for experience completion (next_obs).
    
    This mirrors the feature extraction done in /infer, but fetches CURRENT state.
    
    Returns:
        pod_features: [num_pods, 10] - Current pod metrics
        kv_hit_ratios: [num_pods, 1] - Current cache hit ratios (zeros for now)
        request_features: [3] - Dummy request features (not used for next_obs)
    """
    global RL_MODEL_HYPERPARAMETERS, stats_instance
    
    try:
        # Get current running pods (same as init())
        running_pods = utils.get_running_pods_by_label(POD_LABEL_SELECTOR)
        sorted_pod_ips = utils.fetch_running_pod_ips(running_pods)
        num_pods = len(sorted_pod_ips)
        
        if num_pods == 0:
            logger.warning("No running pods found for cluster state fetch")
            # Return dummy state
            return (
                np.zeros((1, 10), dtype=np.float32),
                np.zeros((1, 1), dtype=np.float32),
                np.zeros(3, dtype=np.float32)
            )
        
        # Initialize arrays for pod features
        pod_features = np.zeros((num_pods, 10), dtype=np.float32)
        
        # Get pod_ip to general_pod_id mapping
        pod_ip_to_generalpodid = RL_MODEL_HYPERPARAMETERS.get('pod_ip_to_generalpodid', {})
        pod_ip_to_gpu_model_encoded = RL_MODEL_HYPERPARAMETERS.get('pod_ip_to_gpu_model_encoded', {})
        
        # For each pod, fetch current metrics
        for i, pod_ip in enumerate(sorted_pod_ips):
            try:
                # These are the same features used in preprocess.py
                # The exact column order depends on your feature extraction
                # Adjust indices based on your actual feature schema
                
                # Example feature extraction (adjust to your schema):
                # Column 0: running_requests (from inflight)
                inflight_requests = utils.GetInflightRequestsForPod(pod_ip) if hasattr(utils, 'GetInflightRequestsForPod') else 0
                pod_features[i, 0] = float(inflight_requests)
                
                # Column 1: queue_length (from waiting requests)
                # Use stored metrics or default to 0
                pod_features[i, 1] = 0  # Placeholder - implement if you track queue length
                
                # Column 2-3: GPU/CPU KV cache usage
                # These would come from vLLM metrics if available
                pod_features[i, 2] = 0  # GPU cache usage (implement if tracked)
                pod_features[i, 3] = 0  # CPU cache usage (implement if tracked)
                
                # Column 4-5: Num requests running/waiting
                pod_features[i, 4] = 0  # Running requests (implement if tracked)
                pod_features[i, 5] = 0  # Waiting requests (implement if tracked)
                
                # Column 6-7: Prefill/decode token counts
                prefill_tokens = utils.GetNumPrefillTokensForPod(pod_ip) if hasattr(utils, 'GetNumPrefillTokensForPod') else 0
                decode_tokens = utils.GetNumDecodeTokensForPod(pod_ip) if hasattr(utils, 'GetNumDecodeTokensForPod') else 0
                pod_features[i, 6] = float(prefill_tokens)
                pod_features[i, 7] = float(decode_tokens)
                
                # Column 8: GPU type (encoded)
                if pod_ip in pod_ip_to_gpu_model_encoded:
                    pod_features[i, 8] = float(pod_ip_to_gpu_model_encoded[pod_ip])
                else:
                    pod_features[i, 8] = 0.0
                
                # Column 9: Availability (1.0 = available)
                pod_features[i, 9] = 1.0  # Assume available (or check pod status)
                
            except Exception as e:
                logger.warning(f"Error fetching metrics for pod {pod_ip}: {e}")
                # Keep zeros for this pod
        
        # Normalize features using stats_instance (same as /infer)
        if stats_instance is not None and stats_instance.get_max_count() > 0:
            # Get normalizable feature names (adjust to your schema)
            # This should match the features in your processed_df
            feature_names = [
                'running_requests', 'queue_length', 'gpu_cache', 'cpu_cache',
                'num_running', 'num_waiting', 'prefill_tokens', 'decode_tokens',
                'gpu_type', 'availability'
            ]
            
            for col_idx, feature_name in enumerate(feature_names):
                if feature_name in stats_instance.feature_stats:
                    stats = stats_instance.feature_stats[feature_name]
                    # Z-score normalization: (x - mean) / std
                    if stats.std > 0:
                        pod_features[:, col_idx] = (pod_features[:, col_idx] - stats.mean) / stats.std
        
        # KV hit ratios - for next_obs, we don't have per-request cache hit info
        # So we use zeros (or could use average cache hit rates if tracked)
        kv_hit_ratios = np.zeros((num_pods, 1), dtype=np.float32)
        
        # Request features - dummy (not used for next_obs, but needed for consistency)
        request_features = np.zeros(3, dtype=np.float32)
        
        logger.debug(f"Fetched current cluster state: {num_pods} pods")
        return pod_features, kv_hit_ratios, request_features
        
    except Exception as e:
        logger.error(f"Error in get_current_cluster_features: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # Return minimal valid state
        return (
            np.zeros((1, 10), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
            np.zeros(3, dtype=np.float32)
        )


def graceful_shutdown(sig=None, frame=None):
    """Handle graceful shutdown when receiving SIGTERM or SIGINT"""
    logger.info(f"Received signal {sig if sig else 'shutdown'}, shutting down gracefully...")
    
    # Stop RL update worker
    stop_rl_update_worker()
    
    # Stop scalable RL training worker
    stop_scalable_rl_training_worker()
    
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


def init_test_mode():
    """Initialize in test mode with real model config but mock Kubernetes setup"""
    global RL_MODEL_HYPERPARAMETERS, stats_instance
    
    logger.info(f"{GREEN_COLOR}Initializing in TEST MODE{RESET_COLOR}")
    logger.info(f"{GREEN_COLOR}Loading REAL model config and normalization stats{RESET_COLOR}")
    
    # Load real hyperparameters from model_config.json
    RL_MODEL_HYPERPARAMETERS = {}
    RL_MODEL_HYPERPARAMETERS['TTFT_REWARD_WEIGHT'] = TTFT_REWARD_WEIGHT
    RL_MODEL_HYPERPARAMETERS['EXPLORATION_ENABLED'] = EXPLORATION_ENABLED
    RL_MODEL_HYPERPARAMETERS['ENABLE_ONLINE_LEARNING'] = ENABLE_ONLINE_LEARNING
    
    logger.info(f"Loading hyperparameters from: {hyperparameter_file_path}")
    utils.load_rl_hyperparameters(hyperparameter_file_path, RL_MODEL_HYPERPARAMETERS)
    
    logger.info(f"Model type: {RL_MODEL_HYPERPARAMETERS['MODEL_TYPE']}")
    logger.info(f"Latency metric: {RL_MODEL_HYPERPARAMETERS.get('LATENCY_METRIC', 'ttft')}")
    
    # Load real normalization statistics from CSV file
    logger.info(f"Loading normalization statistics from: {feature_normalization_stats_file}")
    if os.path.exists(feature_normalization_stats_file):
        try:
            stats_instance = data_normalizer.FeatureStats.load_from_csv(feature_normalization_stats_file)
            if stats_instance is not None:
                logger.info(f"Successfully loaded stats for {len(stats_instance.feature_stats)} features")
            else:
                logger.error("Failed to load normalization statistics")
                assert False
        except Exception as e:
            logger.error(f"Failed to load normalization statistics: {e}")
            import traceback
            logger.error(traceback.format_exc())
            assert False
    else:
        logger.error(f"Normalization statistics file not found: {feature_normalization_stats_file}")
        assert False
    
    # Initialize with mock pod configuration (will be overridden per test)
    setup_mock_environment(5)  # Default 5 pods
    
    logger.info(f"{GREEN_COLOR}Test mode initialization complete!{RESET_COLOR}")
    logger.info(f"  - Using real model config from: {hyperparameter_file_path}")
    logger.info(f"  - Using real normalization stats from: {feature_normalization_stats_file}")
    logger.info(f"  - Using real latency predictor model from: {final_model_dir}")
    logger.info(f"  - Mock pod configurations: will be set per test")


def init():
    global RL_MODEL_HYPERPARAMETERS, stats_instance
    if RL_MODEL_HYPERPARAMETERS is None:

        logger.info(f"{GREEN_COLOR}RL_MODEL_HYPERPARAMETERS is None{RESET_COLOR}")

        RL_MODEL_HYPERPARAMETERS = {}
        RL_MODEL_HYPERPARAMETERS['TTFT_REWARD_WEIGHT'] = TTFT_REWARD_WEIGHT
        RL_MODEL_HYPERPARAMETERS['EXPLORATION_ENABLED'] = EXPLORATION_ENABLED
        RL_MODEL_HYPERPARAMETERS['ENABLE_ONLINE_LEARNING'] = ENABLE_ONLINE_LEARNING
        logger.info("Loading RL hyperparameters from model_config.json")
        utils.load_rl_hyperparameters(hyperparameter_file_path, RL_MODEL_HYPERPARAMETERS)
        model_type = RL_MODEL_HYPERPARAMETERS.get('MODEL_TYPE', 'contextual_bandit')
        logger.info(f"Model type configured: {model_type}")
        if model_type == 'latency_predictor':
            latency_metric = RL_MODEL_HYPERPARAMETERS.get('LATENCY_METRIC', 'ttft')
            logger.info(f"Latency metric for prediction: {latency_metric}")
        else:
            logger.info(f"Using contextual bandit with exploration rate: {RL_MODEL_HYPERPARAMETERS.get('exploration_rate', 0)}")
        # Test permissions first
        logger.info("Testing Kubernetes API permissions...")
        if not test_kubernetes_permissions():
            logger.error("Insufficient Kubernetes permissions - using fallback GPU mapping")
            assert False
        running_vllm_pods = utils.get_running_pods_by_label(POD_LABEL_SELECTOR)
        sorted_running_pod_ips = utils.fetch_running_pod_ips(running_vllm_pods)
        pod_ip_to_generalpodid = utils.create_pod_ip_to_generalpodid_mapping(sorted_running_pod_ips)
        generalpodid_to_pod_ip = {}
        for pod_ip, generalpodid in pod_ip_to_generalpodid.items():
            generalpodid_to_pod_ip[generalpodid] = pod_ip
        generalpodid_to_gpu_model = utils.fetch_generalpodid_to_gpu_model(running_vllm_pods, pod_ip_to_generalpodid)
        pod_ip_to_gpu_model, pod_ip_to_gpu_model_encoded = utils.create_pod_ip_to_gpu_model_mapping(generalpodid_to_gpu_model, pod_ip_to_generalpodid)
        
        logger.info(f"POD_LABEL_SELECTOR: {POD_LABEL_SELECTOR}")
        logger.info(f"len(sorted_running_pod_ips): {len(sorted_running_pod_ips)}, sorted_running_pod_ips: {sorted_running_pod_ips}")
        logger.info(f"pod_ip_to_generalpodid: {pod_ip_to_generalpodid}")
        logger.info(f"generalpodid_to_gpu_model: {generalpodid_to_gpu_model}")
        logger.info(f"pod_ip_to_gpu_model: {pod_ip_to_gpu_model}")
        logger.info(f"pod_ip_to_gpu_model_encoded: {pod_ip_to_gpu_model_encoded}")

        RL_MODEL_HYPERPARAMETERS['pod_ip_to_generalpodid'] = pod_ip_to_generalpodid
        RL_MODEL_HYPERPARAMETERS['generalpodid_to_pod_ip'] = generalpodid_to_pod_ip
        logger.info(f"RL_MODEL_HYPERPARAMETERS['generalpodid_to_pod_ip']: {RL_MODEL_HYPERPARAMETERS['generalpodid_to_pod_ip']}")
        RL_MODEL_HYPERPARAMETERS['sorted_running_pod_ips'] = sorted_running_pod_ips
        RL_MODEL_HYPERPARAMETERS['pod_ip_to_gpu_model'] = pod_ip_to_gpu_model
        RL_MODEL_HYPERPARAMETERS['pod_ip_to_gpu_model_encoded'] = pod_ip_to_gpu_model_encoded
        RL_MODEL_HYPERPARAMETERS['generalpodid_to_gpu_model'] = generalpodid_to_gpu_model
        # Additional mappings for GPU features expected by preprocess/encoding
        RL_MODEL_HYPERPARAMETERS['pod_gpu_mapping'] = generalpodid_to_gpu_model
        GPU_MODEL_TO_ENCODE = {
            'NVIDIA-L20': 0,
            'NVIDIA-L40': 1,
            'NVIDIA-A10': 2,
            'NVIDIA-A100': 3,
            'NVIDIA-H100': 4,
        }
        pod_gpu_id_mapping = {}
        for generalpodid, gpu_model in generalpodid_to_gpu_model.items():
            if gpu_model in GPU_MODEL_TO_ENCODE:
                pod_gpu_id_mapping[generalpodid] = GPU_MODEL_TO_ENCODE[gpu_model]
            else:
                logger.error(f"Unknown GPU model for {generalpodid}: {gpu_model}")
                assert False
        RL_MODEL_HYPERPARAMETERS['pod_gpu_id_mapping'] = pod_gpu_id_mapping
        
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
    
    # # Add checkpointing configuration to hyperparameters
    # RL_MODEL_HYPERPARAMETERS['CHECKPOINT_INTERVAL_STEPS'] = 100
    # RL_MODEL_HYPERPARAMETERS['CHECKPOINT_DIR'] = os.path.join(final_model_dir, 'checkpoints')
    
    # # Create checkpoint directory if it doesn't exist
    # os.makedirs(RL_MODEL_HYPERPARAMETERS['CHECKPOINT_DIR'], exist_ok=True)
    # logger.info(f"Checkpoint directory: {RL_MODEL_HYPERPARAMETERS['CHECKPOINT_DIR']}")
    # logger.info(f"Checkpointing every {RL_MODEL_HYPERPARAMETERS['CHECKPOINT_INTERVAL_STEPS']} steps")

    # Load offline training data for online learning
    global TRAINING_DF
    if ENABLE_ONLINE_LEARNING:
        offline_csv_path = "/app/offline_training_data.csv"
        if os.path.exists(offline_csv_path):
            try:
                with TRAINING_DF_LOCK:
                    TRAINING_DF = pd.read_csv(offline_csv_path)
                    logger.info(f"✅ Loaded offline training data: {len(TRAINING_DF)} samples from {offline_csv_path}")
                    logger.info(f"   Columns: {list(TRAINING_DF.columns[:10])}...")  # Show first 10 columns
            except Exception as e:
                logger.error(f"Failed to load offline training data: {e}")
                TRAINING_DF = pd.DataFrame()
                logger.warning("Starting with empty training dataframe")
        else:
            logger.warning(f"Offline training data not found at {offline_csv_path}")
            logger.warning("Online learning will start from scratch with only new data")
            TRAINING_DF = pd.DataFrame()
    else:
        logger.info("Online learning disabled, skipping offline data load")
    
    # Initialize scalable RL agent if configured

    # logger.info(f"{BLUE_COLOR}model_type: {RL_MODEL_HYPERPARAMETERS['MODEL_TYPE']}{RESET_COLOR}")


    # # if RL_MODEL_HYPERPARAMETERS['MODEL_TYPE'] == 'scalable_rl_agent':
    # if RL_MODEL_HYPERPARAMETERS['MODEL_TYPE'] == 'scalable_rl_agent':
    #     import scalable_rl_routing_agent
    #     from scalable_rl_routing_agent import BROKER
        
    #     logger.info("Initializing scalable_rl_agent, scalable RL agent...")
    #     global SCALABLE_RL_AGENT, BROKER_LOCK
    #     with BROKER_LOCK.write():
    #         # Get number of pods from current running pods
    #         num_pods = len(sorted_running_pod_ips)
    #         logger.info(f"Creating scalable RL agent with {num_pods} pods")
            
    #         # # Create agent with hyperparameters
    #         # SCALABLE_RL_AGENT = create_scalable_rl_agent(
    #         #     per_pod_dim=RL_MODEL_HYPERPARAMETERS.get('per_pod_dim', 8),
    #         #     request_dim=RL_MODEL_HYPERPARAMETERS.get('request_dim', 3),
    #         #     max_pods=RL_MODEL_HYPERPARAMETERS.get('max_pods', 100),
    #         #     learning_rate=RL_MODEL_HYPERPARAMETERS.get('learning_rate', 3e-4),
    #         #     reward_decay_factor=RL_MODEL_HYPERPARAMETERS.get('reward_decay_factor', 1.0),
    #         #     gae_lambda=RL_MODEL_HYPERPARAMETERS.get('gae_lambda', 0.95),
    #         #     n_steps=RL_MODEL_HYPERPARAMETERS.get('n_steps', 256),
    #         #     horizon=RL_MODEL_HYPERPARAMETERS.get('horizon', 1024),
    #         #     batch_size=RL_MODEL_HYPERPARAMETERS.get('batch_size', 64),
    #         #     last_layer_dim_vf=RL_MODEL_HYPERPARAMETERS.get('last_layer_dim_vf', 1),
    #         #     rl=RL_MODEL_HYPERPARAMETERS.get('rl_algorithm', 'PPO'),
    #         #     static_num_pods=True,
    #         # )
            
    #         SCALABLE_RL_AGENT = scalable_rl_routing_agent.ScalableRLRoutingAgent(
    #             per_pod_dim=8, 
    #             # per_pod_dim=11, 
    #             request_dim=3, 
    #             max_pods=100, 
    #             num_requests_per_episode=RL_MODEL_HYPERPARAMETERS['num_requests_per_episode'], 
    #             num_episodes_per_iteration=RL_MODEL_HYPERPARAMETERS['num_episodes_per_iteration'],
    #             num_iterations=RL_MODEL_HYPERPARAMETERS['num_iterations'],   
    #             rl=RL_MODEL_HYPERPARAMETERS['rl_algorithm'], 
    #             static_num_pods=True, 
    #             learning_rate=RL_MODEL_HYPERPARAMETERS['rl_learning_rate'], 
    #             hidden_dim=RL_MODEL_HYPERPARAMETERS['hidden_dim'], 
    #             gamma=RL_MODEL_HYPERPARAMETERS['gamma'], 
    #             gae_lambda=RL_MODEL_HYPERPARAMETERS['gae_lambda'], 
    #             tb_log_dir=os.path.join(RL_MODEL_HYPERPARAMETERS['CHECKPOINT_DIR'], 'tb_logs'), 
    #             batch_size=RL_MODEL_HYPERPARAMETERS['batch_size'], 
    #             n_epochs=RL_MODEL_HYPERPARAMETERS['training_epochs'], 
    #             clip_range=RL_MODEL_HYPERPARAMETERS['clip_range'], 
    #             entropy_coeff=RL_MODEL_HYPERPARAMETERS['entropy_coeff'], 
    #             vf_coef=RL_MODEL_HYPERPARAMETERS['vf_coef'], 
    #             max_grad_norm=RL_MODEL_HYPERPARAMETERS['max_grad_norm'], 
    #             last_layer_dim_pi=RL_MODEL_HYPERPARAMETERS['last_layer_dim_pi'], 
    #             last_layer_dim_vf=RL_MODEL_HYPERPARAMETERS['last_layer_dim_vf'], 
    #             use_prioritized_replay=False, 
    #             buffer_size=RL_MODEL_HYPERPARAMETERS['buffer_size'], 
    #             priority_alpha=RL_MODEL_HYPERPARAMETERS['priority_alpha'], 
    #             priority_beta=RL_MODEL_HYPERPARAMETERS['priority_beta'],
    #             lr_scheduler_type=RL_MODEL_HYPERPARAMETERS['lr_scheduler_type'],
    #             load_tb_best='/app/final_model/init_model/best_model.zip',
    #             )
            
    #         logger.info(f"scalable_rl_routing_agent, Scalable RL agent created successfully")
            
    #         # Load checkpoint if available
    #         checkpoint_path = RL_MODEL_HYPERPARAMETERS.get('RL_CHECKPOINT_PATH')
    #         if checkpoint_path and os.path.exists(checkpoint_path):
    #             try:
    #                 SCALABLE_RL_AGENT.load(checkpoint_path)
    #                 logger.info(f"scalable_rl_routing_agent, Loaded scalable RL checkpoint from {checkpoint_path}")
    #             except Exception as e:
    #                 logger.warning(f"scalable_rl_routing_agent, Failed to load checkpoint: {e}")
        
    #     # Start training thread
    #     logger.info(f"{GREEN_COLOR}Starte scalable rl training worker in init()...{RESET_COLOR}")
    #     start_scalable_rl_training_worker()
    #     logger.info(f"{GREEN_COLOR}Scalable RL agent initialized and training thread started{RESET_COLOR}")
    #     logger.info("scalable_rl_routing_agent, Scalable RL agent initialized and training thread started")


def periodic_checkpoint_scalable_rl():
    """
    Periodically checkpoint the scalable RL agent with comprehensive metadata.
    
    This runs in a background thread and saves:
    - Model weights
    - Training progress (steps, episodes)
    - Performance metrics (rewards, success rate)
    - Buffer statistics
    - Human-readable JSON metadata
    """
    global SCALABLE_RL_AGENT, RL_MODEL_HYPERPARAMETERS
    
    try:
        if SCALABLE_RL_AGENT is None:
            return  # Agent not initialized yet
        
        with SCALABLE_RL_AGENT_LOCK.read():
            # Check if it's time to checkpoint
            checkpoint_interval = RL_MODEL_HYPERPARAMETERS.get('CHECKPOINT_INTERVAL_STEPS', 1000)
            total_steps = SCALABLE_RL_AGENT.total_steps
            
            # Checkpoint at regular intervals
            if total_steps > 0 and total_steps % checkpoint_interval < 64:  # 64 = typical batch size
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                checkpoint_name = f"scalable_rl_step_{total_steps}_{timestamp}"
                checkpoint_path = os.path.join(RL_MODEL_HYPERPARAMETERS['CHECKPOINT_DIR'], checkpoint_name)
                
                try:
                    # Upgrade to write lock for saving
                    with SCALABLE_RL_AGENT_LOCK.write():
                        logger.info(f"scalable_rl_routing_agent, Checkpointing scalable RL agent at step {total_steps}")
                        
                        # Save with comprehensive metadata
                        SCALABLE_RL_AGENT.save(
                            checkpoint_path,
                            save_buffer=False  # Don't save buffer by default (can be large)
                        )
                        
                        # Log metrics
                        metrics = SCALABLE_RL_AGENT.get_metrics()
                        if metrics.get('reward_stats'):
                            logger.info(f"scalable_rl_routing_agent, Avg reward (recent 100): {metrics['reward_stats']['avg_reward_recent']:.3f}")
                            logger.info(f"scalable_rl_routing_agent, Success rate: {metrics['success_rate']:.2%}")
                        
                        # Clean up old checkpoints (keep only last 5)
                        # cleanup_old_checkpoints(RL_MODEL_HYPERPARAMETERS['CHECKPOINT_DIR'], keep_latest=5)
                        
                except Exception as e:
                    logger.error(f"scalable_rl_routing_agent, Failed to save checkpoint: {e}")
    
    except Exception as e:
        logger.error(f"scalable_rl_routing_agent, Error in periodic_checkpoint_scalable_rl: {e}")


def cleanup_old_checkpoints(checkpoint_dir, keep_latest=5):
    """
    Remove old checkpoints, keeping only the most recent ones.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_latest: Number of recent checkpoints to keep
    """
    try:
        # Find all checkpoint files (main model file, not metadata)
        import glob
        checkpoint_files = []
        
        # Look for .zip files (SB3 PPO saves as .zip)
        for f in glob.glob(os.path.join(checkpoint_dir, "scalable_rl_step_*.zip")):
            # Get modification time
            mtime = os.path.getmtime(f)
            checkpoint_files.append((f, mtime))
        
        # Sort by modification time (newest first)
        checkpoint_files.sort(key=lambda x: x[1], reverse=True)
        
        # Delete old ones
        if len(checkpoint_files) > keep_latest:
            for checkpoint_path, _ in checkpoint_files[keep_latest:]:
                try:
                    # Remove checkpoint and associated files
                    base_path = checkpoint_path.replace('.zip', '')
                    
                    # Remove .zip, _metadata.pkl, _metadata.json
                    for ext in ['.zip', '_metadata.pkl', '_metadata.json', '_buffer.pkl']:
                        file_to_remove = base_path + ext if ext != '.zip' else checkpoint_path
                        if os.path.exists(file_to_remove):
                            os.remove(file_to_remove)
                            logger.info(f"scalable_rl_routing_agent, Removed old checkpoint: {os.path.basename(file_to_remove)}")
                except Exception as e:
                    logger.warning(f"scalable_rl_routing_agent, Failed to remove old checkpoint {checkpoint_path}: {e}")
    except Exception as e:
        logger.error(f"scalable_rl_routing_agent, Error cleaning up old checkpoints: {e}")


def create_mock_request_data(num_pods=7, request_id="test_req_001"):
    """
    Create mock request data for testing inference endpoint.
    Matches the actual gateway format from test_request.json
    
    Args:
        num_pods: Number of pods to simulate
        request_id: Request ID for tracking
    
    Returns:
        Mock log message string that mimics real gateway request format
    """
    global RL_MODEL_HYPERPARAMETERS
    
    # Use the pod IPs from RL_MODEL_HYPERPARAMETERS setup by setup_mock_environment
    sorted_pod_ips = RL_MODEL_HYPERPARAMETERS.get('sorted_running_pod_ips', [])
    if not sorted_pod_ips or len(sorted_pod_ips) != num_pods:
        # Fallback: generate mock pod IPs
        sorted_pod_ips = [f"10.0.0.{100+i}" for i in range(num_pods)]
    
    # Create JSON dictionaries for pod metrics
    kv_cache_hit_ratios = {}
    num_inflight_requests = {}
    vllm_gpu_kv_cache_usage = {}
    vllm_cpu_kv_cache_usage = {}
    vllm_num_requests_running = {}
    vllm_num_requests_waiting = {}
    num_prefill_tokens = {}
    num_decode_tokens = {}
    
    for pod_ip in sorted_pod_ips:
        kv_cache_hit_ratios[pod_ip] = round(np.random.uniform(0, 1), 3)
        num_inflight_requests[pod_ip] = np.random.randint(0, 5)
        vllm_gpu_kv_cache_usage[pod_ip] = round(np.random.uniform(0.3, 0.8), 3)
        vllm_cpu_kv_cache_usage[pod_ip] = round(np.random.uniform(0.1, 0.5), 3)
        vllm_num_requests_running[pod_ip] = np.random.randint(0, 5)
        vllm_num_requests_waiting[pod_ip] = np.random.randint(0, 3)
        num_prefill_tokens[pod_ip] = np.random.randint(0, 1000)
        num_decode_tokens[pod_ip] = np.random.randint(0, 2000)
    
    # Build mock log message in actual gateway format
    # Format: **@latency_metrics@requestID@...@numInputTokens@...@allPodsKvCacheHitRatios@{...}@...
    log_parts = []
    log_parts.append("**@latency_metrics@")
    log_parts.append(f"requestID@{request_id}@")
    log_parts.append("request_start_time@1000@")
    log_parts.append("request_end_time@2000@")
    log_parts.append(f"selectedpod@{sorted_pod_ips[0]}@")  # Dummy selected pod
    log_parts.append("ttft@100@")
    log_parts.append("avg_tpot@50@")
    log_parts.append("total_decode_time@500@")
    log_parts.append("e2e@1000@")
    log_parts.append("numInputTokens@150@")
    log_parts.append("numOutputTokens@100@")
    log_parts.append("numTotalTokens@250@")
    
    # Add pod metrics as JSON dictionaries
    log_parts.append(f"allPodsKvCacheHitRatios@{json.dumps(kv_cache_hit_ratios)}@")
    log_parts.append(f"numInflightRequestsAllPods@{json.dumps(num_inflight_requests)}@")
    log_parts.append(f"vllmGPUKVCacheUsage@{json.dumps(vllm_gpu_kv_cache_usage)}@")
    log_parts.append(f"vllmCPUKVCacheUsage@{json.dumps(vllm_cpu_kv_cache_usage)}@")
    log_parts.append(f"vllmNumRequestsRunning@{json.dumps(vllm_num_requests_running)}@")
    log_parts.append(f"vllmNumRequestsWaiting@{json.dumps(vllm_num_requests_waiting)}@")
    log_parts.append("podMetricsLastSecond@{}@")
    log_parts.append(f"numPrefillTokensForAllPods@{json.dumps(num_prefill_tokens)}@")
    log_parts.append(f"numDecodeTokensForAllPods@{json.dumps(num_decode_tokens)}@")
    log_parts.append("subAlgorithm@latency_predictor@")
    log_parts.append("prev_reward@0.5")
    
    return "".join(log_parts)


def run_scalability_test(test_name, num_pods_list, rps_list, duration_per_test=10):
    """
    Run scalability tests with different pod counts and RPS values.
    
    Args:
        test_name: Name of the test suite
        num_pods_list: List of pod counts to test (e.g., [5, 10, 20, 50])
        rps_list: List of RPS values to test (e.g., [10, 50, 100, 200])
        duration_per_test: Duration in seconds for each test configuration
    """
    import concurrent.futures
    import threading
    from collections import defaultdict
    
    logger.info(f"\n{'='*80}")
    logger.info(f"SCALABILITY TEST: {test_name}")
    logger.info(f"{'='*80}\n")
    
    results = []
    
    for num_pods in num_pods_list:
        # Reinitialize with mock pod configuration
        logger.info(f"\n{BLUE_COLOR}Configuring for {num_pods} pods...{RESET_COLOR}")
        setup_mock_environment(num_pods)
        
        for target_rps in rps_list:
            logger.info(f"\n{GREEN_COLOR}Testing: {num_pods} pods @ {target_rps} RPS{RESET_COLOR}")
            
            # Calculate total requests to send
            total_requests = int(target_rps * duration_per_test)
            request_interval = 1.0 / target_rps
            
            # Metrics collection
            latencies = []
            errors = 0
            lock = threading.Lock()
            
            # Pipeline breakdown metrics
            breakdown_metrics = {
                'preprocess': [],
                'normalization': [],
                'encoding': [],
                'inference': []
            }
            
            # Detailed subcomponent breakdowns for each category
            preprocess_breakdown = {
                'json_parse': [],
                'column_check': [],
                'numeric_conversion': [],
                'time_logging': [],
                'get_value': [],
                'create_df': [],
                'pod_index': [],
                'fill_nan': [],
                'preprocess_unified': [],  # E2E total (contains all above)
            }
            
            encoding_breakdown = {
                'extract_pod_columns': [],
                'classify_feature_timing': [],
                'extract_request_feature': [],
                'vectorized_extraction': [],
                'process_pod_feature': [],
                'extract_actions': [],
                'positional_encoding': [],
                'interaction_features': [],
                'post_process': [],
                'end_to_end': [],  # E2E total (contains all above)
            }
            
            inference_breakdown = {
                'prepare_tensors': [],
                'model_inference': [],
                'format_results': [],
            }
            
            def send_request(req_id):
                """Send a single inference request - calls actual handle_infer endpoint"""
                nonlocal errors
                try:
                    start_time = time.time()
                    request_id = f"scalability_test_{req_id}"
                    
                    # Create mock request data in gateway format
                    log_message = create_mock_request_data(num_pods=num_pods, request_id=request_id)
                    
                    # Call actual handle_infer endpoint using Flask test client
                    with app.test_client() as client:
                        response = client.post('/infer', 
                                             json=log_message,
                                             content_type='application/json')
                        
                        elapsed = time.time() - start_time
                        
                        if response.status_code == 200:
                            response_data = response.get_json()
                            
                            with lock:
                                latencies.append(elapsed * 1000)  # Convert to ms
                                
                                # Parse overhead log if available
                                if 'overhead_log' in response_data:
                                    overhead_log = response_data['overhead_log']
                                    # Parse overhead metrics from the log string
                                    # Format: "oh, handle_infer_X: Yms, encode_X: Yms, ..."
                                    
                                    # DEBUG: Log first few requests to see what keys are available
                                    if req_id < 3:
                                        logger.info(f"DEBUG Request {req_id} overhead_log keys: {overhead_log[:500]}")
                                    
                                    try:
                                        parts = overhead_log.split(', ')
                                        for part in parts:
                                            if ':' in part:
                                                key, value = part.split(': ')
                                                value_ms = float(value.replace('ms', ''))
                                                
                                                # Map overhead metrics to breakdown categories (be specific to avoid double-counting)
                                                if key == 'handle_infer_preprocess_overhead':
                                                    breakdown_metrics['preprocess'].append(value_ms)
                                                elif key == 'handle_infer_normalize':
                                                    breakdown_metrics['normalization'].append(value_ms)
                                                elif key == 'handle_infer_encode':
                                                    breakdown_metrics['encoding'].append(value_ms)
                                                elif 'infer_from_tensor' in key.lower() and 'forward' in key.lower():
                                                    # Get the actual model inference time, not the wrapper
                                                    breakdown_metrics['inference'].append(value_ms)
                                                
                                                # Collect preprocess subcomponents
                                                # Keys from preprocess.py overhead_summary
                                                elif key == 'preprocess_json_parse_overhead':
                                                    preprocess_breakdown['json_parse'].append(value_ms)
                                                elif key == 'preprocess_column_check_overhead':
                                                    preprocess_breakdown['column_check'].append(value_ms)
                                                elif key == 'preprocess_numeric_conversion_overhead':
                                                    preprocess_breakdown['numeric_conversion'].append(value_ms)
                                                elif key == 'preprocess_time_logging_overhead':
                                                    preprocess_breakdown['time_logging'].append(value_ms)
                                                elif key == 'preprocess_get_value_overhead':
                                                    preprocess_breakdown['get_value'].append(value_ms)
                                                elif key == 'preprocess_create_df_overhead':
                                                    preprocess_breakdown['create_df'].append(value_ms)
                                                elif key == 'preprocess_pod_index_overhead':
                                                    preprocess_breakdown['pod_index'].append(value_ms)
                                                elif key == 'preprocess_fill_nan_overhead':
                                                    preprocess_breakdown['fill_nan'].append(value_ms)
                                                elif key == 'preprocess_preprocess_unified_inference' or key == 'preprocess_preprocess_unified_training':
                                                    preprocess_breakdown['preprocess_unified'].append(value_ms)
                                                # Also try without double preprocess_ prefix
                                                elif key == 'preprocess_parse_log_message':
                                                    preprocess_breakdown['json_parse'].append(value_ms)
                                                
                                                # Collect encoding subcomponents
                                                elif 'encode_prepare_for_encoding.extract_pod_columns' in key:
                                                    encoding_breakdown['extract_pod_columns'].append(value_ms)
                                                elif 'encode_prepare_for_encoding.classify_feature_timing' in key:
                                                    encoding_breakdown['classify_feature_timing'].append(value_ms)
                                                elif 'encode_prepare_for_encoding.extract_request_feature' in key:
                                                    encoding_breakdown['extract_request_feature'].append(value_ms)
                                                elif 'encode_prepare_for_encoding.vectorized_extraction' in key:
                                                    encoding_breakdown['vectorized_extraction'].append(value_ms)
                                                elif 'encode_prepare_for_encoding.process_pod_feature' in key:
                                                    encoding_breakdown['process_pod_feature'].append(value_ms)
                                                elif 'encode_prepare_for_encoding.extract_actions' in key:
                                                    encoding_breakdown['extract_actions'].append(value_ms)
                                                elif 'encode_prepare_for_encoding.positional_encoding' in key:
                                                    encoding_breakdown['positional_encoding'].append(value_ms)
                                                elif 'encode_prepare_for_encoding.interaction_features' in key:
                                                    encoding_breakdown['interaction_features'].append(value_ms)
                                                elif key == 'encode_post_process':
                                                    encoding_breakdown['post_process'].append(value_ms)
                                                elif key == 'encode_end_to_end':
                                                    encoding_breakdown['end_to_end'].append(value_ms)
                                                
                                                # Collect inference subcomponents
                                                elif key == 'infer_from_tensor_prepare_tensors':
                                                    inference_breakdown['prepare_tensors'].append(value_ms)
                                                elif key == 'infer_from_tensor_model_inference':
                                                    inference_breakdown['model_inference'].append(value_ms)
                                                elif key == 'infer_from_tensor_format_results':
                                                    inference_breakdown['format_results'].append(value_ms)
                                    except Exception as parse_error:
                                        # If parsing fails, just skip breakdown metrics
                                        pass
                        else:
                            with lock:
                                errors += 1
                            if errors <= 10:
                                logger.error(f"Request {req_id} failed with status {response.status_code}: {response.data}")
                
                except Exception as e:
                    with lock:
                        errors += 1
                    if errors <= 10:  # Log more errors to debug
                        logger.error(f"Request {req_id} raised exception: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                    elif errors == 11:
                        logger.error(f"Suppressing further error logs...")
            
            # Run load test
            test_start_time = time.time()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(target_rps * 2, 200)) as executor:
                futures = []
                
                for i in range(total_requests):
                    # Submit request
                    future = executor.submit(send_request, i)
                    futures.append(future)
                    
                    # Sleep to maintain target RPS
                    time.sleep(request_interval)
                
                # Wait for all requests to complete
                concurrent.futures.wait(futures)
            
            test_duration = time.time() - test_start_time
            
            # Calculate metrics
            if len(latencies) > 0:
                latencies_sorted = sorted(latencies)
                p50 = latencies_sorted[int(len(latencies_sorted) * 0.50)]
                p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
                p99 = latencies_sorted[int(len(latencies_sorted) * 0.99)]
                avg = np.mean(latencies)
                min_lat = min(latencies)
                max_lat = max(latencies)
                actual_rps = len(latencies) / test_duration
                success_rate = len(latencies) / total_requests * 100
            else:
                p50 = p95 = p99 = avg = min_lat = max_lat = actual_rps = 0
                success_rate = 0
            
            # Calculate breakdown averages
            breakdown_avg = {}
            for stage, times in breakdown_metrics.items():
                if len(times) > 0:
                    breakdown_avg[f'{stage}_avg_ms'] = np.mean(times)
                else:
                    breakdown_avg[f'{stage}_avg_ms'] = 0
            
            # Calculate detailed subcomponent averages
            preprocess_avg = {}
            for component, times in preprocess_breakdown.items():
                if len(times) > 0:
                    preprocess_avg[f'{component}_avg_ms'] = np.mean(times)
                else:
                    preprocess_avg[f'{component}_avg_ms'] = 0
            
            encoding_avg = {}
            for component, times in encoding_breakdown.items():
                if len(times) > 0:
                    encoding_avg[f'{component}_avg_ms'] = np.mean(times)
                else:
                    encoding_avg[f'{component}_avg_ms'] = 0
            
            inference_avg = {}
            for component, times in inference_breakdown.items():
                if len(times) > 0:
                    inference_avg[f'{component}_avg_ms'] = np.mean(times)
                else:
                    inference_avg[f'{component}_avg_ms'] = 0
            
            # Store results
            result = {
                'num_pods': num_pods,
                'target_rps': target_rps,
                'actual_rps': actual_rps,
                'total_requests': total_requests,
                'successful': len(latencies),
                'errors': errors,
                'success_rate': success_rate,
                'latency_avg_ms': avg,
                'latency_min_ms': min_lat,
                'latency_max_ms': max_lat,
                'latency_p50_ms': p50,
                'latency_p95_ms': p95,
                'latency_p99_ms': p99,
                'duration_sec': test_duration,
                'breakdown': breakdown_avg,
                'breakdown_raw': breakdown_metrics,  # Store raw data for detailed plotting
                'preprocess_breakdown': preprocess_avg,
                'encoding_breakdown': encoding_avg,
                'inference_breakdown': inference_avg,
            }
            results.append(result)
            
            # Print results
            logger.info(f"\n{CYAN_COLOR}Results:{RESET_COLOR}")
            logger.info(f"  Pods: {num_pods}, Target RPS: {target_rps}, Actual RPS: {actual_rps:.1f}")
            logger.info(f"  Requests: {total_requests} total, {len(latencies)} successful, {errors} errors")
            logger.info(f"  Success Rate: {success_rate:.1f}%")
            logger.info(f"  Latency (ms): avg={avg:.2f}, p50={p50:.2f}, p95={p95:.2f}, p99={p99:.2f}")
            logger.info(f"  Latency Range: min={min_lat:.2f}ms, max={max_lat:.2f}ms")
            logger.info(f"  Test Duration: {test_duration:.2f}s")
            
            # Print pipeline breakdown
            if breakdown_avg:
                logger.info(f"  {CYAN_COLOR}Pipeline Breakdown (avg):{RESET_COLOR}")
                logger.info(f"    Preprocess: {breakdown_avg.get('preprocess_avg_ms', 0):.2f}ms, "
                          f"Normalize: {breakdown_avg.get('normalization_avg_ms', 0):.2f}ms, "
                          f"Encode: {breakdown_avg.get('encoding_avg_ms', 0):.2f}ms, "
                          f"Inference: {breakdown_avg.get('inference_avg_ms', 0):.2f}ms")
            
            # Print detailed subcomponent breakdowns
            if preprocess_avg and any(v > 0 for v in preprocess_avg.values()):
                logger.info(f"  {CYAN_COLOR}Preprocess Subcomponents:{RESET_COLOR}")
                logger.info(f"    JSON Parse: {preprocess_avg.get('json_parse_avg_ms', 0):.2f}ms, "
                          f"Column Check: {preprocess_avg.get('column_check_avg_ms', 0):.2f}ms, "
                          f"Numeric Conv: {preprocess_avg.get('numeric_conversion_avg_ms', 0):.2f}ms")
                logger.info(f"    Time Logging: {preprocess_avg.get('time_logging_avg_ms', 0):.2f}ms, "
                          f"Get Value: {preprocess_avg.get('get_value_avg_ms', 0):.2f}ms, "
                          f"Create DF: {preprocess_avg.get('create_df_avg_ms', 0):.2f}ms")
                logger.info(f"    Pod Index: {preprocess_avg.get('pod_index_avg_ms', 0):.2f}ms, "
                          f"Fill NaN: {preprocess_avg.get('fill_nan_avg_ms', 0):.2f}ms")
                logger.info(f"    Unified Total: {preprocess_avg.get('preprocess_unified_avg_ms', 0):.2f}ms")
                
                # Calculate "Other" time
                subcomponents_sum = (preprocess_avg.get('json_parse_avg_ms', 0) +
                                    preprocess_avg.get('column_check_avg_ms', 0) +
                                    preprocess_avg.get('numeric_conversion_avg_ms', 0) +
                                    preprocess_avg.get('time_logging_avg_ms', 0) +
                                    preprocess_avg.get('get_value_avg_ms', 0) +
                                    preprocess_avg.get('create_df_avg_ms', 0) +
                                    preprocess_avg.get('pod_index_avg_ms', 0) +
                                    preprocess_avg.get('fill_nan_avg_ms', 0))
                unified_total = preprocess_avg.get('preprocess_unified_avg_ms', 0)
                other_time = unified_total - subcomponents_sum
                if other_time > 0.5:
                    logger.warning(f"    {RED_COLOR}⚠️  Other (untracked): {other_time:.2f}ms "
                                 f"({other_time/unified_total*100:.1f}% of total){RESET_COLOR}")
            
            if encoding_avg and any(v > 0 for v in encoding_avg.values()):
                logger.info(f"  {CYAN_COLOR}Encoding Subcomponents:{RESET_COLOR}")
                logger.info(f"    Extract Pod Cols: {encoding_avg.get('extract_pod_columns_avg_ms', 0):.2f}ms, "
                          f"Classify Timing: {encoding_avg.get('classify_feature_timing_avg_ms', 0):.2f}ms, "
                          f"Extract Req: {encoding_avg.get('extract_request_feature_avg_ms', 0):.2f}ms")
                logger.info(f"    Vectorized: {encoding_avg.get('vectorized_extraction_avg_ms', 0):.2f}ms, "
                          f"Process Pod: {encoding_avg.get('process_pod_feature_avg_ms', 0):.2f}ms, "
                          f"Post Process: {encoding_avg.get('post_process_avg_ms', 0):.2f}ms")
            
            if inference_avg and any(v > 0 for v in inference_avg.values()):
                logger.info(f"  {CYAN_COLOR}Inference Subcomponents:{RESET_COLOR}")
                logger.info(f"    Prepare Tensors: {inference_avg.get('prepare_tensors_avg_ms', 0):.2f}ms, "
                          f"Model Inference: {inference_avg.get('model_inference_avg_ms', 0):.2f}ms, "
                          f"Format Results: {inference_avg.get('format_results_avg_ms', 0):.2f}ms")
    
        # Print summary table and export CSV
        csv_path = print_scalability_summary(results)
        
        # Generate plots
        logger.info(f"\n{CYAN_COLOR}Generating plots...{RESET_COLOR}")
        plot_path, comparison_path = plot_scalability_results(results, output_dir=".")
        
        # Generate pipeline breakdown plot
        logger.info(f"{CYAN_COLOR}Generating pipeline breakdown plot...{RESET_COLOR}")
        breakdown_path = plot_pipeline_breakdown(results, output_dir=".")
        
        # Generate detailed subcomponent breakdown plots
        logger.info(f"{CYAN_COLOR}Generating detailed subcomponent breakdown plots...{RESET_COLOR}")
        detailed_breakdown_path = plot_detailed_breakdowns(results, output_dir=".")
        
        if plot_path and breakdown_path and detailed_breakdown_path:
            logger.info(f"{GREEN_COLOR}✓ All plots generated successfully{RESET_COLOR}")
        else:
            logger.warning(f"{YELLOW_COLOR}! Some plots may have failed{RESET_COLOR}")
        
        return results


class MockLatencyPredictor:
    """Mock latency predictor for testing that returns random pod selections"""
    def __init__(self, state_dims, hyperparameters, model_dir):
        self.state_dims = state_dims
        self.hyperparameters = hyperparameters
        self.model_dir = model_dir
        logger.info(f"Initialized MockLatencyPredictor with state_dims={state_dims}")
    
    def predict(self, tensor_data):
        """Return a random pod index"""
        num_pods = tensor_data['pod_features_with_staleness'].shape[1]
        # Return random pod index
        selected_pod_idx = np.random.randint(0, num_pods)
        return selected_pod_idx
    
    def load(self, model_dir):
        """Mock load - does nothing"""
        logger.info(f"MockLatencyPredictor: Skipping model load from {model_dir}")
        pass


def setup_mock_environment(num_pods):
    """
    Setup mock environment with specified number of pods.
    This updates RL_MODEL_HYPERPARAMETERS with mock pod configurations.
    """
    global RL_MODEL_HYPERPARAMETERS, stats_instance
    
    # Generate mock pod IPs and mappings
    mock_pod_ips = [f"10.0.0.{100+i}" for i in range(num_pods)]
    
    # Create pod mappings
    pod_ip_to_generalpodid = {}
    generalpodid_to_pod_ip = {}
    pod_ip_to_gpu_model = {}
    pod_ip_to_gpu_model_encoded = {}
    generalpodid_to_gpu_model = {}
    pod_gpu_id_mapping = {}
    
    gpu_models = ['NVIDIA-L20', 'NVIDIA-L40', 'NVIDIA-A10', 'NVIDIA-A100', 'NVIDIA-H100']
    GPU_MODEL_TO_ENCODE = {
        'NVIDIA-L20': 0,
        'NVIDIA-L40': 1,
        'NVIDIA-A10': 2,
        'NVIDIA-A100': 3,
        'NVIDIA-H100': 4,
    }
    
    for i, pod_ip in enumerate(mock_pod_ips):
        general_pod_id = f"generalpod{i}"
        gpu_model = gpu_models[i % len(gpu_models)]
        
        pod_ip_to_generalpodid[pod_ip] = general_pod_id
        generalpodid_to_pod_ip[general_pod_id] = pod_ip
        pod_ip_to_gpu_model[pod_ip] = gpu_model
        pod_ip_to_gpu_model_encoded[pod_ip] = GPU_MODEL_TO_ENCODE[gpu_model]
        generalpodid_to_gpu_model[general_pod_id] = gpu_model
        pod_gpu_id_mapping[general_pod_id] = GPU_MODEL_TO_ENCODE[gpu_model]
    
    # Update hyperparameters
    RL_MODEL_HYPERPARAMETERS['pod_ip_to_generalpodid'] = pod_ip_to_generalpodid
    RL_MODEL_HYPERPARAMETERS['generalpodid_to_pod_ip'] = generalpodid_to_pod_ip
    RL_MODEL_HYPERPARAMETERS['sorted_running_pod_ips'] = sorted(mock_pod_ips)
    RL_MODEL_HYPERPARAMETERS['pod_ip_to_gpu_model'] = pod_ip_to_gpu_model
    RL_MODEL_HYPERPARAMETERS['pod_ip_to_gpu_model_encoded'] = pod_ip_to_gpu_model_encoded
    RL_MODEL_HYPERPARAMETERS['generalpodid_to_gpu_model'] = generalpodid_to_gpu_model
    RL_MODEL_HYPERPARAMETERS['pod_gpu_mapping'] = generalpodid_to_gpu_model
    RL_MODEL_HYPERPARAMETERS['pod_gpu_id_mapping'] = pod_gpu_id_mapping
    
    logger.info(f"Mock environment setup complete: {num_pods} pods configured")


def plot_scalability_results(results, output_dir="."):
    """
    Create professional plots of scalability test results.
    
    Args:
        results: List of result dictionaries
        output_dir: Directory to save plots
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        import pandas as pd
        
        # Convert results to DataFrame
        df = pd.DataFrame(results)
        
        # Set professional style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # Define large font sizes
        TITLE_SIZE = 24
        LABEL_SIZE = 20
        TICK_SIZE = 18
        LEGEND_SIZE = 18
        
        # Get unique pod counts for color coding
        unique_pods = sorted(df['num_pods'].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_pods)))
        pod_colors = {pod: colors[i] for i, pod in enumerate(unique_pods)}
        
        # ============ Plot 1: Latency vs RPS (for each pod count) ============
        ax1 = plt.subplot(2, 3, 1)
        for pod_count in unique_pods:
            pod_data = df[df['num_pods'] == pod_count]
            ax1.plot(pod_data['target_rps'], pod_data['latency_avg_ms'], 
                    marker='o', linewidth=3, markersize=10,
                    label=f'{pod_count} pods', color=pod_colors[pod_count])
        ax1.set_xlabel('Target RPS', fontsize=LABEL_SIZE, fontweight='bold')
        ax1.set_ylabel('Average Latency (ms)', fontsize=LABEL_SIZE, fontweight='bold')
        ax1.set_title('Average Latency vs Request Rate', fontsize=TITLE_SIZE, fontweight='bold', pad=20)
        ax1.legend(fontsize=LEGEND_SIZE, loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=TICK_SIZE)
        
        # ============ Plot 2: P95 Latency vs RPS ============
        ax2 = plt.subplot(2, 3, 2)
        for pod_count in unique_pods:
            pod_data = df[df['num_pods'] == pod_count]
            ax2.plot(pod_data['target_rps'], pod_data['latency_p95_ms'], 
                    marker='s', linewidth=3, markersize=10,
                    label=f'{pod_count} pods', color=pod_colors[pod_count])
        ax2.set_xlabel('Target RPS', fontsize=LABEL_SIZE, fontweight='bold')
        ax2.set_ylabel('P95 Latency (ms)', fontsize=LABEL_SIZE, fontweight='bold')
        ax2.set_title('P95 Latency vs Request Rate', fontsize=TITLE_SIZE, fontweight='bold', pad=20)
        ax2.legend(fontsize=LEGEND_SIZE, loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=TICK_SIZE)
        
        # ============ Plot 3: P99 Latency vs RPS ============
        ax3 = plt.subplot(2, 3, 3)
        for pod_count in unique_pods:
            pod_data = df[df['num_pods'] == pod_count]
            ax3.plot(pod_data['target_rps'], pod_data['latency_p99_ms'], 
                    marker='^', linewidth=3, markersize=10,
                    label=f'{pod_count} pods', color=pod_colors[pod_count])
        ax3.set_xlabel('Target RPS', fontsize=LABEL_SIZE, fontweight='bold')
        ax3.set_ylabel('P99 Latency (ms)', fontsize=LABEL_SIZE, fontweight='bold')
        ax3.set_title('P99 Latency vs Request Rate', fontsize=TITLE_SIZE, fontweight='bold', pad=20)
        ax3.legend(fontsize=LEGEND_SIZE, loc='best')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(labelsize=TICK_SIZE)
        
        # ============ Plot 4: Latency vs Pod Count (for each RPS) ============
        ax4 = plt.subplot(2, 3, 4)
        unique_rps = sorted(df['target_rps'].unique())
        rps_colors = plt.cm.viridis(np.linspace(0, 1, len(unique_rps)))
        for i, rps in enumerate(unique_rps):
            rps_data = df[df['target_rps'] == rps]
            ax4.plot(rps_data['num_pods'], rps_data['latency_avg_ms'], 
                    marker='o', linewidth=3, markersize=10,
                    label=f'{rps} RPS', color=rps_colors[i])
        ax4.set_xlabel('Number of Pods', fontsize=LABEL_SIZE, fontweight='bold')
        ax4.set_ylabel('Average Latency (ms)', fontsize=LABEL_SIZE, fontweight='bold')
        ax4.set_title('Average Latency vs Pod Count', fontsize=TITLE_SIZE, fontweight='bold', pad=20)
        ax4.legend(fontsize=LEGEND_SIZE, loc='best')
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(labelsize=TICK_SIZE)
        
        # ============ Plot 5: Success Rate Heatmap ============
        ax5 = plt.subplot(2, 3, 5)
        pivot_success = df.pivot(index='num_pods', columns='target_rps', values='success_rate')
        im = ax5.imshow(pivot_success.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        ax5.set_xticks(range(len(pivot_success.columns)))
        ax5.set_yticks(range(len(pivot_success.index)))
        ax5.set_xticklabels(pivot_success.columns, fontsize=TICK_SIZE)
        ax5.set_yticklabels(pivot_success.index, fontsize=TICK_SIZE)
        ax5.set_xlabel('Target RPS', fontsize=LABEL_SIZE, fontweight='bold')
        ax5.set_ylabel('Number of Pods', fontsize=LABEL_SIZE, fontweight='bold')
        ax5.set_title('Success Rate (%)', fontsize=TITLE_SIZE, fontweight='bold', pad=20)
        
        # Add text annotations
        for i in range(len(pivot_success.index)):
            for j in range(len(pivot_success.columns)):
                text = ax5.text(j, i, f'{pivot_success.values[i, j]:.1f}',
                               ha="center", va="center", color="black", fontsize=16, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax5)
        cbar.ax.tick_params(labelsize=TICK_SIZE)
        
        # ============ Plot 6: Throughput vs Latency ============
        ax6 = plt.subplot(2, 3, 6)
        for pod_count in unique_pods:
            pod_data = df[df['num_pods'] == pod_count]
            ax6.scatter(pod_data['actual_rps'], pod_data['latency_p95_ms'], 
                       s=200, alpha=0.7, label=f'{pod_count} pods', 
                       color=pod_colors[pod_count])
        ax6.set_xlabel('Actual Throughput (RPS)', fontsize=LABEL_SIZE, fontweight='bold')
        ax6.set_ylabel('P95 Latency (ms)', fontsize=LABEL_SIZE, fontweight='bold')
        ax6.set_title('Throughput vs P95 Latency', fontsize=TITLE_SIZE, fontweight='bold', pad=20)
        ax6.legend(fontsize=LEGEND_SIZE, loc='best')
        ax6.grid(True, alpha=0.3)
        ax6.tick_params(labelsize=TICK_SIZE)
        
        plt.tight_layout(pad=3.0)
        
        # Save plot
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(output_dir, f"scalability_test_plots_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plots saved to: {plot_path}")
        plt.close()
        
        # Create additional detailed comparison plot
        fig2, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot: Latency distribution comparison
        ax_left = axes[0]
        width = 0.8 / len(unique_pods)
        x_pos = np.arange(len(unique_rps))
        
        for i, pod_count in enumerate(unique_pods):
            pod_data = df[df['num_pods'] == pod_count].sort_values('target_rps')
            offset = width * (i - len(unique_pods)/2 + 0.5)
            ax_left.bar(x_pos + offset, pod_data['latency_avg_ms'], width,
                       label=f'{pod_count} pods', alpha=0.8, color=pod_colors[pod_count])
        
        ax_left.set_xlabel('Target RPS', fontsize=LABEL_SIZE, fontweight='bold')
        ax_left.set_ylabel('Average Latency (ms)', fontsize=LABEL_SIZE, fontweight='bold')
        ax_left.set_title('Latency Comparison by Pod Count', fontsize=TITLE_SIZE, fontweight='bold', pad=20)
        ax_left.set_xticks(x_pos)
        ax_left.set_xticklabels([str(rps) for rps in unique_rps], fontsize=TICK_SIZE)
        ax_left.legend(fontsize=LEGEND_SIZE, loc='best')
        ax_left.grid(True, alpha=0.3, axis='y')
        ax_left.tick_params(labelsize=TICK_SIZE)
        
        # Plot: Percentile comparison for highest RPS
        ax_right = axes[1]
        max_rps = df['target_rps'].max()
        max_rps_data = df[df['target_rps'] == max_rps].sort_values('num_pods')
        
        x_pos_pods = np.arange(len(max_rps_data))
        bar_width = 0.25
        
        ax_right.bar(x_pos_pods - bar_width, max_rps_data['latency_p50_ms'], bar_width, 
                    label='P50', alpha=0.8, color='#2ecc71')
        ax_right.bar(x_pos_pods, max_rps_data['latency_p95_ms'], bar_width, 
                    label='P95', alpha=0.8, color='#f39c12')
        ax_right.bar(x_pos_pods + bar_width, max_rps_data['latency_p99_ms'], bar_width, 
                    label='P99', alpha=0.8, color='#e74c3c')
        
        ax_right.set_xlabel('Number of Pods', fontsize=LABEL_SIZE, fontweight='bold')
        ax_right.set_ylabel('Latency (ms)', fontsize=LABEL_SIZE, fontweight='bold')
        ax_right.set_title(f'Latency Percentiles @ {max_rps} RPS', fontsize=TITLE_SIZE, fontweight='bold', pad=20)
        ax_right.set_xticks(x_pos_pods)
        ax_right.set_xticklabels([str(int(p)) for p in max_rps_data['num_pods']], fontsize=TICK_SIZE)
        ax_right.legend(fontsize=LEGEND_SIZE, loc='best')
        ax_right.grid(True, alpha=0.3, axis='y')
        ax_right.tick_params(labelsize=TICK_SIZE)
        
        plt.tight_layout(pad=3.0)
        
        # Save comparison plot
        comparison_path = os.path.join(output_dir, f"scalability_test_comparison_{timestamp}.png")
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison plot saved to: {comparison_path}")
        plt.close()
        
        return plot_path, comparison_path
        
    except Exception as e:
        logger.error(f"Failed to create plots: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None


def plot_pipeline_breakdown(results, output_dir="."):
    """
    Plot pipeline breakdown showing time spent in each stage
    """
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        from datetime import datetime
        
        # Large font sizes for professional presentation
        TITLE_SIZE = 20
        LABEL_SIZE = 16
        TICK_SIZE = 14
        LEGEND_SIZE = 14
        
        # Create figure with 2 rows x 2 columns layout
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Routing Agent Pipeline Breakdown Analysis', fontsize=TITLE_SIZE+4, fontweight='bold', y=0.995)
        
        # Extract data for plotting
        breakdown_data = []
        for r in results:
            if 'breakdown' in r and len(r['breakdown']) > 0:
                breakdown_data.append({
                    'num_pods': r['num_pods'],
                    'target_rps': r['target_rps'],
                    'actual_rps': r['actual_rps'],
                    'preprocess': r['breakdown'].get('preprocess_avg_ms', 0),
                    'normalization': r['breakdown'].get('normalization_avg_ms', 0),
                    'encoding': r['breakdown'].get('encoding_avg_ms', 0),
                    'inference': r['breakdown'].get('inference_avg_ms', 0),
                    'total': r['latency_avg_ms']
                })
        
        if not breakdown_data:
            logger.warning("No breakdown data available for plotting")
            return None
        
        df = pd.DataFrame(breakdown_data)
        
        # Color scheme
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        stage_names = ['Preprocess', 'Normalization', 'Encoding', 'Inference']
        
        # Plot 1: Stacked bar chart by RPS for first pod count
        ax1 = axes[0, 0]
        first_pod_count = df['num_pods'].iloc[0]
        df_subset = df[df['num_pods'] == first_pod_count].sort_values('target_rps')
        
        x = np.arange(len(df_subset))
        width = 0.6
        
        bottom = np.zeros(len(df_subset))
        for i, (col, name, color) in enumerate(zip(['preprocess', 'normalization', 'encoding', 'inference'],
                                                     stage_names, colors)):
            values = df_subset[col].values
            ax1.bar(x, values, width, label=name, bottom=bottom, color=color, alpha=0.9, edgecolor='white', linewidth=2)
            
            # Add value labels on bars (only for significant stages)
            for j, (val, bot) in enumerate(zip(values, bottom)):
                if val > 0.5:  # Only show if > 0.5ms
                    ax1.text(x[j], bot + val/2, f'{val:.1f}', ha='center', va='center', 
                            fontsize=TICK_SIZE-2, fontweight='bold', color='white')
            
            bottom += values
        
        ax1.set_xlabel('Target RPS', fontsize=LABEL_SIZE, fontweight='bold')
        ax1.set_ylabel('Latency (ms)', fontsize=LABEL_SIZE, fontweight='bold')
        ax1.set_title(f'Pipeline Breakdown by RPS\n({first_pod_count} pods)', fontsize=TITLE_SIZE, fontweight='bold', pad=15)
        ax1.set_xticks(x)
        ax1.set_xticklabels(df_subset['target_rps'].values, fontsize=TICK_SIZE)
        ax1.tick_params(axis='y', labelsize=TICK_SIZE)
        ax1.legend(fontsize=LEGEND_SIZE, loc='upper left', framealpha=0.95)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_axisbelow(True)
        
        # Plot 2: Percentage breakdown (pie-like stacked bars)
        ax2 = axes[0, 1]
        
        # Calculate average percentage for each stage across all tests
        avg_breakdown = {
            'Preprocess': df['preprocess'].mean(),
            'Normalization': df['normalization'].mean(),
            'Encoding': df['encoding'].mean(),
            'Inference': df['inference'].mean()
        }
        
        total_avg = sum(avg_breakdown.values())
        percentages = {k: (v/total_avg)*100 for k, v in avg_breakdown.items()}
        
        x = [0]
        width = 0.8
        bottom = 0
        
        for i, ((stage, pct), color) in enumerate(zip(percentages.items(), colors)):
            ax2.barh(x, pct, width, left=bottom, color=color, alpha=0.9, edgecolor='white', linewidth=3)
            # Add percentage label
            if pct > 5:  # Only show label if > 5%
                ax2.text(bottom + pct/2, 0, f'{stage}\n{pct:.1f}%\n({avg_breakdown[stage]:.2f}ms)', 
                        ha='center', va='center', fontsize=LEGEND_SIZE, fontweight='bold', color='white')
            bottom += pct
        
        ax2.set_xlim(0, 100)
        ax2.set_ylim(-0.5, 0.5)
        ax2.set_xlabel('Percentage of Total Latency (%)', fontsize=LABEL_SIZE, fontweight='bold')
        ax2.set_title('Average Pipeline Stage Distribution', fontsize=TITLE_SIZE, fontweight='bold', pad=15)
        ax2.set_yticks([])
        ax2.tick_params(axis='x', labelsize=TICK_SIZE)
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.set_axisbelow(True)
        
        # Plot 3: Line plot showing how each stage scales with RPS
        ax3 = axes[1, 0]
        
        for i, (col, name, color) in enumerate(zip(['preprocess', 'normalization', 'encoding', 'inference'],
                                                     stage_names, colors)):
            # Group by RPS and take mean across different pod counts
            grouped = df.groupby('target_rps')[col].mean()
            ax3.plot(grouped.index, grouped.values, marker='o', linewidth=3, markersize=10,
                    label=name.replace('\n', ' '), color=color, alpha=0.9)
        
        ax3.set_xlabel('Target RPS', fontsize=LABEL_SIZE, fontweight='bold')
        ax3.set_ylabel('Latency (ms)', fontsize=LABEL_SIZE, fontweight='bold')
        ax3.set_title('Stage Latency Scaling with Load', fontsize=TITLE_SIZE, fontweight='bold', pad=15)
        ax3.legend(fontsize=LEGEND_SIZE, loc='best', framealpha=0.95)
        ax3.tick_params(axis='both', labelsize=TICK_SIZE)
        ax3.grid(True, alpha=0.3)
        ax3.set_axisbelow(True)
        
        # Plot 4: Heatmap showing breakdown for all configurations
        ax4 = axes[1, 1]
        
        # Create pivot table for heatmap
        pivot_data = []
        for stage, color in zip(['preprocess', 'normalization', 'encoding', 'inference'], colors):
            row = []
            for rps in sorted(df['target_rps'].unique()):
                val = df[df['target_rps'] == rps][stage].mean()
                row.append(val)
            pivot_data.append(row)
        
        pivot_data = np.array(pivot_data)
        
        im = ax4.imshow(pivot_data, cmap='YlOrRd', aspect='auto')
        
        # Set ticks
        ax4.set_xticks(np.arange(len(sorted(df['target_rps'].unique()))))
        ax4.set_yticks(np.arange(len(stage_names)))
        ax4.set_xticklabels([f'{int(x)}' for x in sorted(df['target_rps'].unique())], fontsize=TICK_SIZE)
        ax4.set_yticklabels([s.replace('\n', ' ') for s in stage_names], fontsize=TICK_SIZE)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax4)
        cbar.set_label('Latency (ms)', fontsize=LABEL_SIZE, fontweight='bold')
        cbar.ax.tick_params(labelsize=TICK_SIZE)
        
        # Add values to heatmap
        for i in range(len(stage_names)):
            for j in range(len(sorted(df['target_rps'].unique()))):
                text = ax4.text(j, i, f'{pivot_data[i, j]:.1f}',
                              ha="center", va="center", color="white" if pivot_data[i, j] > pivot_data.max()/2 else "black",
                              fontsize=TICK_SIZE-2, fontweight='bold')
        
        ax4.set_xlabel('Target RPS', fontsize=LABEL_SIZE, fontweight='bold')
        ax4.set_ylabel('Pipeline Stage', fontsize=LABEL_SIZE, fontweight='bold')
        ax4.set_title('Latency Heatmap by Stage and Load', fontsize=TITLE_SIZE, fontweight='bold', pad=15)
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(output_dir, f"pipeline_breakdown_{timestamp}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Pipeline breakdown plot saved to: {plot_path}")
        return plot_path
        
    except Exception as e:
        logger.error(f"Failed to create pipeline breakdown plot: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def plot_detailed_breakdowns(results, output_dir="."):
    """
    Create detailed breakdown plots for each major category (preprocess, encoding, inference)
    Shows subcomponents for each category
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import pandas as pd
        from datetime import datetime
        
        # Font sizes
        TITLE_SIZE = 18
        LABEL_SIZE = 14
        TICK_SIZE = 12
        LEGEND_SIZE = 12
        
        # Create figure with 3 subplots (one for each category)
        fig, axes = plt.subplots(3, 1, figsize=(20, 18))
        fig.suptitle('Detailed Subcomponent Breakdown Analysis', fontsize=TITLE_SIZE+4, fontweight='bold', y=0.995)
        
        # Extract data
        breakdown_data = []
        for r in results:
            data_point = {
                'num_pods': r['num_pods'],
                'target_rps': r['target_rps'],
                'actual_rps': r['actual_rps'],
            }
            
            # Add preprocess subcomponents
            if 'preprocess_breakdown' in r:
                for key, value in r['preprocess_breakdown'].items():
                    data_point[f'preprocess_{key}'] = value
            
            # Add encoding subcomponents
            if 'encoding_breakdown' in r:
                for key, value in r['encoding_breakdown'].items():
                    data_point[f'encoding_{key}'] = value
            
            # Add inference subcomponents
            if 'inference_breakdown' in r:
                for key, value in r['inference_breakdown'].items():
                    data_point[f'inference_{key}'] = value
            
            breakdown_data.append(data_point)
        
        if not breakdown_data:
            logger.warning("No detailed breakdown data available for plotting")
            return None
        
        df = pd.DataFrame(breakdown_data)
        
        # ============ Plot 1: Preprocess Breakdown ============
        ax1 = axes[0]
        # Note: preprocess_unified is the total e2e time, subcomponents are parts of it
        preprocess_components = ['json_parse', 'column_check', 'numeric_conversion', 'time_logging', 'get_value', 'create_df', 'pod_index', 'fill_nan']
        preprocess_cols = [f'preprocess_{comp}_avg_ms' for comp in preprocess_components]
        preprocess_labels = ['JSON Parse', 'Column Check', 'Numeric Conv', 'Time Log', 'Get Value', 'Create DF', 'Pod Index', 'Fill NaN']
        colors_preprocess = ['#FF6B6B', '#E74C3C', '#4ECDC4', '#3498DB', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
        
        # Group by target_rps and average across pod counts
        grouped = df.groupby('target_rps')[[col for col in preprocess_cols if col in df.columns]].mean()
        
        # Also get the unified total
        unified_col = 'preprocess_preprocess_unified_avg_ms'
        if unified_col in df.columns:
            grouped_unified = df.groupby('target_rps')[unified_col].mean()
        else:
            grouped_unified = None
        
        if not grouped.empty:
            x = np.arange(len(grouped))
            width = 0.6
            
            # First, show the unified (total) bar as a lighter background
            if grouped_unified is not None and np.any(grouped_unified.values > 0):
                ax1.bar(x, grouped_unified.values, width, label='Total (Unified)', 
                       color='#E8E8E8', alpha=0.7, edgecolor='black', linewidth=2, zorder=1)
            
            # Then stack the detailed subcomponents on top
            bottom = np.zeros(len(grouped))
            for i, (col, label, color) in enumerate(zip(preprocess_cols, preprocess_labels, colors_preprocess)):
                if col in grouped.columns:
                    values = grouped[col].values
                    if np.any(values > 0):
                        ax1.bar(x, values, width, label=label, bottom=bottom, 
                               color=color, alpha=0.9, edgecolor='white', linewidth=1, zorder=2)
                        bottom += values
            
            # Add "Other" category if subcomponents don't sum to unified
            if grouped_unified is not None:
                other = grouped_unified.values - bottom
                if np.any(other > 0.5):
                    ax1.bar(x, other, width, label='Other', bottom=bottom,
                           color='#CCCCCC', alpha=0.8, edgecolor='white', linewidth=1, zorder=2)
            
            ax1.set_xlabel('Target RPS', fontsize=LABEL_SIZE, fontweight='bold')
            ax1.set_ylabel('Time (ms)', fontsize=LABEL_SIZE, fontweight='bold')
            ax1.set_title('Preprocess Subcomponent Breakdown\n(Gray bar = Total E2E, Colors = Subcomponents)', 
                         fontsize=TITLE_SIZE, fontweight='bold', pad=15)
            ax1.set_xticks(x)
            ax1.set_xticklabels(grouped.index.astype(int), fontsize=TICK_SIZE)
            ax1.tick_params(axis='y', labelsize=TICK_SIZE)
            ax1.legend(fontsize=LEGEND_SIZE-2, loc='upper left', framealpha=0.95, ncol=3)
            ax1.grid(True, alpha=0.3, axis='y')
            ax1.set_axisbelow(True)
        
        # ============ Plot 2: Encoding Breakdown ============
        ax2 = axes[1]
        # Note: end_to_end is the total e2e time, subcomponents are parts of it
        encoding_components = ['extract_pod_columns', 'classify_feature_timing', 'extract_request_feature', 
                              'vectorized_extraction', 'process_pod_feature', 'post_process']
        encoding_cols = [f'encoding_{comp}_avg_ms' for comp in encoding_components]
        encoding_labels = ['Extract Pods', 'Classify Timing', 'Extract Req', 'Vectorized', 'Process Pod', 'Post Process']
        colors_encoding = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C']
        
        grouped = df.groupby('target_rps')[[col for col in encoding_cols if col in df.columns]].mean()
        
        # Also get the end_to_end total
        e2e_col = 'encoding_end_to_end_avg_ms'
        if e2e_col in df.columns:
            grouped_e2e = df.groupby('target_rps')[e2e_col].mean()
        else:
            grouped_e2e = None
        
        if not grouped.empty:
            x = np.arange(len(grouped))
            width = 0.6
            
            # First, show the end_to_end (total) bar as a lighter background
            if grouped_e2e is not None and np.any(grouped_e2e.values > 0):
                ax2.bar(x, grouped_e2e.values, width, label='Total (E2E)', 
                       color='#E8E8E8', alpha=0.7, edgecolor='black', linewidth=2, zorder=1)
            
            # Then stack the detailed subcomponents on top
            bottom = np.zeros(len(grouped))
            for i, (col, label, color) in enumerate(zip(encoding_cols, encoding_labels, colors_encoding)):
                if col in grouped.columns:
                    values = grouped[col].values
                    if np.any(values > 0):
                        ax2.bar(x, values, width, label=label, bottom=bottom, 
                               color=color, alpha=0.9, edgecolor='white', linewidth=1, zorder=2)
                        bottom += values
            
            # Add "Other" category if subcomponents don't sum to e2e
            if grouped_e2e is not None:
                other = grouped_e2e.values - bottom
                if np.any(other > 0.5):
                    ax2.bar(x, other, width, label='Other', bottom=bottom,
                           color='#CCCCCC', alpha=0.8, edgecolor='white', linewidth=1, zorder=2)
            
            ax2.set_xlabel('Target RPS', fontsize=LABEL_SIZE, fontweight='bold')
            ax2.set_ylabel('Time (ms)', fontsize=LABEL_SIZE, fontweight='bold')
            ax2.set_title('Encoding Subcomponent Breakdown\n(Gray bar = Total E2E, Colors = Subcomponents)', 
                         fontsize=TITLE_SIZE, fontweight='bold', pad=15)
            ax2.set_xticks(x)
            ax2.set_xticklabels(grouped.index.astype(int), fontsize=TICK_SIZE)
            ax2.tick_params(axis='y', labelsize=TICK_SIZE)
            ax2.legend(fontsize=LEGEND_SIZE-2, loc='upper left', framealpha=0.95, ncol=3)
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.set_axisbelow(True)
        
        # ============ Plot 3: Inference Breakdown ============
        ax3 = axes[2]
        inference_components = ['prepare_tensors', 'model_inference', 'format_results']
        inference_cols = [f'inference_{comp}_avg_ms' for comp in inference_components]
        inference_labels = ['Prepare Tensors', 'Model Inference', 'Format Results']
        colors_inference = ['#16A085', '#C0392B', '#8E44AD']
        
        grouped = df.groupby('target_rps')[[col for col in inference_cols if col in df.columns]].mean()
        
        if not grouped.empty:
            x = np.arange(len(grouped))
            width = 0.6
            bottom = np.zeros(len(grouped))
            
            for i, (col, label, color) in enumerate(zip(inference_cols, inference_labels, colors_inference)):
                if col in grouped.columns:
                    values = grouped[col].values
                    if np.any(values > 0):
                        ax3.bar(x, values, width, label=label, bottom=bottom, color=color, alpha=0.9, edgecolor='white', linewidth=2)
                        # Add value labels for significant components
                        for j, (val, bot) in enumerate(zip(values, bottom)):
                            if val > 0.5:
                                ax3.text(x[j], bot + val/2, f'{val:.1f}', ha='center', va='center',
                                       fontsize=TICK_SIZE-2, fontweight='bold', color='white')
                        bottom += values
            
            ax3.set_xlabel('Target RPS', fontsize=LABEL_SIZE, fontweight='bold')
            ax3.set_ylabel('Time (ms)', fontsize=LABEL_SIZE, fontweight='bold')
            ax3.set_title('Inference Subcomponent Breakdown', fontsize=TITLE_SIZE, fontweight='bold', pad=15)
            ax3.set_xticks(x)
            ax3.set_xticklabels(grouped.index.astype(int), fontsize=TICK_SIZE)
            ax3.tick_params(axis='y', labelsize=TICK_SIZE)
            ax3.legend(fontsize=LEGEND_SIZE, loc='upper left', framealpha=0.95)
            ax3.grid(True, alpha=0.3, axis='y')
            ax3.set_axisbelow(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(output_dir, f"detailed_subcomponent_breakdown_{timestamp}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Detailed breakdown plot saved to: {plot_path}")
        return plot_path
        
    except Exception as e:
        logger.error(f"Failed to create detailed breakdown plots: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def print_scalability_summary(results):
    """Print formatted summary table of scalability test results"""
    logger.info(f"\n{'='*120}")
    logger.info(f"SCALABILITY TEST SUMMARY")
    logger.info(f"{'='*120}")
    
    # Header
    header = f"{'Pods':<8} {'Target RPS':<12} {'Actual RPS':<12} {'Success%':<10} " \
             f"{'Avg(ms)':<10} {'P50(ms)':<10} {'P95(ms)':<10} {'P99(ms)':<10}"
    logger.info(header)
    logger.info("-" * 120)
    
    # Data rows
    for r in results:
        row = f"{r['num_pods']:<8} {r['target_rps']:<12} {r['actual_rps']:<12.1f} {r['success_rate']:<10.1f} " \
              f"{r['latency_avg_ms']:<10.2f} {r['latency_p50_ms']:<10.2f} {r['latency_p95_ms']:<10.2f} {r['latency_p99_ms']:<10.2f}"
        logger.info(row)
    
    logger.info(f"{'='*120}\n")
    
    # Export to CSV
    csv_path = None
    try:
        import csv
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = f"scalability_test_results_{timestamp}.csv"
        
        with open(csv_path, 'w', newline='') as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
        
        logger.info(f"Results exported to: {csv_path}")
    except Exception as e:
        logger.error(f"Failed to export results to CSV: {e}")
    
    return csv_path


if __name__ == "__main__":
    # Check if running in test mode
    test_mode = os.environ.get("SCALABILITY_TEST", "1")
    
    if test_mode == "1":
        logger.info(f"\n{MAGENTA_COLOR}{'='*80}{RESET_COLOR}")
        logger.info(f"{MAGENTA_COLOR}ROUTING AGENT SERVICE - SCALABILITY TEST MODE{RESET_COLOR}")
        logger.info(f"{MAGENTA_COLOR}Uses REAL latency predictor via actual handle_infer calls{RESET_COLOR}")
        logger.info(f"{MAGENTA_COLOR}{'='*80}{RESET_COLOR}\n")
        
        # Initialize in test mode (no Kubernetes, no file loading)
        init_test_mode()
        
        logger.info(f"{GREEN_COLOR}Initialization complete. Starting scalability tests...{RESET_COLOR}\n")
        
        # Define test configurations
        # Adjust these based on your testing needs:
        # - NUM_PODS_TO_TEST: Test different cluster sizes (affects tensor dimensions)
        # - RPS_TO_TEST: Test different request loads (affects throughput/latency)
        # - DURATION_PER_TEST: Duration of each test (longer = more stable metrics)
        
        # Reasonable test configuration for quick scalability assessment
        # Adjust based on your needs - more configs = longer test time
        # NUM_PODS_TO_TEST = [5, 10, 20, 50]    # Different pod counts
        NUM_PODS_TO_TEST = [5]    # Different pod counts
        RPS_TO_TEST = [50, 100]       # Different request rates (removed 500 - too saturated)
        DURATION_PER_TEST = 20                  # Seconds per test
        
        # Expected test time: ~5 minutes (4 pods × 4 RPS × 5 seconds + overhead)
        
        # NOTE: The overhead being measured includes:
        # 1. Request preprocessing (parsing gateway log format into DataFrame)
        # 2. Feature normalization (real overhead from data_normalizer)
        # 3. Tensor encoding (extracts features from DataFrame and creates PyTorch tensors)
        # 4. Model inference (REAL latency predictor with actual neural network forward pass)
        
        # Run scalability tests
        results = run_scalability_test(
            test_name="Latency Predictor Inference Scalability",
            num_pods_list=NUM_PODS_TO_TEST,
            rps_list=RPS_TO_TEST,
            duration_per_test=DURATION_PER_TEST
        )
        
        logger.info(f"\n{GREEN_COLOR}Scalability test completed!{RESET_COLOR}")
        logger.info(f"Total test configurations: {len(results)}")
        
        sys.exit(0)
    
    else:
        # Normal service mode
        signal.signal(signal.SIGTERM, graceful_shutdown)
        signal.signal(signal.SIGINT, graceful_shutdown)
        atexit.register(graceful_shutdown)
        
        port = int(os.environ.get("PORT", 8080))
        if not utils.wait_for_port_available(port, max_wait=5):
            logger.error(f"Cannot start Flask app - port {port} is not available")
            sys.exit(1)
            
        logger.info(f"Port {port} is available, starting Flask app properly!")
        
        init()

        logger.info(f"{RED_COLOR}init() finished in main()...{RESET_COLOR}")

        scheduler = BackgroundScheduler()
        # If online learning is disabled, just use the pretrained model
        if ENABLE_ONLINE_LEARNING:
            scheduler.add_job(func=online_train_routine, trigger="interval", seconds=30)
        else:
            logger.info("Online learning disabled. online_train_routine will not be invoked at all - using pretrained model only in inference")
        
        scheduler.start()
        atexit.register(lambda: scheduler.shutdown())
        
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