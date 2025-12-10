## routing_agent_service.py

# import threading
# import joblib
import pandas as pd
import numpy as np
import random
# import uvicorn
# from pydantic import BaseModel
import time
# import asyncio
# from concurrent.futures import ThreadPoolExecutor
import sys
# import concurrent.futures
import encoding
import os
# import sac
# import ppo
# import contextual_bandit
import simpler_contextual_bandit
import latency_predictor
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
from logger import logger
import queue
from collections import deque
from rwlock import RWLock

# GPU features are now always included as one-hot encoded features
INIT_DONE=False
## colors for logging
BLUE_COLOR = "\033[94m"
RED_COLOR = "\033[91m"
GREEN_COLOR = "\033[92m"
PURPLE_COLOR = "\033[95m"
CYAN_COLOR = "\033[96m"
MAGENTA_COLOR = "\033[95m"
RESET_COLOR = "\033[0m" 

app = Flask(__name__)
NUM_FLUSH = 0
ENCODED_DATA_DIR = "encoded_data"

TARGET_GPU_MODEL = os.getenv("TARGET_GPU_MODEL", "GPU-L3c")
WORKLOAD = os.getenv("WORKLOAD", "Unknown")

# Feature normalization statistics are ALWAYS based on GPU-L3c (reference baseline)
# This ensures consistent feature space across all GPU types and experiments
feature_normalization_stats_file = None
final_model_dir = None
hyperparameter_file_path = None
offline_csv_path = None
ROUTING_STRATEGY = os.getenv("ROUTING_STRATEGY", "latency_predictor")
REWARD_FUNCTION = os.getenv("REWARD_FUNCTION", "linear_simple_extended")

# hyperparameter_file_path = '/app/final_model/model_config.json'
# final_model_dir = "/app/final_model"
# feature_normalization_stats_file = f"{final_model_dir}/feature_normalization_statistics.csv"


MAX_TOTAL_DATA = int(os.getenv("MAX_TOTAL_DATA", 20000))

NUM_TRAINS = 0
MODEL_UPDATED = True
LOCK_TRAINING_DATA = threading.Lock()
first_request_starting_time = None
stats_instance = None
TOTAL_NUM_DATA = 0
NUM_NEW_DATA = 0
TOTAL_NUM_NEW_DATA = 0
TRAINING_RIGHT_NOW = False
# Training data accumulation (offline + online)
TRAINING_DF = None  # Holds all training data (offline CSV + online appended data)
TRAINING_DF_LOCK = threading.Lock()  # Thread safety for concurrent flush/train
OFFLINE_DATA_SIZE = 0  # Tracks the size of offline data portion in TRAINING_DF (shrinks as we remove overflow)
PRINT_ONCE_AT_THE_FIRST_REQUEST = True
# RL agent globals
RL_AGENT = None  # Old RL agent (entire cluster as input) - for 'rl_agent' subAlgorithm
SCALABLE_RL_AGENT = None  # New scalable RL agent (pod-independent) - for 'scalable_rl_agent' subAlgorithm
LATENCY_PREDICTOR = None  # Latency predictor model - for 'latency_predictor' subAlgorithm
CONTEXTUAL_BANDIT_AGENT = None  # Neural Contextual Bandit - for 'contextual_bandit' subAlgorithm
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

MIN_NUM_TRAINING_DATA = int(os.getenv("MIN_NUM_TRAINING_DATA", 5000))
MIN_NUM_UPDATE_DATA = int(os.getenv("MIN_NUM_UPDATE_DATA", 500))
POD_LABEL_SELECTOR = os.getenv("POD_LABEL_SELECTOR", "model.aibrix.ai/name=llama3-1-8b")
logger.info(f"POD_LABEL_SELECTOR: {POD_LABEL_SELECTOR}")
if POD_LABEL_SELECTOR == "":
    logger.error(f"POD_LABEL_SELECTOR is empty")
    assert False
ENABLE_ONLINE_LEARNING = int(os.getenv("ENABLE_ONLINE_LEARNING", 0))
EXPLORATION_ENABLED = int(os.getenv("EXPLORATION_ENABLED", 0))
TTFT_REWARD_WEIGHT = float(os.getenv("TTFT_REWARD_WEIGHT", 0.5))
EXPLORATION_RATE = float(os.getenv("EXPLORATION_RATE", 0.1))
SMOOTHING_ENABLED = int(os.getenv("SMOOTHING_ENABLED", 1))  # Enable smoothing for latency predictor
SMOOTHING_THRESHOLD = float(os.getenv("SMOOTHING_THRESHOLD", 0.1))  # 10% threshold by default
LOAD_PRETRAINED_MODEL = int(os.getenv("LOAD_PRETRAINED_MODEL", 1))  # 1=load pretrained model+offline data, 0=train from scratch (random weights, no offline data, but USE pretrained normalization stats)
INCLUDE_GPU_FEATURES = int(os.getenv("INCLUDE_GPU_FEATURES", 1))  # 1=include GPU one-hot encoding, 0=exclude GPU features
logger.info(f"Routing configuration: EXPLORATION_ENABLED={EXPLORATION_ENABLED}, EXPLORATION_RATE={EXPLORATION_RATE}, "
           f"SMOOTHING_ENABLED={SMOOTHING_ENABLED}, SMOOTHING_THRESHOLD={SMOOTHING_THRESHOLD}, "
           f"LOAD_PRETRAINED_MODEL={LOAD_PRETRAINED_MODEL}, INCLUDE_GPU_FEATURES={INCLUDE_GPU_FEATURES}")
HYPERPARAMETERS = None

BROKER_LOCK = RWLock()

# request_features_train will be set dynamically after loading HYPERPARAMETERS
# to respect EXCLUDED_REQUEST_FEATURES from the offline trained model
request_features_train = None

# Fixed handle_flush function
@app.route("/flush", methods=["POST"])
def handle_flush():
    global NUM_FLUSH, ENCODED_DATA_DIR, TOTAL_NUM_DATA, NUM_NEW_DATA, TOTAL_NUM_NEW_DATA, feature_normalization_stats_file, stats_instance, TRAINING_DF
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
        processed_df, sorted_all_pod_ids, _ = preprocess.main(podip_replaced_data_path, "", HYPERPARAMETERS)
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
        TOTAL_NUM_NEW_DATA += len(log_data)
        return jsonify({"status": "success", "message": f"Successfully processed {len(log_data)} log messages"}), 200

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Unhandled exception: {str(e)}")
        logger.error(f"Traceback: {error_traceback}")
        return jsonify({"status": "error", "message": str(e), "traceback": error_traceback}), 500


@app.route("/infer", methods=["POST"])
def handle_infer():
    global final_model_dir, NUM_TRAINS, MODEL_UPDATED, first_request_starting_time, stats_instance, HYPERPARAMETERS, PRINT_ONCE_AT_THE_FIRST_REQUEST, INIT_DONE
    if not INIT_DONE:
        logger.error(f"init() was not done yet. return 503")
        return jsonify({"error": "init() was not done yet. return 503"}), 503
    handle_infer_overhead_summary = {}
    if first_request_starting_time == None:
        first_request_starting_time = time.time()
        logger.info(f"First request starting time set to {first_request_starting_time}")
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
        processed_df, sorted_all_pod_ids, preprocess_dataset_overhead_summary = preprocess.main(None, log_message, HYPERPARAMETERS)
        if PRINT_ONCE_AT_THE_FIRST_REQUEST:
            logger.info(f"processed_df.columns: {list(processed_df.columns)}")
            logger.info(f"sorted_all_pod_ids: {sorted_all_pod_ids}")
            PRINT_ONCE_AT_THE_FIRST_REQUEST = False
        handle_infer_overhead_summary["preprocess_overhead"] = time.time() - preprocess_start_time # quite slow 65ms

        normalize_start = time.time()
        if stats_instance is None:
            logger.error(f"No running statistics available, stats_instance: {stats_instance}")
            logger.error("Cannot perform inference without normalization statistics")
            return jsonify({"error": "No normalization statistics available"}), 500
            assert False
        if stats_instance.get_max_count() == 0:
            logger.error(f"Stats instance count is 0, no data available for normalization")
            assert False

        normalizable_features, non_normalizable_features, pod_feature_types = data_normalizer._get_normalizable_features(processed_df, HYPERPARAMETERS.get('NO_NORMALIZE_FEATURES', []))
        if stats_instance.get_max_count() == 0:
            logger.error(f"request_id,{request_id},No normalization statistics available for inference")
            assert False
            
        # Validate that pooled pod feature types have statistics
        for feature_type in pod_feature_types:
            if feature_type not in stats_instance.feature_stats:
                logger.error(f"request_id,{request_id},Pooled feature type '{feature_type}' not found in stats_instance")
                logger.error(f"Available stats features: {list(stats_instance.feature_stats.keys())}")
                assert False
        
        # Validate that request features have statistics  
        for feature in normalizable_features:
            if feature not in pod_feature_types and feature not in stats_instance.feature_stats:
                logger.error(f"request_id,{request_id},Request feature '{feature}' not found in stats_instance")
                logger.error(f"Available stats features: {list(stats_instance.feature_stats.keys())}")
                assert False
                
        # Normalize features (with special handling for pooled pod features)
        for feature in normalizable_features:
            ##################################################
            # Check if this is a pooled pod feature type (not a column name)
            if feature in pod_feature_types:
                # Find all pod columns for this feature type and normalize each
                matching_columns = [col for col in processed_df.columns 
                                  if data_normalizer._extract_pod_feature_type(col) == feature]
                for col in matching_columns:
                    data_normalizer._normalize_single_feature(processed_df, col, stats_instance, update_statistics=False, request_id=request_id)
            else:
                # Regular feature - normalize directly
                data_normalizer._normalize_single_feature(processed_df, feature, stats_instance, update_statistics=False, request_id=request_id)
            ##################################################
        handle_infer_overhead_summary["normalize"] = time.time() - normalize_start
        
        # 🔍 CHECKPOINT 2: Log NORMALIZED values (for request_id % 10 == 0)
        if int(request_id) % 10 == 0:
            logger.info(f"🔍 CHECKPOINT 2 - AFTER NORMALIZATION (requestID {request_id}):")
            logger.info(f"   Request features: input_tokens={processed_df['input_tokens'].iloc[0]:.4f}, output_tokens={processed_df['output_tokens'].iloc[0]:.4f}")
            for pod_id in sorted_all_pod_ids[:3]:
                kv = processed_df[f'{pod_id}-kv_hit_ratio'].iloc[0]
                inflight = processed_df[f'{pod_id}-inflight_requests'].iloc[0]
                prefill = processed_df[f'{pod_id}-prefill_tokens'].iloc[0]
                logger.info(f"   {pod_id}: kv_hit_ratio={kv:.4f}, inflight={inflight:.4f}, prefill={prefill:.4f}")

        ## Encode data (normalization already done)
        encode_start_time = time.time()
        tensor_data, encode_for_inference_overhead_summary = encoding.encode_for_inference(sorted_all_pod_ids, processed_df, request_features_train, HYPERPARAMETERS)
        handle_infer_overhead_summary["encode"] = time.time() - encode_start_time
        
        # 🔍 CHECKPOINT 3: Log TENSOR values after encoding (for request_id % 10 == 0)
        if int(request_id) % 10 == 0:
            logger.info(f"🔍 CHECKPOINT 3 - AFTER ENCODING TO TENSORS (requestID {request_id}):")
            logger.info(f"   request_features shape: {tensor_data['request_features'].shape}, values: {tensor_data['request_features'][0].numpy()}")
            logger.info(f"   pod_features_with_staleness shape: {tensor_data['pod_features_with_staleness'].shape}")
            logger.info(f"   pod_0000 features: {tensor_data['pod_features_with_staleness'][0, 0, :].numpy()}")
            logger.info(f"   kv_hit_ratios shape: {tensor_data['kv_hit_ratios'].shape}")
            logger.info(f"   pod_0000 kv: {tensor_data['kv_hit_ratios'][0, 0, :].numpy()}")

        infer_from_tensor_start_time = time.time()
        
        get_subAlgorithm_start_time = time.time()
        subAlgorithm = processed_df['subAlgorithm'].iloc[0]
        logger.debug(f"requestID: {request_id}, subAlgorithm: {subAlgorithm}")
        handle_infer_overhead_summary["get_subAlgorithm"] = time.time() - get_subAlgorithm_start_time
        
        # "random"
        # "least-latency"
        # "least-request"
        # "least-kv-cache"
        if subAlgorithm == 'latency_predictor' or subAlgorithm in {'random', 'least_latency', 'least_request', 'least_kv_cache', 'prefix_cache_1', 'prefix_cache_2'}:
            global LATENCY_PREDICTOR, LOAD_PRETRAINED_MODEL
            # OPTIMIZATION: Check without lock first (fast path for initialized state)
            # Only acquire expensive write lock if truly uninitialized
            if LATENCY_PREDICTOR is None:
                latency_predictor_write_lock_contention_start_time = time.time()
                with LATENCY_PREDICTOR_LOCK.write():
                    latency_predictor_write_lock_contention_overhead = time.time() - latency_predictor_write_lock_contention_start_time
                    handle_infer_overhead_summary['latency_predictor_write_lock_contention'] = latency_predictor_write_lock_contention_overhead
                    
                    # Double-check after acquiring lock (another thread may have initialized)
                    latency_predictor_object_create_start_time = time.time()
                    if LATENCY_PREDICTOR is None:
                        state_dims = {
                            'pod_features': tensor_data['pod_features_with_staleness'].shape[2],
                            'kv_hit_ratios': tensor_data['kv_hit_ratios'].shape[2],
                            'request_features': tensor_data['request_features'].shape[1],
                            'num_pods': tensor_data['pod_features_with_staleness'].shape[1]
                        }
                        
                        logger.info(f"🔄 One-time initialization of LATENCY_PREDICTOR with state_dims={state_dims}")
                        LATENCY_PREDICTOR = latency_predictor.LatencyPredictor(state_dims, HYPERPARAMETERS, final_model_dir)
                        # Load model weights based on LOAD_PRETRAINED_MODEL setting
                        model_path = os.path.join(final_model_dir, 'latency_predictor.pth')
                        if LOAD_PRETRAINED_MODEL:
                            # LOAD_PRETRAINED_MODEL=1: Must load pretrained model (fail if missing)
                            if os.path.exists(model_path):
                                try:
                                    LATENCY_PREDICTOR.load(final_model_dir)
                                    logger.info(f"✅ LOAD_PRETRAINED_MODEL=1, loaded pretrained model from final_model_dir: {final_model_dir}, model_path: {model_path}, offline_csv_path: {offline_csv_path}, hyperparameter_file_path: {hyperparameter_file_path}")
                                except Exception as e:
                                    logger.error(f"❌ LOAD_PRETRAINED_MODEL=1 but failed to load: {e}")
                                    raise
                            else:
                                logger.error(f"❌ LOAD_PRETRAINED_MODEL=1 but no model found at {model_path}")
                                logger.error("Set LOAD_PRETRAINED_MODEL=0 to use random weights, or provide a trained model")
                                raise FileNotFoundError(f"Missing model file: {model_path}")
                        else:
                            # LOAD_PRETRAINED_MODEL=0: Only load if model was created by online training
                            # Check if model exists AND was created by our training (NUM_TRAINS > 0)
                            if NUM_TRAINS > 0 and os.path.exists(model_path):
                                try:
                                    LATENCY_PREDICTOR.load(final_model_dir)
                                    logger.info(f"✅ LOAD_PRETRAINED_MODEL=0, loaded model from online training (NUM_TRAINS={NUM_TRAINS})")
                                except Exception as e:
                                    logger.warning(f"⚠️  Failed to load online-trained model: {e}")
                                    logger.warning(f"⚠️  Continuing with random weights")
                            else:
                                logger.warning(f"⚠️  LOAD_PRETRAINED_MODEL=0: Using RANDOM WEIGHTS (NUM_TRAINS={NUM_TRAINS})")
                                logger.warning(f"⚠️  Model will make poor decisions until first training completes ({MIN_NUM_TRAINING_DATA} samples)")
                    handle_infer_overhead_summary["latency_predictor_object_create"] = time.time() - latency_predictor_object_create_start_time
            
            latency_predictor_read_lock_contention_start_time = time.time()
            # Inference with read lock (allows concurrent requests)
            with LATENCY_PREDICTOR_LOCK.read():
                global EXPLORATION_RATE, SMOOTHING_ENABLED, SMOOTHING_THRESHOLD
                handle_infer_overhead_summary["latency_predictor_read_lock_contention"] = time.time() - latency_predictor_read_lock_contention_start_time
                # Get exploration and smoothing parameters
                result, infer_from_tensor_overhead_summary = latency_predictor.infer_latency_predictor_with_model(
                    predictor=LATENCY_PREDICTOR,
                    tensor_data=tensor_data,
                    request_id=request_id,
                    sorted_all_pod_ids=sorted_all_pod_ids,
                    exploration_rate=EXPLORATION_RATE,
                    smoothing=bool(SMOOTHING_ENABLED),
                    smoothing_threshold=SMOOTHING_THRESHOLD,
                    generalpodid_to_gpu_model=HYPERPARAMETERS['generalpodid_to_gpu_model']
                )
        elif subAlgorithm == 'contextual_bandit':
            # === NEURAL CONTEXTUAL BANDIT (NEW IMPLEMENTATION) ===
            logger.info(f"requestID: {request_id}, subAlgorithm: {subAlgorithm}, Using Neural Contextual Bandit for inference")
            
            global CONTEXTUAL_BANDIT_AGENT
            import neural_contextual_bandit
            
            # OPTIMIZATION: Check without lock first (fast path for initialized state)
            if CONTEXTUAL_BANDIT_AGENT is None:
                contextual_bandit_write_lock_start = time.time()
                with LATENCY_PREDICTOR_LOCK.write():  # Reuse existing lock infrastructure
                    handle_infer_overhead_summary['contextual_bandit_write_lock'] = time.time() - contextual_bandit_write_lock_start
                    
                    # Double-check after acquiring lock
                    if CONTEXTUAL_BANDIT_AGENT is None:
                        contextual_bandit_create_start = time.time()
                        state_dims = {
                            'pod_features': tensor_data['pod_features_with_staleness'].shape[2],
                            'kv_hit_ratios': tensor_data['kv_hit_ratios'].shape[2],
                            'request_features': tensor_data['request_features'].shape[1],
                            'num_pods': tensor_data['pod_features_with_staleness'].shape[1]
                        }
                        
                        logger.info(f"🔄 One-time initialization of CONTEXTUAL_BANDIT_AGENT with state_dims={state_dims}")
                        
                        # NOTE: We don't create the agent here - neural_contextual_bandit.infer_from_tensor() 
                        # handles lazy initialization and caching internally
                        CONTEXTUAL_BANDIT_AGENT = "initialized"  # Marker to prevent re-initialization
                        
                        handle_infer_overhead_summary["contextual_bandit_create"] = time.time() - contextual_bandit_create_start
            
            # Inference with the neural contextual bandit
            contextual_bandit_infer_start = time.time()
            result, infer_from_tensor_overhead_summary = neural_contextual_bandit.infer_from_tensor(
                tensor_data=tensor_data,
                request_id=request_id,
                model_updated=MODEL_UPDATED,
                HYPERPARAMETERS=HYPERPARAMETERS,
                final_model_dir=final_model_dir,
                sorted_all_pod_ids=sorted_all_pod_ids
            )
            # Reset MODEL_UPDATED after agent has processed it (to avoid reloading on every request)
            if MODEL_UPDATED:
                MODEL_UPDATED = False
            handle_infer_overhead_summary['contextual_bandit_infer'] = time.time() - contextual_bandit_infer_start
            
            # Format result to match expected interface
            # The neural contextual bandit returns 'selected_pod_index' and 'predicted_rewards'
            # We need to add the missing fields that routing service expects
            result['pod_probabilities'] = {
                sorted_all_pod_ids[i]: 1.0 / len(sorted_all_pod_ids) 
                for i in range(len(sorted_all_pod_ids))
            }
            result['explore_mask'] = 1 if result.get('epsilon', 0) > 0 else 0
            
            logger.info(f"Neural CB inference for {request_id}: pod={result['selected_pod_index']}, predicted_rewards={result['predicted_rewards']}, chosen_pod_predicted_reward={result['confidence']}, epsilon={result.get('epsilon', 0):.3f}")
            
        elif subAlgorithm == 'rl_agent':
            from rl_routing_agent_sb3 import create_rl_routing_agent_sb3, infer_rl_agent
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
                        **HYPERPARAMETERS
                    )
                    ckpt_path = HYPERPARAMETERS.get('RL_CHECKPOINT_PATH')
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
                hyperparameters=HYPERPARAMETERS,
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
            }
            
            logger.info(f"scalable_rl_routing_agent, requestID: {request_id}, action={pod_idx}, prev_reward={prev_reward:.2f}, confidence={confidence:.3f}, num_pods={num_pods}")
        elif subAlgorithm == 'scalable_rl_agent_old':
            # === NEW SCALABLE RL AGENT (pod-count independent) ===
            logger.info(f"requestID: {request_id}, subAlgorithm: {subAlgorithm}, Using SCALABLE RL agent (pod-independent) for inference")
            
            global SCALABLE_RL_AGENT
            
            with SCALABLE_RL_AGENT_LOCK.write():
                if SCALABLE_RL_AGENT is None:
                    # Initialize ONCE - works for any number of pods!
                    pod_features_t = tensor_data['pod_features']
                    per_pod_dim = int(pod_features_t.shape[2])  # e.g., 10
                    kv_hit_t = tensor_data['kv_hit_ratios']
                    kv_dim = int(kv_hit_t.shape[2])  # e.g., 1
                    req_features_t = tensor_data['request_features']
                    req_dim = int(req_features_t.shape[1])  # e.g., 3
                    
                    # Per-pod dimension = pod_features + kv_hit_ratios
                    total_per_pod_dim = per_pod_dim + kv_dim
                    
                    SCALABLE_RL_AGENT = create_scalable_rl_agent(
                        per_pod_dim=total_per_pod_dim,  # 11 (10 pod + 1 kv)
                        request_dim=req_dim,             # 3
                        max_pods=100,                    # Max expected pods
                        **HYPERPARAMETERS
                    )
                    
                    # Load checkpoint if available
                    ckpt_path = HYPERPARAMETERS.get('RL_CHECKPOINT_PATH')
                    if ckpt_path and os.path.exists(ckpt_path):
                        try:
                            SCALABLE_RL_AGENT.load(ckpt_path)
                            logger.info(f"✅ Loaded scalable RL checkpoint from {ckpt_path}")
                        except Exception as e:
                            logger.warning(f"⚠️  Failed to load RL checkpoint {ckpt_path}: {e}")
                    
                    logger.info(f"🚀 Initialized SCALABLE RL agent: per_pod_dim={total_per_pod_dim}, "
                              f"request_dim={req_dim}, max_pods=100 (works with ANY #pods!)")
                
                current_agent = SCALABLE_RL_AGENT
            
            # Inference (no lock needed - thread-safe in new design)
            current_agent, result, infer_from_tensor_overhead_summary = infer_scalable_rl_agent(
                tensor_data=tensor_data,
                request_id=request_id,
                sorted_all_pod_ids=sorted_all_pod_ids,
                processed_df=processed_df,
                rl_agent=current_agent,
                hyperparameters=HYPERPARAMETERS,
                agent_lock=None  # New agent doesn't need lock for prediction
            )
        else:
            logger.error(f"Unknown subAlgorithm: {subAlgorithm}")
            assert False
        handle_infer_overhead_summary["calling_infer_from_tensor"] = time.time() - infer_from_tensor_start_time
        
        remaining_work_start = time.time()
        result["requestID"] = request_id
        result["num_trains"] = NUM_TRAINS
        logger.info(f"requestID: {request_id}, inference result: {result}")
        
        # Map the pod index back to the actual pod ID
        selected_pod_index = result['selected_pod_index']
        if selected_pod_index >= len(sorted_all_pod_ids):
            logger.error(f"Selected pod index {selected_pod_index} out of range, defaulting to first pod")
            assert False
        selected_pod_generalpodid = sorted_all_pod_ids[selected_pod_index]
        if selected_pod_generalpodid not in HYPERPARAMETERS['generalpodid_to_pod_ip']:
            logger.error(f"selected_pod_generalpodid: {selected_pod_generalpodid} not found in HYPERPARAMETERS['generalpodid_to_pod_ip']")
            logger.error(f"HYPERPARAMETERS['generalpodid_to_pod_ip']: {HYPERPARAMETERS['generalpodid_to_pod_ip']}")
            assert False
        selected_pod_ip = HYPERPARAMETERS['generalpodid_to_pod_ip'][selected_pod_generalpodid]
            
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
            
        logger.debug(f"overhead_log: {overhead_log}")
        
        response = {
            "num_trains": NUM_TRAINS,
            "num_flush": NUM_FLUSH,
            "request_timestamp": int(time.time() - first_request_starting_time),
            "selected_pod": selected_pod_ip,
            "selected_pod_generalpodid": selected_pod_generalpodid,
            "selected_pod_gpu_type": result.get('selected_pod_gpu_type', 'unknown'),
            "confidence": round(result['confidence'], 2),
            "exploration": result['explore_mask'],
            "exploration_enabled": EXPLORATION_ENABLED,
            "request_id": request_id,
            "overhead_log": overhead_log,
            "predicted_latencies": result.get('predicted_latencies', {pod_id: -99 for pod_id in sorted_all_pod_ids}),
            "chosen_pod_predicted_latency": result.get('chosen_pod_predicted_latency', -99),
            "predicted_rewards": result.get('predicted_rewards', {pod_id: -99 for pod_id in sorted_all_pod_ids}),
            "chosen_pod_predicted_reward": result.get('chosen_pod_predicted_reward', -99),
            "smoothing": result.get('smoothing_mask', 0),  # Include smoothing indicator
        }
        
        # Return 503 if using random weights (LOAD_PRETRAINED_MODEL=0 and no training yet)
        # Pipeline still runs (for verification/warmup) but client gets proper "not ready" signal
        if LOAD_PRETRAINED_MODEL == 0 and NUM_TRAINS == 0:
            logger.warning(f"⚠️  Inference completed with RANDOM WEIGHTS (NUM_TRAINS=0), returning 503")
            logger.warning(f"⚠️  Need {MIN_NUM_TRAINING_DATA - TOTAL_NUM_NEW_DATA} more samples before first training")
            response["error"] = "Model not trained yet"
            response["reason"] = "LOAD_PRETRAINED_MODEL=0 and NUM_TRAINS=0 (using random weights)"
            response["samples_needed"] = MIN_NUM_TRAINING_DATA
            response["samples_collected"] = TOTAL_NUM_NEW_DATA
            response["message"] = f"Service will be ready after first training ({MIN_NUM_TRAINING_DATA} samples)"
            return jsonify(response), 503
            
        return jsonify(response), 200
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Error in handle_infer: {str(e)}")
        logger.error(f"Traceback: {error_traceback}")
        return jsonify({"error": str(e), "traceback": error_traceback}), 500


def online_train_routine():
    global NUM_TRAINS, MODEL_UPDATED, TOTAL_NUM_DATA, final_model_dir, NUM_NEW_DATA, TOTAL_NUM_NEW_DATA, HYPERPARAMETERS, TRAINING_RIGHT_NOW, LATENCY_PREDICTOR, TRAINING_DF, stats_instance, OFFLINE_DATA_SIZE, MAX_TOTAL_DATA
    if TRAINING_RIGHT_NOW:
        logger.info(f"Previous training still in progress, skipping training")
        return
    if TOTAL_NUM_NEW_DATA < MIN_NUM_TRAINING_DATA:
        logger.info(f"Not enough total training data available, NUM_NEW_DATA: {TOTAL_NUM_NEW_DATA} < {MIN_NUM_TRAINING_DATA}, wait until enough data are added. NUM_TRAINS: {NUM_TRAINS}, TOTAL_NUM_DATA: {TOTAL_NUM_DATA}")
        return
    if NUM_NEW_DATA < MIN_NUM_UPDATE_DATA:
        logger.info(f"Not enough new training data available, NUM_NEW_DATA: {NUM_NEW_DATA} < {MIN_NUM_UPDATE_DATA}, wait until enough data are added. NUM_TRAINS: {NUM_TRAINS}, TOTAL_NUM_DATA: {TOTAL_NUM_DATA}")
        return
    TRAINING_RIGHT_NOW = True
    training_start_time = time.time()
    logger.info(f"online_train_routine start, {NUM_TRAINS}th online training with {NUM_NEW_DATA} new training data")
    try:
        # Route to appropriate training function based on model type
        model_type = HYPERPARAMETERS['MODEL_TYPE']
        if model_type == 'latency_predictor':
            logger.info(f"Training with latency predictor model on entire dataset (offline({len(TRAINING_DF)}) + online({NUM_NEW_DATA}))")

            # Get a copy of TRAINING_DF for training (thread-safe)
            with TRAINING_DF_LOCK:
                if TRAINING_DF is None or len(TRAINING_DF) == 0:
                    logger.error("TRAINING_DF is empty, cannot train")
                    TRAINING_RIGHT_NOW = False
                    return
                training_df_copy = TRAINING_DF.copy()
                total_samples = len(training_df_copy)
                current_offline_size = OFFLINE_DATA_SIZE

            # logger.info(f"Training on (offline data: {ENCODED_DATA_DIR}, online data: {ENCODED_DATA_DIR}, total data: {total_samples}")
            logger.info(f"Training on total data: {total_samples} (offline: {current_offline_size}, online: {total_samples - current_offline_size})")

            # Remove overflow from offline data if total exceeds 20,000
            if total_samples > MAX_TOTAL_DATA and current_offline_size > 0:
                overflow = total_samples - MAX_TOTAL_DATA
                
                # Calculate how much to remove from offline portion
                to_remove = min(overflow, current_offline_size)
                
                # Remove from the beginning of dataframe (where offline data resides)
                training_df_copy = training_df_copy.iloc[to_remove:].reset_index(drop=True)
                logger.info(f"⚠️  Total data ({total_samples}) exceeds limit ({MAX_TOTAL_DATA}). "
                           f"Removed {to_remove} oldest offline samples (overflow={overflow}, offline_size={current_offline_size})")
                
                # Update the global TRAINING_DF and OFFLINE_DATA_SIZE
                with TRAINING_DF_LOCK:
                    TRAINING_DF = training_df_copy.copy()
                    OFFLINE_DATA_SIZE = max(0, current_offline_size - to_remove)
                    logger.info(f"✅ Updated TRAINING_DF: new size={len(TRAINING_DF)} (offline: {OFFLINE_DATA_SIZE}, online: {len(TRAINING_DF) - OFFLINE_DATA_SIZE})")
                
                # Update total_samples for the rest of the function
                total_samples = len(training_df_copy)

            # Drop non-numeric metadata columns from offline CSV that are absent online
            metadata_cols_to_drop = ['source_file', 'reward_function_used']
            cols_present_to_drop = [c for c in metadata_cols_to_drop if c in training_df_copy.columns]
            if cols_present_to_drop:
                logger.info(f"Dropping metadata columns not used for training: {cols_present_to_drop}")
                training_df_copy = training_df_copy.drop(columns=cols_present_to_drop)

            ############################################################################
            # Handle missing values proactively to avoid intermittent failures due to dynamic pod columns
            try:
                # numpy already imported as np at top of file
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
            normalizable_features, non_normalizable_features, pod_feature_types = data_normalizer._get_normalizable_features(
                training_df_copy, HYPERPARAMETERS.get('NO_NORMALIZE_FEATURES', []))
            
            # CRITICAL: Skip pooled statistics computation during online training
            # The pretrained model was trained with PER-POD normalization (each pod normalized separately).
            # Computing pooled statistics (all pods normalized together) creates incompatible feature distributions
            # that cause catastrophic model divergence (loss explodes to millions, R² becomes negative).
            # During online training, we MUST maintain the same normalization strategy as the pretrained model.
            if pod_feature_types:
                logger.info(f"🔄 Found {len(pod_feature_types)} pod feature types: {pod_feature_types}")
                logger.info(f"⚠️  SKIPPING pooled statistics computation during online training")
                logger.info(f"   Reason: Pretrained model uses per-pod normalization - must maintain compatibility")
                logger.info(f"   Using existing per-pod stats from loaded stats_instance")
            # Note: stats_instance already contains per-pod stats loaded from pretrained model
            # We do NOT call _compute_pooled_pod_statistics() during online training
            
            # VERIFICATION: Log normalization stats BEFORE normalization
            logger.info(f"VERIFICATION: Checking normalization stats before online training #{NUM_TRAINS}")
            for feature in normalizable_features:
                if feature in stats_instance.feature_stats:
                    stats = stats_instance.feature_stats[feature]
                    logger.info(f"VERIFICATION BEFORE: {feature} - OLD stats: count={stats.count}, mean={stats.mean}, std={stats.std}")
                    
                    # For request features, log actual values
                    if feature not in pod_feature_types and feature in training_df_copy.columns:
                        actual_values = training_df_copy[feature].values
                        new_mean = actual_values.mean()
                        new_std = actual_values.std()
                        logger.info(f"VERIFICATION BEFORE: {feature} - NEW data: min={actual_values.min():.3f}, max={actual_values.max():.3f}, mean={new_mean:.3f}, std={new_std:.3f}")
                        
                        # **PROOF OF CAUSATION**: Calculate what normalization would produce with OLD vs NEW stats
                        # Take a sample value from the new data
                        sample_value = actual_values[0]
                        old_mean = stats.mean[0] if hasattr(stats.mean, '__getitem__') else stats.mean
                        old_std = stats.std[0] if hasattr(stats.std, '__getitem__') else stats.std
                        if old_std > 0 and new_std > 0:
                            normalized_with_old_stats = (sample_value - old_mean) / old_std
                            normalized_with_new_stats = (sample_value - new_mean) / new_std
                            logger.info(f"VERIFICATION PROOF: {feature} sample_value={sample_value:.3f}")
                            logger.info(f"VERIFICATION PROOF: {feature} normalized_with_OLD_stats = ({sample_value:.3f} - {old_mean:.3f}) / {old_std:.3f} = {normalized_with_old_stats:.3f}")
                            logger.info(f"VERIFICATION PROOF: {feature} normalized_with_NEW_stats = ({sample_value:.3f} - {new_mean:.3f}) / {new_std:.3f} = {normalized_with_new_stats:.3f}")
                            logger.info(f"VERIFICATION PROOF: {feature} DIFFERENCE = {abs(normalized_with_old_stats - normalized_with_new_stats):.3f}")
                            if abs(normalized_with_old_stats) > 5:
                                logger.error(f"VERIFICATION PROOF: {feature} OUTLIER CREATED! Using old stats produces {normalized_with_old_stats:.3f} (should be ~{normalized_with_new_stats:.3f})")

            # Normalize features (with special handling for pooled pod features)
            for feature in normalizable_features:
                # Check if this is a pooled pod feature type (not a column name)
                if feature in pod_feature_types:
                    # Find all pod columns for this feature type and normalize each
                    matching_columns = [col for col in training_df_copy.columns 
                                      if data_normalizer._extract_pod_feature_type(col) == feature]
                    for col in matching_columns:
                        data_normalizer._normalize_single_feature(training_df_copy, col, stats_instance, update_statistics=False)
                else:
                    # Regular feature - normalize directly
                    data_normalizer._normalize_single_feature(training_df_copy, feature, stats_instance, update_statistics=False)
            
            # VERIFICATION: Log normalization stats AFTER normalization
            logger.info(f"VERIFICATION: Checking normalization stats after online training #{NUM_TRAINS}")
            for feature in normalizable_features:
                if feature in stats_instance.feature_stats:
                    stats = stats_instance.feature_stats[feature]
                    logger.info(f"VERIFICATION AFTER: {feature} - count={stats.count}, mean={stats.mean}, std={stats.std}")
                    
                    # For request features, log normalized values
                    if feature not in pod_feature_types and feature in training_df_copy.columns:
                        normalized_values = training_df_copy[feature].values
                        logger.info(f"VERIFICATION AFTER: {feature} - normalized: min={normalized_values.min():.3f}, max={normalized_values.max():.3f}, mean={normalized_values.mean():.3f}, std={normalized_values.std():.3f}")

            # Get sorted pod IDs from the training data (preprocessed CSV format)
            sorted_all_pod_ids = utils.get_sorted_all_pod_ids('processed_csv_columns', training_df_copy.columns.tolist())
            logger.info(f"Training with pods: {sorted_all_pod_ids}")

            # Encode the entire dataset
            encode_start_time = time.time()
            os.makedirs(ENCODED_DATA_DIR, exist_ok=True)
            encoded_training_dir = os.path.join(ENCODED_DATA_DIR, "full_training_data")
            encoding.encode_for_train(sorted_all_pod_ids, training_df_copy, encoded_training_dir, request_features_train, HYPERPARAMETERS)
            logger.info(f"Encoded {total_samples} samples to {encoded_training_dir}, encode time: {time.time() - encode_start_time} seconds")

            # Train on the encoded dataset
            # Pass NUM_NEW_DATA so the training function can evaluate separately on new samples
            train_start_time = time.time()
            latency_predictor.train_latency_predictor(
                encoded_training_dir, 
                final_model_dir, 
                HYPERPARAMETERS, 
                NUM_TRAINS,
                num_new_samples=NUM_NEW_DATA  # Track newly collected online data for separate evaluation
            )
            logger.info(f"train_latency_predictor done, train time: {time.time() - train_start_time} seconds")

            # Reload model in training thread (truly non-blocking for inference)
            # Strategy: Load new model WITHOUT lock, then atomic swap with brief lock
            if LATENCY_PREDICTOR is not None:
                load_start_time = time.time()
                
                # Step 1: Load new model WITHOUT holding lock (no blocking!)
                # Get model architecture from existing predictor
                new_predictor = latency_predictor.LatencyPredictor(
                    LATENCY_PREDICTOR.state_dims, 
                    HYPERPARAMETERS, 
                    final_model_dir
                )
                new_predictor.load(final_model_dir)
                load_time = time.time() - load_start_time
                logger.info(f"Loaded new model weights (no blocking), load time: {load_time:.3f}s")
                
                # Step 2: Atomic swap with VERY brief write lock (microseconds, not milliseconds!)
                swap_start_time = time.time()
                with LATENCY_PREDICTOR_LOCK.write():
                    LATENCY_PREDICTOR = new_predictor  # Atomic reference swap
                swap_time = time.time() - swap_start_time
                logger.info(f"Swapped model reference (atomic), swap time: {swap_time*1000:.2f}ms")
                logger.info(f"✅ Model reload complete with ZERO inference blocking during load")
        
        elif model_type == 'contextual_bandit':
            logger.info(f"Training Neural Contextual Bandit on entire dataset (offline + online: {NUM_NEW_DATA} new)")
            
            # Get a copy of TRAINING_DF for training (same as latency predictor)
            with TRAINING_DF_LOCK:
                if TRAINING_DF is None or len(TRAINING_DF) == 0:
                    logger.error("TRAINING_DF is empty, cannot train")
                    TRAINING_RIGHT_NOW = False
                    return
                training_df_copy = TRAINING_DF.copy()
                total_samples = len(training_df_copy)
                current_offline_size = OFFLINE_DATA_SIZE
            
            logger.info(f"Training on total data: {total_samples} (offline: {current_offline_size}, online: {total_samples - current_offline_size})")
            
            # Remove overflow if needed (same logic as latency predictor)
            if total_samples > MAX_TOTAL_DATA and current_offline_size > 0:
                overflow = total_samples - MAX_TOTAL_DATA
                to_remove = min(overflow, current_offline_size)
                training_df_copy = training_df_copy.iloc[to_remove:].reset_index(drop=True)
                logger.info(f"⚠️  Removed {to_remove} oldest offline samples")
                
                with TRAINING_DF_LOCK:
                    TRAINING_DF = training_df_copy.copy()
                    OFFLINE_DATA_SIZE = max(0, current_offline_size - to_remove)
                
                total_samples = len(training_df_copy)
            
            # Drop metadata columns
            metadata_cols_to_drop = ['source_file', 'reward_function_used']
            cols_present_to_drop = [c for c in metadata_cols_to_drop if c in training_df_copy.columns]
            if cols_present_to_drop:
                training_df_copy = training_df_copy.drop(columns=cols_present_to_drop)
            
            # Handle missing values (numpy already imported at top of file)
            numeric_columns = training_df_copy.select_dtypes(include=[np.number]).columns
            total_missing_numeric = int(training_df_copy[numeric_columns].isna().sum().sum()) if len(numeric_columns) > 0 else 0
            if total_missing_numeric > 0:
                logger.warning(f"Filling {total_missing_numeric} missing numeric values with 0")
                training_df_copy[numeric_columns] = training_df_copy[numeric_columns].fillna(0)
            
            # Normalize the entire dataset (reuse same logic)
            normalizable_features, non_normalizable_features, pod_feature_types = data_normalizer._get_normalizable_features(
                training_df_copy, HYPERPARAMETERS.get('NO_NORMALIZE_FEATURES', []))
            
            # Normalize features
            for feature in normalizable_features:
                if feature in pod_feature_types:
                    matching_columns = [col for col in training_df_copy.columns if data_normalizer._extract_pod_feature_type(col) == feature]
                    for col in matching_columns:
                        data_normalizer._normalize_single_feature(training_df_copy, col, stats_instance, update_statistics=False)
                else:
                    data_normalizer._normalize_single_feature(training_df_copy, feature, stats_instance, update_statistics=False)
            
            # Get sorted pod IDs
            sorted_all_pod_ids = utils.get_sorted_all_pod_ids('processed_csv_columns', training_df_copy.columns.tolist())
            logger.info(f"Training with pods: {sorted_all_pod_ids}")
            
            # Encode the entire dataset
            encode_start_time = time.time()
            os.makedirs(ENCODED_DATA_DIR, exist_ok=True)
            encoded_training_dir = os.path.join(ENCODED_DATA_DIR, "full_training_data")
            encoding.encode_for_train(sorted_all_pod_ids, training_df_copy, encoded_training_dir, request_features_train, HYPERPARAMETERS)
            logger.info(f"Encoded {total_samples} samples to {encoded_training_dir}, encode time: {time.time() - encode_start_time} seconds")
            
            # Train Neural Contextual Bandit
            train_start_time = time.time()
            import neural_contextual_bandit
            neural_contextual_bandit.train_batch(
                encoded_training_dir=encoded_training_dir,
                final_model_dir=final_model_dir,
                HYPERPARAMETERS=HYPERPARAMETERS,
                num_epochs=HYPERPARAMETERS.get('num_epochs', 3)
            )
            logger.info(f"Neural CB batch training done, train time: {time.time() - train_start_time} seconds")
        
        else:
            logger.error(f"Unknown MODEL_TYPE: {model_type}")
            TRAINING_RIGHT_NOW = False
            return
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
                    checkpoint_file = os.path.join(HYPERPARAMETERS['CHECKPOINT_DIR'], 'scalable_rl_agent')
                    
                    eval_freq = (HYPERPARAMETERS['num_requests_per_episode'] + 1) * HYPERPARAMETERS['num_episodes_per_iteration'] # how often to evaluate the model, the eval will be triggered every eval_freq steps. Hence, eval_freq should be the number of requests per iteration
                    SCALABLE_RL_AGENT.train(
                        save_path=checkpoint_file,
                        eval_freq=eval_freq,
                        n_eval_episodes=HYPERPARAMETERS['n_eval_episodes'],
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
    global HYPERPARAMETERS, stats_instance
    
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
        pod_ip_to_generalpodid = HYPERPARAMETERS.get('pod_ip_to_generalpodid', {})
        pod_ip_to_gpu_model_encoded = HYPERPARAMETERS.get('pod_ip_to_gpu_model_encoded', {})
        
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


def init():
    global HYPERPARAMETERS, TARGET_GPU_MODEL, stats_instance, INIT_DONE, final_model_dir, offline_csv_path, hyperparameter_file_path
    
    # Model directory and offline data are GPU-specific
    if TARGET_GPU_MODEL == "NVIDIA-A30" or TARGET_GPU_MODEL == "NVIDIA-L4":
        final_model_dir = "/app/final_model/NVIDIA-A30"
        hyperparameter_file_path = f"${final_model_dir}/model_config.json"
        offline_csv_path = "/app/NVIDIA-A30/offline_training_data.csv"
        feature_normalization_stats_file = "/app/final_model/NVIDIA-A30/feature_normalization_statistics.csv"
    elif TARGET_GPU_MODEL == "NVIDIA-A10":
        # Check if we need contextual_bandit or latency_predictor model
        # Priority: Check if contextual_bandit model exists, otherwise use latency_predictor
        if ROUTING_STRATEGY == "contextual_bandit":
            final_model_dir = f"/app/final_model/NVIDIA-A10/PrefillOnly/contextual_bandit/${REWARD_FUNCTION}"
            logger.info(f"Using CONTEXTUAL BANDIT model from {final_model_dir}")
        elif ROUTING_STRATEGY == "latency_predictor":
            final_model_dir = "/app/final_model/NVIDIA-A10/PrefillOnly/latency_predictor"
            logger.info(f"Using LATENCY PREDICTOR model from {final_model_dir}")
        hyperparameter_file_path = f"{final_model_dir}/model_config.json"
        feature_normalization_stats_file = f"{final_model_dir}/feature_normalization_statistics.csv"
        offline_csv_path = "/app/NVIDIA-A10/PrefillOnly/offline_training_data.csv"
    elif TARGET_GPU_MODEL == "GPU-L3c" or TARGET_GPU_MODEL == "NVIDIA-L40" or TARGET_GPU_MODEL == "NVIDIA-L40S" or TARGET_GPU_MODEL == "NVIDIA-L20":
        hyperparameter_file_path = "/app/final_model/NVIDIA-L20/model_config.json"
        final_model_dir = "/app/final_model/NVIDIA-L20"
        offline_csv_path = "/app/NVIDIA-L20/offline_training_data.csv"
        feature_normalization_stats_file = "/app/final_model/NVIDIA-L20/feature_normalization_statistics.csv"
    elif TARGET_GPU_MODEL == "hetero":
        hyperparameter_file_path = "/app/final_model/hetero/model_config.json"
        final_model_dir = "/app/final_model/hetero"
        offline_csv_path = "/app/hetero/offline_training_data.csv"
        feature_normalization_stats_file = "/app/final_model/hetero/feature_normalization_statistics.csv"
    else:
        logger.error(f"Unknown target GPU model: {TARGET_GPU_MODEL}")
        assert False

    logger.info(f"TARGET_GPU_MODEL: {TARGET_GPU_MODEL}")
    logger.info(f"WORKLOAD: {WORKLOAD}")
    logger.info(f"final_model_dir: {final_model_dir}")
    logger.info(f"hyperparameter_file_path: {hyperparameter_file_path}")
    logger.info(f"offline_csv_path: {offline_csv_path}")
    logger.info(f"Feature normalization stats: {feature_normalization_stats_file}")
    
    assert final_model_dir is not None, f"final_model_dir is None for {TARGET_GPU_MODEL}"
    assert hyperparameter_file_path is not None, f"hyperparameter_file_path is None for {TARGET_GPU_MODEL}"
    assert offline_csv_path is not None, f"offline_csv_path is None for {TARGET_GPU_MODEL}"
    assert feature_normalization_stats_file is not None, f"feature_normalization_stats_file is None for {TARGET_GPU_MODEL}"
    assert os.path.exists(final_model_dir), f"final_model_dir does not exist: {final_model_dir}"
    assert os.path.exists(hyperparameter_file_path), f"hyperparameter_file_path does not exist: {hyperparameter_file_path}"
    assert os.path.exists(offline_csv_path), f"offline_csv_path does not exist: {offline_csv_path}"
    assert os.path.exists(feature_normalization_stats_file), f"feature_normalization_stats_file does not exist: {feature_normalization_stats_file}"
    
    if HYPERPARAMETERS is None:
        logger.info(f"{GREEN_COLOR}HYPERPARAMETERS is None{RESET_COLOR}")
        HYPERPARAMETERS = utils.load_hyperparameter_file(hyperparameter_file_path)
        HYPERPARAMETERS['TTFT_REWARD_WEIGHT'] = TTFT_REWARD_WEIGHT
        HYPERPARAMETERS['EXPLORATION_ENABLED'] = EXPLORATION_ENABLED
        HYPERPARAMETERS['ENABLE_ONLINE_LEARNING'] = ENABLE_ONLINE_LEARNING
        HYPERPARAMETERS['INCLUDE_GPU_FEATURES'] = INCLUDE_GPU_FEATURES
        
        # 🔧 CRITICAL: Apply EXCLUDED_REQUEST_FEATURES to match offline trained model architecture
        global request_features_train
        all_request_features = ['input_tokens', 'output_tokens', 'total_tokens']
        excluded_features = set(HYPERPARAMETERS.get('EXCLUDED_REQUEST_FEATURES', []))
        # Filter out 'none' placeholder
        if 'none' in excluded_features:
            excluded_features.remove('none')
        request_features_train = [f for f in all_request_features if f not in excluded_features]
        logger.info(f"🔧 Request features for inference (after excluding {excluded_features}): {request_features_train}")
        
        model_type = HYPERPARAMETERS.get('MODEL_TYPE', 'contextual_bandit')
        logger.info(f"Model type configured: {model_type}")
        if model_type == 'latency_predictor':
            latency_metric = HYPERPARAMETERS.get('LATENCY_METRIC', 'ttft')
            logger.info(f"Latency metric for prediction: {latency_metric}")
        else:
            logger.info(f"Using contextual bandit with exploration rate: {HYPERPARAMETERS.get('exploration_rate', 0)}")
        # Test permissions first
        logger.info("Testing Kubernetes API permissions...")
        if not test_kubernetes_permissions():
            logger.error("Insufficient Kubernetes permissions - using fallback GPU mapping")
            assert False
        # running_vllm_pods = utils.get_running_pods_by_label(POD_LABEL_SELECTOR)
        # sorted_running_pod_ips = utils.fetch_running_pod_ips(running_vllm_pods)
        # pod_ip_to_generalpodid = utils.create_pod_ip_to_generalpodid_mapping(sorted_running_pod_ips)
        # generalpodid_to_pod_ip = {}
        # for pod_ip, generalpodid in pod_ip_to_generalpodid.items():
        #     generalpodid_to_pod_ip[generalpodid] = pod_ip
        # generalpodid_to_gpu_model = utils.fetch_generalpodid_to_gpu_model(running_vllm_pods, pod_ip_to_generalpodid)
        # pod_ip_to_gpu_model, pod_ip_to_gpu_model_encoded = utils.create_pod_ip_to_gpu_model_mapping(generalpodid_to_gpu_model, pod_ip_to_generalpodid)
        
        # logger.info(f"POD_LABEL_SELECTOR: {POD_LABEL_SELECTOR}")
        # logger.info(f"len(sorted_running_pod_ips): {len(sorted_running_pod_ips)}, sorted_running_pod_ips: {sorted_running_pod_ips}")
        # logger.info(f"pod_ip_to_generalpodid: {pod_ip_to_generalpodid}")
        # logger.info(f"generalpodid_to_gpu_model: {generalpodid_to_gpu_model}")
        # logger.info(f"pod_ip_to_gpu_model: {pod_ip_to_gpu_model}")
        # logger.info(f"pod_ip_to_gpu_model_encoded: {pod_ip_to_gpu_model_encoded}")

        # HYPERPARAMETERS['pod_ip_to_generalpodid'] = pod_ip_to_generalpodid
        # HYPERPARAMETERS['generalpodid_to_pod_ip'] = generalpodid_to_pod_ip
        # logger.info(f"HYPERPARAMETERS['generalpodid_to_pod_ip']: {HYPERPARAMETERS['generalpodid_to_pod_ip']}")
        # HYPERPARAMETERS['sorted_running_pod_ips'] = sorted_running_pod_ips
        # HYPERPARAMETERS['pod_ip_to_gpu_model'] = pod_ip_to_gpu_model
        # HYPERPARAMETERS['pod_ip_to_gpu_model_encoded'] = pod_ip_to_gpu_model_encoded
        # HYPERPARAMETERS['generalpodid_to_gpu_model'] = generalpodid_to_gpu_model
        # # Additional mappings for GPU features expected by preprocess/encoding
        # HYPERPARAMETERS['pod_gpu_mapping'] = generalpodid_to_gpu_model
        # pod_gpu_id_mapping = {}
        # for generalpodid, gpu_model in generalpodid_to_gpu_model.items():
        #     if gpu_model in utils.GPU_MODEL_TO_ENCODE:
        #         pod_gpu_id_mapping[generalpodid] = utils.GPU_MODEL_TO_ENCODE[gpu_model]
        #     else:
        #         logger.error(f"Unknown GPU model for {generalpodid}: {gpu_model}")
        #         assert False
        # HYPERPARAMETERS['pod_gpu_id_mapping'] = pod_gpu_id_mapping
        
        # Load normalization statistics from CSV file
        # NOTE: Feature stats are ALWAYS based on GPU-L3c (reference baseline)
        # This ensures consistent feature space across all GPU types and experiments
        # Stats can be updated by online learning, but the base is always GPU-L3c
        if os.path.exists(feature_normalization_stats_file):
            logger.info(f"Loading normalization statistics (ALWAYS GPU-L3c baseline): {feature_normalization_stats_file}")
            try:
                stats_instance = data_normalizer.FeatureStats.load_from_csv(feature_normalization_stats_file)
                if stats_instance is not None:
                    logger.info(f"✅ Successfully loaded GPU-L3c baseline stats for {len(stats_instance.feature_stats)} features")
                    logger.info(f"   These stats will be used regardless of TARGET_GPU_MODEL={TARGET_GPU_MODEL}")
                    if LOAD_PRETRAINED_MODEL == 0:
                        logger.info(f"   LOAD_PRETRAINED_MODEL=0: Will train model from scratch but use GPU-L3c normalization baseline")
                else:
                    logger.error("Failed to load normalization statistics")
                    assert False
            except Exception as e:
                logger.error(f"Failed to load normalization statistics: {e}")
                assert False
        else:
            logger.error(f"Normalization statistics file not found: {feature_normalization_stats_file}")
            logger.error(f"Expected GPU-L3c baseline stats at: /app/final_model/GPU-L3c/feature_normalization_statistics.csv")
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

    HYPERPARAMETERS['pod_ip_to_generalpodid'] = pod_ip_to_generalpodid
    HYPERPARAMETERS['generalpodid_to_pod_ip'] = generalpodid_to_pod_ip
    logger.info(f"HYPERPARAMETERS['generalpodid_to_pod_ip']: {HYPERPARAMETERS['generalpodid_to_pod_ip']}")
    HYPERPARAMETERS['sorted_running_pod_ips'] = sorted_running_pod_ips
    HYPERPARAMETERS['pod_ip_to_gpu_model'] = pod_ip_to_gpu_model
    HYPERPARAMETERS['pod_ip_to_gpu_model_encoded'] = pod_ip_to_gpu_model_encoded
    HYPERPARAMETERS['generalpodid_to_gpu_model'] = generalpodid_to_gpu_model
    # Additional mappings for GPU features expected by preprocess/encoding
    HYPERPARAMETERS['pod_gpu_mapping'] = generalpodid_to_gpu_model
    pod_gpu_id_mapping = {}
    for generalpodid, gpu_model in generalpodid_to_gpu_model.items():
        if gpu_model in utils.GPU_MODEL_TO_ENCODE:
            pod_gpu_id_mapping[generalpodid] = utils.GPU_MODEL_TO_ENCODE[gpu_model]
        else:
            logger.error(f"Unknown GPU model for {generalpodid}: {gpu_model}")
            assert False
    HYPERPARAMETERS['pod_gpu_id_mapping'] = pod_gpu_id_mapping
        
    # Print feature statistics if available
    if stats_instance is not None:
        logger.info("Per-feature statistics loaded:")
        for feature_name, stats in stats_instance.feature_stats.items():
            logger.info(f"stats_instance, {feature_name}: count={stats.count}, mean={stats.mean}, std={stats.std}")
    else:
        logger.warning("No normalization statistics available - inference will fail")
        assert False
    
    # Add checkpointing configuration to hyperparameters
    HYPERPARAMETERS['CHECKPOINT_INTERVAL_STEPS'] = 100
    HYPERPARAMETERS['CHECKPOINT_DIR'] = os.path.join(final_model_dir, 'checkpoints')
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(HYPERPARAMETERS['CHECKPOINT_DIR'], exist_ok=True)
    logger.info(f"Checkpoint directory: {HYPERPARAMETERS['CHECKPOINT_DIR']}")
    logger.info(f"Checkpointing every {HYPERPARAMETERS['CHECKPOINT_INTERVAL_STEPS']} steps")

    # Load offline training data for online learning
    global TRAINING_DF, OFFLINE_DATA_SIZE
    if ENABLE_ONLINE_LEARNING:
        # Skip loading offline data if LOAD_PRETRAINED_MODEL=0 (pure online learning from scratch)
        if LOAD_PRETRAINED_MODEL == 0:
            logger.warning(f"⚠️  LOAD_PRETRAINED_MODEL=0: Skipping offline training data")
            logger.warning(f"⚠️  Will train from scratch using ONLY online data")
            logger.warning(f"⚠️  Inference will use RANDOM WEIGHTS until first training completes ({MIN_NUM_TRAINING_DATA} samples)")
            logger.info(f"ℹ️  Normalization stats are still loaded to maintain consistent feature space")
            TRAINING_DF = pd.DataFrame()
            OFFLINE_DATA_SIZE = 0
        elif os.path.exists(offline_csv_path):
            try:
                with TRAINING_DF_LOCK:
                    TRAINING_DF = pd.read_csv(offline_csv_path)
                    # shuffle the training data
                    TRAINING_DF = TRAINING_DF.sample(frac=1).reset_index(drop=True)
                    OFFLINE_DATA_SIZE = len(TRAINING_DF)
                    logger.info(f"✅ Loaded offline training data: {len(TRAINING_DF)} samples from {offline_csv_path}")
                    logger.info(f"   Columns: {list(TRAINING_DF.columns[:10])}...")  # Show first 10 columns
            except Exception as e:
                logger.error(f"Failed to load offline training data: {e}")
                TRAINING_DF = pd.DataFrame()
                OFFLINE_DATA_SIZE = 0
                logger.warning("Starting with empty training dataframe")
        else:
            logger.warning(f"Offline training data not found at {offline_csv_path}")
            logger.warning("Online learning will start from scratch with only new data")
            TRAINING_DF = pd.DataFrame()
            OFFLINE_DATA_SIZE = 0
    else:
        logger.info("Online learning disabled, skipping offline data load")
    
    # Initialize scalable RL agent if configured
    TOTAL_NUM_NEW_DATA = len(TRAINING_DF)
    logger.info(f"{BLUE_COLOR}model_type: {HYPERPARAMETERS['MODEL_TYPE']}{RESET_COLOR}")

    # if HYPERPARAMETERS['MODEL_TYPE'] == 'scalable_rl_agent':
    if HYPERPARAMETERS['MODEL_TYPE'] == 'scalable_rl_agent':
        import scalable_rl_routing_agent
        from scalable_rl_routing_agent import BROKER
        
        logger.info("Initializing scalable_rl_agent, scalable RL agent...")
        global SCALABLE_RL_AGENT, BROKER_LOCK
        with BROKER_LOCK.write():
            # Get number of pods from current running pods
            num_pods = len(sorted_running_pod_ips)
            logger.info(f"Creating scalable RL agent with {num_pods} pods")
            
            # # Create agent with hyperparameters
            # SCALABLE_RL_AGENT = create_scalable_rl_agent(
            #     per_pod_dim=HYPERPARAMETERS.get('per_pod_dim', 8),
            #     request_dim=HYPERPARAMETERS.get('request_dim', 3),
            #     max_pods=HYPERPARAMETERS.get('max_pods', 100),
            #     learning_rate=HYPERPARAMETERS.get('learning_rate', 3e-4),
            #     reward_decay_factor=HYPERPARAMETERS.get('reward_decay_factor', 1.0),
            #     gae_lambda=HYPERPARAMETERS.get('gae_lambda', 0.95),
            #     n_steps=HYPERPARAMETERS.get('n_steps', 256),
            #     horizon=HYPERPARAMETERS.get('horizon', 1024),
            #     batch_size=HYPERPARAMETERS.get('batch_size', 64),
            #     last_layer_dim_vf=HYPERPARAMETERS.get('last_layer_dim_vf', 1),
            #     rl=HYPERPARAMETERS.get('rl_algorithm', 'PPO'),
            #     static_num_pods=True,
            # )
            
            SCALABLE_RL_AGENT = scalable_rl_routing_agent.ScalableRLRoutingAgent(
                per_pod_dim=8, 
                # per_pod_dim=11, 
                request_dim=3, 
                max_pods=100, 
                num_requests_per_episode=HYPERPARAMETERS['num_requests_per_episode'], 
                num_episodes_per_iteration=HYPERPARAMETERS['num_episodes_per_iteration'],
                num_iterations=HYPERPARAMETERS['num_iterations'],   
                rl=HYPERPARAMETERS['rl_algorithm'], 
                static_num_pods=True, 
                learning_rate=HYPERPARAMETERS['rl_learning_rate'], 
                hidden_dim=HYPERPARAMETERS['hidden_dim'], 
                gamma=HYPERPARAMETERS['gamma'], 
                gae_lambda=HYPERPARAMETERS['gae_lambda'], 
                tb_log_dir=os.path.join(HYPERPARAMETERS['CHECKPOINT_DIR'], 'tb_logs'), 
                batch_size=HYPERPARAMETERS['batch_size'], 
                n_epochs=HYPERPARAMETERS['training_epochs'], 
                clip_range=HYPERPARAMETERS['clip_range'], 
                entropy_coeff=HYPERPARAMETERS['entropy_coeff'], 
                vf_coef=HYPERPARAMETERS['vf_coef'], 
                max_grad_norm=HYPERPARAMETERS['max_grad_norm'], 
                last_layer_dim_pi=HYPERPARAMETERS['last_layer_dim_pi'], 
                last_layer_dim_vf=HYPERPARAMETERS['last_layer_dim_vf'], 
                use_prioritized_replay=False, 
                buffer_size=HYPERPARAMETERS['buffer_size'], 
                priority_alpha=HYPERPARAMETERS['priority_alpha'], 
                priority_beta=HYPERPARAMETERS['priority_beta'],
                lr_scheduler_type=HYPERPARAMETERS['lr_scheduler_type'],
                load_tb_best='/app/final_model/init_model/best_model.zip',
                )
            
            logger.info(f"scalable_rl_routing_agent, Scalable RL agent created successfully")
            
            # Load checkpoint if available
            checkpoint_path = HYPERPARAMETERS.get('RL_CHECKPOINT_PATH')
            if checkpoint_path and os.path.exists(checkpoint_path):
                try:
                    SCALABLE_RL_AGENT.load(checkpoint_path)
                    logger.info(f"scalable_rl_routing_agent, Loaded scalable RL checkpoint from {checkpoint_path}")
                except Exception as e:
                    logger.warning(f"scalable_rl_routing_agent, Failed to load checkpoint: {e}")
        
        # Start training thread
        logger.info(f"{GREEN_COLOR}Starte scalable rl training worker in init()...{RESET_COLOR}")
        start_scalable_rl_training_worker()
        logger.info(f"{GREEN_COLOR}Scalable RL agent initialized and training thread started{RESET_COLOR}")
        logger.info("scalable_rl_routing_agent, Scalable RL agent initialized and training thread started")
    INIT_DONE = True

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
    global SCALABLE_RL_AGENT, HYPERPARAMETERS
    
    try:
        if SCALABLE_RL_AGENT is None:
            return  # Agent not initialized yet
        
        with SCALABLE_RL_AGENT_LOCK.read():
            # Check if it's time to checkpoint

            checkpoint_interval = HYPERPARAMETERS.get('CHECKPOINT_INTERVAL_STEPS', 1000)
            total_steps = SCALABLE_RL_AGENT.total_steps
            
            # Checkpoint at regular intervals
            if total_steps > 0 and total_steps % checkpoint_interval < 64:  # 64 = typical batch size
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                checkpoint_name = f"scalable_rl_step_{total_steps}_{timestamp}"
                checkpoint_path = os.path.join(HYPERPARAMETERS['CHECKPOINT_DIR'], checkpoint_name)
                
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
                        # cleanup_old_checkpoints(HYPERPARAMETERS['CHECKPOINT_DIR'], keep_latest=5)
                        
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

    logger.info(f"{RED_COLOR}init() finished in main()...{RESET_COLOR}")

    scheduler = BackgroundScheduler()
    # If online learning is disabled, just use the pretrained model
    if ENABLE_ONLINE_LEARNING:
        scheduler.add_job(func=online_train_routine, trigger="interval", seconds=30)
    else:
        logger.info("Online learning disabled. online_train_routine will not be invoked at all - using pretrained model only in inference")
    
    # # Add periodic checkpointing for scalable RL agent (every 2 minutes)
    # scheduler.add_job(func=periodic_checkpoint_scalable_rl, trigger="interval", seconds=120)
    # logger.info("Periodic checkpointing scheduled (every 2 minutes)")
    
    scheduler.start()
    atexit.register(lambda: scheduler.shutdown())
    
    # Start RL update worker thread
    if HYPERPARAMETERS['MODEL_TYPE'] == 'scalable_rl_agent':
        logger.info(f"{GREEN_COLOR}Starting RL update worker in main()...{RESET_COLOR}")
        start_rl_update_worker()
        
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