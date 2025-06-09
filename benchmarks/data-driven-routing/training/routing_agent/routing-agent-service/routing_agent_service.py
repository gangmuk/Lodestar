# routing_agent_service.py

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
import feature_normalization

app = Flask(__name__)

FLUSH_BATCH_ID = 0
ENCODED_DATA_DIR = "encoded_data"
final_model_path = "final_model"
feature_normalization_stats_file = f"{final_model_path}/feature_normalization_statistics.pkl"  # Add this near the top with your other constants;
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
    global FLUSH_BATCH_ID, ENCODED_DATA_DIR, TRAINING_DATA_UPDATED, TOTAL_NUM_DATA, NUM_NEW_DATA, feature_normalization_stats_file, stats_instance
    flush_start_time = time.time()
    log_data = request.json
    try:
        logger.info(f"Received log data with {len(log_data) if log_data else 0} entries")
        if not os.path.exists("raw_training_data"):
            os.mkdir("raw_training_data")
        raw_data_path = f"raw_training_data/batch_{FLUSH_BATCH_ID}.csv"
        FLUSH_BATCH_ID += 1
        
        # Write raw data to file
        ts_write_raw_data = time.time()
        write_to_file(log_data, raw_data_path)
        logger.info(f"wrote {len(log_data)} entries to {raw_data_path}, took {time.time() - ts_write_raw_data} seconds")

        # Preprocess raw data
        ts_preprocess = time.time()
        processed_df, _, all_pods, _ = preprocess.main(raw_data_path, "", TTFT_SLO, AVG_TPOT_SLO)
        logger.info(f"Successfully parsed data, took {time.time() - ts_preprocess} seconds")
        try:
            if stats_instance is None:
                stats_instance = feature_normalization.get_stats_instance(feature_normalization_stats_file)
        except Exception as e:
            logger.error(f"Could not load feature normalization stats: {e}")
            assert False
            
        # if ENABLE_ONLINE_LEARNING:
        processed_df, stats_instance, _ = feature_normalization.normalize_features_for_training(processed_df, stats_instance)
        stats_instance.write_stats_to_file(feature_normalization_stats_file)

        # ===== SHARED REWARD ENGINEERING =====
        processed_df = feature_normalization.try_reward_amplification(processed_df)
        
        # Encode preprocessed data
        ts_encode = time.time()
        encoded_data_subdir = f"{ENCODED_DATA_DIR}/batch_{FLUSH_BATCH_ID}"
        encoding.encode_for_train(all_pods, processed_df, encoded_data_subdir, request_features_train)
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
    global NUM_TRAINS, MODEL_UPDATED, first_request_starting_time, stats_instance
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
        logger.debug(f"Received inference request:\n{log_message}")

        # Extract request ID for logging purposes
        parts = log_message.split("requestID@")
        if len(parts) > 1:
            request_id_parts = parts[1].split("@")
            if request_id_parts:
                request_id = request_id_parts[0]
        handle_infer_total_prep_overhead = time.time() - prep_start_time

        # Use the existing preprocessing function to parse the log
        preprocess_start_time = time.time()
        processed_df, _, all_pods, preprocess_dataset_overhead_summary = preprocess.main(None, log_message, TTFT_SLO, AVG_TPOT_SLO)
        logger.debug(f"Successfully parsed data for request_{request_id}")
        handle_infer_total_total_preprocess_overhead = time.time() - preprocess_start_time

        # Get running statistics and apply normalization (SAME AS TRAINING)
        get_stat_start_time = time.time()
        if stats_instance is None:
            stats_instance = feature_normalization.get_stats_instance(feature_normalization_stats_file)
        if stats_instance is None:
            logger.error(f"No running statistics available, stats_instance: {stats_instance}")
            assert False
        if stats_instance.count == 0:
            logger.warning(f"Stats instance count is 0, no data available for normalization")
        processed_df = feature_normalization.normalize_features_for_inference(processed_df, stats_instance)
        handle_infer_total_get_stat_overhead = time.time() - get_stat_start_time

        # Encode data (normalization already done)
        encode_start_time = time.time()
        tensor_data, encode_for_inference_overhead_summary = encoding.      encode_for_inference(all_pods, processed_df, request_features_train)
        logger.debug(f"Successfully encoded data in memory for inference")
        handle_infer_total_total_encoding_overhead = time.time() - encode_start_time

        infer_from_tensor_start_time = time.time()
        if MODEL == "simpler_contextual_bandit":
            result, infer_from_tensor_overhead_summary = simpler_contextual_bandit.infer_from_tensor(tensor_data=tensor_data, model_updated=MODEL_UPDATED)
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
    global NUM_TRAINS, MODEL_UPDATED, TRAINING_DATA_UPDATED, TOTAL_NUM_DATA, final_model_path, NUM_NEW_DATA

    if  TRAINING_DATA_UPDATED and NUM_NEW_DATA > MIN_NUM_TRAINING_DATA:
        training_start_time = time.time()
        logger.info(f"online_train_routine, Starting {NUM_TRAINS}th online training iteration")
        try:
            if MODEL == "simpler_contextual_bandit":
                simpler_contextual_bandit.train(ENCODED_DATA_DIR, final_model_path)
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


if __name__ == "__main__":
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