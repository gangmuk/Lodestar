#!/usr/bin/env python3

## utils.py

from logger import logger, INCLUDE_GPU_IN_FEATURE
from kubernetes import client, config
import re
import random
import numpy as np
import torch
import os
import subprocess
import threading
import json
import time
import datetime
import pytz
from typing import List, Dict, Any
from kubernetes.config import ConfigException
import hashlib
import traceback
import psutil
import socket
import ast

def load_rl_hyperparameters(file_path, RL_MODEL_HYPERPARAMETERS):
    if not os.path.exists(file_path):
        logger.error(f"Hyperparameter file {file_path} does not exist")
        assert False
    with open(file_path, 'r') as f:
        data = json.load(f)
    for key, value in data.items():
        if key == 'normalization':
            # Handle nested normalization parameters
            RL_MODEL_HYPERPARAMETERS[key] = {}
            for sub_key, sub_value in value.items():
                if sub_key == 'FEATURES_NORMALIZED':
                    # Convert string representation of set to actual set
                    if sub_value and sub_value != "set()":
                        RL_MODEL_HYPERPARAMETERS[key][sub_key] = ast.literal_eval(sub_value)
                    else:
                        RL_MODEL_HYPERPARAMETERS[key][sub_key] = set()
                elif sub_key == 'FEATURES_AMPLIFIED':
                    # Convert string representation of set to actual set
                    if sub_value and sub_value != "set()":
                        RL_MODEL_HYPERPARAMETERS[key][sub_key] = ast.literal_eval(sub_value)
                    else:
                        RL_MODEL_HYPERPARAMETERS[key][sub_key] = set()
                else:
                    RL_MODEL_HYPERPARAMETERS[key][sub_key] = sub_value
        else:
            RL_MODEL_HYPERPARAMETERS[key] = value

    for key, value in RL_MODEL_HYPERPARAMETERS.items():
        if key == 'normalization':
            for sub_key, sub_value in value.items():
                logger.info(f"load_rl_hyperparameters, {key}.{sub_key}: {sub_value}")
        else:
            logger.info(f"load_rl_hyperparameters, {key}: {value}")


def get_sorted_all_pod_ids(source_type, data=None):
    """
    Centralized function to extract sorted_all_pod_ids from different sources.
    
    Args:
        source_type (str): Type of source - 'batch_dataframe', 'single_row', 'processed_csv_columns'
        data: The data source (DataFrame, dict, or list of column names)
    
    Returns:
        list: Sorted list of pod IDs
    """
    if source_type == 'batch_dataframe':
        # Extract from batch dataframe for training (parsed_df)
        all_pods_set = set()
        for col in ['allPodsKvCacheHitRatios', 'numInflightRequestsAllPods']:
            if col in data.columns:
                for row_data in data[col]:
                    if row_data:
                        if type(row_data) is not dict:
                            logger.error(f"Expected dict but got {type(row_data)}: {row_data}")
                            assert False
                        all_pods_set.update(row_data.keys())
        sorted_all_pod_ids = sorted(list(all_pods_set))
        logger.info(f"Extracted {len(sorted_all_pod_ids)} pod IDs from batch dataframe: {sorted_all_pod_ids}")
        return sorted_all_pod_ids
        
    elif source_type == 'single_row':
        # Extract from single row data (dict)
        kv_cache = data.get('allPodsKvCacheHitRatios', {})
        inflight = data.get('numInflightRequestsAllPods', {})
        gpu_cache = data.get('vllmGPUKVCacheUsage', {})
        cpu_cache = data.get('vllmCPUKVCacheUsage', {})
        running = data.get('vllmNumRequestsRunning', {})
        waiting = data.get('vllmNumRequestsWaiting', {})
        
        sorted_all_pod_ids = sorted(list(set(
            list(kv_cache.keys()) +
            list(inflight.keys()) +
            list(gpu_cache.keys()) +
            list(cpu_cache.keys()) +
            list(running.keys()) +
            list(waiting.keys())
        )))
        
        if not sorted_all_pod_ids:
            logger.error("Error: No pod IDs found in the single row data.")
            logger.error(f"Row data keys: {data.keys()}")
            assert False
        logger.info(f"Extracted {len(sorted_all_pod_ids)} pod IDs from single row: {sorted_all_pod_ids}")
        return sorted_all_pod_ids
        
    elif source_type == 'processed_csv_columns':
        # Extract from already processed CSV columns
        pod_ids_set = set()
        for col in data:  # data is list of column names
            if '-kv_hit_ratio' in col:
                pod_id = col.replace('-kv_hit_ratio', '')
                pod_ids_set.add(pod_id)
        sorted_all_pod_ids = sorted(list(pod_ids_set))
        logger.info(f"Extracted {len(sorted_all_pod_ids)} pod IDs from processed CSV columns: {sorted_all_pod_ids}")
        return sorted_all_pod_ids
        
    else:
        logger.error(f"Unknown source_type: {source_type}")
        assert False


def set_all_seeds(seed=42):
    """Set seeds for all sources of randomness to ensure reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch, 'use_deterministic_algorithms'):
        torch.use_deterministic_algorithms(True, warn_only=True)
    print(f"All seeds set to {seed} for reproducible results")
    
def replace_pod_ip_with_generalpodid(data_input):
    """
    Replace pod IPs with general pod IDs in either a file or a log message string.
    
    Args:
        data_input: Either a file path (str) or a log message (str)
    
    Returns:
        str: Path to the processed file or the processed log message
    """
    # Determine if input is a file path or log message
    is_file = os.path.isfile(data_input) if isinstance(data_input, str) and len(data_input) < 260 else False
    
    if is_file:
        # Handle file input (existing logic)
        all_pod_ips_from_training_data = get_all_pod_ips_from_data_file(data_input)
        if not all_pod_ips_from_training_data:
            logger.error(f"No pod IPs found in data file {data_input}")
            assert False
        logger.info(f"Deterministic pod IP order: {all_pod_ips_from_training_data}")
        pod_ip_to_generalpodid = create_pod_ip_to_generalpodid_mapping(all_pod_ips_from_training_data)
        logger.info(f"Deterministic mapping: {pod_ip_to_generalpodid}")
        
        with open(data_input, 'r') as f:
            content = f.read()
        
        for pod_ip, generalpodid in pod_ip_to_generalpodid.items():
            content = content.replace(pod_ip, generalpodid)
        
        replaced_data_file = data_input.replace('.csv', '_replaced.csv')
        with open(replaced_data_file, 'w') as f:
            f.write(content)
        
        logger.info(f"File written {replaced_data_file} with replaced generalpodids")
        return replaced_data_file
    else:
        # Handle log message string input
        pod_ips = extract_pod_ips_from_content(data_input)
        if not pod_ips:
            logger.error(f"No pod IPs found in log message")
            assert False
        
        all_pod_ips_from_log = sorted(list(pod_ips))
        logger.info(f"Deterministic pod IP order from log: {all_pod_ips_from_log}")
        pod_ip_to_generalpodid = create_pod_ip_to_generalpodid_mapping(all_pod_ips_from_log)
        logger.info(f"Deterministic mapping: {pod_ip_to_generalpodid}")
        
        content = data_input
        for pod_ip, generalpodid in pod_ip_to_generalpodid.items():
            content = content.replace(pod_ip, generalpodid)
        
        return content

def get_all_pod_ips_from_data_file(data_file):
    """Extract pod IPs from a data file."""
    pod_ips = set()
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            content = f.read()
        pod_ips = extract_pod_ips_from_content(content)
    except FileNotFoundError:
        logger.error(f"Data file {data_file} not found")
        return []
    except Exception as e:
        logger.error(f"Error reading data file {data_file}: {e}")
        return []
    
    return sorted(list(pod_ips))



def extract_pod_ips_from_content(content):
    """
    Extract pod IPs from content using the existing patterns plus the new data format.
    
    Args:
        content (str): The content to search for pod IPs
    
    Returns:
        set: Set of unique pod IPs found
    """
    pod_ips = set()
    
    # EXISTING Pattern 1: Extract from selectedpod@ field
    # Pattern: selectedpod@IP@
    selectedpod_pattern = r'selectedpod@(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})@'
    selectedpod_matches = re.findall(selectedpod_pattern, content)
    pod_ips.update(selectedpod_matches)
    
    # EXISTING Pattern 2: Extract from JSON-like structures
    # Pattern: "IP":value in JSON-like structures
    json_ip_pattern = r'"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})":'
    json_ip_matches = re.findall(json_ip_pattern, content)
    pod_ips.update(json_ip_matches)
    
    # EXISTING Pattern 3: General IP pattern as fallback (more comprehensive)
    # This catches any IP that might be in other formats
    general_ip_pattern = r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b'
    general_ip_matches = re.findall(general_ip_pattern, content)
    
    # Filter to only include IPs that look like pod IPs (10.x.x.x range)
    pod_like_ips = [ip for ip in general_ip_matches if ip.startswith('10.')]
    pod_ips.update(pod_like_ips)
    
    # Additional validation: ensure IPs are valid
    validated_ips = set()
    for ip in pod_ips:
        parts = ip.split('.')
        if len(parts) == 4 and all(0 <= int(part) <= 255 for part in parts if part.isdigit()):
            validated_ips.add(ip)
    
    logger.info(f"Extracted {len(validated_ips)} unique pod IPs: {sorted(validated_ips)}")
    return validated_ips

def create_pod_ip_to_generalpodid_mapping(unique_pod_ips):
    """Create mapping from pod IPs to general pod IDs."""
    pod_ip_to_generalpodid = {}
    sorted_unique_pod_ips = sorted(unique_pod_ips)  # Ensure consistent ordering
    for idx, pod_ip in enumerate(sorted_unique_pod_ips):
        if idx < 10:
            pod_ip_to_generalpodid[pod_ip] = f"pod_000{idx}"
        elif idx < 100:
            pod_ip_to_generalpodid[pod_ip] = f"pod_00{idx}"
        elif idx < 1000:
            pod_ip_to_generalpodid[pod_ip] = f"pod_0{idx}"
        else:
            logger.error(f"Too many pods ({len(sorted_unique_pod_ips)}) to map to generalpodid, only supporting up to 1000 pods")
            assert False
    logger.info(f"pod_ip_to_generalpodid: {pod_ip_to_generalpodid}")
    return pod_ip_to_generalpodid


def create_pod_ip_to_gpu_model_mapping(generalpodid_to_gpu_model, pod_ip_to_generalpodid):
    pod_ip_to_gpu_model = {}
    for pod_ip, generalpodid in pod_ip_to_generalpodid.items():
        if generalpodid in generalpodid_to_gpu_model:
            pod_ip_to_gpu_model[pod_ip] = generalpodid_to_gpu_model[generalpodid]
        else:
            logger.error(f"GeneralPodID {generalpodid} not found in generalpodid_to_gpu_model mapping for pod IP {pod_ip}")
            assert False
    GPU_MODEL_TO_ENCODE = {
        'NVIDIA-L20': 0,
        'NVIDIA-L40': 1,
        'NVIDIA-A10': 2,
        'NVIDIA-A100': 3,
        'NVIDIA-H100': 4,
    }
    pod_ip_to_gpu_model_encoded = {}
    for pod_ip, gpu_model in pod_ip_to_gpu_model.items():
        if gpu_model in GPU_MODEL_TO_ENCODE:
            pod_ip_to_gpu_model_encoded[pod_ip] = GPU_MODEL_TO_ENCODE[gpu_model]
        else:
            logger.error(f"Unknown GPU model: {gpu_model}, for pod IP {pod_ip}")
            logger.error(f"pod_ip_to_gpu_model: {pod_ip_to_gpu_model}")
            logger.error(f"generalpodid_to_gpu_model: {generalpodid_to_gpu_model}")
            assert False
    return pod_ip_to_gpu_model, pod_ip_to_gpu_model_encoded

def get_running_pods_by_label(label_selector):
    try:
        config.load_incluster_config()
    except ConfigException:
        # try:
        #     config.load_kube_config()
        # except ConfigException:
        #     raise ConfigException("Could not load Kubernetes config from cluster or kubeconfig file")
        raise ConfigException("Could not load Kubernetes config from cluster or kubeconfig file")
        assert False
    
    v1 = client.CoreV1Api()
    return v1.list_pod_for_all_namespaces(label_selector=label_selector)
    
def fetch_running_pod_ips(running_pods: client.V1PodList):
    pod_ips = []
    for pod in running_pods.items:
        if pod.status.phase == "Running" and pod.status.pod_ip:
            pod_ips.append(pod.status.pod_ip)
        else:
            logger.warning(f"Pod {pod.metadata.name} is not running or has no IP assigned")
    pod_ips = sorted(pod_ips)  # Sort to ensure consistent order
    return pod_ips

def fetch_generalpodid_to_gpu_model(running_pods: client.V1PodList, pod_ip_to_generalpodid):
    try:
        config.load_incluster_config()
    except ConfigException:
        # try:
        #     config.load_kube_config()
        # except ConfigException:
        #     raise ConfigException("Could not load Kubernetes config from cluster or kubeconfig file")
        raise ConfigException("Could not load Kubernetes config from cluster or kubeconfig file")
        assert False
    v1 = client.CoreV1Api()
    generalpodid_to_gpu_model = {}
    for pod in running_pods.items:
        if pod.status.phase == "Running" and pod.status.pod_ip:
            pod_ip = pod.status.pod_ip
            if pod_ip not in pod_ip_to_generalpodid:
                logger.error(f"Pod IP {pod_ip} not found in pod_ip_to_generalpodid mapping")
                assert False
            generalpodid = pod_ip_to_generalpodid[pod_ip]
            node_name = pod.spec.node_name
            if node_name:
                try:
                    node = v1.read_node(name=node_name)
                    node_labels = node.metadata.labels or {}
                    if 'machine.cluster.vke.volcengine.com/gpu-name' not in node_labels:
                        logger.error(f"machine.cluster.vke.volcengine.com/gpu-name must exist in node.metadata.labels. It does not.")
                        assert False
                    gpu_model = node_labels['machine.cluster.vke.volcengine.com/gpu-name']
                    generalpodid_to_gpu_model[generalpodid] = gpu_model
                except Exception as e:
                    logger.warning(f"Failed to get node info for {node_name}: {e}")
                    generalpodid_to_gpu_model[generalpodid] = 'unknown'
            else:
                logger.warning(f"Pod {pod.metadata.name} has no node assignment")
                generalpodid_to_gpu_model[generalpodid] = 'unknown'
    return generalpodid_to_gpu_model



#######################################################################################
#######################################################################################

def load_workload(input_path: str) -> List[Any]:
    load_struct = None
    if input_path.endswith(".jsonl"):
        with open(input_path, "r") as file:
            load_struct = [json.loads(line) for line in file]
    else:
        with open(input_path, "r") as file:
            load_struct = json.load(file)
    return load_struct

# Function to wrap the prompt into OpenAI's chat completion message format.
def prepare_prompt(prompt: str, 
                   lock: threading.Lock,
                   session_id: str = None, 
                   history: Dict = None) -> List[Dict]:
    """
    Wrap the prompt into OpenAI's chat completion message format.

    :param prompt: The user prompt to be converted.
    :return: A list containing chat completion messages.
    """
    if session_id is not None:
        with lock:
            past_history = history.get(session_id, [])
            user_message = {"role": "user", "content": f"{prompt}"}
            past_history.append(user_message) 
            history[session_id] = past_history
            return past_history
    else:    
        user_message = {"role": "user", "content": prompt}
        return [user_message]
    
def update_response(response: str, 
                    lock: threading.Lock,
                    session_id: str = None, 
                    history: Dict = None):
    """
    Wrap the prompt into OpenAI's chat completion message format.

    :param prompt: The user prompt to be converted.
    :return: A list containing chat completion messages.
    """
    if session_id is not None:
        with lock:
            past_history = history.get(session_id, [])
            assistant_message = {"role": "assistant", "content": f"{response}"}
            past_history.append(assistant_message) 


######################################################################################


def run_command(command, required=True, print_error=True, nonblock=False):
    """Run shell command and return its output or process handle.

    Args:
        command (str): The shell command to execute.
        required (bool): If True, the function will assert on failure.
        print_error (bool): If True, errors will be printed.
        nonblock (bool): If True, run the command non-blocking.

    Returns:
        tuple: 
            - If nonblock is False: (True, output) on success or (False, error) on failure.
            - If nonblock is True: (True, process) on success or (False, error) on failure.
    """
    try:
        if nonblock:
            # Start the process without waiting for it to complete
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            return True, process
        else:
            # Run the command and wait for it to complete
            output = subprocess.check_output(
                command,
                shell=True,
                stderr=subprocess.STDOUT,
                text=True
            )
            return True, output.strip()
    except subprocess.CalledProcessError as e:
        if print_error:
            print(f"ERROR command: {command}")
            print(f"ERROR output: {e.output.strip()}")
        if required:
            print("Exiting due to required command failure...")
            raise  # Instead of assert False, it's better to raise an exception
        else:
            return False, e.output.strip()


def check_deployment_ready_kubernetes(deployment_name, namespace):
    """
    Checks if all pods of a deployment and all their containers are in a ready state using the Kubernetes Python client.

    Args:
        deployment_name (str): The name of the deployment.
        namespace (str): The namespace where the deployment is located.

    Returns:
        bool: True if all pods and their containers are ready, False otherwise (will keep checking).
    """
    try:
        # Load Kubernetes configuration (assuming you have a valid kubeconfig file)
        try:
            config.load_incluster_config()
        except ConfigException:
            # try:
            #     config.load_kube_config()
            # except ConfigException:
            #     raise ConfigException("Could not load Kubernetes config from cluster or kubeconfig file")
            raise ConfigException("Could not load Kubernetes config from cluster or kubeconfig file")
            assert False
        apps_v1 = client.AppsV1Api()
        core_v1 = client.CoreV1Api()
        
        # Add a max retry counter to prevent infinite loops
        max_retries = 60  # 1 minute with 1-second intervals
        retry_count = 0

        while retry_count < max_retries:
            try:
                deployment = apps_v1.read_namespaced_deployment(name=deployment_name, namespace=namespace)
                selector = deployment.spec.selector.match_labels

                if not selector:
                    print(f"No selector found for deployment '{deployment_name}'. Cannot find associated pods. Retrying in 1 second...")
                    time.sleep(1)
                    retry_count += 1
                    continue

                label_selector = ",".join([f"{k}={v}" for k, v in selector.items()])
                pod_list = core_v1.list_namespaced_pod(namespace=namespace, label_selector=label_selector)
                pods = pod_list.items

                if not pods:
                    print(f"No pods found for deployment '{deployment_name}' in namespace '{namespace}' with selector '{label_selector}'. Retrying in 1 second...")
                    time.sleep(1)
                    retry_count += 1
                    continue

                all_ready = True
                for pod in pods:
                    pod_name = pod.metadata.name
                    conditions = pod.status.conditions or []
                    ready_condition = next(
                        (c for c in conditions if c.type == "Ready"), None
                    )

                    if not ready_condition or ready_condition.status != "True":
                        print(f"Pod '{pod_name}' is not ready. Retrying in 1 second...")
                        all_ready = False
                        break

                    container_statuses = pod.status.container_statuses or []
                    for container_status in container_statuses:
                        if not container_status.ready:
                            container_name = container_status.name
                            print(f"Container '{container_name}' in pod '{pod_name}' is not ready. Retrying in 1 second...")
                            all_ready = False
                            break
                    if not all_ready:
                        break

                if all_ready:
                    print(f"Deployment '{deployment_name}' is ready!")
                    return True

                print(f"Deployment '{deployment_name}' is not ready yet.")
                time.sleep(1)
                retry_count += 1
                
            except client.ApiException as e:
                if e.status == 404:
                    print(f"Deployment '{deployment_name}' not found in namespace '{namespace}'. Please check the name and namespace.")
                    assert False
                else:
                    print(f"Kubernetes API exception: {e}")
                    print("Retrying in 1 second...")
                    time.sleep(1)
                    retry_count += 1
                    
        print(f"Max retries ({max_retries}) reached. Deployment '{deployment_name}' is not ready.")
        return False

    except config.ConfigException as e:
        print(f"Kubernetes configuration error: {e}")
        print("Please ensure your kubeconfig file is properly configured.")
        return False  # Exit if configuration is invalid
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False



def restart_deploy(deployment_name, namespace):
    run_command(f"kubectl rollout restart deploy {deployment_name} -n {namespace}")


def save_k8s_logs(namespace, deployment_name, output_dir, label=None, keyword=None):
    try:
        logger.info(f"Collecting logs from {deployment_name} in namespace {namespace}")
        
        app_label = label
        
        # If no label was provided, try to extract it from the deployment
        if not app_label:
            try:
                # Get the actual selector used by the deployment
                cmd = ["kubectl", "get", "deployment", deployment_name, "-n", namespace, 
                       "-o", "jsonpath={.spec.selector.matchLabels.app}"]
                logger.info(f"Executing command: {' '.join(cmd)}")
                app_label = subprocess.check_output(cmd).decode('utf-8').strip()
                logger.info(f"Found selector app label from deployment: {app_label}")
                
                # If still no label found, default to deployment name
                if not app_label:
                    app_label = deployment_name
                    logger.info(f"No app label found, defaulting to deployment name: {app_label}")
            except subprocess.CalledProcessError:
                # If command fails, default to deployment name
                app_label = deployment_name
                logger.warning(f"Could not get selector for deployment, defaulting to: {app_label}")
        
        logger.info(f"Using app label for pod selection: {app_label}")
        
        # Get the pod name using the app label
        cmd = ["kubectl", "get", "pods", "-n", namespace, "-o", "jsonpath={.items[0].metadata.name}", 
               "-l", f"app={app_label}"]
        logger.info(f"Executing command: {' '.join(cmd)}")
        
        try:
            pod_name = subprocess.check_output(cmd).decode('utf-8').strip()
            
            if not pod_name:
                raise ValueError(f"No pods found with label app={app_label}")
                
        except (subprocess.CalledProcessError, ValueError) as e:
            logger.error(f"Error finding pods with label app={app_label}: {e}")
            logger.info("Trying alternative method - listing all pods and searching for match...")
            
            # Fall back to getting all pods and finding ones related to the deployment
            cmd = ["kubectl", "get", "pods", "-n", namespace, "--field-selector=status.phase=Running"]
            logger.info(f"Executing command: {' '.join(cmd)}")
            all_pods_output = subprocess.check_output(cmd).decode('utf-8')
            
            # Process the output to find pods that might be related to our deployment
            lines = all_pods_output.strip().split('\n')
            if len(lines) < 2:  # Headers only, no pods
                logger.error("No running pods found in namespace")
                return False
                
            potential_pods = []
            for line in lines[1:]:  # Skip header line
                parts = line.split()
                if not parts:
                    continue
                    
                pod_name_part = parts[0]
                # Collect pods that might belong to our deployment
                if deployment_name in pod_name_part or (app_label and app_label in pod_name_part):
                    potential_pods.append(pod_name_part)
                    
            if not potential_pods:
                logger.error(f"No pods found that match the deployment name or label")
                return False
                
            # Use the first matching pod
            pod_name = potential_pods[0]
                
        logger.info(f"Found pod: {pod_name}")
        
        # Get logs from the pod
        cmd = ["kubectl", "logs", "-n", namespace, pod_name]
        logger.info(f"Executing command: {' '.join(cmd)}")
        logs = subprocess.check_output(cmd)
        logs_str = logs.decode('utf-8')
        
        # Save all logs
        all_log_output_file = f"{output_dir}/all-{deployment_name}.log.txt"
        with open(all_log_output_file, 'w', encoding='utf-8') as f:
            f.write(logs_str)
        
        output_size = len(logs_str) / 1024  # Size in KB
        logger.info(f"Unfiltered Logs saved to {all_log_output_file} ({output_size:.2f} KB)")
            
        # Process logs based on keyword filter
        if keyword:
            logger.info(f"Filtering logs for lines containing keyword: '{keyword}'")
            filtered_lines = []
            total_lines = 0
            filtered_count = 0
            for line in logs_str.splitlines():
                total_lines += 1
                if keyword in line:
                    filtered_lines.append(line)
                    filtered_count += 1
            filtered_output_content = '\n'.join(filtered_lines)
            logger.info(f"Filtered {filtered_count} lines containing keyword from {total_lines} total lines")
            filtered_log_output_file = f"{output_dir}/filtered-{deployment_name}.log.csv"
            with open(filtered_log_output_file, 'w', encoding='utf-8') as f:
                f.write(filtered_output_content)
            output_size = len(filtered_output_content) / 1024  # Size in KB
            logger.info(f"Filtered Logs saved to {filtered_log_output_file} ({output_size:.2f} KB)")
        return True
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing kubectl command: {e}")
        logger.error(f"Command output: {e.output.decode('utf-8') if hasattr(e, 'output') else 'No output'}")
        return False
    except Exception as e:
        logger.error(f"Error collecting logs: {e}")
        logger.error(traceback.format_exc())
        return False

def get_current_pdt_time():
    """Get the current time in PDT timezone"""
    pdt_tz = pytz.timezone('America/Los_Angeles')
    utc_now = datetime.datetime.utcnow()
    utc_now = utc_now.replace(tzinfo=pytz.utc)
    pdt_now = utc_now.astimezone(pdt_tz)
    return pdt_now.strftime("%Y%m%d_%H%M%S")



def static_hash(value: str) -> str:
    hash_object = hashlib.sha256(value.encode())
    return hash_object.hexdigest()[:8]

def write_to_file(log_data, output_path):
    """
    Write log data to file with proper filename handling.
    
    Args:
        log_data (dict): Dictionary of log data where keys are request IDs and values are log messages
        output_path (str): Desired output file path
    """
    import time
    
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # If the filename would be too long, create a shorter one
        if len(os.path.basename(output_path)) > 200:  # Leave some margin under 255
            # Create a hash-based filename
            timestamp = str(int(time.time()))
            hash_obj = hashlib.md5(output_path.encode())
            short_name = f"batch_{timestamp}_{hash_obj.hexdigest()[:8]}.csv"
            output_path = os.path.join(os.path.dirname(output_path), short_name)
            logger.info(f"Long filename detected, using shortened name: {output_path}")
        
        with open(output_path, "w") as log_file:
            for request_id, log_message in log_data.items():
                log_file.write(f"{log_message}\n")
        
        logger.info(f"Successfully wrote {len(log_data)} entries to {output_path}")
        return output_path
        
    except OSError as e:
        if "File name too long" in str(e):
            # Fallback: use a temporary file with a short name
            temp_dir = os.path.dirname(output_path)
            timestamp = str(int(time.time()))
            fallback_name = f"batch_{timestamp}.csv"
            fallback_path = os.path.join(temp_dir, fallback_name)
            
            logger.warning(f"Filename too long, using fallback: {fallback_path}")
            
            with open(fallback_path, "w") as log_file:
                for request_id, log_message in log_data.items():
                    log_file.write(f"{log_message}\n")
            
            logger.info(f"Successfully wrote {len(log_data)} entries to {fallback_path}")
            return fallback_path
        else:
            raise e
        
        
def get_process_using_port(port):
    """Find the process ID using a specific port"""
    try:
        # Method 1: Using psutil (more reliable)
        for conn in psutil.net_connections():
            if conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
                try:
                    process = psutil.Process(conn.pid)
                    return conn.pid, process.name()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
    except Exception as e:
        logger.warning(f"psutil method failed: {e}, trying alternative method")
    
    try:
        # Method 2: Using lsof command (fallback)
        result = subprocess.run(['lsof', '-t', f'-i:{port}'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            pid = int(result.stdout.strip().split('\n')[0])
            try:
                process = psutil.Process(pid)
                return pid, process.name()
            except:
                return pid, "unknown"
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    try:
        # Method 3: Using netstat (another fallback)
        result = subprocess.run(['netstat', '-tlnp'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if f':{port} ' in line and 'LISTEN' in line:
                    parts = line.split()
                    if len(parts) > 6 and '/' in parts[6]:
                        pid_program = parts[6].split('/')
                        if pid_program[0].isdigit():
                            return int(pid_program[0]), pid_program[1] if len(pid_program) > 1 else "unknown"
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    return None, None


def kill_process_using_port(port, force=False):
    """Kill the process using a specific port"""
    pid, process_name = get_process_using_port(port)
    
    if not pid:
        logger.info(f"No process found using port {port}")
        return True
    
    logger.info(f"Found process {process_name} (PID: {pid}) using port {port}")
    
    try:
        process = psutil.Process(pid)
        logger.info(f"Attempting to terminate process {process_name} (PID: {pid})")
        
        if not force:
            # Try graceful termination first
            process.terminate()
            
            # Wait up to 5 seconds for graceful termination
            try:
                process.wait(timeout=5)
                logger.info(f"Process {process_name} (PID: {pid}) terminated gracefully")
                return True
            except psutil.TimeoutExpired:
                logger.warning(f"Process {process_name} (PID: {pid}) did not terminate gracefully, forcing kill")
        
        # Force kill if graceful termination failed or force=True
        process.kill()
        
        # Wait up to 3 seconds for force kill
        try:
            process.wait(timeout=3)
            logger.info(f"Process {process_name} (PID: {pid}) force killed successfully")
            return True
        except psutil.TimeoutExpired:
            logger.error(f"Failed to kill process {process_name} (PID: {pid})")
            return False
            
    except psutil.NoSuchProcess:
        logger.info(f"Process {pid} no longer exists")
        return True
    except psutil.AccessDenied:
        logger.error(f"Access denied when trying to kill process {pid}. Try running with sudo/admin privileges")
        return False
    except Exception as e:
        logger.error(f"Error killing process {pid}: {e}")
        return False
    
def is_port_in_use(port):
    """Check if a port is already in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('0.0.0.0', port))
            return False
        except OSError:
            return True

def ensure_port_available(port, max_attempts=3):
    """Ensure port is available by killing any process using it if necessary"""
    logger.info(f"Ensuring port {port} is available...")
    
    for attempt in range(max_attempts):
        if not is_port_in_use(port):
            logger.info(f"Port {port} is available")
            return True
        
        logger.info(f"Port {port} is in use (attempt {attempt + 1}/{max_attempts})")
        
        # Kill the process using the port
        success = kill_process_using_port(port, force=(attempt > 0))
        
        if success:
            # Wait a moment for the port to be released
            time.sleep(2)
            
            # Check if port is now available
            if not is_port_in_use(port):
                logger.info(f"Port {port} is now available after killing previous process")
                return True
        
        # If we're not on the last attempt, wait before trying again
        if attempt < max_attempts - 1:
            logger.info(f"Waiting 3 seconds before next attempt...")
            time.sleep(3)
    
    logger.error(f"Failed to make port {port} available after {max_attempts} attempts")
    return False

def wait_for_port_available(port, max_wait=30):
    """Wait for port to become available (original function, kept for compatibility)"""
    logger.info(f"Checking if port {port} is available...")
    
    start_time = time.time()
    while is_port_in_use(port):
        if time.time() - start_time > max_wait:
            logger.error(f"Port {port} is still in use after {max_wait} seconds")
            return False
        logger.info(f"Port {port} is in use, waiting...")
        time.sleep(1)
    
    logger.info(f"Port {port} is now available")
    return True