#!/usr/bin/env python3

# preprocess.py

from logger import logger, INCLUDE_GPU_IN_FEATURE
from kubernetes import client, config
import re

def get_all_pod_ips_from_data_file(data_file):
    pod_ips = set()
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Method 1: Extract from selectedpod@ field
            # Pattern: selectedpod@IP@
            selectedpod_pattern = r'selectedpod@(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})@'
            selectedpod_matches = re.findall(selectedpod_pattern, content)
            pod_ips.update(selectedpod_matches)
            # Method 2: Extract from JSON-like structures
            # Pattern: "IP":value in JSON-like structures
            json_ip_pattern = r'"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})":'
            json_ip_matches = re.findall(json_ip_pattern, content)
            pod_ips.update(json_ip_matches)
            # Method 3: General IP pattern as fallback (more comprehensive)
            # This catches any IP that might be in other formats
            general_ip_pattern = r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b'
            general_ip_matches = re.findall(general_ip_pattern, content)
            
            # Filter to only include IPs that look like pod IPs (10.x.x.x range)
            pod_like_ips = [ip for ip in general_ip_matches if ip.startswith('10.')]
            pod_ips.update(pod_like_ips)
            
    except FileNotFoundError:
        logger.error(f"Data file {data_file} not found")
        return set()
    except Exception as e:
        logger.error(f"Error reading data file {data_file}: {e}")
        return set()
    
    return sorted(list(pod_ips)) # THIS WAS ALSO THE BUG! Without sorted, the pod IPs were not in order, causing issues in mapping.

def create_pod_ip_to_generalpodid_mapping(unique_pod_ips):
    pod_ip_to_generalpodid = {}
    unique_pod_ips = sorted(unique_pod_ips) ## Without sorted, THIS WAS THE BUG!!!! 
    for idx, pod_ip in enumerate(unique_pod_ips):
        if idx < 10:
            pod_ip_to_generalpodid[pod_ip] = f"pod_000{idx}"
        elif idx < 100:
            pod_ip_to_generalpodid[pod_ip] = f"pod_00{idx}"
        else:
            logger.error(f"Too many pods ({len(unique_pod_ips)}) to map to generalpodid, only supporting up to 100 pods")
            assert False
    print(f"pod_ip_to_generalpodid: {pod_ip_to_generalpodid}")
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
            logger.error(f"Unknown GPU model {gpu_model} for pod IP {pod_ip}")
            assert False
    return pod_ip_to_gpu_model, pod_ip_to_gpu_model_encoded

def get_running_pods_by_label(label_selector):
    config.load_kube_config()
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
    config.load_kube_config()
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
                    gpu_model = node_labels.get('machine.cluster.vke.volcengine.com/gpu-name', 'unknown')
                    generalpodid_to_gpu_model[generalpodid] = gpu_model
                except Exception as e:
                    logger.warning(f"Failed to get node info for {node_name}: {e}")
                    generalpodid_to_gpu_model[generalpodid] = 'unknown'
            else:
                logger.warning(f"Pod {pod.metadata.name} has no node assignment")
                generalpodid_to_gpu_model[generalpodid] = 'unknown'
    return generalpodid_to_gpu_model