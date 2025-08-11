from logger import logger, INCLUDE_GPU_IN_FEATURE
from kubernetes import client, config

def create_pod_ip_to_generalpodid_mapping(unique_pod_ips):
    pod_ip_to_generalpodid = {}
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

def get_pods_by_label(label_selector):
        config.load_kube_config()
        v1 = client.CoreV1Api()
        return v1.list_pod_for_all_namespaces(label_selector=label_selector)
    
def fetch_pod_ips(label_selector):
    pods = get_pods_by_label(label_selector)
    if len(pods.items) == 0:
        logger.error(f"No pods found with label selector: {label_selector}")
        assert False
    return [pod.status.pod_ip for pod in pods.items()]

def fetch_generalpodid_to_gpu_model(label_selector, pod_ip_to_generalpodid):
    config.load_kube_config()
    v1 = client.CoreV1Api()
    pods = get_pods_by_label(label_selector)
    if len(pods.items) == 0:
        logger.error(f"No pods found with label selector: {label_selector}")
        assert False
    generalpodid_to_gpu_model = {}
    for pod in pods.items:
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