from kubernetes import client, config
import json
import threading
from typing import List, Any, Dict
import time
import os
import subprocess
import traceback
from logger import logger

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

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


def check_deployment_ready_kubernetes(deployment_name, k8s_cluster, namespace):
    """
    Checks if all pods of a deployment and all their containers are in a ready state using the Kubernetes Python client.

    Args:
        deployment_name (str): The name of the deployment.
        namespace (str): The namespace where the deployment is located.

    Returns:
        bool: True if all pods and their containers are ready, False otherwise (will keep checking).
    """
    try:
        if k8s_cluster == "vke":
            kube_config_file = os.path.expanduser('~/.kube/config')
        else:
            kube_config_file = os.path.expanduser('~/.kube/config-local')
        if not os.path.exists(kube_config_file):
            logger.error(f"Error: {kube_config_file} does not exist")
            assert False
        config.load_kube_config(config_file=kube_config_file)
        
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
                    logger.error(f"No selector found for deployment '{deployment_name}'. Cannot find associated pods. Retrying in 1 second...")
                    time.sleep(1)
                    retry_count += 1
                    continue

                label_selector = ",".join([f"{k}={v}" for k, v in selector.items()])
                pod_list = core_v1.list_namespaced_pod(namespace=namespace, label_selector=label_selector)
                pods = pod_list.items

                if not pods:
                    logger.error(f"No pods found for deployment '{deployment_name}' in namespace '{namespace}' with selector '{label_selector}'. Retrying in 1 second...")
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
                        logger.error(f"Pod '{pod_name}' is not ready. Retrying in 1 second...")
                        all_ready = False
                        break

                    container_statuses = pod.status.container_statuses or []
                    for container_status in container_statuses:
                        if not container_status.ready:
                            container_name = container_status.name
                            logger.error(f"Container '{container_name}' in pod '{pod_name}' is not ready. Retrying in 1 second...")
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
                    logger.error(f"Deployment '{deployment_name}' not found in namespace '{namespace}'. Please check the name and namespace.")
                    assert False
                else:
                    logger.error(f"Kubernetes API exception: {e}")
                    logger.error("Retrying in 1 second...")
                    time.sleep(1)
                    retry_count += 1
                    
        logger.error(f"Max retries ({max_retries}) reached. Deployment '{deployment_name}' is not ready.")
        return False

    except config.ConfigException as e:
        logger.error(f"Kubernetes configuration error: {e}")
        logger.error("Please ensure your kubeconfig file is properly configured.")
        return False  # Exit if configuration is invalid
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
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