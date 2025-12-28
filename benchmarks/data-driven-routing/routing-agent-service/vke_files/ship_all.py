#!/usr/bin/env python3
"""
Enhanced Hot Deploy Script for start.sh setup
Now supports copying entire directories and individual files
Can safely restart Flask without killing the pod
"""

import os
import sys
import time
import tempfile
import tarfile
import glob
from pathlib import Path
from kubernetes import client, config
from kubernetes.stream import stream
import logging
import utils as utils
import argparse
from logger import logger, INCLUDE_GPU_IN_FEATURE

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)


class K8sDeployment:
    def __init__(self, namespace="default", app_label="routing-agent-service"):
        self.namespace = namespace
        self.app_label = app_label
        # kube_config_file = os.path.expanduser('~/.kube/config')
        kube_config_file = os.path.expanduser('~/.kube/config')
        if not os.path.exists(kube_config_file):
            print(f"Error: {kube_config_file} does not exist")
            assert False
        config.load_kube_config(config_file=kube_config_file)
        self.v1 = client.CoreV1Api()
    
    def get_pods(self):
        """Get running pods"""
        try:
            label_selector = f"app={self.app_label}"
            pods = self.v1.list_namespaced_pod(namespace=self.namespace, label_selector=label_selector)
            
            return [pod.metadata.name for pod in pods.items if pod.status.phase == "Running"]
        except Exception as e:
            logger.error(f"Failed to get pods: {e}")
            return []
    
    def copy_file_to_pod(self, pod_name, local_path, remote_path, max_size_mb=10):
        """Copy a single file to pod. Uses kubectl cp for large files."""
        try:
            if not Path(local_path).exists():
                logger.error(f"Local file {local_path} does not exist")
                return False
            
            # Check file size
            file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
            
            # For large files, use kubectl cp which is more robust
            if file_size_mb > max_size_mb:
                logger.info(f"File size ({file_size_mb:.1f}MB) exceeds {max_size_mb}MB, using kubectl cp")
                return self.copy_file_with_kubectl(pod_name, local_path, remote_path)
            
            # Create parent directory if it doesn't exist
            remote_dir = os.path.dirname(remote_path)
            if remote_dir:
                self.execute_command(pod_name, f"mkdir -p {remote_dir}")
            
            with tempfile.NamedTemporaryFile(suffix='.tar', delete=False) as temp_tar:
                with tarfile.open(temp_tar.name, 'w') as tar:
                    # Use the desired remote filename, not the local basename
                    tar.add(local_path, arcname=os.path.basename(remote_path))
                command = ['sh', '-c', f'tar -xmf - -C {os.path.dirname(remote_path)}']
                resp = stream(
                    self.v1.connect_get_namespaced_pod_exec,
                    pod_name, self.namespace, command=command,
                    stderr=True, stdin=True, stdout=True, tty=False, _preload_content=False
                )
                with open(temp_tar.name, 'rb') as tar_data:
                    resp.write_stdin(tar_data.read())
                resp.close()
            os.unlink(temp_tar.name)
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to copy {local_path} to {pod_name}: {e}")
            # Fallback to kubectl cp
            logger.info("Retrying with kubectl cp...")
            return self.copy_file_with_kubectl(pod_name, local_path, remote_path)
    
    def copy_file_with_kubectl(self, pod_name, local_path, remote_path):
        """Copy file using kubectl cp command (more reliable for large files)"""
        try:
            # Create parent directory if it doesn't exist
            remote_dir = os.path.dirname(remote_path)
            if remote_dir:
                self.execute_command(pod_name, f"mkdir -p {remote_dir}")
            
            # Use kubectl cp with the kubeconfig
            kubeconfig_path = os.path.expanduser('~/.kube/config')
            cmd = f"kubectl --kubeconfig={kubeconfig_path} cp {local_path} {self.namespace}/{pod_name}:{remote_path}"
            
            logger.info(f"Executing: {cmd}")
            result = os.system(cmd)
            
            if result == 0:
                logger.info(f"✅ Successfully copied {local_path} using kubectl cp")
                return True
            else:
                logger.error(f"❌ kubectl cp failed with exit code {result}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to copy with kubectl: {e}")
            return False
    
    def copy_directory_to_pod(self, pod_name, local_dir, exclude_dir_or_file_list, remote_dir):
        """Copy entire directory to pod by copying individual files"""
        try:
            local_path = Path(local_dir)
            if not local_path.exists() or not local_path.is_dir():
                logger.error(f"Local directory {local_dir} does not exist or is not a directory")
                return False
            
            # Get all files in the directory
            files_to_copy = []
            for file_path in local_path.rglob('*'):
                if file_path.is_file():
                    rel_path = file_path.relative_to(local_path)
                    if any(part in exclude_dir_or_file_list for part in rel_path.parts):
                        # print(f"Excluding {file_path} because it is in the exclude list")
                        continue
                    files_to_copy.append(file_path)
            
            if not files_to_copy:
                logger.error(f"No files found in {local_dir}")
                return False
            
            print(f"Found {len(files_to_copy)} files in {local_dir}")
            
            # Step 1: Delete remote directory if it exists
            print(f"Cleaning up remote directory {remote_dir}")
            self.execute_command(pod_name, f"rm -rf {remote_dir}")
            
            # Step 2: Create empty remote directory
            print(f"Creating remote directory {remote_dir}")
            result = self.execute_command(pod_name, f"mkdir -p {remote_dir}")
            print(f"Created remote directory {remote_dir}")
            # Step 3: Copy each file individually
            success_count = 0
            for file_path in files_to_copy:
                rel_path = file_path.relative_to(local_path)
                remote_file_path = f"{remote_dir}/{rel_path}"
                remote_parent_dir = os.path.dirname(remote_file_path)
                if remote_parent_dir != remote_dir:
                    self.execute_command(pod_name, f"mkdir -p {remote_parent_dir}")
                if self.copy_file_to_pod(pod_name, str(file_path), remote_file_path):
                    success_count += 1
                    # print(f"Copied: {file_path} -> {remote_file_path}")
                    print(f"Copied: {file_path}")
                else:
                    logger.error(f"Failed to copy: {file_path}")
                    return False
            print(f"✅ Successfully copied {success_count}/{len(files_to_copy)} files to {pod_name}:{remote_dir}")
            # Verify by listing some files
            result = self.execute_command(pod_name, f"find {remote_dir} -type f | head -10")
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            self.execute_command(pod_name, f"touch {remote_dir}/copied_from_{local_dir}_at_{current_time}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to copy directory {local_dir} to {pod_name}: {e}")
            return False
    
    def execute_command(self, pod_name, command):
        """Execute command in pod"""
        try:
            if isinstance(command, str):
                command = ['sh', '-c', command]
            
            resp = stream(
                self.v1.connect_get_namespaced_pod_exec,
                pod_name, self.namespace, command=command,
                stderr=True, stdin=False, stdout=True, tty=False
            )
            
            return resp if isinstance(resp, str) else resp.read_stdout()
        except Exception as e:
            logger.error(f"Failed to execute command in {pod_name}: {e}")
            return ""

    def kill_routing_service(self, pod_name):
        """Kill running routing_agent_service.py process"""
        try:
            # Find the PID of routing_agent_service.py
            find_pid_cmd = "ps aux | grep 'routing_agent_service.py' | grep -v grep | awk '{print $2}'"
            pid_output = self.execute_command(pod_name, find_pid_cmd)
            
            if pid_output.strip():
                pid = pid_output.strip().split('\n')[0]  # Get first PID if multiple
                print(f"Found routing_agent_service.py process with PID: {pid}")
                
                # Kill the process
                kill_result = self.execute_kubectl_command(pod_name, f"kill {pid}")
                if kill_result:
                    print(f"✅ Successfully killed routing_agent_service.py (PID: {pid}) in {pod_name}")
                    
                    # Wait a moment and verify it's killed
                    time.sleep(2)
                    verify_output = self.execute_command(pod_name, find_pid_cmd)
                    if not verify_output.strip():
                        print(f"✅ Confirmed process is terminated in {pod_name}")
                        return True
                    else:
                        logger.warning(f"⚠️ Process may still be running in {pod_name}")
                        return False
                else:
                    return False
            else:
                print(f"No routing_agent_service.py process found running in {pod_name}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to kill routing_agent_service.py in {pod_name}: {e}")
            exit(1)

    def execute_kubectl_command(self, pod_name, command, background=False):
        """Execute kubectl command in pod"""
        try:
            if background:
                # Run command in background using nohup and &
                if isinstance(command, str):
                    bg_command = f"nohup {command} > /dev/null 2>&1 &"
                    cmd = ['sh', '-c', bg_command]
                else:
                    # Join list command and run in background
                    bg_command = f"nohup {' '.join(command)} > /dev/null 2>&1 &"
                    cmd = ['sh', '-c', bg_command]
            else:
                if isinstance(command, str):
                    cmd = ['sh', '-c', command]
                else:
                    cmd = command
            
            resp = stream(
                self.v1.connect_get_namespaced_pod_exec,
                pod_name, self.namespace, command=cmd,
                stderr=True, stdin=False, stdout=True, tty=False
            )
            
            output = resp if isinstance(resp, str) else resp.read_stdout()
            
            if background:
                print(f"✅ Started '{command}' in background in {pod_name}")
            else:
                print(f"✅ Executed '{command}' in {pod_name}")
                if output.strip():
                    print(f"Output: {output.strip()}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to execute '{command}' in {pod_name}: {e}")
            return False
    
    def restart_flask_safely(self, pod_name):
        """Restart Flask using the wrapper script's signal handling"""
        print(f"Restarting Flask in {pod_name} using wrapper script...")
        
        # Method 1: Send SIGUSR1 to wrapper script (PID 1)
        result = self.execute_command(pod_name, "kill -USR1 1")
        print(f"Sent SIGUSR1 to wrapper script (PID 1)")
        
        # Method 2: Alternative - create restart trigger file
        self.execute_command(pod_name, "touch /app/.restart_trigger")
        print(f"Created restart trigger file")
        
        # Wait for restart
        time.sleep(8)
        
        # Verify Flask is running
        ps_output = self.execute_command(pod_name, "ps aux | grep routing_agent_service.py")
        if "routing_agent_service.py" in ps_output and "grep" not in ps_output.replace("grep routing_agent_service.py", ""):
            print(f"✅ Flask is running after restart in {pod_name}")
            return True
        else:
            logger.error(f"❌ Flask may not be running after restart in {pod_name}")
            print(f"Process check: {ps_output}")
            return False
    
    def deploy_to_pods(self, agent_related_files, final_model_dir):
        """Deploy files, directories and optionally restart Flask"""
        pods = self.get_pods()
        
        if not pods:
            logger.error(f"No running pods found")
            return False
        
        print(f"Found {len(pods)} running pods: {pods}")
        
        overall_success = True
        for pod_name in pods:
            print(f"\n🚀 Deploying to pod: {pod_name}")
            
            # Copy individual files
            files_copied = 0
            for local_path, remote_path in agent_related_files.items():
                print(f"Starting to copy file: {local_path} to {pod_name}")
                if self.copy_file_to_pod(pod_name, local_path, remote_path):
                    print(f"Copied {local_path}")
                    files_copied += 1
                else:
                    logger.error(f"Failed to copy {local_path} to {pod_name}:{remote_path}")
                    logger.error(f"Exiting...")
                    exit()
            
            # Copy directories
            exclude_dir_or_file_list = ['weights_csv', 'checkpoints', 'tensor_dataset.pt', 'xai_report']
            dirs_copied = 0
            for local_dir, remote_dir in final_model_dir.items():
                print(f"Copying directory {local_dir} to {pod_name}:{remote_dir}")
                if self.copy_directory_to_pod(pod_name, local_dir, exclude_dir_or_file_list, remote_dir):
                    dirs_copied += 1
                else:
                    logger.error(f"Failed to copy directory {local_dir} to {pod_name}:{remote_dir}")
                    logger.error(f"Exiting...")
                    exit()
            total_expected = len(agent_related_files) + len(final_model_dir)
            total_copied = files_copied + dirs_copied
            
            if total_copied == total_expected:
                print(f"✅ Copied {files_copied} files and {dirs_copied} directories to {pod_name}")
            else:
                logger.error(f"❌ Only copied {total_copied}/{total_expected} items to {pod_name}")
                overall_success = False
        
        return overall_success
    


def main():
    parser = argparse.ArgumentParser(description='Ship all files and directories')
    parser.add_argument('--ship_code', type=int, default=1, help='ship_code')
    parser.add_argument('--ship_model', type=int, default=1, help='ship_model')
    parser.add_argument('--ship_offline_training_data', type=int, default=0, help='ship_offline_training_data')
    parser.add_argument('--final_model_dir', type=str, default=None, help='Final model directory')
    parser.add_argument('--k8s_cluster', type=str, default='vke', choices=['vke', 'aws', 'local'], help='Kubernetes cluster')
    args = parser.parse_args()
    
    if args.ship_code == 0 and args.ship_model == 0:
        print("Nothing to ship")
        return 

    print("Restarting gateway and routing-agent-service")
    os.system("kubectl rollout restart deployment routing-agent-service -n default & kubectl rollout restart deployment aibrix-gateway-plugins -n aibrix-system")
    time.sleep(3)
    print("Check if routing-agent-service is ready")
    utils.check_deployment_ready_kubernetes('routing-agent-service', args.k8s_cluster, 'default')
    time.sleep(2)

    NAMESPACE = "default"
    APP_LABEL = "routing-agent-service"
    deployment = K8sDeployment(namespace=NAMESPACE, app_label=APP_LABEL)
    iter = 0
    pod_name = None
    while True:
        pods = deployment.get_pods()
        if len(pods) == 0:
            logger.error("❌ No running pods found. Please ensure the routing-agent-service is deployed.")
            sys.exit(1)
        elif len(pods) > 1:
            print(f"Multiple pods found: {pods}. Will try again in 5 seconds...")
            time.sleep(5)
        elif len(pods) == 1:
            print(f"Found 1 running pod: {pods[0]}")
            pod_name = pods[0]
            break
        if iter >= 10:
            logger.error("❌ Unable to find a single running pod after multiple attempts. Exiting...")
            sys.exit(1)
        iter += 1
    assert pod_name is not None
    
    # Individual files to deploy
    agent_related_files = {}
    if args.ship_code == 1:
        print("Shipping all files and directories")
        agent_related_files = {
            "../agent_codes/routing_agent_service.py": "/app/routing_agent_service.py",
            "../agent_codes/preprocess.py": "/app/preprocess.py",
            "../agent_codes/data_processor.py": "/app/data_processor.py",
            "../agent_codes/data_normalizer.py": "/app/data_normalizer.py",
            "../agent_codes/encoding.py": "/app/encoding.py",
            # "../agent_codes/rl_routing_agent_sb3.py": "/app/rl_routing_agent_sb3.py",
            "../agent_codes/neural_contextual_bandit_perpodmodel_advanced.py": "/app/neural_contextual_bandit_perpodmodel_advanced.py",
            "../agent_codes/neural_contextual_bandit_perpodmodel_checkpoint.py": "/app/neural_contextual_bandit_perpodmodel_checkpoint.py",
            "../agent_codes/neural_contextual_bandit_perpodmodel_policygradient.py": "/app/neural_contextual_bandit_perpodmodel_policygradient.py",
            "../agent_codes/distribution_shift_detector.py": "/app/distribution_shift_detector.py",
            "../agent_codes/rwlock.py": "/app/rwlock.py",
            "../agent_codes/scalable_rl_routing_agent.py": "/app/scalable_rl_routing_agent.py",
            "../agent_codes/simpler_contextual_bandit.py": "/app/simpler_contextual_bandit.py",
            "../agent_codes/latency_predictor.py": "/app/latency_predictor.py",
            "../agent_codes/logger.py": "/app/logger.py",
            "../agent_codes/utils.py": "/app/utils.py",
            
            # # Agents module
            # "../agent_codes/agents/__init__.py": "/app/agents/__init__.py",
            # "../agent_codes/agents/rout_agent.py": "/app/agents/rout_agent.py",
            # "../agent_codes/agents/reinforce.py": "/app/agents/reinforce.py",
            # "../agent_codes/agents/replay_buffer.py": "/app/agents/replay_buffer.py",
            # "../agent_codes/agents/tracker.py": "/app/agents/tracker.py",
            # # Envs module
            # "../agent_codes/envs/__init__.py": "/app/envs/__init__.py",
            # "../agent_codes/envs/broker.py": "/app/envs/broker.py",
            # "../agent_codes/envs/request.py": "/app/envs/request.py",
            # "../agent_codes/envs/request_source_gateway.py": "/app/envs/request_source_gateway.py",
            # "../agent_codes/envs/rout_env.py": "/app/envs/rout_env.py",
            # "../agent_codes/envs/rl_env_wrappers.py": "/app/envs/rl_env_wrappers.py",
            # # Policies module
            # "../agent_codes/policies/__init__.py": "/app/policies/__init__.py",
            # "../agent_codes/policies/policy.py": "/app/policies/policy.py",
            # "../agent_codes/policies/nets.py": "/app/policies/nets.py",
        }
    FINAL_MODEL_DIR = {}
    if args.ship_model == 1:
        print("Shipping only final_model directory")
        # Convert to absolute path for consistency
        final_model_abs_path = os.path.abspath(args.final_model_dir)
        FINAL_MODEL_DIR = {final_model_abs_path: "/app/final_model"}

    if args.ship_offline_training_data:
        # NEW: Ship offline training CSV for online learning
        offline_csv_path = os.path.join(os.path.dirname(final_model_abs_path), "data_replaced-processed.csv")
        print(f"offline training CSV path: source in host: {offline_csv_path}, destination in pod: /app/offline_training_data.csv")
        if os.path.exists(offline_csv_path):
            agent_related_files[offline_csv_path] = "/app/offline_training_data.csv"
            print(f"✅ Will ship offline training CSV: {offline_csv_path}")
        else:
            logger.error(f"❌  offline training CSV not found at {offline_csv_path}")
            logger.error("Online learning will start from scratch with only new data")
            logger.error("Exiting...")
            exit(1)

    # Check if files exist
    missing_files = [f for f in agent_related_files.keys() if not os.path.exists(f)]
    if missing_files:
        logger.error(f"Missing files: {missing_files}")
        sys.exit(1)
    missing_dirs = [d for d in FINAL_MODEL_DIR.keys() if not os.path.exists(d)]
    if missing_dirs:
        logger.error(f"Missing directories: {missing_dirs}")
        sys.exit(1)
    print(f"Files: {list(agent_related_files.keys())}")
    print(f"final_model_dir: {list(FINAL_MODEL_DIR.keys())}")
    if deployment.deploy_to_pods(agent_related_files, FINAL_MODEL_DIR):
        print(f"✅ Files and directories copied to {pod_name} (no restart)")
    else:
        logger.error("\n💥 Deployment failed!")
        sys.exit(1)
    restart = False
    if restart:
        print("🚀 Restarting Flask server process in the routing-agent-service pod...")
        if deployment.restart_flask_safely(pod_name):
            print(f"✅ Successfully restarted Flask in {pod_name}")
        else:
            logger.error(f"❌ Failed to restart Flask in {pod_name}")
            sys.exit(1)
    print("\n🎉 Deployment completed successfully!")
    if deployment.kill_routing_service(pod_name):
        print(f"✅ Successfully killed routing_agent_service.py in {pod_name}")
    else:
        logger.error(f"❌ Failed to kill routing_agent_service.py in {pod_name}. Exiting...")
        sys.exit(1)
    print(f"\n🔧 Executing additional commands in {pod_name}")
    deployment.execute_kubectl_command(pod_name, "python routing_agent_service.py", background=True)

if __name__ == "__main__":
    main()