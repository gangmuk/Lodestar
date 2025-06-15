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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class K8sHotDeployer:
    def __init__(self, namespace="default", app_label="routing-agent-service"):
        self.namespace = namespace
        self.app_label = app_label
        
        try:
            config.load_incluster_config()
            logger.info("Loaded in-cluster config")
        except config.ConfigException:
            config.load_kube_config()
            logger.info("Loaded local kubeconfig")
            
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
    
    def copy_file_to_pod(self, pod_name, local_path, remote_path):
        """Copy a single file to pod"""
        try:
            if not Path(local_path).exists():
                logger.error(f"Local file {local_path} does not exist")
                return False
            
            with tempfile.NamedTemporaryFile(suffix='.tar', delete=False) as temp_tar:
                with tarfile.open(temp_tar.name, 'w') as tar:
                    tar.add(local_path, arcname=os.path.basename(local_path))
                
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
            logger.info(f"✅ Copied {local_path} to {pod_name}:{remote_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to copy {local_path} to {pod_name}: {e}")
            return False
    
    def copy_directory_to_pod(self, pod_name, local_dir, remote_dir):
        """Copy entire directory to pod"""
        try:
            local_path = Path(local_dir)
            if not local_path.exists() or not local_path.is_dir():
                logger.error(f"Local directory {local_dir} does not exist or is not a directory")
                return False
            
            # Get all files in the directory
            files_to_copy = []
            for file_path in local_path.rglob('*'):
                if file_path.is_file():
                    files_to_copy.append(file_path)
            
            if not files_to_copy:
                logger.warning(f"No files found in {local_dir}")
                return True
            
            logger.info(f"Found {len(files_to_copy)} files in {local_dir}")
            
            # Create tar archive with directory structure
            with tempfile.NamedTemporaryFile(suffix='.tar', delete=False) as temp_tar:
                with tarfile.open(temp_tar.name, 'w') as tar:
                    for file_path in files_to_copy:
                        # Preserve directory structure relative to local_dir
                        arcname = file_path.relative_to(local_path.parent)
                        tar.add(file_path, arcname=str(arcname))
                        logger.debug(f"Added to tar: {file_path} as {arcname}")
                
                # Ensure remote directory exists and extract
                command = ['sh', '-c', f'mkdir -p {os.path.dirname(remote_dir)} && tar -xmf - -C {os.path.dirname(remote_dir)}']
                
                resp = stream(
                    self.v1.connect_get_namespaced_pod_exec,
                    pod_name, self.namespace, command=command,
                    stderr=True, stdin=True, stdout=True, tty=False, _preload_content=False
                )
                
                with open(temp_tar.name, 'rb') as tar_data:
                    resp.write_stdin(tar_data.read())
                resp.close()

            os.unlink(temp_tar.name)
            logger.info(f"✅ Copied directory {local_dir} to {pod_name}:{remote_dir}")
            
            # List files to verify
            result = self.execute_command(pod_name, f"find {remote_dir} -type f | head -10")
            if result.strip():
                logger.info(f"Sample files in {remote_dir}: {result.strip().replace(chr(10), ', ')}")
            
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
                logger.info(f"Found routing_agent_service.py process with PID: {pid}")
                
                # Kill the process
                kill_result = self.execute_kubectl_command(pod_name, f"kill {pid}")
                if kill_result:
                    logger.info(f"✅ Successfully killed routing_agent_service.py (PID: {pid}) in {pod_name}")
                    
                    # Wait a moment and verify it's killed
                    time.sleep(2)
                    verify_output = self.execute_command(pod_name, find_pid_cmd)
                    if not verify_output.strip():
                        logger.info(f"✅ Confirmed process is terminated in {pod_name}")
                        return True
                    else:
                        logger.warning(f"⚠️ Process may still be running in {pod_name}")
                        return False
                else:
                    return False
            else:
                logger.info(f"No routing_agent_service.py process found running in {pod_name}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to kill routing_agent_service.py in {pod_name}: {e}")
            return False

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
                logger.info(f"✅ Started '{command}' in background in {pod_name}")
            else:
                logger.info(f"✅ Executed '{command}' in {pod_name}")
                if output.strip():
                    logger.info(f"Output: {output.strip()}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to execute '{command}' in {pod_name}: {e}")
            return False
    
    def restart_flask_safely(self, pod_name):
        """Restart Flask using the wrapper script's signal handling"""
        logger.info(f"Restarting Flask in {pod_name} using wrapper script...")
        
        # Method 1: Send SIGUSR1 to wrapper script (PID 1)
        result = self.execute_command(pod_name, "kill -USR1 1")
        logger.info(f"Sent SIGUSR1 to wrapper script (PID 1)")
        
        # Method 2: Alternative - create restart trigger file
        self.execute_command(pod_name, "touch /app/.restart_trigger")
        logger.info(f"Created restart trigger file")
        
        # Wait for restart
        time.sleep(8)
        
        # Verify Flask is running
        ps_output = self.execute_command(pod_name, "ps aux | grep routing_agent_service.py")
        if "routing_agent_service.py" in ps_output and "grep" not in ps_output.replace("grep routing_agent_service.py", ""):
            logger.info(f"✅ Flask is running after restart in {pod_name}")
            return True
        else:
            logger.error(f"❌ Flask may not be running after restart in {pod_name}")
            logger.info(f"Process check: {ps_output}")
            return False
    
    def deploy_to_pods(self, file_mappings, directory_mappings):
        """Deploy files, directories and optionally restart Flask"""
        pods = self.get_pods()
        
        if not pods:
            logger.error(f"No running pods found")
            return False
        
        logger.info(f"Found {len(pods)} running pods: {pods}")
        
        overall_success = True
        for pod_name in pods:
            logger.info(f"\n🚀 Deploying to pod: {pod_name}")
            
            # Copy individual files
            files_copied = 0
            for local_path, remote_path in file_mappings.items():
                if self.copy_file_to_pod(pod_name, local_path, remote_path):
                    files_copied += 1
                else:
                    logger.error(f"Failed to copy {local_path} to {pod_name}:{remote_path}")
                    logger.error(f"Exiting...")
                    exit()
            
            # Copy directories
            dirs_copied = 0
            for local_dir, remote_dir in directory_mappings.items():
                if self.copy_directory_to_pod(pod_name, local_dir, remote_dir):
                    dirs_copied += 1
                else:
                    logger.error(f"Failed to copy directory {local_dir} to {pod_name}:{remote_dir}")
                    logger.error(f"Exiting...")
                    exit()
            
            total_expected = len(file_mappings) + len(directory_mappings)
            total_copied = files_copied + dirs_copied
            
            if total_copied == total_expected:
                logger.info(f"✅ Copied {files_copied} files and {dirs_copied} directories to {pod_name}")
            else:
                logger.error(f"❌ Only copied {total_copied}/{total_expected} items to {pod_name}")
                overall_success = False
        
        return overall_success

def main():
    """Main function"""
    
    # Configuration
    NAMESPACE = "default"
    APP_LABEL = "routing-agent-service"
    
    # Individual files to deploy
    FILES_TO_DEPLOY = {
        "./routing_agent_service.py": "/app/routing_agent_service.py",
        "./preprocess.py": "/app/preprocess.py",
        "./feature_normalization.py": "/app/feature_normalization.py",
        "./encoding.py": "/app/encoding.py",
        "./simpler_contextual_bandit.py": "/app/simpler_contextual_bandit.py",
        "./contextual_bandit.py": "/app/contextual_bandit.py",
        "./logger.py": "/app/logger.py",
    }
    
    # Directories to deploy
    DIRECTORIES_TO_DEPLOY = {
        "./final_model": "/app/final_model",
    }
    # Check if files exist
    missing_files = [f for f in FILES_TO_DEPLOY.keys() if not os.path.exists(f)]
    if missing_files:
        logger.error(f"Missing files: {missing_files}")
        sys.exit(1)
    
    # Check if directories exist
    missing_dirs = [d for d in DIRECTORIES_TO_DEPLOY.keys() if not os.path.exists(d)]
    if missing_dirs:
        logger.error(f"Missing directories: {missing_dirs}")
        sys.exit(1)
    
    restart = False
    
    # Deploy
    deployer = K8sHotDeployer(namespace=NAMESPACE, app_label=APP_LABEL)
    
    pods = deployer.get_pods()
    if len(pods) == 0:
        logger.error("❌ No running pods found. Please ensure the routing-agent-service is deployed.")
        sys.exit(1)
    if len(pods) > 1:
        logger.error(f"❌ Multiple pods found: {pods}. Please specify a single pod to deploy to.")
        sys.exit(1)
    pod_name = pods[0]
    
    logger.info("🚀 Starting deployment...")
    logger.info(f"Files: {list(FILES_TO_DEPLOY.keys())}")
    logger.info(f"Directories: {list(DIRECTORIES_TO_DEPLOY.keys())}")
    logger.info(f"Restart Flask: {restart}")
    
    if deployer.deploy_to_pods(FILES_TO_DEPLOY, DIRECTORIES_TO_DEPLOY):
        if restart:
            if deployer.restart_flask_safely(pod_name):
                logger.info(f"✅ Successfully deployed and restarted Flask in {pod_name}")
            else:
                logger.error(f"❌ Failed to restart Flask in {pod_name}")
                sys.exit(1)
        else:
            logger.info(f"✅ Files and directories copied to {pod_name} (no restart)")
        logger.info("\n🎉 Deployment completed successfully!")
        if restart:
            logger.info("Flask has been restarted with new code and models.")
        else:
            logger.info("Files and directories copied. Flask still running with old code.")

        if deployer.kill_routing_service(pod_name):
            logger.info(f"✅ Successfully killed routing_agent_service.py in {pod_name}")
        else:
            logger.error(f"❌ Failed to kill routing_agent_service.py in {pod_name}")
            sys.exit(1)

        for pod_name in pods:
            logger.info(f"\n🔧 Executing additional commands in {pod_name}")
            deployer.execute_kubectl_command(pod_name, "rm /app/llm_router.log")
            deployer.execute_kubectl_command(pod_name, "python routing_agent_service.py", background=True)
    else:
        logger.error("\n💥 Deployment failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()