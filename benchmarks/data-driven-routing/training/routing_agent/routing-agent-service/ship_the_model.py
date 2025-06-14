#!/usr/bin/env python3
"""
Kubernetes Hot Deploy Script for Models
Copies a local 'final_model' directory to running pods, replacing existing one,
and then attempts to restart the application process within the same pod.
"""

import os
import sys
import time
import subprocess
import tempfile
import tarfile
from pathlib import Path
from kubernetes import client, config
from kubernetes.stream import stream
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class K8sModelDeployer:
    def __init__(self, namespace="default", app_label="your-app-label",
                 main_app_file="routing_agent_service.py",
                 app_startup_command="python routing_agent_service.py"):
        """
        Initialize the deployer
        
        Args:
            namespace: Kubernetes namespace
            app_label: Label selector to find your pods
            main_app_file: The main Python application file to look for when killing the process.
            app_startup_command: The command to explicitly start the Flask server if needed.
                                 Leave empty if your container's entrypoint/CMD handles restarts.
        """
        self.namespace = namespace
        self.app_label = app_label
        self.main_app_file = main_app_file
        self.app_startup_command = app_startup_command
        
        # Load kubernetes config
        try:
            config.load_incluster_config()
            logger.info("Loaded in-cluster config")
        except:
            config.load_kube_config()
            logger.info("Loaded local kubeconfig")
            
        self.v1 = client.CoreV1Api()
    
    def get_pods(self):
        """Get running pods matching the label selector"""
        try:
            label_selector = f"app={self.app_label}"
            pods = self.v1.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=label_selector
            )
            
            running_pods = []
            for pod in pods.items:
                if pod.status.phase == "Running":
                    running_pods.append(pod.metadata.name)
            
            return running_pods
        except Exception as e:
            logger.error(f"Failed to get pods: {e}")
            return []
            
    def copy_directory_to_pod(self, pod_name, local_dir_path, remote_parent_path):
        """
        Copy a local directory to a pod, replacing the existing one if it exists.
        
        Args:
            pod_name: Name of the target pod.
            local_dir_path: Path to the local directory to copy (e.g., "./final_model").
            remote_parent_path: The parent directory on the pod where the local directory
                                will be copied (e.g., "/app/"). The directory itself
                                will be created/replaced inside this path.
        """
        local_dir = Path(local_dir_path)
        if not local_dir.is_dir():
            logger.error(f"Local path {local_dir_path} is not a directory or does not exist.")
            return False

        remote_target_path = Path(remote_parent_path) / local_dir.name

        try:
            # 1. Create a temporary tar file of the local directory
            with tempfile.NamedTemporaryFile(suffix='.tar', delete=False) as temp_tar:
                # tar.add(local_dir_path, arcname=local_dir.name) # Add the directory itself, not just its contents
                # For `kubectl cp` behavior, we often want to copy the *contents* of the directory
                # into the target, or ensure the target directory is named correctly.
                # Let's ensure the tar includes the directory itself at the root.
                with tarfile.open(temp_tar.name, 'w') as tar:
                    tar.add(local_dir_path, arcname=local_dir.name)
                
                # 2. Check if the remote directory exists and remove it if it does
                logger.info(f"Checking for existing directory {remote_target_path} in pod {pod_name}...")
                # Use `rm -rf` to ensure it's gone
                remove_command = ['sh', '-c', f'rm -rf {remote_target_path}']
                
                try:
                    resp_rm = stream(
                        self.v1.connect_get_namespaced_pod_exec,
                        pod_name,
                        self.namespace,
                        command=remove_command,
                        stderr=True,
                        stdin=False,
                        stdout=True,
                        tty=False
                    )
                    # We usually don't need to log stdout/stderr of rm -rf unless it fails
                    # logger.info(f"Removal command output for {remote_target_path}:\n{resp_rm}")
                    logger.info(f"Attempted to remove existing {remote_target_path} in {pod_name}.")
                except Exception as e_rm:
                    # This might fail if the directory doesn't exist, which is fine.
                    logger.warning(f"Could not remove existing directory {remote_target_path} in {pod_name}. It might not exist or permissions issue. Error: {e_rm}")

                # 3. Upload the tar file and extract it to the remote_parent_path
                # The tar file contains 'final_model/' at its root.
                # We want to extract it into remote_parent_path (e.g., /app/)
                # so the result is /app/final_model/
                command = [
                    'sh', '-c',
                    f'mkdir -p {remote_parent_path} && tar -xmf - -C {remote_parent_path}'
                ]
                
                with open(temp_tar.name, 'rb') as tar_data:
                    resp = stream(
                        self.v1.connect_get_namespaced_pod_exec,
                        pod_name,
                        self.namespace,
                        command=command,
                        stderr=True,
                        stdin=True,
                        stdout=True,
                        tty=False,
                        _preload_content=False
                    )
                    
                    # Send tar data to stdin
                    resp.write_stdin(tar_data.read())
                    resp.close()
                
                # Clean up temp file
                os.unlink(temp_tar.name)
                
            logger.info(f"✅ Copied directory {local_dir_path} to {pod_name}:{remote_target_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to copy directory {local_dir_path} to {pod_name}: {e}")
            return False

    def execute_command(self, pod_name, command):
        """Execute a command in a pod"""
        try:
            if isinstance(command, str):
                command = ['sh', '-c', command]
            
            resp = stream(
                self.v1.connect_get_namespaced_pod_exec,
                pod_name,
                self.namespace,
                command=command,
                stderr=True,
                stdin=False,
                stdout=True,
                tty=False
            )
            
            return resp
        except Exception as e:
            logger.error(f"Failed to execute command in {pod_name}: {e}")
            return None

    def kill_flask_process(self, pod_name):
        """
        Kill the Flask process in the pod based on self.main_app_file.
        Assumes the container's entrypoint or a supervisor will restart it.
        """
        logger.info(f"Sending SIGTERM to Flask process ({self.main_app_file}) in {pod_name}...")
        
        process_pattern = f"python.*{self.main_app_file}"
        
        # First, check what processes are running
        check_command = "ps aux | grep python"
        result = self.execute_command(pod_name, check_command)
        logger.info(f"Current Python processes:\n{result}")
        
        # Check if the target process is actually running
        if self.main_app_file in result:
            logger.info("Found target Python process. Attempting to kill it...")
            # Try different approaches to kill the process
            commands = [
                f"pkill -TERM -f '{process_pattern}'", # More robust if it finds the exact command
                "kill -TERM 1", # If your app is PID 1, this sends TERM to the main container process
                "pkill -TERM python" # More aggressive, kills all python processes
            ]
            
            for cmd in commands:
                logger.info(f"Executing: {cmd}")
                self.execute_command(pod_name, cmd)
                time.sleep(2) # Give it a moment to terminate
                
                # Check if process is gone
                result = self.execute_command(pod_name, "ps aux | grep python")
                if self.main_app_file not in result:
                    logger.info("✅ Process terminated successfully")
                    return True # Process is gone, likely restarting
            
            logger.warning(f"❌ Could not confirm termination of {self.main_app_file} in {pod_name}. It might not have restarted yet or something is blocking.")
            return False # Process not terminated
        else:
            logger.warning(f"Process pattern '{process_pattern}' not found in {pod_name}. Assuming it's not running or already restarted.")
            # If the process isn't found, it might have already restarted or wasn't running.
            # In this context, we'll consider it "successful" for the kill step.
            return True

    def start_flask_server(self, pod_name):
        """
        Explicitly starts the Flask server if a command is provided.
        Only call this if kill_flask_process doesn't trigger an automatic restart.
        """
        if not self.app_startup_command:
            logger.info("No explicit startup command provided. Assuming application will auto-restart.")
            return True # Consider successful because no action is needed
        
        logger.info(f"Attempting to explicitly start Flask server in {pod_name} using: {self.app_startup_command}")
        resp = self.execute_command(pod_name, self.app_startup_command + " &") # Run in background
        if resp:
            logger.info(f"Startup command output for {pod_name}:\n{resp}")
            logger.info(f"✅ Startup command sent to {pod_name}. Please check pod logs for actual startup.")
            return True
        else:
            logger.error(f"❌ Failed to send startup command to {pod_name}.")
            return False


def main():
    """Main deployment function for the model directory"""
    
    # Configuration
    NAMESPACE = "default"  # Change this to your namespace
    APP_LABEL = "routing-agent-service"  # Change this to match your pod labels
    
    # Define your main application file and startup command (if needed)
    # The `main_app_file` is used to identify the process to kill.
    # The `app_startup_command` is optional. If your Dockerfile's CMD/ENTRYPOINT
    # handles restarting the app upon its termination (e.g., using a loop or an init system),
    # you can leave `app_startup_command=""`.
    MAIN_APP_FILE = "routing_agent_service.py" 
    APP_STARTUP_COMMAND = "" # e.g., "python /app/routing_agent_service.py"
                             # Leave empty if your entrypoint self-restarts.
                             # If your container runs a supervisor like gunicorn,
                             # you might need to restart gunicorn.

    # Directory to deploy
    LOCAL_MODEL_DIR = "./final_model" # This should be the local directory you want to copy
    REMOTE_MODEL_PARENT_PATH = "/app/" # The parent directory on the pod where final_model will reside
                                       # Resulting path will be /app/final_model
    
    # Check if the local model directory exists
    if not Path(LOCAL_MODEL_DIR).is_dir():
        logger.error(f"Error: Local model directory '{LOCAL_MODEL_DIR}' does not exist or is not a directory.")
        logger.error("Please ensure you run this script from the directory containing your 'final_model' folder.")
        sys.exit(1)
            
    # Create deployer
    deployer = K8sModelDeployer(
        namespace=NAMESPACE,
        app_label=APP_LABEL,
        main_app_file=MAIN_APP_FILE,
        app_startup_command=APP_STARTUP_COMMAND
    )
    
    logger.info("🚀 Starting model directory hot deployment...")
    logger.info(f"Namespace: {NAMESPACE}")
    logger.info(f"App Label: {APP_LABEL}")
    logger.info(f"Local model directory: {LOCAL_MODEL_DIR}")
    logger.info(f"Remote model parent path: {REMOTE_MODEL_PARENT_PATH}")
    logger.info(f"Main application file: {MAIN_APP_FILE}")
    
    # Get pods before deployment
    pods = deployer.get_pods()
    if not pods:
        logger.error("No running pods found matching the label selector!")
        sys.exit(1)
    
    # Deploy model directory to each pod
    success_count = 0
    for pod_name in pods:
        logger.info(f"\n🚀 Deploying model to pod: {pod_name}")
        
        # Copy the final_model directory
        if deployer.copy_directory_to_pod(pod_name, LOCAL_MODEL_DIR, REMOTE_MODEL_PARENT_PATH):
            logger.info(f"✅ Model directory copied successfully to {pod_name}")
            
            # Now, trigger application restart within the *same* pod
            if deployer.kill_flask_process(pod_name):
                # If your application doesn't automatically restart after being killed,
                # you might need to uncomment and configure the start_flask_server method below.
                # However, for typical Flask apps in containers, killing the main process
                # should cause the container to restart it or the entrypoint to re-execute.
                # If using a supervisor like Gunicorn, you might need to send a HUP signal or restart Gunicorn.
                
                # If your app *doesn't* self-restart, uncomment this:
                # if APP_STARTUP_COMMAND:
                #     if deployer.start_flask_server(pod_name):
                #         success_count += 1
                #         logger.info(f"✅ Pod {pod_name} application restart initiated (explicit start)")
                #     else:
                #         logger.error(f"❌ Failed to explicitly start server in {pod_name}")
                # else:
                success_count += 1
                logger.info(f"✅ Pod {pod_name} application restart initiated (via process kill)")
            else:
                logger.error(f"❌ Failed to restart application process in {pod_name}")
        else:
            logger.error(f"❌ Failed to copy model directory to {pod_name}")
    
    logger.info(f"\n🎉 Model deployment complete! Success: {success_count}/{len(pods)} pods")
    logger.info("Note: Application processes are restarting - wait a few seconds for them to come back online and load the new model.")
    
    if success_count == len(pods):
        logger.info("🎉 Model hot deployment completed successfully!")
        sys.exit(0)
    else:
        logger.error("💥 Model hot deployment failed for one or more pods!")
        sys.exit(1)


if __name__ == "__main__":
    main()