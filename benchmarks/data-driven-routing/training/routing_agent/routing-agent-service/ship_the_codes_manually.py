#!/usr/bin/env python3
"""
Kubernetes Hot Deploy Script - Fixed Version
Copies Python files to running pods and restarts the Flask server
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

class K8sHotDeployer:
    def __init__(self, namespace="default", app_label="your-app-label"):
        """
        Initialize the deployer
        
        Args:
            namespace: Kubernetes namespace
            app_label: Label selector to find your pods
        """
        self.namespace = namespace
        self.app_label = app_label
        
        # Load kubernetes config
        try:
            config.load_incluster_config()
            logger.info("Loaded in-cluster config")
        except config.ConfigException:
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
    
    def copy_file_to_pod(self, pod_name, local_path, remote_path):
        """
        Copy a single file to a pod using kubectl cp equivalent
        """
        try:
            local_file = Path(local_path)
            if not local_file.exists():
                logger.error(f"Local file {local_path} does not exist")
                return False
            
            # Create a temporary tar file
            with tempfile.NamedTemporaryFile(suffix='.tar', delete=False) as temp_tar:
                with tarfile.open(temp_tar.name, 'w') as tar:
                    tar.add(local_path, arcname=os.path.basename(local_path))
                
                # Upload the tar file and extract it
                command = [
                    'sh', '-c',
                    f'tar -xmf - -C {os.path.dirname(remote_path)}'
                ]
                
                # Connect to the pod's exec stream
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
                with open(temp_tar.name, 'rb') as tar_data:
                    resp.write_stdin(tar_data.read())
                resp.close()
                
                # Read stdout and stderr to check for errors
                stdout_data = resp.read_stdout()
                stderr_data = resp.read_stderr()

                if stdout_data:
                    logger.debug(f"stdout from copy: {stdout_data}")
                if stderr_data:
                    logger.error(f"stderr from copy: {stderr_data}")
                    return False

            # Clean up temp file
            os.unlink(temp_tar.name)
                
            logger.info(f"✅ Copied {local_path} to {pod_name}:{remote_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to copy {local_path} to {pod_name}: {e}")
            return False
    
    def copy_files_to_pod(self, pod_name, file_mappings):
        """
        Copy multiple files to a pod
        
        Args:
            pod_name: Name of the target pod
            file_mappings: Dict of {local_path: remote_path}
        """
        success_count = 0
        for local_path, remote_path in file_mappings.items():
            if self.copy_file_to_pod(pod_name, local_path, remote_path):
                success_count += 1
        
        logger.info(f"Copied {success_count}/{len(file_mappings)} files to {pod_name}")
        return success_count == len(file_mappings)
    
    def execute_command(self, pod_name, command):
        """Execute a command in a pod and return the output as a string"""
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
            
            # Handle both string and stream object responses
            if isinstance(resp, str):
                return resp
            else:
                try:
                    return resp.read_stdout()
                except AttributeError:
                    logger.error(f"Unexpected response type from execute_command: {type(resp)}")
                    return ""
            
        except Exception as e:
            logger.error(f"Failed to execute command in {pod_name}: {e}")
            return ""
    
    def kill_flask_process(self, pod_name, process_pattern="python.*routing_agent_service.py"):
        """Kill the Flask process in the pod"""
        logger.info(f"Sending SIGTERM to Flask process in {pod_name}...")
        
        # First, check what processes are running
        check_command = "ps aux | grep python"
        result_str = self.execute_command(pod_name, check_command)
        
        logger.info(f"Current Python processes:\n{result_str}")
        
        # The process pattern matching needs to be fixed - look for the actual process
        # From your logs, the process is "python routing_agent_service.py" (not python.*routing_agent_service.py)
        if "python routing_agent_service.py" in result_str:
            logger.info(f"Found Flask process. Attempting to terminate...")
            
            # Try different approaches to kill the process
            commands = [
                "pkill -TERM -f 'routing_agent_service.py'",
                "kill -TERM 1",  # PID 1 is your Flask process
                "pkill -TERM python"
            ]
            
            process_terminated = False
            for cmd in commands:
                logger.info(f"Executing: {cmd}")
                kill_result = self.execute_command(pod_name, cmd)
                if kill_result:
                    logger.debug(f"Kill command output: {kill_result}")

                time.sleep(2)
                
                # Check if process is gone
                result_after_kill = self.execute_command(pod_name, "ps aux | grep python")
                
                if "python routing_agent_service.py" not in result_after_kill:
                    logger.info("✅ Process terminated successfully")
                    process_terminated = True
                    break
            
            if not process_terminated:
                logger.warning(f"❌ Failed to terminate Flask process in {pod_name}")
                return False
        else:
            logger.info(f"Flask process not found or already terminated.")
            
        logger.info(f"✅ Flask process handling completed for {pod_name}")
        return True
    
    def start_flask_server(self, pod_name, command="python /app/routing_agent_service.py"):
        """Start the Flask process in the pod"""
        logger.info(f"Starting Flask process in {pod_name} with command: '{command}'...")
        try:
            # Create logs directory if it doesn't exist
            self.execute_command(pod_name, "mkdir -p /app/logs")
            
            # Start the process in background
            start_command = f"nohup {command} > /app/logs/flask_app.log 2>&1 &"
            result = self.execute_command(pod_name, start_command)
            
            if result:
                logger.debug(f"Start command output: {result}")

            # Give it a moment to start
            time.sleep(5)
            
            # Verify the process is running
            check_result = self.execute_command(pod_name, "ps aux | grep python")
            if "python /app/routing_agent_service.py" in check_result or "routing_agent_service.py" in check_result:
                logger.info(f"✅ Flask server started successfully in {pod_name}")
                return True
            else:
                logger.error(f"❌ Flask server may not have started properly in {pod_name}")
                logger.info(f"Process check result: {check_result}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to start Flask server in {pod_name}: {e}")
            return False

    def deploy_to_pods(self, file_mappings, restart_server=True):
        """
        Deploy files to all matching pods
        
        Args:
            file_mappings: Dict of {local_path: remote_path}
            restart_server: Whether to restart the Flask server
        """
        pods = self.get_pods()
        
        if not pods:
            logger.error("No running pods found for label selector 'app=%s' in namespace '%s'!", self.app_label, self.namespace)
            return False
        
        logger.info(f"Found {len(pods)} running pods: {pods}")
        
        overall_success = True
        for pod_name in pods:
            logger.info(f"\n🚀 Deploying to pod: {pod_name}")
            
            # Copy files
            if self.copy_files_to_pod(pod_name, file_mappings):
                if restart_server:
                    # Kill existing Flask process
                    if self.kill_flask_process(pod_name):
                        # Start new Flask server
                        if self.start_flask_server(pod_name):
                            logger.info(f"✅ Successfully deployed and restarted server in {pod_name}")
                        else:
                            logger.error(f"❌ Failed to restart server in {pod_name}")
                            overall_success = False
                    else:
                        logger.error(f"❌ Failed to terminate existing process in {pod_name}")
                        overall_success = False
                else:
                    logger.info(f"✅ Files copied to {pod_name} (no server restart requested)")
            else:
                logger.error(f"❌ Failed to copy files to {pod_name}")
                overall_success = False
        
        return overall_success
    
    def debug_pod_status(self, pod_name):
        """Debug what's running in the pod"""
        logger.info(f"🔍 Debugging pod {pod_name}...")
        
        debug_commands = {
            "All processes": "ps aux",
            "Python processes": "ps aux | grep python",
            "Port 8080 usage": "netstat -tulpn | grep :8080 || ss -tulpn | grep :8080 || echo 'Port 8080 not in use'",
            "Working directory": "pwd",
            "App directory contents": "ls -la /app/",
            "Recent logs": "ls -la /app/logs/ && tail -20 /app/logs/flask_app.log 2>/dev/null || echo 'No logs found'",
            "Environment": "env | grep -E '(PORT|MODEL|FLASK)'"
        }
        
        for desc, cmd in debug_commands.items():
            logger.info(f"\n--- {desc} ---")
            result = self.execute_command(pod_name, cmd)
            logger.info(f"{result}")


def debug_mode():
    """Run in debug mode to troubleshoot"""
    deployer = K8sHotDeployer(namespace="default", app_label="routing-agent-service")
    pods = deployer.get_pods()
    
    if not pods:
        logger.error("No pods found!")
        return
    
    pod_name = pods[0]
    logger.info(f"Debugging pod: {pod_name}")
    
    # Debug current status
    deployer.debug_pod_status(pod_name)
    
    # Try manual restart
    logger.info("\n🔄 Attempting manual restart of Flask server...")
    deployer.kill_flask_process(pod_name)
    deployer.start_flask_server(pod_name)
    
    # Debug after restart
    logger.info("\n🔍 Status after restart:")
    deployer.debug_pod_status(pod_name)


def main():
    """Main deployment function"""
    
    # Configuration
    NAMESPACE = "default"
    APP_LABEL = "routing-agent-service"
    
    # Files to deploy (local_path: remote_path)
    FILES_TO_DEPLOY = {
        "./routing_agent_service.py": "/app/routing_agent_service.py",
        "./preprocess.py": "/app/preprocess.py",
        "./feature_normalization.py": "/app/feature_normalization.py",
        "./encoding.py": "/app/encoding.py",
        "./simpler_contextual_bandit.py": "/app/simpler_contextual_bandit.py",
        "./contextual_bandit.py": "/app/contextual_bandit.py",
        "./logger.py": "/app/logger.py",
    }
    
    # Check if files exist
    missing_files = []
    for local_path in FILES_TO_DEPLOY.keys():
        if not os.path.exists(local_path):
            missing_files.append(local_path)
    
    if missing_files:
        logger.error(f"Missing local files: {missing_files}")
        logger.error("Please run this script from the directory containing your Python files")
        sys.exit(1)
    
    # Create deployer and deploy
    deployer = K8sHotDeployer(namespace=NAMESPACE, app_label=APP_LABEL)
    
    logger.info("🚀 Starting hot deployment...")
    logger.info(f"Namespace: {NAMESPACE}")
    logger.info(f"App Label: {APP_LABEL}")
    logger.info(f"Files to deploy: {list(FILES_TO_DEPLOY.keys())}")
    
    if deployer.deploy_to_pods(FILES_TO_DEPLOY, restart_server=True):
        logger.info("\n🎉 Hot deployment completed successfully for all targeted pods!")
        logger.info("Note: Applications in pods are restarting - wait a few seconds for them to come back online")
        sys.exit(0)
    else:
        logger.error("\n💥 Hot deployment failed for one or more pods!")
        sys.exit(1)


if __name__ == "__main__":
    # Check if debug mode is requested
    if len(sys.argv) > 1 and sys.argv[1] == "debug":
        debug_mode()
    else:
        main()