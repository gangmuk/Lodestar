#!/usr/bin/env python3
"""
Simple Hot Deploy Script for start.sh setup
Now we can safely restart Flask without killing the pod
"""

import os
import sys
import time
import tempfile
import tarfile
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
        """Copy a file to pod"""
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
    
    def deploy_to_pods(self, file_mappings, restart=True):
        """Deploy files and optionally restart Flask"""
        pods = self.get_pods()
        
        if not pods:
            logger.error(f"No running pods found")
            return False
        
        logger.info(f"Found {len(pods)} running pods: {pods}")
        
        overall_success = True
        for pod_name in pods:
            logger.info(f"\n🚀 Deploying to pod: {pod_name}")
            
            # Copy files
            files_copied = 0
            for local_path, remote_path in file_mappings.items():
                if self.copy_file_to_pod(pod_name, local_path, remote_path):
                    files_copied += 1
                    logger.info(f"Copied {local_path} to {pod_name}:{remote_path}")
                else:
                    logger.error(f"Failed to copy {local_path} to {pod_name}:{remote_path}")
                    logger.error(f"Exiting...")
                    exit()
            
            if files_copied == len(file_mappings):
                logger.info(f"✅ Copied {files_copied} files to {pod_name}")
                
                if restart:
                    if self.restart_flask_safely(pod_name):
                        logger.info(f"✅ Successfully deployed and restarted Flask in {pod_name}")
                    else:
                        logger.error(f"❌ Failed to restart Flask in {pod_name}")
                        overall_success = False
                else:
                    logger.info(f"✅ Files copied to {pod_name} (no restart)")
            else:
                logger.error(f"❌ Only copied {files_copied}/{len(file_mappings)} files to {pod_name}")
                overall_success = False
        
        return overall_success

def main():
    """Main function"""
    
    # Configuration
    NAMESPACE = "default"
    APP_LABEL = "routing-agent-service"
    
    # Files to deploy
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
    missing_files = [f for f in FILES_TO_DEPLOY.keys() if not os.path.exists(f)]
    if missing_files:
        logger.error(f"Missing files: {missing_files}")
        sys.exit(1)
    
    
    restart = True
    
    # Deploy
    deployer = K8sHotDeployer(namespace=NAMESPACE, app_label=APP_LABEL)
    
    logger.info("🚀 Starting deployment...")
    logger.info(f"Files: {list(FILES_TO_DEPLOY.keys())}")
    logger.info(f"Restart Flask: {restart}")
    
    if deployer.deploy_to_pods(FILES_TO_DEPLOY, restart):
        logger.info("\n🎉 Deployment completed successfully!")
        if restart:
            logger.info("Flask has been restarted with new code.")
        else:
            logger.info("Files copied. Flask still running with old code.")
    else:
        logger.error("\n💥 Deployment failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()