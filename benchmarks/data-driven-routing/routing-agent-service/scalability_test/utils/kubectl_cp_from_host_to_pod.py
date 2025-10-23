import sys
import os
import subprocess
from typing import Tuple, Optional
from logger import logger

def kubectl_cp_from_host_to_pod(src: str, dst: str, deployment: str, namespace: str, container: Optional[str] = None) -> Tuple[bool, str]:
    """
    Copy files or directories from local filesystem to a Kubernetes pod using kubectl.
    
    Args:
        src (str): Source file/directory path on local filesystem
        dst (str): Destination file/directory path in pod
        deployment (str): Deployment name
        namespace (str): Kubernetes namespace
        container (str, optional): Container name
    
    Returns:
        Tuple[bool, str]: (Success status, Output message)
    """
    try:
        # Get pod name from deployment using the correct jsonpath format
        cmd = ['kubectl', 'get', 'pods', 
               '-n', namespace, 
               '--selector=app=' + deployment, 
               '-o=jsonpath={.items[0].metadata.name}']
        
        logger.debug(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        pod_name = result.stdout.strip()
        
        # If pod not found with app label, try listing all pods
        if not pod_name:
            logger.error("Pod not found with app label, listing all pods in namespace")
            list_cmd = ['kubectl', 'get', 'pods', '-n', namespace]
            list_result = subprocess.run(list_cmd, capture_output=True, text=True)
            logger.info(f"Available pods:\n{list_result.stdout}")
            
            # Try getting pod by deployment name prefix
            pod_cmd = ['kubectl', 'get', 'pods', 
                      '-n', namespace, 
                      '-o=jsonpath={.items[*].metadata.name}']
            pod_result = subprocess.run(pod_cmd, capture_output=True, text=True)
            
            all_pods = pod_result.stdout.strip().split()
            for pod in all_pods:
                if pod.startswith(deployment):
                    pod_name = pod
                    break
        
        if not pod_name:
            return False, f"No pods found for deployment '{deployment}' in namespace '{namespace}'"
        
        logger.debug(f"Found pod: {pod_name}")
        
        # Get container name if not specified
        if not container:
            container_cmd = ['kubectl', 'get', 'pod', pod_name, 
                            '-n', namespace, 
                            '-o=jsonpath={.spec.containers[0].name}']
            container_result = subprocess.run(container_cmd, capture_output=True, text=True)
            container = container_result.stdout.strip()
            logger.debug(f"Using container: {container}")
        
        # Verify source exists locally
        if not os.path.exists(src):
            return False, f"Source path '{src}' does not exist locally"
        
        is_src_dir = os.path.isdir(src)
        logger.debug(f"Source locally is a {'directory' if is_src_dir else 'file'}: {src}")
        
        # Ensure destination directory exists in pod for file transfers
        if not is_src_dir:
            # If source is a file, ensure destination directory exists in pod
            dst_dir = os.path.dirname(dst)
            if dst_dir:
                mkdir_cmd = ['kubectl', 'exec', pod_name, '-n', namespace]
                if container:
                    mkdir_cmd.extend(['-c', container])
                mkdir_cmd.extend(['--', 'mkdir', '-p', dst_dir])
                
                mkdir_result = subprocess.run(mkdir_cmd, capture_output=True, text=True)
                if mkdir_result.returncode != 0:
                    logger.error(f"Warning: Could not create destination directory in pod: {mkdir_result.stderr}")
        else:
            # If source is a directory, ensure destination directory exists in pod
            mkdir_cmd = ['kubectl', 'exec', pod_name, '-n', namespace]
            if container:
                mkdir_cmd.extend(['-c', container])
            mkdir_cmd.extend(['--', 'mkdir', '-p', dst])
            
            mkdir_result = subprocess.run(mkdir_cmd, capture_output=True, text=True)
            if mkdir_result.returncode != 0:
                logger.error(f"Warning: Could not create destination directory in pod: {mkdir_result.stderr}")
        
        # Build kubectl cp command (host to pod)
        cp_cmd = ['kubectl', 'cp', '-n', namespace]
        
        # Add container flag if specified
        if container:
            cp_cmd.extend(['-c', container])
        
        # Prepare paths for kubectl cp (local to pod)
        dst_with_pod = f"{namespace}/{pod_name}:{dst}"
        cp_cmd.extend([src, dst_with_pod])
        
        logger.debug(f"Running copy command: {' '.join(cp_cmd)}")
        cp_result = subprocess.run(cp_cmd, capture_output=True, text=True)
        
        if cp_result.returncode != 0:
            # Check for specific error messages
            if "No such file or directory" in cp_result.stderr:
                return False, f"Copy failed: Source path '{src}' does not exist locally"
            return False, f"Copy command failed: {cp_result.stderr}"
            
        if is_src_dir:
            return True, f"Copied directory from {src} to {dst} in pod"
        else:
            return True, f"Copied file from {src} to {dst} in pod"
            
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode('utf-8') if e.stderr else str(e)
        return False, f"!!! ERROR !!! Command failed: {error_msg}"
    except Exception as e:
        return False, f"!!! ERROR !!!: {str(e)}"

if __name__ == "__main__":
    if len(sys.argv) < 5:
        logger.info("Usage: python kubectl_cp_host_to_pod.py <src> <dst> <deployment> <namespace> [container]")
        logger.info("  <src>: Source file/directory path on local filesystem")
        logger.info("  <dst>: Destination file/directory path in pod")
        logger.info("  <deployment>: Kubernetes deployment name")
        logger.info("  <namespace>: Kubernetes namespace")
        logger.info("  [container]: Optional container name (defaults to first container)")
        sys.exit(1)
    
    src = sys.argv[1]
    dst = sys.argv[2]
    deployment = sys.argv[3]
    namespace = sys.argv[4]
    container = sys.argv[5] if len(sys.argv) > 5 else None
    
    logger.debug(f"Copying from {src} to {dst} in deployment {deployment} in namespace {namespace}")
    success, output = kubectl_cp_from_host_to_pod(src, dst, deployment, namespace, container)
    
    if success:
        logger.info(f"Success: {output}")
    else:
        logger.info(f"!!! ERROR !!!: {output}")