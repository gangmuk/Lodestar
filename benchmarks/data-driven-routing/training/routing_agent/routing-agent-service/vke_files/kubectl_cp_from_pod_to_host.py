import sys
import os
import subprocess
from typing import Tuple, Optional
from logger import logger

def kubectl_cp_from_pod_to_host(src: str, dst: str, deployment: str, namespace: str, container: Optional[str] = None) -> bool:
    """
    Copy files or directories between a Kubernetes pod and local filesystem using kubectl.
    
    Args:
        src (str): Source file/directory path
        dst (str): Destination file/directory path
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
        
        # Determine if source is local or in pod
        is_src_local = not src.startswith('/') or os.path.exists(src)
        
        # If source is in pod, verify it exists
        if not is_src_local:
            # Check if source exists in the pod
            check_cmd = ['kubectl', 'exec', pod_name, '-n', namespace]
            if container:
                check_cmd.extend(['-c', container])
            check_cmd.extend(['--', 'sh', '-c', f'[ -e "{src}" ] && echo "exists" || echo "not found"'])
            
            check_result = subprocess.run(check_cmd, capture_output=True, text=True)
            if check_result.stdout.strip() != "exists":
                logger.error(f"Source path '{src}' does not exist in the pod")
                return False
            
            # Now check if it's a directory
            check_dir_cmd = ['kubectl', 'exec', pod_name, '-n', namespace]
            if container:
                check_dir_cmd.extend(['-c', container])
            check_dir_cmd.extend(['--', 'sh', '-c', f'[ -d "{src}" ] && echo "directory" || echo "file"'])
            
            check_dir_result = subprocess.run(check_dir_cmd, capture_output=True, text=True)
            is_src_dir = check_dir_result.stdout.strip() == "directory"
            logger.debug(f"Source in pod is a {'directory' if is_src_dir else 'file'}: {src}")
        else:
            # Local source
            if not os.path.exists(src):
                logger.error(f"Source path '{src}' does not exist locally")
                return False
            
            is_src_dir = os.path.isdir(src)
            logger.debug(f"Source locally is a {'directory' if is_src_dir else 'file'}: {src}")
        
        # Ensure destination directory exists for pod-to-local transfers
        if not is_src_local:
            if is_src_dir and not os.path.exists(dst):
                # If source is a directory and destination doesn't exist, create the directory
                os.makedirs(dst, exist_ok=True)
            elif not is_src_dir:
                # If source is a file, ensure destination directory exists
                dst_dir = os.path.dirname(dst)
                if dst_dir:
                    os.makedirs(dst_dir, exist_ok=True)
        
        # Build kubectl cp command
        cp_cmd = ['kubectl', 'cp', '-n', namespace]
        
        # Add container flag if specified
        if container:
            cp_cmd.extend(['-c', container])
        
        # Prepare paths for kubectl cp
        if is_src_local:
            # Local to pod
            dst_with_pod = f"{namespace}/{pod_name}:{dst}"
            cp_cmd.extend([src, dst_with_pod])
            
            # For directories, make sure the destination directory exists in the pod
            if is_src_dir:
                mkdir_cmd = ['kubectl', 'exec', pod_name, '-n', namespace]
                if container:
                    mkdir_cmd.extend(['-c', container])
                mkdir_cmd.extend(['--', 'mkdir', '-p', dst])
                
                mkdir_result = subprocess.run(mkdir_cmd, capture_output=True, text=True)
                if mkdir_result.returncode != 0:
                    logger.error(f"Warning: Could not create destination directory in pod: {mkdir_result.stderr}")
        else:
            # Pod to local
            src_with_pod = f"{namespace}/{pod_name}:{src}"
            cp_cmd.extend([src_with_pod, dst])
        
        logger.debug(f"Running copy command: {' '.join(cp_cmd)}")
        cp_result = subprocess.run(cp_cmd, capture_output=True, text=True)
        
        if cp_result.returncode != 0:
            # Check for specific error messages
            if "No such file or directory" in cp_result.stderr:
                logger.error(f"Copy failed: Source path '{src}' does not exist")
                return False
            logger.error(f"Copy command failed: {cp_result.stderr}")
            return False
            
        if is_src_dir:
            print(f"** {dst}")
            return True
        else:
            print(f"** {dst}")
            return True
            
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode('utf-8') if e.stderr else str(e)
        logger.error(f"!!! ERROR !!! Command failed: {error_msg}")
        return False
    except Exception as e:
        logger.error(f"!!! ERROR !!!: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 5:
        logger.info("Usage: python kubectl_cp.py <src> <dst> <deployment> <namespace> [container]")
        logger.info("  <src>: Source file/directory path (local or in pod)")
        logger.info("  <dst>: Destination file/directory path")
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
    success = kubectl_cp_from_pod_to_host(src, dst, deployment, namespace, container)