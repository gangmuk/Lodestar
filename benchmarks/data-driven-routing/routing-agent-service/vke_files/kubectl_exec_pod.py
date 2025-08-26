import sys
import os
import subprocess
from typing import Tuple, Optional, List
from logger import logger

def kubectl_exec_in_pod(command: List[str], deployment: str, namespace: str, container: Optional[str] = None, interactive: bool = False) -> Tuple[bool, str]:
    """
    Execute a command in a Kubernetes pod using kubectl exec.
    
    Args:
        command (List[str]): Command and arguments to execute in the pod
        deployment (str): Deployment name
        namespace (str): Kubernetes namespace
        container (str, optional): Container name
        interactive (bool): Whether to run in interactive mode (-it flags)
    
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
        
        # Build kubectl exec command
        exec_cmd = ['kubectl', 'exec']
        
        # Add interactive flags if requested
        if interactive:
            exec_cmd.extend(['-it'])
        
        # Add pod and namespace
        exec_cmd.extend([pod_name, '-n', namespace])
        
        # Add container flag if specified
        if container:
            exec_cmd.extend(['-c', container])
        
        # Add separator and the actual command
        exec_cmd.append('--')
        exec_cmd.extend(command)
        
        logger.debug(f"Running exec command: {' '.join(exec_cmd)}")
        
        # Execute the command
        if interactive:
            # For interactive mode, don't capture output to allow user interaction
            exec_result = subprocess.run(exec_cmd)
            if exec_result.returncode == 0:
                return True, f"Interactive command executed successfully in pod {pod_name}"
            else:
                return False, f"Interactive command failed with exit code {exec_result.returncode}"
        else:
            # For non-interactive mode, capture output
            exec_result = subprocess.run(exec_cmd, capture_output=True, text=True)
            
            if exec_result.returncode != 0:
                error_output = exec_result.stderr.strip() if exec_result.stderr else "Command failed"
                return False, f"Command failed with exit code {exec_result.returncode}: {error_output}"
            
            output = exec_result.stdout.strip() if exec_result.stdout else "Command executed successfully (no output)"
            return True, f"Command output:\n{output}"
            
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode('utf-8') if e.stderr else str(e)
        return False, f"!!! ERROR !!! Command failed: {error_msg}"
    except Exception as e:
        return False, f"!!! ERROR !!!: {str(e)}"

if __name__ == "__main__":
    if len(sys.argv) < 4:
        logger.info("Usage: python kubectl_exec_pod.py <deployment> <namespace> <command> [args...] [--container=<container>] [--interactive]")
        logger.info("  <deployment>: Kubernetes deployment name")
        logger.info("  <namespace>: Kubernetes namespace")
        logger.info("  <command>: Command to execute in the pod")
        logger.info("  [args...]: Optional command arguments")
        logger.info("  [--container=<container>]: Optional container name (defaults to first container)")
        logger.info("  [--interactive]: Run in interactive mode (-it flags)")
        logger.info("")
        logger.info("Examples:")
        logger.info("  python kubectl_exec_pod.py myapp default ls -la")
        logger.info("  python kubectl_exec_pod.py myapp default bash --interactive")
        logger.info("  python kubectl_exec_pod.py myapp default cat /etc/hostname --container=sidecar")
        sys.exit(1)
    
    # Parse arguments
    deployment = sys.argv[1]
    namespace = sys.argv[2]
    
    # Parse remaining arguments for command, container, and interactive flag
    remaining_args = sys.argv[3:]
    container = None
    interactive = False
    command_args = []
    
    i = 0
    while i < len(remaining_args):
        arg = remaining_args[i]
        if arg.startswith('--container='):
            container = arg.split('=', 1)[1]
        elif arg == '--interactive':
            interactive = True
        else:
            command_args.append(arg)
        i += 1
    
    if not command_args:
        logger.error("No command specified")
        sys.exit(1)
    
    logger.debug(f"Executing command {' '.join(command_args)} in deployment {deployment} in namespace {namespace}")
    if container:
        logger.debug(f"Using container: {container}")
    if interactive:
        logger.debug("Running in interactive mode")
    
    success, output = kubectl_exec_in_pod(command_args, deployment, namespace, container, interactive)
    
    if success:
        logger.info(f"Success: {output}")
    else:
        logger.error(f"!!! ERROR !!!: {output}")
        sys.exit(1)