import sys
import os
import subprocess
import argparse
import fnmatch
from typing import Tuple, Optional, List
# from logger import logger

def should_skip_file(filename: str, skip_patterns: Optional[List[str]]) -> bool:
    """Check if a file should be skipped based on patterns."""
    if not skip_patterns:
        return False

    for pattern in skip_patterns:
        if fnmatch.fnmatch(filename, pattern):
            return True
    return False

def kubectl_cp_from_pod_to_host(src: str, dst: str, deployment: str, namespace: str, container: Optional[str] = None, skip_files: Optional[List[str]] = None) -> bool:
    """
    Copy files or directories between a Kubernetes pod and local filesystem using kubectl.

    Args:
        src (str): Source file/directory path
        dst (str): Destination file/directory path
        deployment (str): Deployment name
        namespace (str): Kubernetes namespace
        container (str, optional): Container name
        skip_files (list, optional): List of filename patterns to skip

    Returns:
        bool: Success status
    """
    try:
        # Get pod name from deployment using the correct jsonpath format
        cmd = ['kubectl', 'get', 'pods', 
               '-n', namespace, 
               '--selector=app=' + deployment, 
               '-o=jsonpath={.items[0].metadata.name}']
        
        # logger.debug(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        pod_name = result.stdout.strip()
        
        # If pod not found with app label, try listing all pods
        if not pod_name:
            list_cmd = ['kubectl', 'get', 'pods', '-n', namespace]
            list_result = subprocess.run(list_cmd, capture_output=True, text=True)
            print("Error: Pod not found with app label, listing all pods in namespace")
            print(f"Error: Available pods:\n{list_result.stdout}")
            
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
            print(f"Error: No pods found for deployment '{deployment}' in namespace '{namespace}'")
            return False
        
        # Get container name if not specified
        if not container:
            container_cmd = ['kubectl', 'get', 'pod', pod_name, 
                            '-n', namespace, 
                            '-o=jsonpath={.spec.containers[0].name}']
            container_result = subprocess.run(container_cmd, capture_output=True, text=True)
            container = container_result.stdout.strip()

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
                print(f"Error: Source path '{src}' does not exist in the pod")
                return False
            
            # Now check if it's a directory
            check_dir_cmd = ['kubectl', 'exec', pod_name, '-n', namespace]
            if container:
                check_dir_cmd.extend(['-c', container])
            check_dir_cmd.extend(['--', 'sh', '-c', f'[ -d "{src}" ] && echo "directory" || echo "file"'])
            
            check_dir_result = subprocess.run(check_dir_cmd, capture_output=True, text=True)
            is_src_dir = check_dir_result.stdout.strip() == "directory"
        else:
            # Local source
            if not os.path.exists(src):
                print(f"Error: Source path '{src}' does not exist locally")
                return False
            
            is_src_dir = os.path.isdir(src)
        
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
        
        # Handle single file copy with skip check
        if not is_src_dir:
            filename = os.path.basename(src)
            if should_skip_file(filename, skip_files):
                print(f"Skipping file: {filename}")
                return True

            # Build kubectl cp command for single file
            cp_cmd = ['kubectl', 'cp', '-n', namespace]

            # Add container flag if specified
            if container:
                cp_cmd.extend(['-c', container])

            # Prepare paths for kubectl cp
            if is_src_local:
                # Local to pod
                dst_with_pod = f"{namespace}/{pod_name}:{dst}"
                cp_cmd.extend([src, dst_with_pod])
            else:
                # Pod to local
                src_with_pod = f"{namespace}/{pod_name}:{src}"
                cp_cmd.extend([src_with_pod, dst])

            cp_result = subprocess.run(cp_cmd, capture_output=True, text=True)

            # If kubectl cp fails with EOF error, try alternative method using cat
            if cp_result.returncode != 0 and "unexpected EOF" in cp_result.stderr:
                print(f"Warning: kubectl cp failed with EOF error, trying alternative method...")
                try:
                    # Use kubectl exec + cat as fallback
                    cat_cmd = ['kubectl', 'exec', pod_name, '-n', namespace]
                    if container:
                        cat_cmd.extend(['-c', container])
                    cat_cmd.extend(['--', 'cat', src])

                    with open(dst, 'wb') as f:
                        cat_result = subprocess.run(cat_cmd, stdout=f, stderr=subprocess.PIPE)
                        if cat_result.returncode == 0:
                            print(f"Successfully copied file using alternative method: {dst}")
                            return True
                        else:
                            print(f"Error: Alternative copy method also failed: {cat_result.stderr.decode('utf-8')}")
                            return False
                except Exception as e:
                    print(f"Error: Alternative copy method failed with exception: {str(e)}")
                    return False
        else:
            # Handle directory copy with filtering
            success = True

            if is_src_local:
                # Local directory to pod
                # Use os.walk to get all files recursively
                try:
                    all_files = []
                    for root, dirs, files in os.walk(src):
                        for file in files:
                            full_path = os.path.join(root, file)
                            all_files.append(full_path)
                except OSError as e:
                    print(f"Error: Could not list local directory '{src}': {e}")
                    return False

                # Create destination directory in pod
                mkdir_cmd = ['kubectl', 'exec', pod_name, '-n', namespace]
                if container:
                    mkdir_cmd.extend(['-c', container])
                mkdir_cmd.extend(['--', 'mkdir', '-p', dst])

                mkdir_result = subprocess.run(mkdir_cmd, capture_output=True, text=True)
                if mkdir_result.returncode != 0:
                    print(f"Warning: Could not create destination directory in pod: {mkdir_result.stderr}")

                # Copy each file individually, skipping as needed
                for file_path in all_files:
                    filename = os.path.basename(file_path)

                    if should_skip_file(filename, skip_files):
                        # Get relative path for display
                        relative_path = os.path.relpath(file_path, src)
                        print(f"Skipping file: {relative_path}")
                        continue

                    # Get relative path from source directory
                    relative_path = os.path.relpath(file_path, src)
                    item_dst = os.path.join(dst, relative_path)

                    # Create destination directory in pod
                    dst_dir_cmd = ['kubectl', 'exec', pod_name, '-n', namespace]
                    if container:
                        dst_dir_cmd.extend(['-c', container])
                    dst_dir_cmd.extend(['--', 'mkdir', '-p', os.path.dirname(item_dst)])

                    mkdir_result = subprocess.run(dst_dir_cmd, capture_output=True, text=True)

                    cp_cmd = ['kubectl', 'cp', '-n', namespace]
                    if container:
                        cp_cmd.extend(['-c', container])

                    dst_with_pod = f"{namespace}/{pod_name}:{item_dst}"
                    cp_cmd.extend([file_path, dst_with_pod])

                    cp_result = subprocess.run(cp_cmd, capture_output=True, text=True)
                    if cp_result.returncode != 0:
                        print(f"Error: Failed to copy '{relative_path}': {cp_result.stderr}")
                        success = False
            else:
                # Pod directory to local
                # Use find to get all files recursively
                find_cmd = ['kubectl', 'exec', pod_name, '-n', namespace]
                if container:
                    find_cmd.extend(['-c', container])
                find_cmd.extend(['--', 'find', src, '-type', 'f'])

                find_result = subprocess.run(find_cmd, capture_output=True, text=True)
                if find_result.returncode != 0:
                    print(f"Error: Could not list pod directory '{src}': {find_result.stderr}")
                    return False

                all_files = find_result.stdout.strip().split('\n') if find_result.stdout.strip() else []

                # Copy each file individually, skipping as needed
                for file_path in all_files:
                    if not file_path:
                        continue

                    # Get relative path from source directory
                    if file_path.startswith(src):
                        relative_path = file_path[len(src):].lstrip('/')
                    else:
                        relative_path = file_path

                    filename = os.path.basename(file_path)

                    if should_skip_file(filename, skip_files):
                        print(f"Skipping file: {relative_path}")
                        continue

                    # Create destination path
                    dst_file = os.path.join(dst, relative_path)
                    dst_dir = os.path.dirname(dst_file)

                    # Ensure destination directory exists
                    if dst_dir and not os.path.exists(dst_dir):
                        os.makedirs(dst_dir, exist_ok=True)

                    cp_cmd = ['kubectl', 'cp', '-n', namespace]
                    if container:
                        cp_cmd.extend(['-c', container])

                    src_with_pod = f"{namespace}/{pod_name}:{file_path}"
                    cp_cmd.extend([src_with_pod, dst_file])

                    cp_result = subprocess.run(cp_cmd, capture_output=True, text=True)
                    if cp_result.returncode != 0:
                        print(f"Error: Failed to copy '{relative_path}': {cp_result.stderr}")
                        success = False

            cp_result = None  # Set to None for directory case
        
        # Handle results
        if not is_src_dir:
            # Single file copy result
            if cp_result.returncode != 0:
                # Check for specific error messages
                if "No such file or directory" in cp_result.stderr:
                    print(f"Error: Copy failed: Source path '{src}' does not exist")
                    return False
                print(f"Error: Copy command failed: {cp_result.stderr}")
                return False

            print(f"** {dst}")
            return True
        else:
            # Directory copy result (handled above)
            if success:
                print(f"** {dst}")
                return True
            else:
                print(f"Error: Some files failed to copy from directory '{src}'")
                return False
            
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode('utf-8') if e.stderr else str(e)
        print(f"ERROR, Command failed: {error_msg}")
        return False
    except Exception as e:
        print(f"ERROR, Command failed: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Copy files or directories between a Kubernetes pod and local filesystem using kubectl.')
    parser.add_argument('src', help='Source file/directory path (local or in pod)')
    parser.add_argument('dst', help='Destination file/directory path')
    parser.add_argument('deployment', help='Kubernetes deployment name')
    parser.add_argument('namespace', help='Kubernetes namespace')
    parser.add_argument('-c', '--container', help='Container name (defaults to first container)')
    parser.add_argument('-s', '--skip-files', nargs='*', help='Filename patterns to skip (supports wildcards)')

    args = parser.parse_args()

    success = kubectl_cp_from_pod_to_host(
        args.src,
        args.dst,
        args.deployment,
        args.namespace,
        args.container,
        args.skip_files
    )