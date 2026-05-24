#!/usr/bin/env python3
"""
Script to update vLLM command-line arguments in Kubernetes deployments.
Handles quantization, chunked-prefill, and prefix-caching flags.
"""

import argparse
import sys
import subprocess
import json


def run_kubectl(cmd):
    """Run kubectl command and return result"""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"kubectl command failed: {' '.join(cmd)}")
        print(f"Error: {e.stderr}")
        sys.exit(1)


def get_current_command(deployment_name, namespace, container_name):
    """Get the current command list from the container"""
    cmd = [
        'kubectl', 'get', 'deployment', deployment_name, '-n', namespace,
        '-o', f'jsonpath={{.spec.template.spec.containers[?(@.name=="{container_name}")].command}}'
    ]
    result = run_kubectl(cmd)
    if result:
        return json.loads(result)
    return []


def update_command_args(current_command, enable_quantization, quantization_method,
                        enable_chunked_prefill, enable_prefix_caching):
    """
    Update command list with the specified vLLM arguments.

    Args:
        current_command: List of current command arguments
        enable_quantization: 0 or 1 to disable/enable quantization
        quantization_method: Quantization method (e.g., "bitsandbytes")
        enable_chunked_prefill: 0 or 1 to disable/enable chunked prefill
        enable_prefix_caching: 0 or 1 to disable/enable prefix caching

    Returns:
        Updated command list
    """
    # Remove existing flags first
    new_command = []
    skip_next = False

    for i, arg in enumerate(current_command):
        if skip_next:
            skip_next = False
            continue

        # Skip quantization-related args
        if arg == '--quantization':
            skip_next = True  # Skip the value too
            continue

        # Skip boolean flags we're managing
        if arg in ('--enable-chunked-prefill', '--enable-prefix-caching',
                   '--no-enable-chunked-prefill', '--no-enable-prefix-caching'):
            continue

        new_command.append(arg)

    # Add the flags based on settings
    if enable_quantization:
        new_command.append('--quantization')
        new_command.append(quantization_method)

    if enable_chunked_prefill:
        new_command.append('--enable-chunked-prefill')

    if enable_prefix_caching:
        new_command.append('--enable-prefix-caching')

    return new_command


def update_deployment_command(deployment_name, namespace, container_name, new_command):
    """
    Update the container command in the deployment using kubectl patch.
    """
    # Build the JSON patch
    patch = {
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": container_name,
                            "command": new_command
                        }
                    ]
                }
            }
        }
    }

    patch_json = json.dumps(patch)

    cmd = [
        'kubectl', 'patch', 'deployment', deployment_name,
        '-n', namespace,
        '--type', 'strategic',
        '-p', patch_json
    ]

    run_kubectl(cmd)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Update vLLM command-line arguments in Kubernetes deployments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Enable quantization with bitsandbytes
  python3 update_vllm_args.py --enable-quantization 1 --quantization-method bitsandbytes

  # Disable prefix caching
  python3 update_vllm_args.py --enable-prefix-caching 0

  # Full configuration
  python3 update_vllm_args.py --enable-quantization 1 --quantization-method bitsandbytes \\
      --enable-chunked-prefill 1 --enable-prefix-caching 1

  # Specify custom deployment
  python3 update_vllm_args.py --deployment my-model --namespace default \\
      --enable-quantization 0 --enable-chunked-prefill 1
        """
    )
    parser.add_argument(
        "--deployment",
        type=str,
        default="llama-3-8b-instruct",
        help="Name of the deployment to update (default: llama-3-8b-instruct)"
    )
    parser.add_argument(
        "--container",
        type=str,
        default="vllm-openai",
        help="Name of the container to update (default: vllm-openai)"
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="default",
        help="Kubernetes namespace (default: default)"
    )
    parser.add_argument(
        "--enable-quantization",
        type=int,
        choices=[0, 1],
        default=None,
        help="Enable quantization (0=disable, 1=enable)"
    )
    parser.add_argument(
        "--quantization-method",
        type=str,
        default="bitsandbytes",
        help="Quantization method (default: bitsandbytes)"
    )
    parser.add_argument(
        "--enable-chunked-prefill",
        type=int,
        choices=[0, 1],
        default=None,
        help="Enable chunked prefill (0=disable, 1=enable)"
    )
    parser.add_argument(
        "--enable-prefix-caching",
        type=int,
        choices=[0, 1],
        default=None,
        help="Enable prefix caching (0=disable, 1=enable)"
    )

    args = parser.parse_args()

    # Check if any arguments need to be updated
    if (args.enable_quantization is None and
        args.enable_chunked_prefill is None and
        args.enable_prefix_caching is None):
        print("No vLLM arguments specified to update. Use --help for usage.")
        sys.exit(0)

    # Get current command
    print(f"Getting current command from deployment '{args.deployment}'...")
    current_command = get_current_command(args.deployment, args.namespace, args.container)

    if not current_command:
        print(f"Error: Could not get command from container '{args.container}' in deployment '{args.deployment}'")
        sys.exit(1)

    # Determine current state for any unspecified args (preserve existing)
    # Check what's currently in the command
    has_quantization = '--quantization' in current_command
    has_chunked_prefill = '--enable-chunked-prefill' in current_command
    has_prefix_caching = '--enable-prefix-caching' in current_command

    # Get current quantization method if exists
    current_quant_method = None
    for i, arg in enumerate(current_command):
        if arg == '--quantization' and i + 1 < len(current_command):
            current_quant_method = current_command[i + 1]
            break

    # Use specified values or preserve current state
    enable_quantization = args.enable_quantization if args.enable_quantization is not None else (1 if has_quantization else 0)
    quantization_method = args.quantization_method if args.enable_quantization == 1 else (current_quant_method or args.quantization_method)
    enable_chunked_prefill = args.enable_chunked_prefill if args.enable_chunked_prefill is not None else (1 if has_chunked_prefill else 0)
    enable_prefix_caching = args.enable_prefix_caching if args.enable_prefix_caching is not None else (1 if has_prefix_caching else 0)

    print(f"Configuration:")
    print(f"  Quantization: {'enabled (' + quantization_method + ')' if enable_quantization else 'disabled'}")
    print(f"  Chunked Prefill: {'enabled' if enable_chunked_prefill else 'disabled'}")
    print(f"  Prefix Caching: {'enabled' if enable_prefix_caching else 'disabled'}")

    # Update command
    new_command = update_command_args(
        current_command,
        enable_quantization,
        quantization_method,
        enable_chunked_prefill,
        enable_prefix_caching
    )

    # Check if anything changed
    if new_command == current_command:
        print("No changes needed - command already matches desired configuration.")
        sys.exit(0)

    print(f"Updating deployment '{args.deployment}'...")
    success = update_deployment_command(args.deployment, args.namespace, args.container, new_command)

    if success:
        print("vLLM args updated successfully!")
        sys.exit(0)
    else:
        print("Update failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
