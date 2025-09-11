#!/usr/bin/env python3
"""
Script to update environment variables in Kubernetes deployments.
Uses kubectl directly to avoid context confusion issues.
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
        print(f"❌ kubectl command failed: {' '.join(cmd)}")
        print(f"Error: {e.stderr}")
        sys.exit(1)


def update_deployment_env_vars(deployment_name, namespace, container_name, env_vars):
    """
    Update environment variables in a Kubernetes deployment using kubectl.
    
    Args:
        deployment_name (str): Name of the deployment to update
        namespace (str): Kubernetes namespace
        container_name (str): Name of the container to update
        env_vars (dict): Dictionary of environment variable key-value pairs
    """
    try:
        # Get current context
        current_context = run_kubectl(['kubectl', 'config', 'current-context'])
        print(f"🎯 Using kubectl context: {current_context}")
        
        # Check if deployment exists
        print(f"🔍 Checking deployment '{deployment_name}' in namespace '{namespace}'...")
        run_kubectl(['kubectl', 'get', 'deployment', deployment_name, '-n', namespace])
        print("✅ Deployment found")
        
        # Get current env vars to show before
        print(f"📋 Current environment variables for container '{container_name}':")
        try:
            env_output = run_kubectl([
                'kubectl', 'get', 'deployment', deployment_name, '-n', namespace,
                '-o', f'jsonpath={{.spec.template.spec.containers[?(@.name=="{container_name}")].env[*]}}'
            ])
            
            if env_output:
                env_data = json.loads(env_output.replace("'", '"'))
                if isinstance(env_data, list):
                    for env_var in env_data:
                        print(f"  {env_var['name']}={env_var['value']}")
                else:
                    print(f"  {env_data['name']}={env_data['value']}")
            else:
                print("  (no environment variables)")
        except:
            print("  (could not parse current env vars)")
        
        print("-" * 50)
        
        # Set each environment variable
        for env_key, env_value in env_vars.items():
            print(f"🔧 Setting {env_key}={env_value}...")
            run_kubectl([
                'kubectl', 'set', 'env', f'deployment/{deployment_name}',
                '-n', namespace,
                f'{env_key}={env_value}',
                f'--containers={container_name}'
            ])
            
            # Verify it was set
            print(f"🔍 Verifying {env_key} was set...")
            result = run_kubectl([
                'kubectl', 'get', 'deployment', deployment_name, '-n', namespace,
                '-o', f'jsonpath={{.spec.template.spec.containers[?(@.name=="{container_name}")].env[?(@.name=="{env_key}")].value}}'
            ])
            
            if result == env_value:
                print(f"✅ SUCCESS: {env_key}={result}")
            else:
                print(f"❌ FAILED: Expected {env_value}, got {result}")
                return False
        
        # Show rollout status
        print("🚀 Watching rollout...")
        subprocess.run(['kubectl', 'rollout', 'status', f'deployment/{deployment_name}', '-n', namespace])
        
        return True
        
    except Exception as e:
        print(f"💥 ERROR: {e}")
        return False


def parse_env_vars(env_strings):
    """
    Parse environment variable strings in the format KEY=VALUE.
    
    Args:
        env_strings (list): List of strings in KEY=VALUE format
        
    Returns:
        dict: Dictionary of key-value pairs
    """
    env_vars = {}
    for env_string in env_strings:
        if '=' not in env_string:
            print(f"ERROR: Invalid environment variable format: {env_string}")
            print("Expected format: KEY=VALUE")
            sys.exit(1)
        
        key, value = env_string.split('=', 1)  # Split only on first '=' to handle values with '='
        env_vars[key] = value
    
    return env_vars


def main():
    parser = argparse.ArgumentParser(
        description="Update environment variables in Kubernetes deployments using kubectl",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update single environment variable
  python3 update_k8s_env.py --env DATABASE_URL=postgres://localhost:5432/mydb

  # Update multiple environment variables
  python3 update_k8s_env.py --env API_KEY=secret123 --env DEBUG=true --env PORT=8080

  # Specify custom deployment and container
  python3 update_k8s_env.py --deployment my-app --container api-server --env VERSION=1.2.3

  # Update in specific namespace
  python3 update_k8s_env.py --namespace production --env ENVIRONMENT=prod
        """
    )
    parser.add_argument(
        "--env",
        action="append",
        required=True,
        help="Environment variable in KEY=VALUE format (can be used multiple times)"
    )
    parser.add_argument(
        "--deployment",
        type=str,
        default="routing-agent-service",
        help="Name of the deployment to update (default: routing-agent-service)"
    )
    parser.add_argument(
        "--container",
        type=str,
        default="routing-agent",
        help="Name of the container to update (default: routing-agent)"
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="default",
        help="Kubernetes namespace (default: default)"
    )
    
    args = parser.parse_args()
    
    # Parse environment variables
    env_vars = parse_env_vars(args.env)
    
    print(f"Updating deployment '{args.deployment}' in namespace '{args.namespace}'")
    print(f"Container: {args.container}")
    print("Environment variables to update:")
    for key, value in env_vars.items():
        print(f"  {key}: {value}")
    print("-" * 50)
    
    success = update_deployment_env_vars(
        deployment_name=args.deployment,
        namespace=args.namespace,
        container_name=args.container,
        env_vars=env_vars
    )
    
    if success:
        print("✅ Update completed successfully!")
        sys.exit(0)
    else:
        print("❌ Update failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()