#!/usr/bin/env python3
"""
Script to update environment variables in Kubernetes deployments.
Supports updating multiple key-value pairs in a single operation.
"""

import argparse
import sys
from kubernetes import client, config
from kubernetes.client.rest import ApiException


def update_deployment_env_vars(deployment_name, namespace, container_name, env_vars):
    """
    Update environment variables in a Kubernetes deployment.
    
    Args:
        deployment_name (str): Name of the deployment to update
        namespace (str): Kubernetes namespace
        container_name (str): Name of the container to update
        env_vars (dict): Dictionary of environment variable key-value pairs
    """
    try:
        # Load Kubernetes config
        try:
            # Try to load in-cluster config first (if running inside a pod)
            config.load_incluster_config()
            print("Loaded in-cluster Kubernetes config")
        except config.ConfigException:
            # Fall back to loading from ~/.kube/config
            config.load_kube_config()
            print("Loaded Kubernetes config from ~/.kube/config")
        
        # Create API client
        apps_v1 = client.AppsV1Api()
        
        # Get the current deployment
        print(f"Fetching deployment '{deployment_name}' in namespace '{namespace}'...")
        deployment = apps_v1.read_namespaced_deployment(
            name=deployment_name,
            namespace=namespace
        )
        
        # Find the container and update environment variables
        containers = deployment.spec.template.spec.containers
        container_found = False
        
        for container in containers:
            if container.name == container_name:
                container_found = True
                print(f"Found container '{container.name}', updating environment variables...")
                
                # Initialize env list if it doesn't exist
                if container.env is None:
                    container.env = []
                
                # Process each environment variable
                for env_key, env_value in env_vars.items():
                    env_updated = False
                    
                    # Update existing environment variable
                    for env_var in container.env:
                        if env_var.name == env_key:
                            old_value = env_var.value
                            env_var.value = env_value
                            print(f"Updated {env_key}: {old_value} -> {env_value}")
                            env_updated = True
                            break
                    
                    # Add environment variable if it doesn't exist
                    if not env_updated:
                        container.env.append(client.V1EnvVar(name=env_key, value=env_value))
                        print(f"Added {env_key}: {env_value}")
                
                break
        
        if not container_found:
            print(f"ERROR: Container '{container_name}' not found in deployment")
            return False
        
        # Update the deployment
        print("Applying changes to deployment...")
        apps_v1.patch_namespaced_deployment(
            name=deployment_name,
            namespace=namespace,
            body=deployment
        )
        
        print(f"Successfully updated deployment '{deployment_name}'")
        print("The deployment will automatically roll out the changes.")
        return True
        
    except ApiException as e:
        print(f"Kubernetes API error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
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
        description="Update environment variables in Kubernetes deployments",
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