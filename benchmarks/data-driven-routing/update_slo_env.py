#!/usr/bin/env python3
"""
Script to update TTFT_SLO and AVG_TPOT_SLO environment variables 
in the routing-agent-service Kubernetes deployment.
"""

import argparse
import sys
from kubernetes import client, config
from kubernetes.client.rest import ApiException


def update_deployment_env_vars(deployment_name, namespace, ttft_slo, avg_tpot_slo):
    """
    Update environment variables in a Kubernetes deployment.
    
    Args:
        deployment_name (str): Name of the deployment to update
        namespace (str): Kubernetes namespace
        ttft_slo (str): New value for TTFT_SLO
        avg_tpot_slo (str): New value for AVG_TPOT_SLO
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
            if container.name == "routing-agent":
                container_found = True
                print(f"Found container '{container.name}', updating environment variables...")
                
                # Update or add environment variables
                if container.env is None:
                    container.env = []
                
                # Track which env vars were updated
                ttft_updated = False
                avg_tpot_updated = False
                
                # Update existing environment variables
                for env_var in container.env:
                    if env_var.name == "TTFT_SLO":
                        old_value = env_var.value
                        env_var.value = ttft_slo
                        print(f"Updated TTFT_SLO: {old_value} -> {ttft_slo}")
                        ttft_updated = True
                    elif env_var.name == "AVG_TPOT_SLO":
                        old_value = env_var.value
                        env_var.value = avg_tpot_slo
                        print(f"Updated AVG_TPOT_SLO: {old_value} -> {avg_tpot_slo}")
                        avg_tpot_updated = True
                
                # Add environment variables if they don't exist
                if not ttft_updated:
                    container.env.append(client.V1EnvVar(name="TTFT_SLO", value=ttft_slo))
                    print(f"Added TTFT_SLO: {ttft_slo}")
                
                if not avg_tpot_updated:
                    container.env.append(client.V1EnvVar(name="AVG_TPOT_SLO", value=avg_tpot_slo))
                    print(f"Added AVG_TPOT_SLO: {avg_tpot_slo}")
                
                break
        
        if not container_found:
            print("ERROR: Container 'routing-agent' not found in deployment")
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


def main():
    parser = argparse.ArgumentParser(
        description="Update TTFT_SLO and AVG_TPOT_SLO environment variables in routing-agent-service deployment"
    )
    parser.add_argument(
        "--ttft-slo",
        type=str,
        required=True,
        help="New value for TTFT_SLO environment variable"
    )
    parser.add_argument(
        "--avg-tpot-slo",
        type=str,
        required=True,
        help="New value for AVG_TPOT_SLO environment variable"
    )
    parser.add_argument(
        "--deployment-name",
        type=str,
        default="routing-agent-service",
        help="Name of the deployment to update (default: routing-agent-service)"
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="default",
        help="Kubernetes namespace (default: default)"
    )
    
    args = parser.parse_args()
    
    print(f"Updating deployment '{args.deployment_name}' in namespace '{args.namespace}'")
    print(f"TTFT_SLO: {args.ttft_slo}")
    print(f"AVG_TPOT_SLO: {args.avg_tpot_slo}")
    print("-" * 50)
    
    success = update_deployment_env_vars(
        deployment_name=args.deployment_name,
        namespace=args.namespace,
        ttft_slo=args.ttft_slo,
        avg_tpot_slo=args.avg_tpot_slo
    )
    
    if success:
        print("✅ Update completed successfully!")
        sys.exit(0)
    else:
        print("❌ Update failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()