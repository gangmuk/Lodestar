import utils
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check if the deployment is ready')
    parser.add_argument('--deployment', type=str, required=True, help='Deployment name')
    parser.add_argument('--namespace', type=str, required=True, help='Namespace')
    parser.add_argument('--k8s_cluster', type=str, default='vke', choices=['vke', 'local'], help='Kubernetes cluster')
    args = parser.parse_args()
    
    utils.check_deployment_ready_kubernetes(args.deployment, args.k8s_cluster, args.namespace)