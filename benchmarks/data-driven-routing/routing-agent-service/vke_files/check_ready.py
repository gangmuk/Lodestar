import utils
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check if the deployment is ready')
    parser.add_argument('--deployment', type=str, required=True, help='Deployment name')
    parser.add_argument('--namespace', type=str, required=True, help='Namespace')
    args = parser.parse_args()
    utils.check_deployment_ready_kubernetes(args.deployment, args.namespace)