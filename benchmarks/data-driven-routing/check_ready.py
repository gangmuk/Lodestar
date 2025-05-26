import utils
import time

if __name__ == '__main__':
    # utils.restart_deploy('aibrix-gateway-plugins', 'aibrix-system')
    # utils.restart_deploy('latency-predictor-service', 'default')
    # utils.restart_deploy('llama-3-8b-instruct', 'default')
    time.sleep(3)
    utils.check_deployment_ready_kubernetes('aibrix-gateway-plugins', 'aibrix-system')
    utils.check_deployment_ready_kubernetes('llama-3-8b-instruct', 'default')
    # utils.check_deployment_ready_kubernetes('latency-predictor-service', 'default')
    utils.check_deployment_ready_kubernetes('routing-agent-service', 'default')