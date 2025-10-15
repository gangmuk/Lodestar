from dataclasses import dataclass
from typing import Any, Dict
from .broker import PendingReq, RequestBroker

@dataclass
class Request:
    pending: PendingReq
    broker: RequestBroker

    @property
    def state(self) -> Dict[str, Any]:
        return self.pending.state

    def get_obs(self) -> Dict[str, Any]:
        return self.pending.state["obs"]

    def route(self, pod_idx: int):
        self.broker.set_decision(self.pending.request_id, pod_idx)

    # def __init__(self, state=None, poll_interval=0.01):
    #     '''
    #     Args:
    #         state: all relevant information of the request, should be a dict with "reward" key, should contain all observation information
    #         poll_interval: polling time unit
    #     '''
    #     self._state = state
    #     self._poll_interval = poll_interval

    # @property
    # def state(self):
    #     if self._state is None:
    #         raise ValueError("Empty request")
    #     return self._state
    
    # def get_reward(self):
    #     return self._state['reward']

    # def get_obs(self):
    #     ## TODO: create the observation based on the request, need to match the observation space of the environment
    #     ## Example:
    #         # observation_space = spaces.Dict({
    #         # 'pod_features': spaces.Box(
    #         #     -np.inf, np.inf,
    #         #     shape=(num_pods, per_pod_dim - 1),
    #         #     dtype=np.float32
    #         # ),
    #         # 'kv_hit_ratios': spaces.Box(
    #         #     0.0, 1.0,
    #         #     shape=(num_pods, 1),
    #         #     dtype=np.float32
    #         # ),
    #         # 'request_features': spaces.Box(
    #         #     -np.inf, np.inf,
    #         #     shape=(request_dim,),
    #         #     dtype=np.float32
    #         # ),
    #         # 'temporal_features': spaces.Box(
    #         #     -np.inf, np.inf,
    #         #     shape=(0,),
    #         #     dtype=np.float32
    #         # )
    #     # })
    #     obs = Dict({})

    #     return obs

    # def wait_for_request(self):
    #     # TODO: wait for next request and return the new request
    #     # HTTP request come in from Gateway
    #     # It should set the state of the new request
    #     # Keep trying to get the next request until it comes in
        
    #     new_request = Request()
    #     while new_request._state is None:
    #         time.sleep(self._poll_interval)
        
    #     return new_request

    # def route(self, pod_id: int):
    #     # TODO: route request
    #     # This should answer HTTP request from Gateway with the pod_id
    #     pass