from dataclasses import dataclass
from typing import Any, Dict
from broker import PendingReq, RequestBroker

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
        # TODO: action probabilities for debugging
        self.broker.set_decision(self.pending.request_id, pod_idx)
