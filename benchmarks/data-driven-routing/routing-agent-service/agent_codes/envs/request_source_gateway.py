from typing import Optional
from .broker import RequestBroker, PendingReq

class GatewayRequestSource:
    def __init__(self, broker: RequestBroker):
        self.broker = broker

    def get_next(self, timeout: Optional[float] = None) -> PendingReq:
        return self.broker.get_next(timeout=timeout)
