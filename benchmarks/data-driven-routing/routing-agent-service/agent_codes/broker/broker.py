import threading, queue
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from logger import logger

@dataclass
class PendingReq:
    request_id: str
    state: Dict[str, Any]                      # must include "obs"
    prev_reward: Optional[float] = None        # reward for the *previous* request
    decision_event: threading.Event = field(default_factory=threading.Event)
    decision_action: Optional[int] = None
    decision_probs: Optional[Any] = None       # action probabilities from policy

class RequestBroker:
    def __init__(self, maxsize: int = 10000):
        '''
        _by_id access is protected by _lock.
        This is for sb3 SubprocVecEnv, where multiple workers run independently (XXX)
        '''
        self._queue = queue.Queue(maxsize=maxsize)
        self._by_id: Dict[str, PendingReq] = {}
        self._lock = threading.Lock()

    def submit(self, request_id: str, state: Dict[str, Any],
               prev_reward: Optional[float]) -> PendingReq:
        pr = PendingReq(request_id=request_id, state=state, prev_reward=prev_reward)
        with self._lock:
            self._by_id[request_id] = pr
        self._queue.put(pr)
        return pr

    def get_next(self, timeout: Optional[float] = None) -> PendingReq:
        return self._queue.get(timeout=timeout)

    def set_decision(self, request_id: str, action: int, probs: Optional[Any] = None):
        with self._lock:
            pr = self._by_id.get(request_id)
        if pr:
            pr.decision_action = int(action)
            pr.decision_probs = probs # TODO: action probabilities for debugging
            pr.decision_event.set()
        else:
            logger.error(f"Request {request_id} sets decision but not found in broker")
            assert False

    def wait_for_decision(self, request_id: str, timeout: Optional[float]):
        """
        Wait for decision and return (action, probs) tuple.
        Returns None if timeout or not found.
        """
        with self._lock:
            pr = self._by_id.get(request_id)
        if pr is None:
            logger.error(f"Request {request_id} waits for decision but not found in broker")
            assert False
        ok = pr.decision_event.wait(timeout)
        if not ok:
            logger.error(f"pr.decision_event timed out (timeout={timeout})...  Request {request_id}")
            return None
        return (pr.decision_action, pr.decision_probs) if ok else None

    def pop(self, request_id: str):
        with self._lock:
            self._by_id.pop(request_id, None)
