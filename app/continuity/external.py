from __future__ import annotations

import threading
import time
from collections import deque


class ExternalContinuityRateLimiter:
    """Small in-process abuse boundary for the single Custom GPT credential."""

    def __init__(self, requests_per_minute: int) -> None:
        self._limit = max(1, requests_per_minute)
        self._requests: deque[float] = deque()
        self._lock = threading.Lock()

    def allow(self) -> bool:
        now = time.monotonic()
        cutoff = now - 60.0
        with self._lock:
            while self._requests and self._requests[0] <= cutoff:
                self._requests.popleft()
            if len(self._requests) >= self._limit:
                return False
            self._requests.append(now)
            return True
