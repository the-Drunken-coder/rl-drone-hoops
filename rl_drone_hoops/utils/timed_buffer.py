from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Generic, List, Optional, TypeVar

T = TypeVar("T")


@dataclass
class TimedItem(Generic[T]):
    deliver_ts: float
    capture_ts: float
    payload: T


class TimedDeliveryBuffer(Generic[T]):
    """
    FIFO of items that become available at deliver_ts.
    """

    def __init__(self) -> None:
        # Deque avoids O(n) list slicing during pop_ready().
        self._items: Deque[TimedItem[T]] = deque()

    def push(self, *, deliver_ts: float, capture_ts: float, payload: T) -> None:
        self._items.append(TimedItem(deliver_ts=deliver_ts, capture_ts=capture_ts, payload=payload))

    def pop_ready(self, now_ts: float) -> List[TimedItem[T]]:
        if not self._items:
            return []
        ready: List[TimedItem[T]] = []
        while self._items and self._items[0].deliver_ts <= now_ts:
            ready.append(self._items.popleft())
        return ready

    def next_deliver_ts(self) -> Optional[float]:
        if not self._items:
            return None
        return self._items[0].deliver_ts
