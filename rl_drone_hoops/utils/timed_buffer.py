from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, List, Optional, Tuple, TypeVar

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
        self._items: List[TimedItem[T]] = []

    def push(self, *, deliver_ts: float, capture_ts: float, payload: T) -> None:
        self._items.append(TimedItem(deliver_ts=deliver_ts, capture_ts=capture_ts, payload=payload))

    def pop_ready(self, now_ts: float) -> List[TimedItem[T]]:
        if not self._items:
            return []
        ready: List[TimedItem[T]] = []
        i = 0
        while i < len(self._items) and self._items[i].deliver_ts <= now_ts:
            ready.append(self._items[i])
            i += 1
        if i:
            self._items = self._items[i:]
        return ready

    def next_deliver_ts(self) -> Optional[float]:
        if not self._items:
            return None
        return self._items[0].deliver_ts

