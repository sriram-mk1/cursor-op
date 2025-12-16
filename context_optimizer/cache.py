import time
from collections import OrderedDict
from typing import Any, Dict, Tuple


class QueryCache:
    def __init__(self, max_entries: int = 256):
        self.max_entries = max_entries
        self.store: "OrderedDict[Tuple[Any, ...], Dict[str, Any]]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: Tuple[Any, ...], ttl: int) -> Any | None:
        entry = self.store.get(key)
        if not entry:
            self.misses += 1
            return None
        if time.monotonic() - entry["ts"] > ttl:
            self.misses += 1
            del self.store[key]
            return None
        self.hits += 1
        self.store.move_to_end(key)
        return entry["value"]

    def set(self, key: Tuple[Any, ...], value: Any) -> None:
        self.store[key] = {"value": value, "ts": time.monotonic()}
        self.store.move_to_end(key)
        if len(self.store) > self.max_entries:
            self.store.popitem(last=False)
