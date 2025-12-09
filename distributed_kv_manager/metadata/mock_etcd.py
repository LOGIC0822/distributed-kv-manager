import threading
from types import SimpleNamespace
from typing import Dict, Iterator, Tuple


class MockEtcd3Client:
    """
    A lightweight in-memory stand-in for etcd3.Etcd3Client used in tests and
    local runs where a real etcd cluster is unavailable.
    """

    _global_store: Dict[str, bytes] = {}
    _global_lock = threading.RLock()

    def __init__(self, endpoint: str = "mock://local", store: Dict[str, bytes] | None = None):
        # Allow sharing a custom store for isolated tests; default is a process-wide store.
        self._store = store if store is not None else MockEtcd3Client._global_store
        self._lock = threading.RLock() if store is not None else MockEtcd3Client._global_lock
        self.endpoint = endpoint
        self._url = endpoint if endpoint.startswith("mock://") else f"mock://{endpoint}"

    # ----------------- etcd3 compatible API ----------------- #
    def status(self):
        return {"health": "ok", "endpoint": self._url, "count": len(self._store)}

    def put(self, key, value) -> bool:
        skey = self._normalize_key(key)
        with self._lock:
            self._store[skey] = value
        return True

    def get(self, key) -> Tuple[bytes | None, SimpleNamespace]:
        skey = self._normalize_key(key)
        with self._lock:
            val = self._store.get(skey)
        meta = SimpleNamespace(key=skey.encode("utf-8"))
        return val, meta

    def get_prefix(self, prefix: str) -> Iterator[Tuple[bytes, SimpleNamespace]]:
        sprefix = self._normalize_key(prefix)
        with self._lock:
            items = [(k, v) for k, v in self._store.items() if k.startswith(sprefix)]
        for k, v in items:
            yield v, SimpleNamespace(key=k.encode("utf-8"))

    def delete(self, key) -> bool:
        skey = self._normalize_key(key)
        with self._lock:
            existed = skey in self._store
            self._store.pop(skey, None)
        return existed

    def watch(self, key):
        # Returns an empty iterator and a no-op cancel function.
        return iter([]), (lambda: None)

    # ----------------- Helpers ----------------- #
    @staticmethod
    def _normalize_key(key) -> str:
        if isinstance(key, bytes):
            return key.decode("utf-8")
        return str(key)


def reset_mock_etcd_store():
    """Clear the shared in-memory mock etcd store."""
    with MockEtcd3Client._global_lock:
        MockEtcd3Client._global_store.clear()
