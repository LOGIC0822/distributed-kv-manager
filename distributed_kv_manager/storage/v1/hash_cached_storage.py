from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Optional, Tuple

import torch

from distributed_kv_manager.storage.base import AbstractStorage


class V1HashDRAMCache:
    """Hash 级别的 DRAM 缓存。

    - key 为完整 file_path，例如 "<hash>/model.layers.N.self_attn.attn.safetensors"
    - LRU 单位为 hash 目录：淘汰时一次性驱逐该 hash 下的所有 file_path
    """

    def __init__(self, capacity_bytes: int) -> None:
        self._capacity = max(int(capacity_bytes), 1)
        # OrderedDict: hash_id -> { file_path: bytes }
        self._groups: "OrderedDict[str, dict[str, bytes]]" = OrderedDict()
        self._size: int = 0
        self._lock = threading.RLock()

    @staticmethod
    def _hash_id(file_path: str) -> str:
        try:
            if "/" in file_path:
                return file_path.split("/", 1)[0]
        except Exception:
            pass
        return file_path

    def get(self, file_path: str) -> Optional[bytes]:
        hid = self._hash_id(file_path)
        with self._lock:
            group = self._groups.get(hid)
            if not group:
                return None
            data = group.get(file_path)
            if data is not None:
                # 按 hash 组 LRU
                self._groups.move_to_end(hid)
            return data

    def put(self, file_path: str, data: bytes) -> None:
        if data is None:
            return
        hid = self._hash_id(file_path)
        size = len(data)
        with self._lock:
            group = self._groups.get(hid)
            if group is None:
                group = {}
                self._groups[hid] = group
            old = group.get(file_path)
            if old is not None:
                self._size -= len(old)
            group[file_path] = data
            self._groups.move_to_end(hid)
            self._size += size
            self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        while self._size > self._capacity and self._groups:
            # 按 hash 组淘汰
            hid, files = self._groups.popitem(last=False)
            for _, data in files.items():
                self._size -= len(data)


class V1HashCachedStorage(AbstractStorage):
    """基于 hash 目录的 DRAM 缓存封装，仅用于 v1 per-layer safetensors。

    upload/download 都会经过 hash 级 DRAM 缓存：
    - upload 成功后，将该 file_path 对应的 bytes 写入缓存；
    - download 先查缓存，miss 再访问后端并写入缓存。
    """

    def __init__(self, backend: AbstractStorage, capacity_bytes: int) -> None:
        self._backend = backend
        self._cache = V1HashDRAMCache(capacity_bytes)

    # ----- 基本 IO -----
    def upload(self, file_path: str, data: bytes) -> bool:  # type: ignore[override]
        ok = self._backend.upload(file_path, data)
        if ok:
            self._cache.put(file_path, data)
        return ok

    def download(self, file_path: str) -> Optional[bytes]:  # type: ignore[override]
        data = self._cache.get(file_path)
        if data is not None:
            return data
        data = self._backend.download(file_path)
        if data is not None:
            self._cache.put(file_path, data)
        return data

    def exists(self, file_path: str) -> bool:  # type: ignore[override]
        return self._backend.exists(file_path)

    # ----- v0 兼容接口：直接委托给后端 -----
    def pack_kv_data(  # type: ignore[override]
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        input_tokens: torch.Tensor,
        roi: torch.Tensor,
    ) -> bytes:
        return self._backend.pack_kv_data(k_cache, v_cache, input_tokens, roi)

    def unpack_kv_data(  # type: ignore[override]
        self, data: bytes
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self._backend.unpack_kv_data(data)

