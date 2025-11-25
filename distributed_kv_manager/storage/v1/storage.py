from __future__ import annotations

from typing import Optional, Tuple

import torch

from distributed_kv_manager.storage.base import AbstractStorage
from distributed_kv_manager.storage.v1.hash_cached_storage import V1HashCachedStorage


class V1Storage:
    """Thin wrapper on top of an AbstractStorage implementation.

    workflow: upload/download and pack/unpack helper utilities.
    """

    def __init__(self, backend: AbstractStorage) -> None:
        self._backend = backend

    # Basic IO
    def upload(self, file_path: str, data: bytes) -> bool:
        return self._backend.upload(file_path, data)

    def download(self, file_path: str) -> Optional[bytes]:
        return self._backend.download(file_path)

    def exists(self, file_path: str) -> bool:
        return self._backend.exists(file_path)

    # ----- Full-payload helpers (v0 layout; v1 engine 不直接使用) -----
    def pack_full_payload(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        input_tokens: torch.Tensor,
        roi: torch.Tensor,
        slot_mapping: Optional[torch.Tensor] = None,
        payload_meta: Optional[dict] = None,
    ) -> bytes:
        return self._backend.pack_full_payload(
            k_cache, v_cache, input_tokens, roi, slot_mapping, payload_meta
        )

    def unpack_kv_data(self, data: bytes) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self._backend.unpack_kv_data(data)

    def extract_payload_info(self, data: bytes) -> dict:
        return self._backend.extract_payload_info(data)


def create_v1_storage(config) -> V1Storage:
    """Create a V1Storage by delegating to the project's StorageFactory.

    The `config` is expected to have `kv_transfer_config` fields compatible
    with the existing StorageFactory.
    """
    # 延迟导入 StorageFactory 以避免与 storage.factory 之间的循环依赖
    from distributed_kv_manager.storage.factory import StorageFactory

    backend = StorageFactory.create_storage(config)

    # 如果配置显式标记为 v1 版本，则在后端外再包一层 hash 级 DRAM 缓存。
    kvt = getattr(config, "kv_transfer_config", config)
    use_v1 = bool(getattr(kvt, "use_v1", False))
    if use_v1:
        # 容量：优先使用 mem_cache_capacity_gb，其次兼容旧的 mem_cache_capacity_bytes。
        try:
            cap_gb = getattr(kvt, "mem_cache_capacity_gb", None)
        except Exception:
            cap_gb = None
        if cap_gb is not None:
            try:
                capacity_bytes = int(float(cap_gb) * 1024 * 1024 * 1024)
            except Exception:
                capacity_bytes = 256 * 1024 * 1024
        else:
            try:
                capacity_bytes = int(
                    getattr(kvt, "mem_cache_capacity_bytes", 256 * 1024 * 1024)
                )
            except Exception:
                capacity_bytes = 256 * 1024 * 1024
        backend = V1HashCachedStorage(backend, capacity_bytes)

    return V1Storage(backend)
