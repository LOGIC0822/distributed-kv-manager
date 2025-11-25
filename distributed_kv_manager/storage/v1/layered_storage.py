from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

from distributed_kv_manager.storage.base import AbstractStorage

logger = logging.getLogger("V1LayeredStorage")


class V1LayeredStorage(AbstractStorage):
    """V1 专用的“按层分层”存储封装。

    语义：
    - 远端（remote_tier）始终持久化所有层的文件；
    - 本地（local_tier）只缓存前 `layer_split_front` 个层；
    - 读取时优先从本地读取被缓存的层，否则回落到远端。

    适用于 v1 per-layer safetensors 路径：
    file_path 形如::

        "<hash>/model.layers.<N>.self_attn.attn.safetensors"
    """

    def __init__(
        self,
        local_tier: AbstractStorage,
        remote_tier: AbstractStorage,
        layer_split_front: Optional[int] = None,
    ) -> None:
        self._local = local_tier
        self._remote = remote_tier
        self._front_n = layer_split_front

    # -------- helpers --------

    @staticmethod
    def _extract_layer_index(file_path: str) -> Optional[int]:
        """从路径中解析出 layer 索引 N（基于 'model.layers.N.' 模式）。"""
        marker = "model.layers."
        idx = file_path.find(marker)
        if idx == -1:
            return None
        start = idx + len(marker)
        end = start
        length = len(file_path)
        while end < length and file_path[end].isdigit():
            end += 1
        if end == start:
            return None
        try:
            return int(file_path[start:end])
        except Exception:
            return None

    def _should_use_local(self, file_path: str) -> bool:
        """判断该路径对应的层是否应缓存在本地。

        - `_front_n is None`：表示不做任何层缓存。
        - `_front_n < 0`：表示所有层都在本地缓存一份（如 `-1`）。
        - 其它情况：使用 `[0, front_n)`（0-based layer index）作为本地层范围。
        """
        if self._front_n is None:
            return False
        try:
            front_n = int(self._front_n)
        except Exception:
            return False
        if front_n < 0:
            # -1 及其它负数均视为“全部层均应缓存到本地”
            return True
        layer_idx = self._extract_layer_index(file_path)
        if layer_idx is None:
            return False
        # vLLM 层号目前是 0-based，这里约定 [0, front_n) 层在本地缓存。
        return layer_idx < front_n

    # -------- AbstractStorage 接口 --------

    def upload(self, file_path: str, data: bytes) -> bool:  # type: ignore[override]
        """远端始终写入；需要缓存的层额外写本地。"""
        remote_ok = self._remote.upload(file_path, data)
        local_ok = True
        if self._should_use_local(file_path):
            local_ok = self._local.upload(file_path, data)
        logger.debug(
            "V1LayeredStorage.upload path=%s use_local=%s remote_ok=%s local_ok=%s",
            file_path,
            self._should_use_local(file_path),
            remote_ok,
            local_ok,
        )
        return bool(remote_ok and local_ok)

    def download(self, file_path: str) -> Optional[bytes]:  # type: ignore[override]
        """优先从本地读取被缓存的层，否则回退到远端。"""
        if self._should_use_local(file_path):
            data = self._local.download(file_path)
            if data is not None:
                logger.debug("V1LayeredStorage.download hit local for %s", file_path)
                return data
        data = self._remote.download(file_path)
        if data is not None:
            logger.debug("V1LayeredStorage.download hit remote for %s", file_path)
        return data

    def exists(self, file_path: str) -> bool:  # type: ignore[override]
        """本地/远端只要有一个存在即认为存在。"""
        if self._should_use_local(file_path) and self._local.exists(file_path):
            return True
        return self._remote.exists(file_path)

    def pack_kv_data(  # type: ignore[override]
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        input_tokens: torch.Tensor,
        roi: torch.Tensor,
    ) -> bytes:
        """v0 兼容接口：直接委托给远端后端。"""
        if hasattr(self._remote, "pack_kv_data"):
            return self._remote.pack_kv_data(k_cache, v_cache, input_tokens, roi)  # type: ignore[no-any-return]
        # 后备：使用本地后端（理论上 v1_layered 只在 v1 路径使用，不依赖该接口）
        return self._local.pack_kv_data(k_cache, v_cache, input_tokens, roi)

    def unpack_kv_data(  # type: ignore[override]
        self, data: bytes
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """v0 兼容接口：直接委托给远端后端。"""
        if hasattr(self._remote, "unpack_kv_data"):
            return self._remote.unpack_kv_data(data)  # type: ignore[no-any-return]
        return self._local.unpack_kv_data(data)

    # 可选：删除时同时尝试删除本地和远端
    def delete(self, file_path: str) -> bool:
        local_ok = True
        remote_ok = True
        del_local = getattr(self._local, "delete", None)
        del_remote = getattr(self._remote, "delete", None)
        if callable(del_local):
            try:
                local_ok = bool(del_local(file_path))
            except Exception:
                local_ok = False
        if callable(del_remote):
            try:
                remote_ok = bool(del_remote(file_path))
            except Exception:
                remote_ok = False
        return bool(local_ok and remote_ok)

