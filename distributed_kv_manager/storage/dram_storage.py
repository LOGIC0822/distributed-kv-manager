import logging
import time
from typing import Optional, Tuple

import torch

from .base import AbstractStorage
from .caching_storage import MemoryCache  # 复用已有的 LRU 实现
from distributed_kv_manager.storage.v0.layout import (
    pack_kv_local_v0,
    unpack_kv_local_v0,
)

logger = logging.getLogger("MemoryStorage")


class MemoryStorage(AbstractStorage):
    """
    纯内存存储后端，用于模拟一个“内存盘”。
    - 底层用 MemoryCache 存 bytes（有容量限制 + LRU）
    - upload / download 中通过 sleep 模拟延时和带宽
    """

    def __init__(
        self,
        capacity_bytes: int = 256 * 1024 * 1024,
        base_latency_ms: float = 0.0,
        bandwidth_MBps: Optional[float] = None,
    ):
        """
        :param capacity_bytes: 内存总容量（字节），超过后按 LRU 淘汰
        :param base_latency_ms: 每次 IO 的基础延时（毫秒）
        :param bandwidth_MBps: 带宽上限（MB/s）。
                               None 表示不限制，只用 base_latency。
        """
        self._cache = MemoryCache(capacity_bytes)

        # 延时相关配置
        self._base_latency_s = max(float(base_latency_ms), 0.0) / 1000.0
        if bandwidth_MBps is None:
            self._bandwidth_Bps = None
        else:
            # 避免除零
            self._bandwidth_Bps = max(float(bandwidth_MBps), 1e-6) * 1024 * 1024

        logger.info(
            "Initialized MemoryStorage: cap=%dB base_latency=%.3fms bw=%sMB/s",
            capacity_bytes,
            base_latency_ms,
            str(bandwidth_MBps),
        )

    # ========= 核心：IO 延时/带宽模拟 ==========

    def _simulate_io(self, nbytes: int) -> None:
        delay = self._base_latency_s
        if self._bandwidth_Bps is not None and nbytes > 0:
            delay += nbytes / self._bandwidth_Bps
        if delay > 0:
            time.sleep(delay)

    # ========= AbstractStorage 接口实现 ==========

    def upload(self, file_path: str, data: bytes) -> bool:
        """把数据写入内存（并模拟一次写 IO）"""
        try:
            nbytes = len(data)
        except Exception:
            nbytes = 0

        self._simulate_io(nbytes)

        try:
            self._cache.put(file_path, data)
            logger.debug(
                "MemoryStorage.upload: key=%s size=%dB", file_path, nbytes
            )
            return True
        except Exception as e:
            logger.error("MemoryStorage.upload(%s) failed: %s", file_path, e)
            return False

    def download(self, file_path: str) -> Optional[bytes]:
        """从内存读取数据（并模拟一次读 IO）"""
        try:
            data = self._cache.get(file_path)
            if data is None:
                logger.debug("MemoryStorage.download: miss key=%s", file_path)
                return None

            self._simulate_io(len(data))
            logger.debug(
                "MemoryStorage.download: hit key=%s size=%dB",
                file_path,
                len(data),
            )
            return data
        except Exception as e:
            logger.error("MemoryStorage.download(%s) failed: %s", file_path, e)
            return None

    def exists(self, file_path: str) -> bool:
        """检查 key 是否存在（注意这里会触发一次 LRU 访问）"""
        v = self._cache.get(file_path)
        return v is not None

    # ======== KV pack/unpack：直接复用 local_storage 的 v0 layout ========

    def pack_kv_data(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        input_tokens: torch.Tensor,
        roi: torch.Tensor,
    ) -> bytes:
        logger.debug(
            "MemoryStorage.pack_kv_data(v0): k=%s, v=%s",
            tuple(k_cache.shape),
            tuple(v_cache.shape),
        )
        return pack_kv_local_v0(k_cache, v_cache, input_tokens, roi)

    def unpack_kv_data(
        self, data: bytes
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        try:
            logger.debug(
                "MemoryStorage.unpack_kv_data(v0): data_size=%d", len(data)
            )
        except Exception:
            pass

        k_cache, v_cache = unpack_kv_local_v0(data)
        if k_cache is not None and v_cache is not None:
            logger.debug(
                "MemoryStorage.unpack_kv_data(v0) done: k=%s, v=%s",
                tuple(k_cache.shape),
                tuple(v_cache.shape),
            )
        return k_cache, v_cache

    # 可选：支持删除和前缀枚举，方便调试和测试

    def delete(self, file_path: str) -> bool:
        try:
            ok = self._cache.delete(file_path)
            if ok:
                logger.debug("MemoryStorage.delete: %s", file_path)
            return ok
        except Exception as e:
            logger.error("MemoryStorage.delete(%s) failed: %s", file_path, e)
            return False

    def list_files_with_prefix(self, prefix: str) -> list[str]:
        """简单遍历 MemoryCache 的 key，按前缀过滤。仅用于测试/调试。"""
        try:
            # MemoryCache._map 是 OrderedDict，key 就是 file_path
            keys = list(self._cache._map.keys())  # type: ignore[attr-defined]
        except Exception:
            return []

        if not prefix:
            return list(keys)
        return [k for k in keys if k.startswith(prefix)]