from typing import Optional
import logging
import importlib

from .base import AbstractStorage
from .crail_storage import CrailStorage
from .local_storage import LocalStorage
from .caching_storage import CachingStorage
from .composite_storage import CompositeStorage
from .path_mapper import HashBucketPathMapper, MappedStorage
from .v1.layered_storage import V1LayeredStorage

logger = logging.getLogger("StorageFactory")


class _NoopRemoteStorage(AbstractStorage):
    """占位远端存储实现：
    - upload: 始终返回 True（不真正持久化）
    - download: 始终返回 None
    - exists: 始终返回 False
    仅用于 composite / v1_layered 中的“noop 远端”场景。
    """

    def upload(self, file_path: str, data: bytes) -> bool:  # type: ignore[override]
        return True

    def download(self, file_path: str):  # type: ignore[override]
        return None

    def exists(self, file_path: str) -> bool:  # type: ignore[override]
        return False

    def pack_kv_data(self, *args, **kwargs) -> bytes:  # type: ignore[override]
        return b""

    def unpack_kv_data(self, data: bytes):  # type: ignore[override]
        return None, None


class StorageFactory:
    """存储后端工厂：根据 kv_transfer_config 创建具体的 AbstractStorage 实例。"""

    @staticmethod
    def create_storage(config) -> Optional[AbstractStorage]:
        """按配置创建存储实例，支持缓存包装 / 预取等特性。

        说明：
        - config.kv_transfer_config 可能是：
          * vLLM 内部的 KVTransferConfig（包含 extra_config 字段）；
          * 或本项目 config.json 解析出的 SimpleNamespace。
        - 为了兼容 CLI 的 --kv-transfer-config，我们优先从 kv_transfer_config.<name>
          读取字段，其次再从 kv_transfer_config.extra_config[name] 读取。
        """
        kvt = getattr(config, "kv_transfer_config", config)

        def _get(name: str, default=None):
            val = getattr(kvt, name, None)
            if val is not None:
                return val
            extra = getattr(kvt, "extra_config", None)
            if isinstance(extra, dict) and name in extra:
                return extra[name]
            return default

        storage_type = _get("storage_type", "local")
        enable_caching = bool(_get("enable_ssd_caching", False))
        cache_dir = _get("ssd_cache_dir", "/tmp/ssd_cache")
        enable_prefetch = bool(_get("enable_prefetch", True))
        cache_mode = _get("cache_mode", None)  # none | only_mem | mem_and_ssd
        # 内存缓存容量：优先支持 GB 配置，其次兼容旧的 bytes 配置
        mem_capacity_bytes = int(_get("mem_cache_capacity_bytes", 256 * 1024 * 1024))
        mem_capacity_gb = _get("mem_cache_capacity_gb", None)
        if mem_capacity_gb is not None:
            try:
                mem_capacity_bytes = int(float(mem_capacity_gb) * 1024 * 1024 * 1024)
            except Exception:
                pass

        logger.info(
            "创建存储实例: storage_type=%s, enable_caching=%s, cache_dir=%s",
            storage_type,
            enable_caching,
            cache_dir,
        )

        # ---------- 创建基础存储后端 ----------
        base_storage: Optional[AbstractStorage] = None
        layout = _get("directory_layout", "flat")
        mapper_enabled = str(layout).lower() in ("hash2", "hash_bucket", "hash")
        mapper = HashBucketPathMapper(enable=mapper_enabled)

        stype = None if storage_type is None else str(storage_type).lower()

        if stype == "crail":
            crail_dir = _get("crail_dir", "/crail/kvcache")
            base_storage = CrailStorage(str(crail_dir))
            if mapper_enabled:
                base_storage = MappedStorage(base_storage, mapper)

        elif stype == "local":
            # 本地文件系统存储
            local_dir = _get("local_dir", "/tmp/kvcache")
            base_storage = LocalStorage(str(local_dir))
            if mapper_enabled:
                base_storage = MappedStorage(base_storage, mapper)

        elif stype == "composite":
            # 组合后端：前半部分在本地，后半部分在远端
            local_dir = _get("local_dir", "/tmp/kvcache")
            remote_dir = _get("remote_dir", _get("crail_dir", "/crail/kvcache"))
            front_n = _get("layer_split_front", None)
            ratio = float(_get("layer_split_ratio", 0.5))
            local_backend: AbstractStorage = LocalStorage(str(local_dir))

            # 远端后端选择：remote_backend_class > remote_backend_type / remote_storage_type
            remote_backend_type = _get("remote_backend_type", _get("remote_storage_type", None))
            remote_backend_class = _get("remote_backend_class", None)

            def _import_backend(class_path: str):
                try:
                    module_name, cls_name = str(class_path).rsplit(".", 1)
                    mod = importlib.import_module(module_name)
                    return getattr(mod, cls_name)
                except Exception as e:
                    logger.warning("远端后端动态导入失败: %s (%s)", class_path, e)
                    return None

            if remote_backend_class:
                cls = _import_backend(remote_backend_class)
                remote_backend: AbstractStorage = cls(str(remote_dir)) if cls else _NoopRemoteStorage()
            else:
                typ = None if remote_backend_type is None else str(remote_backend_type).lower()
                if typ in ("noop", "none", "placeholder"):
                    remote_backend = _NoopRemoteStorage()
                elif typ == "local":
                    remote_backend = LocalStorage(str(remote_dir))
                elif typ == "crail" or (typ is None and getattr(kvt, "crail_dir", None) is not None):
                    remote_backend = CrailStorage(str(remote_dir))
                elif typ and "." in typ:
                    cls = _import_backend(typ)
                    remote_backend = cls(str(remote_dir)) if cls else _NoopRemoteStorage()
                else:
                    remote_backend = _NoopRemoteStorage()

            # 目录映射：优先在本地 / 远端后端上包 MappedStorage
            if mapper_enabled and not isinstance(local_backend, _NoopRemoteStorage):
                local_backend = MappedStorage(local_backend, mapper)
            if mapper_enabled and not isinstance(remote_backend, _NoopRemoteStorage):
                remote_backend = MappedStorage(remote_backend, mapper)

            # 若远端为占位实现且未显式指定 front_n，则强制将所有层留在本地
            if isinstance(remote_backend, _NoopRemoteStorage) and front_n is None:
                front_n = 1_000_000

            base_storage = CompositeStorage(
                local_backend,
                remote_backend,
                layer_split_front=front_n,
                layer_split_ratio=ratio,
            )

        elif stype == "v1_layered":
            # v1 专用：远端持久化所有层，本地只缓存前 front_n 层（per-layer safetensors）
            local_dir = _get("local_dir", "/tmp/kvcache")
            remote_dir = _get("remote_dir", _get("crail_dir", "/crail/kvcache"))
            front_n = _get("layer_split_front", None)
            remote_backend_type = _get("remote_backend_type", _get("remote_storage_type", None))
            remote_backend_class = _get("remote_backend_class", None)

            def _import_backend_v1(class_path: str):
                try:
                    module_name, cls_name = str(class_path).rsplit(".", 1)
                    mod = importlib.import_module(module_name)
                    return getattr(mod, cls_name)
                except Exception as e:
                    logger.warning("v1_layered 远端后端动态导入失败: %s (%s)", class_path, e)
                    return None

            local_backend: AbstractStorage = LocalStorage(str(local_dir))
            if remote_backend_class:
                cls = _import_backend_v1(remote_backend_class)
                remote_backend: AbstractStorage = cls(str(remote_dir)) if cls else _NoopRemoteStorage()
            else:
                typ = None if remote_backend_type is None else str(remote_backend_type).lower()
                if typ in ("noop", "none", "placeholder"):
                    remote_backend = _NoopRemoteStorage()
                elif typ == "local":
                    remote_backend = LocalStorage(str(remote_dir))
                elif typ == "crail" or (typ is None and getattr(kvt, "crail_dir", None) is not None):
                    remote_backend = CrailStorage(str(remote_dir))
                elif typ and "." in typ:
                    cls = _import_backend_v1(typ)
                    remote_backend = cls(str(remote_dir)) if cls else _NoopRemoteStorage()
                else:
                    remote_backend = _NoopRemoteStorage()

            if mapper_enabled and not isinstance(local_backend, _NoopRemoteStorage):
                local_backend = MappedStorage(local_backend, mapper)
            if mapper_enabled and not isinstance(remote_backend, _NoopRemoteStorage):
                remote_backend = MappedStorage(remote_backend, mapper)

            base_storage = V1LayeredStorage(local_backend, remote_backend, layer_split_front=front_n)

        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")

        # ---------- 根据缓存配置包装 CachingStorage ----------
        if cache_mode is None:
            # 兼容旧的 enable_ssd_caching 逻辑
            if enable_caching:
                logger.info("启用 SSD 缓存（兼容模式），目录: %s", cache_dir)
                return CachingStorage(
                    base_storage,
                    cache_dir,
                    enable_prefetch,
                    cache_mode=None,
                    mem_capacity_bytes=mem_capacity_bytes,
                )
            logger.info("未启用缓存（cache_mode 未设置，enable_ssd_caching=False），直接返回基础存储")
            return base_storage

        mode = str(cache_mode).lower()
        if mode == "none":
            logger.info("缓存模式: none（直接使用基础存储）")
            return base_storage
        elif mode == "only_mem":
            logger.info("缓存模式: only_mem（仅内存缓存，容量=%dB）", mem_capacity_bytes)
            return CachingStorage(
                base_storage,
                cache_dir,
                enable_prefetch,
                cache_mode="only_mem",
                mem_capacity_bytes=mem_capacity_bytes,
            )
        elif mode == "mem_and_ssd":
            logger.info(
                "缓存模式: mem_and_ssd（内存+SSD，SSD目录=%s，容量=%dB）",
                cache_dir,
                mem_capacity_bytes,
            )
            return CachingStorage(
                base_storage,
                cache_dir,
                enable_prefetch,
                cache_mode="mem_and_ssd",
                mem_capacity_bytes=mem_capacity_bytes,
            )
        else:
            logger.warning("未知 cache_mode: %s，返回基础存储", cache_mode)
            return base_storage
