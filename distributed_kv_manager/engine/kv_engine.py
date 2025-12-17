import os
import torch
import logging
import json
import hashlib
import time
import struct
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple
from types import SimpleNamespace
from .base import DistributedKVEngineBase, StoreStatus, RetrieveStatus
from distributed_kv_manager.metadata.etcd import KVMetadataManager, KVMetadata
from distributed_kv_manager.metadata.metadata_cache import MetadataCache
from distributed_kv_manager.storage.factory import StorageFactory
from distributed_kv_manager.storage.base import AbstractStorage


# ------------------ Logger 配置 ------------------ #
logger = logging.getLogger("KVEngine")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
ch.setFormatter(formatter)
logger.addHandler(ch)

class KVEngine(DistributedKVEngineBase):
    """
    分布式 KV 引擎，封装KV存取。
    """

    def __init__(self, config):
        self.rank = getattr(config, "rank", 0)
        self.local_rank = getattr(config, "local_rank", 0)
        self.config = config
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._futures = []
        # 最近一次检索统计（便于外部读取验证节省量）
        self._last_retrieve_stats: dict = {}
        # 最小存储阈值（按 token 数），小于该阈值的序列不落盘
        extra_cfg = getattr(getattr(config, "kv_transfer_config", None), "extra_config", None)
        try:
            self._min_store_tokens = int(extra_cfg.get("min_store_tokens", 16)) if isinstance(extra_cfg, dict) else 16
        except Exception:
            self._min_store_tokens = 16

        # 使用存储工厂创建存储实例
        _storage_inst = StorageFactory.create_storage(config)
        if _storage_inst is None:
            raise RuntimeError("StorageFactory.create_storage 返回 None")
        self._storage = _storage_inst
        logger.info(f"创建的存储实例类型: {type(self._storage)}")
        self.storage_dir = getattr(config.kv_transfer_config, "storage_dir", "/kvcache")

    # metadata
        endpoints = getattr(config.kv_transfer_config, "etcd_endpoints", ["127.0.0.1:2379"])
        self._meta = KVMetadataManager(endpoints=endpoints, prefix="/kvmeta")

        # metadata cache
        self._meta_cache = MetadataCache(meta_manager=self._meta)
        
        # cleanup manager
        # 读取配置的清理间隔时间，默认1小时
        cleanup_interval = getattr(config.kv_transfer_config, "cleanup_interval", 3600)
        from distributed_kv_manager.metadata.cleanup import KVCleanupManager
        self._cleanup_manager = KVCleanupManager(self._meta, cleanup_interval, self._storage)
        self._cleanup_manager.start()

        # debug dump config
        env_debug = os.getenv("KV_DEBUG_DUMP", "")
        env_debug_on = str(env_debug).lower() not in ("", "0", "false", "none")
        self._debug_dump: bool = bool(getattr(config.kv_transfer_config, "enable_debug_dump", False) or env_debug_on)
        self._debug_dump_file: Optional[str] = getattr(
            config.kv_transfer_config, "debug_dump_file", None
        ) or os.getenv("KV_DEBUG_DUMP_FILE", None)
        if self._debug_dump:
            self._maybe_setup_debug_file_handler()

    # ------------------ 维护/清理 ------------------ #
    def clear_cache(
        self,
        session_id: Optional[bytes] = None,
        layer_id: Optional[int] = None,
        contains: Optional[str] = None,
        delete_storage: bool = True,
    ) -> dict:
        """清理 ETCD 下的元数据缓存，并可选删除对应存储文件。

        过滤条件（全部满足才清理）：
        - session_id: 仅清理匹配该会话ID的项（与元数据中的 session_id 比较，bytes）。
        - layer_id: 仅清理指定层ID的项。
        - contains: 仅清理 file_path 中包含该子串的项（例如特定 token 哈希）。

        返回统计信息：{"scanned":N, "matched":M, "deleted_etcd":X, "deleted_files":Y}
        """
        stats = {"scanned": 0, "matched": 0, "deleted_etcd": 0, "deleted_files": 0}
        try:
            # 扫描全部完整键（带前缀）
            full_keys = self._meta.scan_all_metadata_keys()
            stats["scanned"] = len(full_keys)
            if not full_keys:
                logger.info("[clear_cache] 没有可清理的元数据键")
                return stats

            # 前缀（用于构造相对键）
            etcd_prefix = getattr(self._meta, "prefix", "/kvmeta")
            if not isinstance(etcd_prefix, str):
                etcd_prefix = str(etcd_prefix)

            for full_key in full_keys:
                try:
                    meta = self._meta.get_metadata_by_full_key(full_key)
                except Exception:
                    meta = None
                if meta is None:
                    continue

                # 过滤条件
                if session_id is not None and getattr(meta, "session_id", None) != session_id:
                    continue
                if layer_id is not None and getattr(meta, "layer_id", None) != int(layer_id):
                    continue
                if contains is not None and contains not in getattr(meta, "file_path", ""):
                    continue

                stats["matched"] += 1

                # 删除存储文件
                if delete_storage:
                    try:
                        if self._storage_delete(getattr(meta, "file_path", "")):
                            stats["deleted_files"] += 1
                    except Exception as e:
                        logger.warning(f"[clear_cache] 删除存储文件失败: {meta.file_path}: {e}")

                # 删除 etcd 键（转成相对键）
                try:
                    # full_key 形如 "/kvmeta/<rel>"，转换为相对键
                    rel_key = full_key
                    if rel_key.startswith(etcd_prefix + "/"):
                        rel_key = rel_key[len(etcd_prefix) + 1 :]
                    rel_key = rel_key.lstrip("/")
                    self._meta.delete_metadata(rel_key, replicate=True)
                    stats["deleted_etcd"] += 1
                except Exception as e:
                    logger.warning(f"[clear_cache] 删除ETCD键失败: {full_key}: {e}")

            logger.info(
                f"[clear_cache] 扫描={stats['scanned']} 匹配={stats['matched']} 删除ETCD={stats['deleted_etcd']} 删除文件={stats['deleted_files']}"
            )
            return stats
        except Exception as e:
            logger.error(f"[clear_cache] 清理过程失败: {e}", exc_info=True)
            return stats

    def _storage_delete(self, file_path: str) -> bool:
        """尽力删除存储中的文件，兼容多种后端包装。

        - 若存储实现 delete(file_path)，直接调用。
        - 若为分层缓存包装（CachingStorage），尝试清理内存/SSD，并调用后端 delete。
        - 不抛异常，返回是否删除了后端文件（缓存删除不计数）。
        """
        if not file_path:
            return False
        st = self._storage
        # 直接支持 delete
        delete_fn = getattr(st, "delete", None)
        if callable(delete_fn):
            try:
                return bool(delete_fn(file_path))
            except Exception as e:
                logger.warning(f"[storage_delete] 后端删除失败 {file_path}: {e}")
                return False
        # 可能是 CachingStorage：尝试删除缓存及后端
        deleted_backend = False
        try:
            ssd = getattr(st, "ssd_cache", None)
            if ssd is not None and hasattr(ssd, "delete"):
                try:
                    ssd.delete(file_path)
                except Exception:
                    pass
            mem = getattr(st, "mem_cache", None)
            if mem is not None and hasattr(mem, "delete"):
                try:
                    mem.delete(file_path)
                except Exception:
                    pass
            backend = getattr(st, "storage_backend", None)
            if backend is not None:
                backend_delete = getattr(backend, "delete", None)
                if callable(backend_delete):
                    try:
                        deleted_backend = bool(backend_delete(file_path))
                    except Exception as e:
                        logger.warning(f"[storage_delete] 底层后端删除失败 {file_path}: {e}")
        except Exception:
            pass
        return deleted_backend

    # ------------------ KV 存储/检索 ------------------ #
    def should_store(self, model_input, **kwargs) -> StoreStatus:
        return StoreStatus.STORED

    def should_retrieve(self, model_input, **kwargs) -> RetrieveStatus:
        """
        判断 KV 是否命中缓存，通过元数据缓存确认：
        - status == 1 表示写入完成，可以命中
        - 其他情况视为 MISS
        """
        input_tokens = model_input.input_tokens
        seq_lens = model_input.attn_metadata.seq_lens
        stable_key_arg = kwargs.get("stable_key")

        for seq_idx, seq_len in enumerate(seq_lens):
            start_pos = sum(seq_lens[:seq_idx])
            end_pos = start_pos + seq_len
            current_tokens = input_tokens[start_pos:end_pos]
            session_id = getattr(model_input, "session_id", None)
            layer_id = getattr(model_input, "layer_id", None)
            stable_key_override = stable_key_arg
            session_str = session_id.hex() if isinstance(session_id, (bytes, bytearray)) else str(session_id or "session_0000")
            default_stable_key = f"kv_{session_str}_layer_{layer_id or 0}_seq{seq_idx}.pt"
            stable_key = stable_key_override or default_stable_key
            file_path = self._make_key(current_tokens, session_id, layer_id, stable_key=stable_key)

            meta = self._meta_cache.get_metadata(
                key=file_path,
                layer_id=layer_id,
                session_id=session_id,
            )

            if meta is None or meta.status != 1:
                return RetrieveStatus.MISS
            try:
                kv_bytes_exist = self._storage_download_kv_bytes(file_path)
                if kv_bytes_exist is None:
                    return RetrieveStatus.MISS
                info_exist = self._storage.extract_payload_info(kv_bytes_exist)
                pm_exist = info_exist.get("payload_meta", {}) if isinstance(info_exist, dict) else {}
                exist_slots_len = int(pm_exist.get("slots_len", 0) or 0)
                if exist_slots_len < int(getattr(self, "_min_store_tokens", 16)):
                    return RetrieveStatus.MISS
                try:
                    expected_hash = self._tensor_hash(current_tokens[:exist_slots_len] if exist_slots_len > 0 else current_tokens)
                    stored_hash = str(pm_exist.get("tokens_hash", "")) if isinstance(pm_exist, dict) else ""
                    if not stored_hash or stored_hash != expected_hash:
                        return RetrieveStatus.MISS
                except Exception:
                    return RetrieveStatus.MISS
            except Exception:
                return RetrieveStatus.MISS
            if meta.is_expired():
                return RetrieveStatus.MISS

        return RetrieveStatus.HIT

    def store_kv(
        self,
        model_config,
        parallel_config,
        transfer_config,
        model_executable,
        model_input,
        kv_caches,
        store_status,
        hidden_or_intermediate_states=None,
        **kwargs
        ):
        """基于元数据缓存的两阶段提交写入 KV 缓存（异步提交）。

        Args:
            hidden_or_intermediate_states: 兼容旧接口的占位参数（当前未使用）。
        """
        _ = hidden_or_intermediate_states  # 占位以保持接口兼容
        
        # 显式参数传递 override
        stable_key_arg = kwargs.get("stable_key")
        
        input_tokens = model_input.input_tokens
        seq_lens = model_input.attn_metadata.seq_lens
        slot_mapping = model_input.attn_metadata.slot_mapping  # 保持二维形状
        num_layers = len(kv_caches)

        for seq_idx, seq_len in enumerate(seq_lens):
            start_pos = sum(seq_lens[:seq_idx])
            end_pos = start_pos + seq_len
            current_tokens = input_tokens[start_pos:end_pos]
            # 获取session_id和layer_id
            session_id = getattr(model_input, "session_id", None)
            layer_id = getattr(model_input, "layer_id", None)
            # 分块场景下，为同一序列使用稳定键，避免因切片不同导致 key 不一致
            stable_key_override = stable_key_arg or getattr(model_input, "stable_key", None)
            session_str = session_id.hex() if isinstance(session_id, (bytes, bytearray)) else str(session_id or "session_0000")
            try:
                seq_hash = self._tensor_hash(current_tokens)
            except Exception:
                seq_hash = "empty"
            stable_key = stable_key_override or f"kv_{session_str}_layer_{layer_id or 0}_seq{seq_idx}_{seq_hash}.pt"
            stable_key_prefix = (stable_key[:-3] if isinstance(stable_key, str) and stable_key.endswith(".pt") else stable_key)
            file_path = self._make_key(current_tokens, session_id, layer_id, stable_key=stable_key)

            # ------------------ 第一步：查询元数据缓存（仅记录，是否跳过/升级在计算长度后决定） ------------------ #
            existing_meta = self._meta_cache.get_metadata(
                key=file_path,
                layer_id=layer_id,
                session_id=session_id,
            )

            # ------------------ 第三步：准备 KV 数据 ------------------ #
            # 先解析槽位（仅用于持久化/恢复），存储时不以槽位索引读取 4D KV，避免越界
            resolved_slots = self._resolve_sequence_slots(seq_idx, seq_len, slot_mapping, file_path, start_pos)
            if resolved_slots.numel() == 1 and seq_len > 1:
                logger.warning("序列 %s 槽位映射仅包含单个元素，回退为顺序索引 0..%d", file_path, seq_len - 1)
                resolved_slots = torch.arange(seq_len, device=resolved_slots.device)

            # debug dump: store context
            self._dump_store_context(
                seq_idx=seq_idx,
                start_pos=start_pos,
                end_pos=end_pos,
                seq_len=seq_len,
                file_path=file_path,
                input_tokens=input_tokens,
                current_tokens=current_tokens,
                slot_mapping=slot_mapping,
                current_slots=resolved_slots,
            )

            # 第一遍：计算各层可用的最短长度，确保拼接不报错
            min_take_len = seq_len
            for layer_idx in range(num_layers):
                kv_cache = kv_caches[layer_idx]
                key_cache, value_cache = self.split_kv_cache(kv_cache)
                if key_cache.dim() == 4:
                    if seq_idx >= key_cache.shape[0]:
                        logger.error("seq_idx=%d 超过 batch 维度大小=%d", seq_idx, key_cache.shape[0])
                        candidate = 0
                    else:
                        candidate = min(seq_len, int(key_cache[seq_idx].shape[0]))
                elif key_cache.dim() == 3:
                    total_tokens = int(key_cache.shape[0])
                    candidate = min(seq_len, max(0, total_tokens - int(start_pos)))
                else:
                    logger.error("无法识别的KV缓存形状: %s", key_cache.shape)
                    candidate = 0
                min_take_len = min(min_take_len, candidate)

            if min_take_len <= 0:
                logger.warning("序列 %s 在各层无可用 KV token，跳过存储", file_path)
                continue
            if int(min_take_len) < int(getattr(self, "_min_store_tokens", 16)):
                logger.debug("序列 %s 可用长度(%d)小于最小存储阈值(%d)，跳过存储", file_path, int(min_take_len), int(getattr(self, "_min_store_tokens", 16)))
                continue

            # 先初始化 persisted_slots，稍后根据已存在长度进行重切片
            persisted_slots = resolved_slots[:min_take_len]

            # 如果存在已提交的旧条目，检查是否需要“升级”（从更短长度升级到更长）
            existing_slots_len = None
            if existing_meta is not None and existing_meta.status == 1:
                try:
                    kv_bytes_exist = self._storage_download_kv_bytes(file_path)
                    if kv_bytes_exist is not None:
                        info_exist = self._storage.extract_payload_info(kv_bytes_exist)
                        pm_exist = info_exist.get("payload_meta", {}) if isinstance(info_exist, dict) else {}
                        existing_slots_len = int(pm_exist.get("slots_len", 0) or 0)
                except Exception:
                    existing_slots_len = None
            persist_start = int(existing_slots_len or 0)
            persist_len = max(0, min(int(min_take_len), int(seq_len - persist_start)))
            if persist_len <= 0:
                logger.debug(
                    "序列 %s persist_len<=0 跳过存储, start=%d seq_len=%d min_take=%d",
                    file_path, persist_start, seq_len, min_take_len
                )
                continue
            persisted_slots = resolved_slots[persist_start:persist_start + persist_len]

            # 第二遍：按统一长度收集各层 KV
            all_keys, all_values = [], []
            for layer_idx in range(num_layers):
                kv_cache = kv_caches[layer_idx]
                key_cache, value_cache = self.split_kv_cache(kv_cache)
                if key_cache.dim() < 3:
                    continue
                num_heads = int(key_cache.shape[-2])
                head_dim = int(key_cache.shape[-1])
                import torch as _torch
                gseq = _torch.arange(int(persist_len), dtype=_torch.long, device=current_tokens.device)
                if key_cache.dim() == 4:
                    try:
                        selected_k = key_cache[seq_idx][gseq]
                        selected_v = value_cache[seq_idx][gseq]
                    except Exception as e:
                        logger.error("store gather dim4 failed: %s", e, exc_info=True)
                        continue
                else:
                    global_idx = gseq + int(start_pos) + int(persist_start)
                    try:
                        selected_k = key_cache[global_idx]
                        selected_v = value_cache[global_idx]
                    except Exception as e:
                        logger.error("store gather dim3 failed: %s", e, exc_info=True)
                        continue
                all_keys.append(selected_k.unsqueeze(0))
                all_values.append(selected_v.unsqueeze(0))

            all_keys = torch.cat(all_keys, dim=0) if all_keys else torch.tensor([])
            all_values = torch.cat(all_values, dim=0) if all_values else torch.tensor([])
            self._dbg(f"[store] concat keys_shape={None if all_keys.numel()==0 else tuple(all_keys.shape)} values_shape={None if all_values.numel()==0 else tuple(all_values.shape)}")

            # 若未成功收集任何一层的KV（例如索引越界导致全部跳过），直接放弃本序列的存储，避免写入空payload
            if all_keys.numel() == 0 or all_values.numel() == 0:
                logger.warning("序列 %s 未采集到有效KV数据，跳过存储与提交状态", file_path)
                continue

            # tokens/roi 与存储的 KV 对齐
            selected_tokens = current_tokens[persist_start:persist_start + persist_len]
            roi = torch.ones(persist_len, dtype=torch.bool, device=selected_tokens.device)

            # ------------------ 第四步：写入元数据缓存（锁定状态），再异步写入 KV 并更新元数据 ------------------ #
            # 读取配置的过期时间，默认1天
            expire_time = getattr(self.config.kv_transfer_config, "kv_expire_time", 86400)

            # 若连接器传递了分块位置信息（token_offset），使用其覆盖元数据 token_idx 的起止范围。
            # 这样检索聚合可以依据绝对 offset 识别多个块，而不是都从 0 开始。
            connector_payload_meta = getattr(model_input, "payload_meta", None)
            override_offset = None
            try:
                if isinstance(connector_payload_meta, dict) and "token_offset" in connector_payload_meta:
                    # token_offset 为全局序列中的起始绝对位置
                    override_offset = int(connector_payload_meta["token_offset"])
            except Exception:
                override_offset = None
            meta_start_pos = (override_offset if override_offset is not None else (start_pos + persist_start))
            meta_end_pos = meta_start_pos + persist_len

            # 若已存在元数据，则复用并短暂置为写入中；否则新建
            if existing_meta is not None:
                meta = existing_meta
                meta.status = 0  # 正在写入（避免并发读取不一致）
                meta.last_access = int(time.time())
                # 更新 token_idx：使用覆盖后的绝对范围
                try:
                    meta.token_idx = f"{meta_start_pos}-{meta_end_pos}"
                except Exception:
                    pass
            else:
                meta = KVMetadata(
                    session_id=session_id if session_id is not None else b"session_0000",
                    layer_id=layer_id if layer_id is not None else 0,
                    token_idx=f"{meta_start_pos}-{meta_end_pos}",
                    file_path=file_path,
                    file_size=0,
                    create_time=int(time.time()),
                    last_access=int(time.time()),
                    expire_time=expire_time,
                    replica_locations=[b"" for _ in range(3)],
                    status=0,
                    schema_version=1,
                    ext_flags=0,
                    ext_data=b"",
                    ext_data_len=0,
                )
            # 使用缓存 put，自动写入 Pool1/2/3 及 etcd
            self._meta_cache.put_metadata(meta)
            # 使用辅助类确保变量正确捕获
            class InsertAndUpdateTask:
                def __init__(
                    self,
                    engine,
                    seq_idx,
                    file_path,
                    all_keys,
                    all_values,
                    current_tokens,
                    roi,
                    current_slots,
                    meta,
                    connector_payload_meta=None,
                ):
                    self.engine = engine
                    self.seq_idx = seq_idx
                    self.file_path = file_path
                    self.all_keys = all_keys
                    self.all_values = all_values
                    self.current_tokens = current_tokens
                    self.roi = roi
                    self.current_slots = current_slots
                    self.meta = meta
                    self.connector_payload_meta = connector_payload_meta
                
                def __call__(self):
                    try:
                        logger.debug(f"开始处理序列 {self.seq_idx}: {self.file_path}")
                        try:
                            self.all_keys = self.all_keys.detach().cpu()
                            self.all_values = self.all_values.detach().cpu()
                            self.current_tokens = self.current_tokens.detach().cpu()
                            self.current_slots = self.current_slots.detach().cpu()
                        except Exception:
                            pass
                        
                        # 构造 payload_meta
                        payload_meta = {
                            "schema_version": 1,
                            "tokens_hash": self.engine._tensor_hash(self.current_tokens),
                            "num_layers": int(self.all_keys.shape[0]) if self.all_keys.numel() > 0 else 0,
                            "kv_dtype": str(self.all_keys.dtype) if self.all_keys.numel() > 0 else "unknown",
                            "kv_tail_shape": list(self.all_keys.shape[2:]) if self.all_keys.numel() > 0 else [],
                            "slots_len": int(self.current_slots.numel()),
                            "token_offset": meta_start_pos,
                        }

                        # If the connector attached a payload_meta to the model_input
                        # (e.g. token_offset/block_index hints), merge it here so the
                        # stored payload contains that diagnostic information.
                        try:
                            if isinstance(self.connector_payload_meta, dict):
                                # do not let connector override critical engine fields
                                for k, v in self.connector_payload_meta.items():
                                    if k in ("schema_version", "tokens_hash", "num_layers", "kv_dtype", "kv_tail_shape", "slots_len"):
                                        # skip core engine-managed keys
                                        continue
                                    payload_meta[k] = v
                        except Exception:
                            pass

                        try:
                            kvb_exist = self.engine._storage_download_kv_bytes(self.file_path)
                            if kvb_exist is not None:
                                k_old, v_old = self.engine._storage.unpack_kv_data(kvb_exist)
                            else:
                                k_old, v_old = None, None
                        except Exception:
                            k_old, v_old = None, None
                        if k_old is not None and v_old is not None:
                            import torch as _torch
                            old_len = int(k_old.shape[1]) if k_old.dim() >= 2 else 0
                            start_off = 0
                            try:
                                pm = self.engine._storage.extract_payload_info(kvb_exist).get("payload_meta", {}) if kvb_exist is not None else {}
                                start_off = int(pm.get("token_offset", 0) or 0)
                            except Exception:
                                start_off = 0
                            new_off = int(payload_meta.get("token_offset", start_off))
                            end_len = max(old_len, new_off + int(self.current_slots.numel()))
                            num_layers = int(self.all_keys.shape[0])
                            num_heads = int(self.all_keys.shape[-2])
                            head_dim = int(self.all_keys.shape[-1])
                            k_merged = _torch.empty((num_layers, end_len, num_heads, head_dim), dtype=self.all_keys.dtype)
                            v_merged = _torch.empty((num_layers, end_len, num_heads, head_dim), dtype=self.all_values.dtype)
                            k_merged[:, :old_len] = k_old.to(dtype=self.all_keys.dtype)
                            v_merged[:, :old_len] = v_old.to(dtype=self.all_values.dtype)
                            seg_len = int(self.current_slots.numel())
                            if seg_len > 0:
                                k_merged[:, new_off:new_off+seg_len] = self.all_keys
                                v_merged[:, new_off:new_off+seg_len] = self.all_values
                            final_keys = k_merged
                            final_values = v_merged
                            payload_meta["slots_len"] = end_len
                        else:
                            final_keys = self.all_keys
                            final_values = self.all_values
                            payload_meta["slots_len"] = int(self.current_slots.numel())
                        self.engine._storage_insert(
                            self.file_path,
                            final_keys,
                            final_values,
                            self.current_tokens,
                            self.roi,
                            self.current_slots,
                            payload_meta,
                        )
                        logger.debug(f"序列 {self.seq_idx} 存储完成")
                        
                        # 写入完成后更新元数据
                        keys_size = self.all_keys.numel() * self.all_keys.element_size() if self.all_keys.numel() > 0 else 0
                        values_size = self.all_values.numel() * self.all_values.element_size() if self.all_values.numel() > 0 else 0
                        self.meta.file_size = keys_size + values_size
                        self.meta.status = 1  # 写入完成
                        self.meta.last_access = int(time.time())
                        
                        # 使用缓存 put 更新元数据
                        self.engine._meta_cache.put_metadata(self.meta)
                        logger.debug(f"序列 {self.seq_idx} 元数据更新完成，状态: {self.meta.status}")
                        
                    except Exception as e:
                        logger.error(f"存储序列 {self.seq_idx} ({self.file_path}) 失败: {e}", exc_info=True)

            # 创建任务实例并提交
            # pass through any connector-provided payload_meta (diagnostic)
            connector_payload_meta = None
            try:
                connector_payload_meta = getattr(model_input, "payload_meta", None)
            except Exception:
                connector_payload_meta = None

            task = InsertAndUpdateTask(
                self,
                seq_idx,
                file_path,
                all_keys,
                all_values,
                selected_tokens,
                roi,
                persisted_slots,
                meta,
                connector_payload_meta=connector_payload_meta,
            )
            future = self._executor.submit(task)
            self._futures.append(future)

    def retrieve_kv(self, model_executable, model_input, kv_caches, retrieve_status, **kwargs):
        """从存储中恢复 KV 缓存，并提供结果。"""
        # 显式参数传递 override
        stable_key_arg = kwargs.get("stable_key")
        
        input_tokens = model_input.input_tokens
        seq_lens = model_input.attn_metadata.seq_lens
        slot_mapping = model_input.attn_metadata.slot_mapping  # 保持二维形状
        num_layers = len(kv_caches)

        # 允许在 MISS 场景下也尝试按块聚合恢复（部分恢复不会 bypass）。
        if retrieve_status != RetrieveStatus.HIT:
            logger.debug("检索状态为 MISS，尝试按块聚合进行部分恢复")

        logger.debug(f"开始检索KV，序列数量: {len(seq_lens)}")

        # 标记全部序列是否都成功恢复（用于决定是否 bypass 前向）
        recovered_sequences = 0
        # 收集统计信息
        total_tokens = int(sum(int(x) for x in seq_lens))
        restored_per_seq: list[int] = []

        slot_full_override = None

        for seq_idx, seq_len in enumerate(seq_lens):
            start_pos = sum(seq_lens[:seq_idx])
            end_pos = start_pos + seq_len

            current_tokens = input_tokens[start_pos:end_pos]
            session_id = getattr(model_input, "session_id", None)
            layer_id = getattr(model_input, "layer_id", None)
            stable_key_override = stable_key_arg or getattr(model_input, "stable_key", None)
            # normalize default session/layer to align with store path generation
            session_eff = session_id if session_id is not None else b"session_0000"
            layer_eff = layer_id if layer_id is not None else 0
            session_str = (session_eff.hex() if isinstance(session_eff, (bytes, bytearray)) else str(session_eff or "session_0000"))
            # 与 store_kv 对齐：若未显式传入稳定键，则使用基于序号的稳定键
            try:
                seq_hash = self._tensor_hash(current_tokens)
            except Exception:
                seq_hash = "empty"
            stable_key = stable_key_override or f"kv_{session_str}_layer_{layer_eff}_seq{seq_idx}_{seq_hash}.pt"
            file_path = self._make_key(current_tokens, session_eff, layer_eff, stable_key=stable_key)
            self._dbg(
                f"[retrieve] seq_idx={seq_idx} key={file_path} seq_len={seq_len} "
                f"start_pos={start_pos} session={session_str} layer={layer_eff} "
                f"stable_key={stable_key} override={stable_key_override}"
            )
            # 用 stable_key 前缀约束块检索，避免不同输入之间互相聚合
            stable_key_prefix = (
                stable_key[:-3] if isinstance(stable_key, str) and stable_key.endswith(".pt") else stable_key
            )

            # 跳过基础稳定键的哈希/槽位校验，直接尝试聚合块
            meta = None
            kv_bytes = None
            payload_info = {}
            persisted_slots = None
            payload_meta = {}
            expected_layers = None
            tokens_hash = None

            # 基础文件的收集改为统一通过 _collect_from_file

            current_slots = self._resolve_sequence_slots(seq_idx, seq_len, slot_mapping, file_path, start_pos)
            self._dump_slots_context("retrieve.pre", seq_idx, seq_len, start_pos, file_path, slot_mapping, current_slots)
            if current_slots.numel() == 0:
                logger.warning("序列 %s 的有效槽位数为0，跳过", file_path)
                continue
            if current_slots.numel() == 1 and seq_len > 1:
                logger.warning("序列 %s 槽位映射仅一个元素，回退为顺序索引 0..%d", file_path, seq_len - 1)
                current_slots = torch.arange(seq_len, device=current_slots.device)
            self._dbg(f"[retrieve] current_slots_len={int(current_slots.numel())} max={None if current_slots.numel()==0 else int(current_slots.max().item())}")

            # 若存在持久化的 slot_mapping，则优先使用
            if persisted_slots is not None:
                try:
                    persisted_slots = persisted_slots.reshape(-1)
                    if persisted_slots.numel() != current_slots.numel():
                        limit = min(persisted_slots.numel(), current_slots.numel())
                        persisted_slots = persisted_slots[:limit]
                        current_slots = current_slots[:limit]
                        if limit == 0:
                            continue
                    slots_to_use = persisted_slots.to(current_slots.device)
                except Exception:
                    slots_to_use = current_slots
            else:
                slots_to_use = current_slots
            self._dbg(f"[retrieve] slots_to_use_len={int(slots_to_use.numel())}")
            self._dump_slots_context("retrieve.use", seq_idx, seq_len, start_pos, file_path, slot_mapping, slots_to_use)

            # 轻量校验：层数与token哈希（存在时）
            if expected_layers is not None and int(expected_layers) != len(kv_caches):
                logger.warning(
                    "层数不一致: payload=%s, runtime=%s; 跳过该序列",
                    expected_layers,
                    len(kv_caches),
                )
                continue
            if tokens_hash is not None:
                # 使用与存储一致的长度计算哈希：
                # 优先使用持久化的 input_tokens 长度；否则使用 payload 中的 slots_len；
                # 若仍不可用，则回退为 slots_to_use 的长度。
                try:
                    if isinstance(payload_info, dict) and "input_tokens" in payload_info:
                        tokens_len = int(getattr(payload_info["input_tokens"], "numel", lambda: 0)())
                    else:
                        tokens_len = int(payload_meta.get("slots_len", 0) or 0)
                    if tokens_len <= 0:
                        tokens_len = int(slots_to_use.numel())
                    cur_hash = self._tensor_hash(current_tokens[:tokens_len])
                except Exception:
                    # 回退：使用完整序列（可能导致误判）
                    cur_hash = self._tensor_hash(current_tokens)
                if tokens_hash != cur_hash:
                    logger.warning("token哈希不一致，跳过该序列: %s", file_path)
                    continue

            # 存储的实际可恢复长度（第二维）
            # 尝试按块聚合（单文件或多文件）进行恢复：
            # - 首先收集所有候选的已提交元数据条目（同一 session_id 与 layer_id），
            #   并从每个文件的 payload_meta 中读取 token_offset/block_size，若存在则用于排序和写回。
            # - 如果只有单个文件也可走同一路径。
            try:
                blocks = []  # list of (offset, key_tensor, value_tensor, block_len, block_slots)
                def _collect_from_file(rel_path, meta_obj=None):
                    try:
                        kvb = self._storage_download_kv_bytes(rel_path)
                        if kvb is None:
                            try:
                                import os as _os
                                import io as _io
                                for root in ("/kvcache_v1/local", "/kvcache_v1/remote"):
                                    p = f"{root}/{rel_path}"
                                    try:
                                        if _os.path.exists(p):
                                            with open(p, "rb") as f:
                                                kvb = _io.BytesIO(f.read()).getvalue()
                                            break
                                    except Exception:
                                        continue
                            except Exception:
                                kvb = None
                            if kvb is None:
                                self._dbg(f"[retrieve] skip {rel_path}: download None")
                                return
                        info = self._storage.extract_payload_info(kvb)
                        pm = info.get("payload_meta", {}) if isinstance(info, dict) else {}
                        try:
                            nonlocal tokens_hash, expected_layers, payload_meta, payload_info
                            if rel_path == stable_key:
                                try:
                                    t_h = pm.get("tokens_hash", None)
                                    tokens_hash = str(t_h) if t_h is not None else None
                                except Exception:
                                    tokens_hash = tokens_hash
                                try:
                                    n_l = pm.get("num_layers", None)
                                    expected_layers = int(n_l) if n_l is not None else None
                                except Exception:
                                    expected_layers = expected_layers
                                payload_meta = pm if isinstance(pm, dict) else {}
                                payload_info = info if isinstance(info, dict) else {}
                        except Exception:
                            pass
                        # prefer connector-provided token_offset (absolute) and normalize
                        # to the current sequence-relative offset
                        if isinstance(pm, dict) and "token_offset" in pm:
                            try:
                                abs_off = int(pm.get("token_offset", 0))
                            except Exception:
                                abs_off = 0
                            # normalize to current sequence [0, seq_len)
                            try:
                                offset = int(abs_off) - int(start_pos)
                            except Exception:
                                offset = None
                        else:
                            # try to infer from metadata token_idx if available (absolute range "start-end")
                            offset = None
                            if meta_obj is not None:
                                try:
                                    toks = str(getattr(meta_obj, "token_idx", "")).split("-")
                                    if len(toks) == 2:
                                        offset = int(toks[0]) - int(start_pos)
                                except Exception:
                                    offset = None
                        # fallback: treat base full file as offset 0
                        if offset is None:
                            try:
                                if rel_path == stable_key:
                                    offset = 0
                                elif isinstance(rel_path, str) and stable_key and rel_path.endswith(".pt"):
                                    # any file that starts with the base stable_key prefix and is not a block is considered base at offset 0
                                    if rel_path.startswith(stable_key_prefix) and ("_blk" not in rel_path):
                                        offset = 0
                            except Exception:
                                pass
                        if offset is None:
                            # cannot determine offset -> skip
                            self._dbg(f"[retrieve] skip {rel_path}: no offset")
                            return
                        # bounds check & clamp
                        try:
                            if offset < 0:
                                # if the file starts before this sequence window, trim effective block later
                                pass
                            if offset >= seq_len:
                                return
                        except Exception:
                            return
                        try:
                            k_tensor, v_tensor = self._storage.unpack_kv_data(kvb)
                        except Exception:
                            return
                        try:
                            if getattr(k_tensor, "dim", lambda: 0)() >= 2:
                                shape = getattr(k_tensor, "shape", None)
                                if shape is not None and len(shape) > 1:
                                    block_len = int(shape[1])
                                else:
                                    block_len = 0
                            else:
                                block_len = 0
                        except Exception:
                            block_len = 0
                        try:
                            if int(block_len) < int(getattr(self, "_min_store_tokens", 16)):
                                return
                        except Exception:
                            return
                        if block_len <= 0:
                            self._dbg(f"[retrieve] skip {rel_path}: block_len<=0")
                            return
                        # cap block_len to remaining seq_len, trimming for negative offsets
                        try:
                            eff_start = max(0, int(offset))
                            # if offset < 0, we trim the front of the block accordingly
                            front_trim = 0 if offset >= 0 else int(-offset)
                            block_len = max(0, min(int(block_len - front_trim), int(seq_len - eff_start)))
                            if block_len <= 0:
                                return
                            # If we trimmed the front, slice the stored tensors accordingly
                            if front_trim > 0:
                                try:
                                    k_tensor = k_tensor[:, front_trim: front_trim + block_len]
                                    v_tensor = v_tensor[:, front_trim: front_trim + block_len]
                                except Exception:
                                    self._dbg(f"[retrieve] skip {rel_path}: trim fail")
                                    return
                        except Exception:
                            return
                        # 放宽 token hash 校验，避免因统一前缀键导致误判
                        # （当前连接器统一使用 kv_session_0000_layer_0 前缀）
                        # Prefer per-file persisted slot mapping if present
                        block_slots = None
                        try:
                            if isinstance(info, dict) and "slot_mapping" in info:
                                s = info["slot_mapping"]
                                if hasattr(s, 'dim') and s.dim() > 1:
                                    s = s.reshape(-1)
                                import torch as _torch
                                s = s.to(dtype=_torch.long)
                                if int(s.numel()) != int(block_len):
                                    # trim or pad fallback to ensure exact length
                                    if int(s.numel()) > int(block_len):
                                        s = s[:int(block_len)]
                                    else:
                                        s = _torch.arange(int(block_len), dtype=_torch.long)
                                block_slots = s
                        except Exception:
                            block_slots = None
                        # store with sequence-relative offset (effective start after clamp)
                        blocks.append((eff_start, k_tensor, v_tensor, block_len, block_slots))
                        self._dbg(f"[retrieve] collected block rel={rel_path} offset={eff_start} len={block_len} slots_len={None if block_slots is None else block_slots.numel()}")
                    except Exception:
                        return
                _collect_from_file(stable_key, meta_obj=None)

                # 1) 如果存在完全匹配的元数据（即基于完整 token hash），优先收集
                # 1) 扫描所有元数据键，收集属于同一 session/layer 的块条目
                try:
                    full_keys = self._meta.scan_all_metadata_keys()
                except Exception:
                    self._dbg("[retrieve] scan_all_metadata_keys failed")
                    full_keys = []
                self._dbg(f"[retrieve] scan meta keys count={len(full_keys) if full_keys is not None else 0}")

                # 当上游没有显式提供 session_id / layer_id 时，聚合应当与键生成规则保持一致：
                # - session_id 缺省等同于 b"session_0000"
                # - layer_id 缺省等同于 0
                session_id_effective = session_id if session_id is not None else b"session_0000"
                layer_id_effective = layer_id if layer_id is not None else 0

                etcd_prefix = getattr(self._meta, "prefix", "/kvmeta")
                for full_key in full_keys:
                    try:
                        # full_key 带前缀，转换为相对 file_path
                        rel = full_key
                        if rel.startswith(etcd_prefix + "/"):
                            rel = rel[len(etcd_prefix) + 1 :]
                        rel = rel.lstrip("/")
                        m = self._meta.get_metadata_by_full_key(full_key)
                        if m is None:
                            continue
                        if getattr(m, "status", None) != 1:
                            continue
                        # 使用规范化后的会话与层编号进行过滤
                        if getattr(m, "session_id", None) != session_id_effective:
                            continue
                        if getattr(m, "layer_id", None) != layer_id_effective:
                            continue
                        if stable_key_prefix:
                            rel_s = str(rel)
                            if not (rel_s == stable_key or (rel_s.startswith(stable_key_prefix) and ("_blk" in rel_s))):
                                continue
                        else:
                            continue
                        _collect_from_file(rel, meta_obj=m)
                    except Exception:
                        continue

                if not blocks or len(blocks) < 2:
                    try:
                        import os as _os
                        for root in ("/kvcache_v1/local", "/kvcache_v1/remote"):
                            try:
                                for name in _os.listdir(root):
                                    try:
                                        if not isinstance(name, str):
                                            continue
                                        if stable_key_prefix and name.startswith(stable_key_prefix):
                                            rel = name
                                            _collect_from_file(rel, meta_obj=None)
                                    except Exception:
                                        continue
                            except Exception:
                                continue
                    except Exception:
                        pass

                if not blocks:
                    # 无可用块
                    logger.debug(f"序列 {seq_idx} 未找到可用的块进行聚合，跳过")
                    self._dbg(f"[retrieve] seq_idx={seq_idx} blocks_empty")
                    restored_per_seq.append(0)
                    continue

                blocks.sort(key=lambda x: int(x[0]))
                self._dbg(f"[retrieve] seq_idx={seq_idx} blocks_sorted_count={len(blocks)} first_offset={blocks[0][0] if blocks else None}")
                try:
                    import torch as _torch
                    expected = _torch.arange(int(current_slots.numel()), device=current_slots.device, dtype=_torch.long)
                    seq_ok = bool((current_slots[:expected.numel()] == expected).all().item())
                except Exception:
                    seq_ok = True
                if not seq_ok:
                    try:
                        import torch as _torch
                        current_slots = _torch.arange(int(current_slots.numel()), device=current_slots.device, dtype=_torch.long)
                        seq_ok = True
                    except Exception:
                        restored_per_seq.append(0)
                        continue
                blocks = sorted(blocks, key=lambda x: int(x[0]))
                restored_tokens_for_seq = 0
                for offset, k_tensor, v_tensor, block_len, block_slots in blocks:
                    # move to device
                    k_tensor = k_tensor.to(input_tokens.device)
                    v_tensor = v_tensor.to(input_tokens.device)
                    for layer_idx in range(num_layers):
                        try:
                            layer_k = k_tensor[layer_idx]
                            layer_v = v_tensor[layer_idx]
                        except Exception:
                            continue
                        kv_cache = kv_caches[layer_idx]
                        key_cache, value_cache = self.split_kv_cache(kv_cache)

                        if key_cache.dim() < 3:
                            logger.error("无法识别的KV缓存形状(写回): %s", key_cache.shape)
                            raise ValueError("Unsupported KV cache layout")
                        num_heads = int(key_cache.shape[-2])
                        head_dim = int(key_cache.shape[-1])
                        try:
                            import torch as _torch
                            if block_slots is not None:
                                slice_slots = block_slots.to(dtype=_torch.long, device=input_tokens.device)
                            else:
                                slice_slots = current_slots[offset: offset + block_len].to(dtype=_torch.long, device=input_tokens.device)
                        except Exception:
                            slice_slots = current_slots
                        if layer_k.shape[0] < slice_slots.numel():
                            logger.warning("块的 KV 长度小于目标槽位数，尝试按最小长度写回")
                        limit = min(int(layer_k.shape[0]), int(slice_slots.numel()))
                        if limit <= 0:
                            continue
                        if int(layer_k.shape[-2]) != num_heads or int(layer_k.shape[-1]) != head_dim:
                            continue
                        import torch as _torch
                        if key_cache.dim() == 4:
                            max_win = int(key_cache[seq_idx].shape[0])
                            eff = min(limit, max_win)
                            if eff <= 0:
                                continue
                            dest = _torch.arange(eff, dtype=_torch.long, device=input_tokens.device)
                            key_cache[seq_idx][dest] = layer_k[:eff].to(key_cache.dtype)
                            value_cache[seq_idx][dest] = layer_v[:eff].to(value_cache.dtype)
                            restored_tokens_for_seq += int(eff)
                        else:
                            max_win = int(key_cache.shape[0]) - int(start_pos)
                            eff = min(limit, max_win)
                            if eff <= 0:
                                continue
                            dest = _torch.arange(eff, dtype=_torch.long, device=input_tokens.device) + int(start_pos)
                            key_cache[dest] = layer_k[:eff].to(key_cache.dtype)
                            value_cache[dest] = layer_v[:eff].to(value_cache.dtype)
                            restored_tokens_for_seq += int(eff)

                    # restored_tokens_for_seq updated per layer writes above

                # 统计
                restored_len = min(restored_tokens_for_seq, seq_len)
                if restored_len == seq_len:
                    recovered_sequences += 1
                else:
                    self._dbg(f"[retrieve] partial aggregate restore: seq_idx={seq_idx} restored_len={restored_len} < seq_len={seq_len}")
                restored_per_seq.append(int(restored_len))

            except Exception as e:
                logger.error(
                    f"在序列 {seq_idx} 聚合恢复 KV 时出错: {e}",
                    exc_info=True,
                )
                restored_per_seq.append(0)
                continue

        # 若某些序列在循环中因 continue 未写入统计，则补充 0
        while len(restored_per_seq) < len(seq_lens):
            restored_per_seq.append(0)

        # 记录本次检索统计
        total_restored = int(sum(restored_per_seq))
        stats = {
            "sequences": int(len(seq_lens)),
            "restored_per_seq": restored_per_seq,
            "total_restored_tokens": total_restored,
            "total_tokens": total_tokens,
            "full_hits": int(recovered_sequences),
            "bypass": False,
        }

        if recovered_sequences != len(seq_lens):
            logger.debug(
                "KV 恢复未全部成功 (成功 %d / 期望 %d)，继续执行模型前向",
                recovered_sequences,
                len(seq_lens),
            )
            # 输出简要节省比例（仅针对前缀 prefill 的估算）
            if total_tokens > 0:
                saved_ratio = total_restored / total_tokens
                logger.info(
                    "[retrieve] summary: restored_tokens=%d/%d (%.1f%%), sequences_full_hit=%d/%d, bypass=%s",
                    total_restored,
                    total_tokens,
                    saved_ratio * 100.0,
                    recovered_sequences,
                    len(seq_lens),
                    False,
                )
            self._last_retrieve_stats = stats
            return None, False, model_input

        # 全量命中：不绕过模型前向，仅将聚合的 KV 写回，再继续正常解码
        logger.debug("KV 全量命中，继续模型前向（不 bypass）")
        stats["bypass"] = False
        self._last_retrieve_stats = stats
        logger.info(
            "[retrieve] summary: restored_tokens=%d/%d (100.0%%), sequences_full_hit=%d/%d, bypass=%s",
            total_restored,
            total_tokens,
            recovered_sequences,
            len(seq_lens),
            False,
        )
        return None, False, model_input

    # ------------------ Debug helpers ------------------ #
    def _maybe_setup_debug_file_handler(self):
        """If debug dump is enabled, optionally write KVEngine logs to a dedicated file."""
        try:
            if not self._debug_dump:
                return
            # prevent duplicate handlers
            for h in logger.handlers:
                if isinstance(h, logging.FileHandler) and getattr(h, "name", None) == "KVEngineDebugFile":
                    return
            dump_file = self._debug_dump_file
            if not dump_file:
                # default next to storage_dir
                base_dir = getattr(self, "storage_dir", ".")
                os.makedirs(base_dir, exist_ok=True)
                dump_file = os.path.join(base_dir, f"kv_engine_debug_{os.getpid()}.log")
            fh = logging.FileHandler(dump_file, encoding="utf-8")
            fh.setLevel(logging.INFO)
            fh.set_name("KVEngineDebugFile")
            fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            logger.addHandler(fh)
            logger.info(f"[debug_dump] 写入调试日志到: {dump_file}")
        except Exception as e:
            logger.warning(f"无法创建调试日志文件: {e}")

    def _dbg(self, msg: str):
        if self._debug_dump:
            try:
                logger.info(msg)
            except Exception:
                pass

    def _tensor_brief(self, name: str, t: Optional[torch.Tensor], max_items: int = 16) -> str:
        try:
            if t is None:
                return f"{name}: None"
            shape = tuple(t.shape)
            dtype = str(t.dtype)
            device = str(t.device)
            numel = int(t.numel())
            preview = None
            if numel > 0 and len(shape) == 1 and numel <= 64:
                preview = t.detach().reshape(-1)[:max_items].cpu().tolist()
            return f"{name}: shape={shape} dtype={dtype} device={device} numel={numel} preview={preview}"
        except Exception as e:
            return f"{name}: <error {e}>"

    def _dump_store_context(
        self,
        seq_idx: int,
        start_pos: int,
        end_pos: int,
        seq_len: int,
        file_path: str,
        input_tokens: torch.Tensor,
        current_tokens: torch.Tensor,
        slot_mapping: torch.Tensor,
        current_slots: torch.Tensor,
    ):
        if not self._debug_dump:
            return
        try:
            msgs = [
                f"[store] seq_idx={seq_idx} start={start_pos} end={end_pos} seq_len={seq_len} file={file_path}",
                self._tensor_brief("input_tokens", input_tokens),
                self._tensor_brief("current_tokens", current_tokens),
                f"slot_mapping_shape={getattr(slot_mapping, 'shape', None)} dim={getattr(slot_mapping, 'dim', lambda: 'n/a')() if hasattr(slot_mapping,'dim') else 'n/a'}",
                self._tensor_brief("current_slots", current_slots),
            ]
            self._dbg(" | ".join(msgs))
        except Exception:
            pass

    def _dump_slots_context(
        self,
        tag: str,
        seq_idx: int,
        seq_len: int,
        start_pos: int,
        file_path: str,
        slot_mapping: torch.Tensor,
        current_slots: torch.Tensor,
    ):
        if not self._debug_dump:
            return
        try:
            msgs = [
                f"[{tag}] seq_idx={seq_idx} start={start_pos} seq_len={seq_len} file={file_path}",
                f"slot_mapping_shape={getattr(slot_mapping, 'shape', None)} dim={getattr(slot_mapping, 'dim', lambda: 'n/a')() if hasattr(slot_mapping,'dim') else 'n/a'}",
                self._tensor_brief("current_slots", current_slots),
            ]
            self._dbg(" | ".join(msgs))
        except Exception:
            pass

    # ------------------ Helper: KV 结构 ------------------ #
    @staticmethod
    def split_kv_cache(kv_cache: torch.Tensor):
        if kv_cache.dim() < 2 or kv_cache.size(0) != 2:
            raise ValueError(f"无法识别的KV缓存结构: {kv_cache.shape}")
        return kv_cache[0], kv_cache[1]

    # ------------------ Helper: 文件键 ------------------ #
    def _tensor_hash(self, tensor: torch.Tensor) -> str:
        """为张量生成唯一哈希键"""
        if tensor.numel() == 0:
            return "empty"
        tensor_bytes = tensor.cpu().numpy().tobytes()
        # 默认 blake2b 输出 64 bytes -> 128 hex chars，会导致文件名过长（>128字节时
        # KVMetadata 打包的 128B file_path 字段被截断，进而产生两个 key：完整 key 与
        # 被截断的 key，影响后续命中与检索（见 issue: v1 roundtrip 文件存在但检索 miss）。
        # 为避免截断，这里缩短 digest_size；16 bytes(32 hex) 可能碰撞率略高，折中取 24 bytes(48 hex)。
        # 这样典型文件名长度: prefix(~25)+48+".pt"≈76 < 128，不会被截断。
        return hashlib.blake2b(tensor_bytes, digest_size=24).hexdigest()

    def _make_key(
        self,
        input_tokens: torch.Tensor,
        session_id: Optional[bytes] = None,
        layer_id: Optional[int] = None,
        stable_key: Optional[str] = None,
    ) -> str:
        """生成文件键。

        优先使用显式传入的 stable_key（用于分块存储时确保同一序列使用同一键），
        否则回退为基于 tokens 的哈希。
        """
        # 优先使用外部提供的稳定键
        if stable_key:
            try:
                logger.info(f"[naming] use_stable_key={stable_key}")
            except Exception:
                pass
            return stable_key

        seq_hash = self._tensor_hash(input_tokens)
        # 如果没有提供session_id和layer_id，使用默认值
        if session_id is None:
            session_id = b"session_0000"
        if layer_id is None:
            layer_id = 0
        # 将session_id转换为字符串
        session_str = session_id.hex() if isinstance(session_id, (bytes, bytearray)) else str(session_id)
        # 构造文件名，不包含路径前缀
        key = f"kv_{session_str}_layer_{layer_id}_{seq_hash}.pt"
        try:
            logger.info(f"[naming] gen_key session={session_str} layer={layer_id} hash={seq_hash} key={key}")
        except Exception:
            pass
        return key

    # ------------------ Helper: 解析序列槽位映射 ------------------ #
    def _resolve_sequence_slots(
        self,
        seq_idx: int,
        seq_len: int,
        slot_mapping: torch.Tensor,
        file_path: str,
        start_pos: int,
    ) -> torch.Tensor:
        """根据 batch 序号解析该序列对应的 token 槽位索引。

        约束：
        - slot_mapping 应具有形状 [batch, seq] 或 [batch, variable_seq]。
        - 返回的一维张量长度需与 seq_len 匹配，若不匹配则裁剪到最小长度并记录日志。
        - 若出现空序列，返回空张量以便调用方跳过。
        """
        try:
            # 支持两种格式：
            # - 2D: [batch, seq] 或 [batch, variable_seq] → 使用 batch 索引
            # - 1D: [total_tokens] → 使用 [start_pos:end_pos] 切片
            if slot_mapping.dim() == 1:
                current_slots = slot_mapping[start_pos:start_pos + seq_len]
                try:
                    current_slots = current_slots - int(start_pos)
                except Exception:
                    pass
            else:
                current_slots = slot_mapping[seq_idx]
        except Exception:
            logger.error("无法获取槽位映射，slot_mapping shape=%s seq_idx=%d start_pos=%d", getattr(slot_mapping, 'shape', None), seq_idx, start_pos)
            return torch.tensor([], dtype=torch.long)

        if current_slots.dim() > 1:
            current_slots = current_slots.reshape(-1)
        if current_slots.dim() == 0:
            current_slots = current_slots.unsqueeze(0)

        if current_slots.numel() != seq_len:
            logger.warning(
                "序列 %s 的槽位映射长度(%d)与序列长度(%d)不一致，使用最小长度",
                file_path,
                current_slots.numel(),
                seq_len,
            )
            limit = min(current_slots.numel(), seq_len)
            current_slots = current_slots[:limit]
        try:
            import torch as _torch
            current_slots = current_slots.to(dtype=_torch.long)
            if current_slots.numel() > 0:
                current_slots = _torch.clamp(current_slots, min=0, max=max(0, int(seq_len) - 1))
        except Exception:
            current_slots = current_slots.to(dtype=torch.long)
        return current_slots
    
    def _storage_insert(
        self,
        file_path: str,
        k_cache,
        v_cache,
        input_tokens,
        roi,
        slot_mapping: Optional[torch.Tensor] = None,
        payload_meta: Optional[dict] = None,
    ):
        """使用存储后端打包并上传数据，并嵌入元数据用于恢复（包含slot_mapping与meta）。"""
        # 使用扩展打包完整负载
        data = self._storage.pack_full_payload(
            k_cache, v_cache, input_tokens, roi, slot_mapping, payload_meta
        )

        # 获取文件的元数据并嵌入（如可用）
        meta = self._meta_cache.get_metadata(key=file_path)
        if meta:
            metadata_bytes = meta.pack_with_embedding()
            data = metadata_bytes + data

        success = self._storage.upload(file_path, data)
        if not success:
            logger.error(f"Failed to upload KV data to {file_path}")

    def _storage_download(
        self, file_path: str
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """使用存储后端下载并解包数据"""
        logger.debug(f"开始下载文件: {file_path}")
        data = self._storage.download(file_path)
        if data is None:
            logger.warning(f"下载文件 {file_path} 失败，文件可能不存在")
            return None, None
            
        logger.debug(f"成功下载文件 {file_path}，数据大小: {len(data)} 字节")
            
        # 检查数据是否包含嵌入的元数据
        # 元数据头部是METADATA_HEADER + METADATA_VERSION
        from distributed_kv_manager.metadata.etcd import METADATA_HEADER, METADATA_VERSION, HEADER_SIZE, METADATA_SIZE
        header_size = HEADER_SIZE  # 8 bytes
        min_data_size = header_size + METADATA_SIZE  # 至少需要头部和元数据大小
        
        # 如果数据足够长，检查是否有元数据头部
        if len(data) >= min_data_size:
            try:
                # 提取头部
                header, version = struct.unpack("<4sI", data[:header_size])
                logger.debug(f"文件 {file_path} 包含元数据头部: {header}, 版本: {version}")
                # 如果有元数据头部，跳过元数据部分
                if header == METADATA_HEADER and version == METADATA_VERSION:
                    # 跳过头部和元数据，只解包KV数据
                    kv_data = data[header_size + METADATA_SIZE:]
                    logger.debug(f"跳过元数据部分，KV数据大小: {len(kv_data)} 字节")
                    return self._storage.unpack_kv_data(kv_data)
                else:
                    logger.warning(f"文件 {file_path} 的元数据头部不匹配: {header} != {METADATA_HEADER} 或版本不匹配: {version} != {METADATA_VERSION}")
            except Exception as e:
                logger.warning(f"从文件 {file_path} 中提取元数据失败: {e}")
        else:
            logger.warning(f"文件 {file_path} 数据太短，无法包含元数据: {len(data)} < {min_data_size}")
        
        # 如果没有元数据嵌入或提取失败，直接解包整个数据
        logger.debug(f"直接解包整个数据，大小: {len(data)} 字节")
        return self._storage.unpack_kv_data(data)

    def _storage_download_kv_bytes(self, file_path: str) -> Optional[bytes]:
        """下载文件并返回纯KV负载字节（去掉嵌入的元数据头）。"""
        data = self._storage.download(file_path)
        if data is None:
            return None
        try:
            from distributed_kv_manager.metadata.etcd import (
                METADATA_HEADER,
                METADATA_VERSION,
                HEADER_SIZE,
                METADATA_SIZE,
            )
            if len(data) >= HEADER_SIZE + METADATA_SIZE:
                header, version = struct.unpack("<4sI", data[:HEADER_SIZE])
                if header == METADATA_HEADER and version == METADATA_VERSION:
                    return data[HEADER_SIZE + METADATA_SIZE :]
        except Exception:
            pass
        return data

    def _storage_exists(self, file_path: str) -> bool:
        """使用存储后端检查文件是否存在"""
        logger.debug(f"检查文件是否存在: {file_path}, 存储实例类型: {type(self._storage)}")
        # 直接使用file_path，因为_storage.exists会处理路径前缀
        exists = self._storage.exists(file_path)
        logger.debug(f"文件 {file_path} 存在: {exists}")
        return exists
        
    def _update_last_access_time(self, file_path: str):
        """更新元数据的最后访问时间（异步）"""
        try:
            # 获取当前元数据
            meta = self._meta_cache.get_metadata(key=file_path)
            if meta:
                # 更新最后访问时间
                meta.last_access = int(time.time())
                # 异步更新到缓存和ETCD
                self._meta_cache.put_metadata(meta)
        except Exception as e:
            logger.error(f"更新元数据最后访问时间失败: {e}")

    def close(self):
        """关闭KV引擎，等待所有异步任务完成"""
        # 关闭清理器
        if hasattr(self, '_cleanup_manager'):
            self._cleanup_manager.stop()
        
        # 等待异步任务完成
        for f in self._futures:
            try:
                f.result(timeout=60)
            except Exception as e:
                logger.error(f"KV 异步存储失败: {e}")
        self._executor.shutdown(wait=True)
        print("[KV ENGINE CLOSED]")

# --- module-level engine wrapper ---
from .base import StoreStatus, RetrieveStatus
from ..config_loader import load_config_from_json

_engine_singleton: KVEngine | None = None  

def init_engine(config=None, config_path: Optional[str] = None):
    """
    初始化并返回 engine 单例。传入 vllm 的 config 或从 config.json 读取配置。
    
    Args:
        config: vllm 的配置对象
        config_path: 配置文件路径，默认为项目根目录的config.json
    """
    global _engine_singleton
    if _engine_singleton is None:
        # 如果没有提供config，则从配置文件加载
        if config is None:
            config = load_config_from_json(config_path) if config_path is not None else load_config_from_json()
        else:
            # 尝试加载config.json，并与传入的config合并，使JSON生效
            json_cfg = None
            try:
                # 优先使用显式提供的config_path，否则使用默认查找
                json_cfg = load_config_from_json(config_path) if config_path is not None else load_config_from_json()
            except FileNotFoundError:
                # 若找不到则保持传入的config
                json_cfg = None

            if json_cfg is not None:
                combined = SimpleNamespace()
                # 优先使用传入config的rank/local_rank，其次使用JSON，最后默认0
                combined.rank = getattr(config, "rank", getattr(json_cfg, "rank", 0))
                combined.local_rank = getattr(config, "local_rank", getattr(json_cfg, "local_rank", 0))
                # 引擎标识（如存在）
                if hasattr(config, "engine_id"):
                    combined.engine_id = getattr(config, "engine_id")
                elif hasattr(json_cfg, "engine_id"):
                    combined.engine_id = getattr(json_cfg, "engine_id")
                # 使用JSON中的kv_transfer_config覆盖，以确保config.json生效
                combined.kv_transfer_config = getattr(
                    json_cfg, "kv_transfer_config",
                    getattr(config, "kv_transfer_config", SimpleNamespace())
                )
                config = combined

        _engine_singleton = KVEngine(config)  # 创建KVEngine实例
    return _engine_singleton

def destroy_engine():
    """销毁 engine（如果存在）"""
    global _engine_singleton
    if _engine_singleton is not None:
        try:
            _engine_singleton.close()
        finally:
            _engine_singleton = None

def should_store(model_input):
    """
    模块级 should_store 接口，返回 StoreStatus。
    """
    engine = _engine_singleton
    if engine is None:
        # 如果尚未初始化，默认返回 STORED
        return StoreStatus.STORED
    return engine.should_store(model_input)

def store_kv(model_config, parallel_config, transfer_config,
             model_executable, model_input, kv_caches, store_status,
             hidden_or_intermediate_states=None):
    """
    模块级 store_kv 接口，委托给 engine 的 store_kv。
    """
    engine = _engine_singleton
    if engine is None:
        raise RuntimeError("Engine not initialized. Call init_engine(config) first.")
    return engine.store_kv(model_config, parallel_config, transfer_config,
                           model_executable, model_input, kv_caches, store_status,
                           hidden_or_intermediate_states)

def should_retrieve(model_input):
    """
    模块级 should_retrieve 接口，返回 RetrieveStatus。
    """
    engine = _engine_singleton
    if engine is None:
        return RetrieveStatus.MISS
    return engine.should_retrieve(model_input)

def retrieve_kv(model_executable, model_input, kv_caches, retrieve_status):
    """
    模块级 retrieve_kv 接口，委托给 engine 的 retrieve_kv。
    """
    engine = _engine_singleton
    if engine is None:
        raise RuntimeError("Engine not initialized. Call init_engine(config) first.")
    return engine.retrieve_kv(model_executable, model_input, kv_caches, retrieve_status)

def get_last_retrieve_stats() -> dict:
    """返回最近一次 retrieve_kv 的统计信息，便于外部验证节省比例。

    返回示例：
    {
        "sequences": 1,
        "restored_per_seq": [16],
        "total_restored_tokens": 16,
        "total_tokens": 24,
        "full_hits": 0,
        "bypass": False,
    }
    """
    engine = _engine_singleton
    if engine is None:
        raise RuntimeError("Engine not initialized. Call init_engine(config) first.")
    return getattr(engine, "_last_retrieve_stats", {})

def clear_kv_cache(session_id: Optional[bytes] = None,
                   layer_id: Optional[int] = None,
                   contains: Optional[str] = None,
                   delete_storage: bool = True) -> dict:
    """模块级清理接口：清理ETCD元数据并可选删除存储文件。

    - session_id: 仅清理该会话（bytes）；不传表示不过滤会话。
    - layer_id: 仅清理该层；不传表示不过滤层。
    - contains: 文件名包含该子串才清理（例如特定哈希）；不传表示不限。
    - delete_storage: 是否同时删除存储中的文件（默认 True）。
    返回统计信息字典。
    """
    engine = _engine_singleton
    if engine is None:
        raise RuntimeError("Engine not initialized. Call init_engine(config) first.")
    return engine.clear_cache(session_id=session_id,
                              layer_id=layer_id,
                              contains=contains,
                              delete_storage=delete_storage)
