from __future__ import annotations
from typing import TYPE_CHECKING, Union
import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.sequence import IntermediateTensors
from vllm.logger import init_logger

# 默认的预填/分块参数，可由配置覆盖
STORE_AFTER_PREFILL = False
ALLOW_PARTIAL_PREFILL_STORE = False  # 预填未完成时是否允许落部分
DEBUG_BLOCK_ACCOUNTING = True
BLOCK_SIZE = 16
FORCE_STORE_THRESHOLD = 3  # 最多跳过次数

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

logger = init_logger(__name__)

import sys as _sys  # diag
try:
    _sys.stderr.write(
        f"[dkv.connector] module import: "
        f"STORE_AFTER_PREFILL={STORE_AFTER_PREFILL} "
        f"ALLOW_PARTIAL_PREFILL_STORE={ALLOW_PARTIAL_PREFILL_STORE} "
        f"BLOCK_SIZE={BLOCK_SIZE}\\n")
except Exception:
    pass
try:
    with open("/tmp/connector_debug.log", "a", encoding="utf-8") as _f:
        _f.write("[dkv.connector] module import\\n")
        _f.flush()
except Exception:
    pass

class DistributedKVConnector(KVConnectorBase):
    """
    DistributedKVConnector
    """
    def __init__(self, rank: int, local_rank: int, config: VllmConfig):
        """
        engine: DistributedKVEngineBase 子类实例
        """
        from distributed_kv_manager.engine import(
            StoreStatus, RetrieveStatus, init_engine,
            retrieve_kv, should_retrieve, store_kv, should_store,
            destroy_engine)
        self.rank = rank
        self.local_rank = local_rank
        self.config = config
        self.engine = init_engine(config)
        self.transfer_config = config.kv_transfer_config
        self.vllm_config = config
        kv_cfg = getattr(config, "kv_transfer_config", None)
        cache_cfg = getattr(config, "cache_config", None)
        extra = getattr(kv_cfg, "kv_connector_extra_config", {}) if kv_cfg is not None else {}
        # 运行时开关：优先从 extra_config 获取
        try:
            self._store_after_prefill = bool(
                extra["store_after_prefill"] if isinstance(extra, dict) and ("store_after_prefill" in extra)
                else STORE_AFTER_PREFILL
            )
        except Exception:
            self._store_after_prefill = STORE_AFTER_PREFILL
        self._allow_partial_prefill_store = (
            (extra.get("allow_partial_prefill_store")
             if isinstance(extra, dict) and ("allow_partial_prefill_store" in extra)
             else getattr(kv_cfg, "allow_partial_prefill_store", ALLOW_PARTIAL_PREFILL_STORE))
        )
        self._debug_block_accounting = (
            (extra.get("debug_block_accounting")
             if isinstance(extra, dict) and ("debug_block_accounting" in extra)
             else getattr(kv_cfg, "debug_block_accounting", DEBUG_BLOCK_ACCOUNTING))
        )
        self._block_size = getattr(cache_cfg, "block_size", BLOCK_SIZE)
        self._use_flexkv_slot_mapping = bool(
            extra.get("use_flexkv_slot_mapping") if isinstance(extra, dict) and ("use_flexkv_slot_mapping" in extra)
            else getattr(kv_cfg, "use_flexkv_slot_mapping", False)
        )
        # 直接写文件留痕，确保初始化阶段一定能看到日志
        try:
            with open("/tmp/connector_debug.log", "a", encoding="utf-8") as f:
                f.write(f"[connector-init] store_after_prefill={self._store_after_prefill} "
                        f"allow_partial={self._allow_partial_prefill_store} block_size={self._block_size}\n")
                f.flush()
        except Exception:
            pass
        logger.warning(f"[connector] store_after_prefill={self._store_after_prefill}")
        self._dbg(f"[connector] store_after_prefill={self._store_after_prefill}")
        logger.warning(f"[connector] allow_partial_prefill_store={self._allow_partial_prefill_store}")
        self._dbg(f"[connector] allow_partial_prefill_store={self._allow_partial_prefill_store}")
        logger.warning(f"[connector] block_size={self._block_size}")
        self._dbg(f"[connector] block_size={self._block_size}")
        # 预填延迟计数（避免永远跳过）：按文件键统计尝试次数
        self._prefill_attempts = {}
        # 预填块统计与日志开关
        # 跟踪每个文件键的上次可见长度（用于计算Δ与封口数）
        self._prefill_prev_len = {}
        self.retrieve_kv = retrieve_kv
        self.should_retrieve = should_retrieve
        self.store_kv = store_kv
        self.should_store = should_store
        self._destroy_engine = destroy_engine
        # 记录 (session, layer, seq_idx) -> base_stable_key，仅首块计算哈希，后续块直接复用
        self._active_seq_keys: dict[tuple[str, int, int], str] = {}
        # 记录 (session, layer) -> base_stable_key，用于相同会话层的多次请求复用同一稳定键
        self._stable_key_cache: dict[tuple[str, int], str] = {}
        self.store_status = StoreStatus
        self.retrieve_status = RetrieveStatus
        self.engine_name = getattr(config, "engine_id", "unknown_engine")

        logger.info(f"DistributedKVConnector initialized with engine {self.engine_name}")

    def _dbg(self, msg: str) -> None:
        """打印调试信息到 stderr 并写入 /tmp/connector_debug.log。"""
        try:
            import sys
            sys.stderr.write(msg + "\n")
        except Exception:
            pass
        try:
            with open("/tmp/connector_debug.log", "a", encoding="utf-8") as f:
                f.write(msg + "\n")
                f.flush()
        except Exception as e:
            try:
                self._dbg(f"[connector] recv prep failed: {e}")
            except Exception:
                pass
            pass

    def _get_stable_key(self, model_input: "ModelInputForGPUWithSamplingMetadata") -> str:
        """
        生成或获取请求的稳定键（base_key）。
        优先使用 request_ids (attn_metadata) 作为唯一标识。
        若无 request_ids，则回退到 (session, layer) + 首个 chunk 哈希的缓存机制。
        """
        # 1. 优先尝试复用已有的 _dkv_base_key
        if hasattr(model_input, "_dkv_base_key") and model_input._dkv_base_key:
            return model_input._dkv_base_key

        # 准备 session/layer
        session_id = getattr(model_input, "session_id", None) or b"session_0000"
        layer_id = getattr(model_input, "layer_id", None) or 0
        session_norm = session_id.hex() if isinstance(session_id, (bytes, bytearray)) else str(session_id)
        layer_norm = int(layer_id)

        # 2. 尝试从 attn_metadata 获取 request_ids / seq_ids
        sa = getattr(model_input, "attn_metadata", None)
        req_ids = getattr(sa, "request_ids", None)
        seq_ids = getattr(sa, "seq_ids", None)
        
        req_id_str = None
        if req_ids is not None:
            # Handle list/tuple
            if isinstance(req_ids, (list, tuple)) and len(req_ids) > 0:
                req_id_str = str(req_ids[0])
            # Handle tensor
            elif hasattr(req_ids, "flatten"):
                try:
                    if req_ids.numel() > 0:
                        req_id_str = str(req_ids.flatten()[0].item())
                except:
                    pass
            # Handle single value
            elif isinstance(req_ids, (str, int)):
                req_id_str = str(req_ids)

        if req_id_str:
            # 清理特殊字符
            req_id_str = req_id_str.replace("/", "_").replace("\\", "_")
            # 格式: kv_{session}_layer_{layer}_req_{req_id}
            base_key = f"kv_{session_norm}_layer_{layer_norm}_req_{req_id_str}"
            # 保存到 model_input 以便后续使用
            try:
                model_input._dkv_base_key = base_key
            except:
                pass
            return base_key

        # 其次使用 seq_id (vLLM 中通常是唯一的，且必传)
        seq_id_str = None
        if seq_ids is not None:
             if isinstance(seq_ids, (list, tuple)) and len(seq_ids) > 0:
                 seq_id_str = str(seq_ids[0])
             elif hasattr(seq_ids, "flatten"):
                 try:
                     if seq_ids.numel() > 0:
                         seq_id_str = str(seq_ids.flatten()[0].item())
                 except:
                     pass
             elif isinstance(seq_ids, (int, str)):
                 seq_id_str = str(seq_ids)
        
        if seq_id_str:
            base_key = f"kv_{session_norm}_layer_{layer_norm}_seq_{seq_id_str}"
            try:
                model_input._dkv_base_key = base_key
            except:
                pass
            try:
                self._dbg(f"[connector] base_key.seq_id={base_key}")
            except Exception:
                pass
            return base_key

        # 3. 回退逻辑：直接基于当前 input_tokens 计算哈希，避免跨请求复用导致命名冲突

        # 4. 计算哈希 (First chunk case)
        try:
            full_hash = self.engine._tensor_hash(model_input.input_tokens)
        except Exception:
            full_hash = "nohash"
            
        base_key = f"kv_{session_norm}_layer_{layer_norm}_{full_hash}"
        try:
            model_input._dkv_base_key = base_key
        except:
            pass
        try:
            self._dbg(f"[connector] base_key.hash={base_key}")
        except Exception:
            pass
        return base_key

    def recv_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: list[torch.Tensor]
    ) -> tuple[Union[torch.Tensor, IntermediateTensors], bool, "ModelInputForGPUWithSamplingMetadata"]:
        # 默认填充会话/层与稳定键，避免 None 导致键不一致
        stable_key_arg = None
        try:
            # 使用统一逻辑获取 stable_key
            base_key = self._get_stable_key(model_input)
            stable_key_arg = f"{base_key}.pt"
            
            # 尝试写入 model_input，失败则忽略
            pass
            
            self._dbg(f"[connector] recv prep stable_key={stable_key_arg} base={base_key}")
        except Exception as e:
            try:
                self._dbg(f"[connector] recv prep failed: {e}")
            except Exception:
                pass
        
        try:
            self._dbg(f"[connector] recv enter session_id={getattr(model_input,'session_id',None)} layer_id={getattr(model_input,'layer_id',None)} stable_key={stable_key_arg}")
        except Exception:
            pass

        # 保持 vLLM 上游的 slot_mapping，不做改写，避免影响 KV 槽位一致性

        retrieve_status = self.engine.should_retrieve(model_input, stable_key=stable_key_arg)
        hidden_or_intermediate_states, bypass_model_exec, model_input  = self.engine.retrieve_kv(
            model_executable, model_input, kv_caches, retrieve_status, stable_key=stable_key_arg
        )
        return hidden_or_intermediate_states, bypass_model_exec, model_input

    def send_kv_caches_and_hidden_states(
        self,
        model_executable,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches,
        hidden_or_intermediate_states,
    ) -> None:
        block_size = int(self._block_size) if hasattr(self, "_block_size") else BLOCK_SIZE
        # 入口埋点，确认是否进入 send
        self._dbg("[connector-print] send_enter")
        try:
            logger.info("[connector] send_enter")
        except Exception:
            pass
        # 调试：打印 vLLM 传入的元数据是否包含 seq_ids/request_ids
        try:
            sa = getattr(model_input, "attn_metadata", None)
            seq_ids = getattr(sa, "seq_ids", None)
            req_ids = getattr(sa, "request_ids", None)
            self._dbg(f"[connector] send meta seq_ids={seq_ids} request_ids={req_ids}")
        except Exception:
            pass

        # 发送前补充默认的 session/layer/stable_key（与 recv 同步），确保存取使用同一哈希键
        stable_key_arg = None
        try:
            # 使用统一逻辑获取 stable_key
            base_key = self._get_stable_key(model_input)
            stable_key_arg = f"{base_key}.pt"
            
            # 尝试写入 model_input，失败则忽略
            pass
            
            self._dbg(f"[connector] send prep session={getattr(model_input,'session_id',None)} layer={getattr(model_input,'layer_id',None)} stable_key={stable_key_arg}")
        except Exception:
            pass

        def _compute_avail(seq_idx: int, seq_len: int, start_pos: int) -> int:
            """Estimate visible tokens for the sequence using slot_mapping first,
            then fall back to raw KV cache view."""
            avail = 0
            try:
                sm = getattr(model_input.attn_metadata, "slot_mapping", None)
                if sm is not None:
                    if hasattr(sm, "dim") and sm.dim() == 1:
                        sm_slice = sm[start_pos:start_pos + seq_len]
                    else:
                        sm_slice = sm[seq_idx][:seq_len]
                    try:
                        avail = int((sm_slice >= 0).sum().item())
                    except Exception:
                        avail = int(getattr(sm_slice, "shape", [seq_len])[0])
                    avail = min(avail, int(seq_len))
            except Exception:
                avail = 0
            if avail <= 0 and kv_caches:
                try:
                    k0 = kv_caches[0][0]
                    if k0.dim() == 4:
                        avail = int(k0[seq_idx].shape[0]) if seq_idx < k0.shape[0] else 0
                    elif k0.dim() == 3:
                        total_tokens = int(k0.shape[0])
                        avail = min(max(0, total_tokens - start_pos), int(seq_len))
                except Exception:
                    pass
            # 保底：若无法估计可见长度，退回整个 seq_len，避免 new_sealed 恒为 0
            if avail <= 0:
                avail = int(seq_len)
            self._dbg(f"[connector-print] avail_check seq={seq_idx} seq_len={seq_len} start_pos={start_pos} avail={avail}")
            return int(max(0, avail))

        # 若启用“预填充完成后再存储”，当4D/3D视图尚未覆盖完整prompt时跳过本次存储；
        # 判定基于首层K缓存的可见长度与真实seq_len的直接比较；
        # 并加入有限次重试以避免永远不落盘。
        # 硬编码开关：不做运行期动态重评估
        if self._store_after_prefill:
            try:
                seq_lens = model_input.attn_metadata.seq_lens
                input_tokens = model_input.input_tokens
                session_id = getattr(model_input, "session_id", None)
                layer_id = getattr(model_input, "layer_id", None)

                prefill_initial_ready = True
                for seq_idx, seq_len in enumerate(seq_lens):
                    seq_len = int(seq_len)
                    start_pos = int(sum(int(x) for x in seq_lens[:seq_idx]))
                    end_pos = start_pos + seq_len
                    try:
                        current_tokens = input_tokens[start_pos:end_pos]
                        file_key = self.engine._make_key(current_tokens, session_id, layer_id, stable_key=stable_key_arg)
                    except Exception:
                        file_key = f"seq_{seq_idx}_len_{seq_len}"

                    avail = _compute_avail(seq_idx, seq_len, start_pos)

                    if avail < seq_len:
                        prefill_initial_ready = False
                        if not self._allow_partial_prefill_store:
                            logger.info(
                                "[connector] store_after_prefill: 可见长度不足 (avail=%d < seq_len=%d)，等待本次 forward 结束再复查 | key=%s",
                                avail, seq_len, file_key)
                        break
                # 注意：不再提前 return，改为在函数尾部再次复查最终可见长度
                try:
                    setattr(model_input, "_prefill_initial_ready", prefill_initial_ready)
                except Exception:
                    pass
            except Exception as e:
                logger.warning(f"[connector] store_after_prefill 检查失败: {e}，回退为末尾复查")

        # 当未启用 store_after_prefill 时：按块增量落盘；
        # 启用后：跳过按块逻辑，等待预填完成后一次性全量落盘。
        try:
            seq_lens = list(model_input.attn_metadata.seq_lens)
        except Exception:
            seq_lens = []

        # track whether we performed any per-block stores in this call
        self._last_performed_block_store = False

        def _make_block_kv_cache(orig_kv_cache, seq_idx: int, blk_start: int, blk_end: int):
            if orig_kv_cache is None:
                return None
            try:
                key_cache = orig_kv_cache[0]
                value_cache = orig_kv_cache[1]
                if key_cache.dim() == 4:
                    # [batch, seq, num_heads, head_dim]
                    key_block = key_cache[seq_idx:seq_idx+1, blk_start:blk_end].contiguous()
                    value_block = value_cache[seq_idx:seq_idx+1, blk_start:blk_end].contiguous()
                elif key_cache.dim() == 3:
                    # [total_tokens, num_heads, head_dim]
                    key_block = key_cache[blk_start:blk_end].contiguous()
                    value_block = value_cache[blk_start:blk_end].contiguous()
                else:
                    logger.warning("[connector] unsupported kv cache dim=%d", key_cache.dim())
                    return None
                return torch.stack([key_block, value_block], dim=0)
            except Exception as e:
                logger.warning("[connector] make_block_kv_cache failed: %s", e)
                return None

        def _build_and_store_block(blk_idx: int, blk_start: int, blk_end: int,
                                   seq_idx: int, seq_len: int, start_pos: int,
                                   current_tokens, session_id, layer_id,
                                   base_file_key, token_offset_base: int) -> bool:
            blk_len = blk_end - blk_start
            if blk_len <= 0:
                return False
            # 每个块使用独立的稳定键，避免覆盖；全局偏移确保可排序
            block_global_offset = int(token_offset_base + blk_start)
            key_base = base_file_key[:-3] if base_file_key.endswith(".pt") else base_file_key
            block_file_key = f"{key_base}_blk{blk_idx}_off{block_global_offset}.pt"

            # build per-block kv_caches (list per layer)
            block_kv_caches = []
            for layer_kv in kv_caches:
                block_kv = _make_block_kv_cache(layer_kv, seq_idx, blk_start, blk_end)
                if block_kv is None:
                    logger.warning("[connector] block idx=%d key=%s 缺少层KV，跳过该块", blk_idx, block_file_key)
                    return False
                block_kv_caches.append(block_kv)

            # build a tiny model_input-like object with required attrs
            from types import SimpleNamespace
            small_mi = SimpleNamespace()
            small_mi.input_tokens = current_tokens[blk_start:blk_end]
            sa = SimpleNamespace()
            sa.seq_lens = [blk_len]
            # try to derive slot_mapping
            try:
                # 对于分块持久化，槽位映射统一使用相对索引 0..blk_len-1，避免带入全局 slot 导致越界
                sa.slot_mapping = torch.arange(blk_len, device=getattr(current_tokens, "device", None))
            except Exception:
                sa.slot_mapping = torch.arange(blk_len)

            # normalize slot mapping to 1D Long with exact blk_len
            try:
                sm = sa.slot_mapping
                dev = getattr(current_tokens, 'device', None)
                if not isinstance(sm, torch.Tensor):
                    sm = torch.as_tensor(sm, dtype=torch.long, device=dev)
                else:
                    sm = sm.to(dtype=torch.long, device=dev)
                if sm.dim() > 1:
                    sm = sm.reshape(-1)
                if int(sm.numel()) != int(blk_len):
                    sm = torch.arange(blk_len, dtype=torch.long, device=dev)
                sa.slot_mapping = sm
            except Exception:
                sa.slot_mapping = torch.arange(blk_len, dtype=torch.long, device=getattr(current_tokens, 'device', None))
            small_mi.attn_metadata = sa
            small_mi.session_id = session_id
            small_mi.layer_id = layer_id
            # 显式指定稳定键，驱动引擎按块落盘
            small_mi.stable_key = base_file_key

            # Attach minimal payload_meta for diagnostic purposes so we can
            # observe what the connector attempted to persist for this block.
            # Engine currently builds its own payload_meta, but adding this
            # field makes it easy to trace intended offsets in logs and to
            # later wire-through engine behavior if desired.
            small_mi.payload_meta = {
                "token_offset": block_global_offset,
                "block_index": int(blk_idx),
                "block_size": int(blk_len),
                # 存储侧的总长使用块长，避免 slot_mapping 与 seq_len 不匹配警告
                "total_tokens": int(blk_len),
                "stable_key": base_file_key,
                "base_stable_key": base_file_key,
            }

            # compute store_status
            try:
                store_status = self.engine.should_store(small_mi, stable_key=base_file_key)
            except Exception:
                store_status = None

            # call engine.store_kv for this block
            try:
                # Diagnostic log: show the payload_meta we attached so it's
                # easy to trace in the server logs what offsets the
                # connector attempted to persist.
                try:
                    logger.info("[connector] storing block payload_meta=%s key=%s", getattr(small_mi, "payload_meta", {}), base_file_key)
                except Exception:
                    pass

                self.engine.store_kv(
                    self.vllm_config.model_config,
                    self.vllm_config.parallel_config,
                    self.transfer_config,
                    model_executable,
                    small_mi,
                    block_kv_caches,
                    store_status,
                    None,
                    stable_key=base_file_key,
                )
                # mark that we've persisted a block in this invocation
                try:
                    self._last_performed_block_store = True
                except Exception:
                    pass
                __import__("sys").stderr.write(f"[connector] stored block idx={blk_idx} range=[{blk_start},{blk_end}) len={blk_len} key={base_file_key}\n")
                logger.info("[connector] stored block idx=%d range=[%d,%d) len=%d key=%s", blk_idx, blk_start, blk_end, blk_len, base_file_key)
                return True
            except Exception as e:
                logger.exception("[connector] block store failed: %s", e)
                return False

        if not self._store_after_prefill:
            for seq_idx, seq_len in enumerate(seq_lens):
                try:
                    seq_len = int(seq_len)
                    input_tokens = model_input.input_tokens
                    session_id = getattr(model_input, "session_id", None)
                    layer_id = getattr(model_input, "layer_id", None)

                    # 针对 chunk prefill，使用全局累积长度作为偏移（不要用 query_start_loc，它在分块请求时每个 chunk 都从 0 开始）
                    if session_id is None or (isinstance(session_id, str) and session_id.lower() == "none") or (isinstance(session_id, (bytes, bytearray)) and session_id.decode(errors="ignore").lower() == "none"):
                        session_id = b"session_0000"
                    session_norm_bytes = session_id if isinstance(session_id, (bytes, bytearray)) else str(session_id).encode()
                    session_norm = session_norm_bytes.decode("utf-8", errors="ignore")
                    if isinstance(layer_id, str) and layer_id.lower() == "none":
                        layer_id = None
                    layer_norm = layer_id if layer_id is not None else 0
                    
                    # 使用统一逻辑获取 stable_key (request_id 优先 -> session+layer缓存 -> 计算hash)
                    stable_base_key = self._get_stable_key(model_input)
                    
                    engine_stable_key = f"{stable_base_key}.pt"
                    stable_key = engine_stable_key  # 用同一键跟踪累计长度
                    prev_len_global = int(self._prefill_prev_len.get(stable_key, 0))

                    # 简化偏移：按累积长度顺序追加，不做跨请求基准折算，避免负偏移
                    chunk_start = prev_len_global
                    chunk_len = seq_len

                    # 当前 chunk 的 token 视图
                    start_pos_chunk = 0
                    end_pos_chunk = chunk_len
                    current_tokens = input_tokens[start_pos_chunk:end_pos_chunk]

                    try:
                        file_key = self.engine._make_key(current_tokens, session_norm_bytes, layer_norm, stable_key=engine_stable_key)
                    except Exception:
                        file_key = f"seq_{seq_idx}_len_{seq_len}_offset_{chunk_start}"

                    # chunk 内可见长度（一般等于 chunk 长度）
                    avail_chunk = _compute_avail(seq_idx, chunk_len, start_pos_chunk)
                    # 使用可见长度作为实际 chunk 长度，避免后续块判断失效
                    if avail_chunk <= 0:
                        continue
                    chunk_len = avail_chunk
                    chunk_end = chunk_start + chunk_len

                    # 调试：记录 slot_mapping/长度与块偏移
                    try:
                        sm = getattr(model_input.attn_metadata, "slot_mapping", None)
                        sm_dim = getattr(sm, "dim", lambda: None)()
                        sm_len = int(sm.numel()) if hasattr(sm, "numel") else None
                    except Exception:
                        sm_dim = None
                        sm_len = None
                    logger.info(
                        "[connector] seq=%d len=%d chunk_start=%d chunk_end=%d "
                        "avail=%d slot_map_dim=%s slot_map_len=%s prev_len_global=%d",
                        seq_idx, seq_len, chunk_start, chunk_end, avail_chunk, sm_dim, sm_len, prev_len_global)
                    self._dbg(f"[connector-print] seq={seq_idx} len={seq_len} chunk_start={chunk_start} chunk_end={chunk_end} "
                              f"avail={avail_chunk} slot_map_dim={sm_dim} slot_map_len={sm_len} prev_len_global={prev_len_global}")

                    # 计算全局封口块（基于累积长度）
                    sealed_prev = prev_len_global // block_size
                    sealed_now = (chunk_start + avail_chunk) // block_size
                    new_sealed = max(0, sealed_now - sealed_prev)

                    if new_sealed <= 0:
                        # 更新累积长度
                        self._prefill_prev_len[stable_key] = max(prev_len_global, chunk_end)
                        logger.info("[connector] no new blocks seq=%d stable_key=%s prev_len_global=%d chunk_end=%d", seq_idx, stable_key, prev_len_global, chunk_end)
                        self._dbg(f"[connector-print] no_new_blocks seq={seq_idx} stable_key={stable_key} prev_len_global={prev_len_global} chunk_end={chunk_end}")
                        continue

                    logger.info("[connector] detected new sealed blocks key=%s prev_global=%d now_global=%d new_sealed=%d chunk_len=%d", file_key, prev_len_global, chunk_start + avail_chunk, new_sealed, seq_len)
                    self._dbg(f"[connector-print] detected new blocks key={file_key} prev_global={prev_len_global} now_global={chunk_start + avail_chunk} new_sealed={new_sealed} chunk_len={seq_len}")

                    # 逐个新封口块存储（块起止为全局偏移，需换算到当前 chunk 内的相对切片）
                    for i in range(new_sealed):
                        blk_idx = sealed_prev + i
                        blk_start_global = blk_idx * block_size
                        blk_end_global = min(blk_start_global + block_size, chunk_end)
                        blk_len = blk_end_global - blk_start_global
                        if blk_len <= 0:
                            continue
                        blk_start_rel = max(0, blk_start_global - chunk_start)
                        blk_end_rel = blk_start_rel + blk_len
                        self._dbg(f"[connector-print] store_block blk_idx={blk_idx} blk_start_global={blk_start_global} blk_end_global={blk_end_global} blk_start_rel={blk_start_rel} blk_end_rel={blk_end_rel} token_offset_base={chunk_start}")
                        _build_and_store_block(blk_idx, blk_start_rel, blk_end_rel, seq_idx, blk_len, start_pos_chunk, current_tokens, session_norm_bytes, layer_norm, engine_stable_key, chunk_start)

                    # 更新累积长度
                    self._prefill_prev_len[stable_key] = max(prev_len_global, chunk_end)
                    logger.info("[connector] update prev_len stable_key=%s new_prev_len=%d", stable_key, self._prefill_prev_len[stable_key])
                    self._dbg(f"[connector-print] update prev_len stable_key={stable_key} new_prev_len={self._prefill_prev_len[stable_key]}")
                except Exception as e:
                    logger.exception("[connector] per-seq block store failed (debug)")
                    try:
                        import traceback
                        self._dbg(traceback.format_exc())
                    except Exception:
                        pass
            # 仅保留按块存储，跳过后续全量存储逻辑
            return

            # 如果按块已落盘，避免再用截断的全量视角覆盖
            try:
                performed_block_store = getattr(self, "_last_performed_block_store", False)
            except Exception:
                performed_block_store = False
            if performed_block_store:
                return

        # 如果我们已经为某些块执行了存储，则避免再次对整个输入执行store（以免用截断的KV覆盖按块持久化）。
        try:
            performed_block_store = getattr(self, "_last_performed_block_store", False)
        except Exception:
            performed_block_store = False

        # Attach payload_meta to top-level model_input for diagnostic runs too
        pass

        # 仅在启用 store_after_prefill 且按块未执行时执行一次性全量落盘。
        if self._store_after_prefill and not performed_block_store:
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:
                pass
            pass
            store_status = self.engine.should_store(model_input, stable_key=stable_key_arg)
            self.engine.store_kv(
                self.vllm_config.model_config,
                self.vllm_config.parallel_config,
                self.transfer_config,
                model_executable,
                model_input,
                kv_caches,
                store_status,
                hidden_or_intermediate_states,
                stable_key=stable_key_arg,
            )
        else:
            return

    def close(self):
        # 使用模块函数销毁全局引擎单例
        self._destroy_engine()
        logger.info(f"DistributedKVConnector engine {self.engine_name} destroyed")
