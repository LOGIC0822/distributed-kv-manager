from typing import TYPE_CHECKING, Union
import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.sequence import IntermediateTensors
from vllm.logger import init_logger

# 硬编码的预填/分块参数，便于调试；后续可改回从配置读取
STORE_AFTER_PREFILL = True   # 开启：预填充完成后一次性落全量
ALLOW_PARTIAL_PREFILL_STORE = False  # 关闭：不在预填未完成时落部分
DEBUG_BLOCK_ACCOUNTING = True
BLOCK_SIZE = 16
FORCE_STORE_THRESHOLD = 3  # 最多跳过次数

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

logger = init_logger(__name__)

import sys as _sys  # diag
try: _sys.stderr.write(f"[dkv.connector] module import: STORE_AFTER_PREFILL={STORE_AFTER_PREFILL} ALLOW_PARTIAL_PREFILL_STORE={ALLOW_PARTIAL_PREFILL_STORE} BLOCK_SIZE={BLOCK_SIZE}\\n")
except Exception: pass
# Force connector-side per-block persistence during prefill so we don't rely on
# external injection. These assignments override the defaults above at import
# time and will be read when the connector is constructed.
STORE_AFTER_PREFILL = False
ALLOW_PARTIAL_PREFILL_STORE = True

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
        # 硬编码开关（不从配置读取，便于调试）
        self._store_after_prefill = STORE_AFTER_PREFILL
        self._allow_partial_prefill_store = ALLOW_PARTIAL_PREFILL_STORE
        logger.info(f"[connector] store_after_prefill(hardcoded)={self._store_after_prefill}"); __import__("sys").stderr.write(f"[connector] store_after_prefill={self._store_after_prefill}\n")
        logger.info(f"[connector] allow_partial_prefill_store(hardcoded)={self._allow_partial_prefill_store}"); __import__("sys").stderr.write(f"[connector] allow_partial_prefill_store={self._allow_partial_prefill_store}\n")
        # 预填延迟计数（避免永远跳过）：按文件键统计尝试次数
        self._prefill_attempts = {}
        # 预填块统计与日志开关
        self._debug_block_accounting = DEBUG_BLOCK_ACCOUNTING
        # 跟踪每个文件键的上次可见长度（用于计算Δ与封口数）
        self._prefill_prev_len = {}
        self.retrieve_kv = retrieve_kv
        self.should_retrieve = should_retrieve
        self.store_kv = store_kv
        self.should_store = should_store
        self._destroy_engine = destroy_engine
        self.store_status = StoreStatus
        self.retrieve_status = RetrieveStatus
        self.engine_name = getattr(config, "engine_id", "unknown_engine")

        logger.info(f"DistributedKVConnector initialized with engine {self.engine_name}")

    def recv_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: list[torch.Tensor]
    ) -> tuple[Union[torch.Tensor, IntermediateTensors], bool, "ModelInputForGPUWithSamplingMetadata"]:

        retrieve_status = self.engine.should_retrieve(model_input)
        hidden_or_intermediate_states, bypass_model_exec, model_input  = self.engine.retrieve_kv(
            model_executable, model_input, kv_caches, retrieve_status
        )
        return hidden_or_intermediate_states, bypass_model_exec, model_input

    def send_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: list[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor, IntermediateTensors],
    ) -> None:
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

                if not kv_caches:
                    logger.debug("[connector] 无 kv_caches，跳过判定")
                else:
                    first_k = kv_caches[0][0]
                    prefill_initial_ready = True
                    for seq_idx, seq_len in enumerate(seq_lens):
                        seq_len = int(seq_len)
                        start_pos = int(sum(int(x) for x in seq_lens[:seq_idx]))
                        end_pos = start_pos + seq_len
                        try:
                            current_tokens = input_tokens[start_pos:end_pos]
                            file_key = self.engine._make_key(current_tokens, session_id, layer_id)
                        except Exception:
                            file_key = f"seq_{seq_idx}_len_{seq_len}"

                        if first_k.dim() == 4:
                            avail = int(first_k[seq_idx].shape[0]) if seq_idx < first_k.shape[0] else 0
                        elif first_k.dim() == 3:
                            total_tokens = int(first_k.shape[0])
                            avail = min(max(0, total_tokens - start_pos), seq_len)
                        else:
                            avail = 0

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

        def _make_block_kv_cache(orig_kv_cache: torch.Tensor, seq_idx: int, blk_start: int, blk_end: int):
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

        # iterate sequences and detect newly sealed blocks using first layer visibility
        if kv_caches:
            first_layer = kv_caches[0]
        else:
            first_layer = None

        if not self._store_after_prefill:
            for seq_idx, seq_len in enumerate(seq_lens):
                try:
                    seq_len = int(seq_len)
                    start_pos = int(sum(int(x) for x in seq_lens[:seq_idx]))
                    end_pos = start_pos + seq_len
                    input_tokens = model_input.input_tokens
                    current_tokens = input_tokens[start_pos:end_pos]
                    session_id = getattr(model_input, "session_id", None)
                    layer_id = getattr(model_input, "layer_id", None)
                    try:
                        file_key = self.engine._make_key(current_tokens, session_id, layer_id)
                    except Exception:
                        file_key = f"seq_{seq_idx}_len_{seq_len}"

                    # compute avail
                    avail = 0
                    if first_layer is not None:
                        try:
                            k0 = first_layer[0]
                            if k0.dim() == 4:
                                avail = int(k0[seq_idx].shape[0]) if seq_idx < k0.shape[0] else 0
                            elif k0.dim() == 3:
                                total_tokens = int(k0.shape[0])
                                avail = max(0, total_tokens - start_pos)
                                avail = min(avail, seq_len)
                        except Exception:
                            avail = 0

                    prev_len = int(self._prefill_prev_len.get(file_key, 0))
                    sealed_prev = prev_len // BLOCK_SIZE
                    sealed_now = avail // BLOCK_SIZE
                    new_sealed = max(0, sealed_now - sealed_prev)
                    if new_sealed <= 0:
                        # update prev_len
                        self._prefill_prev_len[file_key] = max(prev_len, avail)
                        continue

                    logger.info("[connector] detected new sealed blocks key=%s prev=%d now=%d new_sealed=%d", file_key, prev_len, avail, new_sealed)

                    # for each newly sealed block, slice and store
                    for i in range(new_sealed):
                        blk_idx = sealed_prev + i
                        blk_start = blk_idx * BLOCK_SIZE
                        blk_end = min(blk_start + BLOCK_SIZE, seq_len)
                        blk_len = blk_end - blk_start
                        if blk_len <= 0:
                            continue

                        # build per-block kv_caches (list per layer)
                        block_kv_caches = []
                        for layer_kv in kv_caches:
                            block_kv = _make_block_kv_cache(layer_kv, seq_idx, blk_start, blk_end)
                            block_kv_caches.append(block_kv)

                        # build a tiny model_input-like object with required attrs
                        from types import SimpleNamespace
                        small_mi = SimpleNamespace()
                        small_mi.input_tokens = current_tokens[blk_start:blk_end]
                        sa = SimpleNamespace()
                        sa.seq_lens = [blk_len]
                        # try to derive slot_mapping
                        try:
                            orig_slot = model_input.attn_metadata.slot_mapping
                            if orig_slot is None:
                                sa.slot_mapping = torch.arange(blk_len)
                            else:
                                if orig_slot.dim() == 1:
                                    sa.slot_mapping = orig_slot[start_pos + blk_start: start_pos + blk_end]
                                else:
                                    sa.slot_mapping = orig_slot[seq_idx][:blk_len]
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

                        # Attach minimal payload_meta for diagnostic purposes so we can
                        # observe what the connector attempted to persist for this block.
                        # Engine currently builds its own payload_meta, but adding this
                        # field makes it easy to trace intended offsets in logs and to
                        # later wire-through engine behavior if desired.
                        small_mi.payload_meta = {
                            "token_offset": int(start_pos + blk_start),
                            "block_index": int(blk_idx),
                            "block_size": int(blk_len),
                            "total_tokens": int(seq_len),
                        }

                        # compute store_status
                        try:
                            store_status = self.should_store(small_mi)
                        except Exception:
                            store_status = None

                        # call engine.store_kv for this block
                        try:
                            # Diagnostic log: show the payload_meta we attached so it's
                            # easy to trace in the server logs what offsets the
                            # connector attempted to persist.
                            try:
                                logger.info("[connector] storing block payload_meta=%s key=%s", getattr(small_mi, "payload_meta", {}), file_key)
                            except Exception:
                                pass

                            self.store_kv(
                                self.vllm_config.model_config,
                                self.vllm_config.parallel_config,
                                self.transfer_config,
                                model_executable,
                                small_mi,
                                block_kv_caches,
                                store_status,
                                None,
                            )
                            # mark that we've persisted a block in this invocation
                            try:
                                self._last_performed_block_store = True
                            except Exception:
                                pass
                            __import__("sys").stderr.write(f"[connector] stored block idx={blk_idx} range=[{blk_start},{blk_end}) len={blk_len} key={file_key}\n"); logger.info("[connector] stored block idx=%d range=[%d,%d) len=%d key=%s", blk_idx, blk_start, blk_end, blk_len, file_key)
                        except Exception as e:
                            logger.exception("[connector] block store failed: %s", e)

                    # update prev_len
                    self._prefill_prev_len[file_key] = max(prev_len, avail)
                except Exception as e:
                    logger.warning("[connector] per-seq block store failed: %s", e)

        # 如果我们已经为某些块执行了存储，则避免再次对整个输入执行store（以免用截断的KV覆盖按块持久化）。
        try:
            performed_block_store = getattr(self, "_last_performed_block_store", False)
        except Exception:
            performed_block_store = False

        # Attach payload_meta to top-level model_input for diagnostic runs too
        try:
            if not hasattr(model_input, "payload_meta"):
                model_input.payload_meta = {}
            model_input.payload_meta.update({"connector_prev_len_map": self._prefill_prev_len})
        except Exception:
            pass

        # 仅在启用 store_after_prefill 且按块未执行时执行一次性全量落盘。
        if self._store_after_prefill and not performed_block_store:
            # 再次稳妥校验：所有序列在本次 forward 结束时是否已完全可见
            try:
                # 等待 CUDA 异步内核完成，避免读取到未写完的可见长度
                try:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                except Exception:
                    pass
                first_k = kv_caches[0][0]
                seq_lens = list(model_input.attn_metadata.seq_lens)
                all_final_ready = True
                for seq_idx, seq_len in enumerate(seq_lens):
                    seq_len = int(seq_len)
                    if first_k.dim() == 4:
                        # 4D 视图下 shape 的 seq 维通常是容量而非可见长度，改用 slot_mapping 切片长度进行判定
                        try:
                            sm = getattr(model_input.attn_metadata, "slot_mapping", None)
                            if sm is not None and sm.dim() >= 1:
                                start_pos = int(sum(int(x) for x in seq_lens[:seq_idx]))
                                end_pos = start_pos + seq_len
                                if sm.dim() == 1:
                                    avail_final = int((sm[start_pos:end_pos]).shape[0])
                                else:
                                    avail_final = int((sm[seq_idx][:seq_len]).shape[0])
                            else:
                                # 回退：以期望长度作为可见长度（交由引擎内部按最小可取长度裁剪）
                                avail_final = seq_len
                        except Exception:
                            avail_final = seq_len
                    else:
                        # 3D 视图回退：交由引擎以最小长度处理
                        avail_final = seq_len
                    if avail_final < seq_len:
                        all_final_ready = False
                        break

                # 调试：对比初始与最终可见判定
                try:
                    init_flag = getattr(model_input, "_prefill_initial_ready", None)
                    logger.info("[connector] store_after_prefill: initial_ready=%s, final_ready=%s", init_flag, all_final_ready)
                except Exception:
                    pass

                if not all_final_ready:
                    return
            except Exception:
                # 如果复查失败，则保守起见不做一次性落盘
                return

            store_status = self.engine.should_store(model_input)
            self.engine.store_kv(
                self.vllm_config.model_config,
                self.vllm_config.parallel_config,
                self.transfer_config,
                model_executable,
                model_input,
                kv_caches,
                store_status,
                hidden_or_intermediate_states,
            )

    def close(self):
        # 使用模块函数销毁全局引擎单例
        self._destroy_engine()
        logger.info(f"DistributedKVConnector engine {self.engine_name} destroyed")
