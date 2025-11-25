"""Minimal injection to capture and store sealed KV blocks from vLLM.

This file provides a small, non-invasive monkey-patch that wraps
`vllm.worker.model_runner.ModelRunner.execute_model` and, after the
forward, inspects the visible KV lengths per-sequence. When it detects
newly sealed blocks (by BLOCK_SIZE), it slices the corresponding token
range and per-layer KV slices and calls the existing engine.store_kv to
persist each block individually.

This is intended as a minimal prototype placed under `vllm_adapter/`.
It avoids deep changes in vLLM and reuses the project's engine API.
"""
from types import SimpleNamespace
import copy
import torch
import logging

from distributed_kv_manager.engine import init_engine, store_kv, should_store

logger = logging.getLogger("inject_multi_block")
logger.setLevel(logging.INFO)

# Hardcoded parameters for the prototype
BLOCK_SIZE = 16
FORCE_STORE_THRESHOLD = 3

# state: remember previous visible length per key
_prefill_prev_len = {}

# guard to avoid double-patching
_patched = False


def _make_block_kv_cache(orig_kv_cache: torch.Tensor, seq_idx: int, blk_start: int, blk_end: int) -> torch.Tensor:
    """Return a new small kv_cache tensor shaped like [2, batch?, seq_block, ...]
    matching the engine.split_kv_cache expectation.
    """
    # orig_kv_cache expected shape: [2, ...]
    key_cache = orig_kv_cache[0]
    value_cache = orig_kv_cache[1]

    if key_cache.dim() == 4:
        # layout: [batch, seq, num_heads, head_dim]
        key_block = key_cache[seq_idx:seq_idx+1, blk_start:blk_end].contiguous()
        value_block = value_cache[seq_idx:seq_idx+1, blk_start:blk_end].contiguous()
    elif key_cache.dim() == 3:
        # layout: [total_tokens, num_heads, head_dim]
        key_block = key_cache[blk_start:blk_end].contiguous()
        value_block = value_cache[blk_start:blk_end].contiguous()
    else:
        raise RuntimeError(f"Unsupported kv layout dim={key_cache.dim()}")

    return torch.stack([key_block, value_block], dim=0)


def inject():
    global _patched
    if _patched:
        return

    import os as _os
    mode = str(_os.environ.get("DKV_INJECTOR", "auto")).lower()
    force_multi = str(_os.environ.get("DKV_FORCE_MULTI", "0")).lower() in ("1","true","yes","on")

    # If user explicitly asked for multi, skip minimal_hooks path
    use_minimal = (mode in ("auto", "minimal")) and not force_multi

    if use_minimal:
        # Prefer to use the repo-local minimal hooks implementation if available
        try:
            # try package-qualified import first
            try:
                from vllm_adapter import minimal_hooks as mh
            except Exception:
                # fallback to local import
                import minimal_hooks as mh
            logger.info("inject_multi_block: found minimal_hooks, calling inject() to install hooks")
            try:
                mh.inject()
                _patched = True
                return
            except Exception as e:
                logger.warning("inject_multi_block: minimal_hooks.inject failed: %s", e)
                # fall through to lightweight patch
        except Exception:
            # minimal_hooks not available; continue with lightweight patch
            pass

    try:
        import vllm.worker.model_runner as mr
    except Exception as e:
        logger.warning("无法导入 vllm.worker.model_runner，注入取消: %s", e)
        return

    original_execute = getattr(mr.ModelRunner, "execute_model", None)
    if original_execute is None:
        logger.warning("ModelRunner.execute_model 未找到，注入取消")
        return

    def wrapped_execute_model(self, model_input, kv_caches, intermediate_tensors, num_steps: int = 1):
        # ensure engine is initialized (loads config.json if needed)
        try:
            eng = init_engine()
        except Exception:
            eng = None

        # call original forward
        outputs = original_execute(self, model_input, kv_caches, intermediate_tensors, num_steps=num_steps)

        # If engine not available, skip block store
        if eng is None:
            return outputs

        # basic sanity
        try:
            seq_lens = list(model_input.attn_metadata.seq_lens)
        except Exception:
            return outputs

        # iterate sequences and detect newly sealed blocks using first layer visibility
        for seq_idx, seq_len in enumerate(seq_lens):
            try:
                start_pos = int(sum(int(x) for x in seq_lens[:seq_idx]))
                end_pos = start_pos + int(seq_len)
                current_tokens = model_input.input_tokens[start_pos:end_pos]
                session_id = getattr(model_input, "session_id", None)
                layer_id = getattr(model_input, "layer_id", None)
                file_key = eng._make_key(current_tokens, session_id, layer_id)

                # compute avail from kv_caches first layer
                avail = 0
                if not kv_caches:
                    continue
                first_layer = kv_caches[0]
                if first_layer is None:
                    continue
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

                prev_len = int(_prefill_prev_len.get(file_key, 0))
                sealed_prev = prev_len // BLOCK_SIZE
                sealed_now = avail // BLOCK_SIZE
                new_sealed = max(0, sealed_now - sealed_prev)
                if new_sealed <= 0:
                    # update prev_len to current avail to avoid repeated negatives
                    _prefill_prev_len[file_key] = max(prev_len, avail)
                    continue

                logger.info("[inject_mb] key=%s prev=%d now=%d sealed_new=%d", file_key, prev_len, avail, new_sealed)

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
                        try:
                            block_kv = _make_block_kv_cache(layer_kv, seq_idx, blk_start, blk_end)
                        except Exception as e:
                            logger.warning("[inject_mb] 无法构造 block kv (layer): %s", e)
                            block_kv = None
                        block_kv_caches.append(block_kv)

                    # build a tiny model_input-like object with required attrs
                    small_mi = SimpleNamespace()
                    small_mi.input_tokens = current_tokens[blk_start:blk_end]
                    # attn_metadata must have seq_lens and slot_mapping
                    sa = SimpleNamespace()
                    sa.seq_lens = [blk_len]
                    # derive small slot mapping if possible
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
                        if sm.numel() != blk_len:
                            sm = torch.arange(blk_len, dtype=torch.long, device=dev)
                        sa.slot_mapping = sm
                    except Exception:
                        sa.slot_mapping = torch.arange(blk_len, dtype=torch.long, device=getattr(current_tokens, 'device', None))
                    small_mi.attn_metadata = sa
                    small_mi.session_id = session_id
                    small_mi.layer_id = layer_id

                    # Attach minimal payload_meta so engine can embed token_offset
                    # into metadata for correct aggregation during retrieve.
                    try:
                        small_mi.payload_meta = {
                            "token_offset": int(start_pos + blk_start),
                            "block_index": int(blk_idx),
                            "block_size": int(blk_len),
                            "total_tokens": int(seq_len),
                        }
                    except Exception:
                        pass

                    # compute store_status (engine-level policy)
                    try:
                        store_status = should_store(small_mi)
                    except Exception:
                        store_status = None

                    # call engine.store_kv with model info available on ModelRunner
                    try:
                        model_executable = getattr(self, "model", None)
                        # transfer_config not required by current engine implementation; pass None
                        store_kv(self.model_config, self.parallel_config, None, model_executable, small_mi, block_kv_caches, store_status)
                        logger.info("[inject_mb] stored block idx=%d range=[%d,%d) len=%d key=%s", blk_idx, blk_start, blk_end, blk_len, file_key)
                    except Exception as e:
                        logger.error("[inject_mb] block store failed: %s", e, exc_info=True)

                # update prev_len
                _prefill_prev_len[file_key] = max(prev_len, avail)

            except Exception as e:
                logger.debug("[inject_mb] per-seq handling failed: %s", e)

        return outputs

    # patch
    setattr(mr.ModelRunner, "execute_model", wrapped_execute_model)
    _patched = True
    logger.info("inject_multi_block: patched ModelRunner.execute_model (minimal multi-block injector)")
    try:
        import sys as _sys
        _sys.stderr.write("[inject_mb] patched ModelRunner.execute_model\n")
    except Exception:
        pass


if __name__ == "__main__":
    inject()
