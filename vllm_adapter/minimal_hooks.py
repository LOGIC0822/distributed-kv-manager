"""Minimal, focused hooks extracted to capture sealed blocks.

This module patches model-level forward (for Llama-like models) to run a small
block-capture routine right after the model forward, where `input_ids`,
`kv_caches` and `attn_metadata` are available and up-to-date. It then calls
the project's engine.store_kv for each newly sealed block.

"""
import logging
from types import SimpleNamespace
import torch

from distributed_kv_manager.engine import init_engine, store_kv, should_store

logger = logging.getLogger("minimal_hooks")
logger.setLevel(logging.INFO)

# prototype constants
BLOCK_SIZE = 16

# track previous visible len per key
_prev_len = {}


def _make_block_kv_cache(orig_kv_cache: torch.Tensor, seq_idx: int, blk_start: int, blk_end: int) -> torch.Tensor:
    key_cache = orig_kv_cache[0]
    value_cache = orig_kv_cache[1]
    if key_cache.dim() == 4:
        key_block = key_cache[seq_idx:seq_idx+1, blk_start:blk_end].contiguous()
        value_block = value_cache[seq_idx:seq_idx+1, blk_start:blk_end].contiguous()
    elif key_cache.dim() == 3:
        key_block = key_cache[blk_start:blk_end].contiguous()
        value_block = value_cache[blk_start:blk_end].contiguous()
    else:
        raise RuntimeError("Unsupported kv layout")
    return torch.stack([key_block, value_block], dim=0)


def _capture_blocks(model, input_ids, kv_caches, attn_metadata):
    """Capture newly sealed blocks and store them via engine.store_kv.

    model: the model module (used as model_executable param when calling store_kv)
    input_ids: tensor of tokens for the entire sequence(s)
    kv_caches: list of per-layer kv cache tensors
    attn_metadata: attention metadata object containing seq_lens and slot_mapping
    """
    try:
        eng = init_engine()
    except Exception:
        eng = None
    if eng is None:
        return

    try:
        seq_lens = list(attn_metadata.seq_lens)
    except Exception:
        return

    # iterate sequences
    for seq_idx, seq_len in enumerate(seq_lens):
        start_pos = int(sum(int(x) for x in seq_lens[:seq_idx]))
        end_pos = start_pos + int(seq_len)
        current_tokens = input_ids[start_pos:end_pos]
        session_id = getattr(attn_metadata, 'session_id', None)
        layer_id = getattr(attn_metadata, 'layer_id', None)

        # build key using engine helper
        try:
            file_key = eng._make_key(current_tokens, session_id, layer_id)
        except Exception:
            file_key = f"seq_{seq_idx}_len_{seq_len}"

        # compute avail from first layer if present
        if not kv_caches or kv_caches[0] is None:
            continue
        try:
            k0 = kv_caches[0][0]
            if k0.dim() == 4:
                avail = int(k0[seq_idx].shape[0]) if seq_idx < k0.shape[0] else 0
            elif k0.dim() == 3:
                total_tokens = int(k0.shape[0])
                avail = max(0, total_tokens - start_pos)
                avail = min(avail, seq_len)
            else:
                avail = 0
        except Exception:
            avail = 0

        prev = int(_prev_len.get(file_key, 0))
        sealed_prev = prev // BLOCK_SIZE
        sealed_now = avail // BLOCK_SIZE
        new_sealed = max(0, sealed_now - sealed_prev)
        if new_sealed <= 0:
            _prev_len[file_key] = max(prev, avail)
            continue

        logger.info("[minimal_hooks] key=%s prev=%d now=%d sealed_new=%d", file_key, prev, avail, new_sealed)

        # for each new sealed block, build block kv and call store_kv
        for i in range(new_sealed):
            blk_idx = sealed_prev + i
            blk_start = blk_idx * BLOCK_SIZE
            blk_end = min(blk_start + BLOCK_SIZE, seq_len)
            blk_len = blk_end - blk_start
            if blk_len <= 0:
                continue

            block_kv_caches = []
            for layer_kv in kv_caches:
                try:
                    block_kv = _make_block_kv_cache(layer_kv, seq_idx, blk_start, blk_end)
                except Exception as e:
                    logger.debug("[minimal_hooks] skip layer block slice: %s", e)
                    block_kv = None
                block_kv_caches.append(block_kv)

            # build small model_input
            small_mi = SimpleNamespace()
            small_mi.input_tokens = current_tokens[blk_start:blk_end]
            sa = SimpleNamespace()
            sa.seq_lens = [blk_len]
            try:
                orig_slot = attn_metadata.slot_mapping
                if orig_slot is None:
                    sa.slot_mapping = torch.arange(blk_len)
                else:
                    if orig_slot.dim() == 1:
                        sa.slot_mapping = orig_slot[start_pos + blk_start: start_pos + blk_end]
                    else:
                        sa.slot_mapping = orig_slot[seq_idx][:blk_len]
            except Exception:
                sa.slot_mapping = torch.arange(blk_len)
            small_mi.attn_metadata = sa
            small_mi.session_id = None
            small_mi.layer_id = None

            # Attach per-block payload meta so the engine/retriever can
            # place each stored chunk at the correct global token offset.
            # This mirrors the richer connector path and enables correct
            # aggregation of multiple blocks during retrieve.
            try:
                small_mi.payload_meta = {
                    "token_offset": int(start_pos + blk_start),
                    "block_index": int(blk_idx),
                    "block_size": int(blk_len),
                    "total_tokens": int(seq_len),
                }
            except Exception:
                # Best-effort; if something goes wrong, proceed without extra meta
                pass

            try:
                store_status = should_store(small_mi)
            except Exception:
                store_status = None

            try:
                # model passed as model_executable parameter
                store_kv(None, None, None, model, small_mi, block_kv_caches, store_status)
                logger.info("[minimal_hooks] stored block idx=%d range=[%d,%d) len=%d key=%s", blk_idx, blk_start, blk_end, blk_len, file_key)
                try:
                    import sys as _sys
                    _sys.stderr.write(f"[minimal_hooks] stored block idx={blk_idx} range=[{blk_start},{blk_end}) len={blk_len} key={file_key}\n")
                except Exception:
                    pass
            except Exception as e:
                logger.error("[minimal_hooks] store_kv failed: %s", e, exc_info=True)

        _prev_len[file_key] = max(prev, avail)


def inject_minimal_llama():
    try:
        import vllm.model_executor.models.llama as llama_mod
    except Exception as e:
        logger.warning("minimal_hooks: cannot import llama model module: %s", e)
        return

    orig = getattr(llama_mod.LlamaModel, "forward", None)
    if orig is None:
        logger.warning("minimal_hooks: LlamaModel.forward not found")
        return

    def wrapped_forward(self, input_ids, positions, kv_caches, attn_metadata, intermediate_tensors, inputs_embeds=None):
        out = orig(self, input_ids, positions, kv_caches, attn_metadata, intermediate_tensors, inputs_embeds=inputs_embeds)
        try:
            _capture_blocks(self, input_ids, kv_caches, attn_metadata)
        except Exception as e:
            logger.debug("minimal_hooks: capture failed: %s", e)
        return out

    setattr(llama_mod.LlamaModel, "forward", wrapped_forward)
    logger.info("minimal_hooks: injected LlamaModel.forward wrapper for block capture")
    try:
        import sys as _sys
        _sys.stderr.write("[minimal_hooks] patched LlamaModel.forward\n")
    except Exception:
        pass


def inject():
    inject_minimal_llama()
    try:
        import sys as _sys
        _sys.stderr.write("[minimal_hooks] inject() called\n")
    except Exception:
        pass


if __name__ == '__main__':
    inject()
