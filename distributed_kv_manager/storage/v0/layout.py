from __future__ import annotations

from typing import Tuple, Optional

import torch


def pack_kv_local_v0(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    input_tokens: torch.Tensor,
    roi: torch.Tensor,
) -> bytes:
    """v0 layout: 单层/多层 KV 的本地打包逻辑。

    形状假设与老引擎一致：
    - k_cache/v_cache: 任意形状张量（通常为 [num_layers, seq, ...] 或 [seq, ...]）
    - input_tokens: [seq]
    - roi: [seq] bool
    """
    import io

    payload = {
        "k_cache": k_cache.cpu(),
        "v_cache": v_cache.cpu(),
        "input_tokens": input_tokens.cpu(),
        "roi": roi.cpu(),
    }
    buf = io.BytesIO()
    torch.save(payload, buf)
    return buf.getvalue()


def unpack_kv_local_v0(
    data: bytes,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """v0 layout: 从本地打包格式中恢复 KV（k_cache, v_cache）。"""
    import io

    try:
        buf = io.BytesIO(data)
        loaded = torch.load(buf, map_location="cpu")
        k_cache = loaded["k_cache"]
        v_cache = loaded["v_cache"]
        return k_cache, v_cache
    except Exception:
        return None, None

