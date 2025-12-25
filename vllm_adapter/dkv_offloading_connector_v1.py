# SPDX-License-Identifier: Apache-2.0
# A vLLM v1-style KV Offloading connector that uses distributed-kv-manager engine
# for persistence and retrieval, avoiding any in-place injection or monkey patches.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator
from types import SimpleNamespace
import hashlib

import logging
import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata

# Import our engine public API
from distributed_kv_manager.engine import (
    init_engine,
    retrieve_kv,
    should_retrieve,
    store_kv,
    should_store,
    destroy_engine,
)

logger = init_logger(__name__)


@dataclass
class DKVTransferItem:
    req_id: str
    # token range in absolute token coordinates within the request prompt
    start_token: int
    end_token: int


@dataclass
class DKVOffloadingConnectorMetadata(KVConnectorMetadata):
    # scheduler -> worker plan: which slices to store and which to load
    reqs_to_store: dict[str, list[DKVTransferItem]]
    reqs_to_load: dict[str, list[DKVTransferItem]]


class DKVOffloadingConnector(KVConnectorBase_V1):
    """A minimal v1 connector that delegates KV persistence to our engine.

    Scheduler role: decides which token slices (per request) should be stored
    and which should be loaded, based on scheduler_output.

    Worker role: executes the plan by slicing kv_caches and calling engine.store_kv
    and engine.retrieve_kv with minimal model_input shims.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        # KVConnectorBase_V1 only takes (vllm_config, role); kv_cache_config kept for signature parity.
        super().__init__(vllm_config, role)
        self.vllm_config = vllm_config
        self._engine = init_engine(vllm_config)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.info("DKVOffloadingConnector initialized (role=%s)", role.name)
        try:
            self._block_size = int(getattr(getattr(vllm_config, "v1_config", None), "gpu_block_size", 16))
        except Exception:
            try:
                self._block_size = int(getattr(vllm_config, "gpu_block_size", 16))
            except Exception:
                self._block_size = 16

        # lazy state used by scheduler
        self._requests: dict[str, Request] = {}
        self._request_block_ids: dict[str, list[int]] = {}
        # remember last planned stored token end per request to build incremental slices
        self._planned_end_token: dict[str, int] = {}
        # mark requests that should load from external KV (scheduler sets via get_num_new_matched_tokens)
        self._requests_need_load: dict[str, Request] = {}
        # stable key cache per request to ensure all slices share the same prompt hash
        self._stable_key_cache: dict[str, str] = {}

    # --------------- Worker-side API ---------------
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        # Nothing special needed; worker receives live kv_caches from vLLM
        pass

    def start_load_kv(self, forward_context: Any, **kwargs) -> None:
        # Execute load plan: try retrieving full prompt KV via engine to fill caches.
        meta = self._ensure_worker_meta()
        if not meta.reqs_to_load:
            return
        # In v1 worker, we can access forward_context.attn_metadata and kv caches mapping by layer name.
        # Here we simply call engine.retrieve_kv once per request to fill caches, which is coarse but functional.
        for req_id, items in meta.reqs_to_load.items():
            try:
                # Build a minimal model_input shim for full prompt of the request.
                req = forward_context.requests.get(req_id)
                if req is None:
                    continue
                self._engine_should_retrieve_and_retrieve_full(forward_context, req)
            except Exception:
                self._logger.exception("load_kv failed for req=%s", req_id)

    def wait_for_layer_load(self, layer_name: str) -> None:
        # Synchronous path: nothing to wait.
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: Any,
        **kwargs,
    ) -> None:
        # We perform store at the end via wait_for_save where the full kv_caches are visible.
        return

    def wait_for_save(self) -> None:
        # Execute store plan: slice per request and call engine.store_kv per slice.
        meta = self._ensure_worker_meta()
        if not meta.reqs_to_store:
            return
        try:
            fc = self._last_forward_context  # set by scheduler->worker plumbing in vLLM
        except Exception:
            fc = None
        for req_id, items in meta.reqs_to_store.items():
            try:
                if fc is None:
                    continue
                req = fc.requests.get(req_id)
                if req is None:
                    continue
                for it in items:
                    self._engine_store_slice(fc, req, it.start_token, it.end_token)
            except Exception:
                self._logger.exception("wait_for_save store failed for req=%s", req_id)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        # We run synchronously, so we can return finished immediately
        return set(finished_req_ids), set()

    # --------------- Scheduler-side API ---------------
    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        # 尝试基于完整 prompt 判断是否已有外部缓存可复用。
        matched_tokens = self._find_hit_tokens(request)
        if matched_tokens <= 0:
            return 0, False
        # 记录命中以便后续 build_connector_meta 下发加载计划
        self._requests_need_load[request.request_id] = request
        # 返回还需加载的 token 数（scheduler 期望对齐块大小）
        delta = max(0, matched_tokens - num_computed_tokens)
        return delta, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        # Track request and block ids for later store planning
        self._requests[request.request_id] = request
        block_groups = blocks.get_block_ids()
        block_ids = block_groups[0]
        self._request_block_ids[request.request_id] = block_ids
        if num_external_tokens > 0:
            # scheduler 认为有外部 token 可用，标记为需要加载
            self._requests_need_load[request.request_id] = request

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        # Plan: for each request, store any newly produced prompt tokens since last plan.
        reqs_to_store: dict[str, list[DKVTransferItem]] = {}
        reqs_to_load: dict[str, list[DKVTransferItem]] = {}

        for req_data in scheduler_output.scheduled_new_reqs:
            req_id = req_data.req_id
            req = self._requests.get(req_id)
            if req is None:
                continue
            total_tokens = len(getattr(req, "prompt_token_ids", []) or getattr(req, "input_ids", []) or [])
            aligned_full = (total_tokens // self._block_size) * self._block_size
            if aligned_full <= 0:
                continue
            # 若此前判定可命中，则准备加载计划；否则准备存储计划
            if req_id in self._requests_need_load:
                reqs_to_load.setdefault(req_id, []).append(
                    DKVTransferItem(req_id, 0, aligned_full)
                )
                self._logger.info("[v1_conn] plan load req=%s span=[0,%d)", req_id, aligned_full)
            else:
                prev_stored = self._planned_end_token.get(req_id, 0)
                if prev_stored < aligned_full:
                    reqs_to_store.setdefault(req_id, []).append(
                        DKVTransferItem(req_id, 0, aligned_full)
                    )
                    self._planned_end_token[req_id] = aligned_full
                    self._logger.info("[v1_conn] plan store req=%s span=[0,%d)", req_id, aligned_full)

        # cached requests updates
        cached = scheduler_output.scheduled_cached_reqs
        for idx, req_id in enumerate(cached.req_ids):
            req = self._requests.get(req_id)
            if req is None:
                continue
            new_block_ids_tuple = cached.new_block_ids[idx]
            # If new blocks allocated, map to token span using gpu_block_size
            try:
                gpu_bs = int(getattr(self.vllm_config.v1_config, "gpu_block_size", 16))
            except Exception:
                gpu_bs = int(getattr(self.vllm_config, "gpu_block_size", 16))
            if not new_block_ids_tuple:
                continue
            # new_block_ids_tuple may be nested; flatten to ints
            flat_blocks: list[int] = []
            def _flatten(obj):
                if isinstance(obj, (list, tuple)):
                    for el in obj:
                        _flatten(el)
                else:
                    try:
                        flat_blocks.append(int(obj))
                    except Exception:
                        pass
            _flatten(new_block_ids_tuple)
            if not flat_blocks:
                continue
            min_blk = min(flat_blocks)
            max_blk = max(flat_blocks)
            start_tok = 0  # store from prefix to keep key stable
            end_tok = (max_blk + 1) * gpu_bs
            total_tokens = len(getattr(req, "prompt_token_ids", []) or getattr(req, "input_ids", []) or [])
            aligned_full = (total_tokens // self._block_size) * self._block_size
            end_tok = min(end_tok, aligned_full)
            if end_tok <= start_tok or aligned_full <= 0:
                continue
            prev_end = self._planned_end_token.get(req_id, 0)
            if end_tok > prev_end:
                reqs_to_store.setdefault(req_id, []).append(
                    DKVTransferItem(req_id, start_tok, end_tok)
                )
                self._planned_end_token[req_id] = end_tok

        meta = DKVOffloadingConnectorMetadata(
            reqs_to_store=reqs_to_store,
            reqs_to_load=reqs_to_load,
        )
        # vLLM worker will pass this metadata to the worker side; stash latest fc for worker
        try:
            self._last_forward_context = scheduler_output.forward_context
        except Exception:
            self._last_forward_context = None
        self._connector_metadata = meta
        # 新一轮计划后清理待加载标记（只用一次）
        self._requests_need_load.clear()
        return meta

    def update_connector_output(self, connector_output: Any):
        # No-op for now
        return

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        # Clean state; we save synchronously so return False to free blocks
        req_id = request.request_id
        self._requests.pop(req_id, None)
        self._request_block_ids.pop(req_id, None)
        self._planned_end_token.pop(req_id, None)
        self._stable_key_cache.pop(req_id, None)
        return False, None

    def take_events(self) -> Iterable:
        # No custom events
        return []

    # --------------- Engine bridging helpers ---------------
    def _engine_should_retrieve_and_retrieve_full(self, fc: Any, req: "Request"):
        # Build a model_input-like object compatible with our engine
        # We assume single-sequence per request for prompt phase; extend as needed for batched
        mi = SimpleNamespace()
        # Request carries prompt token ids in req.input_ids (list[int]) or tensor; normalize to tensor
        input_tokens = self._get_aligned_prompt_tokens(req, self._infer_device(fc))
        if input_tokens is None or input_tokens.numel() == 0:
            return
        base_key = self._get_or_build_stable_key(req, input_tokens, layer_id=0)
        mi.input_tokens = input_tokens
        sa = SimpleNamespace()
        seq_len = int(input_tokens.shape[0])
        sa.seq_lens = [seq_len]
        # Build slot mapping 0..N-1 as a fallback; v1 maintains internal maps but we only need contiguous indices
        sa.slot_mapping = torch.arange(sa.seq_lens[0], device=input_tokens.device)
        mi.attn_metadata = sa
        # Use per-request stable session id to avoid cross-request collisions
        try:
            sid = str(getattr(req, "request_id", "v1_session")).encode("utf-8")
        except Exception:
            sid = b"v1_session"
        mi.session_id = sid
        mi.layer_id = 0
        # keep a stable filename (hash of full prompt) to avoid metadata truncation
        mi.stable_key = f"{base_key}_off0_len{seq_len}.pt"

        # Pack kv_caches as a list ordered by model layers, using fc.kv_caches if available
        kv_caches = self._collect_kv_caches(fc)
        rs = should_retrieve(mi)
        retrieve_kv(fc.model, mi, kv_caches, rs)

    def _engine_store_slice(self, fc: Any, req: "Request", start_tok: int, end_tok: int):
        # Slice the prompt tokens and kv caches for the requested span and call engine.store_kv
        if end_tok <= start_tok:
            return
        dev = self._infer_device(fc)
        aligned_tokens = self._get_aligned_prompt_tokens(req, dev)
        if aligned_tokens is None or aligned_tokens.numel() == 0:
            return
        curr_tokens = aligned_tokens[start_tok:end_tok]
        seq_len = int(curr_tokens.shape[0])
        base_key = self._get_or_build_stable_key(req, aligned_tokens, layer_id=0)
        block_key = f"{base_key}_off{start_tok}_len{seq_len}.pt"

        # Build small model_input namespace
        mi = SimpleNamespace()
        mi.input_tokens = curr_tokens
        sa = SimpleNamespace()
        sa.seq_lens = [seq_len]
        sa.slot_mapping = torch.arange(seq_len, device=dev)
        mi.attn_metadata = sa
        try:
            sid = str(getattr(req, "request_id", "v1_session")).encode("utf-8")
        except Exception:
            sid = b"v1_session"
        mi.session_id = sid
        mi.layer_id = 0
        # Annotate token_offset to let engine write absolute offsets into metadata
        mi.payload_meta = {"token_offset": int(start_tok), "block_size": int(seq_len)}
        mi.stable_key = block_key

        kv_caches = self._collect_kv_caches(fc, span=(start_tok, end_tok))
        ss = should_store(mi)
        store_kv(self.vllm_config.model_config, self.vllm_config.parallel_config, None, fc.model, mi, kv_caches, ss, None)

    def _ensure_worker_meta(self) -> DKVOffloadingConnectorMetadata:
        meta = getattr(self, "_connector_metadata", None)
        if isinstance(meta, DKVOffloadingConnectorMetadata):
            return meta
        # default empty plan
        return DKVOffloadingConnectorMetadata(reqs_to_store={}, reqs_to_load={})

    def _collect_kv_caches(self, fc: Any, span: tuple[int, int] | None = None) -> list[torch.Tensor]:
        # Try to collect per-layer kv tensors from forward context; fall back to empty list
        try:
            caches_by_layer = fc.kv_caches  # dict[layer_name, Tensor]
            # Order by insertion for determinism
            tensors = list(caches_by_layer.values())
            if span is None:
                return tensors
            # Optionally slice by token span for 4D or 3D layouts
            start, end = span
            out = []
            for t in tensors:
                try:
                    k = t[0]
                    v = t[1]
                    if k.dim() == 4:
                        # [batch, seq, heads, dim]; assume batch 0
                        out.append(torch.stack([k[0:1, start:end].contiguous(), v[0:1, start:end].contiguous()], dim=0))
                    elif k.dim() == 3:
                        out.append(torch.stack([k[start:end].contiguous(), v[start:end].contiguous()], dim=0))
                    else:
                        out.append(t)
                except Exception:
                    out.append(t)
            return out
        except Exception:
            return []

    def _get_aligned_prompt_tokens(self, req: "Request", device: torch.device) -> torch.Tensor | None:
        tokens = getattr(req, "prompt_token_ids", None) or getattr(req, "input_ids", None)
        if tokens is None:
            return None
        if not torch.is_tensor(tokens):
            tok_tensor = torch.tensor(tokens, dtype=torch.long, device=device)
        else:
            tok_tensor = tokens.to(device=device)
        aligned_len = (int(tok_tensor.shape[0]) // self._block_size) * self._block_size
        if aligned_len <= 0:
            return None
        return tok_tensor[:aligned_len]

    def _build_model_input(self, token_ids: torch.Tensor, slot_mapping: torch.Tensor) -> Any:
        dev = token_ids.device if token_ids.is_cuda else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        toks = token_ids.to(dev)
        smap = slot_mapping.to(dev)
        mi = SimpleNamespace()
        mi.input_tokens = toks
        sa = SimpleNamespace()
        sa.seq_lens = [int(smap.numel())]
        sa.slot_mapping = smap
        mi.attn_metadata = sa
        mi.session_id = None
        mi.layer_id = 0
        mi.payload_meta = {"token_offset": 0, "block_size": int(smap.numel())}
        return mi

    def _make_transfer_item(self, req_id: str, block_ids: list[int]) -> DKVTransferItem:
        if not block_ids:
            return DKVTransferItem(req_id, 0, 0)
        bs = self._block_size
        start = min(block_ids) * bs
        end = (max(block_ids) + 1) * bs
        return DKVTransferItem(req_id=req_id, start_token=start, end_token=end)

    def _find_hit_tokens(self, req: "Request") -> int:
        """调用引擎 should_retrieve 判定是否已有完整前缀命中，返回可复用长度。"""
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        aligned_tokens = self._get_aligned_prompt_tokens(req, dev)
        if aligned_tokens is None or aligned_tokens.numel() == 0:
            return 0
        seq_len = int(aligned_tokens.shape[0])
        smap = torch.arange(seq_len, device=dev)
        mi = SimpleNamespace()
        mi.input_tokens = aligned_tokens
        sa = SimpleNamespace()
        sa.seq_lens = [seq_len]
        sa.slot_mapping = smap
        mi.attn_metadata = sa
        mi.session_id = str(getattr(req, "request_id", "v1_session")).encode("utf-8")
        mi.layer_id = 0
        mi.stable_key = f"{self._get_or_build_stable_key(req, aligned_tokens, 0)}_off0_len{seq_len}.pt"
        rs = should_retrieve(mi)
        # 判定枚举命中
        try:
            if rs is None:
                return 0
            if hasattr(rs, "name"):
                return seq_len if rs.name.lower() == "hit" else 0
            if str(rs).lower().endswith("hit"):
                return seq_len
        except Exception:
            return 0
        return 0

    def _infer_device(self, fc: Any) -> torch.device:
        try:
            device = next(fc.model.parameters()).device
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device

    # --------------- Stable key helpers ---------------
    def _session_hash(self, session_bytes: bytes) -> str:
        try:
            return hashlib.blake2b(session_bytes, digest_size=8).hexdigest()
        except Exception:
            return "session"

    def _get_or_build_stable_key(self, req: "Request", full_tokens: torch.Tensor, layer_id: int = 0) -> str:
        """基于完整 prompt 生成稳定文件名前缀，所有块共享同一哈希。"""
        req_id = getattr(req, "request_id", None)
        if req_id is not None and req_id in self._stable_key_cache:
            return self._stable_key_cache[req_id]
        try:
            full_hash = self._engine._tensor_hash(full_tokens)
        except Exception:
            full_hash = "nohash"
        try:
            sid_bytes = str(req_id or "v1_session").encode("utf-8")
        except Exception:
            sid_bytes = b"v1_session"
        session_part = self._session_hash(sid_bytes)
        base = f"kv_s{session_part}_l{layer_id}_{full_hash}"
        if req_id is not None:
            self._stable_key_cache[req_id] = base
        return base

    # --------------- Cleanup ---------------
    def close(self):
        try:
            destroy_engine()
        except Exception:
            pass


# Alias class name to avoid collisions with any pre-registered connector names in vLLM
# This allows using a different kv_connector name together with kv_connector_module_path
# to force import from this module even if a same-name entry exists in vLLM registry.
class DKVEngineConnectorV1(DKVOffloadingConnector):
    pass
