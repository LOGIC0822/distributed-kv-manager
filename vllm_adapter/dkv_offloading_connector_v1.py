# SPDX-License-Identifier: Apache-2.0
# A vLLM v1-style KV Offloading connector that uses distributed-kv-manager engine
# for persistence and retrieval, avoiding any in-place injection or monkey patches.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator
from types import SimpleNamespace

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
        super().__init__(vllm_config, role, kv_cache_config)
        self.vllm_config = vllm_config
        self.role = role
        self._engine = init_engine(vllm_config)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.info("DKVOffloadingConnector initialized (role=%s)", role.name)

        # lazy state used by scheduler
        self._requests: dict[str, Request] = {}
        self._request_block_ids: dict[str, list[int]] = {}
        # remember last planned token end per request to build incremental slices
        self._planned_end_token: dict[str, int] = {}

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
        # We don't perform remote hits here; return 0 to avoid extra async loads.
        return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        # Track request and block ids for later store planning
        self._requests[request.request_id] = request
        block_groups = blocks.get_block_ids()
        block_ids = block_groups[0]
        self._request_block_ids[request.request_id] = block_ids

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        # Plan: for each request, store any newly produced prompt tokens since last plan.
        reqs_to_store: dict[str, list[DKVTransferItem]] = {}
        reqs_to_load: dict[str, list[DKVTransferItem]] = {}

        for req_data in scheduler_output.scheduled_new_reqs:
            req_id = req_data.req_id
            req = self._requests.get(req_id)
            if req is None:
                continue
            new_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            if new_tokens <= 0:
                continue
            prev_end = self._planned_end_token.get(req_id, 0)
            end_tok = req.num_computed_tokens + new_tokens
            if end_tok > prev_end:
                reqs_to_store.setdefault(req_id, []).append(
                    DKVTransferItem(req_id, prev_end, end_tok)
                )
                self._planned_end_token[req_id] = end_tok

        # cached requests updates
        cached = scheduler_output.scheduled_cached_reqs
        for idx, req_id in enumerate(cached.req_ids):
            req = self._requests.get(req_id)
            if req is None:
                continue
            new_block_ids_tuple = cached.new_block_ids[idx]
            # If new blocks allocated, map to token span using gpu_block_size
            try:
                gpu_bs = self.vllm_config.v1_config.gpu_block_size
            except Exception:
                gpu_bs = getattr(self.vllm_config, "gpu_block_size", 16)
            if not new_block_ids_tuple:
                continue
            # Compute token span for the new contiguous block run
            # Simple heuristic: from min(new_block_ids) to max(new_block_ids)+1 in tokens
            min_blk = min(new_block_ids_tuple)
            max_blk = max(new_block_ids_tuple)
            start_tok = min_blk * gpu_bs
            end_tok = (max_blk + 1) * gpu_bs
            prev_end = self._planned_end_token.get(req_id, 0)
            if end_tok > prev_end:
                reqs_to_store.setdefault(req_id, []).append(
                    DKVTransferItem(req_id, max(prev_end, start_tok), end_tok)
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
        input_ids = getattr(req, "input_ids", None)
        if input_ids is None:
            return
        if not torch.is_tensor(input_ids):
            input_tokens = torch.tensor(input_ids, dtype=torch.long, device=self._infer_device(fc))
        else:
            input_tokens = input_ids
        mi.input_tokens = input_tokens
        sa = SimpleNamespace()
        sa.seq_lens = [int(input_tokens.shape[0])]
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

        # Pack kv_caches as a list ordered by model layers, using fc.kv_caches if available
        kv_caches = self._collect_kv_caches(fc)
        rs = should_retrieve(mi)
        retrieve_kv(fc.model, mi, kv_caches, rs)

    def _engine_store_slice(self, fc: Any, req: "Request", start_tok: int, end_tok: int):
        # Slice the prompt tokens and kv caches for the requested span and call engine.store_kv
        if end_tok <= start_tok:
            return
        input_ids = getattr(req, "input_ids", None)
        if input_ids is None:
            return
        dev = self._infer_device(fc)
        if not torch.is_tensor(input_ids):
            full_tokens = torch.tensor(input_ids, dtype=torch.long, device=dev)
        else:
            full_tokens = input_ids
        curr_tokens = full_tokens[start_tok:end_tok]
        seq_len = int(curr_tokens.shape[0])

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

    def _infer_device(self, fc: Any) -> torch.device:
        try:
            device = next(fc.model.parameters()).device
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device

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
