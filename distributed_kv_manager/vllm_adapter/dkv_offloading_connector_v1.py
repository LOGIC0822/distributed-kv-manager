# SPDX-License-Identifier: Apache-2.0
# This module duplicates the v1 connector so it lives inside the distributed_kv_manager package path
# allowing `kv_connector_module_path = "distributed_kv_manager.vllm_adapter.dkv_offloading_connector_v1"`

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from types import SimpleNamespace
import logging
import torch

try:
    from vllm.config import VllmConfig
    from vllm.logger import init_logger
except Exception:  # pragma: no cover - compatibility shim
    class VllmConfig:  # type: ignore
        def __init__(self) -> None:
            self.model_config = None
            self.parallel_config = None
            self.v1_config = type("X", (), {"gpu_block_size": 16})()
            self.gpu_block_size = 16

    def init_logger(name: str):  # type: ignore
        import logging as _logging
        return _logging.getLogger(name)
try:
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.distributed.kv_transfer.kv_connector.v1 import (
        KVConnectorBase_V1,
        KVConnectorRole,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
except Exception:  # pragma: no cover - compatibility shim
    # 轻量级兜底 stub，避免不同 vLLM 版本缺失符号导致 ImportError。
    from typing import Any, Dict, List

    class KVConnectorBase_V1:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:
            pass

    class KVConnectorRole:  # type: ignore
        name = "unknown"

    class KVConnectorMetadata:  # type: ignore
        pass

    class KVCacheConfig:  # type: ignore
        pass

    class Request:  # type: ignore
        def __init__(self) -> None:
            self.request_id = "stub"
            self.num_computed_tokens = 0
            # Some versions use req_id
            self.req_id = self.request_id

    class KVCacheBlocks:  # type: ignore
        def get_block_ids(self):
            return [[0]]

    class _Cached:  # helper for scheduled_cached_reqs
        def __init__(self) -> None:
            self.req_ids: List[str] = []
            self.new_block_ids: List[List[int]] = []

    class SchedulerOutput:  # type: ignore
        def __init__(self) -> None:
            self.scheduled_new_reqs: List[Request] = []
            self.num_scheduled_tokens: Dict[str, int] = {}
            self.scheduled_cached_reqs = _Cached()
            # forward_context 可能存在于部分版本，这里让它自引用以复用字段名
            self.forward_context = self
            # requests map, used by some code paths
            self.requests: Dict[str, Request] = {}

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
    start_token: int
    end_token: int


@dataclass
class DKVOffloadingConnectorMetadata(KVConnectorMetadata):  # type: ignore[misc]
    reqs_to_store: dict[str, list[DKVTransferItem]]
    reqs_to_load: dict[str, list[DKVTransferItem]]


from distributed_kv_manager.engine.v1 import init_v1_engine, destroy_v1_engine
from distributed_kv_manager.engine.v1.v1_engine import V1KVEngineImpl


class DKVOffloadingConnector(KVConnectorBase_V1):  # type: ignore[misc]
    """超薄连接器：所有逻辑均委托给 engine.v1.V1KVEngineImpl。"""

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole, kv_cache_config: KVCacheConfig | None = None):  # noqa: D401,E501
        super().__init__(vllm_config, role)
        self._logger = logging.getLogger(self.__class__.__name__)
        # 在 engine 层创建核心实现，并由本类1:1转发
        self._core: V1KVEngineImpl = init_v1_engine(vllm_config, role)
        self._logger.info("DKVOffloadingConnector(thin) forwarding to engine.v1 core, role=%s", getattr(role, "name", "?"))

    # Worker-side -------------------------------------------------------
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        return self._core.register_kv_caches(kv_caches)

    def start_load_kv(self, forward_context: Any, **kwargs) -> None:
        return self._core.start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        return self._core.wait_for_layer_load(layer_name)

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: Any, **kwargs) -> None:
        return self._core.save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

    def wait_for_save(self) -> None:
        return self._core.wait_for_save()

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        return self._core.get_finished(finished_req_ids)

    # Scheduler-side ----------------------------------------------------
    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:  # noqa: E501
        return self._core.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):  # noqa: E501
        return self._core.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        # 直接返回核心实现构建的元数据，供 vLLM 调度器传递到 worker
        return self._core.build_connector_meta(scheduler_output)

    def update_connector_output(self, connector_output: Any):
        # 目前无状态需要更新，保持空实现
        return

    def request_finished(self, request: "Request", block_ids: list[int]) -> tuple[bool, dict[str, Any] | None]:  # noqa: E501
        return self._core.request_finished(request, block_ids)

    def take_events(self):
        return self._core.take_events()

    # --- Metadata binding bridge ---
    # vLLM 会在适配器上绑定/清理 metadata；这里转发到核心，使 worker 侧核心可读取。
    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        try:
            self._core.bind_connector_metadata(connector_metadata)  # type: ignore[arg-type]
        except Exception:
            # 兜底：保存在适配器上
            super().bind_connector_metadata(connector_metadata)

    def clear_connector_metadata(self) -> None:
        try:
            self._core.clear_connector_metadata()
        except Exception:
            pass

    # vLLM 生命周期钩子：与 close 等价，供框架统一调用
    def shutdown(self) -> None:
        self.close()
        super().clear_connector_metadata()

    def close(self):  # noqa: D401
        try:
            destroy_v1_engine()
        except Exception:
            pass


class DKVEngineConnectorV1(DKVOffloadingConnector):
    pass
