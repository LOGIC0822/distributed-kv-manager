from __future__ import annotations
from typing import TYPE_CHECKING, Union
import torch
from vllm.sequence import IntermediateTensors
from vllm.logger import init_logger
from .distributed_kv_connector import DistributedKVConnector as _Base

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

logger = init_logger(__name__)

class DistributedKVConnector(_Base):
    def __init__(self, rank: int, local_rank: int, config):
        super().__init__(rank, local_rank, config)
        try:
            kv_cfg = getattr(config, "kv_transfer_config", None)
            extra = getattr(kv_cfg, "kv_connector_extra_config", {}) if kv_cfg is not None else {}
            val = None
            if isinstance(extra, dict) and ("store_after_prefill" in extra):
                val = extra.get("store_after_prefill")
            elif hasattr(kv_cfg, "store_after_prefill"):
                val = getattr(kv_cfg, "store_after_prefill")
            if val is not None:
                self._store_after_prefill = bool(val)
        except Exception:
            pass
    def recv_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: list[torch.Tensor]
    ) -> tuple[Union[torch.Tensor, IntermediateTensors], bool, "ModelInputForGPUWithSamplingMetadata"]:
        stable_key_arg = None
        try:
            base_key = self._get_stable_key(model_input)
            stable_key_arg = f"{base_key}.pt"
            try:
                model_input.stable_key = stable_key_arg
            except Exception:
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
        retrieve_status = self.engine.should_retrieve(model_input, stable_key=stable_key_arg)
        hidden_or_intermediate_states, bypass_model_exec, model_input  = self.engine.retrieve_kv(
            model_executable, model_input, kv_caches, retrieve_status, stable_key=stable_key_arg
        )
        return hidden_or_intermediate_states, bypass_model_exec, model_input
