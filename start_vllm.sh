#!/bin/bash
set -euo pipefail

# Choose connector by vLLM version (v1 vs v0), similar to LMCache example.
# v0  -> DistributedKVConnector (legacy), with chunked prefill enabled.
# v1  -> DKVEngineConnectorV1 (new offloading connector).
if [[ "${USE_VLLM_V1:-0}" == "1" || "${VLLM_USE_V1:-0}" == "1" ]]; then
  KV_JSON='{"kv_connector":"DKVEngineConnectorV1","kv_connector_module_path":"distributed_kv_manager.vllm_adapter.dkv_offloading_connector_v1","kv_role":"kv_both"}'
  export USE_VLLM_V1=1
  export VLLM_USE_V1=1
else
  KV_JSON='{"kv_connector":"DistributedKVConnector","kv_connector_module_path":"distributed_kv_manager.vllm_adapter.distributed_kv_connector","kv_role":"kv_both"}'
  export USE_VLLM_V1=0
  export VLLM_USE_V1=0
  # v0 路径需要 chunked prefill 支持才能按块落盘/复用
  export KV_FORCE_CHUNKED_PREFILL=${KV_FORCE_CHUNKED_PREFILL:-1}
fi

export USE_VLLM_V1=${USE_VLLM_V1:-0}
export VLLM_USE_V1=${VLLM_USE_V1:-0}
export KV_FORCE_CHUNKED_PREFILL=${KV_FORCE_CHUNKED_PREFILL:-1}
export KV_MAX_NUM_BATCHED_TOKENS=${KV_MAX_NUM_BATCHED_TOKENS:-16}
export KV_MAX_NUM_SEQS=${KV_MAX_NUM_SEQS:-4}

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export KV_DEBUG_DUMP=${KV_DEBUG_DUMP:-1}
export KV_DEBUG_DUMP_FILE=${KV_DEBUG_DUMP_FILE:-/tmp/kv_engine_debug.log}
export VLLM_NO_USAGE_STATS=1
export VLLM_LOG_LEVEL=${VLLM_LOG_LEVEL:-DEBUG}

if [[ "${USE_VLLM_V1}" == "1" ]]; then
  export PORT=${PORT:-8100}
  python -u -m vllm.entrypoints.openai.api_server \
    --model /tmp/ckpt/Qwen3-0.6B \
    --port "${PORT}" \
    --max-model-len 512 \
    --gpu-memory-utilization 0.7 \
    --max-num-batched-tokens "${KV_MAX_NUM_BATCHED_TOKENS}" \
    --max-num-seqs "${KV_MAX_NUM_SEQS}" \
    --kv-transfer-config "${KV_JSON}" \
    --disable-log-requests
else
  export PORT=${PORT:-8200}
  exec bash scripts/start_v0_native.sh
fi
