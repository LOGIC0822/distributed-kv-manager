#!/usr/bin/env bash
# Start vLLM v0 with DistributedKVConnector without using inject wrapper.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
JSON='{"kv_connector":"DistributedKVConnector","kv_connector_module_path":"distributed_kv_manager.vllm_adapter.distributed_kv_connector","kv_role":"kv_both","kv_connector_extra_config":{"config_path":"config_v0.json"}}'

# 默认开启引擎 debug dump，便于问题排查
export KV_DEBUG_DUMP=${KV_DEBUG_DUMP:-1}
export KV_DEBUG_DUMP_FILE=${KV_DEBUG_DUMP_FILE:-/tmp/kv_engine_debug.log}

USE_VLLM_V1=0 \
VLLM_USE_V1=0 \
KV_FORCE_CHUNKED_PREFILL=1 \
VLLM_LOG_LEVEL=DEBUG \
KV_DEBUG_DUMP=$KV_DEBUG_DUMP \
KV_DEBUG_DUMP_FILE=$KV_DEBUG_DUMP_FILE \
python -u -m vllm.entrypoints.openai.api_server \
  --model /tmp/ckpt/Qwen3-0.6B \
  --port 8200 \
  --max-model-len 512 \
  --gpu-memory-utilization 0.7 \
  --max-num-batched-tokens 16 \
  --max-num-seqs 4 \
  --enable-chunked-prefill \
  --kv-transfer-config "$JSON" \
  --disable-log-requests
