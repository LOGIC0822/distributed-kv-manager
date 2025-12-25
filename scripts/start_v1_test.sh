#!/bin/bash
# Helper script to start vLLM v1 with our DKV connector and write logs to /tmp/v1.log.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export USE_VLLM_V1=1
export VLLM_USE_V1=1
export KV_DEBUG_DUMP=1
export KV_DEBUG_DUMP_FILE=${KV_DEBUG_DUMP_FILE:-/tmp/kv_engine_debug.log}

KV_JSON='{"kv_connector":"DKVEngineConnectorV1","kv_connector_module_path":"distributed_kv_manager.vllm_adapter.dkv_offloading_connector_v1","kv_role":"kv_both"}'

nohup python -m vllm.entrypoints.api_server \
  --model /tmp/ckpt/Qwen3-0.6B \
  --port ${PORT:-8200} \
  --no-enable-prefix-caching \
  --max-model-len 2048 \
  --gpu-memory-utilization ${GPU_MEM_UTIL:-0.6} \
  --kv-transfer-config "$KV_JSON" \
  > /tmp/v1.log 2>&1 &

echo $! > /tmp/v1.pid
echo "Started v1 server pid=$(cat /tmp/v1.pid), log=/tmp/v1.log"
