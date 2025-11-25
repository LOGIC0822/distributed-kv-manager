# vLLM Adapter for Distributed KV Manager

This directory contains the adapter files needed to integrate the Distributed KV Manager with vLLM. To use this adapter, place these files under the kv_connector submodule in your vLLM installation.

## Installation

To integrate the Distributed KV Manager with your vLLM installation, follow these steps:

1. Locate your vLLM installation directory.
2. Navigate to `vllm/distributed/kv_transfer/kv_connector/` within your vLLM installation.
3. Copy the files from this directory into the `kv_connector` directory:
   - `distributed_kv_connector.py`
   - `factory.py`

Note: This replaces the `factory.py` under `kv_connector/`. Back up the original file if needed.

## File Placement

The files should be placed at the following path inside your vLLM installation:

## Startup Modes with vLLM

You can run with either the legacy inject style (v0) or the direct v1 API.

- v0 (inject):
  ```bash
  python3 vllm_adapter/vllm_start_with_inject.py \
    --model /tmp/ckpt --port 8100 --max-model-len 10000 \
    --kv-transfer-config '{"kv_connector":"DistributedKVConnector","kv_role":"kv_both"}'
  ```

- v1 (direct):
  ```bash
  python3 -m vllm.entrypoints.openai.api_server \
    --model /tmp/ckpt/Qwen3-0.6B --port 8100 --max-model-len 10000 \
    --gpu-memory-utilization 0.8 \
    --kv-transfer-config '{"kv_connector":"DKVOffloadingConnector","kv_connector_module_path":"distributed_kv_manager.vllm_adapter.dkv_offloading_connector_v1","kv_role":"kv_both"}'
  ```

```
vllm/
└── vllm/
    └── distributed/
        └── kv_transfer/
            └── kv_connector/
                ├── distributed_kv_connector.py
                └── factory.py
```

## Configuration

After placing the files, configure vLLM to use the Distributed KV Manager by specifying the KV connector in your config.

### Registering the Connector

The provided `factory.py` registers `DistributedKVConnector` with the KV connector factory. Ensure your vLLM configuration includes:

```python
# In your vLLM configuration
kv_transfer_config = {
    "kv_connector": "DistributedKVConnector",
    # Add other configuration options as needed
}
```

### Dependencies

Ensure the `distributed_kv_manager` package is available in your Python environment:

```bash
pip install -e /path/to/distributed_kv_manager
```

## Usage

Once installed and configured, Distributed KV Manager will handle KV cache storage and retrieval during vLLM inference.

The adapter provides two main functionalities:
1. `recv_kv_caches_and_hidden_states`: Retrieves KV caches and hidden states from distributed storage.
2. `send_kv_caches_and_hidden_states`: Stores KV caches and hidden states to distributed storage.

## Troubleshooting

If you encounter issues:

1. Ensure files are copied to `kv_transfer/kv_connector/`.
2. Verify `distributed_kv_manager` is installed and importable.
3. Check vLLM version compatibility.
4. Confirm your configuration specifies `DistributedKVConnector`.

## Updating

When updating Distributed KV Manager, you may also need to update these adapter files. Back up any original files before updating.

