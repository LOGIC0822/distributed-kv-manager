# Distributed KV Manager

A distributed key-value cache manager for large language model (LLM) inference. It provides pluggable storage backends, composite (local + remote) layer splitting, multi‑tier data caching (memory + SSD), ETCD‑backed metadata with recovery, and optional prefetch + IO aggregation.

## Features

- **Distributed KV Caching**: Store / retrieve per‑sequence KV tensors for reuse
- **Pluggable Storage Backends**: Local filesystem, Crail, or custom (dynamic class import)
- **Composite Storage (NEW)**: Split KV along layer dimension: front part local, back part remote; transparent merge on download
- **Two‑Level Data Cache (NEW)**: In‑memory + optional SSD cache (`cache_mode`: `none` | `only_mem` | `mem_and_ssd`)
- **Prefetch + IO Aggregation (NEW)**: Windowed batching with bandwidth budgeting; prefetch data lands in memory first
- **Metadata Management**: ETCD with connection pooling, failover, buffered writes, and embedded file headers for recovery
- **Three‑Tier Metadata Cache**: Session → Layer → LRU hierarchy
- **Expiration + Cleanup**: Background thread removes expired entries (metadata + payload)
- **Metadata Recovery**: Reconstruct lost ETCD entries from file headers
- **Flexible Configuration Merge**: Runtime config overridden by `config.json` values
- **Enhanced Logging & Extensibility**: Clear instrumentation points, dynamic backend selection

## Architecture Overview

1. **KV Engine** – Decides store/retrieve, packs KV payloads, integrates metadata & prefetch
2. **Storage Layer** – `AbstractStorage` implementations (local / Crail / composite / custom)
3. **Composite Storage** – Layer‑wise split into `.front` + `.back` physical files
4. **Metadata Manager** – ETCD ops (pooling, buffering, failover)
5. **Metadata Cache** – 3 tiers (session, layer, LRU) to reduce ETCD hits
6. **Cleanup Manager** – Scans + expires KV entries
7. **Caching Wrapper** – Memory + SSD + prefetch & IO aggregation

### Key Components

#### KV Engine
The main engine that orchestrates KV cache storage and retrieval operations. It handles:
- Determining when to store or retrieve KV caches
- Managing asynchronous storage operations
- Coordinating with storage backends and metadata systems
- Checking metadata expiration during retrieval operations

#### Storage Backends
Supported / pluggable backends:
- **LocalStorage**: Filesystem reference implementation
- **CrailStorage**: High‑performance distributed storage (optional)
- **CompositeStorage**: Composes a local + remote backend with layer splitting
- **Custom**: Dynamic import via config (`remote_backend_class` or new `storage_type`)

#### Metadata Management
Robust metadata management using ETCD with:
- Connection pooling for multiple ETCD nodes
- Automatic failover and replication
- Structured metadata with fixed-size serialization
- Watch capabilities for metadata changes
- Metadata scanning for cleanup operations
- Metadata recovery from storage files

#### Metadata Caching
Three-tier caching system for metadata:
- **Pool 1**: Session-based metadata cache
- **Pool 2**: Layer-based metadata cache
- **Pool 3**: LRU cache for recent accesses

#### Cleanup Manager
Automatic cleanup of expired KV caches:
- Periodic scanning of metadata for expired entries
- Configurable cleanup interval
- Deletion of both metadata and storage files for expired entries
- Extension of metadata lifetime on access (touch mechanism)

git clone https://github.com/Yan-tx/distributed-kv-manager.git
## Installation

```bash
# Clone
git clone https://github.com/Yan-tx/distributed-kv-manager.git
cd distributed-kv-manager

# Editable install (ensure torch + etcd3 installed in environment)
pip install -e .

# Start ETCD (example)
nohup ./etcd \
  --data-dir /tmp/etcd-data \
  --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://127.0.0.1:2379 \
  > /tmp/etcd.log 2>&1 &
```

ETCD should listen on `127.0.0.1:2379` by default for tests.


The system can be configured in two ways:

### 1. Using Python Configuration Objects
The system can be configured through the `kv_transfer_config` object with the following options:

```python
    storage_type="crail",        # or "local"
    storage_dir="/kvcache",      # Base directory for storage
    etcd_endpoints=["127.0.0.1:2379"],  # ETCD endpoints
    # SSD缓存配置
    enable_ssd_caching=False,    # 是否启用SSD缓存
    ssd_cache_dir="/tmp/ssd_cache",  # SSD缓存目录
    enable_prefetch=True,        # 是否启用预取
    # KV缓存自动淘汰配置
    kv_expire_time=86400,        # KV缓存过期时间（秒），默认1天
    cleanup_interval=3600,       # 清理间隔时间（秒），默认1小时
    # 存储特定配置
    crail_dir="./crail_kvcache", # Crail存储目录
    local_dir="./local_kvcache"  # 本地存储目录
)
```

### 2. Using JSON Configuration File

Alternatively, you can create a `config.json` file in the project root directory:

```json
{
  "kv_transfer_config": {
    "storage_type": "crail",
    "storage_dir": "/kvcache",
    "etcd_endpoints": ["127.0.0.1:2379"],
    "enable_ssd_caching": false,
    "ssd_cache_dir": "/tmp/ssd_cache",
    "enable_prefetch": true,
    "kv_expire_time": 86400,
    "cleanup_interval": 3600,
    "crail_dir": "./crail_kvcache",
    "local_dir": "./local_kvcache"
  },
  "rank": 0,
  "local_rank": 0
}
```

Then initialize the engine without passing a config object:

```python
from distributed_kv_manager import init_engine

# Will automatically load from config.json
engine = init_engine()
```

### Storage Configuration (Basic)

Supported storage configuration fields under `kv_transfer_config`:

- `storage_type` (str)
  - `"local"`: use `LocalStorage(local_dir)` (filesystem backend).
  - `"crail"`: use `CrailStorage(crail_dir)` (distributed storage).
  - `"composite"`: use `CompositeStorage(local_backend, remote_backend, ...)` to split layers between local and remote.
  - `"v1_layered"`: v1-only layered layout; remote backend persists all layers, local backend caches only the first `layer_split_front` layers (per-layer safetensors). Use `layer_split_front = -1` to cache all layers locally.

- `storage_dir` (str)
  - Logical root directory for KV files, typically aligned with `local_dir`.

- `local_dir` (str)
  - Root directory for local storage (required when `storage_type="local"`).

- `crail_dir` (str)
  - Root directory for Crail storage; used when `storage_type="crail"` or as composite's remote default.

- `remote_dir` (str, optional)
  - Root directory for remote storage in composite mode; falls back to `crail_dir` if omitted.

- `remote_backend_type` / `remote_storage_type` (str, optional)
  - Remote backend type in composite mode:
    - `"noop"` / `"none"`: placeholder backend (no real persistence).
    - `"local"`: remote side is also `LocalStorage(remote_dir)`.
    - `"crail"`: remote side is `CrailStorage(remote_dir)`.
    - Any dotted path (e.g. `"my_pkg.my_mod.MyBackend"`): dynamically imported class.

- `remote_backend_class` (str, optional)
  - Explicit class path for remote backend, e.g. `"my_pkg.my_mod.MyStorage"`. Takes precedence over `remote_backend_type`.

- `layer_split_front` (int, optional)
  - In composite mode, number of front layers stored locally.

- `layer_split_ratio` (float, optional, default `0.5`)
  - In composite mode, ratio of front layers stored locally when `layer_split_front` is not set.

- `directory_layout` (str, optional, default `"flat"`)
  - `"hash"`, `"hash2"`, `"hash_bucket"`: enable hash-bucket directory mapping (`MappedStorage(HashBucketPathMapper)`).
  - Other values: keep a flat directory layout.

- `dkv_storage_path` (str, v1 only, optional)
  - Root directory for v1 external KV storage. If not set, v1 falls back to `storage_dir/local_dir` or `/tmp/kvcache/v1`.

- `enable_ssd_caching` (bool)
  - Legacy flag: when true and `cache_mode` is not set, wraps base storage with `CachingStorage(base, ssd_cache_dir, ...)`.

- `cache_mode` (str, optional)
  - `"none"`: no caching, return base storage as-is.
  - `"only_mem"`: in-memory LRU cache only, no SSD.
  - `"mem_and_ssd"`: two-level cache (memory + SSD).

- `ssd_cache_dir` (str, optional, default `"/tmp/ssd_cache"`)
  - Root directory for SSD cache files.

- `mem_cache_capacity_bytes` (int, optional, default `256 * 1024 * 1024`)
  - Capacity of the in-memory cache in bytes.

- `enable_prefetch` (bool, optional, default `true`)
  - Whether to enable prefetching logic inside `CachingStorage`.

- `etcd_endpoints` (list[str])
  - ETCD cluster endpoints used by v0 KVEngine for metadata.

- `kv_expire_time` (int, seconds)
  - TTL for KV entries; used by cleanup manager.

- `cleanup_interval` (int, seconds)
  - Interval for background cleanup scans.

- `store_after_prefill` (bool)
  - For v0: whether to store KV immediately after prefill.

- `v1_storage_mode` (str, optional)
  - Reserved for v1 layered modes (e.g. `"layered"`). Current v1 safetensors-based path does not use it yet.

- `enable_debug_dump` / `debug_dump_file`
  - Whether to enable debug dump for KV/metadata and where to write it.

Examples:

- Local only:
  ```json
  {"kv_transfer_config": {"storage_type": "local", "local_dir": "/tmp/kvcache"}}
  ```
- Crail only (legacy name):
  ```json
  {"kv_transfer_config": {"storage_type": "crail", "crail_dir": "/crail/kvcache"}}
  ```
- V1 layered (front 16 layers cached locally, all layers persisted remotely):
  ```json
  {"kv_transfer_config": {
    "storage_type": "v1_layered",
    "local_dir": "/data/kvcache_v1/local",
    "remote_dir": "/data/kvcache_v1/remote",
    "remote_backend_type": "local",
    "layer_split_front": 16
  }}
  ```

### V1 External KV Storage

The v1 engine stores KV as per-layer safetensors under a configurable root directory:

- Root directory:
  - `dkv_storage_path` controls the v1 external KV root.
  - If not set on `vllm_config.kv_transfer_config`, v1 falls back to `config.json.kv_transfer_config.dkv_storage_path`, then `storage_dir/local_dir`, and finally `/tmp/kvcache/v1`.
- File layout:
  - For each request, v1 hashes the prompt tokens (and optional multimodal hashes) into a folder name: `<prompt_hash> = md5(prompt_token_ids_bytes [+ mm_hashes])`.
  - Files are written as: `dkv_storage_path/<prompt_hash>/model.layers.<N>.self_attn.attn.safetensors` (one file per attention layer).
- Caching / composite support:
  - V1 delegates to `StorageFactory.create_storage`, so all existing options (`storage_type`, `composite`, `cache_mode`, `enable_ssd_caching`, `directory_layout`, etc.) still apply to the v1 path.

#### V1 Mode Switch + DRAM Hash Cache

- `use_v1` (bool): When `true`, v1-specific behavior is enabled across the stack.
  - The storage backend returned by the factory is wrapped with a v1-only DRAM cache: `V1HashCachedStorage`.
  - The LRU eviction unit is a full prompt hash directory (group of safetensors), not per-file.
  - This wrapper is independent of the legacy `CachingStorage` and remains active even when `cache_mode: "none"`.

- DRAM capacity setting (prefer GB):
  - `mem_cache_capacity_gb` (float/int, preferred) → converted to bytes.
  - `mem_cache_capacity_bytes` (int, fallback) → kept for backward compatibility.

- Etcd-only hit/miss semantics:
  - Existence check uses etcd metadata only. If etcd is down/unavailable, v1 treats it as a miss and does not scan the filesystem.
  - After a STORE, a single aggregate metadata entry is written per hash directory (`status=1`, refreshed `last_access`).

- Recommended minimal config for v1 hash-group DRAM caching:
  ```json
  {
    "kv_transfer_config": {
      "use_v1": true,
      "storage_type": "v1_layered",
      "local_dir": "/kvcache_v1/local",
      "remote_dir": "/kvcache_v1/remote",
      "remote_backend_type": "local",
      "layer_split_front": -1,
      "dkv_storage_path": "/kvcache/v1",
      "cache_mode": "none",
      "mem_cache_capacity_gb": 0.25,
      "etcd_endpoints": ["127.0.0.1:2379"],
      "kv_expire_time": 86400
    }
  }
  ```

#### V1 Prefetch

- `enable_prefetch` (bool): enables prefetch at engine init for v1.
- `v1_prefetch_top_k` (int): prefetches top-K hashes by `last_access` (from etcd) into DRAM.
- Expected logs on startup (when enabled):
  - `[v1_engine] scheduled initial v1 prefetch for top_k=K hashes`
  - `[v1_engine] prefetch: top_k=K, actual=X`

Testing quick recipe:
- Clear previous state: `python3 scripts/clear_kv_store.py --yes`
- Run a long request once to populate files + metadata.
- Restart server with `use_v1: true`, `cache_mode: "none"`, and a small `mem_cache_capacity_gb` if you want to observe eviction.
- Watch logs for prefetch lines above; DRAM hits then come from `V1HashCachedStorage`.

### Composite Storage Configuration

Split KV layers so early layers (front) stay local, remaining (back) remote.

Key options:
- `storage_type": "composite"`
- `local_dir`: Local base path
- `remote_dir`: Remote base path (preferred); falls back to `crail_dir` if absent
- `layer_split_front`: Explicit front layer count (overrides ratio)
- `layer_split_ratio`: Fraction (0~1) if `layer_split_front` not set (default 0.5)
- Remote backend selection:
  - `remote_backend_type`: `crail` | `local` | `noop` (placeholder) | custom class path
  - `remote_backend_class`: Fully‑qualified `module.ClassName` (overrides type)

Examples:

1) Ratio split (half local):
```json
{"kv_transfer_config": {
  "storage_type": "composite",
  "local_dir": "/data/kvcache_local",
  "remote_dir": "/data/kvcache_remote",
  "layer_split_ratio": 0.5
}}
```

2) Explicit front count:
```json
{"kv_transfer_config": {
  "storage_type": "composite",
  "local_dir": "/data/kvcache_local",
  "remote_dir": "/data/kvcache_remote",
  "layer_split_front": 12
}}
```

3) Placeholder remote (development):
```json
{"kv_transfer_config": {
  "storage_type": "composite",
  "local_dir": "/data/kvcache_local",
  "remote_dir": "/data/unused_remote",
  "remote_backend_type": "noop"
}}
```

4) Dynamic custom class:
```json
{"kv_transfer_config": {
  "storage_type": "composite",
  "local_dir": "/data/kvcache_local",
  "remote_dir": "/data/remote",
  "remote_backend_class": "my_pkg.backends.MyRemoteStorage"
}}
```

If the remote backend is a noop and `layer_split_front` is not set, the factory forces a large front count so all layers stay local.

### Data Cache Configuration

Multi‑tier caching wrapper around any base storage:
- `cache_mode`: `none` | `only_mem` | `mem_and_ssd`
- `mem_cache_capacity_bytes`: In‑memory LRU capacity
- `enable_ssd_caching`: Backward‑compat flag (used when `cache_mode` omitted)
- `ssd_cache_dir`: Path for SSD tier when enabled
- Prefetch: `enable_prefetch`: True/False (prefetch lands into memory layer)

Example (memory + SSD + prefetch):
```json
{"kv_transfer_config": {
  "storage_type": "local",
  "local_dir": "/data/kvcache",
  "cache_mode": "mem_and_ssd",
  "mem_cache_capacity_bytes": 536870912,
  "ssd_cache_dir": "/data/ssd_cache",
  "enable_prefetch": true
}}
```

## Usage

```python
from distributed_kv_manager import init_engine, destroy_engine
from distributed_kv_manager import should_store, store_kv, should_retrieve, retrieve_kv

# Initialize the engine
engine = init_engine(config)

# Prepare optional hidden/intermediate states captured from the model output.
# This is required when you expect full bypass on a cache hit.
hidden_states = hidden_or_intermediate_states_from_model_output(...)

# Store KV cache if needed
if store_status == StoreStatus.STORED:
    store_kv(
        model_config,
        parallel_config,
        transfer_config,
        model_executable,
        model_input,
        kv_caches,
        store_status,
        hidden_or_intermediate_states=hidden_states,
    )

# Check if we can retrieve KV cache
retrieve_status = should_retrieve(model_input)

# Retrieve KV cache if available
if retrieve_status == RetrieveStatus.HIT:
    hidden_state, bypass_model, new_input = retrieve_kv(
        model_executable, model_input, kv_caches, retrieve_status)

# Clean up
destroy_engine()
```

### Automatic Cleanup

The KV manager automatically cleans up expired KV caches based on the configured expiration time. The cleanup process runs periodically in a background thread. You can configure the expiration time and cleanup interval in the configuration:

```python
config.kv_transfer_config = SimpleNamespace(
    # ... other configuration options
    kv_expire_time=86400,        # KV缓存过期时间（秒），默认1天
    cleanup_interval=3600        # 清理间隔时间（秒），默认1小时
)
```

Expired KV caches are automatically removed from both storage and metadata, freeing up resources without manual intervention.

### Testing

Run the test suite:
```bash
python -m pytest
```
Focused:
```bash
python -m pytest tests/test_kv_engine.py -q
```

## API Reference

### Enums

- `StoreStatus`: Status for storage operations (STORED, SKIPPED)
- `RetrieveStatus`: Status for retrieval operations (HIT, MISS)

### Functions

- `init_engine(config)`: Initialize the KV engine with the given configuration
- `destroy_engine()`: Clean up and destroy the engine instance
- `should_store(model_input)`: Determine if KV cache should be stored
- `store_kv(...)`: Store KV cache with the given parameters
- `should_retrieve(model_input)`: Determine if KV cache can be retrieved
- `retrieve_kv(...)`: Retrieve KV cache if available

## Development

### Project Structure

```
distributed_kv_manager/
├── engine/              # Core KV engine implementation
├── metadata/            # Metadata management and caching
├── storage/             # Storage backend implementations
├── __init__.py          # Package initialization
└── tests/               # Test files
```

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test files
python test_kv_engine.py

# Run cleanup tests
python test_kv_cleanup.py
```

The cleanup tests verify the automatic expiration and cleanup functionality. These tests use a shorter expiration time and cleanup interval to quickly validate the cleanup mechanism.

## vLLM Integration (Quick Start)

- Start a local ETCD first (if not already):

```bash
cd ~/etcd
nohup ./etcd \
  --data-dir /tmp/etcd-data \
  --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://127.0.0.1:2379 \
  > /tmp/etcd.log 2>&1 &
```

- Then start the vLLM OpenAI-compatible server with this connector (v0 API):

```bash
python3 vllm_adapter/vllm_start_with_inject.py \
  --model /tmp/ckpt/Qwen --port 8100 --max-model-len 10000 \
  --gpu-memory-utilization 0.8 \
  --kv-transfer-config '{"kv_connector":"DistributedKVConnector","kv_role":"kv_both"}'

python3 vllm_adapter/vllm_start_with_inject.py \
  --model /tmp/ckpt/Qwen3-0.6B --port 8100 --max-model-len 10000 \
  --gpu-memory-utilization 0.8 \
  --kv-transfer-config '{"kv_connector":"DistributedKVConnector","kv_role":"kv_both"}'

- Or start directly with the v1 API (no inject script needed):

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model /tmp/ckpt/Qwen3-0.6B --port 8100 --max-model-len 10000 \
  --gpu-memory-utilization 0.8 \
  --kv-transfer-config '{"kv_connector":"DKVOffloadingConnector","kv_connector_module_path":"distributed_kv_manager.vllm_adapter.dkv_offloading_connector_v1","kv_role":"kv_both"}'
```

Note: `kv_connector_module_path` ensures vLLM imports the connector from
`distributed_kv_manager.vllm_adapter.dkv_offloading_connector_v1` even if a
same-name connector exists in vLLM. Keep it when running outside editable dev.
```

- Simple requests to test:

```bash
curl http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/tmp/ckpt/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "写一首关于春天的诗。"}],
    "stream": true
  }'

curl http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/tmp/ckpt/Qwen3-0.6B",
    "messages": [
      {"role": "system", "content": "You are a helpful AI assistant."},
      {"role": "user", "content": "请用中文详细解释GAN的原理。"}
    ],
    "max_tokens": 200,
    "temperature": 0.7
  }'

curl http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/tmp/ckpt/Qwen3-0.6B",
    "messages": [
      {"role": "system", "content": "You are a helpful AI assistant."},
      {"role": "user", "content": "请帮我写一首冬天的诗歌。"}
    ],
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

### Important notes (path and model)

- Do not launch the `api_server` from `~`, `~/vllm`, or any directory whose
  name conflicts with installed packages (e.g., a folder named `vllm`). Such
  paths can shadow the real Python package and cause import/startup failures.
- Use an independent working directory instead (e.g., `/workspace`).
- Model path consistency: The `--model` path you pass when launching the
  server must exactly match the `"model"` field in your subsequent `curl`
  requests (e.g., both `/tmp/ckpt/Qwen3-0.6B`). A mismatch may cause
  load/routing failures.

## Contact

For questions or support, please contact the maintainers.

## Implementing a Custom Storage Backend

If you want to plug in your own storage (S3, OSS, NFS, etc.), implement the `AbstractStorage` interface and register it in the factory.

- Contract (see `distributed_kv_manager/storage/base.py`):
  - `upload(file_path: str, data: bytes) -> bool`
  - `download(file_path: str) -> Optional[bytes]`
  - `exists(file_path: str) -> bool`
  - `pack_kv_data(k_cache, v_cache, hidden, input_tokens, roi) -> bytes`
  - `unpack_kv_data(data: bytes) -> (k_cache, v_cache, hidden)`
  - Optional helpers: `delete`, `list_files`, `extract_metadata_from_data`

- Packing/unpacking guidance
  - Follow `LocalStorage` as a reference. It packs CPU tensors via `torch.save` and returns CPU tensors from `unpack_kv_data`; the engine moves them to the right device later.
  - The engine may prepend embedded metadata bytes; it strips them before calling your `unpack_kv_data`, so you typically do not need to parse metadata.

- Minimal example

```python
# distributed_kv_manager/storage/my_storage.py
import io, os, torch, logging
from typing import Optional, Tuple
from .base import AbstractStorage

logger = logging.getLogger("MyStorage")

class MyStorage(AbstractStorage):
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)

    def upload(self, file_path: str, data: bytes) -> bool:
        try:
            full = os.path.join(self.root_dir, file_path)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "wb") as f:
                f.write(data)
            return True
        except Exception as e:
            logger.error("upload failed %s: %s", file_path, e)
            return False

    def download(self, file_path: str) -> Optional[bytes]:
        full = os.path.join(self.root_dir, file_path)
        if not os.path.exists(full):
            return None
        with open(full, "rb") as f:
            return f.read()

    def exists(self, file_path: str) -> bool:
        return os.path.exists(os.path.join(self.root_dir, file_path))

    def pack_kv_data(self, k_cache, v_cache, hidden, input_tokens, roi) -> bytes:
        buf = io.BytesIO()
        torch.save({
            "k_cache": k_cache.cpu(),
            "v_cache": v_cache.cpu(),
            "hidden": None if hidden is None else hidden.cpu(),
            "input_tokens": input_tokens.cpu(),
            "roi": roi.cpu(),
        }, buf)
        return buf.getvalue()

    def unpack_kv_data(self, data: bytes):
        obj = torch.load(io.BytesIO(data), map_location="cpu")
        return obj["k_cache"], obj["v_cache"], obj.get("hidden", None)
```

- Register in the factory (`distributed_kv_manager/storage/factory.py`):

```python
from .my_storage import MyStorage  # add import

class StorageFactory:
    @staticmethod
    def create_storage(config):
        storage_type = getattr(config.kv_transfer_config, "storage_type", "local")
        # ...
        if storage_type == "my_storage":
            root = getattr(config.kv_transfer_config, "my_dir", "/tmp/kvcache_my")
            base_storage = MyStorage(root)
        # wrap with SSD cache if enable_ssd_caching=True
```

- Configure it (JSON or Python):

```json
{
  "kv_transfer_config": {
    "storage_type": "my_storage",
    "my_dir": "/data/kvcache",
    "enable_ssd_caching": false
  }
}
```

Tips
- Ensure `file_path` is relative (factory/backends handle base paths)
- Keep `pack/unpack` stable; shapes + dtypes must round‑trip
- When wrapping with caching, identical keys must work across tiers
- For composite, avoid mutating tensor layer order after split planning

## Prefetch module layout

The prefetch and IO aggregation logic now lives in a dedicated package:

- Package: `distributed_kv_manager.prefetch`
- Implementation: `distributed_kv_manager/prefetch/core.py`
- Public API exported by the package `__init__.py`

Import examples:

```python
from distributed_kv_manager.prefetch import (
  PrefetchBuffer, BudgetEstimator, RateLimiter, IOAggregator, PlanBuilder,
)
```

Note: the legacy top‑level module `distributed_kv_manager/prefetch.py` was removed.

## Notes & Pitfalls

- Always call `destroy_engine()` to stop cleanup + flush async tasks
- Provide `session_id` (bytes) and `layer_id` (int) in `model_input` for stable keys
- Remote path: prefer `remote_dir`; `crail_dir` only kept for backward compatibility
- Prefetch is asynchronous; wait for futures or engine destroy before asserting hits in tests
- If ETCD is unreachable, retrieval falls back to MISS and may log warnings

## Roadmap (Future Enhancements)

- Adaptive layer split based on access latency & hit rates
- Dynamic prefetch budgeting (utilization feedback loop)
- Extended telemetry export (prometheus hooks)
- Additional remote backends (AS13000, Crail) via dynamic import
