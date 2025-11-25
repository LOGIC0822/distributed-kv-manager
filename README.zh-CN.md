# 分布式KV管理器

一个面向大语言模型 (LLM) 推理场景的分布式键值 (KV) 缓存管理器：支持可插拔存储后端、复合分层存储（本地 + 远端）、内存 + SSD 两级数据缓存、ETCD 元数据与恢复、以及可选的预取 + I/O 聚合。

## 功能特性

- **分布式KV缓存**：为LLM推理高效存储和检索KV缓存
- **多种存储后端**：支持本地、Crail（可选）、以及自定义动态导入后端
- **元数据管理**：使用ETCD进行强大的元数据管理，支持连接池和故障转移
- **缓存层**：三层元数据缓存，提升性能
- **异步操作**：基于线程池的非阻塞存储操作
- **容错性**：ETCD元数据的自动故障转移和复制
- **灵活配置**：轻松配置存储后端和缓存策略
- **自动清理**：基于可配置过期时间自动清理过期的KV缓存
- **元数据恢复**：能够从存储文件中恢复元数据
- **增强日志**：详细的日志记录便于调试和监控
 - **双层数据缓存（新增）**：支持内存缓存与SSD缓存，可选三种缓存模式：`none`、`only_mem`、`mem_and_ssd`
 - **预取 + I/O 聚合（新增）**：窗口聚合 + 带宽预算；按层顺序的会话前视；预取优先落入内存
 - **复合存储（新增）**：KV 按层拆分：前段本地、后段远端；下载自动合并

## 架构

系统由以下组件构成：

1. **KV引擎**：处理存储和检索操作的核心引擎
2. **存储层**：抽象存储接口（本地 / Crail / 复合 / 自定义）
3. **元数据管理器**：基于ETCD的元数据管理，支持连接池
4. **元数据缓存**：三层缓存系统，提升元数据访问性能
5. **存储工厂**：工厂模式，用于创建适当的存储后端
6. **清理管理器**：自动清理过期的KV缓存

### 核心组件

#### KV引擎
协调KV缓存存储和检索操作的主引擎。它处理：
- 确定何时存储或检索KV缓存
- 管理异步存储操作
- 协调存储后端和元数据系统
- 在检索操作期间检查元数据过期

#### 存储后端
支持或可扩展的后端：
- **LocalStorage**：文件系统参考实现
- **CrailStorage**：高性能分布式存储（可选）
- **CompositeStorage**：分层拆分 + 合并
- **自定义后端**：通过动态类路径导入

#### 元数据管理
使用ETCD进行强大的元数据管理，具备：
- 多个ETCD节点的连接池
- 自动故障转移和复制
- 固定大小序列化的结构化元数据
- 元数据变更的监听能力
- 用于清理操作的元数据扫描
- 从存储文件中恢复元数据

#### 元数据缓存
三层元数据缓存系统：
- **Pool 1**：基于会话的元数据缓存
- **Pool 2**：基于层的元数据缓存
- **Pool 3**：最近访问的LRU缓存

#### 清理管理器
自动清理过期的KV缓存：
- 定期扫描元数据中的过期条目
- 可配置的清理间隔
- 删除过期条目的元数据和存储文件
- 访问时延长元数据生命周期（touch机制）

## 安装

```bash
# 克隆仓库
git clone https://github.com/Yan-tx/distributed-kv-manager.git
cd distributed_kv_manager

# 安装包
pip install -e .
```

### 本地ETCD设置用于开发

为了本地开发和测试，您可以使用以下命令运行一个最小化的ETCD实例：

```bash
# 导航到您的etcd目录
cd ~/etcd

# 在后台启动ETCD
nohup ./etcd \
  --data-dir /tmp/etcd-data \
  --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://127.0.0.1:2379 \
  > /tmp/etcd.log 2>&1 &
```

这将启动一个监听2379端口的ETCD服务器，这是KV管理器默认期望的端口。

## 配置

系统可以通过两种方式进行配置：

### 1. 使用Python配置对象

系统可以通过`kv_transfer_config`对象进行配置，包含以下选项：

```python
config.kv_transfer_config = SimpleNamespace(
    storage_type="crail",        # 或 "local"
    storage_dir="/kvcache",      # 存储的基础目录
    etcd_endpoints=["127.0.0.1:2379"],  # ETCD端点
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

### 2. 使用JSON配置文件

或者，您可以在项目根目录创建一个`config.json`文件：

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

然后在不传递配置对象的情况下初始化引擎：

```python
from distributed_kv_manager import init_engine

# 将自动从config.json加载配置
engine = init_engine()
```

### 存储与缓存配置（基础）

存储类型：
- `local`：本地存储
- `crail`：Crail 分布式存储
- `composite`：分层存储，本地 + 远端
- `v1_layered`：v1 专用的按层分级缓存模式，远端保存所有层，本地只缓存前 `layer_split_front` 层（基于 per-layer safetensors 文件）
分层拆分（`composite`）：
- `layer_split_front`：前段层数（优先）
- `layer_split_ratio`：比例（未设置前段层数时生效，默认 0.5）

数据缓存：
- `cache_mode`: `none` | `only_mem` | `mem_and_ssd`
- `mem_cache_capacity_bytes`: 内存 LRU 容量（默认 256MB）
- `enable_prefetch`: 预取 + 聚合（默认 true）
- `ssd_cache_dir`: SSD 缓存目录（在 `mem_and_ssd` 或兼容旧模式下使用）
- 兼容：未设置 `cache_mode` 时沿用旧字段 `enable_ssd_caching`

远端后端抽象（在 `composite` 中）：
- `remote_dir`：远端基路径（优先）；缺省回退 `crail_dir`
- `remote_backend_type`：`crail` | `local` | `noop`/`none`/`placeholder` | 自定义类路径
- `remote_backend_class`：完全限定类路径（优先级高于 `remote_backend_type`）
占位远端（noop）场景下若未指定 `layer_split_front`，工厂会强制全部层落本地避免读取空后段。

示例：

1）仅内存缓存 + 预取
```json
{
  "kv_transfer_config": {
    "storage_type": "local",
    "local_dir": "/kvcache",
    "cache_mode": "only_mem",
    "mem_cache_capacity_bytes": 536870912,
    "enable_prefetch": true
  }
}
```

2）内存 + SSD 两级缓存 + 预取
```json
{
  "kv_transfer_config": {
    "storage_type": "local",
    "local_dir": "/kvcache",
    "cache_mode": "mem_and_ssd",
    "ssd_cache_dir": "/tmp/ssd_cache",
    "mem_cache_capacity_bytes": 1073741824,
    "enable_prefetch": true
  }
}
```

6）仅 v1 分层缓存 + 远端持久化示例
```json
{
  "kv_transfer_config": {
    "storage_type": "v1_layered",
    "local_dir": "/kvcache_v1/local",
    "remote_dir": "/kvcache_v1/remote",
    "remote_backend_type": "local",
    "layer_split_front": 16
  }
}
```

3）复合存储（按层拆分） + 两级缓存（使用旧字段 crail_dir，兼容）
4）复合存储 + 占位远端（开发调试）
```json
{
  "kv_transfer_config": {
    "storage_type": "composite",
    "local_dir": "/kvcache_local",
    "remote_dir": "/unused/remote",
    "remote_backend_type": "noop",
    "layer_split_ratio": 0.5
  }
}
```

5）复合存储 + 自定义远端类
```json
{
  "kv_transfer_config": {
    "storage_type": "composite",
    "local_dir": "/kvcache_local",
    "remote_dir": "/remote/kvcache",
    "remote_backend_class": "my_pkg.backends.MyRemoteStorage",
    "layer_split_front": 16
  }
}
```

```json
{
  "kv_transfer_config": {
    "storage_type": "composite",
    "local_dir": "/kvcache_local",
    "crail_dir": "/crail/kvcache",
    "layer_split_front": 20,
    "cache_mode": "mem_and_ssd",
    "ssd_cache_dir": "/tmp/ssd_cache",
    "mem_cache_capacity_bytes": 1073741824,
    "enable_prefetch": true
  }
}
```

说明：
- 未指定 `layer_split_front` 时使用 `layer_split_ratio`（默认 0.5）
- 预取结果在 `only_mem` 与 `mem_and_ssd` 模式下都会优先落入内存缓存
- 推荐新配置使用 `remote_dir`，旧的 `crail_dir` 保留兼容

### v1 外部 KV（模式开关 + DRAM Hash 组缓存）

- `use_v1`（bool）：打开后，激活 v1 专属路径：
  - 工厂返回的后端会被 v1 专用的 DRAM 封装 `V1HashCachedStorage` 包裹；
  - LRU 淘汰的最小单位是“一个 hash 目录”（即该 hash 下的所有 per-layer `.safetensors` 文件一起被驱逐），而不是单个文件；
  - 该封装与旧的通用 `CachingStorage` 无关，即使 `cache_mode: "none"` 也会生效。

- DRAM 容量配置（推荐使用 GB）：
  - `mem_cache_capacity_gb`（float/int，推荐）→ 内部换算为字节；
  - `mem_cache_capacity_bytes`（int，兼容旧字段）。

- 命中/存在性判断（仅 etcd）：
  - v1 只通过 etcd 元数据判断命中；etcd 不可达则直接视为 MISS，不会去遍历/探测本地或远端文件系统以避免放大延迟；
  - 完成一次 STORE 后，会按“hash 目录”聚合写入一条元数据记录（`status=1`），并刷新 `last_access`。

- v1 最小示例（推荐配置）：
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

#### v1 预取（Prefetch）

- `enable_prefetch`（bool）：启用后，在引擎初始化时进行一次预取；
- `v1_prefetch_top_k`（int）：从 etcd 读取未过期条目，按 `last_access` 降序取前 K 个 hash 目录，将其 per-layer 文件拉入 DRAM 缓存；
- 启动日志期望：
  - `[v1_engine] scheduled initial v1 prefetch for top_k=K hashes`
  - `[v1_engine] prefetch: top_k=K, actual=X`

测试建议：
- 使用脚本清理：`python3 scripts/clear_kv_store.py --yes`；
- 首次运行发送一次较长请求以产生文件与元数据；
- 重启服务、确保 `use_v1: true` 与 `cache_mode: "none"`；将 `mem_cache_capacity_gb` 调小便于观察按“hash 目录”整组淘汰；
- 留意上述预取日志；后续 DRAM 命中由 `V1HashCachedStorage` 提供。

## 使用方法

### 基本使用

```python
from distributed_kv_manager import init_engine, destroy_engine
from distributed_kv_manager import should_store, store_kv, should_retrieve, retrieve_kv

# 初始化引擎
engine = init_engine(config)

# 从模型输出中获取可选的隐藏态 / 中间结果，用于完全命中时的 bypass。
hidden_states = hidden_or_intermediate_states_from_model_output(...)

# 检查是否应该存储KV缓存
store_status = should_store(model_input)

# 如需要则存储KV缓存
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

# 检查是否可以检索KV缓存
retrieve_status = should_retrieve(model_input)

# 如可用则检索KV缓存
if retrieve_status == RetrieveStatus.HIT:
    hidden_state, bypass_model, new_input = retrieve_kv(
        model_executable, model_input, kv_caches, retrieve_status)

# 清理
destroy_engine()
```

### 自动清理

KV管理器会根据配置的过期时间自动清理过期的KV缓存。清理过程在后台线程中定期运行。您可以在配置中设置过期时间和清理间隔：

```python
config.kv_transfer_config = SimpleNamespace(
    # ... 其他配置选项
    kv_expire_time=86400,        # KV缓存过期时间（秒），默认1天
    cleanup_interval=3600        # 清理间隔时间（秒），默认1小时
)
```

过期的KV缓存会自动从存储和元数据中删除，无需手动干预即可释放资源。

### 测试

运行测试套件：
```bash
python -m pytest
```
聚焦单文件：
```bash
python -m pytest tests/test_kv_engine.py -q
```

附加说明（新增能力）：
- 预取与 I/O 聚合在缓存包装启用后后台运行；不修改 ETCD 元数据
- 复合存储对引擎透明：ETCD 仍记录逻辑文件键，物理拆分在内部完成

## API参考

### 枚举

- `StoreStatus`：存储操作状态（STORED, SKIPPED）
- `RetrieveStatus`：检索操作状态（HIT, MISS）

### 函数

- `init_engine(config)`：使用给定配置初始化KV引擎
- `destroy_engine()`：清理并销毁引擎实例
- `should_store(model_input)`：确定是否应存储KV缓存
- `store_kv(...)`：使用给定参数存储KV缓存
- `should_retrieve(model_input)`：确定是否可检索KV缓存
- `retrieve_kv(...)`：如可用则检索KV缓存

## 开发

### 项目结构

```
distributed_kv_manager/
├── engine/              # 核心KV引擎实现
├── metadata/            # 元数据管理和缓存
├── storage/             # 存储后端实现
├── __init__.py          # 包初始化
└── tests/               # 测试文件
```

### 运行测试

```bash
# 运行所有测试
python -m pytest

# 运行特定测试文件
python test_kv_engine.py

# 运行清理测试
python test_kv_cleanup.py
```

清理测试验证自动过期和清理功能。这些测试使用较短的过期时间和清理间隔来快速验证清理机制。

## vLLM 快速启动与测试

- 先启动本地 ETCD（如未启动）：

```bash
cd ~/etcd
nohup ./etcd \
  --data-dir /tmp/etcd-data \
  --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://127.0.0.1:2379 \
  > /tmp/etcd.log 2>&1 &
```

- 然后使用本连接器启动 vLLM 的 OpenAI 兼容服务（v0 接口）：

```bash
python3 vllm_adapter/vllm_start_with_inject.py \
  --model /tmp/ckpt/Qwen --port 8100 --max-model-len 10000 \
  --gpu-memory-utilization 0.8 \
  --kv-transfer-config '{"kv_connector":"DistributedKVConnector","kv_role":"kv_both"}'

python3 vllm_adapter/vllm_start_with_inject.py \
  --model /tmp/ckpt/Qwen3-0.6B --port 8100 --max-model-len 10000 \
  --gpu-memory-utilization 0.8 \
  --kv-transfer-config '{"kv_connector":"DistributedKVConnector","kv_role":"kv_both"}'
```

- 或者直接使用 v1 接口启动（无需注入脚本）：

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model /tmp/ckpt/Qwen3-0.6B --port 8100 --max-model-len 10000 \
  --gpu-memory-utilization 0.8 \
  --kv-transfer-config '{"kv_connector":"DKVOffloadingConnector","kv_connector_module_path":"distributed_kv_manager.vllm_adapter.dkv_offloading_connector_v1","kv_role":"kv_both"}'
```

说明：`kv_connector_module_path` 用于强制 vLLM 从
`distributed_kv_manager.vllm_adapter.dkv_offloading_connector_v1` 导入连接器，
即使 vLLM 内部已有同名连接器也不会冲突。若非编辑模式或存在多版本混用，建议保留。

- 基础请求测试：

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

### 重要提醒（路径与模型）

- 启动 `api_server` 时，请不要在 `~` 或 `~/vllm` 等与源码包同名/冲突的目录下执行，
  否则可能因 Python 模块搜索路径冲突（本地目录遮蔽 `vllm` 包）导致导入/启动失败。
- 建议在独立的工作目录（例如 `/workspace`）中启动服务。
- 模型路径一致性：启动 vLLM 时使用的 `--model` 路径必须与后续 `curl` 请求体中的 `"model"` 字段完全一致
  （例如都为 `/tmp/ckpt/Qwen3-0.6B`）。不一致可能导致加载/路由失败。

## 联系方式

如有问题或需要支持，请联系维护人员。

## 自定义存储后端实现

如果你希望接入自定义存储（S3/OSS/NFS/自研分布式存储等），可实现 `AbstractStorage` 并在工厂中注册。

- 接口约定（见 `distributed_kv_manager/storage/base.py`）：
  - `upload(file_path: str, data: bytes) -> bool`
  - `download(file_path: str) -> Optional[bytes]`
  - `exists(file_path: str) -> bool`
  - `pack_kv_data(k_cache, v_cache, hidden, input_tokens, roi) -> bytes`
  - `unpack_kv_data(data: bytes) -> (k_cache, v_cache, hidden)`
  - 可选：`delete`、`list_files`、`extract_metadata_from_data`

- 打包/解包建议
  - 可参考 `LocalStorage`：用 `torch.save` 将 CPU 张量写入字节流；`unpack_kv_data` 返回 CPU 张量，设备迁移由引擎负责。
  - 引擎可能会在写入前嵌入元数据（头部+定长结构），读取时会先剥离再调用你的 `unpack_kv_data`，通常无需自行解析元数据。

- 最小示例

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

- 在工厂中注册（`distributed_kv_manager/storage/factory.py`）：

```python
from .my_storage import MyStorage  # 新增导入

class StorageFactory:
    @staticmethod
    def create_storage(config):
        storage_type = getattr(config.kv_transfer_config, "storage_type", "local")
        # ...
        if storage_type == "my_storage":
            root = getattr(config.kv_transfer_config, "my_dir", "/tmp/kvcache_my")
            base_storage = MyStorage(root)
        # 如需 SSD 包裹，在 enable_ssd_caching=True 时会被 CachingStorage 包裹
```

- 配置示例（JSON 或 Python 均可）：

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
- 确保 `file_path` 为相对键（基目录由后端/工厂拼接）。
- 保证 `pack/unpack` 的兼容性：形状与 dtype 必须能无损还原。
- 启用 SSD 包裹时，`upload/download/exists` 要保持对同一 `file_path` 的一致语义。

## 预取组件结构（Prefetch）

预取与 I/O 聚合逻辑已独立为一个包：

- 包路径：`distributed_kv_manager.prefetch`
- 实现文件：`distributed_kv_manager/prefetch/core.py`
- 公共 API：通过包的 `__init__.py` 统一导出

导入示例：

```python
from distributed_kv_manager.prefetch import (
  PrefetchBuffer, BudgetEstimator, RateLimiter, IOAggregator, PlanBuilder,
)
```

注意：旧的顶层模块 `distributed_kv_manager/prefetch.py` 已移除，请改用上述包路径。

## 注意事项 

- 使用 `destroy_engine()` 结束：确保后台清理与异步写任务退出
- 提供稳定键：`session_id` (bytes) + `layer_id` (int) 必须在 `model_input` 中
- 远端路径优先 `remote_dir`；仅当需要兼容旧配置时使用 `crail_dir`
- 占位远端(noop) 不会真正持久化数据，适合本地验证流程
- ETCD 不可达时：检索自动 MISS，并可能输出日志警告

## 路线图

- 自适应层拆分（基于延迟与命中率）
- 动态预取预算调节（利用率反馈）
- 远端后端扩展（Crail/ AS13000）
