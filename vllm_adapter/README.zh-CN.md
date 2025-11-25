# vLLM 适配器：Distributed KV Manager 集成

本目录包含将 Distributed KV Manager 集成进 vLLM 的适配器文件。使用时需将这些文件放到 vLLM 安装目录的 kv_connector 子目录下。

## 安装

在你的 vLLM 环境中集成 Distributed KV Manager 的步骤：

1. 找到 vLLM 的安装目录。
2. 进入 `vllm/distributed/kv_transfer/kv_connector/` 目录。
3. 将本目录中的文件复制到该 `kv_connector` 目录下：
   - `distributed_kv_connector.py`
   - `factory.py`

注意：这会替换 `kv_connector/` 下的 `factory.py`，如有需要请先备份原文件。

## 文件位置

复制后的目录结构应如下所示：

## 与 vLLM 的启动方式

支持两种方式：旧版注入 (v0) 与新版直接启动 (v1)。

- v0（注入脚本）：
  ```bash
  python3 vllm_adapter/vllm_start_with_inject.py \
    --model /tmp/ckpt --port 8100 --max-model-len 10000 \
    --kv-transfer-config '{"kv_connector":"DistributedKVConnector","kv_role":"kv_both"}'
  ```

- v1（直接启动）：
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

## 配置

复制完成后，需要在 vLLM 配置中指定要使用的 KV 连接器。

### 注册连接器

提供的 `factory.py` 会在工厂中注册 `DistributedKVConnector`。请确保你的 vLLM 配置包含：

```python
# vLLM 配置中
kv_transfer_config = {
    "kv_connector": "DistributedKVConnector",
    # 可按需添加其他配置项
}
```

### 依赖

确保 `distributed_kv_manager` 已安装并可被 Python 环境导入：

```bash
pip install -e /path/to/distributed_kv_manager
```

## 使用说明

完成安装与配置后，Distributed KV Manager 会在 vLLM 推理过程中自动处理 KV Cache 的存取。

该适配器提供两项核心能力：
1. `recv_kv_caches_and_hidden_states`：从分布式存储中读取 KV cache 与 hidden states。
2. `send_kv_caches_and_hidden_states`：将 KV cache 与 hidden states 写入分布式存储。

## 故障排查

如遇问题，请检查：

1. 文件是否已复制到 `kv_transfer/kv_connector/`。
2. `distributed_kv_manager` 是否正确安装并可导入。
3. vLLM 版本与该适配器是否兼容。
4. 配置中是否正确指定了 `DistributedKVConnector`。

## 更新

当升级 Distributed KV Manager 时，可能也需要同步更新这些适配器文件。更新前请备份原文件。

