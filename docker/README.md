# Docker 部署说明

## 概述

本目录包含用于部署分布式KV管理器的Docker配置文件。我们提供了两个不同的Dockerfile以满足不同的部署需求：

- `Dockerfile`: 完整版本，包含Crail分布式存储支持
- `Dockerfile.no-crail`: 精简版本，仅支持本地存储

## Dockerfile 说明

### Dockerfile (完整版)
包含所有组件：
- Crail分布式存储系统
- Disni RDMA库
- SPDK存储驱动
- ETCD元数据管理
- vLLM推理引擎
- 分布式KV管理器

适用于生产环境和需要高性能分布式存储的场景。

### Dockerfile.no-crail (精简版)
仅包含核心组件：
- ETCD元数据管理
- vLLM推理引擎
- 分布式KV管理器（仅本地存储）

适用于开发测试环境或不需要分布式存储的场景。

## 构建镜像

### 构建完整版镜像
```bash
cd docker
docker build -t crail .
```

### 构建精简版镜像
```bash
cd docker
docker build -t kvcache .
```

## 使用说明

### 使用完整版
如果您需要使用Crail分布式存储功能：
1. 确保您的环境中可以安装并配置Crail集群
2. 构建完整版Docker镜像
3. 运行容器时需要挂载相关配置和数据卷

### 使用精简版
如果您只需要本地存储功能：
1. 构建精简版Docker镜像
2. 直接运行容器即可

## 运行脚本（run.sh）说明与用法

为方便快速启动，仓库提供了两份运行脚本，分别对应两套镜像：

- `docker/docker_no-crail/run.sh`（精简版镜像，默认镜像名 `kvcache`）
- `docker/docker/run.sh`（带 Crail 的完整版镜像，默认镜像名 `crail`）

两份脚本均在文件顶部集中定义了用户可修改的变量，建议根据环境自行调整后再执行。

### docker_no-crail/run.sh（本地存储，Lite）

关键变量：
- `IMAGE_NAME`：镜像名，默认 `kvcache`。
- `CONTAINER_NAME`：容器名，默认与镜像名一致。
- `CKPT_HOST_DIR`：宿主机上模型权重/检查点目录（需存在）。
- `CKPT_MOUNT_TARGET`：容器内挂载路径（默认 `/tmp/ckpt`）。
- `DOCKER_EXTRA_ARGS`：其他 Docker 参数（GPU/Host 网络/特权等）。

用法：
```bash
cd docker/docker_no-crail
chmod +x run.sh
./run.sh
```

进入容器后，参考根目录 README 的“vLLM 快速启动与测试”，先启动 ETCD，再启动 vLLM，并确保 `--model` 与 `curl` 请求中的 `"model"` 路径一致。

### docker/run.sh（含 Crail，Full）

关键变量：
- `IMAGE_NAME` / `CONTAINER_NAME`：默认 `crail`。
- `NVME_DEV`：宿主机 NVMe 设备名（例如 `nvme0`，脚本会映射 `/dev/$NVME_DEV`）。
- `CKPT_HOST_DIR` / `CKPT_MOUNT_TARGET`：模型权重目录映射。
- `HUGEPAGES_DATA` / `HUGEPAGES_CACHE` / `HUGEPAGES_SYS` / `HUGEPAGES_DEV`：HugePages 相关挂载路径。
- `DOCKER_EXTRA_ARGS`：包含 Infiniband 与 NVMe 设备映射。

用法：
```bash
cd docker/docker
chmod +x run.sh
./run.sh
```

运行前请确认：
- 宿主机存在并已配置 HugePages，对应目录已创建（参考集群/内核配置）。
- Infiniband 与 NVMe 设备可用且具有访问权限。
- 已按需要构建对应镜像名（或修改脚本中的 `IMAGE_NAME`）。

## 目录与脚本说明（docker/docker）

该目录除 Dockerfile 外，还包含两类脚本/文件：

- `bash/` 目录（在宿主机执行，用于环境准备）
  - `bash/prepare.sh`：基于 1G HugePages 的主机环境准备脚本。执行内容包括：加载 RDMA/NVMe/MLX5 模块、挂载 `/mnt/hugepages/{data,cache}`、检查并（可选）使用 `spdk/scripts/setup.sh` 绑定 NVMe、生成 `hostnqn` 并写回 `crail_conf/crail-site.conf`、配置 IB 网卡 IP。
  - `bash/2M_prepare.sh`：与上类似，但使用 2MB HugePages。请选择其中一个版本在宿主机执行，确保容器运行时所需的 HugePages 与 NVMe/IB 环境就绪。

- 将被 Dockerfile 拷入容器内的脚本与配置
  - `run_node.sh`（复制至 `/root/run_node.sh`）：容器内的 Crail/ SPDK 启动脚本。内容包含：配置 Transparent HugePage、挂载 hugetlbfs、启动 `spdk nvmf_tgt`、创建 NVMe bdev、创建 NVMf 传输/子系统/监听器、最后启动 Crail NameNode 或 DataNode（通过参数 `namenode|datanode` 控制，默认 datanode）。
  - `test.sh`（复制至 `/root/test.sh`）：容器内的 Crail FS 基础验证脚本（mkdir/put/cat/ls/cp/rm）。
  - `clean.sh`（复制至 `/root/clean.sh`）：容器内的清理脚本（停止 nvmf_tgt、删除 NVMf 子系统、卸载 HugePages、结束 Crail 进程）。
  - `crail_conf/`（复制至 `/root/crail/conf`）：Crail 配置目录，包括 `crail-site.conf`、`crail-env.sh`、`core-site.xml`、`log4j.properties`、`slaves` 等。
  - `crail-example/`（复制至 `/root/crail-example` 并由 Maven 构建）：包含 JNI 库与示例客户端工程（如 `libcrail_jni.so`、Java 源码等）。
  - `test_jni_call.py`（复制至 `/root/test_jni_call.py`）：在容器内验证 JNI 调用链是否正常（依赖 `crail-example` 构建产物）。

> 提示：如果 IB 地址、NVMe PCI BDF、NQN、HugePages 页面大小等与默认脚本不一致，请在使用前修改相应脚本中的变量（例如 `run_node.sh` 中的 `IB_ADDR`/`IB_PORT`、`bash/*prepare*.sh` 中的 `NVME_BDF`/页面大小、`crail_conf` 中的 Crail 参数等）。

## 注意事项

1. **依赖关系**: 完整版镜像包含大量依赖组件，构建时间较长，请确保网络连接稳定。

2. **资源需求**: 完整版镜像较大，运行时需要足够的磁盘空间和内存。

3. **配置文件**: 运行容器时可能需要根据实际环境调整配置文件。

## 常见问题

### 构建失败
如果在构建过程中遇到网络问题，可以：
- 使用国内镜像源（Dockerfile中已配置）
- 手动下载依赖包并复制到构建上下文

### 运行时错误
如果容器运行时出现问题：
- 检查日志输出
- 验证配置文件是否正确
- 确认依赖服务（如ETCD）是否正常运行

## vLLM 快速启动与测试（容器内/宿主机均可）

以下为使用本项目连接器启动 vLLM（v0 接口）并做简单验证的命令示例：

### 启动 vLLM OpenAI 兼容服务（v0 接口）

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

### 启动本地 ETCD（如未启动）

```bash
cd ~/etcd
nohup ./etcd \
  --data-dir /tmp/etcd-data \
  --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://127.0.0.1:2379 \
  > /tmp/etcd.log 2>&1 &
```

### 基础请求测试

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

- 目录路径：启动上述 `api_server` 时，请不要在 `~` 或 `~/vllm` 等与源码重名/冲突的目录下执行，以免因 Python 模块搜索路径冲突（例如本地目录名覆盖 `vllm` 包）导致导入失败或启动异常。建议在独立的工作目录（如 `/workspace` 或任意非 `vllm` 命名的目录）中启动服务。
- 模型路径一致性：`--model` 指定的路径需与后续 `curl` 请求体中的 `"model"` 字段完全一致（例如都为 `/tmp/ckpt/Qwen3-0.6B`）。如不一致，将无法正确路由到对应权重，可能导致 404/加载失败或命中错误模型。
