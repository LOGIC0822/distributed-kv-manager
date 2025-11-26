# 基础镜像：使用包含 CUDA 的 PyTorch 镜像
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

# 设置源并安装依赖
RUN sed -i 's@http://.*archive.ubuntu.com@https://mirrors.aliyun.com@g' /etc/apt/sources.list && \
    sed -i 's@http://.*security.ubuntu.com@https://mirrors.aliyun.com@g' /etc/apt/sources.list && \
    apt update && \
    apt install -y \
        build-essential git wget libtool autoconf apt-utils curl \
        openjdk-11-jdk maven \
        python3 python3-pip libibverbs-dev librdmacm-dev \
        nvme-cli sudo iproute2 pciutils libnuma-dev \
        libaio-dev libssl-dev uuid-dev kmod ninja-build cmake python3-dev \
        software-properties-common && \
    apt clean

# 安装 Rust 工具链
# RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# ENV PATH="/root/.cargo/bin:${PATH}"

# 设置 JAVA 环境
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# 安装 Python 依赖（使用国内镜像源）
RUN python3 -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install numpy==1.24.4 sentencepiece==0.1.99 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 拷贝 vLLM 源码和 cutlass 子模块（防止编译阶段失败）
RUN git clone -b v0.9.2 https://github.com/vllm-project/vllm.git /root/vllm
# COPY cutlass /root/vllm/.deps/cutlass-src

# 设置编译环境变量
ENV MAX_JOBS=20
ENV MAKEFLAGS="-j${MAX_JOBS}"
ENV CMAKE_BUILD_PARALLEL_LEVEL=${MAX_JOBS}

# 安装 vLLM 所需依赖，启用 editable 模式
RUN pip install \
        setuptools==77.0.1 \
        packaging>=24.2 \
        cmake \
        ninja \
        regex \
        setuptools_scm \
        wheel \
        pytest \
        -i https://pypi.tuna.tsinghua.edu.cn/simple 

#测试元数据时若若不用构建vLLM 可以先注释掉 
RUN pip install -e /root/vllm
RUN pip install transformers==4.53.2

RUN pip install --upgrade pip setuptools wheel packaging
RUN pip install "protobuf>=3.19.0,<3.21" \
                etcd3

# 安装 etcd 如果在线安装失败可以自行下载解压后直接复制
# 下载并解压 etcd
RUN mkdir -p /root/etcd && \
    cd /root/etcd && \
    wget -q https://github.com/etcd-io/etcd/releases/download/v3.6.4/etcd-v3.6.4-linux-amd64.tar.gz   && \
    tar -xzf etcd-v3.6.4-linux-amd64.tar.gz --strip-components=1 && \
    rm etcd-v3.6.4-linux-amd64.tar.gz
RUN mkdir -p /tmp/etcd-data

RUN git clone -b dev https://github.com/Yan-tx/distributed-kv-manager.git /root/distributed_kv_manager

# 构建 Rust 扩展模块
# RUN cd /root/distributed_kv_manager/rust_extensions/kv_serializer && cargo build --release

RUN cp /root/distributed_kv_manager/vllm_adapter/* /root/vllm/vllm/distributed/kv_transfer/kv_connector/
RUN pip install -e /root/distributed_kv_manager

WORKDIR /root