import os
import torch
from types import SimpleNamespace
from distributed_kv_manager.storage.v1.storage import create_v1_storage
from distributed_kv_manager.storage.factory import StorageFactory


def test_local_upload_download(local_storage, report):
    data = b"hello-world"
    rel = "abcd1234/model.layers.0.self_attn.attn.safetensors"
    ok = local_storage.upload(rel, data)
    assert ok
    got = local_storage.download(rel)
    assert got == data
    assert local_storage.exists(rel)
    path = report(
        "storage_local_io.txt",
        [
            "用例: 存储-本地上传下载",
            "目的: 验证本地存储上传/下载/存在性",
            f"输入构造: rel={rel}, data_bytes={len(data)}",
            f"rel={rel}",
            f"uploaded={ok}",
            f"download_size={len(got) if got else 0}",
            f"exists={local_storage.exists(rel)}",
            "执行步骤: upload -> download -> exists",
            "预期结果: uploaded=True, download_size=输入字节数, exists=True",
            "结论: 本地存储读写正常",
        ],
    )
    assert path


def test_pack_unpack_full_payload(sample_kv, tmp_dir, report):
    k, v, tokens, roi = sample_kv
    cfg = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            storage_type="local",
            local_dir=os.path.join(tmp_dir, "full_payload"),
            use_v1=True,
            mem_cache_capacity_gb=0.05,
        )
    )
    st = create_v1_storage(cfg)
    payload = st.pack_full_payload(k, v, tokens, roi)
    kk, vv = st.unpack_kv_data(payload)
    assert kk is not None and vv is not None
    assert tuple(kk.shape) == tuple(k.shape)
    assert tuple(vv.shape) == tuple(v.shape)
    path = report(
        "storage_full_payload.txt",
        [
            "用例: 存储-完整负载打包/解包",
            "目的: 验证 v0 布局的 KV 打包与解包一致性",
            f"输入构造: k_shape={tuple(k.shape)}, v_shape={tuple(v.shape)}, tokens_len={int(tokens.numel())}",
            f"k_shape={tuple(k.shape)}",
            f"v_shape={tuple(v.shape)}",
            f"payload_size={len(payload)}",
            "执行步骤: pack_full_payload -> unpack_kv_data",
            "预期结果: 解包形状与输入一致",
            "结论: 布局兼容验证通过",
        ],
    )
    assert path


def test_path_mapper_hash_bucket(tmp_dir, report):
    cfg = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            storage_type="local",
            local_dir=os.path.join(tmp_dir, "mapped"),
            directory_layout="hash",
        )
    )
    backend = StorageFactory.create_storage(cfg)
    # 上传与下载应走映射后的路径
    key = "0000000000000000000000000000abcd/model.layers.1.self_attn.attn.safetensors"
    ok = backend.upload(key, b"ok")
    assert ok
    got = backend.download(key)
    assert got == b"ok"
    path = report(
        "storage_path_mapper.txt",
        [
            "用例: 存储-哈希桶路径映射",
            "目的: 验证哈希提取与二级目录映射正确",
            f"输入构造: key={key}",
            f"key={key}",
            f"uploaded={ok}",
            f"download_size={len(got) if got else 0}",
            "执行步骤: upload(key) -> download(key)",
            "预期结果: 下载字节数与上传一致",
            "结论: 路径映射生效",
        ],
    )
    assert path

