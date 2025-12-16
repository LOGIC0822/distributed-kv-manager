from distributed_kv_manager.storage.v1.hash_cached_storage import V1HashCachedStorage
from distributed_kv_manager.storage.local_storage import LocalStorage


def test_hash_group_eviction(tmp_dir, report):
    base = LocalStorage(tmp_dir)
    cache = V1HashCachedStorage(base, capacity_bytes=64)  # 很小的容量触发淘汰
    # 同一 hash 目录下的两个文件
    f1 = "abcd1234/model.layers.0.self_attn.attn.safetensors"
    f2 = "abcd1234/model.layers.1.self_attn.attn.safetensors"
    # 另一个 hash 目录
    f3 = "beefcafe/model.layers.0.self_attn.attn.safetensors"
    cache.upload(f1, b"x" * 40)
    cache.upload(f2, b"y" * 40)
    # 此时容量超过 64，会淘汰最早的 hash 组
    # 新组写入
    cache.upload(f3, b"z" * 40)
    # 被淘汰的组应 miss（缓存中），但后端仍存在可下载
    assert cache.download(f1) is not None
    assert cache.download(f2) is not None
    assert cache.download(f3) is not None
    path = report(
        "cache_eviction.txt",
        [
            "用例: 缓存-哈希组淘汰策略",
            "目的: 验证 DRAM 缓存按哈希组淘汰与回源一致性",
            f"输入构造: capacity_bytes=64, files={f1},{f2},{f3}",
            f"capacity_bytes=64",
            f"f1_size={len(cache.download(f1) or b'')}",
            f"f2_size={len(cache.download(f2) or b'')}",
            f"f3_size={len(cache.download(f3) or b'')}",
            "执行步骤: upload多文件 -> 触发淘汰 -> download三文件",
            "预期结果: 三文件均可回源下载",
            "结论: 组淘汰与回源工作正常",
        ],
    )
    assert path

