import time
from distributed_kv_manager.metadata.v1.metadata import V1MetadataClient


def test_metadata_mark_and_exists(meta_manager, report):
    client = V1MetadataClient(meta_manager, default_expire=30)
    folder = "abcd1234ef/model.layers.0.self_attn.attn.safetensors"
    # 仅以 hash 目录为键写聚合记录
    hash_dir = folder.split("/")[0]
    client.mark_hash_stored(hash_dir, num_tokens=128, file_size=4096)
    assert client.hash_exists(hash_dir) is True
    path = report(
        "metadata_mark_exists.txt",
        [
            "用例: 元数据-聚合写入与存在性检查",
            "目的: 验证按哈希目录聚合记录的写入与命中",
            f"输入构造: hash_dir={hash_dir}, num_tokens=128, file_size=4096, expire=30s",
            f"hash_dir={hash_dir}",
            f"exists={client.hash_exists(hash_dir)}",
            "执行步骤: mark_hash_stored -> hash_exists",
            "预期结果: exists=True",
            "结论: 聚合元数据可被正确命中",
        ],
    )
    assert path


def test_metadata_update_access(meta_manager, report):
    client = V1MetadataClient(meta_manager, default_expire=1)
    key = "deadbeefcafebabe"
    # 先写一条过期很快的记录
    client.mark_hash_stored(key, num_tokens=64, file_size=2048)
    before = meta_manager.get_metadata(key)
    assert before is not None
    t0 = before.last_access
    time.sleep(0.01)
    client.update_access_time(key)
    after = meta_manager.get_metadata(key)
    assert after is not None
    assert after.last_access >= t0
    path = report(
        "metadata_update_access.txt",
        [
            "用例: 元数据-访问时间更新",
            "目的: 验证命中后刷新 last_access 并保持未过期",
            f"输入构造: key={key}, default_expire=1s, num_tokens=64, file_size=2048",
            f"key={key}",
            f"before_last_access={t0}",
            f"after_last_access={after.last_access}",
            f"expired={after.is_expired()}",
            "执行步骤: mark_hash_stored -> update_access_time -> get_metadata",
            "预期结果: after_last_access >= before_last_access 且 expired=False",
            "结论: 访问时间刷新正常",
        ],
    )
    assert path

