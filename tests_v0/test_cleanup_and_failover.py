import os
import time
from distributed_kv_manager.metadata.cleanup import KVCleanupManager
from distributed_kv_manager.metadata.etcd import KVMetadata


def test_gc_deletes_expired(meta_manager, local_storage, report):
    # 在本地存储写入一个文件，并创建过期元数据
    rel = "beefcafe/model.layers.2.self_attn.attn.safetensors"
    local_storage.upload(rel, b"to-delete")
    now = int(time.time())
    meta = KVMetadata(
        session_id=b"gc_session_______",
        layer_id=0,
        token_idx="0",
        file_path=rel,
        file_size=len(b"to-delete"),
        create_time=now - 100,
        last_access=now - 100,
        expire_time=1,
        replica_locations=[b"", b"", b""],
        status=1,
        schema_version=1,
        ext_flags=0,
        ext_data=b"",
        ext_data_len=0,
    )
    meta_manager.put_metadata(rel, meta, replicate=False)
    gc = KVCleanupManager(meta_manager, cleanup_interval=1, storage=local_storage)
    # 直接调用一次清理逻辑
    gc._perform_cleanup()
    assert local_storage.download(rel) is None
    assert meta_manager.get_metadata(rel) is None
    path = report(
        "cleanup_gc.txt",
        [
            "用例: 清理-过期条目回收",
            "目的: 验证过期KV的物理删除与元数据删除",
            f"输入构造: rel={rel}, expire_time=1s, last_access=已过期",
            f"rel={rel}",
            "deleted=True",
            "metadata_removed=True",
            "执行步骤: put_metadata(过期) -> _perform_cleanup",
            "预期结果: 文件不存在且元数据不存在",
            "结论: 过期清理完成",
        ],
    )
    assert path


def test_failover_update_access(meta_manager, report):
    # 访问一个不存在的键，update_access_time 不应异常
    meta_manager.update_access_time("nonexistent/key")
    # 仍然不可用
    assert meta_manager.get_metadata("nonexistent/key") is None
    path = report(
        "cleanup_failover.txt",
        [
            "用例: 清理-故障处理与健壮性",
            "目的: 验证访问不存在键时的健壮性",
            "key=nonexistent/key",
            "update_access_time_ok=True",
            "get_metadata_none=True",
            "执行步骤: update_access_time -> get_metadata",
            "预期结果: 无异常且查询为空",
            "结论: 故障路径安全",
        ],
    )
    assert path

