from distributed_kv_manager.metadata.metadata_cache import MetadataCache
from distributed_kv_manager.metadata.etcd import KVMetadata


def test_metadata_cache_pool_hits(meta_manager, report):
    cache = MetadataCache(meta_manager, pool2_layers=[0, 1], pool3_size=10)
    m1 = KVMetadata(
        session_id=b"ses1____________",
        layer_id=0,
        token_idx="0",
        file_path="aa/bb/cccc",
        file_size=1,
        create_time=1,
        last_access=1,
        expire_time=0,
        replica_locations=[b"", b"", b""],
        status=1,
        schema_version=1,
        ext_flags=0,
        ext_data=b"",
        ext_data_len=0,
    )
    meta_manager.put_metadata("aa/bb/cccc", m1, replicate=False)
    # 第一次查询：ETCD 命中
    got = cache.get_metadata("aa/bb/cccc")
    assert got is not None
    # 第二次查询：Pool3 命中
    got2 = cache.get_metadata("aa/bb/cccc")
    assert got2 is not None
    # 带 layer_id：Pool2 命中
    got3 = cache.get_metadata("aa/bb/cccc", layer_id=0)
    assert got3 is not None
    stats = cache.get_stats()
    assert stats["total_queries"] >= 3
    assert stats["pool2_hits"] >= 1
    assert stats["pool3_hits"] >= 1
    path = report(
        "metadata_cache_stats.txt",
        [
            "用例: 元数据-多层缓存命中与统计",
            "目的: 验证 Pool2/Pool3 命中与统计聚合",
            "输入构造: session_id=ses1, layer_id=0, file_path=aa/bb/cccc",
            "执行步骤: put_metadata -> get_metadata(多次/带layer)",
            f"total_queries={stats['total_queries']}",
            f"hit_rate={stats['hit_rate']:.2%}",
            f"pool2_hits={stats['pool2_hits']}",
            f"pool3_hits={stats['pool3_hits']}",
            f"etcd_hits={stats['etcd_hits']}",
            "预期结果: 命中率>0 且 Pool2/Pool3 均有命中",
            "结论: 多层缓存工作正常",
        ],
    )
    assert path

