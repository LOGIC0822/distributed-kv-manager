import unittest
import types
from unittest.mock import patch
try:
    import torch
except ImportError:  # pragma: no cover
    from typing import Any, cast
    torch = cast(Any, None)

from distributed_kv_manager.engine import (
    init_engine, destroy_engine, store_kv, retrieve_kv,
    should_retrieve, StoreStatus, RetrieveStatus
)

class DummyAttnMeta:
    def __init__(self, seq_lens, slot_mapping):
        self.seq_lens = seq_lens
        self.slot_mapping = slot_mapping

class DummyModelInput:
    def __init__(self, input_tokens, attn_metadata, session_id=b"test_session", layer_id=0):
        self.input_tokens = input_tokens
        self.attn_metadata = attn_metadata
        self.session_id = session_id
        self.layer_id = layer_id
        # optional payload_meta (connector would attach it); engine will merge it
        self.payload_meta = {}

class DummyModelExecutable:
    class DummyEmbed:
        def __init__(self, dim):
            self.embedding_dim = dim
    def __init__(self, hidden_dim=16):
        self.model = types.SimpleNamespace(embed_tokens=DummyModelExecutable.DummyEmbed(hidden_dim))

@unittest.skipIf(torch is None, "torch 未安装，跳过引擎相关测试")
class EngineAggregateBlocksTests(unittest.TestCase):
    def setUp(self):
        cfg = types.SimpleNamespace(
            rank=0,
            local_rank=0,
            kv_transfer_config=types.SimpleNamespace(
                storage_type="local",
                local_dir="./tmp_local_storage",
                storage_dir="./tmp_local_storage",
                etcd_endpoints=["127.0.0.1:2379"],
                kv_expire_time=60,
                cleanup_interval=1000,  # keep cleanup thread idle
                enable_ssd_caching=False,
                enable_prefetch=False,
                enable_debug_dump=False,
            )
        )
        # A simple in-memory KVMetadataManager replacement that supports list/get/put
        class _MemMM:
            def __init__(self, *a, **kw):
                self.prefix = "/kvmeta"
                self._store = {}  # full_key -> meta
            def put_metadata(self, key, meta, replicate=True):
                full_key = f"{self.prefix}/{str(key).lstrip('/')}"
                self._store[full_key] = meta
            def get_metadata(self, key):
                full_key = f"{self.prefix}/{str(key).lstrip('/')}"
                return self._store.get(full_key)
            def get_metadata_by_full_key(self, full_key):
                return self._store.get(full_key)
            def scan_all_metadata_keys(self):
                return list(self._store.keys())
            def delete_metadata(self, key, replicate=True):
                full_key = f"{self.prefix}/{str(key).lstrip('/')}"
                self._store.pop(full_key, None)
        with patch("distributed_kv_manager.engine.kv_engine.KVMetadataManager", _MemMM):
            self.engine = init_engine(cfg)

    def tearDown(self):
        destroy_engine()

    def _make_block_kv(self, num_layers=2, block_tokens=8, heads=2, head_dim=4):
        kv_caches = []
        for _ in range(num_layers):
            k = torch.randn(block_tokens, heads, head_dim)
            v = torch.randn(block_tokens, heads, head_dim)
            kv = torch.stack([k, v], dim=0)
            kv_caches.append(kv)
        return kv_caches

    def test_aggregate_two_blocks_full_bypass(self):
        model_exec = DummyModelExecutable(hidden_dim=8)
        # two blocks of 8 tokens each
        blk1_tokens = torch.randint(0, 1000, (8,))
        blk2_tokens = torch.randint(0, 1000, (8,))
        # store block 1
        attn1 = DummyAttnMeta(seq_lens=[8], slot_mapping=torch.arange(8))
        mi1 = DummyModelInput(blk1_tokens, attn1)
        mi1.payload_meta = {"token_offset": 0, "block_index": 0, "block_size": 8, "total_tokens": 16}
        kv1 = self._make_block_kv()
        store_kv(None, None, None, model_exec, mi1, kv1, StoreStatus.STORED)
        # store block 2
        attn2 = DummyAttnMeta(seq_lens=[8], slot_mapping=torch.arange(8))
        mi2 = DummyModelInput(blk2_tokens, attn2)
        mi2.payload_meta = {"token_offset": 8, "block_index": 1, "block_size": 8, "total_tokens": 16}
        kv2 = self._make_block_kv()
        store_kv(None, None, None, model_exec, mi2, kv2, StoreStatus.STORED)
        # wait async
        for f in getattr(self.engine, "_futures", []):
            try:
                f.result(timeout=15)
            except Exception:
                pass
        # retrieve for full 16 tokens (no exact metadata key exists for the full hash)
        full_tokens = torch.cat([blk1_tokens, blk2_tokens], dim=0)
        attn_full = DummyAttnMeta(seq_lens=[16], slot_mapping=torch.arange(16))
        mi_full = DummyModelInput(full_tokens, attn_full)
        # provide empty kv caches with capacity 16
        num_layers = 2
        heads, head_dim = 2, 4
        empty_kv = []
        for _ in range(num_layers):
            k = torch.zeros(16, heads, head_dim)
            v = torch.zeros(16, heads, head_dim)
            empty_kv.append(torch.stack([k, v], dim=0))
        # even if should_retrieve says MISS, retrieve_kv should aggregate blocks and may bypass if full
        rstatus = should_retrieve(mi_full)
        self.assertEqual(rstatus, RetrieveStatus.MISS)
        hidden, bypass, _ = retrieve_kv(model_exec, mi_full, empty_kv, rstatus)
        # 引擎在 retrieve_kv 中会更新 _last_retrieve_stats，可直接访问实例属性
        stats = getattr(self.engine, "_last_retrieve_stats", {})
        self.assertEqual(stats.get("total_restored_tokens"), 16, f"stats={stats}")
        self.assertTrue(stats.get("bypass"), f"stats={stats}")
        self.assertTrue(bypass)
        self.assertIsNotNone(hidden)

if __name__ == "__main__":
    unittest.main()
