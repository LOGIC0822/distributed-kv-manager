import tempfile
from types import SimpleNamespace
from typing import Any, Dict

try:
    import pytest  # type: ignore
except Exception:
    pytest = None

HAS_TORCH = True
try:
    import torch as _torch  # type: ignore
    torch: Any = _torch
except Exception:
    HAS_TORCH = False
    torch = object()  # type: ignore[assignment]

HAS_ETCD3 = True
try:
    import etcd3 as _etcd3  # type: ignore
    etcd3: Any = _etcd3
except Exception:
    HAS_ETCD3 = False
    etcd3 = object()  # type: ignore[assignment]

if pytest is not None and (not HAS_TORCH or not HAS_ETCD3):
    pytest.skip("Requires torch and etcd3 runtime; skipping v1 engine test.", allow_module_level=True)


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)


class _VLLMConfig:
    def __init__(self, storage_dir: str):
        # minimal kv_transfer_config used by engine
        self.kv_transfer_config = SimpleNamespace(
            storage_type="local",
            storage_dir=storage_dir,
            local_dir=storage_dir,
            etcd_endpoints=["127.0.0.1:2379"],
            enable_ssd_caching=False,
            ssd_cache_dir=storage_dir,
            enable_prefetch=False,
            chunk_size=8,
            force_sync_store=True,
            engine_id="v1_test_session",
            # 传递一个临时 config_path 防止默认 config.json 覆盖测试的storage_dir
            config_path=None,
        )
        # placeholders required by store_kv signature
        self.model_config = SimpleNamespace()
        self.parallel_config = SimpleNamespace()


class _Request:
    def __init__(self, request_id: str, tokens: list[int]):
        self.request_id = request_id
        self.prompt_token_ids = tokens
        self.num_tokens = len(tokens)
        self.num_computed_tokens = 0
        self.model = _FakeModel()


class _ForwardContext:
    def __init__(self, model: Any, kv_caches: Dict[str, Any]):
        self.model = model
        self.kv_caches = kv_caches
        # requests map is not strictly required here, engine keeps unfinished map


def _make_layer_kv(batch: int, seq: int, num_heads: int, head_dim: int):
    k = torch.randn(batch, seq, num_heads, head_dim)
    v = torch.randn(batch, seq, num_heads, head_dim)
    return torch.stack([k, v], dim=0)


def _make_zero_layer_kv(batch: int, seq: int, num_heads: int, head_dim: int):
    k = torch.zeros(batch, seq, num_heads, head_dim)
    v = torch.zeros(batch, seq, num_heads, head_dim)
    return torch.stack([k, v], dim=0)


def test_v1_engine_roundtrip_full_hit():
    from distributed_kv_manager.engine.v1 import init_v1_engine, destroy_v1_engine

    with tempfile.TemporaryDirectory() as temp_dir:
        vcfg = _VLLMConfig(temp_dir)
        # 写入覆盖用的 config.json 到临时路径，避免项目根的 config.json 覆盖本测试参数
        import json, os
        tmp_cfg_path = os.path.join(temp_dir, "config.json")
        with open(tmp_cfg_path, "w", encoding="utf-8") as f:
            json.dump({
                "kv_transfer_config": {
                    "storage_type": "local",
                    "local_dir": temp_dir,
                    "storage_dir": temp_dir,
                    "etcd_endpoints": ["127.0.0.1:2379"],
                    "enable_ssd_caching": False,
                    "enable_prefetch": False,
                    "chunk_size": 8,
                    "engine_id": "v1_test_session"
                }
            }, f)
        vcfg.kv_transfer_config.config_path = tmp_cfg_path

        # build v1 engine instance
        core = init_v1_engine(vcfg)

        # build original KV caches and register
        batch, seq, num_heads, head_dim = 1, 16, 2, 8
        num_layers = 2
        orig_kvs: Dict[str, Any] = {
            f"layer_{i}": _make_layer_kv(batch, seq, num_heads, head_dim)
            for i in range(num_layers)
        }
        core.register_kv_caches(orig_kvs)

        # new request and scheduling
        tokens = list(range(seq))
        req = _Request("req1", tokens)
        # scheduler: update after alloc so engine tracks unfinished request
        core.update_state_after_alloc(req, None, 0)

        # build scheduler_output and forward_context
        num_sched = {req.request_id: len(tokens)}
        fc = _ForwardContext(req.model, orig_kvs)
        sched = SimpleNamespace(
            finished_req_ids=set(),
            scheduled_new_reqs=[req],
            num_scheduled_tokens=num_sched,
            scheduled_cached_reqs=[],
            forward_context=fc,
        )
        core.build_connector_meta(sched)

        # simulate compute completion for prefill
        req.num_computed_tokens = len(tokens)
        # trigger save
        # 触发保存（前缀完整）
        core.wait_for_save()
        # 强制等待底层 futures 完成，确保文件已写入再重建引擎
        try:
            engine = core._engine  # type: ignore[attr-defined]
            for f in list(getattr(engine, '_futures', []) or []):
                try:
                    f.result(timeout=20)
                except Exception:
                    pass
        except Exception:
            pass
        # 验证物理文件是否存在
        import os
        t = torch.tensor(tokens, dtype=torch.long)
        # 使用实际引擎的 _make_key 生成期望文件名，避免与内部哈希长度策略不一致
        fname = engine._make_key(t, session_id=b"v1_test_session", layer_id=0)  # type: ignore[attr-defined]
        full_path = os.path.join(temp_dir, fname)
        assert os.path.exists(full_path), f"expected saved file missing: {full_path}"

        # reconstruct engine and perform retrieval into zero buffers
        destroy_v1_engine()
        core = init_v1_engine(vcfg)

        zero_kvs: Dict[str, Any] = {
            f"layer_{i}": _make_zero_layer_kv(batch, seq, num_heads, head_dim)
            for i in range(num_layers)
        }
        core.register_kv_caches(zero_kvs)

        # second request: same tokens, expect full hit minus last token allocation
        req2 = _Request("req2", tokens)
        need, can_load = core.get_num_new_matched_tokens(req2, num_computed_tokens=0)
        # should_retrieve 直接调用底层构造的 model_input 验证命中
        from distributed_kv_manager.engine import should_retrieve as _should
        mi2 = core._build_model_input(SimpleNamespace(model=req2.model), req2, None)  # type: ignore
        hit_status = _should(mi2)
        # 同步实现：第二返回值为是否异步加载，恒为 False
        assert can_load is False
        assert need == max(0, len(tokens) - 1)
        # HIT 枚举值验证（RetrieveStatus.HIT）
        from distributed_kv_manager.engine import RetrieveStatus as _RS
        assert hit_status == _RS.HIT

        core.update_state_after_alloc(req2, None, need)
        num_sched2 = {req2.request_id: len(tokens)}
        fc2 = _ForwardContext(req2.model, zero_kvs)
        sched2 = SimpleNamespace(
            finished_req_ids=set(),
            scheduled_new_reqs=[req2],
            num_scheduled_tokens=num_sched2,
            scheduled_cached_reqs=[],
            forward_context=fc2,
        )
        core.build_connector_meta(sched2)
        core.start_load_kv(fc2)
        # 再次等待可能的读取后写入（虽然 retrieve 是同步）
        try:
            engine2 = core._engine  # type: ignore[attr-defined]
            for f in list(getattr(engine2, '_futures', []) or []):
                try:
                    f.result(timeout=10)
                except Exception:
                    pass
        except Exception:
            pass

        # verify caches have been filled to match original
        for i in range(num_layers):
            name = f"layer_{i}"
            assert torch.allclose(zero_kvs[name], orig_kvs[name], atol=1e-6)

    destroy_v1_engine()
