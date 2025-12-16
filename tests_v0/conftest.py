import os
import time
from types import SimpleNamespace
import pytest
import torch
from pathlib import Path

from distributed_kv_manager.metadata.etcd import KVMetadataManager, KVMetadata
from distributed_kv_manager.storage.v1.storage import create_v1_storage
from distributed_kv_manager.storage.factory import StorageFactory


@pytest.fixture(scope="session")
def tmp_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("dkv_v0_tests")
    return str(d)


@pytest.fixture(scope="session")
def meta_manager():
    # 使用不可用端点触发 MockEtcd3Client 回退
    mgr = KVMetadataManager(endpoints=["127.0.0.1:12345"], prefix="/kvmeta_v0_test")
    return mgr


@pytest.fixture(scope="session")
def local_storage(tmp_dir):
    cfg = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            storage_type="local",
            local_dir=os.path.join(tmp_dir, "kvcache"),
            use_v1=True,
            mem_cache_capacity_gb=0.1,
        )
    )
    return create_v1_storage(cfg)


def make_meta(file_path: str, expire_seconds: int = 0, status: int = 1) -> KVMetadata:
    now = int(time.time())
    return KVMetadata(
        session_id=b"test_session____",
        layer_id=0,
        token_idx="0",
        file_path=file_path,
        file_size=0,
        create_time=now,
        last_access=now,
        expire_time=int(expire_seconds),
        replica_locations=[b"", b"", b""],
        status=int(status),
        schema_version=1,
        ext_flags=0,
        ext_data=b"",
        ext_data_len=0,
    )


@pytest.fixture
def sample_kv():
    k = torch.randn(2, 8, 16)
    v = torch.randn(2, 8, 16)
    tokens = torch.randint(0, 100, (8,), dtype=torch.int64)
    roi = torch.ones(8, dtype=torch.bool)
    return k, v, tokens, roi


@pytest.fixture(scope="session")
def artifact_dir():
    d = Path(os.getcwd()) / "tests_v0" / "artifacts"
    os.makedirs(str(d), exist_ok=True)
    return str(d)


@pytest.fixture
def report(artifact_dir):
    def _write(name: str, lines: list[str]):
        p = Path(artifact_dir) / name
        with open(p, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"[report:{name}]")
        for ln in lines:
            print(ln)
        print(f"[/report:{name}]")
        return str(p)
    return _write

