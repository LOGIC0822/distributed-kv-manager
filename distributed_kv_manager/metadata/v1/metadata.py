from __future__ import annotations

import time
from typing import List, Optional

from distributed_kv_manager.metadata.etcd import KVMetadata, KVMetadataManager


class V1MetadataClient:
    """v1 外部 KV 的元数据访问封装。

    只依赖 KVMetadataManager，不使用本地三层缓存，避免与外部清理脚本状态不一致。

    提供：
    - hash_exists: 判断某个 hash 目录是否存在有效 KV 记录；
    - mark_hash_stored: 在完成一次 STORE 后写入聚合元数据。
    """

    def __init__(self, manager: KVMetadataManager, default_expire: int = 0) -> None:
        self._manager = manager
        self._default_expire = int(default_expire or 0)

    def get_metadata(
        self,
        key: str,
        layer_id: Optional[int] = None,
        session_id: Optional[bytes] = None,
    ) -> Optional[KVMetadata]:
        """直接从 etcd 获取一条元数据（不经过本地缓存）。"""
        _ = layer_id, session_id
        return self._manager.get_metadata(key)

    def put_metadata(self, meta: KVMetadata) -> None:
        """写入元数据到 etcd。"""
        self._manager.put_metadata(meta.file_path, meta, replicate=True)

    def update_access_time(self, key: str) -> None:
        """更新某条记录的 last_access 并同步写回 etcd。"""
        meta = self._manager.get_metadata(key)
        if meta is None:
            return
        meta.last_access = int(time.time())
        self._manager.put_metadata(key, meta, replicate=True)

    # ---- v1 专用 helper：按 hash 目录维护存在性 / 聚合记录 ----

    def hash_exists(self, folder_abs: str) -> bool:
        """判断某个 hash 目录是否存在有效 KV（status==1 且未过期）。"""
        meta = self.get_metadata(key=folder_abs)
        if meta is None or getattr(meta, "status", 0) != 1:
            return False
        if meta.is_expired():
            return False
        # 刷新访问时间（同步写回）
        self.update_access_time(folder_abs)
        return True

    def mark_hash_stored(
        self,
        folder_abs: str,
        num_tokens: int,
        file_size: int,
    ) -> None:
        """为某个 hash 目录写入一条聚合元数据记录。"""
        now = int(time.time())
        expire = int(self._default_expire or 0)
        session_id = b"v1_external_kv__"  # 16B 占位
        layer_id = 0  # v1 聚合 entry 统一用 0 作为占位
        token_idx = str(int(num_tokens))
        meta = KVMetadata(
            session_id=session_id,
            layer_id=layer_id,
            token_idx=token_idx,
            file_path=folder_abs,
            file_size=int(file_size),
            create_time=now,
            last_access=now,
            expire_time=expire,
            replica_locations=[b"", b"", b""],
            status=1,
            schema_version=1,
            ext_flags=0,
            ext_data=b"",
            ext_data_len=0,
        )
        self.put_metadata(meta)

    # ---- 通用扫描接口：目前主要用于调试 ----

    def scan_by_session_layer(
        self,
        session_id: Optional[bytes],
        layer_id: Optional[int],
    ) -> List[KVMetadata]:
        """按 session_id + layer_id 扫描已提交的元数据（可能为空）。"""
        out: List[KVMetadata] = []
        try:
            full_keys = self._manager.scan_all_metadata_keys()
        except Exception:
            full_keys = []

        prefix = getattr(self._manager, "prefix", "/kvmeta")
        sid = session_id
        lid = layer_id

        for fk in full_keys:
            try:
                rel = fk
                if isinstance(rel, str) and rel.startswith(prefix + "/"):
                    rel = rel[len(prefix) + 1 :]
                m = self._manager.get_metadata_by_full_key(fk)
                if m is None or getattr(m, "status", None) != 1:
                    continue
                if sid is not None and getattr(m, "session_id", None) != sid:
                    continue
                if lid is not None and getattr(m, "layer_id", None) != lid:
                    continue
                out.append(m)
            except Exception:
                continue
        return out

    def stop(self) -> None:
        """v1 当前没有内部后台线程，保留接口仅为对齐。"""
        return

