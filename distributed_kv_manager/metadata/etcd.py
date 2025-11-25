import struct
import time
import threading
import os
import json
from dataclasses import dataclass
from typing import List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import etcd3
from etcd3 import Etcd3Client


# 元数据头部常量
METADATA_HEADER = b"KVMD"
METADATA_VERSION = 1
HEADER_SIZE = 8  # METADATA_HEADER (4 bytes) + METADATA_VERSION (4 bytes)
METADATA_SIZE = 320  # KVMetadata 序列化后的大小


@dataclass
class KVMetadata:
    """KV 元数据结构，定长二进制布局。"""

    session_id: bytes       # 16B
    layer_id: int           # 4B
    token_idx: str          # 例如 "0-1024"，存为 16B
    file_path: str          # 128B
    file_size: int          # 8B
    create_time: int        # 8B (epoch)
    last_access: int        # 8B
    expire_time: int        # 8B (秒，0 表示永不过期)
    replica_locations: List[bytes]  # 3 * 16B = 48B
    status: int             # 1B
    schema_version: int     # 2B
    ext_flags: int          # 4B
    ext_data: bytes         # 64B
    ext_data_len: int       # 4B
    reserved: bytes = b"\x00" * 1  # 1B 保留位（对齐用）

    def pack(self) -> bytes:
        """序列化为固定 320 字节。"""
        return struct.pack(
            "<16sI16s128sQQQQ48sBHI64sI1s",
            self.session_id.ljust(16, b"\x00"),
            self.layer_id,
            self.token_idx.encode("utf-8").ljust(16, b"\x00"),
            self.file_path.encode("utf-8").ljust(128, b"\x00"),
            self.file_size,
            self.create_time,
            self.last_access,
            self.expire_time,
            b"".join(self.replica_locations).ljust(48, b"\x00"),
            self.status,
            self.schema_version,
            self.ext_flags,
            self.ext_data.ljust(64, b"\x00"),
            self.ext_data_len,
            self.reserved,
        )

    @staticmethod
    def unpack(data: bytes) -> "KVMetadata":
        fields = struct.unpack("<16sI16s128sQQQQ48sBHI64sI1s", data)
        return KVMetadata(
            session_id=fields[0].rstrip(b"\x00"),
            layer_id=fields[1],
            token_idx=fields[2].decode("utf-8").rstrip("\x00"),
            file_path=fields[3].decode("utf-8").rstrip("\x00"),
            file_size=fields[4],
            create_time=fields[5],
            last_access=fields[6],
            expire_time=fields[7],
            replica_locations=[fields[8][i:i + 16] for i in range(0, 48, 16)],
            status=fields[9],
            schema_version=fields[10],
            ext_flags=fields[11],
            ext_data=fields[12].rstrip(b"\x00"),
            ext_data_len=fields[13],
            reserved=fields[14],
        )

    def to_dict(self) -> Dict:
        """元数据转为普通 dict，方便 JSON 持久化。"""
        return {
            "session_id": self.session_id.hex(),
            "layer_id": self.layer_id,
            "token_idx": self.token_idx,
            "file_path": self.file_path,
            "file_size": self.file_size,
            "create_time": self.create_time,
            "last_access": self.last_access,
            "expire_time": self.expire_time,
            "replica_locations": [loc.hex() for loc in self.replica_locations],
            "status": self.status,
            "schema_version": self.schema_version,
            "ext_flags": self.ext_flags,
            "ext_data": self.ext_data.hex(),
            "ext_data_len": self.ext_data_len,
        }

    @staticmethod
    def from_dict(data: Dict) -> "KVMetadata":
        """从 dict 还原元数据。"""
        return KVMetadata(
            session_id=bytes.fromhex(data["session_id"]),
            layer_id=data["layer_id"],
            token_idx=data["token_idx"],
            file_path=data["file_path"],
            file_size=data["file_size"],
            create_time=data["create_time"],
            last_access=data["last_access"],
            expire_time=data.get("expire_time", 0),
            replica_locations=[
                bytes.fromhex(loc) for loc in data["replica_locations"]
            ],
            status=data["status"],
            schema_version=data["schema_version"],
            ext_flags=data["ext_flags"],
            ext_data=bytes.fromhex(data["ext_data"]),
            ext_data_len=data["ext_data_len"],
        )

    def pack_with_embedding(self) -> bytes:
        """带文件头的嵌入格式，用于嵌入到 kv_*.pt 等文件中。"""
        header = struct.pack("<4sI", METADATA_HEADER, METADATA_VERSION)
        metadata_bytes = self.pack()
        return header + metadata_bytes

    @staticmethod
    def unpack_from_file(data: bytes) -> Optional["KVMetadata"]:
        """从文件数据中解析嵌入的元数据。"""
        if len(data) < HEADER_SIZE:
            return None

        header, version = struct.unpack("<4sI", data[:HEADER_SIZE])
        if header != METADATA_HEADER or version != METADATA_VERSION:
            return None

        if len(data) < HEADER_SIZE + METADATA_SIZE:
            return None

        return KVMetadata.unpack(data[HEADER_SIZE:HEADER_SIZE + METADATA_SIZE])

    def is_expired(self) -> bool:
        """检查当前元数据是否已过期。"""
        # expire_time 为 0 表示永不过期
        if self.expire_time == 0:
            return False

        current_time = int(time.time())
        return (current_time - self.last_access) >= self.expire_time


class EtcdConnectionPool:
    """简单的 etcd 连接池，负责端点探活和重连。"""

    def __init__(self, endpoints: list[str]):
        self.endpoints = endpoints
        self.clients: Dict[str, Optional[Etcd3Client]] = {}
        self.lock = threading.RLock()
        self._init_connections()

    def _init_connections(self) -> None:
        """初始化所有 etcd 连接。"""
        for ep in self.endpoints:
            try:
                host, port = ep.split(":")
                client = etcd3.client(host=host, port=int(port))
                client.status()
                self.clients[ep] = client
                print(f"[EtcdConnectionPool] Connected to etcd {ep}")
            except Exception as e:
                print(f"[EtcdConnectionPool] Failed to connect {ep}: {e}")
                self.clients[ep] = None

        if not any(self.clients.values()):
            raise RuntimeError("No available etcd endpoints!")

    def get_connection(self, preferred_endpoint: Optional[str] = None) -> Etcd3Client:
        """返回一个可用的 etcd 连接。"""
        with self.lock:
            # 1) 优先使用指定端点
            if preferred_endpoint and preferred_endpoint in self.clients:
                client = self.clients[preferred_endpoint]
                if client and self._check_connection(client):
                    return client

            # 2) 选择任意一个可用端点
            for endpoint, client in self.clients.items():
                if client and self._check_connection(client):
                    return client

            # 3) 尝试重连所有端点
            self._reconnect_all()
            for endpoint, client in self.clients.items():
                if client and self._check_connection(client):
                    return client

            raise RuntimeError("No available etcd connections!")

    def get_all_connections(self) -> list[Etcd3Client]:
        """返回所有可用的 etcd 连接。"""
        with self.lock:
            available_clients: list[Etcd3Client] = []
            for client in self.clients.values():
                if client and self._check_connection(client):
                    available_clients.append(client)
            return available_clients

    def _check_connection(self, client: Etcd3Client) -> bool:
        """简单探活。"""
        try:
            client.status()
            return True
        except Exception:
            return False

    def _reconnect_all(self) -> None:
        """重连所有当前不可用的端点。"""
        for endpoint in self.clients:
            if not self._check_connection(self.clients[endpoint]):
                try:
                    host, port = endpoint.split(":")
                    client = etcd3.client(host=host, port=int(port))
                    client.status()
                    self.clients[endpoint] = client
                    print(f"[EtcdConnectionPool] Reconnected to etcd {endpoint}")
                except Exception as e:
                    print(f"[EtcdConnectionPool] Failed to reconnect {endpoint}: {e}")
                    self.clients[endpoint] = None


class KVMetadataManager:
    """基于 etcd 的 KVMetadata 管理器，带简单 failover。"""

    def __init__(
        self,
        endpoints: list[str],
        prefix: str = "/kvmeta",
        max_workers: int = 5,
    ) -> None:
        """
        endpoints: ["ip1:2379", "ip2:2379", "ip3:2379"]
        使用连接池管理多个 etcd 节点，并提供简单的 failover。
        """
        self.endpoints = endpoints
        self.prefix = prefix
        self.connection_pool = EtcdConnectionPool(endpoints)
        self.max_workers = max_workers
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)

    def _execute_with_failover(self, operation, *args, **kwargs):
        """在可用的 etcd 端点上执行操作，失败自动切换。"""
        last_exception: Optional[Exception] = None

        for client in self.connection_pool.get_all_connections():
            try:
                return operation(client, *args, **kwargs)
            except Exception as e:
                last_exception = e
                print(f"Operation failed on {getattr(client, '_url', '?')}: {e}")
                continue

        raise last_exception or RuntimeError("Operation failed on all etcd nodes")

    def put_metadata(self, key: str, meta: KVMetadata, replicate: bool = True) -> None:
        """存储元数据，可选是否对所有节点进行复制。"""
        clean_key = key.lstrip("/")

        if not replicate:
            # 单节点写入
            def _put(client: Etcd3Client) -> None:
                full_key = f"{self.prefix}/{clean_key}"
                print(f"写入元数据: {full_key}")
                client.put(full_key, meta.pack())

            self._execute_with_failover(_put)
        else:
            # 多节点并行写入
            self._replicate_operation("put", clean_key, meta)

    def get_metadata_by_full_key(self, full_key: str) -> Optional[KVMetadata]:
        """通过完整 etcd key 获取元数据。"""

        def _get(client: Etcd3Client):
            print(f"尝试获取元数据，完整键: {full_key}")
            value, _ = client.get(full_key)
            return value

        value = self._execute_with_failover(_get)
        if value is None:
            return None
        return KVMetadata.unpack(value)

    def get_metadata(self, key: str) -> Optional[KVMetadata]:
        """按逻辑 key 获取元数据。"""

        def _get(client: Etcd3Client):
            clean_key = key.lstrip("/")
            full_key = f"{self.prefix}/{clean_key}"
            print(f"尝试获取元数据，键: {full_key}")
            value, _ = client.get(full_key)
            return value

        value = self._execute_with_failover(_get)
        if value is None:
            return None
        return KVMetadata.unpack(value)

    def update_access_time(self, key: str, replicate: bool = True) -> None:
        """更新 last_access，并可选写回所有节点。"""
        meta = self.get_metadata(key)
        if meta:
            meta.last_access = int(time.time())
            self.put_metadata(key, meta, replicate)

    def delete_metadata(self, key: str, replicate: bool = True) -> None:
        """删除元数据。"""
        if not replicate:

            def _delete(client: Etcd3Client) -> None:
                client.delete(f"{self.prefix}/{key}")

            self._execute_with_failover(_delete)
        else:
            self._replicate_operation("delete", key)

    def _replicate_operation(
        self,
        operation: str,
        key: str,
        meta: Optional[KVMetadata] = None,
    ) -> None:
        """在所有 etcd 节点上并行执行 put/delete 操作。"""
        clients = self.connection_pool.get_all_connections()
        futures = []

        for client in clients:
            if operation == "put" and meta is not None:
                futures.append(
                    self.thread_pool.submit(
                        self._safe_put,
                        client,
                        f"{self.prefix}/{key}",
                        meta.pack(),
                    )
                )
            elif operation == "delete":
                futures.append(
                    self.thread_pool.submit(
                        self._safe_delete,
                        client,
                        f"{self.prefix}/{key}",
                    )
                )

        failed_operations = []
        for i, future in enumerate(as_completed(futures)):
            try:
                future.result()
            except Exception as e:
                failed_operations.append((clients[i], e))

        if failed_operations:
            print(
                f"Failed to replicate operation to {len(failed_operations)} nodes:"
            )
            for client, error in failed_operations:
                print(f"  {getattr(client, '_url', '?')}: {error}")

    def _safe_put(self, client: Etcd3Client, key: str, value: bytes) -> None:
        try:
            client.put(key, value)
        except Exception as e:
            print(f"Failed to put {key} to {getattr(client, '_url', '?')}: {e}")
            raise

    def _safe_delete(self, client: Etcd3Client, key: str) -> None:
        try:
            client.delete(key)
        except Exception as e:
            print(f"Failed to delete {key} from {getattr(client, '_url', '?')}: {e}")
            raise

    def watch_metadata(self, key: str, callback) -> None:
        """监听某个 key 的变更。"""
        client = self.connection_pool.get_connection()
        events_iterator, cancel = client.watch(f"{self.prefix}/{key}")
        for event in events_iterator:
            callback(event)

    def recover_metadata_from_storage(
        self, storage_path: str
    ) -> Dict[str, KVMetadata]:
        """
        从本地存储中扫描并提取嵌入元数据。
        返回: {file_path: KVMetadata}
        """
        recovered_metadata: Dict[str, KVMetadata] = {}

        for root, _, files in os.walk(storage_path):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, storage_path)

                metadata = self._extract_embedded_metadata_from_file(file_path)
                if metadata:
                    metadata.file_path = relative_path
                    recovered_metadata[relative_path] = metadata

        return recovered_metadata

    def _extract_embedded_metadata_from_file(
        self,
        file_path: str,
    ) -> Optional[KVMetadata]:
        """尝试从文件前缀解析嵌入元数据。"""
        try:
            with open(file_path, "rb") as f:
                data = f.read(HEADER_SIZE + METADATA_SIZE)

            metadata = KVMetadata.unpack_from_file(data)
            if metadata:
                return metadata
            return None
        except Exception as e:
            print(f"Failed to extract embedded metadata from {file_path}: {e}")
            return None

    def write_metadata_to_etcd(self, metadata_dict: Dict[str, KVMetadata]) -> None:
        """将本地恢复的元数据写回 etcd。"""
        for file_path, metadata in metadata_dict.items():
            try:
                key = file_path
                self.put_metadata(key, metadata)
                print(f"Successfully wrote metadata for {file_path} to etcd")
            except Exception as e:
                print(f"Failed to write metadata for {file_path} to etcd: {e}")

    def scan_all_metadata_keys(self) -> List[str]:
        """
        扫描当前前缀下所有元数据 key。

        Returns:
            完整 etcd key 列表（包含前缀）。
        """

        def _scan(client: Etcd3Client) -> List[str]:
            keys: List[str] = []
            try:
                for _value, meta in client.get_prefix(self.prefix + "/"):
                    key = meta.key.decode("utf-8")
                    if key.startswith(self.prefix):
                        keys.append(key)
                        print(f"发现元数据 key: {key}")
            except Exception as e:
                print(f"Failed to scan metadata keys: {e}")
                import traceback

                traceback.print_exc()
            return keys

        try:
            return self._execute_with_failover(_scan)
        except Exception as e:
            print(f"Failed to scan metadata keys with failover: {e}")
            import traceback

            traceback.print_exc()
            return []

