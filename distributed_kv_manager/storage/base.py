import abc
from typing import Optional, Tuple, Dict
import torch
import struct
from distributed_kv_manager.metadata.etcd import KVMetadata, METADATA_HEADER, METADATA_VERSION, HEADER_SIZE, METADATA_SIZE


class AbstractStorage(abc.ABC):
    """抽象存储接口，所有存储后端都需要实现这些方法"""
    
    @abc.abstractmethod
    def upload(self, file_path: str, data: bytes) -> bool:
        """上传数据到存储"""
        pass
    
    @abc.abstractmethod
    def download(self, file_path: str) -> Optional[bytes]:
        """从存储下载数据"""
        pass
    
    @abc.abstractmethod
    def exists(self, file_path: str) -> bool:
        """检查文件是否存在"""
        pass
    
    @abc.abstractmethod
    def pack_kv_data(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        input_tokens: torch.Tensor,
        roi: torch.Tensor,
    ) -> bytes:
        """打包KV数据为字节流"""
        pass
    
    @abc.abstractmethod
    def unpack_kv_data(
        self, data: bytes
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """从字节流解包KV数据"""
        pass

    # ---- 扩展：完整负载打包/信息提取 ----
    def pack_full_payload(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        input_tokens: torch.Tensor,
        roi: torch.Tensor,
        slot_mapping: Optional[torch.Tensor] = None,
        payload_meta: Optional[dict] = None,
    ) -> bytes:
        """打包带有额外信息的完整负载（默认实现基于 torch.save 的 dict）。

        字段说明：
        - k_cache/v_cache: [num_layers, seq_len, ...]
        - input_tokens: [seq_len]
        - roi: [seq_len] bool
        - slot_mapping: [seq_len]（可选）
        - payload_meta: dict（可选，包含 tokens_hash/num_layers/kv_dtype/kv_tail_shape/schema_version 等）
        """
        import io as _io
        payload: Dict[str, object] = {
            "k_cache": k_cache.cpu(),
            "v_cache": v_cache.cpu(),
            "input_tokens": input_tokens.cpu(),
            "roi": roi.cpu(),
        }
        if slot_mapping is not None:
            payload["slot_mapping"] = slot_mapping.cpu()
        if payload_meta is not None:
            payload["payload_meta"] = payload_meta
        _buf = _io.BytesIO()
        torch.save(payload, _buf)
        return _buf.getvalue()

    def extract_payload_info(self, data: bytes) -> dict:
        """从完整负载中提取扩展信息（默认解析 torch.save 的 dict）。

        返回字典包含可能的键：slot_mapping, payload_meta, input_tokens, roi。
        若数据格式不兼容或缺失，对应键不会出现。
        """
        import io as _io
        info: dict = {}
        try:
            _buf = _io.BytesIO(data)
            _loaded = torch.load(_buf, map_location="cpu")
            if isinstance(_loaded, dict):
                for k in ("slot_mapping", "payload_meta", "input_tokens", "roi"):
                    if k in _loaded:
                        info[k] = _loaded[k]
        except Exception:
            # 忽略解析失败，返回空信息
            return {}
        return info
        
    def extract_metadata_from_data(self, data: bytes) -> Optional[KVMetadata]:
        """
        从数据中提取嵌入的元数据
        """
        if len(data) < HEADER_SIZE:
            return None
            
        try:
            header, version = struct.unpack("<4sI", data[:HEADER_SIZE])
            if header != METADATA_HEADER or version != METADATA_VERSION:
                return None
                
            if len(data) < HEADER_SIZE + METADATA_SIZE:
                return None
                
            return KVMetadata.unpack(data[HEADER_SIZE:HEADER_SIZE + METADATA_SIZE])
        except Exception as e:
            print(f"Failed to extract metadata from data: {e}")
            return None