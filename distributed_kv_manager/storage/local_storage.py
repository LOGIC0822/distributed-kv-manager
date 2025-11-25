# distributed_kv_manager/storage/local_storage.py
import os
import io
import torch
import logging
from typing import Optional, Tuple

from .base import AbstractStorage
from distributed_kv_manager.storage.v0.layout import (
    pack_kv_local_v0,
    unpack_kv_local_v0,
)

logger = logging.getLogger("LocalStorage")


class LocalStorage(AbstractStorage):
    """本地文件系统存储实现"""

    def __init__(self, local_dir: str):
        self.local_dir = local_dir
        os.makedirs(self.local_dir, exist_ok=True)
        logger.info(f"Initialized LocalStorage with directory: {self.local_dir}")

    def upload(self, file_path: str, data: bytes) -> bool:
        """上传数据到本地文件系统"""
        try:
            full_path = os.path.join(self.local_dir, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            with open(full_path, 'wb') as f:
                f.write(data)
            # 追加完整路径长度与存在性校验，便于后续排查截断或挂载问题
            exists_flag = os.path.exists(full_path)
            logger.debug(
                "Successfully uploaded data to %s (exists=%s, path_len=%d)",
                full_path,
                exists_flag,
                len(full_path),
            )
            return True
        except Exception as e:
            logger.error(f"Failed to upload data to {file_path}: {e}")
            return False

    def download(self, file_path: str) -> Optional[bytes]:
        """从本地文件系统下载数据"""
        try:
            full_path = os.path.join(self.local_dir, file_path)
            # 若路径超长，分段打印，避免某些终端截断末尾
            if len(full_path) > 160:
                segmented = [full_path[i:i + 120] for i in range(0, len(full_path), 120)]
                logger.debug("尝试下载文件(分段):\n%s", "\n".join(segmented))
            else:
                logger.debug(f"尝试下载文件: {full_path}")

            if not os.path.exists(full_path):
                logger.warning(
                    "文件不存在: %s (path_len=%d, local_dir=%s)",
                    full_path,
                    len(full_path),
                    self.local_dir,
                )
                return None

            with open(full_path, 'rb') as f:
                data = f.read()

            logger.debug(f"成功下载文件 {full_path}，数据大小: {len(data)} 字节")
            return data
        except Exception as e:
            logger.error(f"下载文件 {file_path} 失败: {e}")
            return None

    def exists(self, file_path: str) -> bool:
        """检查文件是否存在"""
        full_path = os.path.join(self.local_dir, file_path)
        return os.path.exists(full_path)

    def pack_kv_data(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        input_tokens: torch.Tensor,
        roi: torch.Tensor,
    ) -> bytes:
        """打包KV数据为字节流（v0 layout），通过 storage.v0.layout 统一实现。"""
        logger.debug(
            "打包KV数据(v0): k_cache形状=%s, v_cache形状=%s",
            k_cache.shape,
            v_cache.shape,
        )
        return pack_kv_local_v0(k_cache, v_cache, input_tokens, roi)

    def unpack_kv_data(
        self, data: bytes
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """从字节流解包KV数据（v0 layout），通过 storage.v0.layout 统一实现。"""
        try:
            logger.debug(f"开始解包KV数据(v0)，数据大小: {len(data)} 字节")
        except Exception:
            pass
        k_cache, v_cache = unpack_kv_local_v0(data)
        if k_cache is not None and v_cache is not None:
            logger.debug(
                "成功解包KV数据(v0): k_cache形状=%s, v_cache形状=%s",
                k_cache.shape,
                v_cache.shape,
            )
        return k_cache, v_cache

    def delete(self, file_path: str) -> bool:
        """删除文件（可选方法）"""
        try:
            full_path = os.path.join(self.local_dir, file_path)
            if os.path.exists(full_path):
                os.remove(full_path)
                logger.debug(f"Successfully deleted file: {full_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {e}")
            return False

    def list_files(self, prefix: str = "") -> list:
        """列出所有文件（可选方法）"""
        try:
            full_dir = os.path.join(self.local_dir, prefix)
            if not os.path.exists(full_dir):
                return []

            files = []
            for root, _, filenames in os.walk(full_dir):
                for filename in filenames:
                    rel_path = os.path.relpath(os.path.join(root, filename), self.local_dir)
                    files.append(rel_path)
            return files
        except Exception as e:
            logger.error(f"Failed to list files with prefix {prefix}: {e}")
            return []
