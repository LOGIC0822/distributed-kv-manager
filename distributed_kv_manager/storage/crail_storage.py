import os
import io
import subprocess
import torch
import logging
from typing import Optional, Tuple

from .base import AbstractStorage
from distributed_kv_manager.storage.v0.layout import (
    pack_kv_local_v0,
    unpack_kv_local_v0,
)

logger = logging.getLogger("CrailStorage")


class CrailStorage(AbstractStorage):
    """封装Crail文件操作，实现AbstractStorage接口"""

    def __init__(self, crail_dir: str):
        self.crail_dir = crail_dir
        os.makedirs(self.crail_dir, exist_ok=True)

    def upload(self, file_path: str, data: bytes) -> bool:
        """上传数据到Crail"""
        jar_path = os.environ.get("CRAIL_KVCACHE_JAR", "crail-kvcache-client.jar")
        conf_dir = os.environ.get("CRAIL_CONF_DIR", "/root/crail/conf")
        cmd = [
            "java", "-Djava.library.path=/root/crail/lib",
            f"-Dcrail.conf.dir={conf_dir}",
            "-cp", f"{jar_path}:{conf_dir}:/root/crail/jars/*",
            "com.example.CrailKVCacheManager", "upload-stream", file_path,
        ]
        try:
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            _, stderr = process.communicate(input=data, timeout=30)
            if process.returncode != 0:
                logger.error(f"Crail upload failed: {stderr.decode()}")
                return False
            return True
        except subprocess.TimeoutExpired:
            process.kill()
            logger.error("Crail upload timed out")
            return False

    def download(self, file_path: str) -> Optional[bytes]:
        """从Crail下载数据"""
        jar_path = os.environ.get("CRAIL_KVCACHE_JAR", "crail-kvcache-client.jar")
        conf_dir = os.environ.get("CRAIL_CONF_DIR", "/root/crail/conf")
        cmd = [
            "java", "-Djava.library.path=/root/crail/lib",
            f"-Dcrail.conf.dir={conf_dir}",
            "-cp", f"{jar_path}:{conf_dir}:/root/crail/jars/*",
            "com.example.CrailKVCacheManager", "download-stream", file_path,
        ]
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            chunks: list[bytes] = []
            stdout_pipe = process.stdout
            if stdout_pipe is None:
                process.kill()
                logger.error("Crail download failed: stdout is None")
                return None
            while True:
                chunk = stdout_pipe.read(16 * 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            stderr_pipe = process.stderr
            stderr = b""
            if stderr_pipe is not None:
                stderr = stderr_pipe.read()
            process.wait()
            if process.returncode != 0:
                logger.error(f"Crail download failed: {stderr.decode()}")
                return None
            return b"".join(chunks)
        except Exception as e:
            logger.error(f"Crail download error: {e}")
            return None

    def exists(self, file_path: str) -> bool:
        """检查文件是否存在"""
        jar_path = os.environ.get("CRAIL_KVCACHE_JAR", "crail-kvcache-client.jar")
        conf_dir = os.environ.get("CRAIL_CONF_DIR", "/root/crail/conf")
        cmd = [
            "java", "-Djava.library.path=/root/crail/lib",
            f"-Dcrail.conf.dir={conf_dir}",
            "-cp", f"{jar_path}:{conf_dir}:/root/crail/jars/*",
            "com.example.CrailKVCacheManager", "exists", file_path,
        ]
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate(timeout=10)
            return process.returncode == 0
        except subprocess.TimeoutExpired:
            return False

    def pack_kv_data(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        input_tokens: torch.Tensor,
        roi: torch.Tensor,
    ) -> bytes:
        """打包KV数据为字节流（v0 layout）"""
        return pack_kv_local_v0(k_cache, v_cache, input_tokens, roi)

    def unpack_kv_data(
        self, data: bytes
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """从字节流解包KV数据（v0 layout）"""
        k_cache, v_cache = unpack_kv_local_v0(data)
        if k_cache is None or v_cache is None:
            logger.error("Failed to unpack KV data via v0 layout")
        return k_cache, v_cache
