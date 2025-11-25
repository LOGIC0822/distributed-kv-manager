from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import safetensors
import torch

from .base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from distributed_kv_manager.config_loader import load_config_from_json
from distributed_kv_manager.metadata.etcd import KVMetadataManager
from distributed_kv_manager.metadata.v1.metadata import V1MetadataClient
from distributed_kv_manager.storage.v1.storage import V1Storage, create_v1_storage

try:
    from vllm.logger import init_logger as _vllm_init_logger  # type: ignore

    logger = _vllm_init_logger(__name__)
except Exception:  # pragma: no cover
    logger = logging.getLogger(__name__)


# ---------------- Data structures ----------------


@dataclass
class _ReqMeta:
    token_ids: torch.Tensor
    slot_mapping: torch.Tensor
    is_store: bool
    mm_hashes: list[str]

    @staticmethod
    def make_meta(
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
        is_store: bool,
        mm_hashes: list[str],
    ) -> "_ReqMeta":
        valid_num_tokens = _align_to_block_size(len(token_ids), block_size)
        token_ids_tensor = torch.tensor(token_ids)[:valid_num_tokens]
        block_ids_tensor = torch.tensor(block_ids)
        num_blocks = int(block_ids_tensor.shape[0])
        block_offsets = torch.arange(0, block_size)
        slot_mapping = (
            block_offsets.reshape((1, block_size))
            + block_ids_tensor.reshape((num_blocks, 1)) * block_size
        )
        slot_mapping = slot_mapping.flatten()[:valid_num_tokens]
        return _ReqMeta(
            token_ids=token_ids_tensor,
            slot_mapping=slot_mapping,
            is_store=is_store,
            mm_hashes=mm_hashes,
        )


@dataclass
class _V1Metadata(KVConnectorMetadata):
    requests: list[_ReqMeta] = field(default_factory=list)

    def add_request(
        self,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
        is_store: bool,
        mm_hashes: list[str],
    ) -> None:
        self.requests.append(
            _ReqMeta.make_meta(token_ids, block_ids, block_size, is_store, mm_hashes)
        )


# ---------------- Engine (connector) implementation ----------------


class V1KVEngineImpl(KVConnectorBase_V1):
    """V1 KV connector for vLLM v1 external cache.
    """

    def __init__(self, vllm_config: Any, role: KVConnectorRole):
        super().__init__(vllm_config, role)
        self.vllm_config = vllm_config
        try:
            bs = getattr(getattr(vllm_config, "cache_config", None), "block_size", 16)
        except Exception:
            bs = 16
        self._block_size = int(bs)
        self._requests_need_load: dict[str, Any] = {}
        self._folders_with_meta: set[str] = set()

        self._storage_path = self._resolve_dkv_path(vllm_config) or "/tmp/kvcache/v1"
        logger.info("[v1_engine] dkv_storage_path=%s", self._storage_path)

        # v1 storage backend (A1) + metadata (B1)
        self._v1_storage: Optional[V1Storage]
        self._v1_meta: Optional[V1MetadataClient] = None
        self._v1_kv_expire_time: int = 0
        try:
            from types import SimpleNamespace

            # 优先用 config.json 里的 kv_transfer_config
            try:
                cfg_json = load_config_from_json()
                kvt_json = getattr(cfg_json, "kv_transfer_config", None)
            except Exception:
                kvt_json = None

            if kvt_json is not None:
                if getattr(kvt_json, "storage_dir", None) is None:
                    setattr(kvt_json, "storage_dir", self._storage_path)
                if getattr(kvt_json, "local_dir", None) is None:
                    setattr(kvt_json, "local_dir", self._storage_path)
                effective_kvt = kvt_json
            else:
                # 退回到 vLLM 自带的 kv_transfer_config
                kvt_cfg = getattr(vllm_config, "kv_transfer_config", None)
                if kvt_cfg is not None:
                    if getattr(kvt_cfg, "storage_dir", None) is None:
                        setattr(kvt_cfg, "storage_dir", self._storage_path)
                    if getattr(kvt_cfg, "local_dir", None) is None:
                        setattr(kvt_cfg, "local_dir", self._storage_path)
                    effective_kvt = kvt_cfg
                else:
                    # 再退回：完全没配时使用 local
                    effective_kvt = SimpleNamespace(
                        storage_type="local",
                        storage_dir=self._storage_path,
                        local_dir=self._storage_path,
                    )

            self._kv_transfer_config = effective_kvt
            cfg = SimpleNamespace(kv_transfer_config=effective_kvt)
            self._v1_storage = create_v1_storage(cfg)
            logger.info(
                "[v1_engine] v1_storage backend initialized for %s (storage_type=%s)",
                self._storage_path,
                getattr(effective_kvt, "storage_type", "local"),
            )

            # 初始化 v1 元数据 (etcd). 如果失败则视为关闭 External Cache
            try:
                endpoints = getattr(
                    self._kv_transfer_config, "etcd_endpoints", ["127.0.0.1:2379"]
                )
                prefix = "/kvmeta_v1"
                self._v1_kv_expire_time = int(
                    getattr(self._kv_transfer_config, "kv_expire_time", 0)
                )
                meta_manager = KVMetadataManager(endpoints=endpoints, prefix=prefix)
                self._v1_meta = V1MetadataClient(
                    meta_manager, default_expire=self._v1_kv_expire_time
                )
                logger.info(
                    "[v1_engine] v1 metadata initialized: endpoints=%s prefix=%s expire=%s",
                    endpoints,
                    prefix,
                    self._v1_kv_expire_time,
                )
                # 可选：基于 last_access 预取部分热点 hash 到 DRAM 缓存
                try:
                    enable_prefetch = bool(
                        getattr(self._kv_transfer_config, "enable_prefetch", False)
                    )
                    top_k = int(
                        getattr(self._kv_transfer_config, "v1_prefetch_top_k", 0) or 0
                    )
                except Exception:
                    enable_prefetch = False
                    top_k = 0
                if enable_prefetch and top_k > 0:
                    import threading

                    t = threading.Thread(
                        target=self._run_initial_prefetch,
                        args=(top_k,),
                        daemon=True,
                    )
                    t.start()
                    logger.info(
                        "[v1_engine] scheduled initial v1 prefetch for top_k=%d hashes",
                        top_k,
                    )
            except Exception:
                self._v1_meta = None
                self._v1_kv_expire_time = 0
        except Exception:
            # 存储初始化失败时，只保留 debug 路径
            self._v1_storage = None

    # ---------------- Worker-side ----------------

    def start_load_kv(self, forward_context: Any, **kwargs) -> None:
        metadata: _V1Metadata = self._get_connector_metadata()  # type: ignore
        if metadata is None or not isinstance(metadata, _V1Metadata):
            logger.warning("[v1_engine] start_load_kv: connector metadata is None")
            return

        attn_metadata = getattr(forward_context, "attn_metadata", None)
        if attn_metadata is None:
            logger.warning("[v1_engine] start_load_kv: attn_metadata is None")

        layers = getattr(forward_context, "no_compile_layers", {})
        for req in metadata.requests:
            if req.is_store:
                continue
            logger.info(
                "[v1_engine] inject %d tokens into paged KV",
                int(req.slot_mapping.numel()),
            )
            for layer_name, layer in layers.items():
                kv_cache_attr = getattr(layer, "kv_cache", None)
                if kv_cache_attr is None:
                    continue
                kv_cache_layer = kv_cache_attr[
                    getattr(forward_context, "virtual_engine", 0)
                ]
                kv_cache = None

                # 通过 v1_storage 从外部存储读取 per-layer safetensors
                folder_abs = self._generate_foldername_debug(
                    req.token_ids, req.mm_hashes, create_folder=False
                )
                folder_rel = os.path.basename(folder_abs)
                rel_path = os.path.join(folder_rel, f"{layer_name}.safetensors")
                logger.info(
                    "[v1_engine] load path: layer=%s, folder_abs=%s, rel_path=%s",
                    layer_name,
                    folder_abs,
                    rel_path,
                )
                if self._v1_storage is not None:
                    try:
                        raw = self._v1_storage.download(rel_path)
                        if raw is not None:
                            kv_cache = safetensors.torch.load(raw)["kv_cache"].cuda()
                    except Exception:
                        kv_cache = None

                # 兜底：从本地 debug 文件路径读取
                if kv_cache is None:
                    filename = self._generate_filename_debug(
                        layer_name, req.token_ids, req.mm_hashes
                    )
                    try:
                        kv_cache = safetensors.torch.load_file(filename)[
                            "kv_cache"
                        ].cuda()
                    except Exception:
                        continue

                _inject_kv_into_layer(
                    kv_cache_layer,
                    kv_cache,
                    req.slot_mapping,
                    attn_metadata,
                )

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: Any,
        **kwargs,
    ) -> None:
        metadata: _V1Metadata = self._get_connector_metadata()  # type: ignore
        if metadata is None or not isinstance(metadata, _V1Metadata):
            return

        for req in metadata.requests:
            if not req.is_store:
                continue

            kv_cache = _extract_kv_from_layer(
                kv_layer, req.slot_mapping, attn_metadata
            )

            folder_abs = self._generate_foldername_debug(
                req.token_ids, req.mm_hashes, create_folder=False
            )
            folder_rel = os.path.basename(folder_abs)
            rel_path = os.path.join(folder_rel, f"{layer_name}.safetensors")
            logger.info(
                "[v1_engine] save path: layer=%s, folder_abs=%s, rel_path=%s, is_store=%s",
                layer_name,
                folder_abs,
                rel_path,
                req.is_store,
            )

            try:
                data = safetensors.torch.save({"kv_cache": kv_cache.detach().cpu()})
            except Exception:
                data = None

            if self._v1_storage is not None and data is not None:
                ok = False
                try:
                    ok = self._v1_storage.upload(rel_path, data)
                except Exception:
                    ok = False

                # 写入成功时，按 hash 目录聚合写一条 v1 元数据记录
                if ok and self._v1_meta is not None and folder_abs not in self._folders_with_meta:
                    try:
                        num_tokens = int(req.slot_mapping.numel())
                    except Exception:
                        num_tokens = 0
                    try:
                        file_size = len(data)
                    except Exception:
                        file_size = 0
                    try:
                        self._v1_meta.mark_hash_stored(
                            folder_abs,
                            num_tokens=num_tokens,
                            file_size=file_size,
                        )
                        self._folders_with_meta.add(folder_abs)
                    except Exception:
                        pass

                if not ok:
                    filename = self._generate_filename_debug(
                        layer_name, req.token_ids, req.mm_hashes
                    )
                    try:
                        safetensors.torch.save_file(
                            {"kv_cache": kv_cache.detach().cpu()}, filename
                        )
                    except Exception:
                        pass
            else:
                filename = self._generate_filename_debug(
                    layer_name, req.token_ids, req.mm_hashes
                )
                try:
                    safetensors.torch.save_file(
                        {"kv_cache": kv_cache.detach().cpu()}, filename
                    )
                except Exception:
                    pass

    def wait_for_save(self):
        return

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        return None, None

    # ---------------- Scheduler-side ----------------

    def get_num_new_matched_tokens(
        self,
        request: "Any",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        # 只根据 etcd/v1 元数据判断 External Cache 是否命中
        try:
            token_ids = list(getattr(request, "prompt_token_ids", []) or [])
            mm_hashes = [
                getattr(f, "identifier", "")
                for f in (getattr(request, "mm_features", []) or [])
            ]
            num_tokens_to_check = _align_to_block_size(
                len(token_ids) - 1, self._block_size
            )
            folder = self._generate_foldername_debug(
                torch.tensor(token_ids)[:num_tokens_to_check],
                mm_hashes,
                create_folder=False,
            )
            req_id = getattr(request, "request_id", getattr(request, "req_id", "?"))

            # 如果 v1 元数据不可用，视为无 External Cache：直接 miss，不做磁盘存在性检查
            if self._v1_meta is None:
                logger.info(
                    "[v1_engine] get_num_new_matched_tokens: req=%s len=%d aligned=%d folder=%s meta_enabled=%s meta_exists=%s num_computed=%d",
                    req_id,
                    len(token_ids),
                    num_tokens_to_check,
                    folder,
                    False,
                    False,
                    int(num_computed_tokens),
                )
                return 0, False

            try:
                meta_exists = self._v1_meta.hash_exists(folder)
            except Exception:
                meta_exists = False

            logger.info(
                "[v1_engine] get_num_new_matched_tokens: req=%s len=%d aligned=%d folder=%s meta_enabled=%s meta_exists=%s num_computed=%d",
                req_id,
                len(token_ids),
                num_tokens_to_check,
                folder,
                True,
                bool(meta_exists),
                int(num_computed_tokens),
            )
            if not meta_exists:
                return 0, False

            logger.info("External Cache Hit!")
            return max(0, num_tokens_to_check - int(num_computed_tokens)), False
        except Exception:
            return 0, False

    def update_state_after_alloc(
        self,
        request: "Any",
        blocks: "Any",
        num_external_tokens: int,
    ):
        if int(num_external_tokens or 0) > 0:
            self._requests_need_load[getattr(request, "request_id", "")] = request
        return

    def build_connector_meta(self, scheduler_output: Any) -> KVConnectorMetadata:
        meta = _V1Metadata()

        # 新调度请求
        for new_req in getattr(scheduler_output, "scheduled_new_reqs", []) or []:
            token_ids = list(getattr(new_req, "prompt_token_ids", []) or [])
            mm_hashes = [
                getattr(f, "identifier", "")
                for f in (getattr(new_req, "mm_features", []) or [])
            ]
            if getattr(new_req, "req_id", None) in self._requests_need_load:
                logger.info(
                    "[v1_engine] build_connector_meta: new_req=%s -> LOAD",
                    getattr(new_req, "req_id", None),
                )
                meta.add_request(
                    token_ids=token_ids,
                    block_ids=(getattr(new_req, "block_ids", [[0]]) or [[0]])[0],
                    block_size=self._block_size,
                    is_store=False,
                    mm_hashes=mm_hashes,
                )
            else:
                num_tokens_to_check = _align_to_block_size(
                    len(token_ids) - 1, self._block_size
                )
                folder = self._generate_foldername_debug(
                    torch.tensor(token_ids)[:num_tokens_to_check],
                    mm_hashes,
                    create_folder=False,
                )

                # 若元数据不可用，则一律视为需要 STORE（不做磁盘存在性判断）
                if self._v1_meta is None:
                    logger.info(
                        "[v1_engine] build_connector_meta: new_req=%s len=%d aligned=%d folder=%s meta_enabled=%s meta_exists=%s -> STORE",
                        getattr(new_req, "req_id", None),
                        len(token_ids),
                        num_tokens_to_check,
                        folder,
                        False,
                        False,
                    )
                    meta.add_request(
                        token_ids=token_ids,
                        block_ids=(getattr(new_req, "block_ids", [[0]]) or [[0]])[
                            0
                        ],
                        block_size=self._block_size,
                        is_store=True,
                        mm_hashes=mm_hashes,
                    )
                else:
                    try:
                        meta_exists = self._v1_meta.hash_exists(folder)
                    except Exception:
                        meta_exists = False
                    logger.info(
                        "[v1_engine] build_connector_meta: new_req=%s len=%d aligned=%d folder=%s meta_enabled=%s meta_exists=%s",
                        getattr(new_req, "req_id", None),
                        len(token_ids),
                        num_tokens_to_check,
                        folder,
                        True,
                        bool(meta_exists),
                    )
                    if not meta_exists:
                        logger.info(
                            "[v1_engine] build_connector_meta: new_req=%s -> STORE",
                            getattr(new_req, "req_id", None),
                        )
                        meta.add_request(
                            token_ids=token_ids,
                            block_ids=(
                                getattr(new_req, "block_ids", [[0]]) or [[0]]
                            )[0],
                            block_size=self._block_size,
                            is_store=True,
                            mm_hashes=mm_hashes,
                        )

        # 已缓存请求（可选）
        cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
        try:
            req_ids = list(getattr(cached_reqs, "req_ids", []) or [])
            resumed = list(
                getattr(cached_reqs, "resumed_from_preemption", []) or []
            )
            new_block_ids = list(getattr(cached_reqs, "new_block_ids", []) or [])
            num_computed = list(
                getattr(cached_reqs, "num_computed_tokens", []) or []
            )
        except Exception:
            req_ids, resumed, new_block_ids, num_computed = [], [], [], []

        for i, req_id in enumerate(req_ids):
            if not (resumed[i] if i < len(resumed) else False):
                break
            if req_id in self._requests_need_load:
                request = self._requests_need_load[req_id]
                total_tokens = int(
                    (num_computed[i] if i < len(num_computed) else 0)
                    + getattr(
                        scheduler_output,
                        "num_scheduled_tokens",
                        {},
                    ).get(req_id, 0)
                )
                token_ids = list(
                    getattr(request, "all_token_ids", [])[:total_tokens]
                )
                nbids = new_block_ids[i] if i < len(new_block_ids) else [[0]]
                block_ids = nbids[0] if nbids else [0]
                mm_hashes = [
                    getattr(f, "identifier", "")
                    for f in (getattr(request, "mm_features", []) or [])
                ]
                meta.add_request(
                    token_ids=token_ids,
                    block_ids=block_ids,
                    block_size=self._block_size,
                    is_store=False,
                    mm_hashes=mm_hashes,
                )

        self._requests_need_load.clear()
        return meta

    # ---------------- helpers ----------------

    def _resolve_dkv_path(self, vllm_config: Any) -> Optional[str]:
        """从 kv_transfer_config / config.json 解析 dkv_storage_path."""
        # 1) 先看 vllm_config.kv_transfer_config 是否显式提供 dkv_storage_path
        try:
            kvt = getattr(vllm_config, "kv_transfer_config", None)
            if kvt is not None:
                path = getattr(kvt, "dkv_storage_path", None)
                if path:
                    return str(path)
                extra = getattr(kvt, "extra_config", None)
                if isinstance(extra, dict):
                    p = extra.get("dkv_storage_path")
                    if p:
                        return str(p)
        except Exception:
            pass

        # 2) 否则从 config.json.kv_transfer_config 读取
        try:
            cfg = load_config_from_json()
            kvt_json = getattr(cfg, "kv_transfer_config", None)
            if kvt_json is not None:
                path = getattr(kvt_json, "dkv_storage_path", None)
                if path:
                    return str(path)
                path2 = getattr(kvt_json, "storage_dir", None) or getattr(
                    kvt_json, "local_dir", None
                )
                if path2:
                    return str(path2)
        except Exception:
            pass

        # 3) 仍然失败则交由调用方使用默认路径
        return None

    def _generate_foldername_debug(
        self,
        token_ids: torch.Tensor,
        mm_hashes: list[str],
        create_folder: bool = False,
    ) -> str:
        try:
            token_bytes = token_ids.cpu().numpy().tobytes()
        except Exception:
            token_bytes = bytes()
        if mm_hashes:
            try:
                token_bytes += "-".join(mm_hashes).encode("utf-8")
            except Exception:
                pass
        try:
            import hashlib as _hash

            input_ids_hash = _hash.md5(
                token_bytes, usedforsecurity=False  # type: ignore[arg-type]
            ).hexdigest()
        except Exception:
            input_ids_hash = ""
        foldername = os.path.join(self._storage_path, input_ids_hash)
        if create_folder:
            try:
                os.makedirs(foldername, exist_ok=True)
            except Exception:
                pass
        return foldername

    def _run_initial_prefetch(self, top_k: int) -> None:
        """一次性预取最近访问的部分 hash 到 DRAM Cache（按 last_access 排序）。"""
        if self._v1_meta is None or self._v1_storage is None:
            return
        try:
            metas = self._v1_meta.scan_by_session_layer(session_id=None, layer_id=None)
        except Exception:
            return
        valid: list[Any] = []
        for m in metas:
            try:
                if getattr(m, "status", 0) != 1:
                    continue
                if m.is_expired():
                    continue
                valid.append(m)
            except Exception:
                continue
        if not valid:
            logger.info("[v1_engine] prefetch: no valid metadata entries found")
            return
        try:
            valid.sort(key=lambda m: getattr(m, "last_access", 0), reverse=True)
        except Exception:
            pass
        selected = valid[: int(top_k)]
        logger.info(
            "[v1_engine] prefetch: top_k=%d, actual=%d",
            int(top_k),
            len(selected),
        )
        for m in selected:
            folder_abs = getattr(m, "file_path", None)
            if not folder_abs:
                continue
            hash_id = os.path.basename(str(folder_abs))
            if not hash_id:
                continue
            try:
                self._prefetch_hash_layers(hash_id)
            except Exception:
                continue

    def _prefetch_hash_layers(self, hash_id: str) -> None:
        """针对某个 hash 目录，遍历已有的 layer 文件并触发一次下载以填充 DRAM Cache。"""
        if self._v1_storage is None:
            return
        base_local = getattr(self._kv_transfer_config, "local_dir", None)
        base_remote = getattr(self._kv_transfer_config, "remote_dir", None) or getattr(
            self._kv_transfer_config, "crail_dir", None
        )
        roots: list[str] = []
        for base in (base_local, base_remote):
            if not base:
                continue
            root = os.path.join(str(base), hash_id)
            if os.path.isdir(root):
                roots.append(root)
        if not roots:
            return
        seen: set[str] = set()
        for root in roots:
            for fn in os.listdir(root):
                    if not fn.endswith(".safetensors"):
                        continue
                    rel_path = os.path.join(hash_id, fn)
                    if rel_path in seen:
                        continue
                    seen.add(rel_path)
                    try:
                        # 调用 download 即可触发 CachingStorage 将 bytes 写入内存缓存
                        _ = self._v1_storage.download(rel_path)
                    except Exception:
                        continue

    def _generate_filename_debug(
        self,
        layer_name: str,
        token_ids: torch.Tensor,
        mm_hashes: list[str],
    ) -> str:
        foldername = self._generate_foldername_debug(
            token_ids,
            mm_hashes,
            create_folder=True,
        )
        return os.path.join(foldername, f"{layer_name}.safetensors")


# ---------------- functional helpers ----------------


def _inject_kv_into_layer(
    dst_kv_cache_layer: torch.Tensor,
    src_kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    attn_metadata: Any,
) -> None:
    shape = dst_kv_cache_layer.shape
    try:
        from vllm.v1.attention.backends.mla.common import (  # type: ignore
            MLACommonMetadata as _MLA,
        )
    except Exception:  # pragma: no cover
        _MLA = object  # type: ignore

    if isinstance(attn_metadata, _MLA):
        num_pages, page_size = int(shape[0]), int(shape[1])
        view = dst_kv_cache_layer.reshape(num_pages * page_size, -1)
        view[slot_mapping, ...] = src_kv_cache
        view.reshape(shape)
    else:
        num_pages, page_size = int(shape[1]), int(shape[2])
        view = dst_kv_cache_layer.reshape(2, num_pages * page_size, -1)
        view[:, slot_mapping, ...] = src_kv_cache
        view.reshape(shape)


def _extract_kv_from_layer(
    layer: torch.Tensor,
    slot_mapping: torch.Tensor,
    attn_metadata: Any,
) -> torch.Tensor:
    try:
        from vllm.v1.attention.backends.mla.common import (  # type: ignore
            MLACommonMetadata as _MLA,
        )
    except Exception:  # pragma: no cover
        _MLA = object  # type: ignore

    if isinstance(attn_metadata, _MLA):
        num_pages, page_size = layer.shape[0], layer.shape[1]
        return layer.reshape(num_pages * page_size, -1)[slot_mapping, ...]
    num_pages, page_size = layer.shape[1], layer.shape[2]
    return layer.reshape(2, num_pages * page_size, -1)[:, slot_mapping, ...]


def _align_to_block_size(num_tokens: int, block_size: int) -> int:
    return (num_tokens - 1) // block_size * block_size


# ---------------- Module-level helpers (singleton) ----------------

_CORE_SINGLETON: Optional[V1KVEngineImpl] = None


def init_v1_engine(vllm_config: Any, role: Any = None) -> V1KVEngineImpl:
    global _CORE_SINGLETON
    if _CORE_SINGLETON is None:
        real_role = role if role is not None else KVConnectorRole.WORKER
        _CORE_SINGLETON = V1KVEngineImpl(vllm_config, real_role)
    return _CORE_SINGLETON


def destroy_v1_engine() -> None:
    global _CORE_SINGLETON
    _CORE_SINGLETON = None

