#!/usr/bin/env python3
"""Clear KV metadata in etcd and stored KV files on disk (v0 + v1).

Usage:
  python scripts/clear_kv_store.py [--yes] [--dry-run]
                                   [--session-prefix PREFIX]
                                   [--etcd-only] [--storage-only]
                                   [--ssd]

By default the script reads `config.json` in the repo root to discover
`kv_transfer_config` fields:

- v0:
  - `etcd_endpoints`
  - `storage_dir` / `local_dir`
  - `ssd_cache_dir`
- v1:
  - `dkv_storage_path`
  - `local_dir`
  - `remote_dir`

It will:
- For etcd:
  - Clear v0 metadata under prefix `/kvmeta`
  - Clear v1 metadata under prefix `/kvmeta_v1`
- For storage:
  - v0: remove `kv_*.pt` files under `storage_dir` and optional `ssd_cache_dir`
  - v1: remove `.safetensors` files under `dkv_storage_path`, `local_dir`, and
    `remote_dir` (if configured)

Options:
  --yes                Actually perform deletions. Without this the script runs in dry-run mode.
  --dry-run            Explicit dry run (default if --yes omitted).
  --session-prefix STR For v0: only delete metadata/files whose key/filename
                       contains this prefix (e.g. 'kv_session_0000').
                       For v1: this is applied as a substring filter on the
                       safetensors file path.
  --etcd-only          Only touch etcd (don't remove files on disk).
  --storage-only       Only remove files on disk (don't touch etcd).
  --ssd                Also clean ssd_cache_dir if present in config.

This is a convenience script intended for development/testing. Use with care.
"""

import argparse
import json
import os
from typing import Iterable, List


def load_config(repo_root: str | None = None) -> dict:
    if repo_root is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg_path = os.path.join(repo_root, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"config.json not found at {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg.get("kv_transfer_config", {})


# -------- v0 storage cleanup (kv_*.pt) --------

def clear_storage_v0(
    storage_dir: str | None,
    session_prefix: str | None = None,
    dry_run: bool = True,
    also_ssd: str | None = None,
) -> int:
    if not storage_dir:
        print("[storage/v0] storage_dir not provided, skipping")
        return 0
    storage_dir = os.path.expanduser(storage_dir)
    if not os.path.exists(storage_dir):
        print(f"[storage/v0] storage dir not found: {storage_dir}")
        return 0

    removed = 0
    print(f"[storage/v0] scanning {storage_dir} (dry_run={dry_run})")
    for root, _dirs, files in os.walk(storage_dir):
        for fn in files:
            # v0 文件通常形如: kv_<session>_layer_<id>_*.pt
            if not (fn.startswith("kv_") and fn.endswith(".pt")):
                continue
            if session_prefix and session_prefix not in fn:
                continue
            path = os.path.join(root, fn)
            if dry_run:
                print("[storage/v0] would remove", path)
            else:
                try:
                    os.remove(path)
                    print("[storage/v0] removed", path)
                    removed += 1
                except Exception as e:
                    print("[storage/v0] failed to remove", path, e)

    if also_ssd:
        ssd_dir = os.path.expanduser(also_ssd)
        if os.path.exists(ssd_dir):
            print(f"[storage/v0-ssd] scanning {ssd_dir} (dry_run={dry_run})")
            for root, _dirs, files in os.walk(ssd_dir):
                for fn in files:
                    if not fn.endswith(".pt"):
                        continue
                    if session_prefix and session_prefix not in fn:
                        continue
                    path = os.path.join(root, fn)
                    if dry_run:
                        print("[storage/v0-ssd] would remove", path)
                    else:
                        try:
                            os.remove(path)
                            print("[storage/v0-ssd] removed", path)
                            removed += 1
                        except Exception as e:
                            print("[storage/v0-ssd] failed to remove", path, e)
    return removed


# -------- v1 storage cleanup (per-layer safetensors) --------

def _iter_existing_dirs(paths: Iterable[str | None]) -> List[str]:
    out: List[str] = []
    for p in paths:
        if not p:
            continue
        p_exp = os.path.expanduser(str(p))
        if os.path.exists(p_exp):
            out.append(p_exp)
    # 去重
    return list(dict.fromkeys(out).keys())


def clear_storage_v1(
    v1_dirs: Iterable[str],
    session_prefix: str | None = None,
    dry_run: bool = True,
) -> int:
    """清理 v1 外部 KV 存储（per-layer safetensors）。

    - v1 当前使用 hashed 路径 + per-layer safetensors:
      <root>/<hash>/model.layers.N.self_attn.attn.safetensors
    - 这里不尝试解析 hash，只是简单删除指定目录下的 .safetensors 文件。
    """
    dirs = _iter_existing_dirs(v1_dirs)
    if not dirs:
        print("[storage/v1] no v1 directories found, skipping")
        return 0

    removed = 0
    for d in dirs:
        print(f"[storage/v1] scanning {d} (dry_run={dry_run})")
        for root, _dirs, files in os.walk(d):
            for fn in files:
                if not fn.endswith(".safetensors"):
                    continue
                path = os.path.join(root, fn)
                if session_prefix and session_prefix not in path:
                    continue
                if dry_run:
                    print("[storage/v1] would remove", path)
                else:
                    try:
                        os.remove(path)
                        print("[storage/v1] removed", path)
                        removed += 1
                    except Exception as e:
                        print("[storage/v1] failed to remove", path, e)
    return removed


# -------- etcd cleanup (v0 + v1) --------

def clear_etcd_prefix(
    endpoints: Iterable[str],
    prefix: str,
    session_prefix: str | None = None,
    dry_run: bool = True,
) -> int:
    try:
        import etcd3
    except Exception as e:
        print(f"[etcd] etcd3 package not available: {e}")
        print("Install with: pip install etcd3")
        return 0

    total_deleted = 0
    for ep in endpoints:
        host, sep, port = ep.partition(":")
        port = int(port) if sep else 2379
        print(f"[etcd] connecting to {host}:{port} (prefix={prefix}, dry_run={dry_run})")
        try:
            client = etcd3.client(host=host, port=port)
        except Exception as e:
            print("[etcd] failed to connect to", ep, e)
            continue

        print(f"[etcd] listing keys with prefix {prefix}")
        try:
            it = client.get_prefix(prefix)
            for _value, meta in it:
                # meta is an etcd3.metadata.KeyValue
                key = meta.key.decode() if isinstance(meta.key, bytes) else meta.key
                if session_prefix and session_prefix not in key:
                    continue
                if dry_run:
                    print("[etcd] would delete", key)
                else:
                    ok = client.delete(key)
                    print("[etcd] deleted", key, "ok=", ok)
                    total_deleted += 1
        except Exception as e:
            print("[etcd] error listing/deleting keys at", ep, e)
    return total_deleted


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--yes", action="store_true", help="Actually perform deletions.")
    p.add_argument("--dry-run", action="store_true", help="Dry run (default).")
    p.add_argument(
        "--session-prefix",
        default=None,
        help="Only delete keys/files containing this prefix "
        "(v0: match kv_* filenames / etcd keys; v1: substring on path).",
    )
    p.add_argument("--etcd-only", action="store_true")
    p.add_argument("--storage-only", action="store_true")
    p.add_argument(
        "--ssd",
        action="store_true",
        help="Also clean ssd cache dir from config if present (v0 only).",
    )
    args = p.parse_args()

    dry = not args.yes or args.dry_run

    cfg = load_config()
    etcd_eps = cfg.get("etcd_endpoints", [])
    storage_dir_v0 = cfg.get("storage_dir") or cfg.get("local_dir")
    ssd_dir = cfg.get("ssd_cache_dir") if args.ssd else None

    # v1 directories: dkv_storage_path (debug/fallback), local_dir, remote_dir
    dkv_storage_path = cfg.get("dkv_storage_path")
    v1_local_dir = cfg.get("local_dir")
    v1_remote_dir = cfg.get("remote_dir") or cfg.get("crail_dir")
    v1_dirs = [dkv_storage_path, v1_local_dir, v1_remote_dir]

    if not args.storage_only:
        if not etcd_eps:
            print("[etcd] no endpoints found in config, skipping etcd")
        else:
            # v0 metadata (/kvmeta)
            n0 = clear_etcd_prefix(
                etcd_eps,
                prefix="/kvmeta",
                session_prefix=args.session_prefix,
                dry_run=dry,
            )
            print("[etcd] v0 total deleted:", n0)

            # v1 metadata (/kvmeta_v1)
            n1 = clear_etcd_prefix(
                etcd_eps,
                prefix="/kvmeta_v1",
                session_prefix=args.session_prefix,
                dry_run=dry,
            )
            print("[etcd] v1 total deleted:", n1)

    if not args.etcd_only:
        # v0 storage
        if not storage_dir_v0:
            print("[storage/v0] storage_dir not found in config, skipping v0 storage cleanup")
        else:
            n = clear_storage_v0(
                storage_dir_v0,
                session_prefix=args.session_prefix,
                dry_run=dry,
                also_ssd=ssd_dir,
            )
            print("[storage/v0] total removed:", n)

        # v1 storage
        n_v1 = clear_storage_v1(
            v1_dirs,
            session_prefix=args.session_prefix,
            dry_run=dry,
        )
        print("[storage/v1] total removed:", n_v1)

    if dry:
        print("\nDRY RUN completed. To actually delete, re-run with --yes")


if __name__ == "__main__":
    main()

