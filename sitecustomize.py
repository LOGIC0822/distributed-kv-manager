"""
Auto-inject Distributed KV Manager hooks for vLLM via sitecustomize.

Behavior
- Loads at Python startup if this repository is on PYTHONPATH.
- Tries minimal_hooks.inject() first, then falls back to inject_multi_block.inject().
- Controlled by environment variables:
  - DKV_AUTO_INJECT: '1'/true to enable (default), '0'/false to disable
  - DKV_INJECTOR: 'auto' (default), 'minimal', or 'multi' (multi_block)

Notes
- Idempotent: internal guards in the injectors prevent double patching.
- Works across exec/spawn since sitecustomize is re-imported in child processes.
"""
from __future__ import annotations

import os
import sys


def _log(msg: str) -> None:
    try:
        import logging
        logging.getLogger("dkv.sitecustomize").info(msg)
    except Exception:
        try:
            sys.stderr.write(f"[dkv.sitecustomize] {msg}\n")
        except Exception:
            pass


def _truthy(val: str | None) -> bool:
    if val is None:
        return False
    return str(val).lower() in {"1", "true", "yes", "on"}


def _inject_minimal() -> None:
    from vllm_adapter import minimal_hooks as mh  # type: ignore
    mh.inject()
    # Unconditional stderr marker so we can see it in any log
    try:
        sys.stderr.write("[dkv.sitecustomize] minimal_hooks.inject() installed\n")
    except Exception:
        pass
    _log("minimal_hooks.inject() installed")


def _inject_multi() -> None:
    from vllm_adapter import inject_multi_block as imb  # type: ignore
    imb.inject()
    try:
        sys.stderr.write("[dkv.sitecustomize] inject_multi_block.inject() installed\n")
    except Exception:
        pass
    _log("inject_multi_block.inject() installed")


def _main() -> None:
    if not _truthy(os.environ.get("DKV_AUTO_INJECT", "1")):
        _log("auto injection disabled by DKV_AUTO_INJECT")
        return

    mode = str(os.environ.get("DKV_INJECTOR", "auto")).lower()
    try:
        sys.stderr.write(f"[dkv.sitecustomize] start (mode={mode}) PYTHONPATH entries={len(sys.path)}\n")
    except Exception:
        pass
    try:
        if mode in ("auto", "minimal"):
            _inject_minimal()
            return
    except Exception as e:
        _log(f"minimal injector failed: {e}")
        # fall through

    try:
        if mode in ("auto", "multi", "multi_block"):
            _inject_multi()
            return
    except Exception as e:
        _log(f"multi_block injector failed: {e}")

    _log("no injector installed (check PYTHONPATH or set DKV_INJECTOR)")


# Execute on import
try:
    _main()
except Exception as _e:  # pragma: no cover
    try:
        _log(f"sitecustomize failed: {_e}")
    except Exception:
        pass
