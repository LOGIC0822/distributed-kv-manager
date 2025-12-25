"""
Shim module to expose the v0 connector via the installed package path.

The actual implementation lives in the top-level ``vllm_adapter`` namespace
to keep parity with earlier iterations. Importing this module re-exports
everything from ``vllm_adapter.distributed_kv_connector`` so that
``kv_connector_module_path=distributed_kv_manager.vllm_adapter.distributed_kv_connector``
continues to work.
"""

try:  # lightweight import marker
    with open("/tmp/connector_debug.log", "a", encoding="utf-8") as _f:
        _f.write("[shim] imported distributed_kv_manager.vllm_adapter.distributed_kv_connector\n")
        _f.flush()
except Exception:
    pass

from vllm_adapter.distributed_kv_connector import *  # noqa: F401,F403
