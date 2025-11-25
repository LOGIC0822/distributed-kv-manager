#!/usr/bin/env python3
"""Wrapper to start vLLM after installing inject_multi_block hooks.

Usage (PowerShell):
  $env:VLLM_USE_V1="0"; python vllm_adapter\vllm_start_with_inject.py --model /tmp/ckpt --port 8100 --max-model-len 10000 --gpu-memory-utilization 0.8 --kv-transfer-config '{"kv_connector":"DistributedKVConnector","kv_role":"kv_both"}'

This script ensures the repo is on sys.path, calls inject(), then runs
the vLLM entrypoint module in-process so the hook is installed before
the ModelRunner / models are created.
"""
import os
import sys

# Compute repository root (allow override via env REPO_ROOT). Default: parent dir of this file.
env_root = os.environ.get("REPO_ROOT")
if env_root:
    REPO_ROOT = os.path.expanduser(env_root)
else:
    # this file is at <repo>/vllm_adapter/vllm_start_with_inject.py
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Try to install our injector (best-effort)
try:
    from vllm_adapter.inject_multi_block import inject
    inject()
    print("[vllm_start_with_inject] inject() called")
except Exception as e:
    print(f"[vllm_start_with_inject] inject() failed: {e}", file=sys.stderr)

# Replace current process with a real python -m invocation so multiprocessing
# child processes have a real __main__ file instead of '<stdin>' (avoids
# FileNotFoundError during spawn when runpy was used).
python_exe = sys.executable or "python3"
cli_args = list(sys.argv[1:])

# Optional: enable chunked prefill to allow long prompts to surface multiple sealed blocks.
# This greatly improves the chance to persist full prompt KV via incremental block stores.
# Opt-in via env KV_FORCE_CHUNKED_PREFILL=1. We only add the flag if it's not already present
# to avoid breaking user-provided explicit configs.
if os.environ.get("KV_FORCE_CHUNKED_PREFILL", "0") in ("1", "true", "True"):
    # Accept several historical spellings; only add if none present.
    known_flags = (
        "--enable-chunked-prefill",
        "--no-enable-chunked-prefill",
        "--chunked-prefill-enabled",
        "--chunked-prefill",
    )
    has_flag = any(str(a).startswith(f) for a in cli_args for f in known_flags)
    if not has_flag:
        # For vLLM that uses BooleanOptionalAction, provide the positive form without an explicit value
        cli_args += ["--enable-chunked-prefill"]

# Optional tuning knobs via env to force multiple partial-prefill passes for long prompts.
# Useful when your prompt length is below vLLM's default 2048 token chunk threshold.
def _append_if_env(env_name: str, flag_name: str, conv=int):
    val = os.environ.get(env_name)
    if val is None:
        return
    # avoid duplicating if user already provided the flag
    if any(str(a) == flag_name for a in cli_args):
        return
    try:
        parsed = conv(val)
    except Exception:
        parsed = val
    cli_args.extend([flag_name, str(parsed)])

# Example: set KV_MAX_NUM_BATCHED_TOKENS=256 to split 1.2k prompt into ~5 chunks.
_append_if_env("KV_MAX_NUM_BATCHED_TOKENS", "--max-num-batched-tokens", int)
# Limit how many partial-prefills can be emitted (optional).
_append_if_env("KV_MAX_NUM_PARTIAL_PREFILLS", "--max-num-partial-prefills", int)
# Force treating prompts longer than this threshold as long-prefill (optional).
_append_if_env("KV_LONG_PREFILL_TOKEN_THRESHOLD", "--long-prefill-token-threshold", int)

args = [python_exe, "-m", "vllm.entrypoints.openai.api_server"] + cli_args

# Ensure child process can import the repository-local packages by setting PYTHONPATH
env = os.environ.copy()
old_pp = env.get('PYTHONPATH', '')
repo_pp = REPO_ROOT
if old_pp:
    env['PYTHONPATH'] = repo_pp + os.pathsep + old_pp
else:
    env['PYTHONPATH'] = repo_pp

# Use execve to pass modified environment (so vLLM sees our PYTHONPATH and optional flags)
os.execve(python_exe, args, env)
