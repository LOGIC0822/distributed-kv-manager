#!/usr/bin/env python3
"""Run three vLLM chat tests (short/medium/long) against a local vLLM server.

Usage:
  python scripts/run_vllm_tests.py --host 127.0.0.1 --port 8100 --model /tmp/ckpt

The script runs:
  - short: streaming chat (small prompt)
  - medium: non-streaming chat with moderate max_tokens
  - long: non-streaming chat with a long prefill (tries to trigger multi-block behavior)

Requires: requests (pip install requests)
"""
import argparse
import json
import time
import sys

try:
    import requests
except Exception:
    print("This script requires the 'requests' library. Install with: pip install requests", file=sys.stderr)
    raise


def short_test(base_url, model):
    print("\n=== SHORT (streaming) test ===")
    url = f"{base_url}/v1/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "写一首关于北方的诗。"}],
        "stream": True,
    }
    print("POST", url, json.dumps(body, ensure_ascii=False))
    start = time.time()
    with requests.post(url, json=body, stream=True) as r:
        r.raise_for_status()
        print("Streaming response:")
        try:
            for chunk in r.iter_lines(decode_unicode=True):
                if not chunk:
                    continue
                print(chunk)
        except KeyboardInterrupt:
            print("(interrupted)")
    print(f"Elapsed: {time.time()-start:.2f}s")


def medium_test(base_url, model):
    print("\n=== MEDIUM test ===")
    url = f"{base_url}/v1/chat/completions"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "请详细解释氧化还原的原理。"},
        ],
        "max_tokens": 200,
        "temperature": 0.7,
    }
    print("POST", url)
    start = time.time()
    r = requests.post(url, json=body)
    try:
        r.raise_for_status()
    except Exception:
        print("Status:", r.status_code, r.text)
        raise
    print("Response JSON:")
    print(json.dumps(r.json(), ensure_ascii=False, indent=2))
    print(f"Elapsed: {time.time()-start:.2f}s")


def long_test(base_url, model):
    print("\n=== LONG test (long prefill) ===")
    url = f"{base_url}/v1/chat/completions"
    # build a long prompt by repeating a phrase to try to generate multiple blocks
    long_prefix = ("这是一个很长的上下文。" * 200)[:20000]
    body = {
        "model": model,
        "messages": [{"role": "user", "content": long_prefix + "请基于上文总结要点并给出示例。"}],
        # attach session id to encourage KV-related behavior
        "extra_body": {"session_id": "kv_session_0000"},
    }
    print("POST", url, "(long prefill)")
    start = time.time()
    r = requests.post(url, json=body)
    try:
        r.raise_for_status()
    except Exception:
        print("Status:", r.status_code, r.text)
        raise
    print("Response JSON (truncated):")
    j = r.json()
    # print a compacted view
    print(json.dumps(j, ensure_ascii=False)[:4000])
    print(f"Elapsed: {time.time()-start:.2f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', default=8100, type=int)
    p.add_argument('--model', default='/tmp/ckpt')
    p.add_argument('--which', choices=['short', 'medium', 'long', 'all'], default='all')
    args = p.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    print(f"Target: {base_url}  model={args.model}  which={args.which}")

    try:
        if args.which in ('short', 'all'):
            short_test(base_url, args.model)
        if args.which in ('medium', 'all'):
            medium_test(base_url, args.model)
        if args.which in ('long', 'all'):
            long_test(base_url, args.model)
    except requests.exceptions.RequestException as e:
        print('Request failed:', e, file=sys.stderr)


if __name__ == '__main__':
    main()
