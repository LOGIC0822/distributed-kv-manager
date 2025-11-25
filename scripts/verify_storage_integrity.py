#!/usr/bin/env python3
"""Scan ETCD metadata keys and verify corresponding storage payload files exist.

Usage (PowerShell/Linux):
  python scripts/verify_storage_integrity.py [--delete-missing]

Requires: etcd3 installed, and a config.json with kv_transfer_config.{etcd_endpoints,storage_dir}.
Outputs a summary table of missing vs existing files. If --delete-missing is passed, it will
remove ONLY the metadata keys for which the underlying file is absent (does not touch other files).
"""
import os
import json
import argparse
import etcd3
from typing import List

PREFIX = '/kvmeta'

def load_config(path: str = 'config.json'):
    with open(path,'r',encoding='utf-8') as f:
        raw = json.load(f)
    return raw.get('kv_transfer_config', {}), raw

def connect_etcd(endpoints: List[str]):
    # use first endpoint for simplicity; could be extended to failover.
    host, port = endpoints[0].split(':')
    return etcd3.client(host=host, port=int(port))


def list_meta_keys(cli) -> List[str]:
    # etcd3 client's get_prefix returns (value, metadata) pairs
    keys = []
    for value, md in cli.get_prefix(PREFIX):  # type: ignore
        if md and md.key:
            keys.append(md.key.decode('utf-8'))
    return keys


def rel_from_full(full: str) -> str:
    if full.startswith(PREFIX + '/'):
        return full[len(PREFIX)+1:]
    return full.lstrip('/')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config.json')
    ap.add_argument('--delete-missing', action='store_true', help='Delete only metadata keys whose files are missing')
    args = ap.parse_args()

    kv_conf, full_conf = load_config(args.config)
    storage_dir = kv_conf.get('storage_dir') or kv_conf.get('local_dir') or '/kvcache'
    endpoints = kv_conf.get('etcd_endpoints', ['127.0.0.1:2379'])

    cli = connect_etcd(endpoints)
    keys = list_meta_keys(cli)
    if not keys:
        print('[integrity] No metadata keys found under', PREFIX)
        return

    print(f'[integrity] Scanning {len(keys)} metadata keys; storage_dir={storage_dir}')
    missing = []
    present = 0
    for full_key in keys:
        rel = rel_from_full(full_key)
        file_path = os.path.join(storage_dir, rel)
        exists = os.path.exists(file_path)
        if not exists:
            missing.append((full_key, file_path))
        else:
            present += 1

    print(f'[integrity] present={present} missing={len(missing)}')
    if missing:
        print('--- Missing detail (up to 50) ---')
        for i,(mk, fp) in enumerate(missing[:50]):
            print(f'[{i}] meta={mk} file_absent={fp}')

    if args.delete_missing and missing:
        deleted = 0
        for mk,_ in missing:
            rel = rel_from_full(mk)
            # delete only metadata; ignore errors
            try:
                cli.delete(rel)  # using relative key
                deleted += 1
            except Exception as e:
                print('[integrity] failed delete', mk, e)
        print(f'[integrity] deleted {deleted} stale metadata keys')

if __name__ == '__main__':
    main()
