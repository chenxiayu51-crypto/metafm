#!/usr/bin/env python3
"""Verify packaged source against SOURCE_RELEASE.json without changing files."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    release = json.loads((root / 'SOURCE_RELEASE.json').read_text(encoding='utf-8'))
    failed = []
    for relative, expected in release['files'].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            failed.append(f'missing/unsafe: {relative}')
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failed.append(f'changed: {relative}')
    if failed:
        print('\n'.join(failed))
        return 1
    print(f"PASS: {len(release['files'])} packaged files; base commit {release['base_commit']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
