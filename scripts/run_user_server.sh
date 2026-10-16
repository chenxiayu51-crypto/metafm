#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${E3_PYTHON:-${META_PYTHON:-python}}"
exec "$PY" scripts/server_dataset.py "$@"
