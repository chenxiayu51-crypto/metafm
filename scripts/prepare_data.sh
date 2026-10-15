#!/usr/bin/env bash
#
# Build a canonical five-corpus data root from raw HDF5.
#
#   bash scripts/prepare_data.sh /srv/datasets/raw /srv/datasets/meta_canonical
#
# Runs the repository's five existing converters, then make_splits, then stops
# for you to supply nature_corpus.csv if it is missing. See docs/DATA_LAYOUT.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RAW_ROOT="${1:-${E3_RAW_ROOT:-}}"
CANONICAL_ROOT="${2:-${E3_DATA_ROOT:-}}"

if [ -z "$RAW_ROOT" ] || [ -z "$CANONICAL_ROOT" ]; then
  echo "usage: bash scripts/prepare_data.sh /path/to/raw /path/to/canonical_meta" >&2
  echo "   or: export E3_RAW_ROOT=... E3_DATA_ROOT=..." >&2
  exit 2
fi

PY="${E3_PYTHON:-${META_PYTHON:-python}}"

exec "$PY" scripts/prepare_data.py \
  --raw-root "$RAW_ROOT" \
  --canonical-root "$CANONICAL_ROOT" \
  "${@:3}"
