#!/usr/bin/env bash
#
# E3 five-corpus foundation-model pretraining -- Linux GPU server launcher.
#
#   bash scripts/run_e3_server.sh /srv/datasets/meta_canonical
#
# This is a thin wrapper. It adds no second training entry point: it validates
# the paths, then delegates to the single audited launcher
# `scripts/e3_oneclick.py server`, which runs the repository's own `pretrain.py`
# through Hydra.
#
# Environment:
#   E3_DATA_ROOT      canonical data root; a positional argument overrides it
#   E3_RUN_ROOT       output root for runs; default /srv/experiments/e3
#   E3_NATURE_CORPUS  official Nature corpus CSV; default <data-root>/nature_corpus.csv
#   E3_PYTHON         python interpreter; falls back to META_PYTHON, then python
#
# Any further arguments are forwarded to e3_oneclick.py, e.g.
#   bash scripts/run_e3_server.sh /srv/datasets/meta_canonical --resume /srv/experiments/e3/e3_server_.../checkpoints/last.ckpt
#
# See docs/SERVER_DEPLOY.md for the data layout and the resume procedure.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# A leading argument that is not a flag is the data root.
if [ "$#" -gt 0 ] && [ "${1#-}" = "$1" ]; then
  DATA_ROOT="$1"
  shift
else
  DATA_ROOT="${E3_DATA_ROOT:-}"
fi

if [ -z "$DATA_ROOT" ]; then
  echo "usage: bash scripts/run_e3_server.sh /path/to/canonical_meta" >&2
  echo "   or: export E3_DATA_ROOT=/srv/datasets/meta_canonical" >&2
  exit 2
fi

if [ ! -d "$DATA_ROOT" ]; then
  echo "[E3] ERROR: data root does not exist: $DATA_ROOT" >&2
  exit 2
fi

# Outputs stay outside the checkout. e3_oneclick.py enforces this too.
: "${E3_RUN_ROOT:=/srv/experiments/e3}"
export E3_RUN_ROOT
export E3_DATA_ROOT="$DATA_ROOT"

mkdir -p "$E3_RUN_ROOT"

PY="${E3_PYTHON:-${META_PYTHON:-python}}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "[E3] ERROR: python interpreter not found: $PY" >&2
  echo "[E3] Set E3_PYTHON=/path/to/python (or activate the environment first)." >&2
  exit 2
fi

echo "[E3] repository : $ROOT"
echo "[E3] data root  : $DATA_ROOT"
echo "[E3] run root   : $E3_RUN_ROOT"
echo "[E3] python     : $PY"

exec "$PY" scripts/e3_oneclick.py server --data-root "$DATA_ROOT" --output "$E3_RUN_ROOT" "$@"
