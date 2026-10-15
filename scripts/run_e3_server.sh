#!/usr/bin/env bash
#
# E3 five-corpus foundation-model pretraining -- Linux GPU server launcher.
#
#   bash scripts/run_e3_server.sh /srv/datasets/meta_canonical
#
# Runs the server preflight first and only then starts training, as two separate
# processes. That separation is deliberate: the preflight performs real optimiser
# steps, and a separate process guarantees the production run cannot inherit its
# weights, optimizer state or LR schedule.
#
#   data root   -> preflight (environment, data, dataloader, GPU smoke)
#               -> STOP unless every section passed
#               -> e3_oneclick.py server  (the single training entry point)
#
# Environment:
#   E3_DATA_ROOT            canonical data root; a positional argument overrides it
#   E3_RUN_ROOT             output root; default /srv/experiments/e3
#   E3_NATURE_CORPUS        official corpus CSV; default <data-root>/nature_corpus.csv
#   E3_PYTHON               interpreter; falls back to META_PYTHON, then python
#   E3_PREFLIGHT_SMOKE_STEPS  preflight optimiser steps (default 5, minimum 5)
#   E3_SKIP_PREFLIGHT       set to 1 to go straight to training (not recommended)
#
# Extra arguments are forwarded to e3_oneclick.py, e.g.
#   bash scripts/run_e3_server.sh /srv/datasets/meta_canonical \
#     --resume /srv/experiments/e3/e3_server_<ts>/checkpoints/last.ckpt
#
# See docs/SERVER_DEPLOY.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Pull our own flags out before deciding what the data root is.
PASSTHROUGH=()
SKIP_PREFLIGHT="${E3_SKIP_PREFLIGHT:-0}"
for arg in "$@"; do
  case "$arg" in
    --skip-preflight) SKIP_PREFLIGHT=1 ;;
    *) PASSTHROUGH+=("$arg") ;;
  esac
done

DATA_ROOT=""
if [ "${#PASSTHROUGH[@]}" -gt 0 ] && [ "${PASSTHROUGH[0]#-}" = "${PASSTHROUGH[0]}" ]; then
  DATA_ROOT="${PASSTHROUGH[0]}"
  PASSTHROUGH=("${PASSTHROUGH[@]:1}")
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

PY="${E3_PYTHON:-${META_PYTHON:-python}}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "[E3] ERROR: python interpreter not found: $PY" >&2
  echo "[E3] Set E3_PYTHON=/path/to/python (or activate the environment first)." >&2
  exit 2
fi

mkdir -p "$E3_RUN_ROOT"

echo "[E3] repository : $ROOT"
echo "[E3] data root  : $DATA_ROOT"
echo "[E3] run root   : $E3_RUN_ROOT"
echo "[E3] python     : $PY"

if [ "$SKIP_PREFLIGHT" = "1" ]; then
  echo "[E3] WARNING: preflight skipped (E3_SKIP_PREFLIGHT=1)."
else
  echo ""
  echo "[E3] == server preflight =="
  # Preflight must validate the SAME batch size training will use: a batch size
  # larger than a corpus's window count makes that corpus contribute zero
  # batches, and a preflight run at a different size would not catch it.
  SMOKE_STEPS="${E3_PREFLIGHT_SMOKE_STEPS:-5}"
  BATCH_SIZE="${E3_BATCH_SIZE:-8}"
  if ! "$PY" scripts/server_preflight.py \
        --data-root "$DATA_ROOT" \
        --output "$E3_RUN_ROOT/preflight" \
        --smoke-steps "$SMOKE_STEPS" \
        --batch-size "$BATCH_SIZE"; then
    echo "" >&2
    echo "[E3] PREFLIGHT FAILED. Training was not started." >&2
    echo "[E3] Fix the FAIL lines above, or review $E3_RUN_ROOT/preflight for the report." >&2
    exit 1
  fi
  echo ""
fi

# Fresh process, fresh model: never resumes the preflight's short run.
TRAIN_ARGS=()
if [ "$SKIP_PREFLIGHT" != "1" ]; then
  TRAIN_ARGS+=(--batch-size "${E3_BATCH_SIZE:-8}")
fi
exec "$PY" scripts/e3_oneclick.py server --data-root "$DATA_ROOT" --output "$E3_RUN_ROOT" \
  "${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"}" "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}"
