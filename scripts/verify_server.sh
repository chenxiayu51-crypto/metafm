#!/usr/bin/env bash
#
# E3 server preflight -- verify a Linux GPU host can launch the five-corpus
# foundation-model pretraining run.
#
#   bash scripts/verify_server.sh /srv/datasets/meta_canonical
#
# Every check is read-only. Nothing is installed, nothing is trained. The deep
# data checks reuse the repository's own validators through
# `scripts/e3_oneclick.py audit --probe-loader`, so the preflight exercises the
# same code path a real launch uses rather than a parallel reimplementation.
#
# Exit status is 0 only when no check FAILED. SKIP is reported when a check
# cannot apply (for example, no data root was supplied).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${E3_PYTHON:-${META_PYTHON:-python}}"
DATA_ROOT="${1:-${E3_DATA_ROOT:-}}"
RUN_ROOT="${E3_RUN_ROOT:-/srv/experiments/e3}"

FAILED=0
pass() { printf '  \033[32mPASS\033[0m  %s\n' "$*"; }
fail() { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; FAILED=$((FAILED + 1)); }
skip() { printf '  \033[33mSKIP\033[0m  %s\n' "$*"; }
section() { printf '\n== %s ==\n' "$*"; }

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "[E3] ERROR: python interpreter not found: $PY" >&2
  echo "[E3] Set E3_PYTHON=/path/to/python. Run scripts/setup_server.sh first." >&2
  exit 2
fi

# ---------------------------------------------------------------------------
section "Repository"
# ---------------------------------------------------------------------------
for path in \
  pretrain.py \
  scripts/e3_oneclick.py \
  configs/pretrain.yaml \
  configs/trainer/e3_server.yaml \
  semg_fm/data/datamodule.py \
  semg_fm/models/ring_criss_cross.py \
  semg_fm/lightning/pretrain_module.py \
  semg_fm/e3_diagnostics.py
do
  if [ -f "$ROOT/$path" ]; then pass "present: $path"; else fail "missing: $path"; fi
done

# ---------------------------------------------------------------------------
section "Python environment and GPU"
# ---------------------------------------------------------------------------
"$PY" - <<'PY'
import sys
ok = True
major, minor = sys.version_info[:2]
if (major, minor) >= (3, 10):
    print(f"  \033[32mPASS\033[0m  python {sys.version.split()[0]}")
else:
    print(f"  \033[31mFAIL\033[0m  python {sys.version.split()[0]} < 3.10"); ok = False
try:
    import torch
except ImportError:
    print("  \033[31mFAIL\033[0m  torch is not installed"); raise SystemExit(1)
print(f"  \033[32mPASS\033[0m  torch {torch.__version__}")
if torch.cuda.is_available():
    print(f"  \033[32mPASS\033[0m  CUDA available; {torch.cuda.device_count()} device(s)")
    for i in range(torch.cuda.device_count()):
        print(f"        cuda:{i} = {torch.cuda.get_device_name(i)}")
    if torch.cuda.is_bf16_supported():
        print("  \033[32mPASS\033[0m  bf16 supported by this device")
    else:
        print("  \033[31mFAIL\033[0m  bf16 NOT supported; E3 server profile requires bf16-mixed"); ok = False
else:
    print("  \033[31mFAIL\033[0m  torch.cuda.is_available() is False; E3 server mode requires a GPU"); ok = False
raise SystemExit(0 if ok else 1)
PY
[ $? -eq 0 ] || FAILED=$((FAILED + 1))

# ---------------------------------------------------------------------------
section "Project imports"
# ---------------------------------------------------------------------------
"$PY" - <<'PY'
import importlib, sys
ok = True
for name in ("hydra", "omegaconf", "lightning", "torchmetrics", "h5py", "pandas", "numpy"):
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "?")
        print(f"  \033[32mPASS\033[0m  import {name} {version}")
    except Exception as exc:
        print(f"  \033[31mFAIL\033[0m  import {name}: {exc}"); ok = False
sys.path.insert(0, ".")
for name in ("semg_fm.data.datamodule", "semg_fm.models.ring_criss_cross",
             "semg_fm.lightning.pretrain_module", "semg_fm.e3_diagnostics",
             "semg_fm.reproducibility"):
    try:
        importlib.import_module(name)
        print(f"  \033[32mPASS\033[0m  import {name}")
    except Exception as exc:
        print(f"  \033[31mFAIL\033[0m  import {name}: {exc}"); ok = False
raise SystemExit(0 if ok else 1)
PY
[ $? -eq 0 ] || FAILED=$((FAILED + 1))

# ---------------------------------------------------------------------------
section "Data root"
# ---------------------------------------------------------------------------
if [ -z "$DATA_ROOT" ]; then
  skip "no data root given; run: bash scripts/verify_server.sh /srv/datasets/meta_canonical"
  skip "canonical HDF5 / manifest / corpus checks"
  skip "five-corpus window and DataLoader checks"
else
  export E3_DATA_ROOT="$DATA_ROOT"
  if [ -d "$DATA_ROOT" ]; then pass "data root exists: $DATA_ROOT"; else fail "data root missing: $DATA_ROOT"; fi

  if [ -f "$DATA_ROOT/splits.csv" ]; then
    pass "splits.csv present"
  else
    fail "splits.csv missing (run scripts/prepare_data.sh; do not hand-write it)"
  fi

  if [ -f "${E3_NATURE_CORPUS:-$DATA_ROOT/nature_corpus.csv}" ]; then
    pass "nature_corpus.csv present"
  else
    fail "nature_corpus.csv missing: the three GNI corpora need the OFFICIAL Meta intervals; they are never inferred"
  fi

  if [ -f "$DATA_ROOT/splits.csv" ] && [ -f "${E3_NATURE_CORPUS:-$DATA_ROOT/nature_corpus.csv}" ]; then
    "$PY" - "$DATA_ROOT" "${E3_NATURE_CORPUS:-$DATA_ROOT/nature_corpus.csv}" <<'PY'
import csv, importlib.util, sys
from collections import Counter
from pathlib import Path
root, corpus_path = Path(sys.argv[1]), Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("e3", "scripts/e3_oneclick.py")
e3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e3)
ok = True

manifest = root / "splits.csv"
with manifest.open(encoding="utf-8-sig", newline="") as fh:
    reader = csv.DictReader(fh); rows = list(reader); columns = reader.fieldnames or []
missing = sorted(set(e3.REQUIRED) - set(columns))
if missing:
    print(f"  \033[31mFAIL\033[0m  splits.csv missing columns: {missing}"); ok = False
elif not rows:
    print("  \033[31mFAIL\033[0m  splits.csv has zero rows"); ok = False
else:
    print(f"  \033[32mPASS\033[0m  splits.csv parses: {len(rows)} records, all required columns present")
    counts = Counter(r["dataset_id"] for r in rows)
    for name in e3.TASKS:
        if counts.get(name):
            print(f"        {name}: {counts[name]} record(s)")
        else:
            print(f"  \033[31mFAIL\033[0m  splits.csv has no records for {name}"); ok = False

try:
    corpus_rows = e3.read_nature_corpus(corpus_path)
    splits = Counter(r["split"] for r in corpus_rows)
    print(f"  \033[32mPASS\033[0m  nature_corpus.csv parses: {len(corpus_rows)} half-open intervals {dict(splits)}")
except e3.E3Error as exc:
    print(f"  \033[31mFAIL\033[0m  nature_corpus.csv: {exc}"); ok = False

raise SystemExit(0 if ok else 1)
PY
    [ $? -eq 0 ] || FAILED=$((FAILED + 1))
  fi

  # -------------------------------------------------------------------------
  section "Canonical HDF5, official intervals, five-corpus windows, DataLoader"
  # -------------------------------------------------------------------------
  echo "  (running the real Dataset/DataModule through e3_oneclick.py audit)"
  # audit writes its JSON report to stdout and progress to stderr. The shell
  # captures stdout and reads stderr itself, so no path is ever handed to a
  # Python interpreter that might resolve a POSIX temp path differently.
  AUDIT_ERR="$(mktemp)"
  if AUDIT_OUT="$("$PY" scripts/e3_oneclick.py audit --data-root "$DATA_ROOT" --probe-loader 2>"$AUDIT_ERR")"; then
    pass "preflight audit + real Train/Val DataLoader probe"
    printf '%s' "$AUDIT_OUT" | "$PY" -c "
import json, sys
report = json.load(sys.stdin)
print('        records=%s  checked_h5=%s' % (report['records'], report['checked_h5']))
for task, splits in sorted(report['counts'].items()):
    print('        %-24s train=%-4s val=%-4s test=%s' % (task, splits['train'], splits['val'], splits['test']))
flow = report.get('real_dataflow', {})
for split in ('train', 'val'):
    if split in flow:
        print('        %s windows=%s tasks=%s' % (split, flow[split]['windows'], sorted(set(flow[split]['first_batches']))))
" || fail "audit produced an unparsable report"
  else
    fail "preflight audit failed"
    sed 's/^/        /' "$AUDIT_ERR" | tail -25
  fi
  rm -f "$AUDIT_ERR"
fi

# ---------------------------------------------------------------------------
section "Output root"
# ---------------------------------------------------------------------------
if mkdir -p "$RUN_ROOT" 2>/dev/null && [ -w "$RUN_ROOT" ]; then
  pass "run root is writable: $RUN_ROOT"
  probe="$RUN_ROOT/.e3_write_probe.$$"
  if ( : >"$probe" ) 2>/dev/null; then rm -f "$probe"; pass "file creation in run root"; else fail "cannot create files in $RUN_ROOT"; fi
else
  fail "run root is not writable: $RUN_ROOT (export E3_RUN_ROOT=/srv/experiments/e3)"
fi

# ---------------------------------------------------------------------------
echo
if [ "$FAILED" -eq 0 ]; then
  echo "== E3 server preflight: PASS (0 failures) =="
  echo "   next: bash scripts/run_e3_server.sh $DATA_ROOT"
  exit 0
fi
echo "== E3 server preflight: FAIL ($FAILED failed check(s)) ==" >&2
echo "   fix the FAIL lines above before launching training." >&2
exit 1
