#!/usr/bin/env bash
#
# E3 server environment setup -- INSPECT AND REPORT ONLY.
#
#   bash scripts/setup_server.sh
#
# This script never installs, upgrades, or removes anything. It must not: an
# unattended `pip install` on a shared GPU host can silently replace a CUDA
# build that other projects depend on, and a wrong torch/CUDA pairing produces
# failures that look like model bugs. Instead it inspects what is present,
# compares it against the verified set in requirements-server.txt, prints the
# exact commands a human should run, and stops.
#
# Environment:
#   E3_PYTHON / META_PYTHON  interpreter to inspect (default: python)
#
# Exit status: 0 = ready to train, 1 = action required, 2 = cannot inspect.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${E3_PYTHON:-${META_PYTHON:-python}}"
ACTION=0
section() { printf '\n== %s ==\n' "$*"; }

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "[E3] ERROR: python interpreter not found: $PY" >&2
  echo "[E3] Create a virtual environment and set E3_PYTHON to it, for example:" >&2
  echo "       python3 -m venv /srv/projects/metafmodel/.venv" >&2
  echo "       export E3_PYTHON=/srv/projects/metafmodel/.venv/bin/python" >&2
  echo "[E3] This script will not create or modify an environment for you." >&2
  exit 2
fi

section "Interpreter"
echo "  path    : $(command -v "$PY")"
"$PY" -c 'import sys; print("  version :", sys.version.split()[0], "->", sys.executable)'
if "$PY" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3,10) else 1)'; then
  printf '  \033[32mPASS\033[0m  python >= 3.10\n'
else
  printf '  \033[31mFAIL\033[0m  python < 3.10; metafmodel requires >= 3.10\n'
  ACTION=1
fi

section "NVIDIA driver"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/  gpu     : /'
  printf '  driver  : %s\n' "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' \r')"
  MAXCUDA="$(nvidia-smi -q 2>/dev/null | sed -n 's/.*CUDA Version *: *\([0-9][0-9.]*\).*/\1/p' | head -n1)"
  printf '  max CUDA: %s\n' "${MAXCUDA:-(not reported by this nvidia-smi build)}"
else
  printf '  \033[31mFAIL\033[0m  no working nvidia-smi; E3 server mode requires an NVIDIA GPU\n'
  MAXCUDA=""
  ACTION=1
fi

section "PyTorch and CUDA pairing"
# The torch/toolkit logic lives in Python so the verdict is computed from the
# real API rather than from parsing version strings in bash.
if ! "$PY" - <<PY
import sys
ok = True
yellow = "\033[33m"; red = "\033[31m"; green = "\033[32m"; off = "\033[0m"

try:
    import torch
except Exception:
    print(f"  {red}FAIL{off}  torch is not installed in this interpreter")
    raise SystemExit(1)

print(f"  torch   : {torch.__version__}")
if not torch.cuda.is_available():
    print(f"  {red}FAIL{off}  torch.cuda.is_available() is False; E3 server mode refuses to fall back to CPU")
    raise SystemExit(1)

count = torch.cuda.device_count()
print(f"  {green}PASS{off}  CUDA available; {count} device(s), runtime {torch.version.cuda}")
for index in range(count):
    print(f"        cuda:{index} = {torch.cuda.get_device_name(index)}")

if torch.cuda.is_bf16_supported():
    print(f"  {green}PASS{off}  bf16 supported (required; e3_server is bf16-mixed)")
else:
    print(f"  {red}FAIL{off}  bf16 NOT supported by this torch/device pairing")
    ok = False

# A wheel may carry sm_86 cubins for an sm_89 device: CUDA guarantees binary
# compatibility within a major version for equal-or-later minor versions. So
# look for any compiled arch with the same major and a minor <= the device's,
# rather than demanding an exact sm_XY match.
archs = {a.replace("sm_", "").split("a")[0] for a in torch.cuda.get_arch_list()}
compiled = sorted(a for a in archs if a.isdigit() and len(a) >= 2)
major, minor = torch.cuda.get_device_capability(0)
usable = [a for a in compiled if int(a[:-1]) == major and int(a[-1]) <= minor]
print(f"  device  : compute capability sm_{major}{minor}")
print(f"  wheel   : {', '.join('sm_' + a for a in compiled) or 'unknown'}")
if not compiled:
    print(f"  {yellow}WARN{off}  wheel reports no compiled architectures; cannot check sm_{major}{minor}")
elif usable:
    best = max(usable, key=lambda a: int(a[-1]))
    note = "" if int(best[-1]) == minor else f" (sm_{best} cubin, binary-compatible)"
    print(f"  {green}PASS{off}  sm_{major}{minor} is covered by the wheel{note}")
else:
    print(f"  {red}FAIL{off}  no compiled arch in the wheel covers sm_{major}{minor}; install a matching CUDA build")
    ok = False

raise SystemExit(0 if ok else 1)
PY
then
  ACTION=1
fi

section "Python dependencies (requirements-server.txt)"
if [ ! -f requirements-server.txt ]; then
  printf '  \033[31mFAIL\033[0m  requirements-server.txt is missing from the repository root\n'
  ACTION=1
elif ! "$PY" - <<'PY'
import importlib.metadata as md
import re
from pathlib import Path

red = "\033[31m"; green = "\033[32m"; off = "\033[0m"
mismatched = []
for line in Path("requirements-server.txt").read_text(encoding="utf-8").splitlines():
    line = line.split("#")[0].strip()
    if not line or line.startswith("-"):
        continue
    name, _, want = line.partition("==")
    try:
        have = md.version(name)
    except md.PackageNotFoundError:
        mismatched.append((name, want, "not installed"))
        continue
    # Wheels carry a local segment (2.8.0+cu128). The release part is what the
    # pin constrains, and the exact CUDA build is chosen by the operator.
    if want and have.split("+")[0] != want:
        mismatched.append((name, want, have))

if mismatched:
    for name, want, have in mismatched:
        print(f"  {red}FAIL{off}  {name}: require {want}, found {have}")
    raise SystemExit(1)
print(f"  {green}PASS{off}  every pinned dependency matches")
PY
then
  ACTION=1
fi

section "Repository importability"
if "$PY" -c 'import semg_fm, semg_h5' >/dev/null 2>&1; then
  printf '  \033[32mPASS\033[0m  semg_fm / semg_h5 import from the checkout\n'
else
  printf '  \033[33mNOTE\033[0m  semg_fm / semg_h5 are not importable from here.\n'
  printf '        Scripts call them by path, so this is not fatal. To install: python -m pip install -e .\n'
fi

section "Summary"
if [ "$ACTION" -eq 0 ]; then
  printf '  \033[32mEnvironment is ready.\033[0m\n\n'
  echo "  next:"
  echo "    export E3_DATA_ROOT=/srv/datasets/meta_canonical"
  echo "    export E3_RUN_ROOT=/srv/experiments/e3"
  echo "    bash scripts/verify_server.sh \"\$E3_DATA_ROOT\""
  exit 0
fi

cat <<'EOF'

  Action required. Nothing was installed or modified.

  PyTorch must come from the CUDA wheel index matching this driver. Do NOT let
  a bare `pip install -r requirements-server.txt` resolve torch for you: pip
  will pick a default wheel that may not match the driver above.

    1) install the CUDA build of torch first, for example on a CUDA 12.8 host:

         python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

       (for an older driver choose the matching index, e.g. cu126 / cu121.)

    2) install the remaining pinned dependencies:

         python -m pip install -r requirements-server.txt

    3) re-run this check:

         bash scripts/setup_server.sh

  If this host already has a working CUDA environment for other projects, create
  a separate virtual environment rather than upgrading the shared one.
EOF
exit 1
