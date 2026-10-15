# PowerShell: run from the metafmodel repository root
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repo
if (Test-Path env:META_PYTHON) {
  & $env:META_PYTHON scripts/e3_oneclick.py local
} else {
  python scripts/e3_oneclick.py local
}
if ($LASTEXITCODE -ne 0) { throw "E3 local end-to-end run failed (exit $LASTEXITCODE)" }
