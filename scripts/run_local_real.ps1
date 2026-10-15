# Verify the full E3 training path on a small REAL dataset, locally.
#
#   .\scripts\run_local_real.ps1 -DataRoot D:\emg_data\real_mini
#
# Produces a real mini dataset on the server with scripts/export_real_mini.py,
# transfer it here, then run this to exercise data, sampling, model, training
# and checkpoint-restore on genuine signal -- before spending A800 hours.
[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string]$DataRoot,

  [int]$Steps = 50,
  [int]$ResumeSteps = 5,
  [int]$BatchSize = 4,
  [int]$NumWorkers = 0,
  [int]$ValEvery = 25,
  [int]$ValBatches = 5,

  # Run sections 1-3 only (data, sampling, model); skip training and checkpoint.
  [switch]$SkipTrain
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repo

$python = if ($env:E3_PYTHON) { $env:E3_PYTHON } elseif ($env:META_PYTHON) { $env:META_PYTHON } else { "python" }

if (-not (Test-Path $DataRoot)) {
  throw "data root does not exist: $DataRoot"
}
foreach ($required in @("splits.csv", "nature_corpus.csv")) {
  if (-not (Test-Path (Join-Path $DataRoot $required))) {
    throw "$required is missing from $DataRoot. Export a mini with scripts/export_real_mini.py first."
  }
}

$arguments = @(
  "scripts/verify_local_real.py",
  "--data-root", $DataRoot,
  "--steps", $Steps,
  "--resume-steps", $ResumeSteps,
  "--batch-size", $BatchSize,
  "--num-workers", $NumWorkers,
  "--val-every", $ValEvery,
  "--val-batches", $ValBatches
)
if ($SkipTrain) { $arguments += "--skip-train" }

Write-Host "[local-real] repository : $repo"
Write-Host "[local-real] data root  : $DataRoot"
Write-Host "[local-real] python     : $python"

& $python @arguments
if ($LASTEXITCODE -ne 0) {
  throw "local real verification FAILED (exit $LASTEXITCODE). See the report above."
}
Write-Host "[local-real] verification PASSED"
