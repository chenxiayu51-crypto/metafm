param(
    [ValidateSet('verify','test','pretrain','finetune','evaluate')]
    [string]$Command='verify',
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$ExtraArgs
)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
$pythonExecutable=$env:META_PYTHON
if (-not $pythonExecutable) {
    $pythonExecutable=Join-Path $projectRoot '.venv/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $pythonExecutable)) {
        $pythonExecutable=Join-Path $projectRoot '../MetaEMGFM/.venv/Scripts/python.exe'
    }
}
if (-not (Test-Path -LiteralPath $pythonExecutable)) {throw 'Set META_PYTHON to a working Python executable.'}
Push-Location -LiteralPath $projectRoot
try {
    switch ($Command) {
        'verify' { & $pythonExecutable scripts/local_verify.py @ExtraArgs }
        'test' { & $pythonExecutable -m pytest tests -q @ExtraArgs }
        default { & $pythonExecutable "$Command.py" @ExtraArgs }
    }
    $commandExitCode=$LASTEXITCODE
} finally {Pop-Location}
exit $commandExitCode
