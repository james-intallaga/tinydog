$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Bootstrap = Join-Path $ProjectDir "scripts\tinydog.py"

if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 $Bootstrap @args
    exit $LASTEXITCODE
}

if (Get-Command python -ErrorAction SilentlyContinue) {
    & python $Bootstrap @args
    exit $LASTEXITCODE
}

Write-Error "Python 3.10 or newer is required. Install Python, then run this command again."
