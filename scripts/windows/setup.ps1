# Offline Windows setup for foundation-model-eval. No dataset download and no GPU.
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $root
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    py -3.12 -m venv .venv
}
$py = Join-Path $root ".venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install -r requirements.lock.txt
& $py -m pip install --no-deps --no-build-isolation -e .
& $py -m pip check
& $py -m foundation_eval.run doctor
Write-Host "Offline setup complete. Next: scripts\windows\check.ps1"
