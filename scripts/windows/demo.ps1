# Re-run only the scripted offline demo (no network, no model).
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Run scripts\windows\setup.ps1 first" }
& $py -m foundation_eval.run demo
