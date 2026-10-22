# Offline checks: contracts, manifest checks, demo and tests. Never calls a model.
$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Run scripts\windows\setup.ps1 first" }
& $py -m foundation_eval.run doctor
& $py -m foundation_eval.run check --offline
& $py -m pytest -q -p no:cacheprovider
& $py -m foundation_eval.run demo
Write-Host "Offline checks complete. Demo artifacts are under runs\ and are explicitly not benchmark results."
