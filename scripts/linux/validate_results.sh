#!/usr/bin/env bash
# Validate all recorded runs: completeness, protocol identity and matrix output.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
. .venv/bin/activate
python -m foundation_eval.run check
python scripts/validate_runs.py "$@"
