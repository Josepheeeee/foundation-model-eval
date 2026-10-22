#!/usr/bin/env bash
# Base offline/data environment (CPU is enough for data preparation and checks).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3.12}"
"$PYTHON" -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
python -m foundation_eval.run doctor
echo "Base environment ready. Training: scripts/linux/install_train.sh; inference: scripts/linux/install_infer.sh"
