#!/usr/bin/env bash
# Inference environment, isolated from training. Run on the GPU serving host.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
python3.12 -m venv .venv-infer
. .venv-infer/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock.txt -r requirements-infer.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
python -c "import vllm; print('vllm', vllm.__version__)"
