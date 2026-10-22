#!/usr/bin/env bash
# Download the public tasks and build the normalized sample files.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
. .venv/bin/activate
python -m pip install -r requirements-data.txt
python scripts/prepare_datasets.py
python -m foundation_eval.run prepare \
  --input data/raw/gsm8k_train.jsonl \
  --test data/raw/gsm8k_test.jsonl
python -m foundation_eval.run ruler-build
echo "Data ready. Review data/manifests/*.json before freezing."
