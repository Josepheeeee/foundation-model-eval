#!/usr/bin/env bash
# Collect a frozen evaluation matrix for one served checkpoint.
# Usage:
#   MODEL_LABEL=B1 TRAINING_SEED=42 scripts/linux/run_eval_matrix.sh
#   MODEL_LABEL=B1 TRAINING_SEED=42 PROTOCOL=chat scripts/linux/run_eval_matrix.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
. .venv/bin/activate
LABEL="${MODEL_LABEL:?set MODEL_LABEL (B0/B1/B2 or the 4B replication)}"
SEED="${TRAINING_SEED:-42}"
PROTOCOL="${PROTOCOL:-completion}"
TASKS="${TASKS:-gsm8k svamp mmlu_pro bbh ifeval humaneval_plus}"
MAX_LIMIT="${MAX_LIMIT:-}"
LIMIT_ARG="${MAX_LIMIT:+--limit $MAX_LIMIT}"

for task in $TASKS; do
  if [ ! -f "data/manifests/freeze/${task}-${PROTOCOL}.json" ]; then
    python -m foundation_eval.run freeze --task "$task" --protocol "$PROTOCOL"
  fi
  python -m foundation_eval.run collect --task "$task" --split test --protocol "$PROTOCOL" \
    --model-label "$LABEL" --training-seed "$SEED" $LIMIT_ARG
done

if [ "$PROTOCOL" = "completion" ]; then
  for length in 4096 8192 16384; do
    for ruler_task in niah_single_1 niah_multikey_1 vt; do
      python -m foundation_eval.run collect --task ruler --split diagnostic \
        --samples "data/processed/ruler/${length}_${ruler_task}.jsonl" \
        --model-label "$LABEL" --training-seed "$SEED"
    done
  done
fi

python scripts/validate_runs.py --label "$LABEL"
