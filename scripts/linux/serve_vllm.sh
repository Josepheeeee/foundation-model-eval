#!/usr/bin/env bash
# Serve one base model or one merged LoRA checkpoint behind an OpenAI endpoint.
# Usage: MODEL=Qwen/Qwen3-8B-Base [ADAPTER=outputs/...] scripts/linux/serve_vllm.sh [port]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
. .venv-infer/bin/activate
MODEL="${MODEL:?set MODEL to the base model id or path}"
PORT="${1:-8000}"
TP="${TP:-4}"
SERVED="${SERVED:-target-model}"
ARGS=(serve "$MODEL" --served-model-name "$SERVED" --tensor-parallel-size "$TP"
      --max-model-len "${MAX_MODEL_LEN:-32768}" --seed 2026 --port "$PORT" --host 127.0.0.1)
if [ -n "${ADAPTER:-}" ]; then
  ARGS+=(--enable-lora --lora-modules "${SERVED}=${ADAPTER}")
fi
echo "Serving ${MODEL} on http://127.0.0.1:${PORT}/v1 (served name: ${SERVED})"
exec vllm "${ARGS[@]}"
