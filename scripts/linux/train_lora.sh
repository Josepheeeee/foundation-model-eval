#!/usr/bin/env bash
# Four-GPU LoRA SFT with LlamaFactory. Usage: scripts/linux/train_lora.sh CONFIG.yaml
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
CONFIG="${1:?usage: train_lora.sh CONFIG.yaml}"
if [ ! -f "$CONFIG" ]; then echo "missing config: $CONFIG" >&2; exit 2; fi
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$GPU_COUNT" -lt 4 ]; then echo "need 4 GPUs, found $GPU_COUNT" >&2; exit 2; fi
. .venv-train/bin/activate
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export FORCE_TORCHRUN=1
export NPROC_PER_NODE=4
mkdir -p logs
LOG="logs/$(basename "$CONFIG" .yaml)-$(date -u +%Y%m%dT%H%M%SZ).log"
echo "Training with $CONFIG; four GPUs; log: $LOG"
llamafactory-cli train "$CONFIG" 2>&1 | tee "$LOG"
test -s "$LOG"
