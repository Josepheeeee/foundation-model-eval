#!/usr/bin/env bash
# Merge a LoRA adapter for serving. Usage: scripts/linux/merge_lora.sh CONFIG.yaml
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
CONFIG="${1:?usage: merge_lora.sh CONFIG.yaml}"
. .venv-train/bin/activate
llamafactory-cli export "$CONFIG"
