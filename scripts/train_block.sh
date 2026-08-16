#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/configs/training/block32.yaml}"
NUM_GPUS="${NUM_GPUS:-1}"

exec torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node="${NUM_GPUS}" \
  --module wam_diff.training.finetune \
  --config "${CONFIG_PATH}" \
  "$@"
