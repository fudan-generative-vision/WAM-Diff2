#!/usr/bin/env bash

# Accelerate/DeepSpeed ZeRO-2 distillation launcher.
#
# Examples:
#   bash scripts/train_distil.sh
#   NUM_GPUS=4 bash scripts/train_distil.sh
#   EXPERIMENT_CONFIG=/path/to/experiment.yaml bash scripts/train_distil.sh

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${REPO_ROOT}/configs/accelerate/deepspeed_zero2.yaml}"
EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-${REPO_ROOT}/configs/distillation/distil.yaml}"
NUM_GPUS="${NUM_GPUS:-8}"

exec accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_processes "${NUM_GPUS}" \
  -m wam_diff.training.distil_accelerate \
  --config "${EXPERIMENT_CONFIG}" \
  "$@"
