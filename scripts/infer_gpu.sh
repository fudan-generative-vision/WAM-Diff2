#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
NUM_GPUS="${NUM_GPUS:-1}"

exec torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node="${NUM_GPUS}" \
  --module wam_diff.inference.gpu \
  "$@"
