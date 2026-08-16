#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
NUM_DEVICES="${NUM_DEVICES:-1}"

exec torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node="${NUM_DEVICES}" \
  --module wam_diff.inference.npu \
  "$@"
