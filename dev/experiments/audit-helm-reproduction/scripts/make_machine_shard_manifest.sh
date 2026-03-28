#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

audit::set_defaults

MACHINE_NAME="${1:?machine name required, e.g. namek or yardrat}"
SHARD_INDEX="${2:?shard index required}"
NUM_SHARDS="${3:?num shards required}"
OUTPATH="${4:-${AUDIT_ROOT}/configs/generated/${MACHINE_NAME}.generated.yaml}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ $# -gt 0 ]]; then shift; fi
if [[ $# -gt 0 ]]; then shift; fi
if [[ $# -gt 0 ]]; then shift; fi

"$AIQ_PYTHON" "${AUDIT_ROOT}/python/build_repro_manifest.py" \
  --output "$OUTPATH" \
  --experiment-name "audit-${MACHINE_NAME}" \
  --suite "audit-${MACHINE_NAME}" \
  --single-gpu \
  --devices 0 \
  --num-shards "$NUM_SHARDS" \
  --shard-index "$SHARD_INDEX" \
  "$@"

printf 'Wrote machine shard manifest: %s\n' "$OUTPATH"
printf 'Wrote selection sidecar: %s.selection.yaml\n' "$OUTPATH"
