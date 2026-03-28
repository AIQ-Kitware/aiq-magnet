#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

audit::set_defaults

MACHINE_NAME="${1:?machine name required, e.g. namek or yardrat}"
OUTPATH="${2:-${AUDIT_ROOT}/configs/generated/${MACHINE_NAME}.subset.generated.yaml}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ $# -gt 0 ]]; then shift; fi

"$AIQ_PYTHON" "${AUDIT_ROOT}/python/build_repro_manifest.py" \
  --output "$OUTPATH" \
  --experiment-name "audit-${MACHINE_NAME}-subset" \
  --suite "audit-${MACHINE_NAME}-subset" \
  --single-gpu \
  --devices 0 \
  "$@"

printf 'Wrote machine subset manifest: %s\n' "$OUTPATH"
printf 'Wrote selection sidecar: %s.selection.yaml\n' "$OUTPATH"
