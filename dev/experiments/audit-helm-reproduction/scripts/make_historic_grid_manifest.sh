#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

audit::set_defaults

OUTPATH="${1:-${AUDIT_ROOT}/configs/generated/historic_grid.generated.yaml}"
shift || true

"$AIQ_PYTHON" "${AUDIT_ROOT}/python/build_repro_manifest.py" \
  --output "$OUTPATH" \
  --experiment-name audit-historic-grid \
  --suite audit-historic-grid \
  "$@"

printf 'Wrote historic grid manifest: %s\n' "$OUTPATH"
printf 'Wrote selection sidecar: %s.selection.yaml\n' "$OUTPATH"
