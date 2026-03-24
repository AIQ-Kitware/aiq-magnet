#!/usr/bin/env bash
set -euo pipefail
set +x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

audit::set_defaults

MANIFEST="${1:-${AUDIT_ROOT}/configs/generated/smoke_manifest.generated.yaml}"

if [[ ! -f "$MANIFEST" ]]; then
    "${AUDIT_ROOT}/scripts/make_smoke_manifest.sh" "$MANIFEST"
fi

"${AUDIT_ROOT}/scripts/run_from_manifest.sh" "$MANIFEST"
