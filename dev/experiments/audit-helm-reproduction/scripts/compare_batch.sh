#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

audit::set_defaults

MANIFEST="${1:-${AUDIT_ROOT}/configs/generated/smoke_manifest.generated.yaml}"
audit::require_file "$MANIFEST"

"$AIQ_PYTHON" "${AUDIT_ROOT}/python/compare_batch.py" \
    --manifest "$MANIFEST"
