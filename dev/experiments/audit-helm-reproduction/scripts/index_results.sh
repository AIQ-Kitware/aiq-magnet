#!/usr/bin/env bash
set -euo pipefail
set +x

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/common.sh"
audit::set_defaults

fallback_host="${AUDIT_FALLBACK_HOST:-}"
AUDIT_PYTHON_DIR="${AUDIT_ROOT}/python"
AUDIT_REPORT_ROOT="${AUDIT_ROOT}/reports"

"$AIQ_PYTHON" "$AUDIT_PYTHON_DIR/index_results.py" \
    --results-root "${1:-$AUDIT_RESULTS_ROOT}" \
    --report-dpath "${2:-$AUDIT_REPORT_ROOT/indexes}" \
    ${fallback_host:+--fallback-host "$fallback_host"}
