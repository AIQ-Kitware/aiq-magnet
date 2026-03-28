#!/usr/bin/env bash
set -euo pipefail
set +x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
audit::set_defaults

AUDIT_PYTHON_DIR="${AUDIT_ROOT}/python"
AUDIT_REPORT_ROOT="${AUDIT_ROOT}/reports"

"$AIQ_PYTHON" "$AUDIT_PYTHON_DIR/rebuild_core_report_from_index.py" "$@"
