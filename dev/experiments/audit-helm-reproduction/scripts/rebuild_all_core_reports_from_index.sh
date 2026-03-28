#!/usr/bin/env bash
set -euo pipefail
set +x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
audit::set_defaults

AUDIT_PYTHON_DIR="${AUDIT_ROOT}/python"

"$AIQ_PYTHON" "$AUDIT_PYTHON_DIR/rebuild_all_core_reports_from_index.py" "$@"
