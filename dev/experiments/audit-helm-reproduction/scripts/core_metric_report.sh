#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

audit::set_defaults

"$AIQ_PYTHON" "${AUDIT_ROOT}/python/core_metric_report.py" "$@"
