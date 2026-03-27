#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

audit::set_defaults

RUN_A="${1:?need run A path}"
RUN_B="${2:?need run B path}"
REPORT_DPATH="${3:-${AUDIT_ROOT}/reports/pairwise}"

"$AIQ_PYTHON" "${AUDIT_ROOT}/python/compare_pair.py" \
    --run-a "$RUN_A" \
    --run-b "$RUN_B" \
    --report-dpath "$REPORT_DPATH"
