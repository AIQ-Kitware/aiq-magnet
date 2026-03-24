#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

audit::set_defaults

OUTPUT="${1:-${AUDIT_ROOT}/configs/generated/apples_manifest.generated.yaml}"
if [[ $# -gt 0 ]]; then
    shift
fi
mkdir -p "$(dirname "$OUTPUT")"

"$AIQ_PYTHON" "${AUDIT_ROOT}/python/make_manifest.py" \
    --manifest-type apples \
    --experiment-name audit-smoke-apples \
    --suite audit-smoke-apples \
    --output "$OUTPUT" \
    "$@"

printf 'Wrote apples-to-apples manifest: %s\n' "$OUTPUT"
