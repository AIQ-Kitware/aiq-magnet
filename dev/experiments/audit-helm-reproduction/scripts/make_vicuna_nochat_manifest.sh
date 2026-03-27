#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

audit::set_defaults

OUTPATH="${1:-${AUDIT_ROOT}/configs/generated/vicuna_nochat_overnight.generated.yaml}"
shift || true

"$AIQ_PYTHON" "${AUDIT_ROOT}/python/make_manifest.py" \
  --manifest-type vicuna_nochat \
  --output "$OUTPATH" \
  --experiment-name audit-vicuna-nochat-overnight \
  --suite audit-vicuna-nochat-overnight \
  "$@"

printf 'Wrote Vicuna no-chat manifest: %s\n' "$OUTPATH"
