#!/usr/bin/env bash
set -euo pipefail

AUDIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"$AUDIT_ROOT/scripts/check_env.sh"
"$AUDIT_ROOT/scripts/make_smoke_manifest.sh"
"$AUDIT_ROOT/scripts/run_smoke.sh"
"$AUDIT_ROOT/scripts/compare_batch.sh"
