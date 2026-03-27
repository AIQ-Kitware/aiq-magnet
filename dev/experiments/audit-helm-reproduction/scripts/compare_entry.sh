#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

audit::set_defaults

RESULTS_DPATH="${1:?need kwdagger results root}"
RUN_ENTRY="${2:?need run entry}"
REPORT_DPATH="${3:-${AUDIT_ROOT}/reports/pairwise}"

kwdg_json="$("$AIQ_PYTHON" "${AUDIT_ROOT}/python/resolve_run.py" \
  --mode kwdg \
  --results-dpath "$RESULTS_DPATH" \
  --run-entry "$RUN_ENTRY")"

historic_json="$("$AIQ_PYTHON" "${AUDIT_ROOT}/python/resolve_run.py" \
  --mode historic \
  --precomputed-root "$HELM_PRECOMPUTED_ROOT" \
  --run-entry "$RUN_ENTRY")"

kwdg_status="$(python - <<'PY' "$kwdg_json"
import json, sys
obj = json.loads(sys.argv[1])
print(obj.get('artifact_status', 'unknown'))
PY
)"

kwdg_run_dir="$(python - <<'PY' "$kwdg_json"
import json, sys
obj = json.loads(sys.argv[1])
print(obj.get('run_dir') or '')
PY
)"

if [[ "$kwdg_status" != "ready" ]]; then
  echo "Resolved kwdagger job, but HELM run artifacts are not ready." >&2
  python - <<'PY' "$kwdg_json"
import json, sys
obj = json.loads(sys.argv[1])
print(json.dumps(obj, indent=2))
PY
  exit 1
fi

historic_run_dir="$(python - <<'PY' "$historic_json"
import json, sys
obj = json.loads(sys.argv[1])
matches = obj.get('matches', [])
if not matches:
    raise SystemExit('No historic matches found')
ready = [m for m in matches if m.get('artifact_status') == 'ready']
target = ready[-1] if ready else matches[-1]
print(target['run_dir'])
PY
)"

"${AUDIT_ROOT}/scripts/compare_pair.sh" \
  "$historic_run_dir" \
  "$kwdg_run_dir" \
  "$REPORT_DPATH"
