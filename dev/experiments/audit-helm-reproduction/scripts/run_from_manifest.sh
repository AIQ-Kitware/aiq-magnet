#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

audit::set_defaults

MANIFEST="${1:-${AUDIT_ROOT}/configs/smoke_manifest.yaml}"
audit::require_file "$MANIFEST"

"${AUDIT_ROOT}/scripts/check_env.sh" >/dev/null

EXPERIMENT_NAME="$("$AIQ_PYTHON" "${AUDIT_ROOT}/python/render_schedule_params.py" \
    --manifest "$MANIFEST" --mode experiment_name)"
RESULT_DPATH="$("$AIQ_PYTHON" "${AUDIT_ROOT}/python/render_schedule_params.py" \
    --manifest "$MANIFEST" --mode result_dpath)"
PARAMS="$("$AIQ_PYTHON" "${AUDIT_ROOT}/python/render_schedule_params.py" \
    --manifest "$MANIFEST" --mode params)"
BACKEND="$("$AIQ_PYTHON" "${AUDIT_ROOT}/python/render_schedule_params.py" \
    --manifest "$MANIFEST" --mode backend)"
TMUX_WORKERS="$("$AIQ_PYTHON" "${AUDIT_ROOT}/python/render_schedule_params.py" \
    --manifest "$MANIFEST" --mode tmux_workers)"
DEVICES="$("$AIQ_PYTHON" "${AUDIT_ROOT}/python/render_schedule_params.py" \
    --manifest "$MANIFEST" --mode devices)"

mkdir -p "$RESULT_DPATH"

printf 'Launching experiment: %s\n' "$EXPERIMENT_NAME"
printf 'Results root: %s\n' "$RESULT_DPATH"
printf 'Backend: %s\n' "$BACKEND"
printf 'Devices: %s\n' "$DEVICES"
printf 'tmux_workers: %s\n' "$TMUX_WORKERS"

kwdagger schedule \
    --params="$PARAMS" \
    --devices="$DEVICES" \
    --tmux_workers="$TMUX_WORKERS" \
    --root_dpath="$RESULT_DPATH" \
    --backend="$BACKEND" \
    --skip_existing=1 \
    --run=1
