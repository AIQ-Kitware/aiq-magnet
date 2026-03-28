#!/usr/bin/env bash
set -euo pipefail
set +x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

audit::set_defaults

MANIFEST="${1:-${AUDIT_ROOT}/configs/smoke_manifest.yaml}"
audit::require_file "$MANIFEST"

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
PRECOMPUTED_ROOT="$("$AIQ_PYTHON" "${AUDIT_ROOT}/python/render_schedule_params.py" \
    --manifest "$MANIFEST" --mode precomputed_root)"

if [[ -n "$PRECOMPUTED_ROOT" ]]; then
    AUDIT_REQUIRE_PRECOMPUTED_ROOT=1 "${AUDIT_ROOT}/scripts/check_env.sh" >/dev/null
else
    AUDIT_REQUIRE_PRECOMPUTED_ROOT=0 "${AUDIT_ROOT}/scripts/check_env.sh" >/dev/null
fi

mkdir -p "$RESULT_DPATH"

printf 'Launching experiment: %s\n' "$EXPERIMENT_NAME"
printf 'Results root: %s\n' "$RESULT_DPATH"
printf 'Backend: %s\n' "$BACKEND"
printf 'Devices: %s\n' "$DEVICES"
printf 'tmux_workers: %s\n' "$TMUX_WORKERS"

QUEUE_NAME="$(printf 'audit-%s' "$EXPERIMENT_NAME" | tr -c 'A-Za-z0-9._-' '-')"
printf 'queue_name: %s\n' "$QUEUE_NAME"
printf 'Schedule params:\n%s\n' "$PARAMS"

kwdagger schedule \
    --queue_name="$QUEUE_NAME" \
    --params="$PARAMS" \
    --devices="$DEVICES" \
    --tmux_workers="$TMUX_WORKERS" \
    --root_dpath="$RESULT_DPATH" \
    --backend="$BACKEND" \
    --skip_existing=1 \
    --run=1
