#!/usr/bin/env bash
set -euo pipefail

AUDIT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUDIT_ROOT="$(cd "${AUDIT_SCRIPT_DIR}/.." && pwd)"

audit::set_defaults() {
    export AIQ_MAGNET_ROOT="${AIQ_MAGNET_ROOT:-$HOME/code/aiq-magnet}"
    export HELM_EDITABLE_ROOT="${HELM_EDITABLE_ROOT:-$HOME/code/helm}"
    export AIQ_PYTHON="${AIQ_PYTHON:-python}"
    export HELM_PRECOMPUTED_ROOT="${HELM_PRECOMPUTED_ROOT:-/data/crfm-helm-public}"
    export AUDIT_RESULTS_ROOT="${AUDIT_RESULTS_ROOT:-/data/crfm-helm-audit}"
    export AUDIT_DEFAULT_MAX_EVAL_INSTANCES="${AUDIT_DEFAULT_MAX_EVAL_INSTANCES:-100}"
    export AUDIT_DEFAULT_TMUX_WORKERS="${AUDIT_DEFAULT_TMUX_WORKERS:-2}"
}

audit::print_env() {
    printf 'AIQ_MAGNET_ROOT=%s\n' "$AIQ_MAGNET_ROOT"
    printf 'HELM_EDITABLE_ROOT=%s\n' "$HELM_EDITABLE_ROOT"
    printf 'AIQ_PYTHON=%s\n' "$AIQ_PYTHON"
    printf 'HELM_PRECOMPUTED_ROOT=%s\n' "$HELM_PRECOMPUTED_ROOT"
    printf 'AUDIT_RESULTS_ROOT=%s\n' "$AUDIT_RESULTS_ROOT"
    printf 'AUDIT_DEFAULT_MAX_EVAL_INSTANCES=%s\n' "$AUDIT_DEFAULT_MAX_EVAL_INSTANCES"
    printf 'AUDIT_DEFAULT_TMUX_WORKERS=%s\n' "$AUDIT_DEFAULT_TMUX_WORKERS"
}

audit::require_file() {
    local path="$1"
    if [[ ! -f "$path" ]]; then
        printf 'Missing required file: %s\n' "$path" >&2
        exit 1
    fi
}
