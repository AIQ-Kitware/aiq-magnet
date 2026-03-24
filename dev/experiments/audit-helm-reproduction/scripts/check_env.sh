#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

audit::set_defaults

printf 'Audit environment\n'
printf '=================\n'
audit::print_env
printf '\n'

for path_var in AIQ_MAGNET_ROOT HELM_EDITABLE_ROOT HELM_PRECOMPUTED_ROOT; do
    path="${!path_var}"
    if [[ ! -e "$path" ]]; then
        printf '%s does not exist: %s\n' "$path_var" "$path" >&2
        exit 1
    fi
done

if [[ ! -d "$AUDIT_RESULTS_ROOT" ]]; then
    if mkdir -p "$AUDIT_RESULTS_ROOT" 2>/dev/null; then
        :
    else
        printf 'Warning: unable to create AUDIT_RESULTS_ROOT: %s\n' "$AUDIT_RESULTS_ROOT" >&2
        printf 'The external runner should create this path or override AUDIT_RESULTS_ROOT.\n' >&2
    fi
fi
if ! command -v kwdagger >/dev/null 2>&1; then
    printf 'kwdagger not found on PATH\n' >&2
    exit 1
fi

if ! command -v helm-run >/dev/null 2>&1; then
    printf 'helm-run not found on PATH\n' >&2
    exit 1
fi

if ! command -v "$AIQ_PYTHON" >/dev/null 2>&1; then
    printf 'Configured AIQ_PYTHON not found on PATH: %s\n' "$AIQ_PYTHON" >&2
    exit 1
fi

if ! "$AIQ_PYTHON" -c "import magnet" >/dev/null 2>&1; then
    printf 'Unable to import magnet from %s\n' "$AIQ_PYTHON" >&2
    exit 1
fi

printf 'Environment looks good.\n'
