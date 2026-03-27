#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

audit::set_defaults

RUN_ENTRY="${1:-narrative_qa:model=eleutherai/pythia-6.9b,data_augmentation=canonical}"
BASE_NAME="${2:-audit-narrative-pythia}"
OUTPUT_DIR="${3:-${AUDIT_ROOT}/configs/generated}"
if [[ $# -gt 0 ]]; then
    shift
fi
if [[ $# -gt 0 ]]; then
    shift
fi
if [[ $# -gt 0 ]]; then
    shift
fi

mkdir -p "$OUTPUT_DIR"

R1_OUTPUT="${OUTPUT_DIR}/${BASE_NAME}_r1.yaml"
R2_OUTPUT="${OUTPUT_DIR}/${BASE_NAME}_r2.yaml"

"$AIQ_PYTHON" "${AUDIT_ROOT}/python/make_manifest.py" \
    --manifest-type single \
    --run-entry "$RUN_ENTRY" \
    --experiment-name "${BASE_NAME}-r1" \
    --suite "${BASE_NAME}-r1" \
    --description "Repeat run 1 for ${RUN_ENTRY}" \
    --output "$R1_OUTPUT" \
    "$@"

"$AIQ_PYTHON" "${AUDIT_ROOT}/python/make_manifest.py" \
    --manifest-type single \
    --run-entry "$RUN_ENTRY" \
    --experiment-name "${BASE_NAME}-r2" \
    --suite "${BASE_NAME}-r2" \
    --description "Repeat run 2 for ${RUN_ENTRY}" \
    --output "$R2_OUTPUT" \
    "$@"

printf 'Wrote repeat manifests:\n'
printf '  %s\n' "$R1_OUTPUT"
printf '  %s\n' "$R2_OUTPUT"
