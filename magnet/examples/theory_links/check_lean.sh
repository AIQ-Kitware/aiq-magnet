#!/usr/bin/env bash
# Typecheck the Lean statements the examples point at.
#
# Plain Lean 4, no Mathlib and no lake project: each file states its own
# definitions, so `lean` from any toolchain is enough. A `sorry` warning is
# expected where a statement is well-formed and unproved; anything else is a
# real error.
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
status=0

for fpath in "$HERE"/*/*.lean; do
    printf '%-46s ' "$(basename "$(dirname "$fpath")")/$(basename "$fpath")"
    if output="$(lean "$fpath" 2>&1)"; then
        sorries="$(printf '%s' "$output" | grep -c 'uses .sorry.' || true)"
        echo "ok (${sorries} sorry)"
    else
        echo "FAILED"
        printf '%s\n' "$output"
        status=1
    fi
done

exit "$status"
