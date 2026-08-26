#!/usr/bin/env bash
# Typecheck the Lean statements the examples point at.
#
# They import Mathlib, so they need a Lean project that has one built. Rather
# than carry a lake project and a Mathlib pin in this repository, borrow an
# existing one:
#
#   MAGNET_LEAN_PROJECT=~/code/mathlib-project ./check_lean.sh
#
# A `sorry` is reported rather than treated as a failure: a statement can be
# well-formed and unproved, and which of the two it is belongs in the output.
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${MAGNET_LEAN_PROJECT:-}"

if [ -z "$PROJECT" ]; then
    echo "set MAGNET_LEAN_PROJECT to a lean project with mathlib built" >&2
    exit 2
fi
if [ ! -e "$PROJECT/lakefile.toml" ] && [ ! -e "$PROJECT/lakefile.lean" ]; then
    echo "no lakefile in $PROJECT" >&2
    exit 2
fi
if ! find "$PROJECT/.lake/packages/mathlib/.lake/build/lib" -name 'Mathlib.olean' \
        -print -quit 2>/dev/null | grep -q .; then
    echo "mathlib is not built in $PROJECT (try: lake exe cache get)" >&2
    exit 2
fi

status=0
for fpath in "$HERE"/*/*.lean; do
    label="$(basename "$(dirname "$fpath")")/$(basename "$fpath")"
    printf '%-40s ' "$label"
    if output="$(cd "$PROJECT" && lake env lean "$fpath" 2>&1)"; then
        sorries="$(printf '%s' "$output" | grep -c "declaration uses .sorry." || true)"
        echo "ok (${sorries} sorry)"
    else
        echo "FAILED"
        printf '%s\n' "$output"
        status=1
    fi
done

exit "$status"
