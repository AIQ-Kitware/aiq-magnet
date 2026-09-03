#!/usr/bin/env bash
# Developer-only end-to-end smoke test for the SmolLM example.
#
# This intentionally assumes a developer machine with an NVIDIA GPU, Docker,
# MAGNET + its leasing extra, and an initialized infer-stack backend. It is not
# a CI entry point. The CI workflow exercises only `./run.sh --mock`.
set -euo pipefail

usage() {
    cat <<'EOF_USAGE'
Usage: ./test.sh [--release]

Run all four developer smoke variants:
  1. real endpoints + containerized nodes
  2. mock endpoints + containerized nodes
  3. real endpoints + host nodes
  4. mock endpoints + host nodes

Options:
  --release   Before testing, release every active infer-stack lease and evict
              idle deployments. This is destructive to other work using the
              same infer-stack daemon and is intended for a dedicated developer
              machine.
  -h, --help  Show this help and exit.

Environment:
  SMOLLM_TEST_RUNS  Artifact root (default: runs/dev-smoke-<timestamp>)
  SMOLLM_CATALOG    Optional real-model catalog forwarded to run.sh
EOF_USAGE
}

release_before_start=0
while (($#)); do
    case "$1" in
        --release)
            release_before_start=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown option: %s\n\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
root="${SMOLLM_TEST_RUNS:-$here/runs/dev-smoke-$stamp}"
mkdir -p "$root"
smoke_started=0
summary_printed=0
had_failure=0
lease_pool_dirty=0

variant_names=(
    real-container
    mock-container
    real-host
    mock-host
)
variant_endpoint_modes=(real mock real mock)
variant_node_modes=(container container host host)
declare -A variant_status
for name in "${variant_names[@]}"; do
    variant_status["$name"]='NOT RUN'
done

active_lease_rows() {
    infer-stack leases --json | python -c '
import json
import sys
payload = json.load(sys.stdin)
for lease in payload.get("leases", []):
    if str(lease.get("state", "")).lower().endswith("active"):
        endpoints = ",".join(lease.get("endpoints") or [])
        print("%s\t%s\t%s" % (lease["id"], lease.get("owner", "?"), endpoints))
'
}

require_clean_lease_pool() {
    local context="$1"
    local active
    active="$(active_lease_rows)"
    if [[ -z "$active" ]]; then
        return 0
    fi

    printf '\nSmolLM smoke test refuses to continue: active infer-stack leases %s.\n' "$context" >&2
    printf 'The developer smoke test expects exclusive use of the lease pool so a stale\n' >&2
    printf 'lease cannot turn the next serial variant into a capacity wait.\n\n' >&2
    infer-stack leases >&2 || true
    printf '\nRun ./test.sh --release to clear the infer-stack pool first, or release\n' >&2
    printf 'the specific leases yourself:\n' >&2
    while IFS=$'\t' read -r lease_id owner endpoints; do
        [[ -n "$lease_id" ]] || continue
        printf '  infer-stack release %q --yes --evict  # owner=%s endpoints=%s\n' \
            "$lease_id" "$owner" "$endpoints" >&2
    done <<< "$active"
    return 1
}

release_lease_pool() {
    printf '\n--release: clearing infer-stack leases and idle deployments before testing\n'
    infer-stack release --all --yes --evict
    require_clean_lease_pool 'after --release cleanup'
}

print_summary() {
    local exit_status="${1:-0}"
    ((summary_printed == 0)) || return 0
    summary_printed=1

    printf '\n============================================================\n'
    printf 'SmolLM developer smoke variants\n'
    printf '============================================================\n'
    local i name
    for i in "${!variant_names[@]}"; do
        name="${variant_names[$i]}"
        printf '%d. %-16s endpoints=%-4s nodes=%-9s %-18s %s\n' \
            "$((i + 1))" \
            "$name" \
            "${variant_endpoint_modes[$i]}" \
            "${variant_node_modes[$i]}" \
            "${variant_status[$name]}" \
            "$root/$name"
    done
    printf 'Artifacts: %s\n' "$root"
    if ((exit_status == 0 && had_failure == 0)); then
        printf 'Overall: PASS\n'
    else
        printf 'Overall: FAIL\n'
    fi
}

finish() {
    local status=$?
    set +e
    if ((status != 0 && smoke_started == 1)); then
        local active
        active="$(active_lease_rows 2>/dev/null || true)"
        if [[ -n "$active" ]]; then
            printf '\nActive infer-stack leases remain after the failed/interrupted smoke test:\n' >&2
            printf '%s\n' "$active" >&2
            printf 'Review them with: infer-stack leases\n' >&2
        fi
    fi
    if ((had_failure != 0 && status == 0)); then
        status=1
    fi
    print_summary "$status"
    trap - EXIT
    exit "$status"
}
trap finish EXIT

run_variant() {
    local name="$1"
    shift
    printf '\n============================================================\n'
    printf 'SmolLM developer smoke test: %s\n' "$name"
    printf 'output: %s\n' "$root/$name"
    printf 'command: ./run.sh'
    printf ' %q' "$@"
    printf '\n============================================================\n'

    local run_status=0
    if SMOLLM_RUNS="$root/$name" "$here/run.sh" "$@"; then
        run_status=0
    else
        run_status=$?
        variant_status["$name"]="FAIL (rc=$run_status)"
        had_failure=1
    fi

    # `infer-stack run` releases in a finally block. Catch a regression or an
    # interrupted child here, before another variant tries to use the GPUs.
    if ! require_clean_lease_pool "after $name"; then
        variant_status["$name"]='FAIL (lease leak)'
        had_failure=1
        lease_pool_dirty=1
        return 1
    fi

    if ((run_status == 0)); then
        variant_status["$name"]='PASS'
        return 0
    fi
    return "$run_status"
}

printf 'SmolLM developer smoke test\n'
printf 'Artifacts will be kept under: %s\n' "$root"
printf 'This test expects a usable NVIDIA GPU and Docker.\n'

# Keep help in the smoke test so argument parsing remains usable even if the
# execution prerequisites are broken.
"$here/run.sh" --help >/dev/null

if ((release_before_start == 1)); then
    release_lease_pool
else
    # Default is deliberately non-destructive. Turn stale capacity into an
    # immediate diagnostic instead of modifying another developer's work.
    require_clean_lease_pool 'before starting'
fi
smoke_started=1

# Container modes are adjacent on purpose. The first may build the node image;
# the second should reuse it. In particular, switching from the real catalog to
# the mock catalog must not invalidate the image.
if ! run_variant real-container; then
    :
fi
if ((lease_pool_dirty == 0)); then
    if ! run_variant mock-container --mock; then
        :
    fi
else
    variant_status[mock-container]='SKIPPED (lease pool dirty)'
fi

# Host execution exercises the same real/mock endpoint choices without the
# node-image boundary.
if ((lease_pool_dirty == 0)); then
    if ! run_variant real-host --no-container; then
        :
    fi
else
    variant_status[real-host]='SKIPPED (lease pool dirty)'
fi
if ((lease_pool_dirty == 0)); then
    if ! run_variant mock-host --mock --no-container; then
        :
    fi
else
    variant_status[mock-host]='SKIPPED (lease pool dirty)'
fi

if ((had_failure != 0)); then
    exit 1
fi
