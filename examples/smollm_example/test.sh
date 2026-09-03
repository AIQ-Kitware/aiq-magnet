#!/usr/bin/env bash
# Developer-only end-to-end smoke test for the SmolLM example.
#
# This intentionally assumes a developer machine with an NVIDIA GPU, Docker,
# MAGNET + its leasing extra, and an initialized infer-stack backend. It is not
# a CI entry point. The CI workflow exercises only `./run.sh --mock`.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
root="${SMOLLM_TEST_RUNS:-$here/runs/dev-smoke-$stamp}"
mkdir -p "$root"

run_variant() {
    local name="$1"
    shift
    printf '\n============================================================\n'
    printf 'SmolLM developer smoke test: %s\n' "$name"
    printf 'output: %s\n' "$root/$name"
    printf 'command: ./run.sh'
    printf ' %q' "$@"
    printf '\n============================================================\n'
    SMOLLM_RUNS="$root/$name" "$here/run.sh" "$@"
}

printf 'SmolLM developer smoke test\n'
printf 'Artifacts will be kept under: %s\n' "$root"
printf 'This test expects a usable NVIDIA GPU and Docker.\n'

# Keep help in the smoke test so argument parsing remains usable even if the
# execution prerequisites are broken.
"$here/run.sh" --help >/dev/null

# Container modes are adjacent on purpose. The first may build the node image;
# the second should reuse it. In particular, switching from the real catalog to
# the mock catalog must not invalidate the image.
run_variant real-container
run_variant mock-container --mock

# Host execution exercises the same real/mock endpoint choices without the
# node-image boundary.
run_variant real-host --no-container
run_variant mock-host --mock --no-container

printf '\nAll SmolLM developer smoke-test variants passed.\n'
printf 'Artifacts: %s\n' "$root"
