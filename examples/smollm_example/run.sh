#!/usr/bin/env bash
# Run the SmolLM example.
#
#   ./run.sh              simulated endpoints, no GPU and no weights
#   ./run.sh --gpu        real SmolLM2 135M / 360M on a GPU, via vLLM
#   ./run.sh --container  run the node commands in a container too
#
# The flags combine, and anything else is passed through to
# `magnet evaluate_new` -- so `./run.sh --dry_run=1` compiles the campaign
# without running it.
#
# The two catalogs declare the SAME two aliases, so which one is in play is the
# only difference between those two lines -- the card never learns which
# answered. Images are pulled on demand; nothing needs fetching first.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
examples="$(cd "$here/.." && pwd)"
repo="$(cd "$examples/.." && pwd)"

# This example lives beside magnet, not inside it -- it is a *consumer* of
# magnet, the way a team's own repository is, and it is imported the same way:
# by being on the path. `pipeline.py` and the node commands both need it.
export PYTHONPATH="$examples${PYTHONPATH:+:$PYTHONPATH}"

require_commands() {
    local missing=()
    local command_name
    for command_name in "$@"; do
        if ! command -v "$command_name" >/dev/null 2>&1; then
            missing+=("$command_name")
        fi
    done
    if ((${#missing[@]})); then
        printf 'This example requires MAGNET and infer-stack on PATH. Missing:' >&2
        printf ' %s' "${missing[@]}" >&2
        printf '\n' >&2
        printf 'Install aiq-magnet with its leasing extra in the active environment.\n' >&2
        exit 127
    fi
}

# Detect prerequisites without invoking their CLIs. Calling a modal CLI with no
# subcommand prints its usage text and turns a simple presence check into a
# spurious error.
require_commands magnet infer-stack

catalog="$here/catalog-mock.yaml"
container=0
while [[ "${1:-}" == --gpu || "${1:-}" == --container ]]; do
    case "$1" in
        --gpu)       catalog="$here/catalog.yaml" ;;
        --container) container=1 ;;
    esac
    shift
done

# Built here rather than named, because an image tag that exists on the machine
# that built it is exactly the kind of thing that does not travel. The build is
# a slim Python base plus MAGNET's core -- no GPU, no weights, no extras.
container_args=()
if [[ "$container" == 1 ]]; then
    image=magnet-smollm-example:latest
    docker build -f "$here/Dockerfile" -t "$image" "$repo"
    # Mount the working directory at its own absolute path: kwdagger bakes
    # absolute output paths into every command, so keeping them identical means
    # nothing has to be rewritten and a path in a log is one you can open.
    container_args=(
        --container_image="$image"
        --container_mounts="$PWD"
        # PYTHONPATH is captured by value into the rendered command, and the
        # host path it names does not exist in the image -- the example is
        # COPYed in and already on the image's path. Clear it rather than let
        # a broken host path shadow that.
        --container_env='{"PYTHONPATH": "/opt/magnet/examples"}'
    )
fi

# One-time host setup, not a per-run step: infer-stack needs to know how it
# brings endpoints up. Say so plainly rather than let the run fail deep inside
# a lease with `NullBackend`.
if ! infer-stack status 2>/dev/null | grep -qE '^\s*backend:\s*(compose|kubeai)'; then
    echo "infer-stack has no backend configured. Once per machine:" >&2
    echo "    infer-stack config init --yes --backend compose" >&2
    exit 1
fi

export INFER_STACK_CATALOG="$catalog"

exec python -m magnet.evaluation_new \
    --path="$here/smollm_kwdagger.yaml" \
    --output_path="${SMOLLM_RUNS:-./runs/smollm}" \
    --per_node_leasing \
    --backend=serial \
    "${container_args[@]}" \
    "$@"
