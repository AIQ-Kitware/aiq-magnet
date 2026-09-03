#!/usr/bin/env bash
# Run the SmolLM example.
#
#   ./run.sh                       real SmolLM2 on a GPU; node commands in containers
#   ./run.sh --mock                simulated endpoints; no GPU; nodes still in containers
#   ./run.sh --no-container        real SmolLM2 on a GPU; node commands on the host
#   ./run.sh --mock --no-container simulator plus host node commands
#
# Anything else is passed through to `magnet evaluate_new`, so
# `./run.sh --dry_run=1` compiles the campaign without running it.
#
# The two catalogs declare the SAME two aliases, so the card never learns
# whether the real or simulated endpoints answered. Images and model weights
# are pulled on demand; nothing needs fetching first.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./run.sh [OPTIONS] [MAGNET_EVALUATE_NEW_OPTIONS...]

Run the SmolLM MAGNET demo. By default this serves the real SmolLM2 135M and
360M checkpoints with vLLM on an NVIDIA GPU and runs each MAGNET node inside
the example container.

Options:
  --mock          Use infer-stack simulator endpoints instead of real models.
                  No GPU is required.
  --no-container  Run MAGNET node commands on the host instead of building and
                  using the example container.
  -h, --help      Show this help and exit.

The two mode switches are independent and may be combined. All other arguments
are passed through to `python -m magnet.evaluation_new`.

Examples:
  ./run.sh
  ./run.sh --mock
  ./run.sh --no-container
  ./run.sh --mock --no-container
  ./run.sh --dry_run=1
  ./run.sh --params='matrix: {ask.endpoint: [smol-135]}'
                  Override the kwdagger matrix; endpoint values are infer-stack
                  catalog aliases. See README.md for adding a custom model.

Environment:
  SMOLLM_RUNS     Output root (default: ./runs/smollm)
EOF
}

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
examples="$(cd "$here/.." && pwd)"

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

require_nvidia_gpu() {
    local listing
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        listing=''
    else
        listing="$(nvidia-smi -L 2>/dev/null || true)"
    fi
    if [[ ! "$listing" =~ ^GPU[[:space:]][0-9]+: ]]; then
        cat >&2 <<'MSG'
No usable NVIDIA GPU was detected. The default SmolLM run serves the real
models with vLLM and requires a GPU visible to nvidia-smi.

Use ./run.sh --mock to run the same card against the GPU-free simulator.
MSG
        exit 1
    fi
}

mock=0
container=1
passthrough=()
for arg in "$@"; do
    case "$arg" in
        -h|--help)
            usage
            exit 0
            ;;
        --mock)
            mock=1
            ;;
        --no-container)
            container=0
            ;;
        # Historical positive flags are harmless no-ops now that these are the
        # defaults. Keep them accepted so existing invocations do not break.
        --gpu|--container)
            ;;
        *)
            passthrough+=("$arg")
            ;;
    esac
done

# Detect prerequisites without invoking their CLIs. Calling a modal CLI with no
# subcommand prints its usage text and turns a simple presence check into a
# spurious error.
require_commands magnet infer-stack

if [[ "$mock" == 1 ]]; then
    catalog="$here/catalog-mock.yaml"
else
    require_nvidia_gpu
    catalog="$here/catalog.yaml"
fi

if [[ "$container" == 1 ]] && ! command -v docker >/dev/null 2>&1; then
    cat >&2 <<'MSG'
Docker is required for the default containerized node execution.
Use ./run.sh --no-container to run the MAGNET node commands on the host.
MSG
    exit 127
fi

# One-time host setup, not a per-run step: infer-stack needs to know how it
# brings endpoints up. Say so plainly rather than let the run fail deep inside
# a lease with `NullBackend`.
if ! infer-stack status 2>/dev/null | grep -qE '^\s*backend:\s*(compose|kubeai)'; then
    echo "infer-stack has no backend configured. Once per machine:" >&2
    echo "    infer-stack config init --yes --backend compose" >&2
    exit 1
fi

# The node image installs MAGNET from the commit pinned in Dockerfile and copies
# only this example's CLI package. The Docker build context is this directory,
# not the repository root, so changing cards, catalogs, docs, or unrelated
# MAGNET source does not invalidate the image. infer-stack serves models
# separately on the host.
container_args=()
if [[ "$container" == 1 ]]; then
    image=magnet-smollm-example:latest
    docker build -f "$here/Dockerfile" -t "$image" "$here"
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
        --container_env='{"PYTHONPATH": "/opt/examples"}'
    )
fi

export INFER_STACK_CATALOG="$catalog"

exec python -m magnet.evaluation_new \
    --path="$here/smollm_kwdagger.yaml" \
    --output_path="${SMOLLM_RUNS:-./runs/smollm}" \
    --per_node_leasing \
    --backend=serial \
    "${container_args[@]}" \
    "${passthrough[@]}"
