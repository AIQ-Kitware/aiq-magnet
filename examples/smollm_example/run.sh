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
# The real and mock catalogs expose the same endpoint aliases.
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
  SMOLLM_CATALOG  Real-model catalog path (default: ./catalog.yaml). Use an
                  untracked catalog.local.yaml for custom endpoints.
EOF
}

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
examples="$(cd "$here/.." && pwd)"

# Make the example package importable by the pipeline and node commands.
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
        # Compatibility aliases for the default real/container modes.
        --gpu|--container)
            ;;
        *)
            passthrough+=("$arg")
            ;;
    esac
done

# Check executable availability without invoking modal CLIs.
require_commands magnet infer-stack

if [[ "$mock" == 1 ]]; then
    # The mock catalog is a checked-in fixture.
    catalog="$here/catalog-mock.yaml"
else
    require_nvidia_gpu
    catalog="${SMOLLM_CATALOG:-$here/catalog.yaml}"
fi

if [[ ! -f "$catalog" ]]; then
    printf 'SmolLM catalog does not exist: %s\n' "$catalog" >&2
    exit 1
fi
# Generated jobs run from artifact directories; export an absolute catalog path.
catalog="$(cd "$(dirname "$catalog")" && pwd)/$(basename "$catalog")"

if [[ "$container" == 1 ]] && ! command -v docker >/dev/null 2>&1; then
    cat >&2 <<'MSG'
Docker is required for the default containerized node execution.
Use ./run.sh --no-container to run the MAGNET node commands on the host.
MSG
    exit 127
fi

# Require an initialized infer-stack backend.
if ! infer-stack status 2>/dev/null | grep -qE '^\s*backend:\s*(compose|kubeai)'; then
    echo "infer-stack has no backend configured. Once per machine:" >&2
    echo "    infer-stack config init --yes --backend compose" >&2
    exit 1
fi

# Build the node image from this example directory. infer-stack runs on the host.
container_args=()
if [[ "$container" == 1 ]]; then
    image=magnet-smollm-example:latest
    docker build -f "$here/Dockerfile" -t "$image" "$here"
    # Preserve kwdagger's absolute artifact paths inside the container.
    container_args=(
        --container_image="$image"
        --container_mounts="$PWD"
        # Use the package path baked into the image, not the host PYTHONPATH.
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
