#!/usr/bin/env bash
__doc__='
dockerfiles/uv.build_and_publish.sh

Build and optionally publish a "uv base" image from dockerfiles/uv.dockerfile.

Key features:
  - Accepts BASE_IMAGE (e.g. nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04)
  - Auto-derives BASE_TAG and IMAGE_TAG
  - Reads UV_VERSION default from inside uv.dockerfile (unless overridden)
  - Uses VCS_REF and REPO_URL build-args (via local git state)
  - Generic registry support:
        DOCKER_REGISTRY / DOCKER_USERNAME / DOCKER_TOKEN
  - Summary printed before any docker actions
  - Dedicated registry_login function

Typical usage:

  # Minimal (all defaults)
  ./dockerfiles/uv.build_and_publish.sh

  # Explicit base image, no push
  PUSH_IMAGES=0 \
  PYTHON_VERSION=3.10 \
  UV_VERSION=0.8.8 \
  BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 \
      ./dockerfiles/uv.build_and_publish.sh

Environment variables (override defaults as needed):

  IMAGE_NAME        - Logical image name (default: uv)

  # Base image
  BASE_IMAGE        - Full base image reference
                      (default: nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04)
  PYTHON_VERSION    - Python major.minor (default: 3.10)
  UV_VERSION        - uv version to install. If unset, parsed from uv.dockerfile.

  # Tagging
  IMAGE_TAG         - Primary image tag. If unset, derived as:
                        <base_tag>-python<PYTHON_VERSION>-uv<UV_VERSION>
                      where base_tag is the portion after ":" in BASE_IMAGE.
  DOCKER_REPO       - Registry/namespace for pushed tags
                      (default: docker.io/erotemic)
  DOCKER_REGISTRY   - If unset, derived from DOCKER_REPO (before first "/")
  DOCKER_NAMESPACE  - If unset, derived from DOCKER_REPO (after first "/")

  # Login / push
  PUSH_IMAGES       - If 1, push images to the registry; if 0, build/tag only
                      (default: 1)
  LOGIN_DOCKER      - If 1, attempt login to DOCKER_REGISTRY (default: 1)
  DOCKER_USERNAME   - Registry username
  DOCKER_TOKEN      - Registry token/password

  # Paths
  REPO_ROOT         - Repo root (default: parent of this script)
  UV_DOCKERFILE     - Path to uv.dockerfile
                      (default: ${REPO_ROOT}/dockerfiles/uv.dockerfile)

The script prints a summary of the resolved configuration before building.
'

set -euo pipefail

log(){ printf "\033[1;34m[uv-build]\033[0m %s\n" "$*"; }
die(){ printf "\033[1;31m[error]\033[0m %s\n" "$*" >&2; exit 1; }

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  printf "%s\n" "$__doc__"
  exit 0
fi

# ------------------------------------------------------------------------------
# Global config variables (simple env expansion only)
# ------------------------------------------------------------------------------

: "${IMAGE_NAME:=uv}"

# Base CUDA (or any) image we build FROM
: "${BASE_IMAGE:=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04}"

# Python major.minor
: "${PYTHON_VERSION:=3.10}"

# Will derive UV_VERSION dynamically from dockerfile unless overridden
: "${UV_VERSION:=}"

# Registry / repo info
: "${DOCKER_REPO:=docker.io/erotemic}"     # registry + namespace, e.g. docker.io/erotemic
: "${DOCKER_REGISTRY:=}"                   # optional override
: "${DOCKER_NAMESPACE:=}"                  # optional override

# Login / push
: "${PUSH_IMAGES:=1}"
: "${LOGIN_DOCKER:=1}"

# Back-compat: map DOCKER_* to DOCKER_* if not explicitly set
: "${DOCKER_USERNAME:=}"
: "${DOCKER_TOKEN:=}"
: "${DOCKER_REGISTRY:=${DOCKER_REGISTRY:-}}"
: "${DOCKER_USERNAME:=${DOCKER_USERNAME:-${DOCKER_USERNAME}}}"
: "${DOCKER_TOKEN:=${DOCKER_TOKEN:-${DOCKER_TOKEN}}}"

# Paths
: "${REPO_ROOT:=}"
: "${UV_DOCKERFILE:=}"

# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------

make_vcs_ref(){
    local VCS_REF
    VCS_REF="$(git rev-parse HEAD)"
    local is_dirty=false
    git diff --quiet           || is_dirty=true
    git diff --cached --quiet  || is_dirty=true
    if [ -n "$(git ls-files --others --exclude-standard)" ]; then
      is_dirty=true
    fi
    if $is_dirty; then
      printf "%s-dirty\n" "$VCS_REF"
    else
      printf "%s\n" "$VCS_REF"
    fi
}

get_repo_url(){
    local raw_url repo_url
    raw_url="$(git config --get remote.origin.url || true)"
    repo_url="$raw_url"

    if [[ "$raw_url" =~ ^git@([^:]+):(.+)\.git$ ]]; then
      # git@github.com:org/repo.git → https://github.com/org/repo
      repo_url="https://${BASH_REMATCH[1]}/${BASH_REMATCH[2]}"
    elif [[ "$raw_url" =~ ^https?://.+\.git$ ]]; then
      repo_url="${raw_url%.git}"
    fi

    printf "%s\n" "$repo_url"
}

derive_repo_root_and_dockerfile(){
  local script_dir
  script_dir="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
    pwd
  )"
  : "${REPO_ROOT:=$(realpath "${script_dir}/..")}"
  : "${UV_DOCKERFILE:=${REPO_ROOT}/dockerfiles/uv.dockerfile}"

  [[ -f "$UV_DOCKERFILE" ]] || die "uv.dockerfile not found at $UV_DOCKERFILE"
}

derive_registry_parts(){
  # If DOCKER_REGISTRY / DOCKER_NAMESPACE are unset, derive from DOCKER_REPO
  if [[ -z "${DOCKER_REGISTRY}" || -z "${DOCKER_NAMESPACE}" ]]; then
    local _repo="${DOCKER_REPO}"
    local _reg="${_repo%%/*}"
    local _ns="${_repo#*/}"
    : "${DOCKER_REGISTRY:=${_reg}}"
    : "${DOCKER_NAMESPACE:=${_ns}}"
  fi
}

parse_uv_version_from_dockerfile(){
  if [[ -n "${UV_VERSION}" ]]; then
    return 0
  fi
  local line
  line="$(grep -E '^ARG[[:space:]]+UV_VERSION=' "$UV_DOCKERFILE" || true)"
  if [[ -n "$line" ]]; then
    UV_VERSION="${line#*=}"
    UV_VERSION="${UV_VERSION//\"/}"
    UV_VERSION="${UV_VERSION//\'/}"
  fi
  if [[ -z "${UV_VERSION}" ]]; then
    die "UV_VERSION is unset and could not be parsed from $UV_DOCKERFILE"
  fi
}

derive_base_and_tag(){
  # Allow BASE_IMAGE to be provided as first positional arg if not set
  if [[ -z "${BASE_IMAGE}" && "${1:-}" != "" ]]; then
    BASE_IMAGE="$1"
    shift || true
  fi

  [[ -n "${BASE_IMAGE}" ]] || die "BASE_IMAGE is required (env or positional arg 1)"

  local base_tag
  local base_name
  local unqual_base_image
  if [[ "$BASE_IMAGE" == *:* ]]; then
    # Strip everything before the final '/' (if present)
    unqual_base_image="${BASE_IMAGE##*/}"
    # Split NAME and TAG
    base_name="${unqual_base_image%%:*}"
    base_tag="${unqual_base_image##*:}"
  else
    die "unknown type of base image"
  fi

  parse_uv_version_from_dockerfile

  # Fully qualified image tag is the version of the program/repo suffixed with
  # base versioning.
  : "${IMAGE_TAG:=${UV_VERSION}-python${PYTHON_VERSION}-${base_name}${base_tag}}"

  IMAGE_QUALNAME="${IMAGE_NAME}:${IMAGE_TAG}"
}

registry_login(){
  if [[ "${PUSH_IMAGES}" -ne 1 || "${LOGIN_DOCKER}" -ne 1 ]]; then
    log "Skipping registry login (PUSH_IMAGES=${PUSH_IMAGES}, LOGIN_DOCKER=${LOGIN_DOCKER})"
    return 0
  fi

  if [[ -z "${DOCKER_USERNAME}" || -z "${DOCKER_TOKEN}" ]]; then
    log "DOCKER_USERNAME or DOCKER_TOKEN not set; assuming local-only push or existing docker login"
    return 0
  fi

  log "Logging in to registry ${DOCKER_REGISTRY} as ${DOCKER_USERNAME}"
  printf '%s\n' "${DOCKER_TOKEN}" | docker login "${DOCKER_REGISTRY}" --username "${DOCKER_USERNAME}" --password-stdin
}

build_uv_image(){
  log "Building ${IMAGE_QUALNAME} from ${BASE_IMAGE}"
  docker build \
    --file "${UV_DOCKERFILE}" \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
    --build-arg UV_VERSION="${UV_VERSION}" \
    --build-arg VCS_REF="${VCS_REF}" \
    --build-arg REPO_URL="${REPO_URL}" \
    --build-arg DOCKERFILE_PATH="dockerfile/uv.dockerfile" \
    --tag "${IMAGE_QUALNAME}" \
    "${REPO_ROOT}"
}

make_alias_tags(){
  ALIASES=()
  # Example aliases; adjust to taste.
  #  uv:latest
  #  uv:python3.10
  ALIASES+=( "${IMAGE_NAME}:latest" )
  ALIASES+=( "${IMAGE_NAME}:python${PYTHON_VERSION}" )
}

tag_aliases(){
  for alias in "${ALIASES[@]}"; do
    log "Tagging ${IMAGE_QUALNAME} as ${alias}"
    docker tag "${IMAGE_QUALNAME}" "${alias}"
  done
}

push_all_tags(){
  local remote_tags=()

  # Primary tag
  remote_tags+=( "${DOCKER_REPO}/${IMAGE_NAME}:${IMAGE_TAG}" )

  # Aliases (strip local repo name and reuse tag part)
  for alias in "${ALIASES[@]}"; do
    local tag_part="${alias#"${IMAGE_NAME}":}"
    remote_tags+=( "${DOCKER_REPO}/${IMAGE_NAME}:${tag_part}" )
  done

  log "Remote tags to push (if enabled):"
  for tag in "${remote_tags[@]}"; do
    log "  - ${tag}"
  done

  if [[ "${PUSH_IMAGES}" -eq 1 ]]; then
    registry_login
    for tag in "${remote_tags[@]}"; do
      log "docker push ${tag}"
      docker push "${tag}"
    done
  else
    log "Images were NOT pushed because PUSH_IMAGES=${PUSH_IMAGES}"
  fi
}

print_summary(){
  cat <<EOF
------------------------------------------------------------
uv.build_and_publish.sh summary
------------------------------------------------------------
IMAGE_NAME      = ${IMAGE_NAME}
BASE_IMAGE      = ${BASE_IMAGE}
PYTHON_VERSION  = ${PYTHON_VERSION}
UV_VERSION      = ${UV_VERSION}
IMAGE_TAG       = ${IMAGE_TAG}
IMAGE_QUALNAME  = ${IMAGE_QUALNAME}

DOCKER_REPO      = ${DOCKER_REPO}
DOCKER_REGISTRY  = ${DOCKER_REGISTRY}
DOCKER_NAMESPACE = ${DOCKER_NAMESPACE}
DOCKER_REGISTRY  = ${DOCKER_REGISTRY}
PUSH_IMAGES      = ${PUSH_IMAGES}
LOGIN_DOCKER     = ${LOGIN_DOCKER}

UV_DOCKERFILE   = ${UV_DOCKERFILE}
REPO_ROOT       = ${REPO_ROOT}
------------------------------------------------------------
EOF
}

main(){
  derive_repo_root_and_dockerfile
  derive_registry_parts

  VCS_REF="$(make_vcs_ref)"
  REPO_URL="$(get_repo_url)"

  derive_base_and_tag "$@"
  make_alias_tags

  print_summary

  build_uv_image
  tag_aliases
  push_all_tags
}

if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
  log "Sourcing uv.build_and_publish.sh as a library"
else
  main "$@"
fi

