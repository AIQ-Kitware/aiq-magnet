#!/usr/bin/env bash
__doc__='
dockerfiles/magnet.build_and_publish.sh

Builds a developer-tagged magnet image from a uv base image and optionally
publishes it to a Docker registry (by default docker.io/erotemic).

High-level steps:
  1. Resolve repo root and dockerfiles directory so the script can run from anywhere.
  2. Determine the current git ref and derive a content-addressed image tag that
     incorporates the uv base tag.
  3. Ensure the uv base image is available locally (attempt a pull, but gracefully
     fall back to a local-only image).
  4. Build the magnet image using dockerfiles/magnet.dockerfile.
  5. Tag local developer aliases (e.g. magnet:latest-dev, magnet:latest-dev-python3.10).
  6. Optionally log in to the registry and push all tags.

Typical usage:

  # No push, explicit uv base
  PUSH_IMAGES=0 \
  BASE_IMAGE=uv:latest \
      ./dockerfiles/magnet.build_and_publish.sh

Environment variables:

  IMAGE_NAME          - Logical app name (default: magnet)

  # Base uv image
  BASE_IMAGE        - Canonical name for the uv base image;  (e.g. uv:latest) (positional arg 1 if unset)

  # Tagging
  DOCKER_REPO       - Registry/namespace for pushed tags
                      (default: docker.io/erotemic)

  # Login / push
  PUSH_IMAGES       - If 1, push images to the registry; if 0, build/tag only
                      (default: 1)
  LOGIN_DOCKER      - If 1, attempt login to DOCKER_REGISTRY (default: 1)
  DOCKER_USERNAME   - Registry username
  DOCKER_TOKEN      - Registry token/password

  # Paths
  REPO_ROOT         - Repo root (default: parent of this script)
  DOCKERFILE_PATH - Path to magnet.dockerfile
                      (default: dockerfiles/magnet.dockerfile)
'

set -euo pipefail

log(){ printf "\033[1;34m[magnet-build]\033[0m %s\n" "$*"; }
die(){ printf "\033[1;31m[error]\033[0m %s\n" "$*" >&2; exit 1; }

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  printf "%s\n" "$__doc__"
  exit 0
fi

# ------------------------------------------------------------------------------
# Global config
# ------------------------------------------------------------------------------

: "${IMAGE_NAME:=magnet}"
: "${PYTHON_VERSION:=}"

: "${BASE_IMAGE:=uv:latest-python3.10}"

#: "${DOCKER_REPO:=docker.io/erotemic}"
: "${DOCKER_REPO:=ghcr.io/aiq-kitware}"
: "${DOCKER_REGISTRY:=}"
: "${DOCKER_NAMESPACE:=}"

: "${PUSH_IMAGES:=1}"
: "${LOGIN_DOCKER:=1}"

: "${DOCKER_USERNAME:=}"
: "${DOCKER_TOKEN:=}"

: "${REPO_ROOT:=}"
: "${DOCKERFILE_PATH:=dockerfiles/magnet.dockerfile}"

# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------

make_vcs_ref(){
    git rev-parse HEAD
}

get_repo_url(){
    local raw_url repo_url
    raw_url="$(git config --get remote.origin.url || true)"
    repo_url="$raw_url"

    if [[ "$raw_url" =~ ^git@([^:]+):(.+)\.git$ ]]; then
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

  # Normalize to absolute path
  DOCKERFILE_PATH="$(realpath "$REPO_ROOT/$DOCKERFILE_PATH")"

  # Verify the path is inside the repo
  [[ "$DOCKERFILE_PATH" == "$REPO_ROOT"/* ]] || \
      die "Dockerfile must be inside repo root"

  # Make relative
  DOCKERFILE_PATH="${DOCKERFILE_PATH#"$REPO_ROOT/"}"

  # Ensure the file exists
  [[ -f "${REPO_ROOT}/${DOCKERFILE_PATH}" ]] || \
      die "dockerfile not found at ${REPO_ROOT}/${DOCKERFILE_PATH}"
}


derive_registry_parts(){
  if [[ -z "${DOCKER_REGISTRY}" || -z "${DOCKER_NAMESPACE}" ]]; then
    local _repo="${DOCKER_REPO}"
    local _reg="${_repo%%/*}"
    local _ns="${_repo#*/}"
    : "${DOCKER_REGISTRY:=${_reg}}"
    : "${DOCKER_NAMESPACE:=${_ns}}"
  fi
}

short12(){
  # 12-char short sha (handles "-dirty" suffix)
  local ref="$1"
  local core="${ref%%-*}"
  local suf="${ref#"$core"}"   # SC2295-safe: quote inner expansion
  echo "$(printf "%s" "$core" | cut -c1-12)${suf}"
}

derive_base_and_tag(){
  if [[ -z "${BASE_IMAGE}" && "${1:-}" != "" ]]; then
    BASE_IMAGE="$1"
    shift || true
  fi

  [[ -n "${BASE_IMAGE}" ]] || die "BASE_IMAGE is required"

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
  GIT_REF_SHORT="$(short12 "$VCS_REF")"
  # Fully qualified image tag is the version of the program/repo suffixed with
  # base versioning.
  : "${IMAGE_TAG:=${GIT_REF_SHORT}-${base_name}${base_tag}}"

  IMAGE_QUALNAME="${IMAGE_NAME}:${IMAGE_TAG}"

  # Return remaining args via global "$@"
  set -- "$@"
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

ensure_uv_base_present(){
  log "Ensuring uv base image is present: ${BASE_IMAGE}"
  if ! docker image inspect "${BASE_IMAGE}" >/dev/null 2>&1; then
    log "Local image ${BASE_IMAGE} not found; attempting docker pull"
    if ! docker pull "${BASE_IMAGE}"; then
      log "WARNING: failed to pull ${BASE_IMAGE}; build may still succeed if image exists under a different name"
    fi
  fi
}

build_magnet_image(){
  log "Building ${IMAGE_QUALNAME} from base ${BASE_IMAGE}"
  docker build \
    --file "${REPO_ROOT}/${DOCKERFILE_PATH}" \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    --build-arg GIT_REF="${VCS_REF}" \
    --build-arg USE_LOCKFILE="0" \
    --build-arg REPO_URL="${REPO_URL}" \
    --build-arg VCS_REF="${VCS_REF}" \
    --build-arg DOCKERFILE_PATH="${DOCKERFILE_PATH}" \
    --tag "${IMAGE_QUALNAME}" \
    "${REPO_ROOT}"
}

infer_python_version_from_base_tag(){
  # Parse ...python3.10... from BASE_IMAGE
  if [ -z "$PYTHON_VERSION" ]; then
      if [[ "$BASE_IMAGE" =~ python([0-9]+\.[0-9]+) ]]; then
        PYTHON_VERSION="${BASH_REMATCH[1]}"
      fi
  fi
}

make_alias_tags(){
  ALIASES=()
  ALIASES+=( "${IMAGE_NAME}:latest" )
  if [ -n "$PYTHON_VERSION" ]; then
      ALIASES+=( "${IMAGE_NAME}:latest-python${PYTHON_VERSION}" )
  fi
}

tag_aliases(){
  for alias in "${ALIASES[@]}"; do
    log "Tagging ${IMAGE_QUALNAME} as ${alias}"
    docker tag "${IMAGE_QUALNAME}" "${alias}"
  done
}

push_all_tags(){
  local remote_tags=()
  remote_tags+=( "${DOCKER_REPO}/${IMAGE_NAME}:${IMAGE_TAG}" )

  for alias in "${ALIASES[@]}"; do
    echo "alias = $alias"
    local tag_part="${alias#"${IMAGE_NAME}":}"
    remote_tags+=( "${DOCKER_REPO}/${IMAGE_NAME}:${tag_part}" )
  done

  # Apply remote tags
  log "Remote tags to push (if enabled):"
  for tag in "${remote_tags[@]}"; do
    log "Tagging ${IMAGE_QUALNAME} as ${tag}"
    docker tag "${IMAGE_QUALNAME}" "${tag}"
  done

  if [[ "${PUSH_IMAGES}" -eq 1 ]]; then
    registry_login
    set -x
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
magnet.build_and_publish.sh summary
------------------------------------------------------------
IMAGE_NAME      = ${IMAGE_NAME}
BASE_IMAGE      = ${BASE_IMAGE}
PYTHON_VERSION  = ${PYTHON_VERSION}
IMAGE_TAG       = ${IMAGE_TAG}
IMAGE_QUALNAME  = ${IMAGE_QUALNAME}

DOCKER_REPO      = ${DOCKER_REPO}
DOCKER_REGISTRY  = ${DOCKER_REGISTRY}
DOCKER_NAMESPACE = ${DOCKER_NAMESPACE}
PUSH_IMAGES      = ${PUSH_IMAGES}
LOGIN_DOCKER     = ${LOGIN_DOCKER}

DOCKERFILE_PATH  = ${DOCKERFILE_PATH}
REPO_ROOT        = ${REPO_ROOT}
REPO_URL         = ${REPO_URL}
VCS_REF          = ${VCS_REF}
------------------------------------------------------------
EOF
}

main(){
  derive_repo_root_and_dockerfile
  derive_registry_parts
  infer_python_version_from_base_tag

  VCS_REF="$(make_vcs_ref)"
  REPO_URL="$(get_repo_url)"

  derive_base_and_tag "$@"
  make_alias_tags

  print_summary

  ensure_uv_base_present
  build_magnet_image
  tag_aliases
  push_all_tags
}

if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
  log "Sourcing magnet.build_and_publish.sh as a library"
else
  main "$@"
fi

