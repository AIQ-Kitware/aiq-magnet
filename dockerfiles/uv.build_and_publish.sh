#!/usr/bin/env bash
__doc__='
dockerfiles/uv.build_and_publish.sh

Build and optionally publish a "uv base" image from dockerfiles/uv.dockerfile.

Key features:
  - Accepts BASE_IMAGE (e.g. nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04)
  - Auto-derives BASE_TAG and IMAGE_TAG
  - Reads UV_VERSION default from inside uv.dockerfile
  - Uses VCS_REF and REPO_URL build-args (via local git state)
  - Generic registry support:
        SERVER_URL / SERVER_USERNAME / SERVER_TOKEN
  - Summary printed before any docker actions
  - Dedicated registry_login function

Usage:
  ./dockerfiles/uv.build_and_publish.sh
  BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 ./dockerfiles/uv.build_and_publish.sh
  IMAGE_TAG=my-tag SERVER_URL=… SERVER_USERNAME=… SERVER_TOKEN=… ./dockerfiles/uv.build_and_publish.sh
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

# Registry info
: "${SERVER_URL:=}"
: "${SERVER_USERNAME:=}"
: "${SERVER_TOKEN:=}"

: "${LOGIN_SERVER:=1}"
: "${PUSH_IMAGES:=1}"

# Paths
: "${REPO_ROOT:=}"
: "${UV_DOCKERFILE:=}"

# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------

make_vcs_ref(){
    local VCS_REF="$(git rev-parse HEAD)"
    local is_dirty=false
    git diff --quiet         || is_dirty=true
    git diff --cached --quiet || is_dirty=true
    if [ -n "$(git ls-files --others --exclude-standard)" ]; then
      is_dirty=true
    fi
    local VCS_REF_FULL="$VCS_REF"
    $is_dirty && VCS_REF_FULL="${VCS_REF}-dirty"
    echo "$VCS_REF_FULL"
}

get_repo_url(){
    local _RAW_URL
    _RAW_URL="$(git config --get remote.origin.url)"
    local _REPO_URL="$_RAW_URL"

    if [[ "$_RAW_URL" =~ ^git@([^:]+):(.+)\.git$ ]]; then
      _REPO_URL="https://${BASH_REMATCH[1]}/${BASH_REMATCH[2]}"
    elif [[ "$_RAW_URL" =~ ^https?://.*\.git$ ]]; then
      _REPO_URL="${_RAW_URL%.git}"
    fi
    echo "$_REPO_URL"
}

get_dockerfile_arg_default(){
    local _ARGNAME="$1"
    local _DOCKERFILE_PATH="$2"
    grep -E "^ARG ${_ARGNAME}=" "$_DOCKERFILE_PATH" \
      | head -n1 \
      | cut -d= -f2-
}

registry_login(){
  local registry_host="$1"

  if [[ -z "$registry_host" ]]; then
    log "registry_login: no registry host provided; skipping login"
    return 0
  fi

  if [[ "$PUSH_IMAGES" -ne 1 || "$LOGIN_SERVER" -ne 1 ]]; then
    log "registry_login: not logging in (PUSH_IMAGES=$PUSH_IMAGES, LOGIN_SERVER=$LOGIN_SERVER)"
    return 0
  fi

  if [[ -n "$SERVER_USERNAME" && -n "$SERVER_TOKEN" ]]; then
    log "Logging in to $registry_host as $SERVER_USERNAME"
    printf "%s" "$SERVER_TOKEN" | docker login "$registry_host" -u "$SERVER_USERNAME" --password-stdin
  else
    log "Skipping login: SERVER_USERNAME or SERVER_TOKEN missing.
If needed: docker login $registry_host"
  fi
}

print_summary(){
  local token_status="<unset>"
  [[ -n "${SERVER_TOKEN:-}" ]] && token_status="set"

  cat <<EOF
Build plan (uv base):

  Paths:
    REPO_ROOT:        ${REPO_ROOT}
    UV_DOCKERFILE:    ${UV_DOCKERFILE}

  Git info:
    VCS_REF:          ${VCS_REF}
    REPO_URL:         ${REPO_URL}

  Base image:
    BASE_IMAGE:       ${BASE_IMAGE}
    BASE_TAG:         ${BASE_TAG}

  Versions:
    UV_VERSION:       ${UV_VERSION}
    PYTHON_VERSION:   ${PYTHON_VERSION}

  Image tagging:
    IMAGE_NAME:       ${IMAGE_NAME}
    IMAGE_TAG:        ${IMAGE_TAG}
    IMAGE_QUALNAME:   ${IMAGE_QUALNAME}

  Registry:
    SERVER_URL:       ${SERVER_URL:-<unset>}
    REGISTRY_HOST:    ${REGISTRY_HOST:-<none>}
    REMOTE_IMAGE:     ${REMOTE_IMAGE:-<none>}
    SERVER_USERNAME:  ${SERVER_USERNAME:-<unset>}
    SERVER_TOKEN:     ${token_status}
    LOGIN_SERVER:     ${LOGIN_SERVER}
    PUSH_IMAGES:      ${PUSH_IMAGES}

EOF
}

# ------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------

main() {
  # -----------------------
  # Resolve paths
  # -----------------------
  local script_dir
  script_dir="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
    pwd
  )"

  [[ -z "$REPO_ROOT" ]]      && REPO_ROOT="$(realpath "$script_dir/..")"
  [[ -z "$UV_DOCKERFILE" ]]  && UV_DOCKERFILE="${REPO_ROOT}/dockerfiles/uv.dockerfile"

  [[ -f "$UV_DOCKERFILE" ]] || die "uv.dockerfile not found at $UV_DOCKERFILE"

  # -----------------------
  # Capture VCS metadata
  # -----------------------
  VCS_REF="$(make_vcs_ref)"
  REPO_URL="$(get_repo_url)"
  DOCKERFILE_PATH="$UV_DOCKERFILE"

  # -----------------------
  # If UV_VERSION not supplied, extract from dockerfile default ARG
  # -----------------------
  if [[ -z "$UV_VERSION" ]]; then
    UV_VERSION="$(get_dockerfile_arg_default UV_VERSION "$UV_DOCKERFILE")"
    [[ -n "$UV_VERSION" ]] || die "Failed to infer UV_VERSION from $UV_DOCKERFILE"
  fi

  # -----------------------
  # BASE_TAG + IMAGE_TAG derivation
  # -----------------------
  # Strip everything up to the last '/' …
  BASE_TAG="${BASE_IMAGE##*/}"
  # … then remove all ':' characters
  BASE_TAG="${BASE_TAG//:/}"


  IMAGE_TAG="${UV_VERSION}-python${PYTHON_VERSION}-${BASE_TAG}"

  IMAGE_QUALNAME="${IMAGE_NAME}:${IMAGE_TAG}"

  if [[ -n "$SERVER_URL" ]]; then
    REMOTE_IMAGE="${SERVER_URL}/${IMAGE_NAME}:${IMAGE_TAG}"
    REGISTRY_HOST="${SERVER_URL%%/*}"
    [[ -z "$REGISTRY_HOST" ]] && REGISTRY_HOST="$SERVER_URL"
  else
    REMOTE_IMAGE=""
    REGISTRY_HOST=""
  fi

  # -----------------------
  # Show plan
  # -----------------------
  print_summary

  # -----------------------
  # Login (if applicable)
  # -----------------------
  if [[ -n "$REGISTRY_HOST" ]]; then
    registry_login "$REGISTRY_HOST"
  fi

  # -----------------------
  # Build
  # -----------------------
  log "Building image: ${IMAGE_QUALNAME}"

  DOCKER_BUILDKIT=1 docker build --progress=plain \
    -t "${IMAGE_QUALNAME}" \
    -f "${UV_DOCKERFILE}" \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    --build-arg UV_VERSION="${UV_VERSION}" \
    --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
    --build-arg VCS_REF="${VCS_REF}" \
    --build-arg REPO_URL="${REPO_URL}" \
    --build-arg DOCKERFILE_PATH="${DOCKERFILE_PATH}" \
    "${REPO_ROOT}"

  # Tag local aliases
  ALIASES=(
      "$IMAGE_NAME:latest-python${PYTHON_VERSION}"
      "$IMAGE_NAME:latest"
  )
  for ALIAS in "${ALIASES[@]}"; do
      docker tag "$IMAGE_QUALNAME" "$ALIAS"
  done


  REMOTE_TAGS=()
  REMOTE_TAGS+=("${REMOTE_IMAGE}")

  for alias in "${ALIASES[@]}"; do
    # alias looks like "${APP_NAME}:tag"; we’ll prepend DOCKER_REPO.
    local_tag="${alias#"${IMAGE_NAME}":}"
    REMOTE_TAGS+=("${DOCKER_REPO}/${APP_NAME}:${local_tag}")
  done

  log "Remote tags to push (if enabled):"
  for tag in "${REMOTE_TAGS[@]}"; do
    log "  - ${tag}"
  done

  if [[ "${PUSH_IMAGES}" -eq 1 ]]; then
    log "PUSH_IMAGES=1 → pushing to ${DOCKER_REPO}/${APP_NAME}"
    for tag in "${REMOTE_TAGS[@]}"; do
      log "docker push ${tag}"
      docker push "${tag}"
    done
  else
    log "Images were NOT pushed because PUSH_IMAGES=0."
  fi

  # -----------------------
  # Tag & push
  # -----------------------
  if [[ -n "$REMOTE_IMAGE" ]]; then
    log "Tagging remote image: $REMOTE_IMAGE"
    docker tag "$IMAGE_QUALNAME" "$REMOTE_IMAGE"

    if [[ "$PUSH_IMAGES" -eq 1 ]]; then
      log "Pushing to registry: $REMOTE_IMAGE"
      docker push "$REMOTE_IMAGE"
    else
      log "PUSH_IMAGES=0 → skipping registry push"
    fi
  else
    log "SERVER_URL unset → skipping remote tag/push"
  fi

  # -----------------------
  # done
  # -----------------------
  cat <<EOF

Done.
Built local image:
  ${IMAGE_QUALNAME}

Local aliases:
$(for a in "${ALIASES[@]}"; do printf "    %s\n" "${a}"; done)

Remote image:
  ${REMOTE_IMAGE:-<none>}

EOF
}

if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
    echo "Sourcing as a library"
else
    if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
      printf "%s\n" "$__doc__"
      exit 0
    fi
    main "$@"
fi

