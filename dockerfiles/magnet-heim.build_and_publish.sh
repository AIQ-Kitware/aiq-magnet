#!/usr/bin/env bash
__doc__='
dockerfiles/magnet-heim.build_and_publish.sh

Build and optionally publish a "magnet+HELM(HEIM)" image that extends an
existing magnet image with a custom HELM checkout.

Conceptual behavior:
  - Take an arbitrary existing magnet image (local or remote), specified as
    BASE_IMAGE or as the first positional argument.
  - Prepare or update a HELM staging repo under .staging/helm from HELM_REMOTE.
  - Build a new image using magnet-heim.dockerfile that:
      * uninstalls any site-packages crfm-helm
      * copies the staged HELM repo (with .git) into the image
      * checks out HELM_GIT_REF inside the container
      * installs HELM and the magnet repo with the [heim] extra
  - Tag the resulting heim image and optionally push to Docker Hub.

Tags:
  - Let BASE_TAG be the tag portion of BASE_IMAGE (e.g. "some-tag").
  - This script produces:
      local image: magnet-heim:${BASE_TAG}
      pushed tags (if PUSH_IMAGES=1):
        <DOCKER_REPO>/magnet-heim:${BASE_TAG}
        <DOCKER_REPO>/magnet-heim:latest-heim
        <DOCKER_REPO>/magnet-heim:latest-heim-python<MAJOR.MINOR>

Typical usage:

  # No push, explicit base magnet image
  PUSH_IMAGES=0 \
  BASE_IMAGE=magnet:latest \
      ./dockerfiles/magnet-heim.build_and_publish.sh

Environment variables:

  IMAGE_NAME        - Logical app name (default: magnet-heim)

  # Base magnet image
  BASE_IMAGE        - Magnet image to extend (e.g. magnet:latest).
                      If unset, the first positional argument is used.
  PYTHON_VERSION    - Python major.minor (default: 3.10). Used for tags.

  # HELM repo
  HELM_REMOTE       - Git URL or local path for the HELM repo. REQUIRED.
  HELM_GIT_REF      - HELM ref/branch/sha to check out in the image. REQUIRED.
  HELM_STAGING_DIR  - Local staging dir for HELM (default: .staging/helm)

  # Tagging
  IMAGE_TAG         - Primary tag for the heim image. If unset, derived as:
                        <base_tag>-heim
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
  REPO_ROOT           - Repo root (default: parent of this script)
  MAGNET_HEIM_DOCKERFILE - Path to magnet-heim.dockerfile
                           (default: ${REPO_ROOT}/dockerfiles/magnet-heim.dockerfile)
'

set -euo pipefail

log(){ printf "\033[1;34m[heim-build]\033[0m %s\n" "$*"; }
die(){ printf "\033[1;31m[error]\033[0m %s\n" "$*" >&2; exit 1; }

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  printf "%s\n" "$__doc__"
  exit 0
fi

# ------------------------------------------------------------------------------
# Global config
# ------------------------------------------------------------------------------

: "${IMAGE_NAME:=magnet-heim}"
: "${PYTHON_VERSION:=3.10}"

: "${BASE_IMAGE:=}"
: "${HELM_REMOTE:=}"
: "${HELM_GIT_REF:=}"
: "${HELM_STAGING_DIR:=.staging/helm}"

: "${DOCKER_REPO:=docker.io/erotemic}"
: "${DOCKER_REGISTRY:=}"
: "${DOCKER_NAMESPACE:=}"

: "${PUSH_IMAGES:=1}"
: "${LOGIN_DOCKER:=1}"

: "${DOCKER_USERNAME:=}"
: "${DOCKER_TOKEN:=}"
: "${DOCKER_REGISTRY:=${DOCKER_REGISTRY:-}}"
: "${DOCKER_USERNAME:=${DOCKER_USERNAME:-${DOCKER_USERNAME}}}"
: "${DOCKER_TOKEN:=${DOCKER_TOKEN:-${DOCKER_TOKEN}}}"

: "${REPO_ROOT:=}"
: "${MAGNET_HEIM_DOCKERFILE:=}"

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
  : "${MAGNET_HEIM_DOCKERFILE:=${REPO_ROOT}/dockerfiles/magnet-heim.dockerfile}"

  [[ -f "$MAGNET_HEIM_DOCKERFILE" ]] || die "magnet-heim.dockerfile not found at $MAGNET_HEIM_DOCKERFILE"
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

derive_base_and_tag(){
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
    base_tag="uvlatest"
    die "unknown type of base image"
  fi

  # Fully qualified image tag is the version of the program/repo suffixed with
  # base versioning.
  : "${IMAGE_TAG:=${HELM_GIT_REF}-${base_name}${base_tag}}"

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

ensure_base_present(){
  log "Ensuring base magnet image is present: ${BASE_IMAGE}"
  if ! docker image inspect "${BASE_IMAGE}" >/dev/null 2>&1; then
    log "Local image ${BASE_IMAGE} not found; attempting docker pull"
    if ! docker pull "${BASE_IMAGE}"; then
      log "WARNING: failed to pull ${BASE_IMAGE}; build may still succeed if image exists under a different name"
    fi
  fi
}

prepare_helm_staging(){
  [[ -n "${HELM_REMOTE}" ]] || die "HELM_REMOTE is required"
  [[ -n "${HELM_GIT_REF}" ]] || die "HELM_GIT_REF is required"

  log "Preparing HELM staging directory at ${HELM_STAGING_DIR}"
  mkdir -p "$(dirname "${HELM_STAGING_DIR}")"

  if [[ -d "${HELM_STAGING_DIR}/.git" ]]; then
    log "Existing HELM repo detected; fetching updates"
    git -C "${HELM_STAGING_DIR}" fetch --all --prune
  else
    rm -rf "${HELM_STAGING_DIR}"
    log "Cloning HELM repo from ${HELM_REMOTE}"
    git clone "${HELM_REMOTE}" "${HELM_STAGING_DIR}"
  fi

  log "Checking out HELM_GIT_REF=${HELM_GIT_REF} in staging repo"
  git -C "${HELM_STAGING_DIR}" checkout "${HELM_GIT_REF}"
}

build_heim_image(){
  log "Building ${IMAGE_QUALNAME} from base ${BASE_IMAGE}"
  docker build \
    --file "${MAGNET_HEIM_DOCKERFILE}" \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    --build-arg HELM_STAGING_DIR="${HELM_STAGING_DIR}" \
    --build-arg HELM_GIT_REF="${HELM_GIT_REF}" \
    --build-arg VCS_REF="${VCS_REF}" \
    --build-arg REPO_URL="${REPO_URL}" \
    --build-arg DOCKERFILE_PATH="dockerfile/magnet-heim.dockerfile" \
    --tag "${IMAGE_QUALNAME}" \
    "${REPO_ROOT}"
}

make_alias_tags(){
  ALIASES=()
  ALIASES+=( "${IMAGE_NAME}:latest-heim" )
  ALIASES+=( "${IMAGE_NAME}:latest-heim-python${PYTHON_VERSION}" )
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
magnet-heim.build_and_publish.sh summary
------------------------------------------------------------
IMAGE_NAME        = ${IMAGE_NAME}
BASE_IMAGE        = ${BASE_IMAGE}
PYTHON_VERSION    = ${PYTHON_VERSION}
HELM_REMOTE       = ${HELM_REMOTE}
HELM_GIT_REF      = ${HELM_GIT_REF}
HELM_STAGING_DIR  = ${HELM_STAGING_DIR}
IMAGE_TAG         = ${IMAGE_TAG}
IMAGE_QUALNAME    = ${IMAGE_QUALNAME}

DOCKER_REPO       = ${DOCKER_REPO}
DOCKER_REGISTRY   = ${DOCKER_REGISTRY}
DOCKER_NAMESPACE  = ${DOCKER_NAMESPACE}
DOCKER_REGISTRY   = ${DOCKER_REGISTRY}
PUSH_IMAGES       = ${PUSH_IMAGES}
LOGIN_DOCKER      = ${LOGIN_DOCKER}

MAGNET_HEIM_DOCKERFILE = ${MAGNET_HEIM_DOCKERFILE}
REPO_ROOT              = ${REPO_ROOT}
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

  ensure_base_present
  prepare_helm_staging
  build_heim_image
  tag_aliases
  push_all_tags
}

if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
  log "Sourcing magnet-heim.build_and_publish.sh as a library"
else
  main "$@"
fi

