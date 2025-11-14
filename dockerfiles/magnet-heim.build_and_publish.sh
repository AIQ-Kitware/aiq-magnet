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

Base image:
  - The base magnet image is not derived from the current git ref or UV tag.
  - Instead, you explicitly specify a Docker image to use as the base, e.g.:
      BASE_IMAGE=erotemic/magnet:some-tag
    or:
      ./magnet-heim.build_and_publish.sh erotemic/magnet:some-tag

Tags:
  - Let BASE_TAG be the tag portion of BASE_IMAGE (e.g. "some-tag").
  - This script produces:
      local image: magnet:${BASE_TAG}-heim
      pushed tags (if PUSH_IMAGES=1):
        erotemic/magnet:${BASE_TAG}-heim
        erotemic/magnet:latest-dev-heim
        erotemic/magnet:latest-dev-heim-python<MAJOR.MINOR>

Environment variables (override defaults as needed):

  BASE_IMAGE      - Full base image reference (e.g. erotemic/magnet:foo).
                           If unset, the first positional argument will be used.
                           This is REQUIRED.

  APP_NAME               - Local image name (default: magnet)
  DOCKER_REPO             - Docker Hub namespace for pushed tags (default: erotemic)

  HELM_REMOTE            - Git URL or local path for the HELM repo.
                           REQUIRED (no default).
  HELM_GIT_REF           - HELM ref/branch/sha to check out in the image
                           (default: main)

  STAGING_ROOT           - Directory for staging repos (default: $REPO_ROOT/.staging)

  PYTHON_VERSION         - Python MAJOR.MINOR inside the base magnet image.
                           If unset, this will be inferred from BASE_TAG if
                           it contains "pythonX.Y"; otherwise must be set.

  LOGIN_DOCKERHUB        - If 1, attempt Docker Hub login (default: 1)
  SERVER_USERNAME        - Docker Hub username for non-interactive login
  SERVER_TOKEN           - Docker Hub token/password (never printed, only whether set)

  PUSH_IMAGES            - If 1, push images to Docker Hub; if 0, build/tag only
                           (default: 1)

Usage:
  ./dockerfiles/magnet-heim.build_and_publish.sh erotemic/magnet:some-tag
  BASE_IMAGE=erotemic/magnet:some-tag ./dockerfiles/magnet-heim.build_and_publish.sh
  PUSH_IMAGES=0 ./dockerfiles/magnet-heim.build_and_publish.sh erotemic/magnet:some-tag
  ./dockerfiles/magnet-heim.build_and_publish.sh --help
'

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
	# Running as a script
	set -euo pipefail
fi

if [[ "${SCRIPT_TRACE+x}" != "" ]]; then
	set -x
fi

# -----------------------
# Config / env defaults
# -----------------------
: "${APP_NAME:=magnet}"
: "${DOCKER_REPO:=docker.io/erotemic}"
: "${LATEST_TAG:=latest}"

: "${LOGIN_DOCKERHUB:=1}"
: "${PUSH_IMAGES:=1}"

#: "${HELM_REMOTE:=https://github.com/stanford-crfm/helm.git}"
# Using our fork of helm
: "${HELM_REMOTE:=https://github.com/AIQ-Kitware/helm.git}"
: "${HELM_GIT_REF:=main}"

log(){ printf "\033[1;34m[heim-build]\033[0m %s\n" "$*"; }
die(){ printf "\033[1;31m[error]\033[0m %s\n" "$*" >&2; exit 1; }

# ------------------------------------------------------------------------------
# General helper: prepare_staging_repo
#   Ensures repo_dir is a clean checkout of `remote` at `ref`.
#   If repo_dir does not exist, clone it.
#   If it exists, verify/update remote.origin.url and hard reset to ref.
# ------------------------------------------------------------------------------
prepare_staging_repo() {
  local repo_dir="$1"   # e.g. "$STAGING_ROOT/helm"
  local remote="$2"     # git URL or local path
  local ref="${3:-main}"

  if [[ -z "$repo_dir" || -z "$remote" ]]; then
    die "prepare_staging_repo: repo_dir and remote are required (got '$repo_dir', '$remote')"
  fi

  mkdir -p "$(dirname "$repo_dir")"

  if [[ ! -d "$repo_dir/.git" ]]; then
    log "Cloning staging repo into $repo_dir from $remote"
    rm -rf "$repo_dir"
    git clone --recurse-submodules "$remote" "$repo_dir"
  else
    local origin
    origin="$(git -C "$repo_dir" config --get remote.origin.url || true)"

    if [[ "$origin" != "$remote" ]]; then
      log "Updating remote for $repo_dir:"
      log "  old: $origin"
      log "  new: $remote"
      git -C "$repo_dir" remote set-url origin "$remote"
    fi
    git -C "$repo_dir" remote --verbose

    log "Refreshing $repo_dir → fetch + hard reset to $ref"
    log "Fetch"
    git -C "$repo_dir" fetch --all --tags --prune || true
    log "Reset"
    git -C "$repo_dir" checkout "$ref"
    git -C "$repo_dir" reset --hard "$ref"
    log "ok"
  fi

  if [[ ! -d "$repo_dir/.git" ]]; then
    die "prepare_staging_repo: staging missing .git at $repo_dir"
  fi

  log "Staging repo ready: $repo_dir (ref: $ref)"
}

infer_python_version_from_tag(){
  # Try to parse ...python3.10... from an image tag string
  local tag="$1"
  if [[ "$tag" =~ python([0-9]+\.[0-9]+) ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo ""
  fi
}

print_summary(){
  local token_status
  if [[ -n "${SERVER_TOKEN:-}" ]]; then
    token_status="set"
  else
    token_status="unset"
  fi

  cat <<EOF
Build plan (magnet+HELM):

  Base image:
    BASE_IMAGE:     ${BASE_IMAGE}
    BASE_TAG:              ${BASE_TAG}

  Resulting image:
    LATEST_TAG:       ${LATEST_TAG}
    IMAGE_QUALNAME:      ${IMAGE_QUALNAME}
    Docker Hub tags:       ${REMOTE_IMAGE}

  HELM staging:
    HELM_REMOTE:           ${HELM_REMOTE}
    HELM_GIT_REF:          ${HELM_GIT_REF}
    STAGING_ROOT:          ${STAGING_ROOT}
    STAGING_HELM_DIR:      ${STAGING_HELM_DIR}

  Python:
    PYTHON_VERSION:        ${PYTHON_VERSION}

  Docker Hub login / push:
    PUSH_IMAGES:           ${PUSH_IMAGES} ($([[ "$PUSH_IMAGES" -eq 1 ]] && echo "will push" || echo "build only, no push"))
    LOGIN_DOCKERHUB:       ${LOGIN_DOCKERHUB}
    SERVER_USERNAME:    ${SERVER_USERNAME:-<unset>}
    SERVER_TOKEN:       ${token_status}

No Docker build or push has been run yet.
EOF
}

main() {
  # -----------------------
  # Discover paths
  # -----------------------
  local script_dir repo_root dockerfiles_dir magnet_heim_dockerfile
  script_dir="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
    pwd
  )"
  repo_root="$(realpath "${script_dir}/..")"
  dockerfiles_dir="${script_dir}"
  magnet_heim_dockerfile="${dockerfiles_dir}/magnet-heim.dockerfile"

  if [[ ! -f "${magnet_heim_dockerfile}" ]]; then
    die "magnet-heim.dockerfile not found at ${magnet_heim_dockerfile}"
  fi

  # Base image:
  #   - First positional arg (if provided and not a flag)
  #   - Else BASE_IMAGE from env
  local arg_base="${1:-}"
  if [[ "$arg_base" != "" && "$arg_base" != -* ]]; then
    BASE_IMAGE="$arg_base"
    shift || true
  else
    : "${BASE_IMAGE:=}"
  fi

  if [[ -z "${BASE_IMAGE}" ]]; then
    die "BASE_IMAGE is required (either as env or first argument).
Example:
  BASE_IMAGE=erotemic/magnet:some-tag $0
  $0 erotemic/magnet:some-tag"
  fi

  # HELM remote must be provided
  if [[ -z "${HELM_REMOTE}" ]]; then
    die "HELM_REMOTE is required (git URL or local path to HELM repo)."
  fi

  # Derive base tag from the base image (text after last ':')
  BASE_TAG="${BASE_IMAGE##*:}"
  if [[ "$BASE_TAG" == "$BASE_IMAGE" ]]; then
    die "BASE_IMAGE must include a tag, e.g. erotemic/magnet:some-tag (got '$BASE_IMAGE')"
  fi

  # Derive python version if not already set.
  if [[ -z "${PYTHON_VERSION:-}" ]]; then
    PYTHON_VERSION="$(infer_python_version_from_tag "${BASE_IMAGE}")"
  fi


  # Staging paths for HELM
  : "${STAGING_ROOT:=${repo_root}/.staging}"
  STAGING_HELM_DIR="${STAGING_ROOT}/helm"

  # Heim tag and image names
  MAGNET_HEIM_TAG="${BASE_TAG}-heim"
  IMAGE_QUALNAME="${APP_NAME}:${MAGNET_HEIM_TAG}"

  REMOTE_IMAGE="${DOCKER_REPO}/${APP_NAME}:${MAGNET_HEIM_TAG}"

  ALIASES=()
  # Always have the base latest alias
  ALIASES+=("${APP_NAME}:${LATEST_TAG}-heim")

  # Conditionally add python-specific alias, if we could infer a version.
  if [[ -n "${PYTHON_VERSION:-}" ]]; then
    ALIASES+=("${APP_NAME}:${LATEST_TAG}-heim-python${PYTHON_VERSION}")
  fi

  # -----------------------
  # Print configuration summary
  # -----------------------
  print_summary

  # -----------------------
  # Prepare HELM staging
  # -----------------------
  prepare_staging_repo "${STAGING_HELM_DIR}" "${HELM_REMOTE}" "${HELM_GIT_REF}"

  # -----------------------
  # Docker Hub login (optional)
  # -----------------------
  if [[ "${PUSH_IMAGES}" -eq 1 && "${LOGIN_DOCKERHUB}" -eq 1 ]]; then
    if [[ -n "${SERVER_TOKEN:-}" && -n "${SERVER_USERNAME:-}" ]]; then
      log "Logging in to Docker Hub as ${SERVER_USERNAME}"
      printf "%s" "$SERVER_TOKEN" | docker login -u "$SERVER_USERNAME" --password-stdin
    else
      log "Skipping non-interactive Docker Hub login (env vars not set). If needed, run: docker login"
    fi
  else
    log "Hub login step not required (LOGIN_DOCKERHUB=${LOGIN_DOCKERHUB}, PUSH_IMAGES=${PUSH_IMAGES})"
  fi

  # -----------------------
  # Build magnet+HELM image
  # -----------------------
  log "Building magnet+HELM image: ${IMAGE_QUALNAME}"
  DOCKER_BUILDKIT=1 docker build --progress=plain \
    -t "${IMAGE_QUALNAME}" \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    --build-arg EXPECT_PYTHON="${PYTHON_VERSION}" \
    --build-arg HEIM_GIT_REF="${HELM_GIT_REF}" \
    -f "${magnet_heim_dockerfile}" \
    "${repo_root}"

  # -----------------------
  # Tag local convenience aliases
  # -----------------------
  log "Tagging local developer aliases"
  for alias in "${ALIASES[@]}"; do
    log "docker tag ${IMAGE_QUALNAME} ${alias}"
    docker tag "${IMAGE_QUALNAME}" "${alias}"
  done


  # -----------------------
  # Tag Hub names
  # -----------------------
  REMOTE_TAGS=()
  REMOTE_TAGS+=("${REMOTE_IMAGE}")

  for alias in "${ALIASES[@]}"; do
    # alias looks like "${APP_NAME}:tag"; we’ll prepend DOCKER_REPO.
    local_tag="${alias#"${APP_NAME}":}"
    REMOTE_TAGS+=("${DOCKER_REPO}/${APP_NAME}:${local_tag}")
  done

  log "Remote tags to push (if enabled):"
  for tag in "${REMOTE_TAGS[@]}"; do
    log "  - ${tag}"
    docker tag "${IMAGE_QUALNAME}" "${tag}"
  done

  # =========================
  # Push (optional)
  # =========================

  if [[ "${PUSH_IMAGES}" -eq 1 ]]; then
    log "PUSH_IMAGES=1 → pushing to ${DOCKER_REPO}/${APP_NAME}"
    for tag in "${REMOTE_TAGS[@]}"; do
      log "docker push ${tag}"
      docker push "${tag}"
    done
  else
    log "Images were NOT pushed because PUSH_IMAGES=0."
  fi

  #
  log "Tagging Hub HEIM images"
  docker tag "${IMAGE_QUALNAME}" "${REMOTE_IMAGE}"

  # -----------------------
  # Final summary
  # -----------------------
  cat <<EOF

Done.

Base magnet image:
  ${BASE_IMAGE}

Built magnet+HELM image:
  ${IMAGE_QUALNAME}

  Local aliases:
$(for a in "${ALIASES[@]}"; do printf "    %s\n" "${a}"; done)

  Remote tags:
$(for t in "${REMOTE_TAGS[@]}"; do printf "    %s\n" "${t}"; done)

)

HELM staging:
  HELM_REMOTE   = ${HELM_REMOTE}
  HELM_GIT_REF  = ${HELM_GIT_REF}
  STAGING_HELM  = ${STAGING_HELM_DIR}

Quick smoke tests:
  docker run --rm -it ${IMAGE_QUALNAME} bash -lc 'python -V && uv --version'
  docker run --rm -it ${IMAGE_QUALNAME} python -c "import helm" || echo "import helm failed (check install)"
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
