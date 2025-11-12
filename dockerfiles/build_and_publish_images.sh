#!/usr/bin/env bash
# dockerfiles/build_and_publish_images.sh
#
# Build a developer-tagged magnet image from a uv base and publish to Docker Hub.
# - Can be run from ANY working directory.
# - Assumes this script is inside the repo at: repo_root/dockerfiles/build_and_publish_images.sh
#
# Tags pushed to Docker Hub:
#   erotemic/magnet:<GIT12>-uv<uv-tag>
#   erotemic/magnet:latest-dev
#   erotemic/magnet:latest-dev-python<MAJOR.MINOR>

set -euo pipefail

# -------------------------
# Locate paths relative to this script (runnable from anywhere)
# -------------------------
SCRIPT_DIR="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd
)"
REPO_ROOT="$(realpath "${SCRIPT_DIR}/..")"
DOCKERFILES_DIR="${SCRIPT_DIR}"
MAGNET_DOCKERFILE="${DOCKERFILES_DIR}/magnet.dockerfile"

if [[ ! -f "${MAGNET_DOCKERFILE}" ]]; then
  echo "[error] magnet.dockerfile not found at ${MAGNET_DOCKERFILE}" >&2
  exit 1
fi

# -------------------------
# Config (override via env)
# -------------------------
: "${APP_NAME:=magnet}"
: "${DOCKERHUB_NAMESPACE:=erotemic}"          # Docker Hub namespace
: "${MAGNET_REPO_DIR:=${REPO_ROOT}}"           # Default to repo root resolved above

# UV base selection (default mirrors your example)
: "${UV_IMAGE_TAG:=0.8.4-python3.10-cuda12.4.1-cudnn-devel-ubuntu22.04}"
: "${UV_IMAGE_NAME:=uv}"
: "${UV_IMAGE_QUALNAME:=${UV_IMAGE_NAME}:${UV_IMAGE_TAG}}"

# Registry hosting the uv base (pull source)
#   Option A (default): your GitLab CI registry (private)
#   Option B: Docker Hub copy under erotemic (set UV_PULL_FROM_DOCKERHUB=1)
: "${UV_GITLAB_REGISTRY:=gitlab.kitware.com:4567/computer-vision/ci-docker}"
: "${UV_PULL_FROM_DOCKERHUB:=0}"

# Build args for magnet
: "${USE_LOCKFILE:=0}"

# Optional logins (supply tokens via env for non-interactive)
#   DOCKERHUB_USERNAME / DOCKERHUB_TOKEN
#   GITLAB_KITWARE_USERNAME / GITLAB_KITWARE_TOKEN
: "${LOGIN_GITLAB:=1}"      # set 0 to skip GitLab login/pull if using Docker Hub uv base
: "${LOGIN_DOCKERHUB:=1}"   # set 0 to skip Docker Hub login

# -------------------------
# Helpers
# -------------------------
log(){ printf "\033[1;34m[build]\033[0m %s\n" "$*"; }
die(){ printf "\033[1;31m[error]\033[0m %s\n" "$*" >&2; exit 1; }

make_vcs_ref(){
  # Full commit with "-dirty" suffix if working tree isn't clean
  local sha is_dirty=false
  sha="$(git rev-parse HEAD)"
  git diff --quiet || is_dirty=true
  git diff --cached --quiet || is_dirty=true
  if [ -n "$(git -c core.excludesFile=/dev/null ls-files --others --exclude-standard)" ]; then
    is_dirty=true
  fi
  #$is_dirty && echo "${sha}-dirty" || echo "${sha}"
  # Dont do dirty suffix, it breaks things.
  $is_dirty && echo "${sha}" || echo "${sha}"
}

short12(){
  # 12-char short sha (handles "-dirty" suffix)
  local ref="$1"
  local core="${ref%%-*}"
  local suf="${ref#"$core"}"   # SC2295-safe: quote inner expansion
  echo "$(printf "%s" "$core" | cut -c1-12)${suf}"
}

infer_python_version_from_uv_tag(){
  # Parse ...python3.10... from UV_IMAGE_TAG
  local tag="$1"
  if [[ "$tag" =~ python([0-9]+\.[0-9]+) ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo ""
  fi
}

# -------------------------
# Resolve inputs / context
# -------------------------
[ -d "$MAGNET_REPO_DIR" ] || die "MAGNET_REPO_DIR not found: $MAGNET_REPO_DIR"
cd "$MAGNET_REPO_DIR"

GIT_REF_FULL="$(make_vcs_ref)"
GIT_REF_SHORT="$(short12 "$GIT_REF_FULL")"

PYTHON_VERSION="${PYTHON_VERSION:-$(infer_python_version_from_uv_tag "$UV_IMAGE_TAG")}"
[ -n "$PYTHON_VERSION" ] || die "Could not infer PYTHON_VERSION from UV_IMAGE_TAG='$UV_IMAGE_TAG'. Set PYTHON_VERSION explicitly."

if [ "${UV_PULL_FROM_DOCKERHUB}" = "1" ]; then
  UV_BASE="${DOCKERHUB_NAMESPACE}/${UV_IMAGE_QUALNAME}"   # docker.io/erotemic/uv:...
  LOGIN_GITLAB=0
else
  UV_BASE="${UV_GITLAB_REGISTRY}/${UV_IMAGE_QUALNAME}"    # gitlab.kitware.com:4567/.../uv:...
fi

MAGNET_IMAGE_TAG="${GIT_REF_SHORT}-uv${UV_IMAGE_TAG}"     # e.g. a1b2c3d4e5f6-uv0.8.4-python3.10-...
LOCAL_IMAGE="${APP_NAME}:${MAGNET_IMAGE_TAG}"

# Docker Hub remote names
HUB_IMAGE="${DOCKERHUB_NAMESPACE}/${APP_NAME}:${MAGNET_IMAGE_TAG}"
HUB_ALIAS_LATEST_DEV="${DOCKERHUB_NAMESPACE}/${APP_NAME}:latest-dev"
HUB_ALIAS_LATEST_DEV_PY="${DOCKERHUB_NAMESPACE}/${APP_NAME}:latest-dev-python${PYTHON_VERSION}"

# Optional local developer aliases (handy for quick runs)
LOCAL_ALIAS_LATEST_DEV="${APP_NAME}:latest-dev"
LOCAL_ALIAS_LATEST_DEV_PY="${APP_NAME}:latest-dev-python${PYTHON_VERSION}"

# -------------------------
# Log in (optional)
# -------------------------
if [ "${LOGIN_GITLAB}" = "1" ]; then
  if [ -n "${GITLAB_KITWARE_TOKEN:-}" ] && [ -n "${GITLAB_KITWARE_USERNAME:-}" ]; then
    log "Logging in to GitLab registry"
    printf "%s" "$GITLAB_KITWARE_TOKEN" | docker login "gitlab.kitware.com:4567" -u "$GITLAB_KITWARE_USERNAME" --password-stdin
  else
    log "Skipping non-interactive GitLab login (env vars not set). If needed, run: docker login gitlab.kitware.com:4567"
  fi
fi

if [ "${LOGIN_DOCKERHUB}" = "1" ]; then
  if [ -n "${DOCKERHUB_TOKEN:-}" ] && [ -n "${DOCKERHUB_USERNAME:-}" ]; then
    log "Logging in to Docker Hub"
    printf "%s" "$DOCKERHUB_TOKEN" | docker login -u "$DOCKERHUB_USERNAME" --password-stdin
  else
    log "Skipping non-interactive Docker Hub login (env vars not set). If needed, run: docker login"
  fi
fi

# -------------------------
# Pull uv base
# -------------------------
log "Pulling uv base: ${UV_BASE}"
docker pull "${UV_BASE}"

# -------------------------
# Build magnet (context = repo root; Dockerfile lives in dockerfiles/)
# -------------------------
log "Building ${LOCAL_IMAGE}"
DOCKER_BUILDKIT=1 docker build --progress=plain \
  -t "${LOCAL_IMAGE}" \
  --build-arg USE_LOCKFILE="${USE_LOCKFILE}" \
  --build-arg UV_BASE="${UV_BASE}" \
  --build-arg GIT_REF="${GIT_REF_FULL}" \
  -f "${MAGNET_DOCKERFILE}" \
  "${REPO_ROOT}"

# -------------------------
# Tag local convenience aliases
# -------------------------
log "Tagging local developer aliases"
docker tag "${LOCAL_IMAGE}" "${LOCAL_ALIAS_LATEST_DEV}"
docker tag "${LOCAL_IMAGE}" "${LOCAL_ALIAS_LATEST_DEV_PY}"

# -------------------------
# Tag Docker Hub names
# -------------------------
log "Tagging Docker Hub images"
docker tag "${LOCAL_IMAGE}" "${HUB_IMAGE}"
docker tag "${LOCAL_IMAGE}" "${HUB_ALIAS_LATEST_DEV}"
docker tag "${LOCAL_IMAGE}" "${HUB_ALIAS_LATEST_DEV_PY}"

# -------------------------
# Push
# -------------------------
log "Pushing to Docker Hub → ${DOCKERHUB_NAMESPACE}/${APP_NAME}"
for TAG in \
  "${HUB_IMAGE}" \
  "${HUB_ALIAS_LATEST_DEV}" \
  "${HUB_ALIAS_LATEST_DEV_PY}"
do
  log "docker push ${TAG}"
  docker push "${TAG}"
done

# -------------------------
# Summary + helpful runs
# -------------------------
cat <<EOF

Done.

Built:
  ${LOCAL_IMAGE}

Pushed:
  ${HUB_IMAGE}
  ${HUB_ALIAS_LATEST_DEV}
  ${HUB_ALIAS_LATEST_DEV_PY}

Local dev aliases:
  ${LOCAL_ALIAS_LATEST_DEV}
  ${LOCAL_ALIAS_LATEST_DEV_PY}

Quick smoke tests:
  docker run --rm -it ${LOCAL_IMAGE} bash -lc 'python -V && uv --version'
  docker run --rm --gpus=all -it ${LOCAL_IMAGE} nvidia-smi   # optional
  docker run --rm -it ${LOCAL_IMAGE} pytest                  # optional

Notes:
- Script is location-agnostic: resolves repo root at \`$(basename "${REPO_ROOT}")\` and Dockerfile at \`${MAGNET_DOCKERFILE}\`.
- Set UV_PULL_FROM_DOCKERHUB=1 to pull uv from docker.io/${DOCKERHUB_NAMESPACE}/${UV_IMAGE_QUALNAME}
  instead of ${UV_GITLAB_REGISTRY}/${UV_IMAGE_QUALNAME}.
- To avoid clobbering production 'latest', this script publishes:
    - ${DOCKERHUB_NAMESPACE}/${APP_NAME}:latest-dev
    - ${DOCKERHUB_NAMESPACE}/${APP_NAME}:latest-dev-python${PYTHON_VERSION}
  alongside the content-addressed tag:
    - ${DOCKERHUB_NAMESPACE}/${APP_NAME}:${MAGNET_IMAGE_TAG}
EOF
