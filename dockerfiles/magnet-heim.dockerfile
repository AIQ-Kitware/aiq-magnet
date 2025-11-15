# dockerfiles/magnet-heim.dockerfile
#
# Extends the base magnet image to:
#  - Assert the Python major.minor version
#  - Remove any site-packages-installed crfm-helm
#  - Copy a local HELM checkout (from .staging/helm), including .git
#  - Checkout the requested HELM ref
#  - Install HELM in editable mode with heim dependencies

# ------------------------------------------------------------------------------
# Base image: pass in the magnet image that was built from magnet.dockerfile.
# Example: --build-arg BASE_IMAGE=erotemic/magnet:latest-dev-python3.10
# ------------------------------------------------------------------------------
ARG BASE_IMAGE=erotemic/magnet:latest-dev
FROM ${BASE_IMAGE} AS magnet-heim

# Expected Python MAJOR.MINOR (asserted below)
ARG EXPECT_PYTHON=3.10
# ------------------------------------------------------------------------------
# Assert Python version is exactly the expected major.minor
# ------------------------------------------------------------------------------
RUN python - <<PYCHECK
import sys
expect = "${EXPECT_PYTHON}"
actual = f"{sys.version_info.major}.{sys.version_info.minor}"
if not expect.strip():
    print(f"No expected Python version, actual={actual}")
else:
    if actual != expect:
        raise SystemExit(f"ERROR: Expected Python {expect}, got {actual}")
    print(f"Python version check OK: {actual}")
PYCHECK

# ------------------------------------------------------------------------------
# Which HELM ref/branch/sha to use after copying the checkout
ARG HEIM_GIT_REF=HEAD

# ------------------------------------------------------------------------------
# Bring in your local HELM working tree (must include .git)
# Host path: ./.staging/helm → Image path: /root/code/helm
# ------------------------------------------------------------------------------
# Copy only the git metadata; the checkout will materialize the tree
COPY .staging/helm/.git /root/code/helm/.git

# Faster editable installs with uv; avoid modifying system site-packages
ENV UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache <<'EOF'
set -e

# Ensure we don't accidentally use a site-packages crfm-helm
uv pip uninstall -y crfm-helm || true

cd  /root/code/helm

# Checkout the requested branch 
git checkout "$HEIM_GIT_REF"
git reset --hard "$HEIM_GIT_REF"

# Install the repo in development mode
uv pip install -e .[heim] 

# Fix conflicting duplicate opencv installs
# Note HELM/HEIM does pin the version, so we do graph hacks to pick up whatever
# is currently in pyproject.toml 
HAS_OPENCV_RETCODE="0"
HAS_OPENCV_HEADLESS_RETCODE="0"
uv pip freeze | grep "opencv-python==" || HAS_OPENCV_RETCODE="$?"
uv pip freeze | grep "opencv-python-headless==" || HAS_OPENCV_HEADLESS_RETCODE="$?"

# VAR == 0 means we have it
if [[ "$HAS_OPENCV_HEADLESS_RETCODE" == "0" ]]; then
    if [[ "$HAS_OPENCV_RETCODE" == "0" ]]; then
        uv pip uninstall opencv-python opencv-python-headless
        uv pip install $(grep -oP '"\K([^"]+)' pyproject.toml | sed 's/[;,]\s*.*$//' | grep -E 'opencv-python-headless')
    fi
else
    if [[ "$HAS_OPENCV_RETCODE" == "0" ]]; then
        $PIP_COMMAND uninstall opencv-python
    fi
    uv pip install $(grep -oP '"\K([^"]+)' pyproject.toml | sed 's/[;,]\s*.*$//' | grep -E 'opencv-python-headless')
fi
EOF


ARG VCS_REF=""
ARG REPO_URL=""
ARG DOCKERFILE_PATH=""

LABEL org.opencontainers.image.title="MAGNET + HEIM" \
      org.opencontainers.image.description="MAGNET with HEIM dependencies" \
      org.opencontainers.image.url="$REPO_URL/-/blob/$VCS_REF/$DOCKERFILE_PATH" \
      org.opencontainers.image.source="$REPO_URL" \
      org.opencontainers.image.revision="$VCS_REF" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.authors="Jon Crall <jon.crall@kitware.com>, Kitware Inc." \
      org.opencontainers.image.vendor="Kitware Inc." 
