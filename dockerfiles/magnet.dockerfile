# syntax=docker/dockerfile:1.6
# See tail for quickbuild instructions.
ARG UV_BASE=uv:latest
FROM ${UV_BASE}


# ------------------------------------
# Step 1: Install System Prerequisites
# ------------------------------------

RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt/lists <<EOF
#!/bin/bash
set -e
apt update -q
DEBIAN_FRONTEND=noninteractive apt install -q -y --no-install-recommends \
    unzip \
# Note: normal image cleanup not needed with buildkit cache
EOF

# ---------------------------------
# Step 2: Checkout and install REPO
# ---------------------------------
# Based on the state of the repo this copies the host .git data over and then
# checks out the exact version requested by GIT_REF. It then performs a basic
# install of the project into the virtual environment.

ENV REPO_DNAME=aiq-magnet
ARG USE_LOCKFILE=1
RUN mkdir -p /root/code/$REPO_DNAME
WORKDIR /root/code/${REPO_DNAME}

# Copy only the git metadata; the checkout will materialize the tree
COPY .git /root/code/${REPO_DNAME}/.git

# Control the version/commit to checkout (default: current branch tip)
ARG GIT_REF=HEAD

# Faster editable installs with uv; avoid modifying system site-packages
ENV UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache <<'EOF'
set -e

cd  /root/code/${REPO_DNAME}

# Checkout the requested branch 
git checkout "$GIT_REF"
git reset --hard "$GIT_REF"

# First install pinned requirements for reproducibility
if [[ "$USE_LOCKFILE" -eq 1 ]]; then
  uv pip install -r requirements.lock.txt
fi

# Install the repo in development mode
uv pip install -e .[tests] 

EOF

# Default workdir to the repo
WORKDIR /root/code/${REPO_DNAME}

# See README.md for full usage instructions.


################
### __DOCS__ ###
################
RUN <<EOF
echo '
# Local-only app image that layers on the uv base image.
# Build the base first (tagged as uv:latest), then build this image.

# navigate to the magnet repo
cd $HOME/code/aiq-magnet

docker build -f dockerfiles/uv.dockerfile -t uv:latest .
docker build -f dockerfiles/magnet.dockerfile -t magnet:latest .


' > /dev/null
EOF

