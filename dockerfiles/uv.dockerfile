# syntax=docker/dockerfile:1.5

# Defined in: https://gitlab.kitware.com/computer-vision/ci-docker/-/blob/main/uv.dockerfile

# Allow overriding the base image at build time
# Base should be a compatible ubuntu image.
# Other available tags: https://hub.docker.com/r/nvidia/cuda/tags
ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
FROM ${BASE_IMAGE} AS base

# ------------------------------------
# Step 1: Install System Prerequisites
# ------------------------------------

RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt/lists <<EOF
#!/bin/bash
set -e
apt update -q
DEBIAN_FRONTEND=noninteractive apt install -q -y --no-install-recommends \
    curl \
    wget \
    git \
    unzip \
    ca-certificates \
    build-essential 
# Note: normal image cleanup not needed with buildkit cache
EOF


# Note: the above apt command contains some nice to have extras for developer
# images, we can likely exclude these in a future optimized version.


# Set the shell to bash to auto-activate enviornments
SHELL ["/bin/bash", "-l", "-c"]

# ------------------
# Step 2: Install uv
# ------------------
# Here we take a few extra steps to pin to a verified version of the uv
# installer. This increases reproducibility and security against the main
# astral domain, but not against those linked in the main installer.
# The "normal" way to install the latest uv is:
# curl -LsSf https://astral.sh/uv/install.sh | bash

# Control the version of uv
ARG UV_VERSION=0.9.27

RUN --mount=type=cache,target=/root/.cache <<EOF
#!/bin/bash
set -e
mkdir /bootstrap
cd /bootstrap
# For new releases see: https://github.com/astral-sh/uv/releases
declare -A UV_INSTALL_KNOWN_HASHES=(
    ["0.9.27"]="6a2da893cd56f9e2d1c9140d0c434fd1de097358bedeb89add559ef3179c932e"
    ["0.8.8"]="f86123768d4602c5de570fe2e43d3ef0720e907d420aec8c663da81e41c8de57"
    ["0.8.4"]="601321180a10e0187c99d8a15baa5ccc11b03494c2ca1152fc06f5afeba0a460"
    ["0.7.20"]="3b7ca115ec2269966c22201b3a82a47227473bef2fe7066c62ea29603234f921"
    ["0.7.19"]="e636668977200d1733263a99d5ea66f39d4b463e324bb655522c8782d85a8861"
)
EXPECTED_SHA256="${UV_INSTALL_KNOWN_HASHES[${UV_VERSION}]}"
DOWNLOAD_PATH=uv-install-v${UV_VERSION}.sh
if [[ -z "$EXPECTED_SHA256" ]]; then
    echo "No hash known for UV_VERSION '$UV_VERSION'; version does not exist or hash entry is missing from internal table. Aborting."
    exit 1
fi
curl -LsSf https://astral.sh/uv/$UV_VERSION/install.sh > $DOWNLOAD_PATH
report_bad_checksum(){
    echo "Got unexpected checksum"
    sha256sum "$DOWNLOAD_PATH"
    exit 1
}
echo "$EXPECTED_SHA256  $DOWNLOAD_PATH" | sha256sum --check || report_bad_checksum
# Run the install script
bash /bootstrap/uv-install-v${UV_VERSION}.sh
EOF


# ------------------------------------------
# Step 3: Setup a Python virtual environment
# ------------------------------------------
# This step mirrors a normal virtualenv development environment inside the
# container, which can prevent subtle issues due when running as root inside
# containers. 

# Control which python version we are using
ARG PYTHON_VERSION=3.13

ENV PIP_ROOT_USER_ACTION=ignore

RUN --mount=type=cache,target=/root/.cache <<EOF
#!/bin/bash
export PATH="$HOME/.local/bin:$PATH"
# Use uv to install the requested python version and seed the venv
uv venv "/root/venv$PYTHON_VERSION" --python=$PYTHON_VERSION --seed
BASHRC_CONTENTS='
# setup a user-like environment, even though we are root
export HOME="/root"
export PATH="$HOME/.local/bin:$PATH"
# Auto-activate the venv on login
source $HOME/venv'$PYTHON_VERSION'/bin/activate
'
# It is important to add the content to both so 
# subsequent run commands use the context we setup here.
echo "$BASHRC_CONTENTS" >> $HOME/.bashrc
echo "$BASHRC_CONTENTS" >> $HOME/.profile
echo "$BASHRC_CONTENTS" >> $HOME/.bash_profile
EOF

# -----------------------------------
# Step 4: Ensure venv auto-activation
# -----------------------------------
# This step creates an entrypoint script that ensures any command passed to
# `docker run` is executed inside a login shell where the virtual environment
# is auto-activated. It handles complex cases like multi-arg commands and
# ensures quoting is preserved accurately.
RUN <<EOF
#!/bin/bash
set -e

# We use a quoted heredoc to write the entrypoint script literally, with no variable expansion.
cat <<'__EOSCRIPT__' > /entrypoint.sh
#!/bin/bash
set -e

# Reconstruct the full command line safely, quoting each argument
args=()
for arg in "$@"; do
  args+=("$(printf "%q" "$arg")")
done

# Join arguments into a command string that can be executed by bash -c
# This preserves exact argument semantics (including quotes, spaces, etc.)
cmd="${args[*]}"

# Execute the reconstructed command inside a login shell
# This ensures virtualenv activation via .bash_profile
exec bash -l -c "$cmd"
__EOSCRIPT__

# Print the script at build time for visibility/debugging
cat /entrypoint.sh

chmod +x /entrypoint.sh
EOF


# Set the entrypoint to our script that activates the virtual environment first
ENTRYPOINT ["/entrypoint.sh"]

# Set the default workdir to the user home directory
WORKDIR /root

# Increase startup speed of images:
# https://docs.astral.sh/uv/reference/cli/#uv-run--compile-bytecode
ENV UV_COMPILE_BYTECODE=1

ARG VCS_REF=""
ARG REPO_URL=""
ARG DOCKERFILE_PATH=""

LABEL org.opencontainers.image.title="uv Python" \
      org.opencontainers.image.description="uv ${UV_VERSION} and Python ${PYTHON_VERSION} in an auto-activating virtual environment with base image: ${BASE_IMAGE}." \
      org.opencontainers.image.source="$REPO_URL" \
      org.opencontainers.image.version="uv${UV_VERSION}-python${PYTHON_VERSION}" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.authors="Jon Crall <jon.crall@kitware.com>, Kitware Inc." \
      org.opencontainers.image.vendor="Kitware Inc." 

  # Note: don't include the git hash for an image that does not depend on repo
  # contents. It causes subsequent images to get rebuilt even if there is no
  # change.
  # org.opencontainers.image.url="$REPO_URL/-/blob/$VCS_REF/$DOCKERFILE_PATH" \
  # org.opencontainers.image.revision="$VCS_REF" \

#ARG CREATED=""
# Note sure if we really want a created tag as it messes with hashes
# Keep created layer separate
#LABEL org.opencontainers.image.created="$CREATED"


################
### __DOCS__ ###
################
RUN <<EOF
echo '
# https://www.docker.com/blog/introduction-to-heredocs-in-dockerfiles/

cd ~/code/ci-docker

# Build the image with version-specific tags
export IMAGE_NAME=uv
export UV_VERSION=0.8.8
export PYTHON_VERSION=3.11
export BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
export BASE_TAG=$(echo "$BASE_IMAGE" | sed "s#.*/##; s/://g")
export IMAGE_TAG=${UV_VERSION}-python${PYTHON_VERSION}-$BASE_TAG
export IMAGE_QUALNAME=$IMAGE_NAME:$IMAGE_TAG

echo "IMAGE_QUALNAME=$IMAGE_QUALNAME"

DOCKER_BUILDKIT=1 docker build --progress=plain \
    -t "$IMAGE_QUALNAME" \
    --build-arg UV_VERSION=$UV_VERSION \
    --build-arg PYTHON_VERSION=$PYTHON_VERSION \
    --build-arg BASE_IMAGE=$BASE_IMAGE \
    -f uv.dockerfile .

# Test the build
test_build(){
    docker run --gpus=all -it $IMAGE_QUALNAME nvidia-smi
    docker run --gpus=all -it $IMAGE_QUALNAME echo "hello" "world"
    docker run --gpus=all -it $IMAGE_QUALNAME du -sh /root/.cache
    #docker run --gpus=all -it $IMAGE_QUALNAME bash
}

# Inspect the build
inspect_build(){
    docker image history $IMAGE_QUALNAME --no-trunc
    docker image history $IMAGE_QUALNAME
    docker image inspect $IMAGE_QUALNAME
}

# Add concise tags for easier reuse
export ALIAS1=$IMAGE_NAME:latest
export ALIAS2=$IMAGE_NAME:latest-python${PYTHON_VERSION}
docker tag $IMAGE_QUALNAME $ALIAS1
docker tag $IMAGE_QUALNAME $ALIAS2

# Push to dockerhub
export REMOTE_NAME="erotemic"

# 3) Create remote-qualified tags
docker tag $IMAGE_QUALNAME $REMOTE_NAME/$IMAGE_QUALNAME
docker tag $ALIAS1  $REMOTE_NAME/$ALIAS1
docker tag $ALIAS2  $REMOTE_NAME/$ALIAS2

# 4) Push the tags
docker push $REMOTE_NAME/$IMAGE_QUALNAME
docker push $REMOTE_NAME/$ALIAS1
docker push $REMOTE_NAME/$ALIAS2

' > /dev/null
EOF
