# AIQ-MAGNET Container Build

This repo contains two Dockerfiles:

- `uv.dockerfile` — a fast, reproducible **Python + uv** base (GPU-ready by default).
- `magnet.dockerfile` — a **generic app image** that inherits from the `uv` base and installs the repo code.

The goal is to keep the Python/uv/tooling layer stable and cached, while allowing frequent app changes.

---

## 1) Build the `uv` base image

Choose your Python and uv versions using build args (defaults are defined in `uv.dockerfile`). Example:

```bash
# Tag variables (edit to taste)
# Choose versions
export IMAGE_NAME=uv
export UV_VERSION=0.8.4
export PYTHON_VERSION=3.13
export BASE_IMAGE=nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04
export BASE_TAG=$(echo "$BASE_IMAGE" | sed "s#.*/##; s/://g")
export UV_IMAGE_TAG=${UV_VERSION}-python${PYTHON_VERSION}-$BASE_TAG
export UV_IMAGE_QUALNAME=$IMAGE_NAME:$UV_IMAGE_TAG

echo "UV_IMAGE_QUALNAME=$UV_IMAGE_QUALNAME"

# Build the uv base image
DOCKER_BUILDKIT=1 docker build --progress=plain \
  -t "$UV_IMAGE_QUALNAME" \
  --build-arg UV_VERSION=$UV_VERSION \
  --build-arg PYTHON_VERSION=$PYTHON_VERSION \
  --build-arg BASE_IMAGE=$BASE_IMAGE \
  --build-arg VCS_REF=$(git rev-parse HEAD) \
  --build-arg REPO_URL=$(git config --get remote.origin.url | sed 's/\.git$//') \
  --build-arg DOCKERFILE_PATH=uv.dockerfile \
  -f dockerfiles/uv.dockerfile .
```

**Smoke test the base:**

```bash
docker run --rm -it $UV_IMAGE_QUALNAME uv --version
docker run --rm -it $UV_IMAGE_QUALNAME python -V
# GPU (optional):
docker run --rm --gpus=all -it $UV_IMAGE_QUALNAME nvidia-smi
```

Note: A variant of this image is prebuild on https://hub.docker.com/repository/docker/erotemic/uv/general

---

## 2) Build the app image (`magnet.dockerfile`)


```bash
# --- Build the magnet app image using the uv tag ---
# Build against a specific git version
export GIT_REF=$(git rev-parse --short=12 HEAD)

export MAGNET_IMAGE_NAME=magnet
export UV_BASE_IMAGE=uv:$UV_IMAGE_TAG # exact uv base
export MAGNET_IMAGE_TAG=$GIT_REF-uv$UV_IMAGE_TAG # inherit uv tag
export MAGNET_IMAGE_QUALNAME=$MAGNET_IMAGE_NAME:$MAGNET_IMAGE_TAG

DOCKER_BUILDKIT=1 docker build --progress=plain \
  -t "$MAGNET_IMAGE_QUALNAME" \
  --build-arg UV_BASE="$UV_IMAGE_QUALNAME" \
  --build-arg GIT_REF="$GIT_REF" \
  -f dockerfiles/magnet.dockerfile .

# Helpful aliases
docker tag "$MAGNET_IMAGE_QUALNAME" $MAGNET_IMAGE_NAME:latest
docker tag "$MAGNET_IMAGE_QUALNAME" $MAGNET_IMAGE_NAME:latest-python${PYTHON_VERSION}
```

**Smoke test the repo:**

```bash
docker run --rm -it $MAGNET_IMAGE_QUALNAME bash -lc 'python -V && uv --version'
docker run --rm --gpus=all -it $MAGNET_IMAGE_QUALNAME nvidia-smi   # optional
```


**Run unit tests** 
```bash
docker run --rm -it $MAGNET_IMAGE_QUALNAME pytest
```
