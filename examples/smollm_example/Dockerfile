# The image this example's node commands run in.
#
#   ./run.sh                    builds this and uses it by default
#   ./run.sh --no-container     skips it and runs node commands on the host
#
# MAGNET is installed from a pinned source artifact instead of COPYing the
# checkout into the image. That keeps the node image independent of local card,
# catalog, documentation, and run-output changes. Once aiq-magnet is released,
# replace this default with a pinned package requirement such as
# `aiq-magnet==0.1.0`.
FROM python:3.12-slim

ARG MAGNET_INSTALL="https://github.com/Erotemic/aiq-magnet/archive/cf9cc968d7a88470657e7938addfb3a1a6d0f986.tar.gz"

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN pip install "$MAGNET_INSTALL"

# Only the executable part of the example belongs in the node image. The card,
# pipeline, catalogs, README, and run.sh are orchestration inputs used on the
# host and should not invalidate this image when they change.
WORKDIR /opt/examples
COPY __init__.py /opt/examples/smollm_example/__init__.py
COPY cli /opt/examples/smollm_example/cli
ENV PYTHONPATH=/opt/examples

# A node command is run with `-w "$PWD"` and its working directory bind-mounted
# at the same absolute path it has on the host, so kwdagger's absolute output
# paths resolve identically inside and out.
WORKDIR /
