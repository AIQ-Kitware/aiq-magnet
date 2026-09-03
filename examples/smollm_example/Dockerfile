# Node image for the SmolLM example.
FROM python:3.12-slim

ARG MAGNET_INSTALL="https://github.com/Erotemic/aiq-magnet/archive/cf9cc968d7a88470657e7938addfb3a1a6d0f986.tar.gz"

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN pip install "$MAGNET_INSTALL"

# Only the executable CLI package is needed in the node image.
WORKDIR /opt/examples
COPY __init__.py /opt/examples/smollm_example/__init__.py
COPY cli /opt/examples/smollm_example/cli
ENV PYTHONPATH=/opt/examples

# MAGNET supplies the bind-mounted node working directory at runtime.
WORKDIR /
