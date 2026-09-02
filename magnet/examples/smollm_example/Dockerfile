# The image this example's node commands run in.
#
#   ./run.sh --container        builds this and uses it
#
# Small on purpose. MAGNET's core is cards, claims and kwdagger execution --
# `pip install .` with no extras -- so this is a slim Python base plus about
# eighteen pure-Python packages. It is not a GPU image and it holds no weights.
#
# Note what is NOT installed: the `leasing` extra. `infer-stack run` executes
# on the HOST, outside this container -- acquiring a lease needs the Docker
# daemon and the shared ledger, and the container is what the lease wraps. All
# that reaches in here is OPENAI_BASE_URL and OPENAI_API_KEY, and the node
# talks to that endpoint over plain HTTP. If this image ever needs infer-stack,
# something has been layered the wrong way round.
#
# The build context is the magnet repository root, so `pip install` sees the
# sources. Nothing from the host is baked in beyond that.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/magnet
COPY . /opt/magnet
RUN pip install .

# A node command is run with `-w "$PWD"` and its working directory bind-mounted
# at the same absolute path it has on the host, so kwdagger's absolute output
# paths resolve identically inside and out. Nothing here assumes a location.
WORKDIR /
