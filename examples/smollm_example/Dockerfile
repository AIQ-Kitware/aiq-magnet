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
# sources and the COPY picks up `examples/` as well -- this example lives beside
# magnet rather than inside it, so it is not installed by `pip install .` and has
# to be put on the path itself. Nothing from the host is baked in beyond that.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/magnet
COPY . /opt/magnet
RUN pip install .

# The example is a package beside magnet, not part of it, exactly as a team's
# own repository would be. `python -m smollm_example.cli.*` finds it here.
ENV PYTHONPATH=/opt/magnet/examples

# A node command is run with `-w "$PWD"` and its working directory bind-mounted
# at the same absolute path it has on the host, so kwdagger's absolute output
# paths resolve identically inside and out. Nothing here assumes a location.
WORKDIR /
