#!/usr/bin/env bash
__doc__="
Download HELM benchmark run artifacts from the public GCS bucket.

NOTE:
   This bash script is superceded by a Python script in the main repo.
   For details see:
       python -m magnet.backends.helm.download_helm_results --help


Features:
  - Auto-detect latest version if version=auto (default)
  - Prompt to install gsutil if missing (Debian/Ubuntu auto-install supported)
  - Choose benchmark name (e.g., lite, helm)
  - Key/value args: dir=, version=, benchmark=, checksum=, install=
  - Flags: --list-benchmarks, --list-versions, --verbose
  - Idempotent using 'gsutil rsync' (optionally checksum mode)

Usage:
  ./download_helm_results.sh <download_dir> [version]
  ./download_helm_results.sh dir=<download_dir> [version=auto] [benchmark=lite]
  ./download_helm_results.sh --list-benchmarks
  ./download_helm_results.sh --list-versions [--benchmark=lite]

Examples:
  ./download_helm_results.sh /data/crfm-helm-public
  ./download_helm_results.sh /data/crfm-helm-public --benchmark=helm
  ./download_helm_results.sh /data/crfm-helm-public --benchmark=lite --version=v1.9.0
  ./download_helm_results.sh dir=./data version=auto benchmark=lite

Notes:
  - Requires: gsutil (Google Cloud SDK)
  - Idempotent: uses 'gsutil rsync' to sync only new/changed files
  - See [2]_ for available precomputed results

References:
    .. [1] https://crfm-helm.readthedocs.io/en/latest/downloading_raw_results/
    .. [2] https://console.cloud.google.com/storage/browser/crfm-helm-public
"

set -euo pipefail

# Defaults
DOWNLOAD_DIR=""
VERSION="auto"
BENCHMARK_NAME="lite"
CHECKSUM=0
INSTALL=0
LIST_BENCHMARKS=0
LIST_VERSIONS=0
VERBOSE=0

BUCKET="gs://crfm-helm-public"

logv() { ([[ "$VERBOSE" -eq 1 ]] && echo "$@" >&2) || true; }

# ------------------------------
# Helper: check if gsutil is the right one
is_google_gsutil() {
  gsutil version 2>/dev/null | grep -qi '^gsutil version:'
}

install_gsutil_ubuntu() {
  echo "Installing Google Cloud SDK (gsutil) via apt..."
  sudo apt-get update -y
  sudo apt-get install -y apt-transport-https ca-certificates gnupg curl
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
  echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list >/dev/null
  sudo apt-get update -y
  sudo apt-get install -y google-cloud-cli
}

ensure_gsutil() {
  if command -v gsutil >/dev/null 2>&1 && is_google_gsutil; then
    return 0
  fi
  echo "Google Cloud 'gsutil' not found (or a conflicting 'gsutil' is first on PATH)."

  if [[ "${INSTALL:-0}" == "1" ]]; then
    if command -v apt-get >/dev/null 2>&1; then
      install_gsutil_ubuntu
    else
      echo "Automatic install only implemented for Debian/Ubuntu (apt)." >&2
      echo "Install instructions: https://cloud.google.com/sdk/docs/install" >&2
      exit 1
    fi
  else
    # Only prompt if running in a TTY
    if [[ -t 0 ]] && command -v apt-get >/dev/null 2>&1; then
      read -r -p "Install gsutil now via apt on Debian/Ubuntu? [y/N] " yn || true
      case "${yn:-N}" in
        [Yy]*) install_gsutil_ubuntu ;;
        *)     echo "Please install Google Cloud SDK manually and retry." >&2; exit 1 ;;
      esac
    else
      echo "Please install Google Cloud SDK and retry: https://cloud.google.com/sdk/docs/install" >&2
      exit 1
    fi
  fi

  # Verify post-install
  if ! command -v gsutil >/dev/null 2>&1 || ! is_google_gsutil; then
    echo "Error: gsutil still not available or not the Google Cloud version." >&2
    exit 1
  fi
}

latest_version() {
  local bench="$1"
  local runs_path="${BUCKET}/${bench}/benchmark_output/runs"
  gsutil ls "${runs_path}/" \
    | sed -E "s#${runs_path}/([^/]+)/?#\1#" \
    | sed '/^$/d' \
    | sort -V \
    | tail -n 1
}

list_benchmarks() {
  gsutil ls "${BUCKET}/" \
    | sed -E 's#gs://crfm-helm-public/([^/]+)/?#\1#' \
    | sed '/^$/d'
}

list_versions() {
  local bench="$1"
  local runs_path="${BUCKET}/${bench}/benchmark_output/runs"
  gsutil ls "${runs_path}/" \
    | sed -E "s#${runs_path}/([^/]+)/?#\1#" \
    | sed '/^$/d' \
    | sort -V
}

print_help_and_exit() {
  printf "%s\n" "$__doc__"
  exit 0
}

# ------------------------------
# Parse args
if [[ "${1-}" == "-h" || "${1-}" == "--help" ]]; then
  print_help_and_exit
fi

# Positional compatibility: <download_dir> [version]
if [[ $# -ge 1 && "$1" != -* && "$1" != *=* ]]; then
  DOWNLOAD_DIR="$1"; shift
fi
if [[ $# -ge 1 && "$1" != -* && "$1" != *=* ]]; then
  VERSION="$1"; shift
fi

# Key=value without leading dashes (dir=, version=, etc.)
for arg in "$@"; do
  case "$arg" in
    dir=*) DOWNLOAD_DIR="${arg#dir=}" ;;
    version=*) VERSION="${arg#version=}" ;;
    benchmark=*) BENCHMARK_NAME="${arg#benchmark=}" ;;
    checksum=*) CHECKSUM="${arg#checksum=}" ;;
    install=*) INSTALL="${arg#install=}" ;;
    verbose=*) VERBOSE="${arg#verbose=}" ;;
  esac
done

# Flags and --key=value
for arg in "$@"; do
  case "$arg" in
    --help|-h) print_help_and_exit ;;
    --install) INSTALL=1 ;;
    --list-benchmarks) LIST_BENCHMARKS=1 ;;
    --list-versions) LIST_VERSIONS=1 ;;
    --verbose) VERBOSE=1 ;;
    --dir=*) DOWNLOAD_DIR="${arg#--dir=}" ;;
    --version=*) VERSION="${arg#--version=}" ;;
    --benchmark=*) BENCHMARK_NAME="${arg#--benchmark=}" ;;
    --checksum=*) CHECKSUM="${arg#--checksum=}" ;;
    --install=*) INSTALL="${arg#--install=}" ;;
    --verbose=*) VERBOSE="${arg#--verbose=}" ;;
    --*) echo "Unknown flag: $arg" >&2; exit 2 ;;
  esac
done

logv "DOWNLOAD_DIR=$DOWNLOAD_DIR"
logv "VERSION=$VERSION"
logv "BENCHMARK_NAME=$BENCHMARK_NAME"
logv "CHECKSUM=$CHECKSUM"
logv "INSTALL=$INSTALL"
logv "LIST_BENCHMARKS=$LIST_BENCHMARKS"
logv "LIST_VERSIONS=$LIST_VERSIONS"
logv "VERBOSE=$VERBOSE"

# Listing modes (no dir required)
if [[ "${LIST_BENCHMARKS}" -eq 1 ]]; then
  ensure_gsutil
  echo "Available benchmarks in ${BUCKET}:"
  list_benchmarks
  exit 0
fi

if [[ "${LIST_VERSIONS}" -eq 1 ]]; then
  ensure_gsutil
  echo "Available versions for benchmark '${BENCHMARK_NAME}':"
  list_versions "${BENCHMARK_NAME}" || true
  exit 0
fi

# Require a destination directory for sync
if [[ -z "${DOWNLOAD_DIR}" ]]; then
  echo "Error: download directory not provided." >&2
  echo
  print_help_and_exit
fi

# Ensure gsutil present (or install)
ensure_gsutil

# Resolve version if auto
if [[ "${VERSION}" == "auto" ]]; then
  echo "Resolving latest version for benchmark '${BENCHMARK_NAME}'..."
  VERSION="$(latest_version "${BENCHMARK_NAME}")"
  if [[ -z "${VERSION}" ]]; then
    echo "Error: could not determine latest version." >&2
    exit 1
  fi
  echo "Using latest version: ${VERSION}"
fi

# ------------------------------
# Build paths and sync
BUCKET_BASE="${BUCKET}/${BENCHMARK_NAME}/benchmark_output/runs"
SRC="${BUCKET_BASE}/${VERSION}"
DEST_ROOT="${DOWNLOAD_DIR%/}/${BENCHMARK_NAME}/benchmark_output/runs"
DEST="${DEST_ROOT}/${VERSION}"

echo "HELM benchmark: ${BENCHMARK_NAME}"
echo "Version:        ${VERSION}"
echo "Source:         ${SRC}"
echo "Destination:    ${DEST}"
echo

mkdir -p "${DEST}"

# Idempotent sync
if [[ "${CHECKSUM}" -eq 1 ]]; then
  gsutil -m rsync -r -c "${SRC}" "${DEST}"
else
  gsutil -m rsync -r "${SRC}" "${DEST}"
fi

echo "Done. Files are under: ${DEST}"

