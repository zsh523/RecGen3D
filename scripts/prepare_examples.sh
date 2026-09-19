#!/usr/bin/env bash
# RecGen3D — unpack the example images referenced by examples/full.json.
#
#   bash scripts/prepare_examples.sh                 # download, then unpack
#   bash scripts/prepare_examples.sh path/to/zip     # unpack a local archive
#   ARCHIVE_NAME=recgen3d-paper-samples.zip bash scripts/prepare_examples.sh
#                                                    # the GSO / Toys4k benchmark objects
#
# The archive holds the 27 multi-view sets used for the qualitative results,
# one folder of background-removed views per object.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/examples/data}"
ARCHIVE_NAME="${ARCHIVE_NAME:-recgen3d-examples.zip}"
ARCHIVE_URL="${ARCHIVE_URL:-https://huggingface.co/datasets/zsh523/RecGen3D-examples/resolve/main/${ARCHIVE_NAME}}"

ARCHIVE="${1:-}"
if [ -z "${ARCHIVE}" ]; then
  ARCHIVE="${REPO_ROOT}/examples/${ARCHIVE_NAME}"
  if [ ! -f "${ARCHIVE}" ]; then
    echo "--- downloading example data ---"
    echo "    ${ARCHIVE_URL}"
    # download to a temporary name so an interrupted transfer is never reused
    curl -L --fail -o "${ARCHIVE}.part" "${ARCHIVE_URL}"
    mv "${ARCHIVE}.part" "${ARCHIVE}"
  fi
fi

SUMS="${REPO_ROOT}/SHA256SUMS.txt"
if [ -f "${SUMS}" ] && grep -q "$(basename "${ARCHIVE}")" "${SUMS}"; then
  echo "--- verifying archive ---"
  if ! ( cd "$(dirname "${ARCHIVE}")" && grep "$(basename "${ARCHIVE}")" "${SUMS}" | sha256sum -c - ); then
    echo "checksum mismatch: delete ${ARCHIVE} and retry" >&2
    exit 1
  fi
fi

if [ ! -f "${ARCHIVE}" ]; then
  echo "archive not found: ${ARCHIVE}" >&2
  exit 1
fi

echo "--- unpacking into ${DATA_DIR} ---"
mkdir -p "${DATA_DIR}"
unzip -q -o "${ARCHIVE}" -d "${DATA_DIR}"

echo "--- verifying against examples/full.json ---"
MANIFEST="${MANIFEST:-${REPO_ROOT}/examples/full.json}"
python "${REPO_ROOT}/scripts/check_examples.py" \
  --examples "${MANIFEST}" --data_root "${DATA_DIR}"
