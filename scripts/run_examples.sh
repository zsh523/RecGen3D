#!/usr/bin/env bash
# RecGen3D — generate meshes for the bundled examples.
#
#   bash scripts/run_examples.sh                  # all 27 examples (examples/full.json)
#   MANIFEST=examples/select.json bash scripts/run_examples.sh   # one object, quick check
#   bash scripts/run_examples.sh lego-car colosseum   # only these examples
#
# Override any of the variables below from the environment.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

WEIGHTS_DIR="${WEIGHTS_DIR:-${REPO_ROOT}/pretrained_weights}"
CONFIG="${CONFIG:-configs/recgen3d.yaml}"
CKPT="${CKPT:-${WEIGHTS_DIR}/recgen3d/recgen3d.safetensors}"
VGGT_CKPT="${VGGT_CKPT:-${WEIGHTS_DIR}/recgen3d/vggt_canonical.safetensors}"
MANIFEST="${MANIFEST:-examples/full.json}"
DATA_ROOT="${DATA_ROOT:-examples/data}"
OUTPUT="${OUTPUT:-outputs/examples}"

# hy3dshape_omni resolves `tencent/Hunyuan3D-Omni` under these.
export HY3DGEN_MODELS="${WEIGHTS_DIR}"
export HF_HOME="${WEIGHTS_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if ! python -c "import torch, trimesh" >/dev/null 2>&1; then
  echo "the RecGen3D dependencies are not importable by \`python\` on this PATH." >&2
  echo "activate the environment first:  conda activate recgen3d" >&2
  exit 1
fi

for f in "${CKPT}" "${VGGT_CKPT}"; do
  if [ ! -f "${f}" ]; then
    echo "checkpoint not found: ${f}" >&2
    echo "run scripts/download_weights.sh first, or set CKPT= / VGGT_CKPT=" >&2
    exit 1
  fi
done

ONLY_ARGS=()
if [ "$#" -gt 0 ]; then
  ONLY_ARGS=(--only "$@")
fi

echo "=== RecGen3D ==="
echo "  config   ${CONFIG}"
echo "  ckpt     ${CKPT}"
echo "  manifest ${MANIFEST}"
echo "  output   ${OUTPUT}"
echo

python inference.py \
  --config "${CONFIG}" \
  --ckpt "${CKPT}" \
  --vggt_pretrained "${VGGT_CKPT}" \
  --examples "${MANIFEST}" \
  --data_root "${DATA_ROOT}" \
  --output "${OUTPUT}" \
  "${ONLY_ARGS[@]}"
