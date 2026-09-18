#!/usr/bin/env bash
# RecGen3D — fetch model weights.
#
#   bash scripts/download_weights.sh
#
# Three downloads are needed:
#   1. facebook/VGGT-1B        — the upstream reconstruction backbone weights.
#   2. tencent/Hunyuan3D-Omni  — the VecSet VAE / DiT / conditioner backbone.
#   3. RecGen3D's own weights  — vggt_canonical.safetensors + recgen3d.safetensors.
#
# (1) and (2) come from the HuggingFace Hub; (3) is the RecGen3D release.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_DIR="${WEIGHTS_DIR:-${REPO_ROOT}/pretrained_weights}"
RECGEN3D_REPO="${RECGEN3D_REPO:-zsh523/RecGen3D}"

mkdir -p "${WEIGHTS_DIR}/recgen3d"

# hy3dshape_omni resolves `tencent/Hunyuan3D-Omni` under $HY3DGEN_MODELS.
export HY3DGEN_MODELS="${WEIGHTS_DIR}"
export HF_HOME="${WEIGHTS_DIR}"

echo "=== RecGen3D weights -> ${WEIGHTS_DIR} ==="
echo "    about 23 GB in total; make sure there is room."

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli not found; installing huggingface_hub[cli]"
  pip install -q "huggingface_hub[cli]"
fi

echo "--- 1/3  facebook/VGGT-1B (base reconstruction backbone) ---"
if [ -n "${VGGT_BASE_WEIGHTS:-}" ] && [ -f "${VGGT_BASE_WEIGHTS}" ]; then
  echo "    using the copy already at \$VGGT_BASE_WEIGHTS: ${VGGT_BASE_WEIGHTS}"
elif [ -f "${WEIGHTS_DIR}/vggt/model.safetensors" ]; then
  echo "    already present, skipping"
else
  huggingface-cli download facebook/VGGT-1B model.safetensors \
    --local-dir "${WEIGHTS_DIR}/vggt"
fi

# The VAE / DiT / conditioner all load `pytorch_model.bin`; the 12 GB EMA copy
# and the repo assets are never read, so skip them (~26 GB -> ~14 GB).
echo "--- 2/3  tencent/Hunyuan3D-Omni ---"
huggingface-cli download tencent/Hunyuan3D-Omni \
  --local-dir "${WEIGHTS_DIR}/tencent/Hunyuan3D-Omni" \
  --exclude "*pytorch_model_ema.bin" "assets/*"

echo "--- 3/3  RecGen3D checkpoints (${RECGEN3D_REPO}) ---"
huggingface-cli download "${RECGEN3D_REPO}" \
  --local-dir "${WEIGHTS_DIR}/recgen3d"

# Verify the RecGen3D files against the checksums shipped with the repo.
if [ -f "${REPO_ROOT}/SHA256SUMS.txt" ]; then
  echo "--- verifying checksums ---"
  if ! ( cd "${WEIGHTS_DIR}/recgen3d" && grep -E "safetensors$" "${REPO_ROOT}/SHA256SUMS.txt" | sha256sum -c - ); then
    echo "checksum mismatch: the download is incomplete or corrupted; delete ${WEIGHTS_DIR}/recgen3d and retry" >&2
    exit 1
  fi
fi

echo
echo "=== done ==="
find "${WEIGHTS_DIR}" -maxdepth 3 -name "*.ckpt" -o -maxdepth 3 -name "*.safetensors" | sed 's/^/  /'
cat <<MSG

Add this to your shell (or let scripts/run_examples.sh do it for you):
  export HY3DGEN_MODELS=${WEIGHTS_DIR}
  export HF_HOME=${WEIGHTS_DIR}
MSG
