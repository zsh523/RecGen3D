#!/usr/bin/env bash
# RecGen3D — environment setup.
#
#   bash install.sh                 # create/populate the `recgen3d` conda env
#   CONDA_ENV=myenv bash install.sh # use a different env name
#
# Reference environment: Python 3.10, CUDA 12.1, PyTorch 2.5.1, one 24GB GPU.
# Set CUDA_TAG/TORCH_VERSION below if your toolkit differs.

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-recgen3d}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1}"
CUDA_TAG="${CUDA_TAG:-cu121}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== RecGen3D setup ==="
echo "  env    : ${CONDA_ENV}  (python ${PYTHON_VERSION})"
echo "  torch  : ${TORCH_VERSION} / ${CUDA_TAG}"
echo

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
  echo "--- creating conda env ${CONDA_ENV} ---"
  conda create -y -n "${CONDA_ENV}" "python=${PYTHON_VERSION}"
fi
conda activate "${CONDA_ENV}"

# 1. PyTorch must come from the CUDA-matched index, before anything else.
echo "--- installing PyTorch ${TORCH_VERSION}+${CUDA_TAG} ---"
pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
  --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

# 2. torch-cluster supplies the farthest-point sampling used to build the
#    canonical conditioning point cloud, so it is required, not optional.
#    Its wheels are indexed per torch+CUDA build.
echo "--- installing torch-cluster ---"
pip install torch-cluster==1.6.3 \
  -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html"

# 3. Everything else.
echo "--- installing Python dependencies ---"
pip install -r "${REPO_ROOT}/requirements.txt"

# 4. System libraries needed by pymeshlab / OpenCV / Open3D on headless hosts.
SUDO=""
if [ "$(id -u)" -ne 0 ]; then SUDO="sudo"; fi
if command -v apt-get >/dev/null 2>&1 && [ "${SKIP_APT:-0}" != "1" ] \
   && { [ -z "${SUDO}" ] || command -v sudo >/dev/null 2>&1; }; then
  echo "--- installing system libraries ---"
  ${SUDO} apt-get update
  ${SUDO} apt-get install -y \
    libegl1 libgl1 libopengl0 libglx0 libglvnd0 libgles2 \
    libsm6 libxext6 libxrender-dev libglib2.0-0
else
  echo "--- skipping apt step; ensure libgl1/libegl1/libglib2.0-0 are present ---"
fi

echo
echo "--- verifying ---"
cd "${REPO_ROOT}"
python - <<'PY'
import torch
print(f"  torch          {torch.__version__}  cuda={torch.version.cuda}  available={torch.cuda.is_available()}")
import torch_cluster; print("  torch-cluster  ok")
from hy3dshape_omni.models.diffusion.flow_matching_sit import Diffuser  # noqa: F401
from vggt.models.vggt import VGGT  # noqa: F401
print("  RecGen3D modules import cleanly")
PY

cat <<MSG

=== done ===
Next:
  conda activate ${CONDA_ENV}
  bash scripts/download_weights.sh     # fetch model weights
  bash scripts/prepare_examples.sh     # unpack the example images
  bash scripts/run_examples.sh         # generate meshes

Optional extras (Stage-1 visualisation, background removal, training):
  pip install -r requirements-extra.txt
MSG
