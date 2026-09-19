# RecGen3D: Reconstruction-Guided 3D Generation in a Shared Canonical Space

> **RecGen3D: Reconstruction-Guided 3D Generation in a Shared Canonical Space**
>
> **SIGGRAPH Asia 2026 (Conference Papers)**
>
> [![arXiv](https://img.shields.io/badge/arXiv-2604.01479-b31b1b.svg)](https://arxiv.org/abs/2604.01479)

<p align="center">
  <img src="assets/pipeline.png" alt="RecGen3D Pipeline" width="100%">
</p>

Sparse-view 3D modeling represents a fundamental tension between reconstruction fidelity and generative plausibility. Feed-forward reconstruction excels in efficiency and input alignment but often lacks the global priors needed for structural completeness, while diffusion-based generation provides rich geometric details but struggles with multi-view consistency. **RecGen3D** is a reconstruct-then-condition framework that combines these two paradigms into a cooperative system. We align both models within a shared canonical space and employ decoupled cooperative learning, enabling seamless collaboration during inference. The reconstruction module provides canonical geometric anchors, while the diffusion generator leverages latent-augmented conditioning to refine and complete the geometric structure.

## News

- **[Sep 2026]** Code and pretrained weights are released.
- **[Aug 2026]** RecGen3D is accepted to **SIGGRAPH Asia 2026 (Conference Papers)**! 🎉
- **[Aug 2026]** This repository was renamed from *UniRecGen* to **RecGen3D**, matching the camera-ready title. Old links redirect automatically.

## Demo

https://github.com/user-attachments/assets/1bf54a29-67c0-4d8f-8b69-ba7cdfe63cf3

## Highlights

- **Reconstruct-then-Condition Framework** — Couples feed-forward multi-view 3D reconstruction with native 3D diffusion generation in a shared canonical space for high-fidelity shape modeling from unposed sparse views.
- **Branch Repurposing** — Retargets the pointmap branch to predict in an object-centric canonical space while the depth and camera branches stay in the reference frame. The network is fine-tuned as a whole, so the pretrained 3D priors carry over.
- **Latent-Augmented Multi-View Conditioning** — Enriches dense DINO image tokens with VGGT geometric latents and camera embeddings, allowing the diffusion model to leverage multi-view context while retaining strong appearance priors.
- **Benchmark Results** — Improves on TRELLIS, Hunyuan3D-MV, LucidFusion, SAM 3D and ReconViaGen on the geometric metrics of the Toys4K and GSO benchmarks; see the paper for the tables.

## Method Overview

RecGen3D operates as a two-stage modular pipeline:

**Stage 1: Generation-Compatible Feed-Forward Reconstruction**

Given N unposed input views, a shared feature backbone (VGGT) extracts features and feeds them into three prediction heads — Depth, Camera, and Pointmap. Through our *branch repurposing* strategy, the pointmap head is adapted to predict in canonical object space while keeping depth and camera heads in the reference frame, preserving pretrained geometric priors. A *similarity alignment* step then transforms the more accurate depth-derived 3D points into canonical space, producing a high-quality canonical point cloud.

**Stage 2: Reconstruction-Guided Controllable Generation**

The canonical point cloud is downsampled and used as explicit geometric conditioning for a VecSet Diffusion Transformer (built on Hunyuan3D-Omni). Multi-view DINO tokens are augmented with VGGT geometric latents and camera embeddings via *latent-augmented view conditioning*, enabling the diffusion model to jointly leverage dense visual semantics and precise multi-view geometric context. The VecSet VAE-Decoder then produces a high-fidelity triangular mesh.

## Installation

Tested configuration: **Python 3.10, CUDA 12.1, PyTorch 2.5.1**, a single RTX 4090
(24 GB). `CUDA_TAG` / `TORCH_VERSION` let you build against another toolkit, but only
this combination has been verified — `torch-cluster` must match your torch+CUDA build.

```bash
git clone https://github.com/zsh523/RecGen3D.git
cd RecGen3D
bash install.sh                 # creates the `recgen3d` conda env
conda activate recgen3d
```

`install.sh` installs PyTorch and `torch-cluster` from their CUDA-matched indices before
anything else, then the rest of `requirements.txt`. To do it by hand:

```bash
conda create -n recgen3d python=3.10 && conda activate recgen3d
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install torch-cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip install -r requirements.txt
```

> **`torch-cluster` is required, not optional.** It provides the farthest-point sampling
> that builds the canonical conditioning point cloud. Its wheels are published per
> `torch`+CUDA build, so the `-f` index above must match your PyTorch install exactly.

On a headless machine you also need the usual GL/GLib libraries for `pymeshlab` and OpenCV:

```bash
sudo apt-get install -y libegl1 libgl1 libopengl0 libglx0 libglvnd0 libgles2 \
                        libsm6 libxext6 libxrender-dev libglib2.0-0
```

`requirements-extra.txt` holds a few things inference does not need: the notebook mesh
viewers and background removal for preprocessing your own photographs.

## Weights

```bash
bash scripts/download_weights.sh
```

This populates `pretrained_weights/` (about 23 GB in total):

| Weight | Size | Source | Purpose |
| --- | --- | --- | --- |
| `vggt/model.safetensors` | 5.0 GB | `facebook/VGGT-1B` | base reconstruction backbone |
| `tencent/Hunyuan3D-Omni` | 13.5 GB | HuggingFace Hub | VecSet VAE, DiT backbone, conditioner |
| `recgen3d/vggt_canonical.safetensors` | 4.1 GB | [`zsh523/RecGen3D`](https://huggingface.co/zsh523/RecGen3D) | Stage-1 canonical VGGT |
| `recgen3d/recgen3d.safetensors` | 164 MB | [`zsh523/RecGen3D`](https://huggingface.co/zsh523/RecGen3D) | LoRA + conditioning point encoder |

`recgen3d.safetensors` holds only the LoRA adapters and the conditioning point encoder
(72 M parameters). `download_weights.sh` checks both files against `SHA256SUMS.txt`.

Already have VGGT-1B? Point `$VGGT_BASE_WEIGHTS` at it instead of downloading it again.

Export the weights directory so `hy3dshape_omni` can find the backbone (or let
`scripts/run_examples.sh` do it):

```bash
export HY3DGEN_MODELS=$PWD/pretrained_weights
export HF_HOME=$PWD/pretrained_weights
```

## Quick Start

The fastest check that everything works is a bundled example — no images of your own
needed (see [Examples](#examples)):

```bash
bash scripts/prepare_examples.sh
MANIFEST=examples/select.json bash scripts/run_examples.sh
```

Generate a mesh from your own unposed, background-removed views:

```bash
python inference.py \
    --ckpt pretrained_weights/recgen3d/recgen3d.safetensors \
    --images path/to/view_*.png \
    --name my_object \
    --output outputs/
```

Inputs are RGBA PNGs of a single object with the background removed — the alpha channel
is read as the foreground mask. Two to eight views work well.

Ordinary photographs need preparing first — see [Preparing your own photos](#preparing-your-own-photos).

Each run writes into `outputs/<name>/`:

| File | Description |
| --- | --- |
| `mesh.ply` | the generated mesh — `--mesh_format glb` writes glTF instead, `both` writes both |
| `canonical_points.ply` | the 4096-point Stage-1 canonical point cloud that conditions generation |
| `glbscene_stage1_pointmap.glb` | the full Stage-1 reconstruction, with `--save_intermediate` |

To run a list of objects instead of one, pass a manifest — that is what the
[bundled examples](#examples) use:

```bash
python inference.py --ckpt ... --examples examples/full.json --data_root examples/data \
    --output outputs/ --only lego-car colosseum
```

Useful flags: `--num_inference_steps`, `--guidance_scale`, `--octree_resolution`,
`--seed`, `--overwrite`, `--save_intermediate`, `--mesh_format`, `--device`, and
`--vggt_pretrained` if your Stage-1 weights are not at the config's default path.
Run `python inference.py --help` for the full list.

## Preparing your own photos

`scripts/preprocess_images.py` turns ordinary photographs into the RGBA views the model
expects: background removed with [BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet),
object centred, square, 518x518.

```bash
pip install -r requirements-extra.txt     # BiRefNet needs kornia; .HEIC needs pillow-heif
python scripts/preprocess_images.py --input path/to/photos --output examples/data/my_object
python inference.py --ckpt pretrained_weights/recgen3d/recgen3d.safetensors \
    --images examples/data/my_object/*.png --name my_object --output outputs/
```

Reads JPEG, PNG, WebP, BMP, TIFF and HEIC. Images that already carry an alpha channel
keep it instead of being re-segmented.

| Flag | |
| --- | --- |
| `--batch` | treat each subfolder of `--input` as a separate object |
| `--padding` | bounding-box padding around the object, default 1.2 (matches the bundled examples) |
| `--resolution` | output edge length, default 518 |
| `--keep_names` | keep original filenames instead of `view_NN.png` |
| `--device` | e.g. `cpu` |

## Examples

27 example objects are available to try the model on. The manifests are in this repo;
the images (35 MB) are downloaded from
[`zsh523/RecGen3D-examples`](https://huggingface.co/datasets/zsh523/RecGen3D-examples):

```bash
bash scripts/prepare_examples.sh    # download + unpack into examples/data/
bash scripts/run_examples.sh        # all 27 examples -> outputs/examples/
```

A second archive holds the 8 synthetic objects from the GSO and Toys4k benchmarks that
appear in the paper's figures — 4 rendered views each, already background-free:

```bash
ARCHIVE_NAME=recgen3d-paper-samples.zip MANIFEST=examples/paper-samples.json \
    bash scripts/prepare_examples.sh
MANIFEST=examples/paper-samples.json bash scripts/run_examples.sh
```

Run a subset, or a different manifest — `examples/select.json` holds a single object,
which is the quickest way to check the setup end to end:

```bash
bash scripts/run_examples.sh lego-car colosseum
MANIFEST=examples/select.json bash scripts/run_examples.sh
```

Each example is one folder of views under `examples/data/<name>/`, and
`examples/full.json` lists the object name plus the views to condition on:

```json
{"name": "lego-car", "folder_path": "lego-car",
 "image_names": ["view_00.png", "view_01.png", "view_02.png", "view_03.png"]}
```

Write a manifest in that shape to point the same entry point at your own collections.
The bundled objects range from 2 to 20 views each, and include photographs, video frames
and synthetic renders.

## Repository Layout

```
configs/recgen3d.yaml   inference configuration (model + sampling defaults)
examples/               manifests; example images unpack into examples/data/
hy3dshape_omni/         Stage-2 VecSet diffusion (VAE, DiT, conditioners, pipeline)
vggt/                   Stage-1 reconstruction backbone and canonical-space heads
inference.py            standalone entry point: unposed views -> mesh
install.sh              environment setup
scripts/                weight download, example preparation, preprocessing, batch runs
```

## Citation

If you find RecGen3D useful, please cite:

```bibtex
@inproceedings{huang2026recgen3d,
  title     = {RecGen3D: Reconstruction-Guided 3D Generation in a Shared Canonical Space},
  author    = {Huang, Zhisheng and Chen, Jiahao and Lin, Cheng and Hu, Chenyu and Huang, Hanzhuo and Yu, Zhengming and Li, Mengfei and Liu, Yuheng and Gu, Zekai and Zhao, Zibo and Liu, Yuan and Li, Xin and Wang, Wenping},
  booktitle = {SIGGRAPH Asia 2026 Conference Papers},
  year      = {2026}
}
```

## License

This release is **not** permissively licensed, and no single licence covers it. It
vendors code from two upstream projects, both of which restrict use to non-commercial
research. Read the agreements in [`LICENSES/`](LICENSES/) before using the code or the
weights.

## Acknowledgements

This project builds on [VGGT](https://github.com/facebookresearch/vggt) and
[Hunyuan3D-Omni](https://github.com/Tencent-Hunyuan/Hunyuan3D-Omni) / [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1),
and vendors code from [SiT](https://github.com/willisma/SiT), [DINOv2](https://github.com/facebookresearch/dinov2)
and [diffusers](https://github.com/huggingface/diffusers). Background removal in
`scripts/preprocess_images.py` uses [BiRefNet](https://github.com/ZhengPeng7/BiRefNet).
We thank the authors for making their work available.
