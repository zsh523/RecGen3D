# -*- coding: utf-8 -*-
"""
RecGen3D: Reconstruction-Guided 3D Generation in a Shared Canonical Space.

Standalone inference entry point: turns a set of unposed, background-removed
views of a single object into a triangular mesh.

Examples
--------
Single object, images given directly:

    python inference.py --ckpt pretrained_weights/recgen3d/recgen3d.safetensors \
        --images examples/data/lego-car/*.png \
        --name lego-car --output outputs/

A manifest of several objects (see examples/full.json):

    python inference.py --ckpt pretrained_weights/recgen3d/recgen3d.safetensors \
        --examples examples/full.json --data_root examples/data --output outputs/
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hy3dshape_omni.utils import instantiate_from_config  # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402


def get_args():
    p = argparse.ArgumentParser(
        description="RecGen3D inference: unposed sparse views -> mesh",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, default=str(REPO_ROOT / "configs/recgen3d.yaml"),
                   help="Model configuration file.")
    p.add_argument("--ckpt", type=str, required=True,
                   help="RecGen3D checkpoint (.ckpt / .safetensors).")
    p.add_argument("--vggt_pretrained", type=str, default=None,
                   help="Override the Stage-1 canonical VGGT checkpoint from the config. "
                        "Use 'none' to skip it when --ckpt already carries the VGGT weights.")

    src = p.add_argument_group("input (choose one)")
    src.add_argument("--examples", type=str, default=None,
                     help="JSON manifest with an 'infer_list' of objects.")
    src.add_argument("--images", type=str, nargs="+", default=None,
                     help="Image paths for a single object.")
    p.add_argument("--data_root", type=str, default=None,
                   help="Root that relative 'folder_path' entries in --examples resolve against.")
    p.add_argument("--name", type=str, default="object",
                   help="Output name when using --images.")
    p.add_argument("--only", type=str, nargs="+", default=None,
                   help="Run only these example names from the manifest.")

    p.add_argument("--output", type=str, default="outputs",
                   help="Output directory.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num_inference_steps", type=int, default=None)
    p.add_argument("--guidance_scale", type=float, default=None)
    p.add_argument("--octree_resolution", type=int, default=None)
    p.add_argument("--num_chunks", type=int, default=None)
    p.add_argument("--save_intermediate", action="store_true",
                   help="Also export the full Stage-1 VGGT reconstruction as a GLB scene.")
    p.add_argument("--no_canonical_points", action="store_true",
                   help="Skip writing canonical_points.ply, the Stage-1 conditioning point cloud.")
    p.add_argument("--mesh_format", choices=["ply", "glb", "both"], default="ply",
                   help="Mesh file(s) to write. The formats hold the same geometry; glb is "
                        "handy for Blender and for inline previews on GitHub.")
    p.add_argument("--no_merge_lora", action="store_true",
                   help="Keep the LoRA adapters as separate layers instead of folding them into "
                        "the base weights. Folding is mathematically equivalent and ~13%% faster.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-run examples whose output mesh already exists.")
    return p.parse_args()


def resolve(path, root=REPO_ROOT):
    """Resolve a possibly-relative path against `root`."""
    if path is None:
        return None
    p = Path(path)
    return str(p if p.is_absolute() else (Path(root) / p))


def load_state_dict_any(path):
    """Load a Lightning .ckpt, a raw state_dict, or a .safetensors file."""
    path = str(path)
    if Path(path).suffix.lower() == ".safetensors":
        from safetensors.torch import load_file
        return load_file(path, device="cpu")

    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and isinstance(obj.get("state_dict"), dict):
        return obj["state_dict"]
    if isinstance(obj, dict) and all(torch.is_tensor(v) for v in obj.values()):
        return obj
    raise ValueError(
        f"Unsupported checkpoint structure in {path}: expected a Lightning "
        f"checkpoint with a 'state_dict' entry, or a raw state_dict."
    )


def build_model(args):
    config = OmegaConf.load(args.config)
    model_cfg = config.model

    # Point the Stage-1 VGGT weights at a real file (or disable them).
    vggt_cfg = model_cfg.params.get("vggt_config", None)
    if vggt_cfg is not None:
        pretrained = args.vggt_pretrained if args.vggt_pretrained is not None \
            else vggt_cfg.get("pretrained", None)
        if pretrained in (None, "none", "null", ""):
            vggt_cfg.pretrained = None
        else:
            pretrained = resolve(pretrained)
            if not os.path.exists(pretrained):
                raise FileNotFoundError(
                    f"Stage-1 VGGT checkpoint not found: {pretrained}\n"
                    f"Download it with scripts/download_weights.sh, or pass "
                    f"--vggt_pretrained none if --ckpt already contains the VGGT weights."
                )
            vggt_cfg.pretrained = pretrained

    print("[RecGen3D] building model ...")
    model = instantiate_from_config(model_cfg)

    ckpt = resolve(args.ckpt)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    print(f"[RecGen3D] loading checkpoint: {ckpt}")
    missing, unexpected = model.load_state_dict(load_state_dict_any(ckpt), strict=False)
    print(f"[RecGen3D] missing keys: {len(missing)} | unexpected keys: {len(unexpected)}")
    if missing:
        print("  first missing:", *missing[:5], sep="\n    ")

    model = model.to(args.device).eval()

    if not args.no_merge_lora:
        n = merge_lora_adapters(model)
        if n:
            print(f"[RecGen3D] folded {n} LoRA tensors into the base weights")

    return model, config


def merge_lora_adapters(model):
    """
    Fold the LoRA adapters into the weights they adapt.

    Stage 2 is a LoRA fine-tune of the Hunyuan3D-Omni DiT, so every adapted layer
    otherwise pays for two extra matmuls on all 50 denoising steps. Folding
    `W + BA` into `W` once removes that, which measures ~13% faster end to end and
    is mathematically the same computation.

    Must run after the checkpoint is loaded, since the checkpoint *is* the adapters.
    """
    peft_model = getattr(model, "model", None)
    base = getattr(peft_model, "base_model", None)
    if base is None or not hasattr(base, "merge_and_unload"):
        return 0

    n_adapters = sum(1 for name, _ in peft_model.named_parameters() if "lora_" in name)
    if n_adapters == 0:
        return 0

    merged = base.merge_and_unload()
    # Both the LightningModule and the pipeline hold references to the denoiser.
    model.model = merged
    model.pipeline.model = merged
    return n_adapters


def load_manifest(args):
    """Return a list of {name, image_paths} to run."""
    if bool(args.examples) == bool(args.images):
        raise SystemExit("Provide exactly one of --examples or --images.")

    if args.images:
        if args.only:
            raise SystemExit("--only filters a manifest; it does nothing with --images.")
        return [{"name": args.name, "image_paths": [str(Path(p)) for p in args.images]}]

    manifest_path = Path(resolve(args.examples))
    data = json.loads(manifest_path.read_text())
    # `folder_path` may be absolute, or relative to --data_root (default: the
    # manifest's own directory).
    root = Path(resolve(args.data_root)) if args.data_root else manifest_path.parent

    items = []
    for it in data["infer_list"]:
        folder = Path(it["folder_path"])
        if not folder.is_absolute():
            folder = root / folder
        items.append({
            "name": it["name"],
            "image_paths": [str(folder / n) for n in it["image_names"]],
        })

    if args.only:
        keep = set(args.only)
        items = [i for i in items if i["name"] in keep]
        unknown = keep - {i["name"] for i in items}
        if unknown:
            raise SystemExit(f"--only names not in manifest: {sorted(unknown)}")
    return items


def make_batch(image_paths, device):
    """Preprocess views into the batch the Diffuser expects."""
    missing = [p for p in image_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} input image(s) not found, e.g. {missing[0]}\n"
            f"Run scripts/prepare_examples.sh to unpack the example data."
        )

    images, masks = load_and_preprocess_images(image_paths, mode="pad")
    images = images.to(device)
    masks = masks.to(device)[:, 0, :, :]

    # The Hunyuan branch conditions on the first view, normalised to [-1, 1].
    image = (images[0].clone() * 2 - 1).unsqueeze(0)
    return {"image": image, "images": images.unsqueeze(0), "point_masks": masks}


def save_canonical_points(outputs_vggt, out_dir):
    """
    Export the Stage-1 canonical point cloud that conditions the diffusion model.

    This is the geometric anchor the method is built around, it costs nothing to
    write and needs no optional dependency, so it ships by default.
    """
    try:
        cond_point = outputs_vggt[-1].get("cond_point", None)
        if cond_point is None:
            return None
        pts = cond_point.reshape(-1, 3).detach().cpu().numpy()
        path = os.path.join(out_dir, "canonical_points.ply")
        trimesh.PointCloud(vertices=pts).export(path)
        return pts.shape[0]
    except Exception as e:
        print(f"[RecGen3D] warning: could not export the canonical point cloud ({e})")
        return None


def save_stage1_scene(outputs_vggt, batch, out_dir):
    """
    Export the Stage-1 VGGT reconstruction as a GLB scene, for inspection.

    Diagnostic output, so failures are reported and swallowed: the mesh has
    already been written by the time this runs.
    """
    try:
        import vggt.vggt_misc as misc
        prediction = outputs_vggt[-1]
        glbscene, *_ = misc.process_and_generate_glbscene(
            prediction, batch, out_dir, viz_gt=False, use_HY_token=False,
            prediction_mode="Predicted Pointmap",
        )
        misc.save_glb_scene_with_cleanup(prediction, batch, glbscene, out_dir,
                                         suffix="_stage1_pointmap")
    except ImportError as e:
        print(f"[RecGen3D] warning: skipping the Stage-1 scene export ({e}). "
              f"Reinstall the dependencies: pip install -r requirements.txt")
    except Exception as e:
        print(f"[RecGen3D] warning: could not export the Stage-1 scene ({e})")


def main():
    args = get_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; RecGen3D inference requires a GPU.")

    try:
        items = load_manifest(args)
        model, config = build_model(args)
    except FileNotFoundError as e:
        raise SystemExit(f"[RecGen3D] {e}")

    defaults = config.get("inference", {})
    sample_kwargs = {}
    for key in ("num_inference_steps", "guidance_scale", "octree_resolution", "num_chunks"):
        value = getattr(args, key) if getattr(args, key) is not None else defaults.get(key, None)
        if value is not None:
            sample_kwargs[key] = value
    if defaults.get("mc_level", None) is not None:
        sample_kwargs["mc_level"] = defaults["mc_level"]
    seed = args.seed if args.seed is not None else int(defaults.get("seed", 0))

    out_root = Path(resolve(args.output))
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[RecGen3D] {len(items)} example(s) -> {out_root}")
    print(f"[RecGen3D] sampling: seed={seed} {sample_kwargs}")

    failed = []
    for i, item in enumerate(items, 1):
        out_dir = out_root / item["name"]
        primary_suffix = "glb" if args.mesh_format == "glb" else "ply"
        mesh_path = out_dir / f"mesh.{primary_suffix}"
        if mesh_path.exists() and not args.overwrite:
            print(f"[{i}/{len(items)}] {item['name']}: already done, skipping (--overwrite to redo)")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        n_views = len(item["image_paths"])
        print(f"[{i}/{len(items)}] {item['name']}: {n_views} view(s)")
        t0 = time.time()
        try:
            batch = make_batch(item["image_paths"], args.device)
            with torch.no_grad():
                outputs, outputs_vggt = model.sample(
                    batch=batch, output_type="latents2mesh", seed=seed, **sample_kwargs
                )

            mesh = outputs[0][0]
            generated = trimesh.Trimesh(vertices=mesh.mesh_v, faces=mesh.mesh_f)
            for suffix in (("ply", "glb") if args.mesh_format == "both" else (args.mesh_format,)):
                generated.export(out_dir / f"mesh.{suffix}")

            n_points = None
            if outputs_vggt:
                if not args.no_canonical_points:
                    n_points = save_canonical_points(outputs_vggt, str(out_dir))
                if args.save_intermediate:
                    save_stage1_scene(outputs_vggt, batch, str(out_dir))

            extra = f", {n_points} canonical points" if n_points else ""
            print(f"    -> {mesh_path}  "
                  f"({len(generated.vertices)} verts, {len(generated.faces)} faces{extra}, "
                  f"{time.time() - t0:.1f}s)")
        except Exception as e:
            failed.append((item["name"], repr(e)))
            print(f"    !! failed: {e}")

    print(f"\n[RecGen3D] done: {len(items) - len(failed)}/{len(items)} succeeded -> {out_root}")
    if failed:
        print("[RecGen3D] failures:")
        for name, err in failed:
            print(f"  - {name}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
