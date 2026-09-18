# -*- coding: utf-8 -*-
"""
Prepare your own photographs for RecGen3D.

RecGen3D conditions on RGBA images of a single object: background removed, object
centred, square, 518x518. This script produces exactly that from ordinary photos,
using BiRefNet for segmentation.

    # one object: a folder of photos -> a folder of processed views
    python scripts/preprocess_images.py --input path/to/photos --output examples/data/my_object

    # several objects: one subfolder per object
    python scripts/preprocess_images.py --input path/to/dataset --output processed/ --batch

The framing matters. The object is cropped to its bounding box and then padded by
`--padding` (default 1.2, i.e. 20%) before being resized, which is the framing the
released model was trained and evaluated with. Changing it moves your inputs away
from that distribution, so leave it alone unless you know why you are changing it.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic")


def load_image(path: Path) -> Image.Image:
    """Open an image, including HEIC when pillow-heif is available."""
    if path.suffix.lower() == ".heic":
        try:
            import pillow_heif
        except ImportError:
            raise SystemExit(
                f"{path.name} is HEIC; install pillow-heif to read it:  pip install pillow-heif"
            )
        pillow_heif.register_heif_opener()
    return Image.open(path)


class BiRefNetSegmenter:
    """Foreground segmentation with BiRefNet (downloaded from the HuggingFace Hub)."""

    REPO = "ZhengPeng7/BiRefNet"
    INPUT_SIZE = (1024, 1024)

    def __init__(self, device="cuda"):
        from torchvision import transforms
        from transformers import AutoModelForImageSegmentation

        self.device = torch.device(device)
        print(f"[preprocess] loading {self.REPO} on {self.device}")
        self.model = AutoModelForImageSegmentation.from_pretrained(
            self.REPO, trust_remote_code=True
        ).to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize(self.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.inference_mode()
    def alpha_for(self, rgb: Image.Image) -> np.ndarray:
        """Return a uint8 alpha channel, the same size as `rgb`."""
        from torchvision import transforms

        x = self.transform(rgb).unsqueeze(0).to(self.device)
        pred = self.model(x)[-1].sigmoid().cpu()[0].squeeze()
        mask = transforms.ToPILImage()(pred).resize(rgb.size)
        return np.array(mask)


def square_pad(img: Image.Image, fill) -> Image.Image:
    """Pad to a square without changing the object's scale."""
    w, h = img.size
    if w == h:
        return img
    side = max(w, h)
    out = Image.new(img.mode, (side, side), fill)
    out.paste(img, ((side - w) // 2, (side - h) // 2))
    return out


def preprocess(img: Image.Image, segmenter, resolution=518, padding=1.2,
               alpha_threshold=0.8) -> Image.Image:
    """One photo -> one square RGBA view, background removed and object centred."""
    # An image that already carries a real alpha channel is taken at face value.
    rgba = None
    if img.mode == "RGBA":
        a = np.array(img)[:, :, 3]
        if not np.all(a == 255):
            rgba = img

    if rgba is None:
        rgb = img.convert("RGB")
        # BiRefNet sees at most 1024px; shrinking first keeps large photos cheap.
        longest = max(rgb.size)
        if longest > 1024:
            s = 1024 / longest
            rgb = rgb.resize((int(rgb.width * s), int(rgb.height * s)), Image.Resampling.LANCZOS)
        arr = np.array(rgb.convert("RGBA"))
        arr[:, :, 3] = segmenter.alpha_for(rgb)
        rgba = Image.fromarray(arr, mode="RGBA")

    alpha = np.array(rgba)[:, :, 3]
    fg = np.argwhere(alpha > alpha_threshold * 255)
    if fg.size == 0:
        raise ValueError("no foreground found")

    y0, x0 = fg[:, 0].min(), fg[:, 1].min()
    y1, x1 = fg[:, 0].max(), fg[:, 1].max()
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = int(max(x1 - x0, y1 - y0) * padding)

    # The crop may reach past the edge; PIL fills that with transparent pixels,
    # which keeps the object centred instead of shifting it inwards.
    crop = rgba.crop((int(cx - side // 2), int(cy - side // 2),
                      int(cx + side // 2), int(cy + side // 2)))
    crop = square_pad(crop, (0, 0, 0, 0))
    crop = crop.resize((resolution, resolution), Image.Resampling.LANCZOS)

    # Zero the colour outside the object so stray background pixels cannot leak in.
    out = np.array(crop).astype(np.float32) / 255.0
    out[:, :, :3] *= (out[:, :, 3:4] > alpha_threshold)
    return Image.fromarray((out * 255).astype(np.uint8), mode="RGBA")


def process_folder(src: Path, dst: Path, segmenter, args) -> int:
    images = sorted(p for p in src.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        return 0
    dst.mkdir(parents=True, exist_ok=True)

    written = 0
    for i, path in enumerate(tqdm(images, desc=src.name, leave=False)):
        try:
            view = preprocess(load_image(path), segmenter,
                              resolution=args.resolution, padding=args.padding)
        except ValueError as e:
            print(f"  skipped {path.name}: {e}")
            continue
        name = f"view_{written:02d}.png" if args.rename else f"{path.stem}.png"
        view.save(dst / name, format="PNG")
        written += 1
    return written


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Folder of photos, or of per-object folders with --batch.")
    p.add_argument("--output", required=True, help="Where to write the processed views.")
    p.add_argument("--batch", action="store_true", help="Treat each subfolder of --input as one object.")
    p.add_argument("--resolution", type=int, default=518, help="Output edge length.")
    p.add_argument("--padding", type=float, default=1.2,
                   help="Bounding-box padding before the crop. 1.2 matches the released examples.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--rename", action="store_true", default=True,
                   help="Name outputs view_00.png, view_01.png ... (matches examples/).")
    p.add_argument("--keep_names", dest="rename", action="store_false",
                   help="Keep the original file stems instead.")
    args = p.parse_args()

    src, dst = Path(args.input), Path(args.output)
    if not src.is_dir():
        raise SystemExit(f"--input is not a directory: {src}")

    segmenter = BiRefNetSegmenter(device=args.device)

    if args.batch:
        folders = sorted(d for d in src.iterdir() if d.is_dir())
        if not folders:
            raise SystemExit(f"--batch given but {src} has no subfolders")
        total = 0
        for d in folders:
            n = process_folder(d, dst / d.name, segmenter, args)
            print(f"  {d.name}: {n} view(s)")
            total += n
        print(f"\n[preprocess] {total} view(s) across {len(folders)} object(s) -> {dst}")
    else:
        n = process_folder(src, dst, segmenter, args)
        if n == 0:
            raise SystemExit(f"no images found in {src}")
        print(f"\n[preprocess] {n} view(s) -> {dst}")
        print(f"[preprocess] run them with:  python inference.py --images {dst}/*.png "
              f"--name {dst.name} --ckpt pretrained_weights/recgen3d/recgen3d.safetensors")


if __name__ == "__main__":
    main()
