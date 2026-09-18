# -*- coding: utf-8 -*-
"""
Strip a training checkpoint down to its weights.

PyTorch Lightning checkpoints carry optimizer moments, LR-scheduler state and loop
bookkeeping, which together are usually two to three times the size of the weights
themselves and are useless for inference. This rewrites a checkpoint keeping only
`state_dict`, optionally as safetensors.

    python scripts/strip_checkpoint.py in.ckpt out.safetensors
    python scripts/strip_checkpoint.py in.ckpt out.ckpt --format ckpt --dtype float32
"""

import argparse
import os
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", help="Training checkpoint to read.")
    p.add_argument("dst", help="Where to write the stripped weights.")
    p.add_argument("--format", choices=["safetensors", "ckpt"], default=None,
                   help="Output format (default: inferred from the dst suffix).")
    p.add_argument("--dtype", choices=["keep", "float32", "float16", "bfloat16"],
                   default="keep", help="Cast the weights on the way out.")
    args = p.parse_args()

    fmt = args.format or ("safetensors" if args.dst.endswith(".safetensors") else "ckpt")

    print(f"reading {args.src}")
    obj = torch.load(args.src, map_location="cpu", weights_only=False)
    if not (isinstance(obj, dict) and isinstance(obj.get("state_dict"), dict)):
        raise SystemExit("not a Lightning checkpoint: no 'state_dict' entry")

    state = obj["state_dict"]
    dropped = [k for k in obj if k != "state_dict"]
    print(f"  {len(state)} tensor(s); dropping {', '.join(dropped)}")

    if args.dtype != "keep":
        target = getattr(torch, args.dtype)
        state = {k: (v.to(target) if v.is_floating_point() else v) for k, v in state.items()}

    # safetensors rejects shared storage and non-contiguous tensors.
    state = {k: v.detach().contiguous().clone() for k, v in state.items()}

    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    if fmt == "safetensors":
        from safetensors.torch import save_file
        save_file(state, args.dst)
    else:
        torch.save({"state_dict": state}, args.dst)

    before = os.path.getsize(args.src) / 1e9
    after = os.path.getsize(args.dst) / 1e9
    print(f"wrote {args.dst}\n  {before:.2f} GB -> {after:.2f} GB")


if __name__ == "__main__":
    main()
