# -*- coding: utf-8 -*-
"""Verify that every image referenced by an examples manifest is on disk."""

import argparse
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--examples", required=True, help="Manifest JSON (e.g. examples/full.json).")
    p.add_argument("--data_root", required=True, help="Root the manifest's folder_path entries resolve against.")
    args = p.parse_args()

    manifest = Path(args.examples)
    root = Path(args.data_root)
    items = json.loads(manifest.read_text())["infer_list"]

    total = 0
    incomplete = []
    for it in items:
        folder = Path(it["folder_path"])
        if not folder.is_absolute():
            folder = root / folder
        missing = [n for n in it["image_names"] if not (folder / n).exists()]
        total += len(it["image_names"])
        if missing:
            incomplete.append((it["name"], len(missing), str(folder / missing[0])))

    ok = len(items) - len(incomplete)
    print(f"{ok}/{len(items)} example(s) complete  ({total} image(s) referenced)")
    if incomplete:
        print("incomplete:")
        for name, n, sample in incomplete:
            print(f"  - {name}: {n} missing, e.g. {sample}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
