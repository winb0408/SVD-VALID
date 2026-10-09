#!/usr/bin/env python3
"""Sliding-window volumetric inference for SVD-VALID.

Splits the (possibly odd-sized) volume into overlapping even patches,
denoises each with the trained CRN, and averages the recombination
(VALID v1.0-style overlap reconstruction). The volume is zero-padded
to even dimensions when necessary.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from models.network import NetworkCNR


def pad_to_even(vol):
    pads = [(0, d % 2) for d in vol.shape]
    if not any(p[1] for p in pads):
        return vol, (0, 0, 0)
    padded = np.pad(vol, pads, mode="constant")
    return padded, tuple(p[1] for p in pads)


def sliding_windows(shape, patch, overlap):
    strides = [max(1, int(p * (1 - overlap))) for p in patch]
    starts = []
    for dim, p, s in zip(shape, patch, strides):
        if dim <= p:
            starts.append([0])
            continue
        vals = list(range(0, dim - p + 1, s))
        if vals[-1] != dim - p:
            vals.append(dim - p)
        starts.append(vals)
    z, y, x = starts
    return [(zi, yi, xi) for zi in z for yi in y for xi in x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", required=True, help="Input .npy volume [Z,Y,X].")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--patch", nargs=3, type=int, default=[32, 64, 64])
    ap.add_argument("--overlap", type=float, default=0.1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if any(p % 2 for p in args.patch):
        raise ValueError(f"Patch {args.patch} must be even in every dimension.")

    device = torch.device(
        args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    ckpt = torch.load(args.checkpoint, map_location=device)
    targs = ckpt.get("args", {})
    model = NetworkCNR(
        features=targs.get("features", 16), groups=targs.get("groups", 4)
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    vol = np.load(args.volume).astype(np.float32)
    original_shape = vol.shape
    vol, pad_z = pad_to_even(vol)

    acc = np.zeros_like(vol, dtype=np.float64)
    count = np.zeros_like(vol, dtype=np.float64)

    patch = tuple(args.patch)
    with torch.no_grad():
        for z0, y0, x0 in sliding_windows(vol.shape, patch, args.overlap):
            p = vol[z0:z0 + patch[0], y0:y0 + patch[1], x0:x0 + patch[2]]
            x = torch.from_numpy(p)[None, None].to(device)
            y = model(x)[0, 0].detach().cpu().numpy()
            acc[z0:z0 + patch[0], y0:y0 + patch[1], x0:x0 + patch[2]] += y
            count[z0:z0 + patch[0], y0:y0 + patch[1], x0:x0 + patch[2]] += 1

    denoised = (acc / np.maximum(count, 1)).astype(np.float32)
    denoised = denoised[:original_shape[0], :original_shape[1], :original_shape[2]]

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "denoised.npy", denoised)
    # Quick-look mid slices.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        for col, name in enumerate(("axial (Z)", "coronal (Y)", "sagittal (X)")):
            raw = vol[:original_shape[0], :original_shape[1], :original_shape[2]]
            mid = [s // 2 for s in raw.shape]
            axes[0, col].imshow(raw[mid[0]] if col == 0 else raw[:, mid[1]] if col == 1 else raw[:, :, mid[2]],
                                cmap="magma")
            axes[0, col].set_title(f"input {name}")
            axes[1, col].imshow(denoised[mid[0]] if col == 0 else denoised[:, mid[1]] if col == 1 else denoised[:, :, mid[2]],
                                cmap="magma")
            axes[1, col].set_title(f"denoised {name}")
        for ax in axes.flat:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out / "quicklook.png", dpi=150)
    except ImportError:
        pass

    meta = {
        "volume": args.volume,
        "checkpoint": args.checkpoint,
        "patch": list(args.patch),
        "overlap": args.overlap,
        "output_shape": list(denoised.shape),
        "n_windows": int(count.sum() // np.prod(patch)),
    }
    with open(out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
