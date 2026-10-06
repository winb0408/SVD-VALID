#!/usr/bin/env python3
"""Minimal training entry point for SVD-VALID v0.1."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

from datasets.sampling import guided_tetris_subvolumes, random_tetris_subvolumes
from models.network import NetworkCNR, HessianConstraintLoss3D


def crop3d(arr, shape, rng):
    d, h, w = arr.shape
    pd, ph, pw = shape
    if pd > d or ph > h or pw > w:
        raise ValueError(f"Patch {shape} exceeds volume {arr.shape}")
    z0 = rng.randint(0, d - pd)
    y0 = rng.randint(0, h - ph)
    x0 = rng.randint(0, w - pw)
    return arr[z0:z0 + pd, y0:y0 + ph, x0:x0 + pw], (z0, y0, x0)


def tensor(x, device):
    return torch.from_numpy(np.asarray(x, dtype=np.float32))[None, None].to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-volume", required=True)
    ap.add_argument("--signal-map")
    ap.add_argument("--noise-map")
    ap.add_argument("--mode", choices=["random", "guided"], default="random")
    ap.add_argument("--patch", nargs=3, type=int, default=[32, 64, 64])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--steps-per-epoch", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-reg", type=float, default=1e-4)
    ap.add_argument("--features", type=int, default=16)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="runs/svd-valid")
    args = ap.parse_args()

    if args.mode == "guided" and (not args.signal_map or not args.noise_map):
        raise ValueError("Guided mode requires --signal-map and --noise-map.")

    device = torch.device(
        args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    vol = np.load(args.train_volume).astype(np.float32)
    signal = np.load(args.signal_map).astype(np.float32) if args.signal_map else None
    noise = np.load(args.noise_map).astype(np.float32) if args.noise_map else None

    model = NetworkCNR(features=args.features, groups=args.groups).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    mse = torch.nn.MSELoss()
    hessian = HessianConstraintLoss3D()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    history = []

    for epoch in trange(args.epochs, desc="epochs"):
        model.train()
        running = 0.0

        for _ in range(args.steps_per_epoch):
            patch, origin = crop3d(vol, tuple(args.patch), rng)

            if args.mode == "guided":
                z0, y0, x0 = origin
                sp = signal[z0:z0 + args.patch[0], y0:y0 + args.patch[1], x0:x0 + args.patch[2]]
                npatch = noise[z0:z0 + args.patch[0], y0:y0 + args.patch[1], x0:x0 + args.patch[2]]
                _, views, q = guided_tetris_subvolumes(patch, sp, npatch)
            else:
                _, views = random_tetris_subvolumes(patch)
                q = None

            xa, xb, ya, yb = [tensor(v, device) for v in views]
            ha, hb = model(xa), model(xb)

            loss_neighbor = 0.5 * mse(ha, ya) + 0.5 * mse(hb, yb)
            loss_identity = mse(ha, hb)
            loss_reg = hessian(torch.cat([ha, hb], dim=1))
            loss = loss_neighbor + loss_identity + args.weight_reg * loss_reg

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 20.0)
            optimizer.step()
            running += float(loss.detach().cpu())

        scheduler.step()
        mean_loss = running / args.steps_per_epoch
        history.append({"epoch": epoch + 1, "loss": mean_loss})
        print(f"epoch={epoch + 1:03d} loss={mean_loss:.6f}")

        torch.save(
            {"model": model.state_dict(), "epoch": epoch + 1, "args": vars(args)},
            out / "latest.pt",
        )

    with open(out / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
