#!/usr/bin/env python3
"""Minimal training entry point for SVD-VALID v0.2.

Loss, optimizer, LR schedule and CRN follow VALID v1.0 (the upstream
DWT typo -- the second DWT applied again to output 1 -- is fixed here).

Sampling arms share everything else (loss, optimizer, LR schedule, CRN):
  --mode random       : faithful per-block random Tetris (VALID baseline)
  --mode patch_random : single random pattern per patch, fixed roles
                        (control for guided: removes the guidance, keeps the
                        "one pattern per patch" geometry)
  --mode guided       : SVD-guided single-pattern Tetris (our method)
  --mode fixed        : one fixed pattern for every patch, fixed roles
                        (degeneracy control: if guided keeps picking pattern K,
                        this isolates "guidance" from "that pattern is good")
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

from datasets.sampling import (
    TETRIS_PATTERNS,
    guided_tetris_subvolumes,
    random_tetris_subvolumes,
    tetris_from_pattern,
    valid_random_tetris,
)
from models.network import NetworkCNR, HessianConstraintLoss3D
from models.dwt3d import dwt3d_haar, VALID_REG_BANDS


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
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None, None].to(device)


def wavelet_hessian_loss(out1, out2, hessian):
    """VALID v1.0 loss_reg: Hessian on the low-frequency DWT bands of both outputs."""
    bands1 = dwt3d_haar(out1)
    bands2 = dwt3d_haar(out2)
    stack = torch.cat(
        [torch.cat([bands1[b], bands2[b]], dim=1) for b in VALID_REG_BANDS],
        dim=1,
    )
    return hessian(stack)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-volume", required=True)
    ap.add_argument("--signal-map")
    ap.add_argument("--noise-map")
    ap.add_argument("--mode", choices=["random", "patch_random", "guided", "fixed"],
                    default="random",
                    help="random: faithful per-block VALID Tetris; "
                         "patch_random: single random pattern per patch, fixed "
                         "roles (control for guided); guided: SVD-guided "
                         "pattern per patch; fixed: one fixed pattern "
                         "(--pattern) for every patch.")
    ap.add_argument("--pattern", type=int, default=4,
                    help="Tetris pattern id used by --mode fixed (0..7).")
    ap.add_argument("--role-order", nargs=4, type=int, default=None,
                    metavar=("XA", "XB", "YA", "YB"),
                    help="Permutation of the pattern's four slots assigning them "
                         "to the xa / xb / ya / yb roles. Default is 0,1,2,3. "
                         "The sampled voxel set is unchanged, so this isolates "
                         "the input->target pairing from the sampling phase.")
    ap.add_argument("--patch", nargs=3, type=int, default=[32, 64, 64])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--steps-per-epoch", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-reg", type=float, default=1e-4)
    ap.add_argument("--weight-identity", type=float, default=1.0,
                    help="Weight of loss_identity (mse(ha,hb)). VALID v1.0 uses "
                         "1.0; 0.0 is a diagnostic ablation only.")
    ap.add_argument("--features", type=int, default=16)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="runs/svd-valid")
    args = ap.parse_args()

    if any(p % 2 for p in args.patch):
        raise ValueError(f"Patch {args.patch} must be even in every dimension.")

    if args.mode == "guided" and (not args.signal_map or not args.noise_map):
        raise ValueError("Guided mode requires --signal-map and --noise-map.")

    if not 0 <= args.pattern < len(TETRIS_PATTERNS):
        raise ValueError(f"--pattern must be in [0, {len(TETRIS_PATTERNS) - 1}].")
    if args.role_order is not None and sorted(args.role_order) != [0, 1, 2, 3]:
        raise ValueError("--role-order must be a permutation of 0,1,2,3.")

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
    # VALID v1.0 schedule: linear warmup over the first 10% epochs, cosine decay.
    lr_func = lambda epoch: min(
        (epoch + 1) / (args.epochs * 0.1 + 1e-8),
        0.5 * (math.cos(epoch / args.epochs * math.pi) + 1),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
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
                xa, xb, ya, yb = [tensor(v, device) for v in views]
            elif args.mode == "patch_random":
                _, views = random_tetris_subvolumes(patch)
                xa, xb, ya, yb = [tensor(v, device) for v in views]
            elif args.mode == "fixed":
                views = tetris_from_pattern(patch, args.pattern)
                if args.role_order is not None:
                    views = [views[i] for i in args.role_order]
                xa, xb, ya, yb = [tensor(v, device) for v in views]
            else:
                subs = valid_random_tetris(patch)
                xa, xb, ya, yb = [tensor(v, device) for v in subs]

            ha, hb = model(xa), model(xb)

            loss_neighbor = 0.5 * mse(ha, ya) + 0.5 * mse(hb, yb)
            loss_reg = wavelet_hessian_loss(ha, hb, hessian)
            loss = loss_neighbor + args.weight_reg * loss_reg
            if args.weight_identity:
                loss = loss + args.weight_identity * mse(ha, hb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 20.0)
            optimizer.step()
            running += float(loss.detach().cpu())

        scheduler.step()
        mean_loss = running / args.steps_per_epoch
        history.append({"epoch": epoch + 1, "loss": mean_loss})
        print(f"epoch={epoch + 1:03d} loss={mean_loss:.6f} lr={scheduler.get_last_lr()[0]:.2e}")

        torch.save(
            {"model": model.state_dict(), "epoch": epoch + 1, "args": vars(args)},
            out / "latest.pt",
        )

    with open(out / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
