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
    check_cross_slots,
    guided_tetris_subvolumes,
    random_tetris_subvolumes,
    strided_views,
    tetris_from_pattern,
    valid_random_tetris,
    valid_random_tetris_cross,
    valid_random_tetris_strided,
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
    ap.add_argument("--train-volume", default=None,
                    help="单卷训练体。与 --train-volumes 二选一：给了 --train-volumes 时本参数被忽略"
                         "（参照卷取 vols[0]），可以省略；两者都不给才报错。")
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
    ap.add_argument("--stride", type=int, default=1,
                    help="采样步距 m：一个 (2m)^3 的块里只采 8 个角点。m>1 需要每个轴"
                         "能被 2m 整除；此时覆盖率为 8/(2m)^3，视图每轴长度为 n/(2m)。")
    ap.add_argument("--pairing", choices=["within", "cross"], default="within",
                    help="within: 四个视图全部取自同一个 tet（原有实现）；"
                         "cross: 输入取 tet A 的两个角、目标取互补 tet B 的两个角"
                         "（VALID 的互补配对，V3 预注册角色表的前提）。")
    ap.add_argument("--slots", nargs=4, type=int, default=None,
                    metavar=("XA", "XB", "YA", "YB"),
                    help="cross 配对下的显式相位槽位（0..7）。前两个必须同属一个 tet，"
                         "后两个必须落在它的互补 tet 内。例：体对角 1 4 6 3；"
                         "y 轴向 1 4 3 6（**同一批 4 个体素、同样的输入，只把两个目标对调**）。")
    ap.add_argument("--patch", nargs=3, type=int, default=[32, 64, 64])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--steps-per-epoch", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-schedule", choices=["poly", "cosine"], default="poly",
                    help="poly = 上游 VALID v1.0 的 `utils/general.py:adjust_learning_rate`："
                         "lr·(1−epoch/max_epoch)^0.9，**没有 warmup**；"
                         "cosine = 本仓库此前用的 warmup+余弦（保留以便复现旧 run）。"
                         "2026-10-26 核出此前一直用 cosine，与上游不同，故默认改为 poly。")
    ap.add_argument("--clip-grad", type=float, default=100.0,
                    help="上游 `clip_gradients` 默认 100.0；本仓库此前写死 20.0。")
    ap.add_argument("--train-volumes", nargs="*", default=None,
                    help="多个训练体的 npy 列表（每个都必须是同形状的 log1p 体）。"
                         "给定时每步随机选一个体再裁 patch；不给则用 --train-volume 单卷。")
    ap.add_argument("--weight-reg", type=float, default=1e-4)
    ap.add_argument("--weight-identity", type=float, default=1.0,
                    help="Weight of loss_identity (mse(ha,hb)). VALID v1.0 uses "
                         "1.0; 0.0 is a diagnostic ablation only.")
    ap.add_argument("--sampler-config", default=None,
                    help="R03：物理先验采样器的冻结配置（configs/sampler_config.json）。"
                         "给定时训练前先校验它与磁盘产物（ρ 曲线、角色序、频带、PSF、valid_mask）"
                         "一致，校验不过直接退出，不写半截 checkpoint；通过则把副本 + sha1 写进 run 目录。")
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
    if args.pairing == "cross":
        if args.mode == "guided":
            raise ValueError("cross 配对只支持 random / fixed 两种 mode。")
        if args.mode == "fixed" and args.slots is None:
            raise ValueError("cross + fixed 必须给 --slots。")
        if args.slots is not None:
            check_cross_slots(args.slots)
    elif args.slots is not None:
        raise ValueError("--slots 只在 --pairing cross 下使用。")
    if args.stride > 1 and any(p % (2 * args.stride) for p in args.patch):
        raise ValueError(
            f"--stride {args.stride} 要求每个 --patch 轴能被 {2 * args.stride} 整除。")

    device = torch.device(
        args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    vols = None
    if args.train_volumes:
        # 多卷：--train-volume 可省略（2026-10-27 修：此前 required=True，导致 A/B/C 三个
        # 多卷 launcher 会在 argparse 阶段直接报错退出，一格训练都跑不出来）
        vols = [np.load(p).astype(np.float32) for p in args.train_volumes]
        shapes = {v.shape for v in vols}
        if len(shapes) != 1:
            raise ValueError(f"--train-volumes 的形状不一致：{shapes}")
        vol = vols[0]
    elif args.train_volume:
        vol = np.load(args.train_volume).astype(np.float32)
    else:
        ap.error("必须给 --train-volume 或 --train-volumes 之一")
    signal = np.load(args.signal_map).astype(np.float32) if args.signal_map else None
    noise = np.load(args.noise_map).astype(np.float32) if args.noise_map else None

    model = NetworkCNR(features=args.features, groups=args.groups).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    if args.lr_schedule == "poly":
        # 上游 VALID v1.0：lr·(1 − epoch/max_epoch)^0.9，无 warmup
        lr_func = lambda epoch: max(1.0 - epoch / max(args.epochs, 1), 0.0) ** 0.9
    else:
        # 本仓库此前一直用的（与上游**不同**，保留以复现旧 run）
        lr_func = lambda epoch: min(
            (epoch + 1) / (args.epochs * 0.1 + 1e-8),
            0.5 * (math.cos(epoch / args.epochs * math.pi) + 1),
        )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    mse = torch.nn.MSELoss()
    hessian = HessianConstraintLoss3D()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # R03：采样器配置的**端到端**接入点。校验不过就退出（不写半截 checkpoint），
    # 通过则把副本与 sha1 落在 run 目录里 —— 这样任何一个 run 都能回答
    # "它用的是哪一份物理先验 / 哪个秩带 / 哪条角色序"。
    if args.sampler_config:
        import hashlib
        import subprocess
        import sys as _sys
        cfg_path = Path(args.sampler_config)
        probe = subprocess.run(
            [_sys.executable, str(Path(__file__).parent / "scripts" / "v3_sampler_config.py"),
             "--check"],
            cwd=str(Path(__file__).parent), capture_output=True, text=True)
        if cfg_path.resolve() != (Path(__file__).parent / "configs" / "sampler_config.json").resolve():
            print(f"[R03] 拒绝：--sampler-config 只接受仓库内的 configs/sampler_config.json"
                  f"（校验脚本与它绑定），收到 {cfg_path}")
            raise SystemExit(2)
        if probe.returncode != 0:
            print("[R03] sampler config 校验失败，拒绝开始训练：")
            print(probe.stdout.strip() or probe.stderr.strip())
            raise SystemExit(2)
        raw = cfg_path.read_bytes()
        (out / "sampler_config.json").write_bytes(raw)
        digest = hashlib.sha1(raw).hexdigest()
        (out / "sampler_config.sha1").write_text(digest + "\n")
        print(f"[R03] sampler_config OK  sha1={digest[:12]}  -> {out / 'sampler_config.json'}")

    rng = random.Random(args.seed)
    history = []

    for epoch in trange(args.epochs, desc="epochs"):
        model.train()
        running = 0.0

        for _ in range(args.steps_per_epoch):
            src = vol if vols is None else vols[rng.randrange(len(vols))]
            patch, origin = crop3d(src, tuple(args.patch), rng)

            if args.pairing == "cross":
                # 输入取 tet A，目标取互补 tet B：这是预注册角色表与 VALID 原生配对的前提
                if args.mode == "fixed":
                    views = strided_views(patch, args.slots, args.stride)
                else:
                    views = valid_random_tetris_cross(patch, args.stride)
                xa, xb, ya, yb = [tensor(v, device) for v in views]
            elif args.stride > 1:
                # 同 tet 配对但放大步距（覆盖率诊断臂）
                if args.mode == "fixed":
                    views = strided_views(patch, TETRIS_PATTERNS[args.pattern].tolist(),
                                          args.stride)
                    if args.role_order is not None:
                        views = [views[i] for i in args.role_order]
                else:
                    views = valid_random_tetris_strided(patch, args.stride)
                xa, xb, ya, yb = [tensor(v, device) for v in views]
            elif args.mode == "guided":
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
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
