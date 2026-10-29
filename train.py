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


def dense_views(arr):
    """`--mode dense` 的四个视图：**同一个满分辨率 patch**。

    抽成模块级函数是为了能被单测钉住（Round3 审查 R3-P06）：在 dense 模式下
    `xa is xb` 恒成立，于是同一个 CRN 对同一输入给出**逐位相同**的 `ha = hb`，
    `mse(ha, hb)` 与它的梯度**恒为 0** —— 也就是说 `--weight-identity` 在 dense 臂上
    是**空操作**，不能拿它做"关掉 identity 项会怎样"的消融。
    """
    return [arr, arr, arr, arr]


def loss_terms(ha, hb, ya, yb, hessian, weight_reg, weight_identity):
    """把三项损失分开返回，供 `--audit-loss-grads` 与单测逐项看梯度。

    顺序与原实现逐位一致：`neighbor + weight_reg*reg + weight_identity*identity`。
    """
    neighbor = 0.5 * torch.nn.functional.mse_loss(ha, ya) + 0.5 * torch.nn.functional.mse_loss(hb, yb)
    reg = wavelet_hessian_loss(ha, hb, hessian)
    identity = torch.nn.functional.mse_loss(ha, hb)
    total = neighbor + weight_reg * reg + weight_identity * identity
    return {"neighbor": neighbor, "reg": reg, "identity": identity, "total": total}


def grad_norm(term, model):
    """单个损失项的梯度范数（不写回 `.grad`，可反复调用）。"""
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
    sq = sum(float(g.pow(2).sum()) for g in grads if g is not None)
    return math.sqrt(sq)


def build_model(args, device="cpu"):
    """按 `--arch-head` / `--zero-init-final` 建模型。三臂配置见 Round3 B1：

      direct                      : pred = net(x)                  （旧基线，逐位一致）
      residual                    : pred = x + net(x)              （只加旁路）
      residual + --zero-init-final: pred = x + net(x)，net.out 置零
                                    ⇒ 训练起点**精确**等于输入（安全初始化）

    起点为恒等**不保证**训练损失下降就等于去噪变好（监督 MSE 也可被拟合背景拉低）。
    """
    if args.arch_head == "direct" and args.zero_init_final:
        raise ValueError("--zero-init-final 只对 --arch-head residual 有意义："
                         "direct + 零初始化会让起点恒为 0，是退化配置")
    model = NetworkCNR(features=args.features, groups=args.groups,
                       residual=(args.arch_head == "residual"))
    if args.zero_init_final:
        model.zero_init_out()
    return model.to(device)


def sample_views(args, src, csrc, patch, origin, signal=None, noise=None, gen_seed=None):
    """按当前臂的采样规则抽 4 个视图 `(xa, xb, ya, yb)`；`csrc` 给定时只换目标。

    从 `main()` 里提出来是为了①梯度审计能复用同一套几何；②坐标/冲激单测能直接钉住
    "输入视图与干净目标视图是不是同一批体素"（Round3 审查 R3-P04）。

    `gen_seed` 为 None 时沿用旧行为：从**全局** torch RNG 抽一个种子。
    给定则完全确定（审计/单测用）。
    """

    def build_views(arr, gen=None):
        """按当前臂的采样规则从 arr 抽 4 个视图 (xa,xb,ya,yb)。

        视图是**半尺寸的相位子采样**（每 2×2×2 块取 4 个角），所以有监督臂
        不能在原分辨率上换目标，必须用**同一套采样几何**去抽无噪 patch 的视图。
        因此这里收一个显式的 torch.Generator：同一条随机流跑两遍
        （噪声 patch / 无噪 patch），几何逐体素对齐。
        """
        if args.mode == "dense":
            # 归因诊断臂：**不做**任何相位抽稀，四个视图都是这张满分辨率 patch。
            # 配合 --train-clean-volumes 时 views[2:] 会被替换成同位置干净 patch，
            # 于是训练几何与部署几何（infer.py 的满分辨率滑窗）第一次对齐。
            # **代价（R3-P06）**：xa is xb ⇒ ha == hb 逐位相同 ⇒ `mse(ha,hb)` 与其梯度
            # 恒为 0，`--weight-identity` 在本臂上是空操作。
            return dense_views(arr)
        if args.pairing == "cross":
            if args.mode == "fixed":
                return strided_views(arr, args.slots, args.stride)
            return valid_random_tetris_cross(arr, args.stride, generator=gen)
        if args.stride > 1:
            if args.mode == "fixed":
                v = strided_views(arr, TETRIS_PATTERNS[args.pattern].tolist(),
                                  args.stride)
                return [v[i] for i in args.role_order] if args.role_order is not None else v
            return valid_random_tetris_strided(arr, args.stride, generator=gen)
        if args.mode == "guided":
            z0, y0, x0 = origin
            sp = signal[z0:z0 + args.patch[0], y0:y0 + args.patch[1], x0:x0 + args.patch[2]]
            npatch = noise[z0:z0 + args.patch[0], y0:y0 + args.patch[1], x0:x0 + args.patch[2]]
            _, v, q = guided_tetris_subvolumes(arr, sp, npatch)
            return v
        if args.mode == "patch_random":
            _, v = random_tetris_subvolumes(arr)
            return v
        if args.mode == "fixed":
            v = tetris_from_pattern(arr, args.pattern)
            return [v[i] for i in args.role_order] if args.role_order is not None else v
        return valid_random_tetris(arr, generator=gen)

    if csrc is not None:
        # 有监督上界臂：**只改目标**。输入视图与目标视图都按同一套采样几何抽，
        # 唯一差别是目标来自无噪 patch 而不是噪声 patch 的互补视图。
        # 注意（R3-P04）：相位几何下目标落在**另一些相位位置**上（与输入不重叠），
        # 只有 dense 臂才是同位置（same-voxel）监督。
        cpatch = csrc[origin[0]:origin[0] + args.patch[0],
                      origin[1]:origin[1] + args.patch[1],
                      origin[2]:origin[2] + args.patch[2]]
        gen = torch.Generator()
        if gen_seed is None:
            gen_seed = int(torch.randint(0, 2 ** 31 - 1, (1,)).item())
        gen.manual_seed(int(gen_seed))
        st = gen.get_state()
        views = build_views(patch, gen)
        gen.set_state(st)                       # 同一条随机流 → 几何逐体素一致
        cviews = build_views(cpatch, gen)
        views[2], views[3] = cviews[2], cviews[3]
    else:
        views = build_views(patch)
    return views


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-volume", default=None,
                    help="单卷训练体。与 --train-volumes 二选一：给了 --train-volumes 时本参数被忽略"
                         "（参照卷取 vols[0]），可以省略；两者都不给才报错。")
    ap.add_argument("--signal-map")
    ap.add_argument("--noise-map")
    ap.add_argument("--mode", choices=["random", "patch_random", "guided", "fixed", "dense"],
                    default="random",
                    help="random: faithful per-block VALID Tetris; "
                         "patch_random: single random pattern per patch, fixed "
                         "roles (control for guided); guided: SVD-guided "
                         "pattern per patch; fixed: one fixed pattern "
                         "(--pattern) for every patch. "
                         "dense: **只做归因诊断用**——不做任何相位抽稀，输入输出都是满分辨率 patch；"
                         "必须配 --train-clean-volumes，否则输入=目标=同一张图（平凡恒等）。")
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
    ap.add_argument("--train-clean-volumes", nargs="*", default=None,
                    help="**有监督上界臂**（Round2 S0 归因用）：与 --train-volumes 一一对应、"
                         "同形状的**无噪** log1p 体。给定时目标 y 改成同位置的无噪 patch，"
                         "输入采样（tet / 相位）不变——于是「只把目标从互补视图换成真值」这一个变量"
                         "被单独拿出来。不给时行为与之前逐位相同。")
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
    ap.add_argument("--arm-label", default=None,
                    help="可选：写进 run 元数据的臂名。用于把旧简称换成描述性命名"
                         "（如 S_phase_cross_clean / S_dense_same_voxel / "
                         "D_dense_independent_pair），旧 run 不受影响。")
    ap.add_argument("--audit-loss-grads", action="store_true",
                    help="只做**一项**固定 minibatch 的三项损失 + 逐项梯度范数审计，"
                         "写到 out/loss_grad_audit.json 后退出（不训练）。")
    ap.add_argument("--audit-seed", type=int, default=1234,
                    help="梯度审计用的采样种子（与训练种子分开，审计可重复）。")
    ap.add_argument("--arch-head", choices=["direct", "residual"], default="direct",
                    help="Round3 B1：direct = 现在的直接出图（基线）；"
                         "residual = pred = input + net(input)（只加旁路，不动初始化）。"
                         "与 --zero-init-final 正交，用于把『旁路』和『初始化』两件事拆开。")
    ap.add_argument("--zero-init-final", action="store_true",
                    help="Round3 B1：把输出层（net.out）的权重与偏置置零。"
                         "只对 residual 有意义（起点精确等于输入）；对 direct 会是退化的全零起点，"
                         "脚本会拒绝这种组合。")
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
    # 有监督上界臂：与 vols 一一对应的无噪体
    cleans = None
    if args.train_clean_volumes:
        cleans = [np.load(p).astype(np.float32) for p in args.train_clean_volumes]
        if vols is None:
            raise ValueError("--train-clean-volumes 需要配 --train-volumes（按索引一一对应）")
        if len(cleans) != len(vols):
            raise ValueError(f"--train-clean-volumes 数量 {len(cleans)} 与 "
                             f"--train-volumes 数量 {len(vols)} 不一致")
        for a, b in zip(vols, cleans):
            if a.shape != b.shape:
                raise ValueError(f"干净体形状 {b.shape} 与噪声体 {a.shape} 不一致")
        # 措辞必须中性：这个开关只负责"目标视图来自另一组体"，内容可以是无噪真值
        # （有监督上界臂），也可以是**第二次含噪观测**（满分辨率自监督臂 D_n2n）。
        # 早期日志写成"无噪 patch"，在 D_n2n 的日志里会是错的。
        print(f"[S0] 目标视图来自 --train-clean-volumes（{len(cleans)} 卷；"
              f"真值或第二次观测由调用方决定）")

    if args.mode == "dense" and cleans is None:
        # 满分辨率模式下输入与目标来自同一张图；没有干净体就退化成"学恒等"，
        # 那样的数是**平凡**的，必须直接拒绝，而不是留一个能跑出好看数的口子。
        raise ValueError("--mode dense 必须配 --train-clean-volumes"
                         "（否则输入=目标=同一张图，只能学到恒等）")

    model = build_model(args, device)
    if args.arch_head != "direct" or args.zero_init_final:
        print(f"[B1] arch_head={args.arch_head} zero_init_final={args.zero_init_final}")
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

    if args.audit_loss_grads:
        # R3-P06：训练前先量一次"每项损失各自有多大梯度"。
        # 目的不是评价网络，而是**证明哪些开关在这个臂上有意义**：
        # dense 臂 xa is xb ⇒ identity 项梯度应当**精确为 0**，因此
        # "--weight-identity 0" 在 dense 上是空操作，不能当作消融。
        arng = random.Random(args.audit_seed)
        if vols is None:
            a_src, a_csrc = vol, (cleans[0] if cleans else None)
        else:
            a_src, a_csrc = vols[0], (cleans[0] if cleans else None)
        a_patch, a_origin = crop3d(a_src, tuple(args.patch), arng)
        a_views = sample_views(args, a_src, a_csrc, a_patch, a_origin, signal, noise,
                               gen_seed=args.audit_seed)
        axa, axb, aya, ayb = [tensor(v, device) for v in a_views]
        model.eval()
        aha, ahb = model(axa), model(axb)
        t = loss_terms(aha, ahb, aya, ayb, hessian,
                       args.weight_reg, args.weight_identity)
        rec = {
            "mode": args.mode, "pairing": args.pairing, "arm_label": args.arm_label,
            "audit_seed": args.audit_seed, "patch": list(args.patch),
            "features": args.features, "weight_reg": args.weight_reg,
            "weight_identity": args.weight_identity,
            "inputs_identical": bool(torch.equal(axa, axb)),
            "loss": {k: float(v.detach().cpu()) for k, v in t.items()},
            "grad_norm": {k: (grad_norm(v, model) if v.requires_grad else 0.0)
                          for k, v in t.items()},
            "weighted_grad_norm": {
                "neighbor": grad_norm(t["neighbor"], model),
                "reg": args.weight_reg * grad_norm(t["reg"], model),
                "identity": args.weight_identity * grad_norm(t["identity"], model),
            },
            "out_pred_meanshift": float((torch.cat([aha, ahb]) - torch.cat([axa, axb])).mean()),
        }
        # 每条臂各写一份（同名文件会互相覆盖，导致"两个臂的审计数一样"这种假结论）
        tag = args.arm_label or f"{args.mode}_{args.pairing}"
        (out / f"loss_grad_audit_{tag}.json").write_text(
            json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
        print(f"[audit] inputs_identical={rec['inputs_identical']}")
        for k in ("neighbor", "reg", "identity"):
            print(f"[audit] loss[{k}]={rec['loss'][k]:.6e}  grad_norm={rec['grad_norm'][k]:.6e}"
                  f"  weighted={rec['weighted_grad_norm'][k]:.6e}")
        return 0

    for epoch in trange(args.epochs, desc="epochs"):
        model.train()
        running = 0.0

        for _ in range(args.steps_per_epoch):
            if vols is None:
                src, csrc = vol, (cleans[0] if cleans else None)
            else:
                _i = rng.randrange(len(vols))
                src, csrc = vols[_i], (cleans[_i] if cleans else None)
            patch, origin = crop3d(src, tuple(args.patch), rng)

            views = sample_views(args, src, csrc, patch, origin, signal, noise)
            xa, xb, ya, yb = [tensor(v, device) for v in views]

            ha, hb = model(xa), model(xb)

            terms = loss_terms(ha, hb, ya, yb, hessian,
                               args.weight_reg, args.weight_identity)
            loss = terms["total"]

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
    raise SystemExit(main())
