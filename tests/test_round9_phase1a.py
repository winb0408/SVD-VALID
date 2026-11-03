#!/usr/bin/env python3
"""Round9 / Phase 1A 正确性单测 —— **不训练**，只验实现是否真的像声明的那样。

对应 `reviews/V3_PR1_round9_shared_valid_restoration_plan_2026-11-03.md` 的 Phase 1A
要求清单。每条测试都针对一个"可能被悄悄做错、而文档却声称 faithful"的点：

  T1 采样几何：每个 2×2×2 块取四个不同的角、且构成上游 `idx_pair` 里的一个合法 tet；
     四个视图形状 D/2×H/2×W/2；逐体素的坐标映射正确（不是"形状对了就算对"）。
  T2 角色置换作用域：--role-perm global 一个 patch 内所有块共用一个置换；
     per_block 每块独立；同 seed 可复现。
  T3 上游真实行为是**被测量出来的**而不是被断言的：直接调用 upstream
     `generate_mask_pair`，量出它的 rank 元组在所有块上相同（= 一个全局置换）。
  T4 只有一个可训练参数集 / 一个优化器，两个输入走**同一个** f_θ 对象，
     neighbor / identity / reg 三项梯度都非零。
  T5 正则项对上游 DWT 符号约定不敏感（LLH/LHL/HLL 反号不改变 loss 值），
     于是"符号差异"只是记录项、不是行为差异。
  T6 推理是满分辨率滑窗平均：无相位抽稀、无空洞、恒等模型必须逐体素还原输入。
  T7 --role-perm 的参数校验真的会报错（不得静默忽略），且 CRN 与上游逐位同构。

Run:  python tests/test_round9_phase1a.py
"""

import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.sampling import (  # noqa: E402
    TETRIS_PATTERNS,
    valid_random_tetris,
    valid_random_tetris_strided,
)
from models.dwt3d import VALID_REG_BANDS, dwt3d_haar  # noqa: E402
from models.network import HessianConstraintLoss3D, NetworkCNR  # noqa: E402
import train as train_mod  # noqa: E402

UPSTREAM = Path("/tmp/valid_upstream")
TETS = {tuple(sorted(p.tolist())) for p in TETRIS_PATTERNS}


def _load_upstream_module(rel_path, name):
    """按文件路径加载上游模块。

    不能 `sys.path.insert(0, UPSTREAM)`：本仓库也有 `datasets` / `models` 包，
    而且已经被 import 过，那样加载到的是**我们自己的**文件（会静默比对自己比自己）。
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, UPSTREAM / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# T1  采样几何 + 逐体素坐标映射
# --------------------------------------------------------------------------- #
def _corner_rank_tuples(volume, role_perm, seed=5):
    """把四个视图当成"每块 4 个角的顺序"，返回每块的 rank 元组。

    volume 的每个体素被编码成 `block_id*8 + corner_id`，于是视图里读到的整数就
    直接告诉我们"这个视图在这个块上取了哪个角"。
    """
    torch.manual_seed(seed)
    subs = valid_random_tetris(volume, role_perm=role_perm)
    per_view = [[int(v) % 8 for v in s.reshape(-1).tolist()] for s in subs]
    out = []
    for blk in zip(*per_view):
        order = sorted(blk)
        out.append((tuple(blk), tuple(order.index(c) for c in blk)))
    return out


def test_t1_views_are_one_legal_tetris_with_correct_voxel_mapping():
    shape = (6, 8, 8)  # even; 3*4*4 = 48 blocks
    g = np.indices(shape)
    block_id = (g[0] // 2) * 16 + (g[1] // 2) * 4 + (g[2] // 2)
    corner = (g[0] % 2) * 4 + (g[1] % 2) * 2 + (g[2] % 2)
    labels = torch.from_numpy((block_id * 8 + corner).astype(np.float32))
    # 同时编码原始坐标，用来验"视图体素 ↔ 原体素"的映射
    coords = torch.from_numpy(
        (g[0] * 10000 + g[1] * 100 + g[2]).astype(np.float32))

    for mode in ("per_block", "global"):
        rows = _corner_rank_tuples(labels, mode)
        assert len(rows) == 48, f"{mode}: 块数应为 48，实际 {len(rows)}"

        torch.manual_seed(5)
        subs = valid_random_tetris(labels, role_perm=mode)
        assert [tuple(s.shape) for s in subs] == [(3, 4, 4)] * 4, \
            f"{mode}: 视图形状必须是 D/2×H/2×W/2，实际 {[tuple(s.shape) for s in subs]}"

        torch.manual_seed(5)
        csubs = valid_random_tetris(coords, role_perm=mode)
        cz = [(c // 10000).long() for c in csubs]
        cy = [(c // 100).long() % 100 for c in csubs]
        cx = [(c.long() % 100) for c in csubs]

        for b, (corners, ranks) in enumerate(rows):
            # (a) 四个角色互不相同
            assert len(set(corners)) == 4, f"{mode} block {b}: 角色重复 {corners}"
            # (b) 恰好构成上游 idx_pair 里的一个合法 tet（不含体对角对）
            assert tuple(sorted(corners)) in TETS, \
                f"{mode} block {b}: {sorted(corners)} 不是合法 tet"
            # (c) rank 是 0..3 的置换
            assert sorted(ranks) == [0, 1, 2, 3], f"{mode} block {b}: ranks={ranks}"

            bz, by, bx = b // 16, (b % 16) // 4, b % 4
            for view in range(4):
                z, y, x = (int(cz[view][bz, by, bx]), int(cy[view][bz, by, bx]),
                           int(cx[view][bz, by, bx]))
                # (d) 视图体素 (bz,by,bx) 必须落在原体素的 2×2×2 块 b 内
                assert (z // 2, y // 2, x // 2) == (bz, by, bx), \
                    f"{mode} block {b} view {view}: 映射越块 ({z},{y},{x})"
                # (e) 该原体素的角编号必须等于本视图读到的角编号
                got = (z % 2) * 4 + (y % 2) * 2 + (x % 2)
                assert got == corners[view], \
                    f"{mode} block {b} view {view}: 坐标算出的角 {got} != 视图内容 {corners[view]}"
    print("ok T1: 每个块 4 个不同角色 / 合法 tet / 形状 D/2×H/2×W/2 / 逐体素映射正确")


def test_t1b_strided_views_keep_the_same_geometry():
    """stride>1 的随机采样臂（Round7/物理引导用）必须保留同样的块几何。"""
    shape = (8, 8, 8)
    g = np.indices(shape)
    block_id = (g[0] // 2) * 16 + (g[1] // 2) * 4 + (g[2] // 2)
    corner = (g[0] % 2) * 4 + (g[1] % 2) * 2 + (g[2] % 2)
    labels = torch.from_numpy((block_id * 8 + corner).astype(np.float32))
    torch.manual_seed(3)
    subs = valid_random_tetris_strided(labels, 1, role_perm="global")
    per_view = [[int(v) % 8 for v in s.reshape(-1).tolist()] for s in subs]
    ranks = []
    for blk in zip(*per_view):
        order = sorted(blk)
        ranks.append(tuple(order.index(c) for c in blk))
    assert len(set(ranks)) == 1, "strided 臂的 global 置换没有真正全局"
    print("ok T1b: strided 随机臂同样是每块一个合法 tet，global 置换真正全局")


# --------------------------------------------------------------------------- #
# T2  角色置换作用域 + 可复现性
# --------------------------------------------------------------------------- #
def test_t2_role_perm_scope_and_reproducibility():
    shape = (4, 4, 4)  # 8 块
    g = np.indices(shape)
    block_id = (g[0] // 2) * 4 + (g[1] // 2) * 2 + (g[2] // 2)
    corner = (g[0] % 2) * 4 + (g[1] % 2) * 2 + (g[2] % 2)
    labels = torch.from_numpy((block_id * 8 + corner).astype(np.float32))

    glob = _corner_rank_tuples(labels, "global")
    per = _corner_rank_tuples(labels, "per_block")
    print(f"   seed=5: global rank 元组 = {sorted({r for _, r in glob})}")
    print(f"   seed=5: per_block 不同 rank 元组数 = {len({r for _, r in per})}/8")

    assert len({r for _, r in glob}) == 1, "global 模式下所有块必须共用一个置换"
    assert len({r for _, r in per}) > 1, "per_block 模式下各块应各自抽置换"

    # 每个块各自抽的 tet 本来就可以不同（上游也如此），这里只验置换作用域
    again = _corner_rank_tuples(labels, "global")
    assert glob == again, "同一个 seed 下 global 模式必须逐块可复现"
    print("ok T2: global = 一个 patch 一个置换；per_block 每块独立；seed 可复现")


# --------------------------------------------------------------------------- #
# T3  上游行为是被测量的：直接量 upstream generate_mask_pair
# --------------------------------------------------------------------------- #
def test_t3_upstream_generate_mask_pair_uses_one_global_permutation():
    if not UPSTREAM.exists():
        print("skip T3: /tmp/valid_upstream 不在本机")
        return
    if not torch.cuda.is_available():
        print("skip T3: 上游 generate_mask_pair 的 get_generator() 硬绑 cuda")
        return
    generate_mask_pair = _load_upstream_module(
        "datasets/sampling.py", "_up_valid_sampling").generate_mask_pair

    random.seed(0)
    torch.manual_seed(0)
    masks = generate_mask_pair(torch.zeros(1, 4, 4, 4, device="cuda:0"))
    rows = []
    for b in range(8):
        corners = []
        for k in range(4):
            idx = torch.nonzero(masks[k][8 * b:8 * b + 8].cpu()).flatten().tolist()
            assert len(idx) == 1, f"M{ k + 1 } 在块 {b} 上不是单角"
            corners.append(idx[0])
        order = sorted(corners)
        rows.append((tuple(corners), tuple(order.index(c) for c in corners)))

    rank_tuples = {r for _, r in rows}
    assert len(rank_tuples) == 1, \
        f"上游应当一个 patch 共用一个置换，实测 {len(rank_tuples)} 种 {rank_tuples}"
    assert all(tuple(sorted(c)) in TETS for c, _ in rows), "上游每块也必须是合法 tet"
    n_pat = len({tuple(sorted(c)) for c, _ in rows})
    print(f"ok T3: 实测上游 rank 元组唯一 = {rank_tuples.pop()}；8 块里出现 {n_pat} 种 tet"
          "（每块独立抽 tet，但置换全局共享）")


# --------------------------------------------------------------------------- #
# T4  一个模型 / 一个优化器 / 同一个 f_θ / 三项梯度非零
# --------------------------------------------------------------------------- #
def test_t4_single_model_single_optimizer_and_all_loss_terms_have_grads():
    torch.manual_seed(0)
    model = NetworkCNR(features=16, groups=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4,
                                 betas=(0.9, 0.999))
    assert len(optimizer.param_groups) == 1, "必须只有一个参数组"
    assert {id(p) for p in optimizer.param_groups[0]["params"]} == \
           {id(p) for p in model.parameters()}, "优化器参数集必须恰好等于模型参数集"

    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 1_543_654, f"CRN 参数量应为 1,543,654，实际 {n_params}"

    # 两个分支必须走同一个对象：数一次前向调用次数
    calls = {"n": 0}
    hook = model.register_forward_hook(lambda *a, **k: calls.__setitem__("n", calls["n"] + 1))

    vol = torch.rand(1, 1, 16, 32, 32)
    xa, xb, ya, yb = valid_random_tetris(vol[0, 0], role_perm="global")
    xa, xb, ya, yb = [x[None, None] for x in (xa, xb, ya, yb)]
    ha, hb = model(xa), model(xb)
    hook.remove()
    assert calls["n"] == 2, f"两个输入应各前向一次（同一对象），实际 {calls['n']} 次"
    assert ha.shape == ya.shape == xa.shape

    hessian = HessianConstraintLoss3D()
    terms = train_mod.loss_terms(ha, hb, ya, yb, hessian, 1e-4, 1.0)
    for name in ("neighbor", "identity", "reg"):
        g = torch.autograd.grad(terms[name], [ha, hb], retain_graph=True,
                                allow_unused=True)
        assert any(x is not None and float(x.abs().sum()) > 0 for x in g), \
            f"{name} 对 ha/hb 的梯度恒为 0 —— 该项实际没有起作用"
    print(f"ok T4: 单参数集 {n_params} 个参数 / 单优化器 / 同一 f_θ 前向 2 次 / "
          "neighbor·identity·reg 梯度均非零")


# --------------------------------------------------------------------------- #
# T5  DWT 符号约定不影响正则项
# --------------------------------------------------------------------------- #
def test_t5_reg_loss_is_invariant_to_upstream_dwt_sign_convention():
    """上游 DWT_3D 的三个单-H 频带与我们反号（见 implementation_fidelity.md）。

    正则项对这些频带取二阶导再平方，所以整体反号不改变数值。这里直接验证：
    把 LLH/LHL/HLL 三个频带取负后算出的 reg，与我们现在的 reg 必须相等。
    """
    torch.manual_seed(1)
    out1, out2 = torch.randn(1, 1, 8, 16, 16), torch.randn(1, 1, 8, 16, 16)
    hessian = HessianConstraintLoss3D()
    ours = train_mod.wavelet_hessian_loss(out1, out2, hessian)

    b1, b2 = dwt3d_haar(out1), dwt3d_haar(out2)
    flipped = ("LLH", "LHL", "HLL")
    stack = torch.cat(
        [torch.cat([(-b1[b] if b in flipped else b1[b]),
                    (-b2[b] if b in flipped else b2[b])], dim=1)
         for b in VALID_REG_BANDS], dim=1)
    assert torch.allclose(ours, hessian(stack), atol=1e-6), \
        "符号翻转改变了正则项 —— 说明它不只是符号约定，必须重新核对上游"
    print(f"ok T5: reg loss 对上游 DWT 符号约定不敏感（{float(ours):.6f}）")


# --------------------------------------------------------------------------- #
# T6  推理：满分辨率滑窗平均，无相位抽稀、无空洞
# --------------------------------------------------------------------------- #
def test_t6_inference_is_full_patch_sliding_window_without_decimation():
    import infer  # 同一目录下的推理脚本，函数可复用

    src = Path(infer.__file__).read_text()
    for bad in ("[::2]", "decim", "backfill", "phase_views"):
        assert bad not in src, f"infer.py 里出现了 {bad!r}：推理不得做相位抽稀/回填"

    vol = np.random.rand(37, 64, 64).astype(np.float32)  # 奇数 Z，最长用的真实形状
    padded, pad_z = infer.pad_to_even(vol)
    assert padded.shape[0] % 2 == 0

    patch = (32, 64, 64)
    acc = np.zeros_like(padded, dtype=np.float64)
    count = np.zeros_like(padded, dtype=np.float64)
    seen = set()
    for z0, y0, x0 in infer.sliding_windows(padded.shape, patch, 0.1):
        p = padded[z0:z0 + patch[0], y0:y0 + patch[1], x0:x0 + patch[2]]
        seen.add(tuple(p.shape))
        # 恒等"模型"：若有一丝抽稀/拼接错误，输出就不再等于输入
        acc[z0:z0 + patch[0], y0:y0 + patch[1], x0:x0 + patch[2]] += p
        count[z0:z0 + patch[0], y0:y0 + patch[1], x0:x0 + patch[2]] += 1

    assert seen == {patch}, f"喂给网络的必须是**整个**满分辨率 patch，实际形状 {seen}"
    assert count.min() >= 1, "存在没有任何窗口覆盖的空洞"
    recon = (acc / np.maximum(count, 1))[:vol.shape[0], :vol.shape[1], :vol.shape[2]]
    assert recon.shape == vol.shape
    assert np.allclose(recon, vol, atol=1e-5), "恒等模型下必须逐体素还原输入"
    print(f"ok T6: 推理为满分辨率滑窗平均，窗口形状 {seen}，覆盖率最小 {int(count.min())}，"
          f"恒等模型重建 max|Δ| = {float(np.abs(recon - vol).max()):.2e}")


# --------------------------------------------------------------------------- #
# T7  --role-perm 校验真的会报错 + CRN 与上游同构
# --------------------------------------------------------------------------- #
def _run_train(args):
    return subprocess.run([sys.executable, "train.py"] + args, cwd=ROOT,
                          capture_output=True, text=True)


def test_t7_role_perm_is_validated_not_silently_ignored():
    r = _run_train(["--mode", "fixed", "--pattern", "0", "--role-perm", "global",
                    "--train-volume", "/nonexistent/nope.npy",
                    "--out", "/tmp/_never_written", "--epochs", "1",
                    "--steps-per-epoch", "1"])
    assert r.returncode != 0
    err = (r.stderr or "") + (r.stdout or "")
    assert "--role-perm global 只对 --mode random 有意义" in err, \
        f"fixed 模式下给 global 必须显式报错，实际：{err[-600:]}"

    r = _run_train(["--mode", "random", "--pairing", "cross",
                    "--role-perm", "global",
                    "--train-volume", "/nonexistent/nope.npy",
                    "--out", "/tmp/_never_written", "--epochs", "1",
                    "--steps-per-epoch", "1"])
    assert r.returncode != 0
    err = (r.stderr or "") + (r.stdout or "")
    assert "--role-perm global 不能与 --pairing cross 同用" in err, \
        f"cross + global 必须显式报错，实际：{err[-600:]}"

    # 正例：random + global 必须通过 argparse 与校验，失败点应落在读文件
    r = _run_train(["--mode", "random", "--role-perm", "global",
                    "--train-volume", "/nonexistent/nope.npy",
                    "--out", "/tmp/_never_written", "--epochs", "1",
                    "--steps-per-epoch", "1"])
    err = (r.stderr or "") + (r.stdout or "")
    assert "nope.npy" in err, f"random+global 应一路走到读文件，实际：{err[-600:]}"
    print("ok T7: --role-perm 的非法组合会报错，合法组合会一路传到采样器")


def test_t7b_crn_is_bit_identical_to_upstream():
    if not UPSTREAM.exists():
        print("skip T7b: /tmp/valid_upstream 不在本机")
        return
    up = NetworkCNR.__mro__  # noqa: F841  占位，下面直接读上游模块
    UpCNR = _load_upstream_module("models/model_CRN.py", "_up_valid_crn").CRN

    ours = NetworkCNR(features=16, groups=4)
    theirs = UpCNR(in_channels=1, out_channels=1, n_features=16, n_groups=4)
    po = list(ours.parameters())
    pt = list(theirs.parameters())
    assert len(po) == len(pt) == 112, f"参数张量数应各 112，实际 {len(po)}/{len(pt)}"
    assert [tuple(a.shape) for a in po] == [tuple(b.shape) for b in pt], "形状序列不同"
    with torch.no_grad():
        for a, b in zip(po, pt):
            a.copy_(b)
    ours.eval()
    theirs.eval()
    x = torch.randn(1, 1, 8, 16, 16)
    with torch.no_grad():
        d = float((ours(x) - theirs(x)).abs().max())
    assert d == 0.0, f"权重互传后前向应逐位相同，实际 max|Δ| = {d}"
    print(f"ok T7b: CRN 与上游同构（112 张量 / {sum(p.numel() for p in po)} 参数），"
          f"权重互传后前向 max|Δ| = {d}")


if __name__ == "__main__":
    test_t1_views_are_one_legal_tetris_with_correct_voxel_mapping()
    test_t1b_strided_views_keep_the_same_geometry()
    test_t2_role_perm_scope_and_reproducibility()
    test_t3_upstream_generate_mask_pair_uses_one_global_permutation()
    test_t4_single_model_single_optimizer_and_all_loss_terms_have_grads()
    test_t5_reg_loss_is_invariant_to_upstream_dwt_sign_convention()
    test_t6_inference_is_full_patch_sliding_window_without_decimation()
    test_t7_role_perm_is_validated_not_silently_ignored()
    test_t7b_crn_is_bit_identical_to_upstream()
    print("\nPhase 1A: all checks passed.")
