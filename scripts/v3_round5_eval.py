"""Round5 评估：采样臂（m=1 标准 / m=2 我们的）→ 全幅回填 → 统一域指标。

预注册：`reviews/V3_PR1_round5_prereg_sampling_direct.md`（§3 评价协议、§4 判据、§6 预算）。
本条脚本**只推理，不训练**，也不改任何阈值。

为什么不能复用 `infer.py`
--------------------------
`infer.py` 是满分辨率滑窗，只对 `--mode dense`（有监督上界臂）成立。
本轮三条臂都是**自监督相位采样**臂：训练输入是半尺寸的相位子采样视图，
推理必须按同一套相位几何回填。且 m=2 时 8 个相位只覆盖 12.5%，
要把整幅铺满必须再做 `m³ = 8` 个格点平移各前向一次（预注册 §3 写死：
每个体素恰好被覆盖一次，不做平均）。m=2 的推理代价 = m=1 的 8 倍，这本身就是部署代价。

用法
----
    python -u scripts/v3_round5_eval.py --gpu 2
产物
----
    reports/v3_round5/sampling_direct.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from models.network import NetworkCNR                             # noqa: E402
from v3_round4_metrics import load, masked_gauss                   # noqa: E402
from v3_round4_metrics_v3 import metrics_v3                        # noqa: E402
from v3_round4_b1_prep import fwd, inv                             # noqa: E402
from v3_expA_v2_eval import ckpt_budget                            # noqa: E402

HELD = [902, 903, 904, 905]
LEVEL = "low"

# 臂 -> (域, 步距 m, 建图标签)
ARMS = {
    "S_log": {"domain": "logeps", "stride": 1,
              "desc": "标准 VALID：m=1 随机 tetris + cross 配对"},
    "O_log": {"domain": "logeps", "stride": 2,
              "desc": "我们的采样：m=2 + 预注册体对角角色对（--slots 1 4 6 3）"},
    "O_lin": {"domain": "linear", "stride": 2,
              "desc": "同 O_log，域为 linear"},
}

# `--slots 1 4 6 3` 的实际训练映射：loss_terms = 0.5·mse(ha,ya) + 0.5·mse(hb,yb)，
# 即 model(phase 1) → phase 6、model(phase 4) → phase 3。
SCATTER = {"O_log": {1: 6, 4: 3}, "O_lin": {1: 6, 4: 3}}
OURS_TARGETS = (3, 6)


def npz_path(seed: int) -> Path:
    return ROOT / f"data/processed/v3_round2_s0/{LEVEL}6/eval/s0_{LEVEL}_e{seed}.npz"


def usable(shape, m: int) -> tuple[int, int, int]:
    step = 2 * m
    return tuple((n // step) * step for n in shape)


def phase_slices(shape, m: int, index: int, shift=(0, 0, 0)):
    """相位 index 在步距 m 下的体素切片（与 `scripts/v3_phase3_eval.py` 同一约定）。"""
    dz, dy, dx = index // 4, (index // 2) % 2, index % 2
    return tuple(slice(m * o + s, n, 2 * m)
                 for n, o, s in zip(shape, (dz, dy, dx), shift))


def reconstruct_full(model, vol: np.ndarray, m: int, device,
                     scatter_to: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """全幅回填：8 个相位 × m³ 个格点平移。每个体素恰好覆盖一次（不平均）。

    m=1 时 m³=1，等价于原版 8 相位回填（覆盖率 100%）。
    """
    shape = usable(vol.shape, m)
    vol = vol[:shape[0], :shape[1], :shape[2]]
    out = np.zeros(shape, np.float32)
    filled = np.zeros(shape, bool)
    with torch.no_grad():
        for shift in itertools.product(range(m), repeat=3):
            for i in range(8):
                target = i if scatter_to is None else scatter_to.get(i)
                if target is None:
                    continue                    # 固定角色臂只学过 input→target 这一条映射
                sl_in = phase_slices(shape, m, i, shift)
                sl_out = phase_slices(shape, m, target, shift)
                sub = vol[sl_in]
                if sub.size == 0:
                    continue
                x = torch.from_numpy(np.ascontiguousarray(sub))[None, None].to(device)
                y = model(x)[0, 0].detach().cpu().numpy()
                if y.shape != sub.shape:
                    raise RuntimeError(f"输出形状 {y.shape} 与输入 {sub.shape} 不一致")
                if filled[sl_out].any():
                    raise RuntimeError(f"m={m} shift={shift} phase={i} 出现重叠覆盖")
                out[sl_out] = y
                filled[sl_out] = True
    if scatter_to is None and not filled.all():
        raise RuntimeError(f"m={m} 回填未铺满：{filled.mean():.4f}")
    return out, filled


def predict(arm: str, ckpt: Path, E: dict, norm: dict, gpu: int,
            scatter_to: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """整卷推理 → 反变换回线性 PD → 回到冻结的 log1p 指标域（与 B1 完全同域）。

    返回 `(预测, 覆盖掩膜)`。固定角色臂（我们的采样）只学过 input→target 这一条映射，
    必须按 `scatter_to` 回填，因此只覆盖部分体素；覆盖掩膜交回调用方，
    以便所有臂在**同一批体素**上比较（否则分母不同，比较无意义）。
    """
    spec = ARMS[arm]
    m = spec["stride"]
    dom = spec["domain"]
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt / "latest.pt", map_location=device)
    targs = ck.get("args", {})
    model = NetworkCNR(
        features=targs.get("features", 16), groups=targs.get("groups", 4),
        residual=(targs.get("arch_head", "direct") == "residual"),
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    s = float(E["s"])
    valid = E["valid"]                    # 调用方传入的已是裁剪后的场景（与基线同掩膜）
    x = fwd(dom, np.expm1(E["vol"]) * s, s, valid)
    z = ((x - norm[dom]["mean"]) / norm[dom]["std"]).astype(np.float32)
    y, filled = reconstruct_full(model, z, m, device, scatter_to=scatter_to)
    xhat = y * norm[dom]["std"] + norm[dom]["mean"]
    pd = inv(dom, xhat, s)
    lg = np.log1p(np.maximum(pd, 0.0) / max(s, 1e-12)).astype(np.float32)
    lg[~valid] = 0.0
    return lg, filled


def crop_eval(E: dict, shape) -> dict:
    """把场景裁到与回填一致的可评估形状（所有掩膜逐个裁，避免臂间不同分母）。

    `metrics_v3` 需要 `vol`/`vol2`/`truth_img`/`rho`/`valid`，所以这里**按数组逐个裁**，
    标量（`s` 等）原样带过去，避免漏键导致分母不一致。
    """
    out = {}
    for k, v in E.items():
        out[k] = v[:shape[0], :shape[1], :shape[2]] if isinstance(v, np.ndarray) else v
    return out


def own_pairing_mask(shape, m: int, targets) -> np.ndarray:
    """固定角色臂在部署时能填的体素集合（几何量，与模型无关）。

    `--slots 1 4 6 3` 只学过 `1→6`、`4→3` 两条映射（`loss_terms`：`mse(ha,ya)` / `mse(hb,yb)`），
    所以 m=2 下 8 个相位里只有 {3,6} 有输出，再加上 m³ 个格点平移 ⇒ 覆盖 2/8 = 25%。
    这不是实现细节，是固定角色采样的部署代价。
    """
    mask = np.zeros(shape, bool)
    for shift in itertools.product(range(m), repeat=3):
        for t in targets:
            mask[phase_slices(shape, m, t, shift)] = True
    return mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--recon", choices=["fold_all", "own_pairing"], default="fold_all",
                    help="固定角色臂的回填口径：fold_all = 输入相位→同相位（预注册 §3 字面），"
                         "own_pairing = 输入相位→训练时学的目标相位（+25%% 覆盖）。"
                         "两种都报，主判定用 own_pairing（fold_all 系统低估固定角色臂，"
                         "理由见 reports/v3_phase3_eval.py 的 reconstruct docstring）。")
    ap.add_argument("--out-json", default="reports/v3_round5/sampling_direct.json")
    ap.add_argument("--cache-dir", default="runs/_infer_v2/v3_round5")
    args = ap.parse_args()

    norm = json.loads((ROOT / "reports/v3_round4/b1_norm.json").read_text())
    prereg = "reviews/V3_PR1_round5_prereg_sampling_direct.md"
    out = {"prereg": prereg, "held": HELD, "recon": args.recon,
           "arms": {}, "baselines": {}, "criteria": {}}

    E_all = {s: load(npz_path(s)) for s in HELD}

    # 所有臂与基线共用同一裁剪形状（按最大步距 m=2 → 4 的倍数）
    shape = usable(E_all[HELD[0]]["vol"].shape, 2)

    # 固定角色臂的实际覆盖（几何量）：own_pairing 时所有臂都只在同一批体素上评分。
    # **注意**：只在算指标时收窄 `valid`，绝不能在喂网络前收窄——`fwd()` 会把
    # 非 valid 体素置零，那样等于给模型喂一张 75% 全零的图（2026-11-02 首跑踩过，
    # 会让 S_log 从 0.0166 虚高到 0.59，属于脚本 bug 而不是方法结果）。
    shared = own_pairing_mask(shape, 2, OURS_TARGETS) if args.recon == "own_pairing" else None
    out["shared_mask_fraction"] = float(shared.mean()) if shared is not None else 1.0

    def eval_view(E: dict) -> dict:
        Ec = crop_eval(E, shape)
        if shared is not None:
            Ec["valid"] = Ec["valid"] & shared
        return Ec

    for s, E in E_all.items():
        Ec = eval_view(E)              # 指标口径（own_pairing 时收窄到共享 25% 掩膜）
        Ef = crop_eval(E, shape)       # 参考基线口径：平滑始终用**全 valid**
        # 平滑必须用全 valid：`masked_gauss` 是归一化卷积，若直接喂 25% 的周期性散点掩膜，
        # 归一化分母随之退化、核变成"抽点平均"，同一条 σ*=0.6 的模糊会被削弱成几乎不模糊
        # （实测 e902: 全 valid 平滑后到窄掩膜上评分仍是 0.0097，掩膜内平滑则虚高到 0.0195）。
        # 这只影响参考行，不影响任何一臂（各臂的预测来自全幅 reconstruct_full）。
        out["baselines"][f"e{s}"] = {
            "input": metrics_v3(Ec["vol"], Ec),
            "gauss": metrics_v3(masked_gauss(Ef["vol"], Ef["valid"], 0.6), Ec),
        }

    for arm in args.arms:
        ck = ROOT / f"runs/v3_round5/{arm}_s3407"
        if not (ck / "latest.pt").exists():
            out["arms"][arm] = {"status": "NOT_RUN", "reason": "checkpoint 不存在"}
            print(f"[{arm}] NOT_RUN")
            continue
        rec = {"checkpoint": str(ck.relative_to(ROOT)),
               "spec": ARMS[arm], "budget": ckpt_budget(ck / "latest.pt"), "held": {}}
        for s in HELD:
            E = E_all[s]
            Ec = eval_view(E)          # 指标口径（own_pairing 时已收窄 valid）
            Ein = crop_eval(E, shape)  # 喂网络的口径：始终是完整 valid
            cache = ROOT / args.cache_dir / f"{arm}_{args.recon}_e{s}_pred_log1p.npy"
            if cache.exists():
                pr = np.load(cache)
            else:
                scatter = (SCATTER[arm]
                           if (args.recon == "own_pairing" and arm in SCATTER) else None)
                pr, _filled = predict(arm, ck, Ein, norm, args.gpu, scatter_to=scatter)
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, pr)
            m = metrics_v3(pr, Ec)
            rec["held"][f"e{s}"] = m
            print(f"[{arm} e{s}] rmse_img={m['overall_rmse']:.4f} "
                  f"bias_ves={m['bias_ves']:+.4f} var_bg={m['var_bg']:.2e} n2n={m['n2n']:.3f}")
        out["arms"][arm] = rec

    # ---- 预注册 §4 判据（阈值写死在预注册里，此处只是执行）----
    def rmse(arm: str, s: int) -> float:
        return out["arms"][arm]["held"][f"e{s}"]["overall_rmse"]

    crit = {"rule": "C1 实质差异：逐场景相对变化≥5% 且方向一致≥3/4；"
                    "C2 无实质差异：<2% ≥3/4；C3 其余=无法判定",
            "comparisons": {}}
    if "S_log" in out["arms"] and "O_log" in out["arms"]:
        rel = {f"e{s}": (rmse("O_log", s) - rmse("S_log", s)) / rmse("S_log", s) for s in HELD}
        n_better = sum(1 for v in rel.values() if v <= -0.05)
        n_worse = sum(1 for v in rel.values() if v >= 0.05)
        n_tie = sum(1 for v in rel.values() if abs(v) < 0.02)
        if max(n_better, n_worse) >= 3:
            verdict = "C1_实质差异"
            direction = "OURS_BETTER" if n_better > n_worse else "OURS_WORSE"
        elif n_tie >= 3:
            verdict, direction = "C2_无实质差异", "NONE"
        else:
            verdict, direction = "C3_无法判定", "UNRESOLVED"
        crit["comparisons"]["O_log_vs_S_log"] = {
            "rel_change": rel, "n_ours_better_5pct": n_better,
            "n_ours_worse_5pct": n_worse, "n_tie_2pct": n_tie,
            "verdict": verdict, "direction": direction,
        }
        print(f"[criteria] O_log vs S_log: {verdict} / {direction} / rel={rel}")
    out["criteria"] = crit

    p = ROOT / args.out_json
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
