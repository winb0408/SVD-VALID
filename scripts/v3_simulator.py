"""V3 Phase 0 / T0-6：有真值的三维超声 Power Doppler 仿真器 v0。

目标不是"造得像"，而是**造到能被已知答案校验的程度**：
仿真出来的背景 log-PD 空间自相关，必须和 Phase 1 在真实数据上测到的 ACF 对得上。
对不上就不能拿它的"真值"去做 Phase 4 的主终点。

组成（每一项都可关掉，用于消融）：
  static speckle   固定散斑：同一空间图案 x 宽谱时间起伏（纯常数会退化成秩 1，被 SVD 当杂波删掉）
  dynamic speckle  逐帧独立散斑
  clutter          残余杂波：空间大尺度 x 低频时间谱
  flow             血管内血流（**逐帧独立散斑**；初版写成秩 1，会被带通 SVD 整条删掉，见 2026-10-20 修正）
  thermal          热噪声（更窄的 PSF）

全链：复数 IQ -> 官方带通 SVD [59,350) -> PD = sum_t |.|^2 -> 取 log。
真值 = 同一带通作用于**只有血流那一项**后的 PD。

用法：
    python -u scripts/v3_simulator.py --calibrate          # 标定 PSF，使其 ACF 匹配实测
    python -u scripts/v3_simulator.py --seed 0 --out /tmp/x
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from official_pipeline import svd_blood_band  # noqa: E402
from v3_correlation_length import acf_all_axes  # noqa: E402


# --------------------------------------------------------------------- 配置
@dataclass
class SimCfg:
    shape: tuple = (37, 64, 64)
    T: int = 400
    psf_sigma: tuple = (3.0, 2.37, 3.97)   # (z,y,x) 以体素为单位，标定值见 REPORTS
    thermal_sigma: tuple = (1.0, 1.2, 1.1)
    static_speckle: float = 1.0
    static_mod: float = 0.35               # 固定散斑的时间起伏幅度（0=纯常数，会被 SVD 删干净）
    dynamic_speckle: float = 0.12
    clutter_amp: float = 2.0
    clutter_sigma: tuple = (5.0, 8.0, 8.0)
    clutter_rank: int = 6
    flow_amp: float = 2.5
    vessel_radius: float = 1.1
    n_tubes: int = 14
    thermal: float = 0.25
    seed: int = 0
    # 2026-10-21 新增：把"血管几何"和"噪声实现"的随机流分开。
    # 设了 noise_seed 之后，血管掩膜仍由 seed 决定（同一个 seed 得同一套血管），
    # 而散斑/杂波/热噪声/血流散斑改用 noise_seed 的流——于是可以拿到
    # **同一场景的两次独立观测**。这是 Noise2Noise 式评估的前提：
    # 拿独立实现当检验目标，"保留静态本底"就是对的，不必把背景底抹成 0。
    # 不设（None）时行为与加这个字段之前逐位相同。
    noise_seed: int | None = None
    # 2026-10-21 新增：单独控制**固定散斑**（空间图案 + 慢时间起伏）的随机流。
    # 真实采集里固定散斑由组织微结构和成像几何决定，**两次重复观测共享同一张**
    # （Phase 1 §4 的重复观测差分里它精确抵消）。所以"静态共享、动态独立"才是
    # 物理上对应的两次观测。设成与 seed 相同的值时，这里会把血管幽灵重画一遍
    # 以推进到同一流位置，结果与不设此字段**逐位相同**（已验证）。
    static_seed: int | None = None


# --------------------------------------------------------------- 基础构件
def _fft_convolve(field, sigma):
    """可分离 Gaussian 卷积（周期边界）。field: (Z,Y,X) complex。"""
    out = field
    for ax, s in enumerate(sigma):
        if s <= 0:
            continue
        n = out.shape[ax]
        f = np.fft.fftfreq(n)
        g = np.exp(-2.0 * (np.pi * f * s) ** 2)      # Gaussian 的频域响应
        shape = [1] * out.ndim
        shape[ax] = n
        out = np.fft.ifft(np.fft.fft(out, axis=ax) * g.reshape(shape), axis=ax)
    return out


def gauss_field(rng, shape, sigma):
    """单位方差的复高斯随机场，空间相关由 sigma 决定。"""
    w = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)
    f = _fft_convolve(w, sigma)
    return f / max(np.sqrt(np.mean(np.abs(f) ** 2)), 1e-12)


def broad_time_series(rng, T, n_lines=60, exponent=1.0, dc=1.0):
    """宽谱随机时间序列：n_lines 条随机频率的正弦叠加。DC 分量单独给。"""
    t = np.arange(T)
    s = np.full(T, dc, dtype=np.float64)
    freqs = rng.choice(np.arange(1, T // 2), size=n_lines, replace=False)
    for k in freqs:
        a = (rng.standard_normal() + 1j * rng.standard_normal()) / (k ** exponent)
        s += 2.0 * np.real(a * np.exp(2j * np.pi * k * t / T))
    return s


def low_time_series(rng, T, kmax=40):
    """低频时间序列（杂波用）：白噪声经频域 Gaussian 低通。"""
    f = np.fft.fftfreq(T) * T
    g = np.exp(-0.5 * (f / kmax) ** 2)
    return np.real(np.fft.ifft(np.fft.fft(rng.standard_normal(T)) * g))


def vessel_phantom(rng, shape, n_tubes, radius):
    """随机游走管状血管，返回布尔掩膜 mask。"""
    z, y, x = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]].astype(np.float64)
    mask = np.zeros(shape, bool)
    for _ in range(n_tubes):
        p = np.array([rng.uniform(0, shape[0]), rng.uniform(0, shape[1]), rng.uniform(0, shape[2])])
        d = rng.standard_normal(3)
        d /= np.linalg.norm(d)
        r = radius * rng.uniform(0.5, 1.4)
        for _seg in range(60):
            d = d + 0.35 * rng.standard_normal(3)
            d /= np.linalg.norm(d)
            p = p + d * 1.0
            if not (0 <= p[0] < shape[0] and 0 <= p[1] < shape[1] and 0 <= p[2] < shape[2]):
                break
            dd = (z - p[0]) ** 2 + (y - p[1]) ** 2 + (x - p[2]) ** 2
            mask |= dd <= r * r
    return mask


# ------------------------------------------------------------------ 仿真
def simulate(cfg: SimCfg):
    rng = np.random.default_rng(cfg.seed)
    S, T = tuple(cfg.shape), cfg.T

    vessels = vessel_phantom(rng, S, cfg.n_tubes, cfg.vessel_radius)
    if cfg.noise_seed is not None:
        rng = np.random.default_rng(cfg.noise_seed)

    # --- 固定散斑：空间图案 x 宽谱时间起伏（不是常数，否则秩 1）
    # 注意：静态分量**必须**先从主流程 rng 抽走，否则下游（动态/杂波/血流/热噪声）
    # 的随机流会整体前移，与不设 static_seed 时不再是同一个实现。
    s_spatial = gauss_field(rng, S, cfg.psf_sigma) * np.sqrt(cfg.static_speckle)
    u_static = 1.0 + cfg.static_mod * broad_time_series(rng, T, n_lines=80, exponent=0.7, dc=0.0)
    if cfg.static_seed is not None:
        rng_s = np.random.default_rng(cfg.static_seed)
        vessel_phantom(rng_s, S, cfg.n_tubes, cfg.vessel_radius)   # 推进到同一流位置
        s_spatial = gauss_field(rng_s, S, cfg.psf_sigma) * np.sqrt(cfg.static_speckle)
        u_static = 1.0 + cfg.static_mod * broad_time_series(
            rng_s, T, n_lines=80, exponent=0.7, dc=0.0)
    static = s_spatial[..., None] * u_static[None, None, None, :]

    # --- 动态散斑
    dynamic = np.empty(S + (T,), np.complex64)
    thermal = np.empty(S + (T,), np.complex64)
    for t in range(T):
        dynamic[..., t] = gauss_field(rng, S, cfg.psf_sigma) * np.sqrt(cfg.dynamic_speckle)
        thermal[..., t] = cfg.thermal * gauss_field(rng, S, cfg.thermal_sigma)

    # --- 杂波：空间大尺度 x 低频时间
    clutter = np.zeros(S + (T,), np.complex64)
    for _ in range(cfg.clutter_rank):
        c = gauss_field(rng, S, cfg.clutter_sigma)
        c /= max(np.sqrt(np.mean(np.abs(c) ** 2)), 1e-12)
        g = low_time_series(rng, T)
        g /= max(np.std(g), 1e-12)
        clutter += (cfg.clutter_amp / np.sqrt(cfg.clutter_rank)) * c[..., None] * g[None, None, None, :]

    # --- 血流：只在血管内，**逐帧独立散斑**（宽带时间谱）
    # 2026-10-20 修正：初版写成"一个空间图案 × 一条时间曲线"，那是**秩 1**，
    # 会被官方带通 SVD（删掉前 59 个分量）当杂波整条删掉——血流在 PD 图上等于不存在
    # （`band_and_pd(flow)` 的方差 ≈ 0）。现在用逐帧独立散斑：时间谱平坦，
    # 带通 59..349 留得住约 73% 的能量，PD 图上才真的出现血管结构。
    flow = np.empty(S + (T,), np.complex64)
    for t in range(T):
        flow[..., t] = gauss_field(rng, S, cfg.psf_sigma) * vessels
    if vessels.any():
        v_rms = np.sqrt(np.mean(np.abs(flow[..., 0][vessels]) ** 2))
        flow *= (cfg.flow_amp * 0.6) / max(float(v_rms), 1e-12)

    X = np.zeros(S + (T,), np.complex64)
    for part in (static, dynamic, clutter, flow, thermal):
        X += part

    # --- 数值上凑到和真实数据同量级的动态范围
    X = X.astype(np.complex64) * 1e3

    valid = np.ones(S, bool)
    valid[:2] = False                      # 模仿真实数据的近场死区

    return {"X": X, "flow": flow.astype(np.complex64) * 1e3, "vessels": vessels,
            "valid": valid, "cfg": asdict(cfg)}


def band_and_pd(X, valid, cut=(60, 350)):
    filt, _ = svd_blood_band(X.astype(np.complex64), cut[0], cut[1], True)
    pd = np.sum(np.abs(filt) ** 2, axis=3)
    return pd, filt


def background_mask(pd, valid, q=0.20):
    return valid & (pd <= np.quantile(pd[valid], q))


def acf_vs_target(pd, valid, target, q=0.20, mode="fast"):
    """仿真 PD 背景 log-ACF 与实测目标比较。target: {axis: [rho@lag1..]}"""
    lg = np.log(np.maximum(pd, 1e-12))
    bg = background_mask(pd, valid, q)
    got = acf_all_axes(lg, bg, lags=list(range(1, len(next(iter(target.values()))) + 1)))
    err = {}
    for ax in ("z", "y", "x"):
        g = np.array([c[0] for c in got[ax]], float)
        if mode == "fast":
            plat = float(np.nanmean(g[-2:]))
            g = (g - plat) / max(1.0 - plat, 1e-9)
        t = np.array(target[ax][:len(g)], float)
        err[ax] = {"sim": g.tolist(), "target": t.tolist(),
                   "rmse": float(np.sqrt(np.nanmean((g - t) ** 2)))}
    err["overall_rmse"] = float(np.sqrt(np.mean([err[a]["rmse"] ** 2 for a in ("z", "y", "x")])))
    return err


def load_target(mode="fast"):
    """mode='fast' 去掉远 lag 平台（只对散斑分量做标定）；'raw' 用原始曲线。"""
    j = json.loads(Path("data/processed/v3_correlation_length.json").read_text())
    out = {}
    for seq in j:
        out[seq] = {}
        for ax in ("z", "y", "x"):
            c = np.array([v[0] for v in j[seq]["P1"]["acf"][ax]], float)
            if mode == "fast":
                plat = float(np.nanmean(c[-2:]))
                c = (c - plat) / max(1.0 - plat, 1e-9)
            out[seq][ax] = c.tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--out", default="data/processed/v3_sim")
    ap.add_argument("--seq", default="SA64", help="ACF 标定目标用哪条真实序列")
    ap.add_argument("--target-mode", default="fast", choices=["fast", "raw"])
    ap.add_argument("--noise-seed", type=int, default=None,
                    help="设了之后血管仍由 --seed 决定，噪声改用这条流；"
                         "输出文件名变成 sim_seed{S}_n{N}.npz（同一场景的独立观测）")
    ap.add_argument("--static-seed", type=int, default=None,
                    help="单独控制固定散斑的流；取与 --seed 相同 = 固定散斑与主流程一致，"
                         "用于造'静态共享、动态独立'的两次观测")
    ap.add_argument("--emit-halves", action="store_true",
                    help="额外存 pd_half1/pd_half2：把 400 帧按时间奇偶各半重建 PD。"
                         "这两个量的差给出**与判据同一种噪声**的空间分辨估计，"
                         "用于时间交错自参考（见 scripts/v3_phase4_tire.py）。"
                         "注意带通秩按帧数等比缩放到 [30,175)，保持同一时间频带。")
    args = ap.parse_args()

    targets = load_target(args.target_mode)
    target = targets[args.seq]

    if args.calibrate:
        print(f"标定目标：{args.seq} 的 P1 背景 log-ACF")
        print(f"  target z={np.round(target['z'],3)}\n  target y={np.round(target['y'],3)}"
              f"\n  target x={np.round(target['x'],3)}")
        best = None
        # 各向异性：由目标的快速分量 lag-1 反推 Gaussian 等效长度，作为 PSF 形状的比例
        ell = np.array([float(np.sqrt(-0.5 / np.log(min(t, 0.999)))) for t in
                        (target["z"][0], target["y"][0], target["x"][0])])
        ell = ell / ell[0]
        print(f"  由目标 lag-1 反推的各向异性比例 (z,y,x) = {np.round(ell, 3)}")
        print(f"\n{'scale':>6} {'sig(z,y,x)':<22} {'rmse':>7}  sim@lag1 (z,y,x)")
        for scale in (1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.6):
            cfg = SimCfg(seed=args.seed,
                         psf_sigma=tuple(round(float(scale * e), 2) for e in ell))
            d = simulate(cfg)
            pd, _ = band_and_pd(d["X"], d["valid"])
            e = acf_vs_target(pd, d["valid"], target, mode=args.target_mode)
            l1 = ",".join(f"{e[a]['sim'][0]:.2f}" for a in ("z", "y", "x"))
            print(f"{scale:>6.2f} {str(cfg.psf_sigma):<22} {e['overall_rmse']:>7.4f}  {l1}")
            if best is None or e["overall_rmse"] < best[1]:
                best = (scale, e["overall_rmse"], cfg, e)
        print(f"\n最优 scale={best[0]} rmse={best[1]:.4f} sigma={best[2].psf_sigma}")
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "calibration.json").write_text(json.dumps(
            {"scale": best[0], "rmse": best[1], "sigma": best[2].psf_sigma,
             "detail": best[3], "target_seq": args.seq}, indent=1))
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = simulate(SimCfg(seed=args.seed, noise_seed=args.noise_seed,
                        static_seed=args.static_seed))
    pd, filt = band_and_pd(d["X"], d["valid"])
    gt, _ = band_and_pd(d["flow"], d["valid"])
    name = f"sim_seed{args.seed}"
    if args.noise_seed is not None:
        name += f"_n{args.noise_seed}"
    if args.static_seed is not None:
        name += f"_s{args.static_seed}"
    name += ".npz"
    extra = {}
    if args.emit_halves:
        # 时间奇偶各半的**功率**：用同一个带通滤波结果（不重做 SVD），只把
        # PD = Σ_t |filt_t|² 按帧的奇偶拆成两半。于是 pd = pd_half1 + pd_half2 精确成立，
        # 而且两半的估计噪声互不相关。可以证明 (pd1 − pd2)² 是 Var(pd) 的无偏估计：
        #   Var(pd) = Var(pd1) + Var(pd2)，两半同分布 ⇒ Var(pd1 − pd2) = 2Var(pd1) = Var(pd)。
        # 这就是**与判据同一种噪声**的逐体素估计，且完全来自单次采集。
        pw = np.abs(filt) ** 2
        half1 = pw[..., 0::2].sum(axis=3)
        half2 = pw[..., 1::2].sum(axis=3)
        print(f"  halves: T={d['X'].shape[3]} 帧拆奇偶，"
              f"pd 与两半之和的最大相对偏差 "
              f"{float(np.max(np.abs(half1 + half2 - pd)) / max(float(np.mean(pd)), 1e-12)):.2e}")
        extra = {"pd_half1": half1.astype(np.float32),
                 "pd_half2": half2.astype(np.float32)}
    np.savez_compressed(out / name,
                        pd_linear=pd.astype(np.float32),
                        gt_flow_pd=gt.astype(np.float32),
                        vessels=d["vessels"], valid=d["valid"],
                        cfg=json.dumps(d["cfg"]), **extra)
    e = acf_vs_target(pd, d["valid"], target, mode=args.target_mode)
    print(json.dumps(e, indent=1)[:1200])
    print(f"wrote {out}/{name}")


if __name__ == "__main__":
    main()
