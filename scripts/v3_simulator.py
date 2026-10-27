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
    # ---- 2026-10-26 新增（R06 整改，见 reports/v3_repair_R06_forward_model.md §1）----
    # forward="v1"：血流 = `gauss_field(psf) * vessels`（散斑场乘布尔掩膜），真值在掩膜外**恒为 0**、
    #   管内无内部结构。这就是 R06 指出的"不是血管密度卷积 PSF"。
    # forward="v2"：血流 = `conv(sqrt(ρ) * 白噪, psf)`，ρ 为**连续血管密度**（抗锯齿软管），
    #   即"血管密度 → PSF → 复数 IQ"，真值随之变成管周有晕、管内有脊的平滑场。
    forward: str = "v1"
    # v2 专用：把"血管 ROI（ρ≥0.5）的真值 PD"标定到**背景（ρ=0）PD 中位数**的多少 dB。
    # 口径 = `scripts/v3_r06_v2_audit.py` 用的那个量，即带通之后、
    # 背景用真血管外体素（ρ=0）。2026-10-27 修正后定标改为两阶迭代，seed 100/109/110 的
    # 实测偏差 ≤0.04 dB（此前用未带通功率估计时逐体子散布 −1.7 ~ +9.6 dB）。
    # 取值区间有实测依据：SA64 只有 0.05% 体素高于中位数 +6 dB，HSA64 有 6.6%，默认 6 dB 落在两者之间。
    flow_contrast_db: float = 6.0
    # v2 专用：密度斜坡宽度（体素）。1.5 表示"血管只占体素一部分"的部分容积近似。
    density_soft: float = 1.5


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


def vessel_density(rng, shape, n_tubes, radius, soft=1.5):
    """R06/v2：随机游走管状血管的**连续密度** ρ ∈ [0,1]，取代布尔掩膜。

    为什么要连续：真实血管比体素细，一个体素里只装得下血管的一部分；
    布尔掩膜把"血管占 10% 的体素"和"血管占 100% 的体素"当成同一件事，
    于是真值只能在 0 和 1 之间跳——这正是 R06 判定的"无亚分辨能力"。

    实现：逐段点到轴距离 d，取线性斜坡 `clip((r + 0.5 − d) / soft, 0, 1)`。
    `soft` 是斜坡宽度；它不是真实的部分容积积分，而是一个**声明过的近似**（见报告 §2.3）。
    随机游走用的随机数与 `vessel_phantom` 完全一致，所以同一个 seed 得到同一套管子几何。
    """
    z, y, x = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]].astype(np.float32)
    rho = np.zeros(shape, np.float32)
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
            dist = np.sqrt((z - p[0]) ** 2 + (y - p[1]) ** 2 + (x - p[2]) ** 2)
            # .astype(np.float32)：dist 是 float64（p 来自 rng.uniform），np.maximum 会把
            # rho 升成 float64，导致"同一个 seed 的 rho"在内存里出现两种 dtype
            # （2026-10-27 发现：下游逐位自检因此误报 False，实际差 3e-8＝float32 舍入）
            rho = np.maximum(rho, np.clip((r + 0.5 - dist) / soft, 0.0, 1.0)).astype(np.float32)
    return rho


# ------------------------------------------------------------------ 仿真
def simulate(cfg: SimCfg):
    rng = np.random.default_rng(cfg.seed)
    S, T = tuple(cfg.shape), cfg.T

    valid = np.ones(S, bool)
    valid[:2] = False                      # 模仿真实数据的近场死区

    if cfg.forward == "v2":
        rho = vessel_density(rng, S, cfg.n_tubes, cfg.vessel_radius, cfg.density_soft)
        vessels = rho >= 0.5
    else:
        vessels = vessel_phantom(rng, S, cfg.n_tubes, cfg.vessel_radius)
        rho = vessels.astype(np.float32)
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
    if cfg.forward == "v2":
        # R06/v2：**密度先乘白噪、再经 PSF 卷积**。与 v1 的差别是卷积发生在乘掩膜之后，
        # 于是 PSF 会把血管支撑"糊出晕"，真值不再是硬掩膜；sqrt(ρ) 使 PD ∝ 血流密度。
        sq = np.sqrt(rho).astype(np.float32)
        for t in range(T):
            white = (rng.standard_normal(S) + 1j * rng.standard_normal(S)) / np.sqrt(2)
            flow[..., t] = _fft_convolve(white.astype(np.complex64) * sq, cfg.psf_sigma)
        # 幅度的**物理标定**：把"血管真值 PD 相对背景 PD 中位数"标到 cfg.flow_contrast_db。
        #
        # 2026-10-27 修正：初版用**未带通**的功率估背景与核心，结果逐体子的带通后对比度在
        # −1.7 dB ~ +9.6 dB 之间乱跳（原因：静态散斑与杂波在带通后大幅衰减，而未带通估计
        # 里它们是主导项；另外 ρ≥0.9 的"核心"只有 16–175 个体素，中位数极不稳定）。
        # 现在改成**直接用带通后的功率**定标：先把不带血流的各项加起来做一次同一带通得到
        # 背景 PD，再对血流做一次带通得到真值 PD，按二者之比定标。因为带通是线性的，
        # 定标系数可以精确作用在 IQ 上，`gt = band(flow)` 随之等比变化，无需重算。
        # 定标区改用 ρ≥0.5（体素量级 10²–10³，见 reports/v3_r06/v2_endpoint_audit.json）。
        rest_iq = np.zeros(S + (T,), np.complex64)
        for part in (static, dynamic, clutter, thermal):
            rest_iq += part
        filt_flow, _ = svd_blood_band(flow.astype(np.complex64), 60, 350, True)
        gt_now = np.sum(np.abs(filt_flow) ** 2, axis=3)
        cal = rho >= 0.5                     # 定标区：ρ≥0.5（体素量级 10²–10³）
        bg_true = valid & (rho <= 0.0)       # 背景：真血管外（与审计脚本同一口径）
        target_ratio = 10.0 ** (cfg.flow_contrast_db / 10.0)
        if cal.any() and bg_true.any():
            # 一阶：背景用**不含血流**的带通功率（血流自身的晕先不计）
            pd_rest, _ = band_and_pd(rest_iq, valid)
            r1 = float(np.median(gt_now[cal])) / max(float(np.median(pd_rest[bg_true])), 1e-30)
            k = np.sqrt(target_ratio / max(r1, 1e-30))
            flow *= k
            # 二阶：血流 + 背景一起带通，把血流的"晕"算进背景（审计口径一致），再修一次
            pd_obs, _ = band_and_pd(rest_iq + flow, valid)
            r2 = (k ** 2 * float(np.median(gt_now[cal]))) / max(
                float(np.median(pd_obs[bg_true])), 1e-30)
            flow *= np.sqrt(target_ratio / max(r2, 1e-30))
            del pd_rest, pd_obs
        del rest_iq, filt_flow, gt_now
    else:
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

    return {"X": X, "flow": flow.astype(np.complex64) * 1e3, "vessels": vessels,
            "rho": rho, "valid": valid, "cfg": asdict(cfg)}


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
    ap.add_argument("--forward", default="v1", choices=["v1", "v2"],
                    help="v1=旧前向模型（散斑场乘布尔掩膜）；v2=R06 整改版"
                         "（连续血管密度 → PSF 卷积），真值不再是硬掩膜")
    ap.add_argument("--flow-contrast-db", type=float, default=6.0,
                    help="v2 专用：血管核心真值 PD 相对背景 PD 中位数的 dB 数")
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
                        static_seed=args.static_seed, forward=args.forward,
                        flow_contrast_db=args.flow_contrast_db))
    pd, filt = band_and_pd(d["X"], d["valid"])
    gt, _ = band_and_pd(d["flow"], d["valid"])
    if args.forward == "v2":
        # 如实记录**带通之后**真正达到的血管/背景对比度（标定阶段用的是未带通的一阶估计）
        core = d["rho"] >= 0.5
        bg = background_mask(pd, d["valid"], 0.20)
        ach = 10.0 * np.log10(max(float(np.median(gt[core])), 1e-30)
                              / max(float(np.median(pd[bg])), 1e-30))
        print(f"  v2 对比度：目标 {args.flow_contrast_db} dB → 带通后实测 {ach:.2f} dB"
              f"（核心体素 {int(core.sum())}，背景体素 {int(bg.sum())}）")
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
                        rho=d["rho"].astype(np.float32),   # v2 的连续密度（v1 时等于掩膜）
                        cfg=json.dumps(d["cfg"]), **extra)
    e = acf_vs_target(pd, d["valid"], target, mode=args.target_mode)
    print(json.dumps(e, indent=1)[:1200])
    print(f"wrote {out}/{name}")


if __name__ == "__main__":
    main()
