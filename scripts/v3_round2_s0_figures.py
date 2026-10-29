"""S0 本轮所有图：全部从**已存档的产物**生成（JSON / history.json / checkpoint），不重跑训练。

图的用途是让"这一轮到底证明了什么"在一眼之内可核，而不是装饰。四张图分别对应报告的四条主张：

  fig1_n2n_identity_gap   —— 所有臂的 n2n 与"恒等=1.0"、"最优平稳线性=0.512"两条线的距离
                             （报告 §3、§5.4、§5.5、§7）
  fig2_error_floor        —— 预测误差 vs 输入噪声：点几乎水平 ⇒ 误差与噪声大小无关（§5.5）
  fig3_train_loss         —— 训练损失 / σ² vs epoch：low 降到 0.16、vlow 停在 1.15（§5.5、§5.4）
  fig4_error_map          —— 一个切片的 输入/预测/真值/预测−输入：肉眼可见"预测≈输入+常数扰动"（§5.5）

复算：python -u scripts/v3_round2_s0_figures.py            # 全部
      python -u scripts/v3_round2_s0_figures.py --no-map   # 跳过 fig4（不需要 GPU）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 图里要写中文：用系统里的 Noto Sans CJK（DejaVu 没有中文字形，缺字会渲染成方框）。
for _fam in ["Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei"]:
    try:
        matplotlib.font_manager.findfont(_fam, fallback_to_default=False)
    except Exception:
        continue
    matplotlib.rcParams["font.sans-serif"] = [_fam, "DejaVu Sans"]
    break
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

FIG = ROOT / "reports/v3_round2/figures"
SIG = {"low": 0.0198, "mid": 0.0397, "vlow": 0.0066}   # log1p 域有效 σ（s0_level_match.json / §2.3）
IDENTITY = 1.0
ORACLE = 0.512                                          # oracle_freq_wiener @ low（§2.2）


def _load(p: str) -> dict:
    return json.loads((ROOT / p).read_text())


def fig1() -> Path:
    """每个臂的 n2n：两个评估体分别画点，配 1.0（恒等）与 0.512（最优线性）两条参考线。"""
    rows: list[tuple[str, float, float]] = []

    def add_from(path: str, arms: dict[str, str], prefix: str = ""):
        d = _load(path)
        for arm, label in arms.items():
            if arm not in d["arms"]:
                continue
            pv = d["arms"][arm]["per_volume"]
            vals = [pv[k]["n2n_ratio"] for k in sorted(pv)]
            rows.append((prefix + label, vals[0], vals[1] if len(vals) > 1 else np.nan))

    add_from("reports/v3_round2/s0_low_metrics.json",
             {"A_rand_s3407": "A_rand (相位·自监督) s3407",
              "A_fixed4_s3407": "A_fixed4 (相位·自监督) s3407"})
    add_from("reports/v3_round2/s0_low_metrics_dense.json",
             {"S_dense_s3407": "S_dense (满分辨·真值) s3407",
              "D_n2n_s3407": "D_n2n (满分辨·第二次观测) s3407"})
    add_from("reports/v3_round2/s0_low60_metrics.json",
             {"S_dense_s3407": "S_dense 60 卷 (满分辨·真值) s3407"})
    add_from("reports/v3_round2/s0_vlow_metrics.json",
             {"S_dense_s3407": "S_dense (vlow, 噪声/3) s3407"}, prefix="")
    add_from("reports/v3_round2/s0_mid_metrics.json",
             {"A_rand_s3407": "A_rand (mid = 真实难度) s3407"})
    rows.append(("输入（恒等）", IDENTITY, IDENTITY))

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ys = np.arange(len(rows))[::-1]
    for y, (label, v1, v2) in zip(ys, rows):
        ax.plot([v1, v2] if np.isfinite(v2) else [v1], [y, y] if np.isfinite(v2) else [y],
                color="0.75", lw=2, zorder=1)
        ax.scatter(v1, y, s=70, color="#1f77b4", zorder=3, label="评估体 e900" if y == ys[0] else None)
        if np.isfinite(v2):
            ax.scatter(v2, y, s=70, marker="D", color="#ff7f0e", zorder=3,
                       label="评估体 e901" if y == ys[0] else None)
        ax.text(max(v1, v2 if np.isfinite(v2) else v1) * 1.03, y, f"{v1:.2f}"
                + (f" / {v2:.2f}" if np.isfinite(v2) else ""), va="center", fontsize=8.5, color="0.25")
    ax.axvline(IDENTITY, color="0.35", ls="--", lw=1.2)
    ax.axvline(ORACLE, color="#2ca02c", ls="-.", lw=1.2)
    ax.text(IDENTITY, len(rows) - 0.35, " 恒等 = 1.00（什么都不做）", fontsize=8.5, color="0.35")
    ax.text(ORACLE, -0.75, " 最优平稳线性滤波 = 0.512", fontsize=8.5, color="#2ca02c")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlabel("n2n_ratio  =  mse(pred, 第二次观测) / mse(第一次观测, 第二次观测)\n"
                  "< 1 才算比「不做」好；越低越好", fontsize=9.5)
    ax.set_title("S0：几乎每个格子都在恒等线（1.00）右侧——只有 mid 档一个格子 0.96 略低；"
                 "最优线性滤波是 0.512", fontsize=11)
    ax.set_xlim(0, 8.2)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    fig.tight_layout()
    out = FIG / "fig1_n2n_identity_gap.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig2() -> Path:
    """误差地板：x = 输入误差（≈噪声），y = 预测误差。恒等线、最优线性线、拟合线同图。"""
    pts = []
    for f, tag in [("reports/v3_round2/s0_low_metrics_dense.json", "low  σ=0.0198"),
                   ("reports/v3_round2/s0_vlow_metrics.json", "vlow σ=0.0066")]:
        d = _load(f)
        ct = d["controls"]["input"]
        for vol, m in sorted(ct.items()):
            x = m["rmse_vs_imaging"]
            y = d["arms"]["S_dense_s3407"]["per_volume"][vol]["rmse_vs_imaging"]
            pts.append((x, y, f"{tag} · {vol.split('_')[-1]}"))
    X = np.array([[p[0], 1.0] for p in pts])
    Y = np.array([p[1] for p in pts])
    a, b = np.linalg.lstsq(X, Y, rcond=None)[0]

    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    xs = np.linspace(0, 0.024, 100)
    ax.fill_between(xs, 0, 0.20 * xs, color="#2ca02c", alpha=0.10,
                    label="完美去噪器应在这一带（≈0.17×噪声）")
    ax.plot(xs, xs, color="0.35", ls="--", lw=1.3, label="恒等 y = x（什么都不做）")
    ax.plot(xs, a * xs + b, color="#d62728", lw=1.6,
            label=f"实测拟合 y = {a:.3f}x + {b:.4f}（斜率≈0 ⇒ 误差与噪声无关）")
    for (x, y, lab) in pts:
        ax.scatter(x, y, s=80, color="#1f77b4", zorder=3)
        ax.annotate(f"{lab}\n({x:.4f}, {y:.4f})", (x, y), textcoords="offset points",
                    xytext=(8, -12), fontsize=8, color="0.2")
    ax.set_xlabel("输入误差 rmse(input, 真值)  ≈ 噪声大小", fontsize=10)
    ax.set_ylabel("预测误差 rmse(pred, 真值)", fontsize=10)
    ax.set_title("误差地板：噪声降 3 倍，预测误差几乎不动（4 点，S_dense s3407）", fontsize=11)
    ax.set_xlim(0, 0.024)
    ax.set_ylim(0, 0.034)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    fig.tight_layout()
    out = FIG / "fig2_error_floor.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig3() -> Path:
    """训练损失 / σ²（恒等解 = 1.0）随 epoch：low 降到 0.16，vlow 停在 1.15。"""
    runs = [("runs/v3_round2_s0/low/S_dense_s3407", "low · 12 卷 · s3407", "#1f77b4"),
            ("runs/v3_round2_s0/low/S_dense_s3408", "low · 12 卷 · s3408", "#4c9be8"),
            ("runs/v3_round2_s0/low60/S_dense_s3407", "low · 60 卷 · s3407", "#9467bd"),
            ("runs/v3_round2_s0/vlow/S_dense_s3407", "vlow（噪声/3）· s3407", "#d62728"),
            ("runs/v3_round2_s0/low/D_n2n_s3407", "D_n2n（第二次观测）· s3407", "#2ca02c")]
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for path, label, color in runs:
        h = _load(path + "/history.json")
        L = np.array([rec["loss"] for rec in h])
        lv = "vlow" if "vlow" in path else "low"
        # D_n2n 的损失地板是 2σ²（目标本身带噪声），其余是 σ²
        floor = 2.0 if "D_n2n" in path else 1.0
        ax.plot([r["epoch"] for r in h], L / (SIG[lv] ** 2 * floor), color=color, lw=1.6,
                label=f"{label} → {L[-1] / (SIG[lv] ** 2 * floor):.2f}")
    ax.axhline(1.0, color="0.35", ls="--", lw=1.3)
    ax.text(1.5, 1.05, "1.0 = 恒等解的水平（损失等于噪声方差，等于没去噪）", fontsize=8.5, color="0.3")
    ax.set_yscale("log")
    ax.set_xlabel("epoch", fontsize=10)
    ax.set_ylabel("训练损失 / 噪声方差（D_n2n 除以 2σ²）", fontsize=10)
    ax.set_title("训练损失：低噪声档学得更好，高噪声档几乎没动——但两者在留出体上都比恒等差",
                 fontsize=10.5)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    fig.tight_layout()
    out = FIG / "fig3_train_loss.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig4(gpu: int = 0) -> Path:
    """一个切片：输入 / 预测 / 真值 / (预测−输入)。需要 checkpoint，会跑一次推理。"""
    from v3_expA_v2_eval import infer, write_vol
    from v3_round2_s0_eval import load_eval

    man = _load("data/processed/v3_round2_s0/low/manifest.json")
    E = load_eval(ROOT / man["eval"][0]["npz"])
    name = Path(E["npz"]).stem
    vp = write_vol(E["vol"], ROOT / "runs/_infer_v2/vols" / f"s0_{name}_in.npy")
    ck = ROOT / "runs/v3_round2_s0/low/S_dense_s3407"
    pred = infer(ck, vp, gpu, name).astype(np.float32)

    rho = E["rho"]
    z = int(np.argmax([rho[k].mean() for k in range(rho.shape[0])]))
    sl = lambda a: a[z]
    d_pred = sl(pred) - sl(E["vol"])
    vmax = np.percentile(sl(E["truth"])[E["valid"][z] | True], 99)

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.6))
    for ax, img, title, kw in [
        (axes[0], sl(E["vol"]), "输入（含噪）", dict(vmin=0, vmax=vmax, cmap="gray")),
        (axes[1], sl(pred), "网络输出", dict(vmin=0, vmax=vmax, cmap="gray")),
        (axes[2], sl(E["truth"]), "真值（无噪）", dict(vmin=0, vmax=vmax, cmap="gray")),
        (axes[3], d_pred, "输出 − 输入（放大 20×）", dict(cmap="coolwarm",
                                                     vmin=-np.abs(d_pred).max(), vmax=np.abs(d_pred).max())),
    ]:
        im = ax.imshow(img.T, origin="lower", aspect="equal", **kw)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle("S_dense s3407 @ low（切片 z=%d）：输出几乎等于输入，二者之差是一个弥散的常数型扰动\n"
                 "n2n=1.171（比不变更差），rmse/σ=1.16，r_roi=0.9947" % z, fontsize=10.5)
    fig.tight_layout()
    out = FIG / "fig4_error_map.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-map", action="store_true", help="跳过 fig4（需要 GPU）")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    made = [fig1(), fig2(), fig3()]
    if not args.no_map:
        made.append(fig4(args.gpu))
    for p in made:
        print("wrote", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
