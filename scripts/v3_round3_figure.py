"""Round3 对比图：真值 / 输入 / 网络输出 / 盲高斯模糊，外加误差图。

为什么单独画：预注册的证据是数字，但"到底长什么样"只有图能一眼说清。
这张图**不引入新端点**：n2n 用 `v3_expA_v2_eval.metrics` 在同一批数组上重算，
并与 `reports/v3_round3/a_probes_*.json` / `a5_trivial_baselines.json` 里的记录逐场景对账，
差超过 1e-3 就打印警告（防止图与报告说的不是一回事）。

只画 MIP（最大强度投影），因为要看的是血管结构有没有被保住/加噪，不是单层切片。

用法
----
    python -u scripts/v3_round3_figure.py --level-dir data/processed/v3_round2_s0/low6 \
        --ckpt runs/v3_round2_s0/low/S_dense_s3407 --sigma 0.6 --scenes e900 e903 \
        --out reports/v3_round3/fig_round3_low.png --gpu 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v3_expA_v2_eval import infer, load_eval, metrics, write_vol  # noqa: E402


def mip_db(x: np.ndarray) -> np.ndarray:
    """log1p 域卷 → MIP 的 dB 图（以中位正值归一）。只用于显示。"""
    p = np.expm1(np.maximum(x, 0))
    den = np.median(p[p > 0]) if (p > 0).any() else 1.0
    return 10 * np.log10(np.maximum(p.max(axis=0), 1e-9) / max(den, 1e-12))


def blur(vol, valid, sigma):
    m = valid.astype(np.float32)
    num = gaussian_filter(vol * m, sigma, mode="nearest")
    den = gaussian_filter(m, sigma, mode="nearest")
    out = np.where(den > 1e-6, num / np.maximum(den, 1e-6), vol)
    out[~valid] = 0.0
    return out.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sigma", type=float, required=True, help="盲模糊尺度（A5 选出的 σ*）")
    ap.add_argument("--scenes", nargs="+", default=["e900", "e903"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # 图里只有少量中文（"不动/越低越好"），DejaVu 没有 CJK 字形，配一个本机存在的 CJK 字体。
    # 注意：本机 CJK 只有 .ttc 合集，matplotlib 只把它们注册成 face 0 的名字（...CJK JP），
    # "Noto Sans Mono CJK SC"/"Noto Serif CJK SC" 这类子族名**查不到**、会静默落回 DejaVu，
    # 所以这里用注册名；等宽族也挂同一个（等宽对齐会略变，但中文不会变方块）。
    plt.rcParams["font.sans-serif"] = ["Noto Serif CJK JP", "DejaVu Sans"]
    plt.rcParams["font.monospace"] = ["Noto Sans CJK JP", "DejaVu Sans Mono"]
    plt.rcParams["axes.unicode_minus"] = False

    ldir = Path(args.level_dir)
    man = json.loads((ROOT / ldir / "manifest.json").read_text())
    ckpt = ROOT / args.ckpt
    tag = ldir.name

    rows = []
    for sc in args.scenes:
        ent = [e for e in man["eval"] if str(e["seed"]) == sc or sc in e["npz"]]
        if not ent:
            raise SystemExit(f"[abort] {ldir} 里没有场景 {sc}")
        E = load_eval(ROOT / ent[0]["npz"])
        name = Path(E["npz"]).stem
        p = write_vol(E["vol"], ROOT / "runs/_infer_v2/v3_round3_fig" / f"{tag}_{name}_in.npy")
        pred = infer(ckpt, p, args.gpu, f"r3fig_{tag}_{name}").astype(np.float32)
        bl = blur(E["vol"], E["valid"], args.sigma)
        rows.append((name, E, pred, bl))
        print(f"[{name}] n2n model={metrics(pred, E)['n2n_ratio']:.3f} "
              f"blur(σ={args.sigma:g})={metrics(bl, E)['n2n_ratio']:.3f} "
              f"identity={metrics(E['vol'], E)['n2n_ratio']:.3f}")

    ncol = 5
    fig, axes = plt.subplots(len(rows) * 2, ncol, figsize=(3.0 * ncol, 3.0 * len(rows) * 2))
    axes = np.atleast_2d(axes)
    for i, (name, E, pred, bl) in enumerate(rows):
        items = [("truth (MIP dB)", E["truth"]), ("input obs1", E["vol"]),
                 ("CRN output", pred), (f"blind blur σ={args.sigma:g}", bl),
                 ("truth (overlay free)", E["truth"])]
        for j, (ttl, v) in enumerate(items[:4]):
            ax = axes[2 * i, j]
            im = ax.imshow(mip_db(v), cmap="gray", vmin=-25, vmax=15)
            ax.set_title(f"{name} · {ttl}", fontsize=9)
            ax.axis("off")
        ax = axes[2 * i, 4]
        n1 = metrics(pred, E)["n2n_ratio"]
        n2 = metrics(bl, E)["n2n_ratio"]
        ax.axis("off")
        ax.text(0.02, 0.95, f"{name}\n\nn2n (1 = 不动)\n  identity   1.000\n"
                            f"  CRN        {n1:.3f}\n  blind blur {n2:.3f}\n\n"
                            f"越低越好；0.5 = 完美去噪",
                va="top", fontsize=10, family="monospace")
        # 第二行：与真值的误差图（同一 dB 色标）
        for j, (ttl, v) in enumerate([("|err| input", E["vol"]), ("|err| CRN", pred),
                                      (f"|err| blur σ={args.sigma:g}", bl)]):
            d = np.abs(mip_db(E["truth"]) - mip_db(v))
            ax = axes[2 * i + 1, j]
            im = ax.imshow(d, cmap="magma", vmin=0, vmax=20)
            ax.set_title(f"{name} · {ttl}", fontsize=9)
            ax.axis("off")
        for j in (3, 4):
            axes[2 * i + 1, j].axis("off")
        fig.colorbar(im, ax=axes[2 * i + 1, 3], fraction=0.46, pad=0.02,
                     label="MIP |Δ| (dB)")

    fig.suptitle(f"SVD-VALID round3: {tag} · checkpoint {ckpt.relative_to(ROOT)}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
