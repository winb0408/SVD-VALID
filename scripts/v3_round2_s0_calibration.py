"""S0f：评分正反标定 —— 用**已知改动**验证判据有没有分辨力。

审查 §3 S0f 的原话：用 clean、identity、zero、blur、过锐化、局部断血管、注入假血管等
已知改动验证评分；"全零输"只证明排除一种平凡解，**不保证全部评分具有血管检出有效性**。

本脚本把七种已知改动喂给同一套端点，检查的**顺序关系写死在本文件里**（写在跑之前）：
每一行 `CHECKS` 都是一条必须在数据上成立的序关系。任何一条不成立 → 判据在该端点上
对本类错误不敏感，报告里必须写明，而不是当作"小题大做"。

用法：
    python -u scripts/v3_round2_s0_calibration.py --level low
产物：
    reports/v3_round2/s0_calibration_<level>.json
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

from v3_round2_s0_eval import load_eval, metrics_s0   # noqa: E402
from v3_round2_s0_dataset import PSF_PD              # noqa: E402

# 预先写在跑之前的顺序关系：(端点, 更好的一侧, 更差的一侧, 为什么必须成立)
CHECKS = [
    ("n2n_ratio", "clean", "input", "完美答案必须严格优于不做处理"),
    ("n2n_ratio", "input", "zero", "全零这种平凡解必须被重罚"),
    ("n2n_ratio", "input", "blur", "过度模糊必须被扣分（否则判据奖励抹平）"),
    ("n2n_ratio", "input", "sharp", "过锐化放大噪声必须被扣分"),
    ("n2n_ratio", "input", "broken", "断掉一段血管必须被扣分"),
    ("n2n_ratio", "input", "fake", "凭空注入一条血管必须被扣分"),
    ("r_roi", "clean", "input", "完美答案的结构相关必须高于不做处理"),
    ("r_roi", "input", "blur", "模糊会降低结构相关"),
    ("r_roi", "input", "broken", "断血管会降低结构相关"),
    ("r_roi", "input", "fake", "假血管会降低结构相关"),
    # 2026-10-27 追加（**在跑出结果之前**加，理由见下）：上一条 `r_roi` 对假血管无效，
    # 因为 r_roi 只在 ρ>0 内算，而假血管被注入到 ρ=0 的背景——掩膜里根本没有那个体素。
    # 这是**端点定义**的问题，不是门槛问题：改在同一位置上用全域相关 `r_all` 再查一次。
    ("r_all", "input", "fake", "假血管在 ROI 之外，只有全域相关才看得见"),
]


def corruptions(E: dict, seed: int = 0) -> dict:
    x, valid, rho = E["vol"], E["valid"], E["rho"]
    rng = np.random.default_rng(seed)
    out = {"clean": E["truth"].copy(), "input": x.copy(), "zero": np.zeros_like(x),
           "blur": gaussian_filter(x, sigma=PSF_PD, mode="nearest")}
    smooth = gaussian_filter(x, sigma=PSF_PD, mode="nearest")
    out["sharp"] = (x + 1.5 * (x - smooth)).astype(np.float32)

    # 断血管：找 ρ≥0.5 里最亮的一段（z 中间 8 层），把它拉回背景中位数
    core = valid & (rho >= 0.5)
    broken = x.copy()
    if core.any():
        zs = np.where(core.any(axis=(1, 2)))[0]
        z0 = int(zs[len(zs) // 2])
        seg = core[z0 - 4:z0 + 4]
        broken[z0 - 4:z0 + 4][seg] = float(np.median(x[E["bg"]]))
    out["broken"] = broken

    # 假血管：在背景区（远离真血管）画一条直线管
    fake = x.copy()
    bgvox = np.argwhere(E["bg"])
    if len(bgvox) > 0:
        c = bgvox[rng.integers(len(bgvox))]
        amp = float(np.quantile(x[core], 0.9)) if core.any() else float(np.max(x))
        zz, yy, xx = np.mgrid[0:x.shape[0], 0:x.shape[1], 0:x.shape[2]]
        d = np.sqrt((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2)
        fake = np.where(d <= 1.2, amp, fake).astype(np.float32)
    out["fake"] = fake
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="low")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    man_p = Path(args.manifest or f"data/processed/v3_round2_s0/{args.level}/manifest.json")
    man = json.loads((ROOT / man_p).read_text())
    Es = [load_eval(ROOT / e["npz"]) for e in man["eval"]]

    rec = {"meta": {"level": args.level, "manifest": str(man_p),
                    "prereg": "CHECKS 写在本脚本里，运行前确定（审查 §3 S0f）"},
           "per_volume": {}, "checks": []}
    for E in Es:
        name = Path(E["npz"]).stem
        rec["per_volume"][name] = {tag: metrics_s0(v.astype(np.float32), E)
                                   for tag, v in corruptions(E).items()}

    for ep, better, worse, why in CHECKS:
        rows = []
        for name, m in rec["per_volume"].items():
            a, b = m[better][ep], m[worse][ep]
            ok = (a < b) if ep == "n2n_ratio" else (a > b)
            rows.append({"volume": name, "better": a, "worse": b, "ok": bool(ok)})
        rec["checks"].append({"endpoint": ep, "better": better, "worse": worse,
                              "why": why, "all_ok": all(r["ok"] for r in rows), "rows": rows})

    out = ROOT / (args.out or f"reports/v3_round2/s0_calibration_{args.level}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print(f"wrote {out}")
    for c in rec["checks"]:
        print(f"  [{'OK ' if c['all_ok'] else 'FAIL'}] {c['endpoint']}: "
              f"{c['better']} 应优于 {c['worse']}  —  {c['why']}")
    n_fail = sum(1 for c in rec["checks"] if not c["all_ok"])
    print(f"未通过 {n_fail}/{len(rec['checks'])} 条")


if __name__ == "__main__":
    main()
