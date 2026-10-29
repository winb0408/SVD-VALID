"""实验 B / R04：为"三种表示"对照生成**共享归一化尺度**下的训练体。

审查意见（R04）要求：保持 sampler/net/loss 不变，仅比较 `linear` / `log1p` / `log(PD+ε)`，
验证条件残差与图像质量，并且**固定共享归一化尺度**。

为什么必须重做数据而不是复用 `*_log1p.npy`
--------------------------------------------
现有的 `data/processed/v3_sim_v2/train/*_log1p.npy` 是 `log1p(pd / s_体)`，`s_体` 是**每个体各自**的
中位数（见 `v3_make_sim_dataset.py` 的说明）。逐体中位数会让"同一个表示"在不同体上对应不同的
物理强度：血管亮度的绝对值被一起归一掉了。三种表示如果都这样归一，比较出来的差异里就混着
"每种变换对逐体缩放的敏感度"，而不是"变换本身"。所以这里改成：

    s_glob = median(所有训练体、所有 valid 体素的 pd_linear 的**合并样本**)

一个数、所有体共用、也用于评估体。s_glob 及其来源 hash 落在 `scale.json` 里，不参与任何训练。

三种表示（同一个 s_glob、同一个 ε）
-----------------------------------
    linear : x = pd / s_glob
    log1p  : x = log1p(pd / s_glob)
    logeps : x = log(pd / s_glob + eps),   eps = 1e-3（预注册，事先定死，不调）

`logeps` 的 ε 与 `log1p` 在 `pd > 0` 时几乎只差一个常数平移（∝ log 与 log1p 的差异在 pd/s ≪ 1
时最大），所以这一臂主要回答"PD 允许为 0 时用哪种对数化"，而不是"加一个 ε 能否救回对比度"。

产物
----
    data/processed/v3_sim_v2/repB/scale.json
    data/processed/v3_sim_v2/repB/{linear,log1p,logeps}_train_s{seed}.npy

用法
----
    python scripts/v3_expB_make_representations.py            # 全部 16 个体 × 3 表示
    python scripts/v3_expB_make_representations.py --check    # 只校验，不写盘
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

MANIFEST = ROOT / "data/processed/v3_sim_v2/manifest.json"
OUTDIR = ROOT / "data/processed/v3_sim_v2/repB"
EPS = 1e-3                      # 预注册，写死
REPS = ("linear", "log1p", "logeps")


def sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def transform(pd: np.ndarray, s: float, rep: str) -> np.ndarray:
    z = pd.astype(np.float64) / s
    if rep == "linear":
        return z.astype(np.float32)
    if rep == "log1p":
        return np.log1p(z).astype(np.float32)
    if rep == "logeps":
        return np.log(z + EPS).astype(np.float32)
    raise ValueError(rep)


def inverse(x: np.ndarray, s: float, rep: str) -> np.ndarray:
    """回到**线性 PD / s_glob** 的单位：跨表示比较 n2n 必须在同一域里做。"""
    x = x.astype(np.float64)
    if rep == "linear":
        return x
    if rep == "log1p":
        return np.expm1(x)
    if rep == "logeps":
        return np.exp(x) - EPS
    raise ValueError(rep)


def load_train() -> list[dict]:
    man = json.loads(MANIFEST.read_text())
    return man["train"]


def pooled_scale() -> tuple[float, dict]:
    """s_glob = 合并样本中位数；同时记录逐体中位数的散布，作为"为什么不能用逐体"的证据。"""
    vals = []
    per_vol = {}
    for rec in load_train():
        d = np.load(ROOT / rec["npz"])
        pd, valid = d["pd_linear"], d["valid"].astype(bool)
        vals.append(pd[valid].astype(np.float64))
        per_vol[rec["seed"]] = float(np.median(pd[valid]))
    pooled = np.concatenate(vals)
    s_glob = float(np.median(pooled))
    spread = np.array(list(per_vol.values()))
    stats = {
        "s_glob": s_glob,
        "s_glob_definition": "median(concat over train volumes of pd_linear[valid])",
        "per_volume_median": per_vol,
        "per_volume_spread_rel": float(spread.max() / spread.min() - 1.0),
        "per_volume_cv": float(spread.std() / spread.mean()),
        "n_voxels_pooled": int(pooled.size),
        "pd_percentiles_of_pooled_over_s": {
            f"p{q}": float(np.percentile(pooled, q) / s_glob) for q in (1, 25, 50, 75, 99, 99.9)
        },
        "pd_max_over_s": float(pooled.max() / s_glob),
    }
    return s_glob, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只校验已有产物与 scale.json 一致")
    args = ap.parse_args()

    s_glob, stats = pooled_scale()
    if args.check:
        sc_path = OUTDIR / "scale.json"
        if not sc_path.exists():
            print("missing scale.json")
            return 1
        old = json.loads(sc_path.read_text())
        ok = abs(old["s_glob"] - s_glob) < 1e-12 and old.get("eps") == EPS
        bad = [rec["seed"] for rec in load_train()
               if not (OUTDIR / f"log1p_train_s{rec['seed']}.npy").exists()]
        print(f"s_glob {'OK' if ok else 'CHANGED'}  ({s_glob:.6g})   missing_files={bad[:5]}")
        return 0 if ok and not bad else 1

    OUTDIR.mkdir(parents=True, exist_ok=True)
    recs = []
    for rec in load_train():
        d = np.load(ROOT / rec["npz"])
        pd, valid = d["pd_linear"], d["valid"].astype(bool)
        entry = {"seed": rec["seed"], "npz": rec["npz"], "npz_sha1": sha1(ROOT / rec["npz"]),
                 "valid_coverage": float(valid.mean())}
        for rep in REPS:
            x = transform(pd, s_glob, rep)
            # 往返一致性：反变换回到线性域的最大绝对误差（相对 s_glob 归一）
            rt = np.abs(inverse(x, s_glob, rep) - pd / s_glob)
            np.save(OUTDIR / f"{rep}_train_s{rec['seed']}.npy", x)
            v = x[valid]
            entry[rep] = {
                "file": str((OUTDIR / f"{rep}_train_s{rec['seed']}.npy").relative_to(ROOT)),
                "min": float(v.min()), "max": float(v.max()),
                "mean": float(v.mean()), "std": float(v.std()),
                "rt_maxabs_err_rel": float(rt[valid].max()),
            }
        recs.append(entry)

    scale = {
        "experiment": "B / R04：三种表示对照，共享归一化尺度",
        "eps": EPS,
        "representations": {
            "linear": "pd / s_glob",
            "log1p": "log1p(pd / s_glob)",
            "logeps": "log(pd / s_glob + 1e-3)",
        },
        "inverse_note": "跨表示比较 n2n 时先反变换回线性域：linear=x, log1p=expm1(x), logeps=exp(x)-eps",
        **stats,
        "volumes": recs,
        "prereg": "reports/v3_expB/prereg.md（门槛写在结果之前）",
    }
    (OUTDIR / "scale.json").write_text(json.dumps(scale, ensure_ascii=False, indent=1))
    print(f"s_glob={s_glob:.6g}  eps={EPS}  per-volume median spread={stats['per_volume_spread_rel']:.3f}")
    print(f"pooled pd/s percentiles=" + ", ".join(
        f"{k}={v:.2f}" for k, v in stats["pd_percentiles_of_pooled_over_s"].items()))
    print(f"max pd/s = {stats['pd_max_over_s']:.2f}")
    for rep in REPS:
        rt = max(v[rep]["rt_maxabs_err_rel"] for v in recs)
        rng = (min(v[rep]["min"] for v in recs), max(v[rep]["max"] for v in recs))
        print(f"  {rep:7s} range=[{rng[0]:+.3f}, {rng[1]:+.3f}]  roundtrip maxabs err={rt:.2e}")
    print(f"wrote {OUTDIR.relative_to(ROOT)}/scale.json + {len(recs)*3} 个体文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
