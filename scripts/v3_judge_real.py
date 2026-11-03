"""把**标定过的偏置感知判据 J1.2** 应用到真实数据的四条链上，并与旧口径（单看 n2n）并列。

## 为什么要有这个脚本

用户 2026-10-23 读图发现：SA64 上 ③ 比 ② 好、④ 比 ② 差；HSA64 上 ③ 比 ② **差**、④ 居中。
而旧口径只看 n2n，在 HSA64 上给出的是"③ 最好（0.3830）"。归因见
`reports/v3_phase5_real/HSA64_wiener_diag.json`。判据本身的作废/标定记录见
`reports/v3_judge/JUDGE_v1.md`，标定脚本 `scripts/v3_judge_calibration.py`。

本脚本**不重复任何数据准备逻辑**：全部沿用 `v3_phase5_real_network.py` 的口径
（同 SVD 带、同 `s`、同 log1p、同 z∈[0,32) 裁剪、同 `structure_masks` 常数、同 checkpoint 集）。

## 判定口径（与标定一致，不得就地改）

三道门同时过才算"这条链可接受"：`N_rel<=0.5`、`B_db<=-7.3`、`S_rel>=0.95`
（常数与推导见 `scripts/v3_judge.py`）。**本脚本不产生任何新的通过/不通过判定规则**，
只把已标定的判据跑一遍。

## 旧口径并列（用于对照，不用于取舍）

旧口径 = `n2n <= n2n(①Wiener w=7)` 且 `core 保留 >= 0.95`，即 `v3_phase5_real_network.py`
里已预注册的那条。两条口径的结论都写在 JSON 里，方便看清"换判据到底改变了哪几条链的判定"。

用法：
    python -u scripts/v3_judge_real.py --dataset SA64
    python -u scripts/v3_judge_real.py --dataset HSA64
输出：
    reports/v3_judge/{dataset}_judge.json
    reports/figures/v3_judge_real.png（两数据集合并成一张）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from prepare_svd import load_array, normalize_shape                  # noqa: E402
from v3_phase5_real_compare import (                                  # noqa: E402
    RAW, FRAC_LO, FRAC_HI, band_pd, keep_ratio, mse, setup_cjk,
    structure_masks, svd_basis, to_unit, wiener, band_slice)
from v3_judge import (                                                # noqa: E402
    JUDGE_VERSION, THR_B_DB, THR_N_REL, THR_S_REL, THR_TEXT, judge,
)

NET_Z = 32
CKPT_GLOB = "runs/v3_n2n_multi/res_s*/latest.pt"
OUT_DIR = ROOT / "reports/v3_judge"


def net_estimator(ckpt_paths):
    import torch
    from models.network import NetworkCNR

    out = []
    for ck in ckpt_paths:
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        a = blob["args"]
        if a.get("infer") != "full":
            raise SystemExit(f"{ck} 不是整卷前向的 checkpoint（infer={a.get('infer')}）")
        model = NetworkCNR(features=a["features"], groups=a["groups"])
        model.load_state_dict(blob["model"])
        model.eval()

        def fn(x, model=model, a=a):
            with torch.no_grad():
                xin = torch.as_tensor(np.ascontiguousarray(x))[None, None].float()
                h = model(xin)
                pred = (xin + h if a.get("residual") else h)[0, 0].numpy()
            return pred.astype(np.float32)

        out.append((f"④网络 {ck.parent.name}", fn))
    return out


def run_dataset(ds, frames, emitting_array, w):
    raw = normalize_shape(load_array(RAW[ds]), emitting_array, frames)
    z, x, y, t = raw.shape
    shape3d = (z, x, y)
    C = raw.reshape(-1, t)
    _, V = svd_basis(C)
    del C

    raw4 = normalize_shape(load_array(RAW[ds]), emitting_array, frames)
    halves = {}
    for tag, sl in (("odd", slice(0, None, 2)), ("even", slice(1, None, 2))):
        Ch = raw4.reshape(-1, t)[:, sl]
        th = Ch.shape[1]
        _, Vh = svd_basis(Ch)
        halves[tag] = band_pd(Ch, Vh, *band_slice(th), shape3d)
        del Ch, Vh
    del raw4, raw

    crop = (slice(0, NET_Z), slice(0, x), slice(0, y))
    for tag in halves:
        halves[tag] = halves[tag][crop]
    zc, xc, yc = halves["odd"].shape

    zz, yy, xx = np.mgrid[0:zc, 0:xc, 0:yc]
    valid = ((zz >= 2) & (zz <= zc - 3) & (yy >= 4) & (yy <= xc - 5)
             & (xx >= 4) & (xx <= yc - 5))
    o, e = halves["odd"], halves["even"]
    valid = valid & (0.5 * (o + e) > 0)

    s = float(np.median(0.5 * (o + e)[valid]))
    f = lambda pd: np.log1p(np.maximum(pd, 0.0) / s).astype(np.float32)   # noqa: E731
    xo, xe = f(o), f(e)
    masks = structure_masks(to_unit(0.5 * (xo + xe), valid))

    est = {"②单纯 SVD（不处理）": (xo, xe)}
    est[f"①Wiener w={w}"] = (wiener(xo, w, valid)[0], wiener(xe, w, valid)[0])
    for name, fn in net_estimator(sorted(ROOT.glob(CKPT_GLOB))):
        est[name] = (fn(xo), fn(xe))

    rows = {}
    for k, (p, q) in est.items():
        j = judge(p, q, xo, xe, masks, valid)
        j["core_keep"] = float(keep_ratio(p, xo, masks["core"]))
        j["dark_keep"] = float(keep_ratio(p, xo, masks["dark"]))
        j["corr_odd_even"] = float(np.corrcoef(p[valid], q[valid])[0, 1])
        rows[k] = j

    # 旧口径：n2n <= n2n(①) 且 core>=0.95
    base = f"①Wiener w={w}"
    for k, r in rows.items():
        r["old_pass"] = bool(r["n_rel"] <= rows[base]["n_rel"] and r["core_keep"] >= 0.95)
    meta = {"dataset": ds, "grid": list(halves["odd"].shape), "norm_s": s,
            "valid_voxels": int(valid.sum()),
            "mask_voxels": {k: int(v.sum()) for k, v in masks.items()}}
    return rows, base, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="SA64", choices=sorted(RAW))
    ap.add_argument("--frames", type=int, default=-1)
    ap.add_argument("--emitting-array", type=int, default=0)
    ap.add_argument("--w", type=int, default=7)
    ap.add_argument("--out-dir", default="reports/v3_judge")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, base, meta = run_dataset(args.dataset, args.frames, args.emitting_array, args.w)

    print(f"\n[{args.dataset}] 判据 {JUDGE_VERSION}：{THR_TEXT}")
    print(f"  {'链':28s} {'N_rel':>7s} {'B_db':>7s} {'S_rel':>7s} "
          f"{'旧口径':>6s} {'新判据':>8s}  未过")
    for k, r in rows.items():
        bad = ",".join(t for t, o in (("N", r["ok_n"]), ("B", r["ok_b"]),
                                      ("S", r["ok_s"])) if not o) or "全过"
        print(f"  {k:28s} {r['n_rel']:7.4f} {r['b_db']:7.2f} {r['s_rel']:7.4f} "
              f"{'过' if r['old_pass'] else '不过':>6s} "
              f"{'过' if r['pass'] else '不过':>8s}  {bad}")

    rec = {"judge_version": JUDGE_VERSION, "thresholds": THR_TEXT,
           "thr": {"n_rel": THR_N_REL, "b_db": THR_B_DB, "s_rel": THR_S_REL},
           "reference_chain": base, "rows": rows, "meta": meta,
           "note": ("三道门均为必要条件；全过不等于'去噪变好'。判据来源与标定见 "
                    "reports/v3_judge/JUDGE_v1.md 与 scripts/v3_judge_calibration.py。")}
    (OUT_DIR / f"{args.dataset}_judge.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=1))
    print(f"[{args.dataset}] 记盘 reports/v3_judge/{args.dataset}_judge.json")


if __name__ == "__main__":
    main()
