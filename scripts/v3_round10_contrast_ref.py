#!/usr/bin/env python3
"""Round10 补充诊断：给参照臂补 `contrast` 读数，并落 `overfit_diag.json`。

为什么单独写这一个脚本，而不是改 `v3_round10_eval.py`：

* 预注册 §6 的次要判据要把**训练后模型**的 `slope` / `core ratio` 与**高斯**比
  （"比高斯低 0.05 以上"）。但 `v3_round10_eval.references()` 只给参照臂存了
  `x / x_gain / alpha`，没有 `contrast` 字段 —— 缺了高斯的读数，§6 就无法逐字执行。
* 补这个字段最直接的做法当然是在 `references()` 里加一行，但那会让 `--what grid`
  对**所有** run 目录重跑一遍 dense 推理（28+ 个 GPU 作业的全部代价），
  只为拿几个只依赖 CPU 的参照读数。所以这里独立成脚本：
  `masked_gauss` / `wiener` / SN2N 输出全是确定性、无训练的，重算几秒钟即可。

同时把 `scores.json` 里每个 run 已经算好的 `overfit` 块抽出来，落成预注册 §7 要求的
`reports/v3_round10/overfit_diag.json`（此前只在 scores.json 内联，没有独立文件）。

**本脚本不产生任何判定**：§4 的判定只看 `x` 的 `alpha=1` 值，与 contrast 无关。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v3_round10_eval as E  # noqa: E402
from v3_round4_b1_prep import fwd  # noqa: E402
from v3_round4_metrics import masked_gauss  # noqa: E402

OUT = Path("reports/v3_round10")


def ref_contrast(seq: str, direction: str) -> dict:
    """重算方向 `direction` 下各参照臂的 `contrast` 块（确定性，无训练）。"""
    th = direction.split("2")[0]
    pd_tr, valid, s = E.load_half(seq, th)
    lg = fwd("log1p", pd_tr, s, valid)
    rec: dict = {}
    for sig in E.SIGMAS:
        pd_g = np.maximum(np.expm1(masked_gauss(lg, valid, sig)), 0.0) * s
        rec[f"gauss_{sig:g}"] = E.contrast_block(pd_g, pd_tr, valid, s)
    pd_w = np.maximum(np.expm1(E.scipy_wiener(lg, mysize=3)), 0.0) * s
    rec["wiener3d"] = E.contrast_block(pd_w, pd_tr, valid, s)
    p_sn = E.RUNS / f"sn2n_{seq}_{th}" / "pd_sn2n_linear.npy"
    if p_sn.exists():
        rec["sn2n"] = E.contrast_block(np.load(p_sn).astype(np.float64), pd_tr, valid, s)
    return rec


def main() -> int:
    scores = json.loads((OUT / "scores.json").read_text())
    selections = json.loads((OUT / "selections.json").read_text())

    refs = {}
    for seq in ("SA64", "HSA64"):
        for direction in ("odd2even", "even2odd"):
            refs[f"{seq}_{direction}"] = ref_contrast(seq, direction)
            print(f"  contrast 参照 {seq} {direction} ok")

    # 每个 run 的过拟合诊断（scores.json 里已算好，这里只做搬运）
    runs = {k: v["overfit"] for k, v in scores["runs"].items() if "overfit" in v}
    bad = [k for k, v in runs.items() if not v.get("identity_ok", False)]
    if bad:
        # 预注册 §5：自检不过就是实现错误，不是结果 —— 直接停下，不写产物。
        print("恒等式自检失败：", bad, file=sys.stderr)
        return 4

    # §6 要用的对照：方向 2 上，选中配置 vs 高斯（选中 σ 取自方向 1）
    focus = {}
    for seq in ("SA64", "HSA64"):
        sig = scores["references"][f"{seq}_odd2even"]["gauss"]["best_by_x"]
        for arm in ("A", "B"):
            b = selections[seq]["arms"][arm]
            key = f"{seq}_{arm}_{b['domain']}_e{b['epochs']}_even2odd"
            if key not in scores["runs"]:
                focus[f"{seq}_{arm}"] = {"note": "方向 2 尚未完成"}
                continue
            c = scores["runs"][key]["contrast"]
            g = refs[f"{seq}_even2odd"][f"gauss_{float(sig):g}"]
            focus[f"{seq}_{arm}"] = {
                "run": key, "sigma": float(sig),
                "slope": c["slope"], "slope_gauss": g["slope"],
                "d_slope": c["slope"] - g["slope"],
                "core_ratio": c["core_ratio"], "core_ratio_gauss": g["core_ratio"],
                "d_core": c["core_ratio"] - g["core_ratio"],
                "worse_than_gauss_by_0_05": bool(
                    c["slope"] < g["slope"] - 0.05 or c["core_ratio"] < g["core_ratio"] - 0.05),
            }
            print(f"  §6 {seq} {arm}: d_slope={focus[f'{seq}_{arm}']['d_slope']:+.3f} "
                  f"d_core={focus[f'{seq}_{arm}']['d_core']:+.3f}")

    doc = {
        "note": "预注册 §5 恒等式自检 + §6 次要判据的对照读数；不参与 §4 判定。",
        "identity_all_ok": True,
        "runs": runs,
        "reference_contrast": refs,
        "section6_focus": focus,
    }
    (OUT / "overfit_diag.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    print("saved", OUT / "overfit_diag.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
