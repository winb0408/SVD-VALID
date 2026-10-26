"""V3 Phase 2 / T2-1：在 m=2 下**预先冻结**输入—目标角色序（不训练、不看结果）。

为什么必须冻结：m=2 的单视图泄漏区间是 0.017–0.536（SA64）/ 0.036–0.468（HSA64），
最差角色仍过半。如果训练完再挑"哪个角色好"，角色就成了拟合参数，
H2（可预测采样）和 I1（物理相关性感知采样）都会被自己证伪。
所以本脚本只用**实测 ρ**（去平台快速分量）算出全部有序角色对的泄漏，排序、冻结、写盘。

做法：对每个模式 p（4 个角点），对每个 (输入角点 a, 目标角点 y) 的 4×4=16 种有序对，
算 `L = c^T C^{-1} c`（输入 = a 在 ±3 cell 内的全部位点）。8 个模式共 128 对。
判据只看 (a, y) 的相对几何，所以 8 个模式的结果应当**按几何类重复出现**——
如果它们不重复，说明有位序错误（这正是本脚本的自检）。

输出 `data/processed/v3_role_order_m2.json`。

用法：
    python -u scripts/v3_role_order.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v3_leak_four_role import leak_L, sites_of  # noqa: E402
from v3_precheck_sampling_geometry import off, PATS  # noqa: E402
from v3_leak_nonparametric import fast_curve, rho_sep  # noqa: E402

OUT = Path("data/processed/v3_role_order_m2.json")
CORR = Path("data/processed/v3_correlation_length.json")
M = 2
W = 3


def curves_of(J, seq):
    P1 = J[seq]["summary"]["P1"]
    raw = {a: [v[0] for v in J[seq]["P1"]["acf"][a]] for a in ("z", "y", "x")}
    return {a: list(fast_curve(raw[a], P1[a]["plateau"])) for a in ("z", "y", "x")}


def main():
    J = json.loads(CORR.read_text())
    rec = {"m": M, "w": W, "note": "预注册角色序；L = 实测 ρ（去平台快速分量）下的噪声泄漏上界", "seq": {}}
    for seq in J:
        cur = curves_of(J, seq)
        rho = lambda d, c=cur: rho_sep(d, c)  # noqa: E731
        rows = []
        for p in range(8):
            src = [int(x) for x in PATS[p]]
            tgt = sorted(set(range(8)) - set(src))
            for a in src:
                S = sites_of([a], M, W)
                for y in tgt:
                    L, _ = leak_L(S, M * off(y), rho)
                    sh = tuple(int(round(v)) for v in (off(a) - off(y)))
                    rows.append({"p": p, "a": a, "y": y, "L": L,
                                 "shift": sh,
                                 "d2": float(np.sum((off(y) - off(a)) ** 2))})
        rows.sort(key=lambda r: r["L"])
        # 自检：L 只该依赖 (a,y) 的**有符号格点位移**（ρ 各向异性，所以不能只按平方位移归类）。
        # 同一位移向量下，不同模式给出的 L 必须逐位相同；不等于说明位序或索引写错了。
        classes = {}
        for r in rows:
            classes.setdefault(r["shift"], []).append(r["L"])
        spread = {k: float(np.ptp(v)) for k, v in classes.items()}
        rec["seq"][seq] = {"rows": rows, "class_shift": {str(k): v for k, v in classes.items()},
                           "class_spread": {str(k): v for k, v in spread.items()}}

        print(f"\n=== {seq}  m={M}  {len(rows)} 个有序角色对（按 L 升序，取每类第一例）")
        print("  位移(a→y，格点)   体素位移      L     该类对数")
        seen = set()
        for r in rows:
            if r["shift"] in seen:
                continue
            seen.add(r["shift"])
            print(f"  {r['shift']}      {tuple(int(v) * M for v in r['shift'])}   "
                  f"{r['L']:.3f}   n={len(classes[r['shift']])}")
        Ls = np.array([r["L"] for r in rows])
        print(f"  分布：min {Ls.min():.3f}  中位 {np.median(Ls):.3f}  mean {Ls.mean():.3f}  max {Ls.max():.3f}")
        print("  按位移类自检（同类内不同模式的极差，应全为 0）：")
        for k in sorted(spread, key=lambda k: -spread[k]):
            print(f"    {k}: n={len(classes[k]):>3}  极差 {spread[k]:.2e}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
