"""实验 C / R03 负控制：由**不同的物理先验**推出各自"最干净"的采样角色，再落成可训练的四槽位臂。

为什么先算这个（而不是直接开训）
--------------------------------
实验 C 的问题是"用**对的**物理先验比用**错的**先验强吗"。要回答它，必须先看清一件更基本的事：
**错的先验会不会指向不同的采样方案**。如果换成各向同性 ρ 或轴错位的 ρ，推出来的最优槽位
和实测 ρ 推出来的**一样**，那这个负控制就没有区分力——跑一遍只会得到"两条臂一样好"，
而这既不能证明先验有用，也不能证明它没用。这种"没有信息量的负实验"必须先算出来、写进预注册，
不能等跑完再解释。

用的模型与 `v3_role_order.py` 完全一致（不新造模型）
-----------------------------------------------------
    L = c^T C^{-1} c，输入 = 该角色在 ±w cell 内的全部位点（w=3），m=2，
    ρ = 可分离非参数（逐轴 lag 1..6 线性插值后相乘）或各向同性解析式。

四个先验变体
------------
1. `measured`        实测 ρ（`v3_correlation_length.json`，去平台快速分量）—— 正确先验；
2. `iso`             ρ(δ)=exp(−|δ|/ℓ)，ℓ = 实测三轴 ℓ 的几何平均 —— 去掉**各向异性**；
3. `axis_permuted`   三轴曲线循环错位 (z,y,x)→(y,x,z) —— 轴序错；
4. `strongest_fixed` 不做模型，直接取 `measured` 下 L 最小的那个 tet 固定使用。

产物：`data/processed/v3_expC/prior_arms.json`（含每个变体下 8 个 tet × 36 种槽位组合的排序摘要、
选中槽位、以及与实测先验选中槽位是否相同）。

用法
----
    python -u scripts/v3_expC_prior_arms.py            # 两个序列都算
    python -u scripts/v3_expC_prior_arms.py --seq SA64
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from datasets.sampling import check_cross_slots, TETRIS_PATTERNS  # noqa: E402
from v3_leak_four_role import leak_L  # noqa: E402
from v3_leak_nonparametric import ell_from_lag1, fast_curve, rho_sep  # noqa: E402
from v3_precheck_sampling_geometry import off, PATS  # noqa: E402

M = 2
W = 3
CELLS = np.array(list(itertools.product(range(-W, W + 1), repeat=3)), float)
CORR = ROOT / "data/processed/v3_correlation_length.json"
OUT = ROOT / "data/processed/v3_expC/prior_arms.json"


def sites(roles: list[int]) -> np.ndarray:
    """与 `v3_role_order.sites_of` 同一实现（4 个角色 × 343 个 cell 的绝对坐标）。"""
    return np.concatenate([M * (2.0 * CELLS + off(a)) for a in roles], 0)


def curves_of(J: dict, seq: str) -> dict:
    P1 = J[seq]["summary"]["P1"]
    raw = {a: [v[0] for v in J[seq]["P1"]["acf"][a]] for a in ("z", "y", "x")}
    return {a: list(fast_curve(raw[a], P1[a]["plateau"])) for a in ("z", "y", "x")}


def prior_fns(J: dict, seq: str) -> dict:
    """四个先验变体，全部是 rho(displacement) -> correlation 的可调用对象。"""
    cur = curves_of(J, seq)
    ell = ell_from_lag1(cur)
    ell_iso = float(np.exp(np.mean(np.log(np.asarray(ell, float)))))
    perm = {"z": cur["y"], "y": cur["x"], "x": cur["z"]}

    def _iso(d):
        return np.exp(-np.linalg.norm(np.asarray(d, float), axis=-1) / ell_iso)

    return {
        "measured": {"fn": lambda d: rho_sep(d, cur),
                     "note": "分离非参数 ρ（实测，去平台）", "ell_zyx": list(ell)},
        "iso": {"fn": _iso,
                "note": f"各向同性 exp(-|δ|/ℓ)，ℓ={ell_iso:.3f}（实测 ℓ 的几何平均）",
                "ell_zyx": [ell_iso] * 3},
        "axis_permuted": {"fn": lambda d: rho_sep(d, perm),
                          "note": "三轴曲线循环错位 (z,y,x)->(y,x,z)",
                          "ell_zyx": [ell[1], ell[2], ell[0]]},
    }


def pair_L(rho) -> dict:
    """全部 8 个 tet × 4 输入角色 × 4 个互补目标角色的 L。键 = (p, a, y)。"""
    out = {}
    for p, pat in enumerate(PATS):
        src = [int(x) for x in pat]
        tgt = sorted(set(range(8)) - set(src))
        for a in src:
            S = sites([a])
            for y in tgt:
                L, _ = leak_L(S, M * off(y), rho)
                out[(p, a, y)] = L
    return out


def best_slots(Ls: dict, p: int) -> dict:
    """在一个 tet 内枚举 (2 个输入 × 2 个目标) 的 36 种组合，按**四种配对组合的均值 L** 选最优。"""
    src = [int(x) for x in PATS[p]]
    tgt = sorted(set(range(8)) - set(src))
    rows = []
    for s_in in itertools.combinations(src, 2):
        for t_gt in itertools.combinations(tgt, 2):
            matched = 0.5 * (Ls[(p, s_in[0], t_gt[0])] + Ls[(p, s_in[1], t_gt[1])])
            swapped = 0.5 * (Ls[(p, s_in[0], t_gt[1])] + Ls[(p, s_in[1], t_gt[0])])
            # 训练时按顺序配对（xa→ya, xb→yb），所以槽位顺序必须写成**好的那种配对**，
            # 否则算出来的最优与真正跑起来的臂不是一回事（这是本脚本第一版的一个真 bug）。
            slots = ([int(s_in[0]), int(s_in[1]), int(t_gt[0]), int(t_gt[1])]
                     if matched <= swapped else
                     [int(s_in[0]), int(s_in[1]), int(t_gt[1]), int(t_gt[0])])
            rows.append({"slots": slots, "L_matched": matched, "L_swapped": swapped,
                         "L_best_pairing": min(matched, swapped)})
    rows.sort(key=lambda r: r["L_best_pairing"])
    return {"tet": p, "best": rows[0], "top5": rows[:5], "n_combos": len(rows)}


def summarise(Ls: dict) -> dict:
    by_shift = {}
    for (p, a, y), L in Ls.items():
        sh = tuple(int(round(v)) for v in (off(a) - off(y)))
        by_shift.setdefault(sh, []).append(L)
    classes = {str(k): {"mean_L": float(np.mean(v)), "ptp": float(np.ptp(v)), "n": len(v)}
               for k, v in by_shift.items()}
    ranked = sorted(classes.items(), key=lambda kv: kv[1]["mean_L"])
    return {"n_pairs": len(Ls), "shift_classes": classes,
            "cleanest_shift": ranked[0][0], "cleanest_L": ranked[0][1]["mean_L"],
            "noisiest_shift": ranked[-1][0], "noisiest_L": ranked[-1][1]["mean_L"],
            "L_ratio": ranked[-1][1]["mean_L"] / max(ranked[0][1]["mean_L"], 1e-12)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", nargs="+", default=["SA64", "HSA64"])
    args = ap.parse_args()
    J = json.loads(CORR.read_text())
    rec = {"experiment": "C / R03 负控制：先验变体 → 采样槽位", "m": M, "w": W,
           "leak_model": "L = c^T C^{-1} c（与 v3_role_order.py 同一实现）", "seq": {}}

    for seq in args.seq:
        print(f"\n=== {seq}")
        variants = prior_fns(J, seq)
        per = {}
        for name, spec in variants.items():
            Ls = pair_L(spec["fn"])
            s = summarise(Ls)
            cand = [best_slots(Ls, p) for p in range(8)]
            cand.sort(key=lambda c: c["best"]["L_best_pairing"])
            top = cand[0]
            # 覆盖率/坐标一致性：同一批槽位只能从同一组相位里选，槽位合法由 datasets 校验
            check_cross_slots(top["best"]["slots"])
            per[name] = {"note": spec["note"], "ell_zyx": spec["ell_zyx"], **s,
                         "best_tet": top["tet"], "best_slots": top["best"]["slots"],
                         "best_L_pairing": top["best"]["L_best_pairing"],
                         "top3_tets": [{"tet": c["tet"], "slots": c["best"]["slots"],
                                        "L": c["best"]["L_best_pairing"]} for c in cand[:3]]}
            print(f"  {name:14s} 最干净位移={s['cleanest_shift']} (L={s['cleanest_L']:.4f})  "
                  f"最差={s['noisiest_shift']} (L={s['noisiest_L']:.4f}, 比 {s['L_ratio']:.1f}×)  "
                  f"-> tet{top['tet']} slots={top['best']['slots']}")

        meas = per["measured"]["best_slots"]
        for name in ("iso", "axis_permuted"):
            per[name]["differs_from_measured"] = per[name]["best_slots"] != meas
        # 最强固定模式臂：不随机抽 tet，固定 measured 下最好的 tet
        per["strongest_fixed"] = {
            "note": "不枚举，直接固定 measured 先验下 L 最小的 tet",
            "tet": per["measured"]["best_tet"], "slots": meas,
            "differs_from_measured": False,
        }
        rec["seq"][seq] = {"variants": per,
                           "discriminating": any(per[n]["differs_from_measured"]
                                                 for n in ("iso", "axis_permuted"))}
        print(f"  先验变体之间是否指向**不同**槽位: {rec['seq'][seq]['discriminating']}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
