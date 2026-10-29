"""实验 B 的四条门槛（T-B1…T-B4）的机械化判定。

门槛逐字照抄 `reports/v3_expB/prereg.md` §4（写于任何 B 训练之前）。本脚本不改任何门槛，只做两件事：
把 §4 的措辞翻译成**可复算**的规则，并把聚合口径在结果之前固定下来。

固定下来的聚合口径（预注册没写细节的部分）
------------------------------------------
* 端点均值：先在**每个评估体内对各 seed 求均值**，再对评估体求均值（体之间不加权，两体等重）。
* T-B1 的"种子极差"：`max(两臂各自、两个体上的 (max_seed − min_seed))`——取**最保守**的那个极差，
  即最难通过的口径。
* T-B1 的"同号"：两个评估体上差值的符号一致（不要求差值大小相近）。
* T-B4 的"最高/最低"：三表示严格排序，**并列不算**"最高/最低"（无胜者）。

用法
----
    python -u scripts/v3_expB_gate.py --metrics reports/v3_expB/metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPS = ("linear", "log1p", "logeps")
PRIMARY = "n2n_ratio_lin"
FIDELITY = "r_roi"          # eval 脚本里 `r_roi_log1p` 写进 JSON 的键名就是 r_roi
MECH = {"mass_roi": "res_mass_frac_roi", "spearman": "spearman_abs_res_vs_input"}
KEYMAP = {PRIMARY: "n2n", FIDELITY: "r_roi"}     # 端点名 -> load_arms 的内部键
LOWEST3 = 3                 # T-B4/H2 的"最暗三个十分位"


def load_arms(j: dict) -> dict:
    """→ {rep: {vol: {'n2n': [...], 'r_roi': [...], 'mass': [...], 'spread': [...], 'low3': [...]}}"""
    out: dict = {}
    for key, rec in (j.get("arms") or {}).items():
        for rep in REPS:
            if not key.startswith(f"B_{rep}_s"):
                continue
            per = out.setdefault(rep, {})
            for vol, m in (rec.get("per_volume") or {}).items():
                d = per.setdefault(vol, {k: [] for k in ("n2n", "r_roi", "mass", "spread", "low3")})
                d["n2n"].append(float(m[PRIMARY]))
                d["r_roi"].append(float(m[FIDELITY]))
                d["mass"].append(float(m[MECH["mass_roi"]]))
                d["spread"].append(float(m[MECH["spearman"]]))
                share = (m.get("residual_by_decile") or {}).get("share")
                d["low3"].append(float(sum(share[:LOWEST3])) if share else float("nan"))
    return out


def _mean_or_none(xs) -> float | None:
    xs = [x for x in xs if x is not None and x == x]
    return float(np.mean(xs)) if xs else None


def _seed_spread(xs) -> float:
    xs = [x for x in xs if x == x]
    return float(max(xs) - min(xs)) if len(xs) > 1 else 0.0


def _arm_means(arms: dict, rep: str, key: str) -> dict:
    """每个评估体的均值；缺体/缺臂返回 {}。"""
    if rep not in arms:
        return {}
    return {v: _mean_or_none(d[key]) for v, d in arms[rep].items()}


def tb1(arms: dict) -> dict:
    """两两比较在两条主端点上的可分辨性。"""
    rows = {}
    for i, a in enumerate(REPS):
        for b in REPS[i + 1:]:
            entry = {}
            for key, better in ((PRIMARY, "lower"), (FIDELITY, "higher")):
                ik = KEYMAP[key]
                ma, mb = _arm_means(arms, a, ik), _arm_means(arms, b, ik)
                vols = sorted(set(ma) & set(mb))
                if not vols or any(ma[v] is None or mb[v] is None for v in vols):
                    entry[key] = {"status": "未运行"}
                    continue
                spread = max(
                    [_seed_spread(arms[a][v][ik]) for v in vols if v in arms.get(a, {})]
                    + [_seed_spread(arms[b][v][ik]) for v in vols if v in arms.get(b, {})])
                deltas = {v: ma[v] - mb[v] for v in vols}
                signs = {v: (d < 0 if better == "lower" else d > 0) for v, d in deltas.items()}
                same_sign = len(set(signs.values())) == 1
                beats = {v: abs(deltas[v]) > spread for v in vols}
                entry[key] = {
                    "status": "已判定", "deltas": deltas, "same_sign": same_sign,
                    "a_better_on_all_volumes": all(signs.values()) if same_sign else False,
                    "seed_spread": spread, "beats_spread_on_all_volumes": all(beats.values()),
                    "separable": bool(same_sign and all(beats.values())),
                    "better": better,
                }
            ent = [entry[k] for k in (PRIMARY, FIDELITY) if entry[k].get("status") == "已判定"]
            entry["pair_separable"] = bool(ent) and all(e["separable"] for e in ent)
            rows[f"{a}_vs_{b}"] = entry
    if not rows:
        return {"status": "未运行", "why": "没有可读的臂结果"}
    return {"status": "已判定", "pairs": rows,
            "rule": "两个体上同号 且 |差值| > max(两臂各自、两个体的种子极差) ⇒ 该端点可分辨"}


def tb2(arms: dict) -> dict:
    """胜者必须同时拿下 n2n 最低与 r_roi 最高。"""
    n2n = {r: {v: _mean_or_none(d["n2n"]) for v, d in arms.get(r, {}).items()} for r in REPS}
    rr = {r: {v: _mean_or_none(d["r_roi"]) for v, d in arms.get(r, {}).items()} for r in REPS}
    have = [r for r in REPS if n2n[r] and rr[r]]
    if len(have) < 2:
        return {"status": "未运行", "why": f"只有 {have} 有结果"}
    mean_n2n = {r: _mean_or_none(list(n2n[r].values())) for r in have}
    mean_rr = {r: _mean_or_none(list(rr[r].values())) for r in have}
    best_n2n = min(have, key=lambda r: mean_n2n[r])
    best_rr = max(have, key=lambda r: mean_rr[r])
    # 严格性：并列不算胜者
    tie_n2n = sum(1 for r in have if abs(mean_n2n[r] - mean_n2n[best_n2n]) < 1e-12) > 1
    tie_rr = sum(1 for r in have if abs(mean_rr[r] - mean_rr[best_rr]) < 1e-12) > 1
    if tie_n2n or tie_rr:
        return {"status": "已判定", "winner": None, "no_winner_because": "端点存在并列，不判胜者",
                "mean_n2n_ratio_lin": mean_n2n, "mean_r_roi": mean_rr}
    ok = best_n2n == best_rr
    return {"status": "已判定", "winner": best_rr if ok else None,
            "best_n2n_ratio_lin": best_n2n, "best_r_roi": best_rr,
            "mean_n2n_ratio_lin": mean_n2n, "mean_r_roi": mean_rr,
            "verdict": ("胜者：" + best_rr) if ok else "无胜者（表示在噪声与保真之间存在取舍，不能据此选定）"}


def _avg_all(arms: dict, rep: str, key: str) -> float | None:
    """一条臂在所有 (评估体 × 种子) 格子上的均值；体内先对种子求均值（体等重）。"""
    return _mean_or_none([_mean_or_none(d[key]) for d in arms.get(rep, {}).values()])


def tb3(arms: dict) -> dict:
    """三个表示的 n2n_ratio_lin 都 ≥ 1.0 ⇒ 表示不是瓶颈（不许用它救实验 A）。"""
    have = {r: v for r in REPS if (v := _avg_all(arms, r, "n2n")) is not None}
    if not have:
        return {"status": "未运行"}
    all_ge = all(v >= 1.0 for v in have.values())
    return {"status": "已判定", "mean_n2n_ratio_lin": have, "all_ge_1": all_ge,
            "verdict": ("表示不是瓶颈——实验 A 的结论不得被『换个表示就好了』解释掉"
                        if all_ge else "至少一个表示压到了输入以下（n2n_lin < 1.0）")}


def tb4(arms: dict) -> dict:
    """机制侧的三条方向性预测（可被证伪）。"""
    mass = {r: _avg_all(arms, r, "mass") for r in REPS}
    low3 = {r: _avg_all(arms, r, "low3") for r in REPS}
    spr = {r: _avg_all(arms, r, "spread") for r in REPS}
    have = [r for r in REPS if mass[r] is not None]
    if len(have) < 2:
        return {"status": "未运行"}

    def strictly(x: dict, rep: str, mode: str) -> bool:
        others = [x[r] for r in REPS if r != rep and x.get(r) is not None]
        if not others or x[rep] is None:
            return False
        return all(x[rep] > o for o in others) if mode == "max" else all(x[rep] < o for o in others)

    rows = {
        "H1_linear_highest_res_mass_roi": {"values": mass, "holds": strictly(mass, "linear", "max")},
        "H2_linear_lowest_darkest3_share": {"values": low3, "holds": strictly(low3, "linear", "min")},
        "H3_linear_largest_spearman": {"values": spr, "holds": strictly(spr, "linear", "max")},
    }
    return {"status": "已判定", "hypotheses": rows,
            "note": "任一不成立就照实记录，不得事后改方向重讲一遍故事（prereg §4）",
            "all_hold": all(v["holds"] for v in rows.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="reports/v3_expB/metrics.json")
    ap.add_argument("--out", default="reports/v3_expB/gate_verdict.json")
    args = ap.parse_args()

    rec: dict = {"meta": {
        "prereg": "reports/v3_expB/prereg.md §4（写于任何 B 训练之前）",
        "primary": PRIMARY, "fidelity": FIDELITY, "reps": REPS,
        "aggregation": "体内先对 seed 求均值、再对体求均值（体等重）；"
                       "T-B1 的种子极差取两臂×两体的最大值（最保守）；并列不算最高/最低",
        "do_not_compare_with_A": "B 的 n2n_ratio_lin 与 A 的 n2n_ratio（log1p 域）不是同一个量，不可跨实验比大小",
    }}
    p = ROOT / args.metrics
    if not p.exists():
        for k in ("T-B1", "T-B2", "T-B3", "T-B4"):
            rec[k] = {"status": "未运行", "why": f"{args.metrics} 不存在"}
    else:
        j = json.loads(p.read_text())
        arms = load_arms(j)
        rec["arms_present"] = sorted(arms)
        rec["T-B1"] = tb1(arms)
        rec["T-B2"] = tb2(arms)
        rec["T-B3"] = tb3(arms)
        rec["T-B4"] = tb4(arms)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    for k in ("T-B1", "T-B2", "T-B3", "T-B4"):
        print(f"[{k}] {rec[k].get('status')}  {rec[k].get('verdict', '')}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
