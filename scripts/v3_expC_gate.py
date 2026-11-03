"""实验 C 的判据：T-C0（覆盖率/坐标前提）+ T-C1/C2/C3（配对方式与先验的对照）。

门槛逐字照抄 `reports/v3_expC/prereg.md` §4（写于任何 C 训练之前）。本脚本只做两件事：

1. **T-C0 用代码重算**覆盖率与坐标集合——不看训练结果，**现在就能判**；
2. 训练完成后读 `metrics.json`（由 `scripts/v3_expA_v2_eval.py --arms C_correct C_wrong_iso
   C_role_swap C_random` 产生，端点与实验 A 阶段 2 完全一致），按 §4 的三条判据逐条比。

为什么 T-C0 必须用代码而不是口头
--------------------------------
§4 的 T-C0 写着"覆盖率必须完全相同（12.5%）"。这句话里混了两个量：**单个视图**是 1/8 = 12.5%，
而每条臂用 **4 个视图**，并集是 50%（stride=1，逐块 4/8 体素）。本条与代码实测一起记进 JSON
（`per_view_frac` / `union_frac` / `prereg_number_note`），**不改任何门槛**——T-C0 的实质是
"各臂同覆盖率"，这一点由相位槽位数量直接保证，与那个绝对数字无关。

用法
----
    python -u scripts/v3_expC_gate.py                       # 只跑 T-C0（不需要任何训练结果）
    python -u scripts/v3_expC_gate.py --metrics reports/v3_expC/metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.sampling import strided_views  # noqa: E402

# ------------------------------------------------------------------ 臂（照抄 prereg §2）
ARMS = {                      # 臂名 -> 固定槽位（C_random 为 None：每步随机）
    "C_correct": [0, 1, 7, 6],
    "C_wrong_iso": [2, 7, 5, 0],
    "C_role_swap": [0, 1, 6, 7],
    "C_random": None,
}
PATCH = (32, 64, 64)         # 与 launcher 的 --patch 一致
STRIDE = 1

TEXT = {
    "T-C0_fail": "T-C0 前提不成立：各臂覆盖率/坐标集合不符 → C 不能判定，报告写『前提不成立』。",
    "T-C1_pass": "T-C1 通过：C_correct 相对 C_role_swap 同时 n2n 更低、r_roi 更高，两个体上同号，"
                 "且差值大于两臂种子极差。",
    "T-C1_fail": "T-C1 不成立：在本仿真协议与本次预算下，按实测物理先验挑选采样角色没有可测增益"
                 "（prereg §4 预写的表述），『物理先验感知采样是有效创新』降级为未验证。",
    "T-C1_partial": "T-C1 部分可判：只有 1 个种子，『差值 > 两臂种子极差』这一项无法判（prereg §5）。",
    "T-C2_pass": "T-C2 通过：C_correct 相对 C_random 满足同样条件 → 随机化不是必需的。",
    "T-C2_fail": "T-C2 不成立：先验有用但随机化已经足够 → 创新点应改述为『用先验替代随机抽样、降低成本』。",
    "T-C3_pass": "T-C3 通过：C_correct 相对 C_wrong_iso 满足同样条件（结论只写到『体对角的朝向』）。",
    "T-C3_fail": "T-C3 不成立：各向同性错先验与实测先验在本次预算下无差别。",
}


# ------------------------------------------------------------------ T-C0：代码重算
def _voxel_sets(slots: list[int], shape=PATCH, stride: int = STRIDE) -> list[set]:
    idx = torch.arange(int(np.prod(shape))).reshape(shape).float()
    return [{int(t) for t in v.flatten().tolist()} for v in strided_views(idx, slots, stride)]


def coverage_facts() -> dict:
    n_total = int(np.prod(PATCH))
    facts: dict = {"patch": list(PATCH), "stride": STRIDE, "per_arm": {}}
    for arm, slots in ARMS.items():
        if slots is None:
            facts["per_arm"][arm] = {
                "slots": None, "note": "每步随机抽 tet；覆盖率按**期望**为 4/8，逐块不同 "
                                       "（prereg §4 的『时间平均』口径）",
                "per_view_frac": 1 / 8, "union_frac": 4 / 8}
            continue
        sets = _voxel_sets(slots)
        union = set().union(*sets)
        facts["per_arm"][arm] = {
            "slots": slots,
            "per_view_frac": len(sets[0]) / n_total,
            "union_frac": len(union) / n_total,
            "coordinate_set_sha1": _sha_of(sorted(union)),
            "n_views": len(slots), "n_union": len(union),
        }
    fixed = {a: v for a, v in facts["per_arm"].items() if v["slots"] is not None}
    unions = {a: v["union_frac"] for a, v in fixed.items()}
    facts["same_union_frac"] = len(set(round(x, 12) for x in unions.values())) == 1
    facts["same_coordinate_set"] = {
        "C_correct_vs_C_role_swap": (fixed["C_correct"]["coordinate_set_sha1"]
                                     == fixed["C_role_swap"]["coordinate_set_sha1"]),
        "C_correct_vs_C_wrong_iso": (fixed["C_correct"]["coordinate_set_sha1"]
                                     == fixed["C_wrong_iso"]["coordinate_set_sha1"]),
        "expected": {"C_correct_vs_C_role_swap": True,
                     "C_correct_vs_C_wrong_iso": False,
                     "why": "role_swap 用同样 4 个相位、只对调角色；wrong_iso 是另一组 4 个相位"},
    }
    facts["random_matches_fixed_expectation"] = all(
        abs(v["union_frac"] - facts["per_arm"]["C_random"]["union_frac"]) < 1e-12
        for v in fixed.values())
    facts["prereg_number_note"] = (
        "prereg §2 写『8/64 体素 = 12.5%』，那是**单个视图**的覆盖率（1/8）；每条臂用 4 个视图，"
        "并集是 50%。T-C0 的实质（各臂同覆盖率）成立，绝对数字按此更正，不改任何门槛。")
    ok = (facts["same_union_frac"]
          and facts["same_coordinate_set"]["C_correct_vs_C_role_swap"] is True
          and facts["same_coordinate_set"]["C_correct_vs_C_wrong_iso"] is False
          and facts["random_matches_fixed_expectation"])
    facts["T-C0"] = "通过" if ok else "前提不成立"
    facts["T-C0_text"] = "T-C0 通过：三条固定臂并集覆盖率逐位相同、坐标集合符合设计" if ok else TEXT["T-C0_fail"]
    return facts


def _sha_of(seq) -> str:
    import hashlib
    return hashlib.sha1(",".join(str(x) for x in seq).encode()).hexdigest()[:16]


# ------------------------------------------------------------------ T-C1/C2/C3
def _per_arm(j: dict, seeds: list[str]) -> dict:
    """→ {arm: {vol: {'n2n': [...], 'r_roi': [...]}}}（列表按种子顺序）。"""
    out: dict = {}
    for key, rec in (j.get("arms") or {}).items():
        for arm in ARMS:
            if key.startswith(arm + "_s"):
                per = out.setdefault(arm, {})
                for vol, m in rec["per_volume"].items():
                    d = per.setdefault(vol, {"n2n": [], "r_roi": [], "seeds": []})
                    d["n2n"].append(float(m["n2n_ratio"]))
                    d["r_roi"].append(float(m["r_roi"]))
                    d["seeds"].append(key.rsplit("_s", 1)[1])
    return out


def compare(pa: dict, a: str, b: str, endpoint: str, better: str) -> dict | None:
    """a 相对 b 的差值。better='lower'（n2n）或 'higher'（r_roi）。"""
    if a not in pa or b not in pa:
        return None
    rows = {}
    for vol in sorted(set(pa[a]) & set(pa[b])):
        va, vb = np.array(pa[a][vol][endpoint]), np.array(pa[b][vol][endpoint])
        if len(va) == 0 or len(vb) == 0:
            continue
        delta = float(va.mean() - vb.mean())
        sign_ok = (delta < 0) if better == "lower" else (delta > 0)
        spread = float(max(va.max() - va.min() if len(va) > 1 else 0.0,
                           vb.max() - vb.min() if len(vb) > 1 else 0.0))
        rows[vol] = {"a_mean": float(va.mean()), "b_mean": float(vb.mean()), "delta": delta,
                     "sign_ok": sign_ok, "seed_spread": spread,
                     "beats_spread": abs(delta) > spread,
                     "n_seeds_a": len(va), "n_seeds_b": len(vb)}
    if not rows:
        return None
    return {"rows": rows,
            "all_sign_ok": all(r["sign_ok"] for r in rows.values()),
            "all_beat_spread": all(r["beats_spread"] for r in rows.values()),
            "judgeable_spread": all(r["n_seeds_a"] > 1 and r["n_seeds_b"] > 1 for r in rows.values())}


def judge(label: str, pair: tuple[str, str], pa: dict) -> dict:
    a, b = pair
    c_n2n = compare(pa, a, b, "n2n", "lower")
    c_r = compare(pa, a, b, "r_roi", "higher")
    if c_n2n is None or c_r is None:
        return {"status": "未运行", "why": f"缺 {a} 或 {b} 的结果"}
    sign_ok = c_n2n["all_sign_ok"] and c_r["all_sign_ok"]
    spread_ok = c_n2n["all_beat_spread"] and c_r["all_beat_spread"]
    judgeable = c_n2n["judgeable_spread"] and c_r["judgeable_spread"]
    # 判定顺序（写在结果之前，见 tests）：符号不符 → 不成立；只有 1 个种子 → 部分可判
    # （此时"种子极差"=0，若先判 spread 会把"不可判"误读成"通过"）；符号对且差值够大 → 通过。
    if not sign_ok:
        verdict, text = "不成立", TEXT[f"{label}_fail"]
    elif not judgeable:
        verdict, text = "部分可判", TEXT[f"{label}_partial"]
    elif spread_ok:
        verdict, text = "通过", TEXT[f"{label}_pass"]
    else:
        verdict, text = "不成立", TEXT[f"{label}_fail"]
    return {"status": "已判定", "verdict": verdict, "verdict_text": text,
            "n2n_ratio": c_n2n, "r_roi": c_r,
            "rule": "两个评估体上同时 n2n 更低 且 r_roi 更高，且差值 > 两臂种子极差（prereg §4）"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="reports/v3_expC/metrics.json")
    ap.add_argument("--out", default="reports/v3_expC/gate_verdict.json")
    args = ap.parse_args()

    rec: dict = {"meta": {"prereg": "reports/v3_expC/prereg.md §4（写于任何 C 训练之前）",
                          "arms": ARMS, "patch": list(PATCH), "stride": STRIDE}}
    rec["T-C0"] = coverage_facts()
    print(f"[T-C0] {rec['T-C0']['T-C0']}  并集覆盖率="
          f"{ {a: round(v.get('union_frac', 0), 4) for a, v in rec['T-C0']['per_arm'].items()} }")

    p = ROOT / args.metrics
    if not p.exists():
        rec["T-C1"] = rec["T-C2"] = rec["T-C3"] = {"status": "未运行", "why": f"{args.metrics} 不存在"}
    else:
        j = json.loads(p.read_text())
        pa = _per_arm(j, [])
        if rec["T-C0"]["T-C0"] != "通过":
            rec["T-C1"] = {"status": "前提不成立", "why": rec["T-C0"]["T-C0_text"]}
            rec["T-C2"] = rec["T-C3"] = {"status": "前提不成立", "why": "T-C0 未过，整轮作废（prereg §4）"}
        else:
            rec["T-C1"] = judge("T-C1", ("C_correct", "C_role_swap"), pa)
            rec["T-C2"] = judge("T-C2", ("C_correct", "C_random"), pa)
            rec["T-C3"] = judge("T-C3", ("C_correct", "C_wrong_iso"), pa)
        rec["arms_present"] = sorted(pa)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    for k in ("T-C1", "T-C2", "T-C3"):
        v = rec[k]
        print(f"[{k}] {v.get('status')} {v.get('verdict', '')}")
        if v.get("verdict_text"):
            print(f"      → {v['verdict_text']}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
