"""实验 A 的两道门（T1 / T2）的**机械化判定**。门槛数字一律照抄预注册，不经本脚本改动。

为什么要有这个脚本
------------------
预注册 `reports/v3_expA_v2/prereg.md` §1.3 / §3.3 已经把门槛写在训练结果之前。人工"照着念"有两个
风险：① 结果出来之后无意识地挑有利的聚合方式；② 事后改数字而没人能发现。本脚本把三件事固化成代码：

1. 门槛与判读文字是**字面量**，逐字抄自预注册（`THR` / `TEXT`），改门槛必须改这个文件、留下 diff；
2. **聚合方式写死并说明**（预注册没写，本脚本在结果之前确定，取保守的"逐格全过"为主判据）；
3. 记录**脚本 mtime 与结果文件 mtime 的先后**，用于核对"判定逻辑早于结果"（`predates` 字段）。

用法
----
    python -u scripts/v3_expA_v2_gate.py \
        --stage1 reports/v3_expA_v2/stage1_eval/expA_metrics.json \
        --stage2 reports/v3_expA_v2/metrics.json \
        --history 4k=runs/v3_expA/expA_metrics.json \
        --history 24k=runs/v3_expA_long/expA_metrics.json \
        --history 200k=reports/v3_expA_v2/stage1_eval/expA_metrics.json

不存在的输入一律记为 **未运行**（不猜测、不用别的实验的数字顶替）。

判定纪律
--------
* 阶段 1（T1）只用 `r_ves` 与 `n2n_ratio`：v1 真值退化，其他端点按 R06 §2.1 已证伪。
* 阶段 2（T2）只用 `n2n_ratio` 与 `r_roi`；`zero` 控制若不"输给输入"，T2 的失败**不得**归因于方法
  （预注册 §3.3 失败可归因规则），本脚本把它算成显式检查项 `zero_control_not_degenerate`。
* 落进"未定"就写"未定"，不改门槛、不挑格子。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------- 门槛（字面量）
# 逐字抄自 reports/v3_expA_v2/prereg.md（写入时间 2026-10-27 00:16，早于任何训练结果）。
THR = {
    "T1_input_r_ves": 0.9456,     # §1.3：输入本身的 r_ves，写在 00:16
    "T1_n2n_lt": 1.0,             # §1.3：n2n_ratio < 1.0
    "T2_n2n_lt": 1.0,             # §3.3：n2n_ratio < 1.0
    "T2_r_roi_gt_input": True,    # §3.3：r_roi > 输入本身的 r_roi
    "zero_margin": 0.0,           # §3.3 归因规则的操作化：zero 不得在任何端点 ≥ 输入
}

TEXT = {
    "T1_pass": "T1-通过（预算不足可以解释实验 A）：两条臂的 r_ves 都 > 输入（0.9456）且 n2n_ratio < 1.0"
               " → 实验 A 的负面结论降级为'预算不足'。",
    "T1_fail": "T1-未通过（预算不是解释）：两条臂的 r_ves 都 < 输入且 n2n_ratio > 1.0"
               " → 预算不是解释；实验 A 的结论在 50 倍预算下保持，仍需要阶段 2。",
    "T1_undet": "T1-未定（混合或种子间不一致）：逐格列出，不挑选有利的一格下结论。",
    "T2_pass": "T2-通过（任务定义问题已消除，faithful VALID 可继续）：A_rand 与 A_fixed4 都满足 "
               "n2n_ratio < 1.0 且 r_roi > 输入的 r_roi。",
    "T2_fail": "T2-未通过：两条臂 n2n_ratio 都 > 1.0 或 r_roi 都 < 输入 → faithful VALID 在该仿真上"
               "不成立，实验 B/C/D 继续冻结；结论适用范围写成'Kong 等 2024 的自监督去噪教义 + 本项目仿真协议'。",
    "T2_undet": "T2-未定：不满足通过也不满足未通过的组合，逐格列出。",
    "T2_unattributable": "T2 的失败**不可归因于方法**：zero 对照在至少一个端点上不输给输入"
                         "（预注册 §3.3 第二段的失败可归因规则），只能写'判据仍退化，不能判方法'。",
}

# 聚合方式（预注册未写 → 在结果之前在此固定，取保守口径）
AGG_NOTE = ("预注册只写'两条臂都满足'，未写跨种子/跨体的聚合方式。本脚本在结果产生之前固定："
            "主判据 `strict` = 每条臂的**每一个** (seed × 评估体) 格子都满足门槛；"
            "同时并列报告 `arm_mean` = 每条臂先对体求均值、再对种子求均值后是否满足。"
            "两者不一致时，报告以 strict 为结论、把 arm_mean 标为'口径敏感性'。")


def _sha1(p: Path) -> str:
    return hashlib.sha1(p.read_bytes()).hexdigest()


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p.resolve())


def _load(rel_path: str, key: str, rec: dict) -> dict | None:
    p = ROOT / rel_path
    if not p.exists():
        rec[key] = {"status": "未运行", "path": rel_path,
                    "why": "文件不存在（不猜测、不用别的实验的数字顶替）"}
        return None
    try:
        j = json.loads(p.read_text())
    except Exception as e:                                   # noqa: BLE001
        rec[key] = {"status": "损坏", "path": rel_path, "error": f"{type(e).__name__}: {e}"}
        return None
    return j


# ---------------------------------------------------------------------------- T1
def _t1_cells(j: dict) -> dict:
    """→ {arm: {(seed, 'eval'): {r_ves, n2n_ratio}}}；阶段 1 只有单个评估体，故一格一 seed。"""
    out: dict = {}
    for arm, per_seed in (j.get("arms") or {}).items():
        for seed, m in per_seed.items():
            out.setdefault(arm, {})[(seed, "eval")] = {
                "r_ves": float(m.get("r_ves")), "n2n_ratio": float(m.get("n2n_ratio")),
                "bg_std_ratio": m.get("bg_std_ratio")}
    return out


def check_t1(j: dict) -> dict:
    base = j.get("baseline", {}).get("input", {}).get("mean", {})
    b_rves = float(base.get("r_ves", float("nan")))
    cells = _t1_cells(j)
    if not cells:
        return {"status": "无法判定", "why": "JSON 里没有 arms"}
    per_arm = {}
    for arm, cc in cells.items():
        strict = all(c["r_ves"] > THR["T1_input_r_ves"] and c["n2n_ratio"] < THR["T1_n2n_lt"]
                     for c in cc.values())
        both_bad = all(c["r_ves"] < THR["T1_input_r_ves"] and c["n2n_ratio"] > THR["T1_n2n_lt"]
                       for c in cc.values())
        per_arm[arm] = {
            "strict_pass": strict, "strict_fail": both_bad,
            "r_ves_mean": sum(c["r_ves"] for c in cc.values()) / len(cc),
            "n2n_mean": sum(c["n2n_ratio"] for c in cc.values()) / len(cc),
            "cells": {f"{s}|{v}": c for (s, v), c in cc.items()},
        }
    pass_ = all(a["strict_pass"] for a in per_arm.values())
    fail_ = all(a["strict_fail"] for a in per_arm.values())
    verdict = "T1-通过" if pass_ else ("T1-未通过" if fail_ else "T1-未定")
    return {"status": "已判定", "verdict": verdict,
            "verdict_text": TEXT[{"T1-通过": "T1_pass", "T1-未通过": "T1_fail"}.get(verdict, "T1_undet")],
            "baseline_input_r_ves": b_rves,
            "threshold_literal_check": {
                "input_r_ves_matches_prereg": abs(b_rves - THR["T1_input_r_ves"]) < 5e-4,
                "note": "预注册写 0.9456；此处核对 JSON 里的输入值与之是否一致（不一致要报出来，不改门槛）"},
            "per_arm": per_arm, "aggregation": AGG_NOTE}


# ---------------------------------------------------------------------------- T2
def _t2_cells(j: dict) -> dict:
    """→ {arm: {(seed, volume): metrics}}，从 `A_rand_s3407` 这类键里拆出 arm 与 seed。"""
    out: dict = {}
    for key, rec in (j.get("arms") or {}).items():
        m = re.match(r"^(?P<arm>.+)_s(?P<seed>\d+)$", key)
        if not m or "per_volume" not in rec:
            continue
        arm, seed = m.group("arm"), m.group("seed")
        for vol, met in rec["per_volume"].items():
            out.setdefault(arm, {})[(seed, vol)] = {
                "n2n_ratio": float(met["n2n_ratio"]), "r_roi": float(met["r_roi"]),
                "r_core": met.get("r_core"), "bg_std_ratio": met.get("bg_std_ratio")}
    return out


def _input_ref(j: dict) -> dict:
    """输入本身在每体上的 r_roi / n2n（n2n 恒 1.0，但仍从数据里读，不假设）。"""
    inp = (j.get("controls") or {}).get("input") or {}
    return {vol: {"n2n_ratio": float(m["n2n_ratio"]), "r_roi": float(m["r_roi"])}
            for vol, m in inp.items()}


def zero_control_still_degenerate(j: dict) -> dict:
    """预注册 §3.3 的失败可归因规则：zero 若在任何主端点上 ≥ 输入，判据仍退化。"""
    ctrl = j.get("controls") or {}
    z, inp = ctrl.get("zero") or {}, ctrl.get("input") or {}
    rows, bad = {}, []
    for vol in sorted(set(z) & set(inp)):
        zr = {"n2n_ratio": float(z[vol]["n2n_ratio"]), "r_roi": float(z[vol]["r_roi"])}
        ir = {"n2n_ratio": float(inp[vol]["n2n_ratio"]), "r_roi": float(inp[vol]["r_roi"])}
        # nan 处理（写在此处、结果之前）：全零输出的 r_roi 是 0/0，nan 不等于"赢过输入"，不能静默跳过。
        # 逐项记 nan_treated_as="not_beating"，让它出现在 JSON 里可供核对。
        nan_keys = [k for k in ("n2n_ratio", "r_roi") if zr[k] != zr[k] or ir[k] != ir[k]]
        beats = {"n2n_ratio": zr["n2n_ratio"] <= ir["n2n_ratio"] + THR["zero_margin"],
                 "r_roi": zr["r_roi"] >= ir["r_roi"] - THR["zero_margin"]}
        rows[vol] = {"zero": zr, "input": ir, "zero_beats_or_ties_input": beats,
                     "nan_keys": nan_keys,
                     "nan_treated_as": "not_beating（nan 来自常数输出的 0/0 相关，不构成「赢过输入」）"
                     if nan_keys else None}
        if any(beats.values()):
            bad.append(f"{vol}:{','.join(k for k, v in beats.items() if v)}")
    return {"status": "已检查" if rows else "未运行",
            "zero_still_degenerate": bool(bad), "offending": bad, "rows": rows,
            "rule": "任一评估体上 zero 的 n2n_ratio<=输入 或 r_roi>=输入 ⇒ 判据仍退化 ⇒ 失败不可归因于方法"}


def check_t2(j: dict) -> dict:
    cells = _t2_cells(j)
    inp = _input_ref(j)
    if not cells or not inp:
        return {"status": "无法判定", "why": "缺 arms 或 controls.input"}
    per_arm = {}
    for arm, cc in cells.items():
        ok = all(c["n2n_ratio"] < THR["T2_n2n_lt"] and c["r_roi"] > inp[v]["r_roi"] + THR["zero_margin"]
                 for (s, v), c in cc.items())
        bad = all(c["n2n_ratio"] > THR["T2_n2n_lt"] or c["r_roi"] < inp[v]["r_roi"] - THR["zero_margin"]
                  for (s, v), c in cc.items())
        by_vol = {}
        for (s, v), c in cc.items():
            by_vol.setdefault(v, []).append(c)
        mean_ok = all(
            (sum(c["n2n_ratio"] for c in cs) / len(cs) < THR["T2_n2n_lt"]
             and sum(c["r_roi"] for c in cs) / len(cs) > inp[v]["r_roi"] + THR["zero_margin"])
            for v, cs in by_vol.items())
        per_arm[arm] = {
            "strict_pass": ok, "strict_fail": bad, "arm_mean_pass": mean_ok,
            "n2n_mean": sum(c["n2n_ratio"] for c in cc.values()) / len(cc),
            "r_roi_mean": sum(c["r_roi"] for c in cc.values()) / len(cc),
            "cells": {f"{s}|{v}": c for (s, v), c in cc.items()},
        }
    pass_ = all(a["strict_pass"] for a in per_arm.values())
    fail_ = all(a["strict_fail"] for a in per_arm.values())
    verdict = "T2-通过" if pass_ else ("T2-未通过" if fail_ else "T2-未定")
    zc = zero_control_still_degenerate(j)
    out = {"status": "已判定", "verdict": verdict,
           "verdict_text": TEXT[{"T2-通过": "T2_pass", "T2-未通过": "T2_fail"}.get(verdict, "T2_undet")],
           "input_ref": inp, "per_arm": per_arm, "aggregation": AGG_NOTE,
           "zero_control": zc}
    if verdict == "T2-未通过" and zc.get("zero_still_degenerate"):
        out["attribution"] = TEXT["T2_unattributable"]
    elif verdict == "T2-未通过":
        out["attribution"] = "zero 对照不退化 ⇒ 本次失败可以归因于方法（在预注册的预算与判据下）。"
    return out


# ---------------------------------------------------------------------------- 单调性（机制判读，不是门槛）
def _mean_rves(j: dict, arm: str) -> float | None:
    """兼容两种 JSON：阶段 1 的 summary{arm}{endpoint}{mean}，以及 arms{arm}{seed}。"""
    s = (j.get("summary") or {}).get(arm, {}).get("r_ves")
    if isinstance(s, dict) and "mean" in s:
        return float(s["mean"])
    per_seed = (j.get("arms") or {}).get(arm)
    if isinstance(per_seed, dict) and per_seed:
        vals = [float(m["r_ves"]) for m in per_seed.values() if "r_ves" in m]
        return sum(vals) / len(vals) if vals else None
    return None


def monotonicity(points: list[dict], rec: dict) -> dict:
    rows = []
    for pt in points:
        if pt["json"] is None:
            rows.append({"label": pt["label"], "status": "未运行", "path": pt["path"]})
            continue
        j = pt["json"]
        row = {"label": pt["label"], "status": "已读", "path": pt["path"],
               "arms": {a: _mean_rves(j, a) for a in ("A_rand", "A_fixed4")}}
        row["arms"] = {k: v for k, v in row["arms"].items() if v is not None}
        if pt.get("loss") is not None:
            row["loss"] = pt["loss"]
        rows.append(row)
    have = [r for r in rows if r["status"] == "已读" and r["arms"]]
    trend = {}
    if len(have) >= 2:
        for arm in have[0]["arms"]:
            seq = [(r["label"], r["arms"][arm]) for r in have if arm in r["arms"]]
            if len(seq) >= 2:
                trend[arm] = {"sequence": seq, "decreasing": seq[-1][1] < seq[0][1],
                              "note": "r_ves 随预算是否下降（下降=网络在正确优化错误的目标）"}
    rec["monotonicity"] = {"role": "机制判读，不改门槛（预注册 §1.3 附加检查）",
                           "rows": rows, "trend": trend}
    return rec["monotonicity"]


def parse_loss(log_path: Path) -> dict | None:
    """从训练日志里取最后一个 `epoch=NNN loss=X`；取不到就返回 None（不猜）。"""
    if not log_path.exists():
        return None
    last = None
    for line in log_path.read_text(errors="ignore").splitlines():
        m = re.search(r"epoch=(\d+) loss=([0-9.eE+-]+)", line)
        if m:
            last = {"epoch": int(m.group(1)), "loss": float(m.group(2))}
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default="reports/v3_expA_v2/stage1_eval/expA_metrics.json")
    ap.add_argument("--stage2", default="reports/v3_expA_v2/metrics.json")
    ap.add_argument("--history", action="append", default=[],
                    metavar="LABEL=JSON", help="单调性检查用的历史预算点，可重复")
    ap.add_argument("--loss-log", action="append", default=[],
                    metavar="LABEL=LOG", help="可选：对应预算点的训练日志（取最后一个 epoch 的 loss）")
    ap.add_argument("--out", default="reports/v3_expA_v2/gate_verdict.json")
    args = ap.parse_args()

    self_path = Path(__file__).resolve()
    rec: dict = {
        "meta": {
            "prereg": "reports/v3_expA_v2/prereg.md §1.3（T1）/ §3.3（T2），写入于 2026-10-27 00:16",
            "thresholds_literal": THR,
            "aggregation_note": AGG_NOTE,
            "gate_script": _rel(self_path),
            "gate_script_sha1": _sha1(self_path),
            "gate_script_mtime": self_path.stat().st_mtime,
            "predates": {},   # 判定逻辑是否早于结果（脚本 mtime < 结果 mtime ⇒ True）
        },
    }

    for key, path in (("stage1", args.stage1), ("stage2", args.stage2)):
        p = ROOT / path
        rec["meta"]["predates"][key] = (None if not p.exists()
                                        else bool(self_path.stat().st_mtime < p.stat().st_mtime))

    j1 = _load(args.stage1, "stage1_raw", rec)
    rec["T1"] = check_t1(j1) if j1 else rec.pop("stage1_raw")
    if j1:
        rec.pop("stage1_raw", None)

    j2 = _load(args.stage2, "stage2_raw", rec)
    rec["T2"] = check_t2(j2) if j2 else rec.pop("stage2_raw")
    if j2:
        rec.pop("stage2_raw", None)

    losses = {}
    for item in args.loss_log:
        lab, _, lp = item.partition("=")
        losses[lab] = parse_loss(ROOT / lp)
    pts = []
    for item in args.history:
        lab, _, hp = item.partition("=")
        p = ROOT / hp
        try:
            jj = json.loads(p.read_text()) if p.exists() else None
        except Exception:                                    # noqa: BLE001
            jj = None
        pts.append({"label": lab, "path": hp, "json": jj, "loss": losses.get(lab)})
    monotonicity(pts, rec)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, ensure_ascii=False))

    for k in ("T1", "T2"):
        v = rec[k]
        print(f"[{k}] {v.get('status')}  {v.get('verdict', '')}")
        if v.get("verdict_text"):
            print(f"      → {v['verdict_text']}")
        for arm, a in (v.get("per_arm") or {}).items():
            print(f"      {arm:10s} n2n_mean={a['n2n_mean']:.4f} "
                  f"r={'%.4f' % a['r_roi_mean'] if 'r_roi_mean' in a else '%.4f' % a['r_ves_mean']} "
                  f"strict_pass={a['strict_pass']} strict_fail={a['strict_fail']}")
    if rec.get("T2", {}).get("zero_control", {}).get("status") == "已检查":
        zc = rec["T2"]["zero_control"]
        print(f"[zero] 仍退化={zc['zero_still_degenerate']} 触发={zc['offending']}")
    mt = rec.get("monotonicity", {}).get("trend", {})
    if mt:
        print("[mono] " + "; ".join(f"{a}: {t['sequence']}" for a, t in mt.items()))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
