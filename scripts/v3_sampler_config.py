"""R03：把"物理先验采样器"的全部输入**冻结成一份可校验的配置**（`configs/sampler_config.json`）。

审查意见（R03）原文要求："物理先验端到端接入 sampler"，缺的是 ①`sampler_config.json` 冻结、
②错误先验的负控制。本脚本负责 ①，并把 ② 的**规格**写进配置（`negative_controls`，状态为
`spec_only_not_run`——它是实验 C 的臂，实验 A 的门通过之前不许开跑，见
`reports/v3_expA_v2/prereg.md` §3.3）。

为什么"冻结 + 可校验"而不是写一段注释
--------------------------------------
采样器的行为由**六类外部输入**决定，其中四类是磁盘上的实测产物：

  1. 几何：m（步距）、w（泄漏窗口半径）、8 个 tet 角点表与互补表；
  2. 物理先验 ρ：`data/processed/v3_correlation_length.json`（实测散斑自相关，去平台快速分量）；
  3. 角色序：`data/processed/v3_role_order_m2.json`（由 ρ 推出的有序角色对泄漏 L 排序）；
  4. 频带：官方 1-based `M(:,60:350)` = 291 模式 = 0-based `[59,350)`；
  5. 成像常数：PSF σ 与热噪声 σ（先验的"物理"部分，也用它们做 oracle 对照）；
  6. 有效区掩膜：真实数据是官方 `valid_mask.npy`（含扇形成像足迹），仿真是"前 2 层死区"。

任何一项被改动（换数据、重算 ρ、改 m），采样器就变成了**另一个**方法，而这个改变在日志里
看不出来。所以本脚本把每一项连同它的 **sha1** 写进配置，并提供 `--check` 在每次训练前校验；
`train.py --sampler-config` 校验失败会直接退出，不写半截 checkpoint。

用法
----
    python scripts/v3_sampler_config.py --build      # 生成/刷新 configs/sampler_config.json
    python scripts/v3_sampler_config.py --check      # 校验（不一致 -> 退出码 1）
    python scripts/v3_sampler_config.py --dump       # 打印当前配置

产物：`configs/sampler_config.json`
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from datasets.sampling import COMPLEMENT_PATTERN, TETRIS_PATTERNS  # noqa: E402
from v3_simulator import SimCfg  # noqa: E402

CONFIG = ROOT / "configs" / "sampler_config.json"
CONFIG_VERSION = "R03-v1"

CORR = ROOT / "data/processed/v3_correlation_length.json"
ROLE_ORDER = ROOT / "data/processed/v3_role_order_m2.json"
VALID_SA64 = ROOT / "data/processed/official/SA64/valid_mask.npy"
VALID_HSA64 = ROOT / "data/processed/official/HSA64/valid_mask.npy"

M = 2          # 采样步距（块 = (2m)^3 = 4^3，8 个角点）
W = 3          # 泄漏窗口半径（cell），与 v3_role_order.py 一致
SHAPE = (37, 64, 64)
BAND_OFFICIAL = "M(:,60:350)"
BAND_MODES = 291
BAND_REPO = [59, 350]


def sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def array_sha1(a: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()


def _rel(path: Path) -> str:
    """仓库内给相对路径，仓库外（例如测试里的临时文件）给绝对路径。"""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)



def usable_prefix(n: int, m: int) -> int:
    """`v3_physics_sampler.usable` 的口径：能被 2m 整除的最大前缀（37 -> 36）。"""
    return (n // (2 * m)) * (2 * m)


def valid_mask_facts() -> dict:
    """官方有效区 vs 仿真有效区的**实际差异**（R13 valid_mask 项）。"""
    sa = np.load(VALID_SA64).astype(bool)
    hs = np.load(VALID_HSA64).astype(bool)
    sim = np.ones(SHAPE, bool)
    sim[:2] = False  # 与 v3_simulator.simulate 的 rule 完全一致
    inter = np.logical_and(sa, sim).sum()
    union = np.logical_or(sa, sim).sum()
    return {
        "official_SA64": {
            "path": _rel(VALID_SA64),
            "sha1": sha1(VALID_SA64),
            "coverage": float(sa.mean()),
            "z_all_false": [int(z) for z in np.where(sa.mean(axis=(1, 2)) == 0)[0]],
            "note": "含扇形成像足迹（z=3..13 逐层渐开），不是简单死区",
        },
        "official_HSA64": {
            "path": _rel(VALID_HSA64),
            "sha1": sha1(VALID_HSA64),
            "coverage": float(hs.mean()),
            "same_as_SA64": bool(np.array_equal(sa, hs)),
        },
        "sim": {
            "rule": "valid = ones(shape); valid[:2] = False  # 近场死区",
            "sha1": array_sha1(sim),
            "coverage": float(sim.mean()),
        },
        "sim_vs_official_SA64": {
            "iou": float(inter / union),
            "coverage_gap": float(sim.mean() - sa.mean()),
            "disclaimer": "仿真有效区**不等于**真实有效区：仿真只挖掉近场死区（覆盖 0.919），"
                          "真实官方掩膜还挖掉扇形成像足迹（覆盖 0.820）。"
                          "所以「仿真指标可迁移到真实数据」这句话在有效区口吻上也不成立；"
                          "仿真内部各臂之间可比，跨仿真/真实的绝对值不可比。",
        },
    }


def build() -> dict:
    corr = json.loads(CORR.read_text())
    role = json.loads(ROLE_ORDER.read_text())
    cfg = SimCfg()

    # 角色序的自检摘要：每序列每类位移的最小 L（用来看"体对角是不是真的最干净"）
    seq_top = {}
    for seq, rec in role["seq"].items():
        rows = sorted(rec["rows"], key=lambda r: r["L"])
        best = rows[0]
        worst = rows[-1]
        seq_top[seq] = {
            "best_shift": best["shift"], "best_L": best["L"],
            "worst_shift": worst["shift"], "worst_L": worst["L"],
            "n_pairs": len(rows),
            "shift_class_spread_max": max(abs(v) for v in rec["class_spread"].values()),
        }

    return {
        "sampler_config_version": CONFIG_VERSION,
        "purpose": "冻结物理先验采样器的全部外部输入；任何一项变化都必须刷新本文件并重跑受影响实验",
        "geometry": {
            "m": M, "w": W,
            "block_voxels_per_axis": 2 * M,
            "phases": 8,
            "tetris_patterns": [list(map(int, p)) for p in TETRIS_PATTERNS.tolist()],
            "complement_pattern": list(map(int, COMPLEMENT_PATTERN.tolist())),
            "usable_prefix_rule": "(n // (2m)) * (2m)，各轴独立；z=37 -> 36，其余轴 64 全用",
            "usable_prefix": [usable_prefix(n, M) for n in SHAPE],
            "shape_zyx": list(SHAPE),
            "coverage_m1": 1.0,
            "coverage_m2": 8.0 / (2 * M) ** 3,
        },
        "roles": {
            "preregistered_in": "reports/v3_phase2_role_order.md §4（写于 2026-10-21，先于任何训练）",
            "preferred_class": "body_diagonal",
            "slots": {
                "body_diagonal": [1, 4, 6, 3],
                "y_axial_role_swap": [1, 4, 3, 6],
            },
            "slots_note": "两行是**同一批 4 个体素、同样的输入**，只把两个目标对调 —— 实验 C 的"
                           "'正确 vs 错误先验'配对臂就是把二者对调，覆盖率与坐标完全一致",
            "forbidden": ["x_axial", "z_axial"],
            "forbidden_reason": "x 轴向泄漏 0.536/0.468、z 轴向 0.378/0.318，一条就吃掉目标噪声的 1/3–1/2",
            "role_order_file": {
                "path": _rel(ROLE_ORDER),
                "sha1": sha1(ROLE_ORDER),
                "summary": seq_top,
            },
            "caveat": "本表只冻结**噪声侧**（泄漏 L）排序；信号侧 S(k) 未测，"
                       "「体对角最好」这句话仍然不成立（见 v3_phase2_role_order.md §5）",
        },
        "prior": {
            "kind": "measured_speckle_autocorrelation",
            "source": {
                "path": _rel(CORR),
                "sha1": sha1(CORR),
                "sequences": sorted(corr.keys()),
            },
            "deplateau": "fast_curve(acf, plateau)：去掉静态散斑的平台，只保留快速分量",
            "leak_model": "L = c^T C^{-1} c（输入 = a 在 ±w cell 内全部位点，w=3）",
            "not_used": "静态散斑平台本身不进采样器（它属于「该保留」的一侧，进采样器会把噪声当信号）",
        },
        "band": {
            "official_1based": BAND_OFFICIAL,
            "modes": BAND_MODES,
            "repo_0based": BAND_REPO,
            "frozen_in": "scripts/v3_phase5_real_compare.py::band_slice（4 项单测）",
        },
        "imaging_constants": {
            "psf_sigma_zyx": list(cfg.psf_sigma),
            "thermal_sigma_zyx": list(cfg.thermal_sigma),
            "source": "scripts/v3_simulator.py::SimCfg（标定值，见 reports/v3_r06/）",
            "role": "既用于仿真前向，也用于 oracle_lowpass 对照（同一核，保证对照不偷步）",
        },
        "valid_mask": valid_mask_facts(),
        "negative_controls": {
            "status": "spec_frozen_not_run",
            "why_not_run": "属实验 C；实验 A 的门（reports/v3_expA_v2/prereg.md §3.3 T2）通过前不启动。"
                           "规格与「先验变体会指向哪些槽位」已在 reports/v3_expC/prereg.md §1 算好，"
                           "训练本身尚未运行",
            "prior_arms_source": "data/processed/v3_expC/prior_arms.json（scripts/v3_expC_prior_arms.py）",
            "arms": [
                {"name": "correct_prior", "spec": "实测 ρ 推出的体对角槽位 [0,1,7,6]（与预注册的 "
                                                  "[1,4,6,3] 同类，只是换了 tet）",
                 "isolates": "基线臂，其余三条臂都与它比"},
                {"name": "wrong_role_swap", "spec": "同样 4 个体素、同样覆盖率，只把两个目标对调："
                                                     "[0,1,7,6] -> [0,1,6,7]",
                 "isolates": "采样**角色/位移**，覆盖率与坐标完全不变（最干净的单变量对照）"},
                {"name": "wrong_prior_isotropic", "spec": "ρ 换成 exp(-|δ|/ℓ)；SA64 上推出 [2,7,5,0]",
                 "isolates": "先验的**各向异性**部分——注意 HSA64 上它推出与实测相同的槽位，"
                             "所以只在 SA64 上作数（同分内由枚举顺序决定）"},
                {"name": "wrong_prior_axis_permuted", "spec": "ρ 的三轴曲线循环错位 (z,y,x)->(y,x,z)",
                 "status": "void_undetectable",
                 "why_void": "27 个位移类的取值集合映射到自身，排序完全不变（32.3× 仍 32.3×），"
                             "推出的槽位与实测逐位相同。**这条臂已事前证伪为空操作**，不跑；"
                             "与 R13 known_ambiguity（x/y 等长，轴错位只影响显示）一致"},
                {"name": "random_cross", "spec": "每步随机抽 tet（上游 VALID 的做法）",
                 "isolates": "随机化本身是否必要"},
            ],
        },
        "consumers": {
            "train.py --sampler-config": "训练前校验本文件并把副本+sha1 写进 run 目录（见 train.py）",
            "scripts/v3_role_order.py": "生成角色序（本文件记录它的输出 hash）",
            "scripts/v3_probe_prior_sensitivity.py": "错误先验的负控制（未实现，属实验 C）",
        },
    }


def check(cfg: dict | None = None) -> list[str]:
    """重新计算所有 hash/常量，返回不一致项（空列表 = 通过）。"""
    cfg = cfg or json.loads(CONFIG.read_text())
    bad: list[str] = []
    if cfg.get("sampler_config_version") != CONFIG_VERSION:
        bad.append(f"version: {cfg.get('sampler_config_version')} != {CONFIG_VERSION}")
    for key, path in (("role_order_file", ROLE_ORDER), ("source", CORR)):
        rec = cfg["roles"][key] if key == "role_order_file" else cfg["prior"][key]
        want = rec.get("sha1")
        got = sha1(path)
        if want != got:
            bad.append(f"{rec.get('path', path.name)}: 磁盘 sha1 {got[:12]} != 冻结的 {str(want)[:12]}")
    if cfg["band"]["modes"] != BAND_MODES or cfg["band"]["repo_0based"] != BAND_REPO:
        bad.append("band: 与冻结值不一致")
    cfg_psf = list(SimCfg().psf_sigma)
    if cfg["imaging_constants"]["psf_sigma_zyx"] != cfg_psf:
        bad.append(f"psf_sigma: 代码 {cfg_psf} != 冻结 {cfg['imaging_constants']['psf_sigma_zyx']}")
    if cfg["geometry"]["tetris_patterns"] != [list(map(int, p)) for p in TETRIS_PATTERNS.tolist()]:
        bad.append("tetris_patterns: 与 datasets/sampling.py 不一致")
    if cfg["geometry"]["complement_pattern"] != list(map(int, COMPLEMENT_PATTERN.tolist())):
        bad.append("complement_pattern: 与 datasets/sampling.py 不一致")
    vm = valid_mask_facts()
    if vm["official_SA64"]["sha1"] != cfg["valid_mask"]["official_SA64"]["sha1"]:
        bad.append("valid_mask/official_SA64: sha1 变化")
    if vm["sim"]["sha1"] != cfg["valid_mask"]["sim"]["sha1"]:
        bad.append("valid_mask/sim: 规则或形状变化")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="生成/刷新配置")
    ap.add_argument("--check", action="store_true", help="校验配置与磁盘产物一致")
    ap.add_argument("--dump", action="store_true", help="打印配置")
    args = ap.parse_args()
    if not (args.build or args.check or args.dump):
        ap.print_help()
        return 2

    if args.build:
        CONFIG.parent.mkdir(parents=True, exist_ok=True)
        cfg = build()
        CONFIG.write_text(json.dumps(cfg, ensure_ascii=False, indent=1))
        print(f"wrote {CONFIG.relative_to(ROOT)}  version={cfg['sampler_config_version']}")
        bad = check(cfg)
        print(f"self_check: {'OK' if not bad else bad}")
        return 0 if not bad else 1

    if args.check:
        if not CONFIG.exists():
            print(f"missing {CONFIG}")
            return 1
        bad = check()
        vm = json.loads(CONFIG.read_text())["valid_mask"]["sim_vs_official_SA64"]
        print(f"sampler_config: {'OK' if not bad else 'MISMATCH'}")
        print(f"  sim vs official valid IoU = {vm['iou']:.3f}（覆盖差 {vm['coverage_gap']:+.3f}）")
        for b in bad:
            print(f"  - {b}")
        return 0 if not bad else 1

    print(json.dumps(json.loads(CONFIG.read_text()), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
