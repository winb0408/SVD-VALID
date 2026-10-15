#!/usr/bin/env python3
"""Phase 0 / T0-1: lock down the raw IQ files before any measurement depends on them.

Records, per sequence: file size, SHA256, HDF5 dataset shape/dtype, the axis
order after the h5py transpose, the complex encoding, finite-value counts, and
the full time series of a few fixed voxels.  Writes a human-readable manifest to
`docs/data_manifest.md` and a machine-readable copy next to it.

Deliberately does *not* judge correctness from "the picture looks right".
"""

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets.iq_io import load_iq  # noqa: E402

SEQUENCES = ["SA64", "HSA64", "PW3232", "PW1616"]

# Voxels sampled for the time-series check: (fraction of Z, X, Y).
PROBE_VOXELS = [(0.25, 0.5, 0.5), (0.5, 0.25, 0.75), (0.75, 0.5, 0.25)]


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def hdf5_layout(path):
    """Dataset shapes/dtypes exactly as stored, before any transposition."""
    try:
        import h5py
    except ImportError:
        return None
    try:
        with h5py.File(str(path), "r") as f:
            out = {}
            for key in f.keys():
                obj = f[key]
                if hasattr(obj, "shape"):
                    out[key] = {
                        "stored_shape": list(obj.shape),
                        "stored_dtype": str(obj.dtype),
                        "compound_fields": list(obj.dtype.names or []),
                        "nbytes": int(obj.nbytes),
                    }
            return out
    except OSError:
        return {"__format__": "MATLAB v7.2 or earlier (scipy path)"}


def audit_one(root, name):
    path = root / name / f"Data_{name}.mat"
    rec = {"sequence": name, "path": str(path)}
    if not path.exists():
        rec["error"] = "missing"
        return rec

    rec["size_bytes"] = path.stat().st_size
    rec["size_mib"] = round(path.stat().st_size / 2**20, 1)
    rec["sha256"] = sha256(path)
    rec["storage"] = hdf5_layout(path)

    x = load_iq(path)
    rec["loaded_shape"] = list(x.shape)
    rec["loaded_dtype"] = str(x.dtype)
    rec["is_complex"] = bool(np.iscomplexobj(x))

    # Axis interpretation: [Z, X, Y, T(, Array)].  Only Z is uniquely determined
    # by length (37); X and Y are both 64 and cannot be told apart from the
    # header alone.
    shape = x.shape
    if len(shape) == 5:
        rec["axis_order"] = "[Z, X, Y, T, EmittingArray]"
        rec["emitting_arrays"] = int(shape[4])
    elif len(shape) == 4:
        rec["axis_order"] = "[Z, X, Y, T]"
    else:
        rec["axis_order"] = f"UNEXPECTED ndim={len(shape)}"

    if x.ndim >= 4:
        z, a, b, t = shape[:4]
        rec["voxels"] = int(z * a * b)
        rec["frames"] = int(t)
        rec["voxel_size_mm"] = 0.15
        rec["frame_rate_hz"] = 46.875 if t == 400 else (93.75 if t == 800 else None)
        rec["spatial_axis_lengths"] = {"Z": int(z), "X": int(a), "Y": int(b)}

    if np.iscomplexobj(x):
        rec["finite_fraction"] = float(np.mean(np.isfinite(x)))
        rec["abs_min"] = float(np.abs(x).min())
        rec["abs_max"] = float(np.abs(x).max())
        rec["abs_median"] = float(np.median(np.abs(x)))
        rec["phase_std_rad"] = float(np.std(np.angle(x)))
    else:
        rec["finite_fraction"] = float(np.mean(np.isfinite(x)))
        rec["abs_min"] = float(x.min())
        rec["abs_max"] = float(x.max())

    # Per-voxel time series: length and statistics.
    probe = []
    for fz, fx, fy in PROBE_VOXELS:
        idx = (int(fz * (shape[0] - 1)), int(fx * (shape[1] - 1)), int(fy * (shape[2] - 1)))
        ts = x[idx]
        if ts.ndim > 1:  # emitting-array axis present
            ts = ts[..., 0]
        probe.append({
            "voxel": list(idx),
            "series_length": int(ts.shape[0]),
            "abs_mean": float(np.abs(ts).mean()),
            "abs_std": float(np.abs(ts).std()),
            "phase_mean_rad": float(np.angle(ts).mean()),
            "phase_std_rad": float(np.angle(ts).std()),
            "is_complex": bool(np.iscomplexobj(ts)),
        })
    rec["probe_voxels"] = probe
    return rec


def to_markdown(records, data_root):
    lines = [
        "# 数据清单（Phase 0 / T0-1）",
        "",
        f"> 生成日期：{date.today().isoformat()} ｜ 生成脚本：`scripts/audit_iq_data.py`",
        "> 原始数据根目录：`" + str(data_root) + "`（**不入库**）",
        "",
        "本文件锁定后续所有统计所依赖的数据版本。任何结论都必须能追溯到这里的 SHA256。",
        "",
        "## 来源与许可",
        "",
        "- Dryad 数据集 DOI：<https://doi.org/10.5061/dryad.47d7wm3qg>",
        "- 数据说明（随数据集提供）：`README.md`",
        "- 官方处理脚本（MATLAB）：`main_supplementary_figure_2.m`",
        "- 形态：**波束合成后的复数 IQ 体积序列**（非逐阵元原始 RF）",
        "",
        "## 文件与哈希",
        "",
        "| 序列 | 大小 | SHA256 | 磁盘形状（存储序） | dtype |",
        "|---|---|---|---|---|",
    ]
    for r in records:
        if r.get("error"):
            lines.append(f"| {r['sequence']} | — | **缺失** | — | — |")
            continue
        stored = "—"
        for _k, v in (r.get("storage") or {}).items():
            if isinstance(v, dict):
                stored = str(v.get("stored_shape"))
        lines.append(
            f"| {r['sequence']} | {r['size_mib']} MiB | `{r['sha256'][:16]}…` "
            f"| {stored} | {r.get('loaded_dtype', '—')} |"
        )
    lines += [
        "",
        "> 完整 SHA256 见 `runs/phase0_baseline/metadata.json`。",
        "",
        "## 轴序、复数编码与网格",
        "",
        "| 序列 | 转置后形状 | 轴序 | 帧数 | 体素数 | 帧率 |",
        "|---|---|---|---|---|---|",
    ]
    for r in records:
        if r.get("error"):
            continue
        lines.append(
            f"| {r['sequence']} | {r['loaded_shape']} | {r['axis_order']} "
            f"| {r['frames']} | {r['voxels']:,} | {r['frame_rate_hz']} Hz |"
        )
    lines += [
        "",
        "**已知歧义（必须记录）**：X 与 Y 的长度都是 64，仅凭文件头**无法区分**二者。",
        "这不影响任何测量：展平后的行置换是相似变换，SVD 血流带滤波与逐体素功率在两种解释下完全相同。",
        "它只影响显示方向。",
        "",
        "**展平约定**：本仓库用 C order（`reshape(-1, T)`），官方 MATLAB 用列优先序；",
        "同样只造成行置换，不影响结果（见 `datasets/iq_io.py` 顶部说明）。",
        "",
        "## 逐体素时间序列抽查",
        "",
        "| 序列 | 体素 | 长度 | 幅度均值 | 幅度标准差 | 相位均值 (rad) | 相位标准差 (rad) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in records:
        for p in r.get("probe_voxels", []):
            lines.append(
                f"| {r['sequence']} | {p['voxel']} | {p['series_length']} | {p['abs_mean']:.3e} "
                f"| {p['abs_std']:.3e} | {p['phase_mean_rad']:+.3f} | {p['phase_std_rad']:.3f} |"
            )
    lines += [
        "",
        "抽查判读：时间轴长度与文档一致、值为复数、相位非常数（说明确实是复数 IQ 而非幅度图），",
        "即可认为读取链路正确。**不以「图看起来正常」作为判据**。",
        "",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/ssd1/wjb/Data/Dyrad/data")
    ap.add_argument("--sequences", nargs="*", default=SEQUENCES)
    ap.add_argument("--manifest", default="docs/data_manifest.md")
    ap.add_argument("--json", default="runs/phase0_baseline/metadata.json")
    args = ap.parse_args()

    root = Path(args.data_root)
    records = [audit_one(root, name) for name in args.sequences]

    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(to_markdown(records, root), encoding="utf-8")

    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(
        json.dumps({"data_root": str(root), "sequences": records}, indent=2),
        encoding="utf-8",
    )

    for r in records:
        if r.get("error"):
            print(f"{r['sequence']}: MISSING")
        else:
            print(f"{r['sequence']}: {r['loaded_shape']} {r['loaded_dtype']} "
                  f"sha256={r['sha256'][:16]}…")
    print(f"\nwrote {args.manifest} and {args.json}")


if __name__ == "__main__":
    main()
