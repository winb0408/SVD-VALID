#!/usr/bin/env python3
"""Round8: ZERO-TRAINING comparison of Round7's eight-phase scatter vs full-volume VALID-style inference.

Run on original research host (requires existing Round7 checkpoints and normalized volumes):
  python scripts/v3_round8_zero_train_full_infer.py --self-test
  python scripts/v3_round8_zero_train_full_infer.py --seqs SA64 HSA64 --device cuda
Never trains, modifies, or overwrites checkpoints/old Round7 results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
PAIRS = {"P0": (1, 4, 6, 3), "P1": (0, 2, 7, 5),
         "P2": (3, 6, 4, 1), "P3": (5, 7, 2, 0)}


def parity_slices(i):
    return tuple(slice(x, None, 2) for x in ((i >> 2) & 1, (i >> 1) & 1, i & 1))


def starts(n, p, overlap):
    if p > n:
        raise ValueError(f"patch {p} exceeds volume axis {n}")
    step = max(1, int(p * (1.0 - overlap)))
    v = list(range(0, n - p + 1, step))
    if v[-1] != n - p:
        v.append(n - p)
    return v


@torch.no_grad()
def full_infer(model, x, device, patch, overlap):
    """Full-resolution patch inference: same model everywhere; overlapping predictions averaged."""
    acc = np.zeros_like(x, dtype=np.float64)
    count = np.zeros_like(x, dtype=np.float64)
    zlist, ylist, xlist = (starts(d, p, overlap) for d, p in zip(x.shape, patch))
    for z in zlist:
        for y in ylist:
            for xx in xlist:
                sl = (slice(z, z + patch[0]), slice(y, y + patch[1]),
                      slice(xx, xx + patch[2]))
                t = torch.from_numpy(np.ascontiguousarray(x[sl]))[None, None].to(device)
                yp = model(t)[0, 0].cpu().numpy()
                if yp.shape != tuple(patch):
                    raise RuntimeError(f"unexpected full patch output {yp.shape}")
                acc[sl] += yp
                count[sl] += 1
    if not count.all():
        raise RuntimeError("uncovered voxels")
    return (acc / count).astype(np.float32)


def phase_bias(y, source, mask):
    """Affine-adjusted parity offsets; diagnostic, NOT clean-truth metric."""
    x, v = source[mask].astype(np.float64), y[mask].astype(np.float64)
    slope = np.dot(x - x.mean(), v - v.mean()) / max(np.dot(x - x.mean(), x - x.mean()), 1e-12)
    bias = v.mean() - slope * x.mean()
    residual = y - slope * source - bias
    means = []
    for i in range(8):
        sl = parity_slices(i)
        m = mask[sl]
        means.append(float(residual[sl][m].mean()) if m.any() else float("nan"))
    return {"phase_residual_means": means, "phase_bias_std": float(np.nanstd(means)),
            "phase_bias_range": float(np.nanmax(means) - np.nanmin(means)),
            "affine_slope": float(slope), "affine_intercept": float(bias)}


def y_seam(x, mask):
    m1 = mask[:, :, :-1] & mask[:, :, 1:]
    m2 = mask[:, :, :-2] & mask[:, :, 2:]
    d1 = float(np.mean(np.abs(np.diff(x, axis=2)[m1])))
    d2 = float(np.mean(np.abs((x[:, :, 2:] - x[:, :, :-2])[m2])))
    return {"lag1_abs": d1, "lag2_abs": d2,
            "lag1_over_half_lag2": float(d1 / max(0.5 * d2, 1e-12))}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def self_test():
    """Synthetic only: demonstrate that a four-model phase mosaic creates checkerboard offsets."""
    class Bias(torch.nn.Module):
        def __init__(self, b):
            super().__init__()
            self.b = b
        def forward(self, x):
            return x + self.b
    x = np.ones((32, 64, 64), dtype=np.float32)
    mods = {k: Bias(0.1 * i) for i, k in enumerate(PAIRS)}
    phase = np.zeros_like(x)
    table = {}
    for k, (a, b, c, d) in PAIRS.items():
        table[a] = k, c
        table[b] = k, d
    for i in range(8):
        label, target = table[i]
        t = torch.from_numpy(np.ascontiguousarray(x[parity_slices(i)]))[None, None]
        phase[parity_slices(target)] = mods[label](t)[0, 0].numpy()
    full = np.mean([full_infer(m, x, "cpu", (32, 64, 64), 0.25) for m in mods.values()], axis=0)
    mask = np.ones_like(x, dtype=bool)
    a, b = phase_bias(phase, x, mask), phase_bias(full, x, mask)
    assert a["phase_bias_std"] > 0.04 and b["phase_bias_std"] < 1e-7, (a, b)
    print(json.dumps({"synthetic_test_passed": True, "new_training_jobs": 0,
                      "eight_phase_bias_std": a["phase_bias_std"],
                      "full_patch_ensemble_bias_std": b["phase_bias_std"]}, indent=2))


@torch.no_grad()
def main_sequence(seq, device, patch, overlap):
    # Lazy import: self-test must work even without user-only data and checkpoints.
    from v3_round7_ours_recon import group_models, reconstruct, NORM
    from v3_round4_b1_prep import inv
    source = ROOT / f"data/processed/v3_round7_b1/logeps/{seq}_full_x.npy"
    mask_path = ROOT / f"data/processed/official/{seq}/valid_mask.npy"
    meta = ROOT / f"data/processed/v3_round6/{seq}/metadata.json"
    for p in (source, mask_path, meta):
        if not p.exists():
            raise FileNotFoundError(f"Not run: required original research data missing: {p}")
    for pid in PAIRS:
        p = ROOT / f"runs/v3_round7/{seq}_{pid}/latest.pt"
        if not p.exists():
            raise FileNotFoundError(f"Not run: Round7 checkpoint missing: {p}")
    x = np.load(source).astype(np.float32)
    dims = tuple(2 * (v // 2) for v in x.shape)  # same 36-slice support as Round7
    sl = tuple(slice(0, k) for k in dims)
    x = x[sl]
    mask = np.load(mask_path).astype(bool)[sl]
    table = group_models(seq, device)
    models = {pid: table[slots[0]][0] for pid, slots in PAIRS.items()}
    legacy, filled = reconstruct(table, x, device)
    if not filled.all():
        raise RuntimeError("legacy phase scatter has holes")
    out = {"svd_input": x, "legacy_8phase": legacy}
    for pid, model in models.items():
        out[f"full_{pid}"] = full_infer(model, x, device, patch, overlap)
    out["full_ensemble"] = np.mean([out[f"full_{pid}"] for pid in PAIRS], axis=0).astype(np.float32)
    norm = NORM["logeps"]
    s = float(json.loads(meta.read_text())["s"])
    pd_in = np.maximum(inv("logeps", x * norm["std"] + norm["mean"], s), 0)
    refmax = max(float(pd_in[mask].max()), 1e-12)
    bright = mask & (pd_in >= np.quantile(pd_in[mask], 0.9))
    background = mask & (pd_in <= np.quantile(pd_in[mask], 0.3))
    result = {"sequence": seq, "shape": dims, "patch": list(patch), "overlap": overlap,
              "new_training_jobs": 0,
              "checkpoints_sha256": {pid: _sha256(ROOT / f"runs/v3_round7/{seq}_{pid}/latest.pt")
                                     for pid in PAIRS},
              "methods": {}, "note": "no clean GT; parity/core/background are proxies"}
    save_dir = ROOT / f"data/processed/v3_round8_zero_train/{seq}"
    report_dir = ROOT / "reports/v3_round8_zero_train"
    save_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    for name, y in out.items():
        pd = np.maximum(inv("logeps", y * norm["std"] + norm["mean"], s), 0)
        db = 10 * np.log10(np.maximum(pd, 1e-12) / refmax)  # SAME reference for every method
        np.save(save_dir / f"{name}_logeps_standardized.npy", y.astype(np.float32))
        np.save(save_dir / f"{name}_pd_linear.npy", pd.astype(np.float32))
        np.save(save_dir / f"{name}_db_common_reference.npy", db.astype(np.float32))
        result["methods"][name] = {
            **phase_bias(y, x, mask),
            "seam_y_logeps": y_seam(y, mask),
            "seam_y_db": y_seam(db, mask),
            "core_mean_ratio": float(pd[bright].mean() / max(pd_in[bright].mean(), 1e-12)),
            "bg_std_ratio": float(pd[background].std() / max(pd_in[background].std(), 1e-12))}
    (report_dir / f"{seq}_comparison.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        keys = list(out)
        fig, ax = plt.subplots(2, len(keys), figsize=(3 * len(keys), 6))
        for i, name in enumerate(keys):
            pd = np.maximum(inv("logeps", out[name] * norm["std"] + norm["mean"], s), 0)
            db = 10 * np.log10(np.maximum(pd, 1e-12) / refmax)
            ax[0, i].imshow(db.max(axis=0), cmap="hot", vmin=-25, vmax=0)
            ax[1, i].imshow(db[db.shape[0] // 2], cmap="hot", vmin=-25, vmax=0)
            ax[0, i].set_title(name, fontsize=9)
        for a in ax.ravel(): a.axis("off")
        fig.suptitle(f"{seq}: zero-training inference; common dB reference")
        fig.tight_layout()
        fig.savefig(report_dir / f"{seq}_shared_scale.png", dpi=140)
        plt.close(fig)
    except ImportError:
        pass
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="+", default=["SA64", "HSA64"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--patch", nargs=3, type=int, default=[32, 64, 64])
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if not 0 <= args.overlap < 1:
        raise ValueError("overlap outside [0,1)")
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    for seq in args.seqs:
        r = main_sequence(seq, device, tuple(args.patch), args.overlap)
        print(f"{seq}: wrote {len(r['methods'])} inference-only output methods; training jobs=0")


if __name__ == "__main__":
    main()
