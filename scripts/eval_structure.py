#!/usr/bin/env python3
"""Single source of truth for structural-fidelity metrics.

Rationale (L22): through L14-L20 these metrics were recomputed ad hoc in
inline python each round, and the vascular-core retention drifted between
records (L15/L16 reported 0.69/0.83, L20 reported 0.63/0.82). Every metric
below now has ONE definition, ONE code path, and is reported with the
thresholds that produced it. Any number quoted in the log must come from
here.

Reference domain
----------------
All volumes live in the prepare_svd contrast-remap domain [0, 1]: the input
``power_doppler.npy`` is a monotone quantile remap of the official 400-frame
Power Doppler, so the input volume itself is the high-SNR pseudo ground truth
used as reference (it is the same monotone map the network is trained on).
Structural metrics are therefore computed input-vs-output in this domain;
physical dB conversion is a display concern (see visualize_remap.py).

Metrics
-------
pseudogt_pearson     Pearson r between input and output over all voxels.
vessel_core_keep     mean(out)/mean(in) over input >= CORE_THR (0.65).
bright_top_keep      same over the top 1% of input voxels (model-free bucket).
vessel_body_keep     same over 0.2 <= input < 0.65 (the bulk of vessel voxels).
isolated_blob_keep   same over connected components of input > 0.5 with
                     <= BLOB_MAX_VOX voxels (isolated speckle, not vasculature).
vessel_blob_keep     same over components of input > 0.5 with > BLOB_MAX_VOX.
dark_mean            mean(out) over input <= DARK_THR (0.05) -- should be ~0.
dark_p99             99th percentile of out over the dark region (fill level).
contrast_span        p99(out | input >= CORE_THR) - dark_mean.
grad_keep            mean|grad out| / mean|grad in| (isotropic central diff).
hf_noise_reduction   1 - std(highpass(out)) / std(highpass(in)), highpass =
                     vol - uniform_filter(vol, 3).
out_min/out_max       raw output extrema.
oob_frac             fraction of output outside [0, 1] (model is unconstrained).

Usage
-----
  python scripts/eval_structure.py \
      --input data/processed/sa64_blood_remap/power_doppler.npy \
      --models B3=runs/B3_sa64_remap_random/infer/denoised.npy \
               B3p=runs/B3p_sa64_remap_patchrandom/infer/denoised.npy \
               C3=runs/C3_sa64_remap_guided/infer/denoised.npy \
      --out runs/eval_structure_sa64.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

CORE_THR = 0.65
BODY_LO, BODY_HI = 0.2, 0.65
BLOB_THR = 0.5
BLOB_MAX_VOX = 20
DARK_THR = 0.05
GRAD_KSIZE = 3


def _grad_mag(vol):
    g = np.gradient(vol.astype(np.float64))
    return np.sqrt(sum(gi ** 2 for gi in g))


def _highpass(vol, ksize=GRAD_KSIZE):
    return vol - ndimage.uniform_filter(vol, size=ksize)


def blob_masks(ref, thr=BLOB_THR, max_vox=BLOB_MAX_VOX):
    """Return (isolated, vessels) masks of bright connected components."""
    lab, n = ndimage.label(ref > thr)
    if n == 0:
        z = np.zeros_like(ref, dtype=bool)
        return z, z
    sizes = np.bincount(lab.ravel())
    small_ids = np.where((sizes > 0) & (sizes <= max_vox))[0]
    small_ids = small_ids[small_ids != 0]
    isolated = np.isin(lab, small_ids)
    large_ids = np.where(sizes > max_vox)[0]
    large_ids = large_ids[large_ids != 0]
    vessels = np.isin(lab, large_ids)
    return isolated, vessels


def _keep_ratio(ref, out, mask):
    if mask.sum() == 0:
        return float("nan"), int(mask.sum())
    return float(out[mask].mean() / max(ref[mask].mean(), 1e-12)), int(mask.sum())


def evaluate(ref, out, name="model"):
    ref = np.asarray(ref, dtype=np.float64)
    out = np.asarray(out, dtype=np.float64)
    if ref.shape != out.shape:
        raise ValueError(f"shape mismatch: {ref.shape} vs {out.shape}")

    m = {"name": name, "shape": list(ref.shape)}
    m["pseudogt_pearson"] = float(np.corrcoef(ref.ravel(), out.ravel())[0, 1])

    core = ref >= CORE_THR
    m["vessel_core_keep"], m["n_core_vox"] = _keep_ratio(ref, out, core)

    flat = ref.ravel()
    k = max(1, int(0.01 * flat.size))
    thr_top = np.partition(flat, -k)[-k]
    m["bright_top_keep"], m["n_bright_top_vox"] = _keep_ratio(ref, out, ref >= thr_top)

    m["vessel_body_keep"], m["n_body_vox"] = _keep_ratio(
        ref, out, (ref >= BODY_LO) & (ref < BODY_HI)
    )

    iso, ves = blob_masks(ref)
    m["isolated_blob_keep"], m["n_isolated_vox"] = _keep_ratio(ref, out, iso)
    m["vessel_blob_keep"], m["n_vessel_blob_vox"] = _keep_ratio(ref, out, ves)

    dark = ref <= DARK_THR
    m["dark_mean"] = float(out[dark].mean()) if dark.any() else float("nan")
    m["dark_p99"] = float(np.percentile(out[dark], 99)) if dark.any() else float("nan")

    core_p99 = float(np.percentile(out[core], 99)) if core.any() else float("nan")
    m["contrast_span"] = core_p99 - m["dark_mean"]

    gi, go = _grad_mag(ref), _grad_mag(out)
    m["grad_keep"] = float(go.mean() / max(gi.mean(), 1e-12))
    hi, ho = _highpass(ref), _highpass(out)
    m["hf_noise_reduction"] = float(1.0 - ho.std() / max(hi.std(), 1e-12))

    m["out_min"] = float(out.min())
    m["out_max"] = float(out.max())
    m["oob_frac"] = float(((out < 0.0) | (out > 1.0)).mean())
    m["out_mean"] = float(out.mean())
    m["in_mean"] = float(ref.mean())
    return m


def fmt(row):
    def f(k):
        v = row.get(k)
        return "  n/a " if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:6.3f}"
    return (
        f"{row['name']:>6} | {f('pseudogt_pearson')} | {f('vessel_core_keep')} | "
        f"{f('bright_top_keep')} | {f('vessel_body_keep')} | {f('isolated_blob_keep')} | "
        f"{f('dark_mean')} | {f('contrast_span')} | {f('grad_keep')} | {f('oob_frac')}"
    )


HEADER = (
    "  name | pseudoGT | core   | top1%  | body   | isoBlob | darkMn | span   | "
    "grad   | oobFrac"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="Reference volume (the input the model saw, remap [0,1]).")
    ap.add_argument("--models", nargs="+", required=True,
                    help="Entries NAME=path/to/denoised.npy")
    ap.add_argument("--out", help="Optional JSON output path.")
    args = ap.parse_args()

    ref = np.load(args.input).astype(np.float32)
    # The input as the network sees it is itself a metric row (identity check:
    # every keep ratio must be 1.000 and oob_frac 0.000 -- a self-test).
    rows = [evaluate(ref, ref, name="input")]
    for spec in args.models:
        if "=" not in spec:
            raise ValueError(f"--models entries must be NAME=path, got {spec!r}")
        name, path = spec.split("=", 1)
        rows.append(evaluate(ref, np.load(path).astype(np.float32), name=name))

    print(f"reference: {args.input}")
    print(f"thresholds: core>={CORE_THR}, body=[{BODY_LO},{BODY_HI}), "
          f"blob>{BLOB_THR} split at {BLOB_MAX_VOX} vox, dark<={DARK_THR}")
    print(HEADER)
    print("-" * len(HEADER))
    for r in rows:
        print(fmt(r))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"reference": args.input, "thresholds": {
                "core": CORE_THR, "body": [BODY_LO, BODY_HI],
                "blob": BLOB_THR, "blob_max_vox": BLOB_MAX_VOX, "dark": DARK_THR,
            }, "rows": rows}, f, indent=2)
        print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
