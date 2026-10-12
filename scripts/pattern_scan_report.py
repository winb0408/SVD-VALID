#!/usr/bin/env python3
"""Turn the fixed-pattern scan into a ranking, and ask what predicts it.

The guided selector optimizes rho_signal up / rho_noise down, yet the controlled
comparison showed guided == fixed pattern 4 (L22). So either the criterion is
right and only the estimator is degenerate, or the criterion does not track what
actually makes a sampling geometry good. This script answers that by training one
model per fixed pattern (identical everything else), ranking them, and correlating
the ranking against the candidate explanations available offline:

  rho_signal(k), rho_noise(k)   from runs/phase3/band_stats.json (the criterion)
  geometry(k)                   from TETRIS_PATTERNS alone: how the four
                                subvolume offsets are distributed over the
                                z / y / x axes and how far apart the pairs sit

Usage:
  # after inferring every fixed-pattern run into runs/F<K>_*/infer/
  python scripts/eval_structure.py --input <vol> --models F0=... F1=... ... \
      --out runs/eval_structure_scan.json
  python scripts/pattern_scan_report.py --scan runs/eval_structure_scan.json \
      --phase3 runs/phase3/band_stats.json --out runs/pattern_scan
"""

import argparse
import json
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datasets.sampling import TETRIS_PATTERNS, index_offset  # noqa: E402

# Higher is better for every metric listed here.
RANK_METRICS = ["vessel_core_keep", "bright_top_keep", "pseudogt_pearson",
                "isolated_blob_keep"]

# A model that does nothing scores 1.000 on every *_keep metric, so a keep
# metric on its own rewards NOT denoising. hf_noise_reduction (1 - sigma of the
# highpass of the output over the input's) is the axis that says how much work
# the model actually did; only models at comparable values of it may be
# compared on the keep metrics. Below this value a model has either not
# denoised or amplified high-frequency content, and its keep metrics are not
# evidence of quality.
DENOISE_FLOOR = 0.40


def geometry_descriptors(pattern):
    """Pure-geometry description of one Tetris pattern.

    offsets         : the four (dz,dy,dx) block positions it samples
    axis_counts     : how many of the four sit at value 1 on each axis
    pair_disp       : for the 6 sample pairs, mean |displacement| per axis,
                      i.e. how far apart the retained samples are along z/y/x
    """
    offs = np.array([index_offset(i) for i in pattern], dtype=float)   # (4,3)
    axis_counts = offs.sum(axis=0)
    disp = np.abs(offs[:, None, :] - offs[None, :, :])                 # (4,4,3)
    iu = np.triu_indices(4, k=1)
    pair = disp[iu]                                                    # (6,3)
    return dict(offsets=[[int(v) for v in o] for o in offs],
                axis_counts=[int(v) for v in axis_counts],
                pair_disp_mean=[float(v) for v in pair.mean(axis=0)],
                pair_disp_min=[float(v) for v in pair.min(axis=0)],
                pair_disp_max=[float(v) for v in pair.max(axis=0)])


def spearman(a, b):
    from scipy.stats import spearmanr
    if len(a) < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan"), float("nan")
    r, p = spearmanr(a, b)
    return float(r), float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True,
                    help="eval_structure.py output over the per-pattern models.")
    ap.add_argument("--phase3", default="runs/phase3/band_stats.json")
    ap.add_argument("--domain", default="detrended",
                    help="Which band_stats domain to quote rho from.")
    ap.add_argument("--rank-by", default="vessel_core_keep")
    ap.add_argument("--out", default="runs/pattern_scan")
    args = ap.parse_args()

    scan = json.load(open(args.scan, encoding="utf-8"))
    by_name = {r["name"]: r for r in scan["rows"]}

    phase = json.load(open(args.phase3, encoding="utf-8"))[args.domain]["global_stats"]
    rho_sig = {int(g["pattern"]): g["rho_signal"] for g in phase}
    rho_noi = {int(g["pattern"]): g["rho_noise"] for g in phase}

    rows = []
    for k, pattern in enumerate(TETRIS_PATTERNS.tolist()):
        name = f"F{k}"
        if name not in by_name:
            continue
        m = by_name[name]
        g = geometry_descriptors(pattern)
        rows.append(dict(pattern=k, positions=pattern, **g,
                         rho_signal=rho_sig.get(k), rho_noise=rho_noi.get(k),
                         **{key: m[key] for key in m
                            if key not in ("name", "shape")}))

    if not rows:
        raise SystemExit("no F<K> rows found in the scan file")

    rows.sort(key=lambda r: -r[args.rank_by])
    print(f"ranked by {args.rank_by} (higher = better)\n")
    hdr = (f"{'pat':>4} {'offset(dz,dy,dx)':>26} {'denoise':>8} {'core':>6} "
           f"{'top1%':>6} {'pseudoGT':>8} {'isoBlob':>7} {'span':>6} {'oob%':>6} "
           f"{'rho_sig':>8} {'rho_noi':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        offs = " ".join(f"{o[0]}{o[1]}{o[2]}" for o in r["offsets"])
        flag = " " if r["hf_noise_reduction"] >= DENOISE_FLOOR else "!"
        print(f"{r['pattern']:>4} {offs:>26} {r['hf_noise_reduction']:>+8.3f} "
              f"{r['vessel_core_keep']:>6.3f} {r['bright_top_keep']:>6.3f} "
              f"{r['pseudogt_pearson']:>8.3f} {r['isolated_blob_keep']:>7.3f} "
              f"{r['contrast_span']:>6.3f} {r['oob_frac']*100:>6.2f} "
              f"{r['rho_signal']:>+8.4f} {r['rho_noise']:>+8.4f}{flag}")

    weak = [r["pattern"] for r in rows if r["hf_noise_reduction"] < DENOISE_FLOOR]
    print(f"\n! = hf_noise_reduction < {DENOISE_FLOOR}: the model barely denoised "
          f"(or amplified HF), so its keep metrics are not comparable.")
    if weak:
        print(f"  essentially non-denoising patterns: {weak}")
    # Include the non-pattern references (B3 / C3) if the scan file carries them,
    # so the matched set is the real denoisers rather than one lone pattern.
    matched = [r for r in rows if r["hf_noise_reduction"] >= DENOISE_FLOOR]
    matched_refs = [dict(by_name[n], pattern=None)
                    for n in by_name
                    if not n.startswith("F") and n != "input"
                    and by_name[n].get("hf_noise_reduction", -1) >= DENOISE_FLOOR]
    print(f"\ndenoising-matched models (hf_noise_reduction >= {DENOISE_FLOOR}), "
          f"by {args.rank_by}:")
    for r in sorted(matched + matched_refs, key=lambda r: -r[args.rank_by]):
        label = f"pattern {r['pattern']}" if r.get("pattern") is not None else r["name"]
        print(f"  {label:>10}: core {r['vessel_core_keep']:.3f}  "
              f"body {r['vessel_body_keep']:.3f}  span {r['contrast_span']:.3f}  "
              f"pseudoGT {r['pseudogt_pearson']:.3f}  "
              f"denoise {r['hf_noise_reduction']:+.3f}")

    # Does anything we can measure offline predict the ranking?
    def report_predictors(subset, label):
        if len(subset) < 3:
            print(f"\n({label}: fewer than 3 models, skipping correlation)")
            return {}
        print(f"\npredictors of {args.rank_by} ({label}, n={len(subset)}):")
        target = np.array([r[args.rank_by] for r in subset])
        candidates = {
            "rho_signal (criterion +)": [r["rho_signal"] for r in subset],
            "rho_noise  (criterion -)": [-r["rho_noise"] for r in subset],
            "denoise strength": [r["hf_noise_reduction"] for r in subset],
            "axis_count_z": [r["axis_counts"][0] for r in subset],
            "axis_count_y": [r["axis_counts"][1] for r in subset],
            "axis_count_x": [r["axis_counts"][2] for r in subset],
            "pair_disp_z": [r["pair_disp_mean"][0] for r in subset],
            "pair_disp_y": [r["pair_disp_mean"][1] for r in subset],
            "pair_disp_x": [r["pair_disp_mean"][2] for r in subset],
        }
        res = {}
        for lab, vals in candidates.items():
            v = np.array(vals, dtype=float)
            if np.any(np.isnan(v)):
                continue
            r, p = spearman(v, target)
            res[lab] = dict(spearman_r=r, p=p)
            print(f"  {lab:26s} rho={r:+.3f}  p={p:.3f}")
        return res

    corr_all = report_predictors(rows, "all patterns")
    corr_matched = report_predictors(matched, "denoising-matched only")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "pattern_scan.json", "w", encoding="utf-8") as f:
        json.dump(dict(rank_by=args.rank_by, domain=args.domain,
                       denoise_floor=DENOISE_FLOOR,
                       denoising_matched=[r["pattern"] for r in matched],
                       rows=rows,
                       predictors_all=corr_all,
                       predictors_denoising_matched=corr_matched), f, indent=2)
    print(f"\nsaved {out / 'pattern_scan.json'}")


if __name__ == "__main__":
    main()
