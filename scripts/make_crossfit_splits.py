#!/usr/bin/env python3
"""Phase 1 / T1-1: build the C/A/B frame groupings used by cross-fitted SVD.

For each grouping configuration this writes a `split_manifest.json` recording,
frame by frame, which role that frame plays.  The hard rules enforced here (and
re-asserted by the consumer) are:

  * C, A and B are pairwise disjoint;
  * every frame of the acquisition is assigned to exactly one group;
  * ``T_A == T_B`` exactly.

The last one is not cosmetic.  The power volume is a mean over frames, so if the
two observation groups have different lengths the mean injects a ``T_B / T_A``
scale offset between D_A and D_B -- a systematic intensity bias created by the
definition rather than by the data, which would make the Phase 2 intensity gate
fire for the wrong reason.  When a group split is inherently unequal (400 frames
over 3 interleaved groups is 134/133/133) the surplus frames are dropped from the
*larger observation group only*, from the end, and the dropped indices are
recorded under ``trimmed_for_equal_length`` so the choice stays auditable.

Wording note (see configs/crossfit_pairs.yaml): the official script's blood band
is 1-based inclusive ``60:350`` out of 400 frames = 291 modes.  That band is NOT
transferred to a calibration subgroup with fewer frames -- a calibration matrix
has at most ``T_C`` modes, so the rank rule is re-derived from the calibration
spectrum alone (Phase 1 / T1-2).
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def build_groups(frames, strategy, n_groups):
    """Partition ``range(frames)`` into ``n_groups`` index groups."""
    if n_groups < 2:
        raise ValueError("n_groups must be >= 2")
    if strategy == "interleaved":
        # Round-robin over frame index: the two observation groups interleave
        # across the whole acquisition, so slow flow changes are shared rather
        # than split between them.
        groups = [[t for t in range(frames) if t % n_groups == g] for g in range(n_groups)]
    elif strategy == "contiguous":
        # One contiguous block per group: maximum temporal separation between
        # the observation groups, at the cost of a possible flow-state shift.
        edges = [round(i * frames / n_groups) for i in range(n_groups + 1)]
        groups = [list(range(edges[i], edges[i + 1])) for i in range(n_groups)]
    else:
        raise ValueError(f"Unknown strategy {strategy!r}")
    return groups


def role_assignments(n_groups):
    """(calibration group ids, A group id, B group id) for every fold.

    3 groups -> the 3 rotations of (C, A, B).
    >=4 groups -> every unordered {A, B} pair, with the remaining groups as C.
    """
    if n_groups == 3:
        return [([c], (c + 1) % 3, (c + 2) % 3) for c in range(3)]
    return [([g for g in range(n_groups) if g not in (a, b)], a, b)
            for a, b in combinations(range(n_groups), 2)]


def make_folds(groups, name, strategy):
    folds = []
    for c_ids, a_id, b_id in role_assignments(len(groups)):
        calib = sorted(t for g in c_ids for t in groups[g])
        obs_a = list(groups[a_id])
        obs_b = list(groups[b_id])

        trimmed = {"A": [], "B": []}
        if len(obs_a) != len(obs_b):
            longer, shorter = ("A", "B") if len(obs_a) > len(obs_b) else ("B", "A")
            n_drop = abs(len(obs_a) - len(obs_b))
            if longer == "A":
                trimmed["A"], obs_a = obs_a[-n_drop:], obs_a[:-n_drop]
            else:
                trimmed["B"], obs_b = obs_b[-n_drop:], obs_b[:-n_drop]

        fold = {
            "fold_id": f"{name}_c{c_ids[0]}a{a_id}b{b_id}" if len(c_ids) == 1
                       else f"{name}_c{'_'.join(map(str, c_ids))}a{a_id}b{b_id}",
            "strategy": strategy,
            "calibration_group_ids": c_ids,
            "calibration": calib,
            "A": obs_a,
            "B": obs_b,
            "T_C": len(calib),
            "T_A": len(obs_a),
            "T_B": len(obs_b),
            "trimmed_for_equal_length": trimmed,
        }
        assert_fold(fold)
        folds.append(fold)
    return folds


def assert_fold(fold):
    c, a, b = set(fold["calibration"]), set(fold["A"]), set(fold["B"])
    assert not (c & a), f"{fold['fold_id']}: C and A overlap"
    assert not (c & b), f"{fold['fold_id']}: C and B overlap"
    assert not (a & b), f"{fold['fold_id']}: A and B overlap"
    assert len(fold["A"]) == len(fold["B"]), (
        f"{fold['fold_id']}: T_A={len(fold['A'])} != T_B={len(fold['B'])}; the mean "
        f"over frames would inject a T_B/T_A scale offset between D_A and D_B")
    assert fold["T_A"] > 0 and fold["T_C"] > 0, f"{fold['fold_id']}: empty group"


def build_manifest(sequence, frames, name, strategy, n_groups):
    groups = build_groups(frames, strategy, n_groups)
    assert sorted(t for g in groups for t in g) == list(range(frames)), \
        "groups must cover every frame exactly once"
    folds = make_folds(groups, name, strategy)

    # Frame-level role table: the manifest is the single source of truth for
    # which frames a fold may look at, so it is checked directly rather than
    # inferred.
    roles = []
    for fold in folds:
        role = {t: "C" for t in fold["calibration"]}
        role.update({t: "A" for t in fold["A"]})
        role.update({t: "B" for t in fold["B"]})
        for t in fold["trimmed_for_equal_length"]["A"]:
            role[t] = "trimmed"
        for t in fold["trimmed_for_equal_length"]["B"]:
            role[t] = "trimmed"
        roles.append({"fold_id": fold["fold_id"],
                      "per_frame_role": [role.get(t, "unused") for t in range(frames)]})

    return {
        "sequence": sequence,
        "frames": frames,
        "name": name,
        "strategy": strategy,
        "n_groups": n_groups,
        "group_sizes": [len(g) for g in groups],
        "folds": folds,
        "per_frame_roles": roles,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/crossfit_pairs.yaml")
    ap.add_argument("--out-root", default="data/processed/crossfit/splits")
    ap.add_argument("--sequences", nargs="*", default=None,
                    help="Subset of sequences from the config (default: all).")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    written = []
    for seq, spec in cfg["sequences"].items():
        if args.sequences and seq not in args.sequences:
            continue
        for split in spec["splits"]:
            man = build_manifest(seq, int(spec["frames"]), split["name"],
                                 split["strategy"], int(split["n_groups"]))
            path = out_root / f"{split['name']}.json"
            path.write_text(json.dumps(man, indent=2), encoding="utf-8")
            written.append(man)
            sizes = man["group_sizes"]
            fold_desc = ", ".join(
                f"{f['fold_id'].split('_')[-1]}(C{f['T_C']}/A{f['T_A']}/B{f['T_B']})"
                for f in man["folds"])
            print(f"{seq} {split['name']:<14} groups={sizes} folds={len(man['folds'])}: {fold_desc}")

    print(f"\nwrote {len(written)} split manifests to {out_root}")


if __name__ == "__main__":
    main()
