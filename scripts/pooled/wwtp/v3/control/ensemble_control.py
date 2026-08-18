#!/usr/bin/env python3
"""ensemble_control.py -- combine the per-island champions of a control run into a
single prediction series, written in the EXACT format score_v2.py / score_v3.py
already read, so the ensemble is scored by the identical scorer on the identical
rows as every other arm.

WHY.  ONE-NAS's headline prediction is an ensemble over its island champions.  A
control that reported a single network would be handicapped by ensemble variance
reduction alone -- a difference that has nothing to do with architecture search.
This script gives the control the same number of members combined the same way.

INPUT   <run>/generation_<g>_elites.csv      island,elite_rank,stock,row,predicted
        <run>/generation_<g>_global_best.csv #expected_N2O,naive_N2O,global_best_predicted_N2O
        (--write_elite_predictions must have been passed; run_control.sh does)

OUTPUT  <out>/generation_<g>_global_best.csv   same 3 columns, the ensemble in the
        third one, so the scorer needs no modification whatsoever
        <out>/run_manifest.json                copied, with the ensemble recorded

MEMBERS.  elite_rank == 0 of each island == that island's champion.  With
--number_islands 16 that is 16 networks, matching the ONE-NAS arm's 16 islands.
`--members all` instead uses every elite of every island.

COMBINER.  *** THIS MUST MATCH THE ONE-NAS ARM EXACTLY. ***  Whatever rule the
ONE-NAS arm's scorer uses to collapse its island champions into one series, the
control must use the same rule, or the comparison is between combiners rather
than between architectures.  Two are implemented:

  mean      arithmetic mean of the members' predictions at each timestep.  The
            natural rule when the downstream metric is nMSE on a level forecast,
            and the rule that is optimal for it.

  rankmean  the equity-campaign rule, transposed to a univariate series.  There
            is no cross-section here, so the ranking is taken WITHIN each test
            window (the 143 emitted timesteps of the window): each member's
            predictions are ranked 1..143 within the window, ranks are averaged
            across members, and the averaged rank is mapped back onto the
            quantiles of the pooled member predictions for that window.  This
            keeps the ensemble on the members' own value scale while combining
            only their orderings.  It is applied per window because that is the
            unit ONE-NAS resets state on and the unit predictions are emitted in.

Both are exposed because v3's ONE-NAS scorer did not exist when this was written;
switching the control is a one-word change to --combiner.
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("run", help="control run directory (contains generation_*_elites.csv)")
ap.add_argument("out", help="output directory for the ensembled prediction files")
ap.add_argument("--combiner", choices=["mean", "rankmean"], default="mean",
                help="MUST match the ONE-NAS arm's combiner")
ap.add_argument("--members", choices=["champions", "all"], default="champions",
                help="champions = elite_rank 0 of each island (one per island)")
a = ap.parse_args()

man_path = os.path.join(a.run, "run_manifest.json")
if not os.path.exists(man_path):
    sys.exit(f"FATAL: {man_path} not found -- is {a.run} a control run directory?")
man = json.load(open(man_path))
ISL = man["params"]["ISL"]
L = man["params"]["L"]

elite_files = sorted(glob.glob(os.path.join(a.run, "generation_*_elites.csv")),
                     key=lambda f: int(re.search(r"generation_(\d+)_", f).group(1)))
if not elite_files:
    sys.exit(f"FATAL: no generation_*_elites.csv in {a.run}.  The run must be launched "
             f"with --write_elite_predictions (run_control.sh does this).")

os.makedirs(a.out, exist_ok=True)


def rank_mean(mat):
    """mat: (n_members, n_rows) predictions for ONE window -> (n_rows,) combined.

    Rank each member within the window, average the ranks, then map the averaged
    rank back onto the quantiles of the pooled member predictions.  Ties are
    handled by 'average' ranking, matching scipy.stats.rankdata's default.
    """
    n_members, n_rows = mat.shape
    order = np.argsort(mat, axis=1, kind="stable")
    ranks = np.empty_like(mat, dtype=float)
    rows = np.arange(n_members)[:, None]
    ranks[rows, order] = np.arange(n_rows, dtype=float)
    # average ranking for ties, per member
    for m in range(n_members):
        s = pd.Series(mat[m])
        ranks[m] = s.rank(method="average").values - 1.0
    mean_rank = ranks.mean(axis=0)                       # in [0, n_rows-1]
    pooled = np.sort(mat.ravel())                        # the members' own value scale
    q = mean_rank / max(n_rows - 1, 1)                   # -> [0, 1]
    return np.quantile(pooled, q)


written, skipped, member_counts = 0, [], []
for ef in elite_files:
    g = int(re.search(r"generation_(\d+)_", ef).group(1))
    gb = os.path.join(a.run, f"generation_{g}_global_best.csv")
    if not os.path.exists(gb):
        skipped.append((g, "no matching global_best file"))
        continue

    d = pd.read_csv(gb)
    d.columns = [c.lstrip("#") for c in d.columns]
    n_rows = len(d)

    e = pd.read_csv(ef)
    if a.members == "champions":
        e = e[e["elite_rank"] == 0]
    e = e[e["stock"] == 0]                               # single series (not pooled panel)

    piv = e.pivot_table(index="island", columns="row", values="predicted", aggfunc="first")
    piv = piv.reindex(columns=range(n_rows))
    if piv.isna().any().any() or len(piv) == 0:
        skipped.append((g, f"incomplete elite matrix ({len(piv)} members, "
                           f"{int(piv.isna().sum().sum())} missing cells)"))
        continue
    mat = piv.values                                     # (n_members, n_rows)
    member_counts.append(mat.shape[0])

    comb = mat.mean(axis=0) if a.combiner == "mean" else rank_mean(mat)

    outdf = pd.DataFrame({
        "#expected_N2O": d["expected_N2O"].values,
        "naive_N2O": d["naive_N2O"].values,
        "global_best_predicted_N2O": comb,
    })
    outdf.to_csv(os.path.join(a.out, f"generation_{g}_global_best.csv"), index=False)
    written += 1

# The scorer reads run_manifest.json from the directory it is pointed at.
man["ensemble"] = {
    "source_run": os.path.abspath(a.run),
    "combiner": a.combiner,
    "members": a.members,
    "expected_members": ISL if a.members == "champions" else None,
    "observed_members_min": int(min(member_counts)) if member_counts else None,
    "observed_members_max": int(max(member_counts)) if member_counts else None,
    "note": ("global_best_predicted_N2O in this directory is the ENSEMBLE, not a single "
             "genome.  The column keeps its name so score_v2.py/score_v3.py need no change."),
}
json.dump(man, open(os.path.join(a.out, "run_manifest.json"), "w"), indent=2)

print(f"combiner {a.combiner}  members {a.members}  islands expected {ISL}")
if member_counts:
    print(f"members per generation: min {min(member_counts)} max {max(member_counts)}")
    if a.members == "champions" and min(member_counts) != ISL:
        print(f"WARNING: expected {ISL} champions, saw as few as {min(member_counts)}")
print(f"wrote {written} ensembled generations to {a.out}")
if skipped:
    print(f"skipped {len(skipped)}: {skipped[:5]}{' ...' if len(skipped) > 5 else ''}")
