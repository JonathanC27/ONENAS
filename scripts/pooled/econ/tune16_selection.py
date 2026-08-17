#!/usr/bin/env python3
"""Amendment 6: protocol-symmetric island-width selection on the 2016-2019 span.

Selection objective is mean daily rank IC over the tuning span -- the exact span
and objective protocol.py fixes for every baseline's hyperparameters -- so the
width choice is made without touching post-2019 data.

Per-seed val IC is the mean of the four panels' mean daily ensemble rank IC.
Across seeds we report mean, SE, and spread (SD), plus a paired t vs 8 islands.
"""

import csv
import math
import os

import numpy as np

SETS = ["set1", "set2", "set3", "set4"]
ARMS = [(8, "probe_TUNE16_ISL8"), (16, "probe_TUNE16_ISL16"),
        (20, "probe_TUNE16_ISL20"), (40, "probe_TUNE16_ISL40")]


def seed_ic(d, s, sd):
    p = f"{d}/{s}_seed{sd}/ensemble_diagnostics.csv"
    if not os.path.exists(p):
        return None
    vals = []
    with open(p) as fh:
        for r in csv.DictReader(fh):
            v = float(r["ensemble_rank_ic"])
            if v == v:
                vals.append(v)
    return float(np.mean(vals)) if vals else None


def arm_per_seed(d):
    """seed -> val IC averaged over the four panels (only fully-present seeds)."""
    seeds = sorted({int(x.split("seed")[1]) for x in os.listdir(d)
                    if "seed" in x})
    out = {}
    for sd in seeds:
        ics = [seed_ic(d, s, sd) for s in SETS]
        if all(x is not None for x in ics):
            out[sd] = float(np.mean(ics))
    return out


per_arm = {n: arm_per_seed(d) for n, d in ARMS}
base = per_arm[8]

print("# AMENDMENT 6 RESULT (complete)")
print("# Selection objective: mean daily rank IC on the 2016-2019 tuning span,")
print("# the exact span and objective on which every baseline's hyperparameters")
print("# were tuned (protocol.py). No post-2019 data is consulted.")
print("# islands seeds   val IC      ±SE     seed spread")
rows = {}
for n, _ in ARMS:
    v = np.array(list(per_arm[n].values()))
    m, sd_ = float(v.mean()), float(v.std(ddof=1))
    rows[n] = {"seeds": len(v), "mean": m, "se": sd_ / math.sqrt(len(v)),
               "spread": sd_}
    print(f"#    {n:>3}   {len(v):>3}   {m:+.5f}  {sd_/math.sqrt(len(v)):.5f}    {sd_:.5f}")

print("# Paired vs 8 islands (seeds present in both arms):")
best_n, best_m = None, -9
for n, _ in ARMS:
    if n == 8:
        continue
    common = sorted(set(per_arm[n]) & set(base))
    d = np.array([per_arm[n][k] - base[k] for k in common])
    t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))
    print(f"#   {n:>3} isl  dIC {d.mean():+.5f}  t={t:.2f}  (n={len(d)} seeds)")
    if rows[n]["mean"] > best_m:
        best_m, best_n = rows[n]["mean"], n
if rows[8]["mean"] > best_m:
    best_n = 8
print(f"# SELECTED: {best_n} islands (highest val IC on the tuning span)")
print("# per-seed val IC:")
for n, _ in ARMS:
    s = "  ".join(f"{k}:{v:+.5f}" for k, v in sorted(per_arm[n].items()))
    print(f"#   {n:>3} isl  {s}")
