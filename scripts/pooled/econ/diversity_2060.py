#!/usr/bin/env python3
"""Diversity table (member IC, rho, ensemble IC, theory) for 20/40/60 islands.

score_ensemble.py's per-day `dispersion` column IS the mean pairwise
Spearman correlation of the member cross-sections, i.e. the rho of the
equicorrelated averaging prediction, so the whole table can be rebuilt
from the diagnostics files already on disk. Reproducing the published
40-island row (rho 0.172, member +0.0058, ensemble +0.0128) is the check
that this reads the right column.

    python3 diversity_2060.py
"""
import csv
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SETS = ["set1", "set2", "set3", "set4"]
SEEDS = range(42, 52)
DIAG = "ensemble_diagnostics.csv"
FROM, TO = "2022-01-01", "2024-12-31"

FLEETS = [
    ("20 isl., 2022--24", lambda s, sd: f"probe_ISL20/{s}_seed{sd}/{DIAG}", 20),
    ("40 isl., 2022--24", lambda s, sd: f"probe_ISL40/{s}_seed{sd}/{DIAG}", 40),
    ("60 isl., 2022--24",
     lambda s, sd: f"islands_sweep/islands_60/{s}_seed{sd}/{DIAG}", 60),
]


def main():
    print(f"{'Fleet':>20} {'N':>4} {'rho':>7} {'member':>10} {'ensemble':>10}"
          f" {'predicted':>10} {'ratio':>7}  runs")
    for label, fn, N in FLEETS:
        disp, mem, ens = [], [], []
        runs = 0
        for s in SETS:
            for sd in SEEDS:
                p = os.path.join(HERE, fn(s, sd))
                if not os.path.exists(p):
                    continue
                d, m, e = [], [], []
                with open(p, newline="") as fh:
                    for r in csv.DictReader(fh):
                        # The diagnostics span the whole eval clock; the
                        # table reports 2022-2024, so filter rather than
                        # relabel -- averaging every row under a 2022--24
                        # heading would simply be wrong.
                        if not (FROM <= r.get("date", "") <= TO):
                            continue
                        try:
                            dv = float(r["dispersion"])
                            mv = float(r["mean_member_rank_ic"])
                            ev = float(r["ensemble_rank_ic"])
                        except (KeyError, ValueError):
                            continue
                        if dv == dv and mv == mv and ev == ev:
                            d.append(dv); m.append(mv); e.append(ev)
                if not d:
                    continue
                disp.append(np.mean(d)); mem.append(np.mean(m))
                ens.append(np.mean(e)); runs += 1
        if not runs:
            print(f"{label:>20} {N:>4}   (no data)")
            continue
        rho = float(np.mean(disp))
        m_ic = float(np.mean(mem))
        e_ic = float(np.mean(ens))
        pred = m_ic * math.sqrt(N / (1 + (N - 1) * rho))
        print(f"{label:>20} {N:>4} {rho:>7.3f} {m_ic:>+10.4f} {e_ic:>+10.4f}"
              f" {pred:>+10.4f} {e_ic / pred:>7.3f}  {runs}", flush=True)


if __name__ == "__main__":
    main()
