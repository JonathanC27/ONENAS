#!/usr/bin/env python3
"""Single global-best genome vs island-champion ensemble, at matched seed count.

The original ONE-NAS paper predicts from ONE genome: the global best. Our
extension predicts from the island champions (one member per island). Every
comparison we have run so far confounds the two, because the ONE-NAS arm is
always the island ensemble while the baseline arm is always a single network
per seed.

This script separates them. For each seed it books:

  onenas_single    the global-best genome alone       (1 network / run)
  onenas_islands   that run's island champions        (8 networks / run)
  lstm, gru        the tuned fixed architecture       (1 network / run)

and then rank-mean ensembles each arm across the SAME seeds, using the exact
rule seed_ensemble.py applies to the baselines. That makes

  onenas_single (N seeds)  vs  lstm/gru (N seeds)

doubly matched: same network count, same number of independent training runs,
same combination rule. The only difference left is whether the architecture was
evolved or fixed -- which is the question the paper actually wants to answer.

  onenas_single (N seeds)  vs  onenas_islands (N seeds)

isolates what the island ensemble buys on top of the published prediction rule.

Availability note: raw generation_*_global_best.csv files survive ONLY for the
registered 8-island runs at seeds 42-44 (they were purged after scoring for the
16/20/40-island sweeps), so this runs at 8 islands, 3 seeds.
"""

import csv
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "baselines"))
sys.path.insert(0, os.path.dirname(HERE))

import scoring                     # noqa: E402
import score_stream as ss          # noqa: E402
from panel import Panel            # noqa: E402
from rebook import load_preds      # noqa: E402

TMP = "/Users/jonathanchang/.claude/jobs/fc17658e/tmp"
SETS = ["set1", "set2", "set3", "set4"]
SEEDS = (42, 43, 44)
TOPKS = (5, 10)
HOLD = 10
FROM, TO = "2020-01-01", "2024-12-31"

PAN = {s: Panel(f"{TMP}/{s}_core7", "RET_CS") for s in SETS}


def rank01(v):
    n = len(v)
    r = np.empty(n)
    r[np.argsort(v, kind="stable")] = np.arange(n, dtype=float)
    return r / (n - 1.0)


def sharpe(v):
    v = np.asarray(v)
    return v.mean() / v.std(ddof=1) * math.sqrt(252)


def mdd(v):
    eq = np.cumsum(np.asarray(v))
    return 100 * np.max(np.maximum.accumulate(eq) - eq)


def pred_path(arm, s, sd):
    if arm == "onenas_single":
        return f"onenas_c7e/{s}_seed{sd}/stitched_predictions.csv"
    if arm == "onenas_islands":
        return f"onenas_c7e/{s}_seed{sd}/ensemble_stitched_predictions.csv"
    return f"results_econ/{arm}/{s}_core7_seed{sd}/predictions.csv"


def book_of(panel, preds, rows, top_k):
    days = scoring.build_days(panel, preds, rows)
    bk = ss.run_book(days, scoring.PRED, panel.prc, panel.tc, top_k, None,
                     book="sleeves", hold_days=HOLD)
    dts = [d[1] for d in days]
    ics = [x for x in ss.daily_ics(days, scoring.PRED, ss.spearman) if x == x]
    return dict(zip(dts, bk["daily_ret"])), float(np.mean(ics))


def pooled(per_panel):
    dates = sorted(set.intersection(*[set(v) for v in per_panel.values()]))
    return np.array([[per_panel[s][d] for s in SETS] for d in dates]).mean(1)


def run_arm(arm, top_k):
    """Return (per-seed stats, seed-ensemble stats) for one arm at one top_k."""
    loaded = {}                                   # (set, seed) -> (preds, rows)
    for s in SETS:
        for sd in SEEDS:
            p = pred_path(arm, s, sd)
            if not os.path.exists(p):
                raise SystemExit(f"missing {p}")
            loaded[(s, sd)] = load_preds(p, PAN[s], FROM, TO)

    # --- each seed on its own (the "1 run" baseline for this arm)
    per_seed = []
    for sd in SEEDS:
        pp, ics, nets = {}, [], []
        for s in SETS:
            preds, rows = loaded[(s, sd)]
            ret, ic = book_of(PAN[s], preds, rows, top_k)
            pp[s] = ret
            ics.append(ic)
            nets.append(100 * sum(ret.values()))
        pl = pooled(pp)
        per_seed.append({"seed": sd, "net": float(np.mean(nets)),
                         "sharpe": float(sharpe(pl)), "mdd": float(mdd(pl)),
                         "ic": float(np.mean(ics))})

    # --- rank-mean across the same seeds (seed_ensemble.py's rule)
    pp, ics, nets = {}, [], []
    for s in SETS:
        by_seed = {sd: loaded[(s, sd)][0] for sd in SEEDS}
        common = sorted(set.intersection(*[set(v) for v in by_seed.values()]))
        ens = {r: np.mean([rank01(by_seed[sd][r]) for sd in SEEDS], axis=0)
               for r in common}
        ret, ic = book_of(PAN[s], ens, common, top_k)
        pp[s] = ret
        ics.append(ic)
        nets.append(100 * sum(ret.values()))
    pl = pooled(pp)
    ens_stats = {"net": float(np.mean(nets)),
                 "net_se": float(np.std(nets, ddof=1) / math.sqrt(len(nets))),
                 "sharpe": float(sharpe(pl)), "mdd": float(mdd(pl)),
                 "ic": float(np.mean(ics))}
    return per_seed, ens_stats


ARMS = ["onenas_single", "onenas_islands", "lstm", "gru"]
NETWORKS = {"onenas_single": 1, "onenas_islands": 8, "lstm": 1, "gru": 1}
OUT = {}

for top_k in TOPKS:
    print(f"\n===== top_k {top_k}, H {HOLD}, sleeves, {FROM}..{TO}, "
          f"seeds {','.join(str(x) for x in SEEDS)} =====", flush=True)
    print(f"{'arm':<16} {'nets/run':>8} {'mean 1-run net':>15} {'1-run Sh':>9} "
          f"{'3-seed net':>11} {'3-seed Sh':>10} {'MDD':>6} {'IC':>8}")
    OUT[top_k] = {}
    for arm in ARMS:
        per_seed, ens = run_arm(arm, top_k)
        m_net = float(np.mean([x["net"] for x in per_seed]))
        m_sh = float(np.mean([x["sharpe"] for x in per_seed]))
        OUT[top_k][arm] = {"per_seed": per_seed, "ensemble": ens,
                           "mean_single_run_net": m_net,
                           "mean_single_run_sharpe": m_sh}
        print(f"{arm:<16} {NETWORKS[arm]*len(SEEDS):>8} {m_net:>+15.1f} "
              f"{m_sh:>9.2f} {ens['net']:>+11.1f} {ens['sharpe']:>10.2f} "
              f"{ens['mdd']:>6.1f} {ens['ic']:>+8.4f}", flush=True)

json.dump(OUT, open("results_econ/single_champion.json", "w"), indent=1)
print("\nWROTE results_econ/single_champion.json")
