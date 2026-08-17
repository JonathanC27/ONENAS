#!/usr/bin/env python3
"""Score the no-search controls through the REGISTERED book.

NAS Best Practice 8 (Lindauer & Hutter, JMLR 2020) makes comparison against
random search the one genuinely mandatory fairness item, and the checklist is
what a referee applies. scripts/pooled/controls/ already contains both controls
and 12 core7 runs each (4 panels x seeds 42-44) -- but they were booked at H=1
in their own harness and never entered the evidence map, so the ablation exists
and has never been reported.

This rescores them through the same sleeves book as every other arm
(top_k in {5,10}, H=10, netted costs) so they are directly comparable, at
MATCHED WIDTH (8 members) and MATCHED SEEDS (42-44) against ONE-NAS at 8
islands -- the pre-registered configuration.

  control1 ensemble  8 architectures sampled ONCE at t=0 from a documented
                     distribution, never mutated; rank-mean of all 8.
                     Isolates: does the SEARCH contribute, or just the
                     pipeline + target + architectural heterogeneity?
  control2 ensemble  8 copies of ONE tuned architecture, different inits;
                     rank-mean. The deep-ensemble baseline (Lakshminarayanan
                     et al. 2017). Isolates: does architectural DIVERSITY
                     contribute beyond plain seed ensembling?
  control1/2 single  the trailing-MSE-champion variant of each, i.e. the
                     selection analogue of ONE-NAS's global best.

If ONE-NAS does not beat control1, the neuroevolution is not load-bearing.
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
CTRL = os.path.join(os.path.dirname(HERE), "controls", "results_controls")
SETS = ["set1", "set2", "set3", "set4"]
SEEDS = (42, 43, 44)               # the seeds the controls were run at
TOPKS = (5, 10)
HOLD = 10
FROM, TO = "2020-01-01", "2024-12-31"

PAN = {s: Panel(f"{TMP}/{s}_core7", "RET_CS") for s in SETS}

ARMS = [
    ("control1_ensemble", lambda s, sd: f"{CTRL}/control1/{s}_core7_seed{sd}/ensemble/predictions.csv"),
    ("control1_single",   lambda s, sd: f"{CTRL}/control1/{s}_core7_seed{sd}/single/predictions.csv"),
    ("control2_ensemble", lambda s, sd: f"{CTRL}/control2/{s}_core7_seed{sd}/ensemble/predictions.csv"),
    ("control2_single",   lambda s, sd: f"{CTRL}/control2/{s}_core7_seed{sd}/single/predictions.csv"),
    ("onenas_isl8",       lambda s, sd: f"onenas_c7e/{s}_seed{sd}/ensemble_stitched_predictions.csv"),
    ("onenas_single",     lambda s, sd: f"onenas_c7e/{s}_seed{sd}/stitched_predictions.csv"),
    ("lstm",              lambda s, sd: f"results_econ/lstm/{s}_core7_seed{sd}/predictions.csv"),
    ("gru",               lambda s, sd: f"results_econ/gru/{s}_core7_seed{sd}/predictions.csv"),
]


def sharpe(v):
    v = np.asarray(v)
    return v.mean() / v.std(ddof=1) * math.sqrt(252)


def mdd(v):
    eq = np.cumsum(np.asarray(v))
    return 100 * np.max(np.maximum.accumulate(eq) - eq)


def book_of(panel, preds, rows, top_k):
    days = scoring.build_days(panel, preds, rows)
    bk = ss.run_book(days, scoring.PRED, panel.prc, panel.tc, top_k, None,
                     book="sleeves", hold_days=HOLD)
    dts = [d[1] for d in days]
    ics = [x for x in ss.daily_ics(days, scoring.PRED, ss.spearman) if x == x]
    return dict(zip(dts, bk["daily_ret"])), float(np.mean(ics))


OUT = {}
for top_k in TOPKS:
    print(f"\n===== NO-SEARCH CONTROLS, top_k {top_k}, H {HOLD}, sleeves, "
          f"{FROM}..{TO}, seeds {SEEDS} =====", flush=True)
    print(f"{'arm':<20} {'net':>8} {'±SE':>6} {'Sharpe':>7} {'±SD':>6} "
          f"{'MDD':>6} {'IC':>9}")
    print("-" * 70)
    OUT[top_k] = {}
    for name, pathfn in ARMS:
        nets, shs, mds, ics = [], [], [], []
        per_seed_pool = {}
        ok = True
        for sd in SEEDS:
            pp, pn, pi = {}, [], []
            for s in SETS:
                p = pathfn(s, sd)
                if not os.path.exists(p):
                    ok = False
                    break
                preds, rows = load_preds(p, PAN[s], FROM, TO)
                ret, ic = book_of(PAN[s], preds, rows, top_k)
                pp[s] = ret
                pn.append(100 * sum(ret.values()))
                pi.append(ic)
            if not ok:
                break
            dates = sorted(set.intersection(*[set(v) for v in pp.values()]))
            pool = np.array([[pp[s][d] for s in SETS] for d in dates]).mean(1)
            per_seed_pool[sd] = dict(zip(dates, pool))
            nets.append(float(np.mean(pn)))
            shs.append(float(sharpe(pool)))
            mds.append(float(mdd(pool)))
            ics.append(float(np.mean(pi)))
        if not ok:
            print(f"{name:<20} --- missing runs")
            continue
        OUT[top_k][name] = {
            "net": float(np.mean(nets)),
            "net_se": float(np.std(nets, ddof=1) / math.sqrt(len(nets))),
            "sharpe": float(np.mean(shs)),
            "sharpe_sd": float(np.std(shs, ddof=1)),
            "mdd": float(np.mean(mds)), "ic": float(np.mean(ics)),
            "per_seed_sharpe": [round(x, 3) for x in shs]}
        r = OUT[top_k][name]
        print(f"{name:<20} {r['net']:>+8.1f} {r['net_se']:>6.1f} "
              f"{r['sharpe']:>7.2f} {r['sharpe_sd']:>6.3f} {r['mdd']:>6.1f} "
              f"{r['ic']:>+9.4f}", flush=True)

    # paired contrasts vs ONE-NAS at matched width/seeds
    base = OUT[top_k].get("onenas_isl8")
    if base:
        print(f"\n  paired vs ONE-NAS 8-island (n={len(SEEDS)} seeds):")
        for name in ("control1_ensemble", "control2_ensemble"):
            if name not in OUT[top_k]:
                continue
            d = np.array(base["per_seed_sharpe"]) - np.array(OUT[top_k][name]["per_seed_sharpe"])
            t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d))) if d.std(ddof=1) > 0 else float("nan")
            print(f"    ONE-NAS - {name}: dSharpe {d.mean():+.3f}  t={t:+.2f}")

json.dump(OUT, open("results_econ/controls_scored.json", "w"), indent=1)
print("\nWROTE results_econ/controls_scored.json")
