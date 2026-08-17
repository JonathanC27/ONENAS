#!/usr/bin/env python3
"""The 10-seed comparison table: every arm at one run and at ten runs.

Each arm is booked two ways, so the table separates "what one training run buys"
from "what ten training runs buy":

  1-run     mean over the 10 seeds of that seed's own book (deploy once)
  10-run    rank-mean super-ensemble across all 10 seeds  (deploy ten, combine)

The 10-run column is the like-for-like row across arms: identical seed count,
identical combination rule (seed_ensemble.py's), identical book code. What
differs is how many networks each run contributes -- 1 for a fixed architecture,
`islands` for a ONE-NAS run -- which is the structural asymmetry the paper has
to name rather than hide.

  onenas_single   the global-best genome alone, the ORIGINAL paper's rule.
                  Available at 8 islands / seeds 42-44 ONLY: the raw
                  generation_*_global_best.csv files were purged after scoring
                  for every other configuration. Reported separately at n=3.
"""

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
SEEDS = tuple(range(42, 52))          # 42..51, ten seeds
TOPKS = (5, 10)
HOLD = 10
FROM, TO = "2020-01-01", "2024-12-31"

PAN = {s: Panel(f"{TMP}/{s}_core7", "RET_CS") for s in SETS}


def isl8_dir(sd):
    if sd <= 44:
        return "onenas_c7e"
    if sd <= 47:
        return "probe_s4547"
    return "probe_s4851"


# arm -> (networks contributed per run, path builder)
ARMS = [
    ("onenas_isl8", 8, lambda s, sd: f"{isl8_dir(sd)}/{s}_seed{sd}/ensemble_stitched_predictions.csv"),
    ("onenas_isl20", 20, lambda s, sd: f"probe_ISL20/{s}_seed{sd}/ensemble_stitched_predictions.csv"),
    ("onenas_isl40", 40, lambda s, sd: f"probe_ISL40/{s}_seed{sd}/ensemble_stitched_predictions.csv"),
    ("lstm", 1, lambda s, sd: f"results_econ/lstm/{s}_core7_seed{sd}/predictions.csv"),
    ("gru", 1, lambda s, sd: f"results_econ/gru/{s}_core7_seed{sd}/predictions.csv"),
    ("periodic_lstm_monthly", 1,
     lambda s, sd: f"results_econ/periodic_lstm_monthly/{s}_core7_seed{sd}/predictions.csv"),
]

SINGLE = ("onenas_single", 1, (42, 43, 44),
          lambda s, sd: f"onenas_c7e/{s}_seed{sd}/stitched_predictions.csv")


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


def book_of(panel, preds, rows, top_k):
    days = scoring.build_days(panel, preds, rows)
    bk = ss.run_book(days, scoring.PRED, panel.prc, panel.tc, top_k, None,
                     book="sleeves", hold_days=HOLD)
    dts = [d[1] for d in days]
    ics = [x for x in ss.daily_ics(days, scoring.PRED, ss.spearman) if x == x]
    return dict(zip(dts, bk["daily_ret"])), float(np.mean(ics))


def pool4(per_panel):
    dates = sorted(set.intersection(*[set(v) for v in per_panel.values()]))
    return np.array([[per_panel[s][d] for s in SETS] for d in dates]).mean(1)


def evaluate(name, per_run_nets, pathfn, seeds, top_k, cache):
    """Return (one_run_stats, n_run_ensemble_stats) for one arm at one top_k."""
    # ---- per-seed books
    seed_net, seed_sh, seed_mdd, seed_ic = [], [], [], []
    for sd in seeds:
        pp, ics, nets = {}, [], []
        for s in SETS:
            preds, rows = cache[(s, sd)]
            ret, ic = book_of(PAN[s], preds, rows, top_k)
            pp[s] = ret
            ics.append(ic)
            nets.append(100 * sum(ret.values()))
        pl = pool4(pp)
        seed_net.append(float(np.mean(nets)))
        seed_sh.append(float(sharpe(pl)))
        seed_mdd.append(float(mdd(pl)))
        seed_ic.append(float(np.mean(ics)))
    one = {"nets": per_run_nets,
           "net": float(np.mean(seed_net)),
           "net_se": float(np.std(seed_net, ddof=1) / math.sqrt(len(seed_net))),
           "sharpe": float(np.mean(seed_sh)),
           "sharpe_sd": float(np.std(seed_sh, ddof=1)),
           "mdd": float(np.mean(seed_mdd)),
           "ic": float(np.mean(seed_ic)),
           "worst_sharpe": float(np.min(seed_sh)),
           "per_seed_sharpe": [round(x, 2) for x in seed_sh]}

    # ---- rank-mean super-ensemble over the same seeds
    pp, ics, nets = {}, [], []
    for s in SETS:
        by_seed = {sd: cache[(s, sd)][0] for sd in seeds}
        common = sorted(set.intersection(*[set(v) for v in by_seed.values()]))
        ens = {r: np.mean([rank01(by_seed[sd][r]) for sd in seeds], axis=0)
               for r in common}
        ret, ic = book_of(PAN[s], ens, common, top_k)
        pp[s] = ret
        ics.append(ic)
        nets.append(100 * sum(ret.values()))
    pl = pool4(pp)
    many = {"nets": per_run_nets * len(seeds),
            "net": float(np.mean(nets)),
            "net_se": float(np.std(nets, ddof=1) / math.sqrt(len(nets))),
            "sharpe": float(sharpe(pl)), "mdd": float(mdd(pl)),
            "ic": float(np.mean(ics))}
    return one, many


# ---- load every prediction stream once, reuse across top_k
CACHE = {}
for name, nper, pathfn in ARMS:
    c = {}
    for s in SETS:
        for sd in SEEDS:
            p = pathfn(s, sd)
            if not os.path.exists(p):
                raise SystemExit(f"missing {p}")
            c[(s, sd)] = load_preds(p, PAN[s], FROM, TO)
    CACHE[name] = c
    print(f"loaded {name} ({len(c)} runs)", flush=True)

sname, snper, sseeds, spath = SINGLE
c = {}
for s in SETS:
    for sd in sseeds:
        c[(s, sd)] = load_preds(spath(s, sd), PAN[s], FROM, TO)
CACHE[sname] = c
print(f"loaded {sname} ({len(c)} runs, seeds {sseeds})", flush=True)

OUT = {}
for top_k in TOPKS:
    print(f"\n===== top_k {top_k}, H {HOLD}, sleeves, {FROM}..{TO} =====", flush=True)
    print(f"{'construction':<24} {'nets':>5} | {'1-run net':>10} {'Sh':>5} {'SD':>5} "
          f"{'worst':>6} | {'10-run net':>10} {'±SE':>5} {'Sh':>5} {'MDD':>5} {'IC':>8}")
    print("-" * 104)
    OUT[top_k] = {}
    for name, nper, pathfn in ARMS:
        one, many = evaluate(name, nper, pathfn, SEEDS, top_k, CACHE[name])
        OUT[top_k][name] = {"one_run": one, "ten_run": many}
        print(f"{name:<24} {nper:>5} | {one['net']:>+10.1f} {one['sharpe']:>5.2f} "
              f"{one['sharpe_sd']:>5.3f} {one['worst_sharpe']:>6.2f} | "
              f"{many['net']:>+10.1f} {many['net_se']:>5.1f} {many['sharpe']:>5.2f} "
              f"{many['mdd']:>5.1f} {many['ic']:>+8.4f}", flush=True)
    one, many = evaluate(sname, snper, spath, sseeds, top_k, CACHE[sname])
    OUT[top_k][sname] = {"one_run": one, "three_run": many, "n_seeds": 3}
    print(f"{sname + ' (3 seeds)':<24} {snper:>5} | {one['net']:>+10.1f} "
          f"{one['sharpe']:>5.2f} {one['sharpe_sd']:>5.3f} {one['worst_sharpe']:>6.2f} | "
          f"{many['net']:>+10.1f} {many['net_se']:>5.1f} {many['sharpe']:>5.2f} "
          f"{many['mdd']:>5.1f} {many['ic']:>+8.4f}   <- n=3, NOT 10", flush=True)

json.dump(OUT, open("results_econ/table10.json", "w"), indent=1)
print("\nWROTE results_econ/table10.json")
