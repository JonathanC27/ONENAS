#!/usr/bin/env python3
"""Fixed network budget: is it better to spend it on islands or on seeds?

Every ONE-NAS comparison so far confounds two ways of adding networks. This one
holds the TOTAL network count fixed and varies only how it is partitioned:

    40 networks  =  40 islands x 1 seed
                 =  20 islands x 2 seeds
                 =   8 islands x 5 seeds

    80 networks  =  40 islands x 2 seeds
                 =  20 islands x 4 seeds
                 =   8 islands x 10 seeds

Members from different seeds are independent training runs; members from
different islands of the same run share a data path, a PER buffer, and a
generation clock. The aggregation law (IC_M = IC_1 * sqrt(M / (1 + (M-1)rho)))
says the more-independent partition should win at equal M. If seeds beat islands
at fixed budget, ONE-NAS's width advantage is really a diversity advantage and
the paper should say so; if they tie, island members are effectively independent
and width is genuinely free diversity.

Seed groups are DISJOINT partitions of the ten seeds, redrawn NPART times with a
fixed RNG, so every seed appears exactly once per partition and no seed can be
over-represented (the failure mode that produced the retracted step-0 scaling
curve).
"""

import itertools
import json
import math
import os
import random
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
SEEDS = list(range(42, 52))
TOPKS = (5, 10)
HOLD = 10
FROM, TO = "2020-01-01", "2024-12-31"
NPART = 4                      # independent random partitions per configuration
RNG = random.Random(20260816)  # fixed: reproducible, and not tuned on anything

PAN = {s: Panel(f"{TMP}/{s}_core7", "RET_CS") for s in SETS}

# budget -> list of (islands, seeds_per_group)
BUDGETS = {
    40: [(40, 1), (20, 2), (8, 5)],
    80: [(40, 2), (20, 4), (8, 10)],
}


def isl8_dir(sd):
    return "onenas_c7e" if sd <= 44 else ("probe_s4547" if sd <= 47 else "probe_s4851")


def path_for(width, s, sd):
    if width == 8:
        return f"{isl8_dir(sd)}/{s}_seed{sd}/ensemble_stitched_predictions.csv"
    return f"probe_ISL{width}/{s}_seed{sd}/ensemble_stitched_predictions.csv"


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


def pool4(pp):
    dates = sorted(set.intersection(*[set(v) for v in pp.values()]))
    return np.array([[pp[s][d] for s in SETS] for d in dates]).mean(1)


CACHE = {}
for w in (8, 20, 40):
    c = {}
    for s in SETS:
        for sd in SEEDS:
            c[(s, sd)] = load_preds(path_for(w, s, sd), PAN[s], FROM, TO)
    CACHE[w] = c
    print(f"loaded isl{w}", flush=True)


def group_book(width, group, top_k):
    """Rank-mean ensemble of `group` seeds at `width` islands -> pooled stats."""
    pp, nets, ics = {}, [], []
    for s in SETS:
        by = {sd: CACHE[width][(s, sd)][0] for sd in group}
        if len(group) == 1:
            sd = group[0]
            preds, rows = CACHE[width][(s, sd)]
            ret, ic = book_of(PAN[s], preds, rows, top_k)
        else:
            common = sorted(set.intersection(*[set(v) for v in by.values()]))
            ens = {r: np.mean([rank01(by[sd][r]) for sd in group], axis=0)
                   for r in common}
            ret, ic = book_of(PAN[s], ens, common, top_k)
        pp[s] = ret
        nets.append(100 * sum(ret.values()))
        ics.append(ic)
    pl = pool4(pp)
    return {"net": float(np.mean(nets)), "sharpe": float(sharpe(pl)),
            "mdd": float(mdd(pl)), "ic": float(np.mean(ics))}


def partitions(k):
    """NPART disjoint partitions of the ten seeds into groups of size k."""
    out = []
    for _ in range(NPART):
        pool = SEEDS[:]
        RNG.shuffle(pool)
        out.append([tuple(sorted(pool[i:i + k])) for i in range(0, len(pool), k)])
        if k == len(SEEDS):
            break                      # only one possible grouping
    return out


OUT = {}
for top_k in TOPKS:
    print(f"\n===== FIXED NETWORK BUDGET, top_k {top_k}, H {HOLD}, "
          f"{FROM}..{TO} =====", flush=True)
    OUT[top_k] = {}
    for budget, configs in BUDGETS.items():
        print(f"\n-- {budget} networks --")
        print(f"{'partition':<28} {'groups':>6} {'net':>8} {'±SE':>6} "
              f"{'Sharpe':>7} {'±SE':>6} {'MDD':>6} {'IC':>8}")
        for width, per in configs:
            stats = []
            for part in partitions(per):
                for group in part:
                    stats.append(group_book(width, list(group), top_k))
            nets = np.array([x["net"] for x in stats])
            shs = np.array([x["sharpe"] for x in stats])
            key = f"isl{width}x{per}seed"
            OUT[top_k].setdefault(budget, {})[key] = {
                "n_groups": len(stats),
                "net": float(nets.mean()),
                "net_se": float(nets.std(ddof=1) / math.sqrt(len(nets))) if len(nets) > 1 else None,
                "sharpe": float(shs.mean()),
                "sharpe_se": float(shs.std(ddof=1) / math.sqrt(len(shs))) if len(shs) > 1 else None,
                "mdd": float(np.mean([x["mdd"] for x in stats])),
                "ic": float(np.mean([x["ic"] for x in stats]))}
            r = OUT[top_k][budget][key]
            se_n = f"{r['net_se']:.1f}" if r["net_se"] is not None else "  --"
            se_s = f"{r['sharpe_se']:.3f}" if r["sharpe_se"] is not None else "  --"
            print(f"{f'{width} islands x {per} seed(s)':<28} {len(stats):>6} "
                  f"{r['net']:>+8.1f} {se_n:>6} {r['sharpe']:>7.2f} {se_s:>6} "
                  f"{r['mdd']:>6.1f} {r['ic']:>+8.4f}", flush=True)

json.dump(OUT, open("results_econ/fixed_budget.json", "w"), indent=1)
print("\nWROTE results_econ/fixed_budget.json")
