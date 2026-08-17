#!/usr/bin/env python3
"""THE HEADLINE TABLE: run-for-run, with search cost as a first-class column.

Comparison unit = ONE TRAINING RUN. This is the DARTS / Regularized-Evolution
convention for searched-vs-hand-designed comparisons: the searched method
carries its search cost in a column, the hand-designed baselines carry theirs,
and the headline claim is made on final performance rather than on matched
compute. Lindauer & Hutter's NAS best-practice checklist (JMLR 2020) requires
the cost disclosure; it does NOT require compute-matching against a manual
baseline.

What one run yields differs by method, and the table says so explicitly:
  ONE-NAS       `islands` trained networks from one invocation
  LSTM/GRU/PLM  one trained network from one invocation

That asymmetry IS the claim -- see matched_width.py for the network-matched
counterweight, where the advantage disappears (net +0.45, p=0.97). The honest
statement is amortisation over NETWORKS (one run -> many members), never
amortisation over TIME: one ONE-NAS generation is window_step = 5 trading days,
so the search is the online learning and its cost recurs for the life of the
deployment.

Two fixes over the earlier run-matched table:
  * SHARED CALENDAR. ONE-NAS streams end 2024-12-17/20, the baselines
    2024-12-31, so earlier cells compared 1248 vs 1257 days. Every arm is now
    booked on the intersection of all arms' dates.
  * MDD IS THE 1-RUN MDD. The earlier table printed each baseline's TEN-RUN
    ensemble MDD next to ONE-NAS's 1-run MDD -- a column mix-up that made
    ONE-NAS look like the lowest-drawdown arm. Run for run it is not
    (gru 18.9, plm 16.3 vs ONE-NAS-40 20.4).

Search cost is Anvil-core-hours per 4-panel run, measured (not estimated):
ONE-NAS from `time srun` in the job .err files x 128 charged cores; baselines
from meta.json fit_seconds, converted at the measured M2->EPYC-7763 factor of
1.905 (same binary, same panel, same config, both machines). The "minimum"
figure for ONE-NAS reflects the measured 16.6x over-provisioning of the 128-core
allocation: 10 MPI workers run the identical config as fast as 127, because 64%
of wall time is serial elite evaluation on rank 0.
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
SEEDS = tuple(range(42, 52))
TOPKS = (5, 10)
HOLD = 10
FROM, TO = "2020-01-01", "2024-12-31"

PAN = {s: Panel(f"{TMP}/{s}_core7", "RET_CS") for s in SETS}


def isl8_dir(sd):
    return "onenas_c7e" if sd <= 44 else ("probe_s4547" if sd <= 47 else "probe_s4851")


# label -> (nets per run, charged core-h, minimum core-h or None, path builder)
ARMS = [
    ("ONE-NAS 40 islands", 40, 785.0, 48.0,
     lambda s, sd: f"probe_ISL40/{s}_seed{sd}/ensemble_stitched_predictions.csv"),
    ("ONE-NAS 20 islands", 20, 450.0, None,
     lambda s, sd: f"probe_ISL20/{s}_seed{sd}/ensemble_stitched_predictions.csv"),
    ("ONE-NAS 8 islands", 8, 300.0, None,
     lambda s, sd: f"{isl8_dir(sd)}/{s}_seed{sd}/ensemble_stitched_predictions.csv"),
    ("online LSTM", 1, 0.0295, None,
     lambda s, sd: f"results_econ/lstm/{s}_core7_seed{sd}/predictions.csv"),
    ("online GRU", 1, 0.0258, None,
     lambda s, sd: f"results_econ/gru/{s}_core7_seed{sd}/predictions.csv"),
    ("periodic-LSTM monthly", 1, 0.765, None,
     lambda s, sd: f"results_econ/periodic_lstm_monthly/{s}_core7_seed{sd}/predictions.csv"),
]


def sharpe(v):
    v = np.asarray(v)
    return v.mean() / v.std(ddof=1) * math.sqrt(252)


def mdd(v):
    eq = np.cumsum(np.asarray(v))
    return 100 * np.max(np.maximum.accumulate(eq) - eq)


# ---- pass 1: load everything, find the shared calendar
RAW = {}
for label, nper, cost, cmin, pathfn in ARMS:
    for s in SETS:
        for sd in SEEDS:
            p = pathfn(s, sd)
            if not os.path.exists(p):
                raise SystemExit(f"missing {p}")
            RAW[(label, s, sd)] = load_preds(p, PAN[s], FROM, TO)

# a "date" here is a panel row index; intersect per panel across all arms/seeds
SHARED = {}
for s in SETS:
    sets_ = [set(RAW[(lab, s, sd)][0]) for lab, _, _, _, _ in ARMS for sd in SEEDS]
    SHARED[s] = sorted(set.intersection(*sets_))
    print(f"{s}: shared rows {len(SHARED[s])} "
          f"({PAN[s].dates[SHARED[s][0]]} .. {PAN[s].dates[SHARED[s][-1]]})",
          flush=True)


def book_of(panel, preds, rows, top_k):
    days = scoring.build_days(panel, preds, rows)
    bk = ss.run_book(days, scoring.PRED, panel.prc, panel.tc, top_k, None,
                     book="sleeves", hold_days=HOLD)
    dts = [d[1] for d in days]
    ics = [x for x in ss.daily_ics(days, scoring.PRED, ss.spearman) if x == x]
    return dict(zip(dts, bk["daily_ret"])), float(np.mean(ics))


OUT = {}
for top_k in TOPKS:
    print(f"\n{'='*96}")
    print(f"RUN-FOR-RUN, top_k {top_k}, H {HOLD}, sleeves book, shared calendar, "
          f"{len(SEEDS)} seeds")
    print(f"{'='*96}")
    print(f"{'method':<22} {'nets':>5} {'search':>9} {'net %':>9} {'±SE':>6} "
          f"{'Sharpe':>7} {'SD':>6} {'worst':>6} {'MDD':>6} {'IC':>8}")
    print(f"{'':22} {'/run':>5} {'core-h':>9}")
    print("-" * 96)
    OUT[top_k] = {}
    for label, nper, cost, cmin, pathfn in ARMS:
        nets, shs, mds, ics = [], [], [], []
        for sd in SEEDS:
            pp, pn, pi = {}, [], []
            for s in SETS:
                preds, _ = RAW[(label, s, sd)]
                ret, ic = book_of(PAN[s], preds, SHARED[s], top_k)
                pp[s] = ret
                pn.append(100 * sum(ret.values()))
                pi.append(ic)
            dates = sorted(set.intersection(*[set(v) for v in pp.values()]))
            pool = np.array([[pp[s][d] for s in SETS] for d in dates]).mean(1)
            nets.append(float(np.mean(pn)))
            shs.append(float(sharpe(pool)))
            mds.append(float(mdd(pool)))
            ics.append(float(np.mean(pi)))
        r = {"nets_per_run": nper, "core_h": cost, "core_h_min": cmin,
             "net": float(np.mean(nets)),
             "net_se": float(np.std(nets, ddof=1) / math.sqrt(len(nets))),
             "sharpe": float(np.mean(shs)),
             "sharpe_sd": float(np.std(shs, ddof=1)),
             "worst_sharpe": float(np.min(shs)),
             "mdd": float(np.mean(mds)), "ic": float(np.mean(ics)),
             "per_seed_sharpe": [round(x, 3) for x in shs],
             "per_seed_net": [round(x, 1) for x in nets]}
        OUT[top_k][label] = r
        cs = f"{cost:,.0f}" if cost >= 1 else f"{cost:.3f}"
        print(f"{label:<22} {nper:>5} {cs:>9} {r['net']:>+9.1f} {r['net_se']:>6.1f} "
              f"{r['sharpe']:>7.2f} {r['sharpe_sd']:>6.3f} {r['worst_sharpe']:>6.2f} "
              f"{r['mdd']:>6.1f} {r['ic']:>+8.4f}", flush=True)

    # seed-level (E1) contrasts vs each baseline: Welch on the 10 seeds
    ref = "ONE-NAS 40 islands"
    print(f"\n  {ref} vs each baseline (Welch over {len(SEEDS)} seeds, "
          f"conditional on the 2020-24 path):")
    a_net = np.array(OUT[top_k][ref]["per_seed_net"])
    a_sh = np.array(OUT[top_k][ref]["per_seed_sharpe"])
    for label, nper, _, _, _ in ARMS:
        if nper != 1:
            continue
        b_net = np.array(OUT[top_k][label]["per_seed_net"])
        b_sh = np.array(OUT[top_k][label]["per_seed_sharpe"])

        def welch(x, y):
            n1, n2 = len(x), len(y)
            se = math.sqrt(x.var(ddof=1) / n1 + y.var(ddof=1) / n2)
            return (x.mean() - y.mean()) / se if se > 0 else float("nan")
        print(f"    vs {label:<24} dNet {a_net.mean()-b_net.mean():>+6.1f} "
              f"(t={welch(a_net, b_net):>+5.2f})   "
              f"dSharpe {a_sh.mean()-b_sh.mean():>+5.2f} "
              f"(t={welch(a_sh, b_sh):>+5.2f})", flush=True)

json.dump(OUT, open("results_econ/headline_table.json", "w"), indent=1)
print("\nWROTE results_econ/headline_table.json")
