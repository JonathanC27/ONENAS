#!/usr/bin/env python3
"""Post-hoc argmax-over-k: isolates SELECTION-ON-NOISE from search dynamics.

The island-count result (champion Sharpe FALLS as islands rise: 8->20 dSharpe
-0.347 t=-3.83) is open to a confound. Raising island count changes migration
topology, effective mutation pressure and diversity maintenance -- not only the
size of the candidate pool. A referee can argue the degradation is a
search-dynamics artefact rather than selection on a noisy criterion.

This control removes the confound entirely. The candidate POOL IS FIXED: ten
independently seeded runs at one island count, nothing about the search varies.
The only thing that changes is k, the number of candidates we take the argmax
over. For each k we draw many random subsets of size k, pick the subset member
that looks best on a SELECTION window, and score that pick on a disjoint
EVALUATION window. Alongside it we score the plain AVERAGE of the same k.

  selection window   2020-01-01 .. 2021-12-31
  evaluation window  2022-01-01 .. 2024-12-31   (disjoint, strictly later)

Predictions:
  * If the selection criterion carries information, argmax-over-k IMPROVES with k
    -- more candidates means a better pick.
  * If it is noise, argmax-over-k is FLAT or DECLINES with k (oversearching,
    Quinlan & Cameron-Jones IJCAI 1995) while the mean-of-k improves, because
    averaging is unaffected by the criterion's quality.

No new training runs: this reuses prediction streams already on disk.
"""

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
TOPK = 10
HOLD = 10
SEL_FROM, SEL_TO = "2020-01-01", "2021-12-31"
EVA_FROM, EVA_TO = "2022-01-01", "2024-12-31"
NDRAW = 200
RNG = random.Random(20260817)

PAN = {s: Panel(f"{TMP}/{s}_core7", "RET_CS") for s in SETS}

WIDTHS = [("40 islands", "probe_ISL40"), ("20 islands", "probe_ISL20")]


def isl8(sd):
    return "onenas_c7e" if sd <= 44 else ("probe_s4547" if sd <= 47 else "probe_s4851")


def sharpe(v):
    v = np.asarray(v)
    if len(v) < 3 or v.std(ddof=1) == 0:
        return float("nan")
    return v.mean() / v.std(ddof=1) * math.sqrt(252)


def rank01(v):
    n = len(v)
    r = np.empty(n)
    r[np.argsort(v, kind="stable")] = np.arange(n, dtype=float)
    return r / (n - 1.0)


def book(panel, preds, rows):
    days = scoring.build_days(panel, preds, rows)
    bk = ss.run_book(days, scoring.PRED, panel.prc, panel.tc, TOPK, None,
                     book="sleeves", hold_days=HOLD)
    return dict(zip([d[1] for d in days], bk["daily_ret"]))


OUT = {}
for label, root in WIDTHS:
    # ---- load once per (panel, seed), both windows
    raw = {}
    for s in SETS:
        for sd in SEEDS:
            p = f"{root}/{s}_seed{sd}/ensemble_stitched_predictions.csv"
            if not os.path.exists(p):
                raise SystemExit(f"missing {p}")
            raw[(s, sd, "sel")] = load_preds(p, PAN[s], SEL_FROM, SEL_TO)
            raw[(s, sd, "eva")] = load_preds(p, PAN[s], EVA_FROM, EVA_TO)
    print(f"loaded {label}", flush=True)

    # ---- per-seed pooled Sharpe on each window (the selection score and the truth)
    sel_score, eva_score, eva_ret = {}, {}, {}
    for sd in SEEDS:
        for tag, store in (("sel", sel_score), ("eva", eva_score)):
            pp = {}
            for s in SETS:
                preds, rows = raw[(s, sd, tag)]
                pp[s] = book(PAN[s], preds, rows)
            common = sorted(set.intersection(*[set(v) for v in pp.values()]))
            pool = np.array([[pp[s][d] for s in SETS] for d in common]).mean(1)
            store[sd] = float(sharpe(pool))
            if tag == "eva":
                eva_ret[sd] = dict(zip(common, pool))

    # ---- mean-of-k needs the ENSEMBLE of k seeds, rebooked (not an average of Sharpes)
    def mean_of_k(group):
        pp = {}
        for s in SETS:
            by = {sd: raw[(s, sd, "eva")][0] for sd in group}
            rows = sorted(set.intersection(*[set(v) for v in by.values()]))
            ens = {r: np.mean([rank01(by[sd][r]) for sd in group], axis=0)
                   for r in rows}
            pp[s] = book(PAN[s], ens, rows)
        common = sorted(set.intersection(*[set(v) for v in pp.values()]))
        return float(sharpe(np.array([[pp[s][d] for s in SETS]
                                      for d in common]).mean(1)))

    print(f"\n=== {label}: fixed pool of {len(SEEDS)} runs, argmax over k ===")
    print(f"selection window {SEL_FROM}..{SEL_TO}, "
          f"evaluation window {EVA_FROM}..{EVA_TO}, {NDRAW} draws per k")
    print(f"{'k':>3} {'argmax-of-k':>12} {'±SE':>6} {'mean-of-k':>11} {'±SE':>6} "
          f"{'random pick':>12}")
    print("-" * 56)
    OUT[label] = {}
    for k in (1, 2, 3, 5, 7, 10):
        picks, means, rands = [], [], []
        draws = NDRAW if k < len(SEEDS) else 1
        for _ in range(draws):
            g = RNG.sample(SEEDS, k) if k < len(SEEDS) else SEEDS[:]
            best = max(g, key=lambda sd: sel_score[sd])
            picks.append(eva_score[best])
            rands.append(eva_score[RNG.choice(g)])
            means.append(mean_of_k(g) if k > 1 else eva_score[g[0]])
        f = lambda a: (float(np.mean(a)),
                       float(np.std(a, ddof=1) / math.sqrt(len(a))) if len(a) > 1 else 0.0)
        pm, ps = f(picks); mm, msd = f(means); rm, _ = f(rands)
        OUT[label][k] = {"argmax": pm, "argmax_se": ps, "mean": mm,
                         "mean_se": msd, "random": rm}
        print(f"{k:>3} {pm:>12.3f} {ps:>6.3f} {mm:>11.3f} {msd:>6.3f} {rm:>12.3f}")

    a1, a10 = OUT[label][1]["argmax"], OUT[label][10]["argmax"]
    m1, m10 = OUT[label][1]["mean"], OUT[label][10]["mean"]
    print(f"\n  argmax k=1 -> k=10: {a1:+.3f} -> {a10:+.3f}  ({a10-a1:+.3f})")
    print(f"  mean   k=1 -> k=10: {m1:+.3f} -> {m10:+.3f}  ({m10-m1:+.3f})")
    print(f"  selection-vs-truth rank corr across the pool: ", end="")
    xs = [sel_score[sd] for sd in SEEDS]
    ys = [eva_score[sd] for sd in SEEDS]
    rx, ry = rank01(np.array(xs)), rank01(np.array(ys))
    print(f"{float(np.corrcoef(rx, ry)[0,1]):+.3f}")

json.dump(OUT, open("results_econ/argmax_over_k.json", "w"), indent=1)
print("\nWROTE results_econ/argmax_over_k.json")
