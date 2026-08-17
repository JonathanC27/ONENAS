#!/usr/bin/env python3
"""Network-matched comparison: ONE-NAS at N islands vs an N-seed baseline ensemble.

table10.py matches the number of RUNS. This matches the number of NETWORKS,
which is the comparison a reviewer constructs if we don't:

  ONE-NAS, N islands, ONE run        -> N networks, all from one training run
  LSTM/GRU/PLM, N-seed rank-mean     -> N networks, from N independent runs

Both sides book N networks through identical code. The baselines' N members are
strictly more independent (separate runs, separate initialisations), so this
column is the conservative one for ONE-NAS -- it hands the aggregation law's
main lever to the baseline. Reporting it is the honest counterweight to the
run-matched column, where ONE-NAS's within-run population is free.

ONE-NAS side is the mean over 10 seeds of each seed's own single-run book, so
it is what ONE training run actually delivers, with a seed-dispersion SD.
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
ONENAS_SEEDS = tuple(range(42, 52))
WIDTHS = (8, 20, 40)
BASE_ARMS = ("lstm", "gru", "periodic_lstm_monthly")
TOPKS = (5, 10)
HOLD = 10
FROM, TO = "2020-01-01", "2024-12-31"

PAN = {s: Panel(f"{TMP}/{s}_core7", "RET_CS") for s in SETS}


def isl8_dir(sd):
    return "onenas_c7e" if sd <= 44 else ("probe_s4547" if sd <= 47 else "probe_s4851")


def onenas_path(width, s, sd):
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


# ---------- load baselines, using only seeds present on ALL FOUR panels
# (the killed 48-seed job left set4 short, so availability differs per arm)
BASE, BASE_SEEDS = {}, {}
for arm in BASE_ARMS:
    common = None
    for s in SETS:
        have = {sd for sd in range(42, 140)
                if os.path.exists(f"results_econ/{arm}/{s}_core7_seed{sd}/predictions.csv")}
        common = have if common is None else (common & have)
    seeds = sorted(common)
    BASE_SEEDS[arm] = seeds
    c = {}
    for s in SETS:
        for sd in seeds:
            c[(s, sd)] = load_preds(
                f"results_econ/{arm}/{s}_core7_seed{sd}/predictions.csv",
                PAN[s], FROM, TO)
    BASE[arm] = c
    print(f"loaded {arm} ({len(seeds)} seeds x 4 panels: "
          f"{seeds[0]}..{seeds[-1]})", flush=True)

ONE = {}
for w in WIDTHS:
    c = {}
    for s in SETS:
        for sd in ONENAS_SEEDS:
            c[(s, sd)] = load_preds(onenas_path(w, s, sd), PAN[s], FROM, TO)
    ONE[w] = c
    print(f"loaded onenas_isl{w} (10 seeds x 4 panels)", flush=True)


def onenas_one_run(width, top_k):
    nets, shs, mds, ics = [], [], [], []
    for sd in ONENAS_SEEDS:
        pp, panel_nets, panel_ics = {}, [], []
        for s in SETS:
            preds, rows = ONE[width][(s, sd)]
            ret, ic = book_of(PAN[s], preds, rows, top_k)
            pp[s] = ret
            panel_nets.append(100 * sum(ret.values()))
            panel_ics.append(ic)
        pl = pool4(pp)
        nets.append(float(np.mean(panel_nets)))
        shs.append(float(sharpe(pl)))
        mds.append(float(mdd(pl)))
        ics.append(float(np.mean(panel_ics)))
    return {"net": float(np.mean(nets)),
            "net_se": float(np.std(nets, ddof=1) / math.sqrt(len(nets))),
            "sharpe": float(np.mean(shs)),
            "sharpe_sd": float(np.std(shs, ddof=1)),
            "mdd": float(np.mean(mds)), "ic": float(np.mean(ics))}


def baseline_ens(arm, n, top_k, seeds=None):
    """N-network ensemble; returns None if this arm has fewer than N seeds.

    `seeds` must be supplied by the caller, which averages over several random
    draws -- see baseline_ens_mean. Taking the FIRST n seeds (the original
    implementation) is a single draw and is not comparable to the ONE-NAS cells,
    which are means over 10 seeds; the first-8 LSTM draw happened to sit ~2.3 sd
    below its own subset mean, which made the baseline look far weaker at narrow
    widths than it is.
    """
    if len(BASE_SEEDS[arm]) < n:
        return None
    if seeds is None:
        seeds = BASE_SEEDS[arm][:n]
    pp, nets, ics = {}, [], []
    for s in SETS:
        by = {sd: BASE[arm][(s, sd)][0] for sd in seeds}
        common = sorted(set.intersection(*[set(v) for v in by.values()]))
        ens = {r: np.mean([rank01(by[sd][r]) for sd in seeds], axis=0)
               for r in common}
        ret, ic = book_of(PAN[s], ens, common, top_k)
        pp[s] = ret
        nets.append(100 * sum(ret.values()))
        ics.append(ic)
    pl = pool4(pp)
    return {"net": float(np.mean(nets)),
            "net_se": float(np.std(nets, ddof=1) / math.sqrt(len(nets))),
            "sharpe": float(sharpe(pl)), "sharpe_sd": None,
            "mdd": float(mdd(pl)), "ic": float(np.mean(ics))}


NDRAW = 8
DRAW_RNG = random.Random(20260817)


def baseline_ens_mean(arm, n, top_k):
    """Mean over NDRAW random n-seed draws -- the like-for-like counterpart of
    the ONE-NAS cells, which average over 10 independent runs. When n equals the
    full pool there is only one possible draw."""
    pool = BASE_SEEDS[arm]
    if len(pool) < n:
        return None
    if len(pool) == n:
        r = baseline_ens(arm, n, top_k, seeds=pool)
        r["n_draws"] = 1
        r["net_sd_draw"] = 0.0
        r["sharpe_sd_draw"] = 0.0
        return r
    draws = []
    for _ in range(NDRAW):
        p = pool[:]
        DRAW_RNG.shuffle(p)
        draws.append(baseline_ens(arm, n, top_k, seeds=sorted(p[:n])))
    nets = np.array([d["net"] for d in draws])
    shs = np.array([d["sharpe"] for d in draws])
    return {"net": float(nets.mean()),
            "net_se": float(nets.std(ddof=1) / math.sqrt(len(nets))),
            "net_sd_draw": float(nets.std(ddof=1)),
            "sharpe": float(shs.mean()),
            "sharpe_sd_draw": float(shs.std(ddof=1)),
            "sharpe_sd": None,
            "mdd": float(np.mean([d["mdd"] for d in draws])),
            "ic": float(np.mean([d["ic"] for d in draws])),
            "n_draws": NDRAW}


OUT = {}
for top_k in TOPKS:
    print(f"\n===== NETWORK-MATCHED, top_k {top_k}, H {HOLD}, sleeves, "
          f"{FROM}..{TO} =====", flush=True)
    OUT[top_k] = {}
    for w in WIDTHS:
        print(f"\n-- {w} networks --")
        print(f"{'construction':<34} {'runs':>4} {'net':>8} {'±SE':>6} "
              f"{'Sharpe':>7} {'MDD':>6} {'IC':>8}")
        r = onenas_one_run(w, top_k)
        OUT[top_k][f"onenas_isl{w}_1run"] = r
        print(f"{f'ONE-NAS {w} islands (1 run)':<34} {1:>4} {r['net']:>+8.1f} "
              f"{r['net_se']:>6.1f} {r['sharpe']:>7.2f} {r['mdd']:>6.1f} "
              f"{r['ic']:>+8.4f}", flush=True)
        for arm in BASE_ARMS:
            b = baseline_ens_mean(arm, w, top_k)
            OUT[top_k][f"{arm}_ens{w}"] = b
            if b is None:
                print(f"{f'{arm} {w}-seed ensemble':<34} {w:>4} "
                      f"{'--- only ' + str(len(BASE_SEEDS[arm])) + ' seeds on all 4 panels':>39}",
                      flush=True)
                continue
            tag = "" if b["n_draws"] > 1 else "  (single draw: full pool)"
            print(f"{f'{arm} {w}-seed ensemble':<34} {w:>4} {b['net']:>+8.1f} "
                  f"{b['net_se']:>6.1f} {b['sharpe']:>7.2f} {b['mdd']:>6.1f} "
                  f"{b['ic']:>+8.4f}{tag}", flush=True)

json.dump(OUT, open("results_econ/matched_width.json", "w"), indent=1)
print("\nWROTE results_econ/matched_width.json")
