#!/usr/bin/env python3
"""Does the SEARCH improve with island count? Single global-best genome, 8/20/40.

The original ONE-NAS prediction rule is one genome: the global best. If the
evolutionary search finds better architectures when given more islands, the
single champion must improve from 8 -> 20 -> 40. If the champion is flat while
the island ENSEMBLE climbs (32.1 -> 41.6 -> 45.6 net at top_k 10), then the
search is not what improves with width -- aggregation is -- and adding search
capacity is not the lever.

Raw generation_*_global_best.csv files survive in results_v2/ for all three
widths at 10 seeds x 4 panels; scored on Anvil, book files pulled here.
"""
import csv
import math
import os

import numpy as np

GB = "/Users/jonathanchang/.claude/jobs/fc17658e/tmp/gb"
SETS = ["set1", "set2", "set3", "set4"]
SEEDS = range(42, 52)
ARMS = [("isl8", "8 islands"), ("isl20", "20 islands"), ("isl40", "40 islands")]


def sharpe(v):
    v = np.asarray(v)
    return v.mean() / v.std(ddof=1) * math.sqrt(252)


def mdd(v):
    eq = np.cumsum(np.asarray(v))
    return 100 * np.max(np.maximum.accumulate(eq) - eq)


def load(tag, s, sd):
    """global_best mode writes no ensemble_diagnostics.csv (one member, nothing
    to disagree), so IC comes from the rolling-rank-IC file instead."""
    d = f"{GB}/{tag}/{s}_seed{sd}"
    bp = f"{d}/ensemble_book_daily.csv"
    if not os.path.exists(bp):
        return None
    ret = {}
    with open(bp) as fh:
        for r in csv.DictReader(fh):
            ret[r["date"]] = float(r["model_ret"])
    ics = []
    rp = f"{d}/ensemble_rolling_rank_ic.csv"
    if os.path.exists(rp):
        with open(rp) as fh:
            for r in csv.DictReader(fh):
                for k in ("rank_ic", "rolling_rank_ic", "mean_rank_ic"):
                    if k in r and r[k] not in ("", "nan"):
                        v = float(r[k])
                        if v == v:
                            ics.append(v)
                        break
    return ret, (float(np.mean(ics)) if ics else float("nan"))


print("SINGLE GLOBAL-BEST GENOME by island count (top_k 10, H 10, sleeves, 2020-24)")
print("The original ONE-NAS prediction rule -- one genome, no ensembling.\n")
print(f"{'width':<12} {'seeds':>6} {'net %':>9} {'±SE':>6} {'Sharpe':>8} {'SD':>6} "
      f"{'MDD':>6} {'IC':>9}")
print("-" * 68)

store = {}
for tag, label in ARMS:
    nets, shs, mds, ics = [], [], [], []
    per_seed = {}
    for sd in SEEDS:
        pp, pn, pi = {}, [], []
        ok = True
        for s in SETS:
            r = load(tag, s, sd)
            if r is None:
                ok = False
                break
            ret, ic = r
            pp[s] = ret
            pn.append(100 * sum(ret.values()))
            pi.append(ic)
        if not ok:
            continue
        common = sorted(set.intersection(*[set(v) for v in pp.values()]))
        pool = np.array([[pp[s][d] for s in SETS] for d in common]).mean(1)
        per_seed[sd] = float(sharpe(pool))
        nets.append(float(np.mean(pn)))
        shs.append(float(sharpe(pool)))
        mds.append(float(mdd(pool)))
        ics.append(float(np.mean(pi)))
    store[tag] = {"net": nets, "sh": shs, "per_seed": per_seed}
    print(f"{label:<12} {len(shs):>6} {np.mean(nets):>+9.1f} "
          f"{np.std(nets, ddof=1)/math.sqrt(len(nets)):>6.1f} "
          f"{np.mean(shs):>8.2f} {np.std(shs, ddof=1):>6.3f} "
          f"{np.mean(mds):>6.1f} {np.mean(ics):>+9.4f}")

print("\nPAIRED contrasts on the SINGLE CHAMPION (same seeds):")
base = store["isl8"]["per_seed"]
for tag, label in ARMS[1:]:
    cur = store[tag]["per_seed"]
    common = sorted(set(base) & set(cur))
    d = np.array([cur[k] - base[k] for k in common])
    t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))
    dn = np.mean(store[tag]["net"]) - np.mean(store["isl8"]["net"])
    print(f"  {label} - 8 islands:  dSharpe {d.mean():+.3f}  t={t:+.2f}  "
          f"(n={len(d)})   dNet {dn:+.1f}")

print("\nFor reference, the island-CHAMPION ENSEMBLE at the same widths (top_k 10):")
print("  8 islands  +32.1 net / 0.71 Sharpe")
print(" 20 islands  +41.6 net / 0.86 Sharpe")
print(" 40 islands  +45.6 net / 0.91 Sharpe")
print("\nIf the single champion is FLAT across widths while the ensemble climbs,")
print("width buys AGGREGATION, not a better search -- and more search capacity")
print("(bigger populations, more bp_iterations) is not the binding lever.")
