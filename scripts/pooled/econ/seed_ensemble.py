#!/usr/bin/env python3
"""Build an N-seed rank-mean ensemble of a stochastic baseline arm.

The like-for-like counterpart of the ONE-NAS island-champion ensemble for the
baselines package: take the SAME tuned architecture trained from N different
seeds (seed0 tunes, the rest rerun the frozen config -- run_suite.py's rule),
rank each seed's daily cross-section, average the ranks, and book the combined
signal through the identical sleeves code. Writes a tidy predictions.csv per
panel (so rebook.py-style tooling can consume it) and prints per-panel and
pooled 200-stock statistics.

    python3 seed_ensemble.py --arm lstm --seeds 42,43,44,45,46,47,48,49 \
        --panels /path/set1_core7 ...
"""

import argparse
import csv
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


def rank01(v):
    n = len(v)
    r = np.empty(n)
    r[np.argsort(v, kind="stable")] = np.arange(n, dtype=float)
    return r / (n - 1.0)


def sharpe(v):
    v = np.asarray(v)
    return v.mean() / v.std(ddof=1) * np.sqrt(252)


def mdd(v):
    eq = np.cumsum(np.asarray(v))
    return 100 * np.max(np.maximum.accumulate(eq) - eq)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seeds", default="42,43,44,45,46,47,48,49")
    ap.add_argument("--panels", nargs="+", required=True)
    ap.add_argument("--results-dir", default=os.path.join(HERE, "results_econ"))
    ap.add_argument("--score-from", default="2020-01-01")
    ap.add_argument("--score-to", default="2024-12-31")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]

    per_panel_ret, per_panel_ic, nets = {}, {}, []
    for path in args.panels:
        name = os.path.basename(path.rstrip("/"))
        panel = Panel(path, "RET_CS")
        by_seed = {}
        for sd in seeds:
            p = os.path.join(args.results_dir, args.arm,
                             f"{name}_seed{sd}", "predictions.csv")
            if not os.path.exists(p):
                raise SystemExit(f"missing {p}; run run_suite.py for seed {sd}")
            preds, rows = load_preds(p, panel, args.score_from, args.score_to)
            by_seed[sd] = preds
        common_rows = sorted(set.intersection(*[set(v) for v in by_seed.values()]))
        ens = {r: np.mean([rank01(by_seed[sd][r]) for sd in seeds], axis=0)
               for r in common_rows}
        out_dir = os.path.join(args.results_dir, f"{args.arm}_ens{len(seeds)}",
                               f"{name}_seed{seeds[0]}")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "predictions.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["date", "stock", "pred"])
            for r in common_rows:
                for k, t in enumerate(panel.tickers):
                    w.writerow([panel.dates[r], t, repr(float(ens[r][k]))])
        days = scoring.build_days(panel, ens, common_rows)
        book = ss.run_book(days, scoring.PRED, panel.prc, panel.tc, 10, None,
                           book="sleeves", hold_days=10)
        dts = [d[1] for d in days]
        per_panel_ret[name] = dict(zip(dts, book["daily_ret"]))
        ics = [x for x in ss.daily_ics(days, scoring.PRED, ss.spearman) if x == x]
        per_panel_ic[name] = float(np.mean(ics))
        net = 100 * sum(book["daily_ret"])
        nets.append(net)
        print(f"{args.arm}_ens{len(seeds)} {name}: IC {per_panel_ic[name]:+.4f} "
              f"net {net:+.1f} Sharpe {sharpe(book['daily_ret']):+.2f}",
              flush=True)

    dates = sorted(set.intersection(*[set(v) for v in per_panel_ret.values()]))
    pool = np.array([[per_panel_ret[n][d] for n in per_panel_ret] for d in dates]).mean(1)
    nets = np.asarray(nets)
    print(f"\n{args.arm}_ens{len(seeds)} SUMMARY: IC {np.mean(list(per_panel_ic.values())):+.4f}  "
          f"net {nets.mean():+.1f} ± {nets.std(ddof=1)/np.sqrt(len(nets)):.1f} (4 panels)  "
          f"pooled Sharpe {sharpe(pool):+.2f}  pooled MDD {mdd(pool):.1f}")


if __name__ == "__main__":
    main()
