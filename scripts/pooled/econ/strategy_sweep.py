#!/usr/bin/env python3
"""Strategy sweep: re-book frozen ONE-NAS predictions under alternative books.

Amendment 5 already swept the sleeves book's GEOMETRY (top_k x H, no adoption).
This sweep holds the registered geometry fixed where applicable and varies the
BOOK CONSTRUCTION itself, on the same frozen ensemble predictions:

  sleeves_reg    registered book: equal-weight sleeves, top-10, H=10 (reference)
  sleeves_h5     Amendment-5 best-Sharpe holding, H=5 (reference row)
  rank_weight    sleeves H=10/top-10 but conviction-weighted within each side:
                 weight proportional to (K+1-rank), renormalized per sleeve. Scale-
                 invariant like the registered book (reads only the ordering).
  banded         membership book with a no-trade band: hold exactly 10/10 names;
                 a long exits only when its rank falls below 10+band (band=15),
                 replaced by the best-ranked name not held; symmetric for shorts.
                 Positions drift (algo1-style); costs only on entry/exit.
  beta_neutral   registered sleeves but the short leg is scaled by rolling betas
                 (63-day, vs the panel's equal-weight mean return) so the book
                 targets ~zero market exposure instead of zero net dollars.
  long_only      long leg only: sleeve-rotated top-10, H=10, fully invested
                 ~beta-1 book; the raw-net comparable to equal-weight B&H.

All costs are netted TC/PRC, identical to every other book in the campaign.
Scored on the full 2020-2024 prequential span, 8-island and 40-island suites,
10 seeds x 4 panels each. NO ADOPTION: this is a sensitivity surface over
correlated cells on the development span; paired deltas vs sleeves_reg are
reported per seed so a max-of-N is visible as such.

    tmp/venv/bin/python strategy_sweep.py --panels /path/set1 ... [--out-csv ...]
"""

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "baselines"))

import scoring                     # noqa: E402
import score_stream as ss          # noqa: E402
from panel import Panel            # noqa: E402
from rebook import load_preds      # noqa: E402

SUITES = {"8isl": ("onenas_c7e", "probe_s4547", "probe_s4851"),
          "40isl": ("probe_ISL40",)}
CAP = ss.CAPITAL


def book_metrics(book, n_days):
    net, sharpe, mdd = ss.book_stats(book["daily_ret"])
    return {"net_pct": net, "sharpe": sharpe, "mdd_pct": mdd,
            "turnover": ss.mean(book["traded"]) / ss.GROSS_NOTIONAL,
            "cost_pct": 100.0 * sum(book["cost"]) / CAP, "n_days": n_days}


def weights_for(order, K, weighted):
    """Per-name signed weights for one sleeve, summing to +1/-1 per side."""
    top, bot = order[:K], order[-K:]
    if not weighted:
        w = [1.0 / K] * K
        return list(zip(top, w)) + [(k, -1.0 / K) for k in bot]
    raw = [K - i for i in range(K)]           # K, K-1, .., 1 by rank
    s = float(sum(raw))
    longs = [(k, raw[i] / s) for i, k in enumerate(top)]
    shorts = [(k, -raw[K - 1 - i] / s) for i, k in enumerate(bot)]
    return longs + shorts


def sleeves_book(panel, preds, rows, K, H, weighted=False, long_only=False):
    """Jegadeesh-Titman sleeves, optionally conviction-weighted or long-only."""
    n = panel.n_stocks
    side_cap = CAP / H
    sleeves = []
    daily_ret, costs, traded = [], [], []
    for di, r in enumerate(rows):
        sig = preds[r]
        order = sorted(range(n), key=lambda k: sig[k], reverse=True)
        before = np.zeros(n)
        for _, pos in sleeves:
            before += pos
        sleeves = [s for s in sleeves if s[0] > di - H]
        new = np.zeros(n)
        for k, w in weights_for(order, K, weighted):
            if long_only and w < 0:
                continue
            new[k] = w * side_cap
        sleeves.append([di, new])
        after = np.zeros(n)
        for _, pos in sleeves:
            after += pos
        frac = ss._cost_frac(panel.prc, panel.tc, max(r - 1, 0), None)
        cost = tr = 0.0
        for k in np.nonzero(before != after)[0]:
            d = abs(after[k] - before[k])
            tr += d
            cost += d * frac(k)
        ret = np.asarray(panel.Yscore[r])
        pnl = 0.0
        for s in sleeves:
            pnl += float(np.dot(s[1], ret))
            s[1] = s[1] * (1.0 + ret)
        daily_ret.append((pnl - cost) / CAP)
        costs.append(cost)
        traded.append(tr)
    return {"daily_ret": daily_ret, "cost": costs, "traded": traded}


def banded_book(panel, preds, rows, K, band):
    """Hold exactly K/K names; exit only when rank leaves the K+band band."""
    n = panel.n_stocks
    per = CAP / K
    pos = {}                      # stock -> signed notional (drifts)
    daily_ret, costs, traded = [], [], []
    for r in rows:
        sig = preds[r]
        order = sorted(range(n), key=lambda k: sig[k], reverse=True)
        rank = {k: i for i, k in enumerate(order)}          # 0 = best
        longs = {k for k, v in pos.items() if v > 0}
        shorts = {k for k, v in pos.items() if v < 0}
        frac = ss._cost_frac(panel.prc, panel.tc, max(r - 1, 0), None)
        cost = tr = 0.0

        def trade(k, target):
            nonlocal cost, tr
            d = target - pos.get(k, 0.0)
            if d == 0.0:
                return
            tr += abs(d)
            cost += abs(d) * frac(k)
            if target == 0.0:
                pos.pop(k, None)
            else:
                pos[k] = target

        if not pos:
            for k in order[:K]:
                trade(k, per)
            for k in order[-K:]:
                trade(k, -per)
        else:
            for k in [k for k in longs if rank[k] >= K + band]:
                trade(k, 0.0)
                longs.discard(k)
            for k in [k for k in shorts if rank[k] < n - K - band]:
                trade(k, 0.0)
                shorts.discard(k)
            for k in order:                       # refill longs, best first
                if len(longs) >= K:
                    break
                if k not in longs and k not in shorts:
                    trade(k, per)
                    longs.add(k)
            for k in order[::-1]:                 # refill shorts, worst first
                if len(shorts) >= K:
                    break
                if k not in shorts and k not in longs:
                    trade(k, -per)
                    shorts.add(k)
        ret = panel.Yscore[r]
        pnl = 0.0
        for k in list(pos):
            pnl += pos[k] * ret[k]
            pos[k] *= 1.0 + ret[k]
        daily_ret.append((pnl - cost) / CAP)
        costs.append(cost)
        traded.append(tr)
    return {"daily_ret": daily_ret, "cost": costs, "traded": traded}


def beta_neutral_book(panel, preds, rows, K, H, window=63):
    days = scoring.build_days(panel, preds, rows)
    ret_by_row = {r: panel.Yscore[r] for r in range(rows[0] - window - 1,
                                                    rows[-1] + 1) if r >= 0}
    betas = ss.rolling_betas(days, ret_by_row, window)
    return ss.run_book(days, scoring.PRED, panel.prc, panel.tc, K, None,
                       book="sleeves", hold_days=H, betas=betas)


STRATS = {
    "sleeves_reg": lambda p, pr, rw: sleeves_book(p, pr, rw, 10, 10),
    "sleeves_h5": lambda p, pr, rw: sleeves_book(p, pr, rw, 10, 5),
    "rank_weight": lambda p, pr, rw: sleeves_book(p, pr, rw, 10, 10,
                                                  weighted=True),
    "banded15": lambda p, pr, rw: banded_book(p, pr, rw, 10, 15),
    "banded25": lambda p, pr, rw: banded_book(p, pr, rw, 10, 25),
    "beta_neutral": lambda p, pr, rw: beta_neutral_book(p, pr, rw, 10, 10),
    "long_only": lambda p, pr, rw: sleeves_book(p, pr, rw, 10, 10,
                                                long_only=True),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panels", nargs="+", required=True)
    ap.add_argument("--score-from", default="2020-01-01")
    ap.add_argument("--score-to", default="2024-12-31")
    ap.add_argument("--out-csv", default=os.path.join(HERE, "results_econ",
                                                      "strategy_sweep.csv"))
    args = ap.parse_args()

    panels = {os.path.basename(p.rstrip("/")): Panel(p, "RET_CS")
              for p in args.panels}
    for name, p in panels.items():
        if p.realized != "sidecar":
            raise SystemExit(f"{name}: realized returns not from sidecar")

    out = []
    for suite, dirs in SUITES.items():
        for d in dirs:
            base = os.path.join(HERE, d)
            if not os.path.isdir(base):
                print(f"# missing suite dir {d}", flush=True)
                continue
            for run in sorted(os.listdir(base)):
                ppath = os.path.join(base, run,
                                     "ensemble_stitched_predictions.csv")
                if "_seed" not in run or not os.path.exists(ppath):
                    continue
                panel_name, seed = run.rsplit("_seed", 1)
                if panel_name not in panels:
                    continue
                panel = panels[panel_name]
                preds, rows = load_preds(ppath, panel, args.score_from,
                                         args.score_to)
                for strat, fn in STRATS.items():
                    row = book_metrics(fn(panel, preds, rows), len(rows))
                    rec = {"suite": suite, "strategy": strat,
                           "panel": panel_name, "seed": int(seed)}
                    rec.update(row)
                    out.append(rec)
                    print(f"{suite:<6} {strat:<13} {panel_name:<5} s{seed:<3} "
                          f"net {row['net_pct']:+8.2f}  "
                          f"sharpe {row['sharpe']:+6.2f}  "
                          f"MDD {row['mdd_pct']:5.1f}  "
                          f"turn {row['turnover']:.4f}  "
                          f"cost {row['cost_pct']:6.2f}", flush=True)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"# wrote {args.out_csv} ({len(out)} rows)")


if __name__ == "__main__":
    main()
