#!/usr/bin/env python3
"""Score ONE-NAS stitched predictions under the ICAIF paper's protocol.

The ICAIF paper (EXAMM, offline walk-forward) evaluates per TEST YEAR with:
  * Pearson IC of the raw predictions vs realized next-day simple returns,
  * the Algorithm 1 trigger book -- top-10 long / bottom-10 short, rebalanced
    only when all top-10 preds > 0 and all bottom-10 < 0, else held --
    charged TC/PRC on netted traded notional, fresh book per test year,
  * buy-and-hold and daily equal-weight benchmarks on the same 50 names.

This script re-scores the always-online ONE-NAS ensemble predictions under
exactly those conventions so the two papers' headline rows sit side by side.
Nothing is reimplemented: the book and every metric come from score_stream
(book="algo1"), the same code path the sleeves numbers came from.  Books are
restarted at each year boundary because the ICAIF folds are disjoint.

KNOWN STRUCTURAL CAVEAT (baselines/README.md): these runs were trained on
RET_CS, a per-day cross-sectional rank-normal target, so the prediction scale
is symmetric about zero and the Algorithm 1 trigger fires almost every day,
where EXAMM's raw-return predictions fire it rarely.  The trigger rate and
average holding period are reported per cell so the mismatch is visible.

    tmp/venv/bin/python icaif_protocol.py --panels /path/set1 ... [--out-csv ...]
"""

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "baselines"))

import scoring                     # noqa: E402  (wraps score_stream)
import score_stream as ss          # noqa: E402
from panel import Panel            # noqa: E402
from rebook import load_preds      # noqa: E402

WINDOWS = [("2020", "2020-01-01", "2020-12-31"),
           ("2021", "2021-01-01", "2021-12-31"),
           ("2022", "2022-01-01", "2022-12-31"),
           ("2023", "2023-01-01", "2023-12-31"),
           ("2024", "2024-01-01", "2024-12-31"),
           ("2022-24", "2022-01-01", "2024-12-31"),
           ("2020-24", "2020-01-01", "2024-12-31")]

# suites of stitched ONE-NAS runs, relative to this directory
ARMS = {"onenas_8isl": ("onenas_c7e", "probe_s4547", "probe_s4851"),
        "onenas_20isl": ("probe_ISL20",),
        "onenas_40isl": ("probe_ISL40",)}


def stats_row(days, book):
    net, sharpe, mdd = ss.book_stats(book["daily_ret"])
    pm, pse, pt = ss.mean_se_t(ss.finite(ss.daily_ics(days, scoring.PRED,
                                                      ss.pearson)))
    rm, _, _ = ss.mean_se_t(ss.finite(ss.daily_ics(days, scoring.PRED,
                                                   ss.spearman)))
    return {"n_days": len(days), "pearson_ic": pm, "pearson_t": pt,
            "rank_ic": rm, "net_pct": net, "sharpe": sharpe, "mdd_pct": mdd,
            "turnover": ss.mean(book["traded"]) / ss.GROSS_NOTIONAL,
            "cost_pct": 100.0 * sum(book["cost"]) / ss.CAPITAL,
            "trigger_rate": ss.mean(book["rebalanced"]),
            "avg_hold_days": book.get("avg_holding_days", float("nan"))}


def bench_book(panel, rows, daily_rebalance):
    """$100 equal-weight long book: bought once and held (B&H), or rebalanced
    to equal weights every day (EW), TC/PRC cost on netted traded notional."""
    n = panel.n_stocks
    pos = [0.0] * n
    daily_ret, costs, traded, flags = [], [], [], []
    for i, r in enumerate(rows):
        cost = tr = 0.0
        rebal = daily_rebalance or i == 0
        if rebal:
            frac = ss._cost_frac(panel.prc, panel.tc, max(r - 1, 0), None)
            for k in range(n):
                d = ss.CAPITAL / n - pos[k]
                if d != 0.0:
                    tr += abs(d)
                    cost += abs(d) * frac(k)
            pos = [ss.CAPITAL / n] * n
        pnl = 0.0
        for k in range(n):
            ret = panel.Yscore[r][k]
            pnl += pos[k] * ret
            pos[k] *= 1.0 + ret
        daily_ret.append((pnl - cost) / ss.CAPITAL)
        costs.append(cost)
        traded.append(tr)
        flags.append(1 if rebal else 0)
    return {"daily_ret": daily_ret, "cost": costs, "traded": traded,
            "rebalanced": flags, "avg_holding_days": float("nan")}


def bench_stats(panel, rows, daily_rebalance):
    book = bench_book(panel, rows, daily_rebalance)
    net, sharpe, mdd = ss.book_stats(book["daily_ret"])
    return {"n_days": len(rows), "pearson_ic": float("nan"),
            "pearson_t": float("nan"), "rank_ic": float("nan"),
            "net_pct": net, "sharpe": sharpe, "mdd_pct": mdd,
            "turnover": ss.mean(book["traded"]) / ss.GROSS_NOTIONAL,
            "cost_pct": 100.0 * sum(book["cost"]) / ss.CAPITAL,
            "trigger_rate": ss.mean(book["rebalanced"]),
            "avg_hold_days": float("nan")}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panels", nargs="+", required=True,
                    help="core7 panel dirs named set1..set4")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--out-csv",
                    default=os.path.join(HERE, "results_econ",
                                         "icaif_protocol.csv"))
    args = ap.parse_args()

    panels = {os.path.basename(p.rstrip("/")): Panel(p, "RET_CS")
              for p in args.panels}
    for name, p in panels.items():
        if p.realized != "sidecar":
            raise SystemExit(f"{name}: realized returns not from RET_raw_ "
                             "sidecar; refusing to book")

    out = []

    def emit(arm, panel_name, seed, window, row):
        rec = {"arm": arm, "panel": panel_name, "seed": seed,
               "window": window}
        rec.update(row)
        out.append(rec)
        print(f"{arm:<14} {panel_name:<5} s{seed:<3} {window:<8} "
              f"pIC {row['pearson_ic']:+.4f}  net {row['net_pct']:+8.2f}  "
              f"sharpe {row['sharpe']:+6.2f}  trig {row['trigger_rate']:.2f}  "
              f"cost {row['cost_pct']:6.2f}"
              if row["pearson_ic"] == row["pearson_ic"] else
              f"{arm:<14} {panel_name:<5} s{seed:<3} {window:<8} "
              f"pIC    --    net {row['net_pct']:+8.2f}  "
              f"sharpe {row['sharpe']:+6.2f}", flush=True)

    for arm, dirs in ARMS.items():
        for d in dirs:
            base = os.path.join(HERE, d)
            if not os.path.isdir(base):
                print(f"# skipping missing suite dir {d}", flush=True)
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
                preds, rows = load_preds(ppath, panel,
                                         "2020-01-01", "2024-12-31")
                for wname, w0, w1 in WINDOWS:
                    wrows = [r for r in rows if w0 <= panel.dates[r] <= w1]
                    days = scoring.build_days(panel, preds, wrows)
                    book = ss.run_book(days, scoring.PRED, panel.prc,
                                       panel.tc, args.top_k, None,
                                       book="algo1")
                    emit(arm, panel_name, int(seed), wname,
                         stats_row(days, book))

    for panel_name, panel in sorted(panels.items()):
        for wname, w0, w1 in WINDOWS:
            rows = panel.rows_between(max(w0, "2020-01-02"), w1)
            emit("buy_hold", panel_name, 0, wname,
                 bench_stats(panel, rows, daily_rebalance=False))
            emit("ew_daily", panel_name, 0, wname,
                 bench_stats(panel, rows, daily_rebalance=True))

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"# wrote {args.out_csv} ({len(out)} rows; algo1 book restarted "
          f"per window, top-{args.top_k}, netted TC/PRC costs)")


if __name__ == "__main__":
    main()
