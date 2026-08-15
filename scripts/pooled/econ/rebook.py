#!/usr/bin/env python3
"""Re-book every baseline (and the ONE-NAS runs) under the SLEEVES book.

The paper's economics table books every arm with the identical portfolio
construction the ONE-NAS system was measured under: score_stream.run_book
(book="sleeves", hold_days=10, top-10 per side, netted TC/PRC costs).  This
script does not reimplement anything -- it imports scoring.build_days (which
itself wraps score_stream) and calls score_stream.run_book / book_stats /
daily_ics directly, so a baseline row and the system's row come out of the
same code path.

Inputs re-booked:
  * results_econ/<arm>/<panel>_seed<seed>/predictions.csv  (tidy date,stock,pred)
  * onenas_c7e/<set>_seed<seed>/{ensemble_,}stitched_predictions.csv
    (the C7_E evolution runs; extra real/naive columns are ignored, the
    predictions are re-booked from scratch through the same panel data)

Reference rows booked with the same cost model:
  * ew_buy_hold -- $100 long, equal weight across the 50 names, bought once
    at the prior close of the first scored day (TC/PRC cost on that initial
    build only), then held; positions drift with realised returns.
  * S&P index row: SKIPPED -- the core7 panels carry no sprtrn column and
    there is no clean local source for it.

Also reports each arm's daily rank IC@1 (mean, SE, t) for reference.

    python3 rebook.py --panels /path/set1_core7 ... \
        [--results-dir results_econ] [--onenas-dir onenas_c7e] \
        [--hold-days 10] [--top-k 10]

Writes <results-dir>/sleeves_econ.csv (one row per arm x panel x seed).
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


def load_preds(path, panel, score_from, score_to):
    """Tidy prediction csv -> ({row: ndarray}, rows within the score span)."""
    tix = {t: k for k, t in enumerate(panel.tickers)}
    by_row = {}
    with open(path, newline="") as fh:
        for rec in csv.DictReader(fh):
            d = rec["date"]
            if d < score_from or d > score_to:
                continue
            r = panel.date_index[d]
            v = by_row.setdefault(r, np.full(panel.n_stocks, np.nan))
            v[tix[rec["stock"]]] = float(rec["pred"])
    return by_row, sorted(by_row)


def sleeves_row(panel, days, top_k, hold_days):
    book = ss.run_book(days, scoring.PRED, panel.prc, panel.tc, top_k, None,
                       book="sleeves", hold_days=hold_days)
    return book_row(days, book)


def book_row(days, book):
    net, sharpe, mdd = ss.book_stats(book["daily_ret"])
    return {"n_days": len(days),
            "net_pct": net, "sharpe": sharpe, "mdd_pct": mdd,
            "turnover": ss.mean(book["traded"]) / ss.GROSS_NOTIONAL,
            "cost_pct": 100.0 * sum(book["cost"]) / ss.CAPITAL}


def rank_ic_row(panel, days):
    m, se, t = ss.mean_se_t(ss.finite(ss.daily_ics(days, scoring.PRED,
                                                   ss.spearman)))
    return {"rank_ic_1": m, "rank_ic_se": se, "rank_ic_t": t}


def ew_buy_hold(panel, rows):
    """$100 equal-weight long, bought once, held; cost on the build only."""
    n = panel.n_stocks
    frac = ss._cost_frac(panel.prc, panel.tc, max(rows[0] - 1, 0), None)
    pos = [ss.CAPITAL / n] * n
    build_cost = sum(abs(v) * frac(k) for k, v in enumerate(pos))
    daily_ret, costs, traded = [], [], []
    for i, r in enumerate(rows):
        cost = build_cost if i == 0 else 0.0
        tr = ss.CAPITAL if i == 0 else 0.0
        pnl = 0.0
        for k in range(n):
            ret = panel.Yscore[r][k]
            pnl += pos[k] * ret
            pos[k] *= 1.0 + ret
        daily_ret.append((pnl - cost) / ss.CAPITAL)
        costs.append(cost)
        traded.append(tr)
    book = {"daily_ret": daily_ret, "cost": costs, "traded": traded}
    row = book_row([None] * len(rows), book)
    row.update(rank_ic_1=float("nan"), rank_ic_se=float("nan"),
               rank_ic_t=float("nan"))
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panels", nargs="+", required=True)
    ap.add_argument("--results-dir", default=os.path.join(HERE, "results_econ"))
    ap.add_argument("--onenas-dir", default=os.path.join(HERE, "onenas_c7e"))
    ap.add_argument("--param", default="RET_CS")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--hold-days", type=int, default=10)
    ap.add_argument("--score-from", default="2020-01-01")
    ap.add_argument("--score-to", default="2024-12-31")
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    panels = {os.path.basename(p.rstrip("/")): Panel(p, args.param)
              for p in args.panels}
    for name, p in panels.items():
        if p.realized != "sidecar":
            raise SystemExit(f"{name}: realised returns are not coming from "
                             "the sidecar RET_raw_ columns; refusing to book")

    out_rows = []

    def emit(arm, panel_name, seed, row, src):
        rec = {"arm": arm, "panel": panel_name, "seed": seed, "source": src}
        rec.update({k: row.get(k) for k in
                    ("n_days", "rank_ic_1", "rank_ic_se", "rank_ic_t",
                     "net_pct", "sharpe", "mdd_pct", "turnover", "cost_pct")})
        out_rows.append(rec)
        ic = row.get("rank_ic_1")
        print(f"{arm:<26} {panel_name:<12} s{seed:<4} "
              f"IC {'' if ic == ic else '   --   ':<9}"
              f"{('%+.4f' % ic) if ic == ic else '':<9} "
              f"net {row['net_pct']:+8.2f}  sharpe {row['sharpe']:+6.2f}  "
              f"MDD {row['mdd_pct']:6.2f}  turn {row['turnover']:.4f}  "
              f"cost {row['cost_pct']:6.2f}", flush=True)

    def rebook_file(arm, panel_name, seed, path):
        panel = panels[panel_name]
        preds, rows = load_preds(path, panel, args.score_from, args.score_to)
        days = scoring.build_days(panel, preds, rows)
        row = sleeves_row(panel, days, args.top_k, args.hold_days)
        row.update(rank_ic_row(panel, days))
        emit(arm, panel_name, seed, row, path)

    # ------------------------------------------------------ baseline arms
    if os.path.isdir(args.results_dir):
        for arm in sorted(os.listdir(args.results_dir)):
            arm_dir = os.path.join(args.results_dir, arm)
            if not os.path.isdir(arm_dir):
                continue
            for run in sorted(os.listdir(arm_dir)):
                ppath = os.path.join(arm_dir, run, "predictions.csv")
                if not os.path.exists(ppath) or "_seed" not in run:
                    continue
                panel_name, seed = run.rsplit("_seed", 1)
                if panel_name not in panels:
                    continue
                rebook_file(arm, panel_name, int(seed), ppath)

    # ------------------------------------------------------ ONE-NAS (C7_E)
    if os.path.isdir(args.onenas_dir):
        for run in sorted(os.listdir(args.onenas_dir)):
            if "_seed" not in run:
                continue
            setname, seed = run.rsplit("_seed", 1)
            panel_name = setname + "_core7"
            if panel_name not in panels:
                continue
            for fname, arm in (("ensemble_stitched_predictions.csv",
                                "onenas_ensemble"),
                               ("stitched_predictions.csv", "onenas_single")):
                ppath = os.path.join(args.onenas_dir, run, fname)
                if os.path.exists(ppath):
                    rebook_file(arm, panel_name, int(seed), ppath)

    # ------------------------------------------------------ reference rows
    for panel_name, panel in sorted(panels.items()):
        rows = panel.rows_between(args.score_from, args.score_to)
        emit("ew_buy_hold", panel_name, 42, ew_buy_hold(panel, rows),
             "reference")

    out_csv = args.out_csv or os.path.join(args.results_dir,
                                           "sleeves_econ.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print(f"# wrote {out_csv} ({len(out_rows)} rows; sleeves H="
          f"{args.hold_days}, top-{args.top_k}, netted TC/PRC costs)")


if __name__ == "__main__":
    main()
