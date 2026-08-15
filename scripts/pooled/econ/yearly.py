#!/usr/bin/env python3
"""Per-calendar-year slice of the sleeves economics table.

Same discovery, same panels, same code path as rebook.py (score_stream.run_book,
book="sleeves", netted TC/PRC costs): the book runs once over the full
2020-2024 span -- positions carry across year boundaries exactly as they did in
the headline numbers -- and the daily return series is then grouped by the
calendar year of each scored day.  Per-year net%% is the sum of daily returns
in that year (same additive P&L-over-initial-capital convention as the
full-span net%%), per-year Sharpe is annualised from that year's dailies.

    python3 yearly.py --panels /path/set1_core7 ... \
        [--results-dir results_econ] [--onenas-dir onenas_c7e] \
        [--hold-days 10] [--top-k 10]

Writes <results-dir>/yearly_econ.csv (arm x panel x seed x year) and prints an
arm x year summary (mean +/- SE across panel x seed runs).
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "baselines"))

import scoring                     # noqa: E402
import score_stream as ss          # noqa: E402
from panel import Panel            # noqa: E402

from rebook import load_preds, ew_buy_hold  # noqa: E402


def yearly_rows(dates, daily_ret):
    """(arm-agnostic) group a dated daily return series by calendar year."""
    by_year = defaultdict(list)
    for d, r in zip(dates, daily_ret):
        by_year[d[:4]].append(r)
    out = {}
    for y, rets in sorted(by_year.items()):
        v = np.asarray(rets, dtype=float)
        sd = v.std(ddof=1) if len(v) > 1 else float("nan")
        sharpe = (v.mean() / sd * np.sqrt(252.0)) if sd and sd == sd else float("nan")
        out[y] = {"n_days": len(v), "net_pct": 100.0 * v.sum(),
                  "sharpe": float(sharpe)}
    return out


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

    def emit(arm, panel_name, seed, dates, daily_ret):
        for y, row in yearly_rows(dates, daily_ret).items():
            out_rows.append({"arm": arm, "panel": panel_name, "seed": seed,
                             "year": y, **row})

    def rebook_file(arm, panel_name, seed, path):
        panel = panels[panel_name]
        preds, rows = load_preds(path, panel, args.score_from, args.score_to)
        days = scoring.build_days(panel, preds, rows)
        book = ss.run_book(days, scoring.PRED, panel.prc, panel.tc,
                           args.top_k, None, book="sleeves",
                           hold_days=args.hold_days)
        emit(arm, panel_name, seed, [d[1] for d in days], book["daily_ret"])

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
                print(f"booked {arm} {panel_name} s{seed}", flush=True)

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
                    print(f"booked {arm} {setname} s{seed}", flush=True)

    for panel_name, panel in sorted(panels.items()):
        rows = panel.rows_between(args.score_from, args.score_to)
        # re-derive the daily series the same way rebook.ew_buy_hold does,
        # but keep the dailies instead of collapsing to summary stats
        n = panel.n_stocks
        frac = ss._cost_frac(panel.prc, panel.tc, max(rows[0] - 1, 0), None)
        pos = [ss.CAPITAL / n] * n
        build_cost = sum(abs(v) * frac(k) for k, v in enumerate(pos))
        daily_ret = []
        for i, r in enumerate(rows):
            cost = build_cost if i == 0 else 0.0
            pnl = 0.0
            for k in range(n):
                ret = panel.Yscore[r][k]
                pnl += pos[k] * ret
                pos[k] *= 1.0 + ret
            daily_ret.append((pnl - cost) / ss.CAPITAL)
        emit("ew_buy_hold", panel_name, 42,
             [panel.dates[r] for r in rows], daily_ret)

    out_csv = args.out_csv or os.path.join(args.results_dir, "yearly_econ.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print(f"# wrote {out_csv} ({len(out_rows)} rows)")

    # ------------------------------------------------- arm x year summary
    agg = defaultdict(list)
    for r in out_rows:
        agg[(r["arm"], r["year"])].append(r["net_pct"])
    arms = sorted({a for a, _ in agg})
    years = sorted({y for _, y in agg})
    print(f"\n{'arm':<26}" + "".join(f"{y:>16}" for y in years))
    for a in arms:
        cells = []
        for y in years:
            v = np.asarray(agg[(a, y)], dtype=float)
            se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
            cells.append(f"{v.mean():+7.1f} ±{se:4.1f}")
        print(f"{a:<26}" + "".join(f"{c:>16}" for c in cells))


if __name__ == "__main__":
    main()
