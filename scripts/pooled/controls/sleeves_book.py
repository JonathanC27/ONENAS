#!/usr/bin/env python3
"""Re-book control predictions with score_stream's SLEEVES book (H=10).

baselines/scoring.py books everything with score_stream's default Algorithm 1
trigger, which fires nearly every day on a mean-zero rank signal and makes
turnover/net% incomparable to the evolution arm (scored with --book sleeves
--hold-days 10).  This reads each run's predictions.csv, rebuilds the days
list with scoring.build_days (same panel, same realised returns) and calls
score_stream.run_book(book="sleeves", hold_days=10) -- the identical code
path score_ensemble.py uses for the evolution arm's book.

    python3 sleeves_book.py --panels /path/to/set1_v2 /path/to/set2_v2 \
        [--results-dir results_controls] [--hold-days 10] [--top-k 10]

Writes <results-dir>/sleeves_summary.csv and prints the table.
"""

import argparse
import csv
import os

import numpy as np

import common                      # noqa: F401  (sys.path bootstrap)
import scoring
import score_stream as ss
from panel import Panel

HERE = os.path.dirname(os.path.abspath(__file__))


def load_preds(path, panel):
    """predictions.csv (date,stock,pred) -> ({row: ndarray}, rows)."""
    tix = {t: k for k, t in enumerate(panel.tickers)}
    by_row = {}
    with open(path, newline="") as fh:
        rdr = csv.DictReader(fh)
        for rec in rdr:
            r = panel.date_index[rec["date"]]
            v = by_row.setdefault(r, np.full(panel.n_stocks, np.nan))
            v[tix[rec["stock"]]] = float(rec["pred"])
    rows = sorted(by_row)
    return by_row, rows


def sleeves_stats(panel, days, top_k, hold_days, idx=scoring.PRED):
    book = ss.run_book(days, idx, panel.prc, panel.tc, top_k, None,
                       book="sleeves", hold_days=hold_days)
    net, sharpe, mdd = ss.book_stats(book["daily_ret"])
    turnover = ss.mean(book["traded"]) / ss.GROSS_NOTIONAL
    cost = 100.0 * sum(book["cost"]) / ss.CAPITAL
    return {"net_pct": net, "sharpe": sharpe, "mdd_pct": mdd,
            "turnover": turnover, "cost_pct": cost}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panels", nargs="+", required=True,
                    help="panel directories (basename must match the run "
                         "directory prefix, e.g. .../set1_v2)")
    ap.add_argument("--results-dir",
                    default=os.path.join(HERE, "results_controls"))
    ap.add_argument("--param", default="RET_CS")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--hold-days", type=int, default=10)
    args = ap.parse_args()

    panels = {os.path.basename(p.rstrip("/")): Panel(p, args.param)
              for p in args.panels}
    out_rows = []
    fmt = "%-9s %-16s %-8s  %8s %8s %8s %9s %8s"
    print("# sleeves book, H=%d, top-k %d, TC/PRC costs (score_stream's "
          "run_book, the evolution arm's book)" % (args.hold_days, args.top_k))
    print(fmt % ("control", "run", "variant", "net%", "sharpe", "MDD%",
                 "turnover", "cost%"))
    for control in ("control1", "control2"):
        base = os.path.join(args.results_dir, control)
        if not os.path.isdir(base):
            continue
        for run in sorted(os.listdir(base)):
            pname = run.rsplit("_seed", 1)[0]
            if pname not in panels:
                continue
            panel = panels[pname]
            for variant in ("single", "ensemble"):
                ppath = os.path.join(base, run, variant, "predictions.csv")
                if not os.path.exists(ppath):
                    continue
                preds, rows = load_preds(ppath, panel)
                days = scoring.build_days(panel, preds, rows)
                st = sleeves_stats(panel, days, args.top_k, args.hold_days)
                out_rows.append(dict(control=control, run=run,
                                     variant=variant,
                                     **{k: round(v, 4) for k, v in st.items()}))
                print(fmt % (control, run, variant,
                             "%+.2f" % st["net_pct"], "%+.2f" % st["sharpe"],
                             "%.2f" % st["mdd_pct"],
                             "%.4f" % st["turnover"], "%.2f" % st["cost_pct"]))
    if out_rows:
        path = os.path.join(args.results_dir, "sleeves_summary.csv")
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
            w.writeheader()
            w.writerows(out_rows)
        print("# wrote %s" % path)


if __name__ == "__main__":
    main()
