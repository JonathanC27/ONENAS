#!/usr/bin/env python3
"""M1: island-champion ensemble vs single global-best genome, within-run paired.

The paper's primary test (PAPER_MAP.md S5/M1). Both prediction rules come
from the SAME completed run -- identical population, seed, panel, market
days -- so the per-run difference isolates the prediction rule alone.

Arms (fleet -> local ensemble dirs, global-best dirs under --gb-root):
  C7_E         8 islands,  eval span   (onenas_c7e + probe_s4547 + probe_s4851)
  C7_ISL40     40 islands, eval span   (probe_ISL40)
  TUNE16_ISL8  8 islands,  2016-2019   (probe_TUNE16_ISL8)   <- robustness span
  TUNE16_ISL40 40 islands, 2016-2019   (probe_TUNE16_ISL40)

Windows: eval fleets on 2022-24 and 2020-24; tune16 fleets on 2016-19.
Both rules booked identically: registered sleeves (top-10, H=10, netted
TC/PRC) + daily Spearman rank IC. Paired per (panel, seed), collapsed to
seed level, paired t (df = n_seeds - 1).

    tmp/venv/bin/python m1_ensemble_test.py --panels /path/set1 ... \
        [--gb-root m1_gb] [--out-csv results_econ/m1_ensemble_test.csv]
"""

import argparse
import csv
import os
import sys
import importlib.util

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "baselines"))

import scoring                     # noqa: E402
import score_stream as ss          # noqa: E402
from panel import Panel            # noqa: E402
from rebook import load_preds      # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "sw", os.path.join(HERE, "strategy_sweep.py"))
_sw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sw)
sleeves_book = _sw.sleeves_book

FLEETS = {
    "C7_E":        (("onenas_c7e", "probe_s4547", "probe_s4851"),
                    [("2022-24", "2022-01-01", "2024-12-31"),
                     ("2020-24", "2020-01-01", "2024-12-31")]),
    "C7_ISL40":    (("probe_ISL40",),
                    [("2022-24", "2022-01-01", "2024-12-31"),
                     ("2020-24", "2020-01-01", "2024-12-31")]),
    "TUNE16_ISL8": (("probe_TUNE16_ISL8",),
                    [("2016-19", "2016-01-01", "2019-12-31")]),
    "TUNE16_ISL40": (("probe_TUNE16_ISL40",),
                     [("2016-19", "2016-01-01", "2019-12-31")]),
}


def score_rule(panel, preds, rows):
    days = scoring.build_days(panel, preds, rows)
    book = sleeves_book(panel, preds, rows, 10, 10)
    net, sharpe, mdd = ss.book_stats(book["daily_ret"])
    ic, _, _ = ss.mean_se_t(ss.finite(ss.daily_ics(days, scoring.PRED,
                                                   ss.spearman)))
    return {"rank_ic": ic, "net_pct": net, "sharpe": sharpe, "mdd_pct": mdd}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panels", nargs="+", required=True)
    ap.add_argument("--gb-root", default=os.path.join(HERE, "m1_gb"))
    ap.add_argument("--out-csv", default=os.path.join(HERE, "results_econ",
                                                      "m1_ensemble_test.csv"))
    args = ap.parse_args()

    panels = {os.path.basename(p.rstrip("/")): Panel(p, "RET_CS")
              for p in args.panels}
    for name, p in panels.items():
        if p.realized != "sidecar":
            raise SystemExit(f"{name}: realized returns not from sidecar")

    out = []
    for fleet, (ens_dirs, windows) in FLEETS.items():
        span0 = min(w[1] for w in windows)
        span1 = max(w[2] for w in windows)
        for d in ens_dirs:
            base = os.path.join(HERE, d)
            if not os.path.isdir(base):
                print(f"# missing ensemble dir {d}", flush=True)
                continue
            for run in sorted(os.listdir(base)):
                ens_path = os.path.join(base, run,
                                        "ensemble_stitched_predictions.csv")
                gb_path = os.path.join(args.gb_root, fleet, run,
                                       "ensemble_stitched_predictions.csv")
                if "_seed" not in run or not os.path.exists(ens_path):
                    continue
                if not os.path.exists(gb_path):
                    print(f"# missing global-best stream {fleet}/{run}",
                          flush=True)
                    continue
                pn, seed = run.rsplit("_seed", 1)
                panel = panels[pn]
                for rule, path in (("ensemble", ens_path), ("single", gb_path)):
                    preds, rows = load_preds(path, panel, span0, span1)
                    for wname, w0, w1 in windows:
                        wrows = [r for r in rows
                                 if w0 <= panel.dates[r] <= w1]
                        row = score_rule(panel, preds, wrows)
                        rec = {"fleet": fleet, "rule": rule, "panel": pn,
                               "seed": int(seed), "window": wname}
                        rec.update(row)
                        out.append(rec)
                        print(f"{fleet:<13} {rule:<8} {pn:<5} s{seed:<3} "
                              f"{wname:<8} IC {row['rank_ic']:+.4f}  "
                              f"net {row['net_pct']:+8.2f}  "
                              f"sharpe {row['sharpe']:+6.2f}", flush=True)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"# wrote {args.out_csv} ({len(out)} rows)")

    # ---------------- paired summary: ensemble - single, per fleet x window
    print("\n==== M1 paired summary (ensemble - single; mean over panels "
          "within seed; paired t over seeds) ====")
    for fleet, (_, windows) in FLEETS.items():
        for wname, _, _ in windows:
            per = {}
            for r in out:
                if r["fleet"] != fleet or r["window"] != wname:
                    continue
                key = (r["seed"], r["rule"])
                per.setdefault(key, []).append(r)
            seeds = sorted({s for (s, _) in per})
            for field in ("rank_ic", "net_pct", "sharpe"):
                deltas = []
                for s in seeds:
                    e = per.get((s, "ensemble")); g = per.get((s, "single"))
                    if not e or not g:
                        continue
                    deltas.append(sum(x[field] for x in e) / len(e)
                                  - sum(x[field] for x in g) / len(g))
                if len(deltas) < 2:
                    continue
                m = sum(deltas) / len(deltas)
                sd = (sum((x - m) ** 2 for x in deltas)
                      / (len(deltas) - 1)) ** 0.5
                se = sd / len(deltas) ** 0.5
                t = m / se if se > 0 else float("nan")
                fmt = "%+.4f" if field == "rank_ic" else "%+.2f"
                print(f"{fleet:<13} {wname:<8} d{field:<8} "
                      f"{fmt % m}  (t={t:+.2f}, n={len(deltas)} seeds)")


if __name__ == "__main__":
    main()
