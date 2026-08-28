#!/usr/bin/env python3
"""Score the pre-registered HP screen (HP_SCREEN.md) against control C0.

Each run is the island-champion rank-mean ensemble stream on the 2016-2019
tuning span, booked as daily Spearman rank IC (primary objective) plus the
registered sleeves book (top-10, H=10, netted TC/PRC). Every cell is paired
against hp_C0 on identical (panel, seed) cells (n=6 pairs), collapsed to
seed level for the paired t.

Pre-stated gates (HP_SCREEN.md): KEEP iff paired mean dIC >= +0.0020 with
t >= 2. A cell failing on IC but with dSharpe >= +0.10 and t >= 2 is
FLAGGED (not auto-kept). Everything else: DROP (reported).

    tmp/venv/bin/python hp_screen_score.py --panels /path/set1 /path/set2 \
        [--stitch-root hp_stitch] [--out-csv results_econ/hp_screen_score.csv]
"""

import argparse
import csv
import os
import sys
import importlib.util

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

W0, W1 = "2016-01-01", "2019-12-31"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panels", nargs="+", required=True)
    ap.add_argument("--stitch-root", default=os.path.join(HERE, "hp_stitch"))
    ap.add_argument("--out-csv", default=os.path.join(HERE, "results_econ",
                                                      "hp_screen_score.csv"))
    args = ap.parse_args()

    panels = {os.path.basename(p.rstrip("/")): Panel(p, "RET_CS")
              for p in args.panels}

    out = []
    for arm in sorted(os.listdir(args.stitch_root)):
        adir = os.path.join(args.stitch_root, arm)
        if not os.path.isdir(adir):
            continue
        for run in sorted(os.listdir(adir)):
            path = os.path.join(adir, run, "ensemble_stitched_predictions.csv")
            if "_seed" not in run or not os.path.exists(path):
                continue
            pn, seed = run.rsplit("_seed", 1)
            if pn not in panels:
                continue
            panel = panels[pn]
            preds, rows = load_preds(path, panel, W0, W1)
            days = scoring.build_days(panel, preds, rows)
            book = _sw.sleeves_book(panel, preds, rows, 10, 10)
            net, sharpe, mdd = ss.book_stats(book["daily_ret"])
            ic, _, _ = ss.mean_se_t(ss.finite(ss.daily_ics(days, scoring.PRED,
                                                           ss.spearman)))
            out.append({"arm": arm, "panel": pn, "seed": int(seed),
                        "rank_ic": ic, "net_pct": net, "sharpe": sharpe,
                        "mdd_pct": mdd, "n_days": len(rows)})
            print(f"{arm:<20} {pn:<5} s{seed:<3} IC {ic:+.4f}  "
                  f"net {net:+8.2f}  sharpe {sharpe:+6.2f}", flush=True)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"# wrote {args.out_csv} ({len(out)} rows)")

    # ------------------------------------------------ gates vs hp_C0
    def cells(arm):
        return {(r["panel"], r["seed"]): r for r in out if r["arm"] == arm}

    c0 = cells("hp_C0")
    print("\n==== HP screen gates (paired vs C0 on identical (panel,seed); "
          "KEEP iff dIC >= +0.0020 & t >= 2) ====")
    print(f"C0 levels: IC {sum(r['rank_ic'] for r in c0.values())/len(c0):+.4f}"
          f"  net {sum(r['net_pct'] for r in c0.values())/len(c0):+.1f}"
          f"  sharpe {sum(r['sharpe'] for r in c0.values())/len(c0):+.2f}"
          f"  (n={len(c0)} runs)")
    for arm in sorted({r["arm"] for r in out}):
        if arm == "hp_C0":
            continue
        ce = cells(arm)
        keys = sorted(set(ce) & set(c0))
        if len(keys) < 4:
            print(f"{arm:<20} INSUFFICIENT PAIRS ({len(keys)})")
            continue
        verdicts = {}
        for field in ("rank_ic", "net_pct", "sharpe"):
            # collapse panel pairs to seed level
            per_seed = {}
            for (pn, sd) in keys:
                per_seed.setdefault(sd, []).append(
                    ce[(pn, sd)][field] - c0[(pn, sd)][field])
            deltas = [sum(v) / len(v) for v in per_seed.values()]
            m = sum(deltas) / len(deltas)
            sdv = (sum((x - m) ** 2 for x in deltas) / (len(deltas) - 1)) ** 0.5
            se = sdv / len(deltas) ** 0.5
            verdicts[field] = (m, m / se if se > 0 else float("nan"))
        dic, tic = verdicts["rank_ic"]
        dsh, tsh = verdicts["sharpe"]
        dnet, tnet = verdicts["net_pct"]
        if dic >= 0.0020 and tic >= 2:
            v = "KEEP"
        elif dsh >= 0.10 and tsh >= 2:
            v = "FLAG (economics only)"
        else:
            v = "DROP"
        print(f"{arm:<20} dIC {dic:+.4f} (t{tic:+5.1f})   "
              f"dNet {dnet:+7.1f} (t{tnet:+5.1f})   "
              f"dSharpe {dsh:+.2f} (t{tsh:+5.1f})   -> {v}")


if __name__ == "__main__":
    main()
