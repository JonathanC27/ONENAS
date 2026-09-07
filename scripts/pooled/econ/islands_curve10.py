#!/usr/bin/env python3
"""Width curve on a regular grid: 10/20/30/40/50/60 islands.

Replaces the irregular 8/16/20/40 curve behind paper/islands_scaling.pdf
(HP_SCREEN.md, declared 2026-09-06).  Widths 20 and 40 are the existing
eval-span fleets, reused unchanged; 10/30/50/60 come from the refresh.

Per width this reports pooled net % (the panel-diversified portfolio, which
is what the deployment would actually hold) and the seed distribution of
Sharpe, because the campaign's finding is that width buys reliability more
than it buys mean return.

    python3 islands_curve10.py [--panels-dir DIR] [--fleet-dir DIR]

Writes results_econ/islands_curve10.csv and, unless --no-figure,
paper/islands_scaling.pdf.
"""
import argparse
import csv
import math
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "baselines"))
sys.path.insert(0, os.path.dirname(HERE))

import scoring                        # noqa: E402
import score_stream as ss             # noqa: E402
from panel import Panel               # noqa: E402
from rebook import load_preds         # noqa: E402

SETS = ["set1", "set2", "set3", "set4"]
SEEDS = list(range(42, 52))
WIDTHS = [10, 20, 30, 40, 50, 60]
FROM, TO = "2022-01-01", "2024-12-31"
TOP_K, HOLD = 10, 10
STITCHED = "ensemble_stitched_predictions.csv"


def isl8_dir(sd):
    return "onenas_c7e" if sd <= 44 else ("probe_s4547" if sd <= 47
                                          else "probe_s4851")


def run_path(width, s, sd, fleet_dir):
    """Where this (width, panel, seed) run's stitched predictions live.

    20 and 40 are the pre-existing fleets kept in the repo; the refreshed
    widths live under the sweep's scratch tree (rsynced to --fleet-dir).
    """
    if width == 20:
        return os.path.join(HERE, "probe_ISL20", f"{s}_seed{sd}", STITCHED)
    if width == 40:
        return os.path.join(HERE, "probe_ISL40", f"{s}_seed{sd}", STITCHED)
    return os.path.join(fleet_dir, f"islands_{width}", f"{s}_seed{sd}",
                        STITCHED)


def sharpe(v):
    v = np.asarray(v, dtype=float)
    sd = v.std(ddof=1)
    return float(v.mean() / sd * math.sqrt(252)) if sd > 0 else float("nan")


def mdd(v):
    eq = np.cumsum(np.asarray(v, dtype=float))
    return float(100 * np.max(np.maximum.accumulate(eq) - eq))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels-dir",
                    default="/Users/jonathanchang/.claude/jobs/a28206de/tmp/panels_core7")
    ap.add_argument("--fleet-dir",
                    default=os.path.join(HERE, "islands_sweep"))
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    PAN = {s: Panel(os.path.join(args.panels_dir, s), "RET_CS") for s in SETS}

    rows, missing = [], defaultdict(int)
    # per (width, seed): the pooled daily series across panels, which is the
    # portfolio a deployment holds; and per-run dailies for the seed spread.
    pooled = defaultdict(lambda: defaultdict(list))
    per_run = defaultdict(list)

    for w in WIDTHS:
        for s in SETS:
            for sd in SEEDS:
                p = run_path(w, s, sd, args.fleet_dir)
                if not os.path.exists(p):
                    missing[w] += 1
                    continue
                preds, prows = load_preds(p, PAN[s], FROM, TO)
                prows = [r for r in prows if r > 0]
                if not prows:
                    missing[w] += 1
                    continue
                days = scoring.build_days(PAN[s], preds, prows)
                book = ss.run_book(days, 2, PAN[s].prc, PAN[s].tc, TOP_K,
                                   book="sleeves", hold_days=HOLD)
                dr = book["daily_ret"]
                for d, r in zip([x[1] for x in days], dr):
                    pooled[w][sd].append((d, r))
                per_run[w].append(sharpe(dr))
                rows.append({"islands": w, "panel": s, "seed": sd,
                             "net_pct": 100.0 * float(np.sum(dr)),
                             "sharpe": sharpe(dr), "mdd_pct": mdd(dr)})
        print(f"  width {w:>2}: {len([r for r in rows if r['islands'] == w])}"
              f" runs loaded, {missing[w]} missing", flush=True)

    out = []
    for w in WIDTHS:
        if not pooled[w]:
            continue
        # pooled portfolio per seed: average the panels' daily returns on
        # the shared calendar, then take that seed's net and Sharpe
        seed_net, seed_sh = [], []
        for sd, pairs in pooled[w].items():
            by_date = defaultdict(list)
            for d, r in pairs:
                by_date[d].append(r)
            daily = np.array([np.mean(by_date[d]) for d in sorted(by_date)])
            seed_net.append(100.0 * daily.sum())
            seed_sh.append(sharpe(daily))
        seed_net = np.asarray(seed_net)
        seed_sh = np.asarray(seed_sh)
        out.append({
            "islands": w, "n_seeds": len(seed_sh),
            "net": float(seed_net.mean()),
            "net_se": float(seed_net.std(ddof=1) / math.sqrt(len(seed_net))),
            "sharpe": float(seed_sh.mean()),
            "sharpe_sd": float(seed_sh.std(ddof=1)),
            "worst_sharpe": float(seed_sh.min()),
        })

    print(f"\n{'isl':>4} {'n':>3} {'net':>8} {'±SE':>6} {'Sharpe':>8} "
          f"{'SD':>6} {'worst':>7}")
    for r in out:
        print(f"{r['islands']:>4} {r['n_seeds']:>3} {r['net']:>+8.1f} "
              f"{r['net_se']:>6.1f} {r['sharpe']:>8.2f} {r['sharpe_sd']:>6.3f} "
              f"{r['worst_sharpe']:>7.2f}", flush=True)

    rd = os.path.join(HERE, "results_econ")
    os.makedirs(rd, exist_ok=True)
    with open(os.path.join(rd, "islands_curve10.csv"), "w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(out[0]))
        w_.writeheader()
        w_.writerows(out)
    with open(os.path.join(rd, "islands_curve10_runs.csv"), "w",
              newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=["islands", "panel", "seed",
                                            "net_pct", "sharpe", "mdd_pct"])
        w_.writeheader()
        w_.writerows(rows)
    print(f"\nwrote {rd}/islands_curve10.csv", flush=True)

    if args.no_figure or len(out) < 2:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ISL = [r["islands"] for r in out]
    NET = [r["net"] for r in out]
    NSE = [r["net_se"] for r in out]
    SH = [r["sharpe"] for r in out]
    SSD = [r["sharpe_sd"] for r in out]
    WOR = [r["worst_sharpe"] for r in out]

    BLUE, INK2 = "#2a78d6", "#5a5c61"
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.4, 1.75))

    ax1.errorbar(ISL, NET, yerr=NSE, color=BLUE, lw=1.2, marker="o", ms=3,
                 capsize=3)
    ax1.set_xlabel("Islands")
    ax1.set_ylabel("Net return 2022–24 (%)")
    ax1.set_xticks(ISL)
    ax1.grid(axis="y", color="#ececea", lw=0.6)

    ax2.fill_between(ISL, [m - s for m, s in zip(SH, SSD)],
                     [m + s for m, s in zip(SH, SSD)],
                     color=BLUE, alpha=0.15, lw=0, label="±1 seed SD")
    ax2.plot(ISL, SH, color=BLUE, lw=1.2, marker="o", ms=3,
             label="Mean Sharpe")
    ax2.plot(ISL, WOR, color=INK2, lw=0.9, ls="--", marker="s", ms=2.5,
             label="Worst seed")
    ax2.set_xlabel("Islands")
    ax2.set_ylabel("Sharpe, 2022–24")
    ax2.set_xticks(ISL)
    ax2.grid(axis="y", color="#ececea", lw=0.6)
    ax2.legend(frameon=False, fontsize=5.5, loc="lower right")

    fig.tight_layout()
    for dest in (os.path.join(HERE, "../../../paper/islands_scaling.pdf"),
                 "/Users/jonathanchang/Documents/ONENAS-paper/islands_scaling.pdf"):
        fig.savefig(dest)
    print("wrote islands_scaling.pdf", flush=True)


if __name__ == "__main__":
    main()
