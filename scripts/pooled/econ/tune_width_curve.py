#!/usr/bin/env python3
"""Which island width is best on the 2016-2019 tuning span?

This is the protocol-symmetric view: the tuning span is where a
configuration choice may legitimately be made, so it is the curve that
should govern the width decision, unlike the eval-span curve in
Figure 2.

Widths 8/16/20 exist at seeds 42-46 only; 40/50/60 run to seed 51. Both
views are printed: the common seed set 42-46 (n=20 cells at every width,
so the widths are directly comparable) and each width at its own full
replication.

    python3 tune_width_curve.py
"""
import importlib.util
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

_spec = importlib.util.spec_from_file_location(
    "sw", os.path.join(HERE, "strategy_sweep.py"))
_sw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sw)

PANELS = "/Users/jonathanchang/.claude/jobs/a28206de/tmp/panels_core7"
SETS = ["set1", "set2", "set3", "set4"]
W0, W1 = "2016-01-01", "2019-12-31"
STITCHED = "ensemble_stitched_predictions.csv"
COMMON = list(range(42, 47))          # every width has these

SOURCES = {
    8:  [(42, 46, "probe_TUNE16_ISL8")],
    16: [(42, 46, "probe_TUNE16_ISL16")],
    20: [(42, 46, "probe_TUNE16_ISL20")],
    40: [(42, 46, "probe_TUNE16_ISL40"),
         (47, 51, "tune_sweep/tune_islands_40")],
    50: [(42, 51, "tune_sweep/tune_islands_50")],
    60: [(42, 51, "tune_sweep/tune_islands_60")],
}


def main():
    panels = {s: Panel(os.path.join(PANELS, s), "RET_CS") for s in SETS}
    cells = defaultdict(dict)
    for width, srcs in SOURCES.items():
        for lo, hi, d in srcs:
            base = os.path.join(HERE, d)
            if not os.path.isdir(base):
                print(f"  missing {base}", flush=True)
                continue
            for s in SETS:
                for sd in range(lo, hi + 1):
                    p = os.path.join(base, f"{s}_seed{sd}", STITCHED)
                    if not os.path.exists(p):
                        continue
                    preds, rows = load_preds(p, panels[s], W0, W1)
                    rows = [r for r in rows if r > 0]
                    if not rows:
                        continue
                    days = scoring.build_days(panels[s], preds, rows)
                    book = _sw.sleeves_book(panels[s], preds, rows, 10, 10)
                    net, sharpe, mdd = ss.book_stats(book["daily_ret"])
                    ic, _, _ = ss.mean_se_t(
                        ss.finite(ss.daily_ics(days, scoring.PRED,
                                               ss.spearman)))
                    cells[width][(s, sd)] = (ic, net, sharpe, mdd)
        print(f"  width {width:>2}: {len(cells[width])} cells", flush=True)

    def report(title, seedset):
        print(f"\n=== {title} ===")
        print(f"{'islands':>8}{'n':>5}{'rank IC':>10}{'net%':>9}"
              f"{'Sharpe':>8}{'MDD%':>7}")
        best_ic, best_w = -9, None
        for w in sorted(cells):
            v = [x for k, x in cells[w].items()
                 if seedset is None or k[1] in seedset]
            if not v:
                continue
            ic = float(np.mean([x[0] for x in v]))
            print(f"{w:>8}{len(v):>5}{ic:>+10.4f}"
                  f"{float(np.mean([x[1] for x in v])):>+9.1f}"
                  f"{float(np.mean([x[2] for x in v])):>8.2f}"
                  f"{float(np.mean([x[3] for x in v])):>7.1f}")
            if ic > best_ic:
                best_ic, best_w = ic, w
        print(f"  -> highest tuning-span rank IC: {best_w} islands "
              f"({best_ic:+.4f})")

    report("COMMON SEEDS 42-46 (directly comparable)", set(COMMON))
    report("EACH WIDTH AT FULL REPLICATION", None)


if __name__ == "__main__":
    main()
