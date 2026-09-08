#!/usr/bin/env python3
"""Why the Online AR paired difference is not 27.5 - (-3.3).

There is a CHECK comment on the window table asking this. Online AR is
deterministic and exists at seed 42 only, so its paired comparison uses
the four (panel, seed 42) cells rather than all forty. The ONE-NAS mean
over that subset is not the ONE-NAS mean over the full grid, and the
paired difference is computed on the subset, as it must be.

    python3 check_ar_pairing.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "baselines"))
sys.path.insert(0, os.path.dirname(HERE))

import scoring                        # noqa: E402
import score_stream as ss             # noqa: E402
from panel import Panel               # noqa: E402
from rebook import load_preds         # noqa: E402

PANELS = "/Users/jonathanchang/.claude/jobs/a28206de/tmp/panels_core7"
SETS = ["set1", "set2", "set3", "set4"]
FROM, TO = "2022-01-01", "2024-12-31"


def net(path, pan):
    if not os.path.exists(path):
        return None
    preds, rows = load_preds(path, pan, FROM, TO)
    rows = [r for r in rows if r > 0]
    if not rows:
        return None
    days = scoring.build_days(pan, preds, rows)
    book = ss.run_book(days, 2, pan.prc, pan.tc, 10, book="sleeves",
                       hold_days=10)
    return 100.0 * float(np.sum(book["daily_ret"]))


def main():
    pan = {s: Panel(os.path.join(PANELS, s), "RET_CS") for s in SETS}
    onenas, ar = {}, {}
    for s in SETS:
        for sd in range(42, 52):
            v = net(os.path.join(
                HERE, f"probe_ISL40/{s}_seed{sd}/"
                      "ensemble_stitched_predictions.csv"), pan[s])
            if v is not None:
                onenas[(s, sd)] = v
        v = net(os.path.join(
            HERE, f"results_econ/ar/{s}_core7_seed42/predictions.csv"),
            pan[s])
        if v is not None:
            ar[s] = v

    all_cells = float(np.mean(list(onenas.values())))
    seed42 = float(np.mean([v for (s, sd), v in onenas.items() if sd == 42]))
    ar_mean = float(np.mean(list(ar.values())))
    d = [onenas[(s, 42)] - ar[s] for s in ar if (s, 42) in onenas]

    print(f"ONE-NAS 40 isl, all {len(onenas)} cells : {all_cells:+.1f}")
    print(f"ONE-NAS 40 isl, seed 42 only     : {seed42:+.1f}"
          "   <- the cells AR is paired against")
    print(f"Online AR, {len(ar)} cells            : {ar_mean:+.1f}")
    print(f"paired mean difference           : {float(np.mean(d)):+.1f}"
          f"  (n={len(d)})")
    print(f"naive difference of table means  : {all_cells - ar_mean:+.1f}")


if __name__ == "__main__":
    main()
