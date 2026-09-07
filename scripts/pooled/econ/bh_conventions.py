#!/usr/bin/env python3
"""Buy & hold over 2020-2024 under both conventions, with risk stats.

The paper's window table currently reports buy & hold on the prior study's
convention (raw PRC: no dividends, no split adjustment).  Over the window
that table used, 2022-2024, that row is small.  Over the full registered
evaluation span it is not, so the two conventions need to be side by side
with the risk that produced them.

  prior   raw PRC price return, no dividends, no split adjustment.  This is
          the published benchmark convention and it is the one whose split
          handling ICAIF_BENCH_NOTE.md documents as a defect.
  total   panel RET (split- and dividend-adjusted), the same realized-return
          series every model arm is scored against.

Run:  python3 bh_conventions.py
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "baselines"))
sys.path.insert(0, os.path.dirname(HERE))

import score_stream as ss              # noqa: E402
from panel import Panel                # noqa: E402

PANELS = "/Users/jonathanchang/.claude/jobs/a28206de/tmp/panels_core7"
SETS = ["set1", "set2", "set3", "set4"]
W0, W1 = "2022-01-03", "2024-12-31"
YEARS = [str(y) for y in range(2022, 2025)]


def sharpe(v):
    v = np.asarray(v, dtype=float)
    sd = v.std(ddof=1)
    return float(v.mean() / sd * math.sqrt(252)) if sd > 0 else float("nan")


def mdd(v):
    eq = np.cumsum(np.asarray(v, dtype=float))
    return float(100 * np.max(np.maximum.accumulate(eq) - eq))


def prior_series(panel, rows):
    """Raw-PRC price return on a share count fixed at the first close."""
    n = panel.n_stocks
    r0 = max(rows[0] - 1, 0)
    shares = [(ss.CAPITAL / n) / abs(panel.prc[r0][k]) for k in range(n)]
    prev = sum(shares[k] * abs(panel.prc[r0][k]) for k in range(n))
    out = []
    for r in rows:
        val = sum(shares[k] * abs(panel.prc[r][k]) for k in range(n))
        out.append((val - prev) / ss.CAPITAL)
        prev = val
    return out


def total_series(panel, rows):
    """Equal-weight, bought once and held, compounding on realized RET."""
    n = panel.n_stocks
    pos = [ss.CAPITAL / n] * n
    out = []
    for r in rows:
        pnl = 0.0
        for k in range(n):
            ret = panel.Yscore[r][k]
            pnl += pos[k] * ret
            pos[k] *= 1.0 + ret
        out.append(pnl / ss.CAPITAL)
    return out


def main():
    acc = {"prior": {}, "total": {}}
    for s in SETS:
        panel = Panel(os.path.join(PANELS, s), "RET_CS")
        rows = panel.rows_between(W0, W1)
        for name, fn in (("prior", prior_series), ("total", total_series)):
            for r, v in zip(rows, fn(panel, rows)):
                acc[name].setdefault(panel.dates[r], []).append(v)

    print(f"{'convention':14s}" + "".join(f"{y:>9s}" for y in YEARS)
          + f"{'2020-24':>10s}{'Sharpe':>9s}{'MDD%':>8s}")
    for name, label in (("prior", "prior-hold"), ("total", "total-hold")):
        dates = sorted(acc[name])
        daily = np.array([np.mean(acc[name][d]) for d in dates])
        cells = []
        for y in YEARS:
            m = np.array([d.startswith(y) for d in dates])
            cells.append(100 * daily[m].sum())
        line = f"{label:14s}" + "".join(f"{c:>+9.1f}" for c in cells)
        print(line + f"{100 * daily.sum():>+10.1f}"
              f"{sharpe(daily):>9.2f}{mdd(daily):>8.1f}")

    # The prior study's ACTUAL per-year method: the position is rebuilt at
    # the start of every year and the yearly cells are summed, so a year's
    # return is never scaled by how the position did in earlier years.
    # This is what produced the published -10.1 / +11.2 / +6.8 cells, and
    # it is a different number from holding one position across the span.
    per_year = {y: [] for y in YEARS}
    daily_restart = {}
    for s in SETS:
        panel = Panel(os.path.join(PANELS, s), "RET_CS")
        for y in YEARS:
            rows = panel.rows_between(max(f"{y}-01-01", W0), f"{y}-12-31")
            if not rows:
                continue
            ser = prior_series(panel, rows)
            per_year[y].append(100 * float(np.sum(ser)))
            for r, v in zip(rows, ser):
                daily_restart.setdefault(panel.dates[r], []).append(v)
    dates = sorted(daily_restart)
    dr = np.array([np.mean(daily_restart[d]) for d in dates])
    cells = [float(np.mean(per_year[y])) for y in YEARS]
    print(f"{'prior-yearly':14s}" + "".join(f"{c:>+9.1f}" for c in cells)
          + f"{sum(cells):>+10.1f}{sharpe(dr):>9.2f}{mdd(dr):>8.1f}")
    print("\nprior-yearly is the published convention: position rebuilt each\n"
          "January, yearly cells summed (not compounded).")


if __name__ == "__main__":
    main()
