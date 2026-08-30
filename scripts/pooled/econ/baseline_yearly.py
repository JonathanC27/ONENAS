#!/usr/bin/env python3
"""Per-year sleeves cells for the online baselines (gru, periodic LSTM, AR)."""
import os, sys, importlib.util, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "baselines"))
sys.path.insert(0, os.path.dirname(HERE))
import score_stream as ss                      # noqa: E402
from panel import Panel                         # noqa: E402
from rebook import load_preds                   # noqa: E402
spec = importlib.util.spec_from_file_location("sw", "strategy_sweep.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

ROOT = "/Users/jonathanchang/.claude/jobs/a28206de/tmp/panels_core7"
panels = {f"set{i}": Panel(f"{ROOT}/set{i}", "RET_CS") for i in (1, 2, 3, 4)}
WINDOWS = [("2022", "2022-01-01", "2022-12-31"),
           ("2023", "2023-01-01", "2023-12-31"),
           ("2024", "2024-01-01", "2024-12-31"),
           ("2022-24", "2022-01-01", "2024-12-31")]

for arm in ("gru", "periodic_lstm_monthly", "ar"):
    per = {w: {} for w, _, _ in WINDOWS}
    base = f"results_econ/{arm}"
    for run in sorted(os.listdir(base)):
        p = os.path.join(base, run, "predictions.csv")
        if "_seed" not in run or not os.path.exists(p):
            continue
        pn, seed = run.rsplit("_seed", 1)
        pn = pn.replace("_core7", "")
        if arm != "ar" and not (42 <= int(seed) <= 51):
            continue
        preds, rows = load_preds(p, panels[pn], "2020-01-01", "2024-12-31")
        for wname, w0, w1 in WINDOWS:
            wrows = [r for r in rows if w0 <= panels[pn].dates[r] <= w1]
            net, sh, _ = ss.book_stats(
                m.sleeves_book(panels[pn], preds, wrows, 10, 10)["daily_ret"])
            per[wname].setdefault(seed, []).append(net)
    line = []
    n = 0
    for w, _, _ in WINDOWS:
        vals = [sum(v) / len(v) for v in per[w].values()]
        n = len(vals)
        mm = sum(vals) / n
        se = st.stdev(vals) / n ** 0.5 if n > 1 else float("nan")
        line.append(f"{mm:+.1f}±{se:.1f}" if se == se else f"{mm:+.1f}")
    print(arm, " | ".join(line), f"(n={n} seeds)", flush=True)
