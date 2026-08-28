#!/usr/bin/env python3
"""Declared exploratory eval-span look at bp20 (HP_SCREEN.md, 2026-08-28).

Books the 40 hp2024_bp20 ensemble streams (sleeves + rank IC, 2022-24 and
2020-24) and pairs them per (panel, seed) against the registered primary
(C7_E ensemble rows from results_econ/m1_ensemble_test.csv). Exploratory:
not headline-eligible without confirmation + PRIMARY.md amendment.
"""
import os, csv, sys, importlib.util, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "baselines"))
import scoring, score_stream as ss              # noqa: E402
from panel import Panel                          # noqa: E402
from rebook import load_preds                    # noqa: E402
spec = importlib.util.spec_from_file_location("sw", "strategy_sweep.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

PANEL_ROOT = sys.argv[1] if len(sys.argv) > 1 else "/Users/jonathanchang/.claude/jobs/a28206de/tmp/panels_core7"
panels = {f"set{i}": Panel(f"{PANEL_ROOT}/set{i}", "RET_CS") for i in (1, 2, 3, 4)}
WINDOWS = [("2022-24", "2022-01-01", "2024-12-31"),
           ("2020-24", "2020-01-01", "2024-12-31")]

out = []
for run in sorted(os.listdir("bp20_stitch")):
    if "_seed" not in run:
        continue
    pn, seed = run.rsplit("_seed", 1)
    panel = panels[pn]
    preds, rows = load_preds(f"bp20_stitch/{run}/ensemble_stitched_predictions.csv",
                             panel, "2020-01-01", "2024-12-31")
    for wname, w0, w1 in WINDOWS:
        wrows = [r for r in rows if w0 <= panel.dates[r] <= w1]
        days = scoring.build_days(panel, preds, wrows)
        book = m.sleeves_book(panel, preds, wrows, 10, 10)
        net, sh, mdd = ss.book_stats(book["daily_ret"])
        ic, _, _ = ss.mean_se_t(ss.finite(ss.daily_ics(days, scoring.PRED, ss.spearman)))
        out.append({"panel": pn, "seed": int(seed), "window": wname,
                    "rank_ic": ic, "net_pct": net, "sharpe": sh})
        print(run, wname, f"{net:+.2f}", flush=True)

with open("results_econ/bp20_eval.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)

ctrl = [r for r in csv.DictReader(open("results_econ/m1_ensemble_test.csv"))
        if r["fleet"] == "C7_E" and r["rule"] == "ensemble"]
print("\n==== bp20 (eval span, exploratory) vs registered primary (C7_E), "
      "paired per (panel,seed), collapsed to seeds ====")
for w in ("2022-24", "2020-24"):
    c = {(r["panel"], int(r["seed"])): r for r in ctrl if r["window"] == w}
    b = {(r["panel"], r["seed"]): r for r in out if r["window"] == w}
    for field in ("rank_ic", "net_pct", "sharpe"):
        per, babs, cabs = {}, {}, {}
        for k in b:
            babs.setdefault(k[1], []).append(float(b[k][field]))
            if k in c:
                per.setdefault(k[1], []).append(float(b[k][field]) - float(c[k][field]))
                cabs.setdefault(k[1], []).append(float(c[k][field]))
        d = [sum(v) / len(v) for v in per.values()]
        mm = sum(d) / len(d); se = st.stdev(d) / len(d) ** 0.5
        bl = [sum(v) / len(v) for v in babs.values()]
        cl = [sum(v) / len(v) for v in cabs.values()]
        fmt = "%+.4f" if field == "rank_ic" else "%+.2f"
        print(f"{w} {field:<8} bp20 {fmt % (sum(bl)/len(bl))}  "
              f"primary {fmt % (sum(cl)/len(cl))}  "
              f"delta {fmt % mm} (t={mm/se:+.2f}, n={len(d)} seeds)")
