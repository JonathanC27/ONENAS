#!/usr/bin/env python3
"""Ensembling invariance panel: ensemble - single champion inside every HP cell.

For each of the 15 HP-screen configurations (90 runs, 2016-2019 tuning
clock), score the island-champion rank-mean ensemble (hp_stitch/) and the
single global-best genome (inv_gb/) from the same run under the registered
sleeves book + daily rank IC, and report the within-run paired delta per
cell. Companion to M1: shows the aggregation gain is a property of the
method class, not of any one configuration.
"""
import os, sys, csv, importlib.util, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "baselines"))
sys.path.insert(0, os.path.dirname(HERE))
import scoring, score_stream as ss   # noqa: E402
from panel import Panel               # noqa: E402
from rebook import load_preds         # noqa: E402
_spec = importlib.util.spec_from_file_location("sw", "strategy_sweep.py")
_sw = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_sw)

PANEL_ROOT = sys.argv[1] if len(sys.argv) > 1 else \
    "/Users/jonathanchang/.claude/jobs/a28206de/tmp/panels_core7"
panels = {f"set{i}": Panel(f"{PANEL_ROOT}/set{i}", "RET_CS") for i in (1, 2)}
W0, W1 = "2016-01-01", "2019-12-31"

out = []
for arm in sorted(os.listdir("hp_stitch")):
    adir = os.path.join("hp_stitch", arm)
    if not os.path.isdir(adir):
        continue
    for run in sorted(os.listdir(adir)):
        pn, seed = run.rsplit("_seed", 1)
        if pn not in panels:
            continue
        panel = panels[pn]
        rec = {"arm": arm, "panel": pn, "seed": int(seed)}
        ok = True
        for rule, root in (("ens", "hp_stitch"), ("gb", "inv_gb")):
            path = os.path.join(root, arm, run,
                                "ensemble_stitched_predictions.csv")
            if not os.path.exists(path):
                ok = False
                break
            preds, rows = load_preds(path, panel, W0, W1)
            days = scoring.build_days(panel, preds, rows)
            book = _sw.sleeves_book(panel, preds, rows, 10, 10)
            net, sh, _ = ss.book_stats(book["daily_ret"])
            ic, _, _ = ss.mean_se_t(ss.finite(
                ss.daily_ics(days, scoring.PRED, ss.spearman)))
            rec[f"{rule}_ic"], rec[f"{rule}_net"], rec[f"{rule}_sh"] = ic, net, sh
        if ok:
            out.append(rec)
            print(arm, run, f"dnet {rec['ens_net']-rec['gb_net']:+.2f}",
                  flush=True)

with open("results_econ/invariance_panel.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader()
    w.writerows(out)

print("\n==== ensembling gain (ensemble - single, within-run) per HP cell, "
      "2016-19 ====")
pos = 0
for arm in sorted({r["arm"] for r in out}):
    rows = [r for r in out if r["arm"] == arm]
    dn = [r["ens_net"] - r["gb_net"] for r in rows]
    dsh = [r["ens_sh"] - r["gb_sh"] for r in rows]
    dic = [r["ens_ic"] - r["gb_ic"] for r in rows]
    n = len(dn)
    mn = sum(dn) / n; msh = sum(dsh) / n; mic = sum(dic) / n
    tn = mn / (st.stdev(dn) / n ** 0.5) if n > 1 else float("nan")
    if mn > 0:
        pos += 1
    print(f"{arm:<20} dNet {mn:+7.1f} (t{tn:+5.1f})  dSharpe {msh:+.2f}  "
          f"dIC {mic:+.4f}  (n={n} runs)")
print(f"\ncells with positive ensembling net gain: {pos}/"
      f"{len({r['arm'] for r in out})}")
