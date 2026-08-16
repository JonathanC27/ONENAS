#!/usr/bin/env python3
"""Does frozen 2016-19 tuning handicap the LSTM baseline on 2020-2024?
Runs the top-5 distinct configs (by 2016-19 val rank IC) forward and compares."""
import json, math, os, subprocess, sys, tempfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "baselines"))
sys.path.insert(0, os.path.dirname(HERE))
import scoring, score_stream as ss
from panel import Panel
from rebook import load_preds
TMP = "/Users/jonathanchang/.claude/jobs/fc17658e/tmp"
SETS = ["set1", "set2", "set3", "set4"]
PAN = {s: Panel(f"{TMP}/{s}_core7", "RET_CS") for s in SETS}
m = json.load(open(f"{HERE}/results_econ/lstm/set1_core7_seed42/meta.json"))
tbl = sorted([e for e in m["tuning_table"] if e["val_rank_ic"] == e["val_rank_ic"]],
             key=lambda e: -e["val_rank_ic"])
seen, cfgs = set(), []
for e in tbl:
    k = json.dumps(e["config"], sort_keys=True)
    if k in seen: continue
    seen.add(k); cfgs.append((e["val_rank_ic"], e["config"]))
    if len(cfgs) >= 5: break
chosen = json.dumps(m["hyperparameters"], sort_keys=True)
def sharpe(v): v = np.asarray(v); return v.mean() / v.std(ddof=1) * math.sqrt(252)
print(f"{'rank':>5}{'valIC(16-19)':>14}{'net(20-24)':>12}{'Sharpe':>9}   config", flush=True)
for i, (vic, c) in enumerate(cfgs):
    pp = {}
    for s in SETS:
        out = tempfile.mkdtemp()
        cmd = ["nice", "-n", "15", sys.executable, f"{HERE}/run_one.py", "online_rnn.py",
               "--panel", f"{TMP}/{s}_core7", "--out-dir", out, "--seed", "42",
               "--param", "RET_CS", "--score-from", "2020-01-01", "--score-to", "2024-12-31",
               "--cell", "lstm", "--config", json.dumps(c)]
        subprocess.run(cmd, stdout=open(f"{out}/log", "w"), stderr=subprocess.STDOUT, cwd=HERE)
        panel = PAN[s]
        pr, rows = load_preds(f"{out}/predictions.csv", panel, "2020-01-01", "2024-12-31")
        days = scoring.build_days(panel, pr, rows)
        bk = ss.run_book(days, scoring.PRED, panel.prc, panel.tc, 10, None,
                         book="sleeves", hold_days=10)
        pp[s] = dict(zip([d[1] for d in days], bk["daily_ret"]))
    cc = sorted(set.intersection(*[set(v) for v in pp.values()]))
    pool = np.array([[pp[s][x] for s in SETS] for x in cc]).mean(1)
    tag = "  <-- CHOSEN" if json.dumps(c, sort_keys=True) == chosen else ""
    print(f"{i+1:>5}{vic:>14.5f}{100*pool.sum():>12.1f}{sharpe(pool):>9.2f}   "
          f"h{c['hidden']} lr{c['lr']} sq{c['seq_len']} spd{c['steps_per_day']} "
          f"rp{c['replay_days']} dm{int(c['cs_demean_y'])}{tag}", flush=True)
