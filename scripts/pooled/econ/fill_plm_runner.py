#!/usr/bin/env python3
"""Extend periodic_lstm_monthly to 40 seeds for the network-matched table.

periodic-LSTM-monthly is the strongest baseline wherever it can currently be
compared: at 8 networks it posts Sharpe 0.91 at top_k 10 -- equal to
ONE-NAS-40's headline Sharpe -- and the best baseline MDD. It is stuck at 10
seeds only because it retrains monthly (~611 s/run vs ~15 s for online LSTM),
so its 20- and 40-network cells are the one unrun comparison that could
overturn the paper's headline claim.

Frozen seed-42 hyperparameters per panel (run_suite.py's rule: seed 42 tunes on
2016-2019, every other seed reruns the frozen config), so this adds seeds only.
No tuning, no post-2019 data.

Seeds are ordered so that the 20-network cell completes FIRST (52..61), then the
40-network cell (62..81) -- scoring can start at 20 without waiting ~14 h more.
Cost: ~611 s x 4 panels x 30 seeds ~= 20 h wall clock at nice 15.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PANELS = [f"/Users/jonathanchang/.claude/jobs/fc17658e/tmp/set{i}_core7" for i in (1, 2, 3, 4)]
ARM = "periodic_lstm_monthly"

# 20-network cell first, then the 40-network cell
PHASES = [("20-net", range(52, 62)), ("40-net", range(62, 82))]

t_start = time.time()
done = failed = 0
for label, seeds in PHASES:
    print(f"=== phase {label}: seeds {seeds.start}..{seeds.stop - 1} ===", flush=True)
    for seed in seeds:
        for panel in PANELS:
            pname = os.path.basename(panel)
            out = f"{HERE}/results_econ/{ARM}/{pname}_seed{seed}"
            if os.path.exists(f"{out}/predictions.csv"):
                continue
            hp = json.load(open(f"{HERE}/results_econ/{ARM}/{pname}_seed42/meta.json"))["hyperparameters"]
            os.makedirs(out, exist_ok=True)
            cmd = ["nice", "-n", "15", sys.executable, f"{HERE}/run_one.py",
                   "periodic_retrain.py", "--panel", panel, "--out-dir", out,
                   "--seed", str(seed), "--param", "RET_CS",
                   "--score-from", "2020-01-01", "--score-to", "2024-12-31",
                   "--model", "lstm", "--cadence", "monthly",
                   "--config", json.dumps(hp)]
            t0 = time.time()
            r = subprocess.run(cmd, stdout=open(f"{out}/run.log", "w"),
                               stderr=subprocess.STDOUT, cwd=HERE)
            ok = r.returncode == 0
            done += ok
            failed += (not ok)
            print(f"{ARM} {pname} s{seed}: rc={r.returncode} "
                  f"{time.time()-t0:.0f}s  [{done} done, {failed} failed, "
                  f"{(time.time()-t_start)/3600:.1f}h elapsed]", flush=True)
    print(f"PHASE_{label}_DONE", flush=True)

print(f"FILLPLM_DONE ran={done} failed={failed} "
      f"elapsed={(time.time()-t_start)/3600:.1f}h")
