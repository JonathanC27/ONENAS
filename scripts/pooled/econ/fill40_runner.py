#!/usr/bin/env python3
"""Fill online LSTM/GRU to 40 seeds present on ALL FOUR panels.

Needed for the network-matched column: ONE-NAS at 40 islands (the config
Amendment 6 selects) has to be compared against a 40-network baseline ensemble,
and the killed 48-seed job left set4 short for lstm and skipped seed 52 for gru.

Frozen seed-42 hyperparameters throughout (run_suite.py's rule: seed 42 tunes,
every other seed reruns the frozen config), so this adds seeds only -- it does
not re-tune anything and touches no post-2019 data.

periodic_lstm_monthly is deliberately NOT extended here: it retrains monthly and
costs ~611 s/run, so 40 seeds x 4 panels would be ~20 h of local CPU.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PANELS = [f"/Users/jonathanchang/.claude/jobs/fc17658e/tmp/set{i}_core7" for i in (1, 2, 3, 4)]
TARGET = 82          # seeds 42..81 inclusive -> 40 seeds

done = skipped = failed = 0
t_start = time.time()
for arm, cell in (("lstm", "lstm"), ("gru", "gru")):
    for panel in PANELS:
        pname = os.path.basename(panel)
        hp = json.load(open(f"{HERE}/results_econ/{arm}/{pname}_seed42/meta.json"))["hyperparameters"]
        for seed in range(42, TARGET):
            out = f"{HERE}/results_econ/{arm}/{pname}_seed{seed}"
            if os.path.exists(f"{out}/predictions.csv"):
                skipped += 1
                continue
            os.makedirs(out, exist_ok=True)
            cmd = ["nice", "-n", "15", sys.executable, f"{HERE}/run_one.py",
                   "online_rnn.py", "--panel", panel, "--out-dir", out,
                   "--seed", str(seed), "--param", "RET_CS",
                   "--score-from", "2020-01-01", "--score-to", "2024-12-31",
                   "--cell", cell, "--config", json.dumps(hp)]
            t0 = time.time()
            r = subprocess.run(cmd, stdout=open(f"{out}/run.log", "w"),
                               stderr=subprocess.STDOUT, cwd=HERE)
            ok = r.returncode == 0
            done += ok
            failed += (not ok)
            print(f"{arm} {pname} s{seed}: rc={r.returncode} "
                  f"{time.time()-t0:.0f}s", flush=True)

print(f"FILL40_DONE ran={done} skipped={skipped} failed={failed} "
      f"elapsed={time.time()-t_start:.0f}s")
