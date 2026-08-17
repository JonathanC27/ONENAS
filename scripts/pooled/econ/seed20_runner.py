#!/usr/bin/env python3
"""Extend online LSTM/GRU to 20 seeds (frozen 2016-19 configs) for matched-width ensembles."""
import json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
PANELS = [f"/Users/jonathanchang/.claude/jobs/fc17658e/tmp/set{i}_core7" for i in (1,2,3,4)]
for arm, cell in (("lstm","lstm"), ("gru","gru")):
    for panel in PANELS:
        pname = os.path.basename(panel)
        hp = json.load(open(f"{HERE}/results_econ/{arm}/{pname}_seed42/meta.json"))["hyperparameters"]
        for seed in range(53, 63):
            out = f"{HERE}/results_econ/{arm}/{pname}_seed{seed}"
            if os.path.exists(f"{out}/metrics.json"): continue
            os.makedirs(out, exist_ok=True)
            cmd = ["nice","-n","15",sys.executable,f"{HERE}/run_one.py","online_rnn.py",
                   "--panel",panel,"--out-dir",out,"--seed",str(seed),"--param","RET_CS",
                   "--score-from","2020-01-01","--score-to","2024-12-31",
                   "--cell",cell,"--config",json.dumps(hp)]
            t0=time.time()
            r=subprocess.run(cmd,stdout=open(f"{out}/run.log","w"),stderr=subprocess.STDOUT,cwd=HERE)
            print(f"{arm} {pname} s{seed}: rc={r.returncode} {time.time()-t0:.0f}s", flush=True)
print("SEED20_DONE")
