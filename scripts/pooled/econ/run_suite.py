#!/usr/bin/env python3
"""Sequential driver: the full baseline suite on every core7 panel.

Mirrors baselines/run_all.py's arm list and dependency rules but adds the two
things the economics table needs and run_all does not provide:

  * target-column exclusion on the core7 panels (via econ/run_one.py), with
    --param RET_CS and realised returns from the sidecar RET_raw_ columns;
  * a seed dimension for the stochastic arms (online LSTM/GRU and the
    periodic LSTMs).  Hyperparameters are tuned ONCE per panel at seed 42 on
    2016-2019 (the suite's built-in tuning) and seeds 43/44 re-run the frozen
    config, so seeds measure init/replay noise, never a second tuning pass.

Deterministic arms (trivial, ridge, AR, periodic ridge) run once at seed 42.
Everything runs sequentially; the parent process is expected to be started
under `nice -n 15` (a controls grid shares this machine).

    nice -n 15 python3 run_suite.py --panels /path/set1_core7 ... \
        [--results-dir results_econ] [--seeds 42,43,44]

Idempotent: a run whose metrics.json already exists is skipped.
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_ONE = os.path.join(HERE, "run_one.py")

# arm name -> (script, extra args, arch_from, config_from, stochastic)
# config_from/arch_from name the SEED-42 run of that arm on the same panel.
ARMS = [
    ("naive", "trivial.py", ["--model", "naive"], None, None, False),
    ("str1",  "trivial.py", ["--model", "str1"],  None, None, False),
    ("str5",  "trivial.py", ["--model", "str5"],  None, None, False),
    ("ridge", "online_ridge.py", [], None, None, False),
    ("ar",    "online_ar.py", [],   None, None, False),
    ("lstm",  "online_rnn.py", ["--cell", "lstm"], None, None, True),
    ("gru",   "online_rnn.py", ["--cell", "gru"],  None, None, True),
    # cadence sweep: tuned once (yearly for ridge, quarterly for the LSTM),
    # the other cadences freeze that config -- same rule as run_all.py
    ("periodic_ridge_yearly", "periodic_retrain.py",
     ["--model", "ridge", "--cadence", "yearly"], None, None, False),
    ("periodic_ridge_quarterly", "periodic_retrain.py",
     ["--model", "ridge", "--cadence", "quarterly"],
     None, "periodic_ridge_yearly", False),
    ("periodic_ridge_monthly", "periodic_retrain.py",
     ["--model", "ridge", "--cadence", "monthly"],
     None, "periodic_ridge_yearly", False),
    ("periodic_lstm_quarterly", "periodic_retrain.py",
     ["--model", "lstm", "--cadence", "quarterly"], "lstm", None, True),
    ("periodic_lstm_yearly", "periodic_retrain.py",
     ["--model", "lstm", "--cadence", "yearly"],
     "lstm", "periodic_lstm_quarterly", True),
    ("periodic_lstm_monthly", "periodic_retrain.py",
     ["--model", "lstm", "--cadence", "monthly"],
     "lstm", "periodic_lstm_quarterly", True),
]


def log(results_dir, msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(results_dir, "progress.log"), "a") as fh:
        fh.write(line + "\n")


def hyperparams_of(results_dir, arm, panel_name, seed=42):
    path = os.path.join(results_dir, arm, f"{panel_name}_seed{seed}",
                        "meta.json")
    if not os.path.exists(path):
        return None, path
    with open(path) as fh:
        return json.load(fh).get("hyperparameters", {}), path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panels", nargs="+", required=True)
    ap.add_argument("--results-dir", default=os.path.join(HERE, "results_econ"))
    ap.add_argument("--seeds", default="42,43,44",
                    help="seeds for the stochastic arms; the first one is the "
                         "tuning seed and the deterministic arms' only seed")
    ap.add_argument("--param", default="RET_CS")
    ap.add_argument("--score-from", default="2020-01-01")
    ap.add_argument("--score-to", default="2024-12-31")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    seed0 = seeds[0]
    os.makedirs(args.results_dir, exist_ok=True)
    failures = []

    t_start = time.time()
    for panel in args.panels:
        panel_name = os.path.basename(os.path.abspath(panel).rstrip("/"))
        for arm, script, extra, arch_from, config_from, stochastic in ARMS:
            arm_seeds = seeds if stochastic else [seed0]
            for seed in arm_seeds:
                out_dir = os.path.join(args.results_dir, arm,
                                       f"{panel_name}_seed{seed}")
                if os.path.exists(os.path.join(out_dir, "metrics.json")):
                    log(args.results_dir, f"skip {arm}/{panel_name}/s{seed} "
                                          "(metrics.json exists)")
                    continue
                os.makedirs(out_dir, exist_ok=True)
                cmd = ["nice", "-n", "15", sys.executable, RUN_ONE, script,
                       "--panel", panel, "--out-dir", out_dir,
                       "--seed", str(seed), "--param", args.param,
                       "--score-from", args.score_from,
                       "--score-to", args.score_to] + list(extra)

                # frozen-config reuse: (a) cadence variants inherit the tuned
                # cadence's config; (b) non-tuning seeds inherit the seed-42
                # config of the SAME arm.  (b) wins when both apply.
                cfg_src = None
                if seed != seed0:
                    cfg_src = arm
                elif config_from:
                    cfg_src = config_from
                if cfg_src:
                    hp, src_path = hyperparams_of(args.results_dir, cfg_src,
                                                  panel_name, seed0)
                    if hp is None:
                        failures.append((arm, panel_name, seed,
                                         f"missing config source {src_path}"))
                        log(args.results_dir,
                            f"SKIP {arm}/{panel_name}/s{seed}: needs "
                            f"{cfg_src} seed{seed0} first (not found)")
                        continue
                    cmd += ["--config", json.dumps(hp)]
                elif arch_from:
                    _, arch_path = hyperparams_of(args.results_dir, arch_from,
                                                  panel_name, seed0)
                    if not os.path.exists(arch_path):
                        failures.append((arm, panel_name, seed,
                                         f"missing arch source {arch_path}"))
                        log(args.results_dir,
                            f"SKIP {arm}/{panel_name}/s{seed}: needs "
                            f"{arch_from} seed{seed0} first (not found)")
                        continue
                    cmd += ["--arch-from", arch_path]

                log(args.results_dir,
                    f"run  {arm}/{panel_name}/s{seed}: {' '.join(cmd[3:])}")
                t0 = time.time()
                with open(os.path.join(out_dir, "run.log"), "w") as lg:
                    p = subprocess.run(cmd, stdout=lg,
                                       stderr=subprocess.STDOUT, cwd=HERE)
                dt = time.time() - t0
                if p.returncode != 0:
                    failures.append((arm, panel_name, seed,
                                     f"rc={p.returncode}"))
                    log(args.results_dir,
                        f"FAIL {arm}/{panel_name}/s{seed} rc={p.returncode} "
                        f"({dt:.0f}s), log: {out_dir}/run.log")
                else:
                    log(args.results_dir,
                        f"done {arm}/{panel_name}/s{seed} in {dt:.0f}s")

    log(args.results_dir, f"suite finished in {(time.time()-t_start)/60:.1f} "
                          f"min; {len(failures)} failures")
    for f in failures:
        log(args.results_dir, f"  failure: {f}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
