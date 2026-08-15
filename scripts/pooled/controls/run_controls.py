#!/usr/bin/env python3
"""Run the no-search controls over panels x seeds x {control1, control2}.

    python3 run_controls.py \
        --panels /path/to/set1_v2 /path/to/set2_v2 --seeds 42 43

Results tree (baselines' convention: predictions.csv / book_daily.csv are
gitignored, meta.json / metrics.json / run.log are tracked):

    results_controls/<control>/<panel>_seed<sd>/single/    predictions.csv,
                                                           metrics.json, meta.json
                                               /ensemble/  same
                                               /champion_trace.json
                                               /run.log
    results_controls/summary.csv

Control 2's 2016-2019 grid search runs once per panel (fixed internal tuning
seed, so the config cannot depend on --seed); later seeds of the same panel
reuse the tuned config via --config.  --table-only rebuilds summary.csv from
metrics.json files already on disk.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CONTROLS = {"control1": "control1_random.py",
            "control2": "control2_fixed_ensemble.py"}


def run_one(control, panel, seed, results_dir, extra, frozen_cfg=None):
    name = os.path.basename(panel.rstrip("/"))
    out = os.path.join(results_dir, control, "%s_seed%d" % (name, seed))
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, os.path.join(HERE, CONTROLS[control]),
           "--panel", panel, "--seed", str(seed), "--out-dir", out] + extra
    if control == "control2" and frozen_cfg:
        cmd += ["--config", json.dumps(frozen_cfg)]
    print("== %s %s seed %d" % (control, name, seed), flush=True)
    t0 = time.time()
    with open(os.path.join(out, "run.log"), "w") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    secs = time.time() - t0
    if p.returncode != 0:
        print("   FAILED (%.0fs) -- see %s/run.log" % (secs, out))
        return out, secs, False
    print("   done in %.0fs -> %s" % (secs, out))
    return out, secs, True


def find_frozen_cfg(results_dir, panel_name):
    """Control2 config already tuned for this panel in an earlier run, if any.

    The tuning pass uses a fixed internal seed, so any completed run of the
    same panel carries the identical config; reusing it keeps 'tune once per
    panel, frozen across seeds' true across separate invocations.
    """
    base = os.path.join(results_dir, "control2")
    if not os.path.isdir(base):
        return None
    for run in sorted(os.listdir(base)):
        if run.rsplit("_seed", 1)[0] != panel_name:
            continue
        path = os.path.join(base, run, "ensemble", "meta.json")
        if os.path.exists(path):
            with open(path) as fh:
                return json.load(fh)["hyperparameters"]
    return None


def read_metrics(out, variant):
    path = os.path.join(out, variant, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        m = json.load(fh)
    o = m["summary"]["overall"]["model"]
    return {"pearson_ic": o.get("pearson_ic"),
            "rank_ic_1": o.get("rank_ic_1"),
            "rank_ic_t": o.get("rank_ic_t"),
            "rank_ic_5": o.get("rank_ic_5"),
            "net_pct": o.get("net_pct"),
            "sharpe": o.get("sharpe"),
            "n_days": o.get("n_days")}


def summarize(results_dir):
    rows = []
    for control in sorted(CONTROLS):
        base = os.path.join(results_dir, control)
        if not os.path.isdir(base):
            continue
        for run in sorted(os.listdir(base)):
            out = os.path.join(base, run)
            for variant in ("single", "ensemble"):
                m = read_metrics(out, variant)
                if m:
                    rows.append(dict(control=control, run=run,
                                     variant=variant, **m))
    if not rows:
        print("no metrics.json found under %s" % results_dir)
        return
    path = os.path.join(results_dir, "summary.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    fmt = "%-9s %-16s %-8s %5s  %9s %9s %7s %9s %8s %8s"
    print(fmt % ("control", "run", "variant", "days", "pearsonIC",
                 "rankIC@1", "t", "rankIC@5", "net%", "sharpe"))
    for r in rows:
        print(fmt % (r["control"], r["run"], r["variant"], r["n_days"],
                     "%+.4f" % r["pearson_ic"], "%+.4f" % r["rank_ic_1"],
                     "%+.2f" % r["rank_ic_t"], "%+.4f" % r["rank_ic_5"],
                     "%+.2f" % r["net_pct"], "%+.2f" % r["sharpe"]))
    print("wrote %s" % path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panels", nargs="+", required=True,
                    help="panel directories")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    ap.add_argument("--controls", nargs="+", default=["control1", "control2"],
                    choices=sorted(CONTROLS))
    ap.add_argument("--results-dir",
                    default=os.path.join(HERE, "results_controls"))
    ap.add_argument("--extra", default="",
                    help="extra args passed through to both control scripts, "
                         "e.g. \"--param RET_CS --cost-bps 5\"")
    ap.add_argument("--table-only", action="store_true")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip runs whose single/ and ensemble/ metrics.json "
                         "already exist (resume an interrupted grid)")
    args = ap.parse_args()

    if args.table_only:
        summarize(args.results_dir)
        return

    extra = args.extra.split() if args.extra else []
    t0 = time.time()
    for panel in args.panels:
        frozen_cfg = find_frozen_cfg(args.results_dir,
                                     os.path.basename(panel.rstrip("/")))
        if frozen_cfg:
            print("reusing control2 config already tuned for this panel: %s"
                  % json.dumps(frozen_cfg, sort_keys=True))
        for seed in args.seeds:
            for control in args.controls:
                name = os.path.basename(panel.rstrip("/"))
                out0 = os.path.join(args.results_dir, control,
                                    "%s_seed%d" % (name, seed))
                if args.skip_existing and all(
                        os.path.exists(os.path.join(out0, v, "metrics.json"))
                        for v in ("single", "ensemble")):
                    print("== %s %s seed %d: already complete, skipped"
                          % (control, name, seed), flush=True)
                    continue
                out, _, ok = run_one(control, panel, seed, args.results_dir,
                                     extra, frozen_cfg)
                if ok and control == "control2" and frozen_cfg is None:
                    with open(os.path.join(out, "ensemble",
                                           "meta.json")) as fh:
                        frozen_cfg = json.load(fh)["hyperparameters"]
                    print("   control2 config frozen for later seeds: %s"
                          % json.dumps(frozen_cfg, sort_keys=True))
    print("total %.0fs" % (time.time() - t0))
    summarize(args.results_dir)


if __name__ == "__main__":
    main()
