#!/usr/bin/env python3
"""
watchdog_v3.py --run <run_dir> [options]

SIDECAR collapse watchdog.  Python only -- it touches no C++ and changes no
search behaviour; it reads the files ONE-NAS is already writing.

WHY IT EXISTS.  Three separate things left a 610-generation, 5-seed campaign
with no in-flight signal at all:

  * `--max_pred_sd_ratio` is a ONE-SIDED guard (rnn/rnn_genome.cxx:1465): it
    rejects a genome whose prediction sd is MORE than 3.0x the target sd, and
    never rejects one whose sd is ZERO.  A collapsed genome -- the exact v1
    failure, mean -0.963 sd 0.001 normalised -- is always eligible.
  * The guard INVERTS on low-variance windows.  47 of 610 generations have a
    pooled validation-target sd below 0.010 against a scored-span sd of 0.173,
    so on those windows the 3.0x cap forbids any genome with realistic amplitude
    while a constant sails through.
  * run_v2.sh:74 set `--std_message_level ERROR`, and log.hxx:120-127 orders the
    levels FATAL < ERROR < WARNING < INFO, so ERROR suppressed WARNING and INFO:
    "ALL elite genomes failed the exploding-prediction guard" and every
    global-best update were invisible.  run_v3.sh raises this to INFO into a
    file, with WARNING and above echoed to the job stdout.

WHAT IT DOES.  Every --interval seconds it reads the new
`generation_*_global_best.csv` files, computes sd(global_best_predicted_N2O) /
sd(expected_N2O) on that generation's TEST window, and if the ratio is below
--threshold for --consecutive generations in a row it logs FATAL and cancels the
job.  Every poll prints a heartbeat, so a run that is fine also produces signal.

DELIBERATE NON-BEHAVIOURS:
  * A generation whose target sd is 0 gives an undefined ratio.  It is logged and
    SKIPPED: it neither extends nor resets the streak.  Making it reset the
    streak would let a flat window hide a collapse; making it extend the streak
    would cancel healthy runs on flat windows.
  * Files whose mtime predates the manifest are skipped as stale (they belong to
    an earlier run in the same directory -- see score_v3.py defect 1).
  * Partially-written files (row count != L-1) are skipped and retried.
  * The watchdog never edits anything the scorer reads.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--interval", type=float, default=60.0)
ap.add_argument("--threshold", type=float, default=0.15)
ap.add_argument("--consecutive", type=int, default=20)
ap.add_argument("--job_id", default=os.environ.get("SLURM_JOB_ID", ""))
ap.add_argument("--dry_run", action="store_true",
                help="log what would happen; do not call scancel")
ap.add_argument("--once", action="store_true",
                help="single pass over whatever is already on disk, then exit")
ap.add_argument("--wait_manifest", type=float, default=600.0)
A = ap.parse_args()

RUN = A.run
LOGP = os.path.join(RUN, "watchdog.log")
STATUSP = os.path.join(RUN, "watchdog_status.json")


def log(level, msg):
    line = f"[{level:<7} watchdog_v3 {time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(LOGP, "a") as fh:
            fh.write(line + "\n")
    except Exception:                                        # noqa: BLE001
        pass


# ---- wait for the manifest the runner writes before the binary starts
mp = os.path.join(RUN, "run_manifest.json")
t0 = time.time()
while not os.path.exists(mp):
    if time.time() - t0 > A.wait_manifest:
        log("FATAL", f"no run_manifest.json in {RUN} after {A.wait_manifest:.0f}s; exiting")
        sys.exit(2)
    time.sleep(2.0)
man = json.load(open(mp))
L = man["params"]["L"]
TOTAL_GEN = man["params"].get("total_generation")
try:
    from datetime import datetime
    WRITTEN_AT = datetime.strptime(man["written_at"], "%Y-%m-%dT%H:%M:%S%z").timestamp()
except Exception:                                            # noqa: BLE001
    WRITTEN_AT = 0.0

log("INFO", f"armed on {RUN}: sd(pred)/sd(expected) < {A.threshold} for {A.consecutive} "
            f"consecutive generations => FATAL + scancel {A.job_id or '(no job id)'}"
            + ("  [DRY RUN]" if A.dry_run else ""))
log("INFO", f"L={L} total_generation={TOTAL_GEN} poll={A.interval}s manifest_written_at={man['written_at']}")

seen = {}          # g -> ratio (or None when undefined)
streak = 0
streak_start = None
n_nonfinite_gens = 0
fired = False


def save_status(extra=None):
    st = {"run": RUN, "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
          "threshold": A.threshold, "consecutive_required": A.consecutive,
          "generations_seen": len(seen), "current_streak": streak,
          "streak_started_at_generation": streak_start,
          "nonfinite_generations": n_nonfinite_gens,
          "fired": fired,
          "sd_ratio_by_generation": {str(k): seen[k] for k in sorted(seen)}}
    if extra:
        st.update(extra)
    try:
        with open(STATUSP, "w") as fh:
            json.dump(st, fh, indent=2, default=float)
    except Exception:                                        # noqa: BLE001
        pass


def fire(g):
    global fired
    fired = True
    log("FATAL", f"COLLAPSE: sd(predicted)/sd(expected) < {A.threshold} for {A.consecutive} "
                 f"consecutive generations ({streak_start}..{g}). The one-sided "
                 f"--max_pred_sd_ratio guard cannot reject a collapsed genome, so the search "
                 f"is producing a constant and every further generation is wasted core-hours.")
    ratios = [seen[k] for k in sorted(seen) if seen[k] is not None][-A.consecutive:]
    log("FATAL", "last ratios: " + ", ".join(f"{r:.4f}" for r in ratios))
    save_status({"fired_at_generation": g})
    if not A.job_id:
        log("ERROR", "no SLURM_JOB_ID: cannot scancel, exiting non-zero instead")
        return 3
    if A.dry_run:
        log("FATAL", f"[DRY RUN] would run: scancel {A.job_id}")
        return 3
    log("FATAL", f"running: scancel {A.job_id}")
    try:
        subprocess.run(["scancel", A.job_id], check=False, timeout=60)
    except Exception as e:                                   # noqa: BLE001
        log("ERROR", f"scancel failed: {e}")
    return 3


def poll():
    """Returns an exit code to stop with, or None to keep going."""
    global streak, streak_start, n_nonfinite_gens
    files = sorted(glob.glob(os.path.join(RUN, "generation_*_global_best.csv")),
                   key=lambda f: int(re.search(r"generation_(\d+)_", f).group(1)))
    new = 0
    for f in files:
        g = int(re.search(r"generation_(\d+)_", f).group(1))
        if g in seen:
            continue
        try:
            if os.stat(f).st_mtime < WRITTEN_AT:
                log("WARNING", f"generation {g}: file predates the manifest -- STALE, left over "
                               f"from an earlier run in this directory. Skipping. score_v3.py "
                               f"will refuse it.")
                seen[g] = None
                continue
            d = pd.read_csv(f)
        except Exception as e:                               # noqa: BLE001
            log("DEBUG", f"generation {g}: not readable yet ({e}); will retry")
            continue
        d.columns = [c.lstrip("#") for c in d.columns]
        if len(d) != L - 1 or "global_best_predicted_N2O" not in d.columns:
            continue                                          # still being written
        p = d["global_best_predicted_N2O"].values.astype(float)
        e_ = d["expected_N2O"].values.astype(float)
        new += 1
        if not np.isfinite(p).all():
            n_nonfinite_gens += 1
            log("ERROR", f"generation {g}: {int((~np.isfinite(p)).sum())} of {len(p)} predictions "
                         f"are NON-FINITE. score_v3.py will REFUSE this generation and fail the "
                         f"run; do not expect a score from this job.")
            seen[g] = None
            continue
        sd_e = float(np.nanstd(e_))
        if not np.isfinite(sd_e) or sd_e <= 0.0:
            log("WARNING", f"generation {g}: target sd is {sd_e} -- ratio undefined on this "
                           f"window, skipped (streak unchanged at {streak})")
            seen[g] = None
            continue
        r = float(np.std(p) / sd_e)
        seen[g] = r
        if r < A.threshold:
            if streak == 0:
                globals()["streak_start"] = g
            streak += 1
            log("WARNING", f"generation {g}: sd ratio {r:.4f} < {A.threshold} "
                           f"({streak}/{A.consecutive} consecutive)")
            if streak >= A.consecutive:
                return fire(g)
        else:
            if streak:
                log("INFO", f"generation {g}: sd ratio {r:.4f} -- collapse streak reset "
                            f"(was {streak})")
            streak = 0
            globals()["streak_start"] = None
    if new:
        last = max(seen)
        lr = seen[last]
        log("INFO", f"heartbeat: {len(seen)} generations seen, latest {last} "
                    f"sd ratio {'undefined' if lr is None else f'{lr:.4f}'}, "
                    f"collapse streak {streak}/{A.consecutive}, "
                    f"non-finite generations {n_nonfinite_gens}")
        save_status()
    return None


if A.once:
    rc = poll()
    save_status()
    sys.exit(rc if rc is not None else 0)

while True:
    rc = poll()
    if rc is not None:
        sys.exit(rc)
    if TOTAL_GEN is not None and len(seen) >= TOTAL_GEN:
        log("INFO", f"all {TOTAL_GEN} generations seen; watchdog exiting clean")
        save_status()
        sys.exit(0)
    time.sleep(A.interval)
