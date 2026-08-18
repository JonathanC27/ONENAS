#!/usr/bin/env python3
"""
score_v3.py <run_dir> [expected_baseline_dir]

Scores a ONE-NAS v3 run and returns a VERDICT.  Exit status is part of the
output: 0 = PASS or STRONG PASS, 1 = FAIL (scored, criteria not met),
2 = REFUSED (the run cannot be scored at all).  `score_v3.json` is written in
every case, so a refusal leaves the same machine-readable trail as a pass.

Everything comes from <run_dir>/run_manifest.json, which run_v3.sh wrote at
launch -- including the baseline directory.  The optional second argument is
NOT a source of truth: if given it must MATCH the manifest, and the scorer
refuses if it does not.  This closes the hole that let the scored row set be
changed from outside the recorded run.

WHAT CHANGED FROM score_v2.py, AND WHY (each item is a demonstrated defect):

  (1) STALE PREDICTION FILES WERE SILENTLY SCORED.  prod72.sbatch did
      `mkdir -p "$OUT"` and never cleared it, and the expected-column check is
      data-derived, so a prediction file left behind by ANY earlier run at the
      same (DATA, L, T, V, H) validated perfectly.  A reviewer replaced
      generations 0-4 with a collapsed constant and got "aligned 10 generations,
      refused 0, worst expected-column error 0" and a false
      "ONE-NAS BEATS the persistence gate".  A 12 h timeout plus requeue
      produces exactly this by accident.
      FIX: run_v3.sh does `rm -rf "$OUT"` first, AND every generation file whose
      mtime predates the manifest's `written_at` is REFUSED here, and any such
      refusal fails the run.  The runner fix alone is not enough -- the scorer
      must be able to detect the condition on its own.

  (2) NON-FINITE PREDICTIONS SILENTLY SHRANK THE ROW SET.  score_v2.py:147
      `have = np.isfinite(pred)` let ONE-NAS's own NaNs decide which rows every
      arm was scored on.  Injecting NaN into 71 predictions moved covered rows
      1430 -> 1288 and the persistence gate 1.9433 -> 1.9820 with no message.
      That violates the protocol's own "no arm ever gets its own row set".
      FIX: the row set is COVERAGE (which generations aligned), never finiteness.
      Non-finite values are counted per generation, the generation is refused,
      and any refusal fails the run with the counts printed.

  (3c) THE PROTOCOL'S PASS/FAIL CRITERIA WERE PROSE.  Now executable: sd ratio
      outside [0.3, 3.0], >1% of predictions below the physical floor, or
      nMSE >= the persistence gate stamps "verdict":"FAIL" and exits non-zero.

  (4) THE MANIFEST DID NOT CLOSE THE SCORE.  The baseline directory came from
      argv, preds_*.csv were unhashed, and the scorer intersects the row set
      with EVERY loaded baseline -- so adding, removing or swapping one baseline
      file silently moved the primary row set and therefore the gate.  The
      segment CSVs (the actual model inputs) and total_generation were also
      unrecorded.  FIX: all of it is in the manifest and all of it is verified
      here with the same refusal logic already used for index/windows/prep_meta.

UNCHANGED AND STILL LOAD-BEARING -- the alignment closed form, each step traced
to source:
  test episode of generation g           e = g + T + V
      online_series.cxx:341-372
  episode e -> (segment, window)         windows.csv row window_id == e
      time_series.cxx:757-770, process_arguments.cxx:376+
  file data row i -> episode timestep    j = i + 1
      onenas_island_speciation_strategy.cxx:1119
  timestep j -> source row               seg_row = tgt_start_seg_row + i + 1
      time_series.cxx:479-485
  forecast-issue row                     seg_row - H

METRIC AND CONVENTION -- fixed, identical for every arm:
  nMSE = sum (y - p)^2 / sum (y - mean(y))^2, in RAW mg/L, over exactly the
  target rows the run covered.  The primary number uses ALL rows of every test
  window.  Warm-up-discarded variants are a declared sensitivity with the gate
  recomputed on the same rows, so no number is compared across row sets.
"""
import glob
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

TOL = 1e-4
WARMUPS = (0, 12, 24, 48)      # declared sensitivity; WARMUPS[0] == 0 is primary
SD_RATIO_BAND = (0.3, 3.0)     # PROTOCOL.md section 8
MAX_FRAC_BELOW_FLOOR = 0.01    # PROTOCOL.md section 8

if len(sys.argv) < 2:
    sys.exit("usage: score_v3.py <run_dir> [expected_baseline_dir]")
RUN = sys.argv[1]
ASSERT_BASEDIR = sys.argv[2] if len(sys.argv) > 2 else None

out = {"scorer": "score_v3.py", "run": RUN, "scored_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
       "verdict": "FAIL", "verdict_reasons": [], "refusals": []}
FAIL = out["verdict_reasons"]


def emit(code):
    """Always leave a machine-readable trail, then exit with the given status."""
    try:
        with open(os.path.join(RUN, "score_v3.json"), "w") as fh:
            json.dump(out, fh, indent=2, default=float)
        print(f"\nwrote {os.path.join(RUN, 'score_v3.json')}")
    except Exception as e:                                   # noqa: BLE001
        print(f"WARNING: could not write score_v3.json: {e}")
    print(f"VERDICT {out['verdict']}"
          + ("" if not FAIL else "  because: " + "; ".join(FAIL)))
    sys.exit(code)


def refuse(msg):
    """Cannot score at all -- provenance or inputs are broken."""
    print(f"\nFATAL: {msg}")
    out["verdict"] = "FAIL"
    FAIL.append(msg)
    emit(2)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- manifest
mp = os.path.join(RUN, "run_manifest.json")
if not os.path.exists(mp):
    refuse(f"no run_manifest.json in {RUN}")
man = json.load(open(mp))
if man.get("schema") != "onenas_wwtp_run_manifest/3":
    refuse(f"manifest schema is {man.get('schema')!r}, score_v3.py requires "
           f"'onenas_wwtp_run_manifest/3' (a v2 manifest does not record the baseline "
           f"directory, the segment hashes or total_generation, so it cannot close the score)")

P, D, A = man["params"], man["data"], man["alignment"]
L, T, V, H = P["L"], P["T"], P["V"], P["OFFSET"]
CENTER, SCALE = D["n2o_center"], D["n2o_scale"]
DATA = D["dir"]
BASEDIR = D["baseline_dir"]
TOTAL_GEN = P.get("total_generation")

# written_at is the fence for defect (1): nothing in the run directory may
# predate it.
try:
    WRITTEN_AT = datetime.strptime(man["written_at"], "%Y-%m-%dT%H:%M:%S%z").timestamp()
except Exception as e:                                       # noqa: BLE001
    refuse(f"cannot parse manifest written_at {man.get('written_at')!r}: {e}")

print(f"run       {RUN}")
print(f"manifest  written {man['written_at']} (epoch {WRITTEN_AT:.0f})  slurm {man.get('slurm_job_id')}")
print(f"params    L={L} T={T} V={V} OFFSET={H} step={P['window_step']} "
      f"ISL={P['ISL']} POP={P['POP']} ELITE={P['ELITE']} BP={P['BP']}")
print(f"seeds     online_series_seed={P['seed']}  examm_seed={P.get('examm_seed')}  "
      f"(examm_seed null => mutation/crossover/island RNG came from the wall clock "
      f"and this run is NOT reproducible)")
print(f"total_generation  {TOTAL_GEN}")
print(f"data      {DATA}  ({D['n_segments']} segments, {D['n_windows']} windows, mask={D['mask']})")
print(f"baselines {BASEDIR}  (from the manifest, not from argv)")

if ASSERT_BASEDIR is not None and os.path.realpath(ASSERT_BASEDIR) != os.path.realpath(BASEDIR):
    refuse(f"baseline dir given on the command line ({ASSERT_BASEDIR}) is not the one "
           f"recorded in the manifest ({BASEDIR}). The manifest is authoritative; refusing "
           f"to score against a directory the run did not declare.")

# ---------------------------------------------------------------- provenance
# If any input changed since launch, every number below is meaningless.
for key, fn in (("index_csv_sha256", "index.csv"), ("windows_csv_sha256", "windows.csv"),
                ("prep_meta_sha256", "prep_meta.json")):
    p = os.path.join(DATA, fn)
    if not os.path.exists(p):
        refuse(f"{p} is missing")
    got = sha(p)
    if got != D[key]:
        refuse(f"{fn} changed since launch ({got[:12]} != {D[key][:12]}). The prep was re-run "
               f"after this job started; this run cannot be scored.")
print("provenance OK: index.csv / windows.csv / prep_meta.json match the manifest")

# (4) the 64 segment CSVs are the ACTUAL model inputs and were previously unhashed.
seg_bad = []
for s in D["segment_map"]:
    p = s["file"] if os.path.isabs(s["file"]) else os.path.join(DATA, s["file"])
    if not os.path.exists(p):
        seg_bad.append((s["seg_id"], "missing"))
        continue
    if "sha256" not in s:
        seg_bad.append((s["seg_id"], "no hash in manifest"))
        continue
    if sha(p) != s["sha256"]:
        seg_bad.append((s["seg_id"], "content changed since launch"))
if seg_bad:
    refuse(f"{len(seg_bad)} of {len(D['segment_map'])} segment CSVs failed verification "
           f"(first: seg {seg_bad[0][0]} {seg_bad[0][1]}). These are the model's inputs; "
           f"if they moved, the run is not the run that was recorded.")
print(f"provenance OK: all {len(D['segment_map'])} segment CSVs match the manifest")

# (4) the baseline SET is pinned, because the scorer intersects the row set with
# every loaded baseline -- so adding, removing or swapping one file moves the
# primary row set and therefore the gate.
if not os.path.isdir(BASEDIR):
    refuse(f"baseline dir {BASEDIR} recorded in the manifest does not exist")
declared = D["baseline_files"]                     # {name: sha256}
present = {os.path.basename(f)[6:-4]: f
           for f in sorted(glob.glob(os.path.join(BASEDIR, "preds_*.csv")))}
missing = sorted(set(declared) - set(present))
extra = sorted(set(present) - set(declared))
changed = [k for k in sorted(set(declared) & set(present)) if sha(present[k]) != declared[k]]
if missing or extra or changed:
    refuse(f"the baseline set is not the one this run declared -- missing {missing}, "
           f"unexpected {extra}, content-changed {changed}. The scored row set is the "
           f"intersection over every baseline, so this would silently move the gate.")
print(f"provenance OK: all {len(declared)} baseline files match the manifest "
      f"({', '.join(sorted(declared))})")

# ---------------------------------------------------------------- data
idx = pd.read_csv(os.path.join(DATA, "index.csv"), parse_dates=["time"])
win = pd.read_csv(os.path.join(DATA, "windows.csv"))
n = len(idx)

seg_off = idx.groupby("seg_id", sort=True).size().cumsum().shift(1).fillna(0).astype(int).to_dict()
y_norm = idx["n2o_norm"].values
y_raw = idx["n2o_raw"].values
t_all = idx["time"].values

files = sorted(glob.glob(os.path.join(RUN, "generation_*_global_best.csv")),
               key=lambda f: int(re.search(r"generation_(\d+)_", f).group(1)))
print(f"\nfound {len(files)} generation prediction files")
if not files:
    refuse("no prediction files")

pred = np.full(n, np.nan)       # ONE-NAS prediction, normalised units
oracle5 = np.full(n, np.nan)    # the `naive_N2O` column -- NOT a baseline, see below
covered_gen = np.full(n, -1)
in_window_pos = np.full(n, -1)
double_covered = 0

refused, aligned, worst = [], 0, 0.0
n_stale, n_nonfinite, nonfinite_total = 0, 0, 0

for f in files:
    g = int(re.search(r"generation_(\d+)_", f).group(1))

    # -------- defect (1): a file that predates the manifest is not this run's file.
    mtime = os.stat(f).st_mtime
    if mtime < WRITTEN_AT:
        age = WRITTEN_AT - mtime
        refused.append((g, f"STALE: mtime {time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(mtime))} "
                           f"predates the manifest by {age:.0f}s -- left over from an earlier run "
                           f"in this output directory"))
        n_stale += 1
        continue

    e = g + T + V
    if e >= len(win):
        refused.append((g, "episode past the end of windows.csv"))
        continue
    w = win.iloc[e]
    if int(w["window_id"]) != e:
        refuse("windows.csv is not indexed by window_id")
    d = pd.read_csv(f)
    d.columns = [c.lstrip("#") for c in d.columns]
    if len(d) == 0:
        refused.append((g, "empty file"))
        continue
    if len(d) != L - 1:
        refused.append((g, f"file has {len(d)} rows, expected L-1={L-1}"))
        continue
    for col in ("expected_N2O", "global_best_predicted_N2O", "naive_N2O"):
        if col not in d.columns:
            refuse(f"generation {g}: column {col} missing from {f}")

    pv = d["global_best_predicted_N2O"].values.astype(float)
    ev = d["expected_N2O"].values.astype(float)
    nv = d["naive_N2O"].values.astype(float)

    # -------- defect (2): non-finite values must never quietly redefine the row set.
    nf = {"global_best_predicted_N2O": int((~np.isfinite(pv)).sum()),
          "expected_N2O": int((~np.isfinite(ev)).sum()),
          "naive_N2O": int((~np.isfinite(nv)).sum())}
    if sum(nf.values()) > 0:
        refused.append((g, f"NON-FINITE predictions: {nf} of {len(d)} rows -- a generation "
                           f"with NaN/inf is refused outright, never scored on the rows that "
                           f"happen to be finite"))
        n_nonfinite += 1
        nonfinite_total += sum(nf.values())
        continue

    base = seg_off[int(w["seg_id"])] + int(w["tgt_start_seg_row"])
    sl = np.arange(base + 1, base + 1 + len(d))
    err = float(np.max(np.abs(y_norm[sl] - ev)))
    worst = max(worst, err)
    if err > TOL:
        refused.append((g, f"expected column mismatch {err:.3g} > {TOL}"))
        continue

    double_covered += int((covered_gen[sl] >= 0).sum())
    pred[sl] = pv
    oracle5[sl] = nv
    covered_gen[sl] = g
    in_window_pos[sl] = np.arange(len(d))
    aligned += 1

print(f"aligned {aligned} generations, refused {len(refused)}  "
      f"(worst expected-column error {worst:.3g}, tolerance {TOL})")
print(f"  refused as STALE (predate the manifest):      {n_stale}")
print(f"  refused for NON-FINITE predictions:           {n_nonfinite} "
      f"({nonfinite_total} non-finite values)")
print(f"  refused for other reasons:                    {len(refused) - n_stale - n_nonfinite}")
for g, why in refused[:12]:
    print(f"  REFUSED gen {g}: {why}")
if len(refused) > 12:
    print(f"  ... and {len(refused) - 12} more")
out["refusals"] = [{"generation": g, "reason": why} for g, why in refused]
out["aligned"] = aligned
out["refused"] = len(refused)
out["refused_stale"] = n_stale
out["refused_nonfinite"] = n_nonfinite
out["nonfinite_values"] = nonfinite_total
out["worst_expected_column_error"] = worst

# A refusal is not a row-set adjustment: it fails the run.  These come BEFORE the
# "nothing aligned" check so the message names the actual cause.
if n_stale:
    refuse(f"{n_stale} generation file(s) predate the manifest. The output directory was not "
           f"cleared, so this run's predictions are mixed with an earlier run's. Delete the "
           f"output directory and re-run; run_v3.sh does `rm -rf $OUT` to prevent this.")
if n_nonfinite:
    refuse(f"{n_nonfinite} generation(s) contain {nonfinite_total} non-finite prediction values. "
           f"score_v2.py would have dropped exactly those rows from EVERY arm, moving the gate; "
           f"this scorer refuses instead. Investigate the diverged genomes.")
if aligned == 0:
    refuse("no generation aligned -- refusing to report any number")
if double_covered:
    msg = (f"{double_covered} rows were written by more than one generation "
           f"(window_step={P['window_step']}, L={L})")
    if P["window_step"] >= L:
        refuse(msg + " -- with step >= L the test windows must be disjoint, so alignment is wrong")
    print("WARNING: " + msg)
if TOTAL_GEN is not None and aligned != TOTAL_GEN:
    refuse(f"the run declared total_generation={TOTAL_GEN} but only {aligned} generations "
           f"aligned. An incomplete run must not be scored as if it were complete -- its row "
           f"set is a different, shorter span than the pre-registered one.")

# ---------------------------------------------------------------- row set
# (2) COVERAGE defines the row set. Finiteness never does.
have = covered_gen >= 0
if not np.isfinite(pred[have]).all():
    refuse("internal error: a covered row holds a non-finite prediction after the "
           "non-finite refusal -- the refusal logic is broken, refusing to report")
pred_raw = pred * SCALE + CENTER

pers_raw = np.full(n, np.nan)
src = np.flatnonzero(have)
pers_raw[src] = y_raw[src - H]
if not (idx["seg_id"].values[src] == idx["seg_id"].values[src - H]).all():
    refuse("a persistence anchor fell outside its segment -- segmentation is broken")


def nmse(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    den = np.sum((y - y.mean()) ** 2)
    return float(np.sum((y - p) ** 2) / den) if den > 0 else np.nan


# external baselines, joined onto the SAME rows -- no arm ever gets its own row set
extern = {}
key = idx["seg_id"].astype(str) + ":" + idx["seg_row"].astype(str)
pos = pd.Series(np.arange(n), index=key.values)
for name in sorted(declared):
    b = pd.read_csv(present[name])
    k = b["seg_id"].astype(str) + ":" + b["seg_row"].astype(str)
    arr = np.full(n, np.nan)
    arr[pos.reindex(k.values).values] = b["pred_raw"].values
    extern[name] = arr
    print(f"loaded baseline '{name}': {np.isfinite(arr).sum()} rows")

print(f"\ncovered rows {have.sum()} of {n} emitted ({100*have.mean():.1f}%)   "
      f"span {pd.Timestamp(t_all[have].min())} .. {pd.Timestamp(t_all[have].max())}")
out["covered_rows"] = int(have.sum())
out["params"] = P
out["warmup_sensitivity"] = {}

primary = None
for wu in WARMUPS:
    m = have & (in_window_pos >= wu)
    n_before = int(m.sum())
    for a in extern.values():
        m = m & np.isfinite(a)
    dropped = n_before - int(m.sum())
    if m.sum() < 100:
        continue
    row = {"n": int(m.sum()),
           "rows_dropped_by_baseline_intersection": dropped,
           "onenas": nmse(y_raw[m], pred_raw[m]),
           "persistence_GATE": nmse(y_raw[m], pers_raw[m]),
           "climatology_trailing_mean": nmse(y_raw[m], np.full(int(m.sum()), y_raw[m].mean()))}
    for name, a in extern.items():
        row[name] = nmse(y_raw[m], a[m])
    out["warmup_sensitivity"][wu] = row
    if wu == 0:
        primary = row
    tag = "PRIMARY" if wu == 0 else f"warmup-discard {wu} rows ({wu*5} min)"
    print(f"\n--- {tag}: n={row['n']} (baseline intersection dropped {dropped}) ---")
    for k2, val in row.items():
        if k2 in ("n", "rows_dropped_by_baseline_intersection"):
            continue
        print(f"    {k2:<28} nMSE {val:8.4f}")
    v = "BEATS" if row["onenas"] < row["persistence_GATE"] else "LOSES TO"
    print(f"    => ONE-NAS {v} the persistence gate on these rows")

if primary is None:
    refuse("fewer than 100 primary rows survived -- nothing to score")

# The internally computed gate and the external preds_persistence.csv must be the
# same number.  They are computed from different code paths; a divergence means
# one of them is wrong and the gate is ambiguous.
if "persistence" in primary:
    a, b = primary["persistence_GATE"], primary["persistence"]
    rel = abs(a - b) / max(abs(a), 1e-12)
    out["gate_crosscheck"] = {"internal": a, "external_preds_persistence": b, "rel_diff": rel}
    print(f"\ngate cross-check: internal {a:.6f} vs preds_persistence.csv {b:.6f}  "
          f"(rel diff {rel:.2e})" + ("  <-- DISAGREE" if rel > 1e-6 else "  OK"))

# `naive_N2O` is NOT a baseline: onenas writes test_output[j-1], the target one
# timestep before the scored target, i.e. an oracle that already knows N2O at
# t+H-5min.  Reported only so it can never be mistaken for the gate.
print(f"\n[not a baseline] ONE-NAS 'naive_N2O' column = target at t+H-5min, an oracle: "
      f"nMSE {nmse(y_raw[have], oracle5[have]*SCALE+CENTER):.4f}")

# ---------------------------------------------------------------- collapse check
p = pred_raw[have]
yv = y_raw[have]
floor_norm = (y_raw.min() - CENTER) / SCALE
below = float((pred[have] < floor_norm).mean())
sd_ratio = float(p.std() / yv.std()) if yv.std() else None
print(f"\n--- prediction sanity (v1 collapsed to mean -0.963, sd 0.001 normalised) ---")
print(f"  prediction  raw  mean {p.mean():+.4f}  sd {p.std():.4f}  min {p.min():+.4f}  max {p.max():+.4f}")
print(f"  target      raw  mean {yv.mean():+.4f}  sd {yv.std():.4f}  min {yv.min():+.4f}  max {yv.max():+.4f}")
print(f"  sd ratio pred/target {sd_ratio if sd_ratio is None else round(sd_ratio, 4)}"
      f"   (band {SD_RATIO_BAND}; collapse if << 1, explosion if >> 1)")
print(f"  predictions below the physical floor ({y_raw.min():.4f} mg/L): {100*below:.2f}%"
      f"   (limit {100*MAX_FRAC_BELOW_FLOOR:.0f}%)")
print(f"  distinct predicted values: {len(np.unique(np.round(p,6)))} of {len(p)}")
out["sanity"] = {"pred_mean": float(p.mean()), "pred_sd": float(p.std()),
                 "target_mean": float(yv.mean()), "target_sd": float(yv.std()),
                 "sd_ratio": sd_ratio, "frac_below_physical_floor": below,
                 "n_distinct_predictions": int(len(np.unique(np.round(p, 6))))}

# ---------------------------------------------------------------- per month
print(f"\n--- per-month nMSE (primary rows) ---")
mon = pd.PeriodIndex(pd.to_datetime(t_all), freq="M")
print(f"{'month':<9}{'n':>7}{'ONE-NAS':>11}{'persist':>11}")
pm = {}
for mm in sorted(set(mon[have])):
    k = (mon == mm) & have
    if k.sum() < 200:
        continue
    a, b = nmse(y_raw[k], pred_raw[k]), nmse(y_raw[k], pers_raw[k])
    pm[str(mm)] = {"n": int(k.sum()), "onenas": a, "persistence": b}
    print(f"{str(mm):<9}{k.sum():>7}{a:>11.4f}{b:>11.4f}")
out["permonth"] = pm

# ---------------------------------------------------------------- (3c) VERDICT
# PROTOCOL.md section 8, made executable. Judged on H=72, primary rows.
if sd_ratio is None or not (SD_RATIO_BAND[0] <= sd_ratio <= SD_RATIO_BAND[1]):
    FAIL.append(f"collapse diagnostic: sd ratio {sd_ratio} outside {SD_RATIO_BAND}")
if below > MAX_FRAC_BELOW_FLOOR:
    FAIL.append(f"collapse diagnostic: {100*below:.2f}% of predictions below the physical "
                f"floor (limit {100*MAX_FRAC_BELOW_FLOOR:.0f}%)")
if not (primary["onenas"] < primary["persistence_GATE"]):
    FAIL.append(f"nMSE {primary['onenas']:.4f} >= persistence gate "
                f"{primary['persistence_GATE']:.4f} on the primary rows")

if FAIL:
    out["verdict"] = "FAIL"
    emit(1)

out["verdict"] = "PASS"
if "ridge_d1" in primary and primary["onenas"] < primary["ridge_d1"]:
    out["verdict"] = "STRONG_PASS"
    print("\nSTRONG PASS: ONE-NAS also beats daily-retrained ridge on the same rows.")
print("\nNOTE: this is ONE SEED. PROTOCOL.md section 8 judges the campaign on the MEDIAN "
      "of 5 seeds; a per-seed PASS is not the campaign verdict.")
emit(0)
