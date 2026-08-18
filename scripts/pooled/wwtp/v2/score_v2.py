#!/usr/bin/env python3
"""
score_v2.py <run_dir> [baseline_dir]

Scores a ONE-NAS v2 run.  Takes NOTHING from the environment: every parameter
comes from <run_dir>/run_manifest.json, which run_v2.sh wrote at launch.

WHY.  v1's scorer took L/T/V from env vars and then searched a +/-few-row window
for the offset that made each generation's `expected` column match the sidecar.
For several v1 runs no offset matched the parameters their sbatch advertised, so
they could not be scored at all.  Here alignment is a closed form derived from
the ONE-NAS source, and the `expected` match is used only as an ASSERTION: a
generation whose expected column does not reproduce the sidecar to 1e-4 is
REFUSED, never guessed at.

THE CLOSED FORM (each step traced to source):
  test episode of generation g           e = g + T + V
      online_series.cxx:341-372 -- current_index = g + num_training_sets,
      validation = [cw, cw+V), test = cw + V.
  episode e -> (segment, window)         windows.csv row window_id == e
      time_series.cxx:757-770 assigns training_indexes in argv order and
      process_arguments.cxx:376+ slices each series independently, so episodes
      are numbered file-major.  prep_v2 emits segments chronologically and
      windows.csv is built in the identical order.
  file data row i -> episode timestep    j = i + 1
      onenas_island_speciation_strategy.cxx:1119 -- the writer loops j from 1.
  timestep j -> source row               seg_row = tgt_start_seg_row + i + 1
      time_series.cxx:479-485 -- with --time_offset H, outputs[k] = row k+H.
  forecast-issue row                     seg_row - H
      inputs[k] = row k, so at timestep k the model has seen up to row k.

METRIC AND CONVENTION -- fixed, and identical for every arm:
  nMSE = sum (y - p)^2 / sum (y - mean(y))^2, in RAW mg/L, over exactly the
  target rows the run covered.  Every emitted target row is inside a contiguous
  valid segment, so both the target and its persistence anchor are valid by
  construction: there is no mask choice left to make.  The primary number uses
  ALL rows of every test window (no warm-up discard).  Warm-up-discarded
  variants are printed as a declared sensitivity, with the gate recomputed on
  the same rows each time, so no number is ever compared across row sets.
"""
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np
import pandas as pd

RUN = sys.argv[1]
BASEDIR = sys.argv[2] if len(sys.argv) > 2 else None
TOL = 1e-4
WARMUPS = (0, 12, 24, 48)      # declared sensitivity; WARMUPS[0] == 0 is primary

man = json.load(open(os.path.join(RUN, "run_manifest.json")))
P, D, A = man["params"], man["data"], man["alignment"]
L, T, V, H = P["L"], P["T"], P["V"], P["OFFSET"]
CENTER, SCALE = D["n2o_center"], D["n2o_scale"]
DATA = D["dir"]

print(f"run       {RUN}")
print(f"manifest  written {man['written_at']}  slurm {man.get('slurm_job_id')}")
print(f"params    L={L} T={T} V={V} OFFSET={H} step={P['window_step']} "
      f"ISL={P['ISL']} POP={P['POP']} ELITE={P['ELITE']} BP={P['BP']} seed={P['seed']}")
print(f"data      {DATA}  ({D['n_segments']} segments, {D['n_windows']} windows, "
      f"mask={D['mask']})")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# The manifest pins the data it was run against.  If the prep was re-run since,
# every alignment below is meaningless -- refuse rather than report a number.
for key, fn in (("index_csv_sha256", "index.csv"), ("windows_csv_sha256", "windows.csv"),
                ("prep_meta_sha256", "prep_meta.json")):
    got = sha(os.path.join(DATA, fn))
    if got != D[key]:
        sys.exit(f"FATAL: {fn} changed since launch ({got[:12]} != {D[key][:12]}). "
                 f"The prep was re-run after this job started; this run cannot be scored.")
print("provenance OK: index.csv / windows.csv / prep_meta.json match the manifest")

idx = pd.read_csv(os.path.join(DATA, "index.csv"), parse_dates=["time"])
win = pd.read_csv(os.path.join(DATA, "windows.csv"))
n = len(idx)

# flat key: position of (seg_id, seg_row) inside index.csv
seg_off = idx.groupby("seg_id", sort=True).size().cumsum().shift(1).fillna(0).astype(int)
seg_off = seg_off.to_dict()
y_norm = idx["n2o_norm"].values
y_raw = idx["n2o_raw"].values
t_all = idx["time"].values

files = sorted(glob.glob(os.path.join(RUN, "generation_*_global_best.csv")),
               key=lambda f: int(re.search(r"generation_(\d+)_", f).group(1)))
print(f"\nfound {len(files)} generation prediction files")
if not files:
    sys.exit("no prediction files")

pred = np.full(n, np.nan)       # ONE-NAS prediction, normalised units
oracle5 = np.full(n, np.nan)    # the `naive_N2O` column -- NOT a baseline, see below
covered_gen = np.full(n, -1)
in_window_pos = np.full(n, -1)  # 0-based position within the test window
refused, aligned, worst = [], 0, 0.0

for f in files:
    g = int(re.search(r"generation_(\d+)_", f).group(1))
    e = g + T + V
    if e >= len(win):
        refused.append((g, "episode past the end of windows.csv"))
        continue
    w = win.iloc[e]
    assert int(w["window_id"]) == e, "windows.csv is not indexed by window_id"
    d = pd.read_csv(f)
    d.columns = [c.lstrip("#") for c in d.columns]
    if len(d) == 0:
        refused.append((g, "empty file"))
        continue
    if len(d) != L - 1:
        refused.append((g, f"file has {len(d)} rows, expected L-1={L-1}"))
        continue
    base = seg_off[int(w["seg_id"])] + int(w["tgt_start_seg_row"])
    sl = np.arange(base + 1, base + 1 + len(d))
    err = float(np.max(np.abs(y_norm[sl] - d["expected_N2O"].values)))
    worst = max(worst, err)
    if err > TOL:
        refused.append((g, f"expected column mismatch {err:.3g} > {TOL}"))
        continue
    pred[sl] = d["global_best_predicted_N2O"].values
    oracle5[sl] = d["naive_N2O"].values
    covered_gen[sl] = g
    in_window_pos[sl] = np.arange(len(d))
    aligned += 1

print(f"aligned {aligned} generations, refused {len(refused)}  "
      f"(worst expected-column error {worst:.3g}, tolerance {TOL})")
for g, why in refused[:5]:
    print(f"  REFUSED gen {g}: {why}")
if aligned == 0:
    sys.exit("no generation aligned -- the closed form is wrong, refusing to report")

have = np.isfinite(pred)
pred_raw = pred * SCALE + CENTER
# persistence: the last observed value at forecast-issue time, H rows back.
# Both endpoints are inside one contiguous valid segment by construction.
pers_raw = np.full(n, np.nan)
src = np.flatnonzero(have)
pers_raw[src] = y_raw[src - H]
assert (idx["seg_id"].values[src] == idx["seg_id"].values[src - H]).all(), \
    "a persistence anchor fell outside its segment -- segmentation is broken"


def nmse(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    den = np.sum((y - y.mean()) ** 2)
    return float(np.sum((y - p) ** 2) / den) if den > 0 else np.nan


# optional external baselines: <baseline_dir>/preds_<name>.csv with columns
# seg_id, seg_row, pred_raw.  Joined onto the SAME rows -- no arm ever gets its
# own row set.
extern = {}
if BASEDIR and os.path.isdir(BASEDIR):
    key = idx["seg_id"].astype(str) + ":" + idx["seg_row"].astype(str)
    pos = pd.Series(np.arange(n), index=key.values)
    for bf in sorted(glob.glob(os.path.join(BASEDIR, "preds_*.csv"))):
        name = os.path.basename(bf)[6:-4]
        b = pd.read_csv(bf)
        k = b["seg_id"].astype(str) + ":" + b["seg_row"].astype(str)
        arr = np.full(n, np.nan)
        arr[pos.reindex(k.values).values] = b["pred_raw"].values
        extern[name] = arr
        print(f"loaded baseline '{name}': {np.isfinite(arr).sum()} rows")

print(f"\ncovered rows {have.sum()} of {n} emitted "
      f"({100*have.mean():.1f}%)   span {pd.Timestamp(t_all[have].min())} .. "
      f"{pd.Timestamp(t_all[have].max())}")

out = {"run": RUN, "aligned": aligned, "refused": len(refused),
       "params": P, "covered_rows": int(have.sum()), "warmup_sensitivity": {}}

for wu in WARMUPS:
    m = have & (in_window_pos >= wu)
    # every arm must be finite on the same rows
    for a in extern.values():
        m = m & np.isfinite(a)
    if m.sum() < 100:
        continue
    row = {"n": int(m.sum()),
           "onenas": nmse(y_raw[m], pred_raw[m]),
           "persistence_GATE": nmse(y_raw[m], pers_raw[m]),
           "climatology_trailing_mean": nmse(y_raw[m], np.full(m.sum(), y_raw[m].mean()))}
    for name, a in extern.items():
        row[name] = nmse(y_raw[m], a[m])
    out["warmup_sensitivity"][wu] = row
    tag = "PRIMARY" if wu == 0 else f"warmup-discard {wu} rows ({wu*5} min)"
    print(f"\n--- {tag}: n={row['n']} ---")
    for k, val in row.items():
        if k == "n":
            continue
        print(f"    {k:<28} nMSE {val:8.4f}")
    v = "BEATS" if row["onenas"] < row["persistence_GATE"] else "LOSES TO"
    print(f"    => ONE-NAS {v} the persistence gate on these rows")

# `naive_N2O` is NOT a baseline: onenas writes test_output[j-1], the target one
# timestep before the scored target, i.e. an oracle that already knows N2O at
# t+H-5min.  Reported only so it can never be mistaken for the gate.
m0 = have
print(f"\n[not a baseline] ONE-NAS 'naive_N2O' column = target at t+H-5min, an "
      f"oracle: nMSE {nmse(y_raw[m0], oracle5[m0]*SCALE+CENTER):.4f}")

# ---------------------------------------------------------------- collapse check
p = pred_raw[have]
yv = y_raw[have]
floor_norm = (y_raw.min() - CENTER) / SCALE
below = float((pred[have] < floor_norm).mean())
print(f"\n--- prediction sanity (v1 collapsed to mean -0.963, sd 0.001 normalised) ---")
print(f"  prediction  raw  mean {p.mean():+.4f}  sd {p.std():.4f}  "
      f"min {p.min():+.4f}  max {p.max():+.4f}")
print(f"  target      raw  mean {yv.mean():+.4f}  sd {yv.std():.4f}  "
      f"min {yv.min():+.4f}  max {yv.max():+.4f}")
print(f"  sd ratio pred/target {p.std()/yv.std() if yv.std() else float('nan'):.4f}"
      f"   (collapse if << 1, explosion if >> 1)")
print(f"  predictions below the physical floor ({y_raw.min():.4f} mg/L): {100*below:.2f}%")
print(f"  distinct predicted values: {len(np.unique(np.round(p,6)))} of {len(p)}")
out["sanity"] = {"pred_mean": float(p.mean()), "pred_sd": float(p.std()),
                 "target_mean": float(yv.mean()), "target_sd": float(yv.std()),
                 "sd_ratio": float(p.std()/yv.std()) if yv.std() else None,
                 "frac_below_physical_floor": below,
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

with open(os.path.join(RUN, "score_v2.json"), "w") as fh:
    json.dump(out, fh, indent=2, default=float)
print(f"\nwrote {os.path.join(RUN, 'score_v2.json')}")
