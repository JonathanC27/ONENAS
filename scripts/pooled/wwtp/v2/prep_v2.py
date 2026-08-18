#!/usr/bin/env python3
"""
prep_v2.py -- corrected ONE-NAS data preparation for Avedore dissolved N2O.

WHAT WAS WRONG IN v1 (the thing this file exists to fix)
--------------------------------------------------------
v1 emitted ONE csv of 178,536 rows covering 2022-10-01..2024-06-11 with every
row present, including the 45,667 rows where `n2o_valid == False` (dead sensor,
flatlined, negative-rail).  ONE-NAS has no notion of missing data: consecutive
CSV rows ARE consecutive timesteps.  So

  * every training window that straddled a dead stretch taught the network to
    reproduce a flat, negative, physically impossible signal;
  * the normalisation median/scale were fitted over ALL rows including the dead
    ones, which is why the N2O median came out NEGATIVE (-0.0114) even though
    the median of the *valid* signal is +0.0347.  The target level was biased
    low by construction, before a single genome was evaluated;
  * the ridge baselines DID mask invalid rows on both endpoints, so the two
    sides of the comparison were never scored on the same data.

WHAT THIS FILE DOES INSTEAD
---------------------------
1. Keeps the registered `n2o_valid` mask EXACTLY as-is.  The persistence gates
   (0.4412 / 0.6105 / 1.0199 at H=6/24/72) are registered against this mask;
   redefining it would silently invalidate them.  Known residual artefacts of
   the mask are measured and reported, not patched (see RAIL below).
2. Splits the series into maximal CONTIGUOUS VALID SEGMENTS and writes ONE CSV
   PER SEGMENT.  Verified against the ONE-NAS source (see MULTI-FILE below),
   `slice_input_data` (common/process_arguments.cxx) iterates each input series
   independently, so no training/validation/test window can ever span a
   segment boundary.  Invalid rows are simply not emitted.
3. Truncates each segment to EXACTLY nw*L + H rows, so ONE-NAS's slicer emits
   exactly nw windows with zero remainder.  That makes episode -> source-row a
   closed form with no fudge factor, which is what makes scoring alignment
   deterministic instead of a search.
4. Fits normalisation statistics ONLY on rows inside the burn-in prefix (the
   first T windows in global episode order), and only on emitted (=valid) rows.
   Nothing at or after the first validation window contributes to any statistic.
5. Emits a sidecar index that maps every emitted row back to its source
   timestamp, plus a per-window table giving the exact episode -> row mapping,
   so score_v2.py never has to guess.

MULTI-FILE SEMANTICS (verified by reading AND by experiment, see verify_multifile.sh)
-------------------------------------------------------------------------------------
  time_series.cxx:757-770      --training_filenames -> one TimeSeriesSet per file,
                               training_indexes = [0,1,2,...] in argv order.
  process_arguments.cxx:376+   slice_input_data() loops `for n in series` and, per
                               series, `current_row = 0; while current_row + L <= num_row`.
                               Windows are therefore per-file; the loop restarts at
                               row 0 of every file.  Boundary spanning is impossible.
  Episode IDs are assigned file-major: all windows of file 0, then file 1, ...
  Because we emit segments in chronological order, episode ID order == valid-time
  order, which is exactly what OnlineSeries' clock assumes.
  NOT pooled_panel: --pooled_panel requires all files to have EQUAL row counts
  (online_series.cxx:50-57) and treats window w of every file as CONTEMPORANEOUS.
  Our segments are sequential in time, not parallel series, so pooled_panel would
  make future segments' windows available at generation 0.  It is wrong here.

WINDOW GEOMETRY / --window_step
-------------------------------
  In NON-pooled mode the training pool is {episode i : i < current_index} and the
  validation window is current_index (online_series.cxx:209-213, 341-355).  With
  --window_step < L consecutive windows OVERLAP, so episode current_index-1 shares
  L - window_step rows with the validation window -> direct leakage.  ONE-NAS only
  warns about this (process_arguments.cxx:333).  v1 ran with --window_step 1 and
  L=144, i.e. 143/144 of the validation window was also in the training pool, and
  each generation advanced the clock by 5 MINUTES (so ~700 generations covered
  ~2.5 days out of 620).  v2 therefore pins window_step == L: non-overlapping,
  leak-free, and the adaptation cadence is L*5min of VALID series time.

TIME OFFSET SEMANTICS
---------------------
  time_series.cxx:445-505.  With --time_offset H:
      inputs[k]  = row k        of the file (for non-shift fields)
      outputs[k] = row k + H    of the file
  so the H-step-ahead task is expressed by the flag ALONE.  The target column in
  the emitted CSV must NOT be pre-shifted or the horizon silently doubles.  This
  file emits UNSHIFTED columns.  (v1 also emitted unshifted columns -- the shift
  there lived only in the scoring sidecar -- so this is a confirmation, not a fix.)

Config via environment; defaults are the pre-registered primary arm.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- configuration
SRC = os.environ.get("SRC", "/anvil/scratch/x-jchang5/wwtp/aved_5min.csv")
ROOT = os.environ.get("ROOT", "/anvil/scratch/x-jchang5/wwtp/v2")
START = os.environ.get("START", "2022-10-01")   # matches the v1 campaign span
H = int(os.environ.get("H", "72"))              # forecast horizon in 5-min bins
L = int(os.environ.get("L", "144"))             # --time_series_length (episode length)
T = int(os.environ.get("T", "80"))              # --num_training_sets (burn-in AND batch size)
V = int(os.environ.get("V", "5"))               # --num_validation_sets

# Normalisation shaping.  ONE-NAS output nodes are SIMPLE_NODE and RNN_Node applies
# tanh to every non-GP node, so predictions are hard-bounded to (-1,1).  Any target
# mass outside that band is unreachable by construction and drives MSE backprop
# toward a saturated, zero-gradient constant -- the v1 collapse signature.  We map
# the burn-in robust range of the target to +/-TARGET_BAND, leaving 1-TARGET_BAND
# of headroom for excursions the burn-in never saw.
TARGET_BAND = float(os.environ.get("TARGET_BAND", "0.6"))
N2O_CLIP = float(os.environ.get("N2O_CLIP", "0.95"))
# Covariates feed tanh nodes too.  v1 clipped at +/-8, which is deep in saturation.
# +/-4 is still ~4x the burn-in robust half-range and keeps inputs informative.
COV_CLIP = float(os.environ.get("COV_CLIP", "4.0"))

SIGNALS = ["N2O", "NH4", "NO3", "PO4", "O2_T1", "O2_T2", "O2_SP",
           "AIR_T1", "AIR_T2", "AIR_BLOWER", "SS_T1", "TEMP", "INLET_Q", "SWM"]
TARGET = "N2O"

# MIN_SEG: a segment must be able to produce at least one window.  ONE-NAS drops the
# last H rows of a file to build the shifted target, then slices windows of L rows,
# so a file of R rows yields floor((R-H)/L) windows and needs R >= L + H to yield any.
# Segments shorter than that are dropped -- they cannot contribute a single training
# example, and padding them would reintroduce exactly the fabricated data this file
# exists to remove.  We do NOT impose a larger minimum: a 1-window segment is a real,
# fully-observed L-row stretch of the process and is as legitimate a training example
# as any window carved out of a long segment.  (The distribution is reported below so
# a stricter threshold can be justified from evidence if it is ever wanted.)
MIN_SEG = L + H

OUT = os.path.join(ROOT, f"h{H}_L{L}")
os.makedirs(OUT, exist_ok=True)

print(f"=== prep_v2  H={H} L={L} T={T} V={V} MIN_SEG={MIN_SEG} ===")
print(f"src  {SRC}")
print(f"out  {OUT}")

# ---------------------------------------------------------------- load + mask
df = pd.read_csv(SRC, parse_dates=["time"])
df = df[df["time"] >= pd.Timestamp(START)].reset_index(drop=True)
n_raw = len(df)

# Assert the grid really is a uniform 5-min grid.  Everything downstream (windows,
# the H-step offset, persistence) assumes row distance == time distance.
dt = df["time"].diff().dropna().dt.total_seconds().unique()
assert len(dt) == 1 and dt[0] == 300.0, f"source grid is not uniform 5-min: {dt[:5]}"

valid = df["n2o_valid"].astype(bool).values
print(f"rows from {START}: {n_raw}   valid: {valid.sum()} ({100*valid.mean():.2f}%)"
      f"   invalid (EXCLUDED FROM TRAINING in v2): {(~valid).sum()}")

# ---------------------------------------------------------------- segmentation
# Maximal runs of valid==True.  Half-open [s, e).
edges = np.flatnonzero(np.diff(np.concatenate(([0], valid.astype(np.int8), [0]))) != 0)
seg_s, seg_e = edges[0::2], edges[1::2]
seg_len = seg_e - seg_s
print(f"\n--- contiguous valid segments (before the {MIN_SEG}-row minimum) ---")
print(f"count {len(seg_len)}   total rows {seg_len.sum()}   "
      f"min {seg_len.min()}  median {int(np.median(seg_len))}  "
      f"mean {seg_len.mean():.0f}  max {seg_len.max()}")
for q in (50, 75, 90, 95, 99):
    print(f"  p{q:<3} length {int(np.percentile(seg_len, q)):>6} rows "
          f"({np.percentile(seg_len, q)*5/60/24:.2f} d)")

keep = seg_len >= MIN_SEG
ks, ke = seg_s[keep], seg_e[keep]
print(f"\n--- after the minimum ({MIN_SEG} rows = {MIN_SEG*5/60:.1f} h) ---")
print(f"usable segments {keep.sum()} of {len(seg_len)}   "
      f"rows {seg_len[keep].sum()} of {seg_len.sum()} valid "
      f"({100*seg_len[keep].sum()/seg_len.sum():.1f}% of valid, "
      f"{100*seg_len[keep].sum()/n_raw:.1f}% of the {n_raw} raw rows)")
print(f"dropped {(~keep).sum()} short segments holding {seg_len[~keep].sum()} valid rows")

# ---------------------------------------------------------------- windows
# Truncate each kept segment to exactly nw*L + H rows.  The tail beyond that cannot
# form a window and would only make the episode->row map non-closed-form.
segments, windows = [], []
for sid, (s, e) in enumerate(zip(ks, ke)):
    nw = (e - s - H) // L
    assert nw >= 1
    emit_rows = nw * L + H
    segments.append({"seg_id": sid, "src_start": int(s), "src_end": int(s + emit_rows),
                     "n_rows": int(emit_rows), "n_windows": int(nw),
                     "t_start": str(df["time"].iloc[s]),
                     "t_end": str(df["time"].iloc[s + emit_rows - 1])})
    for w in range(nw):
        in0 = w * L                      # first INPUT row, segment-local
        tg0 = w * L + H                  # first TARGET row, segment-local
        windows.append({"window_id": len(windows), "seg_id": sid, "seg_window": w,
                        "in_start_seg_row": in0, "tgt_start_seg_row": tg0,
                        "tgt_end_seg_row": tg0 + L - 1,
                        "in_start_time": str(df["time"].iloc[s + in0]),
                        "tgt_start_time": str(df["time"].iloc[s + tg0]),
                        "tgt_end_time": str(df["time"].iloc[s + tg0 + L - 1])})

W = len(windows)
emitted_rows = sum(sg["n_rows"] for sg in segments)
seg_windows = np.array([sg["n_windows"] for sg in segments])
print(f"\n--- windows (episodes) ---")
print(f"total windows W = {W}   emitted rows {emitted_rows} "
      f"({100*emitted_rows/n_raw:.1f}% of raw, {100*emitted_rows/valid.sum():.1f}% of valid)")
print(f"windows per segment: min {seg_windows.min()} median {int(np.median(seg_windows))} "
      f"max {seg_windows.max()}")
print(f"rows lost to truncation: {seg_len[keep].sum() - emitted_rows}")
print(f"TOTAL DATA LOST vs the {n_raw} v1 rows: "
      f"{n_raw - emitted_rows} rows ({100*(n_raw-emitted_rows)/n_raw:.1f}%), of which "
      f"{(~valid).sum()} were invalid and should never have been there")

max_gen = W - T - V - 1     # OnlineSeries::get_max_generation, non-pooled
if max_gen < 1:
    sys.exit(f"FATAL: only {W} windows, T={T} V={V} leaves {max_gen} generations")
print(f"\nOnlineSeries budget: max_generation = W - T - V - 1 = {max_gen}")
print(f"  test episode of generation g is  g + T + V  = g + {T+V}")
print(f"  first scored window {T+V} starts {windows[T+V]['tgt_start_time']}")
print(f"  last  scored window {max_gen-1+T+V} ends {windows[max_gen-1+T+V]['tgt_end_time']}")
print(f"  scored rows if every generation lands: {max_gen*L}")

# ---------------------------------------------------------------- burn-in cut
# The burn-in prefix is windows [0, T).  Normalisation may see ONLY source rows at
# or before the LAST TARGET ROW of window T-1.  Window T is the first window that
# can ever be a validation window (generation 0 has current_index = T), so a
# statistic touching row >= its first input row would be lookahead.
bw = windows[T - 1]
burn_cut_src = segments[bw["seg_id"]]["src_start"] + bw["tgt_end_seg_row"]
burn_cut_time = df["time"].iloc[burn_cut_src]
print(f"\n--- normalisation burn-in ---")
print(f"burn-in = windows [0,{T}); statistics use source rows <= {burn_cut_src} "
      f"({burn_cut_time})")

# Rows that are BOTH emitted and inside the burn-in prefix.
emit_mask = np.zeros(n_raw, bool)
for sg in segments:
    emit_mask[sg["src_start"]:sg["src_end"]] = True
burn_mask = emit_mask & (np.arange(n_raw) <= burn_cut_src)
print(f"burn-in rows used for statistics: {burn_mask.sum()} "
      f"(v1 used {43488} rows including dead ones; of those only 10504 were valid)")
if burn_mask.sum() < 2000:
    sys.exit("FATAL: burn-in too small to fit robust statistics")

# ---------------------------------------------------------------- normalisation
stats, norm = {}, {}
score_mask = emit_mask & (np.arange(n_raw) > burn_cut_src)
for c in SIGNALS:
    v = df[c].astype(float).values
    b = v[burn_mask]
    center = float(np.median(b))
    if c == TARGET:
        # Map the burn-in ROBUST RANGE [p0.5, p99.5] onto [-TARGET_BAND, +TARGET_BAND],
        # i.e. centre on the MIDPOINT of that range, not the median.
        #
        # Why the midpoint and not the median: dissolved N2O is a strongly
        # right-skewed positive quantity with a hard floor near 0 (median 0.03,
        # p99 1.17, max 2.07 on the scored span).  Median-centring puts the whole
        # signal into [-0.17, +0.75] -- the entire negative half of the tanh band
        # goes unused and the bulk is compressed to sd 0.085.  Midpoint-centring
        # spends the band symmetrically and gives the bulk sd ~0.13, ~1.6x more
        # resolution, without ever leaving the band.  It is also the mapping with
        # the shortest justification, which matters for something pre-registered.
        #
        # p0.5/p99.5 rather than min/max: the registered mask lets a handful of
        # full-scale rail readings through (RAIL, below) and min/max would let ~10
        # rows set the scale for ~100k.
        lo, hi = float(np.percentile(b, 0.5)), float(np.percentile(b, 99.5))
        center = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        if half <= 0:
            half = float(b.std()) or 1.0
        scale = half / TARGET_BAND
        clip = N2O_CLIP
    else:
        p1, p99 = float(np.percentile(b, 1)), float(np.percentile(b, 99))
        scale = (p99 - p1) / 2.0
        if scale <= 0:
            scale = float(b.std()) or 1.0
        clip = COV_CLIP
    u = (v - center) / scale                       # unclipped
    norm[c] = np.clip(u, -clip, clip)
    stats[c] = {
        "center": center, "scale": scale, "clip": clip,
        "burn_p1": float(np.percentile(b, 1)), "burn_p99": float(np.percentile(b, 99)),
        "clipped_frac_burn": float((np.abs(u[burn_mask]) > clip).mean()),
        "clipped_frac_score": float((np.abs(u[score_mask]) > clip).mean()),
        "outside_tanh_frac_score": float((np.abs(u[score_mask]) > 1.0).mean()),
        "norm_sd_burn": float(norm[c][burn_mask].std()),
        "norm_sd_score": float(norm[c][score_mask].std()),
    }
    print(f"  {c:<11} center {center:>12.4f} scale {scale:>12.4f} clip +/-{clip:<4.1f}"
          f"  clipped burn {100*stats[c]['clipped_frac_burn']:6.3f}%"
          f"  score {100*stats[c]['clipped_frac_score']:6.3f}%")

tg = stats[TARGET]
print(f"\nTARGET check (this is the v1 failure mode):")
print(f"  v1 center was -0.011441 (median of ALL rows incl. dead sensor)")
print(f"  v2 center is  {tg['center']:.6f} (midpoint of the VALID burn-in robust range)")
u_t = (df[TARGET].astype(float).values - tg["center"]) / tg["scale"]
print(f"  scored-span target in normalised units: "
      f"min {u_t[score_mask].min():+.3f} max {u_t[score_mask].max():+.3f}"
      f"  median {np.median(u_t[score_mask]):+.3f}  sd {u_t[score_mask].std():.4f}")
print(f"  PHYSICAL FLOOR of the normalised target (raw N2O = {df[TARGET].min():.4f}) = "
      f"{(df[TARGET].min()-tg['center'])/tg['scale']:+.3f}; any prediction below this is "
      f"nonphysical and score_v2.py counts them (v1's collapse sat at -0.963)")
print(f"  fraction of scored target OUTSIDE the tanh band |y|>1 : "
      f"{100*tg['outside_tanh_frac_score']:.4f}%")
print(f"  fraction of scored target clipped at +/-{N2O_CLIP}      : "
      f"{100*tg['clipped_frac_score']:.4f}%")

# RAIL: the registered mask lets a few full-scale readings through.  Measured, not patched.
rail = int(((df[TARGET].astype(float).values > 2.5) & emit_mask).sum())
print(f"  RESIDUAL ARTEFACT: {rail} emitted rows have N2O > 2.5 mg/L "
      f"(sensor rail, all 2022-11).  Left in: the registered mask defines them as "
      f"valid and the gates are registered against it.  Robust percentiles keep them "
      f"out of the scale; the +/-{N2O_CLIP} clip bounds their influence.")

# ---------------------------------------------------------------- emit
files = []
for sg in segments:
    s, e = sg["src_start"], sg["src_end"]
    block = pd.DataFrame({c: norm[c][s:e] for c in SIGNALS})
    fn = os.path.join(OUT, f"seg_{sg['seg_id']:04d}.csv")
    block.to_csv(fn, index=False, float_format="%.6f")
    sg["file"] = fn
    files.append(fn)
    # ONE-NAS will slice exactly this many windows out of this file.
    assert (len(block) - H) // L == sg["n_windows"]

with open(os.path.join(OUT, "filelist.txt"), "w") as fh:
    fh.write("\n".join(files) + "\n")

# sidecar: one row per EMITTED row, keyed by (seg_id, seg_row)
idx = []
for sg in segments:
    s, e = sg["src_start"], sg["src_end"]
    idx.append(pd.DataFrame({
        "seg_id": sg["seg_id"],
        "seg_row": np.arange(e - s),
        "src_row": np.arange(s, e),
        "time": df["time"].values[s:e],
        "n2o_raw": df[TARGET].astype(float).values[s:e],
        "n2o_norm": norm[TARGET][s:e],
        "n2o_valid": df["n2o_valid"].values[s:e],
    }))
idx = pd.concat(idx, ignore_index=True)
idx.to_csv(os.path.join(OUT, "index.csv"), index=False, float_format="%.6f")
assert bool(idx["n2o_valid"].astype(bool).all()), "an emitted row is not valid -- bug"

pd.DataFrame(windows).to_csv(os.path.join(OUT, "windows.csv"), index=False)

meta = {
    "generator": "prep_v2.py",
    "src": SRC, "start": START,
    "H": H, "L": L, "T": T, "V": V,
    "window_step": L,
    "min_segment_rows": MIN_SEG,
    "mask": "n2o_valid (registered, unmodified)",
    "target_band": TARGET_BAND, "n2o_clip": N2O_CLIP, "cov_clip": COV_CLIP,
    "signals": SIGNALS, "target": TARGET,
    "n_raw_rows_in_span": int(n_raw),
    "n_valid_rows_in_span": int(valid.sum()),
    "n_segments_all": int(len(seg_len)),
    "n_segments_used": int(len(segments)),
    "n_emitted_rows": int(emitted_rows),
    "n_windows": int(W),
    "max_generation": int(max_gen),
    "test_episode_of_generation": "g + T + V",
    "burn_cut_src_row": int(burn_cut_src),
    "burn_cut_time": str(burn_cut_time),
    "burn_rows_used_for_stats": int(burn_mask.sum()),
    "first_scored_window_time": windows[T + V]["tgt_start_time"],
    "stats": stats,
    "segments": segments,
    "rail_rows_gt_2p5": rail,
    "prediction_row_map": (
        "prediction file generation_<g>_global_best.csv data row i (0-based) corresponds "
        "to episode timestep j=i+1 (row 0 of the file is j=1; onenas_island_speciation_"
        "strategy.cxx:1119 starts the loop at j=1).  Episode e = g + T + V.  Look up e in "
        "windows.csv -> (seg_id, tgt_start_seg_row).  The scored target row is "
        "seg_row = tgt_start_seg_row + i + 1, and the forecast-issue row is "
        "seg_row - H.  expected_N2O at file row i MUST equal index.csv n2o_norm at "
        "(seg_id, tgt_start_seg_row + i + 1); score_v2.py refuses the generation otherwise."
    ),
}
with open(os.path.join(OUT, "prep_meta.json"), "w") as fh:
    json.dump(meta, fh, indent=2, default=float)

print(f"\nwrote {len(files)} segment CSVs + filelist.txt + index.csv ({len(idx)} rows)"
      f" + windows.csv ({W} rows) + prep_meta.json")
print(f"OUT={OUT}")
