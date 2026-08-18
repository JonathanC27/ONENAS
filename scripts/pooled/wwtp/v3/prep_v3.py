#!/usr/bin/env python3
"""
prep_v3.py -- ONE-NAS data preparation, v3.  Two stages.

WHAT v3 FIXES RELATIVE TO v2
----------------------------
(1) THE VALIDITY MASK WAS NON-CAUSAL.  prep.py builds `n2o_valid` from three
    components:
        target_real   pointwise, causal
        rail_bin      pointwise on RAW samples, causal
        flatline      n2o.rolling(288, center=True, min_periods=72).std() < 0.005
    The third one is CENTRED: validity at time t depends on samples out to
    t+12 h.  `n2o_valid` decides which rows are emitted, which rows train, which
    rows are scored and where every segment boundary falls, so the scored
    population is selected with hindsight.  v3 computes BOTH the centred mask
    (bit-identical to v1/v2, asserted) and a TRAILING mask (`center=False`,
    same window, same min_periods, same threshold) and carries both through, so
    the impact can be measured instead of assumed.  The rail component is left
    exactly as-is -- it is already causal.

(2) v2 THREW AWAY THE ONLY CLEAN PRE-SPAN.  v2 set START=2022-10-01, discarding
    the 2022-06-11..09-30 rows.  Those rows are strictly BEFORE anything v2
    scores, so using them carries zero lookahead risk, and v2's own worst
    problem is that its burn-in (and therefore its normalisation centre) is
    forced into the atypical March-April 2023 high-N2O excursion.  v3 keeps the
    pre-span and, with T_MODE=preserve, lengthens the burn-in prefix by exactly
    the number of windows the pre-span contributes, so the SCORED population is
    unchanged and the pre-span is used for normalisation and burn-in ONLY.

STAGE grid
    aved_raw.csv -> $ROOT/aved_5min_v3.csv
    Reproduces prep.py's 5-min grid exactly (asserted against aved_5min.csv) and
    additionally emits the mask COMPONENTS and the trailing-mask variant:
        n2o_real, rail_bin, flat_centred, flat_trailing,
        n2o_valid_centred (== v1's n2o_valid), n2o_valid_trailing

STAGE seg
    $ROOT/aved_5min_v3.csv -> $ROOT/<tag>/  (seg_*.csv, filelist.txt, index.csv,
    windows.csv, prep_meta.json), schema-compatible with score_v2.py and
    baselines_v2.py.  Config via env: MASK, SPAN, H, L, T0, V, T_MODE.

Nothing in v1 or v2 is written to or modified.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- shared config
RAW = os.environ.get("RAW", "/anvil/scratch/x-jchang5/wwtp/aved_raw.csv")
V1_GRID = os.environ.get("V1_GRID", "/anvil/scratch/x-jchang5/wwtp/aved_5min.csv")
ROOT = os.environ.get("ROOT", "/anvil/scratch/x-jchang5/wwtp/v3")
GRID_CSV = os.path.join(ROOT, "aved_5min_v3.csv")

GRID = "5min"
DEAD_STD = 0.005
DEAD_WINDOW = "24h"
FFILL_LIMIT = 12
TARGET_FFILL = 3

SIGNALS_MAP = {
    "N2O":        "BIOLOGY.LINE 3 TANK 1.N2O",
    "NH4":        "BIOLOGY.LINE 3 TANK 1.NH4",
    "NO3":        "BIOLOGY.LINE 3 TANK 1.NO3",
    "PO4":        "BIOLOGY.LINE 3 TANK 1.PO4",
    "O2_T1":      "BIOLOGY.LINE 3 TANK 1.O2",
    "O2_T2":      "BIOLOGY.LINE 3 TANK 2.O2",
    "O2_SP":      "BIOLOGY.LINE 3 TANK 1.O2.SETPOINT",
    "AIR_T1":     "BIOLOGY.LINE 3 TANK 1.Q.AIRFLOW",
    "AIR_T2":     "BIOLOGY.LINE 3 TANK 2.Q.AIRFLOW",
    "AIR_BLOWER": "BIOLOGY.BLOWERSTATION 1.Q.AIRFLOW",
    "SS_T1":      "BIOLOGY.LINE 3 TANK 1.SS",
    "TEMP":       "BIOLOGY.LINE 3 TANK 1.TEMPERATURE",
    "INLET_Q":    "INLET.Q",
    "SWM":        "INLET.STATE.SWM INLET FLOW",
}
SIGNALS = list(SIGNALS_MAP)
TARGET = "N2O"


# ================================================================= STAGE: grid
def stage_grid():
    os.makedirs(ROOT, exist_ok=True)
    usecols = ["time"]
    for base in SIGNALS_MAP.values():
        usecols += [base + " value", base + " quality"]

    print("reading raw ...", flush=True)
    raw = pd.read_csv(RAW, usecols=usecols, low_memory=False)
    t = pd.to_datetime(raw["time"], utc=True, format="mixed")
    print(f"raw rows {len(raw)}  span {t.iloc[0]} .. {t.iloc[-1]}")

    vals = {}
    for name, base in SIGNALS_MAP.items():
        v = pd.to_numeric(raw[base + " value"], errors="coerce")
        q = pd.to_numeric(raw[base + " quality"], errors="coerce")
        vals[name] = v.mask(q == 1)
    wide = pd.DataFrame(vals)
    wide.index = t

    # rail detection on RAW samples (pointwise, causal -- unchanged from v1)
    n2o_raw = pd.to_numeric(raw[SIGNALS_MAP[TARGET] + " value"], errors="coerce")
    rail_value = float(np.nanmin(n2o_raw.values))
    at_rail = pd.Series(np.isclose(n2o_raw.values, rail_value, rtol=0, atol=1e-12)
                        & n2o_raw.notna().values, index=t)
    print(f"rail value {rail_value!r}  raw samples at rail {int(at_rail.sum())}")

    print(f"resampling to {GRID} ...", flush=True)
    g = wide.resample(GRID).mean()
    rail_bin = at_rail.resample(GRID).max().reindex(g.index).fillna(False).astype(bool)
    print(f"grid rows {len(g)}  ({g.index[0]} .. {g.index[-1]})")

    # ---- validity, both variants.  Computed on the PRE-FILL resampled target,
    # exactly as v1 does; only `center` differs between the two flatline tests.
    n2o = g[TARGET]
    target_real = n2o.notna()
    win = int(pd.Timedelta(DEAD_WINDOW) / pd.Timedelta(GRID))
    std_c = n2o.rolling(win, center=True, min_periods=win // 4).std()
    std_t = n2o.rolling(win, center=False, min_periods=win // 4).std()
    flat_c = (std_c < DEAD_STD).fillna(False)
    flat_t = (std_t < DEAD_STD).fillna(False)
    valid_c = target_real & (~rail_bin) & (~flat_c)
    valid_t = target_real & (~rail_bin) & (~flat_t)

    print(f"\nwindow {win} bins ({DEAD_WINDOW}), min_periods {win//4}, DEAD_STD {DEAD_STD}")
    print(f"target_real        {int(target_real.sum())}")
    print(f"rail_bin           {int(rail_bin.sum())}   (identical in both variants)")
    print(f"flat CENTRED       {int((target_real & flat_c).sum())}")
    print(f"flat TRAILING      {int((target_real & flat_t).sum())}")
    print(f"valid CENTRED      {int(valid_c.sum())}")
    print(f"valid TRAILING     {int(valid_t.sum())}")

    # ---- gap filling (identical to v1; masks were computed before this)
    feat_cols = [c for c in g.columns if c != TARGET]
    g[feat_cols] = g[feat_cols].ffill(limit=FFILL_LIMIT)
    g[TARGET] = g[TARGET].ffill(limit=TARGET_FFILL)
    g = g.interpolate(method="time", limit_direction="both")
    g = g.bfill().ffill()
    assert not g.isna().any().any(), "dense matrix still has NaN"

    out = g.copy()
    out.insert(0, "n2o_valid_centred", valid_c.astype(int).values)
    out.insert(1, "n2o_valid_trailing", valid_t.astype(int).values)
    out.insert(2, "n2o_real", target_real.astype(int).values)
    out.insert(3, "rail_bin", rail_bin.astype(int).values)
    out.insert(4, "flat_centred", flat_c.astype(int).values)
    out.insert(5, "flat_trailing", flat_t.astype(int).values)
    out.index.name = "time"
    out = out.reset_index()
    out["time"] = out["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    cols = (["time", "n2o_valid_centred", "n2o_valid_trailing", "n2o_real",
             "rail_bin", "flat_centred", "flat_trailing"] + SIGNALS)
    out = out[cols]

    # ---- PROVENANCE ASSERTION: the centred variant must reproduce v1 bit-for-bit
    v1 = pd.read_csv(V1_GRID)
    assert len(v1) == len(out), f"grid length {len(out)} != v1 {len(v1)}"
    assert (v1["time"].values == out["time"].values).all(), "grid timestamps differ from v1"
    nd = int((v1["n2o_valid"].values != out["n2o_valid_centred"].values).sum())
    print(f"\nPROVENANCE: rows where centred mask != v1 n2o_valid: {nd}")
    assert nd == 0, "centred mask does not reproduce v1 -- v3 grid is not comparable"
    for c in SIGNALS:
        d = np.abs(v1[c].values - out[c].values)
        assert np.nanmax(d) < 2e-6, f"{c} differs from v1 grid by {np.nanmax(d)}"
    print("PROVENANCE: signal columns reproduce v1 to 2e-6; centred mask identical.")

    out.to_csv(GRID_CSV, index=False, float_format="%.6f")
    print(f"wrote {GRID_CSV}  shape {out.shape}")
    with open(os.path.join(ROOT, "grid_meta.json"), "w") as fh:
        json.dump({"generator": "prep_v3.py stage=grid", "raw": RAW,
                   "rail_value": rail_value, "dead_std": DEAD_STD,
                   "dead_window": DEAD_WINDOW, "roll_min_periods": win // 4,
                   "grid_rows": int(len(out)),
                   "n_target_real": int(target_real.sum()),
                   "n_rail_bin": int(rail_bin.sum()),
                   "n_flat_centred": int((target_real & flat_c).sum()),
                   "n_flat_trailing": int((target_real & flat_t).sum()),
                   "n_valid_centred": int(valid_c.sum()),
                   "n_valid_trailing": int(valid_t.sum()),
                   "reproduces_v1_mask": True}, fh, indent=2)


# ================================================================== STAGE: seg
def stage_seg():
    MASK = os.environ.get("MASK", "trailing")          # centred | trailing
    SPAN = os.environ.get("SPAN", "full")              # v2 | full
    H = int(os.environ.get("H", "72"))
    L = int(os.environ.get("L", "144"))
    T0 = int(os.environ.get("T0", "80"))               # ONE-NAS burn-in in v2
    V = int(os.environ.get("V", "5"))
    T_MODE = os.environ.get("T_MODE", "preserve")      # preserve | fixed
    TAG = os.environ.get("TAG", f"h{H}_L{L}_{MASK}_{SPAN}_{T_MODE}")


    TARGET_BAND = float(os.environ.get("TARGET_BAND", "0.6"))
    # Robust range of the burn-in target that is mapped onto +/-TARGET_BAND.
    # v2 registered p0.5/p99.5.  Exposed here because reclaiming the pre-span
    # moves the burn-in's upper tail (June 2022 holds the series max, 2.0694),
    # which widens the scale and costs target resolution on the scored span.
    QLO = float(os.environ.get("TARGET_QLO", "0.5"))
    QHI = float(os.environ.get("TARGET_QHI", "99.5"))
    N2O_CLIP = float(os.environ.get("N2O_CLIP", "0.95"))
    COV_CLIP = float(os.environ.get("COV_CLIP", "4.0"))
    MIN_SEG = L + H

    V2_START = "2022-10-01"
    OUT = os.path.join(ROOT, TAG)
    os.makedirs(OUT, exist_ok=True)
    print(f"=== prep_v3 seg  MASK={MASK} SPAN={SPAN} H={H} L={L} T0={T0} V={V} "
          f"T_MODE={T_MODE} ===")
    print(f"out {OUT}")

    df = pd.read_csv(GRID_CSV, parse_dates=["time"])
    start = V2_START if SPAN == "v2" else str(df["time"].iloc[0])
    df = df[df["time"] >= pd.Timestamp(start)].reset_index(drop=True)
    n_raw = len(df)
    dt = df["time"].diff().dropna().dt.total_seconds().unique()
    assert len(dt) == 1 and dt[0] == 300.0, f"grid not uniform 5-min: {dt[:5]}"

    valid = df[f"n2o_valid_{MASK}"].astype(bool).values
    print(f"rows from {start}: {n_raw}   valid: {valid.sum()} ({100*valid.mean():.2f}%)")

    # ---- maximal contiguous valid runs, half-open [s,e)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], valid.astype(np.int8), [0]))) != 0)
    seg_s, seg_e = edges[0::2], edges[1::2]
    seg_len = seg_e - seg_s
    keep = seg_len >= MIN_SEG
    ks, ke = seg_s[keep], seg_e[keep]
    print(f"runs {len(seg_len)}  total valid rows {seg_len.sum()}   "
          f"usable (>= {MIN_SEG}) {keep.sum()} holding {seg_len[keep].sum()} rows")

    # ---- windows, exactly nw*L + H rows per segment
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
            in0, tg0 = w * L, w * L + H
            windows.append({"window_id": len(windows), "seg_id": sid, "seg_window": w,
                            "in_start_seg_row": in0, "tgt_start_seg_row": tg0,
                            "tgt_end_seg_row": tg0 + L - 1,
                            "in_start_time": str(df["time"].iloc[s + in0]),
                            "tgt_start_time": str(df["time"].iloc[s + tg0]),
                            "tgt_end_time": str(df["time"].iloc[s + tg0 + L - 1])})
    W = len(windows)
    emitted_rows = sum(sg["n_rows"] for sg in segments)

    # ---- burn-in length.  T_MODE=preserve lengthens the burn-in prefix by the
    # number of windows that lie ENTIRELY before v2's START, so those windows can
    # only ever be burn-in / normalisation and the scored population is the one
    # v2 would have scored.
    n_pre = sum(1 for w in windows if pd.Timestamp(w["tgt_end_time"]) < pd.Timestamp(V2_START))
    T = T0 + n_pre if (T_MODE == "preserve" and SPAN == "full") else T0
    print(f"windows W={W}  emitted rows {emitted_rows}")
    print(f"windows entirely before {V2_START}: n_pre={n_pre}   -> T={T} (T0={T0})")

    max_gen = W - T - V - 1
    if max_gen < 1:
        sys.exit(f"FATAL: W={W}, T={T}, V={V} leaves {max_gen} generations")
    print(f"max_generation = W - T - V - 1 = {max_gen}")
    print(f"first scored window {T+V} tgt starts {windows[T+V]['tgt_start_time']}")
    print(f"last  scored window {max_gen-1+T+V} tgt ends {windows[max_gen-1+T+V]['tgt_end_time']}")

    # ---- normalisation on the burn-in prefix only
    bw = windows[T - 1]
    burn_cut_src = segments[bw["seg_id"]]["src_start"] + bw["tgt_end_seg_row"]
    burn_cut_time = df["time"].iloc[burn_cut_src]
    emit_mask = np.zeros(n_raw, bool)
    for sg in segments:
        emit_mask[sg["src_start"]:sg["src_end"]] = True
    burn_mask = emit_mask & (np.arange(n_raw) <= burn_cut_src)
    score_mask = emit_mask & (np.arange(n_raw) > burn_cut_src)
    pre_mask = burn_mask & (df["time"].values < np.datetime64(pd.Timestamp(V2_START)))
    print(f"burn-in rows for stats: {burn_mask.sum()}  (of which pre-{V2_START}: "
          f"{pre_mask.sum()})   cut {burn_cut_time}")
    if burn_mask.sum() < 2000:
        sys.exit("FATAL: burn-in too small")

    stats, norm = {}, {}
    for c in SIGNALS:
        v = df[c].astype(float).values
        b = v[burn_mask]
        if c == TARGET:
            lo, hi = float(np.percentile(b, QLO)), float(np.percentile(b, QHI))
            center = 0.5 * (lo + hi)
            half = 0.5 * (hi - lo) or (float(b.std()) or 1.0)
            scale, clip = half / TARGET_BAND, N2O_CLIP
        else:
            center = float(np.median(b))
            p1, p99 = float(np.percentile(b, 1)), float(np.percentile(b, 99))
            scale = (p99 - p1) / 2.0 or (float(b.std()) or 1.0)
            clip = COV_CLIP
        u = (v - center) / scale
        norm[c] = np.clip(u, -clip, clip)
        stats[c] = {"center": center, "scale": scale, "clip": clip,
                    "clipped_frac_burn": float((np.abs(u[burn_mask]) > clip).mean()),
                    "clipped_frac_score": float((np.abs(u[score_mask]) > clip).mean()),
                    "outside_tanh_frac_score": float((np.abs(u[score_mask]) > 1.0).mean()),
                    "norm_sd_burn": float(norm[c][burn_mask].std()),
                    "norm_sd_score": float(norm[c][score_mask].std())}
    tg = stats[TARGET]
    u_t = (df[TARGET].astype(float).values - tg["center"]) / tg["scale"]
    print(f"\nTARGET center {tg['center']:.6f}  scale {tg['scale']:.6f}")
    print(f"  burn-in raw N2O: median {np.median(df[TARGET].values[burn_mask]):.4f} "
          f"mean {df[TARGET].values[burn_mask].mean():.4f}")
    print(f"  scored raw N2O : median {np.median(df[TARGET].values[score_mask]):.4f} "
          f"mean {df[TARGET].values[score_mask].mean():.4f}")
    print(f"  scored normalised: median {np.median(u_t[score_mask]):+.4f} "
          f"sd {u_t[score_mask].std():.4f}  min {u_t[score_mask].min():+.3f} "
          f"max {u_t[score_mask].max():+.3f}")
    print(f"  burn-in normalised: median {np.median(u_t[burn_mask]):+.4f} "
          f"sd {u_t[burn_mask].std():.4f}")

    # ---- emit
    files = []
    for sg in segments:
        s, e = sg["src_start"], sg["src_end"]
        block = pd.DataFrame({c: norm[c][s:e] for c in SIGNALS})
        fn = os.path.join(OUT, f"seg_{sg['seg_id']:04d}.csv")
        block.to_csv(fn, index=False, float_format="%.6f")
        sg["file"] = fn
        files.append(fn)
        assert (len(block) - H) // L == sg["n_windows"]
    with open(os.path.join(OUT, "filelist.txt"), "w") as fh:
        fh.write("\n".join(files) + "\n")

    idx = []
    for sg in segments:
        s, e = sg["src_start"], sg["src_end"]
        idx.append(pd.DataFrame({
            "seg_id": sg["seg_id"], "seg_row": np.arange(e - s),
            "src_row": np.arange(s, e), "time": df["time"].values[s:e],
            "n2o_raw": df[TARGET].astype(float).values[s:e],
            "n2o_norm": norm[TARGET][s:e],
            "n2o_valid": np.ones(e - s, dtype=int)}))
    idx = pd.concat(idx, ignore_index=True)
    idx.to_csv(os.path.join(OUT, "index.csv"), index=False, float_format="%.6f")
    pd.DataFrame(windows).to_csv(os.path.join(OUT, "windows.csv"), index=False)

    meta = {"generator": "prep_v3.py stage=seg", "src": GRID_CSV, "start": start,
            "mask_variant": MASK, "span": SPAN, "t_mode": T_MODE,
            "H": H, "L": L, "T": T, "T0": T0, "V": V, "n_pre_windows": int(n_pre),
            "window_step": L, "min_segment_rows": MIN_SEG,
            "mask": f"n2o_valid_{MASK}",
            "target_band": TARGET_BAND, "target_qlo": QLO, "target_qhi": QHI,
            "n2o_clip": N2O_CLIP, "cov_clip": COV_CLIP,
            "signals": SIGNALS, "target": TARGET,
            "n_raw_rows_in_span": int(n_raw), "n_valid_rows_in_span": int(valid.sum()),
            "n_segments_all": int(len(seg_len)), "n_segments_used": int(len(segments)),
            "n_emitted_rows": int(emitted_rows), "n_windows": int(W),
            "max_generation": int(max_gen),
            "test_episode_of_generation": "g + T + V",
            "burn_cut_src_row": int(burn_cut_src), "burn_cut_time": str(burn_cut_time),
            "burn_rows_used_for_stats": int(burn_mask.sum()),
            "burn_rows_pre_v2start": int(pre_mask.sum()),
            "first_scored_window_time": windows[T + V]["tgt_start_time"],
            "last_scored_window_time": windows[max_gen - 1 + T + V]["tgt_end_time"],
            "stats": stats, "segments": segments}
    with open(os.path.join(OUT, "prep_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=float)
    print(f"wrote {len(files)} segment CSVs, index.csv ({len(idx)}), windows.csv ({W}), "
          f"prep_meta.json -> {OUT}")


if __name__ == "__main__":
    st = sys.argv[1] if len(sys.argv) > 1 else "seg"
    {"grid": stage_grid, "seg": stage_seg}[st]()
