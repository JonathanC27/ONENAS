#!/usr/bin/env python3
"""
baselines_v3.py <prep_dir> [out_json]

Computes the baseline / gate table on EXACTLY the rows ONE-NAS will be scored on.

WHY THIS FILE EXISTS
--------------------
v2's `baselines_v2.py` scored every usable (issue,target) pair in the WHOLE
emitted series -- burn-in included (n = 93,456 at H=72).  `score_v2.py` scores
only the rows covered by test windows (episodes T+V .. T+V+G-1, timesteps
j = 1..L-1; n = 84,177 after the arm intersection).  Those are different row
sets over different calendar spans, so the pre-registered gate
"persistence = 0.8384" is a number from a population that will never be scored.
On the scored rows persistence at H=72 is ~1.05, i.e. WORSE than a constant.

THE SCORED ROW SET (closed form, same derivation as score_v2.py)
---------------------------------------------------------------
  generation g -> test episode e = g + T + V          online_series.cxx:341-372
  g runs 0 .. max_generation-1, max_generation = W - T - V - 1
  episode e -> windows.csv row e -> (seg_id, tgt_start_seg_row)
  prediction file row i -> episode timestep j = i + 1  (i = 0..L-2)
                                        onenas_island_speciation_strategy.cxx:1119
  scored target row = tgt_start_seg_row + j,  j = 1..L-1
  forecast-issue row = scored target row - H
This file assumes every generation lands (the run completes); that is the
maximal scored population and the one the gate must be registered against.

ARM INTERSECTION.  score_v2.py evaluates on `have & finite(every external arm)`.
Ridge is NaN wherever the 72-row N2O lag history crosses a segment start, so
including ridge shrinks the row set for every arm.  Both row sets are reported:
  all_scored  -- every scored row (arms defined everywhere)
  common      -- all_scored intersected with finite ridge (what score_v2.py uses)

WARM-UP AND THE RIDGE FAIRNESS ASYMMETRY
----------------------------------------
ONE-NAS resets its recurrent state at every test window (rnn.cxx:372-381), so at
timestep j it has seen only rows wL..wL+j of the window.  Ridge's N2O lag 72
reaches back to issue_row-72 = wL+j-72, which is BEFORE the window start whenever
j <= 71.  On those rows ridge has history ONE-NAS structurally cannot have.
Every table is therefore reported at wu = 0 (all rows) and wu = 71 (j >= 72 only,
where both models see only in-window history), plus the v2 sensitivity points.
`wu` is `in_window_pos >= wu`, i.e. j >= wu+1, matching score_v2.py's WARMUPS.

UNCERTAINTY.  Paired ratio to persistence R = sum SSE_arm / sum SSE_pers (the
nMSE denominator cancels, so R is exactly the nMSE ratio).  CI from a moving-block
bootstrap over TEST WINDOWS with block length 14 windows, B replicates, resampled
in chronological blocks so within-window and short-range serial correlation is
preserved.  Also reported: median per-window ratio, and the fraction of windows
where the arm beats persistence.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

PREP = sys.argv[1].rstrip("/")
OUTJ = sys.argv[2] if len(sys.argv) > 2 else os.path.join(PREP, "baselines_v3.json")

LAGS = [1, 3, 6, 12, 24, 72]
MAXLAG = max(LAGS)
PER_DAY = 288
RETRAIN_DAYS = [1, 3, 7, 30]
ROLL_DAYS = [1, 3, 7, 30]
ALPHA = 1.0
MIN_TRAIN = 2000
WARMUPS = (0, 12, 24, 48, 71)
BLOCK = 14          # windows per bootstrap block
B = 2000
SHRINK_KS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
RNG = np.random.default_rng(20260908)

meta = json.load(open(os.path.join(PREP, "prep_meta.json")))
H, L, T, V = meta["H"], meta["L"], meta["T"], meta["V"]
G = meta["max_generation"]
SIGNALS = meta["signals"]
print(f"=== baselines_v3  {PREP}")
print(f"    mask={meta.get('mask')} span={meta.get('span')} t_mode={meta.get('t_mode')} "
      f"H={H} L={L} T={T} V={V} W={meta['n_windows']} max_gen={G}")

idx = pd.read_csv(os.path.join(PREP, "index.csv"), parse_dates=["time"])
win = pd.read_csv(os.path.join(PREP, "windows.csv"))
n = len(idx)
seg = idx["seg_id"].values
srow = idx["seg_row"].values
y = idx["n2o_raw"].values.astype(float)
t = idx["time"]

# flat offset of each segment inside index.csv (same construction as score_v2.py)
seg_off = idx.groupby("seg_id", sort=True).size().cumsum().shift(1).fillna(0).astype(int).to_dict()

# ------------------------------------------------------------------ scored rows
ep_of_row = np.full(n, -1, np.int32)     # episode that scores this row
pos_of_row = np.full(n, -1, np.int32)    # in_window_pos i = j-1
for g in range(G):
    e = g + T + V
    w = win.iloc[e]
    assert int(w["window_id"]) == e
    base = seg_off[int(w["seg_id"])] + int(w["tgt_start_seg_row"])
    sl = np.arange(base + 1, base + L)               # j = 1..L-1
    ep_of_row[sl] = e
    pos_of_row[sl] = np.arange(L - 1)
scored = ep_of_row >= 0
print(f"    scored rows {scored.sum()} = {G} episodes x {L-1} timesteps "
      f"({100*scored.sum()/n:.1f}% of {n} emitted)")
print(f"    scored span {t[scored].min()} .. {t[scored].max()}")
# persistence anchors must be in-segment by construction
iss = np.flatnonzero(scored) - H
assert (seg[np.flatnonzero(scored)] == seg[iss]).all(), "persistence anchor left its segment"

# ------------------------------------------------------------------- features
blocks = [pd.read_csv(s["file"]) for s in meta["segments"]]
X_sig = pd.concat(blocks, ignore_index=True)
assert len(X_sig) == n
feats = [X_sig[c].values.astype(float) for c in SIGNALS]
nn = X_sig["N2O"].values.astype(float)
for lag in LAGS:
    v = np.full(n, np.nan)
    v[lag:] = nn[:-lag]
    v[srow < lag] = np.nan                 # crosses the segment start
    feats.append(v)
hod = t.dt.hour.values + t.dt.minute.values / 60.0
dow = t.dt.dayofweek.values
for p, vals in ((24.0, hod), (7.0, dow)):
    feats.append(np.sin(2 * np.pi * vals / p))
    feats.append(np.cos(2 * np.pi * vals / p))
X = np.column_stack(feats)

tgt = np.arange(n) + H
ok = tgt < n
ok[ok] &= seg[tgt[ok]] == seg[np.flatnonzero(ok)]
ok &= np.isfinite(X).all(axis=1)
issue_pos = np.flatnonzero(ok)
tgt_pos = issue_pos + H
Xs, ys = X[issue_pos], y[tgt_pos]
print(f"    ridge-eligible (issue,target) pairs over all emitted rows: {len(issue_pos)}")

# --------------------------------------------------------------------- arms
# All arms predict the target at flat row r from information available at the
# forecast-issue row p = r - H.  Emitted rows are chronological, so a trailing
# statistic over flat rows <= p is causal.  Trailing statistics are allowed to
# cross segment boundaries (that is real observed past); ridge lags are not
# (a lag would silently point at a different calendar time).
ser = pd.Series(y)
clim_exp = ser.expanding().mean().values                      # causal climatology
roll = {k: ser.rolling(k * PER_DAY, min_periods=1).mean().values for k in ROLL_DAYS}

pred = {}
r_idx = np.arange(n)
p_idx = r_idx - H
valid_p = p_idx >= 0


def from_issue(a):
    out = np.full(n, np.nan)
    out[valid_p] = a[p_idx[valid_p]]
    return out


pred["persistence"] = from_issue(y)
pred["climatology_causal_expanding"] = from_issue(clim_exp)
for k in ROLL_DAYS:
    pred[f"rolling_mean_d{k}"] = from_issue(roll[k])
pred["blend_pers_roll7"] = 0.5 * (pred["persistence"] + pred["rolling_mean_d7"])
for kk in SHRINK_KS:
    pred[f"shrunk_causal_k{kk:.1f}"] = (pred["climatology_causal_expanding"]
                                        + kk * (pred["persistence"]
                                                - pred["climatology_causal_expanding"]))


def ridge_fit(Xtr, ytr):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    Z = np.column_stack([(Xtr - mu) / sd, np.ones(len(Xtr))])
    A = Z.T @ Z + ALPHA * np.eye(Z.shape[1])
    A[-1, -1] -= ALPHA
    return mu, sd, np.linalg.solve(A, Z.T @ ytr)


def ridge_pred(m, Xte):
    mu, sd, w = m
    return np.column_stack([(Xte - mu) / sd, np.ones(len(Xte))]) @ w


for days in RETRAIN_DAYS:
    K = days * PER_DAY
    pr = np.full(len(issue_pos), np.nan)
    model, nfit = None, 0
    cps = np.arange(0, n + K, K)
    for c0, c1 in zip(cps[:-1], cps[1:]):
        blk = np.flatnonzero((tgt_pos >= c0) & (tgt_pos < c1))
        if not len(blk):
            continue
        tr = np.flatnonzero(tgt_pos < c0)
        if len(tr) >= MIN_TRAIN:
            model = ridge_fit(Xs[tr], ys[tr])
            nfit += 1
        if model is not None:
            pr[blk] = ridge_pred(model, Xs[blk])
    a = np.full(n, np.nan)
    a[tgt_pos] = pr
    pred[f"ridge_d{days}"] = a
    print(f"    ridge_d{days}: {nfit} refits")

# causally-fitted shrinkage: at each daily checkpoint pick k on a grid by SSE over
# every pair whose TARGET has already been observed.  No oracle.
kgrid = np.linspace(0.0, 1.2, 61)
pk = np.full(n, np.nan)
kser = np.full(n, np.nan)
K = PER_DAY
cps = np.arange(0, n + K, K)
kcur = 1.0
pers_all, clim_all = pred["persistence"], pred["climatology_causal_expanding"]
for c0, c1 in zip(cps[:-1], cps[1:]):
    blk = np.arange(c0, min(c1, n))
    tr = np.flatnonzero((r_idx < c0) & np.isfinite(pers_all) & np.isfinite(clim_all))
    if len(tr) >= MIN_TRAIN:
        d = pers_all[tr] - clim_all[tr]
        e0 = y[tr] - clim_all[tr]
        sse = np.array([np.sum((e0 - kk * d) ** 2) for kk in kgrid])
        kcur = float(kgrid[np.argmin(sse)])
    pk[blk] = clim_all[blk] + kcur * (pers_all[blk] - clim_all[blk])
    kser[blk] = kcur
pred["shrunk_causal_kfit"] = pk

# in-sample-mean shrinkage (ORACLE, reported so it cannot be mistaken for a real arm)
for kk in (0.5,):
    pred[f"ORACLE_shrunk_insample_k{kk:.1f}"] = None   # filled per row-set below

ARMS = [a for a in pred if pred[a] is not None]

# ----------------------------------------------------------------- evaluation
finite_all = np.ones(n, bool)
for a in ARMS:
    finite_all &= np.isfinite(pred[a])
ridge_finite = np.ones(n, bool)
for days in RETRAIN_DAYS:
    ridge_finite &= np.isfinite(pred[f"ridge_d{days}"])
nonridge_finite = np.ones(n, bool)
for a in ARMS:
    if not a.startswith("ridge_"):
        nonridge_finite &= np.isfinite(pred[a])

ROWSETS = {"all_scored": scored & nonridge_finite,
           "common": scored & finite_all}
print(f"    row sets: all_scored {ROWSETS['all_scored'].sum()}   "
      f"common(with ridge) {ROWSETS['common'].sum()}   "
      f"(ridge drops {ROWSETS['all_scored'].sum()-ROWSETS['common'].sum()} rows, "
      f"{100*(1-ROWSETS['common'].sum()/ROWSETS['all_scored'].sum()):.1f}%)")


def nmse(yy, pp):
    den = np.sum((yy - yy.mean()) ** 2)
    return float(np.sum((yy - pp) ** 2) / den) if den > 0 else np.nan


def evaluate(mask, label):
    m = np.flatnonzero(mask)
    if len(m) < 200:
        return None
    yy = y[m]
    eps = ep_of_row[m]
    uep, inv = np.unique(eps, return_index=False), None
    uep = np.unique(eps)
    epi = np.searchsorted(uep, eps)
    nE = len(uep)
    den = np.sum((yy - yy.mean()) ** 2)

    arms = dict(pred)
    arms[f"ORACLE_shrunk_insample_k0.5"] = None
    res = {}
    sse_ep = {}
    for a in ARMS:
        pp = pred[a][m]
        res[a] = float(np.sum((yy - pp) ** 2) / den)
        sse_ep[a] = np.bincount(epi, weights=(yy - pp) ** 2, minlength=nE)
    # oracle arms, defined only relative to this row set
    mu_in = yy.mean()
    res["ORACLE_climatology_insample"] = float(np.sum((yy - mu_in) ** 2) / den)
    sse_ep["ORACLE_climatology_insample"] = np.bincount(epi, weights=(yy - mu_in) ** 2,
                                                        minlength=nE)
    po = mu_in + 0.5 * (pred["persistence"][m] - mu_in)
    res["ORACLE_shrunk_insample_k0.5"] = float(np.sum((yy - po) ** 2) / den)
    sse_ep["ORACLE_shrunk_insample_k0.5"] = np.bincount(epi, weights=(yy - po) ** 2,
                                                        minlength=nE)

    # ---- moving-block bootstrap on the paired ratio to persistence
    nb = int(np.ceil(nE / BLOCK))
    starts = RNG.integers(0, max(nE - BLOCK + 1, 1), size=(B, nb))
    off = np.arange(BLOCK)
    take = (starts[:, :, None] + off[None, None, :]).reshape(B, -1)[:, :nE]
    take = np.clip(take, 0, nE - 1)
    sp = sse_ep["persistence"]
    dp = sp[take].sum(1)
    sc = sse_ep["ORACLE_climatology_insample"]      # nMSE denominator, per episode
    dc = sc[take].sum(1)
    out = {}
    for a, s in sse_ep.items():
        r = s[take].sum(1) / dp
        an = s[take].sum(1) / dc
        ok_e = sp > 0
        pw = np.divide(s, sp, out=np.full(nE, np.nan), where=ok_e)
        out[a] = {"nmse": res[a], "ratio": float(s.sum() / sp.sum()),
                  "ci_lo": float(np.percentile(r, 2.5)),
                  "ci_hi": float(np.percentile(r, 97.5)),
                  "nmse_ci_lo": float(np.percentile(an, 2.5)),
                  "nmse_ci_hi": float(np.percentile(an, 97.5)),
                  "median_window_ratio": float(np.nanmedian(pw)),
                  "win_rate": float(np.nanmean(pw < 1.0))}
    out["_n"] = int(len(m))
    out["_n_windows"] = int(nE)
    return out


results = {}
for rs, rmask in ROWSETS.items():
    for wu in WARMUPS:
        lab = f"{rs}|wu{wu}"
        r = evaluate(rmask & (pos_of_row >= wu), lab)
        if r:
            results[lab] = r
# rows where ridge's 72-row N2O history reaches BEFORE the window start and
# ONE-NAS's recurrent state cannot: the complement of wu71.
r = evaluate(ROWSETS["common"] & (pos_of_row >= 0) & (pos_of_row < 71), "common|j_le_71")
if r:
    results["common|j_le_71"] = r

order = (["persistence", "ORACLE_climatology_insample", "climatology_causal_expanding"]
         + [f"rolling_mean_d{k}" for k in ROLL_DAYS]
         + ["blend_pers_roll7"]
         + [f"ridge_d{k}" for k in RETRAIN_DAYS]
         + [f"shrunk_causal_k{k:.1f}" for k in SHRINK_KS]
         + ["shrunk_causal_kfit", "ORACLE_shrunk_insample_k0.5"])

for lab in ([f"{rs}|wu{wu}" for rs in ROWSETS for wu in WARMUPS] + ["common|j_le_71"]):
    if lab not in results:
        continue
    r = results[lab]
    print(f"\n{'='*104}\n{lab}   n={r['_n']}  windows={r['_n_windows']}\n{'='*104}")
    print(f"{'arm':<34}{'nMSE':>9}{'  95% CI nMSE':>19}{'ratio':>8}"
          f"{'  95% CI paired':>21}{'medWin':>9}{'win%':>7}")
    for a in order:
        if a not in r:
            continue
        if lab.startswith("all_scored") and a.startswith("ridge_"):
            continue          # ridge is undefined on rows it drops; see 'common'
        d = r[a]
        print(f"{a:<34}{d['nmse']:>9.4f}"
              f"  [{d['nmse_ci_lo']:>6.4f},{d['nmse_ci_hi']:>7.4f}]"
              f"{d['ratio']:>8.4f}  [{d['ci_lo']:>6.4f},{d['ci_hi']:>7.4f}]"
              f"{d['median_window_ratio']:>9.4f}{100*d['win_rate']:>7.1f}")

# ------------------------------------------------------ per-month persistence
print(f"\n{'='*104}\nPER-MONTH, primary row set (common|wu0): persistence vs the "
      f"no-skill line\n{'='*104}")
m0 = ROWSETS["common"]
mon = pd.PeriodIndex(t, freq="M")
nb_above = 0
tot = 0
print(f"{'month':<9}{'n':>7}{'pers':>9}{'clim_causal':>13}{'roll7':>9}{'shrunk_kfit':>13}"
      f"{'ridge_d1':>10}")
permonth = {}
for mm in sorted(set(mon[m0])):
    k = np.asarray(mon == mm) & m0
    if k.sum() < 200:
        continue
    yy = y[k]
    row = {a: nmse(yy, pred[a][k]) for a in
           ("persistence", "climatology_causal_expanding", "rolling_mean_d7",
            "shrunk_causal_kfit", "ridge_d1")}
    row["n"] = int(k.sum())
    permonth[str(mm)] = row
    tot += 1
    nb_above += row["persistence"] > 1.0
    print(f"{str(mm):<9}{k.sum():>7}{row['persistence']:>9.4f}"
          f"{row['climatology_causal_expanding']:>13.4f}{row['rolling_mean_d7']:>9.4f}"
          f"{row['shrunk_causal_kfit']:>13.4f}{row['ridge_d1']:>10.4f}")
print(f"\nmonths where persistence nMSE > 1.0 (worse than the in-month constant): "
      f"{nb_above} of {tot}")
print(f"causally fitted shrinkage k over the scored span: median "
      f"{np.nanmedian(kser[m0]):.3f}  p10 {np.nanpercentile(kser[m0],10):.3f}  "
      f"p90 {np.nanpercentile(kser[m0],90):.3f}")

# ------------------------------------------------------------ emit predictions
# Same contract as baselines_v2.py so score_v2.py (and any new baseline arm) can
# join these onto the identical rows: preds_<name>.csv with seg_id, seg_row,
# pred_raw keyed by the TARGET row.  Written to <prep>/baselines_v3/.
BOUT = os.path.join(PREP, "baselines_v3")
os.makedirs(BOUT, exist_ok=True)
EMIT = (["persistence", "climatology_causal_expanding", "blend_pers_roll7",
         "shrunk_causal_kfit"]
        + [f"rolling_mean_d{k}" for k in ROLL_DAYS]
        + [f"ridge_d{k}" for k in RETRAIN_DAYS]
        + [f"shrunk_causal_k{k:.1f}" for k in (0.5, 0.6, 0.7)])
for a in EMIT:
    v = pred[a]
    k = np.isfinite(v)
    pd.DataFrame({"seg_id": seg[k], "seg_row": srow[k], "pred_raw": v[k]}).to_csv(
        os.path.join(BOUT, f"preds_{a}.csv"), index=False)
print(f"wrote {len(EMIT)} preds_*.csv to {BOUT}")

# machine-readable gate, on the primary row set
prim = results.get("common|wu0", {})
gate = {"prep": PREP, "H": H, "row_set": "common|wu0",
        "n_rows": prim.get("_n"), "n_windows": prim.get("_n_windows"),
        "tiers": {}}
if prim:
    best_cheap = min((a for a in prim if a not in ("_n", "_n_windows")
                      and not a.startswith("ridge_") and not a.startswith("ORACLE_")),
                     key=lambda a: prim[a]["nmse"])
    best_any = min((a for a in prim if a not in ("_n", "_n_windows")
                    and not a.startswith("ORACLE_")), key=lambda a: prim[a]["nmse"])
    gate["tiers"] = {
        "tier1_floor_persistence": {"arm": "persistence",
                                    "nmse": prim["persistence"]["nmse"]},
        "tier2_best_nonlearned": {"arm": best_cheap, "nmse": prim[best_cheap]["nmse"]},
        "tier3_best_any_baseline": {"arm": best_any, "nmse": prim[best_any]["nmse"]}}
    print("\nGATE (primary row set common|wu0):")
    for k, v in gate["tiers"].items():
        print(f"  {k:<28} {v['arm']:<28} {v['nmse']:.4f}")
json.dump(gate, open(os.path.join(PREP, "gate_v3.json"), "w"), indent=2, default=float)

json.dump({"prep": PREP, "meta": {k: meta[k] for k in
                                  ("H", "L", "T", "V", "max_generation", "n_windows",
                                   "mask", "span", "t_mode", "n_emitted_rows")},
           "results": results, "permonth": permonth,
           "k_causal_median": float(np.nanmedian(kser[m0]))},
          open(OUTJ, "w"), indent=2, default=float)
print(f"\nwrote {OUTJ}")
