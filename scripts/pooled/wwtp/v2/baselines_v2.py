#!/usr/bin/env python3
"""
baselines_v2.py <prep_dir> [out_dir]

Produces the pre-registered baseline arms as per-row prediction files keyed by
(seg_id, seg_row) of the TARGET row, so score_v2.py can join them onto exactly
the rows ONE-NAS covered.  No baseline ever gets its own row set -- that is the
single most important property here, because v1's ridge baselines masked invalid
rows on both endpoints while ONE-NAS did not, and the two sides were never
actually compared on the same data.

ARMS
  persistence          yhat(t+H) = y(t).  Computed inside score_v2.py from the
                       sidecar (it needs no fitting); emitted here too so the
                       file set is complete and self-describing.
  ridge_dK             ridge on the 14 signals at the issue row + N2O lags
                       {1,3,6,12,24,72} + hour/day-of-week harmonics, refit
                       every K days of VALID series time on all data whose
                       target has already been observed.  K in {1,3,7,30}.
                       K=1 is the "daily-retrained ridge" arm; the best of
                       {3,7,30} is the "best periodic-retrain" arm, chosen on
                       the same rows and with every K reported either way.

NO-LOOKAHEAD.  Rows are walked in emitted order, which is chronological (segments
are emitted chronologically and rows within a segment are contiguous).  A refit
at position tau uses only samples whose TARGET row index is < tau -- i.e. only
forecasts that have already come true.  Feature standardisation is refit from
the same window; nothing is fitted once over the whole series.

ROW ELIGIBILITY.  A row is eligible iff its 72-row (6 h) feature history lies
inside the SAME segment.  This is a property of the row, not of the arm, so it
is applied identically to everything -- score_v2.py intersects arms on finite
values, so ineligible rows drop out of every arm at once.  The count is printed.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

PREP = sys.argv[1]
OUTD = sys.argv[2] if len(sys.argv) > 2 else os.path.join(PREP, "baselines")
os.makedirs(OUTD, exist_ok=True)

meta = json.load(open(os.path.join(PREP, "prep_meta.json")))
H, L = meta["H"], meta["L"]
SIGNALS = meta["signals"]
LAGS = [1, 3, 6, 12, 24, 72]
MAXLAG = max(LAGS)
PER_DAY = 288                      # 5-min bins per day
RETRAIN_DAYS = [1, 3, 7, 30]
ALPHA = 1.0                        # L2 on standardised features

idx = pd.read_csv(os.path.join(PREP, "index.csv"), parse_dates=["time"])
n = len(idx)
seg = idx["seg_id"].values
srow = idx["seg_row"].values
y_raw = idx["n2o_raw"].values
t = idx["time"]

# rebuild the normalised covariates by re-reading the segment CSVs in seg order
blocks = []
for s in meta["segments"]:
    blocks.append(pd.read_csv(s["file"]))
X_sig = pd.concat(blocks, ignore_index=True)
assert len(X_sig) == n, f"segment CSV rows {len(X_sig)} != index rows {n}"

# ---- features at the ISSUE row ------------------------------------------------
feats = [X_sig[c].values.astype(float) for c in SIGNALS]
names = list(SIGNALS)
nn = X_sig["N2O"].values.astype(float)
for lag in LAGS:
    v = np.full(n, np.nan)
    v[lag:] = nn[:-lag]
    v[srow < lag] = np.nan          # would cross the segment start
    feats.append(v)
    names.append(f"N2O_lag{lag}")
hod = t.dt.hour.values + t.dt.minute.values / 60.0
dow = t.dt.dayofweek.values
for p, vals in ((24.0, hod), (7.0, dow)):
    feats.append(np.sin(2 * np.pi * vals / p))
    feats.append(np.cos(2 * np.pi * vals / p))
    names += [f"sin{int(p)}", f"cos{int(p)}"]
X = np.column_stack(feats)

# ---- pair issue row -> target row (H ahead, same segment) ---------------------
tgt = np.arange(n) + H
ok = (tgt < n)
ok[ok] &= seg[tgt[ok]] == seg[np.flatnonzero(ok)]
ok &= np.isfinite(X).all(axis=1)
issue_pos = np.flatnonzero(ok)
tgt_pos = issue_pos + H
print(f"emitted rows {n}; usable (issue,target) pairs {len(issue_pos)} "
      f"({100*len(issue_pos)/n:.1f}%)  -- lost to segment starts/ends and the "
      f"{MAXLAG}-row feature history")

Xs, ys = X[issue_pos], y_raw[tgt_pos]


def ridge_fit(Xtr, ytr):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    Z = (Xtr - mu) / sd
    Z = np.column_stack([Z, np.ones(len(Z))])
    A = Z.T @ Z + ALPHA * np.eye(Z.shape[1])
    A[-1, -1] -= ALPHA                      # do not penalise the intercept
    w = np.linalg.solve(A, Z.T @ ytr)
    return mu, sd, w


def ridge_pred(m, Xte):
    mu, sd, w = m
    Z = (Xte - mu) / sd
    return np.column_stack([Z, np.ones(len(Z))]) @ w


# persistence
pd.DataFrame({"seg_id": seg[tgt_pos], "seg_row": srow[tgt_pos],
              "pred_raw": y_raw[issue_pos]}).to_csv(
    os.path.join(OUTD, "preds_persistence.csv"), index=False)

MIN_TRAIN = 2000
for days in RETRAIN_DAYS:
    K = days * PER_DAY
    pred = np.full(len(issue_pos), np.nan)
    model, nfit = None, 0
    # Refit checkpoints on the global emitted-row clock.  Within a block the model
    # is frozen; at each checkpoint it is refit on every sample whose TARGET has
    # already been observed (target_pos < checkpoint), which is the no-lookahead
    # condition -- a forecast can only train on forecasts that have come true.
    checkpoints = np.arange(0, n + K, K)
    for c0, c1 in zip(checkpoints[:-1], checkpoints[1:]):
        block = np.flatnonzero((tgt_pos >= c0) & (tgt_pos < c1))
        if len(block) == 0:
            continue
        train = np.flatnonzero(tgt_pos < c0)      # target already observed
        if len(train) >= MIN_TRAIN:
            model = ridge_fit(Xs[train], ys[train])
            nfit += 1
        if model is not None:
            pred[block] = ridge_pred(model, Xs[block])
    m = np.isfinite(pred)
    name = f"ridge_d{days}"
    pd.DataFrame({"seg_id": seg[tgt_pos[m]], "seg_row": srow[tgt_pos[m]],
                  "pred_raw": pred[m]}).to_csv(
        os.path.join(OUTD, f"preds_{name}.csv"), index=False)
    den = np.sum((ys[m] - ys[m].mean()) ** 2)
    print(f"  {name:<12} refits {nfit:>4}  rows {m.sum():>7}  "
          f"in-sample-free nMSE {np.sum((ys[m]-pred[m])**2)/den:.4f}  "
          f"(persistence on the same rows "
          f"{np.sum((ys[m]-y_raw[issue_pos[m]])**2)/den:.4f})")

print(f"\nwrote baseline prediction files to {OUTD}")
