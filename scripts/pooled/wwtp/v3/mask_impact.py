#!/usr/bin/env python3
"""
mask_impact.py -- how much does making the flatline detector CAUSAL change?

Compares n2o_valid_centred (v1/v2, non-causal: validity at t depends on data to
t+12h) against n2o_valid_trailing (center=False, same 24h window, same
min_periods, same DEAD_STD).  Reports:
  A. row-level flips, overall and by month, and whether any month flips wholesale
  B. detection latency / persistence at the edges of dead stretches -- the
     concrete cost of a trailing detector
  C. segment + window structure under both masks, for SPAN=v2 and SPAN=full
  D. the pre-span (2022-06-11..09-30) that v2 discarded
"""
import numpy as np
import pandas as pd

ROOT = "/anvil/scratch/x-jchang5/wwtp/v3"
G = pd.read_csv(f"{ROOT}/aved_5min_v3.csv", parse_dates=["time"],
                usecols=["time", "n2o_valid_centred", "n2o_valid_trailing", "n2o_real",
                         "rail_bin", "flat_centred", "flat_trailing", "N2O"])
t = G["time"]
vc = G["n2o_valid_centred"].astype(bool).values
vt = G["n2o_valid_trailing"].astype(bool).values
real = G["n2o_real"].astype(bool).values
rail = G["rail_bin"].astype(bool).values
fc = G["flat_centred"].astype(bool).values
ft = G["flat_trailing"].astype(bool).values
n = len(G)
V2_START = pd.Timestamp("2022-10-01")

print("=" * 78)
print("A. ROW-LEVEL VALIDITY FLIPS")
print("=" * 78)
gain = vt & ~vc          # trailing calls VALID, centred called INVALID
loss = vc & ~vt          # trailing calls INVALID, centred called VALID
print(f"grid rows                              {n}")
print(f"valid under CENTRED  (v1/v2 registered) {vc.sum():>7}  ({100*vc.mean():.2f}%)")
print(f"valid under TRAILING (causal)           {vt.sum():>7}  ({100*vt.mean():.2f}%)")
print(f"  net change                            {vt.sum()-vc.sum():>+7}")
print(f"  rows that FLIP at all                 {(gain|loss).sum():>7}  "
      f"({100*(gain|loss).mean():.3f}% of grid, "
      f"{100*(gain|loss).sum()/vc.sum():.3f}% of centred-valid)")
print(f"  invalid -> valid  (trailing admits)   {gain.sum():>7}")
print(f"  valid -> invalid  (trailing rejects)  {loss.sum():>7}")
print(f"\nflatline component only (within target_real & ~rail):")
base = real & ~rail
print(f"  flat CENTRED  {int((base&fc).sum()):>7}   flat TRAILING {int((base&ft).sum()):>7}")
print(f"  rail-only rows (flat test irrelevant) {int(rail.sum()):>7}")

print("\n--- per month ---")
mon = pd.PeriodIndex(t, freq="M")
rows = []
for m in sorted(set(mon)):
    k = mon == m
    rows.append({"month": str(m), "n": int(k.sum()),
                 "valid_centred": int(vc[k].sum()), "valid_trailing": int(vt[k].sum()),
                 "d": int(vt[k].sum() - vc[k].sum()),
                 "gain": int(gain[k].sum()), "loss": int(loss[k].sum()),
                 "fc": 100 * vc[k].mean(), "ft": 100 * vt[k].mean()})
mdf = pd.DataFrame(rows)
print(mdf.to_string(index=False, float_format=lambda x: f"{x:7.2f}"))
flip_whole = mdf[(mdf["fc"] > 50) != (mdf["ft"] > 50)]
print(f"\nmonths whose validity fraction crosses 50% between masks: "
      f"{len(flip_whole)}  {list(flip_whole['month'])}")
worst = mdf.reindex(mdf["d"].abs().sort_values(ascending=False).index).head(5)
print("largest monthly deltas:")
print(worst.to_string(index=False, float_format=lambda x: f"{x:7.2f}"))

print()
print("=" * 78)
print("B. DETECTION LATENCY AT DEAD-STRETCH EDGES")
print("=" * 78)


def runs(mask):
    e = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))) != 0)
    return e[0::2], e[1::2]


# "truly dead" ground truth independent of either detector: a maximal run where
# the LOCAL (pointwise, causal-free) signal has zero information -- we use the
# union of the two detectors' flags as the stretch definition and then ask, for
# each stretch, when each detector fired relative to the stretch start/end.
fs, fe = runs(fc | ft)
lat_on, lat_off, lens = [], [], []
for s, e in zip(fs, fe):
    if e - s < 12:            # ignore sub-hour specks
        continue
    lens.append(e - s)
    ci = np.flatnonzero(fc[s:e])
    ti = np.flatnonzero(ft[s:e])
    if len(ci) and len(ti):
        lat_on.append((ti[0] - ci[0]) * 5 / 60.0)      # hours trailing fires LATER
        lat_off.append((ti[-1] - ci[-1]) * 5 / 60.0)   # hours trailing stops LATER
lens = np.array(lens); lat_on = np.array(lat_on); lat_off = np.array(lat_off)
print(f"dead stretches (union of detectors, >= 1 h): {len(lens)}")
print(f"  stretch length h: median {np.median(lens)*5/60:.1f}  p90 "
      f"{np.percentile(lens,90)*5/60:.1f}  max {lens.max()*5/60:.1f}")
print(f"  ONSET  latency of trailing vs centred (h): median {np.median(lat_on):+.2f} "
      f"mean {lat_on.mean():+.2f}  p90 {np.percentile(lat_on,90):+.2f}  max {lat_on.max():+.2f}")
print(f"  OFFSET latency of trailing vs centred (h): median {np.median(lat_off):+.2f} "
      f"mean {lat_off.mean():+.2f}  p90 {np.percentile(lat_off,90):+.2f}  max {lat_off.max():+.2f}")
print("  (positive ONSET = trailing flags the dead stretch LATER, i.e. lets dead")
print("   rows through; positive OFFSET = trailing keeps flagging after recovery,")
print("   i.e. discards live rows.  A pure 12 h shift each way is the analytic")
print("   expectation: centred sees +/-12 h, trailing sees -24 h..0.)")

# how bad are the rows trailing lets through?  local variability of admitted rows
n2o = G["N2O"].values
loc_sd = pd.Series(n2o).rolling(72, center=True, min_periods=18).std().values
print(f"\n  local 6h sd of rows trailing ADMITS that centred rejected: "
      f"median {np.nanmedian(loc_sd[gain]):.5f}  (all-valid median "
      f"{np.nanmedian(loc_sd[vc]):.5f})")
print(f"  local 6h sd of rows trailing REJECTS that centred kept    : "
      f"median {np.nanmedian(loc_sd[loss]):.5f}")
print(f"  of the {gain.sum()} admitted rows, {int((loc_sd[gain] < 0.005).sum())} "
      f"have 6h sd < DEAD_STD (i.e. genuinely dead rows the causal mask lets in)")
print(f"  of the {loss.sum()} rejected rows, {int((loc_sd[loss] >= 0.005).sum())} "
      f"have 6h sd >= DEAD_STD (i.e. live rows the causal mask throws away)")

print()
print("=" * 78)
print("C. SEGMENT / WINDOW STRUCTURE  (H=72, L=144, MIN_SEG=216)")
print("=" * 78)
H, L = 72, 144
MIN = L + H
print(f"{'span':<6}{'mask':<10}{'runs':>7}{'valid':>9}{'usable':>8}{'usable_rows':>13}"
      f"{'emitted':>9}{'windows':>9}")
struct = {}
for span, t0 in (("v2", V2_START), ("full", t.iloc[0])):
    sel = (t >= t0).values
    for name, m in (("centred", vc), ("trailing", vt)):
        mm = m[sel]
        s, e = runs(mm)
        ln = e - s
        k = ln >= MIN
        nwin = ((ln[k] - H) // L).sum()
        emit = ((ln[k] - H) // L * L + H).sum()
        struct[(span, name)] = dict(runs=len(ln), valid=int(mm.sum()), usable=int(k.sum()),
                                    usable_rows=int(ln[k].sum()), emitted=int(emit),
                                    windows=int(nwin))
        print(f"{span:<6}{name:<10}{len(ln):>7}{mm.sum():>9}{k.sum():>8}"
              f"{ln[k].sum():>13}{emit:>9}{nwin:>9}")
for span in ("v2", "full"):
    a, b = struct[(span, "centred")], struct[(span, "trailing")]
    print(f"  span={span}: trailing vs centred -> windows {b['windows']-a['windows']:+d} "
          f"({100*(b['windows']-a['windows'])/a['windows']:+.2f}%), emitted rows "
          f"{b['emitted']-a['emitted']:+d} ({100*(b['emitted']-a['emitted'])/a['emitted']:+.2f}%), "
          f"usable segments {b['usable']-a['usable']:+d}")

print()
print("=" * 78)
print("D. THE PRE-SPAN v2 DISCARDED  (2022-06-11 .. 2022-09-30)")
print("=" * 78)
pre = (t < V2_START).values
print(f"source rows before {V2_START.date()}: {pre.sum()}")
for name, m in (("centred", vc), ("trailing", vt)):
    mm = m & pre
    s, e = runs(m[pre])
    ln = e - s
    k = ln >= MIN
    nw = ((ln[k] - H) // L).sum()
    print(f"  {name:<9} valid {mm.sum():>6}   runs {len(ln):>4}   usable(>= {MIN}) {k.sum():>3}"
          f"   rows in usable {ln[k].sum():>6}   windows {nw:>4}")
    if k.sum():
        for s0, e0 in zip(s[k], e[k]):
            print(f"      run {str(t[pre].iloc[s0])} .. {str(t[pre].iloc[e0-1])}  "
                  f"{e0-s0} rows  {(e0-s0-H)//L} windows")
pre_n2o = n2o[pre & vc]
post = (~pre) & vc
print(f"\n  N2O on VALID pre-span rows : median {np.median(pre_n2o):.4f}  "
      f"mean {pre_n2o.mean():.4f}  p0.5 {np.percentile(pre_n2o,0.5):.4f}  "
      f"p99.5 {np.percentile(pre_n2o,99.5):.4f}  sd {pre_n2o.std():.4f}")
print(f"  N2O on VALID post-Oct rows : median {np.median(n2o[post]):.4f}  "
      f"mean {n2o[post].mean():.4f}  p0.5 {np.percentile(n2o[post],0.5):.4f}  "
      f"p99.5 {np.percentile(n2o[post],99.5):.4f}  sd {n2o[post].std():.4f}")
mm = pd.DataFrame({"n2o": n2o, "v": vc}, index=t)
mm = mm[mm["v"]].resample("MS")["n2o"].agg(["count", "median", "mean"])
print("\n  monthly median/mean N2O on centred-valid rows:")
print(mm.to_string(float_format=lambda x: f"{x:9.4f}"))
