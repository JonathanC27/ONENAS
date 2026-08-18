#!/usr/bin/env python3
"""
burnin_impact.py -- what reclaiming the discarded pre-span does to the burn-in
and to the normalisation constants.

For each prep config: the composition of the burn-in prefix (the rows every
normalisation statistic is fitted on, and the rows ONE-NAS trains on before its
clock starts), the resulting target centre/scale, and how well the burn-in
distribution matches the population that will actually be scored.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = "/anvil/scratch/x-jchang5/wwtp/v3"
dirs = sys.argv[1:] or sorted(glob.glob(os.path.join(ROOT, "h72_L144_*")))

for d in dirs:
    m = json.load(open(os.path.join(d, "prep_meta.json")))
    idx = pd.read_csv(os.path.join(d, "index.csv"), parse_dates=["time"])
    cut = m["burn_cut_src_row"]
    burn = idx["src_row"].values <= cut
    y = idx["n2o_raw"].values
    tg = m["stats"]["N2O"]
    print("=" * 92)
    print(f"{os.path.basename(d)}   mask={m['mask']} span={m['span']} t_mode={m['t_mode']} "
          f"T={m['T']} (T0={m['T0']}, n_pre={m['n_pre_windows']})")
    print("=" * 92)
    print(f"burn-in rows {int(burn.sum())}  cut {m['burn_cut_time']}   "
          f"first scored window {m['first_scored_window_time']}")
    print(f"TARGET  center {tg['center']:.6f}   scale {tg['scale']:.6f}   "
          f"(p0.5,p99.5 of burn-in = {tg['center']-0.6*tg['scale']:.4f}, "
          f"{tg['center']+0.6*tg['scale']:.4f})")

    mon = pd.PeriodIndex(idx["time"], freq="M")
    tab = pd.DataFrame({"mon": mon.astype(str), "burn": burn, "y": y})
    b = tab[tab["burn"]].groupby("mon")["y"].agg(["count", "median", "mean", "max"])
    b["pct_of_burnin"] = 100 * b["count"] / b["count"].sum()
    print("\nburn-in composition by month:")
    print(b.to_string(float_format=lambda x: f"{x:9.4f}"))

    exc = b.index.isin(["2023-03", "2023-04", "2023-05"])
    print(f"\n  share of burn-in from the Mar-May 2023 excursion: "
          f"{b['pct_of_burnin'][exc].sum():.1f}%")
    pre = b.index < "2022-10"
    print(f"  share of burn-in from the reclaimed pre-span      : "
          f"{b['pct_of_burnin'][pre].sum():.1f}%")

    # burn-in vs scored-population distribution match
    sc = ~burn
    for nm, k in (("burn-in", burn), ("post-burn-in (scored population)", sc)):
        v = y[k]
        print(f"  {nm:<34} n {k.sum():>6}  median {np.median(v):7.4f}  mean {v.mean():7.4f}"
              f"  p0.5 {np.percentile(v,0.5):7.4f}  p99.5 {np.percentile(v,99.5):7.4f}"
              f"  sd {v.std():6.4f}")
    u = (y - tg["center"]) / tg["scale"]
    print(f"  normalised target on the scored population: median {np.median(u[sc]):+.4f}  "
          f"sd {u[sc].std():.4f}  range [{u[sc].min():+.3f},{u[sc].max():+.3f}]  "
          f"-> uses {100*(u[sc].max()-u[sc].min())/2:.1f}% of the tanh band")
    print(f"  |normalised| > 1 (unreachable through tanh) on scored rows: "
          f"{100*(np.abs(u[sc])>1).mean():.4f}%")
    print()

# --------------------------------------------------------------------------
# Would a narrower robust range fix the resolution loss?  The registered map is
# midpoint/half-width of the burn-in [p0.5,p99.5] -> +/-TARGET_BAND.  The
# burn-in's upper tail is set by an excursion the scored span never repeats, so
# the map spends most of the band on values that never occur.  Diagnostic only:
# these alternatives are NOT applied, they are reported so the choice is evidence-
# based rather than inherited.
print("\n" + "=" * 92)
print("DIAGNOSTIC: sensitivity of the target map to the robust-range choice")
print("=" * 92)
BAND, CLIP = 0.6, 0.95
for d in dirs:
    m = json.load(open(os.path.join(d, "prep_meta.json")))
    idx = pd.read_csv(os.path.join(d, "index.csv"), parse_dates=["time"])
    burn = idx["src_row"].values <= m["burn_cut_src_row"]
    y = idx["n2o_raw"].values
    b, sc = y[burn], y[~burn]
    print(f"\n{os.path.basename(d)}")
    print(f"{'range':<14}{'center':>10}{'scale':>10}{'scored sd':>11}{'band used':>11}"
          f"{'|u|>1 %':>10}{'clip %':>9}")
    for lo_q, hi_q in ((0.5, 99.5), (1, 99), (2.5, 97.5), (5, 95), (10, 90)):
        lo, hi = np.percentile(b, lo_q), np.percentile(b, hi_q)
        c, s = 0.5 * (lo + hi), 0.5 * (hi - lo) / BAND
        u = (sc - c) / s
        print(f"p{lo_q}/p{hi_q:<9}{c:>10.4f}{s:>10.4f}{u.std():>11.4f}"
              f"{100*(u.max()-u.min())/2:>10.1f}%{100*(np.abs(u)>1).mean():>10.4f}"
              f"{100*(np.abs(u)>CLIP).mean():>9.4f}")
