#!/usr/bin/env python3
"""
scored_overlap.py -- do the v3 configs actually score the same rows as v2?

The gate is only transferable between configs if the scored POPULATION is the
same.  This compares the set of scored target TIMESTAMPS (episodes T+V.., j=1..L-1)
across configs, and reports persistence nMSE on the intersection so the mask /
span change can be separated from the row-set change.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

def scored_times(d):
    m = json.load(open(os.path.join(d, "prep_meta.json")))
    L, T, V, G = m["L"], m["T"], m["V"], m["max_generation"]
    idx = pd.read_csv(os.path.join(d, "index.csv"), parse_dates=["time"])
    win = pd.read_csv(os.path.join(d, "windows.csv"))
    off = idx.groupby("seg_id", sort=True).size().cumsum().shift(1).fillna(0).astype(int).to_dict()
    keep = np.zeros(len(idx), bool)
    pos = np.full(len(idx), -1, np.int32)
    for g in range(G):
        e = g + T + V
        w = win.iloc[e]
        b = off[int(w["seg_id"])] + int(w["tgt_start_seg_row"])
        keep[b + 1:b + L] = True
        pos[b + 1:b + L] = np.arange(L - 1)
    t = idx["time"].values[keep]
    y = idx["n2o_raw"].values
    H = m["H"]
    p = np.full(len(idx), np.nan)
    p[H:] = y[:-H]
    return pd.DataFrame({"time": t, "y": y[keep], "pers": p[keep], "pos": pos[keep]}), m


dirs = sys.argv[1:]
base, mb = scored_times(dirs[0])
print(f"reference {os.path.basename(dirs[0])}: {len(base)} scored rows "
      f"{base['time'].min()} .. {base['time'].max()}")


def nmse(y, p):
    return float(np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


for d in dirs[1:]:
    o, mo = scored_times(d)
    s1, s2 = set(base["time"]), set(o["time"])
    inter = s1 & s2
    print(f"\n{os.path.basename(d)}: {len(o)} scored rows")
    print(f"  intersection with reference {len(inter)}  "
          f"({100*len(inter)/len(s1):.2f}% of ref, {100*len(inter)/len(s2):.2f}% of this)")
    print(f"  only in reference {len(s1-s2)}   only in this {len(s2-s1)}")
    ai = base[base["time"].isin(inter)].sort_values("time")
    bi = o[o["time"].isin(inter)].sort_values("time")
    assert np.allclose(ai["y"].values, bi["y"].values), "same timestamp, different target"
    print(f"  persistence nMSE  ref-rows {nmse(base['y'].values, base['pers'].values):.4f}"
          f"   this-rows {nmse(o['y'].values, o['pers'].values):.4f}"
          f"   intersection(ref) {nmse(ai['y'].values, ai['pers'].values):.4f}"
          f"   intersection(this) {nmse(bi['y'].values, bi['pers'].values):.4f}")
    print(f"  in-window position identical on the intersection: "
          f"{bool((ai['pos'].values == bi['pos'].values).all())}")
