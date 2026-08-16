#!/usr/bin/env python3
"""Long-only application of a cross-sectional signal: sleeve-rotated top-K.

Each day 1/H of capital opens an equal-weight position in the signal's top-K
names, held H days (Jegadeesh-Titman rotation, identical to the long leg of
the sleeves book); costs charged on netted position deltas with the panel's
per-stock TC/PRC data. Fully invested, ~beta 1: the raw-net comparable to
equal-weight buy-and-hold.

    python3 long_only.py --pred-csv <predictions.csv> --panel <panel_dir>
"""
import argparse, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "baselines"))
sys.path.insert(0, os.path.dirname(HERE))
import score_stream as ss
from panel import Panel
from rebook import load_preds

def long_only(panel, preds, rows, H=10, K=10, CAP=100.0):
    daily_ret, sleeves, pos, dts = [], [], np.zeros(panel.n_stocks), []
    per = CAP / (H * K)
    for r in rows:
        frac = ss._cost_frac(panel.prc, panel.tc, max(r - 1, 0), None)
        top = np.argsort(preds[r])[::-1][:K]
        new = np.zeros(panel.n_stocks); new[top] = per
        expired = sleeves.pop(0) if len(sleeves) >= H else np.zeros(panel.n_stocks)
        target = pos - expired + new
        delta = target - pos
        cost = sum(abs(delta[k]) * frac(k) for k in np.nonzero(delta)[0])
        pos = target; sleeves.append(new)
        daily_ret.append((float(np.dot(pos, panel.Yscore[r])) - cost) / CAP)
        dts.append(panel.dates[r])
    return dts, np.array(daily_ret)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-csv", required=True)
    ap.add_argument("--panel", required=True)
    ap.add_argument("--score-from", default="2020-01-01")
    ap.add_argument("--score-to", default="2024-12-31")
    a = ap.parse_args()
    panel = Panel(a.panel, "RET_CS")
    preds, rows = load_preds(a.pred_csv, panel, a.score_from, a.score_to)
    dts, v = long_only(panel, preds, rows)
    e = 100 + np.cumsum(v) * 100; pk = np.maximum.accumulate(e)
    roe = np.diff(np.concatenate([[100.0], e])) / np.concatenate([[100.0], e[:-1]])
    print(f"net {e[-1]-100:+.1f}  Sharpe {roe.mean()/roe.std(ddof=1)*math.sqrt(252):.2f}  "
          f"MDD {100*np.max((pk-e)/pk):.1f}")
