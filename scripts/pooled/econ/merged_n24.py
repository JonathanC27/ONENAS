#!/usr/bin/env python3
"""Blind seed-expansion merge (PRIMARY.md amendment 2026-08-16): n=12 -> n=24.

Recomputes every headline statistic at 6 seeds x 4 universes for ONE-NAS and
the stochastic baselines (whose seed 45-47 runs already exist), restates the
every-seed claims, and refreshes the factor alpha on the 6-seed mean series.
"""
import csv, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "baselines"))
sys.path.insert(0, os.path.dirname(HERE))
import scoring, score_stream as ss
from panel import Panel
from rebook import load_preds
from long_only import long_only

TMP = "/Users/jonathanchang/.claude/jobs/fc17658e/tmp"
SETS = ["set1", "set2", "set3", "set4"]
SEEDS = (42, 43, 44, 45, 46, 47)
PAN = {s: Panel(f"{TMP}/{s}_core7", "RET_CS") for s in SETS}
OUT = {}

def sharpe(v): v = np.asarray(v); return v.mean() / v.std(ddof=1) * math.sqrt(252)
def mdd_add(v): eq = np.cumsum(np.asarray(v)); return 100 * np.max(np.maximum.accumulate(eq) - eq)

def onenas_run_series(s, sd):
    d = (f"probe1/island_champions/{s}_seed{sd}" if sd <= 44 else f"probe_s4547/{s}_seed{sd}")
    ret, ic = {}, {}
    with open(f"{d}/ensemble_book_daily.csv") as fh:
        for r in csv.DictReader(fh): ret[r["date"]] = float(r["model_ret"])
    with open(f"{d}/ensemble_diagnostics.csv") as fh:
        for r in csv.DictReader(fh): ic[r["date"]] = float(r["ensemble_rank_ic"])
    return ret, ic

def baseline_run_series(arm, s, sd):
    p = f"results_econ/{arm}/{s}_core7_seed{sd}/predictions.csv"
    if not os.path.exists(p): return None
    panel = PAN[s]
    preds, rows = load_preds(p, panel, "2020-01-01", "2024-12-31")
    days = scoring.build_days(panel, preds, rows)
    bk = ss.run_book(days, scoring.PRED, panel.prc, panel.tc, 10, None,
                     book="sleeves", hold_days=10)
    dts = [d[1] for d in days]
    ics = ss.daily_ics(days, scoring.PRED, ss.spearman)
    return dict(zip(dts, bk["daily_ret"])), dict(zip(dts, [x if x == x else np.nan for x in ics]))

def arm_stats(loader, label):
    nets, ics_run = [], []
    pooled_sh, pooled_md = [], []
    per_seed_pooled = {}
    for sd in SEEDS:
        per_panel = {}
        ok = True
        for s in SETS:
            r = loader(s, sd)
            if r is None: ok = False; break
            per_panel[s] = r
        if not ok: continue
        for s in SETS:
            ret, ic = per_panel[s]
            nets.append(100 * sum(ret.values()))
            ics_run.append(np.nanmean(list(ic.values())))
        common = sorted(set.intersection(*[set(per_panel[s][0]) for s in SETS]))
        pool = np.array([[per_panel[s][0][d] for s in SETS] for d in common]).mean(1)
        per_seed_pooled[sd] = dict(zip(common, pool))
        pooled_sh.append(sharpe(pool)); pooled_md.append(mdd_add(pool))
    nets = np.asarray(nets); n = len(nets)
    res = {"n_runs": n, "n_seeds": len(pooled_sh),
           "ic": round(float(np.mean(ics_run)), 4),
           "net": round(float(nets.mean()), 1),
           "net_se": round(float(nets.std(ddof=1) / math.sqrt(n)), 1),
           "pooled_sharpe": round(float(np.mean(pooled_sh)), 2),
           "pooled_sharpe_se": round(float(np.std(pooled_sh, ddof=1) / math.sqrt(len(pooled_sh))), 2),
           "pooled_mdd": round(float(np.mean(pooled_md)), 1),
           "per_seed_pooled_sharpe": [round(float(x), 2) for x in pooled_sh]}
    print(f"{label:<26} n={n}: IC {res['ic']:+.4f} net {res['net']:+.1f}±{res['net_se']}  "
          f"poolS {res['pooled_sharpe']}±{res['pooled_sharpe_se']}  MDD {res['pooled_mdd']}", flush=True)
    return res, per_seed_pooled

print("== merged n=24 stats", flush=True)
OUT["onenas_ensemble"], onenas_pooled = arm_stats(onenas_run_series, "ONE-NAS ensemble")
for arm in ("lstm", "gru", "periodic_lstm_monthly"):
    OUT[arm], _ = arm_stats(lambda s, sd, a=arm: baseline_run_series(a, s, sd), arm)

# new seeds alone (per-run)
print("== new seeds alone (per-run net / Sharpe / IC)", flush=True)
newruns = {}
for sd in (45, 46, 47):
    for s in SETS:
        ret, ic = onenas_run_series(s, sd)
        v = np.array(list(ret.values()))
        newruns[f"{s}_s{sd}"] = {"net": round(float(100 * v.sum()), 1),
                                 "sharpe": round(float(sharpe(v)), 2),
                                 "ic": round(float(np.nanmean(list(ic.values()))), 4)}
        print(f"  {s} s{sd}: {newruns[f'{s}_s{sd}']}", flush=True)
OUT["new_runs"] = newruns

# factor alpha on 6-seed mean pooled series
def load_ff():
    fac = {}
    with open(f"{TMP}/F-F_Research_Data_Factors_daily.csv") as fh:
        for line in fh:
            p = line.strip().split(",")
            if len(p) == 5 and p[0].strip().isdigit() and len(p[0].strip()) == 8:
                d = p[0].strip(); fac[f"{d[:4]}-{d[4:6]}-{d[6:]}"] = [float(x)/100 for x in p[1:4]]
    with open(f"{TMP}/F-F_Momentum_Factor_daily.csv") as fh:
        for line in fh:
            p = line.strip().split(",")
            if len(p) == 2 and p[0].strip().isdigit() and len(p[0].strip()) == 8:
                d = p[0].strip(); d2 = f"{d[:4]}-{d[4:6]}-{d[6:]}"
                if d2 in fac: fac[d2].append(float(p[1])/100)
    return {d: v for d, v in fac.items() if len(v) == 4}
FF = load_ff()
common = sorted(set.intersection(*[set(v) for v in onenas_pooled.values()]))
mean6 = np.array([[onenas_pooled[sd][d] for sd in onenas_pooled] for d in common]).mean(1)
dts = [d for d in common if d in FF]
y = np.array([dict(zip(common, mean6))[d] for d in dts])
X = np.column_stack([np.ones(len(dts))] + [np.array([FF[d][k] for d in dts]) for k in range(4)])
beta = np.linalg.lstsq(X, y, rcond=None)[0]
e = y - X @ beta; lags = 10
XtXi = np.linalg.inv(X.T @ X)
S = (X * e[:, None]).T @ (X * e[:, None])
for l in range(1, lags + 1):
    w = 1 - l / (lags + 1)
    G = (X[l:] * e[l:, None]).T @ (X[:-l] * e[:-l, None])
    S += w * (G + G.T)
V = XtXi @ S @ XtXi
OUT["alpha_n6"] = {"ann_alpha": round(float(beta[0]*252*100), 1),
                   "nw_t": round(float(beta[0]/math.sqrt(V[0,0])), 2),
                   "beta_mkt": round(float(beta[1]), 2)}
print(f"== 6-seed factor alpha: {OUT['alpha_n6']}", flush=True)

# long-only every-seed restatement (6 seeds)
print("== long-only per-seed pooled net (claim: every seed > +75.5)", flush=True)
lo = {}
for sd in SEEDS:
    per_panel = {}
    for s in SETS:
        d = (f"onenas_c7e/{s}_seed{sd}/ensemble_stitched_predictions.csv" if sd <= 44
             else f"probe_s4547/{s}_seed{sd}/ensemble_stitched_predictions.csv")
        panel = PAN[s]
        preds, rows = load_preds(d, panel, "2020-01-01", "2024-12-31")
        dts_, dr = long_only(panel, preds, rows)
        per_panel[s] = dict(zip(dts_, dr))
    c = sorted(set.intersection(*[set(v) for v in per_panel.values()]))
    pool = np.array([[per_panel[s][d] for s in SETS] for d in c]).mean(1)
    lo[sd] = round(float(100 * pool.sum()), 1)
    print(f"  seed {sd}: long-only pooled net {lo[sd]:+.1f}", flush=True)
OUT["long_only_per_seed"] = lo
OUT["long_only_every_seed_beats_bh"] = all(v > 75.5 for v in lo.values())
OUT["long_only_mean"] = round(float(np.mean(list(lo.values()))), 1)
OUT["long_only_se"] = round(float(np.std(list(lo.values()), ddof=1) / math.sqrt(len(lo))), 1)
print(f"  mean {OUT['long_only_mean']} ± {OUT['long_only_se']}  every-seed: {OUT['long_only_every_seed_beats_bh']}", flush=True)

json.dump(OUT, open("results_econ/merged_n24.json", "w"), indent=1)
print("WROTE results_econ/merged_n24.json")
