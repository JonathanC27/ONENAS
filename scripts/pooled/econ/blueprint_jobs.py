#!/usr/bin/env python3
"""Blueprint calculation battery (chair's Jobs A, D, G, B, C, E).

All arithmetic on existing committed series; no new model runs. Emits a
JSON of every number the rebuilt paper needs, plus a readable log.
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
PAN = {s: Panel(f"{TMP}/{s}_core7", "RET_CS") for s in SETS}
OUT = {}

def load_ff():
    fac = {}
    with open(f"{TMP}/F-F_Research_Data_Factors_daily.csv") as fh:
        for line in fh:
            p = line.strip().split(",")
            if len(p) == 5 and p[0].strip().isdigit() and len(p[0].strip()) == 8:
                d = p[0].strip()
                fac[f"{d[:4]}-{d[4:6]}-{d[6:]}"] = [float(x) / 100 for x in p[1:5]]  # MktRF SMB HML RF
    mom = {}
    with open(f"{TMP}/F-F_Momentum_Factor_daily.csv") as fh:
        for line in fh:
            p = line.strip().split(",")
            if len(p) == 2 and p[0].strip().isdigit() and len(p[0].strip()) == 8:
                d = p[0].strip()
                mom[f"{d[:4]}-{d[4:6]}-{d[6:]}"] = float(p[1]) / 100
    return {d: v[:3] + [mom[d], v[3]] for d, v in fac.items() if d in mom}  # Mkt SMB HML Mom RF
FF = load_ff()

def book_daily(run):
    out = {}
    with open(f"probe1/island_champions/{run}/ensemble_book_daily.csv") as fh:
        for r in csv.DictReader(fh):
            out[r["date"]] = float(r["model_ret"])
    return out

def bh_daily(panel):
    rows = panel.rows_between("2020-01-01", "2024-12-31")
    n = panel.n_stocks
    frac = ss._cost_frac(panel.prc, panel.tc, max(rows[0] - 1, 0), None)
    pos = [100.0 / n] * n
    build = sum(abs(v) * frac(k) for k, v in enumerate(pos))
    out = {}
    for i, r in enumerate(rows):
        pnl = 0.0
        for k in range(n):
            ret = panel.Yscore[r][k]; pnl += pos[k] * ret; pos[k] *= 1 + ret
        out[panel.dates[r]] = (pnl - (build if i == 0 else 0)) / 100.0
    return out

BH = {s: bh_daily(PAN[s]) for s in SETS}

def pooled_series(loader_by_set):
    common = sorted(set.intersection(*[set(v) for v in loader_by_set.values()]))
    return common, np.array([[loader_by_set[s][d] for s in SETS] for d in common]).mean(1)

# seed-mean pooled ONE-NAS book + pooled B&H on common dates
per_seed_book = {}
for sd in (42, 43, 44):
    dts, v = pooled_series({s: book_daily(f"{s}_seed{sd}") for s in SETS})
    per_seed_book[sd] = dict(zip(dts, v))
DTS = sorted(set.intersection(*[set(per_seed_book[sd]) for sd in (42,43,44)],
                              *[set(BH[s]) for s in SETS]))
BOOK = np.array([[per_seed_book[sd][d] for sd in (42,43,44)] for d in DTS]).mean(1)
BHP = np.array([[BH[s][d] for s in SETS] for d in DTS]).mean(1)
RF = np.array([FF[d][4] if d in FF else 0.0 for d in DTS])

def eqstats(pnl, dts):
    e = 100 + np.cumsum(pnl) * 100
    pk = np.maximum.accumulate(e)
    mdd = 100 * np.max((pk - e) / pk)
    roe = np.diff(np.concatenate([[100.0], e])) / np.concatenate([[100.0], e[:-1]])
    y22 = 100 * sum(x for x, d in zip(pnl, dts) if d[:4] == "2022")
    sh = roe.mean() / roe.std(ddof=1) * math.sqrt(252)
    t = roe.mean() / (roe.std(ddof=1) / math.sqrt(len(roe)))
    return dict(net=round(float(e[-1] - 100), 1), sharpe=round(float(sh), 2),
                mdd=round(float(mdd), 1), y22=round(float(y22), 1), t=round(float(t), 2))

def nw_t(x, lags=10):
    x = np.asarray(x); n = len(x); m = x.mean(); e = x - m
    s = e @ e
    for l in range(1, lags + 1):
        s += 2 * (1 - l / (lags + 1)) * (e[l:] @ e[:-l])
    return float(m / math.sqrt(s / n / n))

def nw_alpha(y, dts, lags=10):
    X = np.column_stack([np.ones(len(dts))] + [np.array([FF[d][k] for d in dts]) for k in range(4)])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    XtXi = np.linalg.inv(X.T @ X)
    S = (X * e[:, None]).T @ (X * e[:, None])
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        G = (X[l:] * e[l:, None]).T @ (X[:-l] * e[:-l, None])
        S += w * (G + G.T)
    V = XtXi @ S @ XtXi
    return float(beta[0] * 252 * 100), float(beta[0] / math.sqrt(V[0, 0])), float(beta[1])

# ---------------------------------------------------------------- Job A
print("== Job A: borrow/financing sensitivity", flush=True)
def financed(pnl, short_notional, financed_excess, b_bps):
    drag = (b_bps / 1e4) * (short_notional / 100.0) / 252.0
    fin = (RF + 0.005 / 252) * (financed_excess / 100.0)
    return pnl - drag - fin
A = {}
for name, pnl, short, exc in (("book", BOOK, 100, 0),
                              ("overlay", BHP + BOOK, 100, 100),
                              ("regt", 0.75 * BHP + 0.625 * BOOK, 62.5, 37.5),
                              ("pm", 0.5 * BHP + 1.0 * BOOK, 100, 50)):
    A[name] = {"raw": eqstats(pnl, DTS)}
    for b in (25, 50, 100):
        A[name][f"b{b}"] = eqstats(financed(pnl, short, exc, b), DTS)
A["bh"] = eqstats(BHP, DTS)
OUT["A"] = A
for k, v in A.items():
    print(f"  {k}: {v}", flush=True)

# ---------------------------------------------------------------- Job D
print("== Job D: stationary bootstrap dSharpe", flush=True)
rng = np.random.default_rng(7)
def sb_indices(n, mean_block=20, rng=rng):
    idx = np.empty(n, dtype=int); i = 0
    while i < n:
        start = rng.integers(0, n)
        L = rng.geometric(1 / mean_block)
        for j in range(L):
            if i >= n: break
            idx[i] = (start + j) % n; i += 1
    return idx
def sharpe_np(v): return v.mean() / v.std(ddof=1) * math.sqrt(252)
D = {}
for name, pnl in (("overlay", BHP + BOOK), ("regt", 0.75 * BHP + 0.625 * BOOK)):
    ds, dn = [], []
    for _ in range(2000):
        ix = sb_indices(len(DTS))
        ds.append(sharpe_np(pnl[ix]) - sharpe_np(BHP[ix]))
        dn.append(100 * (pnl[ix].sum() - BHP[ix].sum()))
    ds, dn = np.array(ds), np.array(dn)
    D[name] = {"dSharpe_pt": round(float(sharpe_np(pnl) - sharpe_np(BHP)), 3),
               "dSharpe_ci": [round(float(np.percentile(ds, 2.5)), 3), round(float(np.percentile(ds, 97.5)), 3)],
               "dNet_ci": [round(float(np.percentile(dn, 2.5)), 1), round(float(np.percentile(dn, 97.5)), 1)]}
    print(f"  {name}: {D[name]}", flush=True)
OUT["D"] = D

# ---------------------------------------------------------------- Job G
print("== Job G: cold-start age/calendar curves", flush=True)
def diag_ic(path):
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            out[r["date"]] = float(r["ensemble_rank_ic"])
    return out
mature_ic, cold_ic = {}, {}
for s in SETS:
    for sd in (42, 43, 44):
        for store, p in ((mature_ic, f"probe1/island_champions/{s}_seed{sd}/ensemble_diagnostics.csv"),
                         (cold_ic, f"probe_s22/{s}_seed{sd}/ensemble_diagnostics.csv")):
            if os.path.exists(p):
                for d, v in diag_ic(p).items():
                    store.setdefault(d, []).append(v)
common22 = sorted(d for d in mature_ic if d >= "2022-01-01" and d in cold_ic)
mi = np.array([np.mean(mature_ic[d]) for d in common22])
ci = np.array([np.mean(cold_ic[d]) for d in common22])
halves = {}
for label, lo, hi in (("H1_2022", "2022-01-01", "2023-06-30"), ("H2_2023on", "2023-07-01", "2025-01-01")):
    m = [i for i, d in enumerate(common22) if lo <= d < hi]
    halves[label] = {"mature_ic": round(float(mi[m].mean()), 4), "cold_ic": round(float(ci[m].mean()), 4),
                     "gap_t": round(nw_t(mi[m] - ci[m]), 2), "n_days": len(m)}
OUT["G"] = halves
print(f"  {halves}", flush=True)

# ---------------------------------------------------------------- Job B
print("== Job B: long-only active test + turnover", flush=True)
def lo_pooled(pred_path_fmt, seeds):
    per_seed = []
    turn, cost = [], []
    for sd in seeds:
        per_panel = {}
        for s in SETS:
            panel = PAN[s]
            preds, rows = load_preds(pred_path_fmt.format(s=s, sd=sd), panel, "2020-01-01", "2024-12-31")
            dts_, dr = long_only(panel, preds, rows)
            per_panel[s] = dict(zip(dts_, dr))
        c = sorted(set.intersection(*[set(v) for v in per_panel.values()]))
        per_seed.append(dict(zip(c, np.array([[per_panel[s][d] for s in SETS] for d in c]).mean(1))))
    c = sorted(set.intersection(*[set(v) for v in per_seed], set(DTS)))
    return c, np.array([[v[d] for v in per_seed] for d in c]).mean(1)
c, LO = lo_pooled("onenas_c7e/{s}_seed{sd}/ensemble_stitched_predictions.csv", (42, 43, 44))
bhc = np.array([[BH[s][d] for s in SETS] for d in c]).mean(1)
active = LO - bhc
a_capm = nw_alpha(LO, c)
B = {"active_bps_day": round(float(active.mean() * 1e4), 2),
     "active_nw_t": round(nw_t(active), 2),
     "lo_ff_alpha_ann": round(a_capm[0], 1), "lo_ff_alpha_t": round(a_capm[1], 2),
     "lo_beta_mkt": round(a_capm[2], 2)}
OUT["B"] = B
print(f"  {B}", flush=True)

# ---------------------------------------------------------------- Job C
print("== Job C: paired clustered contrasts vs ONE-NAS ensemble", flush=True)
def arm_pooled_daily(arm, seeds):
    per_seed = []
    for sd in seeds:
        per_panel = {}
        for s in SETS:
            p = f"results_econ/{arm}/{s}_core7_seed{sd}/predictions.csv"
            panel = PAN[s]
            preds, rows = load_preds(p, panel, "2020-01-01", "2024-12-31")
            days = scoring.build_days(panel, preds, rows)
            bk = ss.run_book(days, scoring.PRED, panel.prc, panel.tc, 10, None, book="sleeves", hold_days=10)
            per_panel[s] = dict(zip([d[1] for d in days], bk["daily_ret"]))
        cc = sorted(set.intersection(*[set(v) for v in per_panel.values()]))
        per_seed.append(dict(zip(cc, np.array([[per_panel[s][d] for s in SETS] for d in cc]).mean(1))))
    cc = sorted(set.intersection(*[set(v) for v in per_seed]))
    return dict(zip(cc, np.array([[v[d] for v in per_seed] for d in cc]).mean(1)))
CTRL = os.path.join(os.path.dirname(HERE), "controls", "results_controls")
def ctrl_pooled_daily(ctrl):
    per_seed = []
    for sd in (42, 43, 44):
        per_panel = {}
        for s in SETS:
            p = f"{CTRL}/{ctrl}/{s}_core7_seed{sd}/ensemble/predictions.csv"
            panel = PAN[s]
            preds, rows = load_preds(p, panel, "2020-01-01", "2024-12-31")
            days = scoring.build_days(panel, preds, rows)
            bk = ss.run_book(days, scoring.PRED, panel.prc, panel.tc, 10, None, book="sleeves", hold_days=10)
            per_panel[s] = dict(zip([d[1] for d in days], bk["daily_ret"]))
        cc = sorted(set.intersection(*[set(v) for v in per_panel.values()]))
        per_seed.append(dict(zip(cc, np.array([[per_panel[s][d] for s in SETS] for d in cc]).mean(1))))
    cc = sorted(set.intersection(*[set(v) for v in per_seed]))
    return dict(zip(cc, np.array([[v[d] for v in per_seed] for d in cc]).mean(1)))
C = {}
comparators = {
    "random_arch": ctrl_pooled_daily("control1"),
    "fixed_arch": ctrl_pooled_daily("control2"),
    "lstm_ens8": arm_pooled_daily("lstm_ens8", (42,)),
    "gru_ens8": arm_pooled_daily("gru_ens8", (42,)),
    "periodic_lstm_monthly": arm_pooled_daily("periodic_lstm_monthly", (42, 43, 44)),
}
book_map = dict(zip(DTS, BOOK))
for name, comp in comparators.items():
    cc = sorted(set(book_map) & set(comp))
    d = np.array([book_map[x] - comp[x] for x in cc])
    y = np.array([book_map[x] for x in cc]); z = np.array([comp[x] for x in cc])
    da = nw_alpha(d, cc)
    C[name] = {"dNet_pts": round(float(100 * d.sum()), 1), "dMean_nw_t": round(nw_t(d), 2),
               "dSharpe": round(float(sharpe_np(y) - sharpe_np(z)), 3),
               "dAlpha_ann": round(da[0], 1), "dAlpha_t": round(da[1], 2)}
    print(f"  vs {name}: {C[name]}", flush=True)
OUT["C"] = C

# ---------------------------------------------------------------- Job E
print("== Job E: competitor substitution at published points", flush=True)
E = {}
for name, comp in (("lstm_ens8", comparators["lstm_ens8"]), ("str1", arm_pooled_daily("str1", (42,)))):
    cc = sorted(set(comp) & set(dict(zip(DTS, BHP))))
    bhv = np.array([dict(zip(DTS, BHP))[x] for x in cc]); bkv = np.array([comp[x] for x in cc])
    E[name] = {"regt": eqstats(0.75 * bhv + 0.625 * bkv, cc), "pm": eqstats(0.5 * bhv + 1.0 * bkv, cc)}
    print(f"  {name}: {E[name]}", flush=True)
OUT["E"] = E

json.dump(OUT, open("results_econ/blueprint_jobs.json", "w"), indent=1)
print("WROTE results_econ/blueprint_jobs.json")
