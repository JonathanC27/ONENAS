#!/usr/bin/env python3
"""T3 + F1 refresh: factor-adjusted alphas and cumulative net curves, final arms.

T3: daily pooled book return per arm (mean over panels x seeds of the
registered sleeves book, 2020-2024) regressed on FF3 + Momentum daily
factors (Ken French library), Newey-West (HAC) lag 10. Replaces the ad-hoc
2335859 computation with a committed script.

F1: cumulative net curves (arithmetic sum of pooled daily net, $100 base)
for the paper figure, written to paper/net_curves.pdf and the visible
Documents/ONENAS-paper folder.

    tmp/venv/bin/python t3_alphas_f1_curves.py --panels ... --ff-dir ...
"""
import argparse
import os
import sys
import importlib.util

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "baselines"))
sys.path.insert(0, os.path.dirname(HERE))
import score_stream as ss           # noqa: E402
from panel import Panel              # noqa: E402
from rebook import load_preds        # noqa: E402
_spec = importlib.util.spec_from_file_location("sw", "strategy_sweep.py")
_sw = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_sw)

SEEDS = set(range(42, 52))
W0, W1 = "2020-01-01", "2024-12-31"

ARMS = {
    "onenas_40isl": [("probe_ISL40", "ensemble_stitched_predictions.csv")],
    "onenas_8isl": [(d, "ensemble_stitched_predictions.csv")
                    for d in ("onenas_c7e", "probe_s4547", "probe_s4851")],
    "online_lstm": [("results_econ/lstm", "predictions.csv")],
    "online_gru": [("results_econ/gru", "predictions.csv")],
    "periodic_lstm_monthly": [("results_econ/periodic_lstm_monthly",
                               "predictions.csv")],
}


def book_series(panel, preds, rows):
    book = _sw.sleeves_book(panel, preds, rows, 10, 10)
    return {panel.dates[r]: book["daily_ret"][i] for i, r in enumerate(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", nargs="+", required=True)
    ap.add_argument("--ff-dir", required=True)
    args = ap.parse_args()
    panels = {os.path.basename(p.rstrip("/")): Panel(p, "RET_CS")
              for p in args.panels}

    # ---------------- arm daily series (mean across runs, keyed by date)
    def build_series(w0, w1):
        series = {}
        for arm, sources in ARMS.items():
            acc = {}
            n_runs = 0
            for d, fname in sources:
                if not os.path.isdir(d):
                    print(f"# missing {d}", flush=True)
                    continue
                for run in sorted(os.listdir(d)):
                    path = os.path.join(d, run, fname)
                    if "_seed" not in run or not os.path.exists(path):
                        continue
                    pn, seed = run.rsplit("_seed", 1)
                    pn = pn.replace("_core7", "")
                    if int(seed) not in SEEDS or pn not in panels:
                        continue
                    panel = panels[pn]
                    preds, rows = load_preds(path, panel, w0, w1)
                    rows = [r for r in rows if w0 <= panel.dates[r] <= w1]
                    for dt, r in book_series(panel, preds, rows).items():
                        acc.setdefault(dt, []).append(r)
                    n_runs += 1
            series[arm] = pd.Series({d: np.mean(v) for d, v in acc.items()}
                                    ).sort_index()
            print(f"{arm} [{w0}..{w1}]: {n_runs} runs, "
                  f"{len(series[arm])} days", flush=True)

        # equal-weight buy & hold reference (mean of 4 panel B&H books)
        acc = {}
        for pn, panel in panels.items():
            rows = panel.rows_between(max(w0, "2020-01-02"), w1)
            n = panel.n_stocks
            pos = [ss.CAPITAL / n] * n
            frac = ss._cost_frac(panel.prc, panel.tc, max(rows[0] - 1, 0),
                                 None)
            cost0 = sum(abs(v) * frac(k) for k, v in enumerate(pos))
            for i, r in enumerate(rows):
                pnl = 0.0
                for k in range(n):
                    ret = panel.Yscore[r][k]
                    pnl += pos[k] * ret
                    pos[k] *= 1.0 + ret
            # cost charged on the build day only
                c = cost0 if i == 0 else 0.0
                acc.setdefault(panel.dates[r], []).append((pnl - c) / ss.CAPITAL)
        series["ew_buy_hold"] = pd.Series(
            {d: np.mean(v) for d, v in acc.items()}).sort_index()
        return series

    series = build_series(W0, W1)          # 2020-24: factor alphas

    # ---------------- T3: FF3 + Mom, Newey-West lag 10
    ff = pd.read_csv(os.path.join(args.ff_dir,
                                  "F-F_Research_Data_Factors_daily.csv"),
                     skiprows=4)
    ff = ff.rename(columns={ff.columns[0]: "date"})
    ff = ff[pd.to_numeric(ff["date"], errors="coerce").notna()]
    ff["date"] = pd.to_datetime(ff["date"], format="%Y%m%d")
    mom = pd.read_csv(os.path.join(args.ff_dir,
                                   "F-F_Momentum_Factor_daily.csv"),
                      skiprows=13)
    mom = mom.rename(columns={mom.columns[0]: "date"})
    mom = mom[pd.to_numeric(mom["date"], errors="coerce").notna()]
    mom["date"] = pd.to_datetime(mom["date"], format="%Y%m%d")
    mom.columns = [c.strip() for c in mom.columns]
    fac = ff.merge(mom, on="date").set_index("date")
    fac = fac.apply(pd.to_numeric, errors="coerce") / 100.0  # % -> frac

    out_rows = []
    for arm, s in series.items():
        s = s.copy()
        s.index = pd.to_datetime(s.index)
        df = pd.concat([s.rename("ret"), fac], axis=1, join="inner").dropna()
        X = sm.add_constant(df[["Mkt-RF", "SMB", "HML", "Mom"]])
        res = sm.OLS(df["ret"], X).fit(cov_type="HAC",
                                       cov_kwds={"maxlags": 10})
        a = res.params["const"]
        out_rows.append({
            "arm": arm, "n_days": len(df),
            "alpha_bps_day": 1e4 * a,
            "alpha_ann_pct": 100 * 252 * a,
            "nw_t": res.tvalues["const"],
            "mkt_beta": res.params["Mkt-RF"],
        })
        print(f"{arm:<22} alpha {1e4*a:+.2f} bps/d  "
              f"({100*252*a:+.1f}%/yr)  NW t={res.tvalues['const']:+.2f}  "
              f"beta {res.params['Mkt-RF']:+.3f}  n={len(df)}", flush=True)
    pd.DataFrame(out_rows).to_csv("results_econ/factor_alphas_final.csv",
                                  index=False)

    # ---------------- F1: cumulative net curves (2022-24, the paper's
    # evaluation window; books restarted at the window start so endpoints
    # match the 2022-24 table)
    series = build_series("2022-01-01", "2024-12-31")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    COLORS = {"onenas_40isl": "#2a78d6", "onenas_8isl": "#4a3aa7",
              "online_lstm": "#1baf7a", "ew_buy_hold": "#eb6834",
              "periodic_lstm_monthly": "#eda100"}
    LABELS = {"onenas_40isl": "ONE-NAS (40 islands)",
              "onenas_8isl": "ONE-NAS (8 islands)",
              "online_lstm": "Online LSTM",
              "periodic_lstm_monthly": "Periodic LSTM (monthly)",
              "ew_buy_hold": "Equal-weight buy & hold"}
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ends = []
    for arm in ("onenas_40isl", "onenas_8isl", "online_lstm",
                "periodic_lstm_monthly", "ew_buy_hold"):
        s = series[arm]
        s.index = pd.to_datetime(s.index)
        cum = 100 * s.cumsum()
        ax.plot(cum.index, cum.values, color=COLORS[arm], lw=1.6,
                label=LABELS[arm])
        ends.append((arm, cum.index[-1], cum.iloc[-1]))
    # end labels with a minimum vertical separation so converging arms
    # don't collide
    ends.sort(key=lambda e: e[2])
    ys = [e[2] for e in ends]
    yl = ax.get_ylim()
    MIN_SEP = 0.045 * (yl[1] - yl[0])
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < MIN_SEP:
            ys[i] = ys[i - 1] + MIN_SEP
    for (arm, x, v), y in zip(ends, ys):
        ax.annotate(f"{v:+.0f}", (x, y), xytext=(4, 0),
                    textcoords="offset points", color=COLORS[arm],
                    fontsize=8, va="center", annotation_clip=False)
    ax.axhline(0, color="#8b8d92", lw=1)
    ax.set_ylabel("Cumulative net return (%)")
    ax.grid(axis="y", color="#ececea", lw=0.6)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig("../../../paper/net_curves.pdf")
    fig.savefig("/Users/jonathanchang/Documents/ONENAS-paper/net_curves.pdf")
    print("# wrote factor_alphas_final.csv and net_curves.pdf")


if __name__ == "__main__":
    main()
