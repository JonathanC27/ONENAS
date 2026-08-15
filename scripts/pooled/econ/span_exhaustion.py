#!/usr/bin/env python3
"""Span-exhaustion diagnostic: is there ANY nonlinear signal in core7?

The blend test showed the evolved RNN ensemble carries no alpha orthogonal to
the linear span of the seven core features.  This diagnostic asks the prior
question: does core7 contain nonlinear structure at all, for ANY learner?

Design (identical walk-forward protocol for all three models):
  L  ridge on the 7 features                     (the linear span)
  Q  ridge on the quadratic expansion            (7 + squares + pairwise = 42)
  G  gradient-boosted trees on the 7 features    (free-form nonlinearity)

All models refit each Jan 1 on an expanding window from --train-start
(default 2010-01-01), predicting every day of the following year, pooled
across the 50 stocks, features from build_design_bulk (strictly rows <= r-1).
Hyperparameters are chosen once per panel per model on 2016-2019 (training
through 2015+) by mean daily rank IC, then frozen; scoring is 2020-2024.

The statistic that answers the question is the RESIDUAL rank IC: per day,
rank each model's cross-section, OLS-residualise Q's (G's) ranks on L's, and
correlate the residual with the realised ranks.  If neither Q nor G has
residual t >= 2 in >= 3 of 4 panels, core7 is linearly exhausted -- the
negative result is about the data, not about any particular learner.

    python3 span_exhaustion.py --panels /path/set1_core7 ... \
        [--out-csv results_econ/span_exhaustion.csv]

Writes one row per panel x model to --out-csv and a verdict to stdout.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "baselines"))

from panel import Panel, build_design_bulk  # noqa: E402

EXCLUDE = ("RET_CS", "RET_CS_Z", "RET_CS5")

RIDGE_ALPHAS = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
GBT_GRID = [dict(max_depth=d, learning_rate=lr, max_iter=it)
            for d in (2, 3) for lr in (0.05, 0.1) for it in (200, 400)]


def load_panel(path):
    p = Panel(path, "RET_CS")
    keep = [i for i, f in enumerate(p.features) if f not in EXCLUDE]
    p.features = [p.features[i] for i in keep]
    p.X = p.X[:, :, keep]
    assert "RET_CS5" not in p.features
    return p


def rank01(v):
    """Average-free rank transform to [0,1] (ties broken by order; the panel's
    continuous values make exact ties measure-zero)."""
    n = len(v)
    r = np.empty(n)
    r[np.argsort(v, kind="stable")] = np.arange(n, dtype=float)
    return r / (n - 1.0)


def year_rows(panel, y):
    return [r for r, d in enumerate(panel.dates) if d[:4] == str(y)]


def rows_between(panel, lo, hi):
    return [r for r, d in enumerate(panel.dates) if lo <= d <= hi]


def design(panel, rows):
    """[n_rows, n_stocks, 7] raw lag-1 features and [n_rows, n_stocks] target."""
    X = build_design_bulk(panel, rows, lags=1, intercept=False)
    Y = panel.Y[np.asarray(rows)]
    return X, Y


def expand_quad(Z):
    """Standardised [n, 7] -> [n, 42]: features, squares, pairwise products."""
    n, d = Z.shape
    cols = [Z]
    cols.append(Z * Z)
    prods = [Z[:, i:i + 1] * Z[:, j:j + 1]
             for i in range(d) for j in range(i + 1, d)]
    return np.concatenate(cols + prods, axis=1)


class RidgeModel:
    def __init__(self, alpha, quad):
        self.alpha, self.quad = alpha, quad

    def fit(self, X, y):
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-12
        Z = (X - self.mu) / self.sd
        if self.quad:
            Z = expand_quad(Z)
        d = Z.shape[1]
        A = Z.T @ Z + self.alpha * np.eye(d)
        b = Z.T @ (y - y.mean())
        self.ymu = y.mean()
        self.w = np.linalg.solve(A, b)

    def predict(self, X):
        Z = (X - self.mu) / self.sd
        if self.quad:
            Z = expand_quad(Z)
        return Z @ self.w + self.ymu


class GBTModel:
    def __init__(self, **kw):
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.kw = kw

    def fit(self, X, y):
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.m = HistGradientBoostingRegressor(
            random_state=42, early_stopping=False,
            min_samples_leaf=200, **self.kw)
        self.m.fit(X, y)

    def predict(self, X):
        return self.m.predict(X)


def walk_forward(panel, model_factory, years, train_start):
    """Yearly expanding-window refits; returns {row: [n_stocks] predictions}."""
    preds = {}
    for y in years:
        train_rows = rows_between(panel, train_start, "%d-12-31" % (y - 1))
        test_rows = year_rows(panel, y)
        if not train_rows or not test_rows:
            continue
        Xtr, Ytr = design(panel, train_rows)
        Xte, _ = design(panel, test_rows)
        m = model_factory()
        m.fit(Xtr.reshape(-1, Xtr.shape[2]), Ytr.reshape(-1))
        out = m.predict(Xte.reshape(-1, Xte.shape[2])).reshape(len(test_rows), -1)
        for i, r in enumerate(test_rows):
            preds[r] = out[i]
    return preds


def daily_ic(panel, preds):
    out = {}
    for r, p in preds.items():
        a, b = rank01(p), rank01(panel.Y[r])
        out[r] = float(np.corrcoef(a, b)[0, 1])
    return out


def mean_ic(panel, preds):
    ics = list(daily_ic(panel, preds).values())
    return float(np.mean(ics))


def residual_ic(panel, preds_m, preds_l):
    """Per-day rank IC of model m after OLS-residualising its ranks on L's."""
    out = {}
    for r in preds_m:
        if r not in preds_l:
            continue
        rm, rl, ry = rank01(preds_m[r]), rank01(preds_l[r]), rank01(panel.Y[r])
        A = np.stack([np.ones_like(rl), rl], 1)
        beta, *_ = np.linalg.lstsq(A, rm, rcond=None)
        resid = rm - A @ beta
        if resid.std() < 1e-12:
            continue
        out[r] = float(np.corrcoef(resid, ry)[0, 1])
    return out


def mean_se_t(v):
    v = np.asarray(v, dtype=float)
    m = v.mean()
    se = v.std(ddof=1) / np.sqrt(len(v))
    return m, se, m / se


def tune(panel, factories, train_start, tune_years):
    best, best_ic = None, -np.inf
    for name, fac in factories:
        preds = walk_forward(panel, fac, tune_years, train_start)
        ic = mean_ic(panel, preds)
        print("#   tune %-40s IC %+.4f" % (name, ic), flush=True)
        if ic > best_ic:
            best, best_ic = (name, fac), ic
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panels", nargs="+", required=True)
    ap.add_argument("--train-start", default="2010-01-01")
    ap.add_argument("--tune-years", default="2016,2017,2018,2019")
    ap.add_argument("--score-years", default="2020,2021,2022,2023,2024")
    ap.add_argument("--out-csv",
                    default=os.path.join(HERE, "results_econ",
                                         "span_exhaustion.csv"))
    args = ap.parse_args()
    tune_years = [int(y) for y in args.tune_years.split(",")]
    score_years = [int(y) for y in args.score_years.split(",")]

    rows_out = []
    gate_hits = {"Q": 0, "G": 0}
    for path in args.panels:
        name = os.path.basename(path.rstrip("/"))
        panel = load_panel(path)
        print("# %s: features %s" % (name, panel.features), flush=True)

        ridge_l = tune(panel, [("L ridge a=%g" % a,
                                (lambda a=a: RidgeModel(a, quad=False)))
                               for a in RIDGE_ALPHAS],
                       args.train_start, tune_years)
        ridge_q = tune(panel, [("Q quad-ridge a=%g" % a,
                                (lambda a=a: RidgeModel(a, quad=True)))
                               for a in RIDGE_ALPHAS],
                       args.train_start, tune_years)
        gbt = tune(panel, [("G gbt %s" % kw, (lambda kw=kw: GBTModel(**kw)))
                           for kw in GBT_GRID],
                   args.train_start, tune_years)

        preds = {"L": walk_forward(panel, ridge_l[1], score_years,
                                   args.train_start),
                 "Q": walk_forward(panel, ridge_q[1], score_years,
                                   args.train_start),
                 "G": walk_forward(panel, gbt[1], score_years,
                                   args.train_start)}
        ic = {k: mean_ic(panel, v) for k, v in preds.items()}
        rec = {"panel": name,
               "config_L": ridge_l[0], "config_Q": ridge_q[0],
               "config_G": gbt[0],
               "ic_L": ic["L"], "ic_Q": ic["Q"], "ic_G": ic["G"]}
        for m in ("Q", "G"):
            r = residual_ic(panel, preds[m], preds["L"])
            mu, se, t = mean_se_t(list(r.values()))
            rec["resid_ic_%s" % m] = mu
            rec["resid_se_%s" % m] = se
            rec["resid_t_%s" % m] = t
            if t >= 2.0:
                gate_hits[m] += 1
        rows_out.append(rec)
        print("# %s: IC L %+.4f  Q %+.4f  G %+.4f | resid Q %+.4f (t %+0.2f)"
              "  G %+.4f (t %+0.2f)"
              % (name, ic["L"], ic["Q"], ic["G"],
                 rec["resid_ic_Q"], rec["resid_t_Q"],
                 rec["resid_ic_G"], rec["resid_t_G"]), flush=True)

    import csv
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    print("# wrote %s" % args.out_csv)

    n = len(rows_out)
    print()
    for m, label in (("Q", "quadratic-interaction ridge"),
                     ("G", "gradient-boosted trees")):
        print("GATE %s (%s): residual t>=2 in %d/%d panels -> %s"
              % (m, label, gate_hits[m], n,
                 "NONLINEARITY EXISTS" if gate_hits[m] >= 3
                 else "not established"))
    if max(gate_hits.values()) < 3:
        print("VERDICT: core7 is LINEARLY EXHAUSTED under this protocol -- no "
              "tested nonlinear learner adds alpha orthogonal to the plain "
              "linear model.")
    else:
        print("VERDICT: nonlinear structure detected; an exploratory ONE-NAS "
              "ablation targeting it is justified (chair item 16b).")


if __name__ == "__main__":
    main()
