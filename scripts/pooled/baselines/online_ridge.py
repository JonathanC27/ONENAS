#!/usr/bin/env python3
"""Baseline 2 -- pooled online ridge / RLS with exponential forgetting.

One linear model shared by all 50 stocks (no stock identity), fed the same
feature vector every other baseline gets.  Each day the newly revealed labels
are folded into the sufficient statistics with a forgetting factor lambda:

    A <- lambda * A + sum_k x_k x_k^T          (d x d)
    b <- lambda * b + sum_k x_k y_k            (d,)
    S <- lambda * S + n_stocks                 effective sample count
    w  = (A + delta * S * I)^-1 b

which is exactly recursive least squares with forgetting, written in the
batched (one block of 50 samples per day) normal-equation form -- numerically
steadier than the sequential Sherman-Morrison recursion and identical in
solution.  delta is scaled by S so the ridge penalty means "per sample"
regardless of lambda.  With lambda = 1 this is plain expanding-window online
ridge.  d is small (7-19), so the daily solve is free.

Because RET is itself one of the feature columns and --lags supplies its
history, this model contains a pooled AR(p) as a special case; baseline 3
(online_ar.py) exists to report the per-stock AR variant explicitly.

Tuned pre-2020 over {lambda, delta, lags, cross-sectional demeaning of X and
of y}.  See protocol.py for the tuning rules.

    python3 online_ridge.py --panel .../set1_clean --out-dir out/ridge
"""

import argparse

import numpy as np

import protocol
from protocol import PooledDesignModel


class OnlineRidge(PooledDesignModel):
    def __init__(self, panel, lam=0.999, delta=1e-2, lags=1, cs_demean=False,
                 cs_demean_y=False):
        super().__init__(panel, lags=lags, cs_demean=cs_demean,
                         standardize=True, intercept=True)
        self.lam = float(lam)
        self.delta = float(delta)
        self.cs_demean_y = bool(cs_demean_y)
        d = self.dim
        self.A = np.zeros((d, d))
        self.b = np.zeros(d)
        self.S = 0.0
        self.w = np.zeros(d)
        self._dirty = False
        self._pending = None     # design matrix used for row r, reused on observe

    def predict(self, r):
        X = self.design(r)
        self._pending = (r, X)
        if self._dirty:
            self._refit()
        return X @ self.w

    def _refit(self):
        d = self.A.shape[0]
        reg = self.delta * max(self.S, 1.0)
        try:
            self.w = np.linalg.solve(self.A + reg * np.eye(d), self.b)
        except np.linalg.LinAlgError:
            self.w = np.linalg.lstsq(self.A + reg * np.eye(d), self.b,
                                     rcond=None)[0]
        self._dirty = False

    def observe(self, r):
        if self._pending is None or self._pending[0] != r:
            X = self.design(r)
        else:
            X = self._pending[1]
        y = self.panel.Y[r]
        if self.cs_demean_y:
            y = y - y.mean()
        self.A = self.lam * self.A + X.T @ X
        self.b = self.lam * self.b + X.T @ y
        self.S = self.lam * self.S + X.shape[0]
        self._dirty = True


GRID = {
    "lam": [1.0, 0.999, 0.995, 0.99, 0.98],
    "delta": [1e-3, 1e-2, 1e-1, 1.0],
    "lags": [1, 2, 3],
    "cs_demean": [False, True],
    "cs_demean_y": [False, True],
}
DEFAULT = {"lam": 0.999, "delta": 1e-2, "lags": 1, "cs_demean": False,
           "cs_demean_y": False}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    protocol.add_common_args(ap)
    ap.add_argument("--fast-grid", action="store_true",
                    help="smaller grid (lambda x delta only, lags=1)")
    args = ap.parse_args()
    panel = protocol.load_panel(args)
    grid = dict(GRID)
    if args.fast_grid:
        grid = {"lam": GRID["lam"], "delta": GRID["delta"], "lags": [1],
                "cs_demean": [False], "cs_demean_y": [False]}

    def factory(cfg):
        return OnlineRidge(panel, **cfg)

    protocol.finalize(panel, factory, grid, args, "online-ridge-rls",
                      default_cfg=DEFAULT)


if __name__ == "__main__":
    main()
