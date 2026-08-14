#!/usr/bin/env python3
"""Baseline 3 -- per-stock online ARIMA (ARIMA-OGD / ARIMA-ONS).

Anava, Hazan, Mannor & Shamir (COLT 2013) show that an ARIMA(p,d,q) process is
learnable online by running an AR(p+d+m) predictor on the d-times differenced
series with a no-regret update, and Liu et al. (AAAI 2016) extend the same
scheme.  Daily returns are already a differenced price series, so the natural
instance here is d=0 -- an AR(p) on the return series itself -- but d=1 is kept
in the grid so the "I" of ARIMA is actually exercised and reported.

Per stock k (no pooling -- this is the classical per-series time-series arm):

    z      = d-times differenced return series
    x_k    = [1, z[r-1], ..., z[r-p]] / sigma_k     (sigma_k = causal running
                                                     std of z for that stock)
    zhat   = w_k . x_k * sigma_k
    pred   = zhat undifferenced back to return units

Updates on the squared loss once z[r] is public:

    ogd   w <- Pi_D( w - eta * grad )                     Online Gradient Descent
    ons   A <- A + grad grad^T ;
          w <- Pi_D( w - (1/gamma) A^-1 grad )            Online Newton Step
                                                          (rank-1 Sherman-Morrison)

Both are projected onto an L2 ball of radius D so a single outlier day cannot
blow up the weights.  Everything is vectorised across the 50 stocks.

This model is *nested inside* baseline 2 (online_ridge.py) when lags >= p,
since RET is one of the pooled feature columns -- the difference is that here
each stock carries its own coefficients and there is no cross-sectional
information.  It is reported separately because "online ARIMA" is the standard
comparator a forecasting paper is expected to show.

    python3 online_ar.py --panel .../set1_clean --out-dir out/online_ar
"""

import argparse

import numpy as np

import protocol
from protocol import OnlineModel

FALLBACK_SIGMA = 0.02
MIN_SIGMA_OBS = 20


class OnlineAR(OnlineModel):
    def __init__(self, panel, p=3, eta=1e-2, method="ogd", diff=0,
                 radius=5.0, eps=1e-3):
        self.panel = panel
        self.p = int(p)
        self.eta = float(eta)
        self.method = method
        self.diff = int(diff)
        self.radius = float(radius)
        n, dim = panel.n_stocks, self.p + 1
        self.W = np.zeros((n, dim))
        self.min_row = self.p + self.diff + 1
        # Welford moments of the differenced series, per stock
        self.cnt = 0
        self.mu = np.zeros(n)
        self.m2 = np.zeros(n)
        self._obs_row = -1
        self._pending = None
        if method == "ons":
            self.Ainv = np.repeat((np.eye(dim) / eps)[None], n, axis=0)
        else:
            self.Ainv = None

    # ------------------------------------------------------------ series ops
    def _z(self, r):
        """Differenced target at panel row r ([n_stocks] array)."""
        Y = self.panel.Y
        if self.diff == 0:
            return Y[r]
        return Y[r] - Y[r - 1]

    def _sigma(self):
        if self.cnt < MIN_SIGMA_OBS:
            return np.full(self.panel.n_stocks, FALLBACK_SIGMA)
        sd = np.sqrt(np.maximum(self.m2 / (self.cnt - 1), 0.0))
        return np.where(sd > 1e-10, sd, FALLBACK_SIGMA)

    def _observe_sigma(self, r):
        """Fold z[r] into the running moments (causal: r must be public)."""
        while self._obs_row < r:
            self._obs_row += 1
            if self._obs_row < self.diff:
                continue
            z = self._z(self._obs_row)
            self.cnt += 1
            d = z - self.mu
            self.mu += d / self.cnt
            self.m2 += d * (z - self.mu)

    def _design(self, r):
        """[n_stocks, p+1] normalised AR regressors from rows <= r-1."""
        sig = self._sigma()
        lags = [self._z(r - l) / sig for l in range(1, self.p + 1)]
        return np.column_stack([np.ones(self.panel.n_stocks)] + lags), sig

    # ------------------------------------------------------------- interface
    def predict(self, r):
        self._observe_sigma(r - 1)
        X, sig = self._design(r)
        self._pending = (r, X, sig)
        zhat = (self.W * X).sum(axis=1) * sig
        if self.diff == 1:
            return self.panel.Y[r - 1] + zhat
        return zhat

    def observe(self, r):
        if self._pending is None or self._pending[0] != r:
            self._observe_sigma(r - 1)
            X, sig = self._design(r)
        else:
            _, X, sig = self._pending
        z = self._z(r) / sig                       # normalised label
        err = (self.W * X).sum(axis=1) - z         # normalised residual
        grad = 2.0 * err[:, None] * X
        if self.method == "ogd":
            self.W -= self.eta * grad
        else:                                       # Online Newton Step
            u = np.einsum("nij,nj->ni", self.Ainv, grad)
            denom = 1.0 + (grad * u).sum(axis=1)
            denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
            self.Ainv -= u[:, :, None] * u[:, None, :] / denom[:, None, None]
            step = np.einsum("nij,nj->ni", self.Ainv, grad)
            self.W -= self.eta * step
        norm = np.linalg.norm(self.W, axis=1, keepdims=True)
        scale = np.minimum(1.0, self.radius / np.maximum(norm, 1e-12))
        self.W *= scale
        self._observe_sigma(r)


GRID = {
    "method": ["ogd", "ons"],
    "p": [1, 2, 3, 5, 10],
    "eta": [1e-4, 1e-3, 1e-2, 1e-1],
    "diff": [0, 1],
}
DEFAULT = {"method": "ogd", "p": 3, "eta": 1e-2, "diff": 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    protocol.add_common_args(ap)
    ap.add_argument("--method", default=None, choices=["ogd", "ons"],
                    help="restrict the grid to one update rule")
    args = ap.parse_args()
    panel = protocol.load_panel(args)
    grid = dict(GRID)
    if args.method:
        grid["method"] = [args.method]

    def factory(cfg):
        return OnlineAR(panel, **cfg)

    name = "online-ar" + (f"-{args.method}" if args.method else "")
    protocol.finalize(panel, factory, grid, args, name, default_cfg=DEFAULT)


if __name__ == "__main__":
    main()
