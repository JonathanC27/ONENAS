#!/usr/bin/env python3
"""Baseline 5 -- the "current practice" arm: periodic offline retraining.

A fixed model is retrained *from scratch* on every bar of history available at
a retraining boundary, then frozen and used to predict forward until the next
boundary.  Between boundaries the model never sees a label, which is exactly
what a quarterly-refresh production pipeline does and exactly what an online
learner does not.

    --cadence yearly | quarterly | monthly     (the experimental variable)
    --model   ridge | lstm | gru               (same classes as baselines 2/4)

A boundary falls on the first trading day of each year / quarter / month, plus
a forced retrain on the first predicted day.  Training data at boundary b is
every target row <= b-1 (expanding), or the trailing --lookback-days rows if
that is set, always pooled across all 50 stocks.

Cadence is deliberately NOT tuned: the point is the performance-vs-cadence
curve, so run all three and report them.  Everything else is tuned pre-2020
exactly as the online arms are.

Cost control for the neural variant: each retrain is a fixed budget of
`max_steps` Adam minibatch steps sampled from the training pool rather than a
fixed epoch count, so monthly cadence costs 12x yearly in wall clock rather
than 12x a growing epoch.  The budget itself is tuned pre-2020.

    python3 periodic_retrain.py --panel .../set1_clean --model ridge \
        --cadence quarterly --out-dir out/periodic_ridge_quarterly
"""

import argparse

import numpy as np

import protocol
import rnn_core
from panel import RunningStandardizer, build_design_bulk
from protocol import OnlineModel


def period_key(date, cadence):
    y, m = date[:4], int(date[5:7])
    if cadence == "yearly":
        return y
    if cadence == "quarterly":
        return f"{y}Q{(m - 1) // 3 + 1}"
    if cadence == "monthly":
        return date[:7]
    raise SystemExit(f"unknown cadence {cadence!r}")


class PeriodicRetrain(OnlineModel):
    def __init__(self, panel, kind="ridge", cadence="yearly", lags=1,
                 delta=1e-2, cs_demean=False, cs_demean_y=False,
                 lookback_days=0, hidden=16, lr=3e-3, seq_len=10,
                 max_steps=1500, batch_size=128, seed=42):
        self.panel = panel
        self.kind = kind
        self.cadence = cadence
        self.lags = int(lags)
        self.delta = float(delta)
        self.cs_demean = bool(cs_demean)
        self.cs_demean_y = bool(cs_demean_y)
        self.lookback_days = int(lookback_days)
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.n_retrains = 0
        self.last_key = None
        self._std = None
        self._std_row = -1
        if kind == "ridge":
            self.min_row = max(1, self.lags)
            self.w = None
        else:
            rnn_core.require_torch()
            self.seq_len = int(seq_len)
            self.max_steps = int(max_steps)
            self.rcfg = {"cell": kind, "hidden": int(hidden), "lr": float(lr),
                         "seq_len": self.seq_len, "cs_demean_y": self.cs_demean_y}
            self.cache = rnn_core.CausalCache(panel)
            self.model = None
            self.min_row = self.seq_len + 1
        self._stocks = np.arange(panel.n_stocks)

    # ------------------------------------------------------------- internals
    def _advance_state(self, upto):
        """Make every panel row <= upto public to the feature transforms."""
        if self.kind == "ridge":
            if self._std is None:
                self._std = RunningStandardizer(self.panel.n_feats)
            while self._std_row < upto:
                self._std_row += 1
                self._std.observe(self.panel.X[self._std_row])
        else:
            self.cache.observe(upto)

    def _train_rows(self, r):
        """Target rows usable for training at a boundary landing on row r."""
        lo = self.min_row
        if self.lookback_days > 0:
            lo = max(lo, r - self.lookback_days)
        return np.arange(lo, r)              # strictly before the boundary day

    def _retrain(self, r):
        rows = self._train_rows(r)
        if len(rows) < 30:
            return
        if self.kind == "ridge":
            D = build_design_bulk(self.panel, rows, self.lags, self._std,
                                  self.cs_demean, True)
            d = D.shape[2]
            X = D.reshape(-1, d)
            Y = self.panel.Y[rows]
            if self.cs_demean_y:
                Y = Y - Y.mean(axis=1, keepdims=True)
            y = Y.reshape(-1)
            A = X.T @ X
            b = X.T @ y
            reg = self.delta * X.shape[0]
            try:
                self.w = np.linalg.solve(A + reg * np.eye(d), b)
            except np.linalg.LinAlgError:
                self.w = np.linalg.lstsq(A + reg * np.eye(d), b, rcond=None)[0]
        else:
            seed = self.seed + 1000 * self.n_retrains
            self.model, opt = rnn_core.new_model(self.panel, self.rcfg, seed)
            rng = np.random.default_rng(seed)
            tr = np.repeat(rows, self.panel.n_stocks)
            ts = np.tile(self._stocks, len(rows))
            rnn_core.train_steps(self.model, opt, self.cache, (tr, ts),
                                 self.rcfg, rng, self.max_steps,
                                 self.batch_size)
        self.n_retrains += 1

    # ------------------------------------------------------------- interface
    def predict(self, r):
        self._advance_state(r - 1)
        key = period_key(self.panel.dates[r], self.cadence)
        if key != self.last_key:
            self.last_key = key
            self._retrain(r)
        if self.kind == "ridge":
            if self.w is None:
                return np.zeros(self.panel.n_stocks)
            D = build_design_bulk(self.panel, [r], self.lags, self._std,
                                  self.cs_demean, True)[0]
            return D @ self.w
        if self.model is None:
            return np.zeros(self.panel.n_stocks)
        return rnn_core.predict_all(self.model, self.cache, r, self.seq_len,
                                    self.panel.n_stocks)

    def observe(self, r):
        # labels are stored (they feed the next retrain) but the frozen model
        # is NOT updated -- that is the whole point of this arm
        self._advance_state(r)


RIDGE_GRID = {
    "delta": [1e-3, 1e-2, 1e-1, 1.0],
    "lags": [1, 2, 3],
    "cs_demean": [False, True],
    "cs_demean_y": [False, True],
    "lookback_days": [0, 500, 1250],
}
RIDGE_DEFAULT = {"delta": 1e-2, "lags": 1, "cs_demean": False,
                 "cs_demean_y": False, "lookback_days": 0}
# The spec for this arm is "retrain the SAME fixed LSTM", so the architecture
# is inherited from the tuned online LSTM (--arch-from) rather than re-tuned
# here; only the retraining budget and training-window length are searched.
ARCH_KEYS = ("hidden", "lr", "seq_len", "cs_demean_y")
RNN_STAGES = [
    {"max_steps": [500, 1500, 4000], "lookback_days": [0, 1250]},
]
RNN_DEFAULT = {"hidden": 16, "lr": 3e-3, "seq_len": 10, "max_steps": 1500,
               "lookback_days": 0, "cs_demean_y": False}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    protocol.add_common_args(ap)
    ap.add_argument("--model", default="ridge", choices=["ridge", "lstm", "gru"])
    ap.add_argument("--cadence", default="yearly",
                    choices=["yearly", "quarterly", "monthly"])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--arch-from", default=None,
                    help="meta.json of a tuned online RNN run; its "
                         "hidden/lr/seq_len/cs_demean_y are inherited so this "
                         "arm retrains the SAME architecture")
    # the frozen model carries no state across boundaries, so there is nothing
    # to warm up: start the walk where scoring starts (a retrain is forced on
    # the first predicted day).
    ap.set_defaults(warmup_from="2020-01-01")
    args = ap.parse_args()
    if args.model != "ridge":
        rnn_core.require_torch()
        rnn_core.torch.set_num_threads(max(1, args.threads))

    panel = protocol.load_panel(args)
    grid = RIDGE_GRID if args.model == "ridge" else RNN_STAGES
    default = dict(RIDGE_DEFAULT if args.model == "ridge" else RNN_DEFAULT)
    inherited = None
    if args.arch_from and args.model != "ridge":
        import json as _json
        with open(args.arch_from) as fh:
            hp = _json.load(fh).get("hyperparameters", {})
        inherited = {k: hp[k] for k in ARCH_KEYS if k in hp}
        default.update(inherited)
        if not args.quiet:
            print(f"# architecture inherited from {args.arch_from}: "
                  f"{_json.dumps(inherited, sort_keys=True)}")

    def factory(cfg):
        return PeriodicRetrain(panel, kind=args.model, cadence=args.cadence,
                               batch_size=args.batch_size, seed=args.seed,
                               **cfg)

    name = f"periodic-{args.model}-{args.cadence}"
    protocol.finalize(panel, factory, grid, args, name, default_cfg=default,
                      extra_meta={"cadence": args.cadence,
                                  "retrain_model": args.model,
                                  "inherited_architecture": inherited,
                                  "arch_from": args.arch_from,
                                  "warmup_from": args.warmup_from})


if __name__ == "__main__":
    main()
