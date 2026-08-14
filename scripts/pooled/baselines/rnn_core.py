#!/usr/bin/env python3
"""Shared pooled LSTM/GRU core for baselines 4 (online) and 5 (periodic).

One small recurrent net for all 50 stocks -- no stock identity, no embedding,
same feature vector every other baseline sees.  Deliberately tiny (hidden
8-32): the comparison target is an evolved ONE-NAS RNN of similar capacity,
not a SOTA forecaster.

Causal input caching
--------------------
Feature row t is standardised with the running moments available at the close
of day t and then frozen in a cache.  A row's representation therefore never
changes retroactively, the whole history is a single dense array, and building
a minibatch is one fancy-index -- while remaining strictly causal.

Targets are standardised the same way (pooled running mean/std of the target
column), so the net learns in O(1) units; predictions are mapped back to return
units before scoring, which is monotone and so leaves rank IC untouched while
keeping the long/short book's sign test meaningful.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except Exception:                                    # pragma: no cover
    torch = None
    nn = object
    HAVE_TORCH = False

from panel import RunningStandardizer

TORCH_HELP = (
    "PyTorch is required for the LSTM/GRU baselines.  It is present in this "
    "repo's local environment (torch 2.8.0).  On Anvil the default python3 has "
    "no torch: `module load learning/conda-2021.05-py38-cpu` (or an anaconda/* "
    "module) first, or run these baselines on a laptop -- they take minutes."
)


def require_torch():
    if not HAVE_TORCH:
        raise SystemExit(TORCH_HELP)


class PooledRNN(nn.Module if HAVE_TORCH else object):
    """LSTM or GRU -> last hidden state -> Linear(h, 1)."""

    def __init__(self, n_feats, hidden=16, cell="lstm"):
        require_torch()
        super().__init__()
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(n_feats, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):                # x: [B, T, F]
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class CausalCache:
    """Rolling standardised copies of the features and the target.

    observe(t) must be called exactly once per panel row, in order, at the
    point where row t is public.
    """

    def __init__(self, panel):
        self.panel = panel
        self.Z = np.zeros_like(panel.X)              # standardised features
        self.Ys = np.zeros_like(panel.Y)             # standardised target
        self.fstd = RunningStandardizer(panel.n_feats)
        self.ystd = RunningStandardizer(1)
        self.filled = -1

    def observe(self, t):
        while self.filled < t:
            self.filled += 1
            r = self.filled
            self.fstd.observe(self.panel.X[r])
            self.Z[r] = self.fstd.transform(self.panel.X[r])
            y = self.panel.Y[r].reshape(-1, 1)
            self.ystd.observe(y)
            self.Ys[r] = self.ystd.transform(y).ravel()

    def y_scale(self):
        """(mean, sd) currently used to map normalised output back to returns."""
        if self.ystd.n < 2:
            return 0.0, 1.0
        sd = float(np.sqrt(self.ystd.m2[0] / (self.ystd.n - 1)))
        return float(self.ystd.mean[0]), sd if sd > 1e-12 else 1.0


def make_sequences(Z, target_rows, stocks, seq_len):
    """[B, seq_len, F] batch ending at row-1 for each (target_row, stock)."""
    target_rows = np.asarray(target_rows)
    stocks = np.asarray(stocks)
    offs = np.arange(-seq_len, 0)                    # rows r-seq_len .. r-1
    idx = target_rows[:, None] + offs[None, :]       # [B, T]
    return Z[idx, stocks[:, None], :]                # [B, T, F]


def new_model(panel, cfg, seed):
    require_torch()
    torch.manual_seed(seed)
    m = PooledRNN(panel.n_feats, hidden=int(cfg["hidden"]),
                  cell=cfg.get("cell", "lstm"))
    opt = torch.optim.Adam(m.parameters(), lr=float(cfg["lr"]))
    return m, opt


def train_steps(model, opt, cache, pairs, cfg, rng, n_steps, batch_size):
    """`n_steps` Adam steps on minibatches drawn from `pairs`.

    pairs: (target_rows, stocks) index arrays of usable training examples.
    """
    rows, stocks = pairs
    if len(rows) == 0 or n_steps <= 0:
        return
    seq_len = int(cfg["seq_len"])
    cs = bool(cfg.get("cs_demean_y", False))
    Ys = cache.Ys
    model.train()
    for _ in range(n_steps):
        take = rng.integers(0, len(rows), size=min(batch_size, len(rows)))
        r, s = rows[take], stocks[take]
        x = torch.from_numpy(
            np.ascontiguousarray(make_sequences(cache.Z, r, s, seq_len),
                                 dtype=np.float32))
        y = Ys[r, s]
        if cs:
            y = y - Ys[r].mean(axis=1)               # same-day cross-section
        yt = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
        opt.zero_grad()
        loss = ((model(x) - yt) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


def predict_all(model, cache, row, seq_len, n_stocks):
    """Forecast of every stock's target at panel row `row` (return units)."""
    stocks = np.arange(n_stocks)
    rows = np.full(n_stocks, row)
    x = torch.from_numpy(
        np.ascontiguousarray(make_sequences(cache.Z, rows, stocks, seq_len),
                             dtype=np.float32))
    model.eval()
    with torch.no_grad():
        out = model(x).numpy().astype(np.float64)
    mu, sd = cache.y_scale()
    return out * sd + mu
