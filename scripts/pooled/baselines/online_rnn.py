#!/usr/bin/env python3
"""Baseline 4 -- pooled online LSTM / GRU with daily incremental training.

Fixed architecture (one recurrent layer, hidden 8-32, linear head), one model
shared by all 50 stocks, no stock identity anywhere.  Input is a length-T
window of the same standardised feature vector every other baseline consumes.

Prequential loop, identical to every other baseline:

    predict(r)  forward pass on feature rows r-T .. r-1  -> forecast of Y[r]
    observe(r)  Y[r] becomes public; take `steps_per_day` Adam minibatch steps
                drawn uniformly from the trailing `replay_days` days of
                (sequence, label) pairs -- 50 stocks x replay_days examples

The optimiser state persists across days, so this is genuine incremental
training rather than a fresh fit each day.  The net is deliberately tiny: the
comparison is against evolved ONE-NAS RNNs of similar capacity, not a SOTA
forecaster, and a 32-unit LSTM already has far more parameters than the
evolved networks it is being asked to beat.

Tuning (staged coordinate search on 2016-2019 only, see protocol.tune_staged):
  stage 1  hidden x lr
  stage 2  steps_per_day x replay_days
  stage 3  seq_len x cs_demean_y
A full cross product would be 100+ configs of ~1 min each; the staged search
gets the same knobs covered in ~20 runs.  Nothing after 2019-12-31 is touched.

    python3 online_rnn.py --panel .../set1_clean --cell lstm --out-dir out/lstm
    python3 online_rnn.py --panel .../set1_clean --cell gru  --out-dir out/gru
"""

import argparse

import numpy as np

import protocol
import rnn_core
from protocol import OnlineModel


class OnlineRNN(OnlineModel):
    def __init__(self, panel, cell="lstm", hidden=16, lr=3e-3, seq_len=10,
                 steps_per_day=3, replay_days=250, cs_demean_y=False,
                 batch_size=128, seed=42):
        rnn_core.require_torch()
        self.panel = panel
        self.cfg = {"cell": cell, "hidden": hidden, "lr": lr,
                    "seq_len": int(seq_len), "cs_demean_y": cs_demean_y}
        self.seq_len = int(seq_len)
        self.steps_per_day = int(steps_per_day)
        self.replay_days = int(replay_days)
        self.batch_size = int(batch_size)
        self.cache = rnn_core.CausalCache(panel)
        self.model, self.opt = rnn_core.new_model(panel, self.cfg, seed)
        self.rng = np.random.default_rng(seed)
        self.min_row = self.seq_len + 1
        self._stocks = np.arange(panel.n_stocks)

    def predict(self, r):
        self.cache.observe(r - 1)               # feature rows <= r-1 are public
        if r < self.min_row:
            return np.zeros(self.panel.n_stocks)
        return rnn_core.predict_all(self.model, self.cache, r, self.seq_len,
                                    self.panel.n_stocks)

    def observe(self, r):
        self.cache.observe(r)                   # Y[r] is now public
        lo = max(self.min_row, r - self.replay_days + 1)
        if lo > r:
            return
        rows = np.arange(lo, r + 1)
        tr = np.repeat(rows, self.panel.n_stocks)
        ts = np.tile(self._stocks, len(rows))
        rnn_core.train_steps(self.model, self.opt, self.cache, (tr, ts),
                             self.cfg, self.rng, self.steps_per_day,
                             self.batch_size)


STAGES = [
    {"hidden": [8, 16, 32], "lr": [1e-3, 3e-3, 1e-2]},
    {"steps_per_day": [1, 3, 8], "replay_days": [60, 250, 1000]},
    {"seq_len": [5, 10, 20], "cs_demean_y": [False, True]},
]
DEFAULT = {"hidden": 16, "lr": 3e-3, "seq_len": 10, "steps_per_day": 3,
           "replay_days": 250, "cs_demean_y": False}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    protocol.add_common_args(ap)
    ap.add_argument("--cell", default="lstm", choices=["lstm", "gru"])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--threads", type=int, default=1,
                    help="torch intra-op threads (1 keeps runs reproducible)")
    ap.set_defaults(warmup_from="2010-01-01")
    args = ap.parse_args()
    rnn_core.require_torch()
    rnn_core.torch.set_num_threads(max(1, args.threads))

    panel = protocol.load_panel(args)

    def factory(cfg):
        return OnlineRNN(panel, cell=args.cell, batch_size=args.batch_size,
                         seed=args.seed, **cfg)

    protocol.finalize(panel, factory, STAGES, args, f"online-{args.cell}",
                      default_cfg=DEFAULT,
                      extra_meta={"cell": args.cell,
                                  "batch_size": args.batch_size,
                                  "warmup_from": args.warmup_from})


if __name__ == "__main__":
    main()
