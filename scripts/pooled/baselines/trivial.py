#!/usr/bin/env python3
"""Baseline 1 -- the trivial floor: naive persistence, STR1, STR5.

  naive  pred[r] = Y[r-1]                     (score_stream's `naive` column)
  str1   pred[r] = -Y[r-1]                    one-day reversal
  str5   pred[r] = -(prod(1+Y[r-5..r-1]) - 1) five-day reversal (short-term
                                              reversal, cumulative not mean)
  strk   same with --k

These have no hyperparameters, so no tuning step is needed or performed; they
are the honest floor every learned model must clear.  `naive` is scored here
only so a baseline run and a ONE-NAS run print the same reference number --
score_stream computes it from the C++ output and gets the identical series.

    python3 trivial.py --panel .../set1_clean --model str1 --out-dir out/str1
"""

import argparse

import numpy as np

import protocol
from protocol import OnlineModel


class Trivial(OnlineModel):
    def __init__(self, panel, kind, k=5):
        self.panel = panel
        self.kind = kind
        self.k = 1 if kind in ("naive", "str1") else k
        self.min_row = max(1, self.k)

    def predict(self, r):
        Y = self.panel.Y
        if self.kind == "naive":
            return Y[r - 1].copy()
        if self.kind == "str1":
            return -Y[r - 1]
        cum = np.prod(1.0 + Y[r - self.k:r], axis=0) - 1.0
        return -cum


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    protocol.add_common_args(ap)
    ap.add_argument("--model", default="str1",
                    choices=["naive", "str1", "str5", "strk"])
    ap.add_argument("--k", type=int, default=5,
                    help="lookback for --model strk (str5 forces 5)")
    args = ap.parse_args()
    args.no_tune = True     # nothing to tune

    panel = protocol.load_panel(args)
    k = 5 if args.model == "str5" else args.k

    def factory(cfg):
        return Trivial(panel, args.model, k)

    name = f"trivial-{args.model}" + (f"-k{k}" if args.model == "strk" else "")
    res, meta = protocol.finalize(
        panel, factory, {"kind": [args.model]}, args, name,
        extra_meta={"lookback_k": k if args.model in ("str5", "strk") else 1,
                    "hyperparameters_note": "none -- parameter-free baseline"},
        default_cfg={"kind": args.model})

    if args.model == "naive" and not args.quiet \
            and panel.param == panel.score_param:
        # our naive must be bit-identical to score_stream's naive column
        r = res["summary"]["overall"]
        assert abs(r["model"]["rank_ic_1"] - r["naive"]["rank_ic_1"]) < 1e-12, \
            "naive baseline disagrees with score_stream's naive column"
        print("# check: naive baseline == score_stream naive column (exact)")


if __name__ == "__main__":
    main()
