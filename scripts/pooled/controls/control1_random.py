#!/usr/bin/env python3
"""Control 1 -- fixed-RANDOM-architecture online pipeline (the no-search control).

K=8 tiny RNN architectures are SAMPLED ONCE at t=0 from a documented
distribution (cell ~ Uniform{vanilla RNN, GRU}, hidden ~ UniformInt[4,16],
recurrent-weight density ~ Uniform{1.0, 0.75, 0.5}; lr fixed and shared) and
then NEVER changed: no birth, no death, no mutation, no re-sampling.  Each
architecture trains online under the evolution arm's cadence -- every 5
trading days, ~10 epochs over recency-weighted sampled 40-day windows -- with
weights and Adam state persisting for the entire 2004-2024 walk.

Two variants come out of one pass:
  single    each week the member with the lowest MSE on the trailing 60 days
            is the champion and predicts the next 5 days (the analogue of the
            evolution arm's global best / elite re-scoring)
  ensemble  the rank-mean of all 8 members' cross-sections (the analogue of
            the evolution arm's 8-island-champion rank ensemble)

If the evolution arm's edge comes from the SEARCH, it must beat this; if it
only comes from the pipeline + target + ensembling, it will not.

    python3 control1_random.py --panel /path/to/set1_v2 --seed 42 \
        --out-dir results_controls/control1/set1_v2_seed42
"""

import argparse
import time

import common


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    common.add_common_args(ap)
    ap.add_argument("--lr", type=float, default=3e-3,
                    help="shared Adam learning rate (fixed, not sampled; "
                         "3e-3 is the baseline suite's untuned RNN default)")
    args = ap.parse_args()

    t0 = time.time()
    panel, score_rows, warmup_row = common.setup(args)

    archs = common.sample_architectures(args.seed, args.k, args.lr)
    members = [common.Member(a, panel.n_feats, seed=1000 * args.seed + i)
               for i, a in enumerate(archs)]
    if not args.quiet:
        for i, m in enumerate(members):
            print("# member %d: %s" % (i, m.describe()))

    log = None if args.quiet else (lambda s: print("# " + s, flush=True))
    member_preds, champion_by_row, trace = common.run_weekly(
        panel, members, warmup_row, score_rows, epochs=args.epochs,
        n_windows=args.windows, batch=args.batch, half_life=args.half_life,
        select_days=args.select_days, log=log)
    fit_seconds = time.time() - t0

    meta = common.base_meta(
        args, panel, "control1_random_arch", members,
        extra={"architecture_sampling": common.sampling_distribution(),
               "lr": args.lr,
               "no_search": "architectures sampled once at t=0, never "
                            "mutated, born or killed; only weights change",
               "fit_seconds": round(fit_seconds, 2)})
    results = common.score_both(panel, member_preds, champion_by_row,
                                score_rows, args.out_dir, meta, args,
                                quiet=args.quiet)
    common.dump_trace(args.out_dir, trace, meta)
    if not args.quiet:
        print("# control1: fit %.1fs, total %.1fs%s"
              % (fit_seconds, time.time() - t0,
                 ", wrote " + args.out_dir if args.out_dir else ""))
    return results


if __name__ == "__main__":
    main()
