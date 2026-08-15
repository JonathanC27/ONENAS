#!/usr/bin/env python3
"""Control 2 -- ensemble of 8 IDENTICAL independently-initialised fixed RNNs.

One fixed architecture -- the best config from a small grid tuned on
2016-2019 ONLY (cell x hidden x lr, 8 configs, objective = mean daily rank
IC, protocol.py's tuning convention) -- instantiated 8 times with different
init/sampling seeds, each trained online under the same weekly cadence as the
evolution arm, combined with the same rank-mean rule.

This isolates "ensembling generic RNNs" from "ensembling evolution's island
champions": Control 1 ensembles DIVERSE random architectures; this arm
ensembles seed-diversity only.  If the evolution arm's 8-champion ensemble
only beats its single genome because ensembling helps ANY set of small RNNs,
this control will show the same lift.

The tuning pass uses a FIXED internal seed (7), so the chosen config depends
only on the panel, and --seed varies nothing but the 8 members'
initialisations -- which is the point of the arm.  Pass --config to skip the
grid (run_controls.py does this for the second and later seeds of a panel).

    python3 control2_fixed_ensemble.py --panel /path/to/set1_v2 --seed 42 \
        --out-dir results_controls/control2/set1_v2_seed42
"""

import argparse
import json
import time

import common

GRID = [{"cell": c, "hidden": h, "lr": lr}
        for c in ("rnn", "gru") for h in (8, 16) for lr in (1e-3, 3e-3)]
TUNE_SEED = 7   # fixed: the chosen config must not depend on --seed


def tune(panel, args):
    """Grid search on 2016-2019 with a single online member per config."""
    tune_rows = panel.rows_between(args.tune_from, args.tune_to)
    if not tune_rows:
        raise SystemExit("no panel rows in %s..%s"
                         % (args.tune_from, args.tune_to))
    if panel.dates[tune_rows[-1]] > "2019-12-31":
        raise SystemExit("tuning span extends past 2019-12-31; "
                         "hyperparameters may not see post-2019 data")
    warmup_row = min(panel.first_row_on_or_after(args.warmup_from),
                     tune_rows[0])
    table = []
    if not args.quiet:
        print("# tuning on %s..%s (%d days), %d configs, objective = mean "
              "daily rank IC, tune seed %d"
              % (panel.dates[tune_rows[0]], panel.dates[tune_rows[-1]],
                 len(tune_rows), len(GRID), TUNE_SEED))
    for cfg in GRID:
        t0 = time.time()
        member = common.Member(dict(cfg, rec_density=1.0), panel.n_feats,
                               seed=TUNE_SEED)
        member_preds, champ, _ = common.run_weekly(
            panel, [member], warmup_row, tune_rows, epochs=args.epochs,
            n_windows=args.windows, batch=args.batch,
            half_life=args.half_life, select_days=args.select_days, log=None)
        rows = [r for r in tune_rows if r in champ]
        obj = common.rank_ic_on(panel, member_preds[0], rows)
        table.append({"config": dict(cfg), "val_rank_ic": obj,
                      "seconds": round(time.time() - t0, 2)})
        if not args.quiet:
            print("#   %s  val rank IC %+.5f  (%.1fs)"
                  % (json.dumps(cfg, sort_keys=True), obj, time.time() - t0),
                  flush=True)
    table.sort(key=lambda e: (-e["val_rank_ic"]
                              if e["val_rank_ic"] == e["val_rank_ic"] else 1e9))
    best = dict(table[0]["config"])
    if not args.quiet:
        print("# chosen: %s  (val rank IC %+.5f)"
              % (json.dumps(best, sort_keys=True), table[0]["val_rank_ic"]))
    return best, table


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    common.add_common_args(ap)
    ap.add_argument("--config", default=None,
                    help="JSON dict {cell,hidden,lr}; skips the 2016-2019 "
                         "grid search (use a config tuned by an earlier run "
                         "on the SAME panel)")
    args = ap.parse_args()

    t0 = time.time()
    panel, score_rows, warmup_row = common.setup(args)

    tune_table = None
    if args.config:
        cfg = json.loads(args.config)
    else:
        cfg, tune_table = tune(panel, args)
    tune_seconds = time.time() - t0

    arch = dict(cfg, rec_density=1.0)
    members = [common.Member(arch, panel.n_feats, seed=1000 * args.seed + i)
               for i in range(args.k)]
    if not args.quiet:
        print("# %d identical members, arch %s, seeds %s"
              % (args.k, json.dumps(arch, sort_keys=True),
                 [m.seed for m in members]))

    t1 = time.time()
    log = None if args.quiet else (lambda s: print("# " + s, flush=True))
    member_preds, champion_by_row, trace = common.run_weekly(
        panel, members, warmup_row, score_rows, epochs=args.epochs,
        n_windows=args.windows, batch=args.batch, half_life=args.half_life,
        select_days=args.select_days, log=log)
    fit_seconds = time.time() - t1

    meta = common.base_meta(
        args, panel, "control2_fixed_ensemble", members,
        extra={"hyperparameters": cfg,
               "tuned_on": None if args.config
               else "%s..%s" % (args.tune_from, args.tune_to),
               "tuning_objective": "mean daily rank IC",
               "tuning_seed": None if args.config else TUNE_SEED,
               "tuning_table": tune_table,
               "no_search": "one architecture, tuned pre-2020, instantiated "
                            "%d times with different seeds; never changed"
                            % args.k,
               "tune_seconds": round(tune_seconds, 2),
               "fit_seconds": round(fit_seconds, 2)})
    results = common.score_both(panel, member_preds, champion_by_row,
                                score_rows, args.out_dir, meta, args,
                                quiet=args.quiet)
    common.dump_trace(args.out_dir, trace, meta)
    if not args.quiet:
        print("# control2: tune %.1fs, fit %.1fs, total %.1fs%s"
              % (tune_seconds, fit_seconds, time.time() - t0,
                 ", wrote " + args.out_dir if args.out_dir else ""))
    return results


if __name__ == "__main__":
    main()
