#!/usr/bin/env python3
"""The prequential (test-then-train) protocol shared by every baseline.

One driver, one tuning harness, one CLI surface, so that all five baselines
are compared under bit-identical conditions.

THE PROTOCOL
------------
Panel row r is one trading day.  A forecast for row r is issued at the close of
row r-1 and may use feature rows <= r-1 only.  Walking forward:

    for r in warmup_row .. last_row:
        pred[r] = model.predict(r)     # sees feature rows <= r-1
        model.observe(r)               # Y[r] is now public; model may update

This is exactly the convention score_stream.py uses for ONE-NAS: its stitched
stream is indexed by the *target* date, and its ``naive`` column equals the
previous day's realised return.  So a baseline that returns ``Y[r-1]`` from
``predict(r)`` reproduces score_stream's naive column exactly -- which
``trivial.py --model naive`` asserts.

NO PARAMETER MAY BE TUNED ON POST-2019 DATA
-------------------------------------------
Tuning runs the *same* prequential loop but scores only the pre-2020
validation span (default 2016-01-01 .. 2019-12-31), picking the config with
the highest mean daily rank IC there.  That config is then frozen and the
final run is scored on 2020-01-01 .. 2024-12-31.  Online models keep learning
from post-2019 labels during the final run -- that is the protocol, not
leakage -- but no *hyperparameter* ever sees a post-2019 outcome.
"""

import argparse
import itertools
import json
import os
import time

import numpy as np

from panel import Panel, RunningStandardizer, build_design
import scoring

SCORE_FROM = "2020-01-01"
SCORE_TO = "2024-12-31"
TUNE_FROM = "2016-01-01"
TUNE_TO = "2019-12-31"
WARMUP_FROM = "2005-01-01"   # online learners start here; well before tuning


# ------------------------------------------------------------------ interface

class OnlineModel:
    """Minimal interface the driver needs.

    predict(r) -> ndarray[n_stocks]   forecast of Y[r] from rows <= r-1
    observe(r) -> None                Y[r] just became public
    min_row    -> int                 first row it is able to forecast
    """

    min_row = 1

    def predict(self, r):
        raise NotImplementedError

    def observe(self, r):
        pass


class PooledDesignModel(OnlineModel):
    """Base for models that consume a pooled [n_stocks, d] design matrix.

    Owns the causal standardizer bookkeeping: by the time ``design(r)`` runs,
    the standardizer has observed feature rows 0..r-1 and nothing later.
    """

    def __init__(self, panel, lags=1, cs_demean=False, standardize=True,
                 intercept=True):
        self.panel = panel
        self.lags = lags
        self.cs_demean = cs_demean
        self.intercept = intercept
        self.std = RunningStandardizer(panel.n_feats) if standardize else None
        self._std_row = -1
        self.min_row = max(1, lags)

    def design(self, r):
        if self.std is not None:
            while self._std_row < r - 1:
                self._std_row += 1
                self.std.observe(self.panel.X[self._std_row])
        return build_design(self.panel, r, self.lags, self.std,
                            self.cs_demean, self.intercept)

    @property
    def dim(self):
        d = self.panel.n_feats * self.lags
        return d + 1 if self.intercept else d


# --------------------------------------------------------------------- driver

def walk_forward(model, rows):
    """Run the prequential loop over `rows` (ascending). Returns {row: pred}."""
    preds = {}
    for r in rows:
        preds[r] = np.asarray(model.predict(r), dtype=float)
        model.observe(r)
    return preds


def run_span(panel, model, warmup_row, last_row):
    """Walk from warmup_row to last_row inclusive, respecting model.min_row."""
    start = max(warmup_row, getattr(model, "min_row", 1), 1)
    return walk_forward(model, range(start, last_row + 1))


# --------------------------------------------------------------------- tuning

def grid_dicts(grid):
    """{'a': [1,2], 'b': [3]} -> [{'a':1,'b':3}, {'a':2,'b':3}] (stable order)."""
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def _tuning_rows(panel, args):
    tune_rows = panel.rows_between(args.tune_from, args.tune_to)
    if not tune_rows:
        raise SystemExit(f"no panel rows in {args.tune_from}..{args.tune_to}")
    if panel.dates[tune_rows[-1]] > "2019-12-31":
        raise SystemExit(
            "tuning span extends past 2019-12-31; hyperparameters may not be "
            "selected on post-2019 data"
        )
    return tune_rows


def _warmup_row(panel, args, first_scored_row):
    """Row the walk starts from: --warmup-from, but never after the first
    scored day (baselines whose warmup default sits inside the test span --
    the periodic arm -- must still be tunable on 2016-2019)."""
    return min(panel.first_row_on_or_after(args.warmup_from), first_scored_row)


def _evaluate(panel, factory, cfg, tune_rows, warmup_row):
    model = factory(cfg)
    preds = run_span(panel, model, warmup_row, tune_rows[-1])
    rows = [r for r in tune_rows if r in preds]
    return scoring.rank_ic_mean(panel, preds, rows)


def _key(obj):
    return -obj if obj == obj else 1e9      # NaN sorts last


def tune(panel, factory, grid, args, log=True):
    """Full grid search on the pre-2020 validation span -> (best_cfg, table).

    factory(cfg) -> OnlineModel.  Objective: mean daily rank IC on the
    validation rows, computed with score_stream's spearman.
    """
    tune_rows = _tuning_rows(panel, args)
    warmup_row = _warmup_row(panel, args, tune_rows[0])
    cfgs = grid_dicts(grid)
    table = []
    if log:
        print(f"# tuning on {panel.dates[tune_rows[0]]}..{panel.dates[tune_rows[-1]]} "
              f"({len(tune_rows)} days), {len(cfgs)} configs, "
              f"objective = mean daily rank IC")
    for cfg in cfgs:
        t0 = time.time()
        obj = _evaluate(panel, factory, cfg, tune_rows, warmup_row)
        table.append({"config": cfg, "val_rank_ic": obj,
                      "seconds": round(time.time() - t0, 2)})
        if log:
            print(f"#   {json.dumps(cfg, sort_keys=True)}  val rank IC "
                  f"{obj:+.5f}  ({time.time() - t0:.1f}s)")
    table.sort(key=lambda e: _key(e["val_rank_ic"]))
    best = table[0]["config"]
    if log:
        print(f"# chosen: {json.dumps(best, sort_keys=True)}  "
              f"(val rank IC {table[0]['val_rank_ic']:+.5f})")
    return best, table


def tune_staged(panel, factory, stages, args, default_cfg, log=True):
    """Coordinate search: optimise one group of hyperparameters at a time.

    Each stage is a small grid; the winner of stage i seeds stage i+1.  Used
    where a full cross product would be too expensive (the RNN baselines).
    Still confined entirely to the pre-2020 validation span.
    """
    tune_rows = _tuning_rows(panel, args)
    warmup_row = _warmup_row(panel, args, tune_rows[0])
    cfg = dict(default_cfg)
    table = []
    n = sum(len(grid_dicts(s)) for s in stages)
    if log:
        print(f"# staged tuning on {panel.dates[tune_rows[0]]}.."
              f"{panel.dates[tune_rows[-1]]} ({len(tune_rows)} days), "
              f"{len(stages)} stages / {n} runs, objective = mean daily rank IC")
    for si, stage in enumerate(stages):
        trials = []
        for combo in grid_dicts(stage):
            trial = dict(cfg)
            trial.update(combo)
            t0 = time.time()
            obj = _evaluate(panel, factory, trial, tune_rows, warmup_row)
            rec = {"stage": si, "config": dict(trial), "varied": combo,
                   "val_rank_ic": obj, "seconds": round(time.time() - t0, 2)}
            trials.append(rec)
            table.append(rec)
            if log:
                print(f"#   [stage {si}] {json.dumps(combo, sort_keys=True)}  "
                      f"val rank IC {obj:+.5f}  ({time.time() - t0:.1f}s)")
        trials.sort(key=lambda e: _key(e["val_rank_ic"]))
        cfg = dict(trials[0]["config"])
        if log:
            print(f"#   [stage {si}] best {json.dumps(trials[0]['varied'], sort_keys=True)}"
                  f"  -> running config {json.dumps(cfg, sort_keys=True)}")
    best_obj = min((e["val_rank_ic"] for e in table if e["val_rank_ic"] == e["val_rank_ic"]),
                   key=lambda o: -o, default=float("nan"))
    if log:
        print(f"# chosen: {json.dumps(cfg, sort_keys=True)}  "
              f"(val rank IC {best_obj:+.5f})")
    return cfg, table


# ------------------------------------------------------------------- CLI glue

def add_common_args(ap):
    ap.add_argument("--panel", required=True,
                    help="panel directory (per-stock CSVs + panel_dates.csv)")
    ap.add_argument("--out-dir", default=None,
                    help="write predictions.csv / meta.json / metrics.json here")
    ap.add_argument("--param", default="RET", help="target column")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--score-from", default=SCORE_FROM)
    ap.add_argument("--score-to", default=SCORE_TO)
    ap.add_argument("--tune-from", default=TUNE_FROM)
    ap.add_argument("--tune-to", default=TUNE_TO)
    ap.add_argument("--warmup-from", default=WARMUP_FROM,
                    help="first panel date the online loop runs from")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--cost-bps", type=float, default=None)
    ap.add_argument("--max-horizon", type=int, default=10)
    ap.add_argument("--no-tune", action="store_true",
                    help="skip the grid search and use --config / defaults")
    ap.add_argument("--config", default=None,
                    help="JSON dict of frozen hyperparameters (implies --no-tune)")
    ap.add_argument("--quiet", action="store_true")
    return ap


def finalize(panel, factory, grid, args, model_name, extra_meta=None,
             default_cfg=None):
    """Tune (unless frozen), run the scored span, score, write outputs."""
    t_all = time.time()
    tune_table = None
    staged = isinstance(grid, (list, tuple))
    if args.config:
        cfg = json.loads(args.config)
    elif args.no_tune:
        if default_cfg:
            cfg = dict(default_cfg)
        elif staged:
            raise SystemExit("--no-tune needs a default config or --config")
        else:
            cfg = dict(grid_dicts(grid)[0])
    elif staged:
        cfg, tune_table = tune_staged(panel, factory, grid, args,
                                      default_cfg or {}, log=not args.quiet)
    else:
        cfg, tune_table = tune(panel, factory, grid, args, log=not args.quiet)

    score_rows = panel.rows_between(args.score_from, args.score_to)
    if not score_rows:
        raise SystemExit(f"no panel rows in {args.score_from}..{args.score_to}")
    warmup_row = _warmup_row(panel, args, score_rows[0])
    t0 = time.time()
    model = factory(cfg)
    preds = run_span(panel, model, warmup_row, score_rows[-1])
    fit_seconds = time.time() - t0
    rows = [r for r in score_rows if r in preds]
    dropped = len(score_rows) - len(rows)
    if dropped and not args.quiet:
        print(f"# note: {dropped} scored rows had no prediction (warmup); dropped")

    meta = {"model": model_name,
            "panel": panel.path,
            "panel_name": panel.name(),
            "param": panel.param,
            "features": panel.features,
            "hyperparameters": cfg,
            "tuned_on": None if (args.config or args.no_tune)
                        else f"{args.tune_from}..{args.tune_to}",
            "tuning_objective": "mean daily rank IC",
            "tuning_table": tune_table,
            "warmup_from": args.warmup_from,
            "score_from": args.score_from,
            "score_to": args.score_to,
            "scored_days": len(rows),
            "top_k": args.top_k,
            "cost_bps": args.cost_bps,
            "seed": args.seed,
            "fit_seconds": round(fit_seconds, 2),
            "total_seconds": round(time.time() - t_all, 2)}
    if extra_meta:
        meta.update(extra_meta)

    res = scoring.score_and_write(panel, preds, rows, args.out_dir, meta,
                                  top_k=args.top_k, cost_bps=args.cost_bps,
                                  max_horizon=args.max_horizon,
                                  quiet=args.quiet)
    if not args.quiet:
        print(f"# {model_name}: run {fit_seconds:.1f}s, total "
              f"{time.time() - t_all:.1f}s"
              + (f", wrote {args.out_dir}" if args.out_dir else ""))
    return res, meta


def load_panel(args):
    return Panel(args.panel, args.param)
