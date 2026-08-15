#!/usr/bin/env python3
"""Shared machinery for the two no-search control arms.

These controls answer one reviewer question: does the EVOLVED ARCHITECTURE
contribute anything beyond (a) the online training pipeline, (b) the
CS-rank-normal target, (c) the data, and (d) generic ensembling?  They mirror
the evolution arm's protocol -- weekly clock, recency-weighted 40-day-window
sampling, ~10 epochs per wake, champion re-selection on recent data, 8-member
rank-mean ensemble -- while never changing an architecture after t=0.

Everything metric-shaped is IMPORTED from ../baselines (scoring.py -> the
parent score_stream.py), so a control's numbers come out of exactly the same
code path as the evolution arm's and the baseline suite's.  Nothing outside
scripts/pooled/controls/ is modified.

THE WEEKLY PREQUENTIAL LOOP (one pass produces both control variants)
---------------------------------------------------------------------
Panel row r is one trading day; a forecast for row r may use feature rows
<= r-1 only (the baselines' convention, identical to score_stream's).

    for wake in warmup_row, warmup_row+5, warmup_row+10, ...:
        # all rows <= wake-1 are public
        for each member: train EPOCHS passes over WINDOWS_PER_WAKE sampled
            40-day (stock, window) examples, recency-weighted with a
            ~2-trading-year half-life; weights and Adam state PERSIST
        champion <- member with the lowest MSE on the trailing SELECT_DAYS
            days (re-scored with current weights, targets all public)
        for r in wake .. wake+4:                      # the next week
            every member issues pred[r] from feature rows r-40 .. r-1
            single-model control  = the champion's pred[r]
            ensemble control      = rank-mean of all K members' pred[r]

No member is ever born, killed, or mutated; the ONLY thing that changes over
time is the weights.  That is the entire difference from the evolution arm.

CAUSALITY: features and targets are standardised by rnn_core.CausalCache
(expanding Welford moments, a row frozen the moment it becomes public);
training windows require target rows <= wake-1; champion selection uses
target rows <= wake-1; a day's forecast reads feature rows <= r-1.  There is
no other data path into the models.
"""

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_POOLED = os.path.dirname(_HERE)
for _p in (_POOLED, os.path.join(_POOLED, "baselines")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rnn_core                    # noqa: E402  (baselines/: CausalCache, torch)
import scoring                     # noqa: E402  (baselines/: score_stream metrics)
import score_stream as ss          # noqa: E402  (ranks() for the combiner)
from panel import Panel            # noqa: E402

rnn_core.require_torch()
import torch                       # noqa: E402
import torch.nn as nn              # noqa: E402

# ------------------------------------------------- protocol constants
# Matched to the evolution arm (see README.md for the fairness table).
STEP_DAYS = 5            # weekly clock: train every 5 trading days
WINDOW_LEN = 40          # evolution trains on 40-day windows
EPOCHS = 10              # backprop epochs per weekly wake (evolution: 10)
WINDOWS_PER_WAKE = 512   # sampled (stock, 40-day window) examples per wake
BATCH = 256              # minibatch size within a wake
HALF_LIFE = 504.0        # recency half-life in trading days (~2 years, PER-like)
SELECT_DAYS = 60         # champion = lowest MSE on the trailing 60 days
K_MODELS = 8             # evolution's ensemble is 8 island champions

SCORE_FROM = "2020-01-01"
SCORE_TO = "2024-12-31"
TUNE_FROM = "2016-01-01"
TUNE_TO = "2019-12-31"
WARMUP_FROM = "2004-01-01"   # evolution trains from 2004+; panels start 2004-08


# ------------------------------------------------------------- architecture

class SeqRNN(nn.Module):
    """Tiny recurrent net trained seq-to-seq over 40-day windows.

    cell in {"rnn" (vanilla tanh), "gru"}; one layer, hidden 4-16; per-step
    linear head.  `rec_density` < 1 freezes a random binary mask over the
    recurrent weight matrix (re-applied after every optimiser step), the
    fixed-topology analogue of the evolved genomes' sparse recurrent
    connectivity.
    """

    CELLS = {"rnn": nn.RNN, "gru": nn.GRU}

    def __init__(self, n_feats, hidden, cell, rec_density=1.0, seed=0):
        super().__init__()
        torch.manual_seed(seed)                     # deterministic init
        self.rnn = self.CELLS[cell](n_feats, int(hidden), num_layers=1,
                                    batch_first=True)
        self.head = nn.Linear(int(hidden), 1)
        self.mask = None
        if rec_density < 1.0:
            g = torch.Generator().manual_seed(seed + 1)
            w = self.rnn.weight_hh_l0
            self.mask = (torch.rand(w.shape, generator=g)
                         < rec_density).to(w.dtype)
            with torch.no_grad():
                w.mul_(self.mask)

    def apply_mask(self):
        if self.mask is not None:
            with torch.no_grad():
                self.rnn.weight_hh_l0.mul_(self.mask)

    def forward(self, x):                           # [B, T, F] -> [B, T]
        out, _ = self.rnn(x)
        return self.head(out).squeeze(-1)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class Member:
    """One fixed architecture with persistent weights, optimiser and sampler."""

    def __init__(self, arch, n_feats, seed):
        self.arch = dict(arch)
        self.seed = int(seed)
        self.net = SeqRNN(n_feats, arch["hidden"], arch["cell"],
                          arch.get("rec_density", 1.0), seed=self.seed)
        self.opt = torch.optim.Adam(self.net.parameters(),
                                    lr=float(arch["lr"]))
        self.rng = np.random.default_rng(self.seed)

    def describe(self):
        d = dict(self.arch)
        d["seed"] = self.seed
        d["n_params"] = self.net.n_params()
        return d


# Control 1's architecture sampling distribution, drawn ONCE at t=0 per run
# seed and then frozen forever:
#   cell        ~ Uniform{vanilla RNN, GRU}
#   hidden      ~ UniformInt[4, 16]
#   rec_density ~ Uniform{1.0, 0.75, 0.5}   (recurrent-weight sparsity mask)
#   lr          fixed (not sampled; one shared value, see --lr)
REC_DENSITIES = (1.0, 0.75, 0.5)
HIDDEN_RANGE = (4, 16)


def sample_architectures(seed, k, lr):
    rng = np.random.default_rng(seed)
    archs = []
    for _ in range(k):
        archs.append({
            "cell": ("rnn", "gru")[int(rng.integers(0, 2))],
            "hidden": int(rng.integers(HIDDEN_RANGE[0], HIDDEN_RANGE[1] + 1)),
            "rec_density": float(REC_DENSITIES[int(rng.integers(
                0, len(REC_DENSITIES)))]),
            "lr": float(lr),
        })
    return archs


def sampling_distribution():
    return {"cell": "Uniform{rnn, gru}",
            "hidden": "UniformInt[%d, %d]" % HIDDEN_RANGE,
            "rec_density": "Uniform{%s} (fixed binary mask on weight_hh)"
                           % ", ".join(str(d) for d in REC_DENSITIES),
            "lr": "fixed (shared, not sampled)"}


# --------------------------------------------------------------- training

def _gather_windows(cache, starts, stocks):
    """Standardised (X, Y) for windows: inputs rows a..a+39, targets a+1..a+40."""
    offs = np.arange(WINDOW_LEN)
    xi = starts[:, None] + offs[None, :]
    X = torch.from_numpy(np.ascontiguousarray(
        cache.Z[xi, stocks[:, None], :], dtype=np.float32))
    Y = torch.from_numpy(np.ascontiguousarray(
        cache.Ys[xi + 1, stocks[:, None]], dtype=np.float32))
    return X, Y


def sample_windows(rng, n_public, n_stocks, n, half_life):
    """Recency-weighted (start_row, stock) pairs.

    Window start a needs targets a+1..a+WINDOW_LEN all public, i.e.
    a <= n_public - WINDOW_LEN - 1.  P(a) ~ 2^(-age/half_life) where age is
    the window end's distance from the newest public row (PER-like
    exponential recency, ~2 trading years to half weight).
    """
    a_max = n_public - WINDOW_LEN - 1
    if a_max < 0:
        return None, None
    ages = np.arange(a_max, -1, -1, dtype=np.float64)   # age of a=0 is a_max
    w = np.exp2(-ages / half_life)
    w /= w.sum()
    starts = rng.choice(a_max + 1, size=n, p=w)
    stocks = rng.integers(0, n_stocks, size=n)
    return starts, stocks


def train_wake(member, cache, wake, epochs, n_windows, batch, half_life):
    """One weekly training event: rows <= wake-1 are public."""
    starts, stocks = sample_windows(member.rng, wake, cache.panel.n_stocks,
                                    n_windows, half_life)
    if starts is None:
        return
    X, Y = _gather_windows(cache, starts, stocks)
    net, opt = member.net, member.opt
    net.train()
    for _ in range(epochs):
        perm = member.rng.permutation(n_windows)
        for lo in range(0, n_windows, batch):
            take = torch.from_numpy(perm[lo:lo + batch])
            opt.zero_grad()
            loss = ((net(X[take]) - Y[take]) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            net.apply_mask()


# ------------------------------------------------------------- inference

def recent_mse(member, cache, wake, select_days):
    """MSE (standardised-target space) on the trailing `select_days` rows.

    Re-scores the member's CURRENT weights on target rows wake-select_days ..
    wake-1, all public at the wake -- the analogue of the evolution arm
    re-scoring elites on the most recent windows before picking the global
    best for the coming week.
    """
    n_stocks = cache.panel.n_stocks
    lo = max(WINDOW_LEN, wake - select_days)
    rows = np.arange(lo, wake)
    if len(rows) == 0:
        return float("inf")
    tr = np.repeat(rows, n_stocks)
    tss = np.tile(np.arange(n_stocks), len(rows))
    X = torch.from_numpy(np.ascontiguousarray(
        rnn_core.make_sequences(cache.Z, tr, tss, WINDOW_LEN),
        dtype=np.float32))
    y = cache.Ys[tr, tss]
    member.net.eval()
    with torch.no_grad():
        out = member.net(X)[:, -1].numpy().astype(np.float64)
    return float(np.mean((out - y) ** 2))


def predict_day(member, cache, r):
    """Every stock's forecast for row r from feature rows r-40..r-1 (return units)."""
    n_stocks = cache.panel.n_stocks
    stocks = np.arange(n_stocks)
    rows = np.full(n_stocks, r)
    X = torch.from_numpy(np.ascontiguousarray(
        rnn_core.make_sequences(cache.Z, rows, stocks, WINDOW_LEN),
        dtype=np.float32))
    member.net.eval()
    with torch.no_grad():
        out = member.net(X)[:, -1].numpy().astype(np.float64)
    mu, sd = cache.y_scale()
    return out * sd + mu


# ------------------------------------------------------------ the main loop

def run_weekly(panel, members, warmup_row, pred_rows, epochs=EPOCHS,
               n_windows=WINDOWS_PER_WAKE, batch=BATCH, half_life=HALF_LIFE,
               select_days=SELECT_DAYS, log=None):
    """The weekly prequential loop of the module docstring.

    pred_rows: ascending panel rows that need predictions (the scored span).
    Predictions are produced from the first wake whose week overlaps
    pred_rows[0]; earlier weeks are pure warm-up (training only -- champion
    identity has no effect on training, so skipping selection there is
    exactly equivalent and much cheaper).

    Returns (member_preds, champion_by_row, champion_trace):
      member_preds     list of {row: ndarray[n_stocks]} per member
      champion_by_row  {row: member index}
      champion_trace   list of per-wake records (date, champion, member MSEs)
    """
    cache = rnn_core.CausalCache(panel)
    first_pred, last = pred_rows[0], pred_rows[-1]
    member_preds = [dict() for _ in members]
    champion_by_row = {}
    trace = []
    t0 = time.time()
    wakes = list(range(warmup_row, last + 1, STEP_DAYS))
    for wi, wake in enumerate(wakes):
        cache.observe(wake - 1)                    # rows <= wake-1 are public
        for m in members:
            train_wake(m, cache, wake, epochs, n_windows, batch, half_life)
        week_last = min(wake + STEP_DAYS - 1, last)
        if week_last < first_pred:
            if log and wi % 100 == 0:
                log("warmup wake %d/%d (%s)  %.0fs"
                    % (wi + 1, len(wakes), panel.dates[wake], time.time() - t0))
            continue
        mses = [recent_mse(m, cache, wake, select_days) for m in members]
        champ = int(np.argmin(mses))
        trace.append({"date": panel.dates[wake], "row": wake,
                      "champion": champ,
                      "member_mse": [round(x, 6) for x in mses]})
        for r in range(wake, week_last + 1):
            cache.observe(r - 1)                   # today's inputs are public
            for i, m in enumerate(members):
                member_preds[i][r] = predict_day(m, cache, r)
            champion_by_row[r] = champ
        if log and wi % 20 == 0:
            log("scored wake %d/%d (%s) champion=m%d  %.0fs"
                % (wi + 1, len(wakes), panel.dates[wake], champ,
                   time.time() - t0))
    return member_preds, champion_by_row, trace


# ------------------------------------------------------------- combination

def rank_mean(cross):
    """score_ensemble.py's rank_mean combiner, reimplemented on ss.ranks:
    each member's cross-section becomes average ranks (ties share the mean
    rank), centred by (n-1)/2, then averaged over members.  Scale-free."""
    n = len(cross[0])
    mid = (n - 1) / 2.0
    acc = np.zeros(n)
    for c in cross:
        acc += np.asarray(ss.ranks(list(np.asarray(c, dtype=float)))) - mid
    return acc / len(cross)


def combine_predictions(member_preds, champion_by_row, rows):
    """(single_preds, ensemble_preds) dicts {row: ndarray} over `rows`."""
    single, ens = {}, {}
    for r in rows:
        cross = [mp[r] for mp in member_preds]
        single[r] = cross[champion_by_row[r]]
        ens[r] = rank_mean(cross)
    return single, ens


# ------------------------------------------------------------------ scoring

def protocol_meta(epochs, n_windows, batch, half_life, select_days):
    return {"step_days": STEP_DAYS, "window_len": WINDOW_LEN,
            "epochs_per_wake": epochs, "windows_per_wake": n_windows,
            "minibatch": batch, "recency_half_life_days": half_life,
            "select_days": select_days,
            "loss": "seq-to-seq MSE on the standardised target over all 40 "
                    "window steps",
            "optimizer": "Adam, grad-norm clip 1.0, state persists across "
                         "wakes",
            "combiner": "rank_mean (score_ensemble.py's definition)"}


def score_both(panel, member_preds, champion_by_row, score_rows, out_dir,
               base_meta, args, quiet=False):
    """Score the single-champion and rank-mean-ensemble variants; write
    <out_dir>/single/ and <out_dir>/ensemble/ with scoring.py's outputs."""
    rows = [r for r in score_rows if r in champion_by_row]
    dropped = len(score_rows) - len(rows)
    if dropped:
        print("# note: %d scored rows had no prediction (warmup); dropped"
              % dropped)
    single, ens = combine_predictions(member_preds, champion_by_row, rows)
    results = {}
    for variant, preds in (("single", single), ("ensemble", ens)):
        meta = dict(base_meta)
        meta["model"] = "%s/%s" % (base_meta["control"], variant)
        meta["variant"] = variant
        meta["variant_doc"] = (
            "weekly champion (lowest trailing-%d-day MSE) predicts the week"
            % base_meta["protocol"]["select_days"] if variant == "single" else
            "rank-mean of all %d members' cross-sections, score_ensemble.py's "
            "combiner" % len(member_preds))
        sub = os.path.join(out_dir, variant) if out_dir else None
        results[variant] = scoring.score_and_write(
            panel, preds, rows, sub, meta, top_k=args.top_k,
            cost_bps=args.cost_bps, max_horizon=args.max_horizon, quiet=quiet)
    return results


def rank_ic_on(panel, preds, rows):
    """Mean daily rank IC (the tuning objective), via baselines/scoring.py."""
    return scoring.rank_ic_mean(panel, preds, rows)


# ---------------------------------------------------------------- CLI glue

def add_common_args(ap):
    ap.add_argument("--panel", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--param", default="RET_CS",
                    help="training target column (default RET_CS, the "
                         "evolution arm's target)")
    ap.add_argument("--score-param", default=None)
    ap.add_argument("--realized", default="auto",
                    choices=["auto", "sidecar", "param"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=K_MODELS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--windows", type=int, default=WINDOWS_PER_WAKE)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--half-life", type=float, default=HALF_LIFE)
    ap.add_argument("--select-days", type=int, default=SELECT_DAYS)
    ap.add_argument("--warmup-from", default=WARMUP_FROM)
    ap.add_argument("--score-from", default=SCORE_FROM)
    ap.add_argument("--score-to", default=SCORE_TO)
    ap.add_argument("--tune-from", default=TUNE_FROM)
    ap.add_argument("--tune-to", default=TUNE_TO)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--cost-bps", type=float, default=None)
    ap.add_argument("--max-horizon", type=int, default=10)
    ap.add_argument("--threads", type=int, default=1,
                    help="torch intra-op threads (1 keeps runs reproducible)")
    ap.add_argument("--exclude-features", default="",
                    help="comma-separated columns to DROP from the model "
                         "INPUTS (the target --param column is still read "
                         "for training/scoring).  On the core7 panels this "
                         "must be RET_CS,RET_CS_Z,RET_CS5: RET_CS5 is "
                         "forward-looking and the other two are target "
                         "columns whose causal previous-row information is "
                         "already carried by RET_CS_IN")
    ap.add_argument("--quiet", action="store_true")
    return ap


def apply_feature_selection(panel, exclude_csv, quiet=False):
    """Drop columns from panel.X / panel.features (model INPUTS only).

    Must run after Panel.__init__, which has already extracted Y (--param)
    and Yscore, so excluding the target column from the inputs does not
    affect what the model is trained on or scored against.  Returns
    (input_features, excluded) for the meta record.
    """
    exclude = [c for c in exclude_csv.split(",") if c] if exclude_csv else []
    missing = [c for c in exclude if c not in panel.features]
    if missing:
        print("# note: --exclude-features %s not in the panel header %s; "
              "ignored" % (",".join(missing), panel.features))
    drop = set(exclude) - set(missing)
    if drop:
        keep = [i for i, f in enumerate(panel.features) if f not in drop]
        panel.X = np.ascontiguousarray(panel.X[:, :, keep])
        panel.features = [panel.features[i] for i in keep]
    if not quiet:
        print("# model inputs (%d): %s%s"
              % (panel.n_feats, panel.features,
                 "  [excluded: %s]" % ", ".join(sorted(drop)) if drop else ""))
    return list(panel.features), sorted(drop)


def setup(args):
    torch.set_num_threads(max(1, args.threads))
    panel = Panel(args.panel, args.param, args.score_param, args.realized)
    if not args.quiet:
        print("# realized: %s (%s), trained on %s; panel header (%d) %s"
              % (panel.realized, panel.score_param, panel.param,
                 panel.n_feats, panel.features))
    args._input_features, args._excluded_features = apply_feature_selection(
        panel, getattr(args, "exclude_features", ""), quiet=args.quiet)
    if "RET_CS5" in panel.features:
        raise SystemExit(
            "RET_CS5 is a FORWARD-LOOKING column and may never be a model "
            "input; rerun with --exclude-features RET_CS,RET_CS_Z,RET_CS5"
        )
    score_rows = panel.rows_between(args.score_from, args.score_to)
    if not score_rows:
        raise SystemExit("no panel rows in %s..%s"
                         % (args.score_from, args.score_to))
    warmup_row = min(panel.first_row_on_or_after(args.warmup_from),
                     score_rows[0])
    return panel, score_rows, warmup_row


def base_meta(args, panel, control, members, extra=None):
    meta = {"control": control,
            "panel": panel.path, "panel_name": panel.name(),
            "param": panel.param, "score_param": panel.score_param,
            "realized_source": panel.realized,
            "input_features": getattr(args, "_input_features",
                                      list(panel.features)),
            "excluded_features": getattr(args, "_excluded_features", []),
            "k_members": len(members),
            "members": [m.describe() for m in members],
            "protocol": protocol_meta(args.epochs, args.windows, args.batch,
                                      args.half_life, args.select_days),
            "warmup_from": args.warmup_from,
            "score_from": args.score_from, "score_to": args.score_to,
            "top_k": args.top_k, "cost_bps": args.cost_bps,
            "seed": args.seed,
            "torch_version": torch.__version__}
    if extra:
        meta.update(extra)
    return meta


def dump_trace(out_dir, trace, meta):
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "champion_trace.json"), "w") as fh:
        json.dump({"meta": {k: meta[k] for k in
                            ("control", "panel_name", "seed")},
                   "trace": trace}, fh, indent=1)
