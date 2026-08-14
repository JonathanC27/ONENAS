# Baseline suite

The set of forecasters an online neuroevolution model has to beat before any
claim of the form *"online neuroevolution beats periodic retraining, and beats
online LSTM/GRU and online ARIMA"* is worth making.

Everything here is new code under `scripts/pooled/baselines/`; nothing outside
this directory is modified. The metric definitions are **imported from
`../score_stream.py`**, not re-derived, so a baseline number and a ONE-NAS
number come out of the same code path and are directly comparable.

---

## 1. The protocol

A panel is 50 date-aligned per-stock CSVs plus `panel_dates.csv`. Panel row
`r` is one trading day.

```
for r in warmup_row .. last_row:
    pred[r] = model.predict(r)      # may read feature rows <= r-1 ONLY
    model.observe(r)                # Y[r] is now public; the model may update
```

* A forecast for row `r` is issued at the **close of row `r-1`** and predicts
  the next day's return `Y[r]`.
* Immediately after, `Y[r]` is revealed and may be used to update the model.
  This is prequential / test-then-train, identical for every baseline.
* This matches `score_stream.py` exactly: its stitched stream is indexed by the
  **target** date and its `naive` column equals the previous day's realised
  return. `trivial.py --model naive` asserts that our naive baseline reproduces
  that column bit-for-bit, which is the end-to-end check that our row alignment
  is the same as ONE-NAS's.
* Scored span: `2020-01-01 .. 2024-12-31` (1258 trading days on `set1_clean`).

Feature columns are read from the per-stock CSV header, never hardcoded, so
the `set*_v2` panels (extra `RET_CS`, `RET_CS_Z`, fixed `ILLIQUIDITY`,
normalised `RET`/`sprtrn`) run without a code change. All per-stock CSVs in a
panel must share a header; `panel.py` raises if they do not. `--param` selects
the target column (default `RET`; use `--param RET_CS` on a v2 panel to train
and score against the rank-normal target).

### Causality

Two places could leak, and both are handled explicitly:

* **Feature standardisation** uses expanding-window Welford moments updated
  only as rows become public (`panel.RunningStandardizer`). It is
  parameter-free, so it adds nothing to tune. For the RNNs a row is
  standardised once, at the moment it becomes public, and cached
  (`rnn_core.CausalCache`), so a row's representation never changes
  retroactively and a minibatch is one fancy-index.
* **Cross-sectional transforms** (`cs_demean`, `cs_demean_y`) only ever use the
  same day's cross-section, which is available at that day's close.

### The book

Untouched from `score_stream.py`: `$100` long / `$100` short, top-10 and
bottom-10 by signal, rebalanced only when all top-10 signals are `> 0` and all
bottom-10 are `< 0`, costs `TC/PRC` at the prior close charged on **netted**
traded notional. `turnover` is mean daily traded notional / `$200` gross;
`cost%` is total cost as a percentage of the `$100` capital base.

---

## 2. Pre-2020 tuning

> No parameter may be tuned on data after 2019-12-31.

* Hyperparameters are selected on **2016-01-01 .. 2019-12-31** by running the
  *same* prequential loop and scoring only those days.
* The objective is **mean daily cross-sectional rank IC** (`score_stream`'s
  Spearman), the paper's headline metric.
* The winning config is then frozen and the model is re-run for the scored
  span. `protocol._tuning_rows` hard-fails if `--tune-to` is ever pushed past
  2019-12-31.
* Online models keep learning from post-2019 *labels* during the final run —
  that is the protocol under test, not leakage — but no *hyperparameter* ever
  sees a post-2019 outcome.
* Every config tried and its validation score are recorded in
  `meta.json → tuning_table`, so the search is auditable.

Two search strategies, both in `protocol.py`:

* `tune` — full cross product (used where a config costs < 1 s: ridge, AR,
  periodic ridge).
* `tune_staged` — coordinate search, one group of hyperparameters per stage,
  the winner seeding the next stage (used for the RNNs, where a full cross
  product would be 100+ runs of ~30 s).

Two deliberate reuses of a tuned config, both documented in `meta.json`:

* The periodic LSTM inherits `hidden/lr/seq_len/cs_demean_y` from the tuned
  **online** LSTM (`--arch-from`). The spec for that arm is "retrain the *same*
  fixed LSTM", so only the retraining budget is searched.
* The cadence sweep tunes once (yearly for ridge, quarterly for the LSTM) and
  freezes that config across all three cadences, because cadence is the
  experimental variable — everything else must be held constant for the
  performance-vs-cadence curve to mean anything.

---

## 3. The baselines

| name | file | what it is |
|---|---|---|
| `naive` | `trivial.py` | `pred = Y[r-1]`. Reproduces `score_stream`'s naive column exactly. |
| `str1` | `trivial.py` | `pred = -Y[r-1]`, one-day reversal. **The honest floor.** |
| `str5` | `trivial.py` | `pred = -(cumulative 5-day return)`, short-term reversal. |
| `ridge` | `online_ridge.py` | Pooled online ridge / RLS with exponential forgetting. |
| `ar` | `online_ar.py` | Per-stock online ARIMA (ARIMA-OGD / ARIMA-ONS). |
| `lstm`, `gru` | `online_rnn.py` | Pooled online LSTM/GRU, daily incremental training. |
| `periodic_{ridge,lstm}_{yearly,quarterly,monthly}` | `periodic_retrain.py` | Retrain from scratch at each boundary, freeze, predict forward. |

**1. Trivial floor.** Parameter-free, so no tuning step is run at all. `str1`
is the bar: a free, zero-parameter signal that scores a positive rank IC on
this data. Any learned model that does not clear it is not worth reporting.

**2. `online_ridge.py`** — one linear model for all 50 stocks, fed the same
feature vector everything else gets. Sufficient statistics with a forgetting
factor:

```
A <- lam*A + sum_k x_k x_k^T ;  b <- lam*b + sum_k x_k y_k ;  S <- lam*S + n
w  = (A + delta*S*I)^-1 b
```

which is RLS with forgetting written in batched normal-equation form —
numerically steadier than the sequential Sherman-Morrison recursion, identical
solution, and `d` is 7-19 so the daily solve is free. Because `RET` is one of
the feature columns and `--lags` supplies its history, this **contains a pooled
AR(p)**; that is why the online-AR arm below exists separately rather than as
the only linear baseline. Grid: `lam × delta × lags × cs_demean × cs_demean_y`
(240 configs, ~60 s total).

**3. `online_ar.py`** — the classical per-series arm. Anava et al. (COLT 2013)
and Liu et al. (AAAI 2016) show ARIMA(p,d,q) is learnable online by running an
AR predictor on the `d`-times differenced series with a no-regret update.
Returns are already a differenced price series, so `d=0` (plain AR(p) on
returns) is the natural instance, but `d=1` stays in the grid so the "I" is
actually exercised. Per stock, on a causally-normalised series, with either
**OGD** (`w <- Pi_D(w - eta*grad)`) or **ONS** (`A <- A + gg^T`,
`w <- Pi_D(w - (1/gamma) A^-1 g)`, rank-1 Sherman-Morrison), projected onto an
L2 ball so one outlier day cannot blow up the weights. Vectorised across
stocks. Grid: `method × p × eta × diff` (80 configs, ~17 s total).

**4. `online_rnn.py`** — pooled LSTM/GRU, one recurrent layer, hidden 8-32,
linear head, no stock identity. Predict = forward pass over feature rows
`r-T..r-1`; update = `steps_per_day` Adam minibatch steps sampled from the
trailing `replay_days` days of (sequence, label) pairs. **Optimiser state
persists across days**, so this is genuine incremental training, not a daily
refit. Kept tiny on purpose: the comparison target is an evolved ONE-NAS RNN of
similar capacity, and a 32-unit LSTM already carries far more parameters than
the evolved networks it is being asked to beat. Staged tuning over
`hidden × lr`, then `steps_per_day × replay_days`, then `seq_len × cs_demean_y`.

**5. `periodic_retrain.py`** — the "current practice" arm. At each boundary
(first trading day of each year/quarter/month, plus a forced retrain on the
first predicted day) the model is retrained **from scratch** on every target
row `<= boundary-1` (expanding, or the trailing `--lookback-days`), then frozen.
Between boundaries it never sees a label — which is exactly the thing an online
learner is supposed to beat. `--model ridge|lstm|gru`, `--cadence
yearly|quarterly|monthly`. The neural variant spends a fixed budget of
`max_steps` Adam steps per retrain rather than a fixed epoch count, so monthly
cadence costs 12x yearly in wall clock rather than 12x a growing epoch; the
budget itself is tuned pre-2020.

---

## 4. Reproducing every number

Everything is local, CPU-only, deterministic given `--seed`. **No cluster jobs.**

```bash
cd scripts/pooled/baselines
PANEL=/path/to/set1_clean

# the whole suite (~13 min on a laptop), tuning included
python3 run_all.py --panel $PANEL --baselines all --results-dir results

# or a subset: trivial | cheap | neural | periodic | paper | all
python3 run_all.py --panel $PANEL --baselines cheap

# one baseline on its own, with its tuning table on stdout
python3 online_ridge.py --panel $PANEL --out-dir results/ridge/set1_clean
python3 online_rnn.py   --panel $PANEL --cell gru --out-dir out/gru
python3 periodic_retrain.py --panel $PANEL --model ridge --cadence monthly \
        --out-dir out/periodic_ridge_monthly

# re-run with a frozen config (no tuning) -- exactly what run_all does for the
# cadence sweep
python3 periodic_retrain.py --panel $PANEL --model ridge --cadence quarterly \
        --config '{"delta":0.001,"lags":1,"cs_demean":false,
                   "cs_demean_y":true,"lookback_days":0}'

# rebuild the summary table from results already on disk
python3 run_all.py --panel $PANEL --baselines all --table-only

# cost sensitivity: flat 5 bps one-way instead of the panel's TC/PRC
python3 run_all.py --panel $PANEL --baselines cheap --extra "--cost-bps 5"

# a v2 panel, targeting the rank-normal column
python3 run_all.py --panel /path/to/set1_v2 --baselines paper --extra "--param RET_CS"
```

Output tree:

```
results/<baseline>/<panel>/predictions.csv   date,stock,pred  (tidy, scored span)
                          /meta.json         model + frozen hyperparameters + timings
                          /metrics.json      per-year and overall metric block
                          /book_daily.csv    daily long-short book
                          /run.log           stdout, tuning table included
results/summary_<panel>.csv                  the aggregate table
```

`predictions.csv` is the tidy interchange format (`date,stock,pred`, `stock` =
ticker, `date` = the **target** date). To score any other tidy prediction file
with the same metrics, use `scoring.score_and_write` / `scoring.build_days` —
which is what every baseline already calls, so no adapter step is needed.

### Determinism

`--seed` fixes the torch init, the replay sampler, and each periodic retrain's
init (`seed + 1000*retrain_index`). The RNNs run with `torch.set_num_threads(1)`
by default (`--threads`) so runs are bit-reproducible; raising it trades
reproducibility for a little speed. The non-neural baselines have no stochastic
component at all.

### Sanity checks that must hold

If these do not reproduce, the protocol is wrong and nothing else should be
believed:

* `naive` rank IC ≈ **-0.018**, turnover ≈ **1.17**, `cost%` ≈ **52%** — heavy
  drag, the signature of a signal that flips its whole book almost daily.
* `str1` rank IC ≈ **+0.018** and exactly `-1 ×` the naive rank IC.
* `naive`'s printed `model` row equals its `naive` reference row exactly
  (asserted in `trivial.py`).

### Dependencies

`numpy` everywhere; `torch` only for `lstm`/`gru` and `periodic_*_lstm`
(detected at runtime, with a clear message if absent). Locally: numpy 2.0.2,
torch 2.8.0, python 3.9. On Anvil the default `python3` has **no torch** —
`module load learning/conda-2021.05-py38-cpu` or an `anaconda/*` module first.
The non-neural baselines are pure numpy and run anywhere. The whole suite is a
laptop job; there is no reason to submit it to a cluster.
