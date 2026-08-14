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
panel must share a header; `panel.py` raises if they do not.

Training target and scoring target are separable:

* `--param` — the column the model is **trained** on (default `RET`).
* `--score-param` — the realised return the IC and the long-short book are
  **scored** against (default: same as `--param`).

On a v2 panel, `--param RET_CS --score-param RET` trains on the rank-normal
target while still booking real returns. Daily rank IC comes out *identical*
either way — `RET_CS` is a within-day monotone transform of `RET`, and this is
verified: on a v2-shaped panel STR1 scores `+0.0175` under both settings — but
the book's P&L is only meaningful in return units, so the split matters.

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

# the whole suite (~20 min on an M-series laptop), tuning included
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

# a v2 panel: train on the rank-normal target, book real returns
python3 run_all.py --panel /path/to/set1_v2 --baselines paper \
        --extra "--param RET_CS --score-param RET"
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

---

## 5. Measured results — `set1_clean`, 2020-01-01 .. 2024-12-31

1258 scored days, 50 stocks, long-short top/bottom-10, `TC/PRC` costs, seed 42.
Reproduced in full by `python3 run_all.py --panel set1_clean --baselines all`.
`secs` is wall clock **including** the pre-2020 tuning search.

| baseline | pearsIC | t | rIC@1 | rIC@5 | rIC@10 | net% | Sharpe | MDD% | turnover | cost% | secs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| naive persistence | -0.0172 | -2.28 | -0.0175 | -0.0165 | -0.0157 | -96.04 | -0.81 | 98.93 | 1.166 | 52.32 | 4 |
| **str1** | +0.0172 | +2.28 | **+0.0175** | +0.0165 | +0.0157 | -8.53 | -0.07 | 43.07 | 1.166 | 52.32 | 3 |
| str5 | +0.0162 | +2.18 | +0.0144 | +0.0211 | +0.0122 | +14.99 | +0.11 | 62.75 | 0.594 | 27.06 | 3 |
| online ridge/RLS | +0.0193 | +2.73 | +0.0126 | +0.0188 | +0.0242 | +0.81 | +0.01 | 46.05 | 0.991 | 45.94 | 68 |
| online AR (OGD) | +0.0177 | +3.27 | +0.0147 | +0.0082 | +0.0100 | -9.64 | -0.11 | 51.00 | 0.488 | 22.84 | 18 |
| online LSTM | +0.0162 | +2.39 | +0.0111 | +0.0117 | +0.0132 | +7.92 | +0.08 | 69.93 | 0.379 | 20.46 | 409 |
| online GRU | +0.0103 | +1.64 | +0.0105 | +0.0149 | +0.0130 | +36.63 | +0.40 | 24.61 | 0.435 | 20.62 | 272 |
| periodic ridge, yearly | +0.0197 | +2.70 | +0.0149 | +0.0179 | +0.0194 | +53.12 | +0.44 | 52.03 | 0.933 | 44.66 | 25 |
| periodic ridge, quarterly | +0.0203 | +2.79 | +0.0147 | +0.0198 | +0.0229 | +41.94 | +0.35 | 51.42 | 0.918 | 43.63 | 4 |
| periodic ridge, monthly | +0.0204 | +2.81 | +0.0143 | +0.0194 | +0.0234 | +23.79 | +0.20 | 50.04 | 0.912 | 43.11 | 4 |
| **periodic LSTM, yearly** | +0.0180 | +2.70 | **+0.0181** | +0.0229 | +0.0211 | +41.21 | +0.42 | 30.67 | 0.108 | 6.51 | 20 |
| periodic LSTM, quarterly | +0.0150 | +2.26 | +0.0177 | +0.0185 | +0.0185 | +45.43 | +0.45 | 25.23 | 0.170 | 10.31 | 200 |
| periodic LSTM, monthly | +0.0134 | +2.05 | +0.0117 | +0.0207 | +0.0173 | +44.50 | +0.48 | 32.55 | 0.201 | 11.99 | 175 |

Frozen hyperparameters (all chosen on 2016-2019 only):

| baseline | config |
|---|---|
| online ridge | `lam=1.0, delta=1e-3, lags=1, cs_demean=False, cs_demean_y=True` |
| online AR | `method=ogd, p=1, eta=1e-4, diff=0` |
| online LSTM | `hidden=8, lr=1e-3, seq_len=5, steps_per_day=8, replay_days=1000, cs_demean_y=True` |
| online GRU | `hidden=16, lr=3e-3, seq_len=20, steps_per_day=1, replay_days=1000, cs_demean_y=False` |
| periodic ridge | `delta=1e-3, lags=1, cs_demean=False, cs_demean_y=True, lookback_days=0` |
| periodic LSTM | `hidden=8, lr=1e-3, seq_len=5, max_steps=4000, lookback_days=0, cs_demean_y=True` |

### What this suite says about the paper's claim

Read these numbers before writing the abstract, because two of them are
inconvenient:

1. **`str1` is a genuinely hard bar.** A free, zero-parameter signal scores
   rank IC `+0.0175`. Of the five learned arms, only the periodic LSTM clears
   it, and only barely. Online ridge (`+0.0126`), online AR (`+0.0147`),
   online LSTM (`+0.0111`) and online GRU (`+0.0105`) all score *below* a
   one-line reversal rule. Any headline of the form "our model achieves rank
   IC 0.0x" has to be read against `+0.0175`.
2. **Periodic retraining currently beats the online arms here.** Periodic
   ridge beats online ridge on rank IC (`+0.0149` vs `+0.0126`) and on net
   return (`+53%` vs `+0.8%`); the periodic LSTM beats the online LSTM on
   every column. So "online neuroevolution beats periodic retraining" is a
   real, unearned-by-default claim — this suite does not hand it over.
3. **Costs decide the book, not IC.** The high-IC arms churn ~1.0 turnover and
   pay 43-52% of capital in costs over five years, which is what turns `str1`'s
   positive IC into `-8.5%` net. The periodic LSTM wins net return with a
   *lower* IC than online ridge purely because it trades at 0.11 turnover for
   6.5% cost. Report IC and net% together, or the story is not honest.
4. Pearson IC and rank IC disagree in places (online ridge has the second-best
   Pearson but a middling rank IC), so quote both, as the table above does.

### Dependencies

`numpy` everywhere; `torch` only for `lstm`/`gru` and `periodic_*_lstm`
(detected at runtime, with a clear message if absent). Locally: numpy 2.0.2,
torch 2.8.0, python 3.9. On Anvil the default `python3` has **no torch** —
`module load learning/conda-2021.05-py38-cpu` or an `anaconda/*` module first.
The non-neural baselines are pure numpy and run anywhere. The whole suite is a
laptop job; there is no reason to submit it to a cluster.
