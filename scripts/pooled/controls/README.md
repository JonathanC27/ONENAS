# No-search controls

The two controls a NAS paper cannot survive review without.  The claim under
test: does the EVOLVED ARCHITECTURE contribute anything beyond (a) the online
training pipeline, (b) the CS-rank-normal target, (c) the data, and (d)
generic ensembling?  The evolution arm's edge is only attributable to the
*search* if it beats arms that have everything else and never search.

Everything here is new code under `scripts/pooled/controls/`; nothing outside
this directory is modified.  The metric code is **imported** from
`../baselines/scoring.py`, which imports it from `../score_stream.py` — so a
control's Pearson IC, rank IC, and book numbers come out of the exact code
path that scores the evolution arm and the baseline suite.

## The two controls

**Control 1 — fixed-random-architecture online pipeline**
(`control1_random.py`).  K=8 tiny RNN architectures are sampled ONCE at t=0
from a documented distribution and then never changed:

| knob | distribution |
|---|---|
| cell | Uniform{vanilla RNN (tanh), GRU} |
| hidden units | UniformInt[4, 16] |
| recurrent density | Uniform{1.0, 0.75, 0.5} — a fixed binary mask on `weight_hh`, re-applied after every optimiser step (the fixed-topology analogue of evolved sparse recurrent connectivity) |
| lr | fixed, shared (3e-3 Adam; deliberately NOT sampled or tuned, so the arm has no tuned knob at all) |

Each member trains online forever (weights + Adam state persist, no birth, no
death, no mutation).  One pass produces two variants:

* `single/` — each week, the member with the lowest MSE on the trailing 60
  days is the champion and predicts the next 5 days.  Compare against the
  evolution arm's single global best (**+0.0131 ± 0.0017** rank IC).
* `ensemble/` — the rank-mean of all 8 members' cross-sections.  Compare
  against the 8-island-champion rank ensemble (**≈ +0.018**, pending).

**Control 2 — ensemble of identical independently-initialised fixed RNNs**
(`control2_fixed_ensemble.py`).  One architecture — the winner of an 8-config
grid (`cell × hidden{8,16} × lr{1e-3,3e-3}`) tuned on **2016–2019 only**,
objective = mean daily rank IC, fixed internal tuning seed so the chosen
config cannot depend on `--seed` — instantiated 8 times with different seeds,
trained online under the same cadence, combined with the same rank-mean rule.
Control 1's ensemble carries architecture-diversity; this one carries
seed-diversity only.  Together they separate "ensembling generic small RNNs"
from "ensembling evolution's island champions".

## Fairness: what is matched to the evolution arm

| protocol element | evolution arm | these controls |
|---|---|---|
| clock | weekly (train, then predict the next 5 trading days) | identical: wake every 5 panel rows |
| training unit | 40-day windows | identical: 40-step sequences, seq-to-seq MSE over all 40 steps |
| training data sampling | 2000 windows sampled per generation, PER-style recency | 512 sampled (stock, window) examples per wake, exponential recency with a 504-trading-day (~2yr) half-life |
| epochs per wake | 10 backprop epochs | 10 passes (minibatch 256, Adam, grad-clip 1.0) |
| selection window | elites re-scored on the ~5 most recent windows; global best predicts next week | champion = lowest MSE on the trailing 60 days (~1.5 windows/wk × recent weeks), re-scored with current weights, predicts next week |
| ensemble members / combiner | 8 island champions, rank-mean (`score_ensemble.py --combine rank_mean`) | 8 members, same centred-rank-mean rule (same definition, on `score_stream.ranks`) |
| model capacity | evolved genomes ~7–30 nodes, mixed cell types, sparse recurrence | hidden 4–16, {vanilla RNN, GRU}, optional recurrent sparsity; 103–911 parameters in a typical draw |
| features / target | panel CSV header columns; target RET_CS | identical: header-driven features (works on `set*_v2` and the upcoming `set*_core7` without change), `--param RET_CS` |
| realised returns for scoring | sidecar `RET_raw_` columns (`--realized auto`) | identical (via `baselines/panel.py`) |
| training span / scored span | trains from 2004+, scored 2020-01-01 .. 2024-12-31 | identical: warmup from panel start (2004), scored 2020–2024 |
| causality | forecast for day r uses information ≤ r−1 | identical: `CausalCache` expanding standardisation, windows require targets ≤ wake−1, selection uses targets ≤ wake−1, day-r forecast reads rows r−40..r−1 |
| metric code | `score_stream.py` | the same functions, imported through `baselines/scoring.py` |

## What is necessarily different (and why it is conservative)

* **No search.**  Architectures are frozen at t=0 (control 1) or chosen once
  pre-2020 (control 2).  That is the experimental variable, not a confound.
* **Optimiser.**  Torch Adam with gradient clipping instead of the C++
  backprop implementation.  Adam is the stronger generic choice; if anything
  this biases *against* the evolution arm's search claim, which is the safe
  direction for a control.
* **Weight persistence.**  Evolution's genomes are born and die; a control
  member's weights train continuously from 2004.  Each control member
  therefore sees *more* cumulative gradient than any individual genome —
  again conservative.
* **Window count per wake.**  512 sampled windows vs evolution's 2000 (which
  is amortised over 40 genomes/generation).  Per surviving model, per week,
  the controls take 20 Adam minibatch steps over 512 fresh recency-sampled
  windows; raise `--windows` to check sensitivity (runtime is linear in it).
* **Champion metric.**  Trailing-60-day MSE on the standardised target rather
  than the C++ fitness on the 5 most recent windows; same information set,
  same purpose (pick this week's predictor from public data only).

## No-lookahead checklist

1. Features and targets are standardised with expanding Welford moments,
   frozen per row the moment the row becomes public
   (`baselines/rnn_core.CausalCache`).
2. A training window starting at row `a` requires targets `a+1..a+40 ≤
   wake−1`; the sampler enforces `a ≤ wake−41`.
3. Champion selection at a wake scores target rows `wake−60..wake−1` only.
4. The forecast for row `r` reads feature rows `r−40..r−1` (`make_sequences`,
   the same indexing the audited RNN baselines use).
5. Control 2's grid search runs only on 2016–2019 and hard-fails if
   `--tune-to` passes 2019-12-31.  Control 1 has nothing to tune.
6. Determinism: everything is a pure function of `--seed`
   (`torch.set_num_threads(1)`, per-member torch and numpy generators);
   re-running a config reproduces `metrics.json` bit for bit.

## Running

```bash
cd scripts/pooled/controls

# both controls, both panels, seeds 42 and 43 (the reported grid)
python3 run_controls.py \
    --panels /path/to/set1_v2 /path/to/set2_v2 --seeds 42 43

# one control on its own
python3 control1_random.py --panel /path/to/set1_v2 --seed 42 \
    --out-dir results_controls/control1/set1_v2_seed42
python3 control2_fixed_ensemble.py --panel /path/to/set1_v2 --seed 42 \
    --out-dir results_controls/control2/set1_v2_seed42

# rebuild the summary table from disk
python3 run_controls.py --panels x --table-only
```

Runtime: a control-1 run is ~10–15 min on an M-series laptop; control 2 adds
its once-per-panel tuning pass (~10 min, reused for later seeds via
`--config`, which `run_controls.py` does automatically).  CPU-only, no
cluster jobs.

Output tree (predictions.csv / book_daily.csv are gitignored, as in
`baselines/results`):

```
results_controls/<control>/<panel>_seed<sd>/single/    predictions.csv (date,stock,pred),
                                                       metrics.json, meta.json, book_daily.csv
                                           /ensemble/  same
                                           /champion_trace.json   weekly champion + member MSEs
                                           /run.log
results_controls/summary.csv
```

`meta.json` records the full config: the sampled architectures (control 1)
or the tuning table and frozen config (control 2), every protocol constant,
seeds, and timings.

## Reading the result

* Evolution single genome: **+0.0131 ± 0.0017** rank IC@1; 8-champion
  ensemble ≈ **+0.018** (pending).  `str1` (free one-day reversal) is
  ≈ +0.0175 on set1_clean.
* If a control lands **above +0.02** rank IC, do not believe it before
  re-auditing for lookahead — that would exceed every learner measured on
  this data.
* The interesting comparisons: evolution single vs `control1/single` and
  `control2` members (does search beat a random fixed architecture under the
  same pipeline?), and evolution ensemble vs both `ensemble/` variants (is
  the ensemble lift specific to evolved diversity, or generic?).

## Input features (audited per panel family)

Inputs are read from the per-stock CSV header, then filtered by
`--exclude-features` (recorded per run in `meta.json -> input_features /
excluded_features`):

* **v2 panels** — all 8 header columns as inputs: `RET, VOL_CHANGE,
  BA_SPREAD, ILLIQUIDITY, sprtrn, TURNOVER, RET_CS, RET_CS_Z` (target
  columns appear only at input rows <= r-1, which is causal).
* **core7 panels** — run with `--exclude-features RET_CS,RET_CS_Z,RET_CS5`,
  leaving exactly the evolution arm's 7 inputs: `RET, RET_CS_IN, BA_SPREAD,
  ILLIQUIDITY, REV21_1, TURN_RATIO, VOL21`.  `RET_CS5` is FORWARD-LOOKING
  and `common.setup` hard-fails if it is ever left in the input set; the
  causal previous-row information of the other two target columns is carried
  by `RET_CS_IN`.  Target stays `RET_CS` in both families.

## Measured results — full grid, 2020-01-01 .. 2024-12-31

4 panel sets x {v2, core7} x seeds {42,43,44} x both controls = 48 runs,
~1258 scored days each, trained on RET_CS, scored against the sidecar
RET_raw returns.  Rank IC from `scoring.py` (= `score_stream.py`); net% and
Sharpe from `sleeves_book.py` (`score_stream.run_book(book="sleeves",
hold_days=10)`, the evolution arm's book).  Per-run rows are in
`results_controls/final_table.md`; per-panel means (over 3 seeds):

| set | arm | core7 rIC@1 | core7 net% | core7 Sharpe | v2 rIC@1 | v2 net% | v2 Sharpe |
|---|---|---|---|---|---|---|---|
| set1 | c1 single | +0.0049 | -3.3 | -0.08 | +0.0048 | +16.8 | +0.47 |
| set1 | c1 ensemble | +0.0072 | +19.3 | +0.30 | +0.0133 | +39.6 | +0.85 |
| set1 | c2 single | +0.0058 | -4.1 | -0.10 | +0.0097 | +22.9 | +0.64 |
| set1 | c2 ensemble | +0.0092 | +28.5 | +0.43 | +0.0132 | +38.6 | +0.78 |
| set2 | c1 single | +0.0128 | +21.5 | +0.60 | +0.0169 | +37.7 | +1.10 |
| set2 | c1 ensemble | +0.0235 | +49.4 | +1.06 | +0.0251 | +64.6 | +1.56 |
| set2 | c2 single | +0.0170 | +28.6 | +0.77 | +0.0159 | +33.0 | +1.09 |
| set2 | c2 ensemble | +0.0224 | +50.8 | +1.12 | +0.0270 | +60.6 | +1.57 |
| set3 | c1 single | +0.0055 | +15.2 | +0.33 | +0.0090 | +10.9 | +0.29 |
| set3 | c1 ensemble | +0.0142 | +38.2 | +0.61 | +0.0150 | +42.1 | +0.86 |
| set3 | c2 single | +0.0045 | -1.7 | -0.03 | +0.0141 | +11.1 | +0.27 |
| set3 | c2 ensemble | +0.0067 | +18.2 | +0.33 | +0.0184 | +44.8 | +0.93 |
| set4 | c1 single | +0.0125 | +23.9 | +0.66 | +0.0195 | +18.5 | +0.59 |
| set4 | c1 ensemble | +0.0225 | +40.8 | +0.76 | +0.0276 | +37.3 | +0.89 |
| set4 | c2 single | +0.0155 | +20.3 | +0.58 | +0.0165 | +21.7 | +0.70 |
| set4 | c2 ensemble | +0.0229 | +36.3 | +0.74 | +0.0250 | +39.3 | +1.00 |

Aggregates over all 12 runs per (control, family), mean ± across-run sd:

| family | arm | rank IC@1 | net% (sleeves) | Sharpe |
|---|---|---|---|---|
| core7 | c1 single | +0.0089 ± 0.0044 | +14.3 ± 14.9 | +0.38 ± 0.39 |
| core7 | c1 ensemble | +0.0169 ± 0.0070 | +36.9 ± 14.5 | +0.68 ± 0.31 |
| core7 | c2 single | +0.0107 ± 0.0064 | +10.8 ± 17.0 | +0.30 ± 0.45 |
| core7 | c2 ensemble | +0.0153 ± 0.0078 | +33.5 ± 15.7 | +0.66 ± 0.36 |
| core7 | *evolution single* | *+0.0078* | | |
| core7 | *evolution ensemble* | *+0.0106 ± 0.0021* | *+41.4 ± 6.1* | *+0.63 ± 0.10* |
| v2 | c1 single | +0.0125 ± 0.0076 | +21.0 ± 13.2 | +0.61 ± 0.37 |
| v2 | c1 ensemble | +0.0202 ± 0.0069 | +45.9 ± 13.0 | +1.04 ± 0.33 |
| v2 | c2 single | +0.0140 ± 0.0033 | +22.2 ± 9.4 | +0.68 ± 0.34 |
| v2 | c2 ensemble | +0.0209 ± 0.0060 | +45.9 ± 10.9 | +1.07 ± 0.34 |

Control 2's tuned configs (2016-2019 only, tables in `meta.json`):
`{rnn, hidden 8, lr 3e-3}` on all four v2 panels; `{gru, hidden 8, lr 3e-3}`
on all four core7 panels.

**Lookahead audit of readings above +0.02** (pre-registered rule: triple-check
before believing).  Explained by panel difficulty, not leakage: the free
`str1` one-day reversal rule scores rank IC +0.0175 / +0.0256 / +0.0173 /
+0.0244 on set1-4 (identical for v2 and core7 — RET_CS is the same column).
Measured against that per-panel floor, every control ensemble sits at or
below str1 (v2 within ~0.002, core7 0.002-0.011 below) and every single sits
clearly below it.  The causal chain was audited independently: training
windows require targets <= wake-1, champion selection reads targets <=
wake-1, day-r forecasts read rows r-40..r-1 through the same
`make_sequences`/`CausalCache` code as the audited RNN baselines, the
forward-looking RET_CS5 column is hard-excluded, and runs are
bit-reproducible per seed (verified).

### Reading against the evolution arm (core7 = identical features)

* **IC.**  The no-search controls beat the evolution arm's primary core7
  numbers: singles +0.0089 (c1) / +0.0107 (c2) vs evolution's +0.0078;
  ensembles +0.0169 (c1) / +0.0153 (c2) vs evolution's +0.0106 ± 0.0021.
* **Economics.**  A statistical wash: evolution's +41.4% ± 6.1 net (Sharpe
  0.63 ± 0.10) vs the control ensembles' +36.9 ± 14.5 (0.68 ± 0.31) and
  +33.5 ± 15.7 (0.66 ± 0.36).  Evolution's mean net% is a few points higher
  and much more stable across panels; the controls' Sharpe means are
  slightly higher; the intervals overlap heavily either way.
* **Ensembling is the active ingredient.**  Both arms gain ~+0.006-0.008 IC
  and ~+20 net points from the 8-member rank-mean combination, and
  seed-diversity alone (control 2) captures it; evolved island diversity
  adds nothing detectable on top.
* **The str1 floor stands above everyone.**  No arm — evolution included —
  beats the free per-panel reversal rule on rank IC.
* Honest verdict: on identical features, cadence, target and combiner,
  **architecture search does not look necessary on this data**.  The
  evolution arm's one defensible advantage is the LOWER VARIANCE of its
  economics across panels (sd 6.1 net points vs ~15 for the controls); its
  IC is matched or exceeded by frozen random architectures under the same
  online pipeline.

Raw numbers: `results_controls/summary.csv` (IC + Algorithm-1 book),
`results_controls/sleeves_summary.csv` (sleeves book, H=10),
`results_controls/final_table.md` (side-by-side grid), and the per-run
`metrics.json`.  Rebuild with `run_controls.py --table-only`,
`sleeves_book.py --panels ...`, `final_table.py`.
