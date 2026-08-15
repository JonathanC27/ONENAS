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

## Measured results — set1_v2 / set2_v2, 2020-01-01 .. 2024-12-31, seeds 42/43

1258 scored days each, 50 stocks, trained on RET_CS, scored against the
sidecar RET_raw returns.  ICs come from `scoring.py` (=`score_stream.py`);
net%/Sharpe come from `sleeves_book.py`, which re-books the same
predictions.csv with `score_stream.run_book(book="sleeves", hold_days=10)` —
the evolution arm's book.  (`metrics.json` also carries `scoring.py`'s
default Algorithm-1-trigger book; on a rank-normal signal that trigger fires
nearly daily and is not comparable, so it is not quoted here.)

| control | run | variant | pearsonIC | rankIC@1 | t | rankIC@5 | net% (sleeves) | Sharpe |
|---|---|---|---|---|---|---|---|---|
| control1 | set1_v2 seed42 | single | +0.0011 | -0.0023 | -0.45 | +0.0098 | +9.5 | +0.28 |
| control1 | set1_v2 seed42 | ensemble | +0.0117 | +0.0094 | +1.79 | +0.0226 | +35.0 | +0.72 |
| control1 | set1_v2 seed43 | single | +0.0063 | +0.0085 | +1.71 | +0.0162 | +30.5 | +0.82 |
| control1 | set1_v2 seed43 | ensemble | +0.0168 | +0.0172 | +3.12 | +0.0246 | +39.6 | +0.85 |
| control1 | set2_v2 seed42 | single | +0.0176 | +0.0176 | +3.30 | +0.0200 | +45.5 | +1.22 |
| control1 | set2_v2 seed42 | ensemble | +0.0266 | +0.0261 | +4.25 | +0.0297 | +67.7 | +1.48 |
| control1 | set2_v2 seed43 | single | +0.0219 | +0.0193 | +3.62 | +0.0202 | +30.3 | +0.91 |
| control1 | set2_v2 seed43 | ensemble | +0.0255 | +0.0248 | +4.08 | +0.0313 | +68.9 | +1.63 |
| control2 | set1_v2 seed42 | single | +0.0077 | +0.0077 | +1.53 | +0.0165 | +15.8 | +0.41 |
| control2 | set1_v2 seed42 | ensemble | +0.0182 | +0.0154 | +2.76 | +0.0255 | +46.5 | +0.88 |
| control2 | set1_v2 seed43 | single | +0.0131 | +0.0117 | +2.28 | +0.0117 | +29.0 | +0.85 |
| control2 | set1_v2 seed43 | ensemble | +0.0145 | +0.0150 | +2.74 | +0.0182 | +38.0 | +0.82 |
| control2 | set2_v2 seed42 | single | +0.0151 | +0.0174 | +3.12 | +0.0173 | +39.5 | +1.36 |
| control2 | set2_v2 seed42 | ensemble | +0.0252 | +0.0267 | +4.24 | +0.0277 | +65.5 | +1.75 |
| control2 | set2_v2 seed43 | single | +0.0123 | +0.0133 | +2.45 | +0.0154 | +29.5 | +0.96 |
| control2 | set2_v2 seed43 | ensemble | +0.0278 | +0.0273 | +4.36 | +0.0293 | +64.4 | +1.61 |

Control 2's tuned config: `{"cell": "rnn", "hidden": 8, "lr": 3e-3}` on BOTH
panels (full 8-config tuning tables in `meta.json`).  Runtimes on an M-series
laptop, torch single-threaded: control 1 ≈ 10–15 min/run; control 2 first
seed ≈ 14–15 min (grid search included), later seeds ≈ 7 min.

**The >+0.02 readings on set2_v2 were lookahead-audited before being
believed** (per the pre-registered rule above).  The explanation is panel
difficulty, not leakage: the free `str1` one-day reversal rule scores rank IC
**+0.0256** on set2_v2 vs **+0.0175** on set1_v2 (same scoring path,
`baselines/trivial.py --model str1 --param RET_CS`).  Measured against that
per-panel floor, every control ensemble sits within ±0.002 of str1 and every
control single sits 0.006–0.020 BELOW it — exactly the signature of models
that harvest the panel's reversal effect and nothing more.  The causal chain
was re-audited independently: training windows require targets ≤ wake−1,
champion selection reads targets ≤ wake−1, day-r forecasts read rows
r−40..r−1 through the same `make_sequences`/`CausalCache` code the audited
RNN baselines use, and runs are bit-reproducible per seed.

### Reading against the evolution arm

* **Singles.**  Evolution's single global best is **+0.0131 ± 0.0017**; the
  no-search singles average +0.0108 (control 1) / +0.0125 (control 2) across
  these four runs, with control 1 as low as −0.002 on set1.  At the
  single-model level the searched genome looks modestly better than a random
  or grid-picked fixed architecture under the identical pipeline — but both
  sit BELOW the per-panel str1 floor.
* **Ensembles.**  The generic 8-member rank-mean ensembles average +0.0178
  (control 1) / +0.0211 (control 2) — i.e. they already reach the evolution
  ensemble's reported ≈ +0.018 without any search.  The ensemble lift
  (+0.006–0.010 over own singles) appears to be a property of rank-mean
  ensembling small RNNs, not of evolved diversity.
* Honest read so far: **the case that architecture search is necessary is
  not yet made.**  Whatever edge exists is concentrated in the single-model
  comparison and is smaller than the panel-to-panel spread; the ensemble
  numbers are matched by seed-diversity alone.  A per-panel,
  same-seed-protocol comparison of the evolution ensemble vs these two
  ensembles (and vs the str1 floor) is the decisive table for the paper.

Raw numbers: `results_controls/summary.csv` (IC + Algorithm-1 book),
`results_controls/sleeves_summary.csv` (sleeves book, H=10), and the per-run
`metrics.json`.  Rebuild with `run_controls.py --table-only` and
`sleeves_book.py --panels ...`.
