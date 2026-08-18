# PROTOCOL — ONE-NAS vs. adaptive baselines on Avedore dissolved N2O (v2)

**Status: PRE-REGISTERED. Written and committed before any production run.**
Everything below — horizon, row set, metric, gate, baselines, configuration,
seed count, and the success criterion — is fixed here in advance. This campaign
has already been burned once by choosing conventions after seeing numbers; the
point of this document is that there is nothing left to choose afterwards.

Code: `~/wwtp_scripts/v2/` · Data: `/anvil/scratch/x-jchang5/wwtp/v2/`

---

## 0. What v1 got wrong, in one paragraph

ONE-NAS was trained on `aved_line3.csv`: 178,536 consecutive rows, of which
51,270 (28.7%) had `n2o_valid == False` — dead sensor, flatlined, negative-rail.
ONE-NAS has no notion of missing data; consecutive CSV rows *are* consecutive
timesteps, so every window straddling a dead stretch taught the network to
reproduce a flat, negative, physically impossible signal. The normalisation
median was fitted over all rows including the dead ones, which is why the N2O
centre came out **negative (−0.0114)** when the median of the *valid* signal is
**+0.035**: the target level was biased low before a single genome was
evaluated. Separately, `--window_step 1` with `L=144` put 143 of the validation
window's 144 rows into the training pool (direct leakage) and advanced the
online clock by **5 minutes per generation**, so ~700 generations covered ~2.5
days out of 620. Meanwhile the ridge baselines *did* mask invalid rows on both
endpoints. The two sides of the comparison were never scored on the same data.

## 1. Verified facts about ONE-NAS this protocol depends on

Each was established from the C++ source and, where load-bearing, by experiment.

| Fact | Evidence |
|---|---|
| `--training_filenames` → one `TimeSeriesSet` per file; `training_indexes` in argv order | `time_series.cxx:757-770` |
| Windows are sliced **per file**; the slicer restarts at row 0 of every file, so **no window can span a file boundary** | `process_arguments.cxx:376+`, **and empirically — see below** |
| Episodes are numbered **file-major**: all windows of file 0, then file 1, … | empirically confirmed |
| Test episode of generation *g* is **`g + T + V`** | `online_series.cxx:341-372`; empirically confirmed |
| `--time_offset H` alone expresses the H-ahead task: `inputs[k]=row k`, `outputs[k]=row k+H` | `time_series.cxx:445-505` |
| Prediction file data row *i* is episode timestep *j = i+1* | `onenas_island_speciation_strategy.cxx:1119` |
| A file of *R* rows yields `floor((R−H)/L)` windows | `process_arguments.cxx:376+` |
| `--pooled_panel` requires **equal row counts** and treats window *w* of every file as **contemporaneous** | `online_series.cxx:50-57` |
| `--max_pred_sd_ratio` is a **one-sided** guard (explosion only) | `selection_metric.cxx:85-93` |

### The multi-file experiment (`verify_multifile.sh`, job 20008019)

Two 30-row files in disjoint value bands (file A strictly positive, file B
strictly negative), `L=8`, `H=2`. Per-file slicing predicts **6** windows
(`floor(28/8)=3` each); concatenation predicts **7** (`floor(58/8)`). Windows
dumped with `--write_sliced_files`:

```
windows emitted by ONE-NAS: 6          <- per-file, not 7
  generation_0.csv first=+0.500 last=+0.507 -> fileA
  generation_1.csv first=+0.508 last=+0.515 -> fileA
  generation_2.csv first=+0.516 last=+0.523 -> fileA
  generation_3.csv first=-0.500 last=-0.507 -> fileB
  generation_4.csv first=-0.508 last=-0.515 -> fileB
  generation_5.csv first=-0.516 last=-0.523 -> fileB
windows containing rows from BOTH files: 0
RESULT: PASS
```

The log also printed `Current testing episode ID(s): 1 test episodes, first ID 3`
at generation 0 with `T=2, V=1` — confirming `e = g + T + V = 3` empirically.

**Consequence — `--pooled_panel` is NOT used.** Our segments are *sequential in
time*, not parallel contemporaneous series. Pooled-panel mode would make window
*w* of a 2024 segment available for training at the same clock as window *w* of
a 2022 segment: future leakage. Plain multi-file mode is the correct one, and it
is what gives us the property we need — chronological episode ordering with no
window ever spanning a gap.

## 2. Data preparation (`prep_v2.py`)

Source `aved_5min.csv` (uniform 5-min grid, asserted), span from **2022-10-01**.

* Mask: the **registered `n2o_valid`, unmodified**. The published gates are
  registered against it; redefining it would silently invalidate them. Residual
  artefacts (a handful of 12.0 mg/L rail readings in 2022-11) are *measured and
  reported*, not patched — robust percentiles keep them out of the scale and
  none survive into a usable segment.
* Invalid rows are **excluded from training entirely** — not filled, not
  interpolated.
* The series is split into maximal contiguous valid runs, one CSV per run.
* **Minimum segment length = `L + H`.** Justification: a file of *R* rows yields
  `floor((R−H)/L)` windows, so `R ≥ L+H` is exactly the threshold for
  contributing a single training example. Shorter segments cannot contribute
  anything, and padding them would reintroduce the fabricated data this whole
  exercise removes. No *larger* minimum is imposed: a one-window segment is a
  real, fully-observed L-row stretch of the process and is as legitimate as any
  window carved from a long segment. The length distribution is reported so a
  stricter rule could be justified from evidence if ever wanted.
* Each segment is truncated to exactly `nw*L + H` rows, so ONE-NAS's slicer
  emits exactly `nw` windows with zero remainder — this is what makes
  episode→row a closed form rather than a search.
* Normalisation is fitted **only on emitted (valid) rows inside the burn-in
  prefix** (windows `[0,T)`), i.e. only on source rows at or before the last
  target row of window `T−1`. Nothing at or after the first possible validation
  window contributes to any statistic.
* Target map: burn-in robust range `[p0.5, p99.5]` → `[−0.6, +0.6]`, clipped at
  ±0.95. ONE-NAS output nodes are `SIMPLE_NODE` and `RNN_Node` applies `tanh`,
  so predictions are hard-bounded to (−1,1); any target mass outside is
  unreachable by construction and drives MSE toward a saturated, zero-gradient
  constant — the v1 collapse signature. Midpoint- rather than median-centring
  because N2O is strongly right-skewed with a hard floor near zero; median-
  centring leaves the entire negative half of the band unused and compresses the
  bulk to sd 0.085 instead of 0.133.

### What it produced (primary arm, H=72, L=144, T=80)

| | |
|---|---|
| Raw rows in span | 178,536 |
| Valid rows | 127,266 (71.3%) |
| Contiguous valid runs | **1,762** — min 1, median **6**, mean 72, max 10,138 rows |
| Runs ≥ 216 rows (`L+H`) | **64**, holding 109,004 rows = **85.7% of valid** |
| Dropped short runs | 1,698, holding 18,262 valid rows (14.3% of valid) |
| Emitted after truncation | **104,832 rows** (82.4% of valid, 58.7% of raw) |
| Windows (episodes) | **696** — per segment min 1, median 6, max 69 |
| `max_generation = W−T−V−1` | **610** |
| Burn-in cut | 2023-05-12 14:55, 13,032 rows for statistics |
| First scored window | 2023-05-15 03:00 · last ends 2024-06-11 00:30 |
| Scored rows if all generations land | 87,840 |

**Data lost vs. v1's 178,536 rows: 73,704 rows (41.3%) — of which 51,270 were
invalid and should never have been there.** The genuine loss is the 18,262 valid
rows in runs too short to form a window (14.3% of valid data) plus 4,172 rows
truncated from segment tails.

Target after normalisation: scored span `[−0.600, +0.832]`, sd 0.133, **0.0000%
outside the tanh band, 0.0000% clipped**. Physical floor of the process sits at
−0.601 normalised; anything below is nonphysical and is counted by the scorer.

Secondary arm (H=24): 69 segments, 104,904 rows, **717 windows**, max_generation
631, first scored window 2023-05-11 11:00.

### Known limitation of the split, stated in advance

Valid data is wildly uneven in time: **December 2022 – February 2023 contributes
zero usable rows** (the documented 31.2-day dead run plus neighbours), so the
80-window burn-in is forced to run to 2023-05-12 and is dominated by the
March–April 2023 high-N2O excursion (April median 0.53 mg/L vs 0.03 on the
scored span). The normalisation centre therefore reflects an atypical regime.
This is a property of the dataset, not a choice, and it is unavoidable under the
no-lookahead constraint. It is also precisely the regime in which online
adaptation should pay and a frozen model should fail — which the protocol tests.

## 3. Horizon

* **Primary: H = 72 (6 h).** Largest headroom (the H=72 persistence gate is the
  weakest), largest measured adaptation payoff (daily ridge 0.75 of persistence
  at H=72 vs 0.92 at H=24), and fewest excursions already visible at prediction
  time.
* **Secondary: H = 24 (2 h).** Reported, but the success criterion is judged on
  H=72.
* H=6 is not run.

## 4. Scoring convention — ONE, fixed, identical for every arm

* **Metric:** `nMSE = Σ(y−ŷ)² / Σ(y−ȳ)²` in **raw mg/L**, `ȳ` the mean over the
  scored rows.
* **Rows:** exactly the target rows covered by the test windows the run produced
  (episodes `T+V … T+V+G−1`), **intersected with the rows on which every arm is
  finite**. No arm ever gets its own row set. If a run does not complete all
  generations, every baseline is recomputed on the covered subset and the fact
  is reported — the numbers are never taken from a different row set.
* **Mask:** *there is no mask decision left to make.* Every emitted row lies
  inside a contiguous valid segment, so both the scored target and its
  persistence anchor `H` rows back are valid by construction. This is the single
  biggest structural difference from v1, where "naive vs clean" was a live
  choice that moved results.
* **Span:** determined by the run and reported, not chosen. For the primary arm
  it is 2023-05-15 → 2024-06-11.
* **Warm-up:** the **primary number discards nothing** — all `L−1` rows of every
  test window are scored, the strictest convention. Warm-up-discarded variants
  (12 / 24 / 48 rows) are printed as a declared sensitivity **with the gate
  recomputed on each row set**, so no two numbers are ever compared across
  different rows.
* **Not a baseline:** the `naive_N2O` column ONE-NAS writes is
  `test_output[j−1]` — the target at `t+H−5min`, an *oracle*. It scored 0.0141
  in the smoke test. It is reported explicitly labelled so it can never be
  mistaken for the gate.

### The gate must be recomputed. The registered numbers do not transfer.

The registered persistence gates (0.4412 / 0.6105 / **1.0199** at H=6/24/72) were
computed on v1's row set. On the v2 rows, **persistence at H=72 scores 0.8384**.
That is not a discrepancy to explain away — it is a different row set, and it is
exactly the ~3× convention sensitivity that motivated this document. **The gate
for every v2 claim is persistence recomputed by `score_v2.py` on the identical
rows, in the same run.** The v1 gate numbers are not used for anything.

## 5. Baselines (`baselines_v2.py`) — the exact set

All emit per-row predictions keyed by `(seg_id, seg_row)` of the target row and
are joined onto the identical rows by `score_v2.py`.

1. **Persistence** — `ŷ(t+H) = y(t)`. The gate.
2. **Daily-retrained ridge** (`ridge_d1`) — the strong adaptive baseline.
3. **Best periodic-retrain ridge** — the best of `ridge_d3 / d7 / d30`, chosen on
   the same rows; **all four K are reported regardless**, so the choice cannot
   flatter anyone.
4. **Trailing-mean climatology** — nMSE ≡ 1.0 by construction; the "no skill"
   line.

Ridge features: the 14 normalised signals at the issue row + N2O lags
{1,3,6,12,24,72} + hour-of-day and day-of-week harmonics. Refit at each
checkpoint on every sample whose **target has already been observed** — a
forecast may only train on forecasts that have come true. A row is eligible only
if its 72-row feature history lies inside the same segment (91.2% of emitted
rows); this is a property of the row, applied identically to every arm.

Measured on the v2 rows (already run, both horizons). H=72:

| arm | nMSE | vs persistence |
|---|---|---|
| ridge_d1 (daily) | **0.6701** | 0.80 |
| ridge_d7 | 0.7300 | 0.87 |
| ridge_d3 | 0.7909 | 0.94 |
| persistence | 0.8384 | 1.00 |
| ridge_d30 | 1.4665 | 1.37 |

H=24 (secondary), same construction:

| arm | nMSE | vs persistence |
|---|---|---|
| ridge_d1 (daily) | **0.3171** | 0.86 |
| ridge_d3 | 0.3273 | 0.89 |
| ridge_d7 | 0.3393 | 0.92 |
| persistence | 0.3684 | 1.00 |
| ridge_d30 | 0.4819 | 0.97 |

The direction and magnitude reproduce the campaign's measured
adaptation-frequency effect (daily 0.75 at H=72, 0.92 at H=24; monthly 1.84).
Two things to note, both of which were used to set §3 and both of which are
recorded *before* any ONE-NAS run:

* **Headroom is far larger at H=72 than at H=24.** The best baseline sits 20%
  below persistence at H=72 but only 14% below at H=24, and the absolute nMSE
  scale differs 2×. This is the concrete reason H=72 is primary.
* **The crossover sits between 7 and 30 days on *these* rows, not 3–7 as on
  v1's.** That is not a contradiction — it is a different row set — and it is
  the cleanest available illustration of why every arm must be scored on one row
  set and why the v1 gates cannot be carried over.

## 6. ONE-NAS configuration to test

Pre-registered, one arm:

```
--number_islands 16   --generated_population_size 5   --elite_population_size 8
--bp_iterations 10    --num_mutations 1
--possible_node_types simple UGRNN MGU GRU delta LSTM
--get_train_data_by PER --per_alpha 0.6 --per_lambda 0.007 --per_epsilon 1e-8
--time_series_length 144   --window_step 144
--num_training_sets 80     --num_validation_sets 5
--time_offset 72           --selection_metric mse
--repopulation_frequency 0 --max_pred_sd_ratio 3.0
--normalize none --control_size_method none
```

Rationale for the two settings that were previously wrong or unset:

* **`--window_step 144`, pinned equal to `L`.** In non-pooled mode the training
  pool is `{i < current_index}` and the validation window *is* `current_index`,
  so any step `< L` overlaps them by `L−step` rows — leakage ONE-NAS only warns
  about. Pinning to `L` also fixes the adaptation cadence at `L` rows of valid
  series time = **12 h**, comfortably inside the regime where retraining was
  measured to pay (daily ridge beats persistence; the crossover to failure is
  days, not hours). It is deliberately *not* exposed as a knob in `run_v2.sh`.
* **`--repopulation_frequency 0`.** With a non-zero value `do_repopulation`
  indexes `rank[i]` for `i < islands_to_exterminate` while `rank_islands()` may
  return a shorter vector — out-of-bounds read → SIGSEGV. This killed v1 jobs
  20006592/20006593.
* **`--max_pred_sd_ratio 3.0`** is retained but noted as **one-sided**: it
  rejects exploding predictions and is blind to collapse. v1's collapse (sd
  0.001 against target sd 0.056) was never caught by the search. `score_v2.py`
  checks the low side explicitly (sd ratio, distinct-value count, fraction below
  the physical floor).

## 7. Seeds

**5 seeds** for the primary arm (H=72): `--online_series_seed 41..45`. Reported
as median and full range of the per-seed nMSE, never the best seed. 3 seeds for
the secondary arm (H=24). One arm × 5 seeds; no configuration sweep — this
campaign is testing a corrected pipeline, not searching hyperparameters.

## 8. Pre-declared success criterion

Judged on H=72, primary rows, median over the 5 seeds:

* **PASS** — median ONE-NAS nMSE **< the persistence gate recomputed on the same
  rows**, *and* the collapse diagnostics are clean (prediction/target sd ratio in
  [0.3, 3.0], < 1% of predictions below the physical floor).
* **STRONG PASS** — median ONE-NAS nMSE **< daily-retrained ridge on the same
  rows** (i.e. architecture search buys something over a linear model with the
  same adaptation cadence). This is the result worth writing up.
* **FAIL** — median ≥ persistence, or the collapse diagnostics trip. If it fails,
  the reported conclusion is that ONE-NAS does not beat persistence on this
  series under a correct pipeline. That is a publishable negative result and it
  will be reported as one; it will **not** trigger another convention search.

The gate is fixed before the run. No post-hoc span, mask, row, or warm-up choice
may be substituted for it.

## 9. Alignment mechanism (why these numbers can be trusted)

`run_v2.sh` writes `run_manifest.json` **before the binary starts**, recording
`L, T, V, OFFSET, window_step, ISL, POP, ELITE, BP, seed`, the ordered segment
map, the data span, the mask name, the normalisation constants, the full argv,
the binary's SHA-256 and the ONE-NAS git commit, plus SHA-256 of `index.csv`,
`windows.csv` and `prep_meta.json`.

`score_v2.py` takes **nothing from the environment** — it reads the manifest.
It refuses outright if the prep files' hashes no longer match (i.e. the prep was
re-run after launch). Alignment is the closed form of §1, not a search:

```
e = g + T + V  →  windows.csv[e] → (seg_id, tgt_start_seg_row)
file row i     →  target seg_row = tgt_start_seg_row + i + 1
                  issue  seg_row = target seg_row − H
```

The `expected_N2O` column is still checked against the sidecar, but now only as
an **assertion**: a generation that does not reproduce it to 1e-4 is *refused*,
never guessed at. In the smoke test the worst error over 10 generations was
**exactly 0.0** — the closed form is not approximate.

## 10. Smoke-test evidence (job 20008088, ~3 s of compute)

4 islands, pop 2, elite 2, bp 2, 10 generations — a deliberately trivial search,
run only to prove the pipeline works end to end. **These numbers are not
evidence about the method.**

```
provenance OK: index.csv / windows.csv / prep_meta.json match the manifest
found 10 generation prediction files
aligned 10 generations, refused 0  (worst expected-column error 0, tol 1e-4)
covered rows 1430   span 2023-05-15 03:05 .. 2023-05-20 02:55

prediction raw  mean +0.4088  sd 0.6268  min -0.5831  max +2.2079
target     raw  mean +0.3276  sd 0.3909  min -0.0208  max +1.5940
sd ratio pred/target 1.6037
distinct predicted values: 1428 of 1430
```

What this establishes: 64 segment files ingest; episodes are numbered as
predicted; the closed form aligns every generation exactly; baselines join on
identical rows; and **the collapse is gone** — v1's final global-best emitted sd
0.001 against target sd 0.056 (ratio 0.018); this emits ratio 1.60 with 1,428
distinct values in 1,430 rows. The run *loses* to persistence (2.73 vs 1.94) as a
3-second random-ish search should, on a 5-day slice where persistence itself
scores 1.94. 14.8% of predictions fall below the physical floor — expected of an
untrained network, and a diagnostic the production run must improve on.

### Second smoke run at the exact production configuration (job 20008113)

16 islands, pop 5, elite 8, bp 10 — the pre-registered settings — still only 10
generations, 18.1 s. Same 1,430 rows as above, so the two are directly
comparable:

| | tiny (4/2/2/2) | **production (16/5/8/10)** |
|---|---|---|
| nMSE | 2.7315 | **2.1987** |
| pred/target sd ratio | 1.6037 | **1.0935** |
| below physical floor | 14.76% | **9.86%** |
| distinct predictions | 1428 / 1430 | **1430 / 1430** |

Both still lose to persistence (1.94) at 10 generations out of 610, which is
what should happen. The point is that the alignment path exercised here is
byte-identical to the production one — `worst expected-column error 0` again —
and that more search budget moves every diagnostic the right way.

## 11. What is NOT verified

Listed so nobody assumes otherwise:

1. **That a properly-trained ONE-NAS beats anything.** No production run has been
   made. The smoke test says the plumbing works, nothing more.
2. **Long-run numerical stability** over 610 generations with BPTT across 144
   timesteps — untested at this L. Gradient pathology remains possible.
3. **The chosen `L=144` / 12 h cadence is optimal.** It is *justified* (inside the
   measured good regime, keeps 85.7% of valid data, 696 windows) but not tuned,
   deliberately. `L=288` (daily cadence, one full diurnal cycle, 325 windows) is
   the pre-declared fallback if 144 shows gradient pathology.
4. **That `T=80` is the right burn-in/batch size.** `T` is both the burn-in window
   count and the per-generation batch; 80 was chosen to put the first scored
   window in mid-May 2023 while leaving 610 generations. Not swept.
5. **Whether excluding the 18,262 valid rows in short runs biases the scored
   population.** Short valid runs sit next to dead-sensor stretches and may be
   systematically different; this is not characterised.
6. **The residual sensor rail.** 10–12 valid rows read 12.0 mg/L in 2022-11.
   None survive into a usable segment for H=72, but the mask does not catch
   them and a different `L`/`H` could admit them.
7. **The covariate clip change** (±8 → ±4) is reasoned, not measured; no ablation.
8. **The v1 forensic audit's findings** are assumed, not reproduced here.
9. **Multi-seed variance** — unknown; 5 seeds is a guess at adequate.
10. **Nothing about Agtrup**, the second plant.

---

## 12. The exact next command, and what it costs

Nothing below runs without human approval.

NOTE: the login shell's default `python3` has no pandas. Always use
`PY=/apps/anvil/external/apps/conda/2024.02/bin/python3` (which is what
`run_v2.sh`, `prod72.sbatch` and `verify_multifile.sh` already do internally).

```bash
PY=/apps/anvil/external/apps/conda/2024.02/bin/python3

# 1. prep + baselines -- ALREADY DONE for H=72 and H=24.  Rerun only if the
#    prep changes; note that doing so invalidates any run whose manifest pins
#    the old hashes, and score_v2.py will refuse those runs (by design).
ssh anvil "H=72 L=144 T=80 $PY ~/wwtp_scripts/v2/prep_v2.py"
ssh anvil "$PY ~/wwtp_scripts/v2/baselines_v2.py \
             /anvil/scratch/x-jchang5/wwtp/v2/h72_L144"

# 2. THE COMMAND TO RUN once approved -- the production arm, H=72, 5 seeds.
#    prod72.sbatch scores itself on completion.
ssh anvil 'sbatch --array=41-45 ~/wwtp_scripts/v2/prod72.sbatch'

# 3. re-score / aggregate by hand if needed
ssh anvil "for s in 41 42 43 44 45; do $PY ~/wwtp_scripts/v2/score_v2.py \
             /anvil/scratch/x-jchang5/wwtp/v2/prod72_s\$s/run \
             /anvil/scratch/x-jchang5/wwtp/v2/h72_L144/baselines; done"
```

### Measured cost

Timing probe (job 20008113): **18.06 s wall for 10 generations at the exact
production configuration on 16 ranks** — 1.81 s/generation.

| | |
|---|---|
| Generations per seed | 610 (`max_generation`) |
| Wall per seed, linear extrapolation | 1,102 s ≈ **18 min** |
| Core-hours per seed, linear | 1,102 s × 16 / 3600 = **4.9 core-h** |
| **Growth correction** | genomes grow over 610 generations and `--control_size_method none` imposes no cap; the probe measures generations 0–9, when islands are still seeding from minimal genomes. Assume **3–5×** mean cost. |
| **Budget per seed** | **15–25 core-h** (1–2 h wall) |
| **Budget, 5 seeds (array, parallel)** | **75–125 core-h**, ~2 h wall |

`prod72.sbatch` requests `-N 1 -n 16 -t 12:00:00`, which is ~6× the upper
estimate — deliberate headroom, since the growth factor is the one number here
that is extrapolated rather than measured. The secondary H=24 arm (3 seeds, 631
generations) adds a further **45–75 core-h**.

Total for the pre-registered campaign: **≈ 120–200 core-h**. Do not launch until
this has been read and approved.
