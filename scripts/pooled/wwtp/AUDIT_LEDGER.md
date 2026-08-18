# Forensic audit: what is proven, what is void

Full audit of the Avedore N2O pipeline. Supersedes several claims in
TANH_SATURATION.md and ADAPTATION_FREQUENCY.md. Audit scripts on Anvil at
/anvil/scratch/x-jchang5/wwtp/audit/.

## The conclusion that SURVIVES

**ONE-NAS vs persistence is the one row-for-row fair comparison in the study**
(score_onenas.py computes persistence on exactly the rows ONE-NAS covered, and
persistence needs no training). ONE-NAS loses it in **every run at every
checkpoint**. That headline stands. What is void is every ONE-NAS-vs-*fitted*-
baseline claim, because of the masking asymmetry below.

## PROVEN

**Lineage reconciles exactly**: 831,557 raw -> 210,528 5-min bins -> 178,536 fed to
ONE-NAS (the 2022-10-01 left-trim, 31,992 rows) -> 127,266 of those valid.
Resampling drops nothing.

**The flatlining is REAL, not a pipeline artefact.** My forward-fill hypothesis is
REFUTED. Of 39,908 exactly-equal consecutive pairs, **98.2% are genuine multi-sample
means**. The raw 2-minute series is **34.8% self-identical** — flatter than our
5-min grid, so averaging *reduced* flatness. 178 runs of >=2 h sit exactly at the
sensor rail -0.0237268526107072; the longest is **105 hours** of one bit-identical
value. `n2o_valid` removes almost all of it (22.4% -> 5.4% consecutive-equal); it
is simply never applied to the training input.

**Normalisation is centred on a dead sensor.** make_onenas_fix.py fits median and
percentile span on ALL burn-in rows, of which only **21.3% are valid**. The median
used as the origin is -0.0114 mg/L; on valid rows it would be +0.0153. The lower
scale anchor p1 is **bit-identical to the sensor rail**.

**ONE-NAS has zero gap awareness** (time_series.cxx:241-370): no timestamp column,
no NaN/mask concept, every CSV row is one timestep. The 51,270 invalid rows and a
31-day linearly-interpolated hole are indistinguishable from real data.

**Multi-file segmentation WORKS and is the fix.** process_arguments.cxx:376-416:
`current_row` resets per file, so windows provably never cross a file boundary; the
--time_offset truncation is applied per file (time_series.cxx:1168-1183), so no
target is taken across a segment boundary. Unequal file lengths are fine when
--pooled_panel is absent.

**--time_offset H is a genuine H-step-ahead task**, no double shift
(time_series.cxx:479-503).

**The collapse mechanism, with a natural experiment.** Validation windows landing
in the Nov-2023 dead stretch: 51 generations have a fully dead validation window,
first at generation 491. fix72 saturates at 492-493, locks at -0.9667 by 522, and
**never recovers even after the sensor comes back at generation 553**. Two selection
flags turn degenerate there: `--selection_metric mse` has no gradient on a
zero-variance window, and `--max_pred_sd_ratio 3.0` (selection_metric.cxx:274-286)
**inverts into a pro-constant filter** — any genome with real dynamics exceeds 3x
the near-zero target sd and takes a +1e6 penalty, while a flat constant is never
rejected however wrong. With `--repopulation_frequency 0` there is no escape.
h24big separates the two causes cleanly: saturated by the tanh ceiling at gens
200-250, genuinely RECOVERS and tracks well at 300-475, then re-collapses at
generation 500 when the validation window hits 0.00 valid.

## VOID or WRONG — corrections to what I reported

**fix24 0.753 and fix72 1.948 are INVALID.** They are partial-run scores taken
before the collapse. The on-disk full-run values are **fix24 13.795** and
**fix72 21.785**. (Even at the truncated cut-offs both still lost to persistence.)

**"13.8-17% of the target outside the tanh band; perfect oracle scores 0.2367"** is
OVERSTATED — that was unverified comment text in make_onenas_fix.py. Measured:
**5.51% outside**, perfect-clipped oracle **0.1726**. The max |y| = 3.33 is right,
and the fixed scale genuinely gives 0.0000% out of band.

**"Skill vs ONE-NAS's own naive column"** is meaningless as a forecast measure.
onenas_island_speciation_strategy.cxx:1039 sets `naive_pred = test_output[j-1]` —
the target one row EARLIER, i.e. information from the future relative to the
forecast origin. It is an oracle. My -36.8 -> -788,254 trajectory indicates
divergence and nothing else.

**"46,000 invalid rows in training caused the collapse"** — the row gap is real
(51,270, 28.7%) but it is NOT the trigger. It is present at every generation
including the healthy ones; h24big tracks well for 175 generations despite it. The
trigger is the dead validation window.

**persistence = 0.1445 is unreconciled.** It appears nowhere on disk;
sensitivity.py was written specifically to chase it and reports 0.3925 for the same
task. Nearest on-disk value is 0.1438. Do not quote 0.1445.

**The adaptation-frequency cadence labels are WRONG.** adapt_freq.py compacts to
valid rows and then slices the retrain loop in COMPACTED indices, so `step=days*288`
is 288 *valid rows*, not one calendar day. At 63% validity "daily" is really ~1.6
days and jumps months across a dead episode. The self-check shows it: 59 "weeks"
for a 731-day span (~104 calendar weeks). Ridge and persistence are on identical
rows so the RATIO is fair and the direction is probably right, but the cadence axis
must be relabelled before any claim rests on it.

**The masking asymmetry is CONFIRMED**: baselines.py/conventions.py/adapt_freq.py
all train on masked rows; the ONE-NAS input is unmasked. Refinement: only
adapt_freq.py masks BOTH endpoints. Elsewhere `n2o_valid` flags the origin t while
the target is t+H, so **7,147 rows (5.6% of the "clean" set) grade forecasts
against dead-sensor targets**.

**Alignment warning was real and unacted-on**: 6 generations (537-542) align at
offset -3 instead of +1, exactly where the target is bit-constant so the exact-match
search ties. 864 rows mis-scored by 20 minutes; immaterial to totals.

## Scope limit on the searchability result

rho = +0.617/+0.828 is reproducible (clean basis +0.642/+0.794) but MIS-SCOPED for
this purpose: it measures val->test rank transfer for ridge on **7-day validation
and 7-day test** windows. ONE-NAS validates on 5 x 12 h = 2.5 days and tests on a
single 12-hour window. The estimate comes from windows ~14x larger and far less
noisy than the ones ONE-NAS actually selects on, so it does NOT establish that
ONE-NAS's selection transfers.

## Two ONE-NAS bugs (independently confirmed)

Out-of-bounds read in do_repopulation (onenas_island_speciation_strategy.cxx:885 vs
rank_islands() line 214) segfaulted two jobs at generation 350 while Slurm still
showed RUNNING. Unbounded memory growth OOM'd a third at 618 generations.
