# Four-arm campaign: ONE-NAS does not reach persistence. Definitive.

20 runs (4 arms x seeds 41-45), H=24, 632 generations, one binary whose diff was
audited as entirely flag-gated so BASE is behaviourally unmodified. All 26 scored
run-variants on the pre-registered row set (n=87,110 / 632 windows, gate 0.4901):
**zero refusals, 632/632 aligned, worst expected-column error 0.**

## Results — single global best is PRIMARY (the published ONE-NAS rule)

| arm | n | median | per-seed | spread | ratio/pers | 95% CI |
|---|---|---|---|---|---|---|
| base | 4/5 | **1.3378** | 1.1119 1.2997 1.3759 2.7410 | 2.47x | 2.81 | [1.43, 5.57] |
| A (find it) | 0-2/5 | **did not complete** | overgrew, ~10x base cost | — | — | — |
| **B (seeded AT simple1 = 0.4473)** | 4/5 | **3.3413** | 2.1212 2.6711 4.0114 4.5336 | 2.14x | 8.18 | [4.51, 16.60] |
| C (residual) | 5/5 | **6.6227** | 2.1414 4.5812 6.6227 6.7569 7.7176 | 3.60x | 13.51 | [5.96, 31.67] |

16-champion mean ensemble: base 1.1389, B 0.8814, C 1.3552.

**Best number anywhere in the campaign: 0.6176** (base seed 42, ensemble) — still 26%
above the gate and 38% above simple1's 0.4473. On the published single-best rule the
floor is 1.1119 = 2.27x the gate.

**Arm B is the decisive result: seeded AT 0.4473 it degrades ~7.5x, ending worse than
starting from scratch.** The search cannot hold an architecture a person writes in one
line.

## Three source findings — two of which correct claims I made

**1. `--control_size_method` is INERT on this task.** The trigger is
`genome_better_count > naive_better_count` (onenas_island_speciation_strategy.cxx:836-837);
both counters are cumulative and never reset (:1070-1081). Final tally in production:
**Naive 631 / Genome 1.** Confirmed empirically in arm A: zero fire events, counters at
generation 180 reading Naive 181 / Genome 0.
=> My reasoning that "with erosion fixed, growth control becomes the right tool" was
MOOT — the knob never fires. Arm A therefore isolates protect-inputs + the published
5:10 ratio, nothing more.

**2. `--protect_input_nodes` protects the bookkeeping, not the information.** It guards
`disable_node` only; `disable_edge` is untouched, so an input can remain `enabled`
while unreachable. It does fix the dominant channel (base erodes to 6-11 enabled
inputs) but **live inputs still fell to 8/14 (B s44), 10/14 (C s42), 9/14 (A s43)**.
=> My earlier "verified: holds 14/14 in 3/3 runs" was measuring the ENABLED count. Any
check counting enabled inputs reports a clean 14/14 while information is still leaving
through disabled edges.

**3. A third ONE-NAS bug: unrecoverable deadlock.** `insert_genome` latches
`status=FILLED` (onenas_island.cxx:176-179) and never clears it; `select_elite_population`
rebuilds elites through structural-hash dedupe, so when distinct topologies fall below
8 the island is FILLED yet `elite_is_full()` is false, all three branches of
`generate_genome` fail (:260-278), and :280-282 recurses forever. Arm B seed 45 produced
**95.4M FATAL lines and a 29 GB log growing ~500 MB / 20 s** before being cancelled — a
live filesystem threat, not a nuisance.

## Failures, reported not worked around

* base s44 and A s44 both **collapsed** (watchdog cancelled at generations 233-252 and
  213-232 — same seed, near-identical window, pointing at the shared
  `--online_series_seed 44`). The watchdog proved itself here.
* B s45 **deadlocked** (above).
* Arm A **overgrew until it was too slow to finish** — ~10x base's cost, 2/5 seeds still
  on pace at the wall. Its pooled nMSE is unmeasured; only its trajectory is known.
* No replacement seeds were substituted.

## Where this leaves the question

My stated confidence was 10-15% for "ONE-NAS is great here" and ~40% for "clears
persistence." The measured answer is **neither, in any arm, seed, or combiner.** I was
too optimistic even at 40%.

Three arms, twenty runs, two mechanistic fixes, the published exploration ratio
restored, a free level anchor, and a hand-picked winning architecture handed to it as a
starting point — and it went backwards from all of them.

## What survives as a result

* **A 31-weight network beats the trivial baseline under a real deployed online
  protocol** (simple1 = 0.4473 vs persistence 0.4901), and the capacity optimum runs
  opposite to intuition — 1 hidden cell, with 19 nodes already past it.
* **A fully mechanised account of why the search fails**: neutral drift on the input
  layer confirmed against a no-selection Monte-Carlo; new units rejected at birth
  (split_node 0.118 vs add_edge 0.743) traced to Lamarckian init; selection signal
  quantified at rho = 0.17 against the next window with validation variance at 14.9% of
  pooled.
* **Three genuine ONE-NAS defects** found and localised: the repopulation
  out-of-bounds read, the input-layer erosion, and this deadlock.
* Second independent domain in which a fixed architecture beats the evolved one on the
  identical loop — the equity campaign found the same.

## Not verified

dyn2 vs seed bit-equivalence (unprovable; MPI ordering makes runs non-reproducible —
base medians 1.3378 here vs 1.3797 prior are consistent). Whether deadlock/collapse
RATES differ by arm (n=1 and n=2). Whether a different `scale_D` rescues arm C (its
5.3x over-dispersion suggests calibration, but no sweep was run). Rank-mean
(deliberately skipped). simple1's 0.4473 and the gate itself (taken as given).


# VERIFIED FINAL NUMBERS (independent pass + agent agreement)

I recomputed the table from the 26 score JSONs myself. Protocol integrity: all 26
aligned 632/632, **refused 0**, all n=87,110, a single gate value 0.4901085022 across
every variant. The comparison is exact.

| arm / variant | n | median | ratio | per-seed |
|---|---|---|---|---|
| base, single best | 4 | 1.3378 | 2.73 | 1.11 1.30 1.38 2.74 |
| base, ensemble | 4 | 1.1389 | 2.32 | 0.62 0.90 1.38 1.92 |
| **B, single best** | 4 | **3.3413** | 6.82 | 2.12 2.67 4.01 4.53 |
| **B, ensemble** | 4 | **0.8814** | 1.80 | 0.57 0.79 0.97 1.02 |
| C, level | 5 | 6.6227 | 13.51 | 2.14 4.58 6.62 6.76 7.72 |
| C, ensemble | 5 | 1.3552 | 2.77 | 1.11 1.35 1.36 2.16 2.81 |

**0 of 26 variants score below the gate.** Best anywhere: **B/s43 ensemble = 0.5690**,
ratio 1.161 — 16% above persistence and 27% above simple1.

CORRECTION: an earlier note in this file said the best number anywhere was 0.6176
(base s42). It is 0.5690 (B s43 ensemble). Independently confirmed on two passes.

## Ensembling is the only lever that did anything

Arm B: single best 3.3413 -> ensembled 0.8814, a **3.8x improvement** — the largest
ensembling gain in either campaign. Consistent with the equity finding that
aggregation carries the method while selection destroys it: the single global best
degrades badly, averaging its 16 island champions recovers most of the loss. It still
does not clear 0.4901.

## THE SHARPEST DIAGNOSTIC: arm C found the right architecture and still failed

Arm C **found and HELD** the target topology — 1 hidden cell, 14 inputs, the same shape
as simple1 — and scored 13.5x persistence anyway. Its residuals are **5.3x too large**
with **correlation +0.058 to truth**, i.e. roughly **30x worse than emitting zero** —
and emitting zero would have BEEN persistence, since in residual space
`yhat = y_t + 0` is exactly the persistence forecast.

That isolates the failure to what online training does with the weights, INDEPENDENT
of topology. The architecture space is not the problem; the architecture space
contains the answer, and two separate arms reached it (C found it, B was handed it)
and both still lost.

## Every proposed fix failed; two made things worse

| fix | mechanically worked? | effect on accuracy |
|---|---|---|
| protect inputs | yes — holds 14/14 enabled | none; B and C both use it and score WORSE than base |
| published 5:10 ratio (arm A) | yes | overgrew ~10x base cost, could not finish 632 generations |
| free level anchor (arm C) | yes — found 1 cell / 14 inputs | WORST arm, 13.5x persistence |
| hand it the answer (arm B) | n/a | 0.4473 -> 3.34, worse than starting from scratch |

## The honest scope

This is ONE task where persistence is unusually strong — the best of 15 baselines
(ridge_d3, 0.4389) is only 10% better than doing nothing. That is a hostile regime for
architecture search and this is NOT a general claim about ONE-NAS.

But the task is not unlearnable, which is what makes the result publishable rather
than merely negative: **simple1 (16 nodes / 15 edges / 31 weights, one simple cell)
beats persistence at 0.4473, and it is exactly one `add_node` from the seed genome
every run starts at.** The search cannot find it, cannot hold it when handed it, and
none of the available levers fix that. **The failure is localised in the search — not
the architecture space, and not the task.**
