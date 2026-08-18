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
