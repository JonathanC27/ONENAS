# Independent design review: DO NOT LAUNCH AS DESIGNED

Four changes required first. None costs meaningful compute; one is a single CLI
flag that converts this campaign into the decisive experiment for the diagnostic
paper.

## 1. The pre-registered gate is computed on the wrong rows

PROTOCOL.md §4 states the rule correctly (score every arm on exactly the rows the
run covers). §5's table violates it: baselines_v2.py scores over all
baseline-eligible rows (n=93,456, from 2023-04-01) while ONE-NAS covers episodes
85-695 (n=84,320, from 2023-05-15). The 9,136 extra rows carry **41.7% of the
total target sum-of-squares** — they are the April-May 2023 high-N2O excursion
(mean 0.4638 vs 0.1070 on the scored span) — and inflate the denominator for
every arm.

| H=72 arm | PROTOCOL §5 | On ONE-NAS's actual rows |
|---|---|---|
| persistence (the gate) | 0.8384 | **1.0510** |
| ridge_d1 | 0.6701 | **0.8980** |
| ridge_d3 | 0.7909 | **1.0946 (loses to persistence)** |
| ridge_d7 | 0.7300 | 0.9497 |
| ridge_d30 | 1.4665 | 1.5827 |

Same defect at H=24 (persistence 0.3684 -> 0.4915; ridge_d1 0.3171 -> 0.4351).

## 2. At H=72 a zero-parameter moving average beats everything pre-registered

| H=72, ONE-NAS's rows | nMSE |
|---|---|
| **0.5*(persistence + 7-day rolling mean)** — zero parameters | **0.7488** |
| 7-day rolling mean | 0.8539 |
| daily-retrained ridge (14 signals, 6 lags, harmonics) | 0.8980 |
| in-sample mean (no-skill line) | 1.0000 |
| **persistence — the pre-registered gate** | **1.0510** |

**Persistence has NEGATIVE SKILL at H=72** — worse than a constant. So the
pre-registered PASS criterion is satisfiable by emitting a well-placed constant,
which is precisely the degenerate failure mode the collapse diagnostics exist to
catch. And the STRONG PASS (beat ridge) is cleared by a moving average.

Paired 14-window moving-block bootstrap over 611 windows:
* roll7 vs persistence: ratio 0.8125, CI [0.737, 0.972], wins 64.3% of windows.
* ridge_d1 vs persistence: ratio 0.8544, **CI [0.699, 1.017] includes 1**, and
  ridge **loses the median window** (42.2% win rate). The pooled ridge advantage
  is carried entirely by a few windows where persistence explodes.

At H=24 the ordering is normal. The two horizons are qualitatively different and
the primary one is the anomalous one.

## 3. The "5 seeds" are not seeded

`--online_series_seed` seeds only the PER episode sampler
(time_series/online_series.cxx:118-121). The RNG driving the architecture search
— mutation, crossover, island selection — is **wall-clock seeded**
(onenas/examm.cxx:68, onenas/onenas.cxx:76), and genome RNG is wall-clock seeded
and truncated to int16_t (rnn/rnn_genome.cxx:127). No run is reproducible; the
manifest's `seed` field is misleading provenance; an anomalous seed cannot be
re-run. Benign for variance estimation, unacceptable for a no-rebuttal venue.

## 4. Seed count: 5 is not enough. 10 staged to 20.

Kish effective n on the 611 scored windows is **52.1**, not 84,320 rows — six
windows carry 27.1% of persistence's total SSE. Irreducible row-sampling SD of any
arm's pooled nMSE is ~0.12 before a single seed is drawn. And the dominant
seed-level variance is a discrete catastrophe, not Gaussian jitter: in v1 fix72,
**53.6% of generations emit a global best with prediction sd < 0.01**. Run-level
scores are bimodal (~2 vs ~22); a 5-seed median reports which mode got 3 votes.
At sigma_seed ~ 0.3 (the realistic planning number) n ~ 24 is required.

Report the paired RATIO to each baseline, not raw nMSE (paired block-bootstrap SD
0.082 vs 0.12 marginal).

## 5. The searchability number that justified this domain was measured at the wrong horizon

rho.py hardcodes **H = 6 (30 min)**. ONE-NAS runs at H=72 (6 h). Four mismatches
in total, and the horizon one was previously unnoticed:

| axis | rho was measured at | ONE-NAS operates at |
|---|---|---|
| horizon | **H=6 (30 min)** | **H=72 (6 h)** |
| validation window | 7 days | 5 x 12 h |
| test window | 7 days | one 12 h window |
| candidates | ridge on random feature subsets | evolved recurrent architectures |

At 30 minutes persistence is near-optimal, so every candidate retaining an
autoregressive feature is near-identical; and a random 3-of-19 feature subset often
contains no N2O feature at all, so the Spearman measures a broken/not-broken
partition rather than discrimination among near-equals. **+0.617/+0.828 is not
evidence that ONE-NAS's selection transfers.**

## 6. The fix: one flag that was never switched on

`--write_elite_predictions` is already parsed (common/process_arguments.cxx:173)
and implemented (onenas/onenas_island_speciation_strategy.cxx:679), called at line
822 AFTER evaluate_elite_population(validation) and select_elite_population(). It
writes every elite of every island — 16 x 8 = 128 candidates per generation.

Adding it to the production argv yields **610 generations x 16 island champions =
9,760 (validation, out-of-sample) pairs at ONE-NAS's exact operating point**, for
~128 forward passes per generation and ~300 MB/seed. This is the measurement that
settles whether the diagnostic's GO for Avedore was real.

## 7. Searchability vs learnability is currently an UNEARNED rescue

The distinction is logically real (different functionals, explicit counterexamples
both ways). But as deployed here: it was introduced in the very document reporting
Avedore is not learnable; **only the diagonal of the 2x2 has ever been observed**
(NB-201 both, equities neither); the off-diagonal cell rests on Avedore alone,
whose searchability leg is mis-measured per §5; and the corroborating "island count
helps here" claim was retracted by TANH_SATURATION.md (the 4->8->16 progression
confounds island count with the normalisation fix, and the 8->16 gain is mostly
avoided catastrophe: August 81.15->3.06 while March 0.92->1.61 and April
0.68->0.95 got WORSE).

Ruling: keep the distinction, demote the claim to a proposed refinement with a
stated test, and let §6's measurement decide. If rho collapses toward 0 at the true
operating point, the Avedore GO was a false positive — which is still publishable,
and arguably better: a diagnostic that catches its own false positive by being
measured correctly beats one that was never checked.

## 8. Span and collapse — pre-declared now

**Score-through. Never abort, never choose afterwards.** The favourable v1 cut
(G<=350, giving 0.669) is the **argmin over every prefix tested**
(0.795/0.712/0.669/0.760/0.733/0.720/0.766/3.08/8.36/13.80). Sensor death is the
modal WWTP failure, not an exotic event; a controller that pins to a rail for 60
generations and never recovers is a safety-relevant deployment property, and
reporting the pre-collapse number describes a system that does not exist.

The "it's a v1 bug v2 fixes" defence is NOT available: the audit's mechanism is a
**low-variance validation window**, and a fully-valid window can still be
near-constant. v2 retains `--max_pred_sd_ratio 3.0` (which inverts into a
pro-constant filter there) and `--repopulation_frequency 0` (which removes the
escape route). The collapse mechanism is not provably fixed.

Pre-registered: no run stopped early for performance; primary = pooled nMSE over
the FULL covered span; mandatory co-reported median per-window ratio with sign test
and win rate; mandatory collapse census per seed (first generation with prediction
sd < 0.01 x target sd, longest contiguous run, recovery generation); **no seed ever
dropped for collapsing — the fraction of seeds that collapse is itself a headline
number.**

## 9. The missing arm

A **fixed-architecture online control** — same online loop, same cadence, same
ensembling, hand-picked architectures, no evolution. Not in the protocol. The
equity campaign already ran it and found fixed-architecture controls BEAT the
evolution ensemble. A NAS paper cannot omit the control the authors' own prior
campaign showed to be decisive.

## 10. Required before launch, in priority order

1. Add `--write_elite_predictions` (zero compute; re-run the 10-generation smoke
   test to confirm global-best outputs are byte-identical).
2. Amend §5 to the corrected row set; add the rolling-mean family; replace the
   gates with three tiers: trivial blend 0.7488, then ridge 0.8980, then the
   fixed-architecture control.
3. Go to 10 seeds staged to 20 with a pre-declared escalation rule, and seed the
   search RNG (onenas/examm.cxx:68). ~300-500 core-h.
4. Add the fixed-architecture control arm.

Run M0 first — re-evaluate v1's 295 saved fix72 genomes with
`rnn_examples/evaluate_rnn` (no training, a few core-minutes) — as a cheap read on
whether rho at the true operating point is near +0.6 or near 0.

## Unverified

Seshan et al. 2025 (doi 10.1016/j.watres.2024.122754) could not be retrieved;
the comparison to it is reasoning, not established, and must be checked against the
PDF before writing.
