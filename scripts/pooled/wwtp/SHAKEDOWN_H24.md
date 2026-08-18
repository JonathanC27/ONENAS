# H=24 shakedown: pipeline GO, one blocker, and the one-seed numbers

Both arms ran to full length. 20.8 core-h of the 50 budgeted. No multi-seed array
launched.

## The pipeline is sound at production scale — first time verified

| check | result |
|---|---|
| completion | ONE-NAS 632/632 (31m43s), control 632/632 (45m08s), both exit 0 |
| alignment | both **aligned 632, refused 0**, worst expected-column error **exactly 0** |
| provenance | 73 segment CSVs, 15 baselines, index/windows/prep_meta all hash-match |
| row set | n=87,110 (wu0) / 45,504 (wu71) — **identical to baselines_v3.json** |
| gate cross-check | internal vs preds_persistence.csv, **rel diff 0.00e+00** |
| watchdog | live on both, never fired, 0 non-finite generations |
| repopulation | 10 events at generations 150-600, distinct islands each time, zero FATAL |
| control invariant | topology frozen at 19 nodes / 60 edges / 0 recurrent across all 632 |

## One-seed numbers (single global best is PRIMARY — the published ONE-NAS rule)

Gate: T0 0.4901 / T1 0.4472 / T2 0.4389 (wu0); 0.5164 / 0.4684 / 0.4552 (wu71).

| arm | wu0 | wu71 |
|---|---|---|
| **ONE-NAS single global best** | **1.6827** | 1.7639 |
| ONE-NAS ensemble, mean | 1.4237 | 1.5736 |
| ONE-NAS ensemble, rank-mean | 2.2218 | 2.4438 |
| **Control single global best** | **0.6328** | 0.6386 |
| Control ensemble, mean | 0.8332 | 0.7861 |
| Control ensemble, rank-mean | 1.0351 | 1.0434 |

All six lose to all three tiers. **One seed is not evidence.**

Note the ensembling asymmetry: it HELPS ONE-NAS (1.68 -> 1.42) and HURTS the
control (0.63 -> 0.83). Consistent with ONE-NAS's members being individually
degenerate and the control's being individually sound.

## T=118 RESOLVED — it is a training hyperparameter, not a clock

Three effects, all from source:
* **Batch size.** online_series.cxx:230/281 draw exactly `num_training_sets`
  episodes; that batch is the genome's training set (onenas_mpi.cxx:362-367) and
  rnn_genome.cxx:1179-1211 runs bp_iterations epochs over all of them with one
  update each -> **BP x T updates per genome: 1180 vs 800, +47.5%**.
* **PER selectivity.** Pool is {i < g+T}, batch is T, so the sampled fraction
  T/(g+T) is larger at every generation — prioritisation is WEAKER, closer to
  exhaustive.
* **Recency tilt.** priority = 1/(mse+eps) * exp(-lambda(g - availability_generation));
  the T-1 episodes ahead of the counter have a POSITIVE exponent, max boost
  exp(lambda(T-1)) = **2.27x at T=118 vs 1.74x at T=80**. Those forward episodes
  never receive a priority update, so they keep validation_mse = 1.0 permanently —
  a never-updated band of 117 episodes vs 79.

Does NOT invalidate the gate: prep_v3 emitted at T=118 and baselines_v3 computed
the gate on that same 632-window population, so gate and run are matched.

## Two blockers the shakedown caught before they cost anything

* **The control was unscoreable.** run_control.sh wrote `manifest/1`, which
  score_v3.py refuses outright — and prod_control.sbatch ends its scorer calls in
  `|| true`, so it would have silently produced NO SCORE. Promoted to schema 3.
* **The control ran a DIFFERENT BINARY** (~/ONENAS/build, built before the tree's
  last commit) from the search arm. Both now run the ONENAS_seed build, and the
  control now passes --examm_seed. Without this the comparison was invalid.

## THE REMAINING BLOCKER — the below-floor criterion fails the gate's own baseline

`MAX_FRAC_BELOW_FLOOR = 0.01` would FAIL **ridge_d3 — tier 2 of the gate itself**,
which puts **6.31%** of predictions below the physical floor (d1 6.15%, d7 7.23%,
d30 5.22%). Every persistence-family arm scores 0.00% because it copies observed
values and structurally cannot violate it. The 1% limit misses the gate's best
baseline by 6x, and it fired on both shakedown arms (12.49% / 10.50%).
**Decide this before five seeds, not after.**

Also: watchdog_v3.py:186 skips flat windows only when `sd_e <= 0.0`; generation 300
has target sd 1.11e-16 and produced ratios of 7.4e13 / 5.5e14. Verified to be a
flat TARGET, not exploding predictions (pred sd 8e-3 / 6e-2). Needs a relative
tolerance. This is the source of the "10^14" figures I reported earlier as
numerical instability — they were a denominator artefact.

## Cost — the 3-5x growth correction did not materialise

Measured: ONE-NAS **8.46 core-h/seed** (3.01 s/gen against a 3.0 s/gen probe),
control **12.03 core-h/seed**. The search arm is CHEAPER than the control because
it starts minimal while the control carries lstm4 from generation 0.
**Five seeds of both arms: ~102 core-h, ~45 min wall if run concurrently.**

    sbatch --array=41-45 ~/wwtp_scripts/v3/prod_h24.sbatch
    sbatch --array=41-45 ~/wwtp_scripts/v3/control/prod_control_h24.sbatch

## Verdict: GO on the pipeline, conditional on settling the below-floor criterion
