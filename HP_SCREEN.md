# HP_SCREEN.md — pre-registered hyperparameter screen (committed before launch)

Committed BEFORE any cell below is launched or scored. Companion to
PRIMARY.md; the paper reports every cell in this file regardless of outcome.

## Purpose and span

Single-knob screening of ONE-NAS hyperparameters against the frozen primary,
run and scored ENTIRELY on the baselines' tuning span (2016-01-01..
2019-12-31, the Amendment 6 clock: NTW 511/493/501/536 for sets 1-4, 201
generations). No cell touches 2020-2024. Adoption of any winner into the
scored configuration requires a subsequent dated amendment to PRIMARY.md,
written before the adopted configuration is scored on 2020-2024.

## Design

- Launcher: scripts/pooled/anvil/factorial_v2.sbatch with SPAN=tune16;
  submit lines in scripts/pooled/anvil/hp_screen_submit.sh.
- Replication per cell: panels set1-set2, seeds 42-44 (6 run-pairs/cell).
- Control cell C0 = the frozen primary exactly: TARGET=RET_CS, SELECT=mse,
  NTS=2000, elite 8 / generated 5, repopulation 50, PER alpha 0.6 lambda
  0.007, bp 10, L=40, V=5, 8 islands. (NTS is stated explicitly because the
  launcher's default is 200; the registered primary is 2000.)
- Every cell varies ONE axis from C0 (the single pre-declared interaction
  cell W1-TS varies two, for the stated mechanistic reason).

## Cells

Wave 1 -- fitness signal and target (root cause: val-MSE's rank-correlation
with realized return is +0.02; rank-based statistics detect structure MSE
cannot):
- W1-S1  SELECT=ic_gated (halflife 8, gate 1.5)
- W1-S2  SELECT=ic (ungated)
- W1-T1  TARGET=RET_CS5 (5-day cross-sectional target; horizon-matches the
         books' 10-day holding and the rising multi-horizon IC)
- W1-TS  TARGET=RET_CS5 + SELECT=ic_gated (declared interaction: a
         horizon-matched target may change what the fitness signal can see)

Wave 2 -- published ONE-NAS configuration values never tested on equities:
- W2-R1  elite 5 / generated 10 (the published ratio; ours inverts it)
- W2-R2  elite 5 / generated 15
- W2-F1  repopulation_frequency 25
- W2-F2  repopulation_frequency 100
- W2-N1  NTS=600 (published-scale subsequence count)
- W2-G1  ROUNDS=2 (two generations per window)

Wave 3 -- replay and training budget:
- W3-P1  per_alpha 0.4
- W3-P2  per_alpha 0.8
- W3-P3  per_lambda 0.021
- W3-B1  bp_iterations 20

Conditional (not launched with this screen): an NCL cell (--ncl_lambda)
requires the ncl branch's binary; if run, it is declared by amendment here
first.

Axes deliberately NOT searched, as already resolved by prior campaign
evidence: islands (swept 8/16/20/40; Amendment 6), sequence length (15 vs 40
flat, t=0.32), num_validation_sets (5 vs 20 flat), node-type library
(restriction hurts, P6), window geometry (P1 flat), ensemble width/combiner
(saturated).

## Objective and gates (fixed now)

Primary objective, identical to the baselines' tuning protocol and Amendment
6: mean daily cross-sectional rank IC over 2016-2019, pooled across the
cell's panels and seeds. Economic co-primary, reported for every cell: the
registered sleeves book (top-10, H=10, netted TC/PRC) net % and Sharpe on
the same span.

KEEP rule per cell, paired against C0 on identical (panel, seed) cells
(n=6 pairs): paired mean dIC >= +0.0020 with t >= 2. A cell that fails on
IC but shows paired dSharpe >= +0.10 with t >= 2 is flagged for the
confirmation stage but not auto-kept. All other cells are reported as
negative results.

## Confirmation and adoption

Kept cells are combined into one candidate configuration (single-axis
winners merged; if W1-TS wins over its marginals, it supersedes them). The
candidate is confirmed at 10 seeds x 4 panels on 2016-2019 against C0 at
equal replication. Only a confirmed candidate may be adopted, via a dated
PRIMARY.md amendment committed before it is scored once on 2020-2024. If
nothing survives, the screen is reported as a negative-results table and the
registered primary stands.

## Budget

15 configurations x 2 panels x 3 seeds = 90 runs at the tune16 clock (201
generations, roughly 2/3 of an eval-span run each).
