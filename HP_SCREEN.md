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

## AMENDMENT (2026-08-30): cell W3-P0, declared before launch

One additional Wave-3 cell: W3-P0 GET=Uniform -- prioritized experience
replay OFF, i.e. the published ONE-NAS papers' plain subsequence sampling.
Rationale: PER exists only in the post-publication codebase (added
2025-06), appears in no ONE-NAS publication, and is listed in the paper's
settings table; its effect must be measured, not assumed. Same design,
objective, gates, and replication as every other cell (2 panels x 3 seeds,
tune16 clock, paired vs C0). Reported in full regardless of outcome.

OUTCOME (2026-08-30, scored on arrival): dIC -0.0024 (t=-1.9), dNet +8.7
(t=+2.6), dSharpe +0.34 (t=+2.5) -> FLAG (economics only), not KEEP.
PER mildly helps the registered IC objective and mildly hurts the book --
the campaign's IC/economics dissociation again. Conclusion for the paper:
PER is NOT load-bearing (removing it does not degrade the system; if
anything the book improves within noise at n=6); the settings-table row is
an implementation detail, ablated. Per the bp20 precedent, the economics
flag is not adopted without confirmation.

## Incident log

2026-08-27: the first mass submission placed --export after the script name,
so sbatch passed it as script arguments and every job ran at launcher
defaults (RET_CS + ic_gated, eval-span clock, seed 42). All 45 jobs were
cancelled within ~30 minutes; the only damage is to the PHASE-1 design
artifacts RET_CS_ic_gated/set1_seed42 and set2_seed42 (~85% of files
overwritten by the concurrent default jobs). Those were 1-seed,
pre-registration design probes whose conclusions are already recorded; the
screen's W1-S1 cell re-runs the configuration properly on the tuning span.
Two submit-script fixes followed: option ordering, and pinning SELECT=mse
(the launcher default is ic_gated) for every cell that does not override it.
No screen cell had produced scoreable output before cancellation, so no
gate is affected.

## RESULTS (2026-08-28, all 90 runs complete; scored per the gates above)

ZERO cells KEEP. C0 (the frozen primary) has the highest tuning-span rank IC
(+0.0172; net +23.2, Sharpe +0.81). Both selection-metric cells are strongly
NEGATIVE (ic_gated dSharpe -0.55 t=-3.1; ungated ic -0.90 t=-7.2): selecting
on validation IC overfits selection noise rather than repairing the fitness
signal. RET_CS5 improves the book (+8.1 net, t=+4.3) while degrading daily
IC@1 (slower signal; fails the registered objective). FLAGs (economics-only,
not auto-kept): bp20 (+11.8 net t=+4.6, +0.46 Sharpe t=+4.0), repop100
(+0.25 Sharpe t=+2.4), rounds2 (+0.21 t=+2.2), pa04 (+0.33 t=+3.6; its net
t=+184.9 is a tiny-variance artifact at n=3 seeds). Full table:
scripts/pooled/econ/results_econ/hp_screen_score.csv (commit 8626ddb).

Disposition: the registered primary stands, its configuration now
protocol-symmetric on this axis set. Whether to run the confirmation stage
on the flagged economics-only cells (bp20 strongest) is an open decision;
none may be adopted without that confirmation plus a dated PRIMARY.md
amendment, per the rules above.

## DECLARED EXPLORATORY EVAL-SPAN LOOK (2026-08-28, committed before launch)

By author decision, bp_iterations=20 — the strongest economics-only FLAG of
the screen — is launched on the evaluation clock (2020-2024, 10 seeds x 4
panels, ARM hp2024_bp20, all other flags the frozen primary) and scored on
2020-2024 and 2022-2024 paired against the existing C7_E fleet (identical
seeds/panels, bp=10), BYPASSING the confirmation stage. Standing terms:

- This is an exploratory measurement. It is reported in full regardless of
  outcome and is NOT eligible for any headline, adoption, or the paper's
  primary tables without the confirmation stage plus a dated PRIMARY.md
  amendment.
- Its selection provenance (best of 14 screen cells on the tuning span,
  FLAG not KEEP, n=6) is disclosed wherever the number appears; the
  reported delta must be read as carrying that selection.
- One scoring pass, both windows, both metrics families (rank IC + sleeves
  book); no re-scoring under other books.

OUTCOME (2026-08-28, one pass as declared): bp20 does NOT transfer.
Paired vs the primary over 10 seeds: 2022-24 net +17.6 vs +20.5
(d=-2.9, t=-0.95), Sharpe 0.61 vs 0.70 (d=-0.09, t=-0.92), dIC -0.0014
(t=-1.42); 2020-24 the same shape (all deltas mildly negative, none
significant). The screen's n=6 FLAG (+11.8 net, +0.46 Sharpe on 2016-19)
was selection-inflated, as the confirmation-stage rule anticipated. No
adoption; the registered primary stands; the confirmation stage for bp20
is moot. Full table: results_econ/bp20_eval.csv.

## AMENDMENT (2026-08-29): reporting venue

By author decision the paper reports this search at standard ML granularity
(setup paragraph + settings table, paper/hp_reporting.tex): search space,
tuning span, and selected values, with a pointer to this repository for the
complete per-cell results. The full-reporting commitment above is satisfied
by this file, results_econ/hp_screen_score.csv, and results_econ/
bp20_eval.csv remaining in the repository, referenced from the paper. No
result is altered; the selected configuration is the registered primary,
which predates and is unaffected by every look recorded here.

## Budget

15 configurations x 2 panels x 3 seeds = 90 runs at the tune16 clock (201
generations, roughly 2/3 of an eval-span run each).
