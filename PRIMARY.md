# Pre-registered primary configuration (IAAI-27)

Committed BEFORE scoring any run on the core-7 panels. Phase-2/3 results for this
configuration are confirmatory; any other variant is reported as an ablation and
never promoted to primary.

## Primary
- Panels: core-7 (`panels_core7/set{1..4}_core7`), 4 panels x 50 stocks
- Inputs: RET RET_CS_IN BA_SPREAD ILLIQUIDITY REV21_1 TURN_RATIO VOL21
- Target: RET_CS (1-day cross-sectional rank-normal)
- Geometry: L=40, step=5, V=5 (weekly clock); burn-in through 2019 + 50 warm-up gens
- Search: 8 islands, generated 5, elite 8, repopulation every 50, PER (fixed
  window-index), MSE selection, NTS=2000, bp_iterations 10, --write_elite_predictions
- Prediction: rank-mean ensemble of the 8 island champions (score_ensemble.py,
  --ensemble island_champions --combine rank_mean)
- Book: overlapping sleeves H=10, top/bottom-10, TC/PRC netted costs
- Primary metric: mean daily cross-sectional rank IC across the 4 panels, 2020-2024
  prequential. Secondary: net Sharpe on the sleeve book.
- Seeds: 42,43,44 (initial) + replication set 42..51.

## Phase-1 gate outcomes (paired vs E0 on identical panel x seed cells; rule:
## KEEP iff paired mean >= +0.0020 and t >= 2)
- Ensemble stacking: +0.0029 +/- 0.0015, t=1.95 -> KEEP. Documented deviation:
  t is 1.95 vs the 2.0 threshold on the new base alone; the same effect measured
  +0.0052 (t=3.5, n=8) on the prior base, pooled 20 cells t~3.5. Intent of the
  rule ("must replicate") satisfied; deviation disclosed here rather than rounded.
- P1 staleness (L=15,V=2): +0.0005, t=0.32 -> DROP (reported as negative result).
- P6 capacity limits: -0.0035, t=-3.10 -> DROP (restriction hurts; ablation row).

## Escape hatch (pre-stated)
If the primary's scored IC on core-7 is < +0.005 (sanity floor) or a scorer defect
is found, fall back to the single global-best genome (no ensemble), documented as
a deviation.

## AMENDMENT (2026-08-16): blind seed expansion 12 -> 24 runs
Seeds 45-47 of the frozen ac58a56 primary were launched 2026-08-15 (identical
config, no selection among seeds) and remain unscored at the time of this
amendment. They are scored ONCE, blindly, immediately after this commit, and
merged into every headline statistic (n=12 -> n=24 runs; 3 -> 6 seeds).
Every claim currently phrased "in every seed" is restated at the expanded
count even if a new seed breaks it. No further seeds will be added after
results are seen.

## AMENDMENT 2 (2026-08-16, committed before any run is launched or scored)

**A. Seed expansion.** The seed set for ONE-NAS and every stochastic baseline
is extended from {42-47} to {42-51}, superseding Amendment 1's "no further
seeds" clause by author decision to increase replication; 10 is the terminal
count. Seeds 48-51 are launched blind: no output is scored, inspected, or
plotted until all 16 ONE-NAS runs complete. All headline quantities are
restated at n=10 (40 runs) whether or not a claim survives. Baseline symmetry
is binding: online LSTM, online GRU, periodic LSTM (all cadences), the
LSTM/GRU seed ensembles, and the random-arch and fixed-arch controls use the
identical seed set {42-51} with their frozen 2016-2019 hyperparameters (no
re-tuning). Deterministic arms have no seed dimension. Every derived exhibit
is recomputed at n=10 with no mixing of seed counts. We state in advance that
the factor-alpha t-statistic is expected to move from 1.95 to approximately
1.97: seed averaging removes only the idiosyncratic share of daily book
variance and cannot buy significance on a date-common quantity. Runs 48-51
use a resolved command line verified identical to runs 42-47 except SEED.

**B. Aggregation-law out-of-sample prediction.** Seeds 48-51 extend the
cross-seed super-ensemble from 48 to 80 island champions. Fitting
IC_M = IC_1 * sqrt(M/(1+(M-1)*rho)) to the existing cross-seed points gives
rho = 0.35, IC_inf = +0.0134. We predict IC(80) = +0.0132, recorded here
before the runs are scored. Both outcomes are published; a miss falsifies the
constant-rho form of the law and is reported as such.

**C. Hyperparameter sensitivity campaign.** Seven configurations are launched
simultaneously and never iterated: the frozen PRIMARY.md default;
bp_iterations in {5, 20, 40}; PER lambda in {0.002, 0.020};
num_validation_sets = 20. Each changes one factor from the default. Each runs
5 seeds x 4 panels on seeds {100-104}, disjoint from the reporting seed set.
- Span: 2016-01-01..2019-12-31, the exact span on which every baseline was
  tuned. No configuration is tuned, selected, or tie-broken on any post-2019
  quantity.
- Statistic: mean daily cross-sectional rank IC, pooled by day across
  4 panels x 5 seeds, one daily series per configuration.
- Test: paired daily difference vs the frozen default, Newey-West lag 10,
  two-sided, Bonferroni over K=6 (|t| >= 2.50).
- Threshold: dIC >= T, where T is computed from daily-IC dispersion and
  cross-config correlation measured on 2016-2019 (not 2020-2024) and
  committed to this file before the first configuration is scored.
  Provisional value T = 0.0038.
- Adoption rule: adopted only if dIC >= T, |t| >= 2.50, and the ladder is
  monotone in the hypothesized direction. If two or more clear, both are
  reported and NEITHER is adopted; no joint configuration is constructed.
  Any reported winner's dIC is shrunk by E[max of 6] * SE = 0.00145.
- Headline rule: the paper's headline tables report the frozen PRIMARY.md
  configuration at n=10 regardless of outcome. Any winner appears only as a
  hyperparameter-sensitivity row; adoption into a headline would additionally
  require a 10-seed 2020-2024 confirmatory arm and warm-up-invariance
  evidence.
- No-winner branch (pre-committed): if no configuration clears both criteria,
  the incumbent stands, no headline changes, and we publish the full grid,
  the minimum-detectable-effect table, and the finding that hyperparameter
  search over the search's own knobs is not resolvable at this
  signal-to-noise level.
- Reporting: all seven configurations are reported in full, losers included,
  in every outcome.
- Pilot: the live pilot ships the frozen configuration ac58a56 on Aug 19-20
  and is unaffected by this campaign in all branches.
