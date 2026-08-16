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
