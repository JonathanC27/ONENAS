# PRIMARY_HOLDOUT.md — pre-registered one-shot confirmatory holdout (2015-2019)

> **ADDENDUM (2026-08-15, same day, CANCELLATION).** The confirmatory ONE-NAS
> holdout campaign is cancelled by author decision. At cancellation time no
> ONE-NAS run had touched 2015-2019 and the full 2015-2019 baseline table had
> not been read by anyone (only the declared set1 peek). The paper's evidence
> base is the 2020-2024 development span, reported as such, plus the live
> pilot. The already-computed 2015-2019 BASELINE table (baselines only, tuned
> strictly OOS on 2011-2014) may now be read and used descriptively (e.g. a
> regime-robustness appendix for the baselines); no ONE-NAS 2015-2019 result
> will exist, so no confirmatory claim is made and none of the endpoints below
> will be evaluated. The submit-ready campaign script
> (scripts/pooled/anvil/holdout_b1.sbatch) is retained but must not be
> submitted without a new dated amendment here.

Committed 2026-08-15, BEFORE the 2015-2019 baseline table completed
(`results_econ_2015/` suite in flight at commit time; only the set1 peek below
had been seen). Companion to PRIMARY.md (commit ac58a56), which pre-registers
the system configuration; this file pre-registers the holdout evaluation.

## Why a holdout

2020-2024 is the development span: the CS rank-normal target, core7 features,
ensemble definition, and sleeve book were all selected while looking at it.
2015-2019 has never been touched by any ONE-NAS design decision and is the
only clean span left before the live pilot. It is spent exactly once, on the
terms below, and the result is published unconditionally — pass or fail.

## Peek ledger (declared)

1. 2026-08-15: set1 baselines peeked (ridge, STR1/5, naive, AR, LSTM,
   EW buy&hold) after re-tuning strictly out-of-sample on 2011-2014.
2. 2026-08-15 (in flight): full 4-panel baseline table, baselines only.
   NO decision rule about the ONE-NAS holdout is keyed to this table; the
   endpoints below were fixed before it completed.

No ONE-NAS run has touched 2015-2019. No ONE-NAS design decision postdates
peek (1).

## Frozen configuration

The PRIMARY.md ac58a56 primary, unchanged: core7 panels and inputs, RET_CS
target, L=40 / step 5 / V=5, NTS=2000, PER, MSE selection, 8 islands /
generated 5 / elite 8 / repopulation 50, island-champion rank-mean ensemble,
sleeves H=10 top/bottom-10 raw book with netted TC/PRC costs.

Total generations extend from ~300 to ~550 only so the weekly generation clock
covers 2015-01-01..2019-12-31 (warmup/burn-in through 2014). Nothing else
changes. NO exploratory improvement (signal smoothing, vol-scaled sizing,
H-strip blends, NCL/decorrelation selection, restricted search) may touch the
holdout, regardless of how it performs on 2020-2024.

## Endpoints (fixed now; all reported with sign and CI regardless of outcome)

PRIMARY: paired difference in daily rank IC (lag 1), ONE-NAS island-champion
ensemble MINUS periodic ridge yearly, both scored 2015-2019 under the
identical prequential protocol and identical booking code, paired per
panel x seed cell against the ridge run on the same panel.

SECONDARY (fixed at commit time):
1. Paired IC difference vs the fixed-architecture control ensemble
   (control2, same frozen config family), run on 2015-2019.
2. Paired IC difference vs online LSTM.
3. Pooled ONE-NAS ensemble rank IC (reported with CI; explicitly NOT a
   success criterion on its own — IC>0 vs zero is not the paper's question).
4. Net %, Sharpe, MDD on the identical sleeves H=10 book, plus the
   per-calendar-year slice (econ/yearly.py).

Power note, stated in advance: cells share panels and trading days, so the
effective sample is ~4 panels, not 12 cells; pooled IC SE is realistically
~0.0025-0.003. The design is marginal for detecting half-strength transfer
(+0.005 -> t~2) and the paper will say so.

## Procedure

1. Launch the 12-run confirmatory campaign (4 panels x seeds 42-44) on Anvil
   with the frozen config, target ~2026-08-20.
2. Score ONCE through the existing code path (score_stream + econ/rebook.py +
   econ/yearly.py) against the already-scored frozen baselines in
   results_econ_2015/. No reruns, no additional cuts, no new arms added to
   the comparison after this commit.
3. Write the outcome into the paper as-is. If the primary is negative, the
   paper reports it and stands on the framework, the diagnostics, and the
   pilot.

## What the live pilot is (and is not)

The pilot (Alpaca paper account, ~13 trading days before the deadline) is a
deployment-feasibility demonstration: implementation fidelity (live
predictions match a shadow rebook), latency/uptime of the weekly online loop,
and cost-model plumbing. Alpaca paper fills are simulated, so the pilot
validates neither slippage nor alpha (power ~6-7% at the historical edge);
the paper makes no alpha claim from it.
