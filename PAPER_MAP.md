# PAPER_MAP.md — section map, claims, and the test behind each claim

Thesis: online neuroevolution earns its edge on equities as an ENSEMBLE
GENERATOR, not an architecture finder. The island-champion rank-mean
ensemble makes ONE-NAS the strongest system in a cost-aware prequential
protocol; the population's value is only accessible by aggregation, and the
always-online population appreciates with time.

House statistics (used everywhere): comparison unit = one training run;
paired per-(panel,seed) deltas collapsed to seed level, paired t (df=9);
date-clustered SEs for anything built from daily series; IUT (net AND
Sharpe) where a claim needs both metrics.

--------------------------------------------------------------------------
S1  Introduction — contributions
  (i) island-champion ensemble for ONE-NAS, (ii) cost-aware prequential
  protocol on CRSP mid-cap panels, (iii) mechanism study: aggregation not
  selection, (iv) online-continuity result. No tests.

S2  Method — ONE-NAS/EXAMM + the ensemble construction.
  Fig 1: pipeline (online clock, islands, champion rank-mean).
  Note member correlation 0.19-0.22 (measured) as the diversity source.

S3  Protocol — panels V1-V4, prequential 2020-24 clock, RET_CS target,
  sleeves book, netted TC/PRC costs, 10 seeds, pre-registration ledger
  (PRIMARY.md + amendments). Table: frozen config.

S4  Headline results
  T1  2020-24 headline table: ONE-NAS 8/20/40 isl vs online LSTM, online
      GRU, periodic-LSTM (all cadences), fixed/random-arch controls.
      Test: paired seed-level t vs each baseline. STATUS: DONE
      (results_econ/headline_table.json; 40isl vs LSTM dNet +12.0 t=3.5,
      dSharpe +0.19 t=3.1).
  T2  2022-24 window table vs prior paper (EXAMM, published B&H row).
      STATUS: DONE (paper/icaif_window_tables.tex).
  T3  Factor alpha with MKT + STR1 + STR5 reversal controls, Driscoll-Kraay
      SEs. STATUS: exists (results_econ/factor_alphas.csv); RE-RUN cheaply
      on the final table's constructions for consistency.
  F1  Cumulative net curves (net_curves.py). STATUS: regenerate for final
      arms. Local, minutes.

S5  Mechanism study (the paper's core)
  M1  PRIMARY TEST — island ensemble vs single champion, WITHIN-RUN paired:
      both prediction rules scored from the same runs; paired t over 10
      seeds, both 8isl and 40isl, on rank IC / net / Sharpe, 2022-24 and
      2020-24. STATUS: DONE (1bd1941; paper/m1_ensemble_table.tex).
      40isl 2022-24: dNet +23.0 (t=18.2), dSharpe +0.66 (t=14.7); gap
      widens 8->40 isl (winner's curse confirmed); replicates on the
      2016-19 tuning fleets (t 4.5-7.5 on economics).
  M2  Fitness signal is uninformative: rank-corr(val MSE, realized) +0.02;
      argmax over 10x pool +0.022 vs averaging +0.143; random pick beats
      argmax; NAS-Bench-201 positive control passes (harness valid).
      STATUS: DONE (bf07a9a, 7a64d90, 49192a6).
  M3  Boundary (report, don't hide): random-architecture online ensembles
      match evolved champions at the registered book (t=0.41); aggregation
      doubles Sharpe across ALL member-generation mechanisms. STATUS: DONE
      (BP8, 94a9d14).
  M4  Member-set robustness: island champions vs diverse-8 (loses,
      dIC -0.0007 t=-0.6), top-16, all-elites. STATUS: DONE (G1, ef33d66).
  M5  Width buys reliability, not IC: islands 8->40 SD(Sharpe) .171->.078,
      worst seed .41->.78; IC follows ensemble-averaging math (Amendment 3
      prediction +0.0120 vs measured +0.0123). STATUS: DONE.
  M6  Not "any ensemble": matched-member LSTM/GRU seed-ensembles lose on
      2022-24 under identical books (+16.0/0.61 vs +27.5/0.87). STATUS:
      DONE (books_vs_lstm.csv); caveat in text: ens rows have no seed
      dimension (report as untestable construction, cite balanced-draw
      matched-width analysis fa9843f).

S6  Online continuity
  C1  Cold 2022 start earns ~half the matured system on the identical
      window (+12.3 vs +26.5 pooled, frozen 8isl primary). STATUS: DONE
      (d7370a9); disclose generation-count confound as in the commit.
  C2  OPTIONAL (only if continuity is promoted to headline): 40isl
      cold-start cell, 12 runs on Anvil. NOT REQUIRED for the thesis.

S7  Sensitivity and ablations
  A1  Book geometry surface (top_k x H, 15 cells, no adoption). DONE
      (Amendment 5).
  A2  Book construction: Algorithm-1 degenerates on rank-normal signals
      (trigger fires daily, ~8%/yr drag); sleeves registered; banded
      (buy/hold-spread) +16.7 net t=9.6 at lower turnover — EXPLORATORY,
      no adoption. DONE (strategy_sweep.csv, banded25_yearly.csv).
  A3  HP screen (HP_SCREEN.md): 14 cells + C0 on the 2016-2019 tuning
      span. STATUS: DONE (8626ddb): ZERO KEEPs -- the registered primary
      wins its own screen; IC-based selection actively hurts (fitness-
      repair hypothesis refuted, confirming M2/M3); four economics-only
      FLAGs led by bp20, confirmation stage optional and unadopted.
  A4  Islands scaling table (8/16/20/40 with Amendment 6 protocol-symmetric
      selection). DONE (islands_sweep_final.txt).

S8  Limitations & honest accounting
  Dev-span disclosure (window inherited from prior protocol; full-span in
  appendix); cancelled 2015-19 holdout (documented, PRIMARY_HOLDOUT.md);
  survivorship bias in universe construction; single-market/single-frequency
  scope; prior-paper benchmark convention appendix — DECISION PENDING with
  the professor (ICAIF_BENCH_NOTE.md holds the reproduction).

--------------------------------------------------------------------------
NEW WORK QUEUE (in priority order)

  1. M1 full-n stitching + booking (Anvil ~1h + local ~30m). The paper's
     primary test. No new training.
  2. A3 HP screen launch (90 runs x ~40 min wholenode) + scoring vs C0.
  3. T3/F1 refresh (factor alphas, net curves) on the final arm set.
     Local, fast, do last when tables freeze.
  4. C2 40isl cold-start — only if continuity becomes a headline claim.
  5. Conditional: if the screen KEEPs anything -> confirmation at 10 seeds
     x 4 panels on 2016-19 -> PRIMARY.md amendment -> single 2020-24 run.
