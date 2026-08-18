# Adversarial code review: NO-GO as written, GO after 5 blocking fixes

Independent of the design review, and it converges on the same S1 blocker — two
reviewers computed the true gate separately and got **1.0497** and **1.0510**.

## The three claimed root-cause fixes: ALL PASS, verified by running the binary

**Segment split — PASS at the production config.** Job 20008230 on all 64 segment
files: `696 sets` matches per-file slicing exactly (concatenated slicing would give
727, so the count discriminates). All 696 dumped windows byte-match windows.csv
(worst |diff| 0.0); **0 windows escape their segment**. `e = g+T+V` confirmed at
the real config. Truncation exact: 0 of 64 segments have a window remainder.

**Valid-only normalisation — PASS.** Burn cut at source row 64,403 vs first scored
row 65,125 — no scored row touches any statistic. Centre +0.8539 (v1: -0.0114).
0.000% of scored target clipped or outside the tanh band. Zero NaN across all 64
segment CSVs.

**window_step == L — PASS.** Not overridable; confirmed in the real argv. Target
ranges of distinct episodes are provably disjoint and no training/validation input
range reaches a test target row for any H >= 0.

**Alignment is genuinely closed-form and load-bearing — PASS.** A +-1/2/3 row shift
gives max-error >= 0.00277 (27x tolerance), so **0/610 would silently pass**; an
episode off-by-one gives min error 0.0101, 0/1218 pass.

## S1 — BLOCKING: the gate is on the wrong rows and has no skill

baselines_v2.py scores every usable pair in the whole emitted series (93,456 rows,
including the 13k burn-in); score_v2.py scores only test-window rows (84,177).

| arm | PROTOCOL §5 | true scored rows |
|---|---|---|
| **persistence — THE GATE** | 0.8384 | **1.0497** |
| ridge_d1 | 0.6701 | 0.8976 |
| ridge_d7 | 0.7300 | 0.9493 |
| ridge_d3 | 0.7909 | **1.0944** |
| climatology (in-sample mean) | 1.0 | 1.0000 |

**Persistence is worse than climatology in 11 of 14 months.** §3's rationale for
H=72 primary inverts: claimed headroom 20% vs 14%, actual 14.5% vs 11.4% — and at
H=72 the gate is below no-skill while at H=24 it is a real gate.

**A two-parameter shrinkage rule earns STRONG PASS and passes every declared
collapse diagnostic:**

    k     sd_ratio   nMSE     beats gate?  beats ridge_d1?  sd-ratio in [0.3,3.0]?
    0.50   0.572     0.8649   YES          YES              PASS
    0.60   0.648     0.8480   YES          YES              PASS

The criteria cannot separate that from architecture search buying anything.

## S2 — BLOCKING: no collapse detection, and the logs are muted

`--max_pred_sd_ratio` rejection is one-sided (rnn_genome.cxx:1465) — a collapsed
genome (ratio 0) is always eligible. **47/610 generations have pooled validation
target sd < 0.010** and 140/610 < 0.020, against a scored-span sd of 0.173; there
the 3.0x cap forbids any genome with realistic amplitude. 98/610 test windows have
target sd < 0.01.

`run_v2.sh:74` sets `--std_message_level ERROR`, which suppresses WARNING and INFO
(log.hxx:120-127) — silencing "ALL elite genomes failed the exploding-prediction
guard" and every global-best update. **A 610-generation, 5-seed run would produce
essentially no in-flight signal.** score_v2.py checks collapse only after the fact
and never exits non-zero.

## S3 — HIGH: stale prediction files are silently scored (DEMONSTRATED)

prod72.sbatch does `mkdir -p "$OUT"` and never clears it. The expected-column
assertion is data-derived, so a file from any earlier run at the same
(DATA,L,T,V,H) passes. The reviewer replaced generations 0-4 with a collapsed
constant, left 5-9 original: score_v2.py reported **"aligned 10, refused 0, worst
error 0"** and **"ONE-NAS BEATS the persistence gate."** A 12 h timeout + requeue is
exactly how this happens.

## S4 — HIGH: non-finite predictions silently shrink the row set (DEMONSTRATED)

score_v2.py:147 `have = np.isfinite(pred)` defines the row set for EVERY arm.
Injecting NaN into 71 predictions dropped covered rows 1430 -> 1288 and moved the
gate 1.9433 -> 1.9820 with no warning. Violates §4's "no arm gets its own row set" —
ONE-NAS's own failures define it.

## S6 — MEDIUM: ridge is favoured, on 47.8% of rows

ONE-NAS's recurrent state resets at each test window (rnn.cxx:372-381), so at
timestep j it has seen only rows wL..wL+j. Ridge's N2O lags reach 72 rows back,
crossing the window start whenever j <= 71.

    rows                                    n       pers    ridge_d1  ratio
    ALL primary                           84,177   1.0497   0.8976    0.855
    j<=71 (ridge lags cross window start)  40,257   1.1289   0.8722    0.773
    j>71  (lags inside the window)         43,920   0.9823   0.9193    0.936

**Ridge's edge is 3x larger exactly on the rows where it holds information ONE-NAS
is denied.** The declared warm-up sensitivity (0/12/24/48) does not reach it.
Fix: add wu=71 and make that the headline fairness check, or drop ridge's N2O lags.

Everything else about baseline fairness is clean: identical feature source, lags
guarded in-segment, refit only on observed targets, identical join keys. ONE-NAS
adapts every 144 emitted rows vs ridge_d1's 288 — **ONE-NAS adapts twice as often.**

## S5 / S7-S10 — lower severity

Manifest does not close the score (baseline dir comes from argv, preds unhashed,
`total_generation` unrecorded, segment CSVs unhashed, `onenas_git_dirty` reports
False when git *fails*). Docs claim 87,840 scored rows; actual 87,230.
`climatology_trailing_mean` is the in-sample mean — an oracle, ==1.0 by
construction, and not trailing; a genuine causal trailing mean measures 1.3493.

## Flagged, could not close

**Mask provenance.** 609 of 13,444 distinct N2O values appear as BOTH valid and
invalid, so `n2o_valid` is context-dependent, not pointwise. If that context is
centred or forward-looking, segment boundaries encode "the sensor is about to die"
and the scored row set is selected with hindsight. Impact on ONE-NAS is probably
nil, but the selection effect on the scored population is real and inherited from
v1. **Locate the mask-generation script and confirm it is causal.**

The mask does not exclude all flatlines: 5.38% of valid rows exactly repeat the
previous row, longest exact-flat run inside valid data is **129 rows (10.8 h)** —
nearly a full window. That is the fuel for S2.

## Free improvement left on the table

`START=2022-10-01` discards 5,599 valid rows in 2 usable runs = **37 extra windows
of pure pre-span burn-in with zero lookahead risk**. Using them directly attacks
§2's acknowledged worst problem (burn-in forced into the atypical March-April
excursion, centre 0.854 while the scored span sits at -0.51 +- 0.13).

## Verified clean

No teacher forcing (rnn.cxx:592-608 ignores expected_outputs). No shifted inputs.
Recurrent edges strictly backward. `--normalize none` genuinely skips. Every scored
target AND its persistence anchor is valid — 87,230/87,230 on both endpoints; the
v1 "7,147 live-origin/dead-target" problem is gone. Numerator and denominator share
one row set; sd is ddof=0 on both arms.
