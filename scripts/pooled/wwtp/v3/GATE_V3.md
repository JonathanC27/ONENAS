# Corrected data and gate — and the primary horizon is wrong

Independently re-derived. The agent could not locate the review documents (my
briefing gave repo paths relative to the wrong root) and worked from source and
data alone — yet reproduced 1.0497, 47.8%, 0.773 vs 0.936, 11-of-14 months, and
5,599 rows / 2 runs / 37 windows exactly. Those are independent replications, not
echoes.

Pipeline validated first: prep_v3.py rebuilds the 5-min grid from aved_raw.csv and
ASSERTS it reproduces v1 (0 mask rows differ, all 14 signals to 2e-6), and a
`centred_v2_fixed` config reproduces v2 bit-for-bit (W=696, max_gen=610, centre
0.853937, first scored window 2023-05-15 03:00).

## 1. The non-causal mask is real but IMMATERIAL

| | centred (registered) | trailing (causal) |
|---|---|---|
| valid rows | 132,869 (63.11%) | 132,827 (63.09%) |
| rows that flip | — | 4,016 (1.91% of grid) |
| usable segments (H=72) | 64 | **67** |
| emitted rows | 104,832 | **105,336 (+504)** |
| windows | 696 | **698** |

**Effect on the gate: 0.1%** (persistence 1.0497 centred vs 1.0486 trailing; scored
rows overlap 97.9%). The causal mask GAINS data. Detection latency is the analytic
12 h in both directions and the two error types are near-symmetric in count.

**Switch to trailing** — not because it is more accurate (it is a wash) but because
the hindsight is free to remove and it deletes an unanswerable reviewer objection.
My earlier framing of this as a serious problem was overstated.

## 2. Reclaimed pre-span helps the burn-in, hurts the target map — unless p1/p99

37 windows recovered, provably burn-in-only (T_MODE=preserve sets T=117; the scored
row set is byte-identical to v2's 87,230).

Burn-in excursion share 96.7% -> **68.1%**. But June 2022 holds the series maximum
(2.0694), so under the registered p0.5/p99.5 map the centre moves 0.8539 -> 1.0231
and scored-span target sd falls 16%. Under **p1/p99** the map is invariant to the
reclaim (centre 0.8705, scale 1.4856 — within 2% of v2) with 0% of scored targets
outside the tanh band. Baseline tables are byte-identical either way, so this
affects trainability only, never the gate.

## 3. The corrected gate, H=72 (n=84,250, 612 windows)

ratio = paired SSE ratio to persistence; CI = 2000-replicate moving-block
bootstrap, block = 14 windows.

| arm | nMSE | ratio | 95% CI (paired) | median window ratio | win % |
|---|---|---|---|---|---|
| persistence | 1.0486 | 1.000 | — | 1.000 | — |
| climatology (in-sample mean, ORACLE) | 1.0000 | 0.954 | — | 1.346 | 42.8 |
| climatology (causal expanding mean) | **1.3204** | 1.259 | [1.035, 2.071] | 4.559 | 24.8 |
| rolling mean 7 d | 0.8529 | 0.813 | [0.739, 0.975] | 0.717 | 64.1 |
| **0.5 x (persistence + 7 d mean)** | **0.7477** | **0.713** | **[0.688, 0.761]** | 0.705 | **78.9** |
| ridge_d1 | 0.9192 | 0.877 | [0.710, **1.092**] | **1.400** | **39.4** |
| ridge_d3 | 1.0129 | 0.966 | [0.729, 1.180] | 1.413 | 38.2 |
| ridge_d7 | 1.0995 | 1.049 | [0.733, 1.283] | 1.439 | 37.6 |
| shrunk causal k=0.6 | 0.8424 | 0.803 | [0.765, 0.935] | 1.365 | 40.0 |

Four things the registered table hid:

1. **Persistence is below the no-skill line** (1.0486 vs 1.0000), worse than the
   in-month constant in 11 of 14 months, and its own CI is [0.845, 1.191] — the
   gate value is +-16%.
2. **A genuinely causal climatology scores 1.3204** — worse than persistence. v2's
   "climatology_trailing_mean" was the in-sample mean, an oracle, ==1.0 by
   construction. The no-skill line is not free.
3. **Ridge does not survive.** Every ridge arm's paired CI crosses 1.0. ridge_d1's
   aggregate 0.877 comes entirely from a few windows: **median per-window ratio
   1.40, loses 60.6% of windows.** Ridge is not usable as a gate component at H=72.
4. **A zero-parameter rule beats everything non-oracle** — the blend, whose two
   constants were fixed a priori, is the only arm whose CI is comfortably clear
   of 1.

Extra defect in the registered table: its arms are not on one row set among
themselves — ridge_d7 covers 93,240 rows where persistence is 0.8429, ridge_d30
covers 89,136 where persistence is 1.0723. The single "vs persistence" column
mixes three reference values.

## 4. Ridge fairness confirmed, but trimming is the wrong fix

ridge's lag-72 crosses the window start on **40,186 of 84,250 rows (47.7%)**.
Ratio to persistence: **0.7840** where ridge has out-of-window history, **0.9671**
where it does not. Edge collapses 6.5x.

But `wu=71` fixes only ridge's 72-row lag — the 7-day (2,016 rows) and 30-day
(8,640) rolling means have a j-independent advantage the trim never touches, and
the blend's ratio barely moves (0.684 -> 0.741). The trim also discards 47.7% of
rows and moves the gate 6.4%, and it widens every CI (rolling_d7 stops being
significant).

**Headline wu=0, with wu=71 as a mandatory declared companion.** Binding rule: any
"ONE-NAS beats ridge" claim must hold at wu=71, the only row set where the two
have comparable history. Claims against persistence/blend/rolling means may be made
at wu=0. The clean long-term fix is a WINDOW-LOCAL baseline family, not a row trim.

## 5. Three-tier gate (config h72_L144_trailing_full_preserve_q1)

| tier | criterion | number |
|---|---|---|
| **T0 FLOOR** (necessary, NOT sufficient, not publishable as a positive) | median nMSE < 1.0000 (the constant) and < 1.0486 (persistence), collapse diagnostics clean | **1.0000** |
| **T1 PASS** | median < 0.7477 AND upper CI of paired ratio to the blend < 1.0 | **0.7477** |
| **T2 STRONG** | T1, and < 0.7477 at wu=0 and < 0.7273 at wu=71, and win rate vs blend > 55%, and better in >= 10 of 14 months | **0.7477 / 0.7273** |

T0 uses the CONSTANT, not persistence, because at H=72 persistence sits above the
no-skill line. T1 is not satisfiable by a degenerate estimator by construction — it
IS the best degenerate estimator.

## 6. THE PRIMARY HORIZON IS WRONG

PROTOCOL §3 chose H=72 primary because "the H=72 persistence gate is the weakest."
On the correct rows that is exactly why it is the wrong primary: **the gate is
weakest there because it is broken there.**

H=24 (n=87,110, 632 windows): persistence **0.4901** against a 1.0000 no-skill line
— a genuine skill line, worse than the in-month constant in only 5 of 14 months.
Rolling means are all WORSE than persistence (1.59-2.01x). Ridge_d3 **0.4389**
genuinely beats it. No zero-parameter rule gets near.
Tiers: T0 0.4901, T1 0.4472 (causal shrunk k=0.8), T2 0.4389 (ridge_d3).

**Recommendation: H=24 primary, H=72 secondary** — or at minimum, the H=72 claim
must be stated against the blend (T1), never against persistence.

## 7. Not verified

The review documents (could not locate them — everything above is independently
re-derived). No ONE-NAS run: the scored population assumes every generation lands
and aligns; if generations are refused the gate must be recomputed on the actual
subset. **T=117 as a ONE-NAS argument** — verified only that it preserves
max_generation and the scored windows; how `--num_training_sets` interacts with PER
sampling at 117 vs 80 is a run-side property and is UNCHECKED. Ridge's checkpoint-
phase sensitivity (d7 moves 0.949 -> 1.100 across variants) is reported but not
diagnosed. Per-month nMSE is unstable (low-variance months collapse the
denominator) and is diagnostic only.
