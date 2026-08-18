# The heavy tail explained — and why abstention backfires

Computed directly from the deploy cache (n=87,110 verified, persistence 0.4901085
reproduced to 7 decimals). Scripts on Anvil at ~/wwtp_scripts/v3/deploy/
{storm.py, what.py, abst.py, abst2.py}.

## 1. Storm mode does NOT explain the heavy tail — it is the OPPOSITE

| | value |
|---|---|
| mean SWM over all 632 windows | 0.0927 |
| mean SWM in the **worst 6** windows | **0.0000** |
| mean SWM in the best half | 0.1516 |
| Spearman(window SSE, window SWM fraction) | **-0.166**, p=2.9e-05 |
| mean per-window SSE, storm vs no-storm | **1.147 vs 2.396 — storm is 0.48x** |
| share of total error from storm windows | 3.9% (they are 7.8% of windows) |

**Storm windows are HALF as hard as normal ones.** My hypothesis that the
catastrophic windows were storm events is refuted. Worth reporting: it rules out the
obvious physical explanation.

## 2. Target VARIANCE explains it, and it hurts every model equally

| characteristic | worst 31 windows | all | lift |
|---|---|---|---|
| target variance | 0.1551 | 0.0140 | **11.1x** |
| target mean | 0.5420 | 0.1035 | 5.2x |
| target range | 1.1723 | 0.2819 | 4.2x |

Spearman(SSE, target variance) = **+0.762** (p=4e-121); vs range **+0.765**.

**The same windows hurt persistence**: Spearman(ridge SSE, persistence SSE) =
**+0.787**, and the worst-31-by-ridge windows carry **46.5% of persistence's error
too**. This is a property of the data, not of any model.

Temporally concentrated: octiles 1-2 (windows 123-280) carry **66.8% of all squared
error** with 3-6x the target variance of the rest.

## 3. Abstention BACKFIRES — and nMSE is an invalid metric for it

A causal signal genuinely exists: Spearman(previous-window variance, this-window SSE)
= **+0.562**; variance is autocorrelated at +0.754. So high-error windows ARE
predictable at issue time.

But abstaining on them makes the model **worse relative to persistence**:

| coverage | rows | MSE | nMSE | **ratio to persistence** | random control |
|---|---|---|---|---|---|
| 100% | 86,967 | 0.01661 | 0.4371 | **0.8958** | 0.8958 |
| 90% | 78,028 | 0.01247 | 0.5065 | **0.9827** | 0.8935 |
| 80% | 69,303 | 0.00923 | 0.6592 | **1.0689** | 0.8938 |
| 50% | 42,773 | 0.00619 | 0.9497 | **1.6554** | 0.8982 |

Random abstention stays flat at ~0.895 throughout, so the signal IS doing something —
it is just doing the wrong thing.

**Two findings here.**

**(a) nMSE is invalid for selective prediction.** It renormalises by the retained
rows' own variance, so abstaining on high-variance windows shrinks the denominator
faster than the numerator. MSE falls (0.01661 -> 0.00619) while nMSE RISES
(0.4371 -> 0.9497) on the identical predictions. Only MSE and the paired ratio to a
baseline computed on the same retained rows are valid.

**(b) The model's entire value is concentrated in the volatile windows.** On calm
windows persistence is nearly unbeatable, so abstaining on volatile windows throws
away exactly what the model is for.

## 4. The deployment framing this implies

Not "we know when we don't know" — the opposite:

> **The forecast earns its keep precisely when the plant is volatile**, which is when
> an operator actually needs it. On quiet periods nothing beats assuming no change,
> and the model correctly adds little. Reported as a ratio to persistence on matched
> rows, model value rises monotonically with target variability.

That is a stronger and more honest operational claim than an abstention curve, and it
falls out of the same measurements.

## Caveats

Measured with ridge_d3 as the reference model, not the HGB (whose causally-selected
number is still pending). The direction should hold — the effect is driven by data
structure that hurts persistence equally — but the magnitudes are ridge's. Abstention
was tested at WINDOW granularity on one causal signal (previous-window variance);
a row-level signal or a learned predictor might behave differently, though the
mechanism in (b) would still apply.
