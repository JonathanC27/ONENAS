# Persistence is the ceiling on Avedore N2O — and it is not a ONE-NAS problem

Ridge with 7 lags x 14 channels, 60/40 chronological split, valid-mask applied to
both endpoints, scored on identical rows as persistence (`delta_min.py`):

| horizon | persistence | ridge (levels) | ridge (delta) |
|---|---|---|---|
| 30 min | **0.2013** | 0.2772 | 0.2770 |
| 2 h | **0.4632** | 0.6240 | 0.6239 |
| 6 h | **0.9097** | 1.4827 | 1.4826 |

## Two findings

**1. The delta reparameterisation is a no-op.** Predicting the increment
`y(t+H) - y(t)` and adding back `y(t)` gives 0.2770 vs 0.2772 — identical to four
decimal places at every horizon. This was the leading hypothesis for why learned
models lose (that a levels model burns capacity rediscovering `y(t+H) ~ y(t)`).
It is wrong here, and the reason is mechanical: the feature set already contains
`N2O_l0`, so a linear model can represent the persistence solution exactly. The
reparameterisation moves no information.

**2. A rich linear model LOSES to persistence at every horizon**, by 38%, 35% and
63%. Combined with the earlier baseline table (whole-stream, monthly-retrained:
persistence 0.1445, ridge 0.1492, GBT 0.1875) that is three model families, two
parameterisations, and both static and periodically-retrained regimes — all
losing to the trivial baseline.

## Why this is NOT a ONE-NAS hyperparameter problem

ONE-NAS is improving steadily as its configuration is corrected and enlarged
(H=24: 2.199 at 4 islands mis-specified -> 1.135 at 8 -> 0.753 at 16 with the
horizon fixed). It is converging toward the same place ridge already sits (0.624),
and ridge loses. Tuning ONE-NAS harder buys movement toward a ceiling that is
itself below the gate.

## The distinction this exposes, which matters for the diagnostic paper

Avedore PASSES the searchability screen: validation-to-test rank correlation is
+0.617 mean / +0.828 median across configs (vs +0.021 on equities), and
consistent with that, ONE-NAS improves monotonically with island count here
whereas on equities the champion DEGRADED with island count.

But searchability and learnability are ORTHOGONAL:

* **searchable** = you can reliably tell good configurations from bad ones.
* **learnable**  = the best findable model beats the trivial baseline.

Avedore N2O is searchable and (at these horizons, with these features) not
learnable. Equities were neither. The diagnostic as currently formulated measures
only the first, so it must be paired with an explicit trivial-baseline check
before any claim that architecture search will pay off in a domain. That is a
genuine refinement of the contribution, and this dataset is the evidence for it.

## Caveats

* One feature set (14 channels, lags {0,1,2,3,6,12,24}) and one linear model
  family. A materially different representation could change this; the GBT
  per-window result (median 0.448 vs ridge 0.630 across 23 windows) shows
  nonlinearity helps *relative to ridge* without closing the gap to persistence
  whole-stream.
* The 60/40 chronological split straddles known dead periods; persistence is
  scored on identical rows, so the comparison is fair, but absolute levels differ
  from other splits in this campaign (see the scoring-convention note).
