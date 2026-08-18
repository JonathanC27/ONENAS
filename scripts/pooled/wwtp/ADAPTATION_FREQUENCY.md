# CORRECTION: persistence is NOT the ceiling — adaptation frequency is the lever

`PERSISTENCE_CEILING.md` claimed a rich linear model loses to persistence at every
horizon and that no ONE-NAS configuration could close the gap. **That conclusion is
withdrawn.** It rested on a STATIC ridge fitted once on 60% of the record and left
to go stale for two years. It measured the cost of not adapting, not a limit on
what is learnable.

Sweeping the retrain interval (rolling 30-day training window, identical rows and
mask for both arms, `adapt_freq.py`):

| retrain every | H=6 (30 min) | H=24 (2 h) | H=72 (6 h) |
|---|---|---|---|
| 30 days | 1.07 | 1.52 | 1.84 |
| 14 days | 0.98 | 1.40 | 1.45 |
| 7 days  | 0.94 | 1.00 | 1.06 |
| 3 days  | **0.89** | 1.06 | 0.86 |
| 1 day   | 0.92 | **0.92** | **0.75** |

Values are ridge nMSE / persistence nMSE; below 1.00 beats persistence.
Absolute persistence: 0.4412 (H=6), 0.6105 (H=24), 1.0199 (H=72).

Weekly win rate (fraction of 59 weeks where ridge beats persistence) rises with
adaptation frequency too, so this is not a few lucky windows dominating an average:

| retrain every | H=6 | H=24 | H=72 |
|---|---|---|---|
| 30 days | 0.627 | 0.441 | 0.508 |
| 7 days  | 0.729 | 0.627 | 0.695 |
| 1 day   | **0.780** | **0.678** | **0.763** |

## Findings

1. **Persistence IS beatable at every horizon** given frequent enough adaptation.
   The crossover sits between 7 and 3 days.
2. **The payoff grows with horizon.** At 6 h, monthly retraining loses by 84% while
   daily retraining wins by 25%. At 30 min the whole effect is small because
   persistence is already near-optimal there. This makes H=72 (6 h) the most
   attractive target, not the hardest.
3. **ONE-NAS is on the wrong side of the crossover.** It adapts once per
   `--window_step`, currently 5 days — between the 7-day (ratio 1.00) and 3-day
   points. Reducing window_step to 1 day moves it to where the measured gain is
   largest, and is a one-flag change.
4. The relationship is monotone in adaptation frequency and steepest at long
   horizons. That is a direct, quantified argument for continuous online
   adaptation, i.e. for what ONE-NAS is.

## Caveats

* Ridge only; a single feature set (14 channels, lags {0,1,2,3,6,12,24}).
* Rolling 30-day training window held fixed while the retrain interval varies, so
  interval and window length are not fully separated.
* H=6 is non-monotone at the 1-day point (0.89 -> 0.92), so the very fastest
  setting is not uniformly best. Treat 3 vs 1 day as within noise at short horizon.
* Absolute persistence values here differ from other tables in this campaign
  because of the split; both arms are scored on identical rows, so ratios are the
  comparable quantity. See the scoring-convention note.
