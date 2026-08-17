# The diagnostic passes its positive control (NAS-Bench-201)

The reviewer objection to a screen validated only on equities is: *"you have shown
sensitivity and zero specificity — you validated only where the answer was already
known to be no."* This is that objection answered.

**Artefact.** `NAS-Bench-201-v1_1-096897.pth`, md5 `55e847143ce1f7c2d89b676f6b096897`
(last six hex digits match the official filename, as the distribution intends).
Dong & Yang, *NAS-Bench-201*, ICLR 2020, arXiv:2001.00326. 15,625 architectures,
hp=200, 3 seeds (777/888/999). Every architecture's validation AND test accuracy is
precomputed, so the whole protocol is table resampling — no training.

## The four predictions, all confirmed, on all four benchmark settings

| setting | P1 Spearman | P2 argmax k=1 → k=2500 | P3 argmax − mean | P4 argmax − random |
|---|---|---|---|---|
| CIFAR-10 (cross-run) | **0.9841** | 87.46 → 94.22 (**+6.76**) | **+7.15** ± 0.004 | **+7.19** ± 0.184 |
| CIFAR-10 (within-run) | 0.9964 | 83.47 → 91.19 (+7.71) | +7.77 ± 0.004 | +7.72 ± 0.175 |
| CIFAR-100 | 0.9934 | 61.52 → 72.87 (+11.35) | +11.46 ± 0.007 | +11.26 ± 0.169 |
| ImageNet16-120 | 0.9948 | 33.36 → 46.48 (+13.12) | +12.90 ± 0.006 | +12.97 ± 0.134 |

Seed-noise ceiling (CIFAR-10): sigma_between 12.80 vs sigma_seed 0.56, ICC for a
single seed 0.998. The ranking signal dwarfs seed noise, which is exactly the
regime the diagnostic is supposed to detect.

## Side by side with the negative case

| measurement | equities (ONE-NAS) | NAS-Bench-201 (CIFAR-10) |
|---|---|---|
| (iii) validation→truth rank corr | **+0.021** (and −0.261 on the fixed-pool control) | **+0.984** |
| (ii) argmax over a larger pool | **flat**: +0.022 over a 10× pool | **rises**: +6.76 |
| (i) champion vs population | **population wins** (+15.7 vs +45.6 net) | **champion wins** by +7.15 |
| random pick vs argmax pick | **random WINS** (1.185 vs 1.105) | **argmax wins** by +7.19 |

Every measurement flips sign or direction between a domain where architecture
search demonstrably works and one where it does not. The screen says GO on
NAS-Bench-201 and STOP on equities, which is what a diagnostic has to do to be
worth anything.

## Honest notes

- "mean-of-k" here is the mean TEST ACCURACY of the k sampled architectures, not an
  ensemble — accuracy has no ensembling analogue matching the equity book's
  rank-mean. The argmax-vs-random-pick contrast (P4) is the cleaner comparison and
  is reported alongside.
- The scale of the gap is far larger on NB-201 than any plausible finance effect.
  That is the point (these are opposite regimes) but the paper must not imply the
  magnitudes are commensurable.
- ImageNet16-120 was expected to be the noisiest, intermediate case. It is not — it
  shows the LARGEST separation. The diagnostic does not appear to track a gradient
  of searchability across these three, because all three are firmly in the
  high-signal regime. A synthetic SNR dial remains the way to demonstrate the
  boundary; it was not run.
