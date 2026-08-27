# Why the ICAIF paper's B&H differs from every B&H in this repo

Verified 2026-08-27 against the ICAIF sources in `EXAMM-Extended` and
`Financial_toolbox` (both local).

## Reproduction

The ICAIF paper's benchmark columns are transcribed constants
(`EXAMM-Extended/scripts/stock_run/emit_paper_tables.py`, `BENCH` dict) from a
Financial_toolbox run whose buy-and-hold convention is the per-stock PRICE
return, averaged over the 50 names:

    (PRC[-2] - PRC[0]) / PRC[0]        # do_nothing_return.py; last day dropped

Applying exactly that formula to this repo's
`datasets/mid_highmid_price_clean/set{1..4}` reproduces ALL 12 published
(set, year) B&H cells to within 0.04 pts (e.g. set1 2024: paper -4.73,
reproduction -4.70).

## Why that convention understates B&H

1. CRSP `PRC` is NOT split-adjusted. Under the paper's convention the worst
   "losers" of 2024 are COO -75%, DECK -69%, TSCO -76%, ODFL -55% -- these are
   4:1 / 6:1 / 5:1 / 2:1 stock splits, not losses. (`RET` is split- and
   dividend-adjusted; this repo's books use RET_raw and are unaffected.)
2. Dividends are excluded (~1.5-2%/yr on these names).

Corrected total-return B&H on the same universes: 2022 -7.9, 2023 +13.3,
2024 +12.2 -> mean +5.9%/yr vs the published +2.64%/yr. The published "EXAMM
+11.02 vs B&H +2.64" margin roughly halves under the corrected benchmark; the
2022-only comparison (+15.46 vs -10.05) loses ~2 pts. The published daily-EW
row (-4.78%/yr) is hit harder: a daily-rebalanced book on split-unadjusted
prices books each split as a real one-day loss.

## Model rows are exposed too

`Financial_toolbox/portfolio.py` prices every position in shares at raw `PRC`
(buys at price, clears at price), so any position held across a split day
books a fake +/-50-85% move (fake loss on longs, fake gain on shorts). EXAMM
holds ~5 days in the ICAIF paper, so its net-return cells carry this noise;
splits occurred inside the test years (e.g. TECH 4:1, 2022-04, set3).

## Consequence for cross-paper comparisons

Use total-return B&H everywhere. Three B&H series now exist and none should be
casually compared:
  * ICAIF published: price-return, split-unadjusted, no dividends (biased down)
  * `icaif_protocol.py` benchmarks: total-return, $100 restarted per window
  * `yearly_econ` ew_buy_hold: total-return, book runs continuously from 2020
    (drifted notional scales later years)
