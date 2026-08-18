# Root cause: ONE-NAS output nodes are tanh-bounded, and the target was scaled outside the band

Established by a forensic pass over all six wastewater runs. This supersedes two
earlier hypotheses of mine (a mis-specified horizon; invalid rows in training) —
both were wrong or unproven; see "Corrections" below.

## The mechanism, at source level

* Output nodes are created as `SIMPLE_NODE` — `rnn/rnn_genome.cxx:4113`.
* `RNN_Node::activation_function` applies `tanh` to every node that is not
  `OUTPUT_NODE_GP` / `INPUT_NODE_GP` — `rnn/rnn_node.cxx:82-88`.

So a ONE-NAS prediction **cannot leave (-1, 1)** in normalized units. Measured:
predictions in every original run are hard-bounded to exactly +-1 (raw -0.6367 /
+0.6139, identical across h6s, h24s, h72s, h24big).

But `make_onenas.py` scaled N2O by *half* the burn-in 1-99 percentile span
(0.6253), leaving **13.8-17.0% of the scored target outside the reachable band**,
up to +3.33 normalized.

## Two consequences, both measured

**A hard ceiling.** A *perfect but saturated* oracle — `clip(y, -1, 1)` — already
scores nMSE **0.2367**. That alone loses the H=6 gate (0.164) and consumes half
the H=24 gate. **No ONE-NAS model could ever have cleared the 30-minute horizon**,
regardless of search budget or configuration.

**A training pathology.** MSE backprop toward an unreachable target drives tanh
into saturation, where its derivative is 0; learning stalls and the genome latches
to a constant. At h24s generation 350 the global best emits the literal constant 1
for all 143 rows. Constant-output generations: h6s 2.0%, h24big 1.9%, h24s 14.0%,
**h72s 43.9%** (permanently collapsed from generation 213). The trigger is visible
in the logs: in generation block 75-99 the out-of-band target fraction spikes to
~60% and saturation jumps from ~0% to 45-73% in every run simultaneously.

## After the fix

Scale by `2*(p99-p1)` of the burn-in instead of `(p99-p1)/2` — 4x more generous,
still burn-in-only so leak-free — clipped at +-0.9. Scored-span target max |y|
drops 3.33 -> 0.832 with **0.0000% out of band**.

Matched span (50,193 rows, 2023-03-01 -> 2023-08-23), 16 islands, bp 10, elite 8:

| arm | islands | norm | nMSE | gate (persistence) | verdict |
|---|---|---|---|---|---|
| h24s | 4 | old | 2.1995 | 0.4550 | loses |
| h24big | 8 | old | 1.1180 | 0.4550 | loses |
| **fix24** | 16 | **fixed** | **0.6691** | 0.4550 | **loses 1.47x** |
| h72s | 4 | old | 3.3868 | 1.0407 | loses |
| **fix72** | 16 | **fixed** | **2.1398** | 1.0407 | **loses 2.06x** |

The fix helped enormously (H=24: 2.20 -> 0.67) and fix24 is the first arm to beat
a constant-mean predictor. It still misses the gate by 47% at 2 h and 106% at 6 h.

**Saturation runaway recurs at the new band edge.** Over the full covered span
through 2024-02, fix24 scores 13.80 (persistence 0.3736) and fix72 21.79
(persistence 0.8445), driven by a Nov 2023-Jan 2024 episode (fix24 Dec 2023 nMSE
19,447) where predictions pin to the new edge (+-2.49 raw) while the winter target
is small. So this is a recurring instability of the online search on this series,
not solely an artefact of the original scale.

## Two real ONE-NAS bugs found

**Out-of-bounds read in repopulation.** `do_repopulation`
(`onenas/onenas_island_speciation_strategy.cxx:885`) loops `i < islands_to_exterminate`
and indexes `rank[i]`, but `rank_islands()` (line 214) only pushes islands with
`get_erase_again_num() == 0` when `repeat_extinction` is false, so `rank` can be
shorter than `islands_to_exterminate`. Two jobs SEGFAULTED on rank 0 at generation
350 (the 7th multiple of `repopulation_frequency=50`) while Slurm still reported
RUNNING, burning 16 cores each as zombies. One-line fix: bound the loop by
`min(islands_to_exterminate, (int32_t)rank.size())`. Worked around with
`--repopulation_frequency 0`, which is explicitly supported (line 869).

**Unbounded memory growth.** `wwtp_h24big` died OUT_OF_MEMORY at 618 generations.
Long runs on `-p shared` need an explicit `--mem`.

## Corrections to earlier claims of mine

* **"The horizon was mis-specified."** WRONG, proven two ways: the ONE-NAS input
  CSV is byte-identical across horizons (md5 cf86fbd0...), so `H` in
  `make_onenas.py` only builds the sidecar and the horizon is set solely by
  `--time_offset`; and each run's effective offset recovered from its own outputs
  gives P=6, 24, 24, 72 respectively, consistent against all three sidecars at max
  error 5e-6. The four earlier results must NOT be marked superseded on those grounds.
* **"Scoring alignment could not be reproduced."** My error: each run has its own
  sidecar, and I scored against the untrimmed 210,528-row BASE instead of the
  trimmed 178,536-row one. With the right BASE, all 3,029 generations across six
  runs align at offset 1, zero refused. The refusal was the script working.
* **"Invalid rows in training are the root cause."** NOT ESTABLISHED. The row-count
  gap is real (178,536 fed vs 132,869 valid) and worth fixing, but saturation is
  the mechanism with direct source-level and measured evidence. Do not assert the
  invalid-row story as the cause.
* **"An admin cancelled the three jobs."** That was me (`scancel`), after they had
  reached ~675/683 of ~683 generations — so the fix24/fix72 numbers above are from
  essentially complete runs.

## Standing caveat

8 -> 16 islands did not by itself close the gap; the 16-island gain arrived bundled
with the normalization fix. The 8 -> 16 improvement is also mostly *avoided
catastrophe* rather than better fitting: August 81.15 -> 3.06 while March
(0.92 -> 1.61) and April (0.68 -> 0.95) got worse, tracking the collapse rate
(14.0% -> 1.9%). Do not expect 40 islands to close 47%.
