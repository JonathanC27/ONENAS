# Shrinkage diagnosed, fixed, and shown not to matter — the ceiling is LEVEL, not capacity

Genome binaries parsed directly (a standalone reader now exists at
/anvil/scratch/x-jchang5/wwtp/dyn/analysis/gparse.py). 19,623 scored candidates
instrumented across 4,000 island-generations, 7 test runs.

## THREE CORRECTIONS to the framing I reported earlier

**1. It is the INPUT layer collapsing, not the hidden layer.** The final global best
of prod_h24_s41 has **2 enabled inputs (N2O, O2_T1), 0 hidden, 1 output**, with 13
inputs disabled. `disable_node()` (rnn_genome.cxx:2711-2717) excludes only
`OUTPUT_LAYER`, so `INPUT_LAYER` nodes are legal targets. Disabling an input is
**data ablation, not architecture search.**

**2. The search never had capacity to destroy.** The seed genome is 14 inputs fully
connected to 1 output — **0 hidden nodes, 0 recurrent edges**. Live hidden nodes
never exceeded ~1.0 in any decile. This is a **growth failure**, not destruction of
a working architecture. My "the search actively destroys a working architecture"
claim was wrong.

**3. "Enabled Edges" in fitness_log.csv is an accounting artefact.**
`get_enabled_edge_count()` (:400) tests only `edge->enabled`, ignoring endpoint
reachability. The "22 edges / 12 recurrent" at generation 631 are **2 live forward
and 1 live recurrent**; the rest hang off disabled inputs. Edges never grew.

Also: **saturation is minor.** Only 0.16% of global-best predictions have |p|>0.999;
median per-generation max|p| ~ 0.60. Of the 12.84% out-of-range predictions, **98.8%
are BELOW the target minimum** — a level error, not clipping at +-1.

## MECHANISM (measured, not argued)

**M1 — new UNITS are rejected at birth; new EDGES are not.** Acceptance into the
elite pool, overall mean 0.372:

| operator | acc. rate | vs mean |
|---|---|---|
| add_edge | **0.743** | 1.99 |
| add_recurrent_edge | 0.587 | 1.58 |
| enable_node | 0.364 | 0.98 |
| disable_node | 0.360 | 0.97 |
| add_node | 0.330 | 0.89 |
| **split_node** | **0.118** | **0.32** |

Selection has NO preference for node removal (0.360 vs 0.364 — indistinguishable).
What is rejected is *new units*. Cause: Lamarckian init (rnn_genome.cxx:1850-1852)
draws the new unit's weights from the parent's distribution — a random perturbation
at parent scale — so the child is strictly worse at birth with 10 BP epochs to
recover. Acceptance decays as incumbents converge: add_node 0.542 -> 0.148,
split_node 0.242 -> 0.060. **This is exactly the failure NEAT's speciation prevents;
ONE-NAS islands are parallel populations, not topological niches.**

**M2 — the input layer random-walks off a reflecting boundary.** With no hidden units
to target, both operators act almost entirely on inputs. `disable_node` is always
feasible; `enable_node` is infeasible from the seed (nothing disabled), and the retry
loop (onenas.cxx:290-296) resamples on failure. Measured **3.1:1 proposal imbalance**
in the first 50 generations. A Monte-Carlo of this walk *with no selection at all*
predicts E[enabled inputs] = 7.58 after 600 steps; observed 6.0-6.4. **The erosion is
essentially neutral diffusion.**

**M3 — selection is too weak to arrest it.** The per-generation validation set
(5 x 144 rows) has **median target variance 0.00247 against pooled 0.01661 — 14.9%
of the variance the model is finally scored on**; 35% of generations select on a
slice with variance < 0.001. Elite rank vs that elite's MSE on the next held-out
window: **mean Spearman rho = 0.170**. Plus a leak: `backpropagate_stochastic` picks
`best_parameters` by minimising MSE on the SAME 5 windows later used to rank the
child (rnn_genome.cxx:1214-1219), while elites are re-scored cold — children get a
min-over-10 optimistic score, elites one honest one. **1.82 of 8 elite slots turn
over per generation.**

Resolved: Lamarckian inheritance PENALISES grown children, it does not advantage
shrunk ones (my hypothesis was backwards). `--control_size_method` has only `none`,
`reduce_mutation_rate`, `reduce_add_mutation` — **both non-none options reduce
growth; there is no grow-side size control in the codebase**, so `none` was correct.
Structural dedupe is not size-biased. Repopulation is not an accelerant (+0.400
nodes after an event vs -0.027 elsewhere).

## THE FIX WORKS STRUCTURALLY — AND DOES NOT HELP ACCURACY

One-line change: exclude `INPUT_LAYER` from `disable_node()`.

Live-input trajectory by decile:

    PROD   13.5 13.5 12.6 12.0 10.9  7.8  6.6  5.4  5.7  5.5
    A1     13.4 13.3 13.8 13.1 12.7 13.6 13.8 14.0 14.0 14.0
    B1     13.6 13.8 13.9 13.9 13.9 13.9 13.9 13.4 13.8 14.0

Erosion abolished, 3/3 runs. And as a second-order effect **the search finally builds
structure**: A1 reaches 14 inputs / 2.85 hidden / 29.3 edges / 27.1 recurrent —
comparable in scale to the frozen control. Input erosion was consuming the
`disable_node` budget and crowding out growth.

**But accuracy does not improve.** Three replicates of the IDENTICAL baseline score
**1.0066, 1.3797, 1.9748** — a ~2x run-to-run spread. Both protect-inputs replicates
(2.01, 2.53) sit at or above that band. Honest reading: no improvement, not
resolvable at n=1 per arm.

**Runs are NOT bit-reproducible despite --examm_seed** — worker completion order
determines insertion order. Record as a protocol finding.

**`--linear_output` is a clear negative:** max|p| explodes to 3.88-6.18 because
removing the bound removes the only thing limiting the damage, and
`--max_pred_sd_ratio` is evaluated on validation, not the scored window.

## THE CEILING: 74-93% OF THE ERROR IS LEVEL, AND SHAPE IS PINNED

Decomposing pooled error into shape (after removing an oracle per-window mean) and
level:

| run | nMSE | **shape** | level | level share |
|---|---|---|---|---|
| PROD (2 inputs) | 1.6986 | **0.3398** | 1.3588 | 80% |
| A0 baseline | 1.0066 | **0.3786** | 0.6281 | 62% |
| A1 protect-inputs (14 inputs, 27 rec edges) | 2.0064 | **0.3548** | 1.6516 | 82% |
| A3 both | 6.0836 | **0.4489** | 5.6347 | 93% |

**The shape component is pinned at 0.34-0.45 in every arm.** The 2-input collapsed
network and the 14-input / 27-recurrent-edge network fit the within-window signal
EQUALLY WELL. An oracle per-window constant scores 0.3788.

Persistence reaches 0.4901 because it **anchors the level** — it starts from y_t.
ONE-NAS has y_t as an input and never learns to pass it through.

## VERDICT

* The collapse is real, fully explained, and fixable in one line — verified 3/3.
* **Fixing it does not close the gap.** Restoring all 14 inputs and letting real
  recurrent structure grow leaves the within-window fit unchanged.
* **Persistence (0.4901) is NOT reachable by any architecture-search fix.** Even
  perfectly de-levelled, ONE-NAS caps near 0.35 with an ORACLE level; with a causal
  anchor the published baselines already sit at 0.4389-0.4901.
* The frozen control's 0.6138 is likewise a level-tracking result, not a capacity one.

**The defensible claim is NOT "the search destroyed a working architecture."** It is:
*on a low-variance, level-dominated prequential task, the search's selection signal
(median 15% of the pooled variance, rho = 0.17 against the next window) is too weak
to hold structure against the neutral drift of its own operator set — and even when
structure is held, the task is not capacity-limited.*

The only lever that could produce a competitive number is **predicting the
persistence residual** (y_{t+H} - y_t), a target reparameterisation, not a search
fix. Note this is NOT contradicted by the earlier ridge delta test (which found the
reparameterisation a no-op): ridge already had N2O_l0 in its features and could
represent persistence exactly, whereas ONE-NAS's tanh output cannot represent the
identity on a target with mean -0.517.
