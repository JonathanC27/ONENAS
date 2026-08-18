# Why the search destroys architectures — literature verdict and ranked fixes

## HEADLINE: our config INVERTS the published exploration/exploitation ratio

|  | elite | generated | ratio |
|---|---|---|---|
| **ours** | 8 | 5 | **0.625** |
| **published ONE-NAS** (verbatim, arXiv:2202.13471 and ASC 2023) | 5 | 10 | **2.0** |

A **3.2x reduction in offspring throughput per elite slot.** And ONE-NAS explicitly
does NOT retrain elites: *"while O_t are trained using backpropagation on batches
drawn from the historical data, the RNNs in E_t do not continue to be trained."* So
8 frozen elites occupy the island while only 5 children compete per generation.

NOTE: this exact deviation was flagged during the EQUITY campaign as a proposed
experiment ("elite/generated split sweep: ours elite 8/generated 5 vs the original
paper's elite 5/generated 10") and was never run.

## THE MECHANISM IS IN THE FOUNDING PAPER

Stanley & Miikkulainen, NEAT, *Evolutionary Computation* 10(2):99-127, 2002,
DOI 10.1162/106365602320169811, §3.3, verbatim:

> "Because **smaller structures optimize faster than larger structures**, and adding
> nodes and connections usually initially decreases the fitness of the network,
> recently augmented structures have little hope of surviving more than one
> generation even though the innovations they represent might be crucial towards
> solving the task in the long run. The solution is to **protect innovation by
> speciating the population**."

And their non-speciation ablation, §5 — this is our failure mode, published in 2002:

> "without speciation, the population **quickly converges on whatever topology
> happens to initially perform best**. Thus, a lot of diversity is drained
> immediately (within 10 generations)."

## THE LINEAGE CLAIMS GROWTH — but never measured it

| source | claim |
|---|---|
| Lyu & Desell, ONE-NAS, GECCO 2022 | "will progressively **grow larger** RNNs" |
| Ororbia, ElSaid & Desell, GECCO 2019 | "**Larger networks tended to perform better**" |
| Lyu et al. 2024 (stock returns, arXiv:2410.17212) | "evolves **progressively larger** RNNs" |

**CAVEAT THAT MUST BE HONOURED:** these are prose design claims, not measurements.
**No paper in the lineage plots genome size over generations.** So we cannot say
"published runs grow and ours shrinks" from published DATA — only that the published
CLAIM is growth and our measurement contradicts it. The absence of the diagnostic is
itself worth reporting.

## tanh ON THE OUTPUT IS EXAMM-STANDARD — reframe the bug

It is stock behaviour (rnn_node.cxx:82-88), not a local deviation. The real defect is
the **normalisation contract under non-stationarity**: EXAMM's normalisers fit bounds
on training data and assume targets stay inside tanh's range; under drift a streaming
target leaves it. Frame as a violated contract, NOT as "tanh is wrong."

Hewamalage, Bergmeir & Bandara, *IJF* 37(1):388-427, 2021, §4.2.6, verbatim:
> "The activation functions used in RNN cells... **have a saturation area after which
> the outputs are constant**. Hence... it should be assured that the inputs fed are
> **normalized adequately such that the outputs do not lie in the saturated range**."

Their remedy is per-window local normalisation, and their forecast head is an affine
layer. Also: `OUTPUT_NODE_GP` (identity activation) is already published in-lineage
via EXA-GP (DOI 10.1145/3638530.3654349, 10.1145/3638530.3664173) — so using it is
not a hack.

## UNDER-TRAINING: the best-supported item

* Shu, Wang & Cai, ICLR 2020 (arXiv:1909.09569): architectures with "fast convergence
  ... are **consequently selected by NAS algorithms**. Nonetheless, these architectures
  may not necessarily lead to better generalization."
* Zhou et al., EcoNAS, CVPR 2020: "**more epochs improves rho_sp**" while "decreasing
  the number of channels will lead to an **increase** of rho_sp" — cutting EPOCHS is the
  unreliable proxy axis; cutting CAPACITY is the reliable one. We cut epochs.
* The lineage's own 2024 stock paper uses **20 bp_iterations**, not 10. Raising it is a
  within-lineage move.

## RANKED FIXES

1. **Restore elite:generated to 5:10.** Pure config-deviation restoration — the
   strongest possible answer to "why was the original configuration wrong."
   Justification: NEAT §3.3.
2. **Decouple the training budget from genome size, or raise bp_iterations toward 20.**
   Justification: NEAT; Shu et al.; Zela et al. 2018; Zhou et al. 2020.
3. **Fix the output-range contract** — per-window local normalisation, optionally the
   existing identity output node. Justification: Hewamalage et al. §4.2.6; EXA-GP.
4. **Select on a persistence-scaled skill measure** rather than raw MSE. Justification:
   Deng et al., ECML PKDD 2022 (Auto-PyTorch-TS uses mean MASE as the validation AND
   test objective); Hyndman & Koehler 2006. Within one generation raw MSE is a valid
   comparator, but a global best tracked ACROSS generations is compared across windows
   of differing variance — quiet windows make MSE trivially small for near-constant
   predictors. **Pre-empt Koutsandreas et al. 2022** ("only small discrepancies between
   the different error measures") by citing it ourselves; present as robustness, not
   root cause.
5. **Age regularisation / re-evaluate elites.** Justification: Real et al., AAAI 2019
   (regularized evolution) + Yu et al. ICLR 2020 on ranking distortion under weight
   sharing. Stops a genome that got lucky on one quiet window holding an elite slot for
   hundreds of insertions.
6. **Report genome size over insertions as a first-class diagnostic.** The lineage's own
   precedent: Patterson, Karns, Lyu & Desell, GECCO 2025, DOI 10.1145/3712256.3726457.
7. **DO NOT add a capacity floor / inverse-parsimony term as a "restoration".** No prior
   work supports minimum-complexity pressure; the GP literature runs the other way. A
   referee will read it as special-casing. Present as a contribution or omit.

## THE FRAMING THAT SURVIVES REVIEW

The literature documents NAS **failing to improve** (Yu et al. ICLR 2020; Yang et al.
ICLR 2020; Li & Talwalkar UAI 2019) and DARTS **collapsing as search epochs grow**
(DARTS+ arXiv:1909.06035; RobustDARTS arXiv:1909.09656; DARTS- ICLR 2021). **None is an
evolutionary NAS on a streaming forecasting task, and none reports a fixed hand-picked
architecture beating the evolved one on the identical online loop and binary.** Our 2.8x
control gap with a measured size trajectory appears to be a novel instance. Claim:
*the literature documents NAS failing to improve; we document NAS monotonically
destroying a working architecture over a full run, with the size trajectory measured.*

## Unverified / flagged

No published genome-size-vs-generation curve exists anywhere in the lineage. Smyl &
Kuber 2016 is a symposium talk, cite via Hewamalage et al. No literature found on
degenerate selection under low-variance validation windows or on variance-aware fitness
for forecasting NAS — genuine gaps, and our novelty. `--control_size_method` was NOT
found in the searched tree; confirm which binary produced the run.
