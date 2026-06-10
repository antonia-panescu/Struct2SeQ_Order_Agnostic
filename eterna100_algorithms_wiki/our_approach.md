# Our approach — Struct2SeQ-bidir (order-agnostic RL)

The anchor page. Every external algorithm in this wiki is compared back to this.

## One-paragraph summary
Struct2SeQ-bidir is a **learned, amortized** RNA inverse-folding policy: a single
neural network that, given a target dot-bracket structure, emits a nucleotide
sequence. It extends the original Struct2SeQ (an LSTM/GNN structure→sequence
model that generates strictly **left→right**) into an **order-agnostic /
bidirectional** policy trained with **Q-learning**, so it can fill positions in
*any* order. Rewards during training come from a **RibonanzaNet-SS** folding
oracle (a learned secondary-structure predictor). At inference we draw **K
samples** (best-of-N: K=1000 or K=10k) and refold each to check solves.

## Solving logic (how a puzzle is "solved")
1. Encode the target structure; the policy network amortizes everything it
   learned from training data (Shujun's `top2M.csv`, ~2M genome-scan 240-mer
   windows with OK filtering — **not** bpRNA).
2. **No per-puzzle search loop.** We sample K candidate sequences (random
   permutation order + argmax, or stochastic strategies).
3. Fold each candidate back with an oracle and mark the puzzle solved if **any**
   candidate folds exactly to the target (base-pair Jaccard = 1.0).
4. We evaluate under three fold-back oracles to separate "did we design it"
   from "which folding model agrees": **RibonanzaNet, ViennaRNA 2, EternaFold.**

## Oracle & objective
- **Training reward:** learned (RibonanzaNet-SS), not thermodynamic.
- **Eval / solve criterion:** exact base-pair match after refolding (Jaccard 1.0),
  reported separately per oracle. Plus a graded mean-best-Jaccard fidelity metric.
- No explicit free-energy / ensemble-defect term in the objective.

## Reported Eterna100-V2 results (K=1000)
| Model | RibonanzaNet | ViennaRNA 2 | EternaFold |
|---|---:|---:|---:|
| S2S-bidir (random-perm, argmax) | 59/100 | 52/100 | 64/100 |
| Original S2S argmax | 25 | 30 | 36 |
| Original S2S 3-strategies | 52 | 45 | 58 |
| Original S2S 3-strategies + rescue | 59 | 49 | 61 |

(K=10k done too; conclusions in committed reports — bidir best or tied-best.
All 100 V2 targets are pseudoknot-free.)

## Defining characteristics (for the comparison rubric)
1. **Search vs. amortization:** ❌ no per-puzzle search — pure amortized generation + best-of-N. This is the headline difference vs. search methods like NEMO.
2. **Oracle:** **learned** (RibonanzaNet) for the reward; refold-checked across learned + thermodynamic oracles.
3. **Objective:** exact base-pair match; no energy term.
4. **Domain knowledge:** ❌ none hand-crafted — no GC/AU/GU fill ratios, no loop boosting, no mismatch tables. Everything is learned from `top2M.csv`.
5. **Compute budget:** one network forward pass per sample × K; **seconds, fully parallel**, no per-puzzle iteration. Orders of magnitude cheaper than hours-per-puzzle search.
6. **Generality:** trained on 240-mer genome windows; also evaluated on the **240-mer / OpenKnot-7** in-painting setting — i.e. aims at long, natural-like designs, not just the short hand-authored Eterna puzzles.

## Where this sits
Our bet is **amortization + a learned oracle**: train once, generate fast, no
brittle hand-tuned heuristics, and transfer to long/natural RNAs. The cost is
that we don't (yet) do the per-puzzle thermodynamic search that lets methods
like NEMO grind out the hardest 30–40 puzzles. The interesting design space —
explored on each algorithm page — is **hybridizing**: using our learned policy
as the proposal/prior inside a lightweight search loop.
