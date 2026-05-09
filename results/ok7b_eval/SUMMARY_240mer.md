# OK7b 240mer — clean comparison tables

All numbers from `results/ok7b_eval/MASTER_240mer.csv`. Reproducible
via `evaluation/build_240mer_master.py` (re-runnable aggregator).
Headline numbers shown in **bold**.

## Headline (matched K~1000 budget per puzzle)

| config | n_samples | unique seqs | perfect% | solved |
|---|---:|---:|---:|---:|
| AR L→R argmax-only (originally-broken K=1000 baseline) | 20,032 | 340 | 14.35% | 5/20 |
| AR L→R argmax-only + rescue | 20,137 | 445 | 14.37% | 7/20 |
| AR + faithful 3-strategy (eps0.05+eps0.10+qsoftmax) | 20,160 | 20,120 | 6.55% | 13/20 |
| **AR + faithful 3-strategy + rescue** (matches `test_240.py` exactly) | **20,283** | **20,243** | **6.64%** | **17/20** |
| AR + 4-strategy (paper §2.7: + topk_k=128) | 22,720 | 22,680 | 6.81% | 13/20 |
| AR + 4-strategy + rescue | 23,098 | 23,058 | 6.81% | 16/20 |
| **Ours: bidir_random argmax-only (K=1000)** | **20,032** | **20,032** | **33.24%** | **13/20** |
| Ours: bidir_random argmax + Shujun-rescue (4^k local) | 20,362 | 20,362 | 32.70% | 13/20 |
| Ours: bidir_random argmax + bidir-rescue (single-seed K=100) | 20,732 | 20,087 | 32.12% | 13/20 |
| Ours: bidir_random argmax + bidir-rescue (multi-seed top-5, K=500) | 23,532 | — | 28.30% | 13/20 |

## Headline ratios

- **Per-sample reliability ratio**: ours / AR-with-full-Shujun-protocol = **33.24% / 6.64% = 5.0×**
- **Coverage without rescue (matched diversity)**: ours 13/20, AR 13/20 — tied
- **Coverage with rescue**: AR 17/20 vs ours 13/20 — AR's 4-puzzle lead is ENTIRELY rescue, which exploits AR's narrow-exploration failure modes

## AR per-strategy ablation

| strategy | K | perfect% | solved |
|---|---:|---:|---:|
| epsilon p=0.05 | 333 | 10.77% | 12/20 |
| epsilon p=0.10 | 333 | 8.79% | 11/20 |
| **Q-softmax p=1.0** (Q-values uncalibrated → garbage at low K) | 333 | **0.07%** | 2/20 |
| topk beam k=50 | 50 | 17.10% | 5/20 |
| topk beam k=128 | 128 | 8.91% | 7/20 |

## Our model ablations

| variant | K | perfect% | solved |
|---|---:|---:|---:|
| bidir_random argmax (random-perm) | 1000 | **33.24%** | **13/20** |
| bidir_identity argmax (L→R, same checkpoint) | 1000 | 23.72% | 10/20 |
| bidir + structural motif-preservation in-painting | 1000 | 25.69% | 13/20 |
| bidir_random + epsilon p=0.10 | 333 | 16.89% | 14/20 |
| bidir_random + Q-softmax | 333 | 0.07% | 3/20 |

## Architecture-vs-order ablation

Same checkpoint, only inference changes:
- L→R argmax → 23.72%
- random-perm argmax → 33.24%
- **Δ from decoding order alone: +9.5pp** (clean attribution to our method)

Same L→R protocol, different checkpoints (orig vs ours-bidir-arch):
- orig L→R argmax → 14.35%
- ours bidir_identity L→R argmax → 23.72%
- **Δ from architecture+training: +9.4pp** (partly confounded with 5 extra training episodes)

## Rescue is one-sided (mechanistic finding)

Symmetric rescue applied to per-puzzle best non-perfect:

| | rescue worked | rescue failed |
|---|---:|---:|
| AR + 3-strategy | 4/5 (puzzles 0, 12, 13, 17 → jaccard=1.0) | 1/5 (puzzle 3) |
| Ours bidir_random | 0/6 (max_j 0.956–0.989) | 6/6 |
| Ours + multi-seed bidir-rescue (top-5 × K=100) | 0/7 (max_j 0.902–0.989) | 7/7 |

Interpretation: AR's failure modes are 1-2 base mismatches (narrow
exploration, locally rescuable). Our random-perm decoder explores
broadly; remaining failures are structurally entrenched (whole-context
wrong, not a few bases). Local repair (whether 4^k enumeration or
model-guided in-painting at any seed/K combo) cannot bridge that.

## Q-softmax is essentially garbage at low K (both architectures)

At K=333 budget, Q-softmax sampling produces ~0.2 perfects per puzzle:
- AR + Q-softmax: 0.07% perfect, 2/20 solved
- Ours + Q-softmax: 0.07% perfect, 3/20 solved

Q-values are not calibrated probabilities (Shujun's writeup §2.7
acknowledges this), so softmax sampling drifts far off-peak and
breaks pseudoknot constraints. Works only at K=98,000 budget where
0.07% × 98000 ≈ 70 perfects per puzzle is enough.

## L→R + argmax is effectively deterministic at K=1000

Unique-sequence counts:

| config | total samples | unique seqs |
|---|---:|---:|
| AR L→R argmax-only | 20,032 | **340** (mean 17/puzzle) |
| Ours bidir_identity L→R argmax | 20,032 | **346** (mean 17/puzzle) |
| Ours bidir_random argmax | 20,032 | **20,032** (1000/puzzle = 1 per sample) |

Order-randomization is the diversity engine for our model — token-level
sampling (epsilon, qsoftmax) is not required. AR cannot use this lever
because order is fixed.
