# Sharpening the policy: contrastive / RFT / GRPO ideas

Captured 2026-05-08 from a design discussion. Not committed to a plan
yet — these are options to revisit after the current DQN run lands and
we've done the OpenKnot eval.

## The motivating critique

Struct2SeQ (Shujun's paper) generates 98 000 sequences per target and
screens with Jaccard / OpenKnot score. Even though their best puzzles
yield hundreds–thousands of "perfect" matches, the per-sample acceptance
rate is on the order of 1–10 %. That feels like **search-by-volume with
a weak prior**, not a model that has actually learned the
structure→sequence map. The aesthetic goal: a model that proposes a
small number of sequences and most of them are good — quality moved
from search budget to the model itself.

## Why is the current policy diffuse?

Three compounding causes, all fixable:

1. **Binary per-position reward** (eq. 4 in the paper:
   `r_t = 1[pair correct] · 1[SHAPE compliant]`). A 90 %-correct sequence
   and a 60 %-correct sequence both yield "partial credit" smeared across
   positions. At any single position the gradient signal can't
   distinguish "this is one of 50 mediocre sequences" from "this is
   actually a top candidate". Q-values converge on a blurry policy.
2. **DQN signal is per-position.** The thing we actually care about is a
   sequence-level outcome (does the whole RNA fold to S?). DQN never
   sees that as a single signal — it's smeared across L per-position
   bootstraps. Modern RL methods (PPO, GRPO) get sharper signal by
   ranking *whole rollouts*.
3. **Order-agnostic adds variance.** Different perms produce different
   sequences for the same target. A few orders are excellent, most are
   mediocre. The model never sees the contrast between "this order on
   this target produced 0.95" and "that order on the same target
   produced 0.20" at training time. The 98k-best-of brute force is
   implicitly doing this discrimination — at inference, not in the model.

## Contrastive flavors and what each one buys you

### (a) Inference-time verifier — separable, **band-aid**
Train `E(seq, struct) → score` (regression on reward, or InfoNCE between
seq/struct embeddings using off-target sequences as free negatives).
At inference: generate N candidates, rerank with E, return top-1.

- Pros: the policy is untouched; verifier is small and fast; free
  negatives are abundant.
- Cons: doesn't make the policy any better. Still need to generate
  many candidates — just pick more cleverly. Useful as a backstop, not
  a fix to the diffuse-policy problem.

### (b) Rejection-sampling fine-tuning (RFT / expert iteration) — **start here**
Sample N per target with the current model, keep top-K % by reward,
supervised-fine-tune the policy on those expert samples (plain
teacher-forced cross-entropy on `(struct, perm, seq)` triples). Iterate.

- This is the recipe behind Llama-3 post-training, R1's SFT phase, and
  many recent post-trains.
- Reuses the existing 98k-style brute force productively: instead of
  throwing those samples away after eval, they become training data.
- Implementation: ~200 LOC on top of existing `play()` + training
  code. No new architecture, no new loss math.
- Plausible gain: best-of-1 reward 0.88 → 0.91 range after a single
  round, *guess*; needs measurement.
- Risk: mode collapse if the expert pool is too narrow. Mitigate by
  keeping top-5 % rather than top-1 %, and by mixing in a fraction of
  on-policy DQN updates.

### (c) Group-relative policy contrastive (GRPO / DPO-style) — **right thing for order-agnostic**
Per target, sample K rollouts under different perms (and/or different
epsilon). Compute reward for each. Within each K-tuple, normalise:
`A_i = (r_i - mean) / std`. Loss:
`L = -mean(A_i · log p(seq_i | struct, perm_i))`.

- The K rollouts use *different orders*, so GRPO directly trains the
  model to be consistent across orders. That is exactly the
  consistency property order-agnostic training is supposed to deliver
  but DQN doesn't enforce sharply.
- The within-group advantage normalisation gives a clean ranked signal
  even when raw rewards saturate near 1.0 — fixes the binary-reward
  blur.
- Implementation: ~150 LOC on top of existing `generate_permuted` and
  the play loop. Replace DQN TD loss in `train_epoch` with the
  group-advantage REINFORCE loss.
- Compute: K rollouts per target costs K× the play time, but per-target
  count can drop (we don't need 98k targets if each one yields a sharper
  signal). Tradeoff is favourable.

## Recommended sequence

1. **Finish current DQN run.** Don't throw it away — DQN gives a
   reasonable starting policy that RFT/GRPO can build on.
2. **OpenKnot eval on `best_policy_network.pt`.** Establishes baseline
   numbers (Jaccard, target-OK-score, perfect-solutions-per-puzzle)
   to measure improvements against. Required before any of below.
3. **RFT round 1.** Generate ~100 candidates × 1 000 targets
   (≈ 100k rollouts, ~1–2 days on 4 GPUs). Keep top 5 % by reward.
   SFT the policy on those triples. Re-run OpenKnot eval. Compare.
4. **RFT round 2** if round 1 helped (it almost certainly will).
5. **GRPO** if RFT plateaus or order-consistency is still poor (e.g.,
   if reward variance across orders for the same target is large after
   RFT). This is the bigger lift; spec it before implementing.
6. **(Optional) Verifier reranker** at the end. Once the policy is
   sharp, a small InfoNCE verifier on top squeezes out the last few
   points cheaply.

## How this would frame the paper

> "Struct2SeQ achieves human-competitive performance by sampling 98 000
> sequences per target and screening. We show that order-agnostic
> training combined with contrastive policy refinement achieves the
> same quality at K-fold lower per-target sample budget — the model
> has *learned* the structure→sequence map, not just a wide prior."

That is a concrete, defensible claim and directly answers the
search-by-volume critique. Pairs naturally with the in-painting /
motif-scaffolding capability story (which AR can't do at all).

## Caveat: binary reward still gets in the way

RFT and GRPO sharpen the policy **given** a noisy reward, but they
don't make the reward less noisy. The paper's exponential reward
reweighting (eq. 6) is a hack in this direction. Cleaner alternatives
to consider for the reward itself:

- Sequence-level **MCC** of predicted vs target structure (continuous
  in [-1, 1], smooth gradient).
- Sequence-level **Eterna classic score** as a single scalar.
- Sequence-level **target OpenKnot score** (already what we screen on
  — make it the training signal too).

Any of these as the reward for the GRPO advantage would be more
discriminating than the per-position binary signal. Reward redesign
is independent of the contrastive question and could be a separate
ablation.

## Open questions to settle before committing

- For RFT: top 5 % vs top 1 %? Mix in on-policy data or pure SFT?
  How many rounds before diminishing returns?
- For GRPO: K = 4, 8, 16? Different perms vs different epsilon as the
  source of within-group diversity? Both?
- Reward redesign: stick with binary per-position for comparability
  with Shujun's results, or move to a graded reward and lose the
  apples-to-apples?
- Verifier: train it as a side artifact during RFT/GRPO (cheap, free
  data) or only when needed?
