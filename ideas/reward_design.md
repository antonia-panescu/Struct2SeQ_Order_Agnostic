# Reward design — graded alternatives to partner-match-binary

Captured 2026-05-08 from a comparison between this repo's reward and
the knitnet branch's REINFORCE reward. Things to try later, after the
current DQN run lands and the OpenKnot eval is in.

## Diagnosis: two repos, two rewards, neither universally better

| Repo | Reward | Granularity | Algorithm fit |
|---|---|---|---|
| `struct2seq_bidir_rl/` (this) | per-position binary partner-match (`Env.py:132–167`) | vector `[L]` | DQN per-step TD |
| `knitnet/` REINFORCE | F1 of base pairs (continuous [0,1]) | scalar per sequence | REINFORCE + baseline |

Reward design follows algorithm choice — neither is "better" out of
context.

**Where F1 wins:** graded signal, no 3-positions-cost-per-shift
amplification, drops cleanly into REINFORCE/GRPO advantage
normalisation.

**Where F1 loses:** ignores unpaired-correctness entirely (an
all-unpaired design has F1 = 0 even when target has many unpaired
positions correctly handled), zero-gradient regime when F1 = 0,
unusable for per-step DQN bootstrap.

**Where partner-match-binary wins:** rewards unpaired correctness,
per-position credit assignment for DQN, distinct values across
"mostly wrong" outputs.

**Where partner-match-binary loses:** binary cliff between off-by-one
and on-target, 3-position amplification, no smooth gradient near
optimality.

## Things to implement (priority order)

### 1. Partner-distance reward — graded, per-position (DQN-compatible)

A drop-in replacement for `Env.py:get_reward()` that fixes the
binary cliff while keeping the per-position vector shape DQN needs.

**Definition.** For each position `i`:

```
if i is paired in target with partner j_target:
    if i is paired in design with partner j_design:
        r_i = exp(-|j_design - j_target| / σ)        # smooth credit
    else:
        r_i = 0                                       # missed pair entirely
elif i is unpaired in target:
    r_i = 1 if i unpaired in design else 0           # binary unpaired correctness
```

**Properties.**
- σ is a tunable scale (start σ ≈ 3 nt — gives ~0.72 reward for
  off-by-one, ~0.13 for off-by-five, ~0 for off-by-twenty).
- Reduces to partner-match-binary when σ → 0; reduces to "any pair
  partner counts" when σ → ∞.
- Vector-shaped per position, so DQN's per-step TD still works
  unchanged.
- Mean over positions gives a continuous sequence-level score in
  [0,1], usable as a REINFORCE/GRPO advantage too.

**Implementation.** ~25 LOC in `Env.py`, no changes elsewhere.
Behind a config flag (`reward_mode: "partner_distance"`) so the
existing binary reward stays as a comparison baseline.

### 2. Sequence-level continuous rewards for REINFORCE/GRPO path

If/when we move off DQN onto REINFORCE/GRPO (see
`ideas/contrastive_and_rft.md`), the sequence-level reward options
in priority order:

a. **Target OpenKnot score** — directly the metric Shujun's paper
   evaluates on. Aligning training with eval is almost always a win.
b. **MCC of base-pair adjacency matrix** — symmetric, in [-1,1],
   well-defined for sparse-pair targets, considers TN as well as TP.
c. **Jaccard of base-pair sets** — already used in the paper's
   screening pipeline (`fig 3`, `fig 8`); cheap to compute.
d. **F1 of base-pair sets** — what knitnet uses. Same caveat as
   above (ignores unpaired correctness).
e. **Mean of partner-distance reward** (option 1 above, aggregated)
   — graded, accounts for unpaired correctness.

### 3. Reward-mode ablation as a deliverable in itself

Once we have the partner-distance reward implemented, run a
3-way comparison on the same training setup:
`partner_match_binary` (current) vs `partner_distance` vs `f1_seq_level`.
Hold everything else constant. The ablation is publishable on its
own — the field doesn't have a head-to-head comparison of these
reward choices on inverse folding.

## Suggestion for the knitnet branch

The knitnet REINFORCE training (`src/models/rna_denoiser_rl_module.py:237–249`)
currently uses raw F1 of base pairs. **The partner-distance reward
would be a strict upgrade for knitnet too**, because it fixes F1's
biggest weakness: the lack of credit for correctly-unpaired positions.

Suggested concrete change for the knitnet branch:

1. Add a new `reward_mode: "partner_distance"` (and optionally
   `"partner_distance_with_repair"` to mirror the existing
   `final_wc_repair` variants).
2. Compute partner-distance per position as in section 1 above,
   then take the mean to produce the sequence-level scalar that
   REINFORCE expects.
3. Compare against `raw` (current default) F1 on the existing
   training pipeline. Hold compute and seeds constant.

Why this is worth doing on knitnet specifically:

- Knitnet's REINFORCE setup is *closer* to GRPO than struct2seq's
  DQN, so improvements to the reward there compound when we move
  struct2seq toward GRPO later.
- Sparse-pair targets (most loops, single-stem hairpins, long
  unstructured regions) get zero signal under F1. Partner-distance
  with the unpaired-correctness term gives meaningful gradient on
  those.
- The repair-edit-penalty variant (`final_wc_repair_edit_penalty`)
  combines naturally with partner-distance — the model gets
  graded structural credit *and* edit-distance regularisation in
  one signal.

If this lands well on knitnet, porting it back to struct2seq is
trivial because the per-position reward shape is preserved (DQN
still works).

## Decision tree

```
Are we keeping DQN for struct2seq?
├── Yes: implement partner-distance reward (option 1) in Env.py.
│        Run a 3-way reward ablation (binary / partner-distance / mean-F1).
│        Test on knitnet first as the easier integration point.
│
└── No / moving to REINFORCE+GRPO: implement target-OK-score reward
   (or MCC), wire into the sequence-level advantage path.
   Partner-distance is still a fine fallback.
```

## Open questions

- σ for partner-distance: tune as a hyperparameter or fix at 3 nt
  based on intuition? Probably ablate σ ∈ {1, 3, 5, 10} once.
- Should unpaired-correctness be weighted differently from
  paired-correctness in the per-position reward? E.g., normalise so
  that the *expected* reward is 0.5 under a uniform-random design,
  to avoid imbalance for pair-sparse vs pair-dense targets.
- Does the partner-distance smoothing help or hurt in the long
  pseudoknot regime where partners can be 100+ nt away? σ might need
  to scale with pair distance.
- Compatibility with the paper's eq. 6 exp-reweighting:
  partner-distance is already graded, so the exp-reweight may be
  redundant or even harmful. Test both with and without.
