# Ideas for the writeup

A running scratchpad of points to make in the workshop paper. Append
freely. Each entry is tagged so we can quickly slot ideas into the right
section when we draft.

Tags:
- `[FIG]` — visual / figure caption material
- `[TEXT]` — main-text talking point, not figured
- `[NUMBER]` — a specific quantitative claim with the number written down
- `[METHOD]` — methodology detail to include
- `[CAVEAT]` — limitation or honest acknowledgement
- `[REVIEWER]` — anticipated reviewer pushback + how we'd respond

---

## Headline claims

- `[NUMBER]` Order-agnostic decoding extension of Struct2SeQ produces
  **2.15× more perfect-Jaccard designs** than the original L→R AR model
  at matched K=1000 on the OK7 100mer pseudoknot benchmark
  (9145 / 20000 = 45.5% vs 4254 / 20000 = 21.2%, same scoring pipeline).
- `[NUMBER]` Within our model, **random-perm decoding beats L→R-from-the-same-checkpoint
  by 38.6%** at producing perfect designs (45.5% vs 32.8%) — proving the
  capability lives in the *training distribution*, not just architecture.
- `[NUMBER]` **Principled structural-motif in-painting (motif-preservation
  mode) achieves 37.9% perfect designs** on the OK7 puzzles (vs random
  scatter at matched mean fraction-fixed: 27.7%). Structurally-coherent
  constraints are *easier* for the model than scattered ones.
  **Framing**: this is the RFdiffusion / ProteinMPNN analog —
  *fix the motif (hairpin / pseudoknot-stem), design the surrounding
  scaffold*. The motif is the anchor; the scaffold is the design target.
- `[NUMBER]` Sweet spot of motif size: **56.8% perfect at 16-25 nt motifs**.
  Extreme sizes are harder both ways (too small = unconstrained noise;
  too large = under-determined design problem).
- `[NUMBER]` **AR + teacher-forced motif inpainting (the strongest hack)
  reaches 33.4% perfect — still 4.5pp below our 37.9%.** Teacher-forcing
  alone improves AR from 21.2% (no motif) → 33.4%, so it helps a lot.
  But it doesn't close the gap. Order-agnostic decoding wins natively.
- `[NUMBER]` **Best-of-N saturation:** at K=1000, ours solves
  **95% of OK7 puzzles** (≥1 perfect-Jaccard design); AR L→R only solves
  **75%**. AR + teacher-forced motif catches up to 95% but only at the
  full K=1000 — at K=500 it's at 89.7% vs our 92.2%. So even the
  saturation regime favors order-agnostic.

---

## Main text talking points

- `[TEXT]` **Matched-budget vs unmatched-budget framing.** All our
  comparisons against the original Struct2SeQ are at a *matched* K=1000
  sampling budget. We mention as an aside that the original's published
  numbers used K=98 000 (98× more samples per target on 256 A100s), and
  that even at 100× less compute our model produces more perfect
  designs. **This belongs in the main text only — not in any figure.
  Figures show matched-budget head-to-head; the unmatched mention is a
  one-line aside.**

- `[TEXT]` **No published RNA inverse-folding paper has reported
  structural-element motif preservation.** Position-level fixing exists
  (gRNAde, RNA-DCGen, BAnG), and protein-side work (RFdiffusion,
  ProteinMPNN) does motif scaffolding. We are the first to define and
  evaluate motif-as-structural-element preservation for RNA.

- `[TEXT]` **Our work continues Shujun's §4 future-work paragraph
  directly.** Quote: *"To do this, we simply need to condition the model
  on the incomplete decoding sequence by randomly permuting the decoding
  sequence during training while applying the causal mask. During
  inference, arbitrary decoding orders can be applied."* That is our
  method, full stop. Frame as "completing what Struct2SeQ left open."

- `[TEXT]` **Same scoring pipeline as Struct2SeQ.** RibonanzaNet-SS for
  predicted structure → Hungarian → Jaccard. RibonanzaNet for predicted
  SHAPE → Eterna Classic + Crossed Pair Quality → OK score (the
  pure-Python score functions from `OpenKnotScorePipeline/scoring.py`,
  bit-identical to Shujun's pipeline). No degree of freedom hidden in
  the metric.

- `[TEXT]` **Random-perm vs paired-first vs L→R-from-bidir as an
  ablation triple.** The same checkpoint produces three different output
  distributions under three different inference-time decoding orders.
  Random-perm wins overall. Paired-first wins on p80 OK score (cleaner
  high-scoring tail). L→R loses on hard pseudoknots but matches on easy
  puzzles. This validates Shujun's §4 hypothesis ("decoding paired
  positions together may help") *partially*: it's a meaningful
  perturbation of inference behaviour but not the universally-best
  choice. Random uniform-perm remains our default.

- `[TEXT]` **In-painting K-sweep U-shape.** Random-scatter perfect-rate
  is 45.5% → 27.7% → 18.3% → 18.5% → 31.7% across K_inpaint ∈ {0, 0.25,
  0.5, 0.75, 0.95}. The minimum at K=0.5–0.75 is meaningful: that's
  where the model has the *most* unconstrained context to generate
  while still being heavily constrained. The bounce-back at K=0.95 is
  because almost everything is fixed → easy.

- `[TEXT]` **Constraint held: 100% on a 500-row sample.** The model's
  generated sequences match WT exactly at every fixed position. This
  isn't trivially true — the in-painting is *enforced* by overriding
  the sampled token, but the surrounding context is *generated* by the
  model based on the constraint. The fact that the surrounding
  generates coherent structure ~38% of the time is the result.

---

## Things to NOT put in figures

- The 98k vs 1k compute-budget comparison. It's referenced text only;
  no plot.
- Per-puzzle results across configs at the level of every K_inpaint
  value × every motif kind. Too dense. We pick one or two
  representative views.
- Detailed architecture diagram (we just cite Struct2SeQ paper Fig 1
  and say "we replace LSTM-PE + causal-conv with relative-position
  bias on self+cross attention").

---

## Figures (planned)

- `[FIG]` **Fig 1 — Concept figure.** Single panel showing the
  in-painting idea on one OK7 puzzle (PreQ1-II switch). Antonia is
  redoing this herself.

- `[FIG]` **Fig 2 — Cross-method per-puzzle bar chart.** 20 puzzles ×
  perfect-Jaccard count out of K=1000, grouped bars: ours (random-perm),
  ours (paired_first), orig Struct2SeQ AR. Mirror Shujun's Fig 4 layout.

- `[FIG]` **Fig 3 — In-painting capability.** Quality (mean Jaccard or
  perfect-rate) vs constraint type/size. X-axis = motif size or
  fraction-fixed; lines for random-scatter and structural-motif modes.
  Shows the structural-motif curve sitting *above* the random-scatter
  curve at matched fraction-fixed.

---

## Anticipated reviewer pushback + responses

- `[REVIEWER]` *"Your K=1000 isn't enough; the published baseline used
  K=98 000."*
  Response: (a) we run the original Struct2SeQ at the same K=1000 to
  ensure apples-to-apples; we don't claim to beat them at their K; we
  claim to beat them at matched K. (b) We show the budget-scaling
  curve N ∈ {1, 10, 100, 1000} demonstrating quality saturates by N=1000
  on most puzzles.

- `[REVIEWER]` *"Single seed."*
  Response: acknowledged in limitations. Workshop submission, future
  work to add multi-seed.

- `[REVIEWER]` *"Why no SHAPE supervision?"*
  Response: out of compute scope; SS-only training to keep the
  architecture-vs-decoding-order axis clean. Shujun's Struct2SeQ-SHAPE
  variant adds SHAPE; mention as future work.

- `[REVIEWER]` *"Why not compare to gRNAde / MPNN-RFdiff / Rosetta?"*
  Response: per-puzzle bar chart Fig 2 includes these from the
  OpenKnotBench CSV. We are competitive at K=1000 with their
  cherry-picked top-20 of K=98000.

- `[REVIEWER]` *"You only test one architecture (RPE pure transformer
  decoder). Is the gain from the architecture or the training
  distribution?"*
  Response: the L→R-from-bidir-checkpoint ablation answers this. Same
  weights, same architecture, different inference order. L→R is 32.8%
  perfect; random-perm is 45.5%. The gain lives in the
  training-distribution + decoding-flexibility combination, not the
  architecture alone.

- `[REVIEWER]` *"Wouldn't AR with teacher-forced inpainting close the
  gap?"*
  Response: **No (RESOLVED).** We ran orig Struct2SeQ.pt under L→R
  AR with teacher-forced WT at the motif positions, K=1000, same 20
  OK7 puzzles. Result: 33.4% perfect Jaccard. Compare to ours
  (bidir + structural motif): 37.9% perfect. Teacher-forcing helps
  AR a lot (21.2% → 33.4% on its own scaffold), but it does NOT
  close the gap to order-agnostic — we still win by **+4.5pp
  absolute / +13.5% relative**. Mechanism: even with the motif
  positions teacher-forced, the L→R decoder still commits to upstream
  scaffold positions before the motif is reached and hence cannot
  optimize them around the constraint.

---

## Limitations to acknowledge upfront

- `[CAVEAT]` Single seed on the eval sweep. Reproducibility on our
  end is good (per-rank checkpointing, exact CSVs); statistical
  significance on between-config differences not reported.

- `[CAVEAT]` Computational, not experimental, evaluation. We use
  RibonanzaNet-SS for structure prediction and RibonanzaNet for SHAPE
  prediction. SHAPE values are predicted, not measured. Wet-lab
  validation is future work.

- `[CAVEAT]` 100mer benchmark only. We did not run on the 240mer OK7b
  set (compute-budget reasons). Defer to conference version.

- `[CAVEAT]` Training is ongoing — at submission the model is at episode
  6 of 10 of the latest fine-tune. Numbers reported are the snapshot
  at submission time (`best_policy_network.pt`, test_reward = 0.8937
  from episode 5).

- `[CAVEAT]` Loading the original Struct2SeQ.pt into our new RPE
  architecture is not a faithful AR baseline (zeros out the original's
  LSTM-PE). For the AR comparison we use the original
  `Struct2SeQ_training/` codebase + native arch, separate run.

- `[CAVEAT]` Motif in-painting at K=0% in our `bidir_inpaint_K00` run
  is identical to `bidir_random` (no positions fixed → empty constraint
  set → no-op). Reported as a sanity check that the in-painting
  plumbing doesn't introduce bias when there's nothing to fix.

---

## Open questions / things to write up after running more experiments

- AR Struct2SeQ + teacher-forced motif inpainting result.
- Best-of-N scaling curve from existing samples.csv.
- Mutation distance from WT (mirror Shujun Fig 6).
- Model confidence at WT positions (does the model "agree" with the
  motif constraint without being forced?).
- **Motif-redesign (Framing B) proof-of-concept.** Same selected motifs
  as Framing A, but invert the mask: fix everything *except* the motif
  to WT, regenerate just the hairpin / pseudoknot-stem itself. Pairs
  with the motif-preservation results to demonstrate "fix arbitrary
  subset, design complement" capability — both directions are
  AR-impossible.

---

## Future work (for paper §4)

- `[TEXT]` **Targeted application to aptamer / ligand-binding sites.**
  The motif-redesign capability (Framing B) could be applied specifically
  to redesign sequence around known functional sites: preserve the
  ligand-binding pocket / aptamer recognition motif, explore variants of
  the surrounding structural context. This is the natural synth-bio
  follow-up to the proof-of-concept here, paired with experimental
  validation (Eterna OpenKnot rounds, in-vitro binding assays).
- `[TEXT]` **Order-strategy ablation across all 20 puzzles** (paired-first
  vs loop-outward vs hardest-first).
- `[TEXT]` **Graded reward** (partner-distance or target-OK score)
  combined with REINFORCE/GRPO for sample efficiency. cf.
  `ideas/reward_design.md`.
- `[TEXT]` **240mer OK7b benchmark** at higher per-target compute.
- `[TEXT]` **Multi-seed runs** for statistical significance.
- `[TEXT]` **SHAPE-supervised retraining variant** (analog to
  Struct2SeQ-SHAPE).

---

## Random/uncategorised

(Drop any new ideas here as they come up; we'll re-tag and slot into
the right section above when drafting.)

- **Matched-baseline + budget-asymmetry framing** (Antonia, 22:30 UTC):
  Compare our performance on baselines run with the same 1000-sample
  budget. Highlight as a one-line main-text aside that we needed
  way fewer samples per target to beat the original method which used
  98 000 samples in the published paper. This stays in the main text;
  figures show only the matched-budget comparison.
