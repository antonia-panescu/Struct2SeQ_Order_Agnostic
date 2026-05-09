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

- `[NUMBER]` **OK7b 240mer headline (faithful Shujun protocol).** We
  re-implemented Shujun's exact `test_240.py` eval protocol:
  generate_sequence_batched(p=0.05) + generate_sequence_batched(p=0.10)
  + generate_sequence_batched_sample(p=1.0), each at K=333 per puzzle
  (1:1:1 ratio matching Shujun's `repeat=128` per call), plus the
  post-hoc rescue strategy from `test_240.py:177-225` (recursive 4^k
  enumeration on diff_pos ≤ 4 via Shujun's `search_v2.get_mutated`,
  imported directly — no reimplementation). **AR + faithful 3-strategy
  + rescue: 6.6% perfect, 17/20 puzzles solved**, 20,283 samples,
  20,243 unique sequences. **Ours bidir_random argmax-only (no rescue):
  33.2% perfect, 13/20 solved**, K=1000, 20,032 unique sequences.
  Apples-to-apples (both with rescue): ours 32.7% perfect / 13 solved
  vs AR 6.6% / 17 → **5.0× perfect-rate ratio in our favor on
  per-sample reliability; AR wins on puzzle coverage 17 vs 13.**

- `[NUMBER]` **Per-strategy breakdown on the AR side (240mer, K=333):**
  topk beam k=50 = 17.1% (strongest per-sample but only 5/20 puzzles
  due to k=50 vs K=333 budget gap); epsilon p=0.05 = 10.8% (12/20);
  epsilon p=0.10 = 8.8% (11/20); **Q-softmax = 0.07% (2/20)**. Q-softmax
  is essentially garbage on its own — Q-values aren't calibrated
  probabilities, so softmax sampling drifts far off-peak. At Shujun's
  K=98000 budget per puzzle, even 0.07% gives ~70 perfects per puzzle,
  which works; at K=333 it gives ~0.2 perfects per puzzle = essentially
  zero. This is a real per-sample-reliability indictment of the
  best-of-N regime: token-level sampling on AR Q-values requires
  enormous K to compensate for its low per-sample success rate.

- `[NUMBER]` **4-strategy (paper §2.7 with beam) actually performs
  WORSE than 3-strategy (test_240.py code).** When we add topk beam
  k=128 to the faithful 3-strategy mix and apply rescue, we get
  **16/20 solved** — one fewer than the 3-strategy + rescue (17/20).
  Why: topk produces medium-quality sequences that often become the
  per-puzzle "best non-perfect" seed, displacing the lower-jaccard
  but more-locally-rescuable seeds from the random-search variants.
  Rescue then can't reach jaccard=1.0 from those. **This validates
  Shujun's design choice to comment out beam in `test_240.py`: for the
  full pipeline, beam *hurts* puzzle coverage when paired with rescue.**

- `[NUMBER]` **Even our model in L→R mode beats orig in L→R mode.**
  Same-checkpoint ablation: bidir_identity (our checkpoint forced into
  L→R + argmax) gets **23.7% perfect, 10/20 solved** on 240mer. Orig
  L→R argmax gets **14.4% perfect, 5/20 solved**. So ~9 pp of the
  perfect-rate gain is from architecture (RPE bias replacing LSTM-PE +
  causal-conv) at fixed L→R order; the remaining ~10 pp gain to our
  random-perm 33.2% is from order randomization. Architecture and
  order both contribute roughly equally — paper claim should attribute
  to both, not order alone.

- `[NUMBER]` **Argmax + L→R is effectively deterministic on K=1000.**
  Verified by counting unique sequences: orig L→R argmax-only K=1000
  produces only 340 unique sequences (out of 20,032 samples; mean ~17
  unique per puzzle). Our bidir_identity (same arch, L→R + argmax) gets
  346 unique. **Our bidir_random + argmax K=1000 produces 20,032 unique
  sequences (one per sample).** Order-randomization is the diversity
  engine — token-level sampling is not required for our approach to
  produce diverse-yet-reliable designs.

- `[TEXT]` **DISCUSSION: Headline framing for §3 + §4.** What we
  defensibly claim:
  (i) Per-sample reliability — at K=1000 on OK7b 240mer, we produce
      33.2% perfect-Jaccard designs vs the orig Struct2SeQ baseline's
      6.6% under its full published protocol (eps0.05+eps0.10+qsoftmax
      + rescue, faithful `test_240.py`). **5.0× ratio.**
  (ii) Tied coverage at matched diversity engine — without rescue,
       both AR + 3-strategy and our random-perm-argmax solve 13/20
       puzzles. So the diversity engine alone gives the same coverage
       at K=1000.
  (iii) Rescue is a one-sided tool. With rescue, AR pulls ahead to
        17/20 (we stay at 13/20). The mechanistic story below explains
        why: rescue exploits AR's narrow-exploration failure mode
        (1-2 base mismatches reachable by local mutation). Our
        random-perm exploration leaves only structurally-entrenched
        failures that local repair can't touch.
  We do NOT claim rescue is a contribution of our method. We
  *implemented* a bidirectional analog using in-painting (best
  non-perfect → fix non-diff_pos → regenerate diff_pos with K=100
  random-perm) and verified it produces identical max_jaccard per
  puzzle as Shujun's 4^k enumeration — so neither paradigm rescues
  our seeds. We mention this only as analysis evidence, not as a
  proposed method.

- `[NUMBER]` **Rescue strategy is a Shujun-pipeline crutch that does
  not help our model.** Applied symmetrically to both pipelines:
  AR + 3-strategy + rescue gains +4 puzzles solved (13 → 17). Ours +
  rescue gains 0 puzzles (13 → 13). Why: rescue only fires when the
  best non-perfect sequence has diff_pos ≤ 4 *and* a 1-4 nt mutation
  reaches Jaccard=1.0. AR's failure modes are localised mismatches
  that local repair can fix; our failure modes are structural (entire
  pseudoknot stems wrong) that 1-4 nt edits cannot rescue. We are NOT
  dependent on post-hoc repair to be competitive on puzzle coverage —
  rescue is a useful AR-side tool, not a fundamental capability gap on
  our side.
- `[NUMBER]` Within our model, **random-perm decoding beats L→R-from-the-same-checkpoint
  by 38.6%** at producing perfect designs (45.5% vs 32.8%) — proving the
  capability lives in the *training distribution*, not just architecture.
- `[NUMBER]` **Principled structural-motif in-painting (motif-preservation
  mode, Framing A) achieves 37.9% perfect designs** on the OK7 puzzles
  (vs random scatter at matched mean fraction-fixed: 27.7%).
  Structurally-coherent constraints are *easier* for the model than
  scattered ones. **Framing**: this is the RFdiffusion / ProteinMPNN
  analog — *fix the motif (hairpin / pseudoknot-stem), design the
  surrounding scaffold* (mean fraction designed: 85.2%). The motif is
  the anchor; the scaffold is the design target. Solves 19/20 puzzles.

- `[NUMBER]` **Motif-redesign (Framing B) — same checkpoint, inverted
  mask — achieves 19.8% perfect designs** on the same OK7 puzzles at
  K=1000 (3974 / 20096). Same selected motifs as Framing A; everything
  EXCEPT the motif is fixed to WT (mean fraction designed: 14.8%).
  Solves 13/20 puzzles. Constraint-held: 100% on 500-row sample.
  By motif kind: hairpin 30.8% > internal_loop 19.7% > pseudoknot_stem
  15.7%. Sweet spot at 16-25 nt motifs: 36.6% perfect.
- `[NUMBER]` **Counter-intuitive finding: redesigning the smaller
  fraction of positions is HARDER than redesigning the larger
  fraction.** Framing A designs ~85% of positions and gets 37.9%
  perfect; Framing B designs ~15% of positions and gets 19.8% perfect
  (~half). Per-puzzle: A-wins 15/20, B-wins 4/20, tie 1/20. Both modes
  are AR-impossible — *only the order-agnostic decoding makes both
  directions accessible from a single checkpoint*.
- `[NUMBER]` **The B-vs-A gap is a training-distribution mismatch, not
  a decoding-order artifact.** We tested the natural hypothesis that
  the gap was caused by random-perm decoding leaving parts of the
  fixed scaffold out of the decoder's KV cache when motif positions
  are predicted. Re-ran Framing B with **fixed-first decoding** (decode
  all WT-scaffold positions first in random order, then the motif
  positions in random order — putting the entire scaffold in the
  decoder's KV cache before the motif is generated): **20.3% perfect,
  vs 19.8% with random-perm** — only +0.5 pp. So decoder ordering
  contributes <1 pp to the gap. The remaining ~18 pp gap is *intrinsic
  to the train-test distribution mismatch*: the model was trained on
  full-sequence generation with all positions free, never on "fix 85%
  of positions to externally-specified WT, design the remaining 15%".
  Clear future-work direction: train with mixed fraction-fixed (e.g.,
  random K-of-WT during training) so the model learns to condition on
  partial fixed scaffolds. Constraint-held verified: 100% on 500-row
  sample for fixed-first variant.
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
- `[NUMBER]` **Sample efficiency / moving away from best-of-N:**
  Expected number of samples to find one perfect-Jaccard design is
  **2.2 (ours)** vs **4.7 (orig AR L→R)** at the aggregate level —
  a **2.1× sample-efficiency ratio**. Per-puzzle (geometric mean
  across solvable puzzles): **4.3 (ours) vs 16.1 (orig AR)** — a
  **3.7× ratio**. Ours solves 19/20 puzzles (where ≥1 perfect exists
  in K=1000); orig AR only solves 15/20.

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

- `[TEXT]` **DISCUSSION: Mechanistic insight into Struct2SeQ inference
  protocol (use this in §4).** The mechanistic story we landed on after
  the K=1000 audit: in any AR generation pipeline with a single
  decoding order (L→R), pure argmax is essentially deterministic
  modulo CUDA float non-determinism — the model's full distribution
  collapses to a single point estimate per puzzle, so K=1000 argmax
  samples are ~17 unique sequences. Shujun's published K=98000
  protocol is precisely the workaround: three different sampling rules
  (epsilon-greedy at 5% and 10% uniform-among-allowed, plus Q-softmax
  multinomial) inject token-level randomness to recover diversity, and
  rescue strategy then enumerates local mutations on the best
  near-miss when diff_pos ≤ 4. The K=98000 figure is dominated by the
  cost of the per-step sampling rather than by genuine search: each
  individual sample has low probability of hitting Jaccard=1.0
  (Q-softmax 0.07%, epsilon ~10%), so massive K is needed for
  best-of-N to pay off. **Order randomization replaces this entire
  scaffolding.** A fresh permutation per sample produces 1000+ unique
  sequences from one argmax-only checkpoint at K=1000. We recover the
  full distribution through *order diversity*, not token diversity.
  This is why we don't need rescue: diff_pos for our failure modes is
  not constrained to be ≤ 4 (we don't fail by being one mutation away
  from a fixed near-miss; we fail by exploring a different basin
  entirely), so local repair doesn't apply.

- `[TEXT]` **DISCUSSION: K=1000 argmax-only AR was the wrong baseline
  to start with — note honestly in the paper.** Our initial K=1000 AR
  baseline produced 21.2% perfect on 100mer, 14.4% on 240mer, which
  by face value seemed to reproduce Shujun's published numbers. We
  later discovered that argmax-only AR at K=1000 produces only ~17
  unique sequences (CUDA float non-determinism is the only source of
  variance across samples), so the K=1000 budget is wasted: we are
  effectively running K~17. The numbers happened to match Shujun's
  published K=98000 + 3-strategy + rescue numbers because both end
  up close to the structural ceiling of an AR L→R argmax decoder for
  these puzzles. The real comparison required equipping AR with its
  full published diversity engine, which we did. We are honest about
  this in §4: the per-sample-reliability gap (5.0× perfect-rate ratio)
  is the load-bearing claim; the puzzle-coverage gap (13/20 vs 17/20
  with rescue) goes the other way and is acknowledged as a
  consequence of rescue's helpfulness on AR's localised failure modes.

- `[TEXT]` **L→R + argmax is effectively deterministic: ~17-20 unique
  sequences per K=1000 budget.** A direct check on our generated
  outputs: orig Struct2SeQ at K=1000 with argmax-only L→R produces
  only ~17-20 unique sequences per puzzle (mean 20 on 100mer, 17 on
  240mer; the remaining ~98% of samples are exact duplicates). Our
  own model under bidir_identity (L→R + argmax) shows the same
  effective-K~18. The small variance is CUDA float non-determinism
  occasionally breaking a near-tied argmax. By contrast, our
  bidir_random (random-permutation + argmax) produces 1000+ unique
  sequences per K=1000 budget. **Order-randomization is the diversity
  engine, not token sampling.** This sharpens the
  apples-to-apples comparison: at K=1000 argmax-only we get ~60×
  more unique designs per target than the L→R AR baseline, while
  also winning the perfect-Jaccard rate (45.5% vs 21.2%). The 3
  decoding strategies in Shujun's published K=98000 protocol
  (epsilon-greedy uniform, Q-softmax, topk beam) are how the AR
  baseline gets real diversity — so the comparison "K=1000 random-perm
  argmax (us) vs K=999 3-strategy mix (orig)" is the crucial test.

- `[TEXT]` **Per-sample reliability → moving away from best-of-N as
  a crutch.** The dominant paradigm in the field — including Shujun's
  K=98 000-per-target screening protocol — is brute-force best-of-N
  sampling: generate orders of magnitude more candidates than you need,
  then filter by Jaccard / OK score / experimental SHAPE. This is
  effectively a *workaround for low per-sample reliability*. Our
  order-agnostic decoding raises per-sample reliability (45.5% perfect
  vs 21.2% for orig AR, same K=1000), which means **half as many
  samples are needed** to find a perfect design (E[K] = 2.2 vs 4.7,
  per-puzzle E[K] = 4.3 vs 16.1, ~2-4× efficiency depending on the
  metric). Practical implications: less compute for in-silico screening,
  fewer wet-lab synthesis cycles, faster experimental iteration. Frame
  the paper around "the path forward is more reliable per-sample
  generation, not more samples". This is also why the
  budget-vs-quality scaling curve (Fig "best of N") matters: ours
  saturates at 95% by K=500 while AR L→R only at K=1000+ — for the
  same compute budget you get fundamentally better coverage.

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

## Conclusion / framing

- `[TEXT]` **Random-permutation decoding is the headline algorithmic
  contribution.** Concretely: at inference, for each sample we draw a
  fresh `perm = randperm(L)` and decode the L positions in that order
  using the bidirectionally-trained policy with relative-position bias
  (no causal mask, no L→R commitment). The same checkpoint can also do
  paired-first / loop-outward / fixed-positions-first by swapping the
  permutation rule — random uniform-perm is the default and the
  strongest single-strategy choice on OK7. This is what makes
  in-painting (Framing A and B), motif-redesign, and arbitrary
  conditional generation fall out of one trained model. Frame the
  conclusion around: *the algorithmic pivot from "AR L→R" to
  "stochastic permutation decoding" is the contribution; the perfect-
  Jaccard / sample-efficiency / motif-inpainting numbers are
  consequences of it.*
- `[TEXT]` **Why random-perm wins over the same checkpoint's L→R
  inference.** Same weights, only the inference-time decoding order
  differs (random-perm 45.5% perfect vs L→R-from-bidir 32.8%).
  Mechanism: averaging over many decoding orders lets the model
  "vote" using whatever order is easiest for that particular target
  — paired positions can be committed early when the sample's first
  draws happen to land there, scaffolds can be filled around an early-
  committed motif when the perm orders things that way, etc. Random-
  perm is essentially a **mixture of conditional distributions over
  decoding orders**, and that mixture concentrates more probability
  on perfect-Jaccard sequences than any single fixed order. This is a
  clean, paper-quotable framing that does not require any tuning
  knob — drawing a uniform perm per sample is the algorithm.

---

## Cross-checks vs Shujun's writeup

- `[TEXT]` **What is and isn't directly readable from Shujun's
  writeup.** Figs 4 (Struct2SeQ) and 5 (Struct2SeQ-SHAPE) of the
  reference writeup are exactly the *per-puzzle perfect-Jaccard counts*
  we want as reference bars — the numbers (AK_PK100-3, Diplonema, E.
  coli, …) are printed on each bar at K=98000. So we can read these
  off directly without re-asking Shujun. The thing we still need to
  ask Shujun for is the **raw 80th-percentile OpenKnot score per
  puzzle**, because Fig 7 only publishes the *z-scored* 80th-pct OK
  scores (across-method normalization for the CASP-style summed bar
  chart) — the raw 80th-pct OK values are not given anywhere in the
  paper. So when we follow up with Shujun, narrow the ask to: "raw
  80th-pct OK scores per puzzle for K=98000 Struct2SeQ and
  Struct2SeQ-SHAPE on OK7 100mer."

---

## Future work (for paper §4)

- `[TEXT]` **Targeted application to aptamer / ligand-binding sites.**
  The motif-redesign capability (Framing B) could be applied specifically
  to redesign sequence around known functional sites: preserve the
  ligand-binding pocket / aptamer recognition motif, explore variants of
  the surrounding structural context. This is the natural synth-bio
  follow-up to the proof-of-concept here, paired with experimental
  validation (Eterna OpenKnot rounds, in-vitro binding assays).
- `[TEXT]` **Better rescue mechanisms for order-agnostic models.**
  Local mutation rescue (Shujun's 4^k enumeration) and our model-guided
  in-painting rescue both hit the same structural ceiling on our K=1000
  output: failures are not 1-4 nt off, they're whole-context off.
  Future directions worth investigating: (a) multi-seed bidir-rescue
  (rescue from top-N non-perfect, not just top-1, so different fixed
  scaffolds give different rescue contexts); (b) iterated refinement
  (rescue → re-evaluate diff_pos → rescue again, local-search style);
  (c) structurally-aware order ensembles within each rescue (paired-first,
  loop-outward); (d) RL-fine-tuning specifically on near-misses so the
  model learns to escape structural basins. These are deferred; the
  current paper's claim does not depend on them.

- `[TEXT]` **Train with mixed fraction-fixed (in-painting-aware
  training).** Our motif-redesign result (Framing B) shows the model
  has a real but limited capability when ~85% of positions are
  externally fixed — because training never exposed it to that
  conditioning regime. Adding a random K-of-WT mask to a fraction of
  training samples (sampled K ∈ [0, 1]) would give the model explicit
  experience filling small free regions inside large fixed scaffolds.
  Should close most of the ~18 pp B-vs-A gap.
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
