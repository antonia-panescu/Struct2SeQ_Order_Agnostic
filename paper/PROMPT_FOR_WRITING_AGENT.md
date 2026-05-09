# Prompt for paper-writing agent

Goal: produce a final, ICML-quality 4-page workshop submission for ICML
GenBio 2026 (deadline 2026-05-09 11:59 UTC). Below: current paper state,
required changes, headline numbers, and Shujun's most recent (validating)
external comments to incorporate into the discussion.

---

## Paper

- LaTeX root: `icml2026/paper.tex`
- Bib: `icml2026/references.bib`
- Style: ICML 2026 workshop (anonymous, double-blind, non-archival)
- Page limit: 4 pages main + unlimited refs/appendix
- Section skeleton (already in place): Introduction, Related work, Method
  (order-agnostic decoding + structural-motif in-painting), Experiments,
  "Best-of-N is not the right axis," Limitations, Conclusion, Impact

---

## Required changes

### Style direction (read first)

- **No meta-language about being "an extension of Struct2SeQ".** We
  cite Struct2SeQ as the AR baseline we improve over and as the source
  of the DQN training framework / scoring pipeline / OK7b benchmark.
  We do NOT frame our work as "completing Struct2SeQ's §4 future work"
  or as a derivative project. Frame our contribution as a *new method*
  that happens to share infrastructure with Struct2SeQ for fair
  comparison, not as Struct2SeQ-with-an-extension.
- **Don't over-praise Struct2SeQ.** It's a concurrent baseline, not a
  foundation we stand on. References should be functional: cite for
  the OK7b challenge / RibonanzaNet env / DQN-AR training, not as
  inspiration. ICML reviewers will resent excessive deference; it
  reads as low-confidence.
- **Lead with WHY**: every contribution claim should answer "what
  practical improvement does this enable?" — not "how does this differ
  from Struct2SeQ." E.g.:
  - "Random-permutation decoding is more sample-efficient than the
    best-of-$N$ regime that dominates RNA design at the time of writing,
    by 2--5$\times$ on per-sample perfect-Jaccard rate."
  - "Native in-painting unlocks principled motif scaffolding for RNA
    (RFdiffusion-analog), which is necessary for redesigning aptamers
    around fixed ligand-binding pockets and for engineering riboswitches
    around fixed regulatory motifs."
  - "Order-agnostic decoding enables the same trained checkpoint to
    perform multiple sequence-design tasks (unconstrained generation,
    motif preservation, motif redesign) without retraining or
    architecture changes."
- **RNA-domain motivation**: lift the in-painting framing to its
  synth-bio use cases. Aptamer / riboswitch design — fix the
  ligand-binding aptamer motif, redesign the surrounding regulatory
  scaffold (or vice versa). Cite candidate papers like:
  - Wachsmuth et al. 2013 (computer-designed riboswitches),
  - Townshend et al. 2018 (Eterna OpenRiboswitch),
  - Penchovsky 2014 (review of RNA-based logic),
  - Joshi et al. 2024 (gRNAde, geometric motif scaffolding for RNA).
  Bib agent should pull these via the ICML 2026 references.bib if not
  already present.

### Top-level constraints
- **HARD 4-page limit.** Current draft already overshoots without our
  additions. Cut anything not essential. Concrete cuts below.
- **No concept figure** — drop entirely. Replace its slot with the
  unified centered results table (see §2 below).
- **OK7b 240-mer only.** No OK7 100-mer numbers anywhere.
- **For the original Struct2SeQ baseline, plain generation only.** Do
  not include AR motif in-painting numbers (those are in our
  in-painting section, where AR's "teacher-forced motif" is shown
  as the strongest available AR-side hack).
- **Both metrics on every row**: Perfect-Jaccard \% AND Puzzles solved
  (out of 20). Reported uniformly throughout text and tables.

### 1. Replace the OK7 100-mer headline with the OK7b 240-mer faithful comparison

The current `tab:main` reports OK7 100-mer numbers based on a
*now-obsolete* baseline (`orig L→R argmax-only K=1000` produced only
~17 unique sequences per puzzle due to CUDA float non-determinism — the
K=1000 budget was wasted). The new headline is on **OK7b 240-mer** under
Shujun's exact `test_240.py` protocol.

Switch dataset, switch protocol. All numbers below.

### 2. SINGLE unified results table (replaces all the prior tables)

**No concept figure** — dropped to save page space. Replace the slot
with the centered table below. **OK7b 240-mer only** (no 100-mer
numbers). Both metrics (`Perfect-Jaccard \%` AND `Puzzles solved`)
reported on every row.

LaTeX skeleton (single column, centered, sections via `\multicolumn`
italic headers and `\midrule`). Drop the `Unique seqs` and `Notes`
columns from prior drafts — the 4-page limit doesn't allow them.

```latex
\begin{table*}[t]
  \centering
  \caption{OK7b 240-mer pseudoknot design at matched $K{\sim}1000$
    samples per target. Perfect-Jaccard \% is the fraction of designs
    that reproduce the target structure exactly (Hungarian on
    RibonanzaNet-SS predicted base-pair probabilities, $\theta{=}0.5$).
    Puzzles solved is the count with $\ge 1$ perfect-Jaccard design at
    $K{=}1000$. The original Struct2SeQ row uses the published
    architecture and checkpoint trained for 10 episodes; ``Ours''
    uses our order-agnostic Transformer-decoder variant trained for 5
    additional episodes under the random-permutation distribution.
    The 3-strategy protocol is $\epsilon$-greedy with $p{=}0.05$ +
    $\epsilon$-greedy with $p{=}0.10$ + Q-softmax sampling, each with
    $K{\approx}333$ samples per target, matching
    \citet{he2026struct2seq}'s \texttt{test\_240.py}. Rescue is the
    post-hoc 4$^k$ local repair from \texttt{test\_240.py:177--225}
    on the per-puzzle best non-perfect seed.}
  \label{tab:main}
  \small
  \begin{tabular}{lcc}
    \toprule
    Method & Perfect-Jaccard (\%) & Puzzles solved \\
    \midrule
    \multicolumn{3}{@{}l}{\emph{Plain generation. Original Struct2SeQ (AR L$\to$R, 10 episodes)}} \\
    \midrule
    argmax-only                                          & 14.35  & 5/20  \\
    3-strategy                                           & 6.55   & 13/20 \\
    3-strategy + rescue                                  & 6.64   & 17/20 \\
    \midrule
    \multicolumn{3}{@{}l}{\emph{Plain generation. Ours (bidirectional, +5 extra episodes)}} \\
    \midrule
    L$\to$R argmax-only                                  & 23.72  & 10/20 \\
    L$\to$R + 3-strategy                                 & 9.38   & 12/20 \\
    L$\to$R + 3-strategy + rescue                        & 9.13   & 13/20 \\
    \textbf{Random-perm argmax-only}                     & \textbf{33.24} & \textbf{13/20} \\
    Random-perm + 3-strategy                             & 13.65  & 14/20 \\
    Random-perm + 3-strategy + rescue                    & 13.44  & 15/20 \\
    \midrule
    \multicolumn{3}{@{}l}{\emph{In-painting. Ours (random-perm)}} \\
    \midrule
    \textbf{Motif preservation}                          & \textbf{25.69} & \textbf{13/20} \\
    Motif preservation + rescue                          & 25.28  & 13/20 \\
    Motif redesign (AR-impossible)                       & 8.79   & 5/20  \\
    \bottomrule
  \end{tabular}
\end{table*}
```

**12 rows, three sections:**
- 3 rows orig Struct2SeQ baselines (plain generation, argmax / 3-strat / 3-strat+rescue)
- 6 rows ours plain generation (L$\to$R triple + random-perm triple, parallel structure)
- 3 rows ours in-painting (motif preservation, +rescue, motif redesign)

**No orig row in the in-painting section** per the user's choice — frame
in-painting as a capability story (we can do it natively; AR cannot
natively support motif redesign at all). If a contextual AR anchor is
needed, an optional row could read:

```latex
    Struct2SeQ AR + teacher-forced motif (closest AR-side hack)        & 11.17  & 7/20  \\
```

But default is ours-only.

Notes for the layout:
- `table*` (full-width float) so the table can sit centered across both
  columns even if we use `twocolumn` formatting; if the ICML template
  uses single-column, drop the `*` (use plain `table`).
- `@{}l` left-aligns the section headers without column spacing.
- Italic section markers (`\emph{...}`) instead of horizontal rules
  for sub-sections — saves vertical space.
- Bold the two headline rows: ours-random-perm-argmax (the per-sample
  reliability headline) and ours-motif-preservation (the in-painting
  capability headline).
- `\small` font keeps the table compact.
- Drop the `Unique seqs` column (mention in body text only — "AR
  argmax-only at K=1000 produces only ~340 unique sequences, an
  effective $K{\sim}17$"). Same for the `Notes` column.

### 3. Per-strategy ablation — APPENDIX or compact paragraph in body

Don't put per-strategy AR ablation in the main table. Either:

(a) **Body-paragraph version** (recommended for 4-page limit):

> The 3-strategy protocol breaks down as follows on OK7b 240-mer
> ($K{=}333$ each): for orig Struct2SeQ AR, $\epsilon{=}0.05$ achieves
> $10.77\%$ / $12$, $\epsilon{=}0.10$ achieves $8.79\%$ / $11$,
> Q-softmax achieves $0.07\%$ / $2$. For ours under random-perm,
> the same three strategies achieve $24.03\%$ / $13$, $16.89\%$ /
> $14$, $0.07\%$ / $3$ respectively. Q-softmax sampling is
> essentially zero on both architectures because Q-values are not
> calibrated probabilities (cf.~\citet[\S 2.7]{he2026struct2seq}).

(b) **Appendix table version** (only if there is space):

| Strategy | $K$ | AR perfect\% / solved | Ours rand-perm perfect\% / solved |
|---|---:|---:|---:|
| $\epsilon{=}0.05$       | 333 | 10.77 / 12 | 24.03 / 13 |
| $\epsilon{=}0.10$       | 333 | 8.79 / 11  | 16.89 / 14 |
| Q-softmax $p{=}1.0$     | 333 | 0.07 / 2   | 0.07 / 3   |
| topk beam $k{=}50$ (cf. paper §2.7; commented out in `test_240.py`) | 50  | 17.10 / 5  | n/a |
| topk beam $k{=}128$     | 128 | 8.91 / 7   | n/a |

### Comprehensive discussion-points list (USE THIS for §4 / §5 sectioning)

Below is the curated list of all the discussion ideas we have surfaced
through experiments and conversation. The writing agent should pick
the strongest 3--4 to land in §4 and squeeze the rest into a short
appendix or future-work bullets, given the 4-page limit.

1. **Per-sample reliability vs puzzle coverage as a Pareto trade-off**
   (the headline). At matched K=1000 we offer 5$\times$ more reliable
   designs; with the full Shujun protocol AR pulls coverage by 4
   puzzles via rescue. We frame this honestly: rescue is a real tool
   that helps AR more because of failure-mode locality.

2. **Order-randomization replaces token sampling as the diversity
   engine**. AR-with-argmax produces ~17 unique sequences from K=1000
   (CUDA non-determinism). Our random-perm-with-argmax produces 1000
   unique. AR's 3-strategy (eps-greedy + Q-softmax) is the workaround
   for argmax-determinism that we replace with order randomization.
   Quote attribution: this framing was independently confirmed by
   Shujun He in correspondence ("random order argmax combines high
   average performance of argmax with exploration enabled by random
   order; with L→R this wasn't possible") — anonymise or paraphrase
   as the venue allows.

3. **Argmax + L→R is effectively deterministic at K=1000** (~17
   unique seqs). We verified this empirically. K=1000 budget is wasted
   under L→R argmax. Brief paragraph + table-footer remark.

4. **Q-softmax pathology**. On both architectures, Q-softmax sampling
   gives 0.07\% perfect at K=333 because Q-values are not calibrated
   probabilities (writeup §2.7). Quantitative confirmation of
   uncalibrated-Q caveat. Implication: don't sample from Q at low K.

5. **Why rescue is one-sided: over-pair vs under-pair failure modes**
   (the most novel mechanistic finding). AR seeds skew toward
   over-pairing (rescuable: mutate to break unwanted pair). Our
   random-perm seeds skew toward under-pairing (not rescuable: 4^k
   mutation cannot create a pair the network refuses to predict from
   context). Manually verified on puzzle 12 (UC2414, 240mer): all 16
   mutations of the C-G pair at positions 14, 21 fail to trigger the
   pair RibonanzaNet refuses; max jaccard = 0.984.

6. **Half-blind AR teacher-forced motif vs native random-perm
   in-painting**. Mechanically distinct: AR commits the upstream
   scaffold *before* visiting the motif (no motif info in KV cache);
   downstream of the motif gets motif-aware via the overridden tokens.
   Random-perm pre-fills the motif into rna\_outputs and the
   partner-mask reads it at every decode step. Empirical 2.3$\times$
   gap on motif preservation (25.69 vs 11.17) is the direct
   consequence. (Compare to ours-LR + teacher-forced motif on the same
   checkpoint — TBD numbers in flight.)

7. **Architecture vs decoding-order ablation**. Same-checkpoint
   ablation isolates the +9.5\,pp from random-perm (clean) and
   confounded +9.4\,pp from architecture+training (5 extra episodes).
   Both contribute roughly equally to the perfect-rate gap.

8. **Pseudoknot-specific structural rationale**. L→R commits to one
   half of every crossing pair without seeing the partner. Random
   perms decode partners first ~half the time, so the partner-mask
   has structural constraints to apply. Why our gain is
   pseudoknot-shaped.

9. **Best-of-N is the wrong axis** (this section already exists in the
   draft). Our point: per-sample reliability is a more meaningful
   axis than best-of-N coverage; cite analogous arguments from LM
   literature (lightman2024verify; wang2023selfconsistency) where
   verifier or self-consistency methods replace pure best-of-N.

10. **In-painting capabilities AR cannot do natively**. Motif redesign
    (fix scaffold, design middle motif) is impossible under L→R AR
    teacher-forcing; works natively under random-perm via the
    `fixed=` argument. Capability claim, not a head-to-head.

11. **Rescue is good — for both pipelines**. Don't dismiss as a
    "crutch." It gives ours-3-strategy +1 puzzle (14→15) too. The
    asymmetry in *how much* it helps is mechanistic, not a rhetorical
    point.

### Future-work bullets (recommended, all derived from analyses we did)

- **Closing the under-pair seed bottleneck**: under-pairing failures
  can't be locally rescued because the network refuses to predict the
  missing pair from context. Future work: targeted RL fine-tuning on
  near-miss seeds, larger-radius mutation search, or training the
  model to do "structural pair-retrieval" iterations as a generation
  refinement.
- **In-painting-aware training**: train with a random K-of-WT mask on
  a fraction of training samples. Motif-redesign currently
  underperforms motif-preservation at matched K because training
  never exposed the model to the conditioning regime "fix 85\% of
  positions, design 15\%."
- **Aptamer / riboswitch redesign as targeted application**: fix the
  ligand-binding aptamer pocket as the constrained motif, redesign
  the surrounding regulatory scaffold to vary the ligand response
  threshold or kinetics. Pair with experimental SHAPE / binding
  validation.
- **Order-aware beam search** over joint (order, token) space.
  Defined but non-trivial; deferred.
- **Multi-seed iterated rescue** for under-pair failures.
- **Larger compute budget** (K-sweep) to verify whether structural
  ceilings are real or just K=1000 limitations.
- **Multi-seed runs / error bars** for statistical significance
  between configs (workshop submission, conference version).
- **240-mer Struct2SeQ trained checkpoint** (we used what we believed
  to be the 240-mer-trained Struct2SeQ.pt; full ablation against
  Shujun's separately-trained 240-mer Struct2SeQ-SHAPE deferred).

### 3. Algorithm exposition: keep light, reference Struct2SeQ

**Recommendation: TWO compact algorithm boxes** — one for random-perm
decoding, one for in-painting (which is a one-line modification to
the first). Both fit in single-column space. They are the
methodological backbone the reviewer will look for in an ML paper.

```latex
\begin{algorithm}[t]
  \caption{Random-permutation argmax decoding.}
  \label{alg:randperm}
  \begin{algorithmic}[1]
    \State \textbf{Input:} target structure $S$, model $\pi_\theta$, length $L$.
    \State Sample permutation $\sigma \sim \mathrm{Uniform}(S_L)$.
    \State Initialise output buffer $y \leftarrow \emptyset$, KV cache $\mathcal{K} \leftarrow \emptyset$.
    \For{$t = 0, \ldots, L-1$}
      \State Decode position $\sigma_t$: $\hat{q} \leftarrow \pi_\theta(\sigma_t \mid y, S, \sigma, \mathcal{K})$.
      \State Apply structural mask: $\hat{q} \leftarrow \hat{q} - \mathrm{mask}(\sigma_t, y, S)$.
      \State Emit $y_{\sigma_t} \leftarrow \arg\max \hat{q}$; update $\mathcal{K}$.
    \EndFor
    \State \Return $y$.
  \end{algorithmic}
\end{algorithm}
```

```latex
\begin{algorithm}[t]
  \caption{In-painting decoding (motif preservation / redesign).}
  \label{alg:inpaint}
  \begin{algorithmic}[1]
    \State \textbf{Input:} target $S$, fixed mask $f \in (\{-1\} \cup \{0,1,2,3\})^L$, model $\pi_\theta$.
    \State \textbf{Pre-fill:} $y \leftarrow f$ at positions where $f_i \ge 0$. $\Comment{Constrained nucleotides visible to partner-mask from step 0.}$
    \State Sample permutation $\sigma \sim \mathrm{Uniform}(S_L)$ (or use \emph{fixed-first} order).
    \For{$t = 0, \ldots, L-1$}
      \State Decode position $\sigma_t$ as in Algorithm~\ref{alg:randperm}.
      \If{$f_{\sigma_t} \ge 0$}
        \State Override $y_{\sigma_t} \leftarrow f_{\sigma_t}$.
      \EndIf
    \EndFor
    \State \Return $y$.
  \end{algorithmic}
\end{algorithm}
```

The two algorithms together fit in less than half a column. They make
the methodological contribution explicit: random-perm + structural-
mask is the generation step, and in-painting is a *one-line
pre-fill* that exposes any subset of positions to the partner-mask
from step zero.

**Keep the body method section** focused on:

- **Architecture difference**: original = LSTM-PE encoder + LSTM-PE +
  causal-conv decoder. Ours = same encoder, replaced decoder with pure
  Transformer using a learned relative-position bias on self+cross
  attention scores keyed on the permutation. This single change is what
  enables any decoding order to share the same parameters.
- **Training**: same DQN + experience replay + twice-shifted action-
  value training as~\citet{he2026struct2seq}, but with a fresh random
  permutation $\pi \sim \mathrm{Uniform}(S_L)$ drawn per training
  sample. Causal mask is in **step order**, not RNA-position order.
- **Inference**: fresh permutation per sample at $K{=}1000$;
  argmax decode at each step (no token-level sampling). KV cache is
  consistent across step-order autoregression.
- **Motif in-painting** (one paragraph): the same trained checkpoint
  supports two operations by adjusting the `fixed=` mask in
  `generate_permuted`. Motif preservation = fix the motif positions
  to wild-type, design the surrounding scaffold (RFdiffusion analog).
  Motif redesign = fix the surrounding scaffold to wild-type, design
  just the motif (local mutagenesis analog; AR-impossible natively).
  Constraint-held verified 100% on a 500-row sample.

- **AR vs ours in-painting framing for §3 / §4** (the explicit
  hack-vs-native paragraph). For motif preservation specifically:

  > Teacher-forced motif preservation in an L→R AR decoder is a
  > degraded hack with a structural asymmetry that depends on motif
  > position. Concretely, suppose the motif occupies positions
  > $[a, b]$ in the middle of a length-$L$ target. The decoder visits
  > positions in L→R order: at each step before $a$, the model emits
  > a token from its current decoder state, with no information about
  > the motif WT identity (the motif positions have not been visited
  > yet, so they are absent from the KV cache). At each step in
  > $[a, b]$, the model's emission is overridden to the WT base; the
  > KV cache then contains the (overridden) WT identity going forward.
  > For positions $> b$, the decoder is now motif-aware via the KV
  > cache. The result: the upstream scaffold is generated *blind* to
  > the motif identity, while the downstream scaffold is *informed*.
  > For pseudoknotted targets where long-range pairs cross the motif
  > boundary (e.g., a base at position $\ll a$ paired with one at
  > position $\gg b$), upstream commits before seeing what its
  > eventual partner's context will be.

  > Our random-permutation decoder pre-fills the constrained positions
  > in the structural-output tensor `rna_outputs` *before* the decode
  > loop starts (\texttt{Functions.py:404}). The partner-pairing
  > constraint logic (\texttt{Functions.py:435--445}) reads the motif
  > nucleotide identities from this tensor at every decode step,
  > regardless of where the motif positions land in the random
  > permutation. So both upstream and downstream of the motif (or
  > any subset of "fixed" positions) the design steps are aware of
  > the structural constraints implied by the WT motif, from step
  > zero. We frame this as *native* in-painting: structural awareness
  > of fixed positions is built into the generation loop, not
  > retrofitted post-hoc.

  Numbers backing this: motif preservation on OK7b 240-mer at $K{=}1000$
  gives AR teacher-forced $11.17\%$ / $7$ vs ours random-perm
  $25.69\%$ / $13$ ($2.3\times$ perfect-rate ratio). See \cref{tab:main}.

### 4. Discussion: comprehensive, ICML-style, with RNA insight

The discussion should hit these points (in roughly this order):

#### 4.1 Per-sample reliability vs puzzle coverage (the headline trade-off)

Frame as a Pareto trade-off between two metrics that the field has
historically conflated. Best-of-$N$ at $K{=}98{,}000$ optimizes
coverage-via-sampling; we offer reliability-via-architecture.

- Across-protocol (their best vs ours best at $K{\sim}1000$): 5.0× perfect-rate ratio in our favor.
- Matched-protocol (both under 3-strategy + rescue): 2.0× perfect-rate ratio in our favor; AR leads coverage by 2 puzzles.
- Operating-point story: we offer two extremes (33.24%/13 reliability-optimised, 13.44%/15 coverage-optimised); AR has only one regime (sampling+rescue is required to make K=1000 produce diverse sequences at all).

#### 4.2 The diversity-engine substitution

Cite Shujun's recent confirmation in conversation:

> "Random order argmax combines the high average performance of argmax
> with the exploration enabled by random order. With L→R this wasn't
> possible." (Shujun He, 2026-05-08, personal communication; quote
> with attribution if non-anonymous, paraphrase if anonymous-blind)

Concretely:
- Argmax + L→R is essentially deterministic at K=1000 (~17 unique
  sequences per puzzle for both checkpoints under L→R+argmax — verified
  empirically on samples.csv outputs).
- Argmax + random-permutation produces 20{,}032 unique sequences from
  K=1000 budget (one per sample).
- Token-level sampling (epsilon-greedy + Q-softmax) on AR is the
  workaround for argmax-determinism; we replace that workaround with
  order-randomization, which is *additive* with argmax rather than
  *substituting* for it.

This is the central methodological observation. Frame it as an
algorithm-design decoupling: "diversity from inference order" is a
lever orthogonal to "per-sample reliability of the trained policy,"
and we exploit both.

#### 4.3 Why rescue is one-sided (mechanistic)

Rescue is a real tool, not a "crutch we dismiss" — it gives our
random-perm + 3-strategy +1 puzzle (14→15) too. But it gives AR a
bigger boost (+4 puzzles, 13→17). Why:

- AR's failure modes are 1-2 base mismatches (narrow exploration →
  locally rescuable by 4^k enumeration).
- Random-perm argmax-only failure modes are structurally entrenched
  (whole-context wrong, not a few bases). Verified: Shujun-rescue,
  bidir-rescue (model-guided in-painting), AND multi-seed bidir-rescue
  all give 0/7 successful rescues on our argmax-only seeds.
- When we add 3-strategy generation on top of random-perm, we re-introduce
  some narrow-exploration near-misses (some samples drift into
  AR-like locally-rescuable regimes), which rescue then helps recover —
  hence ours +1 puzzle there.

This is a clean mechanistic finding: **rescue's effectiveness depends
on the failure-mode locality of the generator**. Different decoding
regimes produce different failure modes; whether rescue helps depends
on whether those failures are locally repairable.

#### 4.4 Q-softmax is uncalibrated at low K

Both architectures show Q-softmax sampling at 0.07% perfect for
K=333. Shujun's writeup §2.7 explicitly notes Q-values "do not have a
direct probabilistic interpretation"; our results quantify the
practical consequence: token-level sampling from softmax over Q-values
requires K≥10{,}000 budget per puzzle to be useful (0.07% × 10{,}000 ≈
7 perfects). At K=1000 it's effectively zero. Implication for the
field: don't sample from Q if you're compute-constrained.

#### 4.5 RNA-specific structural intuition

Why does random-permutation help on pseudoknots specifically? Because
pseudoknots have crossing pairs: position $i$ pairs with $j$ where
$|j-i|$ may be large. In L→R AR, when the decoder commits to the
nucleotide at $i$, it has not yet seen any context downstream of $i$,
so it cannot use the to-be-decoded partner $j$ as a constraint. With
random permutations, on average half the time the partner is decoded
*before* the constrained position, so the partner-mask logic in
generation has something to constrain against. This is a clean
RNA-domain rationale for why the order-agnostic extension is
particularly well-suited to pseudoknotted targets — the more
structurally "crossing" a target is, the more L→R loses.

(The OK7 and OK7b challenges are pseudoknot-only; our gain is largest
on the most pseudoknotted puzzles. Per-puzzle correlation analysis
deferred to appendix.)

#### 4.6 Limitations to acknowledge upfront

- Single seed; no error bars on between-config comparisons. Workshop
  paper, deferred to conference version.
- K=1000 budget is 100× smaller than Shujun's published K=98{,}000.
  Our 2-5× per-sample reliability gain compensates, but absolute
  coverage at K=1000 is necessarily lower than at K=98{,}000.
- 240mer model (architecture+training) is confounded with 5 extra
  training episodes. The decoding-order effect (+9.5pp at L→R→
  random-perm on the same checkpoint) is clean; the architecture+
  training effect (+9.4pp at L→R-orig→L→R-ours) is partially
  confounded.
- Predicted (not experimental) SHAPE; predicted (not measured)
  structure via RibonanzaNet-SS. Same as Shujun's published evals.
- Rescue's local repair search is a single-seed, ≤4-position
  enumeration — more sophisticated rescue (multi-seed, iterated, RL-
  finetuning on near-misses) is out of scope for this workshop paper.

### 5. Tone: ICML-style technical depth + RNA insight

This is an ML paper appearing at a biology workshop. Technical depth
should match the ML community's expectations:
- DQN, twice-shifted training, partner-mask logic — cite + brief
  formalism.
- Order-agnostic AR (XLNet, ARDM, ProteinMPNN) — acknowledge prior
  work properly.
- Decoupling of "diversity engine" from "per-sample reliability" — call
  this out as a methodological contribution, not just a result.

But also speak to the RNA design audience:
- Pseudoknot-specific rationale (§4.5 above).
- Synthetic-biology framing for in-painting (RFdiffusion analog for
  motif preservation; mutagenesis analog for motif redesign).
- Future-work bullet on aptamer / ligand-binding-site redesign as the
  natural application of motif redesign.

Keep equations minimal — most of the formal apparatus is identical to
Struct2SeQ's; cite-and-defer for shared content, focus paper space on
what is *different*.

### 6. Specific TODOs in the paper.tex right now

Search for `\TODO` markers — there are several open ones. Replace
with:
- Headline-table discussion paragraph (use numbers from this prompt).
- Motif-redesign vs preservation paragraph (already mostly written;
  decide whether to keep in table or push to text).
- Abstract (write last; bullets in this prompt).
- Decision on whether to keep the OK7 100-mer numbers as a smaller
  reference table in appendix or omit entirely. Recommendation:
  **omit** — 240-mer is the only headline; report 240-mer Pareto
  trade-off table in main; appendix table can show per-strategy.

### 7. Ablation paragraphs to write (using existing data)

- **Architecture-vs-order ablation** (already in `ideas_for_writing.md`):
  ours-L→R-argmax 23.72% vs orig-L→R-argmax 14.35% = +9.37pp
  architecture+training; ours-random-perm-argmax 33.24% vs
  ours-L→R-argmax 23.72% = +9.52pp decoding order. Roughly equal.
- **Q-softmax pathology**: 0.07% perfect on both architectures at
  K=333 → ~0.2 perfects per puzzle → essentially zero. Q-values are
  not probabilities; sampling drifts off-peak. Quantitative version of
  Shujun writeup §2.7 caveat.

---

## Files to read

| File | Why |
|---|---|
| `icml2026/paper.tex` | Current paper draft (with stale 100-mer numbers) |
| `icml2026/references.bib` | Bibliography |
| `paper/ideas_for_writing.md` | Running scratchpad of all paper-relevant numbers and framings (FULLY UPDATED) |
| `lab_notebook/2026-05-09.md` | Today's protocol notes with full per-config breakdown |
| `results/ok7b_eval/SUMMARY_240mer.md` | Clean comparison tables, paper-ready |
| `results/ok7b_eval/MASTER_240mer.csv` | Single source of truth for all per-config aggregates |
| `evaluation/build_240mer_master.py` | Re-runnable aggregator (execute to refresh MASTER + see grouped output) |
| `Struct2SeQ/test_240.py` | Shujun's exact eval protocol (what we replicated) |
| `Struct2SeQ/writeup.pdf` | Shujun's paper, especially §3.2 (240-mer challenge) and §2.7 (inference) |

## Hard deadline

2026-05-09 11:59 UTC. ~5 h from now (as of writing). Anonymisation
pass before submission: grep author names, lab name, repo URL, GitHub
handles. Submit via OpenReview link in the workshop CFP.

## What NOT to do

- Don't claim bidir-rescue or multi-seed bidir-rescue as proposed
  methods. They are analysis-only (verifying the structural-ceiling
  explanation).
- Don't include topk beam in the headline 3-strategy mix. Shujun
  commented it out in his actual eval; reporting it confuses the
  comparison. Mention separately.
- Don't push the "AR is a crutch" framing. Rescue is a real tool;
  the asymmetry is mechanistic (failure-mode locality), not a
  rhetorical dismissal.
- Don't use 100-mer numbers in the headline. The user explicitly chose
  240-mer as the only benchmark.
