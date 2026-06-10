# NEMO — NEsted MOnte Carlo RNA Puzzle Solver

- **Author:** Fernando Portela (citizen scientist, Eterna Massive Open Laboratory)
- **Paper:** *An unexpectedly effective Monte Carlo technique for the RNA inverse folding problem*, bioRxiv [345587](https://www.biorxiv.org/content/10.1101/345587v1) (posted 2018-06-14).
- **Code:** [github.com/eternagame/nemo](https://github.com/eternagame/nemo) — C, links ViennaRNA 2.1.x, OpenMP.
- **Headline result:** **95 / 100** Eterna100 solved within **24 h/puzzle** — the best *program* score reported at the time (humans solve all 100).

> Primary sources used here: the GitHub README (CLI + build), and the
> Cazenave–Fournier follow-up *Monte Carlo Inverse Folding* (arXiv
> [2005.09961](https://arxiv.org/abs/2005.09961), 2020), which documents NEMO's
> internals — score function, fill probabilities, mismatch/boosting tables —
> precisely because it reuses them as priors. Numbers below are quoted from those.

## What it is, in one line
A **per-puzzle stochastic search**: nested Monte Carlo over the sequence,
guided by **hand-crafted RNA domain heuristics**, scored by **ViennaRNA's
thermodynamic fold** (base-pair distance + a free-energy gap term), with
iterated restarts that re-randomize the parts that misfold.

## Core algorithm — NMCS-B (Nested Monte Carlo Search, modified)
NEMO runs **several iterations of NMCS-B**, a lightly modified Nested Monte
Carlo Search, and between iterations keeps part of the best solution, re-opens
the bad parts, and restarts ("**nested iterated**").

**Representation.** A candidate = a sequence string. Assigned positions hold a
letter `A/U/G/C`; unassigned positions hold `N`. The search "works on" the `N`s.

**Level-1 NMCS.** At each state (= a partially filled sequence), for each
possible move it runs a number of **playouts**, plays the move that produced the
best playout, advances to the next state, and repeats until the sequence is
fully assigned (a terminal state).

**Moves.**
- A move takes the **first `N`** in the sequence and assigns it.
- If that position is **paired** in the target, both bases of the pair are
  assigned **simultaneously** — only `AU`, `GC`, `GU` (and reverses) can pair, so
  it's natural to set them together. Paired positions ⇒ 6 ordered move options;
  unpaired ⇒ 4.

**Playouts.** Roll out to a complete sequence using the **biased domain
heuristics** below (not uniform random). **Modification vs. classic NMCS:**
NMCS-B **retains the single best playout seen so far across the whole run**, not
just per-level.

**Scoring** (playouts evaluated by):
```
        ⎧  K / (1 + ΔG)      if K > 0
score = ⎨
        ⎩  K · (1 + ΔG)      otherwise

K   = 1 − BPD / (2 · NumTargetPairs)
```
- `BPD` = base-pair distance: number of differing pairs between the sequence's
  ViennaRNA-folded structure and the target.
- `NumTargetPairs` = number of pairs in the target.
- `ΔG` = (MFE of the sequence's actual fold) − (free energy the sequence would
  have *in the target structure*). I.e. how far the target is from being the
  ground state. `K=1` and `ΔG=0` ⟺ perfect, uniquely-folding solution.

So the objective is **not just "match the pairs"** — it also pushes the target
to be the minimum-free-energy structure (thermodynamic uniqueness), which is
what makes designs robust under ViennaRNA.

**Outer iterated loop / restarts.** If a NMCS-B iteration doesn't solve it,
NEMO **keeps part of the best sequence** and frees up (re-randomizes) the set of
bases that (a) **don't fold correctly**, (b) **their neighborhood**, and (c) a
few **randomly selected** bases — then restarts NMCS-B. This escapes the
chaotic-landscape traps where one base flip wrecks the fold.

## Fill order & inpainting (does it fill L→R? can it inpaint?)
**Order is deterministic/positional, not learned and not free-random.** A move
takes the **first `N`** in the string and assigns it, **paired bases first**, and
fills **both partners of a pair simultaneously**. So the *order* is a fixed,
structure-informed traversal (stems-first, positional); what's stochastic is the
**base value** at each step (heuristic-weighted playouts), **not the order**.
NEMO therefore does **not** exploit order-agnosticism the way we do — its power
is search-over-values + restarts, not order flexibility.
*(Exact within-hole ordering — strictly positional vs. a paired-pass-then-unpaired
pass — to confirm in the C source; the paper wording is slightly ambiguous.)*

**Inpainting: yes, natively — it's the core mechanism.** Holes are `N`, assigned
positions are letters, and the search only works on the `N`s. The CLI even takes
an optional `[<start_sequence>]`. The **iterated-restart loop is repeated
inpainting**: keep the good part of the best sequence, re-open (`→N`) the bases
that *misfold + neighborhood + a few random*, re-search just those. Puzzle-imposed
fixed bases are handled identically (kept as letters, never `N`).

> **vs. ours:** both can inpaint, different mechanism. Ours = one **learned
> forward pass over arbitrary held positions, any order** (our order-agnostic
> edge). NEMO = a **search loop over holes in fixed positional order**.

## Domain knowledge & heuristics (the secret sauce)
These are hand-chosen from Eterna-player experience, *not* learned. They bias
the playout sampling.

**Paired-base fill probabilities** (closing pairs of outermost stacks; "left/right" = inside-the-loop view):
| Context | GC/CG | AU/UA | GU/UG |
|---|---:|---:|---:|
| General case | 60% | 33% | 7% |
| Left-most in junction | 82% | 11% | 7% |
| Right-most in junction | 37% | 56% | 7% |

The 60/33/7 general split roughly matches natural-RNA pair frequencies.

**Unpaired-base fill (general):** A/U/G/C = **93% / 1% / 5% / 1%** (A-rich loops).

**Mismatch handling** — context-dependent tables. E.g. for a mismatch adjacent
to a paired base, the distribution depends on the partner base; in internal
loops it depends on whether the mismatch is already assigned; in junctions /
external loops mismatches are 97/1/1/1 (A-dominant). These encode terminal-
mismatch and loop-closing tricks Eterna players use.

**"Boosting."** Strong, near-deterministic rules (>80% in specific cases —
triloops, 1×1 and 2×2 internal loops, terminal mismatches) that place specific
nucleotide combinations at "boosting points" to **lower the loop's free energy**
and stabilize the intended fold. Crucially NEMO does *not* apply boosting 100%
of the time — the hardest puzzles sometimes need unconventional solutions.

## Build / usage (from the repo)
```
./nemo [-E] [-v] [-i <num_iter>] "<target_structure>" [<start_sequence>]
cat list.e100 | xargs -L 1 ./nemo -E -v     # batch over the Eterna100 list
```
Compiles against the ViennaRNA 2.1.x source tree with GCC + OpenMP. `-E` = Eterna
mode, `-i` = iteration budget, `-v` = verbose. Heuristics documented in a linked
Google doc.

## Reported Eterna100 performance
- **NEMO: 95/100** within 24 h/puzzle (ViennaRNA MFE oracle). The 5 it never
  solved are the benchmark's hardest (the Cazenave follow-up names puzzles
  100, 99, 97, 91, 90, 78 as the ones their related methods struggled with too).
- At publication this was the **best automated score** on Eterna100; prior
  programs (RNAinverse, NUPACK, MODENA, antaRNA, etc.) and even learning methods
  of that era (EternaBrain, SentRNA) sat well below — Portela's framing was that
  a "simple" Monte Carlo + good heuristics *unexpectedly* beat them all.

### Follow-up: Cazenave–Fournier GNRPA (2020)
The arXiv follow-up generalizes NEMO's idea with **Nested Rollout Policy
Adaptation (NRPA)** and a variant **GNRPA** that injects **NEMO's heuristic
probabilities as fixed priors** (`β_ib = log(NEMO prob)`) into a *learned*
rollout policy. Best result: **92/100 in 1 h** wall-clock (leaf-parallel GNRPA
with restarts), **95/100 in 2 h** — matching NEMO with far less per-puzzle time.
Key knobs: search level (1→2→3), policy adaptation rate α, beam size,
"correction" (boosting) on/off, restart threshold = seq length / 5, root vs.
leaf parallelization. *Takeaway: NEMO's hand-priors + an adaptive search policy
is a strong, parallelizable recipe.*

## Prior vs. search — what's actually doing the work
A natural take is "the heuristics do everything, so NEMO's *search* is weak / its
algorithm is worse than ours." Half right — and the wrong half matters.

**Prior-free search is genuinely weak.** Cazenave's independent reimplementation
(Table 1), ablating NEMO's heuristic priors (β):
| Config | Without NEMO priors | With NEMO priors |
|---|---:|---:|
| Level-1 search | **3 / 100** | **30 / 100** |
| Level-2 search | 49 / 100 | 73 / 100 |
Adding search *depth* on top of a good prior then climbs 30 → 73 → 85 (beam +
level 3) → **92 (1 h) / 95 (2 h)** with restarts.

**But "heuristics vs. their algorithm" is a false split.** The heuristics *are*
NEMO's prior — exactly as our network weights are *our* prior. The honest framing:

| | Prior (no deep search) | + search on top |
|---|---:|---|
| NEMO | hand-coded heuristics ≈ **30** (L1, Vienna) | deep NMCS + defect-restarts → **95** |
| Ours | **learned** policy ≈ **52** (Vienna, best-of-N) | only shallow best-of-N (non-adaptive) |

So the real signal **favors us**: our *learned* prior looks **stronger and far
cheaper** than their hand-prior (52 vs ~30 at comparable shallow-search footing —
and ours wasn't even optimized for Vienna; see next section). What we lack isn't a
better prior — it's the **search layer**, the exact thing that takes NEMO 30 → 95.
*Correct takeaway:* we hold the expensive half (a strong, cheap prior); they hold
the half we're missing (per-puzzle adaptive search). **Bolt search onto our prior
→ likely leapfrog.** *(Caveat: 52-vs-30 isn't a clean A/B — different Vienna roles,
K, target set; suggestive, not proof.)*

## Why our Vienna2 number isn't NEMO's Vienna number
NEMO scores 95 under Vienna; we report only 52 under Vienna2 — **not because we're
doing it wrong, but because Vienna plays a different role in each setup.**

- **NEMO: Vienna is the design *target*.** It folds with Vienna *inside its search
  loop*, thousands of times/puzzle, and stops only when Vienna folds to target. A
  solve means **Vienna agrees by construction.**
- **Us: Vienna is a *post-hoc check*.** We design against **RibonanzaNet** (the
  training reward), then refold with Vienna2 only to check. "Unsolved under Vienna"
  mostly means **"we never optimized for Vienna,"** not "the puzzle is unsolvable."

**✅ VERIFIED (2026-06-07): the V2 targets ARE Vienna-solvable; our gap is pure
search.** Test: fold every official **V2/Vienna2 reference solution** with our exact
`RNA.fold` (ViennaRNA 2.7.2, `-d2` defaults) and compare to the V2 target
(`evaluation/check_v2_reference_folds.py`).
- **98/98 valid puzzles reproduce their target exactly** (Jaccard 1.0). The only 2
  gaps are **data artifacts**: #100's reference cell is truncated (13 nt vs 381 nt),
  #22's V2 structure column is empty in the TSV. No alternate Vienna setting
  (`noLP`, `dangles`) was needed — defaults already reproduce all.
- **Positive controls** on the puzzles the per-target FT called "impossible": #66
  (1 lone pair), #88 (2), #65 (1), #62 (bulge), and **#33 (6 lone pairs)** all
  reproduce **exactly**. Vienna (`noLP=0`) forms lone pairs fine given the right
  sequence.

This **rejects both** earlier hypotheses: (a) Vienna config/version mismatch — no,
our engine reproduces the references; (b) targets Vienna-MFE-impossible — no, a
Vienna-folding solution provably exists for essentially all of them.

**⚠️ The 2026-06-06 per-target-FT "lone-pair impossible" conclusion was WRONG** — an
inference error, not a data error. The FT data (plateau at N−1 even with Vienna
reward, RNet saturating to 434/1000) was real, but the conclusion "Vienna refuses
the lone pair → impossible" is falsified by the references. Truth: **designing a
sequence whose Vienna-MFE realizes a specific lone pair is genuinely hard** (narrow
favorable sequence space), and a 20-epoch per-target FT is an *underpowered search*.
NEMO finds these via *hours of deep, defect-guided, Vienna-directed search*. **That
is the entire difference.**

**Net:** our 56/100 Vienna gap vs NEMO's 95 is a **pure search gap** — strongest
evidence yet for bolting a per-puzzle search loop onto our learned policy. Nothing
about the oracle or the targets blocks us.

**EternaFold caveat (softened):** EternaFold's higher count (67) reflects that it's
**more permissive / easier to hit by sampling**, *not* that Vienna is "wrong" about
these targets. Vienna is a valid thermodynamic gold standard we simply under-search.
Choose the oracle by what we want to claim, not by which gives a bigger number.

**Remaining confound (still real):** NEMO's 95 is on the **original V1** set (2018);
ours is **V2** (post-2021) — different puzzles, so the 95-vs-56 number was never a
clean head-to-head regardless. But the *search-gap* conclusion holds either way.

## Head-to-head vs. ours (Struct2SeQ-bidir)
| Dimension | NEMO | Ours (Struct2SeQ-bidir) |
|---|---|---|
| Family | Per-puzzle nested Monte Carlo **search** | **Amortized learned** policy + best-of-N |
| Oracle | ViennaRNA MFE (thermodynamic) | RibonanzaNet-SS (learned); refold-checked on 3 oracles |
| Objective | bp-distance **+ ΔG** (thermo uniqueness) | exact bp-match only |
| Domain heuristics | **Extensive, hand-crafted** (fill ratios, boosting, mismatch tables) | **None** — learned from `top2M.csv` |
| Compute | **Hours/puzzle**, search loop + restarts | **Seconds**, one fwd pass × K, no loop |
| Eterna100 | **95/100** (Vienna) | 52 (Vienna2) / 59 (RNet) / 64 (EternaFold) @K=1000 |
| Generality | Tuned to short Eterna puzzles / Vienna model | Trained on 240-mer genome windows; targets long/natural RNA + OpenKnot |

**Honest read.** Under the same regime NEMO compares to (Vienna MFE), it solves
95 vs. our 52 — but it spends *hours of guided search per puzzle plus decades of
encoded RNA folklore*, while we do *one amortized forward pass with zero
hand-heuristics*. These optimize different things: NEMO maximizes solve-rate on
this specific benchmark under Vienna; we bet on a fast, general, learned
generator. The gap on the hardest ~40 puzzles is essentially the gap between
**search + thermodynamic feedback + boosting** and **single-shot amortized
sampling**.

## Ideas worth stealing (NEMO → us)
1. **Bolt a search loop onto our policy.** Our network is a *much smarter
   playout/prior* than NEMO's hand-weights. Use Struct2SeQ-bidir as the proposal
   distribution inside an NMCS/NRPA-style per-puzzle search (exactly the GNRPA
   move, but with a learned prior instead of `60/33/7`). Likely the single
   highest-leverage upgrade for hard puzzles.
2. **Defect-targeted resampling.** NEMO's restart = re-open *only* the
   mis-folding bases + neighborhood + a few random. We already refold for solve-
   checking; feed the per-position defect back to the order-agnostic policy
   (which can fill in any order!) to *resample just the broken region* instead of
   redrawing whole sequences. This is a natural fit for our bidirectional design.
3. **Add a thermodynamic / ΔG term at eval.** A best-of-N reranker that scores by
   ViennaRNA bp-distance + ΔG (NEMO's score function) before committing — cheap,
   and rewards designs where the target is the true ground state, improving
   transfer under the Vienna oracle.
4. **Boosting as a learned-or-applied post-process.** Triloop / 1×1 / 2×2
   internal-loop boosting rules are deterministic energy wins; applying (or
   learning) them could rescue near-miss designs at near-zero cost.
5. **Time-matched comparison.** To compare fairly we should report solve-rate at
   *matched wall-clock budget*, not just K. NEMO@24h vs ours@(K, seconds) are
   different points on the curve; a search-augmented ours would let us trace the
   whole curve.

## Open questions / to verify later
- Exact NMCS-B playout counts per move and the iteration schedule (in the C
  source / Google doc; README didn't expose constants).
- Whether NEMO's 95/100 used any per-puzzle manual hints or fixed-base handling
  beyond the benchmark's imposed bases.
- How NEMO's solved set overlaps with ours under Vienna2 — a per-puzzle diff
  would show exactly which hard puzzles search unlocks that amortization misses.
