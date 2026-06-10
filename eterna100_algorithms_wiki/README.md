# Eterna100 Algorithms Wiki

A living, comparative knowledge base of algorithms that perform well on the
**Eterna100** RNA inverse-folding benchmark. The goal is to understand *how each
one actually solves puzzles* and contrast its solving logic with **our**
approach (Struct2SeQ-bidir), so we can borrow ideas and position our work.

> Eterna100 = 100 secondary-structure puzzles (dot-bracket), varying difficulty.
> A puzzle is **solved** when at least one designed sequence folds *exactly* to
> the target (base-pair distance 0 / Jaccard 1.0). Some puzzles fix certain bases.
> Human experts solved all 100; for a long time no *program* did.

## How to read this wiki
- **[our_approach.md](our_approach.md)** — the anchor: our Struct2SeQ-bidir method, restated in the same vocabulary we use to describe everyone else. Every algorithm page ends with a head-to-head against this.
- One file per external algorithm. Each page: what it is → solving logic → oracle & objective → domain heuristics → reported Eterna100 score → head-to-head vs ours → ideas worth stealing.

## Algorithms covered
| Page | Algorithm | Family | Status |
|---|---|---|---|
| [nemo.md](nemo.md) | NEMO (Portela 2018) | Nested Monte Carlo search + hand heuristics | ✅ written |

(more to come: EternaBrain, SentRNA, antaRNA, MODENA, NUPACK, RNAinverse, learning-based SOTA…)

## Master comparison table
Folding engine = the oracle the method optimizes *against*. "Search/puzzle" =
does it run a per-puzzle optimization loop, vs. a learned amortized generator.

| Algorithm | Family | Oracle / objective | Per-puzzle search? | Hand-crafted RNA heuristics? | Eterna100 solved | Budget | Source |
|---|---|---|---|---:|---:|---|---|
| **Ours: Struct2SeQ-bidir** | Learned policy (order-agnostic Q-learning) | RibonanzaNet-SS (learned); eval refold under RNet / Vienna2 / EternaFold | ❌ amortized (best-of-N sampling) | ❌ none | 52 (Vienna2) / 59 (RNet) / 64 (EternaFold) @K=1000 | 1 fwd pass × K samples, no per-puzzle loop | [our_approach.md](our_approach.md) |
| **NEMO** (Portela 2018) | Nested iterated Monte Carlo search (NMCS-B) | ViennaRNA 2.1.x MFE + base-pair distance | ✅ heavy (search + restarts) | ✅ extensive (fill probs, boosting, mismatch tables) | **95 / 100** | up to **24 h/puzzle** wall-clock | [nemo.md](nemo.md) |
| Cazenave–Fournier GNRPA (2020, NEMO follow-up) | Nested Rollout Policy Adaptation + NEMO heuristics as priors | same as NEMO (Vienna MFE + bp-distance) | ✅ | ✅ (reuses NEMO priors) | 92 @1h, 95 @2h | parallel, 1–2 h/puzzle | [nemo.md](nemo.md#follow-up-cazenavefournier-gnrpa-2020) |

## Comparison rubric (the dimensions we score every algorithm on)
1. **Search vs. amortization** — per-puzzle optimization loop, or a trained model that generates in one shot?
2. **Oracle** — what folding model defines "correct"? Thermodynamic (Vienna/NUPACK MFE) vs. learned (RibonanzaNet/EternaFold). *This is the single biggest confound when comparing solve counts across methods.*
3. **Objective** — exact base-pair match only, or energy/ensemble terms (ΔG, ensemble defect, probability) too?
4. **Domain knowledge** — hand-crafted RNA heuristics (GC/AU/GU fill ratios, loop boosting, mismatch rules) vs. learned-from-data.
5. **Compute budget** — seconds vs. hours per puzzle; parallel?
6. **Generality** — does it transfer beyond Eterna100 (e.g. longer designs, pseudoknots, our 240-mer / OpenKnot setting)?

## ⚠️ Apples-to-oranges warning (read before quoting any solve count)
Solve counts are **only comparable under the same fold-back oracle.** NEMO's
95/100 is measured under **ViennaRNA MFE** — the engine Eterna100 was originally
authored for. Our headline numbers refold under three oracles and our Vienna2
number (52/100) is the only directly-comparable one to NEMO's regime; even then,
budget differs by orders of magnitude (our amortized best-of-N vs. NEMO's hours
of per-puzzle search). When we tabulate "X solves N", always pin (oracle, budget,
K). See each page's head-to-head for the honest comparison.

**Subtlety that bites:** distinguish the **design oracle** (what a method
optimizes *against*) from the **check oracle** (what we refold with to score). NEMO
*designs against* Vienna, so it solves under Vienna by construction. We *design
against* RibonanzaNet and only *check* under Vienna — so our "fails under Vienna"
usually means "not optimized for Vienna." **Verified 2026-06-07:** the V2 targets
**are** Vienna-solvable — folding every official V2/Vienna2 reference solution under
our exact `RNA.fold` reproduces the target for **98/98** valid puzzles (the 2 gaps
are TSV data artifacts), incl. lone-pair puzzles like #33 (6 lone pairs). So our
56/100 Vienna gap is a **pure search/coverage problem** (our amortized sampling
doesn't find the existing, often narrow Vienna solutions), **not** an oracle config
bug and **not** target impossibility. The earlier "lone-pair Vienna-impossible"
claim is falsified (`evaluation/check_v2_reference_folds.py`). EternaFold's higher
count is just more-permissive, not more-correct. Still mind **V1 vs V2** — NEMO's 95
is on V1. See [nemo.md](nemo.md#why-our-vienna2-number-isnt-nemos-vienna-number).
