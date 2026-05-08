"""Principled motif extraction from RNA secondary structure.

A *motif* here is a complete structural element:
  - hairpin           — a stem of nested pairs ending in an unpaired loop
  - internal_loop     — a stem with exactly one inner stem (with unpaired
                        residues on both sides separating them)
  - bulge             — special case of internal_loop with zero unpaired
                        residues on one side
  - multi_loop        — a stem closing a junction of >=2 inner stems
  - pseudoknot_stem   — a contiguous run of crossing pairs

Each motif's `positions` set is the set of residue indices the motif
"owns" — both its closing pairs and its loop residues. Fixing the
nucleotide identities at these positions to WT preserves the motif as
a unit (intact base pairs + intact loop sequence).

Algorithm (~200 LOC):
  1. Parse the dot-bracket into pairs (allow pseudoknots).
  2. Use OpenKnotScorePipeline.identify_crossing_bps to split nested
     vs crossing pairs.
  3. Group nested pairs into stems by adjacency:
       (i, j), (i+1, j-1), (i+2, j-2), ...
     until the run breaks.
  4. Build a parent-child tree: stem A is parent of stem B if A's
     outermost pair encloses all of B's pairs.
  5. For each stem, classify based on what's inside:
       0 inner stems -> hairpin (positions = stem + loop)
       1 inner stem  -> internal_loop / bulge
                        (positions = outer stem's closing pair +
                         flanking unpaired residues + inner stem's
                         closing pair)
       >=2 inner stems -> multi_loop
                        (positions = outer stem's closing pair +
                         all unpaired junction residues + the
                         closing pairs of each inner stem)
  6. Cluster crossing pairs into pseudoknot stems by adjacency.

CLI:
    python -m evaluation.motif_extraction \\
        --targets-csv /home/.../Round3_targets.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Make sure ARNIEFILE is set so arnie.utils imports cleanly.
os.environ.setdefault(
    "ARNIEFILE", "/home/nvidia/haiwen/antonia/struct2seq_bidir_rl/arnie_file.txt"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, "/home/nvidia/haiwen/antonia/OpenKnotScorePipeline/src")

from arnie.utils import convert_dotbracket_to_bp_list  # noqa: E402
from openknotscore.pipeline.scoring import identify_crossing_bps  # noqa: E402


# --------------------------------------------------------------------------- types


@dataclass(frozen=True)
class Motif:
    kind: str                                   # see module docstring
    positions: frozenset                        # set[int] of residue indices
    closing_pairs: tuple                        # tuple[tuple[int, int], ...]
    size: int                                   # = len(positions)
    span: tuple                                 # (min_pos, max_pos) for plotting

    def to_dict(self):
        return {
            "kind": self.kind,
            "positions": sorted(self.positions),
            "closing_pairs": list(self.closing_pairs),
            "size": self.size,
            "span": list(self.span),
        }


# --------------------------------------------------------------------------- helpers


def _group_into_stems(pairs):
    """Group base pairs `(i, j)` with `i < j` into stems.

    A stem is a contiguous run `(i, j), (i+1, j-1), (i+2, j-2), ...`.
    Returns a list of stems; each stem is a list of `(i, j)` pairs in
    order from the outermost pair to the innermost.
    """
    if not pairs:
        return []
    sorted_pairs = sorted(set(tuple(p) for p in pairs))
    used = [False] * len(sorted_pairs)
    pair_index = {p: idx for idx, p in enumerate(sorted_pairs)}
    stems = []
    for idx, (i, j) in enumerate(sorted_pairs):
        if used[idx]:
            continue
        stem = [(i, j)]
        used[idx] = True
        # extend inward
        ii, jj = i + 1, j - 1
        while (ii, jj) in pair_index and not used[pair_index[(ii, jj)]]:
            stem.append((ii, jj))
            used[pair_index[(ii, jj)]] = True
            ii += 1
            jj -= 1
        stems.append(stem)
    # sort stems by outer pair (i ascending)
    stems.sort(key=lambda s: s[0])
    return stems


def _stem_outer(stem):
    return stem[0]


def _stem_inner(stem):
    return stem[-1]


def _stem_encloses(outer_stem, inner_stem):
    """Does `outer_stem`'s outermost pair enclose all of `inner_stem`?"""
    oi, oj = _stem_outer(outer_stem)
    ii, ij = _stem_outer(inner_stem)
    return oi < ii and ij < oj


def _build_stem_tree(stems):
    """For each stem, find its direct parent stem (the smallest enclosing one).

    Returns a list of children-lists (parallel to `stems`); also returns
    a list of root stem indices (those with no parent).
    """
    n = len(stems)
    # parent[k] = index of the smallest enclosing stem, or -1
    parent = [-1] * n
    for k in range(n):
        outer = _stem_outer(stems[k])
        smallest_enclosing = -1
        smallest_size = None
        for q in range(n):
            if q == k:
                continue
            if _stem_encloses(stems[q], stems[k]):
                qi, qj = _stem_outer(stems[q])
                size = qj - qi
                if smallest_size is None or size < smallest_size:
                    smallest_size = size
                    smallest_enclosing = q
        parent[k] = smallest_enclosing
    children = [[] for _ in range(n)]
    roots = []
    for k, p in enumerate(parent):
        if p < 0:
            roots.append(k)
        else:
            children[p].append(k)
    return children, roots


def _classify_stem(stem, children_of_this_stem, stems):
    """Classify what type of motif this stem closes, and return its
    `positions` set + closing pair list.

    Returns (kind, positions, closing_pairs).
    """
    inner_pair = _stem_inner(stem)              # innermost pair of this stem
    inner_i, inner_j = inner_pair
    outer_i, outer_j = _stem_outer(stem)
    n_kids = len(children_of_this_stem)

    if n_kids == 0:
        # Hairpin: stem + loop
        kind = "hairpin"
        # all residues from outer_i to outer_j (inclusive)
        positions = set(range(outer_i, outer_j + 1))
        closing_pairs = tuple(stem)
        return kind, positions, closing_pairs

    if n_kids == 1:
        # Internal loop or bulge
        child_idx = children_of_this_stem[0]
        child_stem = stems[child_idx]
        ci, cj = _stem_outer(child_stem)
        # unpaired residues between inner pair of this stem and outer pair
        # of the child stem: (inner_i+1 .. ci-1) on the 5' side and
        # (cj+1 .. inner_j-1) on the 3' side.
        kind = "internal_loop"
        # positions = outer pair + entire inner pair + unpaired flanks +
        # closing pair of inner stem (so the inner stem's closing
        # nucleotides are also fixed and complement each other)
        positions = {outer_i, outer_j, inner_i, inner_j, ci, cj}
        positions.update(range(inner_i + 1, ci))
        positions.update(range(cj + 1, inner_j))
        closing_pairs = (
            _stem_outer(stem),
            inner_pair,
            _stem_outer(child_stem),
        )
        return kind, positions, closing_pairs

    # multi-loop
    kind = "multi_loop"
    # positions: this stem's outer + inner pair, all unpaired junction
    # residues (between inner_i and inner_j minus child stems'
    # enclosed regions), plus each child stem's outer pair.
    positions = {outer_i, outer_j, inner_i, inner_j}
    closing_pairs_list = [_stem_outer(stem), inner_pair]
    occupied = set()
    for child_idx in children_of_this_stem:
        ci, cj = _stem_outer(stems[child_idx])
        positions.add(ci)
        positions.add(cj)
        closing_pairs_list.append((ci, cj))
        # mark the inside of each child stem as "not part of junction"
        occupied.update(range(ci, cj + 1))
    # junction residues = anything between (inner_i+1) and (inner_j-1)
    # not enclosed by any child stem.
    for k in range(inner_i + 1, inner_j):
        if k not in occupied:
            positions.add(k)
    return kind, positions, tuple(closing_pairs_list)


def _crossing_stems(crossed_bps):
    """Cluster crossing pairs into pseudoknot stems."""
    return _group_into_stems(crossed_bps)


# --------------------------------------------------------------------------- public API


def extract_motifs(dot_bracket: str) -> list[Motif]:
    """Decompose a dot-bracket (with pseudoknots) into structural-element motifs.

    Returns a list of Motif objects. The motifs do NOT partition the
    structure — a child stem's closing pair is also referenced by its
    parent multi/internal-loop motif. That overlap is intentional:
    fixing a multi-loop's positions includes the closing pair of each
    of its children, but the children's hairpin motifs (with their
    interior loops) are separate motif candidates.
    """
    bps = convert_dotbracket_to_bp_list(dot_bracket, allow_pseudoknots=True)
    bps = [tuple(p) if not isinstance(p, tuple) else p for p in bps]

    # Identify which pairs participate in any crossing
    crossed_residues = set(identify_crossing_bps(bps))
    nested_pairs = [
        (i, j) for (i, j) in bps
        if i not in crossed_residues and j not in crossed_residues
    ]
    crossed_pairs = [
        (i, j) for (i, j) in bps
        if i in crossed_residues or j in crossed_residues
    ]

    motifs: list[Motif] = []

    # Nested side: build stems + tree + classify
    nested_stems = _group_into_stems(nested_pairs)
    children, _roots = _build_stem_tree(nested_stems)
    for k, stem in enumerate(nested_stems):
        kind, positions, closing_pairs = _classify_stem(stem, children[k], nested_stems)
        if not positions:
            continue
        motifs.append(
            Motif(
                kind=kind,
                positions=frozenset(positions),
                closing_pairs=tuple(closing_pairs),
                size=len(positions),
                span=(min(positions), max(positions)),
            )
        )

    # Pseudoknot side: each crossing-pair stem becomes a motif
    for stem in _crossing_stems(crossed_pairs):
        positions = set()
        for (i, j) in stem:
            positions.add(i)
            positions.add(j)
        if not positions:
            continue
        motifs.append(
            Motif(
                kind="pseudoknot_stem",
                positions=frozenset(positions),
                closing_pairs=tuple(stem),
                size=len(positions),
                span=(min(positions), max(positions)),
            )
        )

    # sort: by start position then by size, deterministic
    motifs.sort(key=lambda m: (m.span[0], m.size))
    return motifs


# --------------------------------------------------------------------------- CLI


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--targets-csv", type=str,
        default="/home/nvidia/haiwen/antonia/OpenKnotAIDesignData/Targets/Round3_targets.csv",
    )
    args = p.parse_args()

    import pandas as pd
    df = pd.read_csv(args.targets_csv)
    print(f"{'idx':>3} {'puzzle':<32} {'kind':<16} {'size':>5} {'frac':>6} {'span':<10}")
    print("-" * 80)
    total_motifs = 0
    by_kind = {}
    for i, row in df.iterrows():
        L = int(row["Length"])
        title = row["Title"][:31]
        motifs = extract_motifs(row["Dot-bracket"])
        for m in motifs:
            print(
                f"{i:>3} {title:<32} {m.kind:<16} {m.size:>5} "
                f"{m.size/L:>6.2f} {str(m.span):<10}"
            )
            by_kind[m.kind] = by_kind.get(m.kind, 0) + 1
        total_motifs += len(motifs)
        print()
    print(f"=== summary ===")
    print(f"total motifs:    {total_motifs}")
    for k, c in sorted(by_kind.items()):
        print(f"  {k:<16}: {c}")
    print(f"avg motifs/puzzle: {total_motifs / len(df):.1f}")


if __name__ == "__main__":
    _cli()
