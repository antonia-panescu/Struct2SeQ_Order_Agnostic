"""Thermodynamic / learned folding oracles for Eterna100 re-scoring.

Provides a uniform MFE-style folding interface that takes an RNA sequence and
returns its predicted secondary structure as (dot_bracket, base_pair_set), so
the same downstream Jaccard-vs-target logic used for RibonanzaNet-SS can score
ViennaRNA and EternaFold predictions.

Oracles
-------
- "vienna"     : ViennaRNA 2 MFE structure via the `RNA` Python package
                 (RNA.fold == RNAfold -d2 defaults). This is the canonical
                 reference engine the Eterna100-V2 targets were tuned for.
- "eternafold" : EternaFold prediction via arnie's `mfe(package='eternafold')`,
                 which calls the contrafold binary with EternaFoldParams.v1
                 (default MEA structure -- the standard EternaFold fold-back).

All oracles are pseudoknot-free; the Eterna100-V2 target set contains no
pseudoknots, so this is exact for these targets.

Designed to be import-light so it can be used inside multiprocessing workers:
it does NOT import torch / the training stack. Dot-bracket parsing is a local
stack-based implementation (handles () [] {} <> for robustness on targets).
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List, Set, Tuple

BP = Tuple[int, int]
_BRACKETS = {")": "(", "]": "[", "}": "{", ">": "<"}
_OPENERS = set(_BRACKETS.values())


def dotbracket_to_bps(db: str) -> Set[BP]:
    """Dot-bracket -> set of (i, j) base pairs with i < j.

    Supports nested pseudoknot bracket families () [] {} <>. Unmatched
    brackets are ignored (defensive; oracle output is always balanced).
    """
    stacks: Dict[str, List[int]] = {o: [] for o in _OPENERS}
    pairs: Set[BP] = set()
    for k, ch in enumerate(db):
        if ch in _OPENERS:
            stacks[ch].append(k)
        elif ch in _BRACKETS:
            st = stacks[_BRACKETS[ch]]
            if st:
                i = st.pop()
                pairs.add((i, k))
    return pairs


def _clean(seq: str) -> str:
    return seq.strip().upper().replace("T", "U")


@lru_cache(maxsize=None)
def _vienna_db(seq: str) -> str:
    import RNA  # in-process, fast

    structure, _mfe = RNA.fold(seq)
    return structure


@lru_cache(maxsize=None)
def _eternafold_db(seq: str) -> str:
    # ARNIEFILE must point at arnie_file.txt with an `eternafold:` entry.
    from arnie.mfe import mfe

    return mfe(seq, package="eternafold")


def fold_vienna(seq: str) -> Tuple[str, Set[BP]]:
    db = _vienna_db(_clean(seq))
    return db, dotbracket_to_bps(db)


def fold_eternafold(seq: str) -> Tuple[str, Set[BP]]:
    db = _eternafold_db(_clean(seq))
    return db, dotbracket_to_bps(db)


_ORACLES = {"vienna": fold_vienna, "eternafold": fold_eternafold}


def get_oracle(name: str):
    """Return a callable seq -> (dot_bracket, base_pair_set)."""
    try:
        return _ORACLES[name]
    except KeyError:
        raise ValueError(
            f"unknown oracle {name!r}; choose from {sorted(_ORACLES)}"
        )


def jaccard(pred_bps: Set[BP], target_bps: Set[BP]) -> float:
    """Base-pair-set Jaccard. Two empty sets are perfectly similar (1.0)."""
    if not pred_bps and not target_bps:
        return 1.0
    union = pred_bps | target_bps
    return len(pred_bps & target_bps) / len(union) if union else 1.0


if __name__ == "__main__":  # tiny self-test
    os.environ.setdefault(
        "ARNIEFILE",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "arnie_file.txt"),
    )
    s = "GGGGAAAACCCC"
    for name in ("vienna", "eternafold"):
        db, bps = get_oracle(name)(s)
        print(f"{name:11} {db}  bps={sorted(bps)}")
