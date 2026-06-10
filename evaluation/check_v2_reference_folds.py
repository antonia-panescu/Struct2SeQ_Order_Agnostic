"""Sanity check: do the OFFICIAL Eterna100-V2 reference solutions fold back to
their V2 target under *our* Vienna call (RNA.fold defaults, == RNAfold -d2)?

This forks the "why is our Vienna solve-rate low" question:
  - If a reference does NOT reproduce its target  -> Vienna config/version issue
    (the target IS Vienna-solvable; our engine settings differ), OR a genuinely
    Vienna-MFE-impossible target (lone pair / anti-thermo bulge).
  - We then probe alternate Vienna model settings (noLP, dangles) on the
    failures to separate "config mismatch" from "true impossibility".

Data: data/eterna100/eterna100_puzzles.tsv
  col 'Secondary Structure V2'        = target
  col 'Sample Solution (V2/Vienna2)'  = the canonical V2-under-Vienna2 reference
  col 'Sample Solution (V2/Vienna1)'  = V2-under-Vienna1 reference (cross-check)
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Set, Tuple

import RNA

TSV = os.path.join(os.path.dirname(__file__), "..", "data", "eterna100", "eterna100_puzzles.tsv")
BP = Tuple[int, int]
_BRACKETS = {")": "(", "]": "[", "}": "{", ">": "<"}
_OPENERS = set(_BRACKETS.values())


def dotbracket_to_bps(db: str) -> Set[BP]:
    stacks: Dict[str, List[int]] = {o: [] for o in _OPENERS}
    pairs: Set[BP] = set()
    for k, ch in enumerate(db):
        if ch in _OPENERS:
            stacks[ch].append(k)
        elif ch in _BRACKETS:
            st = stacks[_BRACKETS[ch]]
            if st:
                pairs.add((st.pop(), k))
    return pairs


def clean(seq: str) -> str:
    return seq.strip().upper().replace("T", "U")


def fold_default(seq: str) -> str:
    """Exactly our pipeline's call."""
    structure, _ = RNA.fold(clean(seq))
    return structure


def fold_md(seq: str, dangles: int = 2, noLP: int = 0, temperature: float = 37.0) -> str:
    md = RNA.md()
    md.dangles = dangles
    md.noLP = noLP
    md.temperature = temperature
    fc = RNA.fold_compound(clean(seq), md)
    structure, _ = fc.mfe()
    return structure


def is_lone(pair: BP, pairs: Set[BP]) -> bool:
    i, j = pair
    return (i - 1, j + 1) not in pairs and (i + 1, j - 1) not in pairs


def n_lone(pairs: Set[BP]) -> int:
    return sum(is_lone(p, pairs) for p in pairs)


def classify(target_bps: Set[BP], pred_bps: Set[BP]) -> str:
    missing = target_bps - pred_bps   # target has, prediction lacks (under-pairing)
    extra = pred_bps - target_bps     # prediction has, target lacks (over-pairing)
    if not missing and not extra:
        return "exact"
    missing_lone = sum(is_lone(p, target_bps) for p in missing)
    tags = []
    if missing:
        if missing_lone == len(missing):
            tags.append(f"under:{len(missing)}miss(all-lone)")
        elif missing_lone:
            tags.append(f"under:{len(missing)}miss({missing_lone}lone)")
        else:
            tags.append(f"under:{len(missing)}miss(0lone)")
    if extra:
        tags.append(f"over:{len(extra)}extra")
    return " + ".join(tags)


def main() -> None:
    rows = []
    with open(TSV) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append(r)

    print(f"ViennaRNA {RNA.__version__}  |  RNA.fold defaults (== RNAfold -d2)\n")
    n = 0
    repro_v2v2 = 0
    repro_v2v1 = 0
    failures = []
    for r in rows:
        num = r["Puzzle #"].strip()
        if not num:
            continue
        target = r["Secondary Structure V2"].strip()
        sol_v2v2 = r["Sample Solution (V2/Vienna2)"].strip()
        sol_v2v1 = r["Sample Solution (V2/Vienna1)"].strip()
        if not target or set(target) <= {".", ""}:
            continue
        n += 1
        tbps = dotbracket_to_bps(target)

        ok_v2v2 = ok_v2v1 = False
        if sol_v2v2 and len(sol_v2v2) == len(target):
            ok_v2v2 = dotbracket_to_bps(fold_default(sol_v2v2)) == tbps
        if sol_v2v1 and len(sol_v2v1) == len(target):
            ok_v2v1 = dotbracket_to_bps(fold_default(sol_v2v1)) == tbps
        repro_v2v2 += ok_v2v2
        repro_v2v1 += ok_v2v1

        if not ok_v2v2:
            cls = "(no V2/Vienna2 solution)" if not sol_v2v2 else (
                f"len-mismatch" if len(sol_v2v2) != len(target)
                else classify(tbps, dotbracket_to_bps(fold_default(sol_v2v2)))
            )
            failures.append((num, r["Puzzle Name"].strip()[:34], len(target),
                             len(tbps), n_lone(tbps), cls, sol_v2v2, target, ok_v2v1))

    print(f"Puzzles checked: {n}")
    print(f"V2/Vienna2 reference reproduces V2 target under our RNA.fold: "
          f"{repro_v2v2}/{n}")
    print(f"V2/Vienna1 reference reproduces V2 target under our RNA.fold: "
          f"{repro_v2v1}/{n}")
    print(f"\n=== {len(failures)} puzzles whose V2/Vienna2 reference does NOT reproduce target ===")
    print(f"{'#':>4} {'name':35} {'L':>4} {'pairs':>5} {'lone':>4} {'V1ref?':>6}  failure")
    for num, name, L, npairs, nlone, cls, *_rest in failures:
        v1ok = _rest[2]
        print(f"{num:>4} {name:35} {L:>4} {npairs:>5} {nlone:>4} {('yes' if v1ok else 'no'):>6}  {cls}")

    # --- config-mismatch probe on the failures ---
    print("\n=== Vienna model-setting probe on the failing references ===")
    print("Does any alternate setting make the V2/Vienna2 reference reproduce the target?")
    print(f"{'#':>4}  default  noLP=1  dangles=0  dangles=1")
    recovered = {"noLP1": 0, "d0": 0, "d1": 0}
    for num, name, L, npairs, nlone, cls, sol, target, _v1 in failures:
        if not sol or len(sol) != len(target):
            continue
        tbps = dotbracket_to_bps(target)
        d_ok = dotbracket_to_bps(fold_default(sol)) == tbps
        nolp_ok = dotbracket_to_bps(fold_md(sol, dangles=2, noLP=1)) == tbps
        d0_ok = dotbracket_to_bps(fold_md(sol, dangles=0)) == tbps
        d1_ok = dotbracket_to_bps(fold_md(sol, dangles=1)) == tbps
        recovered["noLP1"] += nolp_ok
        recovered["d0"] += d0_ok
        recovered["d1"] += d1_ok
        print(f"{num:>4}  {str(d_ok):>7}  {str(nolp_ok):>6}  {str(d0_ok):>9}  {str(d1_ok):>9}")
    print(f"\nRecovered by setting (of {len(failures)} failures): "
          f"noLP=1 -> {recovered['noLP1']},  dangles=0 -> {recovered['d0']},  "
          f"dangles=1 -> {recovered['d1']}")


if __name__ == "__main__":
    main()
