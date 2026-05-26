#!/usr/bin/env python3
"""Convert eterna100-benchmarking TSV to Struct2SeQ/OpenKnot-style target CSV.

Input columns from https://github.com/eternagame/eterna100-benchmarking:
  Puzzle #, Puzzle Name, Secondary Structure V1, Secondary Structure V2, ...

For this project's evaluation scripts we need:
  puzzleID, Title, Dot-bracket, Sequence, Length, wild_type_sequence

Default target version is V2 (latest puzzle structure in the TSV); use --version V1
for the historical V1 structures.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

VALID_NT = set("ACGUT")
BAD_SOLUTION_MARKERS = {"", "(Unsolved)", "(Undisclosed)", "Unsolved", "Undisclosed"}


def clean_seq(s: str, L: int) -> str:
    s = (s or "").strip().upper().replace("T", "U")
    if s in BAD_SOLUTION_MARKERS or any(ch not in VALID_NT for ch in s) or len(s) != L:
        # Only used by inpainting / original WT-bias paths; full-design eval does not
        # condition on it. Use neutral all-A when Eterna withholds or lacks a solution.
        return "A" * L
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--version", choices=["V1", "V2"], default="V2")
    args = ap.parse_args()

    structure_col = f"Secondary Structure {args.version}"
    solution_prefs = [
        f"Sample Solution ({args.version}/Vienna2)",
        f"Sample Solution ({args.version}/Vienna1)",
        "Sample Solution (V2/Vienna2)",
        "Sample Solution (V2/Vienna1)",
        "Sample Solution (V1/Vienna2)",
        "Sample Solution (V1/Vienna1)",
    ]

    rows = []
    with open(args.input, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            db = r[structure_col].strip()
            L = len(db)
            seq = None
            for col in solution_prefs:
                cand = clean_seq(r.get(col, ""), L)
                if cand != "A" * L:
                    seq = cand
                    break
            if seq is None:
                seq = "A" * L
            pid = f"Eterna100_{int(r['Puzzle #']):03d}_{args.version}"
            title = f"{int(r['Puzzle #']):03d}: {r['Puzzle Name']}"
            rows.append({
                "puzzleID": pid,
                "Title": title,
                "Dot-bracket": db,
                "Sequence": seq,
                "Length": L,
                "wild_type_sequence": seq,
                "Eterna100_number": int(r["Puzzle #"]),
                "Eterna100_version": args.version,
            })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} rows={len(rows)} version={args.version}")
    print(f"length_min={min(r['Length'] for r in rows)} length_max={max(r['Length'] for r in rows)}")

if __name__ == "__main__":
    main()
