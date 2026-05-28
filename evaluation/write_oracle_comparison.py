"""Build eterna100_oracle_comparison.md: solved + graded-Jaccard fold-back for
the 4 Eterna100 variants across 3 folding oracles (RibonanzaNet / Vienna /
EternaFold).

solved      = # puzzles with >=1 design folding exactly to the target (Jaccard==1).
best_jacc   = mean over 100 puzzles of the closest design's Jaccard (graded
              fold-back fidelity; reveals near-misses exact-match hides).
mean_jacc   = mean Jaccard over all samples.

RNet numbers come from the existing results/eterna100_eval/*/summary.csv (best
Jaccard recomputed from its samples.csv so all three oracles are comparable).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

VARIANTS = [
    ("bidir_random_k1000_v2", "S2S-bidir (random-perm, argmax)"),
    ("orig_l2r_argmax_k1000_v2", "Original S2S argmax"),
    ("orig_3strategies_k1000_v2", "Original S2S 3-strategies"),
    ("orig_3strategies_rescue_v2", "Original S2S 3-strategies + rescue"),
]
ORACLES = [
    ("rnet", "RibonanzaNet", ROOT / "results" / "eterna100_eval"),
    ("vienna", "ViennaRNA 2", ROOT / "results" / "eterna100_eval_vienna"),
    ("eternafold", "EternaFold", ROOT / "results" / "eterna100_eval_eternafold"),
]


def load(oracle_key, root, variant):
    sdir = root / variant
    summ_p, samp_p = sdir / "summary.csv", sdir / "samples.csv"
    if not summ_p.exists():
        return None
    df = pd.read_csv(summ_p)
    solved = int((df["n_perfect_jaccard"] > 0).sum())
    n_targets = len(df)
    mean_jacc = float(df["mean_jaccard"].mean())
    if "best_jaccard" in df.columns:
        best = float(df["best_jaccard"].mean())
    elif samp_p.exists():  # RNet summary lacks best_jaccard; derive from samples
        s = pd.read_csv(samp_p, usecols=["puzzle_idx", "jaccard_vs_target"])
        best = float(s.groupby("puzzle_idx")["jaccard_vs_target"].max().mean())
    else:
        best = float("nan")
    return {"solved": solved, "n": n_targets, "best": best, "mean": mean_jacc}


def fmt(x, n=3):
    return "NA" if x is None or (isinstance(x, float) and x != x) else f"{x:.{n}f}"


def main():
    data = {}  # (variant, oracle_key) -> stats
    for vkey, _ in VARIANTS:
        for okey, _, root in ORACLES:
            data[(vkey, okey)] = load(okey, root, vkey)

    L = []
    L.append("# Eterna100-V2: solved across folding oracles\n")
    L.append("Same 4 models, identical generated sequences (K=1000 budget); only the "
             "fold-back oracle differs. A puzzle is **solved** when >=1 design folds "
             "exactly to the target (base-pair Jaccard == 1.0) — the canonical "
             "Eterna100 criterion. ViennaRNA 2 is the reference engine the V2 targets "
             "were tuned for; all 100 V2 targets are pseudoknot-free.\n")

    # --- headline solved table ---
    L.append("## Solved (puzzles / 100)\n")
    L.append("| Model | RibonanzaNet | ViennaRNA 2 | EternaFold |")
    L.append("|---|---:|---:|---:|")
    for vkey, vlabel in VARIANTS:
        cells = []
        for okey, _, _ in ORACLES:
            d = data[(vkey, okey)]
            cells.append("NA" if d is None else f"{d['solved']}/{d['n']}")
        L.append(f"| {vlabel} | {cells[0]} | {cells[1]} | {cells[2]} |")
    L.append("")

    # --- graded fold-back table ---
    L.append("## Graded fold-back fidelity (mean best-per-puzzle Jaccard; "
             "mean-over-all-samples in parens)\n")
    L.append("| Model | RibonanzaNet | ViennaRNA 2 | EternaFold |")
    L.append("|---|---:|---:|---:|")
    for vkey, vlabel in VARIANTS:
        cells = []
        for okey, _, _ in ORACLES:
            d = data[(vkey, okey)]
            cells.append("NA" if d is None else f"{fmt(d['best'])} ({fmt(d['mean'])})")
        L.append(f"| {vlabel} | {cells[0]} | {cells[1]} | {cells[2]} |")
    L.append("")

    # --- per-puzzle solved breakdown (Vienna vs EternaFold vs RNet) ---
    L.append("## Per-puzzle solved (1 = >=1 perfect design)\n")
    # load all per-puzzle summaries
    persum = {}
    for vkey, _ in VARIANTS:
        for okey, _, root in ORACLES:
            p = root / vkey / "summary.csv"
            persum[(vkey, okey)] = pd.read_csv(p) if p.exists() else None
    targets = pd.read_csv(ROOT / "data" / "eterna100" / "eterna100_targets_v2.csv")
    header = ["#", "puzzle", "L"]
    for vkey, vlabel in VARIANTS:
        for okey, olabel, _ in ORACLES:
            header.append(f"{vlabel.split('(')[0].strip()[:10]}/{okey[:4]}")
    L.append("| " + " | ".join(header) + " |")
    L.append("|" + "---|" * len(header))
    for idx, t in targets.iterrows():
        pid = str(t["puzzleID"])
        rowcells = [str(int(t.get("Eterna100_number", idx + 1))), pid.replace("Eterna100_", "").replace("_V2", ""), str(int(t["Length"]))]
        for vkey, _ in VARIANTS:
            for okey, _, _ in ORACLES:
                df = persum[(vkey, okey)]
                if df is None or "puzzle_id" not in df.columns or pid not in set(df["puzzle_id"]):
                    rowcells.append("·")
                    continue
                r = df[df["puzzle_id"] == pid].iloc[0]
                rowcells.append("1" if r["n_perfect_jaccard"] > 0 else "0")
        L.append("| " + " | ".join(rowcells) + " |")
    L.append("")

    out = ROOT / "eterna100_oracle_comparison.md"
    out.write_text("\n".join(L))
    print(f"wrote {out}")
    # echo headline to stdout
    print("\n".join(L[: 4 + len(VARIANTS) + 6]))


if __name__ == "__main__":
    main()
