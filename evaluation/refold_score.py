"""Re-score a cached Eterna100 samples.csv under a different folding oracle.

The policy networks generate sequences WITHOUT any folding (the oracle is only
used at scoring time), so re-folding the cached `generated_sequence` column with
ViennaRNA / EternaFold is identical to re-running the whole generation pipeline
under that oracle -- same exact sequences, same K=1000 budget. This script does
exactly that for the three generation variants (bidir, orig-argmax, orig-3strat).
Rescue is handled separately by run_rescue_oracle.py.

A puzzle is "solved" (n_perfect_jaccard > 0) iff at least one design folds to a
structure whose base-pair set equals the target exactly (Jaccard == 1.0) -- the
canonical Eterna100 criterion. We additionally record best_jaccard (the closest
any design got) as a graded fold-back fidelity measure.

Output schema matches evaluation/run_ok7_eval.py's merge_and_summarize so the
existing report tooling can consume it; OK/eterna/cpq scores are RibonanzaNet-
SHAPE-specific and are left NaN here (only Jaccard/solved are oracle-comparable).

Usage:
    python evaluation/refold_score.py \\
        --in-samples results/eterna100_eval/bidir_random_k1000_v2/samples.csv \\
        --out-dir    results/eterna100_eval_vienna/bidir_random_k1000_v2 \\
        --targets-csv data/eterna100/eterna100_targets_v2.csv \\
        --oracle vienna --workers 16
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(THIS))
sys.path.insert(0, str(THIS / "evaluation"))
os.environ.setdefault("ARNIEFILE", str(THIS / "arnie_file.txt"))

from fold_oracles import dotbracket_to_bps, get_oracle, jaccard  # noqa: E402

SAMPLE_COLS = [
    "puzzle_idx", "puzzle_id", "title", "sample_idx",
    "generated_sequence", "predicted_structure",
    "jaccard_vs_target", "eterna_score", "cpq_score", "ok_score",
    "motif_kind", "motif_size_nt",
]

# multiprocessing worker state
_ORACLE_NAME = None
_ORACLE_FN = None


def _init_worker(oracle_name, arniefile):
    global _ORACLE_NAME, _ORACLE_FN
    os.environ["ARNIEFILE"] = arniefile
    _ORACLE_NAME = oracle_name
    _ORACLE_FN = get_oracle(oracle_name)


def _fold_one(seq):
    db, _bps = _ORACLE_FN(seq)
    return seq, db


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-samples", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--targets-csv", required=True)
    ap.add_argument("--oracle", required=True, choices=["vienna", "eternafold"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit-puzzles", type=int, default=None,
                    help="Smoke test: keep only the first N puzzle_idx values.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- target base-pair sets, keyed by puzzle_id (== puzzleID) ----------
    tdf = pd.read_csv(args.targets_csv)
    target_bps = {
        str(r["puzzleID"]): {
            (min(i, j), max(i, j)) for i, j in dotbracket_to_bps(str(r["Dot-bracket"]))
        }
        for _, r in tdf.iterrows()
    }

    # --- load cached samples ---------------------------------------------
    df = pd.read_csv(args.in_samples)
    if args.limit_puzzles is not None:
        keep = sorted(df["puzzle_idx"].unique())[: args.limit_puzzles]
        df = df[df["puzzle_idx"].isin(keep)].reset_index(drop=True)
    n_in = len(df)
    df["generated_sequence"] = df["generated_sequence"].astype(str)

    # --- fold every UNIQUE sequence once (CPU-bound, parallel) -----------
    uniq = sorted(df["generated_sequence"].unique())
    print(f"[{args.oracle}] {n_in} rows, {len(uniq)} unique sequences "
          f"-> folding with {args.workers} workers", flush=True)

    db_by_seq = {}
    if args.workers > 1:
        import multiprocessing as mp
        with mp.Pool(args.workers, initializer=_init_worker,
                     initargs=(args.oracle, os.environ["ARNIEFILE"])) as pool:
            for i, (seq, db) in enumerate(
                pool.imap_unordered(_fold_one, uniq, chunksize=64), 1
            ):
                db_by_seq[seq] = db
                if i % 5000 == 0:
                    print(f"  folded {i}/{len(uniq)}", flush=True)
    else:
        _init_worker(args.oracle, os.environ["ARNIEFILE"])
        for seq in uniq:
            db_by_seq[seq] = _fold_one(seq)[1]

    bps_by_seq = {s: dotbracket_to_bps(db) for s, db in db_by_seq.items()}

    # --- re-score every row ----------------------------------------------
    pred_db, jacc = [], []
    for _, r in df.iterrows():
        seq = r["generated_sequence"]
        db = db_by_seq[seq]
        pred_db.append(db)
        jacc.append(jaccard(bps_by_seq[seq], target_bps.get(str(r["puzzle_id"]), set())))

    df["predicted_structure"] = pred_db
    df["jaccard_vs_target"] = jacc
    for c in ("eterna_score", "cpq_score", "ok_score"):
        df[c] = np.nan

    keep_cols = [c for c in SAMPLE_COLS if c in df.columns]
    if "sampling_strategy" in df.columns:
        keep_cols = keep_cols + ["sampling_strategy"]
    df = df[keep_cols]
    assert len(df) == n_in, f"row count changed: {len(df)} != {n_in}"
    df.to_csv(out_dir / "samples.csv", index=False)

    # --- per-puzzle summary ----------------------------------------------
    tag = f"{Path(args.in_samples).parent.name}__{args.oracle}"
    rows = []
    for pi, sub in df.groupby("puzzle_idx"):
        rows.append({
            "config_tag": tag,
            "puzzle_idx": int(pi),
            "puzzle_id": str(sub["puzzle_id"].iloc[0]),
            "title": str(sub["title"].iloc[0]),
            "n_samples": int(len(sub)),
            "n_perfect_jaccard": int((sub["jaccard_vs_target"] == 1.0).sum()),
            "p80_ok_score": float("nan"),
            "mean_jaccard": float(sub["jaccard_vs_target"].mean()),
            "mean_ok_score": float("nan"),
            "best_jaccard": float(sub["jaccard_vs_target"].max()),
        })
    summ = pd.DataFrame(rows).sort_values("puzzle_idx").reset_index(drop=True)
    summ.to_csv(out_dir / "summary.csv", index=False)

    solved = int((summ["n_perfect_jaccard"] > 0).sum())
    print(f"[{args.oracle}] {tag}: solved {solved}/{len(summ)}  "
          f"mean best-Jaccard {summ['best_jaccard'].mean():.3f}  "
          f"-> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
