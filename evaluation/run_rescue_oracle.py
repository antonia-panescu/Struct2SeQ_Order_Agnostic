"""Oracle-parameterized version of run_rescue.py (ViennaRNA / EternaFold).

Same faithful Shujun 4^k rescue as evaluation/run_rescue.py, but the folding
oracle used to (a) derive the seed's predicted structure / diff positions and
(b) score the rescue candidates is ViennaRNA or EternaFold instead of
RibonanzaNet-SS. This keeps the rescue fully oracle-consistent: the 4^k repair
fixes positions THAT oracle disagrees on, and candidates are kept/rejected by
THAT oracle's fold-back.

Input should be the SAME oracle's re-scored 3-strategies samples.csv (produced
by refold_score.py), so the "already solved" skip and seed selection use the
oracle's own Jaccard. Output augments it with rescue rows, in the schema the
comparison report expects (incl. best_jaccard).

Usage:
    python evaluation/run_rescue_oracle.py \\
        --in-samples results/eterna100_eval_vienna/orig_3strategies_k1000_v2/samples.csv \\
        --out-dir    results/eterna100_eval_vienna/orig_3strategies_rescue_v2 \\
        --targets-csv data/eterna100/eterna100_targets_v2.csv \\
        --oracle vienna --workers 16
"""
from __future__ import annotations

import argparse
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

# Shujun's exact rescue helpers (do not reinvent). These pull his repo; only the
# MAIN process needs them -- the fold workers below import only fold_oracles.
sys.path.insert(0, "/home/nvidia/haiwen/antonia/Struct2SeQ")
from search_v2 import get_mutated as _shujun_get_mutated  # noqa: E402
from Functions import cor2vec as _shujun_cor2vec  # noqa: E402


def cor2vec_from_bps(bp_list, seq_len):
    corr = {}
    for i, j in bp_list:
        corr[i] = j
        corr[j] = i
    return _shujun_cor2vec(corr, "x" * seq_len)


def _norm(bps):
    return {(min(i, j), max(i, j)) for i, j in bps}


# --- multiprocessing fold worker (import-light) --------------------------
_ORACLE_FN = None


def _init_worker(oracle_name, arniefile):
    global _ORACLE_FN
    os.environ["ARNIEFILE"] = arniefile
    _ORACLE_FN = get_oracle(oracle_name)


def _fold_one(seq):
    db, _ = _ORACLE_FN(seq)
    return seq, db


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-samples", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--targets-csv", required=True)
    ap.add_argument("--oracle", required=True, choices=["vienna", "eternafold"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-diff-positions", type=int, default=4,
                    help="Match Shujun: skip rescue if diff_pos count exceeds this.")
    args = ap.parse_args()

    oracle = get_oracle(args.oracle)
    in_df = pd.read_csv(args.in_samples)
    in_df["generated_sequence"] = in_df["generated_sequence"].astype(str)
    targets = pd.read_csv(args.targets_csv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_bps_sets, target_dbs = {}, {}
    for _, row in targets.iterrows():
        pid = str(row["puzzleID"])
        target_dbs[pid] = str(row["Dot-bracket"])
        target_bps_sets[pid] = _norm(dotbracket_to_bps(target_dbs[pid]))

    # --- Phase 1: per puzzle, find diff positions and enumerate 4^k -------
    plan = []           # list of dicts describing each puzzle's rescue
    candidate_seqs = set()
    rescue_summary = []
    for puzzle_idx, sub in in_df.groupby("puzzle_idx"):
        if (sub["jaccard_vs_target"] == 1.0).any():
            rescue_summary.append({"puzzle_idx": int(puzzle_idx),
                                   "rescue_status": "skipped_already_solved",
                                   "n_rescue_candidates": 0,
                                   "rescue_jaccard_max": float("nan")})
            continue

        sub = sub.sort_values("jaccard_vs_target", ascending=False).reset_index(drop=True)
        seed = sub.iloc[0]
        seed_seq = seed["generated_sequence"]
        L = len(seed_seq)
        pid = str(seed["puzzle_id"])

        # Oracle-consistent seed prediction (re-fold with the chosen oracle).
        seed_pred_db, _ = oracle(seed_seq)
        target_vec = cor2vec_from_bps(dotbracket_to_bps(target_dbs[pid]), L)
        pred_vec = cor2vec_from_bps(dotbracket_to_bps(seed_pred_db), L)
        diff_pos = np.where(target_vec != pred_vec)[0].tolist()

        if len(diff_pos) > args.max_diff_positions:
            rescue_summary.append({"puzzle_idx": int(puzzle_idx),
                                   "rescue_status": f"too_many_diff_pos_{len(diff_pos)}",
                                   "n_rescue_candidates": 0,
                                   "rescue_jaccard_max": float("nan")})
            continue

        rescue_seqs = sorted(set(_shujun_get_mutated(seed_seq, list(diff_pos), [])) - {seed_seq})
        if not rescue_seqs:
            rescue_summary.append({"puzzle_idx": int(puzzle_idx),
                                   "rescue_status": "no_new_candidates",
                                   "n_rescue_candidates": 0,
                                   "rescue_jaccard_max": float("nan")})
            continue

        candidate_seqs.update(rescue_seqs)
        plan.append({"puzzle_idx": int(puzzle_idx), "pid": pid,
                     "title": seed["title"], "L": L,
                     "diff": len(diff_pos), "seqs": rescue_seqs})

    # --- Phase 2: fold all unique candidates with the oracle (parallel) ---
    uniq = sorted(candidate_seqs)
    print(f"[{args.oracle}] rescue: {len(plan)} puzzles to repair, "
          f"{len(uniq)} unique candidates -> folding", flush=True)
    db_by_seq = {}
    if uniq and args.workers > 1:
        import multiprocessing as mp
        with mp.Pool(args.workers, initializer=_init_worker,
                     initargs=(args.oracle, os.environ["ARNIEFILE"])) as pool:
            for seq, db in pool.imap_unordered(_fold_one, uniq, chunksize=32):
                db_by_seq[seq] = db
    else:
        for seq in uniq:
            db_by_seq[seq], = (oracle(seq)[0],)
    bps_by_seq = {s: dotbracket_to_bps(db) for s, db in db_by_seq.items()}

    # --- Phase 3: assemble rescue rows ------------------------------------
    rescue_rows = []
    for p in plan:
        pid, tgt = p["pid"], target_bps_sets[p["pid"]]
        max_j = -1.0
        for k, seq_str in enumerate(p["seqs"]):
            jacc = jaccard(_norm(bps_by_seq[seq_str]), tgt)
            rescue_rows.append({
                "puzzle_idx": p["puzzle_idx"], "puzzle_id": pid, "title": p["title"],
                "sample_idx": int(1_000_000 + k),
                "generated_sequence": seq_str,
                "predicted_structure": db_by_seq[seq_str],
                "jaccard_vs_target": jacc,
                "eterna_score": np.nan, "cpq_score": np.nan, "ok_score": np.nan,
                "motif_kind": "rescue", "motif_size_nt": p["diff"],
                "sampling_strategy": "rescue",
            })
            max_j = max(max_j, jacc)
        rescue_summary.append({"puzzle_idx": p["puzzle_idx"], "rescue_status": "ran",
                               "n_rescue_candidates": len(p["seqs"]),
                               "rescue_jaccard_max": float(max_j)})
        print(f"  puzzle {p['puzzle_idx']} ({pid}): {p['diff']} diff_pos -> "
              f"{len(p['seqs'])} candidates, max_j={max_j:.3f}", flush=True)

    # --- write augmented samples + summary --------------------------------
    if "sampling_strategy" not in in_df.columns:
        in_df["sampling_strategy"] = "original"
    out_df = pd.concat([in_df, pd.DataFrame(rescue_rows)], ignore_index=True)
    out_df.to_csv(out_dir / "samples.csv", index=False)

    tag = f"orig_3strategies_rescue__{args.oracle}"
    rows = []
    for pi, sub in out_df.groupby("puzzle_idx"):
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
    pd.DataFrame(rescue_summary).to_csv(out_dir / "rescue_log.csv", index=False)

    solved = int((summ["n_perfect_jaccard"] > 0).sum())
    print(f"\n[{args.oracle}] rescue: {len(out_df)} rows "
          f"({len(rescue_rows)} rescue added), solved {solved}/{len(summ)}  "
          f"mean best-Jaccard {summ['best_jaccard'].mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
