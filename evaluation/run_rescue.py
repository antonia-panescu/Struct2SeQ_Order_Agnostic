"""Faithful Shujun rescue strategy from test_240.py:177-225.

For each puzzle's best (highest-jaccard, non-perfect) sample:
  - Compute diff_pos = positions where target's pair-partner vector
    differs from predicted's pair-partner vector
  - If len(diff_pos) <= 4, enumerate all 4^k nucleotide combinations at
    those positions (max 256 candidates)
  - Score each via RibonanzaNet-SS + OpenKnotScorePipeline
  - Append all rescue candidates to a `samples.csv` augmenting the
    input run.

Run from struct2seq_bidir_rl/. Assumes Round4 puzzles + arnie_file.txt.

Usage:
    python evaluation/run_rescue.py \\
        --in-samples results/ok7b_eval/orig_struct2seq_3strategies_faithful_240mer/samples.csv \\
        --out-dir    results/ok7b_eval/orig_struct2seq_3strategies_rescue_240mer \\
        --targets-csv /home/nvidia/haiwen/antonia/OpenKnotAIDesignData/Targets/Round4_targets.csv
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

THIS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(THIS))

OKS_SRC = Path("/home/nvidia/haiwen/antonia/OpenKnotScorePipeline/src")
if str(OKS_SRC) not in sys.path:
    sys.path.insert(0, str(OKS_SRC))

os.environ.setdefault("ARNIEFILE", str(THIS / "arnie_file.txt"))

from Env import DQN_env, mask_diagonal  # noqa: E402
from Functions import convert_dotbracket_to_bp_list  # noqa: E402
from arnie.pk_predictors import _hungarian  # noqa: E402
from openknotscore.pipeline.scoring import (  # noqa: E402
    calculateCrossedPairQualityScore,
    calculateEternaClassicScore,
)

# Use Shujun's exact rescue helpers from his Struct2SeQ repo. Don't reinvent.
sys.path.insert(0, "/home/nvidia/haiwen/antonia/Struct2SeQ")
from search_v2 import get_mutated as _shujun_get_mutated  # noqa: E402
from search_v2 import edit_sequence as _shujun_edit_sequence  # noqa: E402

NT_TO_IDX = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}
IDX_TO_NT = {0: "A", 1: "C", 2: "G", 3: "U"}


# cor2vec imported directly from Shujun's Struct2SeQ/Functions.py.
# His signature is cor2vec(corr_dict, structure_str) -> ndarray. We adapt
# it: build a corr dict from the bp list, pass a length-L placeholder str.
sys.path.insert(0, "/home/nvidia/haiwen/antonia/Struct2SeQ")
from Functions import cor2vec as _shujun_cor2vec  # noqa: E402


def cor2vec_from_bps(bp_list, seq_len):
    """Wrapper that builds Shujun's corr-dict format and calls his cor2vec."""
    corr = {}
    for i, j in bp_list:
        corr[i] = j
        corr[j] = i
    return _shujun_cor2vec(corr, "x" * seq_len)


# get_mutated and edit_sequence imported directly from Shujun's
# /home/nvidia/haiwen/antonia/Struct2SeQ/search_v2.py — no reimplementation.


def jaccard_bp(predicted_bps, target_bps_set):
    pred_set = {(min(i, j), max(i, j)) for i, j in predicted_bps}
    if not pred_set and not target_bps_set:
        return 1.0
    inter = pred_set & target_bps_set
    union = pred_set | target_bps_set
    return len(inter) / len(union) if union else 1.0


def compute_ok_score(predicted_db, shape_list, L):
    e = calculateEternaClassicScore(predicted_db, shape_list, 0, L - 1, filter_singlets=True)
    c = calculateCrossedPairQualityScore(predicted_db, shape_list, 0, L - 1, filter_singlets=True)
    e = float("nan") if e is None else float(e)
    cq = float("nan") if (not isinstance(c, list) or len(c) < 2) else float(c[1])
    ok = float("nan") if (np.isnan(e) or np.isnan(cq)) else 0.5 * (e + cq)
    return e, cq, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-samples", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--targets-csv", required=True)
    ap.add_argument("--max-diff-positions", type=int, default=4,
                    help="Match Shujun: skip rescue if diff_pos count exceeds this.")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device)
    in_df = pd.read_csv(args.in_samples)
    targets = pd.read_csv(args.targets_csv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "samples.csv"

    # Build target-bp sets per puzzle for jaccard computation
    target_bps_sets = {}
    target_dbs = {}
    for _, row in targets.iterrows():
        pid = str(row["puzzleID"])
        db = row["Dot-bracket"]
        target_dbs[pid] = db
        bps = convert_dotbracket_to_bp_list(db, allow_pseudoknots=True)
        target_bps_sets[pid] = {(min(i, j), max(i, j)) for i, j in bps}

    env = DQN_env(use_gpu=True, compile=False)

    # Pass through the existing samples first, then append rescue rows.
    rescue_rows = []
    rescue_summary = []
    for puzzle_idx, sub in in_df.groupby("puzzle_idx"):
        # Skip puzzles already solved (max jaccard == 1)
        if (sub["jaccard_vs_target"] == 1.0).any():
            rescue_summary.append({
                "puzzle_idx": int(puzzle_idx),
                "rescue_status": "skipped_already_solved",
                "n_rescue_candidates": 0,
                "rescue_jaccard_max": float("nan"),
            })
            continue

        # Pick best non-perfect sequence as the rescue seed
        sub = sub.sort_values("jaccard_vs_target", ascending=False).reset_index(drop=True)
        seed = sub.iloc[0]
        seed_seq = seed["generated_sequence"]
        seed_pred_db = seed["predicted_structure"]
        L = len(seed_seq)
        pid = str(seed["puzzle_id"])
        target_db = target_dbs[pid]

        # diff_pos: positions where target pair-partner vec != predicted pair-partner vec
        target_bps = convert_dotbracket_to_bp_list(target_db, allow_pseudoknots=True)
        pred_bps = convert_dotbracket_to_bp_list(seed_pred_db, allow_pseudoknots=True)
        target_vec = cor2vec_from_bps(target_bps, L)
        pred_vec = cor2vec_from_bps(pred_bps, L)
        diff_pos = np.where(target_vec != pred_vec)[0].tolist()

        if len(diff_pos) > args.max_diff_positions:
            rescue_summary.append({
                "puzzle_idx": int(puzzle_idx),
                "rescue_status": f"too_many_diff_pos_{len(diff_pos)}",
                "n_rescue_candidates": 0,
                "rescue_jaccard_max": float("nan"),
            })
            continue

        # Generate all 4^k mutations using Shujun's exact get_mutated
        positions = list(diff_pos)  # mutated by get_mutated (.pop)
        rescue_seqs = _shujun_get_mutated(seed_seq, positions, [])
        # Dedup and exclude the seed
        rescue_seqs = sorted(set(rescue_seqs) - {seed_seq})
        if not rescue_seqs:
            rescue_summary.append({
                "puzzle_idx": int(puzzle_idx),
                "rescue_status": "no_new_candidates",
                "n_rescue_candidates": 0,
                "rescue_jaccard_max": float("nan"),
            })
            continue

        # Tokenize and score in mini-batches to fit memory at L=240
        seqs_tensor = torch.tensor(
            [[NT_TO_IDX[nt] for nt in s] for s in rescue_seqs],
            dtype=torch.long, device=device,
        )
        BS = 16
        bpps_list, shape_list = [], []
        with torch.no_grad():
            for i0 in range(0, seqs_tensor.shape[0], BS):
                chunk = seqs_tensor[i0:i0 + BS]
                bpps_list.append(env.SS_model(chunk).sigmoid().detach().cpu().numpy())
                shape_list.append(env.reactivity_model(chunk)[:, :, 0].detach().cpu().float().numpy())
        bpps = np.concatenate(bpps_list, axis=0)
        shape_arr = np.concatenate(shape_list, axis=0)

        max_j = -1.0
        for k, seq_str in enumerate(rescue_seqs):
            bpp = bpps[k, :L, :L]
            pred_db, pred_bp_list = _hungarian(mask_diagonal(bpp), theta=0.5, min_len_helix=1)
            jacc = jaccard_bp(pred_bp_list, target_bps_sets[pid])
            shape_b = shape_arr[k, :L].tolist()
            e, cq, ok = compute_ok_score(pred_db, shape_b, L)
            rescue_rows.append({
                "puzzle_idx": int(puzzle_idx),
                "puzzle_id": pid,
                "title": seed["title"],
                "sample_idx": int(1_000_000 + k),  # offset to mark as rescue
                "generated_sequence": seq_str,
                "predicted_structure": pred_db,
                "jaccard_vs_target": jacc,
                "eterna_score": e, "cpq_score": cq, "ok_score": ok,
                "motif_kind": "rescue", "motif_size_nt": len(diff_pos),
                "sampling_strategy": "rescue",
            })
            if jacc > max_j:
                max_j = jacc
        rescue_summary.append({
            "puzzle_idx": int(puzzle_idx),
            "rescue_status": "ran",
            "n_rescue_candidates": len(rescue_seqs),
            "rescue_jaccard_max": float(max_j),
        })
        print(f"  puzzle {puzzle_idx} ({pid}): {len(diff_pos)} diff_pos → {len(rescue_seqs)} candidates, max_j={max_j:.3f}")

    # Write rescue-augmented samples (input + rescue rows)
    rescue_df = pd.DataFrame(rescue_rows)
    if "sampling_strategy" not in in_df.columns:
        in_df["sampling_strategy"] = "original"
    out_df = pd.concat([in_df, rescue_df], ignore_index=True)
    out_df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}: {len(out_df)} total rows ({len(rescue_df)} rescue rows added)")

    # Per-puzzle summary
    rows = []
    for pi, sub in out_df.groupby("puzzle_idx"):
        rows.append({
            "config_tag": "orig_struct2seq_3strategies_rescue_240mer",
            "puzzle_idx": int(pi),
            "puzzle_id": str(sub["puzzle_id"].iloc[0]),
            "title": str(sub["title"].iloc[0]),
            "n_samples": len(sub),
            "n_perfect_jaccard": int((sub["jaccard_vs_target"] == 1.0).sum()),
            "p80_ok_score": float(sub["ok_score"].dropna().quantile(0.80)) if sub["ok_score"].dropna().size else float("nan"),
            "mean_jaccard": float(sub["jaccard_vs_target"].mean()),
            "mean_ok_score": float(sub["ok_score"].dropna().mean()) if sub["ok_score"].dropna().size else float("nan"),
        })
    pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(rescue_summary).to_csv(out_dir / "rescue_log.csv", index=False)
    n_perf = (out_df["jaccard_vs_target"] == 1.0).sum()
    solved = sum(1 for r in rows if r["n_perfect_jaccard"] > 0)
    print(f"\n=== AR + 3 strategies + rescue ===")
    print(f"  Total: {len(out_df)} samples, perfect%={100*n_perf/len(out_df):.1f}, puzzles solved={solved}/{len(rows)}")


if __name__ == "__main__":
    main()
