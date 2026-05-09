"""Bidirectional rescue strategy — analog of Shujun's `test_240.py:177-225`
rescue but using our order-agnostic model's in-painting capability instead
of 4^k local enumeration.

For each puzzle's best non-perfect sequence in an existing samples.csv:
  - Compute diff_pos (positions where target's pair-partner vec differs
    from predicted's pair-partner vec) — same as Shujun's rescue.
  - Fix all positions OUTSIDE diff_pos to that seed's nucleotides
    (the sequence that produced the highest-jaccard non-perfect).
  - Regenerate just the diff_pos using our bidir model with random-perm
    decoding, K samples per puzzle.
  - Score each candidate; record any that improve over the seed.

Run from struct2seq_bidir_rl/. Requires the same Env + checkpoint as
run_ok7_eval.py.
"""
from __future__ import annotations
import argparse
import os
import sys
import yaml
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
from Functions import generate_permuted, convert_dotbracket_to_bp_list  # noqa: E402
from Dataset import tokenize_dot_bracket  # noqa: E402
from arnie.pk_predictors import _hungarian  # noqa: E402
from openknotscore.pipeline.scoring import (  # noqa: E402
    calculateCrossedPairQualityScore,
    calculateEternaClassicScore,
)
from evaluation.run_eval import build_model  # noqa: E402
from run import TrainingConfig  # noqa: E402

DROP_OLD_DECODER_PE = {
    "decoder.positional_encoding.weight_ih_l0",
    "decoder.positional_encoding.weight_hh_l0",
    "decoder.positional_encoding.bias_ih_l0",
    "decoder.positional_encoding.bias_hh_l0",
    "decoder.conv.0.conv.weight",
    "decoder.conv.0.conv.bias",
    "decoder.conv_norm.weight",
    "decoder.conv_norm.bias",
}

NT_TO_IDX = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}
IDX_TO_NT = {0: "A", 1: "C", 2: "G", 3: "U"}


def cor2vec_from_bps(bp_list, seq_len):
    vec = np.full(seq_len, -1, dtype=np.int64)
    for i, j in bp_list:
        vec[i] = j
        vec[j] = i
    return vec


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


def build_bidir_model(config_path, checkpoint, device):
    """Same model-loading pattern as run_ok7_eval.py:502-540."""
    config = TrainingConfig.from_yaml(config_path)
    model = build_model(config, device)
    sd = torch.load(checkpoint, map_location="cpu")
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if k not in DROP_OLD_DECODER_PE}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  missing keys ({len(missing)}): {missing[:3]}{'...' if len(missing)>3 else ''}")
    if unexpected:
        print(f"  unexpected keys ({len(unexpected)}): {unexpected[:3]}{'...' if len(unexpected)>3 else ''}")
    model.eval()
    return model.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-samples", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--targets-csv", required=True)
    ap.add_argument("--checkpoint", default=str(THIS / "best_policy_network.pt"))
    ap.add_argument("--config", default=str(THIS / "default_config.yaml"))
    ap.add_argument("--max-diff-positions", type=int, default=20,
                    help="Skip puzzles where diff_pos exceeds this (more lenient than "
                         "Shujun's 4 since our model handles arbitrary in-painting fractions).")
    ap.add_argument("--k-samples", type=int, default=100,
                    help="Number of random-perm samples per seed.")
    ap.add_argument("--num-seeds", type=int, default=1,
                    help="Number of distinct non-perfect seeds to rescue from (top-N by "
                         "jaccard). Different seeds give different fixed scaffolds.")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device)
    in_df = pd.read_csv(args.in_samples)
    targets = pd.read_csv(args.targets_csv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "samples.csv"

    target_dbs = {}
    target_bps_sets = {}
    target_correspondences = {}
    for _, row in targets.iterrows():
        pid = str(row["puzzleID"])
        db = row["Dot-bracket"]
        target_dbs[pid] = db
        bps = convert_dotbracket_to_bp_list(db, allow_pseudoknots=True)
        target_bps_sets[pid] = {(min(i, j), max(i, j)) for i, j in bps}
        tc = {}
        for i, j in bps:
            tc[i] = j
            tc[j] = i
        target_correspondences[pid] = tc

    env = DQN_env(use_gpu=True, compile=False)
    model = build_bidir_model(args.config, args.checkpoint, device)
    rng = np.random.default_rng(0)

    rescue_rows = []
    rescue_log = []
    for puzzle_idx, sub in in_df.groupby("puzzle_idx"):
        if (sub["jaccard_vs_target"] == 1.0).any():
            rescue_log.append({"puzzle_idx": int(puzzle_idx), "status": "already_solved",
                              "n_candidates": 0, "max_jaccard": float("nan")})
            continue
        # Take top-N distinct (by sequence) non-perfect seeds; all must be
        # below jaccard=1.0 (already filtered by the if above).
        sub_sorted = sub.sort_values("jaccard_vs_target", ascending=False)
        # Dedup by generated_sequence to get DIFFERENT fixed scaffolds
        sub_unique = sub_sorted.drop_duplicates("generated_sequence")
        seeds = sub_unique.head(args.num_seeds).reset_index(drop=True)
        if len(seeds) == 0:
            rescue_log.append({"puzzle_idx": int(puzzle_idx), "status": "no_seeds",
                              "n_candidates": 0, "max_jaccard": float("nan")})
            continue
        # Each seed gets its own rescue attempt; track best across seeds
        per_seed_results = []
        for seed_rank, seed_row in seeds.iterrows():
            seed_seq = seed_row["generated_sequence"]
            seed_pred_db = seed_row["predicted_structure"]
            L = len(seed_seq)
            pid = str(seed_row["puzzle_id"])
            target_db = target_dbs[pid]
            target_bps_loop = convert_dotbracket_to_bp_list(target_db, allow_pseudoknots=True)
            pred_bps_loop = convert_dotbracket_to_bp_list(seed_pred_db, allow_pseudoknots=True)
            target_vec = cor2vec_from_bps(target_bps_loop, L)
            pred_vec = cor2vec_from_bps(pred_bps_loop, L)
            diff_pos_loop = np.where(target_vec != pred_vec)[0].tolist()
            if len(diff_pos_loop) > args.max_diff_positions or len(diff_pos_loop) == 0:
                per_seed_results.append({"seed_rank": int(seed_rank), "diff_pos": len(diff_pos_loop),
                                         "skipped": True, "max_j": float("nan")})
                continue
            per_seed_results.append({"seed_rank": int(seed_rank), "seed_seq": seed_seq,
                                     "seed_pred_db": seed_pred_db, "L": L, "pid": pid,
                                     "target_db": target_db, "target_bps": target_bps_loop,
                                     "diff_pos": diff_pos_loop, "skipped": False})

        # If all seeds skipped, log and continue
        live_seeds = [s for s in per_seed_results if not s["skipped"]]
        if not live_seeds:
            rescue_log.append({"puzzle_idx": int(puzzle_idx),
                              "status": f"all_seeds_too_many_diff_pos",
                              "n_candidates": 0, "max_jaccard": float("nan")})
            continue

        # Run rescue per-seed (each seed is a different fixed-scaffold context)
        K = args.k_samples
        max_j_overall = -1.0
        diff_pos_summary = []
        for seed_info in live_seeds:
            L = seed_info["L"]
            pid = seed_info["pid"]
            target_db = seed_info["target_db"]
            target_bps = seed_info["target_bps"]
            seed_seq = seed_info["seed_seq"]
            diff_pos = seed_info["diff_pos"]
            seed_rank = seed_info["seed_rank"]
            diff_pos_summary.append(len(diff_pos))

            src1 = torch.tensor(tokenize_dot_bracket(target_db), dtype=torch.long, device=device)
            src = src1.unsqueeze(0).expand(K, L).contiguous()
            ct1 = torch.zeros((L, L), dtype=torch.float, device=device)
            ct1.fill_diagonal_(1.0)
            for i, j in target_bps:
                ct1[i, j] = 1.0
                ct1[j, i] = 1.0
            ct = ct1.unsqueeze(0).expand(K, L, L).contiguous()
            tcs = [target_correspondences[pid] for _ in range(K)]
            fixed = torch.full((K, L), -1, dtype=torch.long, device=device)
            diff_set = set(int(p) for p in diff_pos)
            for i in range(L):
                if i not in diff_set:
                    fixed[:, i] = NT_TO_IDX[seed_seq[i]]
            perms = [torch.from_numpy(rng.permutation(L)).long() for _ in range(K)]
            perm = torch.stack(perms).to(device)

            with torch.no_grad():
                seqs = generate_permuted(
                    model, src, ct, tcs, perm=perm,
                    mode="epsilon_argmax", p=0.0, fixed=fixed,
                )
                BS = 16
                bpps_list, shape_list = [], []
                for i0 in range(0, K, BS):
                    chunk = seqs[i0:i0 + BS]
                    bpps_list.append(env.SS_model(chunk).sigmoid().detach().cpu().numpy())
                    shape_list.append(env.reactivity_model(chunk)[:, :, 0].detach().cpu().float().numpy())
                bpps = np.concatenate(bpps_list, axis=0)
                shape_arr = np.concatenate(shape_list, axis=0)

            seed_max_j = -1.0
            for k in range(K):
                seq_idx = seqs[k, :L].cpu().tolist()
                seq_str = "".join(IDX_TO_NT.get(int(t), "N") for t in seq_idx)
                bpp = bpps[k, :L, :L]
                pred_db, pred_bp_list = _hungarian(mask_diagonal(bpp), theta=0.5, min_len_helix=1)
                jacc = jaccard_bp(pred_bp_list, target_bps_sets[pid])
                shape_b = shape_arr[k, :L].tolist()
                e, cq, ok = compute_ok_score(pred_db, shape_b, L)
                rescue_rows.append({
                    "puzzle_idx": int(puzzle_idx),
                    "puzzle_id": pid,
                    "title": seeds.iloc[seed_rank]["title"],
                    "sample_idx": int(2_000_000 + seed_rank * K + k),
                    "generated_sequence": seq_str,
                    "predicted_structure": pred_db,
                    "jaccard_vs_target": jacc,
                    "eterna_score": e, "cpq_score": cq, "ok_score": ok,
                    "motif_kind": "bidir_rescue", "motif_size_nt": len(diff_pos),
                    "sampling_strategy": f"bidir_rescue_seed{seed_rank}",
                })
                if jacc > seed_max_j:
                    seed_max_j = jacc
            if seed_max_j > max_j_overall:
                max_j_overall = seed_max_j
            print(f"    seed_rank={seed_rank} diff_pos={len(diff_pos)} K={K} max_j={seed_max_j:.3f}")
        rescue_log.append({
            "puzzle_idx": int(puzzle_idx), "status": "ran",
            "n_candidates": K * len(live_seeds), "max_jaccard": float(max_j_overall),
            "n_seeds": len(live_seeds),
            "diff_pos_summary": str(diff_pos_summary),
        })
        print(f"  puzzle {puzzle_idx} ({pid}): {len(live_seeds)} seeds, total {K*len(live_seeds)} candidates, max_j_overall={max_j_overall:.3f}")

    rescue_df = pd.DataFrame(rescue_rows)
    if "sampling_strategy" not in in_df.columns:
        in_df["sampling_strategy"] = "original"
    out_df = pd.concat([in_df, rescue_df], ignore_index=True)
    out_df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}: {len(out_df)} total rows ({len(rescue_df)} bidir-rescue rows added)")

    rows = []
    for pi, sub in out_df.groupby("puzzle_idx"):
        rows.append({
            "config_tag": "bidir_rescue_240mer",
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
    pd.DataFrame(rescue_log).to_csv(out_dir / "bidir_rescue_log.csv", index=False)
    n_perf = (out_df["jaccard_vs_target"] == 1.0).sum()
    solved = sum(1 for r in rows if r["n_perfect_jaccard"] > 0)
    print(f"\n=== bidir + bidirectional-rescue ===")
    print(f"  Total: {len(out_df)} samples, perfect%={100*n_perf/len(out_df):.2f}, solved={solved}/{len(rows)}")


if __name__ == "__main__":
    main()
