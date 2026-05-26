"""Build centralized MASTER results tables for the paper.

Scans all results/ok7_eval/* (100mer) and results/ok7b_eval/* (240mer)
configs, computes summary statistics (perfect%, puzzles_solved,
mean_jaccard, mean_ok_score), labels them, and writes:

  paper_figs/MASTER_100mer.csv
  paper_figs/MASTER_240mer.csv

Each row has tag, label, group, n_samples, perfect_pct, puzzles_solved,
mean_jaccard, mean_ok_score, n_unique_seqs, status.

Usage: python paper_figs/build_master_tables.py
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "paper_figs"
OUT.mkdir(exist_ok=True)


# Manual labels + group mapping. New tags added at bottom; if a tag is
# missing from this map it falls into "uncategorized" group with a
# best-guess label.
LABELS = {
    # ===== 100mer =====
    "bidir_random": ("Ours: bidir_random argmax-only (K=1000)", "ours"),
    "bidir_identity": ("Ours: bidir_identity (L→R from same checkpoint, argmax)", "ours_ablation"),
    "bidir_paired_first": ("Ours: bidir + paired-first decode order", "ours_ablation"),
    "bidir_motifs_structural": ("Ours: bidir + structural motif-preservation in-painting", "ours_inpaint"),
    "bidir_motifs_redesign": ("Ours: bidir + motif-redesign (Framing B, fix scaffold)", "ours_inpaint"),
    "bidir_motifs_redesign_fixedfirst": ("Ours: motif-redesign + fixed-first decoding", "ours_inpaint"),
    "bidir_inpaint_K00": ("Ours: bidir + scatter-inpaint K=0% (sanity)", "ours_inpaint_kscan"),
    "bidir_inpaint_K25": ("Ours: bidir + scatter-inpaint K=25%", "ours_inpaint_kscan"),
    "bidir_inpaint_K50": ("Ours: bidir + scatter-inpaint K=50%", "ours_inpaint_kscan"),
    "bidir_inpaint_K75": ("Ours: bidir + scatter-inpaint K=75%", "ours_inpaint_kscan"),
    "bidir_inpaint_K95": ("Ours: bidir + scatter-inpaint K=95% (rescue regime)", "ours_inpaint_kscan"),
    "orig_struct2seq_l2r": ("AR L→R argmax-only", "AR_baseline"),

    # ===== 240mer =====
    "bidir_random_240mer": ("Ours: bidir_random argmax-only (K=1000)", "ours"),
    "bidir_random_rescue_240mer": ("Ours: bidir_random + Shujun-rescue", "ours"),
    "bidir_random_bidirescue_240mer": ("Ours: bidir_random + bidir-rescue (model-guided in-paint)", "ours"),
    "bidir_random_multiseed_rescue_240mer": ("Ours: bidir_random + multi-seed bidir-rescue", "ours_rescue_ablation"),
    "bidir_random_epsilon_240mer": ("Ours: bidir_random + epsilon p=0.10 (K=333)", "ours_per_strategy"),
    "bidir_random_epsilon05_240mer": ("Ours: bidir_random + epsilon p=0.05 (K=333)", "ours_per_strategy"),
    "bidir_random_qsoftmax_240mer": ("Ours: bidir_random + Q-softmax (K=333)", "ours_per_strategy"),
    "bidir_random_3strategies_240mer": ("Ours: bidir_random + 3-strategy mix", "ours_full"),
    "bidir_random_3strategies_rescue_240mer": ("Ours: bidir_random + 3-strategy + rescue", "ours_full"),
    "bidir_identity_240mer": ("Ours: bidir_identity (L→R from same checkpoint, argmax)", "ours_ablation"),
    "bidir_identity_rescue_240mer": ("Ours: bidir_identity + Shujun-rescue", "ours_AR"),
    "bidir_identity_epsilon_240mer": ("Ours: bidir_identity + epsilon p=0.10", "ours_AR"),
    "bidir_identity_epsilon05_240mer": ("Ours: bidir_identity + epsilon p=0.05", "ours_AR"),
    "bidir_identity_qsoftmax_240mer": ("Ours: bidir_identity + Q-softmax", "ours_AR"),
    "bidir_identity_3strategies_240mer": ("Ours: bidir_identity + 3-strategy mix", "ours_AR_full"),
    "bidir_identity_3strategies_rescue_240mer": ("Ours: bidir_identity + 3-strategy + rescue", "ours_AR_full"),
    "bidir_motifs_structural_240mer": ("Ours: bidir + structural motif-preservation in-painting", "ours_inpaint"),
    "bidir_motifs_structural_rescue_240mer": ("Ours: bidir + motif-preservation + rescue", "ours_inpaint"),

    "orig_struct2seq_l2r_240mer": ("AR L→R argmax-only", "AR_baseline"),
    "orig_struct2seq_l2r_rescue_240mer": ("AR L→R argmax-only + rescue", "AR_baseline_rescue"),
    "orig_struct2seq_strategy_epsilon05_240mer": ("AR strategy: epsilon p=0.05 (K=333)", "AR_per_strategy"),
    "orig_struct2seq_strategy_epsilon_240mer": ("AR strategy: epsilon p=0.10 (K=333)", "AR_per_strategy"),
    "orig_struct2seq_strategy_qsoftmax_240mer": ("AR strategy: Q-softmax (K=333)", "AR_per_strategy"),
    "orig_struct2seq_strategy_topk_240mer": ("AR strategy: topk beam k=50", "AR_per_strategy"),
    "orig_struct2seq_strategy_topk128_240mer": ("AR strategy: topk beam k=128", "AR_per_strategy"),
    "orig_struct2seq_3strategies_240mer": ("AR + 3-strategy (eps0.05+eps0.10+qsoftmax)", "AR_full"),
    "orig_struct2seq_3strategies_faithful_240mer": ("AR + faithful 3-strategy", "AR_full"),
    "orig_struct2seq_3strategies_rescue_240mer": ("AR + 3-strategy + rescue", "AR_full"),
    "orig_struct2seq_4strategies_240mer": ("AR + 4-strategy (paper §2.7)", "AR_full"),
    "orig_struct2seq_4strategies_rescue_240mer": ("AR + 4-strategy + rescue", "AR_full"),
    "orig_struct2seq_motifs_structural_240mer": ("AR + teacher-forced motif (argmax-only)", "AR_inpaint"),
    "orig_struct2seq_motifs_structural_rescue_240mer": ("AR + teacher-forced motif + rescue", "AR_inpaint"),
    "orig_motifs_structural_epsilon05_240mer": ("AR + teacher-forced motif + epsilon p=0.05 (K=333)", "AR_inpaint_per_strategy"),
    "orig_motifs_structural_epsilon10_240mer": ("AR + teacher-forced motif + epsilon p=0.10 (K=333)", "AR_inpaint_per_strategy"),
    "orig_motifs_structural_qsoftmax_240mer": ("AR + teacher-forced motif + Q-softmax (K=333)", "AR_inpaint_per_strategy"),
    "orig_motifs_3strategies_faithful_240mer": ("AR + teacher-forced motif + 3-strategy mix", "AR_inpaint_full"),
    "orig_motifs_3strategies_rescue_240mer": ("AR + teacher-forced motif + 3-strategy + rescue", "AR_inpaint_full"),
}


def summarize_config(samples_csv: Path) -> dict:
    df = pd.read_csv(samples_csv)
    n_perf = int((df["jaccard_vs_target"] == 1.0).sum())
    n_solved = int(df.groupby("puzzle_idx")
                   .apply(lambda g: (g["jaccard_vs_target"] == 1.0).any(),
                          include_groups=False).sum())
    n_uniq = int(df["generated_sequence"].nunique()) if "generated_sequence" in df.columns else len(df)
    return {
        "n_samples": len(df),
        "n_unique_seqs": n_uniq,
        "perfect_pct": round(100 * n_perf / max(len(df), 1), 2),
        "puzzles_solved": n_solved,
        "n_puzzles": int(df["puzzle_idx"].nunique()),
        "mean_jaccard": round(float(df["jaccard_vs_target"].mean()), 4),
        "mean_ok_score": round(float(df["ok_score"].mean()), 2)
                          if "ok_score" in df.columns else float("nan"),
        "status": "ok",
    }


def build_master(results_dir: Path, out_path: Path, label="100mer"):
    rows = []
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        samples = d / "samples.csv"
        if not samples.exists():
            continue
        tag = d.name
        label_text, group = LABELS.get(tag, (f"(uncategorized) {tag}", "uncategorized"))
        try:
            stats = summarize_config(samples)
        except Exception as e:
            stats = {"n_samples": -1, "n_unique_seqs": -1, "perfect_pct": -1,
                     "puzzles_solved": -1, "n_puzzles": -1,
                     "mean_jaccard": float("nan"), "mean_ok_score": float("nan"),
                     "status": f"err: {type(e).__name__}"}
        rows.append({"tag": tag, "label": label_text, "group": group, **stats})
    df = pd.DataFrame(rows)
    # group-wise sort
    group_order = ["AR_baseline", "AR_baseline_rescue", "AR_per_strategy",
                   "AR_full", "AR_inpaint", "AR_inpaint_per_strategy", "AR_inpaint_full",
                   "ours", "ours_per_strategy", "ours_full",
                   "ours_inpaint", "ours_inpaint_kscan",
                   "ours_ablation", "ours_AR", "ours_AR_full",
                   "ours_rescue_ablation", "uncategorized"]
    df["group_rank"] = df["group"].map({g: i for i, g in enumerate(group_order)}).fillna(99)
    df = df.sort_values(["group_rank", "perfect_pct"], ascending=[True, False]).drop(columns=["group_rank"])
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}: {len(df)} rows")
    return df


def main():
    df100 = build_master(PROJECT_ROOT / "results" / "ok7_eval",
                         OUT / "MASTER_100mer.csv", label="100mer")
    df240 = build_master(PROJECT_ROOT / "results" / "ok7b_eval",
                         OUT / "MASTER_240mer.csv", label="240mer")
    # Also dump a unified condensed view (for the paper writeup).
    print("\n=== 100mer ===")
    print(df100[["tag", "perfect_pct", "puzzles_solved", "mean_jaccard"]].to_string(index=False))
    print("\n=== 240mer ===")
    print(df240[["tag", "perfect_pct", "puzzles_solved", "mean_jaccard"]].to_string(index=False))


if __name__ == "__main__":
    main()
