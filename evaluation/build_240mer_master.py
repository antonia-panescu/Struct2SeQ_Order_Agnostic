"""Aggregate all OK7b 240mer results into a single master comparison
table for the paper. Produces:
  - results/ok7b_eval/MASTER_240mer.csv (one row per config)
  - prints the headline comparison table

Run from struct2seq_bidir_rl/. Re-runnable; reads existing summary CSVs.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

ROOT = Path("results/ok7b_eval")

# (config_tag, dir, label, group)
CONFIGS = [
    # AR baseline (orig Struct2SeQ + L→R argmax-only — the originally-broken baseline)
    ("orig_struct2seq_l2r_240mer",
     "orig_struct2seq_l2r_240mer",
     "AR L→R argmax-only",
     "AR_baseline"),
    ("orig_struct2seq_l2r_rescue_240mer",
     "orig_struct2seq_l2r_rescue_240mer",
     "AR L→R argmax-only + rescue",
     "AR_baseline_rescue"),

    # AR with single strategies (per-strategy breakdown of the 3-strategy mix)
    ("orig_struct2seq_strategy_epsilon05_240mer",
     "orig_struct2seq_strategy_epsilon05_240mer",
     "AR strategy: epsilon p=0.05 (K=333)",
     "AR_per_strategy"),
    ("orig_struct2seq_strategy_epsilon_240mer",
     "orig_struct2seq_strategy_epsilon_240mer",
     "AR strategy: epsilon p=0.10 (K=333)",
     "AR_per_strategy"),
    ("orig_struct2seq_strategy_qsoftmax_240mer",
     "orig_struct2seq_strategy_qsoftmax_240mer",
     "AR strategy: Q-softmax p=1.0 (K=333)",
     "AR_per_strategy"),
    ("orig_struct2seq_strategy_topk_240mer",
     "orig_struct2seq_strategy_topk_240mer",
     "AR strategy: topk beam k=50 (K=50)",
     "AR_per_strategy"),
    ("orig_struct2seq_strategy_topk128_240mer",
     "orig_struct2seq_strategy_topk128_240mer",
     "AR strategy: topk beam k=128 (K=128)",
     "AR_per_strategy"),

    # Faithful Shujun protocol (test_240.py: 3 strategies + rescue, NO beam)
    ("orig_struct2seq_3strategies_faithful_240mer",
     "orig_struct2seq_3strategies_faithful_240mer",
     "AR + faithful 3-strategy (eps0.05+eps0.10+qsoftmax, K~999)",
     "AR_full"),
    ("orig_struct2seq_3strategies_rescue_240mer",
     "orig_struct2seq_3strategies_rescue_240mer",
     "AR + faithful 3-strategy + rescue (matches test_240.py exactly)",
     "AR_full"),
    ("orig_struct2seq_4strategies_240mer",
     "orig_struct2seq_4strategies_240mer",
     "AR + 4-strategy (paper §2.7: 3-strategy + topk_k=128, K~1127)",
     "AR_full"),
    ("orig_struct2seq_4strategies_rescue_240mer",
     "orig_struct2seq_4strategies_rescue_240mer",
     "AR + 4-strategy + rescue (paper §2.7 + post-hoc repair)",
     "AR_full"),

    # Our model
    ("bidir_random_240mer",
     "bidir_random_240mer",
     "Ours: bidir_random argmax-only (K=1000)",
     "ours"),
    ("bidir_random_rescue_240mer",
     "bidir_random_rescue_240mer",
     "Ours: bidir_random argmax-only + Shujun-rescue (4^k local enum)",
     "ours"),
    ("bidir_random_bidirescue_240mer",
     "bidir_random_bidirescue_240mer",
     "Ours: bidir_random argmax-only + bidir-rescue (model-guided in-paint)",
     "ours"),

    # Our model under L→R AR (architecture-vs-order ablation)
    ("bidir_identity_rescue_240mer",
     "bidir_identity_rescue_240mer",
     "Ours: bidir_identity (L→R argmax) + rescue",
     "ours_AR"),
    ("bidir_identity_epsilon05_240mer",
     "bidir_identity_epsilon05_240mer",
     "Ours: bidir_identity (L→R) + epsilon p=0.05",
     "ours_AR"),
    ("bidir_identity_epsilon_240mer",
     "bidir_identity_epsilon_240mer",
     "Ours: bidir_identity (L→R) + epsilon p=0.10",
     "ours_AR"),
    ("bidir_identity_qsoftmax_240mer",
     "bidir_identity_qsoftmax_240mer",
     "Ours: bidir_identity (L→R) + qsoftmax",
     "ours_AR"),
    ("bidir_random_qsoftmax_240mer",
     "bidir_random_qsoftmax_240mer",
     "Ours: bidir_random + Q-softmax (K=333)",
     "ours_per_strategy"),
    ("bidir_random_epsilon_240mer",
     "bidir_random_epsilon_240mer",
     "Ours: bidir_random + epsilon p=0.10 (K=333)",
     "ours_per_strategy"),

    # Other ours-model variants (not core but useful)
    ("bidir_identity_240mer",
     "bidir_identity_240mer",
     "Ours: bidir_identity (L→R from same checkpoint, argmax)",
     "ours_ablation"),
    ("bidir_motifs_structural_240mer",
     "bidir_motifs_structural_240mer",
     "Ours: bidir + structural motif-preservation in-painting",
     "ours_inpaint"),

    # Symmetric ablation: ours under L→R + 3-strategy + rescue (matches AR protocol on our checkpoint)
    ("bidir_identity_3strategies_240mer",
     "bidir_identity_3strategies_240mer",
     "Ours: bidir_identity (L→R) + faithful 3-strategy",
     "ours_AR_full"),
    ("bidir_identity_3strategies_rescue_240mer",
     "bidir_identity_3strategies_rescue_240mer",
     "Ours: bidir_identity (L→R) + faithful 3-strategy + rescue",
     "ours_AR_full"),

    # Symmetric ablation: ours under random-perm + 3-strategy + rescue (NEW symmetric)
    ("bidir_random_epsilon05_240mer",
     "bidir_random_epsilon05_240mer",
     "Ours: bidir_random + epsilon p=0.05 (K=333)",
     "ours_per_strategy"),
    ("bidir_random_3strategies_240mer",
     "bidir_random_3strategies_240mer",
     "Ours: bidir_random + faithful 3-strategy (eps0.05+eps0.10+qsoftmax)",
     "ours_full"),
    ("bidir_random_3strategies_rescue_240mer",
     "bidir_random_3strategies_rescue_240mer",
     "Ours: bidir_random + faithful 3-strategy + rescue",
     "ours_full"),

    # Multi-seed bidir-rescue
    ("bidir_random_multiseed_rescue_240mer",
     "bidir_random_multiseed_rescue_240mer",
     "Ours: bidir_random + multi-seed bidir-rescue (top-5, K=100/seed)",
     "ours_rescue_ablation"),
]


def collect():
    rows = []
    for tag, d, label, group in CONFIGS:
        sd = ROOT / d
        sa = sd / "samples.csv"
        su = sd / "summary.csv"
        if not sa.exists() or not su.exists():
            rows.append({
                "tag": tag, "label": label, "group": group,
                "n_samples": None, "n_unique_seqs": None,
                "perfect_pct": None, "puzzles_solved": None,
                "mean_jaccard": None, "mean_ok_score": None,
                "status": "missing",
            })
            continue
        df = pd.read_csv(sa)
        s = pd.read_csv(su)
        rows.append({
            "tag": tag, "label": label, "group": group,
            "n_samples": int(len(df)),
            "n_unique_seqs": int(df["generated_sequence"].nunique()),
            "perfect_pct": round(100 * (df["jaccard_vs_target"] == 1.0).mean(), 2),
            "puzzles_solved": int((s["n_perfect_jaccard"] > 0).sum()),
            "n_puzzles": int(len(s)),
            "mean_jaccard": round(float(df["jaccard_vs_target"].mean()), 4),
            "mean_ok_score": round(float(df["ok_score"].dropna().mean()), 2)
                             if df["ok_score"].dropna().size else None,
            "status": "ok",
        })
    return pd.DataFrame(rows)


def main():
    df = collect()
    out = ROOT / "MASTER_240mer.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out}")
    print()
    print("=== HEADLINE COMPARISON ===")
    headline = df[df["group"].isin(["AR_baseline", "AR_baseline_rescue", "AR_full", "ours"])]
    cols = ["label", "n_samples", "n_unique_seqs", "perfect_pct", "puzzles_solved", "n_puzzles", "mean_jaccard"]
    print(headline[cols].to_string(index=False))
    print()
    print("=== AR PER-STRATEGY ===")
    ps = df[df["group"] == "AR_per_strategy"]
    print(ps[cols].to_string(index=False))
    print()
    print("=== OURS ABLATIONS ===")
    oa = df[df["group"].isin(["ours_per_strategy", "ours_ablation", "ours_inpaint"])]
    print(oa[cols].to_string(index=False))


if __name__ == "__main__":
    main()
