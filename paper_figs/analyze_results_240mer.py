"""Paper-figure analyses for OK7b 240mer benchmark.

Three sub-analyses (parallels paper_figs/analyze_results.py for 100mer):
  bestofn   — best-of-N scaling at 240mer (defends K=1000 budget)
  baseline  — per-puzzle bar chart vs all OpenKnotBench Round 4 methods
  mutdist   — Hamming-distance histogram from WT (Shujun Fig 6 analog)

Usage:
    python paper_figs/analyze_results_240mer.py --analysis bestofn
    python paper_figs/analyze_results_240mer.py --analysis baseline
    python paper_figs/analyze_results_240mer.py --analysis mutdist
    python paper_figs/analyze_results_240mer.py --analysis all
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "ok7b_eval"
TARGETS_CSV = "/home/nvidia/haiwen/antonia/OpenKnotAIDesignData/Targets/Round4_targets.csv"
BENCH_CSV = "/home/nvidia/haiwen/antonia/OpenKnotAIDesignData/Data/OpenKnotBench_data.v4.5.1.csv"
OUT = PROJECT_ROOT / "paper_figs"
OUT.mkdir(exist_ok=True)


def _load_samples(config_tag):
    p = RESULTS_DIR / config_tag / "samples.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def _save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=200)
    print(f"wrote {OUT / name}.{{pdf,png}}")


def best_of_n_curve():
    Ns = [1, 5, 25, 100, 500, 1000]
    n_repeats = 50
    rng = np.random.default_rng(0)
    configs = [
        ("bidir_random_240mer", "Ours (random-perm)", "#264653", "-"),
        ("orig_struct2seq_l2r_240mer", "Original Struct2SeQ (AR L→R)", "#E76F51", "-"),
        ("bidir_motifs_structural_240mer", "Ours + structural motif in-paint", "#2A9D8F", "--"),
        ("orig_struct2seq_motifs_structural_240mer", "Original + AR teacher-forced motif", "#9D4EDD", "--"),
        ("orig_struct2seq_3strategies_rescue_240mer", "Original + 3-strategy + rescue (full pipeline)", "#F4A261", ":"),
    ]
    rows = []
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for tag, label, color, ls in configs:
        df = _load_samples(tag)
        if df is None:
            print(f"  skip {tag}: no samples.csv")
            continue
        n_puzzles = df["puzzle_idx"].nunique()
        ys = []
        for N in Ns:
            cnt_per_repeat = []
            for _ in range(n_repeats):
                hit = 0
                for pi, gp in df.groupby("puzzle_idx"):
                    n_avail = len(gp)
                    if n_avail < N:
                        idx = rng.integers(0, n_avail, size=N)
                    else:
                        idx = rng.choice(n_avail, size=N, replace=False)
                    sub = gp.iloc[idx]
                    if (sub["jaccard_vs_target"] == 1.0).any():
                        hit += 1
                cnt_per_repeat.append(hit / n_puzzles)
            mean_hit = np.mean(cnt_per_repeat)
            std_hit = np.std(cnt_per_repeat)
            ys.append(mean_hit)
            rows.append({"config": tag, "N": N, "frac_puzzles_with_perfect": mean_hit,
                         "std": std_hit})
        ax.plot(Ns, ys, marker="o", color=color, linestyle=ls, label=label, linewidth=1.6)

    ax.set_xscale("log")
    ax.set_xlabel("Per-puzzle sample budget N")
    ax.set_ylabel("Fraction of puzzles with ≥1 perfect Jaccard")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7.0, loc="lower right")
    ax.set_title("OK7b 240mer", fontsize=10)
    fig.tight_layout()
    _save(fig, "fig_best_of_n_240mer")
    pd.DataFrame(rows).to_csv(OUT / "best_of_n_240mer.csv", index=False)
    plt.close(fig)


def per_puzzle_baseline_chart():
    bench = pd.read_csv(BENCH_CSV, low_memory=False)
    r4 = bench[bench["round"] == 4]
    targets = pd.read_csv(TARGETS_CSV)

    methods = ["Struct2SeQ", "Struct2SeQ-SHAPE", "gRNAde",
               "MPNN-RFdiff", "MPNN-fixbb", "Rosetta"]

    ours = _load_samples("bidir_random_240mer")
    if ours is None:
        print("  skip baseline chart: no bidir_random_240mer samples")
        return
    ours_top20 = (ours
                  .groupby("puzzle_idx", group_keys=False)
                  .apply(lambda d: d.nlargest(20, "jaccard_vs_target"))
                  .reset_index(drop=True))
    ours_per_puzzle = ours_top20.groupby("puzzle_idx")["jaccard_vs_target"].mean()

    rows = []
    puzzles = sorted(r4["puzzle"].unique())
    for i, p in enumerate(puzzles):
        sub = r4[r4["puzzle"] == p]
        for m in methods:
            ms = sub[sub["method"] == m]
            v = ms["RNet_F1"].mean() if len(ms) else float("nan")
            rows.append({"puzzle_idx": i, "puzzle": p, "method": m, "mean_F1": v})
        rows.append({"puzzle_idx": i, "puzzle": p, "method": "Ours (top-20 of K=1000)",
                     "mean_F1": float(ours_per_puzzle.get(i, float("nan")))})
    bar_df = pd.DataFrame(rows)

    methods_plot = ["Ours (top-20 of K=1000)", "Struct2SeQ", "Struct2SeQ-SHAPE",
                    "MPNN-RFdiff", "gRNAde", "MPNN-fixbb"]
    colors = {
        "Ours (top-20 of K=1000)": "#264653",
        "Struct2SeQ": "#E76F51",
        "Struct2SeQ-SHAPE": "#F4A261",
        "MPNN-RFdiff": "#2A9D8F",
        "gRNAde": "#9D4EDD",
        "MPNN-fixbb": "#6C757D",
    }
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    nP = len(puzzles)
    nM = len(methods_plot)
    width = 0.8 / nM
    for j, m in enumerate(methods_plot):
        sub = bar_df[bar_df["method"] == m].sort_values("puzzle_idx")
        x = np.arange(nP) + j * width - 0.4 + width / 2
        ax.bar(x, sub["mean_F1"].values, width=width, color=colors[m], label=m)

    ax.set_xticks(np.arange(nP))
    ax.set_xticklabels(puzzles, rotation=90, fontsize=6)
    ax.set_ylabel("mean RNet_F1 (Jaccard, top designs)")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, linewidth=0.4, color="black", alpha=0.5)
    ax.legend(fontsize=6.5, loc="lower right", ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title("OK7b 240mer per-puzzle baseline comparison", fontsize=10)
    fig.tight_layout()
    _save(fig, "fig_baseline_comparison_240mer")
    bar_df.to_csv(OUT / "baseline_comparison_240mer.csv", index=False)
    plt.close(fig)


def mutation_distance_hist():
    targets = pd.read_csv(TARGETS_CSV)
    wt_seqs = {i: row["Sequence"] for i, row in targets.iterrows()}

    configs = [
        ("bidir_random_240mer", "Ours (random-perm)", "#264653"),
        ("orig_struct2seq_l2r_240mer", "Original Struct2SeQ (AR L→R)", "#E76F51"),
        ("orig_struct2seq_3strategies_rescue_240mer", "Original + 3-strat + rescue", "#F4A261"),
    ]
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    rows = []
    for tag, label, color in configs:
        df = _load_samples(tag)
        if df is None:
            continue
        dists = []
        for _, r in df.iterrows():
            wt = wt_seqs.get(r["puzzle_idx"])
            if wt is None:
                continue
            seq = r["generated_sequence"]
            L = min(len(wt), len(seq))
            d = sum(1 for k in range(L) if wt[k] != seq[k])
            dists.append(d)
        ax.hist(dists, bins=40, color=color, alpha=0.55, label=label,
                density=True, edgecolor="white", linewidth=0.3)
        rows.append({"config": tag, "mean_dist": float(np.mean(dists)),
                     "median_dist": float(np.median(dists)),
                     "std": float(np.std(dists))})
    ax.set_xlabel("Hamming distance from WT (240mer)")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("OK7b 240mer", fontsize=10)
    fig.tight_layout()
    _save(fig, "fig_mutation_distance_240mer")
    pd.DataFrame(rows).to_csv(OUT / "mutation_distance_240mer.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--analysis",
                   choices=["bestofn", "baseline", "mutdist", "all"],
                   default="all")
    args = p.parse_args()
    if args.analysis in ("bestofn", "all"):
        print("=== best-of-N curve (240mer) ===")
        best_of_n_curve()
    if args.analysis in ("baseline", "all"):
        print("=== per-puzzle baseline bar chart (240mer) ===")
        per_puzzle_baseline_chart()
    if args.analysis in ("mutdist", "all"):
        print("=== mutation distance histogram (240mer) ===")
        mutation_distance_hist()


if __name__ == "__main__":
    main()
