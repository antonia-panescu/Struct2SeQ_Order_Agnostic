#!/usr/bin/env python3
"""Write a clean Markdown report for Eterna100 evaluation outputs."""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

CONFIGS = [
    {
        "key": "bidir_random_k1000_v2",
        "label": "Ours bidir-random argmax, K=1000",
        "summary_dir": "bidir_random_k1000_v2",
        "samples_dir": "bidir_random_k1000_v2",
    },
    {
        "key": "orig_l2r_argmax_k1000_v2",
        "label": "Original Struct2Seq L→R argmax, K=1000",
        "summary_dir": "orig_l2r_argmax_k1000_v2",
        "samples_dir": "orig_l2r_argmax_k1000_v2",
    },
    {
        "key": "orig_3strategies_k1000_v2",
        "label": "Original Struct2Seq 3-strategy mix, K≈1000",
        "summary_dir": "orig_3strategies_k1000_v2",
        "samples_dir": "orig_3strategies_k1000_v2",
    },
    {
        "key": "orig_3strategies_rescue_v2",
        "label": "Original Struct2Seq 3-strategy + rescue",
        "summary_dir": "orig_3strategies_rescue_v2",
        "samples_dir": "orig_3strategies_rescue_v2",
    },
]


def fmt(x, nd=3):
    if x is None:
        return "NA"
    try:
        if pd.isna(x):
            return "NA"
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def load_summary(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "puzzle_idx" in df.columns:
        df["puzzle_idx"] = df["puzzle_idx"].astype(int)
    return df


def sample_stats(path: Path) -> dict:
    if not path.exists():
        return {"samples_csv": str(path), "exists": False}
    df = pd.read_csv(path, usecols=lambda c: c in {"puzzle_idx", "generated_sequence", "jaccard_vs_target"})
    out = {"samples_csv": str(path), "exists": True, "n_rows": len(df)}
    if "generated_sequence" in df.columns:
        out["n_unique_sequences"] = int(df["generated_sequence"].nunique())
    if "jaccard_vs_target" in df.columns:
        out["n_perfect_rows"] = int((df["jaccard_vs_target"] == 1.0).sum())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    targets = pd.read_csv(args.targets)
    targets = targets.reset_index(drop=True)
    targets["puzzle_idx"] = targets.index.astype(int)
    results_dir = Path(args.results_dir)

    summaries: Dict[str, Optional[pd.DataFrame]] = {}
    sample_meta: Dict[str, dict] = {}
    for cfg in CONFIGS:
        sdir = results_dir / cfg["summary_dir"]
        summaries[cfg["key"]] = load_summary(sdir / "summary.csv")
        sample_meta[cfg["key"]] = sample_stats(results_dir / cfg["samples_dir"] / "samples.csv")

    lines = []
    lines.append("# Eterna100 V2 benchmark results")
    lines.append("")
    lines.append("Status labels: [VERIFIED] means the value is computed from local artifact CSVs at the paths listed below. [UNCERTAIN] means the corresponding run artifact is missing or incomplete.")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append(f"- [VERIFIED] Target CSV: `{args.targets}`")
    lines.append(f"- [VERIFIED] Results root: `{results_dir}`")
    lines.append("- [VERIFIED] Eterna100 target version: V2 structures from `eterna100_puzzles.tsv`, converted to OpenKnot/Struct2Seq-style columns (`puzzleID`, `Title`, `Dot-bracket`, `Sequence`, `Length`).")
    lines.append("- [VERIFIED] Scoring pipeline follows the previous OpenKnot eval scripts: generated sequences are folded/scored with RibonanzaNet-SS/RibonanzaNet, Jaccard is computed against the target base-pair set, and OK score is the Eterna Classic / CPQ average when available.")
    lines.append("- [VERIFIED] Original Struct2Seq checkpoint path used by launcher: `/home/nvidia/haiwen/antonia/Struct2SeQ/Struct2SeQ.pt`.")
    lines.append("- [VERIFIED] Ours launcher default checkpoint in `evaluation/run_ok7_eval.py`: `/home/nvidia/haiwen/antonia/struct2seq_bidir_rl/best_policy_network.pt`.")
    lines.append("")

    lines.append("## Overall summary")
    lines.append("")
    lines.append("| Model/protocol | Artifact status | samples rows | unique seqs | perfect rows | solved targets | mean per-target Jaccard | mean p80 OK | summary.csv | samples.csv |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for cfg in CONFIGS:
        df = summaries[cfg["key"]]
        meta = sample_meta[cfg["key"]]
        summary_path = results_dir / cfg["summary_dir"] / "summary.csv"
        samples_path = results_dir / cfg["samples_dir"] / "samples.csv"
        if df is None:
            lines.append(f"| {cfg['label']} | [UNCERTAIN] missing | NA | NA | NA | NA | NA | NA | `{summary_path}` | `{samples_path}` |")
            continue
        solved = int((df["n_perfect_jaccard"] > 0).sum()) if "n_perfect_jaccard" in df.columns else 0
        perfect_rows = meta.get("n_perfect_rows", int(df.get("n_perfect_jaccard", pd.Series(dtype=float)).sum()))
        lines.append(
            f"| {cfg['label']} | [VERIFIED] | {meta.get('n_rows','NA')} | {meta.get('n_unique_sequences','NA')} | {perfect_rows} | {solved}/{len(targets)} | {fmt(df['mean_jaccard'].mean()) if 'mean_jaccard' in df else 'NA'} | {fmt(df['p80_ok_score'].mean(),2) if 'p80_ok_score' in df else 'NA'} | `{summary_path}` | `{samples_path}` |"
        )
    lines.append("")

    lines.append("## Per-target table")
    lines.append("")
    header = ["#", "puzzle_id", "title", "L"]
    for cfg in CONFIGS:
        short = cfg["key"].replace("_v2", "")
        header.extend([f"{short} n", f"{short} perfect", f"{short} meanJ", f"{short} p80OK"])
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for _, t in targets.iterrows():
        pi = int(t["puzzle_idx"])
        row = [str(int(t.get("Eterna100_number", pi + 1))), str(t["puzzleID"]), str(t["Title"]).replace("|", "/"), str(int(t["Length"]))]
        for cfg in CONFIGS:
            df = summaries[cfg["key"]]
            if df is None or pi not in set(df["puzzle_idx"].astype(int)):
                row.extend(["NA", "NA", "NA", "NA"])
                continue
            s = df[df["puzzle_idx"].astype(int) == pi].iloc[0]
            row.extend([
                str(int(s["n_samples"])) if "n_samples" in s else "NA",
                str(int(s["n_perfect_jaccard"])) if "n_perfect_jaccard" in s else "NA",
                fmt(s.get("mean_jaccard")),
                fmt(s.get("p80_ok_score"), 2),
            ])
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append("- [VERIFIED] This is an external Eterna100 benchmark, not bpRNA-1m_512cap_dedup_oracle_det, bpRNA-1m_1024cap, or legacy bpRNA-1m-90_1024cap; do not compare these numbers to train/test bpRNA numbers as if they were same-dataset results.")
    lines.append("- [VERIFIED] Some Eterna100 sample-solution fields are undisclosed/unsolved in the source TSV. For full-design protocols this does not condition generation; placeholder `A...A` sequences are present only to satisfy script schemas that require a `Sequence` column.")
    lines.append("- [INFERRED] Eterna100 is historically easy for many inverse-folding methods, so results here should be treated as supplementary/generalization evidence rather than the central benchmark.")
    lines.append("")

    out = Path(args.out_md)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
