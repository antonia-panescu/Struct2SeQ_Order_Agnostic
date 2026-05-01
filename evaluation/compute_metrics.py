"""
Post-process a run_eval.py output directory to produce a knitnet-format
evaluation_summary.json (MCC, Jaccard, F1, sensitivity, PPV, NSR,
gc_deviation, kl_divergence, success rates, per-strategy + global aggregates).

Reads per_sample.csv + per_target.csv from the run_dir (both written by
run_eval.py). Writes:
  evaluation_summary.json     - knitnet-format summary
  per_target_metrics.csv      - per-target × per-strategy metric rows
  per_sample_metrics.csv      - per-sample metric rows (every (target, strategy, sample))

Metric definitions are kept bit-for-bit identical to
knitnet/src/utils/rna_utils.py and knitnet/scripts/eval/evaluate_struct2seq_benchmark.py
so numbers are directly comparable to existing knitnet eval JSONs.

Usage:
  python evaluation/compute_metrics.py --run-dir evaluation/<UTC>__...
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.stats


# ---------------------------------------------------------------------------
# Verbatim from knitnet/src/utils/rna_utils.py
# ---------------------------------------------------------------------------

def dot_bracket_to_pairs(structure: str) -> List[Tuple[int, int]]:
    """Extended dot-bracket → list of (i, j) base pairs (matches knitnet)."""
    OPEN = "([{<"
    CLOSE = ")]}>"
    close_to_open = dict(zip(CLOSE, OPEN))
    stacks: Dict[str, List[int]] = {o: [] for o in OPEN}
    pairs = []
    for i, char in enumerate(structure):
        if char in OPEN:
            stacks[char].append(i)
        elif char in CLOSE:
            open_char = close_to_open[char]
            if stacks[open_char]:
                j = stacks[open_char].pop()
                pairs.append((j, i))
    return pairs


def calculate_jaccard_index(pred_structure: str, target_structure: str) -> float:
    pred_pairs = set(dot_bracket_to_pairs(pred_structure))
    target_pairs = set(dot_bracket_to_pairs(target_structure))
    intersection = len(pred_pairs & target_pairs)
    union = len(pred_pairs | target_pairs)
    if union == 0:
        return 1.0
    return intersection / union


def calculate_structure_mcc(pred_structure: str, true_structure: str) -> float:
    if len(pred_structure) != len(true_structure):
        raise ValueError("Structures must have same length")
    true_paired = set()
    for i, j in dot_bracket_to_pairs(true_structure):
        true_paired.add(i); true_paired.add(j)
    pred_paired = set()
    for i, j in dot_bracket_to_pairs(pred_structure):
        pred_paired.add(i); pred_paired.add(j)
    length = len(true_structure)
    tp = tn = fp = fn = 0
    for i in range(length):
        t = i in true_paired
        p = i in pred_paired
        if p and t: tp += 1
        elif (not p) and (not t): tn += 1
        elif p and (not t): fp += 1
        else: fn += 1
    num = (tp * tn) - (fp * fn)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if den == 0:
        if tp == 0 and fp == 0 and fn == 0:
            return 1.0
        return 0.0
    return num / den


def calculate_structure_accuracy(pred_structure: str, true_structure: str) -> Dict[str, float]:
    pred_pairs = set(dot_bracket_to_pairs(pred_structure))
    true_pairs = set(dot_bracket_to_pairs(true_structure))
    if len(true_pairs) == 0 and len(pred_pairs) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if len(pred_pairs) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp = len(pred_pairs & true_pairs)
    fp = len(pred_pairs - true_pairs)
    fn = len(true_pairs - pred_pairs)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def sequence_recovery(pred_seq: str, true_seq: str, structure: str) -> Dict[str, float]:
    if len(pred_seq) != len(true_seq) or len(pred_seq) != len(structure):
        raise ValueError("Sequences and structure must have same length")
    paired_positions = set()
    for i, j in dot_bracket_to_pairs(structure):
        paired_positions.add(i); paired_positions.add(j)
    total_match = paired_match = unpaired_match = 0
    paired_count = unpaired_count = 0
    for i, (p, t) in enumerate(zip(pred_seq, true_seq)):
        if p == t:
            total_match += 1
            if i in paired_positions: paired_match += 1
            else: unpaired_match += 1
        if i in paired_positions: paired_count += 1
        else: unpaired_count += 1
    return {
        "overall": total_match / len(pred_seq),
        "paired": paired_match / paired_count if paired_count > 0 else 0.0,
        "unpaired": unpaired_match / unpaired_count if unpaired_count > 0 else 0.0,
    }


def compute_nucleotide_metrics(generated_seq: str, native_seq: str) -> Dict[str, float]:
    """GC deviation + KL divergence vs natural distribution (matches knitnet)."""
    if len(generated_seq) == 0:
        return {"gc_deviation": 0.0, "kl_divergence": 0.0}
    gen_gc = (generated_seq.count("G") + generated_seq.count("C")) / len(generated_seq)
    target_gc = (
        (native_seq.count("G") + native_seq.count("C")) / len(native_seq)
        if len(native_seq) > 0 else 0.25
    )
    gc_deviation = abs(gen_gc - target_gc)
    natural_freq = np.array([0.25, 0.22, 0.28, 0.25])
    gen_freq = np.array([generated_seq.count(n) / len(generated_seq) for n in "ACGU"])
    gen_freq = gen_freq + 1e-10
    gen_freq = gen_freq / gen_freq.sum()
    kl_div = float(scipy.stats.entropy(gen_freq, natural_freq))
    return {"gc_deviation": gc_deviation, "kl_divergence": kl_div}


def compute_pairwise_sequence_identity(sequences: List[str]) -> Dict[str, float]:
    """Mean pairwise hamming / identity across N sequences."""
    n = len(sequences)
    if n < 2:
        return {"mean_hamming_distance": 0.0, "mean_sequence_identity": 1.0,
                "min_hamming_distance": 0.0, "max_hamming_distance": 0.0}
    valid = [s for s in sequences if s and len(s) > 0]
    if len(valid) < 2:
        return {"mean_hamming_distance": 0.0, "mean_sequence_identity": 1.0,
                "min_hamming_distance": 0.0, "max_hamming_distance": 0.0}
    distances = []
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            if len(valid[i]) != len(valid[j]):
                continue
            ham = sum(a != b for a, b in zip(valid[i], valid[j]))
            distances.append(ham / len(valid[i]))
    if not distances:
        return {"mean_hamming_distance": 0.0, "mean_sequence_identity": 1.0,
                "min_hamming_distance": 0.0, "max_hamming_distance": 0.0}
    return {
        "mean_hamming_distance": float(np.mean(distances)),
        "mean_sequence_identity": 1.0 - float(np.mean(distances)),
        "min_hamming_distance": float(np.min(distances)),
        "max_hamming_distance": float(np.max(distances)),
    }


# ---------------------------------------------------------------------------
# Per-sample metric assembly (mirrors compute_sample_metrics in knitnet eval)
# ---------------------------------------------------------------------------

def compute_sample_metrics(
    generated_seq: str,
    target_structure: str,
    native_sequence: str,
    pred_structure: str,
) -> Dict:
    """Same metric set as knitnet's evaluate_struct2seq_benchmark.compute_sample_metrics,
    but our pred_structure comes from the eval pipeline (RibonanzaNet+Hungarian)
    instead of being computed here. Length-aligned defensively."""
    if not generated_seq or len(generated_seq) == 0:
        return {"jaccard": 0.0, "mcc": 0.0, "f1": 0.0, "sensitivity": 0.0, "ppv": 0.0,
                "nsr_overall": 0.0, "nsr_paired": 0.0, "nsr_unpaired": 0.0,
                "gc_deviation": 0.0, "kl_divergence": 0.0, "valid": False}

    L = len(target_structure)
    if len(generated_seq) != L:
        if len(generated_seq) < L:
            generated_seq = generated_seq + "A" * (L - len(generated_seq))
        else:
            generated_seq = generated_seq[:L]
    if len(pred_structure) != L:
        if len(pred_structure) < L:
            pred_structure = pred_structure + "." * (L - len(pred_structure))
        else:
            pred_structure = pred_structure[:L]

    jaccard = calculate_jaccard_index(pred_structure, target_structure)
    try:
        mcc = calculate_structure_mcc(pred_structure, target_structure)
    except ValueError:
        mcc = 0.0
    sa = calculate_structure_accuracy(pred_structure, target_structure)

    native_seq = native_sequence or ""
    if len(native_seq) != L:
        if len(native_seq) < L:
            native_seq = native_seq + "A" * (L - len(native_seq))
        else:
            native_seq = native_seq[:L]
    try:
        nsr = sequence_recovery(generated_seq, native_seq, target_structure)
    except ValueError:
        nsr = {"overall": 0.0, "paired": 0.0, "unpaired": 0.0}

    nuc = compute_nucleotide_metrics(generated_seq, native_sequence or "")

    return {
        "jaccard": jaccard, "mcc": mcc, "f1": sa["f1"],
        "sensitivity": sa["recall"], "ppv": sa["precision"],
        "nsr_overall": nsr["overall"], "nsr_paired": nsr["paired"], "nsr_unpaired": nsr["unpaired"],
        "gc_deviation": nuc["gc_deviation"], "kl_divergence": nuc["kl_divergence"],
        "valid": True,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

PER_TARGET_NUMERIC = [
    "jaccard_mean", "jaccard_median", "jaccard_std", "jaccard_best",
    "mcc_mean", "mcc_median", "mcc_std", "mcc_best",
    "success_rate_jaccard_0.8", "success_rate_jaccard_0.9",
    "success_rate_mcc_0.8", "success_rate_mcc_0.9",
    "nsr_overall_mean", "nsr_paired_mean", "nsr_unpaired_mean",
    "f1_mean", "sensitivity_mean", "ppv_mean",
    "gc_deviation_mean", "kl_divergence_mean",
    "pairwise_seq_identity_mean", "pairwise_hamming_mean",
]


def per_target_aggregate(group: pd.DataFrame, target_id: str, target_struct: str,
                         native_seq: str) -> Dict:
    valid = group[group["valid"] == True]
    n_valid = len(valid)
    if n_valid == 0:
        return {"target_id": target_id, "n_samples": 0}
    js = valid["jaccard"].tolist()
    mc = valid["mcc"].tolist()
    sequences = valid["predicted_sequence"].tolist()
    div = compute_pairwise_sequence_identity(sequences)
    return {
        "target_id": target_id,
        "target_structure": target_struct,
        "native_sequence": native_seq,
        "seq_length": len(target_struct),
        "n_samples": n_valid,
        "jaccard_mean": float(np.mean(js)),
        "jaccard_median": float(np.median(js)),
        "jaccard_std": float(np.std(js)),
        "jaccard_best": float(np.max(js)),
        "mcc_mean": float(np.mean(mc)),
        "mcc_median": float(np.median(mc)),
        "mcc_std": float(np.std(mc)),
        "mcc_best": float(np.max(mc)),
        "success_rate_jaccard_0.8": float(np.mean([j >= 0.8 for j in js])),
        "success_rate_jaccard_0.9": float(np.mean([j >= 0.9 for j in js])),
        "success_rate_mcc_0.8":     float(np.mean([m >= 0.8 for m in mc])),
        "success_rate_mcc_0.9":     float(np.mean([m >= 0.9 for m in mc])),
        "nsr_overall_mean":  float(valid["nsr_overall"].mean()),
        "nsr_paired_mean":   float(valid["nsr_paired"].mean()),
        "nsr_unpaired_mean": float(valid["nsr_unpaired"].mean()),
        "f1_mean":          float(valid["f1"].mean()),
        "sensitivity_mean": float(valid["sensitivity"].mean()),
        "ppv_mean":         float(valid["ppv"].mean()),
        "gc_deviation_mean": float(valid["gc_deviation"].mean()),
        "kl_divergence_mean": float(valid["kl_divergence"].mean()),
        "pairwise_seq_identity_mean": div["mean_sequence_identity"],
        "pairwise_hamming_mean":      div["mean_hamming_distance"],
    }


def build_global_summary(summary_df: pd.DataFrame) -> Dict:
    """Match knitnet's build_model_evaluation_summary global block."""
    if summary_df.empty:
        return {}
    return {
        "n_targets": int(len(summary_df)),
        "total_samples_generated": int(summary_df["n_samples"].sum()),
        "jaccard_mean_global":   float(summary_df["jaccard_mean"].mean()),
        "jaccard_median_global": float(summary_df["jaccard_median"].median()),
        "jaccard_best_global":   float(summary_df["jaccard_best"].max()),
        "jaccard_best_mean_global": float(summary_df["jaccard_best"].mean()),
        "jaccard_best_max_global":  float(summary_df["jaccard_best"].max()),
        "mcc_mean_global":   float(summary_df["mcc_mean"].mean()),
        "mcc_median_global": float(summary_df["mcc_median"].median()),
        "mcc_best_global":   float(summary_df["mcc_best"].max()),
        "mcc_best_mean_global": float(summary_df["mcc_best"].mean()),
        "mcc_best_max_global":  float(summary_df["mcc_best"].max()),
        "success_rate_jaccard_0.8_global": float(summary_df["success_rate_jaccard_0.8"].mean()),
        "success_rate_jaccard_0.9_global": float(summary_df["success_rate_jaccard_0.9"].mean()),
        "success_rate_mcc_0.8_global":     float(summary_df["success_rate_mcc_0.8"].mean()),
        "success_rate_mcc_0.9_global":     float(summary_df["success_rate_mcc_0.9"].mean()),
        "nsr_overall_global":  float(summary_df["nsr_overall_mean"].mean()),
        "nsr_paired_global":   float(summary_df["nsr_paired_mean"].mean()),
        "nsr_unpaired_global": float(summary_df["nsr_unpaired_mean"].mean()),
        "pairwise_seq_identity_global": float(summary_df["pairwise_seq_identity_mean"].mean()),
        "f1_mean_global":          float(summary_df["f1_mean"].mean()),
        "sensitivity_mean_global": float(summary_df["sensitivity_mean"].mean()),
        "ppv_mean_global":         float(summary_df["ppv_mean"].mean()),
        "gc_deviation_global":  float(summary_df["gc_deviation_mean"].mean()),
        "kl_divergence_global": float(summary_df["kl_divergence_mean"].mean()),
        "num_structures_jaccard_best_1": int((np.isclose(summary_df["jaccard_best"].values, 1.0)).sum()),
        "num_structures_mcc_best_1":     int((np.isclose(summary_df["mcc_best"].values, 1.0)).sum()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, type=Path,
                   help="run_eval.py output directory containing per_sample.csv")
    p.add_argument("--output-name", default="evaluation_summary.json", type=str)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    sample_csv = run_dir / "per_sample.csv"
    target_csv = run_dir / "per_target.csv"
    if not sample_csv.exists() or not target_csv.exists():
        print(f"[metrics] missing per_sample.csv or per_target.csv in {run_dir}", file=sys.stderr)
        return 1

    print(f"[metrics] reading {sample_csv}")
    samples = pd.read_csv(sample_csv)
    targets = pd.read_csv(target_csv)
    print(f"[metrics] {len(samples)} samples, {len(targets)} targets")

    # Join target metadata onto samples
    target_meta = targets.set_index("name")[["target_structure", "target_sequence"]]
    samples = samples.join(target_meta, on="name", how="left")

    # ---- per-sample metrics --------------------------------------------------
    out_rows = []
    for idx, row in samples.iterrows():
        m = compute_sample_metrics(
            generated_seq=str(row["predicted_sequence"]),
            target_structure=str(row["target_structure"]),
            native_sequence=str(row["target_sequence"]) if not pd.isna(row["target_sequence"]) else "",
            pred_structure=str(row["predicted_structure"]),
        )
        m["row_index"] = row["row_index"]
        m["name"] = row["name"]
        m["length"] = row["length"]
        m["strategy"] = row["strategy"]
        m["sample_idx"] = row["sample_idx"]
        m["predicted_sequence"] = row["predicted_sequence"]
        m["predicted_structure"] = row["predicted_structure"]
        m["reward_from_eval"] = row["reward"]
        out_rows.append(m)
        if (idx + 1) % 5000 == 0:
            print(f"[metrics] processed {idx + 1} / {len(samples)} samples")
    metrics_df = pd.DataFrame(out_rows)
    metrics_df.to_csv(run_dir / "per_sample_metrics.csv", index=False)

    # ---- per-target × per-strategy + per-target across-all aggregates --------
    strategies = sorted(samples["strategy"].unique().tolist())
    per_target_strategy_rows = []
    per_target_global_rows = []
    name_to_struct = dict(zip(targets["name"], targets["target_structure"]))
    name_to_native = dict(zip(targets["name"], targets["target_sequence"].fillna("")))

    for name, target_group in metrics_df.groupby("name"):
        target_struct = name_to_struct.get(name, "")
        native_seq = name_to_native.get(name, "")

        # per strategy
        for strat in strategies:
            sub = target_group[target_group["strategy"] == strat]
            agg = per_target_aggregate(sub, name, target_struct, native_seq)
            agg["strategy"] = strat
            per_target_strategy_rows.append(agg)

        # across-all (best of all 96 samples, mean across all)
        agg_all = per_target_aggregate(target_group, name, target_struct, native_seq)
        agg_all["strategy"] = "_all_"
        per_target_global_rows.append(agg_all)

    per_target_strategy_df = pd.DataFrame(per_target_strategy_rows)
    per_target_all_df = pd.DataFrame(per_target_global_rows)
    pd.concat([per_target_all_df, per_target_strategy_df], ignore_index=True).to_csv(
        run_dir / "per_target_metrics.csv", index=False
    )

    # ---- global aggregates ---------------------------------------------------
    per_strategy_global = {}
    for strat in strategies:
        sdf = per_target_strategy_df[per_target_strategy_df["strategy"] == strat]
        if len(sdf):
            per_strategy_global[strat] = build_global_summary(sdf)

    overall_global = build_global_summary(per_target_all_df)

    # ---- assemble final JSON (knitnet-shaped) --------------------------------
    # Read run_eval.py's summary.json (if present) to copy through metadata
    legacy_summary = {}
    if (run_dir / "summary.json").exists():
        with open(run_dir / "summary.json") as f:
            legacy_summary = json.load(f)

    summary = {
        "run_id": legacy_summary.get("run_id"),
        "timestamp": legacy_summary.get("utc_started"),
        "experiment": legacy_summary.get("experiment"),
        "label": legacy_summary.get("label"),
        "data_label": legacy_summary.get("data_label"),
        "test_csv": legacy_summary.get("test_csv"),
        "decoding_order": legacy_summary.get("decoding_order"),
        "checkpoint": legacy_summary.get("checkpoint"),
        "model": legacy_summary.get("model"),
        "n_targets": overall_global.get("n_targets", 0),
        "n_samples_per_strategy": int(legacy_summary.get("samples_per_strategy", 0)),
        "n_strategies": len(strategies),
        "n_samples_per_target": int(legacy_summary.get("samples_per_strategy", 0)) * len(strategies),
        "total_samples_generated": overall_global.get("total_samples_generated", 0),
        "sampling_strategies": legacy_summary.get("sampling", {}).get("strategies"),
        "folding_engine": "RibonanzaNet-SS+Hungarian (theta=0.5, min_helix=1)",
        "per_strategy_global": per_strategy_global,
        # global = best-of-all-96 for each target, then aggregated across targets
        **overall_global,
    }

    out_json = run_dir / args.output_name
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[metrics] wrote {out_json}")

    # ---- console headline ----------------------------------------------------
    print("\n[metrics] === GLOBAL (best-of-all-strategies per target, then mean across targets) ===")
    print(f"  n_targets                        : {overall_global.get('n_targets')}")
    print(f"  mcc_best_mean_global             : {overall_global.get('mcc_best_mean_global'):.6f}")
    print(f"  jaccard_best_mean_global         : {overall_global.get('jaccard_best_mean_global'):.6f}")
    print(f"  mcc_mean_global                  : {overall_global.get('mcc_mean_global'):.6f}")
    print(f"  jaccard_mean_global              : {overall_global.get('jaccard_mean_global'):.6f}")
    print(f"  f1_mean_global                   : {overall_global.get('f1_mean_global'):.6f}")
    print(f"  sensitivity_mean_global          : {overall_global.get('sensitivity_mean_global'):.6f}")
    print(f"  ppv_mean_global                  : {overall_global.get('ppv_mean_global'):.6f}")
    print(f"  nsr_overall_global               : {overall_global.get('nsr_overall_global'):.6f}")
    print(f"  success_rate_mcc_0.8_global      : {overall_global.get('success_rate_mcc_0.8_global'):.6f}")
    print(f"  success_rate_jaccard_0.8_global  : {overall_global.get('success_rate_jaccard_0.8_global'):.6f}")
    print(f"  num_structures_mcc_best_1        : {overall_global.get('num_structures_mcc_best_1')}")
    print(f"  num_structures_jaccard_best_1    : {overall_global.get('num_structures_jaccard_best_1')}")

    print("\n[metrics] === PER STRATEGY (mean across targets of per-target stats) ===")
    for s in strategies:
        g = per_strategy_global.get(s, {})
        print(f"  {s:18s}  mcc_mean={g.get('mcc_mean_global', 0):.4f}  "
              f"mcc_best_mean={g.get('mcc_best_mean_global', 0):.4f}  "
              f"jaccard_mean={g.get('jaccard_mean_global', 0):.4f}  "
              f"f1_mean={g.get('f1_mean_global', 0):.4f}  "
              f"nsr={g.get('nsr_overall_global', 0):.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
