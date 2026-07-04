"""Generate RNA sequences with an order-agnostic Struct2SeQ checkpoint."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from arnie.utils import convert_dotbracket_to_bp_list
from tqdm import tqdm

from Dataset import tokenize_dot_bracket, tokenize_sequence
from Encoder_Decoder import DotBracketRNATransformer
from Env import DQN_env
from Functions import (
    cor2vec,
    delete_modules,
    generate_permuted,
    hamming_distance,
    jaccard_similarity_base_pairs,
    load_config_from_yaml,
    make_ref_upweighted_params,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_POLICY_WEIGHTS = "weights/order_agnostic_policy.pt"
NT_TO_IDX = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}
IDX_TO_NT = {0: "A", 1: "C", 2: "G", 3: "U"}
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate RNA sequences from target dot-bracket structures."
    )
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID to use")
    parser.add_argument(
        "--n_structures",
        type=int,
        default=100,
        help="Samples per strategy per target",
    )
    parser.add_argument(
        "--target_df",
        type=str,
        required=True,
        help="CSV with Title/Dot-bracket/wild_type_sequence columns",
    )
    parser.add_argument(
        "--out_folder", type=str, default="results", help="Output folder"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="output.csv",
        help="Output CSV filename",
    )
    parser.add_argument(
        "--up_bias",
        type=float,
        default=0.0,
        help="Logit bias toward the wild_type_sequence column, matching Struct2SeQ",
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default=DEFAULT_POLICY_WEIGHTS,
        help=f"Order-agnostic policy checkpoint (default: {DEFAULT_POLICY_WEIGHTS})",
    )
    parser.add_argument(
        "--rnet_weights",
        type=str,
        default=None,
        help="RibonanzaNet weights; defaults to weights/RibonanzaNet.pt",
    )
    parser.add_argument(
        "--rnet_ss_weights",
        type=str,
        default=None,
        help="RibonanzaNet-SS weights; defaults to weights/RibonanzaNet-SS.pt",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_brev_8gpu.yaml",
        help="Model config YAML",
    )
    parser.add_argument(
        "--decoding_order",
        choices=["permuted", "l2r"],
        default="permuted",
        help="permuted = order-agnostic random RNA-position order; l2r = ablation",
    )
    parser.add_argument(
        "--score_batch_size",
        type=int,
        default=32,
        help="Batch size for RibonanzaNet scoring",
    )
    parser.add_argument(
        "--rescue_max_diff",
        type=int,
        default=4,
        help="Run mutation rescue when the best design differs at at most this many pairing positions",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_column(df: pl.DataFrame, names: list[str], required: bool = True):
    for name in names:
        if name in df.columns:
            return name
    if required:
        raise ValueError(f"CSV must contain one of these columns: {names}")
    return None


def build_model(config_path: str, checkpoint_path: str, device: torch.device):
    config = load_config_from_yaml(config_path)
    model = DotBracketRNATransformer(
        config.db_vocab_size,
        config.rna_vocab_size,
        config.embed_size,
        config.nhead,
        config.num_encoder_layers,
        config.num_decoder_layers,
        config.dim_feedforward,
        config.dropout,
    )
    delete_modules(model)

    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model_state_dict" in state:
            state = state["model_state_dict"]

    def clean_key(key: str) -> str:
        for prefix in ("_orig_mod.", "module."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        return key

    state = {clean_key(k): v for k, v in state.items()}
    state = {k: v for k, v in state.items() if k not in DROP_OLD_DECODER_PE}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
    allowed_missing = {"decoder.rpe_self.bias.weight", "decoder.rpe_cross.bias.weight"}
    unallowed = [key for key in missing if key not in allowed_missing]
    if unallowed:
        raise RuntimeError(f"Unexpected missing checkpoint keys: {unallowed}")

    model = model.to(device)
    model.eval()
    return model


def build_target_tensors(structure: str, repeat: int, device: torch.device):
    bps = convert_dotbracket_to_bp_list(structure, allow_pseudoknots=True)
    length = len(structure)
    ct_matrix = np.eye(length, dtype="float32")
    for i, j in bps:
        ct_matrix[i, j] = 1.0
        ct_matrix[j, i] = 1.0

    paired_correspondence = {}
    for i, j in bps:
        paired_correspondence[i] = j
        paired_correspondence[j] = i

    src = torch.tensor(tokenize_dot_bracket(structure), dtype=torch.long, device=device)
    src = src.unsqueeze(0).repeat(repeat, 1)
    ct = torch.tensor(ct_matrix, dtype=torch.float32, device=device)
    ct = ct.unsqueeze(0).repeat(repeat, 1, 1)
    correspondences = [paired_correspondence] * repeat
    return src, ct, correspondences, bps


def make_perm(batch_size: int, length: int, mode: str, device: torch.device):
    if mode == "permuted":
        return torch.stack(
            [torch.randperm(length, device=device) for _ in range(batch_size)],
            dim=0,
        )
    if mode == "l2r":
        return (
            torch.arange(length, device=device)
            .unsqueeze(0)
            .expand(batch_size, length)
        )
    raise ValueError(mode)


def token_rows_to_strings(tokens: torch.Tensor) -> list[str]:
    return ["".join(IDX_TO_NT[int(tok)] for tok in row.tolist()) for row in tokens]


def score_sequences(env: DQN_env, sequences: torch.Tensor, batch_size: int):
    structures, base_pairs, shape_profiles = [], [], []
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        batch_structures, batch_bps = env.get_structure(batch)
        batch_shape = env.get_SHAPE(batch)
        structures.extend(batch_structures)
        base_pairs.extend(batch_bps)
        shape_profiles.extend(batch_shape)
    return structures, base_pairs, shape_profiles


def maybe_hamming(reference: str | None, sequence: str):
    if not reference or len(reference) != len(sequence):
        return None
    return hamming_distance(reference.replace("T", "U"), sequence)


def normalize_reference(reference) -> str | None:
    if reference is None:
        return None
    sequence = str(reference).strip().upper().replace("T", "U")
    if sequence in {"", "NONE", "NAN", "NULL"}:
        return None
    return sequence


def mutate_sequence(sequence: str, positions: list[int]) -> list[str]:
    sequences = [sequence]
    for position in reversed(positions):
        sequences = [
            candidate[:position] + nt + candidate[position + 1 :]
            for candidate in sequences
            for nt in "AUGC"
        ]
    return sequences


def rescue_diff_positions(env: DQN_env, target_structure: str, design_structure: str):
    target_corr = env.get_paired_correspondences(
        convert_dotbracket_to_bp_list(target_structure, allow_pseudoknots=True)
    )
    design_corr = env.get_paired_correspondences(
        convert_dotbracket_to_bp_list(design_structure, allow_pseudoknots=True)
    )
    target_vec = cor2vec(target_corr, target_structure)
    design_vec = cor2vec(design_corr, target_structure)
    return np.where(target_vec != design_vec)[0].tolist()


def sequences_to_tensor(sequences: list[str], device: torch.device):
    return torch.tensor(
        [tokenize_sequence(sequence) for sequence in sequences],
        dtype=torch.long,
        device=device,
    )


def main() -> int:
    args = parse_args()
    start_time = time.time()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu_id))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Set --gpu_id to an available GPU.")
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_folder = Path(args.out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    output_path = out_folder / args.output_csv
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = build_model(
        resolve_repo_path(args.config),
        resolve_repo_path(args.weights_path),
        device,
    )
    env = DQN_env(
        args.rnet_weights,
        args.rnet_ss_weights,
        use_gpu=True,
        compile=False,
    )

    targets = pl.read_csv(args.target_df)
    title_col = get_column(targets, ["Title", "name", "source"], required=False)
    structure_col = get_column(targets, ["Dot-bracket", "structure", "target_structure"])
    sequence_col = get_column(
        targets,
        ["wild_type_sequence", "Sequence", "sequence", "target_sequence"],
        required=False,
    )
    if args.up_bias != 0.0 and sequence_col is None:
        raise ValueError("--up_bias requires a wild_type_sequence/Sequence column")

    strategies = [
        ("eps_argmax_p05", "epsilon_argmax", 0.05),
        ("eps_argmax_p10", "epsilon_argmax", 0.10),
        ("sample", "sample", 1.0),
    ]

    rows = []
    for target_idx, row in enumerate(tqdm(targets.iter_rows(named=True), total=len(targets))):
        title = str(row[title_col]) if title_col else f"target_{target_idx}"
        target_structure = str(row[structure_col])
        reference_sequence = normalize_reference(row[sequence_col]) if sequence_col else None
        src, ct, correspondences, target_bps = build_target_tensors(
            target_structure, args.n_structures, device
        )
        weight, bias = None, None
        if args.up_bias != 0.0 and reference_sequence:
            if len(reference_sequence) != len(target_structure):
                raise ValueError(
                    f"{title}: wild_type_sequence length {len(reference_sequence)} "
                    f"does not match target structure length {len(target_structure)}"
                )
            weight, bias = make_ref_upweighted_params(
                reference_sequence, up_bias=args.up_bias
            )
            weight = weight.to(device)
            bias = bias.to(device)

        target_samples = []
        for strategy_name, mode, p in strategies:
            perm = make_perm(
                args.n_structures,
                len(target_structure),
                args.decoding_order,
                device,
            )
            generated = generate_permuted(
                model,
                src,
                ct,
                correspondences,
                perm=perm,
                mode=mode,
                p=p,
                weight=weight,
                bias=bias,
            )

            predicted_sequences = token_rows_to_strings(generated.detach().cpu())
            predicted_structures, predicted_bps, shape_profiles = score_sequences(
                env, generated, args.score_batch_size
            )

            for sample_idx, sequence in enumerate(predicted_sequences):
                jaccard = jaccard_similarity_base_pairs(
                    predicted_bps[sample_idx], target_bps
                )
                shape_profile = [
                    float(value) for value in np.asarray(shape_profiles[sample_idx])
                ]
                sample_row = {
                    "sequence": sequence,
                    "predicted_structure": predicted_structures[sample_idx],
                    "source": title,
                    "shape_profile": str(shape_profile),
                    "target_structure": target_structure,
                    "jaccard": float(jaccard),
                    "structure_match": predicted_structures[sample_idx]
                    == target_structure,
                    "strategy": strategy_name,
                    "sample_idx": sample_idx,
                    "decoding_order": args.decoding_order,
                    "hamming_distance": maybe_hamming(
                        reference_sequence, sequence
                    ),
                }
                rows.append(sample_row)
                target_samples.append(
                    {
                        **sample_row,
                        "base_pairs": predicted_bps[sample_idx],
                    }
                )

        if target_samples and args.rescue_max_diff >= 0:
            best_sample = max(target_samples, key=lambda sample: sample["jaccard"])
            if best_sample["jaccard"] != 1.0:
                diff_positions = rescue_diff_positions(
                    env, target_structure, best_sample["predicted_structure"]
                )
                if 0 < len(diff_positions) <= args.rescue_max_diff:
                    rescue_sequences = mutate_sequence(
                        best_sample["sequence"], diff_positions
                    )
                    rescue_tokens = sequences_to_tensor(rescue_sequences, device)
                    rescue_structures, rescue_bps, rescue_shapes = score_sequences(
                        env, rescue_tokens, args.score_batch_size
                    )
                    for rescue_idx, sequence in enumerate(rescue_sequences):
                        jaccard = jaccard_similarity_base_pairs(
                            rescue_bps[rescue_idx], target_bps
                        )
                        shape_profile = [
                            float(value)
                            for value in np.asarray(rescue_shapes[rescue_idx])
                        ]
                        rows.append(
                            {
                                "sequence": sequence,
                                "predicted_structure": rescue_structures[rescue_idx],
                                "source": title,
                                "shape_profile": str(shape_profile),
                                "target_structure": target_structure,
                                "jaccard": float(jaccard),
                                "structure_match": rescue_structures[rescue_idx]
                                == target_structure,
                                "strategy": "rescue",
                                "sample_idx": rescue_idx,
                                "decoding_order": args.decoding_order,
                                "hamming_distance": maybe_hamming(
                                    reference_sequence, sequence
                                ),
                            }
                        )

    pl.DataFrame(rows).write_csv(output_path)
    elapsed = time.time() - start_time
    print(f"[generate] wrote {len(rows)} sequences to {output_path}")
    print(f"[generate] elapsed: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
