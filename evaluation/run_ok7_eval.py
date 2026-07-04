"""OpenKnot 7 (100mer) evaluation suite.

Multi-GPU eval via accelerate. Generates K samples per puzzle for the 20
OK7 100mer puzzles supplied with ``--targets-csv``.
Scores each generated sequence with:
  - RibonanzaNet-SS  -> predicted dot-bracket -> Jaccard vs target
  - RibonanzaNet     -> predicted SHAPE -> OK score
    (Eterna Classic + Crossed Pair Quality, from
     ``OpenKnotScorePipeline/src/openknotscore/pipeline/scoring.py``).

Inference modes:
  - l2r_legacy   : original Functions.generate_sequence_batched (use for
                   --model original AR baseline).
  - identity     : generate_permuted with perm = arange(L) (= L->R).
  - random       : generate_permuted with perm = randperm(L) per sample.
  - paired_first : decode all paired-in-target positions first, then
                   unpaired, with random tiebreaks within each block.
  - inpaint      : random perm, with floor(K_inpaint*L) WT positions
                   pre-filled from the puzzle's `Sequence` column.

Per-rank checkpointing: writes append-mode CSV + batches_done.txt after
every batch; resume on restart fast-skips completed batches.

Example:
  export OPENKNOTSCORE_SRC=/path/to/OpenKnotScorePipeline/src
  CUDA_VISIBLE_DEVICES=0,1,3,4 accelerate launch --num_processes 4 \\
      evaluation/run_ok7_eval.py \\
      --targets-csv /path/to/Round3_targets.csv \\
      --model bidir --inference-mode random --k-samples 1000 \\
      --out-dir results/ok7_eval/bidir_random/
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_POLICY_CHECKPOINT = PROJECT_ROOT / "weights" / "order_agnostic_policy.pt"

OKS_SRC = os.environ.get("OPENKNOTSCORE_SRC")
if OKS_SRC:
    oks_path = Path(OKS_SRC).expanduser().resolve()
    if str(oks_path) not in sys.path:
        sys.path.insert(0, str(oks_path))

from Dataset import tokenize_dot_bracket, detokenize_dot_bracket  # noqa: E402
from Env import DQN_env, mask_diagonal  # noqa: E402
from Functions import (  # noqa: E402
    convert_dotbracket_to_bp_list,
    generate_permuted,
    generate_sequence_batched,
)
from arnie.pk_predictors import _hungarian  # noqa: E402
try:
    from openknotscore.pipeline.scoring import (  # noqa: E402
        calculateCrossedPairQualityScore,
        calculateEternaClassicScore,
    )
except ImportError as exc:
    raise ImportError(
        "OpenKnotScorePipeline is required for OK7 scoring. Install it or set "
        "OPENKNOTSCORE_SRC=/path/to/OpenKnotScorePipeline/src."
    ) from exc
from evaluation.run_eval import build_model  # noqa: E402
from run import TrainingConfig  # noqa: E402


# --------------------------------------------------------------------------- constants

NT_TO_IDX = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}  # T treated as U
IDX_TO_NT = {0: "A", 1: "C", 2: "G", 3: "U"}


# --------------------------------------------------------------------------- CLI


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model", choices=["bidir"], default="bidir",
        help=("Only 'bidir' (= weights/order_agnostic_policy.pt with new RPE arch) is "
              "supported here. The original Struct2SeQ.pt must be evaluated "
              "from the Struct2SeQ_training/ codebase since loading its "
              "LSTM-PE weights into the new arch would zero out positional "
              "info and not be a faithful baseline. Compare against published "
              "per-puzzle baseline numbers directly."),
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_POLICY_CHECKPOINT),
        help="Policy checkpoint path (default: weights/order_agnostic_policy.pt).",
    )
    p.add_argument(
        "--inference-mode",
        choices=["l2r_legacy", "identity", "random", "paired_first", "inpaint"],
        required=True,
    )
    p.add_argument("--inpaint-k", type=float, default=0.0,
                   help="Fraction of WT positions to fix (only for scatter inpaint).")
    p.add_argument(
        "--motif-mode",
        choices=["scatter", "structural", "structural_redesign"],
        default="scatter",
        help=("scatter = uniform-random K-of-WT (legacy); "
              "structural = fix one structural motif to WT and redesign "
              "the surrounding scaffold; structural_redesign = keep the "
              "scaffold fixed and redesign only the motif. Only used when "
              "--inference-mode is inpaint."),
    )
    p.add_argument(
        "--max-motif-fraction", type=float, default=0.5,
        help=("Skip motifs covering more than this fraction of the puzzle. "
              "Avoids degenerate 'fix 89%% of structure' cases."),
    )
    p.add_argument(
        "--sampling-mode",
        choices=["argmax", "qsoftmax", "epsilon"],
        default="argmax",
        help=("argmax (default): pure greedy argmax, no token-level "
              "sampling; qsoftmax: multinomial sampling from softmax "
              "over Q-values (matches the published softmax strategy); epsilon: "
              "argmax with p=0.1 of uniform-among-allowed-bases sampling "
              "(matches the published epsilon strategy). Applies to all "
              "inference modes that use generate_permuted."),
    )
    p.add_argument("--sampling-p", type=float, default=0.1,
                   help="epsilon probability for --sampling-mode=epsilon.")
    p.add_argument(
        "--decode-order",
        choices=["random", "fixed_first", "identity"],
        default="random",
        help=("random = uniform-random permutation (default, matches "
              "training); fixed_first = decode all fixed scaffold "
              "positions first then free positions (random within "
              "each); identity = strict L→R order. With identity + "
              "fixed=motif this reproduces AR teacher-forced motif "
              "preservation on the trained checkpoint (same checkpoint as "
              "random-perm, only decoding paradigm differs)."),
    )
    p.add_argument("--targets-csv", type=str, required=True,
                   help="CSV containing OpenKnot target structures.")
    p.add_argument("--k-samples", type=int, default=1000,
                   help="Samples per puzzle.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--config", type=str, default="config_brev_8gpu.yaml",
                   help="YAML matching the trained checkpoint's architecture.")
    p.add_argument("--limit-puzzles", type=int, default=None,
                   help="Smoke test: cap to first N puzzles.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# --------------------------------------------------------------------------- data


@dataclass
class Puzzle:
    idx: int
    title: str
    puzzle_id: str
    target_db: str          # dot-bracket
    wt_seq: str             # WT/starter ACGU
    length: int


def load_puzzles(csv_path: str, limit: Optional[int] = None) -> List[Puzzle]:
    df = pd.read_csv(csv_path)
    puzzles = []
    for i, row in df.iterrows():
        if limit is not None and i >= limit:
            break
        title = row["Title"]
        target_db = row["Dot-bracket"]
        wt_seq = row["Sequence"]
        length = int(row["Length"])
        if len(target_db) != length or len(wt_seq) != length:
            # Some 240mer puzzles have a 1-position discrepancy in the
            # CSV's declared Length column. Trust db/wt instead.
            actual = min(len(target_db), len(wt_seq))
            target_db = target_db[:actual]
            wt_seq = wt_seq[:actual]
            length = actual
        puzzles.append(
            Puzzle(
                idx=i,
                title=title,
                puzzle_id=str(row["puzzleID"]),
                target_db=target_db,
                wt_seq=wt_seq,
                length=length,
            )
        )
    return puzzles


class PuzzleReplicaDataset(Dataset):
    """Each puzzle replicated K times. Yields per-sample dict with metadata."""

    def __init__(self, puzzles: List[Puzzle], k_samples: int):
        self.puzzles = puzzles
        self.k_samples = k_samples
        self._total = len(puzzles) * k_samples

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int):
        puzzle_idx, sample_idx = divmod(idx, self.k_samples)
        return {"puzzle_idx": puzzle_idx, "sample_idx": sample_idx}


def collate(batch, puzzles: List[Puzzle], device_dummy_for_shape=None):
    """Stack puzzle metadata + tokenized targets into a batch.

    All puzzles in OK7 are 100-nt; no padding needed within a batch.
    """
    puzzle_idxs = [b["puzzle_idx"] for b in batch]
    sample_idxs = [b["sample_idx"] for b in batch]
    Ls = [puzzles[pi].length for pi in puzzle_idxs]
    Lmax = max(Ls)
    B = len(batch)
    src = torch.zeros((B, Lmax), dtype=torch.long)
    # ct_matrix starts as identity (matching training: Dataset.py:78 uses
    # np.eye); GraphConv normalises by row-sum, so every row needs >=1 nonzero.
    ct_matrix = torch.zeros((B, Lmax, Lmax), dtype=torch.float)
    for b in range(B):
        ct_matrix[b].fill_diagonal_(1.0)
    target_correspondence = []
    wt_indices = torch.full((B, Lmax), -1, dtype=torch.long)
    for b, (pi, _) in enumerate(zip(puzzle_idxs, sample_idxs)):
        pz = puzzles[pi]
        L = pz.length
        src[b, :L] = torch.tensor(tokenize_dot_bracket(pz.target_db), dtype=torch.long)
        bps = convert_dotbracket_to_bp_list(pz.target_db, allow_pseudoknots=True)
        for i, j in bps:
            ct_matrix[b, i, j] = 1.0
            ct_matrix[b, j, i] = 1.0
        tc = {}
        for i, j in bps:
            tc[i] = j
            tc[j] = i
        target_correspondence.append(tc)
        for i, nt in enumerate(pz.wt_seq):
            wt_indices[b, i] = NT_TO_IDX[nt]
    return {
        "puzzle_idxs": torch.tensor(puzzle_idxs, dtype=torch.long),
        "sample_idxs": torch.tensor(sample_idxs, dtype=torch.long),
        "lengths": torch.tensor(Ls, dtype=torch.long),
        "src": src,
        "ct_matrix": ct_matrix,
        "target_correspondence": target_correspondence,
        "wt_indices": wt_indices,
    }


# --------------------------------------------------------------------------- inference


def make_perm(B: int, L: int, mode: str, src: torch.Tensor, device, rng: np.random.Generator):
    """Return a [B, L] permutation tensor for the chosen mode."""
    if mode == "identity":
        return torch.arange(L, device=device).unsqueeze(0).expand(B, L).contiguous()
    if mode == "random" or mode == "inpaint":
        perms = []
        for _ in range(B):
            p = rng.permutation(L)
            perms.append(torch.from_numpy(p).long())
        return torch.stack(perms).to(device)
    if mode == "fixed_first":
        # Caller must use make_perm_fixed_first instead — this branch is only
        # reachable if someone passes "fixed_first" without the fixed mask.
        raise ValueError(
            "make_perm(mode='fixed_first', ...) requires the fixed mask; "
            "call make_perm_fixed_first(fixed, rng, device) instead."
        )
    if mode == "paired_first":
        # paired-in-target indicator: src token == 0 means '.' (unpaired);
        # any other db token (open/close brackets) is paired. Decode paired
        # positions first, unpaired second, with random tiebreak within each
        # block.
        perms = []
        for b in range(B):
            paired = (src[b] != 0).cpu().numpy().astype(np.int64)  # 1 if paired
            jitter = rng.random(L)
            # primary key: not-paired-flag (0 first); secondary: jitter
            keys = -paired + 1e-3 * jitter  # paired -> negative, unpaired -> positive
            order = np.argsort(keys, kind="stable")
            perms.append(torch.from_numpy(order).long())
        return torch.stack(perms).to(device)
    raise ValueError(f"unknown perm mode: {mode}")


def make_perm_fixed_first(
    fixed: torch.Tensor, rng: np.random.Generator, device,
) -> torch.Tensor:
    """Return [B, L] perm where each row puts fixed positions (fixed[b,i] >= 0)
    first in random order, then the free positions (fixed[b,i] == -1) in
    random order. Used so the decoder's KV cache contains the entire
    fixed scaffold before any free position is designed.
    """
    B, L = fixed.shape
    fixed_cpu = fixed.detach().cpu().numpy()
    perms = []
    for b in range(B):
        is_fixed = fixed_cpu[b] >= 0
        fixed_idx = np.where(is_fixed)[0]
        free_idx = np.where(~is_fixed)[0]
        rng.shuffle(fixed_idx)
        rng.shuffle(free_idx)
        order = np.concatenate([fixed_idx, free_idx])
        perms.append(torch.from_numpy(order).long())
    return torch.stack(perms).to(device)


def make_fixed(
    B: int, L: int, wt_indices: torch.Tensor, k_inpaint: float, rng: np.random.Generator,
    device,
) -> Optional[torch.Tensor]:
    """Build [B, L] tensor with -1 for free positions, WT nucleotide for fixed
    (random scatter mode)."""
    if k_inpaint <= 0.0:
        return None
    n_fixed = int(np.floor(k_inpaint * L))
    if n_fixed <= 0:
        return None
    fixed = torch.full((B, L), -1, dtype=torch.long, device=device)
    for b in range(B):
        idx = rng.choice(L, size=n_fixed, replace=False)
        for i in idx:
            fixed[b, i] = int(wt_indices[b, i].item())
    return fixed


def make_fixed_motif(
    B: int, L: int, wt_indices: torch.Tensor,
    puzzle_idxs_in_batch: List[int], lengths_in_batch: List[int],
    motif_inventories: dict, rng: np.random.Generator, device,
    invert: bool = False,
):
    """Pick one motif at random from each row's puzzle inventory and build
    a [B, L] fixed tensor. Returns (fixed_tensor, list_of_(kind, size)).

    motif_inventories: dict[puzzle_idx] -> list of motifs
                        each motif: {"kind", "positions", "size"}

    invert=False (Framing A, motif-preservation): fix the motif positions
        to WT, regenerate everything else.
    invert=True (Framing B, motif-redesign): fix everything EXCEPT the
        motif to WT, regenerate just the motif.
    """
    fixed = torch.full((B, L), -1, dtype=torch.long, device=device)
    motif_meta = []
    for b in range(B):
        pi = puzzle_idxs_in_batch[b]
        Lb = lengths_in_batch[b]
        motifs = motif_inventories.get(pi, [])
        if not motifs:
            motif_meta.append(("none", 0))
            continue
        # pick uniformly at random
        m = motifs[int(rng.integers(0, len(motifs)))]
        motif_pos_set = {int(p) for p in m["positions"] if 0 <= int(p) < Lb}
        if invert:
            # fix everything except the motif positions
            for pos in range(Lb):
                if pos not in motif_pos_set:
                    fixed[b, pos] = int(wt_indices[b, pos].item())
        else:
            # fix the motif positions
            for pos in motif_pos_set:
                fixed[b, pos] = int(wt_indices[b, pos].item())
        motif_meta.append((m["kind"], m["size"]))
    return fixed, motif_meta


# --------------------------------------------------------------------------- scoring


def jaccard_bp(predicted_bps: List[tuple], target_bps_set) -> float:
    pred_set = {(min(i, j), max(i, j)) for i, j in predicted_bps}
    if not pred_set and not target_bps_set:
        return 1.0
    inter = pred_set & target_bps_set
    union = pred_set | target_bps_set
    return len(inter) / len(union) if union else 1.0


def compute_ok_score(predicted_db: str, shape: List[float], L: int) -> tuple:
    """Returns (eterna, cpq_quality, ok_score) — all 0-100 floats."""
    eterna = calculateEternaClassicScore(predicted_db, shape, 0, L - 1, filter_singlets=True)
    cpq = calculateCrossedPairQualityScore(predicted_db, shape, 0, L - 1, filter_singlets=True)
    if eterna is None:
        eterna = float("nan")
    if cpq is None or not isinstance(cpq, list) or len(cpq) < 2:
        cpq_quality = float("nan")
    else:
        cpq_quality = float(cpq[1])
    if np.isnan(eterna) or np.isnan(cpq_quality):
        ok = float("nan")
    else:
        ok = 0.5 * (eterna + cpq_quality)
    return float(eterna), float(cpq_quality), float(ok)


# --------------------------------------------------------------------------- main


def main():
    args = parse_args()
    accelerator = Accelerator()
    device = accelerator.device
    rng = np.random.default_rng(args.seed + accelerator.process_index)

    # Per-rank dirs & resume state
    out_dir = Path(args.out_dir)
    rank_dir = out_dir / f"rank{accelerator.process_index}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    samples_csv = rank_dir / "samples.csv"
    bdone_path = rank_dir / "batches_done.txt"
    if bdone_path.exists():
        with open(bdone_path) as f:
            batches_done = int(f.read().strip())
        if accelerator.is_main_process:
            print(f"[resume] rank{accelerator.process_index}: skipping first "
                  f"{batches_done} batches.")
    else:
        batches_done = 0

    csv_header = [
        "puzzle_idx", "puzzle_id", "title", "sample_idx",
        "generated_sequence", "predicted_structure",
        "jaccard_vs_target", "eterna_score", "cpq_score", "ok_score",
        "motif_kind", "motif_size_nt",
    ]
    new_csv = not samples_csv.exists()
    csv_f = open(samples_csv, "a", newline="")
    csv_w = csv.writer(csv_f)
    if new_csv:
        csv_w.writerow(csv_header)
        csv_f.flush()

    # Puzzles
    puzzles = load_puzzles(args.targets_csv, limit=args.limit_puzzles)
    if accelerator.is_main_process:
        print(f"Loaded {len(puzzles)} puzzles. K={args.k_samples}/puzzle. "
              f"Total samples: {len(puzzles) * args.k_samples}.")

    target_bps_sets = [
        {(min(i, j), max(i, j)) for i, j in
         convert_dotbracket_to_bp_list(pz.target_db, allow_pseudoknots=True)}
        for pz in puzzles
    ]

    # Build motif inventory for structural in-painting modes (both
    # motif-preservation and motif-redesign use the same inventory).
    motif_inventories: dict = {}
    if args.motif_mode in ("structural", "structural_redesign"):
        from evaluation.motif_extraction import extract_motifs
        for pz in puzzles:
            ms = extract_motifs(pz.target_db)
            ms_kept = [
                {
                    "kind": m.kind,
                    "positions": sorted(m.positions),
                    "size": m.size,
                }
                for m in ms
                if m.size <= args.max_motif_fraction * pz.length
            ]
            motif_inventories[pz.idx] = ms_kept
            if accelerator.is_main_process:
                print(
                    f"  puzzle {pz.idx} {pz.title[:30]:<30} L={pz.length}: "
                    f"{len(ms_kept)}/{len(ms)} motifs kept "
                    f"(<= {args.max_motif_fraction:.0%} of L)"
                )

    # Dataset/loader
    dataset = PuzzleReplicaDataset(puzzles, args.k_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=lambda b: collate(b, puzzles),
    )
    dataloader = accelerator.prepare(dataloader)

    # Build model + env
    config = TrainingConfig.from_yaml(args.config)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path

    model = build_model(config, device)
    if accelerator.is_main_process:
        print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")

    def _clean_key(k):
        for prefix in ("_orig_mod.", "module."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        return k
    state = {_clean_key(k): v for k, v in state.items()}
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
    n_dropped = sum(1 for k in state if k in DROP_OLD_DECODER_PE)
    state = {k: v for k, v in state.items() if k not in DROP_OLD_DECODER_PE}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if accelerator.is_main_process:
        if n_dropped:
            print(f"  Dropped {n_dropped} legacy decoder-PE keys.")
        if unexpected:
            print(f"  WARNING unexpected keys: {unexpected}")
        if missing:
            print(f"  Missing keys (newly-init): {missing}")
    model.eval()
    model = model.to(device)

    env = DQN_env(use_gpu=True, compile=False)

    # Per-batch loop
    n_per_rank_batches = 0
    t0 = time.time()
    for batch_num, batch in enumerate(tqdm(
        dataloader, desc=f"rank{accelerator.process_index}",
        disable=not accelerator.is_main_process,
    )):
        if batch_num < batches_done:
            continue
        n_per_rank_batches += 1

        src = batch["src"].to(device)
        ct_matrix = batch["ct_matrix"].to(device)
        target_correspondence = batch["target_correspondence"]
        lengths = batch["lengths"]
        wt_indices = batch["wt_indices"].to(device)
        puzzle_idxs = batch["puzzle_idxs"].tolist()
        sample_idxs = batch["sample_idxs"].tolist()
        B, L = src.shape

        # Generate sequences (fp32 — bf16 autocast can produce NaN values that
        # collapse the model to all-A on tight pseudoknot puzzles).
        motif_meta = [("none", 0)] * B
        with torch.no_grad():
            if args.inference_mode == "l2r_legacy":
                seqs = generate_sequence_batched(
                    model, src, target_correspondence, ct_matrix, p=0.0,
                )
            else:
                if args.inference_mode == "inpaint":
                    # Build the fixed mask first; perm depends on it when
                    # decode_order=fixed_first.
                    if args.motif_mode in ("structural", "structural_redesign"):
                        fixed, motif_meta = make_fixed_motif(
                            B, L, wt_indices,
                            puzzle_idxs, [int(x.item()) for x in lengths],
                            motif_inventories, rng, device,
                            invert=(args.motif_mode == "structural_redesign"),
                        )
                    else:
                        fixed = make_fixed(B, L, wt_indices, args.inpaint_k, rng, device)
                    if args.decode_order == "fixed_first" and fixed is not None:
                        perm = make_perm_fixed_first(fixed, rng, device)
                    elif args.decode_order == "identity":
                        perm = torch.arange(L, device=device).unsqueeze(0).expand(B, L).contiguous()
                    else:
                        perm = make_perm(B, L, "inpaint", src, device, rng)
                else:
                    perm = make_perm(B, L, args.inference_mode, src, device, rng)
                    fixed = None
                if args.sampling_mode == "argmax":
                    _gen_mode, _gen_p = "epsilon_argmax", 0.0
                elif args.sampling_mode == "qsoftmax":
                    _gen_mode, _gen_p = "sample", 1.0
                elif args.sampling_mode == "epsilon":
                    _gen_mode, _gen_p = "epsilon_argmax", float(args.sampling_p)
                else:
                    raise ValueError(args.sampling_mode)
                seqs = generate_permuted(
                    model, src, ct_matrix, target_correspondence,
                    perm=perm, mode=_gen_mode, p=_gen_p,
                    fixed=fixed,
                )

            # Score: predicted structure via SS_model
            bpps = env.SS_model(seqs).sigmoid().detach().cpu().numpy()
            # Score: predicted SHAPE via reactivity_model
            shape_arr = env.reactivity_model(seqs)[:, :, 0].detach().cpu().float().numpy()

        # Per-sample post-processing on CPU
        for b in range(B):
            L_b = int(lengths[b].item())
            seq_idx = seqs[b, :L_b].cpu().tolist()
            seq_str = "".join(IDX_TO_NT.get(int(t), "N") for t in seq_idx)
            bpp_b = bpps[b, :L_b, :L_b]
            pred_db, pred_bps = _hungarian(
                mask_diagonal(bpp_b), theta=0.5, min_len_helix=1,
            )
            jacc = jaccard_bp(pred_bps, target_bps_sets[puzzle_idxs[b]])
            shape_b = shape_arr[b, :L_b].tolist()
            eterna, cpq_q, ok = compute_ok_score(pred_db, shape_b, L_b)

            pz = puzzles[puzzle_idxs[b]]
            mk, msz = motif_meta[b]
            csv_w.writerow([
                puzzle_idxs[b], pz.puzzle_id, pz.title, sample_idxs[b],
                seq_str, pred_db,
                f"{jacc:.6f}", f"{eterna:.4f}", f"{cpq_q:.4f}", f"{ok:.4f}",
                mk, msz,
            ])
        csv_f.flush()
        os.fsync(csv_f.fileno())

        # Update batches_done
        with open(bdone_path, "w") as f:
            f.write(str(batch_num + 1))

    csv_f.close()
    elapsed = time.time() - t0
    if accelerator.is_main_process:
        print(f"rank0 done in {elapsed:.1f}s ({n_per_rank_batches} batches "
              f"this run).")

    # Wait for all ranks to finish, then rank-0 merges + computes summary
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        merge_and_summarize(out_dir, accelerator.num_processes, puzzles, args)


def merge_and_summarize(out_dir: Path, num_ranks: int, puzzles: List[Puzzle], args):
    """Concatenate per-rank CSVs and write summary.csv."""
    parts = []
    for r in range(num_ranks):
        p = out_dir / f"rank{r}" / "samples.csv"
        if p.exists():
            parts.append(pd.read_csv(p))
    if not parts:
        print("No per-rank CSVs found; skipping summary.")
        return
    all_df = pd.concat(parts, ignore_index=True)
    all_df.to_csv(out_dir / "samples.csv", index=False)

    rows = []
    if args.inference_mode == "inpaint" and args.motif_mode == "structural":
        cfg_tag = f"{args.model}_motifs_structural"
    elif args.inference_mode == "inpaint" and args.motif_mode == "structural_redesign":
        cfg_tag = f"{args.model}_motifs_redesign"
    elif args.inference_mode == "inpaint":
        cfg_tag = f"{args.model}_inpaint_K{args.inpaint_k:.2f}"
    else:
        cfg_tag = f"{args.model}_{args.inference_mode}"
    if args.inference_mode == "inpaint" and args.decode_order == "fixed_first":
        cfg_tag = f"{cfg_tag}_fixedfirst"
    if args.inference_mode == "inpaint" and args.decode_order == "identity":
        cfg_tag = f"{cfg_tag}_LR"
    if args.sampling_mode != "argmax":
        cfg_tag = f"{cfg_tag}_{args.sampling_mode}"
    for pz in puzzles:
        sub = all_df[all_df["puzzle_idx"] == pz.idx]
        if len(sub) == 0:
            continue
        rows.append({
            "config_tag": cfg_tag,
            "puzzle_idx": pz.idx,
            "puzzle_id": pz.puzzle_id,
            "title": pz.title,
            "n_samples": len(sub),
            "n_perfect_jaccard": int((sub["jaccard_vs_target"] == 1.0).sum()),
            "p80_ok_score": float(sub["ok_score"].dropna().quantile(0.80))
                            if sub["ok_score"].dropna().size else float("nan"),
            "mean_jaccard": float(sub["jaccard_vs_target"].mean()),
            "mean_ok_score": float(sub["ok_score"].dropna().mean())
                             if sub["ok_score"].dropna().size else float("nan"),
        })
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    print(f"Wrote {out_dir/'summary.csv'} with {len(summary_df)} rows.")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
