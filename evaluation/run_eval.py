"""
Single-GPU evaluation of a struct2seq_bidir_rl checkpoint on a target test CSV.

Runs the Struct2SeQ evaluation protocol with three sampling strategies and
N samples per strategy per target. By default, this evaluator uses random
per-sample decoding orders to match order-agnostic training:

  strategy A (epsilon-greedy, p=0.05) : argmax with 5% near-uniform-among-allowed picks
  strategy B (epsilon-greedy, p=0.10) : argmax with 10% near-uniform-among-allowed picks
  strategy C (sample, T=1)            : multinomial draw from softmax(values)

(p=0.05/0.10 + softmax-with-tiny-temperature is exactly what
``Functions.generate_sequence_batched`` does; the T=1 sampler is exactly
``Functions.generate_sequence_batched_sample`` with p=1. Original used
``repeat=128`` per strategy; we default to 32.)

Reward identical to training's test-play:
    env.SS_model(seq).sigmoid() -> _hungarian(theta=0.5, min_helix=1)
    -> DQN_env.get_reward (positional MCC-style match vs target).

Run from the repository root. The environment loads RibonanzaNet weights from
``RIBONANZA_WEIGHTS_DIR`` when set, otherwise from ``../weights``.

Outputs land in evaluation/<UTC-timestamp>__<experiment>__<label>__<data>/:
  summary.json     - run metadata, checkpoint hash, per-strategy + best-of metrics
  per_target.csv   - one row per target with best-of-strategy and best-of-all
  per_sample.csv   - one row per sample (target x strategy x sample_idx), with seq+reward
  run.log          - stdout/stderr capture
  checkpoint.pt    - snapshot of evaluated weights (reproducibility)

Example:
  CUDA_VISIBLE_DEVICES=6 python evaluation/run_eval.py \
      --checkpoint best_policy_network.pt \
      --config     config_brev_8gpu.yaml \
      --test-csv   /path/to/test.csv \
      --experiment struct2seq_bidir_rl_orderagnostic \
      --label      ep4_best
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from arnie.pk_predictors import _hungarian  # noqa: E402

from Dataset import (  # noqa: E402
    convert_dotbracket_to_bp_list,
    tokenize_dot_bracket,
)
from Encoder_Decoder import DotBracketRNATransformer  # noqa: E402
from Env import DQN_env, mask_diagonal  # noqa: E402
from Functions import generate_permuted as _generate_permuted_canonical  # noqa: E402
from run import TrainingConfig, delete_modules  # noqa: E402


STRATEGIES = [
    {"name": "eps_argmax_p05", "mode": "epsilon_argmax", "p": 0.05},
    {"name": "eps_argmax_p10", "mode": "epsilon_argmax", "p": 0.10},
    {"name": "sample_T1",      "mode": "sample",         "p": 1.0 },
]


# --------------------------------------------------------------------------- utils

def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_of_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class Tee:
    def __init__(self, path: Path):
        self.f = open(path, "w", buffering=1)
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def write(self, s):
        self.stdout.write(s)
        self.f.write(s)

    def flush(self):
        self.stdout.flush()
        self.f.flush()

    def close(self):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.f.close()


def build_model(config: TrainingConfig, device: torch.device) -> torch.nn.Module:
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
    model = model.to(device)
    model.eval()
    return model


def get_paired_correspondences_from_bp(bps):
    pc = {}
    for i, j in bps:
        pc[i] = j
        pc[j] = i
    return pc


# --------------------------------------------------------------------------- generator

# Watson-Crick + wobble allowed pairs; index of complement(s) for each base.
# Match logic in Functions.create_base_pair_mask (ACGU -> 0/1/2/3):
#   left=A(0) -> only U(3) allowed
#   left=C(1) -> only G(2)
#   left=G(2) -> C(1) or U(3)
#   left=U(3) -> only A(0)
COMPLEMENT_ALLOWED = {
    0: {3},
    1: {2},
    2: {1, 3},
    3: {0},
}

MAX_REPEAT = [4, 4, 3, 5]  # ACGU — same as create_base_pair_mask


def generate_permuted(
    model: torch.nn.Module,
    src: torch.Tensor,                    # [B, L]
    ct_matrix: torch.Tensor,              # [B, L, L]
    target_correspondence: list,          # list[dict] length B, RNA-pos -> partner
    perm: torch.Tensor,                   # [B, L] long, RNA-pos decoded at each step
    mode: str,                            # "epsilon_argmax" or "sample"
    p: float,                             # epsilon for "epsilon_argmax"; ignored for "sample"
    start_token: int = 4,
) -> torch.Tensor:
    """Thin wrapper delegating to the canonical Functions.generate_permuted.

    Kept here for backward compat with the existing eval script structure.
    The canonical version uses the corrected forward-looking paired-encoding
    convention (matches training; no 1-step lag).
    """
    return _generate_permuted_canonical(
        model, src, ct_matrix, target_correspondence,
        perm=perm, mode=mode, p=p, start_token=start_token,
    )


# Legacy local implementation, kept for reference but unused. The canonical
# version lives in Functions.py.
@torch.no_grad()
def _generate_permuted_legacy(
    model: torch.nn.Module,
    src: torch.Tensor,                    # [B, L]
    ct_matrix: torch.Tensor,              # [B, L, L]
    target_correspondence: list,          # list[dict] length B, RNA-pos -> partner
    perm: torch.Tensor,                   # [B, L] long, RNA-pos decoded at each step
    mode: str,                            # "epsilon_argmax" or "sample"
    p: float,                             # epsilon for "epsilon_argmax"; ignored for "sample"
    start_token: int = 4,
) -> torch.Tensor:
    """LEGACY — uses lagged paired_running convention. Kept for reference only.
    Use the wrapper above (delegates to Functions.generate_permuted) instead."""
    B, L = src.shape
    device = src.device
    model.eval()

    memory = model.encoder(src, ct_matrix)
    if memory.shape[0] == 1 and B != 1:
        memory = memory.expand(B, *memory.shape[1:])

    # paired-encoding lookup at each RNA position (1 = unpaired, 0 = paired)
    paired_at_pos = (src == 0).long()                 # [B, L]
    # paired_encoding in step order: [B, L]
    paired_step = paired_at_pos.gather(1, perm)       # [B, L]

    # decoder input in step order (shifted-right by 1, start token at step 0)
    tgt = torch.full((B, 1), start_token, dtype=torch.long, device=device)
    paired_running = torch.full((B, 1), 0, dtype=torch.long, device=device)
    # paired_running[:, 0] is the paired encoding for the start token slot;
    # at step t>=1 it is paired_step[:, t-1] (encoding for the position whose
    # nucleotide just got fed in). Match training's `paired_encoding_perm` semantics.

    # rna_outputs[b, i] = generated nt at RNA position i, or -1 if not yet decoded
    rna_outputs = torch.full((B, L), -1, dtype=torch.long, device=device)
    A_cnt = np.zeros(B, dtype=np.float64)

    past_key_values = None

    for t in range(L):
        cur_rna_pos = perm[:, t]  # [B]

        tgt_mask = torch.triu(
            torch.full((tgt.size(1), tgt.size(1)), float("-inf"), device=device),
            diagonal=1,
        )

        out = model.decoder(
            tgt,
            paired_running,
            memory,
            tgt_mask=tgt_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        if isinstance(out, tuple):
            out, past_key_values = out
        values = out[:, -1, :]                          # [B, vocab-1] = [B, 4]

        # ---- build per-row mask in RNA-position space ----------------------
        mask = torch.zeros(B, 4, dtype=torch.bool, device=device)

        cur_pos_cpu = cur_rna_pos.cpu().tolist()
        rna_outputs_cpu = rna_outputs.cpu().tolist()
        for b in range(B):
            cp = cur_pos_cpu[b]
            tc = target_correspondence[b]
            if cp in tc:
                partner = tc[cp]
                partner_nt = rna_outputs_cpu[b][partner]
                if partner_nt >= 0:
                    allowed = COMPLEMENT_ALLOWED[partner_nt]
                    for nt in range(4):
                        if nt not in allowed:
                            mask[b, nt] = True

        # repeat constraint: look at last `max_repeat[nt]` STEP-order outputs
        if t > 5:
            # tgt[:, 1:] is the step-ordered emitted nucleotides so far (length t)
            recent = tgt[:, 1:]                         # [B, t]
            # for each base nt, check the trailing window of length max_repeat[nt]
            for nt in range(4):
                w = MAX_REPEAT[nt]
                if t >= w:
                    window = recent[:, -w:]
                    triggered = (window == nt).all(dim=1)
                    new_mask_row = mask.clone()
                    new_mask_row[:, nt] = mask[:, nt] | triggered
                    # if masking this nt would zero out everything, back off:
                    full_block = new_mask_row.sum(dim=1) == 4
                    new_mask_row[full_block, nt] = mask[full_block, nt]
                    mask = new_mask_row

        # >40% A constraint
        a_block = torch.from_numpy(A_cnt > L * 0.4).to(device)
        if a_block.any():
            new_mask = mask.clone()
            new_mask[:, 0] = mask[:, 0] | a_block
            full_block = new_mask.sum(dim=1) == 4
            new_mask[full_block, 0] = mask[full_block, 0]
            mask = new_mask

        masked_values = values.masked_fill(mask, float("-inf"))

        # ---- sampling -------------------------------------------------------
        if mode == "epsilon_argmax":
            # exactly the policy used in Functions.generate_sequence_batched
            probs = F.softmax(masked_values / 10000000.0, dim=-1)
            sampled_tokens = torch.multinomial(probs, 1)
            argmax_tokens = torch.argmax(masked_values, dim=-1, keepdim=True)
            random_sample = torch.rand(B, device=device) < p
            next_tokens = torch.where(
                random_sample.unsqueeze(1), sampled_tokens, argmax_tokens
            )
        elif mode == "sample":
            # exactly Functions.generate_sequence_batched_sample with p=1
            probs = F.softmax(masked_values, dim=-1)
            next_tokens = torch.multinomial(probs, 1)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # ---- bookkeeping ----------------------------------------------------
        # write into RNA-position grid
        rna_outputs.scatter_(1, cur_rna_pos.unsqueeze(1), next_tokens)
        A_cnt = A_cnt + (next_tokens.squeeze(-1) == 0).cpu().float().numpy()

        # extend decoder input (step-order) with the just-emitted token,
        # and the paired-encoding for the *next* step's input slot.
        tgt = torch.cat([tgt, next_tokens], dim=1)
        # paired_running[:, t+1] = paired_step[:, t] (encoding of the position whose nt we just fed in)
        next_paired = paired_step[:, t : t + 1]
        paired_running = torch.cat([paired_running, next_paired], dim=1)

    # rna_outputs is now fully populated: [B, L] in RNA-position order
    return rna_outputs


# --------------------------------------------------------------------------- main

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--test-csv", required=True, type=Path)
    p.add_argument("--experiment", required=True, type=str)
    p.add_argument("--label", required=True, type=str)
    p.add_argument("--data-label", default=None, type=str)
    p.add_argument("--out-dir", default=Path(__file__).resolve().parent, type=Path)
    p.add_argument("--samples-per-strategy", default=32, type=int,
                   help="N samples per sampling strategy per target (default 32)")
    p.add_argument("--targets-per-chunk", default=4, type=int,
                   help="How many targets to batch together per forward pass (each one becomes "
                        "samples_per_strategy entries). Effective batch = this * samples_per_strategy")
    p.add_argument("--decoding-order", choices=["permuted", "l2r"], default="permuted",
                   help="permuted = random RNA-pos decoding order (matches order-agnostic training); "
                        "l2r = left-to-right (sanity check)")
    p.add_argument("--max-len", default=512, type=int)
    p.add_argument("--limit", default=None, type=int)
    p.add_argument("--skip-names-from", default=None, type=Path,
                   help="Path to a per_target.csv (or any CSV with a 'name' column); skip those targets")
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--snapshot-checkpoint", action="store_true", default=True)
    p.add_argument("--no-snapshot-checkpoint", dest="snapshot_checkpoint", action="store_false")
    return p.parse_args()


def build_perm(B: int, L: int, mode: str, device, generator) -> torch.Tensor:
    if mode == "permuted":
        # one independent random perm per batch row
        perms = [torch.randperm(L, device=device, generator=generator) for _ in range(B)]
        return torch.stack(perms, dim=0)
    elif mode == "l2r":
        return torch.arange(L, device=device).unsqueeze(0).expand(B, L).contiguous()
    else:
        raise ValueError(mode)


def main() -> int:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — set CUDA_VISIBLE_DEVICES to a free GPU.")
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cuda_gen = torch.Generator(device=device).manual_seed(args.seed)

    # ------------------------------------------------------------------ run dir
    stamp = utc_stamp()
    data_label = args.data_label or args.test_csv.parent.name
    run_id = f"{stamp}__{args.experiment}__{args.label}__{data_label}"
    run_dir = (args.out_dir / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    tee = Tee(run_dir / "run.log")
    print(f"[eval] run_dir = {run_dir}")
    print(f"[eval] cuda visible: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}, "
          f"using {torch.cuda.get_device_name(0)}")

    # ----------------------------------------------------------------- config
    config = TrainingConfig.from_yaml(str(args.config))

    # --------------------------------------------------------- checkpoint meta
    ckpt_src = args.checkpoint.resolve()
    ckpt_stat = ckpt_src.stat()
    ckpt_sha = sha256_of_file(ckpt_src)
    ckpt_meta = {
        "source_path": str(ckpt_src),
        "size_bytes": ckpt_stat.st_size,
        "mtime_iso": datetime.fromtimestamp(ckpt_stat.st_mtime, tz=timezone.utc).isoformat(),
        "sha256": ckpt_sha,
    }
    print(f"[eval] checkpoint: {ckpt_src} (sha256 {ckpt_sha[:12]}…, mtime {ckpt_meta['mtime_iso']})")
    if args.snapshot_checkpoint:
        snapshot_path = run_dir / "checkpoint.pt"
        shutil.copy2(ckpt_src, snapshot_path)
        ckpt_meta["snapshot_path"] = str(snapshot_path)

    # --------------------------------------------------------------- model
    model = build_model(config, device)
    state = torch.load(ckpt_src, map_location="cpu")
    # Drop legacy decoder-PE weights (LSTM + CausalConv) when loading
    # pre-RPE-redesign checkpoints; new RPE bias tables stay at default init.
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
    if n_dropped:
        print(f"[eval] dropped {n_dropped} legacy decoder-PE keys")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected}")
    allowed_missing = {"decoder.rpe_self.bias.weight", "decoder.rpe_cross.bias.weight"}
    unallowed = [k for k in missing if k not in allowed_missing]
    if unallowed:
        raise RuntimeError(f"Unexpected missing keys: {unallowed}")
    if missing:
        print(f"[eval] missing (newly-init RPE): {missing}")
    model.eval()
    print(f"[eval] model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    # --------------------------------------------------------------- env
    env = DQN_env(use_gpu=True, compile=False)
    print("[eval] DQN_env (RibonanzaNet-SS) loaded")

    # --------------------------------------------------------------- data
    df = pl.read_csv(str(args.test_csv))
    print(f"[eval] loaded {len(df)} rows from {args.test_csv}")
    names = df["name"].to_list() if "name" in df.columns else [f"row{i}" for i in range(len(df))]
    structures = df["structure"].to_list()
    target_seqs = df["sequence"].to_list() if "sequence" in df.columns else [None] * len(df)

    rows = list(zip(range(len(df)), names, structures, target_seqs))
    if args.max_len is not None:
        rows = [r for r in rows if len(r[2]) <= args.max_len]
        print(f"[eval] filtered to {len(rows)} rows with len <= {args.max_len}")
    if args.skip_names_from is not None:
        skip_df = pl.read_csv(str(args.skip_names_from))
        skip_names = set(skip_df["name"].to_list())
        before = len(rows)
        rows = [r for r in rows if r[1] not in skip_names]
        print(f"[eval] skipped {before - len(rows)} rows already in {args.skip_names_from}; "
              f"{len(rows)} remaining")
    if args.limit is not None:
        rows = rows[: args.limit]
        print(f"[eval] capped to {len(rows)} rows")
    if not rows:
        print("[eval] no rows to evaluate; aborting.")
        tee.close()
        return 1

    buckets: dict[int, list] = defaultdict(list)
    for r in rows:
        buckets[len(r[2])].append(r)

    # --------------------------------------------------------------- eval loop
    sample_records: list[dict] = []  # one row per sample
    target_records: list[dict] = []  # one row per target
    n_per_strat = args.samples_per_strategy
    chunk = args.targets_per_chunk
    t0 = time.time()

    # incremental flush — preserves work if the run crashes
    sample_csv_path = run_dir / "per_sample.csv"
    target_csv_path = run_dir / "per_target.csv"
    sample_fields = [
        "row_index", "name", "length", "strategy", "sample_idx",
        "reward", "predicted_sequence", "predicted_structure",
    ]
    target_fields = [
        "row_index", "name", "length", "target_structure", "target_sequence",
        "best_reward_any", "mean_reward_any",
    ] + [k for s in STRATEGIES for k in (f"best_{s['name']}", f"mean_{s['name']}")]
    sample_csv_f = open(sample_csv_path, "w", newline="")
    target_csv_f = open(target_csv_path, "w", newline="")
    sample_writer = csv.DictWriter(sample_csv_f, fieldnames=sample_fields); sample_writer.writeheader()
    target_writer = csv.DictWriter(target_csv_f, fieldnames=target_fields); target_writer.writeheader()
    sample_csv_f.flush(); target_csv_f.flush()

    for L, group in sorted(buckets.items()):
        for c_start in tqdm(range(0, len(group), chunk),
                            desc=f"len={L} (n={len(group)})", leave=False):
            sub = group[c_start : c_start + chunk]  # list of (idx, name, struct, tgt_seq)
            n_targets = len(sub)

            # build per-target tensors
            src_per_target = []
            ct_per_target = []
            tc_per_target = []
            for _, _, struct, _ in sub:
                bps = convert_dotbracket_to_bp_list(struct, allow_pseudoknots=True)
                ct = np.zeros((L, L), dtype=np.float32)
                for i, j in bps:
                    ct[i, j] = ct[j, i] = 1.0
                for i in range(L):
                    ct[i, i] = 1.0
                src_per_target.append(torch.tensor(tokenize_dot_bracket(struct), dtype=torch.long))
                ct_per_target.append(torch.from_numpy(ct))
                tc_per_target.append(get_paired_correspondences_from_bp(bps))

            # tile each target n_per_strat times along batch
            src_tile = torch.stack(src_per_target).repeat_interleave(n_per_strat, dim=0).to(device)
            ct_tile = torch.stack(ct_per_target).repeat_interleave(n_per_strat, dim=0).to(device)
            tc_tile: list = []
            for tc in tc_per_target:
                tc_tile.extend([tc] * n_per_strat)

            B_eff = src_tile.shape[0]
            assert B_eff == n_targets * n_per_strat

            # --- run each sampling strategy -----------------------------
            for strat in STRATEGIES:
                perm = build_perm(B_eff, L, args.decoding_order, device, cuda_gen)

                # bf16 autocast for the autoregressive decode + folder.
                # Both the policy network and env.SS_model are bf16-safe;
                # this is the same precision used during training.
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    predicted = generate_permuted(
                        model,
                        src_tile,
                        ct_tile,
                        tc_tile,
                        perm=perm,
                        mode=strat["mode"],
                        p=strat["p"],
                    )  # [B_eff, L] in RNA-position order

                    with torch.no_grad():
                        # cast bf16 -> fp32 before .numpy() (numpy doesn't support bf16)
                        bpps = env.SS_model(predicted).sigmoid().detach().float().cpu().numpy()
                preds_cpu = predicted.cpu()

                for k in range(B_eff):
                    target_idx_in_sub = k // n_per_strat
                    sample_idx = k % n_per_strat
                    orig_idx, name, struct, tgt_seq = sub[target_idx_in_sub]

                    pred_struct, pred_bps = _hungarian(
                        mask_diagonal(bpps[k]), theta=0.5, min_len_helix=1
                    )
                    positional_reward = env.get_reward(pred_bps, struct)
                    reward = float(np.mean(positional_reward))
                    pred_seq = "".join(env.index_to_nt[int(t)] for t in preds_cpu[k].tolist())

                    rec = {
                        "row_index": orig_idx,
                        "name": name,
                        "length": L,
                        "strategy": strat["name"],
                        "sample_idx": sample_idx,
                        "reward": reward,
                        "predicted_sequence": pred_seq,
                        "predicted_structure": pred_struct,
                    }
                    sample_records.append(rec)
                    sample_writer.writerow(rec)

            # aggregate per target for this chunk
            for ti in range(n_targets):
                orig_idx, name, struct, tgt_seq = sub[ti]
                # rewards from sample_records that match this target
                # (last 3*n_per_strat appended belong to this chunk)
                base_offset = len(sample_records) - n_targets * n_per_strat * len(STRATEGIES) \
                              + ti * n_per_strat  # start of strat 0 for this target
                # gather across strategies
                per_strat_rewards: dict[str, list] = {s["name"]: [] for s in STRATEGIES}
                for s_idx, strat in enumerate(STRATEGIES):
                    strat_block_start = (
                        len(sample_records) - n_targets * n_per_strat * len(STRATEGIES)
                        + s_idx * n_targets * n_per_strat
                        + ti * n_per_strat
                    )
                    block = sample_records[strat_block_start : strat_block_start + n_per_strat]
                    per_strat_rewards[strat["name"]] = [r["reward"] for r in block]

                all_rewards = [r for v in per_strat_rewards.values() for r in v]
                rec = {
                    "row_index": orig_idx,
                    "name": name,
                    "length": L,
                    "target_structure": struct,
                    "target_sequence": tgt_seq if tgt_seq is not None else "",
                    "best_reward_any":  float(max(all_rewards)),
                    "mean_reward_any":  float(np.mean(all_rewards)),
                }
                for s in STRATEGIES:
                    rs = per_strat_rewards[s["name"]]
                    rec[f"best_{s['name']}"] = float(max(rs))
                    rec[f"mean_{s['name']}"] = float(np.mean(rs))
                target_records.append(rec)
                target_writer.writerow(rec)
            sample_csv_f.flush()
            target_csv_f.flush()

    elapsed = time.time() - t0
    print(f"[eval] generation + scoring done in {elapsed:.1f}s "
          f"({len(sample_records)} samples / {len(target_records)} targets)")

    # --------------------------------------------------------------- close incremental writers
    sample_csv_f.close()
    target_csv_f.close()

    # corpus-level metrics
    corpus = {}
    for s in STRATEGIES:
        bests = np.array([r[f"best_{s['name']}"] for r in target_records])
        means = np.array([r[f"mean_{s['name']}"] for r in target_records])
        corpus[s["name"]] = {
            "mean_of_per_target_means": float(means.mean()),
            "mean_of_per_target_bests": float(bests.mean()),
            "median_of_per_target_bests": float(np.median(bests)),
        }
    best_any = np.array([r["best_reward_any"] for r in target_records])
    corpus["best_of_all_strategies"] = {
        "mean_of_per_target_bests": float(best_any.mean()),
        "median_of_per_target_bests": float(np.median(best_any)),
    }

    summary = {
        "run_id": run_id,
        "utc_started": stamp,
        "experiment": args.experiment,
        "label": args.label,
        "data_label": data_label,
        "test_csv": str(args.test_csv.resolve()),
        "test_csv_rows_total": len(df),
        "n_targets_evaluated": len(target_records),
        "n_samples_total": len(sample_records),
        "samples_per_strategy": n_per_strat,
        "max_len_filter": args.max_len,
        "limit": args.limit,
        "decoding_order": args.decoding_order,
        "targets_per_chunk": chunk,
        "sampling": {
            "strategies": STRATEGIES,
            "matches_original_struct2seq_generate_py": True,
            "original_repeat": 128,
            "this_run_repeat": n_per_strat,
            "diff_from_original": (
                "decoding order is per-sample random permutation (--order-agnostic "
                "training) instead of left-to-right; samples_per_strategy reduced "
                "from 128 -> {n}.".format(n=n_per_strat)
            ),
            "reward_pipeline": (
                "env.SS_model(seq).sigmoid -> _hungarian(theta=0.5, min_len_helix=1) "
                "-> DQN_env.get_reward (per-position MCC-style match)."
            ),
        },
        "model": {
            "config_path": str(args.config.resolve()),
            "architecture": "DotBracketRNATransformer",
            "embed_size": config.embed_size,
            "nhead": config.nhead,
            "num_encoder_layers": config.num_encoder_layers,
            "num_decoder_layers": config.num_decoder_layers,
            "dim_feedforward": config.dim_feedforward,
            "dropout": config.dropout,
            "trained_sequence_length": getattr(config, "sequence_length", None),
        },
        "checkpoint": ckpt_meta,
        "metrics": corpus,
        "timing": {
            "elapsed_seconds": elapsed,
            "samples_per_second": len(sample_records) / max(elapsed, 1e-9),
        },
        "gpu": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_name": torch.cuda.get_device_name(0),
        },
        "seed": args.seed,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[eval] === RESULTS (eval reward = per-position MCC-style match) ===")
    print(f"  experiment       : {args.experiment}")
    print(f"  label            : {args.label}")
    print(f"  data_label       : {data_label}")
    print(f"  decoding_order   : {args.decoding_order}")
    print(f"  n_targets        : {len(target_records)}")
    print(f"  samples/target   : {n_per_strat * len(STRATEGIES)}  ({n_per_strat} x {len(STRATEGIES)} strategies)")
    for name, m in corpus.items():
        if "mean_of_per_target_bests" in m:
            print(f"  {name:30s}  mean(best per target) = {m['mean_of_per_target_bests']:.6f}"
                  f"   median(best) = {m['median_of_per_target_bests']:.6f}")
    print(f"  outputs          : {run_dir}")

    # ---- run knitnet-format metric computation (MCC, Jaccard, F1, NSR ...) -
    print("\n[eval] running compute_metrics.py for knitnet-format JSON...")
    import subprocess
    rc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parent / "compute_metrics.py"),
         "--run-dir", str(run_dir)],
        check=False,
    ).returncode
    if rc != 0:
        print(f"[eval] compute_metrics.py exited with code {rc} — "
              f"per_sample.csv is intact, you can rerun manually.")
    tee.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
