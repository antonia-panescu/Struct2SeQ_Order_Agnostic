import argparse
import csv
import os
import pickle
import time
import uuid
from dataclasses import asdict, dataclass
from glob import glob
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from arnie.pk_predictors import _hungarian
from Dataset import *
from Encoder_Decoder import *
from Env import *
from Functions import *
from pytorch_ranger import Ranger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm


# Configuration Class
@dataclass
class TrainingConfig:
    # Model Architecture
    db_vocab_size: int = 100
    rna_vocab_size: int = 5
    embed_size: int = 384
    nhead: int = 16
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: Optional[int] = None
    dropout: float = 0.1

    # Training Parameters
    n_episodes: int = 15
    epochs_per_episode: int = 7
    initial_p: float = 0.5
    final_p: float = 0.0
    train_batch_size: int = 8
    max_train_batch_size: int = 64
    inference_batch_size: int = 120
    # K random permutations of the same target per fwd/bwd pass during training
    # (Shujun's recommendation). Each sample is replicated K times in the batch,
    # each replica decoded with an independent random RNA-position permutation.
    # K=1 disables (single perm per sample, original behavior).
    k_perm: int = 1
    learning_rate: float = 0.0001
    weight_decay: float = 1e-4
    gamma_start: float = 0.5  # Starting value for gamma decay
    gamma_end: float = 0.5  # Ending value for gamma decay
    sequence_length: int = 100  # Length of the sequence for gamma calculation

    # Dataset Parameters
    n_targets: int = 720  # 00
    test_n_targets: int = 360  # 00
    test_size: float = 0.2
    update_train_structures: int = 2

    def __post_init__(self):
        if self.dim_feedforward is None:
            self.dim_feedforward = self.embed_size * 4

    @classmethod
    def from_yaml(cls, file_path: str) -> "TrainingConfig":
        with open(file_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, file_path: str):
        with open(file_path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)


def save_default_config():
    config = TrainingConfig()
    config.to_yaml("default_config.yaml")


# Original functions from your code
def load_structures(file_path):
    structures = []
    with open(file_path, "r") as f:
        for line in f:
            s = line.strip()
            structed_ness = 1 - s.count(".") / len(s)
            if len(s) < 100:
                s = s + "." * (100 - len(s))
            if structed_ness > 0.5:
                structures.append(s)
    # structured_ness=[1-s.count(".")/len(s) for s in structures]
    # structures
    return structures


def setup_models(config):
    policy_network = DotBracketRNATransformer(
        config.db_vocab_size,
        config.rna_vocab_size,
        config.embed_size,
        config.nhead,
        config.num_encoder_layers,
        config.num_decoder_layers,
        config.dim_feedforward,
        config.dropout,
    )
    target_network = DotBracketRNATransformer(
        config.db_vocab_size,
        config.rna_vocab_size,
        config.embed_size,
        config.nhead,
        config.num_encoder_layers,
        config.num_decoder_layers,
        config.dim_feedforward,
        config.dropout,
    )

    delete_modules(policy_network)
    delete_modules(target_network)

    # polycy_network.load_state_dict(torch.load("../test2_2M/policy_network.pt"))

    target_network.load_state_dict(policy_network.state_dict())

    return policy_network, target_network


def _make_permuted_causal_mask(perm, device):
    """Build a causal attention mask for a permuted decoding order.

    perm: [L] long tensor — decoding order (e.g. perm[0] is decoded first).
    Returns: [L+1, L+1] float mask (includes START token at position 0).
    The START token (col/row 0) is always visible to everyone.
    Position perm[i] can attend to START and positions perm[0..i].
    """
    L = perm.shape[0]
    full = L + 1  # +1 for START token
    mask = torch.full((full, full), float("-inf"), device=device)
    # START token (row 0) can see itself
    mask[0, 0] = 0.0
    # Build order: rank[pos] = step at which position pos is decoded
    rank = torch.empty(L, dtype=torch.long, device=device)
    rank[perm] = torch.arange(L, device=device)
    for step in range(L):
        pos = perm[step].item()
        row = pos + 1  # +1 for START offset
        # Can see START token
        mask[row, 0] = 0.0
        # Can see all positions decoded at steps <= step
        for prev_step in range(step + 1):
            prev_pos = perm[prev_step].item()
            mask[row, prev_pos + 1] = 0.0
    return mask


def train_epoch(
    policy_network,
    target_network,
    optimizer,
    dataloader,
    loss_fn,
    accelerator,
    config,
    scheduler=None,
    order_agnostic=False,
    log_state=None,
    mid_epoch_ckpt_path=None,
    resume_step=0,
    mid_epoch_save_interval=5000,
):
    """Train one epoch. If log_state is provided (main process only), append
    a row to logs/metrics_step.csv and wandb.log() after every optimizer step.

    log_state: dict with keys {csv_writer, csv_file, episode, epoch_idx,
                               start_time, use_wandb, global_step}.
               global_step is mutated in place.
    mid_epoch_ckpt_path: if set, save model+optimizer+scheduler state here
                         every mid_epoch_save_interval steps (crash recovery).
    resume_step: fast-skip this many batches at the start (resuming from ckpt).
    """
    policy_network.train()
    target_network.eval()
    total_loss = 0

    optimizer_lr = optimizer.param_groups[0]["lr"]
    print(f"Optimizer learning rate: {optimizer_lr}")
    if resume_step > 0:
        print(f"Fast-skipping {resume_step} batches (mid-epoch resume)...")

    for step, batch in enumerate(tqdm(dataloader, desc="Training")):
        if step < resume_step:
            continue
        sequence = batch["sequence"]
        structure = batch["structure"]
        reward = batch["reward"].float()
        ct_matrix = batch["ct_matrix"]

        # Reward reweighting
        exp_reward = torch.exp(reward.mean(1, keepdim=True))
        reward = reward * exp_reward

        B, L = sequence.shape

        if order_agnostic:
            # K-permutation tiling (Shujun's recommendation): replicate each
            # sample K times along the batch dim, with an independent random
            # permutation per replica. Forces the decoder to learn
            # permutation-invariant features per data sample.
            K = max(1, getattr(config, "k_perm", 1))
            if K > 1:
                sequence = sequence.repeat_interleave(K, dim=0)
                structure = structure.repeat_interleave(K, dim=0)
                reward = reward.repeat_interleave(K, dim=0)
                ct_matrix = ct_matrix.repeat_interleave(K, dim=0)
                B = sequence.shape[0]

            # Per-sample [B, L] random permutations: perm[b, t] is the RNA
            # position predicted at decoding step t for batch row b.
            perm = torch.stack(
                [torch.randperm(L, device=sequence.device) for _ in range(B)]
            )

            # Permute sequence and reward to match decoding order
            sequence_perm = sequence.gather(1, perm)
            reward_perm = reward.gather(1, perm)

            # paired_encoding at decoding step t is the paired/unpaired indicator
            # at RNA position perm[t] (the position about to be predicted at this
            # step). Slot t of the shifted-tgt input carries the *previously
            # emitted* nucleotide; the paired indicator is intentionally
            # forward-looking — it conditions the prediction on the kind of
            # position being predicted.
            paired_encoding_perm = (structure == 0).long().gather(1, perm)

            # Build shifted sequence in permuted order
            shifted = torch.full((B, L), 4, dtype=torch.long, device=sequence.device)
            shifted[:, 1:] = sequence_perm[:, :-1]

            # Causal mask on permuted order (standard upper-triangular).
            # Position information lives in the relative-position attention bias
            # inside the decoder, which is computed from `perm`.
            tgt_mask = torch.triu(
                torch.full((L, L), float("-inf"), device=sequence.device),
                diagonal=1,
            )

            with accelerator.autocast():
                # Forward pass with permuted inputs and per-sample perm
                q_vals = policy_network(
                    structure,
                    ct_matrix,
                    shifted,
                    tgt_mask=tgt_mask,
                    paired_encoding=paired_encoding_perm,
                    perm=perm,
                ).float()
                current_q = q_vals.gather(
                    -1,
                    sequence_perm.unsqueeze(-1),
                ).squeeze(-1)

                with torch.no_grad():
                    next_q = (
                        target_network(
                            structure,
                            ct_matrix,
                            shifted,
                            tgt_mask=tgt_mask,
                            paired_encoding=paired_encoding_perm,
                            perm=perm,
                        )
                        .float()
                        .detach()
                    )
                    next_q = next_q.max(-1)[0][:, 1:]
                    next_q = F.pad(next_q, (0, 1, 0, 0))

                gamma = (
                    torch.linspace(
                        config.gamma_start,
                        config.gamma_end,
                        L,
                    )
                    .to(sequence.device)
                    .float()
                )
                expected_q = reward_perm + (gamma * next_q)
                loss = loss_fn(current_q, expected_q)
        else:
            # Original L->R training
            shifted_sequence = torch.full(
                (B, L),
                4,
                dtype=torch.long,
                device=sequence.device,
            )
            shifted_sequence[:, 1:] = sequence[:, :-1]
            tgt_mask = torch.triu(
                torch.full((L, L), float("-inf"), device=sequence.device),
                diagonal=1,
            )

            with accelerator.autocast():
                current_q_values = policy_network(
                    structure,
                    ct_matrix,
                    shifted_sequence,
                    tgt_mask=tgt_mask,
                ).float()
                action = sequence
                current_q_values = current_q_values.gather(
                    -1,
                    action.unsqueeze(-1),
                ).squeeze(-1)

                with torch.no_grad():
                    next_q_values = (
                        target_network(
                            structure,
                            ct_matrix,
                            shifted_sequence,
                            tgt_mask=tgt_mask,
                        )
                        .float()
                        .detach()
                    )
                    next_q_values = next_q_values.max(-1)[0][:, 1:]
                    next_q_values = F.pad(next_q_values, (0, 1, 0, 0))

                gamma = (
                    torch.linspace(
                        config.gamma_start,
                        config.gamma_end,
                        config.sequence_length,
                    )
                    .to(sequence.device)
                    .float()
                )
                expected_q_values = reward + (gamma * next_q_values)
                loss = loss_fn(current_q_values, expected_q_values)

        accelerator.backward(loss)
        accelerator.clip_grad_norm_(policy_network.parameters(), 1)
        optimizer.step()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        step_loss = loss.item()
        total_loss += step_loss

        # Mid-epoch checkpoint (main process only)
        if mid_epoch_ckpt_path is not None and accelerator.is_main_process:
            if (step + 1) % mid_epoch_save_interval == 0:
                unwrapped = accelerator.unwrap_model(policy_network)
                torch.save({
                    "model": unwrapped.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "step": step,
                }, mid_epoch_ckpt_path)

        if log_state is not None:
            log_state["global_step"] += 1
            step_lr = optimizer.param_groups[0]["lr"]
            wall_s = time.time() - log_state["start_time"]
            log_state["csv_writer"].writerow(
                [
                    time.strftime("%Y-%m-%dT%H:%M:%S"),
                    log_state["episode"],
                    log_state["epoch_idx"],
                    log_state["global_step"],
                    f"{step_loss:.6f}",
                    f"{step_lr:.6e}",
                    sequence.shape[0],
                    f"{wall_s:.2f}",
                ]
            )
            if log_state["global_step"] % 50 == 0:
                log_state["csv_file"].flush()
            if log_state["use_wandb"]:
                wandb.log(
                    {
                        "train/step_loss": step_loss,
                        "train/lr": step_lr,
                        "train/episode": log_state["episode"],
                        "train/epoch_within_episode": log_state["epoch_idx"],
                        "global_step": log_state["global_step"],
                    }
                )

    optimizer_lr = optimizer.param_groups[0]["lr"]
    print(f"Optimizer learning rate after training: {optimizer_lr}")

    return total_loss / len(dataloader)


def play(model, env, dataloader, accelerator, episode, save_data=True, p=0.0,
         save_interval=500):
    model.eval()
    data = []
    run_id = accelerator.process_index
    os.makedirs("tmp", exist_ok=True)

    if save_data:
        os.makedirs(f"tmp/episode{episode}/process{run_id}", exist_ok=True)
        out_path = f"tmp/episode{episode}/process{run_id}/data.pt"
        episode_rewards = []
        batches_done = 0
    else:
        # Test play: checkpoint rewards so crashes can resume
        test_dir = f"tmp/test_episode{episode}/process{run_id}"
        os.makedirs(test_dir, exist_ok=True)
        rewards_ckpt = f"{test_dir}/rewards.npy"
        batches_done_file = f"{test_dir}/batches_done.txt"
        if os.path.exists(rewards_ckpt) and os.path.exists(batches_done_file):
            episode_rewards = list(np.load(rewards_ckpt, allow_pickle=True))
            with open(batches_done_file) as f:
                batches_done = int(f.read().strip())
            print(f"Resuming test play from batch {batches_done} ({len(episode_rewards)} rewards already computed)")
        else:
            episode_rewards = []
            batches_done = 0

    # Pick the unwrapped model for generate_permuted (which calls model.encoder
    # / model.decoder directly). DDP-wrapped modules expose `.module`.
    model_for_decode = model.module if hasattr(model, "module") else model

    for batch_num, batch in enumerate(tqdm(dataloader, desc="Playing")):
        if batch_num < batches_done:
            continue  # fast-skip already-computed batches (data loading only, no GPU work)

        src = batch["src"]
        PC = batch["paired_correspondence"]
        ct_matrix = batch["ct_matrix"]
        B_, L_ = src.shape

        # Random per-sample decoding permutation, matching the order-agnostic
        # training distribution. For p=0 (test play) this is the deterministic
        # argmax-with-base-pair-mask under random RNA-position order.
        perm = torch.stack([torch.randperm(L_, device=src.device) for _ in range(B_)])

        with torch.no_grad():
            with accelerator.autocast():
                predicted_sequences = generate_permuted(
                    model_for_decode, src, ct_matrix, PC,
                    perm=perm, mode="epsilon_argmax", p=p,
                )
                bpps = env.SS_model(predicted_sequences).sigmoid().detach().cpu().numpy()

        structures, bps = [], []
        for bpp in bpps:
            structure, bp = _hungarian(mask_diagonal(bpp), theta=0.5, min_len_helix=1)
            structures.append(structure)
            bps.append(bp)

        target_structures = [detokenize_dot_bracket(s.cpu().numpy()) for s in src]
        rewards = [env.get_reward(bp, s) for bp, s in zip(bps, target_structures)]
        episode_rewards.extend(rewards)

        if save_data:
            for t, s, r in zip(batch["src"], predicted_sequences, rewards):
                data.append([t.cpu(), s.cpu(), r])

            # Incremental save every save_interval batches so crashes don't lose all data
            if (batch_num + 1) % save_interval == 0 and data:
                torch.save({
                    "structures": torch.stack([d[0] for d in data]),
                    "sequences":  torch.stack([d[1] for d in data]),
                    "rewards":    torch.stack([torch.from_numpy(np.array(d[2])) for d in data]),
                }, out_path)
        else:
            # Save test play checkpoint after every batch (rewards are tiny)
            np.save(rewards_ckpt, np.array(episode_rewards))
            with open(batches_done_file, "w") as f:
                f.write(str(batch_num + 1))

    # Gather only rewards (scalars) — never gather structure tensors across GPUs;
    # that all-gather times out at NCCL's 600s limit when the play set is large.
    episode_rewards_t = torch.tensor(np.array(episode_rewards)).to(accelerator.device)
    episode_rewards_mask = torch.ones(len(episode_rewards_t)).to(accelerator.device)
    gathered_rewards = accelerator.gather(episode_rewards_t).cpu().numpy()
    gathered_mask = accelerator.gather(episode_rewards_mask).cpu().numpy()
    gathered_rewards = gathered_rewards[gathered_mask.astype("bool")]

    if save_data and data:
        torch.save({
            "structures": torch.stack([d[0] for d in data]),
            "sequences":  torch.stack([d[1] for d in data]),
            "rewards":    torch.stack([torch.from_numpy(np.array(d[2])) for d in data]),
        }, out_path)

    # Compute hard structures (10% worst-reward on this rank) without cross-GPU gather
    if save_data and data:
        item_rewards = [float(np.mean(d[2])) for d in data]
        n_hard = max(1, int(0.1 * len(data)))
        hard_indices = np.argsort(item_rewards)[:n_hard]
        local_hard_structures = [
            detokenize_dot_bracket(data[i][0].numpy()) for i in hard_indices
        ]
    else:
        local_hard_structures = []

    # Clean up test play checkpoint files on successful completion
    if not save_data:
        for f in [rewards_ckpt, batches_done_file]:
            if os.path.exists(f):
                os.remove(f)

    if accelerator.is_main_process:
        print(f"AVG reward for this episode: {np.mean(gathered_rewards)}")
    return np.mean(gathered_rewards), gathered_rewards, local_hard_structures


def log_rewards(filename, episode, train_rewards, test_rewards):
    with open(filename, "a") as f:
        f.write(f"{episode},{train_rewards},{test_rewards}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="RNA Structure Prediction Training")
    parser.add_argument(
        "--config",
        type=str,
        default="default_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--target_structure_file", type=str, help="Path to target structure file"
    )
    parser.add_argument(
        "--order-agnostic",
        action="store_true",
        help="Train with random decoding order (bidirectional)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to pretrained policy_network.pt to resume from",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="knitnet-struct2seq",
        help="WandB project name",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="WandB run name (default: auto-generated)",
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument(
        "--skip-play",
        action="store_true",
        help="Skip play phase for episode 0 and train directly on existing pkls in tmp/",
    )
    parser.add_argument(
        "--start-episode",
        type=int,
        default=0,
        help="Episode index to start from (0-indexed). Use with --checkpoint to resume mid-run.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training for --start-episode and go straight to test play. Use when checkpoint already exists.",
    )
    return parser.parse_args()


def main():
    # Parse arguments and load config
    args = parse_args()
    if not os.path.exists(args.config):
        print(f"Config file not found at {args.config}, creating default config...")
        save_default_config()
    config = TrainingConfig.from_yaml(args.config)

    start_time = time.time()
    accelerator = Accelerator(mixed_precision="bf16")

    # Prefer FlashAttention / mem-efficient SDPA backends on Ampere+. PyTorch
    # 2.3's nn.MultiheadAttention routes through SDPA when need_weights=False
    # (which is now true at every call site after the RPE redesign). Falls
    # back gracefully if conditions aren't met. This is a startup-only hint.
    try:
        torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_mem_efficient=True, enable_math=False
        )
    except (AttributeError, RuntimeError):
        # Older torch may not have sdp_kernel context manager as a callable;
        # ignore — math kernel is the safe default.
        pass

    os.makedirs("tmp", exist_ok=True)
    os.makedirs("stats", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # CSV metric logging (main process only)
    step_csv_file = None
    step_csv_writer = None
    episode_csv_file = None
    episode_csv_writer = None
    if accelerator.is_main_process:
        step_csv_path = os.path.join("logs", "metrics_step.csv")
        episode_csv_path = os.path.join("logs", "metrics_episode.csv")
        step_is_new = not os.path.exists(step_csv_path)
        episode_is_new = not os.path.exists(episode_csv_path)
        step_csv_file = open(step_csv_path, "a", newline="")
        step_csv_writer = csv.writer(step_csv_file)
        if step_is_new:
            step_csv_writer.writerow(
                [
                    "timestamp",
                    "episode",
                    "epoch_within_episode",
                    "global_step",
                    "train_loss",
                    "lr",
                    "batch_size",
                    "wall_s_since_start",
                ]
            )
            step_csv_file.flush()
        episode_csv_file = open(episode_csv_path, "a", newline="")
        episode_csv_writer = csv.writer(episode_csv_file)
        if episode_is_new:
            episode_csv_writer.writerow(
                [
                    "timestamp",
                    "episode",
                    "train_loss_mean",
                    "train_reward",
                    "test_reward",
                    "best_test_reward",
                    "lr_final",
                    "epoch_count",
                    "duration_s",
                ]
            )
            episode_csv_file.flush()

    use_wandb = not args.no_wandb
    if use_wandb and accelerator.is_main_process:
        run_name = args.wandb_run_name or (
            f"struct2seq_bidir_{'orderag' if args.order_agnostic else 'ltr'}_"
            f"{'pretrained' if args.checkpoint else 'scratch'}_"
            f"{time.strftime('%Y%m%d_%H%M%S')}"
        )
        wandb.init(
            project=args.wandb_project,
            entity="antonia-panescu-yale-university",
            name=run_name,
            config={
                **asdict(config),
                "order_agnostic": args.order_agnostic,
                "checkpoint": args.checkpoint,
            },
        )
        print(f"WandB run: {run_name}")

    # print(accelerator.distributed_type)
    # exit()

    # Load data and setup
    structures = pl.read_csv(args.target_structure_file)["structure"].to_list()
    # structed_ness=[1-s.count(".")/len(s) for s in structures]
    # structures = [s for s, z in zip(structures, structed_ness) if z > 0.5]

    # structures = load_structures("../structures.txt")
    print(f"there are {len(structures)} structures in total")
    train_structures, test_structures = train_test_split(
        structures, test_size=config.test_size, random_state=42
    )

    # Setup dataloaders
    np.random.seed(0)
    target_structures = np.random.choice(
        train_structures, config.n_targets, replace=False
    )
    test_target_structures = np.random.choice(
        test_structures, config.test_n_targets, replace=False
    )

    inference_dataset = DotBracketDataset(target_structures)
    inference_dataloader = DataLoader(
        inference_dataset,
        batch_size=config.inference_batch_size,
        shuffle=False,
        collate_fn=collate_dbn,
        num_workers=16,
    )

    test_inference_dataset = DotBracketDataset(test_target_structures)
    test_inference_dataloader = DataLoader(
        test_inference_dataset,
        batch_size=config.inference_batch_size,
        shuffle=False,
        collate_fn=collate_dbn,
        num_workers=16,
    )

    inference_dataloader, test_inference_dataloader = accelerator.prepare(
        inference_dataloader, test_inference_dataloader
    )

    # Setup model and optimizer
    policy_network, target_network = setup_models(config)

    # Load pretrained checkpoint if provided
    if args.checkpoint is not None and os.path.exists(args.checkpoint):
        print(f"Loading pretrained checkpoint: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location="cpu")

        # The decoder PE was redesigned (LSTM + CausalConv → relative-position
        # attention bias). Old checkpoints (Struct2SeQ.pt and any
        # policy_network_*.pt from before the redesign) carry weights for the
        # dropped layers; skip them. New checkpoints don't contain these keys
        # so the filter is a no-op for those.
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
            print(f"  Dropped {n_dropped} legacy decoder-PE keys (LSTM + CausalConv).")

        # Use strict=False so the new RPE bias tables stay at their default init.
        missing_p, unexpected_p = policy_network.load_state_dict(state, strict=False)
        missing_t, unexpected_t = target_network.load_state_dict(state, strict=False)
        if unexpected_p:
            raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected_p}")
        # Allow only the new RPE keys to be missing.
        allowed_missing = {"decoder.rpe_self.bias.weight", "decoder.rpe_cross.bias.weight"}
        unallowed = [k for k in missing_p if k not in allowed_missing]
        if unallowed:
            raise RuntimeError(f"Unexpected missing keys (not new RPE): {unallowed}")
        print(f"  Loaded successfully. Missing (newly-init RPE): {missing_p}")

    policy_network, target_network = accelerator.prepare(policy_network, target_network)

    # torch.compile after accelerator.prepare so the DDP wrapper is in place.
    # mode='default' (no CUDA graphs) avoids breakage on dynamic shapes from
    # variable-length structures. dynamic=True is a hint that shapes may
    # change between calls. env.SS_model and reactivity_model are already
    # compiled in DQN_env.__init__ (see Env.py).
    if os.environ.get("STRUCT2SEQ_DISABLE_COMPILE") != "1":
        try:
            policy_network = torch.compile(policy_network, mode="default", dynamic=True)
            target_network = torch.compile(target_network, mode="default", dynamic=True)
            print("[startup] torch.compile applied to policy and target networks")
        except Exception as e:
            print(f"[startup] torch.compile failed, continuing uncompiled: {e}")

    env = DQN_env()
    env.SS_model = accelerator.prepare(env.SS_model)
    loss_fn = torch.nn.MSELoss()

    # Training loop
    p = config.initial_p
    p_delta = -(config.initial_p - config.final_p) / config.n_episodes
    train_batch_size = config.train_batch_size
    best_test_reward = float("-inf")
    hard_structures = []
    # Fast-forward p and train_batch_size if resuming mid-run
    for _ in range(args.start_episode):
        p += p_delta
        train_batch_size = min(train_batch_size * 2, config.max_train_batch_size)
    # Mutable container for global step count; passed into train_epoch via log_state.
    global_step_counter = [0]
    for episode in range(args.start_episode, config.n_episodes):
        episode_start_time = time.time()
        print(f"Episode {episode + 1}/{config.n_episodes}")

        if (episode + 1) % config.update_train_structures == 0:
            print("updating train structures")
            target_structures = np.random.choice(
                train_structures, config.n_targets, replace=False
            )

        inference_dataset = DotBracketDataset(list(target_structures) + list(hard_structures))
        inference_dataloader = DataLoader(
            inference_dataset,
            batch_size=config.inference_batch_size,
            shuffle=False,
            collate_fn=collate_dbn,
            num_workers=16,
        )
        inference_dataloader = accelerator.prepare(inference_dataloader)

        # Auto-skip play if data already exists for this episode (crash recovery)
        existing_episode_data = (
            glob(f"tmp/episode{episode}/process*/data.pt") or
            glob(f"tmp/episode{episode}/process*/batch*.pkl")
        )
        if existing_episode_data:
            print(f"Skipping play phase — using existing data in tmp/episode{episode}/")
            train_rewards, train_rewards_vector, hard_structures = float("nan"), np.array([]), []
        else:
            train_rewards, train_rewards_vector, hard_structures = play(
                policy_network, env, inference_dataloader, accelerator, episode, p=p
            )
            if accelerator.is_main_process:
                df = pd.DataFrame({"episode_reward": np.mean(train_rewards_vector, axis=1) if train_rewards_vector.ndim == 2 else train_rewards_vector})
                df.to_csv(f"stats/episode{episode}.csv", index=False)

        accelerator.wait_for_everyone()

        # Train (skip if --skip-train and this is the start episode)
        skip_train_this_episode = args.skip_train and episode == args.start_episode
        if skip_train_this_episode:
            print(f"Skipping training for episode {episode} (--skip-train, checkpoint already exists)")
            loss = float("nan")
            epoch_losses = []
        else:
            train_dataset = RNADataset(
                glob("tmp/episode*/process*/data.pt") or
                glob("tmp/episode*/process*/batch*.pkl")
            )
            print(f"using batch size {train_batch_size} for training")
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=train_batch_size,
                shuffle=True,
                num_workers=16,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=4,
            )

            optimizer = Ranger(
                policy_network.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )

            if episode == 0:
                print("initializing LinearWarmupScheduler scheduler (1-epoch warmup)")
                scheduler = LinearWarmupScheduler(
                    optimizer=optimizer,
                    total_steps=len(train_dataloader),
                    final_lr=config.learning_rate,
                )
            else:
                print("initializing CosineAnnealingLR scheduler")
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, len(train_dataloader)
                )

            scheduler, optimizer, train_dataloader = accelerator.prepare(
                scheduler, optimizer, train_dataloader
            )
            # Resume from per-epoch checkpoint if one exists (crash recovery)
            epoch_ckpt_base = f"policy_network_{episode}_epoch"
            completed_epochs = sorted([
                int(f.replace(epoch_ckpt_base, "").replace(".pt", ""))
                for f in os.listdir(".")
                if f.startswith(epoch_ckpt_base) and f.endswith(".pt")
            ])
            start_epoch = completed_epochs[-1] + 1 if completed_epochs else 0
            if start_epoch > 0:
                ckpt_path = f"{epoch_ckpt_base}{completed_epochs[-1]}.pt"
                print(f"Resuming training from epoch {start_epoch} (loaded {ckpt_path})")
                unwrapped = accelerator.unwrap_model(policy_network)
                unwrapped.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
                accelerator.unwrap_model(target_network).load_state_dict(unwrapped.state_dict())

            epoch_losses = []
            for epoch in range(start_epoch, config.epochs_per_episode):

                anneal = epoch == config.epochs_per_episode - 1

                # Mid-epoch checkpoint path and resume step for this epoch
                mid_ckpt_path = f"tmp/train_mid_ckpt_ep{episode}_epoch{epoch}.pt"
                resume_step = 0
                if os.path.exists(mid_ckpt_path):
                    mid_ckpt = torch.load(mid_ckpt_path, map_location="cpu")
                    resume_step = mid_ckpt["step"] + 1
                    print(f"Resuming epoch {epoch} from step {resume_step} (mid-epoch checkpoint)")
                    unwrapped = accelerator.unwrap_model(policy_network)
                    unwrapped.load_state_dict(mid_ckpt["model"])
                    accelerator.unwrap_model(target_network).load_state_dict(unwrapped.state_dict())
                    optimizer.load_state_dict(mid_ckpt["optimizer"])
                    if scheduler is not None and mid_ckpt.get("scheduler") is not None:
                        scheduler.load_state_dict(mid_ckpt["scheduler"])

                log_state = None
                if accelerator.is_main_process:
                    log_state = {
                        "csv_writer": step_csv_writer,
                        "csv_file": step_csv_file,
                        "episode": episode + 1,
                        "epoch_idx": epoch,
                        "start_time": start_time,
                        "use_wandb": use_wandb,
                        "global_step": global_step_counter[0],
                    }

                if episode == 0 or anneal:
                    loss = train_epoch(
                        policy_network,
                        target_network,
                        optimizer,
                        train_dataloader,
                        loss_fn,
                        accelerator,
                        config,
                        scheduler=scheduler,
                        order_agnostic=args.order_agnostic,
                        log_state=log_state,
                        mid_epoch_ckpt_path=mid_ckpt_path,
                        resume_step=resume_step,
                    )
                else:
                    loss = train_epoch(
                        policy_network,
                        target_network,
                        optimizer,
                        train_dataloader,
                        loss_fn,
                        accelerator,
                        config,
                        scheduler=None,
                        order_agnostic=args.order_agnostic,
                        log_state=log_state,
                        mid_epoch_ckpt_path=mid_ckpt_path,
                        resume_step=resume_step,
                    )
                if log_state is not None:
                    global_step_counter[0] = log_state["global_step"]
                epoch_losses.append(loss)

                # Save per-epoch checkpoint and clean up mid-epoch checkpoint
                if accelerator.is_main_process:
                    unwrapped = accelerator.unwrap_model(policy_network)
                    accelerator.save(unwrapped.state_dict(), f"{epoch_ckpt_base}{epoch}.pt")
                    if os.path.exists(mid_ckpt_path):
                        os.remove(mid_ckpt_path)

        print(f"Training loss: {loss:.4f}")
        if use_wandb and accelerator.is_main_process:
            wandb.log({"train/loss": loss, "episode": episode + 1})
        target_network.load_state_dict(policy_network.state_dict())
        accelerator.wait_for_everyone()

        # Save checkpoint before test play so weights aren't lost if test play OOMs
        unwrapped_model = accelerator.unwrap_model(policy_network)
        accelerator.save(unwrapped_model.state_dict(), "policy_network.pt")
        accelerator.save(unwrapped_model.state_dict(), f"policy_network_{episode}.pt")

        # Clean up per-epoch checkpoints now that the final checkpoint is saved
        if accelerator.is_main_process:
            for f in os.listdir("."):
                if f.startswith(f"policy_network_{episode}_epoch") and f.endswith(".pt"):
                    os.remove(f)

        # #reset optimizer learning rate
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = config.learning_rate

        print("testing")
        test_rewards, test_rewards_vector, _ = play(
            policy_network,
            env,
            test_inference_dataloader,
            accelerator,
            episode,
            save_data=False,
        )

        if accelerator.is_main_process:
            df = pd.DataFrame({"episode_reward": np.mean(test_rewards_vector, axis=1) if test_rewards_vector.ndim == 2 else test_rewards_vector})
            df.to_csv(f"stats/test{episode}.csv", index=False)

        # if episode%3==0:
        train_batch_size = min(train_batch_size * 2, config.max_train_batch_size)

        # Track best before writing the CSV row so best_test_reward in the row
        # reflects the correct running maximum (including this episode).
        is_new_best = test_rewards > best_test_reward
        if is_new_best:
            best_test_reward = test_rewards

        # Log rewards
        if accelerator.is_main_process:
            log_rewards("rewards_log.csv", episode + 1, train_rewards, test_rewards)
            lr_final = optimizer.param_groups[0]["lr"]
            train_loss_mean = (
                float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            )
            episode_duration = time.time() - episode_start_time
            episode_csv_writer.writerow(
                [
                    time.strftime("%Y-%m-%dT%H:%M:%S"),
                    episode + 1,
                    f"{train_loss_mean:.6f}",
                    f"{float(train_rewards):.6f}",
                    f"{float(test_rewards):.6f}",
                    f"{float(best_test_reward):.6f}",
                    f"{lr_final:.6e}",
                    len(epoch_losses),
                    f"{episode_duration:.2f}",
                ]
            )
            episode_csv_file.flush()
            if use_wandb:
                wandb.log(
                    {
                        "reward/train": train_rewards,
                        "reward/test": test_rewards,
                        "reward/best_test": best_test_reward,
                        "train/loss_mean_per_episode": train_loss_mean,
                        "train/lr_final": lr_final,
                        "train/episode_duration_s": episode_duration,
                        "episode": episode + 1,
                        "global_step": global_step_counter[0],
                    }
                )

        # Save best checkpoint (regular per-episode checkpoint already saved before test play)
        unwrapped_model = accelerator.unwrap_model(policy_network)
        if is_new_best:
            accelerator.save(unwrapped_model.state_dict(), "best_policy_network.pt")
            print(f"New best test reward: {best_test_reward}. Saved weights.")
        p = p + p_delta
        print(f"updated p to {p}")

    # Save final model and record runtime
    unwrapped_model = accelerator.unwrap_model(policy_network)
    accelerator.save(unwrapped_model.state_dict(), "final_policy_network.pt")
    if use_wandb and accelerator.is_main_process:
        wandb.finish()
    if accelerator.is_main_process:
        if step_csv_file is not None:
            step_csv_file.close()
        if episode_csv_file is not None:
            episode_csv_file.close()

    end_time = time.time()
    runtime = end_time - start_time

    with open("runtime.txt", "w") as f:
        f.write(f"Total runtime: {runtime:.2f} seconds")

    print(f"Runtime recorded in runtime.txt")


if __name__ == "__main__":
    main()
