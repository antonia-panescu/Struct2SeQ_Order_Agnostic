"""
Single-GPU train-step throughput benchmark on the NEW RPE architecture.

Sweeps batch size and K (multi-perm replication) to find:
  - the largest (batch_size × K) that fits in 80 GB
  - the throughput-vs-memory curve, so we can pick the sweet spot for training

Also reports a comparison to the historical OLD-arch number from
logs/metrics_step.csv: ~79 samples/sec at B=256 across 4 GPUs ≈ 20
samples/sec/GPU.

Run:
  CUDA_VISIBLE_DEVICES=2 python evaluation/bench_train_speed.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Encoder_Decoder import DotBracketRNATransformer  # noqa
from run import TrainingConfig, delete_modules  # noqa


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


def build_model(device):
    config = TrainingConfig.from_yaml("config_brev_8gpu.yaml")
    model = DotBracketRNATransformer(
        config.db_vocab_size, config.rna_vocab_size, config.embed_size,
        config.nhead, config.num_encoder_layers, config.num_decoder_layers,
        config.dim_feedforward, config.dropout,
    )
    delete_modules(model)
    state = torch.load("Struct2SeQ.pt", map_location="cpu")
    state = {k: v for k, v in state.items() if k not in DROP_OLD_DECODER_PE}
    model.load_state_dict(state, strict=False)
    return model.to(device), config


def make_dummy_batch(B, L, device):
    structure = torch.randint(0, 10, (B, L), device=device)
    sequence = torch.randint(0, 4, (B, L), device=device)
    reward = torch.rand((B, L), device=device)
    ct_matrix = torch.eye(L, device=device).unsqueeze(0).expand(B, L, L).contiguous()
    return structure, sequence, reward, ct_matrix


def bench(model, B, L, K, device, n_warmup=3, n_iter=10):
    """Run K-perm multi-replication training step. Returns (samples/sec, peak_mem_GB).

    Reproduces the train_epoch order-agnostic block exactly: replicate by K,
    build per-row [B*K, L] perms, fwd+bwd through both policy and target,
    Q-learning style loss.
    """
    structure, sequence, reward, ct_matrix = make_dummy_batch(B, L, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()

    # Build a target_network (DDP semantics: same weights in eval mode)
    # For benchmarking we just use the policy in eval mode for the Q-target.
    target_model, _ = build_model(device)
    target_model.eval()

    torch.cuda.reset_peak_memory_stats(device)

    times = []
    for it in range(n_warmup + n_iter):
        seq = sequence.repeat_interleave(K, dim=0)
        struct = structure.repeat_interleave(K, dim=0)
        rew = reward.repeat_interleave(K, dim=0)
        ct = ct_matrix.repeat_interleave(K, dim=0)
        Beff = seq.shape[0]
        perm = torch.stack([torch.randperm(L, device=device) for _ in range(Beff)])
        seq_perm = seq.gather(1, perm)
        rew_perm = rew.gather(1, perm)
        paired_perm = (struct == 0).long().gather(1, perm)
        shifted = torch.full((Beff, L), 4, dtype=torch.long, device=device)
        shifted[:, 1:] = seq_perm[:, :-1]
        tgt_mask = torch.triu(torch.full((L, L), float("-inf"), device=device), diagonal=1)

        torch.cuda.synchronize()
        t0 = time.time()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            q = model(struct, ct, shifted, tgt_mask=tgt_mask, paired_encoding=paired_perm, perm=perm).float()
            current_q = q.gather(-1, seq_perm.unsqueeze(-1)).squeeze(-1)
            with torch.no_grad():
                next_q = target_model(struct, ct, shifted, tgt_mask=tgt_mask, paired_encoding=paired_perm, perm=perm).float().detach()
                next_q = next_q.max(-1)[0][:, 1:]
                next_q = F.pad(next_q, (0, 1, 0, 0))
            gamma = torch.linspace(1.0, 0.0, L, device=device)
            expected = rew_perm + gamma * next_q
            loss = loss_fn(current_q, expected)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        t1 = time.time()
        if it >= n_warmup:
            times.append(t1 - t0)

    mean_iter = sum(times) / len(times)
    samples_per_sec = (B * K) / mean_iter
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
    del target_model
    torch.cuda.empty_cache()
    return samples_per_sec, peak_mem_gb, mean_iter


def main():
    device = torch.device("cuda:0")
    print(f"[bench] device = {torch.cuda.get_device_name(0)}")
    print(f"[bench] L = 240 (matches config sequence_length)")
    print(f"[bench] OLD arch reference (from logs/metrics_step.csv 2026-04-27):")
    print(f"        ~9.89 it/s × B=256 across 4 GPUs = 79 samples/sec total")
    print(f"        per-GPU: ~19.8 samples/sec at B=256 (4-way DDP)")
    print()

    # Sweep over (B, K) with caps to avoid OOM
    L = 240
    configs = [
        # K=1 (no multi-perm)
        (32,   1),
        (64,   1),
        (128,  1),
        (256,  1),
        (512,  1),
        # K=4 (Shujun's recommendation)
        (32,   4),  # effective batch 128
        (64,   4),  # effective 256
        (128,  4),  # effective 512
        (256,  4),  # effective 1024
        # K=8 if memory permits
        (64,   8),  # effective 512
        (128,  8),  # effective 1024
    ]

    print(f"{'B':>5} {'K':>3} {'eff':>5}   {'samp/s':>10}  {'mem GB':>7}  {'sec/it':>7}  {'speedup vs old':>14}")
    print('-' * 75)
    model, config = build_model(device)
    model.train()

    OLD_PER_GPU = 19.8

    results = []
    for (B, K) in configs:
        try:
            sps, mem, sec = bench(model, B, L, K, device)
        except torch.cuda.OutOfMemoryError as e:
            print(f"{B:>5} {K:>3} {B*K:>5}   OOM ({str(e).split(chr(10))[0][:50]}...)")
            torch.cuda.empty_cache()
            continue
        speedup = sps / OLD_PER_GPU
        print(f"{B:>5} {K:>3} {B*K:>5}   {sps:>10.1f}  {mem:>7.2f}  {sec:>7.3f}  {speedup:>13.1f}x")
        results.append((B, K, sps, mem, sec, speedup))

    print()
    if results:
        # Pick the recommended config: largest samp/s that fits comfortably (< 70 GB)
        valid = [r for r in results if r[3] < 70]
        best = max(valid, key=lambda r: r[2]) if valid else None
        if best:
            B, K, sps, mem, sec, speedup = best
            print(f"[bench] Recommended config: B={B}, K={K}, effective batch={B*K}")
            print(f"        Throughput: {sps:.0f} samp/s ({speedup:.1f}× faster than old arch)")
            print(f"        Memory: {mem:.1f} GB / 80 GB ({mem/80*100:.0f}%)")
            print(f"        On 4 GPUs DDP, projected: {sps*4:.0f} samp/s total")


if __name__ == "__main__":
    main()
