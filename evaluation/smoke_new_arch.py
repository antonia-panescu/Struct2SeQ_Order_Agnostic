"""
Smoke test for the new RPE-based decoder architecture:
  1. Build the model with the new architecture
  2. Load Struct2SeQ.pt with strict=False after dropping the 8 LSTM/Conv keys
  3. Verify `unexpected` is empty and `missing` is exactly the new RPE bias keys
  4. Run a forward+backward pass on a dummy batch with both L→R and randperm
     decoding orders, with and without KV-cache
  5. Report timings for sanity
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Encoder_Decoder import DotBracketRNATransformer, RelativePositionBias  # noqa
from run import TrainingConfig, delete_modules  # noqa


DROP = {
    "decoder.positional_encoding.weight_ih_l0",
    "decoder.positional_encoding.weight_hh_l0",
    "decoder.positional_encoding.bias_ih_l0",
    "decoder.positional_encoding.bias_hh_l0",
    "decoder.conv.0.conv.weight",
    "decoder.conv.0.conv.bias",
    "decoder.conv_norm.weight",
    "decoder.conv_norm.bias",
}


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device = {device}")

    config = TrainingConfig.from_yaml("config_brev_8gpu.yaml")
    model = DotBracketRNATransformer(
        config.db_vocab_size, config.rna_vocab_size, config.embed_size,
        config.nhead, config.num_encoder_layers, config.num_decoder_layers,
        config.dim_feedforward, config.dropout,
    )
    delete_modules(model)
    model = model.to(device)

    # ---- 1. Verify DROP set ---------------------------------------------------
    state = torch.load("Struct2SeQ.pt", map_location="cpu")
    actual_dropped = sorted(k for k in state.keys() if k in DROP)
    print(f"[smoke] keys in DROP found in checkpoint: {len(actual_dropped)}/{len(DROP)}")
    for k in sorted(DROP):
        present = "OK " if k in actual_dropped else "MISS"
        print(f"        {present}  {k}")
    assert len(actual_dropped) == len(DROP), "DROP set is wrong — see MISS above"

    state = {k: v for k, v in state.items() if k not in DROP}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[smoke] unexpected keys: {len(unexpected)}")
    assert not unexpected, f"unexpected keys: {unexpected}"
    print(f"[smoke] missing keys (should be the new RPE tables): {missing}")
    expected_missing_prefix = "decoder.rpe_"
    for k in missing:
        assert k.startswith(expected_missing_prefix), f"unexpected missing key: {k}"
    assert any("rpe_self.bias.weight" in k for k in missing), missing
    assert any("rpe_cross.bias.weight" in k for k in missing), missing

    # ---- 2. Forward / backward, L→R order ------------------------------------
    B, L = 4, 100
    src = torch.randint(0, 10, (B, L), device=device)
    ct_matrix = torch.eye(L, device=device).unsqueeze(0).expand(B, L, L).contiguous()
    sequence = torch.randint(0, 4, (B, L), device=device)
    shifted = torch.full((B, L), 4, dtype=torch.long, device=device)
    shifted[:, 1:] = sequence[:, :-1]
    paired_encoding = (src == 0).long()  # L→R: encoding at the position being predicted = at RNA pos t = step t
    tgt_mask = torch.triu(torch.full((L, L), float("-inf"), device=device), diagonal=1)

    model.train()
    out = model(src, ct_matrix, shifted, tgt_mask=tgt_mask, paired_encoding=paired_encoding)
    print(f"[smoke] L→R forward output shape: {tuple(out.shape)}  (expected ({B}, {L}, {config.rna_vocab_size - 1}))")
    assert out.shape == (B, L, config.rna_vocab_size - 1)
    loss = out.sum()
    loss.backward()
    print(f"[smoke] L→R backward: loss={loss.item():.4f}")

    # ---- 3. Forward / backward, randperm order --------------------------------
    perm = torch.stack([torch.randperm(L, device=device) for _ in range(B)])
    sequence_perm = sequence.gather(1, perm)
    shifted_perm = torch.full((B, L), 4, dtype=torch.long, device=device)
    shifted_perm[:, 1:] = sequence_perm[:, :-1]
    paired_encoding_perm = (src == 0).long().gather(1, perm)

    model.zero_grad()
    out_perm = model(
        src, ct_matrix, shifted_perm,
        tgt_mask=tgt_mask, paired_encoding=paired_encoding_perm, perm=perm,
    )
    print(f"[smoke] randperm forward output shape: {tuple(out_perm.shape)}")
    assert out_perm.shape == (B, L, config.rna_vocab_size - 1)
    loss = out_perm.sum()
    loss.backward()
    print(f"[smoke] randperm backward: loss={loss.item():.4f}")

    # ---- 4. KV-cache decoding sanity (single autoregressive step) -------------
    model.eval()
    with torch.no_grad():
        # Step 0: feed start token only
        memory = model.encoder(src, ct_matrix)
        start_tok = torch.full((B, 1), 4, dtype=torch.long, device=device)
        paired_step0 = paired_encoding_perm[:, :1]  # encoding at position perm[:, 0]
        out0, kv0 = model.decoder(
            start_tok, paired_step0, memory, perm=perm,
            tgt_mask=None, past_key_values=None, use_cache=True,
        )
        print(f"[smoke] cache step 0 out shape: {tuple(out0.shape)}, kv per layer: {tuple(kv0[0][0].shape)}")
        # Step 1: feed previous token + cache from step 0
        prev_tok = torch.argmax(out0[:, -1, :], dim=-1, keepdim=True)
        paired_step1 = paired_encoding_perm[:, 1:2]
        out1, kv1 = model.decoder(
            prev_tok, paired_step1, memory, perm=perm,
            tgt_mask=None, past_key_values=kv0, use_cache=True,
        )
        print(f"[smoke] cache step 1 out shape: {tuple(out1.shape)}, kv per layer: {tuple(kv1[0][0].shape)}")
        assert kv1[0][0].shape[1] == 2, "KV cache should grow by 1 per step"

    # ---- 5. Timing sanity -----------------------------------------------------
    if device.type == "cuda":
        # Time 10 forward passes at L=240 batch=64
        L2, B2 = 240, 64
        src2 = torch.randint(0, 10, (B2, L2), device=device)
        ct2 = torch.eye(L2, device=device).unsqueeze(0).expand(B2, L2, L2).contiguous()
        seq2 = torch.randint(0, 4, (B2, L2), device=device)
        shifted2 = torch.full((B2, L2), 4, dtype=torch.long, device=device)
        shifted2[:, 1:] = seq2[:, :-1]
        perm2 = torch.stack([torch.randperm(L2, device=device) for _ in range(B2)])
        paired_perm2 = (src2 == 0).long().gather(1, perm2)
        tgt_mask2 = torch.triu(torch.full((L2, L2), float("-inf"), device=device), diagonal=1)

        model.train()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            o = model(src2, ct2, shifted2, tgt_mask=tgt_mask2, paired_encoding=paired_perm2, perm=perm2)
            o.sum().backward()
            model.zero_grad()
        torch.cuda.synchronize()
        dt = time.time() - t0
        print(f"[smoke] 10 fwd+bwd at B={B2} L={L2}: {dt:.2f}s  ({10 * B2 / dt:.1f} samp/s)")

    print("\n[smoke] === ALL CHECKS PASSED ===")


if __name__ == "__main__":
    main()
