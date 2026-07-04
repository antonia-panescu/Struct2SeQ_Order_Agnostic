import csv
import os
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import yaml
from arnie.utils import convert_bp_list_to_dotbracket, convert_dotbracket_to_bp_list
from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupScheduler(_LRScheduler):
    """Linear warmup learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer
        total_steps: Total number of steps in one epoch (len(train_loader))
        final_lr: Target learning rate at the end of warmup

    Note:
        Despite the name, self.last_epoch inherited from _LRScheduler
        actually counts steps, not epochs. It starts at -1 and is
        incremented by 1 every time scheduler.step() is called.
    """

    def __init__(self, optimizer, total_steps, final_lr):
        self.total_steps = total_steps
        self.final_lr = final_lr
        super().__init__(optimizer)  # last_epoch=-1 by default

    def get_lr(self):
        # self.last_epoch is actually the current step number (starts at 0)
        current_step = self.last_epoch
        # Calculate current step's learning rate
        progress = float(current_step) / self.total_steps
        # Clip progress to avoid lr going above final_lr
        progress = min(1.0, progress)

        return [self.final_lr * progress for _ in self.base_lrs]


PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("ARNIEFILE", str(PROJECT_ROOT / "arnie_file.txt"))

from arnie.pk_predictors import _hungarian


def standardize_dbn(s):
    bps = convert_dotbracket_to_bp_list(s, allow_pseudoknots=True)
    converted = convert_bp_list_to_dotbracket(bps, len(s))
    return converted


def cor2vec(corr, s):
    seq_len = len(s)
    vec = np.ones(seq_len) * -1

    for i in corr:
        j = corr[i]
        vec[i] = j
        vec[j] = i
    return vec


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries = entries

    def print(self):
        print(self.entries)


def load_config_from_yaml(file_path):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return Config(**config)


def write_config_to_yaml(config, file_path):
    with open(file_path, "w") as file:
        yaml.safe_dump(config, file)


def delete_modules(model):
    del model.encoder.encoder_layers.self_attn.in_proj_weight
    del model.encoder.encoder_layers.self_attn.in_proj_bias
    del model.encoder.encoder_layers.self_attn.out_proj.weight
    del model.encoder.encoder_layers.self_attn.out_proj.bias
    del model.encoder.encoder_layers.linear1.weight
    del model.encoder.encoder_layers.linear1.bias
    del model.encoder.encoder_layers.linear2.weight
    del model.encoder.encoder_layers.linear2.bias
    del model.encoder.encoder_layers.norm1.weight
    del model.encoder.encoder_layers.norm1.bias
    del model.encoder.encoder_layers.norm2.weight
    del model.encoder.encoder_layers.norm2.bias

    # Delete from decoder
    # del model.decoder.decoder_layers.self_attn.in_proj_weight
    # del model.decoder.decoder_layers.self_attn.in_proj_bias
    # del model.decoder.decoder_layers.self_attn.out_proj.weight
    # del model.decoder.decoder_layers.self_attn.out_proj.bias
    # del model.decoder.decoder_layers.multihead_attn.in_proj_weight
    # del model.decoder.decoder_layers.multihead_attn.in_proj_bias
    # del model.decoder.decoder_layers.multihead_attn.out_proj.weight
    # del model.decoder.decoder_layers.multihead_attn.out_proj.bias
    # del model.decoder.decoder_layers.linear1.weight
    # del model.decoder.decoder_layers.linear1.bias
    # del model.decoder.decoder_layers.linear2.weight
    # del model.decoder.decoder_layers.linear2.bias
    # del model.decoder.decoder_layers.norm1.weight
    # del model.decoder.decoder_layers.norm1.bias
    # del model.decoder.decoder_layers.norm2.weight
    # del model.decoder.decoder_layers.norm2.bias
    # del model.decoder.decoder_layers.norm3.weight
    # del model.decoder.decoder_layers.norm3.bias
    return model


def generate_sequence(model, src, target_correspondence, start_token=4, p=0.1):
    seq_len = len(src)
    model.eval()
    # p=0.1
    with torch.no_grad():
        src = src.unsqueeze(0)  # Add batch dimension if not present
        memory = model.encoder(src)

        tgt = torch.tensor(
            [[start_token]], dtype=torch.long
        ).cuda()  # Start token with batch dimension
        outputs = [start_token]

        for position in range(seq_len):
            tgt_mask = torch.triu(
                torch.full(
                    (tgt.size(1), tgt.size(1)), float("-inf"), device=tgt.device
                ),
                diagonal=1,
            )
            paired_encoding = (src == 0).long()
            out = model.decoder(tgt, paired_encoding, memory, tgt_mask=tgt_mask)
            values = out[0, -1]
            # break
            # Apply base pair constraints
            mask = create_base_pair_mask(position, target_correspondence, outputs)
            masked_values = values.masked_fill(mask.cuda(), float("-inf"))
            # masked_values = values
            # break
            # Sample from the policy, not really
            if np.random.uniform() < p:
                probs = F.softmax(masked_values / 10000000, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
            else:
                next_token = torch.argmax(masked_values)

            outputs.append(next_token)
            tgt = torch.cat(
                [tgt, torch.tensor([[next_token]], dtype=torch.long).cuda()], dim=-1
            )

    nts = "ACGU"

    predicted_sequence = torch.tensor(outputs[1:])
    return predicted_sequence


def generate_sequence_norules(model, src, target_correspondence, start_token=4, p=0.1):
    seq_len = len(src)
    model.eval()
    # p=0.1
    with torch.no_grad():
        src = src.unsqueeze(0)  # Add batch dimension if not present
        memory = model.encoder(src)

        tgt = torch.tensor(
            [[start_token]], dtype=torch.long
        ).cuda()  # Start token with batch dimension
        outputs = [start_token]

        for position in range(seq_len):
            tgt_mask = torch.triu(
                torch.full(
                    (tgt.size(1), tgt.size(1)), float("-inf"), device=tgt.device
                ),
                diagonal=1,
            )
            paired_encoding = (src == 0).long()
            out = model.decoder(tgt, paired_encoding, memory, tgt_mask=tgt_mask)
            values = out[0, -1]
            # break
            # Apply base pair constraints
            # mask = create_base_pair_mask(position, target_correspondence, outputs)
            # masked_values = values.masked_fill(mask.cuda(), float('-inf'))
            masked_values = values
            # masked_values = values
            # break
            # Sample from the policy, not really
            if np.random.uniform() < p:
                probs = F.softmax(masked_values / 10000000, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
            else:
                next_token = torch.argmax(masked_values)

            outputs.append(next_token)
            tgt = torch.cat(
                [tgt, torch.tensor([[next_token]], dtype=torch.long).cuda()], dim=-1
            )

    nts = "ACGU"

    predicted_sequence = torch.tensor(outputs[1:])
    return predicted_sequence


# def generate_sequence_batched(model, src_batch, target_correspondence_batch, start_token=4, p=0.1, max_len=None):
#     batch_size, seq_len = src_batch.shape
#     model.eval()
#     device = src_batch.device

#     with torch.no_grad():
#         memory = model.encoder(src_batch)

#         tgt = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
#         outputs = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)

#         if max_len is None:
#             max_len = seq_len

#         for position in range(max_len):
#             tgt_mask = torch.triu(torch.full((tgt.size(1), tgt.size(1)), float('-inf'), device=device), diagonal=1)
#             paired_encoding = (src_batch == 0).long()
#             out = model.decoder(tgt, paired_encoding, memory, tgt_mask=tgt_mask)
#             values = out[:, -1, :]

#             # Apply base pair constraints
#             #mask = create_base_pair_mask_batched(position, target_correspondence_batch, outputs)
#             mask = []
#             for i,tc in enumerate(target_correspondence_batch):
#                 mask = create_base_pair_mask(position, target_correspondence, outputs)
#                 mask.append(create_base_pair_mask(position, tc, outputs[i]))
#             mask = torch.stack(mask,0)
#             masked_values = values.masked_fill(mask, float('-inf'))

#             # Sample from the policy
#             random_sample = torch.rand(batch_size, device=device) < p
#             probs = F.softmax(masked_values / 1e7, dim=-1)
#             sampled_tokens = torch.multinomial(probs, 1)
#             max_tokens = torch.argmax(masked_values, dim=-1, keepdim=True)
#             next_tokens = torch.where(random_sample.unsqueeze(1), sampled_tokens, max_tokens)

#             outputs = torch.cat([outputs, next_tokens], dim=1)
#             tgt = torch.cat([tgt, next_tokens], dim=1)

#     return outputs[:, 1:]  # Remove start token


# def create_base_pair_mask_batch(position, target_correspondence, outputs, batch_size):
#     mask = torch.zeros(batch_size, 4, dtype=torch.bool, device=outputs.device)
#     for b in range(batch_size):
#         if position in target_correspondence[b]:
#             paired_position = target_correspondence[b][position]
#             if position > paired_position:
#                 left_base = outputs[b, paired_position + 1]
#                 if left_base == 0:  # A
#                     mask[b, 0:3] = True  # Only U is allowed
#                 elif left_base == 1:  # C
#                     mask[b, 0:2] = True  # Only G is allowed
#                     mask[b, 3] = True
#                 elif left_base == 2:  # G
#                     mask[b, 0] = True  # C and U are allowed
#                     mask[b, 2] = True
#                 elif left_base == 3:  # U
#                     mask[b, 1:4] = True  # Only A is allowed
#     return mask

# def generate_sequence_batched(model, src, target_correspondence, start_token=4, p=0.1):
#     batch_size, seq_len = src.shape
#     model.eval()

#     with torch.no_grad():
#         memory = model.encoder(src)

#         tgt = torch.full((batch_size, 1), start_token, dtype=torch.long, device=src.device)
#         outputs = torch.full((batch_size, seq_len + 1), start_token, dtype=torch.long, device=src.device)

#         for position in range(seq_len):
#             out = model.decoder(tgt, memory)
#             values = out[:, -1, :]

#             mask = create_base_pair_mask_batch(position, target_correspondence, outputs, batch_size)
#             masked_values = values.masked_fill(mask, float('-inf'))

#             random_sample = torch.rand(batch_size, device=src.device) < p
#             probs = F.softmax(masked_values / 10000000, dim=-1)
#             sampled_tokens = torch.multinomial(probs, 1).squeeze(-1)
#             max_tokens = torch.argmax(masked_values, dim=-1)

#             next_tokens = torch.where(random_sample, sampled_tokens, max_tokens)

#             outputs[:, position + 1] = next_tokens
#             tgt = torch.cat([tgt, next_tokens.unsqueeze(1)], dim=1)

#     predicted_sequences = outputs[:, 1:]  # Remove start tokens
#     return predicted_sequences


def jaccard_similarity_base_pairs(bp_set1, bp_set2):
    """
    Compute Jaccard similarity between two sets of base pairs.

    :param bp_set1: List of base pair tuples for set 1
    :param bp_set2: List of base pair tuples for set 2
    :return: Jaccard similarity as a float between 0 and 1
    """
    # Convert lists of base pairs to sets of frozensets for efficient set operations
    set1 = set(frozenset(bp) for bp in bp_set1)
    set2 = set(frozenset(bp) for bp in bp_set2)

    # Compute intersection and union
    intersection = set1.intersection(set2)
    union = set1.union(set2)

    # Compute Jaccard similarity
    jaccard = (
        len(intersection) / len(union) if union else 1.0
    )  # If both sets are empty, similarity is 1

    return jaccard


# Example usage
# bp_set1 = [[16, 43], [17, 44], [18, 41], [19, 42], [20, 39], [21, 40], [22, 37], [23, 38], [24, 35], [25, 36]]
# bp_set2 = [[16, 43], [17, 44], [18, 41], [19, 42], [20, 39], [21, 40], [22, 37], [23, 38], [26, 35], [27, 34]]


# Watson-Crick + wobble allowed pairs for the base-pair constraint mask.
# Index of complement(s) for each base. ACGU -> 0/1/2/3.
#   left=A(0) -> only U(3) allowed
#   left=C(1) -> only G(2)
#   left=G(2) -> C(1) or U(3)
#   left=U(3) -> only A(0)
_COMPLEMENT_ALLOWED = {0: {3}, 1: {2}, 2: {1, 3}, 3: {0}}
_MAX_REPEAT = [4, 4, 3, 5]  # per-nt repeat caps in step-order, ACGU


@torch.no_grad()
def generate_permuted(
    model,
    src,                                   # [B, L] long, structure tokens
    ct_matrix,                             # [B, L, L]
    target_correspondence,                 # list[dict] length B (RNA-pos -> partner)
    perm,                                  # [B, L] long, RNA pos predicted at each step
    mode: str = "epsilon_argmax",          # "epsilon_argmax" or "sample"
    p: float = 0.0,                        # epsilon for "epsilon_argmax"; ignored for "sample"
    start_token: int = 4,
    fixed=None,                            # optional [B, L] long, -1 free, 0..3 constrained
):
    """Permutation-aware autoregressive decode with KV-cache.

    Matches the order-agnostic training distribution:
      - tgt slot t holds the previously-emitted nucleotide (at perm[t-1] for t>=1,
        start_token for t=0)
      - paired_encoding slot t describes the position about to be predicted
        (at perm[t]) — forward-looking, identical to training's
        `paired_encoding_perm[:, t]`
      - position information is supplied via the decoder's relative-position
        attention bias keyed on `perm`; no additive PE on the input embedding.

    Returns: [B, L] long, generated nucleotides in RNA-position order.

    Sampling modes:
      "epsilon_argmax": with probability p, sample uniformly from allowed bases
        (softmax-with-tiny-temperature trick); otherwise argmax. Matches the
        original L→R generate_sequence_batched policy.
      "sample": multinomial draw from softmax(values), no temperature scaling.
        Matches the original L→R generate_sequence_batched_sample with p=1.
    """
    B, L = src.shape
    device = src.device
    model.eval()

    memory = model.encoder(src, ct_matrix)
    if memory.shape[0] == 1 and B != 1:
        memory = memory.expand(B, *memory.shape[1:])

    # paired/unpaired indicator at each RNA position, then permuted to step order
    paired_at_pos = (src == 0).long()                 # [B, L]
    paired_step = paired_at_pos.gather(1, perm)        # [B, L]; matches training

    rna_outputs = torch.full((B, L), -1, dtype=torch.long, device=device)
    if fixed is not None:
        # In-painting: prefill constrained positions so partner-mask logic
        # (below) sees the fixed nucleotides when checking pairing
        # constraints — required because the partner of a current step may
        # already be a fixed position.
        rna_outputs = torch.where(fixed >= 0, fixed, rna_outputs)
    A_cnt = np.zeros(B, dtype=np.float64)

    # Step-order tgt input. At step t, we feed:
    #   tgt_in[t] = previously emitted nucleotide (or start_token at t=0)
    #   paired_in[t] = paired_step[:, t] (encoding of position being predicted)
    # In cached decoding we only feed the new slot per step.
    past_key_values = None
    prev_tok = torch.full((B, 1), start_token, dtype=torch.long, device=device)

    for t in range(L):
        cur_rna_pos = perm[:, t]                       # [B]
        paired_in = paired_step[:, t : t + 1]          # [B, 1]

        out = model.decoder(
            prev_tok,
            paired_in,
            memory,
            perm=perm,
            tgt_mask=None,                  # cached path uses RPE bias only
            past_key_values=past_key_values,
            use_cache=True,
        )
        if isinstance(out, tuple):
            out, past_key_values = out
        values = out[:, -1, :]                          # [B, 4]

        # ---- per-row mask in RNA-position space ----------------------------
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
                    allowed = _COMPLEMENT_ALLOWED[partner_nt]
                    for nt in range(4):
                        if nt not in allowed:
                            mask[b, nt] = True

        # repeat constraint over recently emitted tokens (step-order window)
        if t > 5:
            recent_step_tokens = []
            # we don't keep the step-order history explicitly; reconstruct from rna_outputs
            # using the perm prefix
            prefix_perm = perm[:, max(0, t - max(_MAX_REPEAT)) : t]   # last few steps
            recent = rna_outputs.gather(1, prefix_perm)               # [B, w]
            for nt in range(4):
                w = _MAX_REPEAT[nt]
                if t >= w:
                    window = recent[:, -w:]
                    triggered = (window == nt).all(dim=1)
                    new_mask = mask.clone()
                    new_mask[:, nt] = mask[:, nt] | triggered
                    full_block = new_mask.sum(dim=1) == 4
                    new_mask[full_block, nt] = mask[full_block, nt]
                    mask = new_mask

        a_block = torch.from_numpy(A_cnt > L * 0.4).to(device)
        if a_block.any():
            new_mask = mask.clone()
            new_mask[:, 0] = mask[:, 0] | a_block
            full_block = new_mask.sum(dim=1) == 4
            new_mask[full_block, 0] = mask[full_block, 0]
            mask = new_mask

        masked_values = values.masked_fill(mask, float("-inf"))

        # Defensive: if every base is masked for some row (can happen on
        # tight pseudoknots when partner-mask already blocks 3 and the
        # recent-repeat constraint blocks the 4th — the escape hatches
        # above don't catch this composition), fall back to the unmasked
        # values for that row so softmax/argmax don't see all-(-inf).
        all_masked_rows = torch.isinf(masked_values).all(dim=-1)
        if all_masked_rows.any():
            masked_values = torch.where(
                all_masked_rows.unsqueeze(-1), values, masked_values,
            )

        # ---- sampling ------------------------------------------------------
        if mode == "epsilon_argmax":
            # bf16-safe deterministic-argmax-via-softmax: cast to fp32 first.
            # Sanitize NaN/inf from bf16 model outputs so argmax & multinomial
            # don't crash on tight pseudoknots / overflow.
            mv32 = masked_values.float()
            mv32 = torch.nan_to_num(mv32, nan=-1e30, posinf=1e30, neginf=-1e30)
            argmax = torch.argmax(mv32, dim=-1, keepdim=True)
            if p > 0.0:
                probs = F.softmax(mv32 / 1e7, dim=-1)
                probs = torch.nan_to_num(probs, nan=0.25, posinf=0.0, neginf=0.0)
                probs = probs.clamp(min=1e-8)
                probs = probs / probs.sum(dim=-1, keepdim=True)
                sampled = torch.multinomial(probs, 1)
                random_sample = torch.rand(B, device=device) < p
                next_tokens = torch.where(random_sample.unsqueeze(1), sampled, argmax)
            else:
                next_tokens = argmax
        elif mode == "sample":
            mv32 = masked_values.float()
            mv32 = torch.nan_to_num(mv32, nan=-1e30, posinf=1e30, neginf=-1e30)
            probs = F.softmax(mv32, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.25, posinf=0.0, neginf=0.0)
            probs = probs.clamp(min=1e-8)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            next_tokens = torch.multinomial(probs, 1)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if fixed is not None:
            # If this step lands on a fixed (in-painted) position, override
            # the sampled token with the constrained nucleotide. KV cache
            # stays consistent because we feed `prev_tok = next_tokens`
            # next iteration regardless.
            fixed_at_step = fixed.gather(1, cur_rna_pos.unsqueeze(1))
            is_fixed = fixed_at_step >= 0
            next_tokens = torch.where(is_fixed, fixed_at_step, next_tokens)
        rna_outputs.scatter_(1, cur_rna_pos.unsqueeze(1), next_tokens)
        A_cnt = A_cnt + (next_tokens.squeeze(-1) == 0).cpu().float().numpy()
        prev_tok = next_tokens                          # feed at next step

    return rna_outputs


def generate_sequence_batched(
    model, src, target_correspondence, ct_matrix, start_token=4, p=0.1, max_len=None
):
    _, seq_len = src.shape
    batch_size = len(target_correspondence)
    model.eval()

    with torch.no_grad():
        memory = model.encoder(src, ct_matrix)
        if len(memory) == 1:
            memory = memory.expand(batch_size, *memory[0].shape)
        tgt = torch.full(
            (batch_size, 1), start_token, dtype=torch.long, device=src.device
        )
        outputs = torch.full(
            (batch_size, 1), start_token, dtype=torch.long, device=src.device
        )
        A_cnt = np.zeros(len(target_correspondence))
        max_len = max_len or seq_len
        past_key_values = None
        total_values = torch.zeros(len(target_correspondence), device=src.device)
        for position in range(max_len):
            tgt_mask = torch.triu(
                torch.full(
                    (tgt.size(1), tgt.size(1)), float("-inf"), device=tgt.device
                ),
                diagonal=1,
            )
            paired_encoding = (src == 0).long()
            # Identity perm: L→R decoding (legacy generate_* functions).
            # The new decoder always requires perm; this preserves the original
            # left-to-right inference behavior.
            B_, L_ = src.shape
            id_perm = torch.arange(L_, device=src.device).unsqueeze(0).expand(B_, L_)
            out, past_key_values = model.decoder(
                tgt,
                paired_encoding,
                memory,
                perm=id_perm,
                past_key_values=past_key_values,
                tgt_mask=tgt_mask,
                use_cache=True,
            )
            values = out[:, -1, :]

            # Apply base pair constraints
            mask = create_base_pair_mask_batched(
                position, target_correspondence, outputs, A_cnt, max_len
            )
            masked_values = values.masked_fill(mask, float("-inf"))

            # Sample from the policy
            random_sample = torch.rand(batch_size, device=src.device) < p
            probs = F.softmax(masked_values / 10000000, dim=-1)
            sampled_tokens = torch.multinomial(probs, 1)
            max_tokens = torch.argmax(masked_values, dim=-1, keepdim=True)
            next_tokens = torch.where(
                random_sample.unsqueeze(1), sampled_tokens, max_tokens
            )
            total_values = total_values + values.gather(-1, next_tokens).squeeze(-1)

            outputs = torch.cat([outputs, next_tokens], dim=1)

            A_cnt = A_cnt + (next_tokens.squeeze() == 0).cpu().float().numpy()
            tgt = torch.cat([tgt, next_tokens], dim=1)

    return outputs[:, 1:]  # , total_values  # Remove start token


def generate_sequence_batched_sample(
    model, src, target_correspondence, ct_matrix, start_token=4, p=1.0, max_len=None
):
    _, seq_len = src.shape
    batch_size = len(target_correspondence)
    model.eval()

    with torch.no_grad():
        memory = model.encoder(src, ct_matrix)
        if len(memory) == 1:
            memory = memory.expand(batch_size, *memory[0].shape)
        tgt = torch.full(
            (batch_size, 1), start_token, dtype=torch.long, device=src.device
        )
        outputs = torch.full(
            (batch_size, 1), start_token, dtype=torch.long, device=src.device
        )
        A_cnt = np.zeros(len(target_correspondence))
        max_len = max_len or seq_len
        past_key_values = None
        total_values = torch.zeros(len(target_correspondence), device=src.device)
        for position in range(max_len):
            tgt_mask = torch.triu(
                torch.full(
                    (tgt.size(1), tgt.size(1)), float("-inf"), device=tgt.device
                ),
                diagonal=1,
            )
            paired_encoding = (src == 0).long()
            # Identity perm: L→R decoding (legacy generate_* functions).
            # The new decoder always requires perm; this preserves the original
            # left-to-right inference behavior.
            B_, L_ = src.shape
            id_perm = torch.arange(L_, device=src.device).unsqueeze(0).expand(B_, L_)
            out, past_key_values = model.decoder(
                tgt,
                paired_encoding,
                memory,
                perm=id_perm,
                past_key_values=past_key_values,
                tgt_mask=tgt_mask,
                use_cache=True,
            )
            values = out[:, -1, :]

            # Apply base pair constraints
            mask = create_base_pair_mask_batched(
                position, target_correspondence, outputs, A_cnt, max_len
            )
            masked_values = values.masked_fill(mask, float("-inf"))

            # Sample from the policy
            random_sample = torch.rand(batch_size, device=src.device) < p
            probs = F.softmax(masked_values, dim=-1)
            sampled_tokens = torch.multinomial(probs, 1)
            max_tokens = torch.argmax(masked_values, dim=-1, keepdim=True)
            next_tokens = torch.where(
                random_sample.unsqueeze(1), sampled_tokens, max_tokens
            )
            total_values = total_values + values.gather(-1, next_tokens).squeeze(-1)

            outputs = torch.cat([outputs, next_tokens], dim=1)

            A_cnt = A_cnt + (next_tokens.squeeze() == 0).cpu().float().numpy()
            tgt = torch.cat([tgt, next_tokens], dim=1)

    return outputs[:, 1:]  # , total_values  # Remove start token


def generate_sequence_batched_accelerate(
    model, src, target_correspondence, ct_matrix, start_token=4, p=0.1, max_len=None
):
    _, seq_len = src.shape
    batch_size = len(target_correspondence)
    model.eval()

    with torch.no_grad():
        memory = model.module.encoder(src, ct_matrix)
        if len(memory) == 1:
            memory = memory.expand(batch_size, *memory[0].shape)
        tgt = torch.full(
            (batch_size, 1), start_token, dtype=torch.long, device=src.device
        )
        outputs = torch.full(
            (batch_size, 1), start_token, dtype=torch.long, device=src.device
        )
        A_cnt = np.zeros(src.shape[0])
        max_len = max_len or seq_len
        past_key_values = None
        total_values = torch.zeros(len(target_correspondence), device=src.device)
        for position in range(max_len):
            tgt_mask = torch.triu(
                torch.full(
                    (tgt.size(1), tgt.size(1)), float("-inf"), device=tgt.device
                ),
                diagonal=1,
            )
            paired_encoding = (src == 0).long()
            B_, L_ = src.shape
            id_perm = torch.arange(L_, device=src.device).unsqueeze(0).expand(B_, L_)
            out, past_key_values = model.module.decoder(
                tgt,
                paired_encoding,
                memory,
                perm=id_perm,
                past_key_values=past_key_values,
                tgt_mask=tgt_mask,
                use_cache=True,
            )
            values = out[:, -1, :]

            # Apply base pair constraints
            mask = create_base_pair_mask_batched(
                position, target_correspondence, outputs, A_cnt, max_len
            )
            masked_values = values.masked_fill(mask, float("-inf"))

            # Sample from the policy
            random_sample = torch.rand(batch_size, device=src.device) < p
            probs = F.softmax(masked_values / 10000000, dim=-1)
            sampled_tokens = torch.multinomial(probs, 1)
            max_tokens = torch.argmax(masked_values, dim=-1, keepdim=True)
            next_tokens = torch.where(
                random_sample.unsqueeze(1), sampled_tokens, max_tokens
            )
            total_values = total_values + values.gather(-1, next_tokens).squeeze(-1)

            outputs = torch.cat([outputs, next_tokens], dim=1)

            A_cnt = A_cnt + (next_tokens.squeeze() == 0).cpu().float().numpy()
            tgt = torch.cat([tgt, next_tokens], dim=1)

    return outputs[:, 1:]  # , total_values  # Remove start token


def create_base_pair_mask(position, target_correspondence, outputs, A_cnt, max_len):
    mask = torch.zeros(4, dtype=torch.bool)  # Default to no mask (all False)
    if position in target_correspondence:
        paired_position = target_correspondence[position]
        if position > paired_position:
            # We're on the right side of the pair, so we need to mask based on the left side
            left_base = outputs[paired_position + 1]
            # ACGU
            if left_base == 0:  # A
                mask[0:3] = True  # Only U is allowed
            elif left_base == 1:  # C
                mask[0:2] = True  # Only G is allowed
                mask[3] = True
            elif left_base == 2:  # G
                mask[0] = True  # C and U are allowed
                mask[2] = True
            elif left_base == 3:  # U
                mask[1:4] = True  # Only A is allowed
    # repeat constraint
    # The OpenTB constraints are no more than 3G, 4C, 4A and <40% adenine.

    max_repeat = [4, 4, 3, 5]  # ACGU
    if position > 5:

        for nt in range(4):
            last = outputs[-max_repeat[nt] :]
            if (last == nt).sum() == max_repeat[nt]:
                mask[nt] = True
                break
        if mask.sum() == 4:
            mask[nt] = False

    if A_cnt > max_len * 0.4:
        mask[0] = True
        if mask.sum() == 4:
            mask[0] = False

    return mask


def create_base_pair_mask_batched(
    position, target_correspondence, outputs, A_cnt, max_len
):
    batch_size = outputs.shape[0]
    masks = []

    for i in range(batch_size):
        mask = create_base_pair_mask(
            position, target_correspondence[i], outputs[i], A_cnt[i], max_len
        )
        masks.append(mask)

    return torch.stack(masks).to(outputs.device)


import csv


def log_rewards(file_path, episode, train_rewards, test_rewards):
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([episode, train_rewards, test_rewards])


def generate_sequence_topk(
    model, src, target_correspondence, ct_matrix, start_token=4, k=5, max_len=None
):
    seq_len = src.shape[1]
    model.eval()

    with torch.no_grad():
        memory = model.encoder(src, ct_matrix)
        _, L, C = memory.shape
        tgt = torch.full((1, 1), start_token, dtype=torch.long, device=src.device)
        candidates = [
            {"sequence": tgt[0], "previous_value": 0.0, "logits": 0.0, "A_cnt": 0.0}
        ]

        max_len = max_len or seq_len
        gamma = torch.linspace(0.25, 0.0, max_len)
        for position in range(max_len):
            # Stack all candidate sequences
            tgt = torch.stack([c["sequence"] for c in candidates])
            cumulative_logits = torch.tensor(
                [c["logits"] for c in candidates], device=src.device
            )

            # Create tgt_mask for all candidates
            tgt_mask = torch.triu(
                torch.full(
                    (tgt.size(1), tgt.size(1)), float("-inf"), device=tgt.device
                ),
                diagonal=1,
            )
            # tgt_mask = tgt_mask.unsqueeze(0).expand(tgt.size(0), -1, -1)

            # Expand src and memory for batch processing
            src_expanded = src.expand(tgt.size(0), -1)
            memory_expanded = memory.expand(tgt.size(0), L, C)
            paired_encoding = (src_expanded == 0).long()

            # Process all candidates in parallel
            out = model.decoder(
                tgt, paired_encoding, memory_expanded, tgt_mask=tgt_mask
            )
            values = out[:, -1, :]

            # Apply base pair constraints
            A_cnt = [a["A_cnt"] for a in candidates]
            mask = create_base_pair_mask_batched(
                position, [target_correspondence[0]] * len(tgt), tgt, A_cnt, max_len
            )
            masked_values = values.masked_fill(mask, float("-inf"))


            # Consider all 4 RNA nucleotides for all candidates
            # new_logits = masked_values + cumulative_logits.unsqueeze(1)
            # top_values, top_indices = torch.topk(new_logits.view(-1), k)

            # Create new candidates
            new_candidates = []
            for i in range(len(candidates)):
                # for value, index in zip(top_values, top_indices):
                sequence, previous_value, logits, A_cnt = (
                    candidates[i]["sequence"],
                    candidates[i]["previous_value"],
                    candidates[i]["logits"],
                    candidates[i]["A_cnt"],
                )

                for nucleotide_idx in range(4):
                    # candidate_idx = index // 4
                    # nucleotide_idx = index % 4
                    new_sequence = torch.cat(
                        [sequence, torch.tensor([nucleotide_idx], device=tgt.device)]
                    )
                    new_candidates.append(
                        {
                            "sequence": new_sequence,
                            "previous_value": masked_values[i, nucleotide_idx].item(),
                            "logits": logits + masked_values[i, nucleotide_idx].item(),
                            "A_cnt": A_cnt + (nucleotide_idx == 0),
                        }
                    )
            # +(1-gamma[position])*masked_values[i,nucleotide_idx].item()
            candidates = sorted(
                new_candidates, key=lambda x: x["logits"], reverse=True
            )[:k]
        # Stack all final candidate sequences
        top_k_sequences = torch.stack([c["sequence"] for c in candidates])
    return top_k_sequences[:, 1:]  # Remove start token and return k x L tensor


def bps2set(bps):
    return set([tuple(bp) for bp in bps])


def get_rescue_sequences(target_dbn, design_dbn, design_sequence):

    target_bps = bps2set(
        convert_dotbracket_to_bp_list(target_dbn, allow_pseudoknots=True)
    )
    design_bps = bps2set(
        convert_dotbracket_to_bp_list(design_dbn, allow_pseudoknots=True)
    )

    paired_bp_candidates = ["AU", "UA", "GC", "CG", "GU", "UG"]

    unpaired_bp_candidates = [
        "AA",
        "AG",
        "AC",
        "UU",
        "UC",
        "GA",
        "GG",
        "CA",
        "CU",
        "CC",
    ]

    # generate new candidates
    missing_bps = [bp for bp in target_bps if bp not in design_bps]
    extra_bps = [bp for bp in design_bps if bp not in target_bps]

    # missing_bps.append((30,31))
    # extra_bps.append((2,32))


    if (len(missing_bps) + len(extra_bps)) > 3:
        return []

    candidates = []

    for i, j in missing_bps:
        new_candidates = []
        if len(candidates) > 0:
            for c in candidates:
                for pairs in paired_bp_candidates:
                    new_c = c[:]
                    new_c.append((i, j, pairs))
                    new_candidates.append(new_c)
        else:
            for pairs in paired_bp_candidates:
                new_c = []
                new_c.append((i, j, pairs))
                new_candidates.append(new_c)

        candidates = new_candidates

    for i, j in extra_bps:

        new_candidates = []
        if len(candidates) > 0:
            for c in candidates:
                for pairs in unpaired_bp_candidates:
                    new_c = c[:]
                    new_c.append((i, j, pairs))
                    new_candidates.append(new_c)
        else:
            for pairs in unpaired_bp_candidates:
                new_c = []
                new_c.append((i, j, pairs))
                new_candidates.append(new_c)

        candidates = new_candidates

    design_sequence = list(design_sequence)
    new_sequences = []
    for c in candidates:
        new_sequence = design_sequence[:]
        for i, j, pair in c:
            new_sequence[i] = pair[0]
            new_sequence[j] = pair[1]
        new_sequences.append("".join(new_sequence))

    new_sequences = list(set(new_sequences))

    return new_sequences


def hamming_distance(str1, str2):
    # Ensure strings are of the same length
    if len(str1) != len(str2):
        raise ValueError(
            "Strings must be of the same length to compute Hamming distance."
        )

    # Calculate the Hamming distance
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))
