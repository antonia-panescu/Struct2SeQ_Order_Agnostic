import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """
    Causal 1D Convolution layer - only operates on past elements in the sequence
    Input: (batch_size, sequence_length, channels)
    Output: (batch_size, sequence_length, out_channels)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding_size = (kernel_size - 1) * dilation

        # Create regular 1D convolution
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=0,  # We'll handle padding manually
        )

    def forward(self, x):
        # Input is BxLxC, conv expects BxCxL
        x = x.transpose(1, 2)

        # Add padding to the left/past
        padding = torch.zeros(
            x.shape[0],  # batch
            x.shape[1],  # channels
            self.padding_size,  # padding size
            device=x.device,
            dtype=x.dtype,
        )
        x_padded = torch.cat([padding, x], dim=-1)

        # Apply convolution
        out = self.conv(x_padded)

        # Return to BxLxC format
        return out.transpose(1, 2)


# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
        )


class GraphConv(nn.Module):
    def __init__(self, input_dim, dropout):
        super(GraphConv, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 4, input_dim),
            nn.Dropout(dropout),
        )
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)

    def forward(self, x, adj_matrix):
        res = x
        adj_matrix = adj_matrix / adj_matrix.sum(-1, keepdim=True)
        x = torch.matmul(adj_matrix, x)
        x = x + res
        res = x
        x = self.layer_norm1(x)
        x = self.ffn(x)
        x = x + res
        x = self.layer_norm2(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self, db_vocab_size, embed_size, num_layers, nhead, dim_feedforward, dropout
    ):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(db_vocab_size, embed_size)

        # self.positional_encoding = PositionalEncoding(
        #     embed_size, dropout=dropout)
        self.positional_encoding = nn.LSTM(embed_size, embed_size)
        self.norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

        self.encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layers, num_layers=num_layers
        )
        self.conv = nn.Sequential(
            nn.Conv1d(embed_size, embed_size, 5, padding=2), nn.Dropout(dropout)
        )

        self.gconv = GraphConv(embed_size, dropout)

    def forward(self, src, ct_matrix, src_mask=None, src_key_padding_mask=None):
        src = self.embedding(src)

        res = src
        src = self.dropout(self.positional_encoding(src)[0])
        src = self.conv(src.permute(0, 2, 1)).permute(0, 2, 1)
        src = self.norm(src + res)
        src = self.gconv(src, ct_matrix)
        # print(src.shape)
        # exit()

        return self.transformer_encoder(
            src, mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )


class Decoder(nn.Module):
    def __init__(
        self, rna_vocab_size, embed_size, num_layers, nhead, dim_feedforward, dropout
    ):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(rna_vocab_size, embed_size)
        self.paired_embedding = nn.Embedding(2, embed_size)
        # self.positional_encoding = PositionalEncoding(
        #     embed_size, dropout=dropout)
        self.positional_encoding = nn.LSTM(embed_size, embed_size)
        self.decoder_layers = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            self.decoder_layers, num_layers=num_layers
        )
        self.fc_out = nn.Linear(embed_size, rna_vocab_size - 1)

    def forward(
        self,
        tgt,
        paired_encoding,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        tgt = self.embedding(tgt)

        # embed paired or not
        L = tgt.shape[1] - 1
        tgt[:, 1:] = tgt[:, 1:] + self.paired_embedding(paired_encoding)[:, :L]

        tgt = self.positional_encoding(tgt)[0]

        output = self.transformer_decoder(
            tgt,
            memory,
            memory_mask=memory_mask,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.fc_out(output)


class RelativePositionBias(nn.Module):
    """T5-style relative-position attention bias.

    Produces per-head additive logit biases B[h, i, j] from the *RNA-position*
    distance between query position i and key position j. Used to give a
    permutation-equivariant decoder explicit position information without
    relying on step-order operators (LSTM / CausalConv) that bake in L→R
    inductive bias.

    For order-agnostic decoding, callers pass `q_positions = perm[:, :Lq]`
    and `k_positions = perm[:, :Lk]` (or `k_positions = arange(L_enc)` for
    cross-attention to the encoder).
    """

    def __init__(self, num_buckets: int = 64, max_distance: int = 512, n_heads: int = 12):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.n_heads = n_heads
        # learned bias per (bucket, head); init small so RPE doesn't dominate at start
        self.bias = nn.Embedding(num_buckets, n_heads)
        nn.init.normal_(self.bias.weight, std=0.02)

    @staticmethod
    def _bucket(relative_position: torch.Tensor, num_buckets: int, max_distance: int) -> torch.Tensor:
        """T5-style bidirectional bucketing.

        Half of `num_buckets` for negative offsets, half for positive.
        Within each half: half exact distances, half log-spaced.
        relative_position is a long tensor of any shape; returns the same
        shape with values in [0, num_buckets).
        """
        ret = torch.zeros_like(relative_position)
        n = -relative_position
        # bidirectional: encode sign in the high bits
        num_buckets_half = num_buckets // 2
        ret = ret + (n < 0).long() * num_buckets_half
        n = n.abs()

        max_exact = num_buckets_half // 2
        is_small = n < max_exact

        # log-spaced for distances >= max_exact
        n_float = n.float().clamp(min=max_exact)
        val_if_large = max_exact + (
            torch.log(n_float / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets_half - max_exact)
        ).long()
        val_if_large = val_if_large.clamp(max=num_buckets_half - 1)

        ret = ret + torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, q_positions: torch.Tensor, k_positions: torch.Tensor) -> torch.Tensor:
        """
        q_positions: [B, Lq] long — RNA positions of queries
        k_positions: [B, Lk] long — RNA positions of keys
        Returns:    [B, n_heads, Lq, Lk] — additive logit bias to add to QK^T
        """
        # rel_pos[b, i, j] = k_positions[b, j] - q_positions[b, i]
        rel_pos = k_positions[:, None, :] - q_positions[:, :, None]  # [B, Lq, Lk]
        buckets = self._bucket(rel_pos, self.num_buckets, self.max_distance)  # [B, Lq, Lk]
        # bias_table[buckets] -> [B, Lq, Lk, n_heads]; permute to [B, n_heads, Lq, Lk]
        return self.bias(buckets).permute(0, 3, 1, 2).contiguous()


class OptimizedTransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, nhead, dropout,
                 rpe_num_buckets: int = 64, rpe_max_distance: int = 512):
        super().__init__()
        embed_size = d_model
        self.nhead = nhead
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                OptimizedDecoderLayer(d_model, nhead, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        # self.norm = nn.LayerNorm(d_model)
        # NOTE: previous architecture had nn.LSTM positional encoder + CausalConv1d here.
        # Both are intrinsically L→R operators that produce nonsense step-order
        # representations under random-permutation decoding (the order-agnostic
        # training/inference regime). They are replaced with relative-position
        # attention biases that are evaluated in RNA-position space (independent
        # of decoding step). See `RelativePositionBias` above.
        self.fc_out = nn.Linear(embed_size, vocab_size - 1)
        self.paired_embedding = nn.Embedding(2, embed_size)

        # Two RPE tables: one for self-attention (decoder ↔ decoder), one for
        # cross-attention (decoder ↔ encoder). Separate parameters because the
        # two distributions differ (e.g. cross-attn keys are at fixed RNA
        # positions while self-attn keys are at permuted positions).
        self.rpe_self = RelativePositionBias(
            num_buckets=rpe_num_buckets, max_distance=rpe_max_distance, n_heads=nhead
        )
        self.rpe_cross = RelativePositionBias(
            num_buckets=rpe_num_buckets, max_distance=rpe_max_distance, n_heads=nhead
        )

    def forward(
        self,
        tgt,
        paired_encoding,
        memory,
        perm,
        encoder_perm=None,
        tgt_mask=None,
        past_key_values=None,
        use_cache=False,
    ):
        """
        Args:
            tgt: [B, Ltgt] long — shifted-right input tokens (start_token at slot 0).
                Ltgt is 1 in cached-decoding steps after the first.
            paired_encoding: [B, L] long — paired/unpaired indicator at the *RNA
                position being predicted* at each decoding step (perm[t]).
            memory: [B, L_enc, D] — encoder output in RNA-position order.
            perm: [B, L] long — RNA position predicted at each decoding step.
                Step t predicts the nucleotide at RNA position perm[:, t].
            encoder_perm: [B, L_enc] long, optional — RNA positions of encoder
                memory slots. Defaults to arange(L_enc) (encoder is always in
                RNA order).
            tgt_mask: [Ltgt, Ltgt] float — additive causal mask in step space
                (-inf upper triangle, 0 lower). When `past_key_values` is set
                we ignore this and let the cache+mask combination be implicit.
            past_key_values: list of (K, V) per layer, or None.
            use_cache: bool — if True, return (output, new_key_values).
        """
        tgt = self.embedding(tgt)
        L = tgt.shape[1]
        tgt = tgt + self.paired_embedding(paired_encoding)[:, :L]

        # All position information now lives in attention biases, not in tgt.

        # Decide which step indices the current `tgt` corresponds to.
        # `OptimizedDecoderLayer` always slices `tgt[:, -1]` as the query in the
        # cached path, regardless of the length of `tgt` it was given. So if
        # past_key_values is set, only the last step is being predicted now —
        # the rest is either replayed input or cached state.
        if past_key_values is not None:
            past_len = past_key_values[0][0].shape[1]    # number of keys already cached
            self_q_pos = perm[:, past_len : past_len + 1]              # [B, 1]
            self_k_pos = perm[:, : past_len + 1]                        # [B, past_len+1]
        else:
            self_q_pos = perm[:, :L]
            self_k_pos = perm[:, :L]

        # Self-attention bias: [B, n_heads, Lq, Lk]
        self_bias = self.rpe_self(self_q_pos, self_k_pos)

        # Cross-attention bias: encoder memory keys are at RNA positions 0..L_enc-1
        if encoder_perm is None:
            B = tgt.shape[0]
            L_enc = memory.shape[1]
            encoder_perm = torch.arange(L_enc, device=tgt.device).unsqueeze(0).expand(B, L_enc)
        cross_bias = self.rpe_cross(self_q_pos, encoder_perm)            # [B, n_heads, Lq, L_enc]

        # Combine self_bias with the causal mask. tgt_mask (additive [-inf, 0]
        # upper-triangular) only applies in the non-cached path; in the cached
        # path the layer slices the new query and the keys come from the cache,
        # so causal ordering is implicit.
        if tgt_mask is not None and past_key_values is None:
            self_attn_mask = tgt_mask.unsqueeze(0).unsqueeze(0) + self_bias
        else:
            self_attn_mask = self_bias

        # Flatten head dim into batch dim — nn.MultiheadAttention expects
        # attn_mask of shape [B*n_heads, Lq, Lk].
        B = tgt.shape[0]
        Lq = self_attn_mask.shape[2]
        Lk = self_attn_mask.shape[3]
        self_attn_mask = self_attn_mask.reshape(B * self.nhead, Lq, Lk)
        cross_attn_mask = cross_bias.reshape(B * self.nhead, Lq, cross_bias.shape[3])

        output = tgt
        new_key_values = [] if use_cache else None

        for idx, layer in enumerate(self.layers):
            layer_past = past_key_values[idx] if past_key_values is not None else None
            output, layer_key_values = layer(
                output, memory, self_attn_mask, cross_attn_mask, layer_past, use_cache
            )
            if use_cache:
                new_key_values.append(layer_key_values)

        # output = self.norm(output)
        output = self.fc_out(output)
        if use_cache:
            return output, new_key_values
        else:
            return output


class OptimizedDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, self_attn_mask=None, cross_attn_mask=None,
                past_key_value=None, use_cache=False):
        """
        self_attn_mask: [B*n_heads, Lq, Lk] additive logit-space mask combining
            causal mask (-inf upper triangle in non-cached path) with
            relative-position bias.
        cross_attn_mask: [B*n_heads, Lq, L_enc] additive logit-space cross-attn
            relative-position bias (no causal component).
        """
        if past_key_value is not None:
            tgt_q = tgt[:, -1, None]
            tgt_k, tgt_v = past_key_value
            tgt_k = torch.cat([tgt_k, tgt_q], dim=1)
            tgt_v = torch.cat([tgt_v, tgt_q], dim=1)
        else:
            tgt_q, tgt_k, tgt_v = tgt, tgt, tgt

        self_attn_output, _ = self.self_attn(
            tgt_q, tgt_k, tgt_v, attn_mask=self_attn_mask, need_weights=False
        )
        tgt_q = self.norm1(tgt_q + self_attn_output)

        mem_k, mem_v = memory, memory
        cross_attn_output, _ = self.multihead_attn(
            tgt_q, mem_k, mem_v, attn_mask=cross_attn_mask, need_weights=False
        )
        tgt_q = self.norm2(tgt_q + cross_attn_output)

        ff_output = self.ffn(tgt_q)
        tgt_q = self.norm3(tgt_q + ff_output)

        if use_cache:
            tgt_q = torch.cat([tgt[:, :-1], tgt_q], 1)

        if use_cache:
            return tgt_q, (tgt_k, tgt_v)
        else:
            return tgt_q, None


class DotBracketRNATransformer(nn.Module):
    def __init__(
        self,
        db_vocab_size,
        rna_vocab_size,
        embed_size,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        dropout,
    ):
        super(DotBracketRNATransformer, self).__init__()
        self.encoder = Encoder(
            db_vocab_size,
            embed_size,
            num_encoder_layers,
            nhead,
            dim_feedforward,
            dropout,
        )
        self.decoder = OptimizedTransformerDecoder(
            rna_vocab_size, embed_size, num_decoder_layers, nhead, dropout
        )

    def forward(
        self,
        src,
        ct_matrix,
        tgt,
        src_mask=None,
        tgt_mask=None,
        src_padding_mask=None,
        tgt_padding_mask=None,
        memory_key_padding_mask=None,
        paired_encoding=None,
        perm=None,
        encoder_perm=None,
    ):
        """
        perm: [B, L] long — RNA position predicted at each decoding step.
            If None, defaults to identity (L→R decoding); the decoder then
            behaves as a standard L→R transformer.
        encoder_perm: [B, L_enc] long — RNA positions of encoder memory slots.
            If None, defaults to arange(L_enc) (encoder always sees fixed order).
        """
        memory = self.encoder(
            src, ct_matrix, src_mask=src_mask, src_key_padding_mask=src_padding_mask
        )
        # paired_encoding lives in decoding-step space. Callers doing L→R decoding
        # can omit it; we then default to (src == 0) which equals the unpaired
        # indicator at RNA positions and matches identity-perm decoding.
        if paired_encoding is None:
            paired_encoding = (src == 0).long()
        if perm is None:
            B, L = src.shape
            perm = torch.arange(L, device=src.device).unsqueeze(0).expand(B, L)
        output = self.decoder(
            tgt, paired_encoding, memory, perm=perm,
            encoder_perm=encoder_perm, tgt_mask=tgt_mask,
        )
        return output + memory.mean() * 0


def generate_square_subsequent_mask(src):
    sz = src.shape[1]
    mask = (torch.triu(torch.ones((sz, sz), device=src.device)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def create_mask(src, tgt):
    PAD_IDX = 4
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=src.device).type(
        torch.bool
    )

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
