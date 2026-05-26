"""Generate the architecture figure for the Struct2SeQ-bidir paper.

Highlights what changed vs the original Struct2SeQ (Shujun et al.):

  NEW   : T5-style relative-position attention biases (RPE_self, RPE_cross),
          random-permutation decoding order, K-permutation tiling at training
          time, per-step paired_encoding gathered to the permutation.
  KEEP  : Encoder stack (LSTM-pos-enc + GraphConv + N x TransformerEncoder),
          per-step paired-vs-unpaired embedding addition, Q-learning TD loss.

Outputs:
    paper_figs/fig_architecture.pdf  (vector, for the paper)
    paper_figs/fig_architecture.png  (rasterised preview)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
COLOR_KEEP = "#E6EEF7"          # cool grey-blue: inherited from Struct2SeQ
EDGE_KEEP = "#5F7A9B"
COLOR_NEW = "#FFE8C2"           # warm cream: new in bidir
EDGE_NEW = "#C46A0F"
COLOR_INPUT = "#FFFFFF"
EDGE_INPUT = "#333333"
COLOR_DATA = "#F4F4F4"
TEXT_COLOR = "#1B1B1B"

FIG_W, FIG_H = 7.0, 9.4         # double-column friendly, tall


def box(ax, xy, wh, label, *, kind="keep", fontsize=9, weight="normal", subtext=None):
    """Draw a rounded rectangle with a centered label and optional subtext."""
    x, y = xy
    w, h = wh
    face = {"keep": COLOR_KEEP, "new": COLOR_NEW, "input": COLOR_INPUT, "data": COLOR_DATA}[kind]
    edge = {"keep": EDGE_KEEP, "new": EDGE_NEW, "input": EDGE_INPUT, "data": EDGE_INPUT}[kind]
    lw = 1.8 if kind == "new" else 1.1
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=lw, edgecolor=edge, facecolor=face, zorder=2,
    )
    ax.add_patch(patch)
    if subtext is None:
        ax.text(x + w / 2, y + h / 2, label,
                ha="center", va="center",
                fontsize=fontsize, weight=weight, color=TEXT_COLOR, zorder=3)
    else:
        ax.text(x + w / 2, y + h * 0.68, label,
                ha="center", va="center",
                fontsize=fontsize, weight=weight, color=TEXT_COLOR, zorder=3)
        ax.text(x + w / 2, y + h * 0.30, subtext,
                ha="center", va="center",
                fontsize=fontsize - 1.5, style="italic",
                color="#444444", zorder=3)


def arrow(ax, p0, p1, *, label=None, label_pos=0.5, style="-|>", lw=1.2, color="#222", side="right",
          dotted=False):
    arr = FancyArrowPatch(
        p0, p1,
        arrowstyle=style, mutation_scale=12, lw=lw,
        color=color, linestyle="dotted" if dotted else "solid",
        zorder=4, shrinkA=2, shrinkB=2,
    )
    ax.add_patch(arr)
    if label:
        mx = p0[0] + label_pos * (p1[0] - p0[0])
        my = p0[1] + label_pos * (p1[1] - p0[1])
        dx = 0.04 if side == "right" else -0.04
        ha = "left" if side == "right" else "right"
        ax.text(mx + dx, my, label, ha=ha, va="center",
                fontsize=7.5, color=color, zorder=5)


def bracket(ax, x, y0, y1, label, *, side="left", color=EDGE_NEW):
    """Vertical curly-bracket-ish indicator labelling a group of stacked boxes."""
    pad = 0.08
    if side == "left":
        ax.plot([x, x - pad, x - pad, x], [y0, y0, y1, y1],
                color=color, lw=1.2, zorder=4)
        ax.text(x - pad - 0.05, (y0 + y1) / 2, label,
                rotation=90, ha="right", va="center", fontsize=8.5,
                color=color, weight="bold")
    else:
        ax.plot([x, x + pad, x + pad, x], [y0, y0, y1, y1],
                color=color, lw=1.2, zorder=4)
        ax.text(x + pad + 0.05, (y0 + y1) / 2, label,
                rotation=-90, ha="left", va="center", fontsize=8.5,
                color=color, weight="bold")


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)
    ax.set_aspect("equal")
    ax.axis("off")

    # -----------------------------------------------------------------------
    # Top inputs
    # -----------------------------------------------------------------------
    box(ax, (0.3, 12.1), (4.2, 0.7),
        "Target structure (dot-bracket)",
        subtext="“((((....))))…”  [B, L]",
        kind="input", fontsize=9, weight="bold")
    box(ax, (5.5, 12.1), (4.2, 0.7),
        "Contact (CT) matrix",
        subtext="base-pair graph  [B, L, L]",
        kind="input", fontsize=9, weight="bold")

    # -----------------------------------------------------------------------
    # Encoder stack (KEEP)
    # -----------------------------------------------------------------------
    enc_x, enc_w = 0.5, 4.0
    box(ax, (enc_x, 11.0), (enc_w, 0.55),
        "DB token embedding", kind="keep")
    box(ax, (enc_x, 10.30), (enc_w, 0.55),
        "LSTM positional encoder", kind="keep")
    box(ax, (enc_x, 9.60), (enc_w, 0.55),
        "Conv1d (k=5) + Dropout", kind="keep")
    box(ax, (enc_x, 8.90), (enc_w, 0.55),
        "LayerNorm  (residual)", kind="keep")
    box(ax, (enc_x, 8.10), (enc_w, 0.65),
        "GraphConv  (GeM pooling)",
        subtext="message-passing over CT matrix",
        kind="keep", fontsize=9)
    box(ax, (enc_x, 7.20), (enc_w, 0.70),
        r"$N_{\mathrm{enc}}\times$  TransformerEncoderLayer",
        subtext="multi-head self-attn  +  FFN",
        kind="keep", fontsize=9)

    # Encoder boundary bracket
    bracket(ax, enc_x - 0.05, 7.20, 11.55,
            "Encoder  (unchanged)", side="left",
            color=EDGE_KEEP)

    # CT-matrix arrow into GraphConv
    arrow(ax, (7.6, 12.1), (enc_x + enc_w - 0.5, 8.55),
          style="-|>", color=EDGE_KEEP, lw=1.0,
          label="ct_matrix", side="right")

    # Structure -> embedding
    arrow(ax, (2.4, 12.1), (2.5, 11.55),
          style="-|>", color=EDGE_KEEP)

    # Vertical chain inside encoder
    for y0, y1 in [(11.0, 10.85), (10.30, 10.15), (9.60, 9.45),
                   (8.90, 8.75), (8.10, 7.90)]:
        arrow(ax, (enc_x + enc_w / 2, y0), (enc_x + enc_w / 2, y1),
              style="-|>", color=EDGE_KEEP, lw=1.0)

    # memory output
    arrow(ax, (enc_x + enc_w / 2, 7.20), (enc_x + enc_w / 2, 6.70),
          style="-|>", color=EDGE_KEEP, lw=1.2)
    ax.text(enc_x + enc_w / 2 + 0.06, 6.95,
            r"memory  $\in\mathbb{R}^{B\times L\times D}$",
            ha="left", va="center", fontsize=8.5, color="#333")

    # -----------------------------------------------------------------------
    # Decoder inputs (right column)
    # -----------------------------------------------------------------------
    dec_x, dec_w = 5.3, 4.4

    # Permutation pathway (NEW)
    box(ax, (dec_x, 11.55), (dec_w, 0.75),
        "Random permutation  perm[B, L]",
        subtext=r"$\mathrm{perm}[t] = $ RNA position decoded at step $t$",
        kind="new", weight="bold", fontsize=9)

    # K-perm tiling (NEW, training-only)
    box(ax, (dec_x, 10.55), (dec_w, 0.75),
        r"K-permutation tiling  ($K=k_{\mathrm{perm}}$)",
        subtext="training only — replicate each target K times",
        kind="new", weight="bold", fontsize=9)

    # Shifted sequence + paired_encoding (gathered through perm)
    box(ax, (dec_x, 9.65), (dec_w, 0.65),
        "Shifted-right tgt  +  paired_encoding",
        subtext=r"both gathered through $\mathrm{perm}$  (gather(1, perm))",
        kind="new", fontsize=9)

    arrow(ax, (dec_x + dec_w / 2, 11.55), (dec_x + dec_w / 2, 11.30),
          color=EDGE_NEW, lw=1.1)
    arrow(ax, (dec_x + dec_w / 2, 10.55), (dec_x + dec_w / 2, 10.30),
          color=EDGE_NEW, lw=1.1)

    # -----------------------------------------------------------------------
    # Decoder body
    # -----------------------------------------------------------------------
    body_y_top, body_y_bot = 9.10, 3.10
    box(ax, (dec_x - 0.05, body_y_bot), (dec_w + 0.10, body_y_top - body_y_bot),
        "", kind="data")
    # token embed + paired_embed
    box(ax, (dec_x + 0.20, 8.40), (dec_w - 0.40, 0.55),
        "Token embedding + paired_embedding", kind="keep", fontsize=9)
    arrow(ax, (dec_x + dec_w / 2, 9.65), (dec_x + dec_w / 2, 8.95),
          color=EDGE_NEW, lw=1.1)
    # downstream arrow: token-embed -> N x layers (skipping the removed block).
    # nl_y_top = 7.50 (defined below); reference the literal here to avoid forward-ref.
    arrow(ax, (dec_x + dec_w / 2, 8.40), (dec_x + dec_w / 2, 7.50),
          color=EDGE_KEEP, lw=1.0)

    # REMOVED block (greyed out, no strikethrough — keeps text legible)
    box(ax, (dec_x + 0.20, 7.70), (dec_w - 0.40, 0.55),
        "", kind="data")
    ax.text(dec_x + dec_w / 2, 7.97,
            r"$\bf{removed:}$  LSTM pos-enc  +  CausalConv1d",
            ha="center", va="center",
            fontsize=8.5, color="#888")
    ax.text(dec_x + dec_w / 2, 7.55,
            "(both bake in L→R; replaced by RPE below)",
            ha="center", va="center",
            fontsize=7.0, color="#888", style="italic")

    # N x OptimizedDecoderLayer
    nl_y_top, nl_y_bot = 7.50, 4.40
    box(ax, (dec_x + 0.10, nl_y_bot), (dec_w - 0.20, nl_y_top - nl_y_bot),
        "", kind="data")
    ax.text(dec_x + dec_w / 2, nl_y_top - 0.20,
            r"$N_{\mathrm{dec}}\times$  OptimizedDecoderLayer",
            ha="center", va="top", fontsize=9.5, weight="bold", color=TEXT_COLOR)

    # Inside the layer: self-attn + RPE_self
    sa_y = 6.45
    box(ax, (dec_x + 0.30, sa_y), (dec_w - 0.60, 0.55),
        "Self-attention", kind="keep", fontsize=9)
    # RPE_self callout on the right
    box(ax, (dec_x + dec_w + 0.10, sa_y - 0.05), (1.40, 0.65),
        r"$+$  RPE$_{\!\mathrm{self}}$",
        subtext=r"bias($\mathrm{perm}[t_1]\!\!-\!\mathrm{perm}[t_2]$)",
        kind="new", weight="bold", fontsize=9)
    arrow(ax, (dec_x + dec_w + 0.10, sa_y + 0.27),
          (dec_x + dec_w - 0.30, sa_y + 0.27),
          color=EDGE_NEW, lw=1.2)

    # Causal mask callout
    ax.text(dec_x + dec_w / 2, sa_y - 0.18,
            "+ causal mask (in step-space)",
            ha="center", va="top", fontsize=7.5, color="#555", style="italic")

    arrow(ax, (dec_x + dec_w / 2, nl_y_top - 0.30),
          (dec_x + dec_w / 2, sa_y + 0.55), color=EDGE_KEEP, lw=1.0)

    # cross-attn + RPE_cross
    ca_y = 5.45
    box(ax, (dec_x + 0.30, ca_y), (dec_w - 0.60, 0.55),
        r"Cross-attention  $\rightarrow$  memory", kind="keep", fontsize=9)
    box(ax, (dec_x + dec_w + 0.10, ca_y - 0.05), (1.40, 0.65),
        r"$+$  RPE$_{\!\mathrm{cross}}$",
        subtext=r"bias($\mathrm{perm}[t]\!\!-\!\!j$)",
        kind="new", weight="bold", fontsize=9)
    arrow(ax, (dec_x + dec_w + 0.10, ca_y + 0.27),
          (dec_x + dec_w - 0.30, ca_y + 0.27),
          color=EDGE_NEW, lw=1.2)
    arrow(ax, (dec_x + dec_w / 2, sa_y), (dec_x + dec_w / 2, ca_y + 0.55),
          color=EDGE_KEEP, lw=1.0)

    # encoder_perm
    box(ax, (dec_x + dec_w + 0.10, ca_y - 0.95), (1.40, 0.55),
        r"encoder_perm",
        subtext=r"$= \mathrm{arange}(L_{\mathrm{enc}})$",
        kind="data", fontsize=8)
    arrow(ax, (dec_x + dec_w + 0.80, ca_y - 0.40),
          (dec_x + dec_w + 0.80, ca_y - 0.05),
          color="#888", lw=0.9, dotted=True)

    # FFN + LayerNorm
    ffn_y = 4.55
    box(ax, (dec_x + 0.30, ffn_y), (dec_w - 0.60, 0.50),
        "FFN  +  LayerNorm", kind="keep", fontsize=9)
    arrow(ax, (dec_x + dec_w / 2, ca_y), (dec_x + dec_w / 2, ffn_y + 0.50),
          color=EDGE_KEEP, lw=1.0)

    # encoder memory feeding cross-attn
    arrow(ax, (enc_x + enc_w, 6.70), (dec_x + 0.30, ca_y + 0.27),
          style="-|>", color=EDGE_KEEP, lw=1.4,
          label="memory  K, V", label_pos=0.55, side="right")

    # fc_out
    box(ax, (dec_x + 0.10, 3.30), (dec_w - 0.20, 0.55),
        r"fc_out  $\rightarrow$  Q-values  $Q[t, a]$",
        kind="keep", fontsize=9.2, weight="bold")
    arrow(ax, (dec_x + dec_w / 2, nl_y_bot),
          (dec_x + dec_w / 2, 3.85), color=EDGE_KEEP, lw=1.0)

    # Q-learning TD update box at the bottom
    box(ax, (1.6, 1.85), (6.8, 0.95),
        r"Q-learning TD target:   $r_t^{\mathrm{perm}} + \gamma \max_{a} Q'(s_{t+1}, a)$",
        subtext=r"reward gathered through $\mathrm{perm}$;  $K$ replicas average gradient per target",
        kind="new", weight="bold", fontsize=10)
    arrow(ax, (dec_x + dec_w / 2, 3.30), (5.0, 2.80),
          color=EDGE_NEW, lw=1.2)

    # Decoder bracket
    bracket(ax, dec_x + dec_w + 1.65, 3.30, 11.55 + 0.75,
            "Decoder  (NEW components highlighted)",
            side="right", color=EDGE_NEW)

    # -----------------------------------------------------------------------
    # Legend
    # -----------------------------------------------------------------------
    lg_x, lg_y = 0.5, 0.55
    box(ax, (lg_x, lg_y - 0.05), (0.35, 0.30), "", kind="keep")
    ax.text(lg_x + 0.45, lg_y + 0.10,
            "Inherited from Struct2SeQ (Shujun et al.)",
            ha="left", va="center", fontsize=9)
    box(ax, (lg_x + 4.55, lg_y - 0.05), (0.35, 0.30), "", kind="new")
    ax.text(lg_x + 5.00, lg_y + 0.10,
            "New in this work (bidir / order-agnostic)",
            ha="left", va="center", fontsize=9, color=EDGE_NEW, weight="bold")

    # Title
    ax.text(5.0, 12.85, "Struct2SeQ-bidir  —  architecture",
            ha="center", va="bottom",
            fontsize=12, weight="bold", color=TEXT_COLOR)

    fig.tight_layout()

    out_dir = Path(__file__).resolve().parent
    pdf = out_dir / "fig_architecture.pdf"
    png = out_dir / "fig_architecture.png"
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    print(f"wrote {pdf}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
