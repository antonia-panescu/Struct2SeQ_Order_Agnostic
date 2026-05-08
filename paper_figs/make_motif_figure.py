"""Single workshop-paper figure illustrating motif-based in-painting.

Three panels stacked vertically. Each panel uses a fixed pixel canvas so
the long arcs across the 59-nt PreQ1-II switch don't fight with text.

Output: paper_figs/fig_motif_inpainting.{pdf,png}.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("ARNIEFILE", str(PROJECT_ROOT / "arnie_file.txt"))
from arnie.utils import convert_dotbracket_to_bp_list  # noqa: E402

sys.path.insert(0, "/home/nvidia/haiwen/antonia/OpenKnotScorePipeline/src")
from openknotscore.pipeline.scoring import identify_crossing_bps  # noqa: E402

from evaluation.motif_extraction import extract_motifs  # noqa: E402


ORANGE = "#F4A261"
PURE_ORANGE = "#E76F51"
NESTED = "#264653"
PSEUDO = "#9D4EDD"
GREEN_OK = "#2A9D8F"
RED_FAIL = "#D62828"
GRAY = "#6C757D"


def _semicircle(ax, x1, x2, *, above=True, height_unit=1.0,
                color="black", lw=0.85, zorder=2):
    """Draw a semicircular arc between (x1, 0) and (x2, 0).

    `height_unit` is the y-unit height of the peak (independent of x-span,
    which is what the user wants when aspect != equal so all arcs have the
    same visual height regardless of base-pair distance).

    Actually for an arc-diagram, peak height is *proportional* to span so
    longer pairs visually stand out. Use peak_h = (x2-x1) * height_unit.
    """
    if x1 > x2:
        x1, x2 = x2, x1
    cx = (x1 + x2) / 2
    r = (x2 - x1) / 2
    h = r * height_unit
    sign = 1 if above else -1
    n = 60
    t = np.linspace(0, np.pi, n)
    xs = cx - r * np.cos(t)
    ys = sign * h * np.sin(t)
    ax.plot(xs, ys, color=color, linewidth=lw, zorder=zorder,
            solid_capstyle="round")


def _draw_panel(ax, dot_bracket, motif_positions=None, designed_seq=None,
                outcome=None, panel_label=None, panel_subtitle=None,
                ar_overlay=False, *, max_arc_h=18.0, show_seq_track=False):
    """Render one panel into ax. The y-budget is fixed so panels stack
    cleanly: arcs live in y∈[0, max_arc_h] above and y∈[-max_arc_h, 0]
    below; title sits at y=max_arc_h + 2; designed-seq sits at
    y = -(max_arc_h + 2)."""
    L = len(dot_bracket)
    bps = convert_dotbracket_to_bp_list(dot_bracket, allow_pseudoknots=True)
    crossed = set(identify_crossing_bps(bps))

    # 1. orange motif highlight — tall vertical band so motif region is
    # visually obvious and overlaps the arcs that participate in it.
    if motif_positions:
        # Use one wide band per contiguous run of motif positions to look
        # cleaner than per-position rectangles.
        sorted_pos = sorted(motif_positions)
        runs = []
        cur_start = sorted_pos[0]
        cur_end = sorted_pos[0]
        for p in sorted_pos[1:]:
            if p == cur_end + 1:
                cur_end = p
            else:
                runs.append((cur_start, cur_end))
                cur_start = cur_end = p
        runs.append((cur_start, cur_end))
        for s, e in runs:
            ax.add_patch(Rectangle(
                (s - 0.5, -max_arc_h - 0.4), e - s + 1.0,
                2 * max_arc_h + 0.8,
                facecolor=ORANGE, edgecolor="none", alpha=0.30, zorder=0,
            ))
        # Also a deeper-colored thin strip on the baseline for emphasis
        for s, e in runs:
            ax.add_patch(Rectangle(
                (s - 0.5, -0.55), e - s + 1.0, 1.10,
                facecolor=ORANGE, edgecolor="none", alpha=0.95, zorder=1.5,
            ))

    # 2. baseline + ticks
    ax.plot([-0.5, L - 0.5], [0, 0], color="black", linewidth=0.8, zorder=1)
    for i in range(0, L, 10):
        ax.plot([i, i], [-0.22, 0.22], color="black", linewidth=0.5, zorder=1)
        ax.text(i, -0.85, str(i + 1), ha="center", va="top",
                fontsize=5.5, color=GRAY)

    # 3. arcs — uniform height_unit so peak ∝ span
    span_to_h = max_arc_h / max(1, max((j - i) for i, j in bps))
    for (i, j) in bps:
        i, j = min(i, j), max(i, j)
        is_pseudo = i in crossed or j in crossed
        color = PSEUDO if is_pseudo else NESTED
        _semicircle(ax, i, j, above=not is_pseudo,
                    height_unit=span_to_h,
                    color=color, lw=1.0,
                    zorder=2 + (1 if is_pseudo else 0))

    # 4. AR overlay
    if ar_overlay and motif_positions:
        mleft = min(motif_positions)
        # Arrow well above the arcs
        y_arrow = max_arc_h + 1.2
        arrow = FancyArrowPatch(
            (-0.5, y_arrow), (mleft - 0.5, y_arrow),
            arrowstyle="->", mutation_scale=12, color=GRAY, linewidth=1.4,
        )
        ax.add_patch(arrow)
        ax.text((mleft - 0.5) / 2, y_arrow + 0.4, "L→R decoder",
                ha="center", va="bottom", fontsize=6.5, color=GRAY,
                style="italic")
        # ?'s above each motif position
        for p in motif_positions:
            ax.text(p, y_arrow, "?", ha="center", va="center",
                    fontsize=7, color=RED_FAIL, weight="bold")

    # 5. designed-sequence track below
    if designed_seq is not None:
        track_y = -(max_arc_h + 1.5)
        ax.plot([-0.5, L - 0.5], [track_y, track_y],
                color="black", linewidth=0.6)
        for i, ch in enumerate(designed_seq):
            in_motif = motif_positions and i in motif_positions
            color = PURE_ORANGE if in_motif else "black"
            weight = "bold" if in_motif else "normal"
            ax.text(i, track_y - 0.9, ch, ha="center", va="top",
                    fontsize=5.6, family="monospace",
                    color=color, weight=weight)
        ax.text(-1.0, track_y - 0.9, "design:", ha="right", va="top",
                fontsize=6.5, color=GRAY, style="italic")

    # 6. outcome badge (top right)
    if outcome:
        kind, text = outcome
        col = GREEN_OK if kind == "success" else RED_FAIL
        sym = "✓" if kind == "success" else "✗"
        ax.text(L - 1, max_arc_h + (3.0 if ar_overlay else 1.6),
                f"{sym} {text}",
                fontsize=7.5, color=col, weight="bold",
                ha="right", va="bottom")

    # 7. panel label + subtitle (top, well above arcs)
    title_y = max_arc_h + (3.0 if ar_overlay else 1.6)
    if panel_label:
        ax.text(-2.5, title_y, panel_label, fontsize=10, weight="bold",
                color="black", ha="left", va="bottom")
    if panel_subtitle:
        ax.text(2.0, title_y, panel_subtitle, fontsize=7, color="black",
                ha="left", va="bottom")

    # 8. axes — fixed budget so panels are uniform
    pad_top = 4.5 if ar_overlay else 3.0
    pad_bot = 3.5 if designed_seq is not None else 1.5
    ax.set_xlim(-3.5, L + 1.5)
    ax.set_ylim(-(max_arc_h + pad_bot), max_arc_h + pad_top)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    puz = pd.read_csv(
        "/home/nvidia/haiwen/antonia/OpenKnotAIDesignData/Targets/Round3_targets.csv"
    )
    row = puz.iloc[4]
    title_txt = row["Title"]
    target_db = row["Dot-bracket"]
    L = int(row["Length"])

    motifs = extract_motifs(target_db)
    motifs = [m for m in motifs if m.size <= 0.5 * L]
    chosen = next(m for m in motifs if m.kind == "hairpin")
    motif_positions = set(chosen.positions)
    print(f"Demo: {title_txt}, L={L}, motif={chosen.kind}, "
          f"size={chosen.size}, span={chosen.span}")

    samples = pd.read_csv("results/ok7_eval/bidir_motifs_structural/samples.csv")
    sub = samples[(samples.puzzle_idx == 4)
                  & (samples.motif_kind == "hairpin")
                  & (samples.motif_size_nt == chosen.size)
                  & (samples.jaccard_vs_target == 1.0)]
    if len(sub) > 0:
        designed_seq = sub.iloc[0]["generated_sequence"]
        designed_jacc = float(sub.iloc[0]["jaccard_vs_target"])
    else:
        sub = samples[(samples.puzzle_idx == 4) & (samples.motif_kind == "hairpin")]
        best = sub.nlargest(1, "jaccard_vs_target").iloc[0]
        designed_seq = best["generated_sequence"]
        designed_jacc = float(best["jaccard_vs_target"])

    # Each panel needs ~y-range = 2*max_arc_h + ~6 for padding+text
    fig = plt.figure(figsize=(7.0, 9.0))
    gs = fig.add_gridspec(
        3, 1, height_ratios=[1.0, 1.15, 1.45], hspace=0.10,
        left=0.02, right=0.99, top=0.99, bottom=0.04,
    )

    # Panel (a)
    ax_a = fig.add_subplot(gs[0, 0])
    _draw_panel(
        ax_a, target_db, motif_positions=motif_positions,
        panel_label="(a)",
        panel_subtitle=f"Target: {title_txt} (L={L} nt). "
                       f"Hold the orange hairpin ({chosen.size} nt) to "
                       f"wild-type; design the rest.",
    )

    # Panel (b)
    ax_b = fig.add_subplot(gs[1, 0])
    _draw_panel(
        ax_b, target_db, motif_positions=motif_positions,
        ar_overlay=True,
        outcome=("fail", "incompatible"),
        panel_label="(b)",
        panel_subtitle="AR L→R: must commit to position 1 with no\n"
                       "knowledge of the downstream fixed motif.",
    )

    # Panel (c)
    ax_c = fig.add_subplot(gs[2, 0])
    _draw_panel(
        ax_c, target_db, motif_positions=motif_positions,
        designed_seq=designed_seq,
        outcome=("success", f"Jaccard = {designed_jacc:.2f}"),
        panel_label="(c)",
        panel_subtitle="Order-agnostic (ours): random-permutation\n"
                       "decoding places the motif first and designs around it.",
    )

    handles = [
        mpatches.Patch(facecolor=ORANGE, alpha=0.6, label="Fixed motif (WT)"),
        plt.Line2D([0], [0], color=NESTED, lw=2, label="Nested base pair"),
        plt.Line2D([0], [0], color=PSEUDO, lw=2, label="Pseudoknot base pair"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, 0.005))

    out = PROJECT_ROOT / "paper_figs" / "fig_motif_inpainting.pdf"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    print(f"Wrote {out}")
    print(f"Wrote {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
