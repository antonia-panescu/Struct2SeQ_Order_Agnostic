"""Cohesive paper-figure generator.

One script, one style, all figures rendered as high-quality SVG (with PNG
preview alongside). Designed to be paste-able into a Jupyter notebook
cell-by-cell — sections are separated by `# %%` markers.

Usage::

    python paper_figs/make_paper_figures.py --figs all
    python paper_figs/make_paper_figures.py --figs 1,3,5
    python paper_figs/make_paper_figures.py --list

Figures::

    1. methods_comparison_100mer  Headline bar: perfect-Jaccard rate by
                                  method at matched K=1000 on OK7 100mer.
    2. per_puzzle_100mer          Per-puzzle mean F1 / Jaccard across
                                  baselines (the OpenKnotBench grid).
    3. best_of_n                  Puzzles-solved fraction vs K, ours vs
                                  AR baseline, on 100mer.
    4. sample_efficiency          E[K*] aggregate and per-puzzle (geo
                                  mean) — the "moving away from BoN"
                                  punchline.
    5. apples_to_apples_240mer    240mer apples-to-apples: AR full
                                  Shujun pipeline (3-strategy + rescue)
                                  vs ours bidir_random argmax.
    6. arch_vs_order_240mer       240mer ablation: orig L->R, bidir L->R
                                  (identity), bidir random-perm.
    7. inpaint_ksweep             U-shape: random-scatter perfect-rate
                                  vs K_inpaint fraction-fixed.

Numbers come from `paper_figs/*.csv` where available; otherwise from
the headline numbers in `paper/ideas_for_writing.md`.

Style: black text, purple primary accent, grey for non-ours baselines,
sans-serif, minimal spines, light horizontal grid only.
"""
# %% imports
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    import scienceplots  # noqa: F401  (registers the 'science' style)
    _HAS_SCIENCE = True
except ImportError:
    _HAS_SCIENCE = False

warnings.filterwarnings("ignore")

# %% paths

HERE = Path(__file__).resolve().parent
OUT = HERE  # save SVG/PNG next to the CSVs
OUT.mkdir(exist_ok=True)


# %% style: cohesive purple-accent, black-text, minimal-spine
#
# Anchor colour PURPLE_PRIMARY is used for the "ours" headline series.
# Lighter purple shades are used for our ablation rows so the family is
# visually distinct from the grey baselines but still reads as one
# group. Baselines stay neutral grey so the eye is drawn to purple.

PURPLE_PRIMARY = "#6B21A8"   # deep, paper-friendly purple (ours, default)
PURPLE_MID     = "#8B5CF6"   # mid purple (ours, secondary)
PURPLE_LIGHT   = "#C4B5FD"   # light purple (ours, tertiary / ablations)
GREY_DARK      = "#374151"   # dark grey (text)
GREY_MID       = "#6B7280"   # medium grey (baselines, primary)
GREY_LIGHT     = "#9CA3AF"   # light grey (baselines, secondary)
GREY_FAINT     = "#D1D5DB"   # faint grey (gridlines, unsolved)
BLACK          = "#000000"

# A small named palette so figure functions can stay declarative.
PALETTE = {
    "ours_random":      PURPLE_PRIMARY,
    "ours_pairedfirst": PURPLE_MID,
    "ours_l2r":         PURPLE_LIGHT,
    "ours_motifA":      PURPLE_MID,
    "ours_motifB":      PURPLE_LIGHT,
    "ar_l2r":           GREY_MID,
    "ar_teacher":       GREY_LIGHT,
    "other_baseline":   GREY_FAINT,
}


def _set_style():
    """Single style block; safe to call repeatedly."""
    if _HAS_SCIENCE:
        plt.style.use(["science", "no-latex"])
    else:
        plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("paper")

    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size":          10,
        "axes.titlesize":     11,
        "axes.titleweight":   "bold",
        "axes.labelsize":     10,
        "axes.labelcolor":    BLACK,
        "axes.edgecolor":     BLACK,
        "axes.linewidth":     0.8,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.titlecolor":    BLACK,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "xtick.color":        BLACK,
        "ytick.color":        BLACK,
        "legend.fontsize":    9,
        "legend.frameon":     False,
        "legend.title_fontsize": 9,
        "grid.color":         GREY_FAINT,
        "grid.linewidth":     0.5,
        "grid.alpha":         0.7,
        "figure.dpi":         140,
        "savefig.bbox":       "tight",
        "savefig.transparent": False,
        "svg.fonttype":       "none",   # keep text as text in SVG
    })


def _save(fig, name):
    """Save SVG + PNG side by side."""
    svg = OUT / f"{name}.svg"
    png = OUT / f"{name}.png"
    fig.savefig(svg, format="svg")
    fig.savefig(png, format="png", dpi=300)
    print(f"wrote {svg.name}  +  {png.name}")


# %% fig 1 — headline methods comparison (100mer)

def fig1_methods_comparison():
    """Perfect-Jaccard rate by method at matched K=1000 on OK7 100mer.

    Data are the locked headline numbers from ideas_for_writing.md.
    Bars sorted by perfect-rate ascending so 'ours, random-perm' is
    visibly tallest.
    """
    rows = [
        ("Struct2SeQ AR (L→R)",                      21.2, "ar_l2r"),
        ("Struct2SeQ AR + teacher-forced motif",     33.4, "ar_teacher"),
        ("Ours, L→R from bidir checkpoint",          32.8, "ours_l2r"),
        ("Ours, Framing B (motif redesign)",         19.8, "ours_motifB"),
        ("Ours, Framing A (motif preservation)",     37.9, "ours_motifA"),
        ("Ours, random-perm decoding",               45.5, "ours_random"),
    ]
    df = pd.DataFrame(rows, columns=["method", "perfect_rate", "color_key"])
    df = df.sort_values("perfect_rate").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    colors = [PALETTE[k] for k in df["color_key"]]
    bars = ax.barh(df["method"], df["perfect_rate"], color=colors,
                   edgecolor=BLACK, linewidth=0.6, height=0.7)

    for bar, val in zip(bars, df["perfect_rate"]):
        ax.text(val + 0.6, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", ha="left",
                fontsize=9, color=BLACK)

    ax.set_xlim(0, max(df["perfect_rate"]) * 1.18)
    ax.set_xlabel("Perfect-Jaccard rate (%) at K=1000")
    ax.set_title("OK7 100mer: matched-budget comparison",
                 loc="left", pad=8)
    ax.xaxis.grid(True, linestyle="-", linewidth=0.5, alpha=0.7)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(left=False)

    _save(fig, "fig1_methods_comparison_100mer")
    plt.close(fig)


# %% fig 2 — per-puzzle bars (100mer)

def fig2_per_puzzle_100mer():
    """Per-puzzle mean F1 / Jaccard across all OpenKnotBench methods.

    Source: paper_figs/baseline_comparison.csv (already exists).
    Highlight 'Ours' in purple; everything else stays grey.
    """
    csv = HERE / "baseline_comparison.csv"
    if not csv.exists():
        print(f"  skipping fig2: {csv.name} not found")
        return
    df = pd.read_csv(csv)

    # Order methods: ours first (left, purple), baselines after
    method_order = ["Ours (top-20 of K=1000)",
                    "Struct2SeQ", "Struct2SeQ-SHAPE",
                    "gRNAde", "MPNN-fixbb", "MPNN-RFdiff", "Rosetta"]
    method_order = [m for m in method_order if m in df["method"].unique()]
    df["method"] = pd.Categorical(df["method"], method_order, ordered=True)
    df = df.sort_values(["method", "puzzle_idx"])

    puzzles = sorted(df["puzzle"].unique(), key=lambda s: int(s[1:]))
    n_methods = len(method_order)
    n_puzzles = len(puzzles)

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    width = 0.8 / n_methods
    xs = np.arange(n_puzzles)

    for i, m in enumerate(method_order):
        color = PURPLE_PRIMARY if "Ours" in m else GREY_MID if i == 1 else GREY_LIGHT
        if "Ours" not in m and i > 1:
            color = GREY_FAINT
        sub = df[df["method"] == m].set_index("puzzle").reindex(puzzles)
        ax.bar(xs + i * width, sub["mean_F1"].values,
               width=width, label=m, color=color,
               edgecolor=BLACK, linewidth=0.3)

    ax.set_xticks(xs + width * (n_methods - 1) / 2)
    ax.set_xticklabels(puzzles, rotation=45, ha="right")
    ax.set_ylabel("Mean F1 (Jaccard, base pairs)")
    ax.set_ylim(0, 1.0)
    ax.set_title("OK7 100mer: per-puzzle quality across methods",
                 loc="left", pad=8)
    ax.legend(loc="lower right", ncol=2, fontsize=7.5)
    ax.yaxis.grid(True, linestyle="-", linewidth=0.5, alpha=0.7)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    _save(fig, "fig2_per_puzzle_100mer")
    plt.close(fig)


# %% fig 3 — best-of-N saturation

def fig3_best_of_n():
    """Puzzles-solved fraction vs K, ours vs AR baseline.

    Source: paper_figs/best_of_n.csv. Plots 4 series:
        bidir_random, bidir_motifs_structural,
        orig_struct2seq_l2r, orig_struct2seq_motifs_structural
    """
    csv = HERE / "best_of_n.csv"
    if not csv.exists():
        print(f"  skipping fig3: {csv.name} not found")
        return
    df = pd.read_csv(csv)

    label_map = {
        "bidir_random":                       ("Ours, random-perm",     PURPLE_PRIMARY, "-"),
        "bidir_motifs_structural":            ("Ours, motif preservation (A)", PURPLE_MID, "--"),
        "orig_struct2seq_l2r":                ("Struct2SeQ AR (L→R)",   GREY_MID,       "-"),
        "orig_struct2seq_motifs_structural":  ("AR + teacher-forced motif", GREY_LIGHT, "--"),
    }

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for cfg, (label, color, ls) in label_map.items():
        sub = df[df["config"] == cfg].sort_values("N")
        if sub.empty:
            continue
        ax.plot(sub["N"], sub["frac_puzzles_with_perfect"] * 100,
                marker="o", markersize=4, color=color,
                linewidth=1.6, linestyle=ls, label=label)
        if "std" in sub.columns:
            ax.fill_between(sub["N"],
                            (sub["frac_puzzles_with_perfect"] - sub["std"]) * 100,
                            (sub["frac_puzzles_with_perfect"] + sub["std"]) * 100,
                            color=color, alpha=0.12, linewidth=0)

    ax.set_xscale("log")
    ax.set_xlabel("Sampling budget K (per puzzle)")
    ax.set_ylabel("Puzzles solved (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Best-of-N saturation, OK7 100mer",
                 loc="left", pad=8)
    ax.legend(loc="lower right", fontsize=7.5)
    ax.yaxis.grid(True, linestyle="-", linewidth=0.5, alpha=0.7)
    ax.xaxis.grid(True, linestyle="-", linewidth=0.5, alpha=0.4, which="both")
    ax.set_axisbelow(True)

    _save(fig, "fig3_best_of_n_100mer")
    plt.close(fig)


# %% fig 4 — sample efficiency E[K*]

def fig4_sample_efficiency():
    """Aggregate and geometric-mean per-puzzle E[K*] for ours vs AR L→R.

    The lower the bar, the more sample-efficient.
    """
    rows = [
        ("Aggregate",   "Struct2SeQ AR (L→R)",  4.7,  "ar_l2r"),
        ("Aggregate",   "Ours, random-perm",    2.2,  "ours_random"),
        ("Per-puzzle\n(geom. mean)", "Struct2SeQ AR (L→R)", 16.1, "ar_l2r"),
        ("Per-puzzle\n(geom. mean)", "Ours, random-perm",    4.3, "ours_random"),
    ]
    df = pd.DataFrame(rows, columns=["scope", "method", "ek", "color_key"])

    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    scope_order = ["Aggregate", "Per-puzzle\n(geom. mean)"]
    method_order = ["Struct2SeQ AR (L→R)", "Ours, random-perm"]
    width = 0.36
    xs = np.arange(len(scope_order))

    for i, m in enumerate(method_order):
        color = PALETTE["ar_l2r" if "Struct2SeQ" in m else "ours_random"]
        vals = [df[(df["scope"] == s) & (df["method"] == m)]["ek"].values[0]
                for s in scope_order]
        ax.bar(xs + (i - 0.5) * width, vals, width=width,
               color=color, edgecolor=BLACK, linewidth=0.6, label=m)
        for x, v in zip(xs + (i - 0.5) * width, vals):
            ax.text(x, v + max(vals) * 0.025, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=9, color=BLACK)

    ax.set_xticks(xs)
    ax.set_xticklabels(scope_order)
    ax.set_ylabel(r"$\mathbb{E}[K^\star]$  (samples to one perfect design)")
    ax.set_title("Sample efficiency, OK7 100mer", loc="left", pad=8)
    ax.set_ylim(0, max(df["ek"]) * 1.25)
    ax.yaxis.grid(True, linestyle="-", linewidth=0.5, alpha=0.7)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=8)
    ax.tick_params(bottom=False)

    _save(fig, "fig4_sample_efficiency_100mer")
    plt.close(fig)


# %% fig 5 — 240mer apples-to-apples

def fig5_apples_to_apples_240mer():
    """240mer with Shujun's full faithful pipeline applied to AR.

    Numbers from ideas_for_writing.md: AR + 3-strategy + rescue gets
    6.6% perfect / 17 puzzles; ours bidir_random argmax-only (no rescue)
    gets 33.2% / 13. With rescue applied symmetrically: ours 32.7% / 13.
    """
    rows = [
        # (method,                          variant,                 perfect_rate, puzzles)
        ("Struct2SeQ AR\n(3-strategy)",     "no rescue",             0.6,    6),
        ("Struct2SeQ AR\n(3-strategy)",     "+ rescue (Shujun pipeline)", 6.6,  17),
        ("Ours, random-perm",               "no rescue",            33.2,   13),
        ("Ours, random-perm",               "+ rescue (symmetric)",  32.7,  13),
    ]
    df = pd.DataFrame(rows, columns=["method", "variant", "perfect_rate", "puzzles"])

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=False)

    methods = ["Struct2SeQ AR\n(3-strategy)", "Ours, random-perm"]
    variants = ["no rescue", "+ rescue (Shujun pipeline)"]
    # display variants for the AR column / ours column
    var_for = {
        "Struct2SeQ AR\n(3-strategy)": ["no rescue", "+ rescue (Shujun pipeline)"],
        "Ours, random-perm":           ["no rescue", "+ rescue (symmetric)"],
    }
    width = 0.36
    xs = np.arange(len(methods))

    for ax, metric, ylabel, ymax in [
        (axes[0], "perfect_rate",
         "Perfect-Jaccard rate (%)", 40),
        (axes[1], "puzzles",
         "Puzzles solved (out of 20)", 20),
    ]:
        for i, vlabel_pair in enumerate(zip(variants, ["+ rescue (Shujun pipeline)", "+ rescue (symmetric)"])):
            # i indexes "no rescue" (0) vs "with rescue" (1)
            vals = []
            for m in methods:
                vname = var_for[m][i]
                vals.append(df[(df["method"] == m) & (df["variant"] == vname)][metric].values[0])
            color = GREY_LIGHT if i == 0 else PURPLE_MID
            ax.bar(xs + (i - 0.5) * width, vals, width=width,
                   color=[PALETTE["ar_l2r"] if i == 1 and "AR" in m else
                          GREY_LIGHT if i == 0 else PURPLE_PRIMARY
                          for m in methods],
                   edgecolor=BLACK, linewidth=0.6,
                   label="No rescue" if i == 0 else "With rescue")
            for x, v in zip(xs + (i - 0.5) * width, vals):
                ax.text(x, v + ymax * 0.02,
                        f"{v:.1f}" if metric == "perfect_rate" else f"{int(v)}",
                        ha="center", va="bottom", fontsize=8.5, color=BLACK)
        ax.set_xticks(xs)
        ax.set_xticklabels(methods)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, ymax * 1.1)
        ax.yaxis.grid(True, linestyle="-", linewidth=0.5, alpha=0.7)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)
        ax.tick_params(bottom=False)

    axes[0].set_title("Per-sample reliability", loc="left", pad=6)
    axes[1].set_title("Puzzle coverage",        loc="left", pad=6)
    axes[0].legend(loc="upper left", fontsize=8)

    fig.suptitle("OK7b 240mer: apples-to-apples comparison",
                 x=0.05, ha="left", fontsize=11, fontweight="bold")
    fig.subplots_adjust(top=0.84, wspace=0.32)

    _save(fig, "fig5_apples_to_apples_240mer")
    plt.close(fig)


# %% fig 6 — architecture vs decoding-order ablation (240mer)

def fig6_arch_vs_order_240mer():
    """Disaggregates ~half-from-arch / ~half-from-order on 240mer.

    Three bars at fixed budget K=1000, argmax-only:
      orig L→R         14.4%   (LSTM-PE + causal-conv + L→R)
      bidir L→R        23.7%   (RPE + L→R)         <- arch contrib
      bidir random-perm 33.2%  (RPE + random-perm) <- + order contrib
    """
    rows = [
        ("Original Struct2SeQ\n(LSTM-PE, L→R)",          14.4, "ar_l2r"),
        ("Ours architecture only\n(RPE, L→R)",            23.7, "ours_l2r"),
        ("Ours full\n(RPE, random-perm)",                 33.2, "ours_random"),
    ]
    df = pd.DataFrame(rows, columns=["config", "perfect_rate", "color_key"])

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    colors = [PALETTE[k] for k in df["color_key"]]
    bars = ax.bar(df["config"], df["perfect_rate"],
                  color=colors, edgecolor=BLACK, linewidth=0.6, width=0.6)
    for bar, v in zip(bars, df["perfect_rate"]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.6,
                f"{v:.1f}%", ha="center", va="bottom",
                fontsize=9, color=BLACK)

    # annotate contributions
    ax.annotate(
        "", xy=(1, 23.7 + 0.3), xytext=(0, 14.4 + 0.3),
        arrowprops=dict(arrowstyle="->", color=GREY_DARK, lw=1.0),
    )
    ax.text(0.5, 19.0, "+9.3 pp\narchitecture", ha="center", va="center",
            fontsize=8, color=GREY_DARK, fontstyle="italic")
    ax.annotate(
        "", xy=(2, 33.2 + 0.3), xytext=(1, 23.7 + 0.3),
        arrowprops=dict(arrowstyle="->", color=PURPLE_PRIMARY, lw=1.0),
    )
    ax.text(1.5, 28.5, "+9.5 pp\norder", ha="center", va="center",
            fontsize=8, color=PURPLE_PRIMARY, fontstyle="italic")

    ax.set_ylabel("Perfect-Jaccard rate (%) at K=1000")
    ax.set_ylim(0, max(df["perfect_rate"]) * 1.3)
    ax.set_title("OK7b 240mer: architecture vs. order contribution",
                 loc="left", pad=8)
    ax.yaxis.grid(True, linestyle="-", linewidth=0.5, alpha=0.7)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(bottom=False)

    _save(fig, "fig6_arch_vs_order_240mer")
    plt.close(fig)


# %% fig 7 — in-painting K-sweep U-shape

def fig7_inpaint_ksweep():
    """Random-scatter perfect-rate vs K_inpaint (fraction-fixed).

    From ideas_for_writing.md:
      0.00 -> 45.5%  (no constraint = same as random-perm)
      0.25 -> 27.7%
      0.50 -> 18.3%
      0.75 -> 18.5%
      0.95 -> 31.7%
    """
    k_inpaint   = [0.00, 0.25, 0.50, 0.75, 0.95]
    perfect     = [45.5, 27.7, 18.3, 18.5, 31.7]
    structural  = (0.50, 37.9)  # mean fraction-fixed ~15% for Framing A
    # Actually: Framing A fixes ~15% (mean fraction-designed 85%). At
    # matched fraction-fixed 0.15, random-scatter is interpolated between
    # 45.5 and 27.7 -> ~ 0.15 * (27.7-45.5)/0.25 + 45.5 = 34.8%, well
    # below 37.9. Plot the structural point as a single annotated dot.

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(k_inpaint, perfect,
            marker="o", markersize=6, color=GREY_MID,
            linewidth=1.5, label="Random-scatter in-painting")

    # Highlight the structural-motif (Framing A) data point
    # at mean fraction-fixed ~0.15 -> 37.9%.
    ax.scatter([0.15], [37.9], s=70, color=PURPLE_PRIMARY,
               zorder=5, edgecolor=BLACK, linewidth=0.6,
               label="Structural motif (Framing A)")
    ax.annotate("Framing A: 37.9%\n(structural is\neasier than\nscattered)",
                xy=(0.15, 37.9), xytext=(0.32, 42),
                fontsize=8, color=BLACK,
                arrowprops=dict(arrowstyle="->", color=PURPLE_PRIMARY,
                                lw=0.8, alpha=0.8))

    ax.set_xlabel("Fraction of positions fixed to wild-type")
    ax.set_ylabel("Perfect-Jaccard rate (%) at K=1000")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(0, 55)
    ax.set_title("Structural motifs are easier than scattered constraints",
                 loc="left", pad=8)
    ax.legend(loc="lower left", fontsize=8)
    ax.yaxis.grid(True, linestyle="-", linewidth=0.5, alpha=0.7)
    ax.xaxis.grid(True, linestyle="-", linewidth=0.5, alpha=0.4)
    ax.set_axisbelow(True)

    _save(fig, "fig7_inpaint_ksweep_100mer")
    plt.close(fig)


# %% main / CLI

FIGS = {
    1: ("methods_comparison_100mer",    fig1_methods_comparison),
    2: ("per_puzzle_100mer",            fig2_per_puzzle_100mer),
    3: ("best_of_n_100mer",             fig3_best_of_n),
    4: ("sample_efficiency_100mer",     fig4_sample_efficiency),
    5: ("apples_to_apples_240mer",      fig5_apples_to_apples_240mer),
    6: ("arch_vs_order_240mer",         fig6_arch_vs_order_240mer),
    7: ("inpaint_ksweep_100mer",        fig7_inpaint_ksweep),
}


def _parse_figs(arg: str):
    if arg == "all":
        return list(FIGS.keys())
    return [int(x.strip()) for x in arg.split(",")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--figs", default="all",
                   help="comma-separated fig numbers, or 'all' (default)")
    p.add_argument("--list", action="store_true",
                   help="list available figures and exit")
    args = p.parse_args()

    if args.list:
        for n, (name, _) in FIGS.items():
            print(f"  {n}. {name}")
        return

    _set_style()
    for n in _parse_figs(args.figs):
        if n not in FIGS:
            print(f"unknown fig {n}; --list to see available")
            continue
        name, fn = FIGS[n]
        print(f"--- fig {n}: {name}")
        fn()


if __name__ == "__main__":
    main()
