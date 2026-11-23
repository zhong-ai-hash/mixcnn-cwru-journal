"""Column-width architecture diagram for the v6 manuscript."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "results_v2"


def box(ax, x, y, w, h, text, fc, ls="-", lw=0.75, fs=5.8):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.35,rounding_size=1.2",
        linewidth=lw, linestyle=ls, edgecolor="0.20", facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, linespacing=1.15)


def arrow(ax, x1, y1, x2, y2, ls="-", color="0.25"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=6,
        linewidth=0.65, linestyle=ls, color=color, shrinkA=1, shrinkB=1,
    ))


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 5.8,
        "axes.linewidth": 0,
    })
    fig, ax = plt.subplots(figsize=(3.45, 2.25))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(1, 97, "(a) Reference path", ha="left", va="top",
            fontsize=6.3, fontweight="bold")

    box(ax, 1, 68, 13, 15, "Input\n1 x 1024", "#f2f2f2", fs=5.2)
    box(ax, 17, 67, 17, 17, "Stem\nk32 / s4\n4,480", "#dce9f5", fs=5.2)
    box(ax, 38, 81, 19, 10,
        "Ref: pad 387|388\n128 x 1024", "#fbe5ce", ls="--", fs=5.0)
    box(ax, 38, 60, 19, 10,
        "NoPad: bypass\n128 x 249", "#dff0df", ls="--", fs=5.0)
    box(ax, 61, 60, 22, 31,
        "3 x MixConv\n25,216 each\n75,648 total", "#e6dff2", fs=5.2)
    box(ax, 87, 65, 12, 21,
        "BN + GAP\nFC\n1,546", "#f3dddd", fs=5.2)

    arrow(ax, 14, 75.5, 17, 75.5)
    arrow(ax, 34, 77, 38, 84)
    arrow(ax, 34, 73, 38, 66)
    arrow(ax, 57, 85, 61, 80, ls="--", color="#a55a1b")
    arrow(ax, 57, 66, 61, 69, ls="--", color="#357a35")
    arrow(ax, 83, 75.5, 87, 75.5)

    ax.text(47.5, 92.3, "Ref only", ha="center", va="bottom",
            fontsize=4.8, color="#a55a1b")
    ax.text(47.5, 57.5, "NoPad only", ha="center", va="top",
            fontsize=4.8, color="#357a35")

    ax.plot([1, 98.5], [52, 52], color="0.75", linewidth=0.55)
    ax.text(1, 49, "(b) Variant switches", ha="left", va="top",
            fontsize=6.3, fontweight="bold")
    box(ax, 1, 25, 29, 17,
        "Ref\npad + k64 block", "#fbe5ce", fs=5.2)
    box(ax, 34, 25, 29, 17,
        "NoPad\nremove pad path", "#dff0df", fs=5.2)
    box(ax, 66.5, 25, 32.5, 17,
        "MS\nk8 / k64 parallel\nDW branches", "#e6dff2", fs=5.0)
    box(ax, 1, 4, 63.5, 16,
        "NoRes: remove identity    Soft: learnable\nper-channel threshold after DW",
        "#eeeeee", fs=5.0)
    box(ax, 67, 4, 32, 16,
        "Params\nRef/NoPad/NoRes 81,674\nMS 84,746 | Soft 82,058",
        "#f2f2f2", fs=4.5)

    fig.tight_layout(pad=0.05)
    fig.savefig(OUT / "fig_architecture_v6.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_architecture_v6.png", dpi=600, bbox_inches="tight")


if __name__ == "__main__":
    main()
