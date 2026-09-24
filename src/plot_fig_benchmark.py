
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["DejaVu Sans"],
    "font.size":        10,
    "axes.linewidth":   0.8,
    "xtick.direction":  "out",
    "ytick.direction":  "out",
    "figure.dpi":       150,
    "savefig.dpi":      200,
})

TOTAL = 549_999

SPLITS = [
    ( "Train" , 319_999 , "#1565c0" ) ,
    ( "Val" , 40_000 , "#29b6f6" ) ,
    ( "Test-\nKnown" , 40_000 , "#e53935" ) ,
    ( "Unseen-\nFamily" , 50_000 , "#43a047" ) ,
    ( "Real-World\nOOD" , 100_000 , "#7b1fa2" ) ,
]

TRAIN_DGA = 160_082
TRAIN_BENIGN = 159_917

C_KNOWN = "#2563eb"
C_HELDOUT = "#ea580c"
C_MAL = "#b91c1c"
C_LEGIT = "#0e7490"


def _chip(ax, x, y, color, w=0.018, h=0.060):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y - h / 2), w, h,
        boxstyle="round,pad=0.002",
        linewidth=0, facecolor=color, clip_on=False, zorder=3,
    ))


def plot_benchmark(out_dir: Path) -> None:
    fig = plt.figure(figsize=(11, 7.8))

    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1.4, 1.0],
        hspace=0.42,
        wspace=0.30,
        left=0.08, right=0.96,
        top=0.93, bottom=0.05,
    )
    ax_bar  = fig.add_subplot(gs[0, 0])
    ax_pie  = fig.add_subplot(gs[0, 1])
    ax_info = fig.add_subplot(gs[1, :])

    fig.suptitle(f"Dataset Overview  ({TOTAL:,} domains)",
                 fontsize=13, fontweight="bold")

    labels = [s[0] for s in SPLITS]
    counts = [s[1] for s in SPLITS]
    colors = [s[2] for s in SPLITS]
    x = np.arange(len(labels))

    bars = ax_bar.bar(x, counts, color=colors, width=0.55, alpha=0.88, zorder=3)

    for bar, n in zip(bars, counts):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.013,
            f"{n:,}",
            ha="center", va="bottom",
            fontsize=8.0, fontweight="bold",
        )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, fontsize=9.0)
    ax_bar.set_ylabel("Sample count", fontsize=9.5)
    ax_bar.set_title("Sample count by split",
                     fontsize=10.5, fontweight="bold", pad=6)
    ax_bar.set_ylim(0, max(counts) * 1.22)
    ax_bar.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, zorder=0)
    ax_bar.set_axisbelow(True)
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.tick_params(axis="x", bottom=False)
    ax_bar.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{int(v/1000)}k" if v >= 1000 else str(int(v)))
    )

    wedges, texts, autotexts = ax_pie.pie(
        [TRAIN_DGA, TRAIN_BENIGN],
        labels=["DGA", "Benign"],
        colors = [ "#1565c0" , "#e53935" ] ,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(linewidth=1.5, edgecolor="white"),
        textprops=dict(fontsize=10.5),
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_color("white")
        at.set_fontweight("bold")
    ax_pie.set_title("Label distribution in Train set",
                     fontsize=10.5, fontweight="bold", pad=8)

    ax_info.set_xlim(0, 1)
    ax_info.set_ylim(0, 1)
    ax_info.axis("off")

    ax_info.text(0.5, 0.93, "Benchmark Design",
                 ha="center", va="top",
                 fontsize = 11 , fontweight = "bold" , color = "#1e293b" )
    ax_info . plot ( [ 0.01 , 0.99 ] , [ 0.83 , 0.83 ] , color = "#cbd5e1" , lw = 0.9 )

    ax_info . plot ( [ 0.50 , 0.50 ] , [ 0.06 , 0.81 ] , color = "#e2e8f0" , lw = 0.9 )

    Y_SEC = 0.77
    Y_IT1 = 0.60
    Y_DT1 = 0.47
    Y_IT2 = 0.28
    Y_DT2 = 0.15

    CX1 = 0.03

    ax_info.text(CX1, Y_SEC,
                 "DGA Families  (23 total: 18 known + 5 held-out)",
                 ha="left", va="top",
                 fontsize = 9.5 , fontweight = "bold" , color = "#1e3a8a" )

    _chip(ax_info, CX1 + 0.01, Y_IT1 - 0.025, C_KNOWN)
    ax_info.text(CX1 + 0.04, Y_IT1,
                 "18 known families",
                 ha="left", va="top",
                 fontsize=9.0, fontweight="bold", color=C_KNOWN)
    ax_info.text(CX1 + 0.04, Y_DT1,
                 "->  Train / Validation / Test-Known",
                 ha="left", va="top",
                 fontsize = 8.0 , color = "#64748b" , style = "italic" )

    _chip(ax_info, CX1 + 0.01, Y_IT2 - 0.025, C_HELDOUT)
    ax_info.text(CX1 + 0.04, Y_IT2,
                 "5 held-out families",
                 ha="left", va="top",
                 fontsize=9.0, fontweight="bold", color=C_HELDOUT)
    ax_info.text(CX1 + 0.04, Y_DT2,
                 "->  Unseen-Family OOD  (50,000 domains)",
                 ha="left", va="top",
                 fontsize = 8.0 , color = "#64748b" , style = "italic" )

    CX2 = 0.53

    ax_info.text(CX2, Y_SEC,
                 "Real-World OOD  (100,000 domains)",
                 ha="left", va="top",
                 fontsize = 9.5 , fontweight = "bold" , color = "#7f1d1d" )

    _chip(ax_info, CX2 + 0.01, Y_IT1 - 0.025, C_MAL)
    ax_info.text(CX2 + 0.04, Y_IT1,
                 "6 malicious feeds",
                 ha="left", va="top",
                 fontsize=9.0, fontweight="bold", color=C_MAL)
    ax_info.text(CX2 + 0.04, Y_DT1,
                 "phishing.army,  openphish,  hagezi,  stamparm,  URLhaus,  360netlab",
                 ha="left", va="top",
                 fontsize = 8.5 , color = "#64748b" , style = "italic" )

    _chip(ax_info, CX2 + 0.01, Y_IT2 - 0.025, C_LEGIT)
    ax_info.text(CX2 + 0.04, Y_IT2,
                 "2 legitimate sources",
                 ha="left", va="top",
                 fontsize=9.0, fontweight="bold", color=C_LEGIT)
    ax_info.text(CX2 + 0.04, Y_DT2,
                 "Tranco tail,  crt.sh",
                 ha="left", va="top",
                 fontsize = 8.5 , color = "#64748b" , style = "italic" )


    for fmt in ("pdf", "png"):
        path = out_dir / f"fig_benchmark_design.{fmt}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        print(f"  Saved {path}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "figures",
                    help="Output directory (default: figures/)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print("=== Benchmark Design Figure ===")
    plot_benchmark(args.out)
    print("Done.")


if __name__ == "__main__":
    main()
