
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent

plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size":       9,
    "figure.dpi":      150,
    "savefig.dpi":     200,
})

C_BENIGN = "#2563eb"
C_DGA = "#dc2626"
C_UF = "#7c3aed"
C_RW = "#525252"
C_KNOWN_BORDER = "#374151"

XLO, XHI = -6.5, 6.5
YLO, YHI = -5.5, 5.5


def plot_gap(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 7.0))
    ax.set_xlim(XLO, XHI)
    ax.set_ylim(YLO, YHI)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax . set_facecolor ( "#efefef" )
    for sp in ax.spines.values():
        sp.set_linewidth(1.2)
        sp . set_color ( "#9ca3af" )
    fig.patch.set_facecolor("white")

    ax.add_patch(Ellipse((0, 0), width=10.0, height=8.2,
                         facecolor="white", edgecolor=C_KNOWN_BORDER,
                         linewidth=2.0, linestyle="--", zorder=1))
    ax.text(0, -4.5, "Known space (existing benchmarks stop here)",
            ha="center", fontsize=9, color=C_KNOWN_BORDER, style="italic", zorder=2)

    ax.add_patch(Ellipse((-1.4, 0.4), width=3.2, height=2.6, angle=12,
                         facecolor=C_BENIGN, alpha=0.22,
                         edgecolor=C_BENIGN, linewidth=2.0, zorder=2))
    rng = np.random.default_rng(42)
    bpts = rng.normal([-1.4, 0.4], 0.55, (20, 2))
    ax.scatter(bpts[:, 0], bpts[:, 1], s=55, color=C_BENIGN,
               alpha=0.65, linewidths=0, zorder=3)
    ax.text(-1.4, 2.0, "Benign", ha="center", fontsize=11,
            color=C_BENIGN, fontweight="bold", zorder=4)

    for dx, dy, ew, eh, ang, alp in [
        ( 0.0,  0.0, 3.6, 3.0, -8, 0.18),
        (-0.4,  0.5, 2.2, 1.6, 15, 0.12),
        ( 0.5, -0.5, 2.0, 1.4, -20, 0.10),
    ]:
        ax.add_patch(Ellipse((1.8+dx, -0.3+dy), width=ew, height=eh, angle=ang,
                             facecolor=C_DGA, alpha=alp,
                             edgecolor=C_DGA, linewidth=1.6, zorder=2))
    dpts = rng.normal([1.8, -0.3], 0.62, (20, 2))
    ax.scatter(dpts[:, 0], dpts[:, 1], s=55, color=C_DGA,
               alpha=0.65, linewidths=0, zorder=3)
    ax.text(1.8, 1.5, "Known DGA\n(18 families)", ha="center", fontsize=11,
            color=C_DGA, fontweight="bold", zorder=4)

    uf_cx, uf_cy = 4.4, 2.2
    ax.add_patch(Ellipse((uf_cx, uf_cy), width=2.2, height=1.8, angle=-20,
                         facecolor=C_UF, alpha=0.28,
                         edgecolor=C_UF, linewidth=2.2,
                         linestyle="-", zorder=2))
    uf_pts = rng.normal([uf_cx, uf_cy], 0.38, (10, 2))
    ax.scatter(uf_pts[:, 0], uf_pts[:, 1], s=65, color=C_UF,
               alpha=0.80, marker="^", linewidths=0, zorder=3)

    ax.annotate("", xy=(3.0, 0.5), xytext=(3.7, 1.6),
                arrowprops=dict(arrowstyle="->", color=C_UF,
                                lw=1.4, mutation_scale=12))
    ax.text(uf_cx + 0.2, uf_cy + 1.35,
            "Unseen-Family OOD\n(5 held-out DGA families)",
            ha="center", fontsize=9.5, color=C_UF, fontweight="bold", zorder=4)
    ax.text(uf_cx + 0.2, uf_cy + 0.78,
            "DGA-like, but unseen at training",
            ha="center", fontsize=8, color=C_UF, style="italic", zorder=4)

    rw_groups = {
        "phishing":   (rng.normal([-5.2,  3.8], 0.30, (4, 2))),
        "blocklist":  (rng.normal([-4.8, -3.8], 0.28, (4, 2))),
        "tranco_tail":(rng.normal([-0.5,  4.8], 0.32, (4, 2))),
        "urlhaus":    (rng.normal([ 4.0, -4.0], 0.30, (4, 2))),
        "netlab":     (rng.normal([-5.8,  0.5], 0.25, (3, 2))),
        "crtsh":      (rng.normal([ 5.5, -1.8], 0.25, (3, 2))),
    }
    for pts in rw_groups.values():
        ax.scatter(pts[:, 0], pts[:, 1], s=65, color=C_RW,
                   alpha=0.72, marker="s", linewidths=0, zorder=3)

    ax.text(-5.0, 5.0, "Real-World OOD\n(8 external sources:\nphishing, blocklists, crt.sh...)",
            ha="center", fontsize=9.5, color=C_RW, fontweight="bold", zorder=4)
    ax.text(-5.0, 3.4,
            "Heterogeneous — some malicious,\nsome legitimate-looking",
            ha="center", fontsize=8, color=C_RW, style="italic", zorder=4)

    legend_items = [
        mpatches.Patch(color=C_BENIGN, alpha=0.7, label="Benign (known)"),
        mpatches.Patch(color=C_DGA,    alpha=0.7, label="Known DGA families (18, seen at training)"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor=C_UF,
                   markersize=9, label="Unseen-Family OOD  — missed by existing benchmarks"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=C_RW,
                   markersize=9, label="Real-World OOD  — missed by existing benchmarks"),
    ]
    fig.legend(handles=legend_items, loc="lower center",
               bbox_to_anchor=(0.5, 0.01), ncol=2, frameon=False,
               fontsize=8.5, handlelength=1.4, columnspacing=1.5)

    ax.set_title(
        "Research Gap: Existing DGA benchmarks evaluate only within the known space\n"
        "OpenSetDGA adds two OOD scenarios to reflect real deployment conditions",
        fontsize=10, fontweight="bold", pad=10, linespacing=1.5
    )

    fig.subplots_adjust(bottom=0.18)

    for fmt in ("pdf", "png"):
        path = out_dir / f"fig_gap.{fmt}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        print(f"  Saved {path}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "figures")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print("=== Gap Figure ===")
    plot_gap(args.out)
    print("Done.")


if __name__ == "__main__":
    main()
