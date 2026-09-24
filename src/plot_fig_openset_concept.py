
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from matplotlib.colors import to_rgba
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

CLASSES = {
    "B" : { "center" : ( - 2.2 , 1.6 ) , "color" : "#7c3aed" , "name" : "Benign" } ,
    "1" : { "center" : ( 2.2 , 1.6 ) , "color" : "#dc2626" , "name" : "Known DGA family 1" } ,
    "2" : { "center" : ( 2.2 , - 1.6 ) , "color" : "#059669" , "name" : "Known DGA family 2" } ,
    "3" : { "center" : ( - 2.2 , - 1.6 ) , "color" : "#d97706" , "name" : "Known DGA family 3" } ,
}

ELLIPSES = {
    "B": dict(width=2.6, height=2.0, angle=20),
    "1": dict(width=2.6, height=2.0, angle=-20),
    "2": dict(width=2.6, height=2.0, angle=20),
    "3": dict(width=2.6, height=2.0, angle=-20),
}

XLO, XHI = -4.0, 4.0
YLO, YHI = -3.6, 3.8


def _base_ax(ax, title, facecolor="white"):
    ax.set_xlim(XLO, XHI)
    ax.set_ylim(YLO, YHI)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(facecolor)
    for sp in ax.spines.values():
        sp.set_linewidth(1.4)
        sp . set_color ( "#374151" )
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)


def _draw_ellipses(ax, outline_color="white", outline_lw=1.5, alpha=0.82):
    for label, info in CLASSES.items():
        cx, cy = info["center"]
        ep = ELLIPSES[label]
        ax.add_patch(Ellipse((cx, cy), **ep,
                             facecolor=info["color"], alpha=alpha,
                             edgecolor=outline_color, linewidth=outline_lw,
                             zorder=3))
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=20, fontweight="bold", color="white", zorder=4)


def _draw_voronoi_fill(ax):
    centers = np.array([info["center"] for info in CLASSES.values()])
    colors  = [info["color"] for info in CLASSES.values()]

    nx, ny = 400, 400
    xs = np.linspace(XLO, XHI, nx)
    ys = np.linspace(YLO, YHI, ny)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)

    dists   = np.sum((pts[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    nearest = np.argmin(dists, axis=1)

    rgba = np.array([to_rgba(colors[i], alpha=0.55) for i in nearest])
    rgba = rgba.reshape(ny, nx, 4)

    ax.imshow(rgba, extent=[XLO, XHI, YLO, YHI], origin="lower",
              aspect="auto", zorder=0, interpolation="nearest")


def _draw_voronoi_boundaries(ax):
    centers = np.array([info["center"] for info in CLASSES.values()])

    nx, ny = 600, 600
    xs = np.linspace(XLO, XHI, nx)
    ys = np.linspace(YLO, YHI, ny)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)

    dists    = np.sum((pts[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    sorted_d = np.sort(dists, axis=1)
    diff     = (sorted_d[:, 1] - sorted_d[:, 0]).reshape(ny, nx)

    ax.contour(xx, yy, diff, levels=[0.4],
               colors = "#1d4ed8" , linewidths = 2.8 , zorder = 2 )


def plot_concept(out_dir: Path) -> None:
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(14.5, 5.4))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.15, top=0.88, wspace=0.08)

    _base_ax(ax_a, "(a) Training")
    _draw_ellipses(ax_a)

    _base_ax(ax_b, "(b) Closed-Set Classification")
    _draw_voronoi_fill(ax_b)
    _draw_voronoi_boundaries(ax_b)
    _draw_ellipses(ax_b)

    _base_ax ( ax_c , "(c) Open-Set Recognition (Ours)" , facecolor = "#cccccc" )
    _draw_ellipses ( ax_c , outline_color = "#1d4ed8" , outline_lw = 2.8 )
    ax_c.text(XHI - 0.15, YLO + 0.20, "unknown",
              ha="right", va="bottom", fontsize=11,
              style = "italic" , color = "#444444" , zorder = 5 )

    legend_items = [
        mpatches.Patch(color=CLASSES["B"]["color"], label="B  =  Benign"),
        mpatches.Patch(color=CLASSES["1"]["color"], label="1/2/3  =  Known DGA families"),
        mpatches.Patch(facecolor="#cccccc", edgecolor="#888888", linewidth=1.0,
                       label="Gray  =  Unknown / OOD"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=3,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.01),
               handlelength=1.4, columnspacing=2.0)

    for fmt in ("pdf", "png"):
        path = out_dir / f"fig_openset_concept.{fmt}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        print(f"  Saved {path}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "figures")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print("=== Open-Set Concept Figure ===")
    plot_concept(args.out)
    print("Done.")


if __name__ == "__main__":
    main()
