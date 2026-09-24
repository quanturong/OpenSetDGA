
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

ROOT = Path(__file__).parent.parent

plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size":       9,
    "figure.dpi":      150,
    "savefig.dpi":     200,
})

C_KNOWN_D = "#1d4ed8"
C_KNOWN_L = "#eff6ff"
C_UF_D = "#6d28d9"
C_UF_L = "#f5f3ff"
C_OOD_D = "#c2410c"
C_OOD_L = "#fff7ed"
C_TEXT = "#111827"
C_SUB = "#6b7280"
C_BORDER = "#d1d5db"
C_BADGE = "#1e3a8a"


def _rect(ax, x, y, w, h, fc="white", ec=C_BORDER, lw=1.0, radius=0.02, zorder=2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        clip_on=False, zorder=zorder,
    ))


def _badge(ax, cx, cy, n, r=0.22):
    ax.add_patch(mpatches.Circle((cx, cy), r, color=C_BADGE, zorder=6, clip_on=False))
    ax.text(cx, cy, str(n), ha="center", va="center",
            fontsize=8, fontweight="bold", color="white", zorder=7)


def _t(ax, x, y, s, ha="left", va="top", color=C_TEXT, **kw):
    return ax.text(x, y, s, ha=ha, va=va, color=color, **kw)


def _bullets(ax, x, y0, items, fontsize=7.8, dy=0.32):
    for i, item in enumerate(items):
        _t(ax, x, y0 - i * dy, f"•  {item}", fontsize=fontsize, color=C_SUB)


def _tag(ax, cx, y, w, h, text, color, zorder=4):
    _rect(ax, cx - w/2, y, w, h, fc=color, ec=color, lw=0, radius=0.012, zorder=zorder)
    _t(ax, cx, y + h/2, text, ha="center", va="center",
       fontsize=7, color="white", fontweight="bold", zorder=zorder + 1)


def plot_pipeline(out_dir: Path) -> None:
    W, H = 14.5, 6.8
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    _t(ax, W / 2, H - 0.18, "Overview of the OpenSetDGA Benchmark",
       ha="center", fontsize=13, fontweight="bold")

    PY0, PH = 0.5, 5.9
    panels = [
        ( 0.15 , PY0 , 2.80 , PH ) ,
        ( 3.50 , PY0 , 2.80 , PH ) ,
        ( 6.85 , PY0 , 4.30 , PH ) ,
        ( 11.70 , PY0 , 2.65 , PH ) ,
    ]

    for px, py, pw, ph in panels:
        _rect ( ax , px , py , pw , ph , fc = "#fafafa" , ec = "#9ca3af" , lw = 1.3 , radius = 0.06 )

    mid_y = PY0 + PH / 2
    for (px, _, pw, _), (nx, *_) in zip(panels, panels[1:]):
        ax.annotate("", xy=(nx - 0.04, mid_y), xytext=(px + pw + 0.04, mid_y),
                    arrowprops=dict(arrowstyle="->", color="#9ca3af",
                                   lw=2.0, mutation_scale=18))

    px, py, pw, ph = panels[0]
    _badge(ax, px + 0.30, py + ph - 0.35, 1)
    _t(ax, px + 0.60, py + ph - 0.22, "DATA SOURCES", fontsize=8.5, fontweight="bold")

    sb_x, sb_w = px + 0.16, pw - 0.32
    src_boxes = [
        (py + 3.70, 1.80, C_KNOWN_L, C_KNOWN_D, "Benign",      ["Tranco"]),
        (py + 2.10, 1.40, C_UF_L,    C_UF_D,    "DGA",         ["23 families"]),
        (py + 0.18, 1.72, C_OOD_L,   C_OOD_D,   "Real-world",
         ["6 malicious feeds", "Tranco tail", "crt.sh"]),
    ]
    for (by, bh, bfc, bec, btitle, bullets) in src_boxes:
        _rect(ax, sb_x, by, sb_w, bh, fc=bfc, ec=bec, lw=1.0, radius=0.025)
        _t(ax, sb_x + sb_w / 2, by + bh - 0.14, btitle,
           ha="center", fontsize=8.5, fontweight="bold", color=bec)
        _bullets(ax, sb_x + 0.22, by + bh - 0.44, bullets)

    px, py, pw, ph = panels[1]
    _badge(ax, px + 0.30, py + ph - 0.35, 2)
    _t(ax, px + 0.60, py + ph - 0.22, "DATA CONSTRUCTION", fontsize=8.5, fontweight="bold")

    steps = [
        "eTLD+1 normalization",
        "Validity filtering",
        "Deduplication",
        "Source capping",
        "Stratified benign\nsampling",
    ]
    for i, s in enumerate(steps):
        _t(ax, px + 0.28, py + ph - 0.82 - i * 0.92,
           f"•  {s}", fontsize=8.0, color=C_SUB)

    px, py, pw, ph = panels[2]
    _badge(ax, px + 0.30, py + ph - 0.35, 3)
    _t(ax, px + 0.60, py + ph - 0.22, "OPENSETDGA BENCHMARK", fontsize=8.5, fontweight="bold")
    _t(ax, px + 0.60, py + ph - 0.54, "549,999 domains",
       fontsize=7.5, color=C_SUB, style="italic")

    sb_x, sb_w = px + 0.20, pw - 0.40
    cx = sb_x + sb_w / 2

    splits = [
        (py + 3.18, 2.00, C_KNOWN_L, C_KNOWN_D, "KNOWN / ID",
         ["Train: 319,999", "Validation: 40,000",
          "Test-Known: 40,000", "18 known families + benign"],
         "Known-class classification"),
        (py + 1.64, 1.42, C_UF_L, C_UF_D, "UNSEEN-FAMILY OOD",
         ["50,000 domains", "5 held-out DGA families"],
         "Unseen-family OOD"),
        (py + 0.10, 1.42, C_OOD_L, C_OOD_D, "REAL-WORLD OOD",
         ["100,000 domains", "8 sources (6 malicious + 2 legitimate)"],
         "Real-world OOD"),
    ]
    for (sy, sh, sfc, sec, stitle, sbullets, stag) in splits:
        _rect(ax, sb_x, sy, sb_w, sh, fc=sfc, ec=sec, lw=1.0, radius=0.025)
        _t(ax, cx, sy + sh - 0.14, stitle, ha="center",
           fontsize=8.5, fontweight="bold", color=sec)
        _bullets(ax, sb_x + 0.22, sy + sh - 0.42, sbullets, dy=0.30)
        _tag(ax, cx, sy + 0.07, sb_w * 0.68, 0.28, stag, sec)

    px, py, pw, ph = panels[3]
    _badge(ax, px + 0.30, py + ph - 0.35, 4)
    _t(ax, px + 0.60, py + ph - 0.22, "EVALUATION", fontsize=8.5, fontweight="bold")

    evals = [
        (py + 4.30, "Baselines", C_KNOWN_D,
         ["LightGBM", "CNN", "BiLSTM"]),
        (py + 2.80, "OOD Scoring", C_UF_D,
         ["MSP, Energy, KNN,", "ReAct, Hybrid E+KNN"]),
        (py + 1.50, "Metrics", C_OOD_D,
         ["AUROC, FPR@TPR95", "classification +", "OOD metrics"]),
    ]
    for sy, label, color, lines in evals:
        _t(ax, px + 0.22, sy, label, fontsize=8.5, fontweight="bold", color=color)
        for i, line in enumerate(lines):
            _t(ax, px + 0.22, sy - 0.33 - i * 0.28, line, fontsize=7.8, color=C_SUB)

    for dy in [4.00, 2.55]:
        ax.plot([px + 0.14, px + pw - 0.14], [py + dy] * 2,
                color=C_BORDER, lw=0.8)

    for fmt in ("pdf", "png"):
        path = out_dir / f"fig_pipeline.{fmt}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        print(f"  Saved {path}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "figures")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print("=== Pipeline Figure ===")
    plot_pipeline(args.out)
    print("Done.")


if __name__ == "__main__":
    main()
