
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import FancyBboxPatch
from scipy.stats import gaussian_kde

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent

C_UF = "#2563eb"
C_OOD = "#ea580c"

GROUP_COLORS = {
    "Known benign" : "#2563eb" ,
    "Known DGA" : "#dc2626" ,
    "Tranco tail" : "#0891b2" ,
    "phishing.army" : "#f97316" ,
    "OpenPhish" : "#84cc16" ,
    "Hagezi" : "#7c3aed" ,
    "Stamparm" : "#78350f" ,
    "URLhaus" : "#059669" ,
    "360netlab" : "#d97706" ,
    "crt.sh" : "#db2777" ,
}

SOURCE_MAP = {
    "tranco":                                              "Known benign",
    "chrmor_25dga":                                        "Known DGA",
    "tranco_tail":                                         "Tranco tail",
    "phishing.army:phishing_army_blocklist.txt":           "phishing.army",
    "openphish.com:feed.txt":                              "OpenPhish",
    "raw.githubusercontent.com:hagezi/dns-blocklists":     "Hagezi",
    "raw.githubusercontent.com:stamparm/blackbook":        "Stamparm",
    "urlhaus.abuse.ch:text_online":                        "URLhaus",
    "360netlab":                                           "360netlab",
    "crtsh":                                               "crt.sh",
}

GROUPS_ORDER = [
    ("Known benign",  "test_known"),
    ("Known DGA",     "test_known"),
    ("Tranco tail",   "unknown_ood"),
    ("phishing.army", "unknown_ood"),
    ("OpenPhish",     "unknown_ood"),
    ("Hagezi",        "unknown_ood"),
    ("Stamparm",      "unknown_ood"),
    ("URLhaus",       "unknown_ood"),
    ("360netlab",     "unknown_ood"),
    ("crt.sh",        "unknown_ood"),
]

GROUP_LABELS = {
    "Known benign":  "Known\nbenign",
    "Known DGA":     "Known\nDGA",
    "Tranco tail":   "Tranco\ntail",
    "phishing.army": "phishing.\narmy",
    "OpenPhish":     "OpenPhish",
    "Hagezi":        "Hagezi",
    "Stamparm":      "Stamparm",
    "URLhaus":       "URLhaus",
    "360netlab":     "360netlab",
    "crt.sh":        "crt.sh",
}

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["DejaVu Sans"],
    "font.size":        9,
    "axes.linewidth":   0.8,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width":0.8,
    "ytick.major.width":0.8,
    "xtick.direction":  "out",
    "ytick.direction":  "out",
    "figure.dpi":       150,
    "savefig.dpi":      200,
})


SCORER_KEYS = [
    ("BiLSTM / MSP",             "MSP"),
    ("BiLSTM / Energy",          "Energy"),
    ("BiLSTM / KNN-k5",          "KNN-k5"),
    ("BiLSTM / ReAct-p95",       "ReAct\n(p95)"),
    ("BiLSTM / Hybrid E+KNN-k5", "Hybrid\nE+KNN"),
]


def load_fig2_data(psj: dict) -> dict:
    rows = {}
    for model_key, label in SCORER_KEYS:
        for split in ("unknown_family", "unknown_ood"):
            entry = psj["models"].get(f"{model_key} | {split}")
            if entry:
                rows.setdefault(label, {})[split] = entry["summary"]
    return rows


def plot_fig2(out_dir: Path, psj: dict) -> None:
    data   = load_fig2_data(psj)
    labels = [lab for _, lab in SCORER_KEYS]
    n      = len(labels)
    x      = np.arange(n)
    bw, off = 0.30, 0.17

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    fig.subplots_adjust(wspace=0.38, left=0.07, right=0.97,
                        bottom=0.18, top=0.88)

    panels = [
        (axes[0], "auroc",      "AUROC ↑"),
        (axes[1], "fpr_at_tpr", "FPR@TPR95 ↓"),
    ]
    for ax, metric, title in panels:
        for split, color, dx in (
            ("unknown_family", C_UF,  -off),
            ("unknown_ood",    C_OOD, +off),
        ):
            vals = [data[lab][split][metric]["mean"] for lab in labels]
            cis  = [data[lab][split][metric]["ci_half"] for lab in labels]
            ax.bar(x + dx, vals, bw,
                   color=color, alpha=0.82, zorder=3)
            ax.errorbar(x + dx, vals, yerr=cis,
                        fmt = "none" , ecolor = "#374151" ,
                        elinewidth=1.3, capsize=3.5, capthick=1.3, zorder=4)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 1.08)
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.tick_params(axis="y", labelsize=8)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=9.5, fontweight="bold", pad=5)

    handles = [
        mpatches.Patch(color=C_UF,  alpha=0.82, label="Unseen-Family OOD"),
        mpatches.Patch(color=C_OOD, alpha=0.82, label="Real-World OOD"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.02))
    fig.text(0.5, -0.02, "Error bars: 95% CI  (t-distribution, n = 5 seeds)",
             ha = "center" , fontsize = 7.5 , color = "#64748b" , style = "italic" )

    for fmt in ("pdf", "png"):
        path = out_dir / f"fig2_bilstm_ood_scorers.{fmt}"
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close(fig)


SAMPLE_CAP = 3000
JITTER_N   = 150
KDE_PTS    = 300


def load_fig3_data(data_path: Path, rng_seed: int = 42) -> dict[str, np.ndarray]:
    raw = json.load(open(data_path))
    pools: dict[str, list] = defaultdict(list)
    for row in raw:
        grp = SOURCE_MAP.get(row["source"])
        if grp:
            pools[grp].append(row["benign_prob"])

    rng = np.random.default_rng(rng_seed)
    out: dict[str, np.ndarray] = {}
    for grp, vals_list in pools.items():
        arr = np.array(vals_list, dtype=np.float64)
        if len(arr) > SAMPLE_CAP:
            idx = rng.choice(len(arr), size=SAMPLE_CAP, replace=False)
            arr = arr[idx]
        out[grp] = arr
    return out


def violin_kde(vals: np.ndarray, y_pts: int = KDE_PTS):
    kde = gaussian_kde(vals, bw_method="silverman")
    y   = np.linspace(0, 1, y_pts)
    d   = kde(y)
    d  /= d.max()
    return y, d


def plot_fig3(out_dir: Path, data_path: Path) -> None:
    groups_data = load_fig3_data(data_path)

    n_grp     = len(GROUPS_ORDER)
    n_known = sum ( 1 for _ , sp in GROUPS_ORDER if sp == "test_known" )
    n_ood = n_grp - n_known
    div_x = n_known - 0.5

    fig, ax = plt.subplots(figsize=(12.0, 4.2))
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.16, top=0.80)

    xlim = (-0.55, n_grp - 0.45)

    fig.suptitle("Benign Confidence Across Domain Sources",
                 fontsize=11, fontweight="bold", y=0.98)

    total_w      = xlim[1] - xlim[0]
    known_mid_ax = (0.5 - xlim[0]) / total_w
    ood_mid_ax   = (n_known + n_ood / 2 - 0.5 - xlim[0]) / total_w

    for ax_frac, label, clr in (
        ( known_mid_ax , "Known / ID" , "#1d4ed8" ) ,
        ( ood_mid_ax , "Real-World OOD" , "#c2410c" ) ,
    ):
        ax.text(ax_frac, 1.03, label,
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color=clr)

    ax . axvspan ( xlim [ 0 ] , div_x , color = "#dbeafe" , alpha = 0.35 , lw = 0 , zorder = 0 )
    ax . axvspan ( div_x , xlim [ 1 ] , color = "#ffedd5" , alpha = 0.35 , lw = 0 , zorder = 0 )

    ax . axvline ( div_x , color = "#94a3b8" , lw = 0.8 , linestyle = "--" , zorder = 1 )

    half_vln = 0.36
    rng      = np.random.default_rng(42)

    for i, (name, _split) in enumerate(GROUPS_ORDER):
        vals  = groups_data.get(name, np.array([]))
        if len(vals) == 0:
            print(f"  WARNING: no data for group '{name}'")
            continue
        color = GROUP_COLORS[name]
        pos   = float(i)

        y, d = violin_kde(vals)
        xL   = pos - d * half_vln
        xR   = pos + d * half_vln
        ax.fill_betweenx(y, xL, xR, alpha=0.22, color=color, linewidth=0)
        ax.plot(xL, y, color=color, lw=1.3, alpha=0.6)
        ax.plot(xR, y, color=color, lw=1.3, alpha=0.6)

        q50 = float(np.median(vals))
        ax.scatter([pos], [q50], s=18, color="white",
                   zorder = 6 , linewidths = 1.0 , edgecolors = "#111827" )
        ax.scatter([pos], [q50], s=6, color="#111827",
                   zorder=7, linewidths=0)

        n_jit = min(80, len(vals))
        idx   = rng.choice(len(vals), size=n_jit, replace=False)
        jx    = pos + rng.uniform(-half_vln * 0.28, half_vln * 0.28, size=n_jit)
        ax.scatter(jx, vals[idx], s=3.5, color=color,
                   alpha=0.18, linewidths=0, zorder=2)

    ax.set_xticks(range(n_grp))
    ax.set_xticklabels([GROUP_LABELS[n] for n, _ in GROUPS_ORDER], fontsize=8.5)
    ax.set_ylim(-0.04, 1.08)
    ax.set_xlim(*xlim)
    ax.set_ylabel("Predicted benign probability", fontsize=9)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.25, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    for fmt in ("pdf", "png"):
        path = out_dir / f"fig3_benign_confidence.{fmt}"
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "figures",
                    help="Output directory (default: figures/)")
    ap.add_argument("--psj", type=Path,
                    default=ROOT / "multi_seed_results" / "per_seed_results.json",
                    help="per_seed_results.json")
    ap.add_argument("--fig3data", type=Path,
                    default=ROOT / "fig3_data_5seeds.json",
                    help="BiLSTM inference output (use generate_fig3_data.py to create)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print("=== Fig 2 ===")
    psj = json.load(open(args.psj))
    plot_fig2(args.out, psj)

    print("=== Fig 3 ===")
    plot_fig3(args.out, args.fig3data)

    print("Done.")


if __name__ == "__main__":
    main()
