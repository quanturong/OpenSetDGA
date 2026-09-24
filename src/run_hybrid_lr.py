
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from ood_utils import ood_metrics


def load_scores(path: Path, tag: str) -> pd.Series:
    df = pd.read_csv(path)
    return df.set_index("domain")["ood_score"].rename(tag)


def merge_scores(bilstm_dir: Path, extra_ood_dir: Path, split: str) -> pd.DataFrame:
    e = load_scores(bilstm_dir / f"scores_energy_{split}.csv", "energy")
    k = load_scores(extra_ood_dir / f"scores_bilstm_knn_k5_{split}.csv", "knn")
    return pd.concat([e, k], axis=1).dropna()


def run_one_seed(bilstm_dir: Path, extra_ood_dir: Path, split_seed: int = 42) -> dict:
    known_df = merge_scores(bilstm_dir, extra_ood_dir, "known")
    uf_df    = merge_scores(bilstm_dir, extra_ood_dir, "unknown_family")
    ood_df   = merge_scores(bilstm_dir, extra_ood_dir, "unknown_ood")

    rng = np.random.default_rng(split_seed)

    def split_half(df):
        idx = rng.choice(len(df), size=len(df) // 2, replace=False)
        mask = np.zeros(len(df), dtype=bool)
        mask[idx] = True
        return df.iloc[idx], df.iloc[~mask]

    uf_cal,  uf_holdout  = split_half(uf_df)
    ood_cal, ood_holdout = split_half(ood_df)

    X_id      = known_df[["energy", "knn"]].values
    X_ood_mix = np.vstack([uf_cal[["energy", "knn"]].values,
                           ood_cal[["energy", "knn"]].values])
    X_cal = np.vstack([X_id, X_ood_mix])
    y_cal = np.concatenate([np.zeros(len(X_id)),
                            np.ones(len(X_ood_mix))]).astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X_cal, y_cal, test_size=0.2, random_state=split_seed, stratify=y_cal)

    scaler = StandardScaler()
    clf    = LogisticRegression(C=1.0, max_iter=1000, random_state=split_seed)
    clf.fit(scaler.fit_transform(X_train), y_train)

    val_auroc = roc_auc_score(
        y_val, clf.predict_proba(scaler.transform(X_val))[:, 1])

    def hscore(df):
        return clf.predict_proba(
            scaler.transform(df[["energy", "knn"]].values))[:, 1]

    ks   = hscore(known_df)
    ufs  = hscore(uf_holdout)
    oods = hscore(ood_holdout)

    m_uf  = ood_metrics(ks, ufs)
    m_ood = ood_metrics(ks, oods)

    return {
        "coef_energy":  float(clf.coef_[0][0]),
        "coef_knn":     float(clf.coef_[0][1]),
        "val_auroc":    float(val_auroc),
        "unknown_family": m_uf,
        "unknown_ood":    m_ood,
    }


def print_metrics(label: str, m: dict):
    print(f"  {label}: "
          f"AUROC={m['auroc']:.4f}  "
          f"AUPR={m['aupr_out']:.4f}  "
          f"FPR@95={m['fpr_at_tpr']:.4f}  "
          f"Prec@95={m['precision_at_tpr']:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--bilstm_dir", type=Path,
                      help="Single-seed: BiLSTM output dir (contains scores_energy_*.csv)")
    mode.add_argument("--seeds", type=int, nargs="+",
                      help="Multi-seed: list of seed integers")

    ap.add_argument("--extra_ood_dir", type=Path,
                    help="Single-seed: extra_ood output dir (contains scores_bilstm_knn_k5_*.csv)")
    ap.add_argument("--base_dir", type=Path, default=Path("baseline_out"),
                    help="Multi-seed: root dir containing bilstm_s{S} and extra_ood_s{S} "
                         "(default: baseline_out)")
    ap.add_argument("--split_seed", type=int, default=42,
                    help="RNG seed for the 50%% calibration/holdout split (default: 42)")
    ap.add_argument("--save", type=Path, default=None,
                    help="Optional JSON file to write results to")

    args = ap.parse_args()

    if args.seeds is not None:
        per_seed = {}
        for S in args.seeds:
            bilstm_dir    = args.base_dir / f"bilstm_s{S}"
            extra_ood_dir = args.base_dir / f"extra_ood_s{S}"
            if not bilstm_dir.exists() or not extra_ood_dir.exists():
                print(f"[WARN] s{S}: dirs not found, skipping "
                      f"({bilstm_dir}, {extra_ood_dir})")
                continue
            print(f"\n=== seed {S} ===")
            res = run_one_seed(bilstm_dir, extra_ood_dir, args.split_seed)
            print(f"  coef: energy={res['coef_energy']:.4f}  knn={res['coef_knn']:.4f}"
                  f"  val_auroc={res['val_auroc']:.4f}")
            print_metrics("unknown_family", res["unknown_family"])
            print_metrics("unknown_ood",    res["unknown_ood"])
            per_seed[S] = res

        if not per_seed:
            print("No seeds completed — check --base_dir.")
            sys.exit(1)

        print("\n=== Multi-seed summary (mean ± std) ===")
        for split_key in ("unknown_family", "unknown_ood"):
            print(f"\n  {split_key}:")
            for metric in ("auroc", "aupr_out", "fpr_at_tpr", "precision_at_tpr"):
                vals = [per_seed[S][split_key][metric] for S in per_seed]
                mn, sd = np.mean(vals), np.std(vals)
                print(f"    {metric:25s}: {mn:.4f} ± {sd:.4f}")

        if args.save:
            with open(args.save, "w") as f:
                json.dump({"seeds": list(per_seed.keys()),
                           "split_seed": args.split_seed,
                           "per_seed": {str(k): v for k, v in per_seed.items()}},
                          f, indent=2)
            print(f"\nSaved -> {args.save}")

    else:
        if args.extra_ood_dir is None:
            ap.error("--extra_ood_dir is required with --bilstm_dir")

        res = run_one_seed(args.bilstm_dir, args.extra_ood_dir, args.split_seed)
        print(f"coef: energy={res['coef_energy']:.4f}  knn={res['coef_knn']:.4f}"
              f"  val_auroc={res['val_auroc']:.4f}")
        print("\nHybrid LR (BiLSTM Energy + BiLSTM KNN k=5):")
        print_metrics("unknown_family", res["unknown_family"])
        print_metrics("unknown_ood",    res["unknown_ood"])

        if args.save:
            with open(args.save, "w") as f:
                json.dump(res, f, indent=2)
            print(f"\nSaved -> {args.save}")


if __name__ == "__main__":
    main()
