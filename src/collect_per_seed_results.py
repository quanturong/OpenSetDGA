
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

SEEDS = [1, 7, 42, 99, 314]
METRICS = ["auroc", "aupr_out", "fpr_at_tpr", "precision_at_tpr"]
T_CRIT = float ( stats . t . ppf ( 0.975 , df = len ( SEEDS ) - 1 ) )


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.load(open(path))


def extract_metric(d: dict, key: str, metric: str):
    sub = d.get(key, {})
    if isinstance(sub, dict):
        return sub.get(metric)
    return None


def collect_entry(seeds_data: list[dict | None], key: str, metric: str) -> list | None:
    vals = []
    for d in seeds_data:
        if d is None:
            return None
        v = extract_metric(d, key, metric)
        if v is None:
            return None
        vals.append(float(v))
    return vals


def summarize(vals: list[float]) -> dict:
    mn = float(np.mean(vals))
    sd = float(np.std(vals, ddof=0))
    ci = float(T_CRIT * sd / np.sqrt(len(vals)))
    return {"mean": mn, "std": sd, "ci_half": ci}


def build_record(seeds_data: list[dict | None], result_key: str) -> dict | None:
    record = {"per_seed": {}, "summary": {}}
    for metric in METRICS:
        vals = collect_entry(seeds_data, result_key, metric)
        if vals is None:
            return None
        for seed, v in zip(SEEDS, vals):
            record["per_seed"].setdefault(seed, {})[metric] = v
        record["summary"][metric] = summarize(vals)
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kaggle", type=Path, default=Path("E:/kaggle_results"))
    ap.add_argument("--baseline", type=Path,
                    default=Path("E:/OpenSetDGA-Detection/baseline_out"))
    ap.add_argument("--out", type=Path,
                    default=Path("E:/OpenSetDGA-Detection/multi_seed_results/per_seed_results.json"))
    args = ap.parse_args()

    KG = args.kaggle
    BL = args.baseline

    SOURCES = [
        ("LGBM Binary",     "MSP",              "lgbm_binary", KG, ["ood_msp_{split}"]),
        ("LGBM Binary",     "Energy",           "lgbm_binary", KG, ["ood_energy_{split}"]),
        ("LGBM Multiclass", "MSP",              "lgbm_multi",  KG, ["ood_msp_{split}"]),
        ("LGBM Multiclass", "Energy",           "lgbm_multi",  KG, ["ood_energy_{split}"]),
        ("LGBM Multiclass", "KNN-k5",           "lgbm_multi",  KG, ["ood_knn_{split}"]),
        ("CNN",             "MSP",              "cnn",         KG, ["ood_msp_{split}"]),
        ("CNN",             "Mahalanobis",      "cnn",         KG, ["ood_mahalanobis_{split}"]),
        ("CNN",             "Energy",           "extra_ood",   KG, ["cnn_energy_{split}"]),
        ("CNN",             "ODIN T=1000",      "extra_ood",   KG, ["cnn_odin_T1000_e0.0_{split}"]),
        ("CNN",             "KNN-k5",           "extra_ood",   KG, ["cnn_knn_k5_{split}"]),
        ("CNN",             "KNN-k20",          "extra_ood",   KG, ["cnn_knn_k20_{split}"]),
        ("CNN",             "KNN-k50",          "extra_ood",   KG, ["cnn_knn_k50_{split}"]),
        ("CNN",             "Hybrid E+KNN-k5",  "extra_ood",   KG, ["cnn_hybrid_energy_knn_k5_{split}"]),
        ("BiLSTM",          "MSP",              "bilstm",      KG, ["ood_msp_{split}"]),
        ("BiLSTM",          "Energy",           "bilstm",      KG, ["ood_energy_{split}"]),
        ("BiLSTM",          "Mahalanobis",      "bilstm",      KG, ["ood_mahalanobis_{split}"]),
        ("BiLSTM",          "ODIN T=1000",      "extra_ood",   KG, ["bilstm_odin_T1000_e0.0_{split}"]),
        ("BiLSTM",          "KNN-k5",           "extra_ood",   KG, ["bilstm_knn_k5_{split}"]),
        ("BiLSTM",          "KNN-k20",          "extra_ood",   KG, ["bilstm_knn_k20_{split}"]),
        ("BiLSTM",          "KNN-k50",          "extra_ood",   KG, ["bilstm_knn_k50_{split}"]),
        ("BiLSTM",          "Hybrid E+KNN-k5",  "extra_ood",   KG, ["bilstm_hybrid_energy_knn_k5_{split}"]),
        ("BiLSTM",          "ReAct-p65",        "extra_ood",   BL, ["bilstm_react_p65_{split}"]),
        ("BiLSTM",          "ReAct-p75",        "extra_ood",   BL, ["bilstm_react_p75_{split}"]),
        ("BiLSTM",          "ReAct-p85",        "extra_ood",   BL, ["bilstm_react_p85_{split}"]),
        ("BiLSTM",          "ReAct-p90",        "extra_ood",   BL, ["bilstm_react_p90_{split}"]),
        ("BiLSTM",          "ReAct-p95",        "extra_ood",   BL, ["bilstm_react_p95_{split}"]),
        ("BiLSTM-OE",       "MSP",              "bilstm_oe",   KG, ["ood_msp_{split}"]),
        ("BiLSTM-OE",       "Energy",           "bilstm_oe",   KG, ["ood_energy_{split}"]),
    ]

    SPLITS = ["unknown_family", "unknown_ood"]

    output = {
        "seeds": SEEDS,
        "t_crit": T_CRIT,
        "metrics": METRICS,
        "models": {},
    }

    available_count = 0
    missing_count = 0

    for (model, scorer, prefix, base, _rkeys) in SOURCES:
        label = f"{model} / {scorer}"

        seeds_data = [load_json(base / f"{prefix}_s{s}" / "results.json")
                      for s in SEEDS]

        for split in SPLITS:
            rkey = _rkeys[0].format(split=split)
            record = build_record(seeds_data, rkey)

            entry_key = f"{label} | {split}"
            if record is not None:
                output["models"][entry_key] = {
                    "model":   model,
                    "scorer":  scorer,
                    "split":   split,
                    "result_key": rkey,
                    "source":  str(base),
                    **record,
                }
                available_count += 1
            else:
                missing_count += 1

    hlr_path = BL / "hybrid_lr_multiseed_results.json"
    if hlr_path.exists():
        hlr = json.load(open(hlr_path))
        per_seed_raw = hlr.get("per_seed", {})
        if len(per_seed_raw) >= len(SEEDS):
            for split_key, split_label in [("unknown_family", "unknown_family"),
                                            ("unknown_ood", "unknown_ood")]:
                entry_key = f"BiLSTM / Hybrid LR | {split_label}"
                record = {"per_seed": {}, "summary": {}}
                ok = True
                for metric in METRICS:
                    vals = []
                    for S in SEEDS:
                        v = per_seed_raw.get(str(S), {}).get(split_key, {}).get(metric)
                        if v is None:
                            ok = False
                            break
                        vals.append(float(v))
                    if not ok:
                        break
                    for seed, v in zip(SEEDS, vals):
                        record["per_seed"].setdefault(seed, {})[metric] = v
                    record["summary"][metric] = summarize(vals)
                if ok:
                    output["models"][entry_key] = {
                        "model":   "BiLSTM",
                        "scorer":  "Hybrid LR",
                        "split":   split_label,
                        "result_key": "hybrid_lr",
                        "source":  str(hlr_path),
                        **record,
                    }
                    available_count += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Collected {available_count} (model, scorer, split) entries "
          f"with full 5-seed data.")
    print(f"Skipped {missing_count} entries (incomplete across seeds).")
    print(f"Saved -> {args.out}")

    print(f"\n{'Entry':55} {'AUROC':20} {'FPR@95':20} {'AUPR':20} {'Prec@95':20}")
    print("-" * 115)
    for k, v in output["models"].items():
        s = v["summary"]
        def fmt(m):
            return f"{s[m]['mean']:.4f}±{s[m]['std']:.4f}"
        print(f"{k:55} {fmt('auroc'):20} {fmt('fpr_at_tpr'):20} "
              f"{fmt('aupr_out'):20} {fmt('precision_at_tpr'):20}")


if __name__ == "__main__":
    main()
