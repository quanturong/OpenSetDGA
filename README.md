# OpenSetDGA: A Benchmark for Open-Set DGA Detection

**OpenSetDGA** is a benchmark of 549,999 normalized (eTLD+1) domains for evaluating DGA detection under two distinct open-set scenarios:

- **Unknown-family OOD** — 5 DGA families held out entirely from training
- **Real-world OOD** — heterogeneous threat feeds and uncommon-legitimate sources

Key finding: the best scorer for unknown-family OOD (BiLSTM + ReAct) differs from the best for real-world OOD (BiLSTM + KNN-k5), showing the two scenarios require different detection strategies.

> **Paper:** OpenSetDGA: A Benchmark for Open-Set Domain Generation Algorithm Detection — FDSE 2026, Springer
> **Dataset:** [HuggingFace — ThanhPhuongtphz/OpenSetDGA-Detection](https://huggingface.co/datasets/ThanhPhuongtphz/OpenSetDGA-Detection)
> **License:** MIT

---

## Repository layout

```
OpenSetDGA-Detection/
├── data/processed/              # download from HuggingFace (see below)
│   ├── known/
│   │   ├── train.csv            # 319,999 samples (18 DGA families + benign)
│   │   ├── val.csv              # 40,000 samples
│   │   └── test_known.csv       # 40,000 samples
│   ├── unknown_family/
│   │   └── test_unknown_family.csv   # 50,000 samples (5 unseen families)
│   └── unknown_ood/
│       └── test_ood.csv         # 100,000 samples (threat feeds + legitimate)
├── notebooks/
│   └── 01_data_collection_and_cleaning.ipynb
├── src/
│   ├── features.py              # 38 lexical features
│   ├── train_baseline.py        # LightGBM Binary
│   ├── train_multiclass.py      # LightGBM Multiclass (19 classes)
│   ├── train_neural.py          # CNN (DomainCNN)
│   ├── train_bilstm.py          # BiLSTM + MSP / Energy / Mahalanobis / KNN
│   ├── train_bilstm_oe.py       # BiLSTM with Outlier Exposure
│   ├── eval_extra_ood.py        # ReAct, ODIN, KNN, Hybrid E+KNN
│   ├── hybrid_scorer.py         # Hybrid Logistic Regression scorer
│   ├── run_multi_seed.py        # Run full pipeline over N seeds
│   └── data_assertions.py       # Zero-leakage verification
├── requirements.txt
├── setup.py
├── LICENSE
└── CITATION.cff
```

---

## Quick start

```bash
git clone https://github.com/quanturong/OpenSetDGA-A-Benchmark-for-Open-Set-Domain-Generation-Algorithm-Detection.git
cd OpenSetDGA-A-Benchmark-for-Open-Set-Domain-Generation-Algorithm-Detection
pip install -r requirements.txt
```

### 1. Download the dataset

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="ThanhPhuongtphz/OpenSetDGA-Detection",
    repo_type="dataset",
    local_dir="data/processed"
)
```

### 2. Verify data integrity

```bash
python src/data_assertions.py
```

Checks zero leakage across all 10 pairs of the 5 splits, and no conflicting labels within any split.

### 3. Reproduce all results

```bash
# Train all models, 5 seeds (1, 7, 42, 99, 314)
python src/run_multi_seed.py

# Optional: Hybrid LR scorer (uses 50% OOD holdout for calibration)
python src/hybrid_scorer.py

# Aggregate per-seed results into a single summary
python src/aggregate_seeds.py
```

Output is written to `baseline_out/<model>_s<seed>/results.json`.

---

## Results

Mean ± std over 5 seeds. **Bold** = best per column. Full tables (AUPR, Prec@95) are in the paper.

### Known-Class Classification

| Model | MC-Acc ↑ | MC-F1 ↑ | Bin-F1 ↑ | Bin-AUC ↑ |
|---|---|---|---|---|
| LGBM Binary | — | — | 0.9075 ± 0.0002 | 0.9702 ± 0.0001 |
| LGBM Multiclass | 0.7596 ± 0.0252 | 0.7258 ± 0.0110 | 0.8464 ± 0.0203 | 0.9693 ± 0.0002 |
| CNN | 0.9290 ± 0.0024 | 0.8879 ± 0.0029 | 0.9714 ± 0.0019 | 0.9961 ± 0.0003 |
| BiLSTM | 0.9405 ± 0.0007 | **0.9046 ± 0.0011** | 0.9758 ± 0.0002 | **0.9970 ± 0.0001** |
| BiLSTM-OE | **0.9412 ± 0.0007** | 0.8961 ± 0.0032 | **0.9761 ± 0.0004** | **0.9970 ± 0.0001** |

### Unknown-Family OOD

| Model | Scorer | AUROC ↑ | FPR@95 ↓ |
|---|---|---|---|
| LGBM Binary | MSP = Energy† | 0.4963 ± 0.0029 | 0.9030 ± 0.0155 |
| LGBM Multiclass | MSP | 0.6134 ± 0.0072 | 0.8011 ± 0.0248 |
| CNN | Energy | 0.8130 ± 0.0174 | 0.5878 ± 0.0477 |
| CNN | Hybrid E+KNN | 0.7952 ± 0.0076 | 0.5837 ± 0.0244 |
| BiLSTM | MSP | 0.6971 ± 0.0146 | 0.7931 ± 0.0269 |
| BiLSTM | Energy | 0.8570 ± 0.0086 | **0.4507 ± 0.0419** |
| BiLSTM | ODIN (T=1000) | 0.7915 ± 0.0102 | 0.5917 ± 0.0393 |
| BiLSTM | Mahalanobis | 0.6119 ± 0.0107 | 0.8844 ± 0.0128 |
| BiLSTM | KNN k=5 | 0.6109 ± 0.0113 | 0.8689 ± 0.0116 |
| BiLSTM | Hybrid E+KNN | 0.8122 ± 0.0088 | 0.5116 ± 0.0143 |
| BiLSTM | **ReAct p=95** | **0.8636 ± 0.0116** | 0.4623 ± 0.0557 |
| BiLSTM-OE | MSP | 0.7037 ± 0.0100 | 0.8192 ± 0.0207 |
| BiLSTM-OE | Energy | 0.7008 ± 0.0096 | 0.6893 ± 0.0204 |
| BiLSTM | Hybrid LR‡ | 0.8114 ± 0.0108 | 0.5111 ± 0.0220 |

### Real-World OOD

| Model | Scorer | AUROC ↑ | FPR@95 ↓ |
|---|---|---|---|
| LGBM Binary | MSP = Energy† | 0.5651 ± 0.0015 | 0.9765 ± 0.0018 |
| LGBM Multiclass | MSP | 0.5450 ± 0.0033 | 0.9047 ± 0.0060 |
| CNN | MSP | 0.4906 ± 0.0278 | 0.9914 ± 0.0173 |
| CNN | KNN k=5 | 0.6857 ± 0.0071 | 0.8411 ± 0.0140 |
| BiLSTM | MSP | 0.4874 ± 0.0072 | 0.9714 ± 0.0064 |
| BiLSTM | Energy | 0.5786 ± 0.0185 | 0.8929 ± 0.0329 |
| BiLSTM | ReAct p=90 | 0.5338 ± 0.0147 | 0.9297 ± 0.0167 |
| BiLSTM | Hybrid E+KNN | 0.7012 ± 0.0100 | 0.7263 ± 0.0075 |
| BiLSTM | **KNN k=5** | **0.7224 ± 0.0021** | **0.6827 ± 0.0054** |
| BiLSTM | Hybrid LR‡ | 0.7018 ± 0.0067 | 0.7263 ± 0.0034 |

† For a binary classifier, MSP and Energy are monotone in the same logit — rankings are identical.  
‡ Hybrid LR uses 50% of target OOD samples for calibration and is not directly comparable to zero-shot scorers.

