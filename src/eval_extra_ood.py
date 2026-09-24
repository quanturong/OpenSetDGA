
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

from ood_utils import ood_metrics, print_ood_metrics
from logger import get_logger

log = get_logger(__name__)


_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789-."
_CHAR2IDX = {c: i + 2 for i, c in enumerate(_CHARS)}
VOCAB_SIZE = len ( _CHARS ) + 2
MAX_LEN = 75


def tokenize_batch(domains, max_len=MAX_LEN):
    out = np.zeros((len(domains), max_len), dtype=np.int32)
    for i, d in enumerate(domains):
        d = d.lower().strip()
        for j, ch in enumerate(d[:max_len]):
            out[i, j] = _CHAR2IDX.get(ch, 1)
    return out


def _compute_lengths(tokens):
    return (tokens != 0).sum(axis=1).astype(np.int64)


class DomainCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, feat_dim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Sequential(
            nn.Conv1d(embed_dim, 128, 3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2))
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 256, 3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2))
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 256, 3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU())
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.pre = nn.Sequential(nn.Linear(256, feat_dim), nn.ReLU())
        self.head = nn.Linear(feat_dim, n_classes)
        self.feat_dim = feat_dim

    def forward(self, x):
        e = self . embed ( x ) . permute ( 0 , 2 , 1 )
        h = self.conv1(e)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.pool(h).squeeze(-1)
        feat = self.pre(h)
        return self.head(feat), feat

    def forward_from_embed(self, e_ble):
        e = e_ble . permute ( 0 , 2 , 1 )
        h = self.conv1(e)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.pool(h).squeeze(-1)
        feat = self.pre(h)
        return self.head(feat), feat


class DomainBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes,
                 hidden_dim=64, num_layers=1, feat_dim=128, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.pre = nn.Sequential(nn.Linear(hidden_dim * 2, feat_dim), nn.ReLU())
        self.head = nn.Linear(feat_dim, n_classes)
        self.feat_dim = feat_dim

    def _pool_from_embed(self, e, lengths=None):
        if lengths is not None:
            lc = lengths.cpu().clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                e, lc, batch_first=True, enforce_sorted=False)
            po, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(po, batch_first=True)
            B, H = out.size(0), out.size(2) // 2
            idx = (lengths - 1).clamp(min=0).long()
            fwd = out[torch.arange(B, device=out.device), idx, :H]
            bwd = out[:, 0, H:]
            return self.drop(torch.cat([fwd, bwd], -1))
        else:
            out, _ = self.lstm(e)
            return self.drop(out[:, -1, :])

    def forward(self, x, lengths=None):
        e = self.embed(x)
        p = self._pool_from_embed(e, lengths)
        feat = self.pre(p)
        return self.head(feat), feat

    def forward_from_embed(self, e, lengths=None):
        p = self._pool_from_embed(e, lengths)
        feat = self.pre(p)
        return self.head(feat), feat


def extract_logits_feats(model, tokens, device, batch_size=1024, is_bilstm=False):
    model.eval()
    all_logits, all_feats = [], []
    xt = torch.tensor(tokens, dtype=torch.long)
    lengths = torch.tensor(_compute_lengths(tokens), dtype=torch.long) if is_bilstm else None
    with torch.no_grad():
        for i in range(0, len(tokens), batch_size):
            xb = xt[i:i + batch_size].to(device)
            if is_bilstm:
                lb = lengths[i:i + batch_size].to(device)
                logit, feat = model(xb, lb)
            else:
                logit, feat = model(xb)
            all_logits.append(logit.cpu())
            all_feats.append(feat.cpu().numpy())
    return torch.cat(all_logits, 0), np.vstack(all_feats)


def energy_score(logits, T=1.0):
    return -(T * torch.logsumexp(logits / T, dim=-1)).numpy()


def msp_score(logits):
    return 1.0 - F.softmax(logits, dim=-1).numpy().max(axis=1)


def odin_score(model, tokens, device, T=1000.0, epsilon=0.001,
               batch_size=1024, is_bilstm=False):
    model.eval()
    all_scores = []
    xt = torch.tensor(tokens, dtype=torch.long)
    lengths = torch.tensor(_compute_lengths(tokens), dtype=torch.long) if is_bilstm else None

    for i in range(0, len(tokens), batch_size):
        xb = xt[i:i + batch_size].to(device)

        emb = model . embed ( xb ) . detach ( ) . requires_grad_ ( True )

        if is_bilstm:
            lb = lengths[i:i + batch_size].to(device)
            with torch.backends.cudnn.flags(enabled=False):
                logit, _ = model.forward_from_embed(emb, lb)
                scaled_soft = F.softmax(logit / T, dim=-1)
                loss = scaled_soft.max(dim=-1)[0].sum()
                loss.backward()
        else:
            logit, _ = model.forward_from_embed(emb)
            scaled_soft = F.softmax(logit / T, dim=-1)
            max_soft = scaled_soft.max(dim=-1)[0]
            loss = max_soft.sum()
            loss.backward()

        if epsilon > 0 and emb.grad is not None:
            perturbed = emb.data + epsilon * emb.grad.sign()
        else:
            perturbed = emb.data

        with torch.no_grad():
            if is_bilstm:
                logit2, _ = model.forward_from_embed(perturbed, lb)
            else:
                logit2, _ = model.forward_from_embed(perturbed)

            soft2 = F.softmax(logit2 / T, dim=-1)
            score = 1.0 - soft2.max(dim=-1)[0]

        all_scores.append(score.cpu().numpy())

    return np.concatenate(all_scores)


def knn_score(feats_train, feats_query, k=5):
    nn_model = NearestNeighbors(n_neighbors=k, metric='euclidean',
                                algorithm='auto', n_jobs=-1)
    nn_model.fit(feats_train)
    dists, _ = nn_model.kneighbors(feats_query)
    return dists.mean(axis=1).astype(np.float32)


def react_score_from_feats(head, feats_train, feats_query, percentile=90):
    threshold = float(np.percentile(feats_train, percentile))
    clipped = np.clip(feats_query, a_min=None, a_max=threshold)
    device = next(head.parameters()).device
    with torch.no_grad():
        logits = head(torch.tensor(clipped, dtype=torch.float32).to(device)).cpu()
    return energy_score(logits)


def _csv_paths(run_dir):
    rd = Path(run_dir)
    return {
        "train": rd / "known" / "train.csv",
        "val": rd / "known" / "val.csv",
        "test_known": rd / "known" / "test_known.csv",
        "unknown_family": rd / "unknown_family" / "test_unknown_family.csv",
        "unknown_ood": rd / "unknown_ood" / "test_unknown_ood.csv",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="data/processed")
    ap.add_argument("--cnn_dir", default="baseline_out/neural_20260414_211427")
    ap.add_argument("--bilstm_dir", default="baseline_out/bilstm_20260414_222503")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--odin_T", type=float, default=1000.0)
    ap.add_argument("--odin_eps", type=float, nargs="+", default=[0.0, 0.001, 0.005])
    ap.add_argument("--knn_k", type=int, nargs="+", default=[5, 20, 50])
    ap.add_argument("--react_pcts", type=int, nargs="+", default=[65, 75, 85, 90, 95])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path("baseline_out") / f"extra_ood_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    csvs = _csv_paths(args.run_dir)

    log.info("Loading CSVs …")
    dfs = {k: pd.read_csv(str(p)) for k, p in csvs.items()}
    for k, df in dfs.items():
        log.info(f"  {k:20s}: {len(df):>8,} rows")

    le = LabelEncoder()
    le.fit(dfs["train"]["class_label"].values)
    n_classes = len(le.classes_)
    log.info(f"\n  {n_classes} classes")

    log.info("\nTokenizing …")
    tok = {}
    for k, df in dfs.items():
        tok[k] = tokenize_batch(df["domain"].tolist())
        log.info(f"  {k:20s}: {tok[k].shape}")

    results = {}
    ood_splits = ["unknown_family", "unknown_ood"]

    log.info("\n" + "=" * 70)
    log.info("  EXPERIMENT 1: CNN + Energy score on raw logits")
    log.info("=" * 70)

    cnn = DomainCNN(VOCAB_SIZE, embed_dim=32, n_classes=n_classes,
                    feat_dim=128).to(device)
    cnn.load_state_dict(torch.load(
        str(Path(args.cnn_dir) / "model.pt"),
        map_location=device, weights_only=True))
    cnn.eval()
    log.info("  CNN model loaded.")

    cnn_logits, cnn_feats = {}, {}
    for k in ["test_known"] + ood_splits:
        cnn_logits[k], cnn_feats[k] = extract_logits_feats(
            cnn, tok[k], device, is_bilstm=False)
        log.info(f"  {k:20s}: logits={tuple(cnn_logits[k].shape)}")

    log.info("\n── cnn_energy ──")
    id_e = energy_score(cnn_logits["test_known"])
    for split in ood_splits:
        ood_e = energy_score(cnn_logits[split])
        m = ood_metrics(id_e, ood_e)
        results[f"cnn_energy_{split}"] = m
        print_ood_metrics(m, split)

    log.info("\n── cnn_msp (reference) ──")
    id_m = msp_score(cnn_logits["test_known"])
    for split in ood_splits:
        ood_m = msp_score(cnn_logits[split])
        m = ood_metrics(id_m, ood_m)
        results[f"cnn_msp_{split}"] = m
        print_ood_metrics(m, split)

    log.info("\n" + "=" * 70)
    log.info("  EXPERIMENT 2: ODIN (temperature + embedding perturbation)")
    log.info("=" * 70)

    bilstm = DomainBiLSTM(VOCAB_SIZE, embed_dim=32, n_classes=n_classes,
                           hidden_dim=64, num_layers=1, feat_dim=128,
                           dropout=0.2).to(device)
    bilstm.load_state_dict(torch.load(
        str(Path(args.bilstm_dir) / "model.pt"),
        map_location=device, weights_only=True))
    bilstm.eval()
    log.info("  BiLSTM model loaded.")

    T = args.odin_T
    for eps in args.odin_eps:
        tag = f"cnn_odin_T{int(T)}_e{eps}"
        log.info(f"\n── {tag} ──")
        t0 = time.time()
        id_s = odin_score(cnn, tok["test_known"], device,
                          T=T, epsilon=eps, is_bilstm=False)
        for split in ood_splits:
            ood_s = odin_score(cnn, tok[split], device,
                               T=T, epsilon=eps, is_bilstm=False)
            m = ood_metrics(id_s, ood_s)
            results[f"{tag}_{split}"] = m
            print_ood_metrics(m, split)
        log.info(f"  ({time.time() - t0:.0f}s)")

        tag = f"bilstm_odin_T{int(T)}_e{eps}"
        log.info(f"\n── {tag} ──")
        t0 = time.time()
        id_s = odin_score(bilstm, tok["test_known"], device,
                          T=T, epsilon=eps, is_bilstm=True)
        for split in ood_splits:
            ood_s = odin_score(bilstm, tok[split], device,
                               T=T, epsilon=eps, is_bilstm=True)
            m = ood_metrics(id_s, ood_s)
            results[f"{tag}_{split}"] = m
            print_ood_metrics(m, split)
        log.info(f"  ({time.time() - t0:.0f}s)")

    log.info("\n" + "=" * 70)
    log.info("  EXPERIMENT 3: KNN on neural features + Hybrid (energy + KNN)")
    log.info("=" * 70)

    log.info("\nExtracting BiLSTM features …")
    bilstm_logits, bilstm_feats = {}, {}
    for k in ["train", "test_known"] + ood_splits:
        bilstm_logits[k], bilstm_feats[k] = extract_logits_feats(
            bilstm, tok[k], device, is_bilstm=True)
        log.info(f"  {k:20s}: feats={bilstm_feats[k].shape}")

    log.info("\nExtracting CNN train features …")
    cnn_logits["train"], cnn_feats["train"] = extract_logits_feats(
        cnn, tok["train"], device, is_bilstm=False)
    log.info(f"  train               : feats={cnn_feats['train'].shape}")

    for k_val in args.knn_k:
        tag = f"bilstm_knn_k{k_val}"
        log.info(f"\n-- {tag} --")
        t0 = time.time()
        id_s = knn_score(bilstm_feats["train"], bilstm_feats["test_known"], k=k_val)

        known_domains = dfs["test_known"]["domain"].tolist()
        pd.DataFrame({"domain": known_domains, "ood_score": id_s}).to_csv(
            out_dir / f"scores_{tag}_known.csv", index=False)

        for split in ood_splits:
            ood_s = knn_score(bilstm_feats["train"], bilstm_feats[split], k=k_val)
            m = ood_metrics(id_s, ood_s)
            results[f"{tag}_{split}"] = m
            print_ood_metrics(m, split)

            split_key = "unknown_family" if split == "unknown_family" else "unknown_ood"
            split_domains = dfs[split]["domain"].tolist()
            pd.DataFrame({"domain": split_domains, "ood_score": ood_s}).to_csv(
                out_dir / f"scores_{tag}_{split_key}.csv", index=False)

        log.info(f"  ({time.time() - t0:.0f}s)")

    for k_val in args.knn_k:
        tag = f"cnn_knn_k{k_val}"
        log.info(f"\n── {tag} ──")
        t0 = time.time()
        id_s = knn_score(cnn_feats["train"], cnn_feats["test_known"], k=k_val)
        for split in ood_splits:
            ood_s = knn_score(cnn_feats["train"], cnn_feats[split], k=k_val)
            m = ood_metrics(id_s, ood_s)
            results[f"{tag}_{split}"] = m
            print_ood_metrics(m, split)
        log.info(f"  ({time.time() - t0:.0f}s)")

    for k_val in args.knn_k:
        for model_name, m_logits, m_feats_train, m_feats in [
            ("bilstm", bilstm_logits, bilstm_feats["train"], bilstm_feats),
            ("cnn", cnn_logits, cnn_feats["train"], cnn_feats),
        ]:
            tag = f"{model_name}_hybrid_energy_knn_k{k_val}"
            log.info(f"\n── {tag} ──")

            id_energy = energy_score(m_logits["test_known"])
            id_knn = knn_score(m_feats_train, m_feats["test_known"], k=k_val)

            mu_e, sig_e = id_energy.mean(), id_energy.std() + 1e-8
            mu_k, sig_k = id_knn.mean(), id_knn.std() + 1e-8

            id_hybrid = (id_energy - mu_e) / sig_e + (id_knn - mu_k) / sig_k

            for split in ood_splits:
                ood_energy = energy_score(m_logits[split])
                ood_knn = knn_score(m_feats_train, m_feats[split], k=k_val)
                ood_hybrid = (ood_energy - mu_e) / sig_e + (ood_knn - mu_k) / sig_k

                m = ood_metrics(id_hybrid, ood_hybrid)
                results[f"{tag}_{split}"] = m
                print_ood_metrics(m, split)

    log.info("\n" + "=" * 70)
    log.info("  EXPERIMENT 4: BiLSTM ReAct (clip penultimate feats + energy)")
    log.info("=" * 70)

    for pct in args.react_pcts:
        tag = f"bilstm_react_p{pct}"
        log.info(f"\n-- {tag} --")
        t0 = time.time()
        id_s = react_score_from_feats(
            bilstm.head, bilstm_feats["train"], bilstm_feats["test_known"], pct)
        for split in ood_splits:
            ood_s = react_score_from_feats(
                bilstm.head, bilstm_feats["train"], bilstm_feats[split], pct)
            m = ood_metrics(id_s, ood_s)
            results[f"{tag}_{split}"] = m
            print_ood_metrics(m, split)
        log.info(f"  ({time.time() - t0:.0f}s)")

    with open(str(out_dir / "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    log.info("\n" + "=" * 70)
    log.info("  SUMMARY — Best AUROC per split")
    log.info("=" * 70)
    for split in ood_splits:
        log.info(f"\n  {split}:")
        split_results = {k: v for k, v in results.items() if k.endswith(f"_{split}")}
        ranked = sorted(split_results.items(),
                        key=lambda x: x[1]["auroc"], reverse=True)
        for rank, (name, met) in enumerate(ranked[:10], 1):
            tag = name.replace(f"_{split}", "")
            log.info(f"    {rank:2d}. {tag:45s}  "
                  f"AUROC={met['auroc']:.4f}  FPR@95={met['fpr_at_tpr']:.4f}")

    log.info(f"\nAll results → {out_dir}")


if __name__ == "__main__":
    main()
