
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
SEEDS = [1, 7, 42, 99, 314]

_CHARS   = "abcdefghijklmnopqrstuvwxyz0123456789-."
_C2I     = {c: i + 2 for i, c in enumerate(_CHARS)}
VOCAB = len ( _CHARS ) + 2
MAX_LEN  = 75

def tokenize(domains: list[str]) -> np.ndarray:
    out = np.zeros((len(domains), MAX_LEN), dtype=np.int32)
    for i, d in enumerate(domains):
        for j, ch in enumerate(d.lower().strip()[:MAX_LEN]):
            out[i, j] = _C2I.get(ch, 1)
    return out


class DomainBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes,
                 hidden_dim=64, num_layers=1, feat_dim=128, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm  = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                              batch_first=True, bidirectional=True,
                              dropout=dropout if num_layers > 1 else 0.0)
        self.pre   = nn.Sequential(
            nn.Linear(hidden_dim * 2, feat_dim), nn.ReLU()
        )
        self.head  = nn.Linear(feat_dim, n_classes)
        self . drop = nn . Identity ( )

    def _pool(self, x, lengths):
        e = self.embed(x)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                e, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
            B, _, H2 = out.shape
            H = H2 // 2
            idx = (lengths - 1).clamp(min=0).long()
            fwd = out[torch.arange(B), idx, :H]
            bwd = out[:, 0, H:]
            return torch.cat([fwd, bwd], dim=-1)
        else:
            out, _ = self.lstm(e)
            return out[:, -1, :]

    def forward(self, x, lengths=None):
        pooled = self._pool(x, lengths)
        feat   = self.pre(pooled)
        return self.head(feat), feat


def load_model(ckpt_path: Path, n_classes: int) -> DomainBiLSTM:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    has_l1 = any("l1" in k for k in state)
    model = DomainBiLSTM(VOCAB, 32, n_classes, num_layers=2 if has_l1 else 1)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def infer_benign_prob(model: DomainBiLSTM, domains: list[str],
                      batch_size: int = 2048) -> np.ndarray:
    toks = tokenize(domains)
    lengths_all = (toks != 0).sum(axis=1).clip(min=1)
    probs = []
    for i in range(0, len(domains), batch_size):
        x   = torch.from_numpy(toks[i:i + batch_size]).long()
        lng = torch.from_numpy(lengths_all[i:i + batch_size]).long()
        logits, _ = model(x, lng)
        p = F.softmax(logits, dim=-1).numpy()
        probs . append ( p [ : , 0 ] )
    return np.concatenate(probs)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref",    type=Path, default=ROOT / "fig3_data.json",
                    help="Seed-42 reference file with domain+source labels")
    ap.add_argument("--kaggle", type=Path, default=Path("E:/kaggle_results"),
                    help="kaggle_results directory")
    ap.add_argument("--out",    type=Path, default=ROOT / "fig3_data_5seeds.json",
                    help="Output path")
    args = ap.parse_args()

    if not args.ref.exists():
        sys.exit(f"ERROR: reference file not found: {args.ref}")

    print(f"Loading reference: {args.ref}")
    ref_rows = json.load(open(args.ref))
    domains  = [r["domain"] for r in ref_rows]
    sources  = [r["source"] for r in ref_rows]
    splits   = [r["split"]  for r in ref_rows]
    N        = len(domains)
    print(f"  {N:,} domains, {len(set(sources))} sources")

    seed_probs = {}
    for s in SEEDS:
        ckpt = args.kaggle / f"bilstm_s{s}" / "model.pt"
        if not ckpt.exists():
            print(f"  [SKIP] seed {s}: {ckpt} not found")
            continue
        results_json = args.kaggle / f"bilstm_s{s}" / "results.json"
        n_classes    = len(json.load(open(results_json))["classes"])
        print(f"  Seed {s}: loading model ({n_classes} classes)...", end=" ", flush=True)
        model = load_model(ckpt, n_classes)
        probs = infer_benign_prob(model, domains)
        seed_probs[s] = probs
        print(f"done  mean_P_benign={probs.mean():.4f}")

    if not seed_probs:
        sys.exit("ERROR: no model checkpoints found.")

    all_probs = np . stack ( list ( seed_probs . values ( ) ) , axis = 0 )
    mean_prob = all_probs.mean(axis=0)
    std_prob  = all_probs.std(axis=0)

    out_rows = [
        {
            "domain":         domains[i],
            "source":         sources[i],
            "split":          splits[i],
            "benign_prob":    float(mean_prob[i]),
            "benign_prob_std": float(std_prob[i]),
        }
        for i in range(N)
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out_rows, open(args.out, "w"))
    print(f"\nSaved {len(out_rows):,} rows -> {args.out}")
    print(f"Seeds used: {sorted(seed_probs)}")

    from collections import defaultdict
    by_src = defaultdict(list)
    for r in out_rows:
        by_src[r["source"]].append(r["benign_prob"])
    print(f"\n{'Source':50s} {'n':>6} {'median':>8} {'mean':>8}")
    print("-" * 76)
    for src, vals in sorted(by_src.items(), key=lambda x: -np.median(x[1])):
        print(f"{src:50s} {len(vals):6d} {np.median(vals):8.4f} {np.mean(vals):8.4f}")


if __name__ == "__main__":
    main()
