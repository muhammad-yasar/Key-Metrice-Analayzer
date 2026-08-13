"""
step2_train_classifier.py
=========================
Updated with SAPC-paper optimizations:
  1. all-MiniLM-L6-v2 embedding (faster, better sentence-level similarity)
  2. Feature fusion: embedding + PageRank dependency score + KPI regex score
  3. UMAP dimensionality reduction before classifier head

Run:
    python step2_train_classifier.py --data training_data.csv

Output files:
    level_classifier.pt / level_classifier_labels.json
    class_classifier.pt / class_classifier_labels.json
    umap_reducer.pkl  (shared UMAP model — needed by step3 API)
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
import argparse
import time
import os
import re
import pickle
import logging
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("pip install sentence-transformers")

try:
    import umap
except ImportError:
    raise ImportError("pip install umap-learn")

try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    HAS_SPACY = True
except Exception:
    HAS_SPACY = False
    logging.warning("spaCy not available — PageRank scoring disabled. "
                    "pip install spacy && python -m spacy download en_core_web_sm")

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False
    logging.warning("networkx not available — PageRank scoring disabled. "
                    "pip install networkx")

# ── Config ──────────────────────────────────────────────────────────────────
# Opt 1: switch to all-MiniLM-L6-v2 (384d, faster, better sentence similarity)
MPNET_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM    = 384          # MiniLM is 384d (was 768 for mpnet)
UMAP_DIM     = 16           # sqrt(470 samples)≈21, use 16 to avoid overfitting
FUSED_DIM    = UMAP_DIM + 2 # 16 + pagerank + kpi = 18
HIDDEN_DIM   = 64   # smaller network for small dataset
EPOCHS       = 80
BATCH_SIZE   = 32
LEARNING_RATE = 5e-4
MIN_F1       = 0.70

# KPI detection patterns (same as step4)
_KPI_RE = re.compile(
    r"|".join([
        r"\b\d[\d,\.]*\s*(ha|hectares?|km2?|%|percent|tonne)\b",
        r"€\s*\d[\d,\.]*",
        r"\bby\s+(20\d{2}|19\d{2})\b",
        r"\bper\s+annum\b",
        r"\b(DAFM|NPWS|EPA|OPW|Coillte|the\s+Minister|the\s+Government)\b"
        r".{0,80}\b(shall|will|must|deliver|establish|implement)\b",
        r"\b(publish|develop|establish)\b.{0,60}\b(plan|report|database|guidance)\b",
    ]),
    re.IGNORECASE,
)


# ── Opt 2: PageRank dependency scoring ──────────────────────────────────────
def compute_pagerank_scores(texts: list) -> np.ndarray:
    """
    Build a directed dependency graph over sentences and compute
    PageRank scores. High score = sentence referenced by many others
    = likely a key policy commitment.
    Returns normalised scores array of shape (len(texts),).
    """
    if not HAS_SPACY or not HAS_NX:
        return np.zeros(len(texts), dtype=np.float32)

    # Extract dependency tokens per sentence
    dep_tokens = []
    for text in texts:
        doc = _nlp(text[:300])  # limit for speed
        tokens = frozenset(
            t.lemma_.lower() for t in doc
            if t.dep_ in ("nsubj", "dobj", "ROOT") and len(t.lemma_) > 2
        )
        dep_tokens.append(tokens)

    # Build directed graph: edge j→i if j shares a dependency token with i
    G = nx.DiGraph()
    n = len(texts)
    G.add_nodes_from(range(n))

    for i in range(n):
        if not dep_tokens[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if dep_tokens[i] & dep_tokens[j]:
                G.add_edge(j, i)

    if G.number_of_edges() == 0:
        return np.zeros(n, dtype=np.float32)

    pr = nx.pagerank(G, alpha=0.85)
    scores = np.array([pr.get(i, 0.0) for i in range(n)], dtype=np.float32)

    # Normalise 0-1
    mn, mx = scores.min(), scores.max()
    if mx > mn:
        scores = (scores - mn) / (mx - mn)
    return scores


def compute_kpi_scores(texts: list) -> np.ndarray:
    """Binary KPI flag as float (0 or 1)."""
    return np.array(
        [1.0 if _KPI_RE.search(t) else 0.0 for t in texts],
        dtype=np.float32,
    )


# ── Opt 3: UMAP reduction ───────────────────────────────────────────────────
def fit_umap(embeddings: np.ndarray, n_components: int = UMAP_DIM) -> tuple:
    """Fit UMAP on training embeddings. Returns (reducer, reduced_embeddings)."""
    print(f"  Fitting UMAP {embeddings.shape[1]}d → {n_components}d ...")
    t0 = time.time()
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=min(15, len(embeddings) // 10),  # scale to dataset size
        min_dist=0.1,
        metric="cosine",
        random_state=42,
        verbose=False,
    )
    reduced = reducer.fit_transform(embeddings)
    print(f"  UMAP done in {time.time()-t0:.1f}s — shape: {reduced.shape}")
    return reducer, reduced


# ── Model definition ────────────────────────────────────────────────────────
class ClassifierHead(nn.Module):
    """
    Linear head on fused features (UMAP-reduced embedding + pagerank + kpi).
    Input dim = FUSED_DIM = UMAP_DIM + 2
    """
    def __init__(self, n_classes: int, in_dim: int = FUSED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(HIDDEN_DIM // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Dataset ─────────────────────────────────────────────────────────────────
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(embeddings, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# ── Core training function ───────────────────────────────────────────────────
def train_one_classifier(
    texts: list,
    labels_raw: list,
    label_encoder: LabelEncoder,
    save_name: str,
    device: str,
    sentence_model,
    umap_reducer=None,
    pr_scores: np.ndarray = None,
    kpi_scores: np.ndarray = None,
) -> tuple:
    """
    Returns (best_f1, umap_reducer) — reducer is fitted here if not provided.
    """
    print(f"\n{'='*60}")
    print(f"Training: {save_name}")
    print(f"{'='*60}")
    print(f"Classes ({len(label_encoder.classes_)}): "
          f"{label_encoder.classes_.tolist()}")
    print(f"Total samples: {len(texts)}")

    y = label_encoder.transform(labels_raw)
    n_classes = len(label_encoder.classes_)

    counts = Counter(y)
    for cls_idx, count in sorted(counts.items()):
        flag = "" if count >= 30 else "  <-- LOW"
        print(f"  {label_encoder.classes_[cls_idx]:<28} {count:>4}{flag}")

    # Stratified split
    indices = list(range(len(texts)))
    idx_tr, idx_val, y_tr, y_val = train_test_split(
        indices, y, test_size=0.15, stratify=y, random_state=42,
    )
    texts_tr  = [texts[i] for i in idx_tr]
    texts_val = [texts[i] for i in idx_val]
    print(f"\nTrain: {len(texts_tr)}  Val: {len(texts_val)}")

    # ── Opt 1: Embed with MiniLM ──────────────────────────────────────────
    print(f"\nEmbedding with MiniLM ({EMBED_DIM}d) on {device}...")
    t0 = time.time()
    with torch.no_grad():
        emb_tr  = sentence_model.encode(
            texts_tr,  batch_size=128, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        emb_val = sentence_model.encode(
            texts_val, batch_size=128, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True,
        )
    print(f"Embedding done in {time.time()-t0:.1f}s")

    # ── Opt 3: UMAP reduction ─────────────────────────────────────────────
    if umap_reducer is None:
        umap_reducer, emb_tr_r = fit_umap(emb_tr)
    else:
        print(f"  Applying existing UMAP reducer...")
        emb_tr_r = umap_reducer.transform(emb_tr)

    emb_val_r = umap_reducer.transform(emb_val)
    print(f"  UMAP reduced: train {emb_tr_r.shape}  val {emb_val_r.shape}")

    # ── Opt 2: Feature fusion ─────────────────────────────────────────────
    pr_tr  = pr_scores[idx_tr].reshape(-1, 1) if pr_scores is not None else np.zeros((len(idx_tr), 1))
    pr_val = pr_scores[idx_val].reshape(-1, 1) if pr_scores is not None else np.zeros((len(idx_val), 1))
    kp_tr  = kpi_scores[idx_tr].reshape(-1, 1) if kpi_scores is not None else np.zeros((len(idx_tr), 1))
    kp_val = kpi_scores[idx_val].reshape(-1, 1) if kpi_scores is not None else np.zeros((len(idx_val), 1))

    X_tr  = np.hstack([emb_tr_r,  pr_tr,  kp_tr]).astype(np.float32)
    X_val = np.hstack([emb_val_r, pr_val, kp_val]).astype(np.float32)
    print(f"  Fused feature dim: {X_tr.shape[1]}  "
          f"(UMAP:{UMAP_DIM} + PageRank:1 + KPI:1)")

    tr_ds = EmbeddingDataset(X_tr, y_tr)
    va_ds = EmbeddingDataset(X_val, y_val)
    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE)

    # ── Build and train the head ──────────────────────────────────────────
    head = ClassifierHead(n_classes, in_dim=X_tr.shape[1]).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2
    )

    class_counts = np.bincount(y_tr, minlength=n_classes).astype(float)
    # Use sqrt inverse frequency for smoother weighting on very small classes
    class_weights = torch.tensor(
        1.0 / np.sqrt(np.maximum(class_counts, 1)), dtype=torch.float32,
    ).to(device)
    class_weights = class_weights / class_weights.sum() * n_classes
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    best_f1, best_state = 0.0, None

    print(f"\n{'Epoch':<8}{'Loss':<12}{'Val Acc':<12}{'Val F1'}")
    print("-" * 46)

    for epoch in range(1, EPOCHS + 1):
        head.train()
        total_loss = 0.0
        for emb, lbl in tr_dl:
            emb, lbl = emb.to(device), lbl.to(device)
            optimizer.zero_grad()
            loss = loss_fn(head(emb), lbl)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / len(tr_dl)

        if epoch % 5 == 0 or epoch == EPOCHS:
            head.eval()
            all_preds, all_true = [], []
            with torch.no_grad():
                for emb, lbl in va_dl:
                    preds = head(emb.to(device)).argmax(dim=1).cpu()
                    all_preds.extend(preds.numpy())
                    all_true.extend(lbl.numpy())

            acc = np.mean(np.array(all_preds) == np.array(all_true))
            f1  = f1_score(all_true, all_preds, average="macro",
                           zero_division=0)
            print(f"{epoch:<8}{avg_loss:<12.4f}{acc:<12.2%}{f1:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.clone()
                              for k, v in head.state_dict().items()}

    # ── Save ──────────────────────────────────────────────────────────────
    head.load_state_dict(best_state)
    torch.save(head.state_dict(), f"{save_name}.pt")
    with open(f"{save_name}_labels.json", "w") as f:
        json.dump(label_encoder.classes_.tolist(), f, indent=2)

    print(f"\nBest val macro-F1: {best_f1:.4f}")
    print(f"Saved: {save_name}.pt  +  {save_name}_labels.json")

    # Full report
    head.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for emb, lbl in va_dl:
            preds = head(emb.to(device)).argmax(dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_true.extend(lbl.numpy())

    print(f"\nClassification report ({save_name}):")
    print(classification_report(
        all_true, all_preds,
        target_names=label_encoder.classes_,
        zero_division=0,
    ))

    report = classification_report(
        all_true, all_preds,
        target_names=label_encoder.classes_,
        output_dict=True, zero_division=0,
    )
    low = [n for n in label_encoder.classes_
           if report.get(n, {}).get("f1-score", 0) < MIN_F1]
    if low:
        print(f"WARNING: low f1 labels (< {MIN_F1}):")
        for l in low:
            print(f"  {l:<28} f1={report[l]['f1-score']:.2f}  "
                  f"support={report[l]['support']}")
    else:
        print(f"All labels passed f1 >= {MIN_F1} — ready to deploy.")

    return best_f1, umap_reducer


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Train KMA sentence classifiers (SAPC-optimised)."
    )
    parser.add_argument("--data",   default="training_data.csv")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch",  type=int, default=BATCH_SIZE)
    parser.add_argument("--no-pagerank", action="store_true",
                        help="Skip PageRank scoring (faster, less accurate)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("WARNING: No GPU — training will be slow (~30-40 min).")

    if not os.path.exists(args.data):
        print(f"Training data not found: {args.data}")
        print("Run step1_prepare_training_data.py first.")
        return

    df = pd.read_csv(args.data).dropna(
        subset=["text", "level_label", "class_label"]
    )
    print(f"Loaded {len(df)} training rows from {args.data}")
    texts = df["text"].tolist()

    # Load MiniLM once — shared between both classifiers
    print(f"\nLoading {MPNET_MODEL}...")
    sentence_model = SentenceTransformer(MPNET_MODEL, device=device)
    for p in sentence_model.parameters():
        p.requires_grad = False
    print("Model loaded and frozen.")

    # Opt 2: Compute PageRank + KPI scores once for all sentences
    print("\nComputing PageRank dependency scores...")
    if args.no_pagerank or not HAS_SPACY or not HAS_NX:
        pr_scores  = np.zeros(len(texts), dtype=np.float32)
        kpi_scores = compute_kpi_scores(texts)
        print("  PageRank disabled — using zeros")
    else:
        pr_scores  = compute_pagerank_scores(texts)
        kpi_scores = compute_kpi_scores(texts)
        pr_nonzero = (pr_scores > 0).sum()
        print(f"  PageRank: {pr_nonzero}/{len(texts)} non-zero scores")
        print(f"  KPI flag: {(kpi_scores > 0).sum()}/{len(texts)} sentences")

    t_start = time.time()

    # Train level classifier — also fits the shared UMAP reducer
    le_level = LabelEncoder().fit(df["level_label"])
    _, umap_reducer = train_one_classifier(
        texts, df["level_label"].tolist(),
        le_level, "level_classifier",
        device, sentence_model,
        umap_reducer=None,   # fit here
        pr_scores=pr_scores, kpi_scores=kpi_scores,
    )

    # Save UMAP reducer — step3 API needs this at inference time
    with open("umap_reducer.pkl", "wb") as f:
        pickle.dump(umap_reducer, f)
    print("\nSaved: umap_reducer.pkl")

    # Train class classifier — reuse the same UMAP reducer
    le_class = LabelEncoder().fit(df["class_label"])
    train_one_classifier(
        texts, df["class_label"].tolist(),
        le_class, "class_classifier",
        device, sentence_model,
        umap_reducer=umap_reducer,  # reuse
        pr_scores=pr_scores, kpi_scores=kpi_scores,
    )

    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Training complete in {total/60:.1f} minutes")
    print(f"Files produced:")
    print(f"  level_classifier.pt + level_classifier_labels.json")
    print(f"  class_classifier.pt + class_classifier_labels.json")
    print(f"  umap_reducer.pkl  <-- NEW: needed by step3 API")
    print(f"\nCopy all 5 files to your FastAPI server.")


if __name__ == "__main__":
    main()
