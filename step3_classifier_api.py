"""
step3_classifier_api.py
=======================
Updated with SAPC-paper optimizations:
  1. all-MiniLM-L6-v2 embedding (matches step2)
  2. PageRank + KPI feature fusion at inference time
  3. UMAP reduction (loads umap_reducer.pkl from step2)

Start:
    uvicorn step3_classifier_api:app --host 0.0.0.0 --port 8002
"""

import torch
import torch.nn as nn
import json
import re
import time
import pickle
import logging
import numpy as np
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("pip install sentence-transformers")

try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    HAS_SPACY = True
except Exception:
    HAS_SPACY = False

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI(title="KMA Sentence Classifier", version="2.0")

MODEL_DIR  = Path(__file__).parent
MPNET_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM   = 384
UMAP_DIM    = 64
FUSED_DIM   = UMAP_DIM + 2
CONFIDENCE_THRESHOLD = 0.60

# Globals loaded at startup
DEVICE       = None
sentence_model = None
umap_reducer   = None
level_head     = None
class_head     = None
level_labels   = None
class_labels   = None

# KPI regex
_KPI_RE = re.compile(
    "|".join([
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

INDIRECT_PATTERNS = [
    r"regulation\s*\(eu\)\s*\d{4}/\d+",
    r"regulation\s*\(ec\)\s*\d{4}/\d+",
    r"directive\s*\d{4}/\d+",
    r"\barticle\s+\d+\s+of\s+(regulation|directive|the)",
    r"pursuant\s+to\s+(regulation|directive|article|the\s+act)",
    r"in\s+accordance\s+with\s+(regulation|directive|article)",
    r"as\s+required\s+by\s+(regulation|directive|the\s+act)",
    r"under\s+regulation\s*\(",
    r"member\s+states?\s+shall",
]
_indirect_re = re.compile("|".join(INDIRECT_PATTERNS), re.IGNORECASE)


# ── Model definition (must match step2) ─────────────────────────────────────
class ClassifierHead(nn.Module):
    def __init__(self, n_classes: int, in_dim: int = FUSED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── PageRank scoring ─────────────────────────────────────────────────────────
def compute_pagerank_scores(texts: list) -> np.ndarray:
    """Compute PageRank importance scores for a batch of sentences."""
    if not HAS_SPACY or not HAS_NX:
        return np.zeros(len(texts), dtype=np.float32)
    try:
        dep_tokens = []
        for text in texts:
            doc = _nlp(text[:300])
            tokens = frozenset(
                t.lemma_.lower() for t in doc
                if t.dep_ in ("nsubj", "dobj", "ROOT") and len(t.lemma_) > 2
            )
            dep_tokens.append(tokens)

        G = nx.DiGraph()
        n = len(texts)
        G.add_nodes_from(range(n))
        for i in range(n):
            if not dep_tokens[i]:
                continue
            for j in range(n):
                if i != j and dep_tokens[i] & dep_tokens[j]:
                    G.add_edge(j, i)

        if G.number_of_edges() == 0:
            return np.zeros(n, dtype=np.float32)

        pr = nx.pagerank(G, alpha=0.85)
        scores = np.array([pr.get(i, 0.0) for i in range(n)], dtype=np.float32)
        mn, mx = scores.min(), scores.max()
        if mx > mn:
            scores = (scores - mn) / (mx - mn)
        return scores
    except Exception as e:
        logging.warning(f"PageRank failed: {e}")
        return np.zeros(len(texts), dtype=np.float32)


# ── Startup ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
def load_models():
    global DEVICE, sentence_model, umap_reducer
    global level_head, class_head, level_labels, class_labels

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {DEVICE}")

    # Load MiniLM
    logging.info(f"Loading {MPNET_MODEL}...")
    sentence_model = SentenceTransformer(MPNET_MODEL, device=DEVICE)
    for p in sentence_model.parameters():
        p.requires_grad = False
    logging.info("MiniLM loaded")

    # Load UMAP reducer
    umap_path = MODEL_DIR / "umap_reducer.pkl"
    if umap_path.exists():
        with open(umap_path, "rb") as f:
            umap_reducer = pickle.load(f)
        logging.info(f"UMAP reducer loaded from {umap_path}")
    else:
        logging.error(f"Missing: {umap_path} — run step2_train_classifier.py first")
        return

    # Load level classifier
    lv_pt   = MODEL_DIR / "level_classifier.pt"
    lv_lbls = MODEL_DIR / "level_classifier_labels.json"
    if not lv_pt.exists():
        logging.error(f"Missing: {lv_pt}")
        return
    with open(lv_lbls) as f:
        level_labels = json.load(f)
    level_head = ClassifierHead(len(level_labels)).to(DEVICE)
    level_head.load_state_dict(
        torch.load(lv_pt, map_location=DEVICE, weights_only=True)
    )
    level_head.eval()
    logging.info(f"Level classifier: {level_labels}")

    # Load class classifier
    cl_pt   = MODEL_DIR / "class_classifier.pt"
    cl_lbls = MODEL_DIR / "class_classifier_labels.json"
    with open(cl_lbls) as f:
        class_labels = json.load(f)
    class_head = ClassifierHead(len(class_labels)).to(DEVICE)
    class_head.load_state_dict(
        torch.load(cl_pt, map_location=DEVICE, weights_only=True)
    )
    class_head.eval()
    logging.info(f"Class classifier: {class_labels}")
    logging.info("KMA classifier service ready (v2 — SAPC optimised).")


# ── Core classify function ───────────────────────────────────────────────────
def classify_batch(sentences: list) -> list:
    if not sentences:
        return []

    # Opt 1: MiniLM embeddings
    with torch.no_grad():
        raw_emb = sentence_model.encode(
            sentences, batch_size=128,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    # Opt 3: UMAP reduction
    reduced = umap_reducer.transform(raw_emb).astype(np.float32)

    # Opt 2: Feature fusion
    pr_scores  = compute_pagerank_scores(sentences).reshape(-1, 1)
    kpi_scores = np.array(
        [1.0 if _KPI_RE.search(t) else 0.0 for t in sentences],
        dtype=np.float32,
    ).reshape(-1, 1)

    fused = np.hstack([reduced, pr_scores, kpi_scores])

    # Classify
    with torch.no_grad():
        X = torch.tensor(fused, dtype=torch.float32).to(DEVICE)
        lv_probs = torch.softmax(level_head(X), dim=1).cpu()
        cl_probs = torch.softmax(class_head(X), dim=1).cpu()

    results = []
    for i, text in enumerate(sentences):
        lv_conf, lv_idx = float(lv_probs[i].max()), int(lv_probs[i].argmax())
        cl_conf, cl_idx = float(cl_probs[i].max()), int(cl_probs[i].argmax())

        results.append({
            "text":                 text,
            "level":                level_labels[lv_idx],
            "class":                class_labels[cl_idx],
            "lv_confidence":        round(lv_conf, 4),
            "cl_confidence":        round(cl_conf, 4),
            "is_direct":            not bool(_indirect_re.search(text)),
            "has_metric":           bool(_KPI_RE.search(text)),
            "pagerank_score":       round(float(pr_scores[i, 0]), 4),
            "flagged_by":           "classifier",
            "needs_mistral_review": (
                lv_conf < CONFIDENCE_THRESHOLD or
                cl_conf < CONFIDENCE_THRESHOLD
            ),
        })

    return results


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.post("/kma/classify")
def classify_endpoint(payload: dict = Body(...)):
    sentences = payload.get("sentences", [])
    if not sentences:
        return JSONResponse({"results": []})
    t0 = time.time()
    results = classify_batch(sentences)
    elapsed = round(time.time() - t0, 3)
    return JSONResponse({
        "results": results,
        "count":   len(results),
        "elapsed": elapsed,
    })


@app.get("/kma/health")
def health():
    ready = all([
        sentence_model is not None,
        umap_reducer   is not None,
        level_head     is not None,
        class_head     is not None,
    ])
    return {
        "status":        "ready" if ready else "not_ready",
        "model":         MPNET_MODEL,
        "embed_dim":     EMBED_DIM,
        "umap_dim":      UMAP_DIM,
        "fused_dim":     FUSED_DIM,
        "level_labels":  level_labels,
        "class_labels":  class_labels,
        "pagerank":      HAS_SPACY and HAS_NX,
    }


@app.get("/kma/labels")
def labels():
    return {
        "level_labels": level_labels,
        "class_labels": class_labels,
    }
