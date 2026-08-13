---
language:
- en
- fr
- de
- sk
- pl
tags:
- text-classification
- policy-analysis
- environmental-policy
- peatlands
- nlp
- sentence-classification
license: mit
---

# KMA — Key Metric Analyser

**Automated extraction and classification of policy commitments from environmental policy documents.**

Developed at the **Insight SFI Research Centre for Data Analytics, University of Galway**
as part of the EU LIFE MultiPeat / ASPECT project.

---

## ⚠️ Important — Do NOT use `AutoModel`

This is a **custom PyTorch classifier**, not a HuggingFace transformer model.
Loading with `AutoModel.from_pretrained()` will fail.
Use the loading code provided below.

---

## What it does

KMA classifies policy sentences along two dimensions:

### Level (3 classes)
| Label | Description |
|---|---|
| `Policy Action` | Direct commitment — named actor + shall/will/must |
| `Policy Outcome` | Result or goal the policy aims to achieve |
| `Unsure` | Vague, hedged, aspirational, or background text |

### Class (9 classes)
| Label | Description |
|---|---|
| `Area` | Land area, hectares, geographic scope |
| `Emissions` | Specific CO2/GHG reduction target with % or deadline |
| `Site Status` | Named protected sites, SACs, NHAs, restoration programmes |
| `Spending` | Actual euro/dollar amounts, annual payments, grants |
| `Policy Action` | Reference to another law, EU regulation, directive |
| `Knowledge Resource` | Plans, reports, databases, guidance documents produced |
| `Practical Resource` | Compensation packages, relocation schemes, training |
| `Environment Quality` | Water quality, WFD, habitat condition, biodiversity |
| `Miscellaneous` | Does not fit any above |

---

## Model Architecture

Based on the **SAPC (Smart Agile Prioritization and Clustering)** framework
(Radwan et al., 2025, IEEE Access):

```
Policy sentence
      │
      ▼
all-MiniLM-L6-v2  (384d, frozen)
      │
      ▼
UMAP  (384d → 16d, cosine metric)
      │
      ├── PageRank dependency score
      └── KPI regex flag
      │
      ▼
Fused features (18d)
      │
      ▼
ClassifierHead:
  Linear(18→64) + BatchNorm1d + ReLU + Dropout(0.4)
  Linear(64→32) + ReLU + Dropout(0.3)
  Linear(32→n_classes)
      │
      ├── level_classifier.pt  →  Level label (3 classes)
      └── class_classifier.pt  →  Class label (9 classes)
```

**Files in this repo:**

| File | Description |
|---|---|
| `level_classifier.pt` | PyTorch weights for Level classifier |
| `class_classifier.pt` | PyTorch weights for Class classifier |
| `umap_reducer.pkl` | Fitted UMAP reducer (384d → 16d) |
| `level_classifier_labels.json` | Index → Level label mapping |
| `class_classifier_labels.json` | Index → Class label mapping |

---

## Installation

```bash
pip install torch sentence-transformers umap-learn huggingface_hub numpy
```

---

## Usage

```python
import torch
import torch.nn as nn
import pickle
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download

# ── Step 1: Download model files ──────────────────────────────────────────────
repo_id = "link2yasar/kma-classifier"
for f in ["level_classifier.pt", "class_classifier.pt",
          "umap_reducer.pkl", "level_classifier_labels.json",
          "class_classifier_labels.json"]:
    hf_hub_download(repo_id=repo_id, filename=f, local_dir="./kma_model")

# ── Step 2: Load components ───────────────────────────────────────────────────
with open("kma_model/umap_reducer.pkl", "rb") as f:
    umap_reducer = pickle.load(f)

with open("kma_model/level_classifier_labels.json") as f:
    level_labels = json.load(f)

with open("kma_model/class_classifier_labels.json") as f:
    class_labels = json.load(f)

# ── Step 3: Define model architecture (must match exactly) ────────────────────
class ClassifierHead(nn.Module):
    def __init__(self, n_classes, in_dim=18):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes),
        )
    def forward(self, x):
        return self.net(x)

level_head = ClassifierHead(len(level_labels))
class_head = ClassifierHead(len(class_labels))
level_head.load_state_dict(torch.load("kma_model/level_classifier.pt",
                                       map_location="cpu", weights_only=True))
class_head.load_state_dict(torch.load("kma_model/class_classifier.pt",
                                       map_location="cpu", weights_only=True))
level_head.eval()
class_head.eval()

# ── Step 4: Load embedding model ──────────────────────────────────────────────
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ── Step 5: Classify sentences ────────────────────────────────────────────────
def classify(sentences: list) -> list:
    # Embed with MiniLM
    emb = embedder.encode(sentences, normalize_embeddings=True,
                          convert_to_numpy=True)
    # Reduce with UMAP
    reduced = umap_reducer.transform(emb).astype(np.float32)
    # Fuse (PageRank=0, KPI=0 when used standalone without spaCy)
    fused = np.hstack([
        reduced,
        np.zeros((len(sentences), 1), dtype=np.float32),  # PageRank
        np.zeros((len(sentences), 1), dtype=np.float32),  # KPI flag
    ])
    # Classify
    with torch.no_grad():
        X       = torch.tensor(fused)
        lv_prob = torch.softmax(level_head(X), dim=1)
        cl_prob = torch.softmax(class_head(X), dim=1)

    return [{
        "text":          s,
        "level":         level_labels[int(lv_prob[i].argmax())],
        "class":         class_labels[int(cl_prob[i].argmax())],
        "lv_confidence": round(float(lv_prob[i].max()), 3),
        "cl_confidence": round(float(cl_prob[i].max()), 3),
    } for i, s in enumerate(sentences)]

# ── Example ───────────────────────────────────────────────────────────────────
results = classify([
    "The Government shall restore 40,000 ha of peatland by 2027.",
    "A payment of €1,500 per annum will be made to affected turf-cutters.",
    "Forests can help to provide temporary mitigation of climate change.",
    "DAFM will establish a national database of peatland sites.",
])

for r in results:
    print(f"{r['level']:<18} {r['class']:<22} "
          f"conf={r['lv_confidence']:.2f}/{r['cl_confidence']:.2f}")
    print(f"  {r['text']}")
```

**Expected output:**
```
Policy Action      Area                   conf=0.85/0.78
  The Government shall restore 40,000 ha of peatland by 2027.
Policy Action      Spending               conf=0.82/0.71
  A payment of €1,500 per annum will be made to affected turf-cutters.
Unsure             Environment Quality    conf=0.74/0.69
  Forests can help to provide temporary mitigation of climate change.
Policy Action      Knowledge Resource     conf=0.88/0.75
  DAFM will establish a national database of peatland sites.
```

---

## Training Data

Trained on **614 reviewed sentences** from 5 environmental policy documents:

| Policy | Country | Language | Reviewed rows |
|---|---|---|---|
| National Peatlands Strategy 2015 | Ireland | EN | 145 |
| UNCCD COP21 Strategic Framework | International | EN | 15 |
| England Peat Action Plan 2021 | England | EN | 61 |
| Scotland Peatland Strategy | Scotland | EN | 44 |
| Nature Recovery Strategy | England | EN | 157 |
| Synthetic rare-class examples | — | EN | 48 |

Class distribution after training:

| Class | Count |
|---|---|
| Environment Quality | 177 |
| Knowledge Resource | 149 |
| Miscellaneous | 99 |
| Site Status | 63 |
| Practical Resource | 50 |
| Policy Action (class) | 31 |
| Area | 27 |
| Spending | 13 |
| Emissions | 5 |

---

## Performance

| Metric | Level classifier | Class classifier |
|---|---|---|
| Macro F1 | 0.42 | 0.36 |
| Accuracy | 70% | 38% |
| Training samples | 614 | 614 |

> **Note:** Performance is limited by small training set size, particularly
> for rare classes (Emissions, Spending). More annotated policies will
> improve accuracy significantly. Active development ongoing.

---

## Limitations

- Trained primarily on English peatland/wetland policy documents
- Emissions (5 examples) and Spending (13 examples) classes have very low support
- Not yet evaluated on non-English documents
- Performance will improve as more annotated data is added

---

## Full Pipeline

For the complete annotation and training pipeline including:
- PDF extraction (PyMuPDF)
- Bootstrap annotation with Zephyr
- Human review workflow
- Odoo + Qdrant integration

See: **[GitHub Repository](https://github.com/link2yasar/kma-classifier)**

---

## Citation

```bibtex
@software{khan2025kma,
  author    = {Khan, Muhammad Yasar},
  title     = {KMA: Key Metric Analyser for Environmental Policy Documents},
  year      = {2025},
  url       = {https://huggingface.co/link2yasar/kma-classifier},
  note      = {Insight SFI Research Centre, University of Galway.
               EU LIFE MultiPeat / ASPECT project.}
}
```

---

## License

MIT License
