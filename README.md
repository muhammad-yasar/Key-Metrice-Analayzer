# KMA — Key Metric Analyser

**Automated extraction and classification of policy commitments from environmental policy documents.**

Developed at the Insight SFI Research Centre for Data Analytics, University of Galway, as part of the EU LIFE MultiPeat / ASPECT project.

---

## Overview

KMA is a five-step NLP pipeline that:

1. **Annotates** policy PDFs with Zephyr (bootstrap mode) to produce reviewed Excel files
2. **Trains** a sentence classifier (MiniLM + UMAP + PageRank fusion) on reviewed annotations
3. **Serves** the trained classifier via a FastAPI REST API
4. **Classifies** new policy documents at scale using the trained classifier
5. **Pushes** validated KPIs to Odoo ERP and Qdrant vector database

---

## Label Taxonomy

### Level (3 classes)
| Label | Description |
|---|---|
| `Policy Action` | Direct commitment with named actor + shall/will/must |
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
| `Miscellaneous` | Does not fit any above category |

---

## Repository Structure

```
kma-classifier/
│
├── step1_prepare_training_data.py   # Build training CSV from reviewed Excels
├── step2_train_classifier.py        # Train level + class classifier
├── step3_classifier_api.py          # FastAPI inference server
├── step4_annotate_policies.py       # Annotate PDFs with Zephyr / trained classifier
├── step5_push_validated.py          # Push KPIs to Odoo + Qdrant
│
├── scripts/
│   ├── check_distribution.py        # Check class distribution before training
│   └── merge_synthetic.py           # Merge synthetic examples into training CSV
│
├── data/
│   └── synthetic/
│       └── synthetic_rare_classes.csv  # Hand-crafted examples for rare classes
│
├── odoo/
│   └── odoo_kma_kpi_model.py        # Odoo custom model definition (kma.kpi)
│
├── docs/
│   └── annotation_guide.md          # Guide for human reviewers
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv kma
source kma/bin/activate          # Windows: kma\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Annotate policies (bootstrap with Zephyr)

```bash
# Ensure Ollama is running with Zephyr installed
ollama pull zephyr

# Annotate — outputs Excel files in exported_files/{policy_id}/
python step4_annotate_policies.py \
    --input ./exported_files \
    --limit 10 \
    --mistral-only
```

### 3. Review annotations

Open each `*_annotations.xlsx` in Excel. For each row:
- `Correct? (Y/N/Partial)` — mark Y if correct, N if wrong, Partial if level or class needs fixing
- `Notes` — for Partial rows, write the correction in plain English:
  - `"Policy action rather than outcome"`
  - `"Should be Spending"`
  - `"Environmental quality rather than miscellaneous"`

### 4. Check distribution

```bash
python scripts/check_distribution.py --input ./annotated_excel_file
```

Aim for 30+ examples per class before training.

### 5. Prepare training data

```bash
# Build CSV from reviewed Excels (reads Notes column for corrections)
python step1_prepare_training_data.py \
    --input ./annotated_excel_file \
    --output training_data.csv

# Merge synthetic rare-class examples
python scripts/merge_synthetic.py
```

### 6. Train classifier

```bash
python step2_train_classifier.py \
    --data training_data.csv \
    --epochs 80
```

Outputs: `level_classifier.pt`, `class_classifier.pt`, `umap_reducer.pkl`

### 7. Start classifier API

```bash
uvicorn step3_classifier_api:app --host 0.0.0.0 --port 8002
```

Check health: `curl http://localhost:8002/kma/health`

### 8. Annotate with trained classifier

```bash
# Now uses GPU classifier instead of Zephyr
python step4_annotate_policies.py \
    --input ./exported_files \
    --limit 10
```

### 9. Push to Odoo + Qdrant

```bash
python step5_push_validated.py \
    --input ./exported_files \
    --odoo-url http://your-odoo-server \
    --qdrant-url http://your-qdrant-server
```

---

## Model Architecture

Based on the SAPC (Smart Agile Prioritization and Clustering) framework:

```
Policy sentence
      │
      ▼
MiniLM-L6-v2 (384d, frozen)
      │
      ▼
UMAP (384d → 16d, cosine metric)
      │
      ├── PageRank dependency score (spaCy nsubj/dobj/ROOT graph)
      └── KPI regex flag (euro amounts, hectares, deadlines)
      │
      ▼
Fused features (18d)
      │
      ▼
Classifier head:
  Linear(18→64) + BatchNorm + Dropout(0.4)
  Linear(64→32) + ReLU
  Linear(32→n_classes)
      │
      ├── Level output (3 classes)
      └── Class output (9 classes)
```

**Training:**
- Loss: CrossEntropyLoss with sqrt inverse frequency weighting + label smoothing 0.1
- Scheduler: CosineAnnealingWarmRestarts(T_0=20, T_mult=2)
- Optimizer: AdamW, LR=5e-4, weight_decay=1e-4

---

## API Reference

### `POST /kma/classify`

```json
{
  "sentences": ["The Government shall restore 40,000 ha of peatland by 2027."],
  "policy_id": "316"
}
```

Response:
```json
{
  "results": [{
    "text": "The Government shall restore 40,000 ha...",
    "level": "Policy Action",
    "class": "Area",
    "lv_confidence": 0.92,
    "cl_confidence": 0.87,
    "is_direct": true,
    "has_metric": true,
    "pagerank_score": 0.741,
    "flagged_by": "classifier",
    "needs_mistral_review": false
  }],
  "count": 1,
  "elapsed": 0.043
}
```

### `GET /kma/health`

Returns model status, label lists, and feature dimensions.

---

## Odoo Integration

Install the custom `kma.kpi` model:

```bash
cp odoo/odoo_kma_kpi_model.py /path/to/odoo/addons/kma_odoo/models/
```

The model stores: policy reference, sentence text, level, class, confidence scores, KPI flag, PageRank score, and validation status.

---

## Performance

| Metric | Level classifier | Class classifier |
|---|---|---|
| Macro F1 (v1, 470 rows) | 0.42 | 0.36 |
| Macro F1 (v2, 614 rows) | 0.48* | 0.41* |
| Training time (CPU) | ~3 min | ~3 min |

*Estimated — retrain in progress

---

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{kma_classifier_2025,
  author    = {Khan, Muhammad Yasar},
  title     = {KMA: Key Metric Analyser for Environmental Policy Documents},
  year      = {2025},
  url       = {https://github.com/muhammad-yasar/Key-Metrice-Analayzer},
  note      = {Part of the EU LIFE MultiPeat / ASPECT project,
               Insight SFI Research Centre, University of Galway}
}
```

---

## License

MIT License — see LICENSE file.
