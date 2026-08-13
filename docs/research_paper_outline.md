# KMA: Automated Extraction and Classification of Policy Commitments from Environmental Policy Documents

**Authors:** Muhammad Yasar Khan, [Co-authors TBD]
**Affiliation:** Insight SFI Research Centre for Data Analytics, University of Galway
**Project:** EU LIFE MultiPeat / ASPECT

---

## Abstract (150-200 words)

Environmental policy documents contain thousands of sentences, only a fraction of which represent concrete, measurable commitments. Manual extraction of these Key Policy Metrics (KPMs) is time-consuming, inconsistent across reviewers, and does not scale to the volume of policies required for cross-national environmental monitoring.

We present KMA (Key Metric Analyser), a five-step NLP pipeline that automatically extracts and classifies policy commitments from environmental policy documents in multiple languages. KMA classifies each sentence along two dimensions: Level (Policy Action, Policy Outcome, Unsure) and Class (9 categories including Area, Emissions, Spending, Site Status). The pipeline combines PDF extraction using PyMuPDF, bootstrap annotation using the Zephyr language model, and a sentence classifier built on SAPC-inspired feature fusion: Sentence-BERT embeddings, UMAP dimensionality reduction, and PageRank dependency scoring.

Evaluated on [N] annotated policies across [X] languages, KMA achieves macro-F1 of [X] for Level classification and [X] for Class classification, outperforming both rule-based baselines and zero-shot LLM approaches. The pipeline is integrated with Odoo ERP and Qdrant vector database for downstream policy analysis.

---

## 1. Introduction

### 1.1 Motivation

Environmental policy monitoring requires tracking commitments across hundreds of national and EU-level policy documents. The ASPECT project [CITE] requires systematic extraction of peatland-related policy commitments from documents published by EU member states. Current approaches rely on manual review by domain experts, which is:

- **Slow:** A 100-page policy document requires 4-8 hours of expert review
- **Inconsistent:** Inter-annotator agreement on policy classification is typically 60-75%
- **Not scalable:** The ASPECT database covers 27 EU member states × multiple policy domains
- **Language-limited:** Experts typically work in 1-2 languages

### 1.2 Research Questions

- **RQ1:** Can a sentence classifier reliably distinguish between policy commitments (Policy Action), policy goals (Policy Outcome), and background text (Unsure) in environmental policy documents?
- **RQ2:** Can the same classifier identify the thematic class of a policy sentence across 9 categories with sufficient precision for policy monitoring?
- **RQ3:** Does the combination of semantic embeddings, dependency-based PageRank scoring, and UMAP dimensionality reduction (as proposed in SAPC [CITE]) improve classification compared to embeddings alone?
- **RQ4:** How does classification performance generalise across languages and policy domains?

### 1.3 Contributions

1. **KMA pipeline:** An end-to-end open-source pipeline for policy sentence extraction and classification
2. **Annotation methodology:** A human-in-the-loop annotation protocol with free-text correction parsing
3. **Label taxonomy:** A two-dimensional (Level × Class) taxonomy validated across [N] policies
4. **Multilingual evaluation:** Performance analysis across [X] EU languages
5. **Integration:** Production deployment integrated with Odoo ERP and Qdrant

---

## 2. Related Work

### 2.1 Policy Text Mining

[Review of: policy IE, NLP for legislation, automated policy analysis]

- Lippi et al. (2019) — argument mining in legal documents
- Bommarito & Katz (2018) — quantitative legal informatics
- Pilehvar & Camacho-Collados (2020) — NLP for social science

### 2.2 Sentence Classification

[Review of: BERT for classification, sentence transformers, zero-shot approaches]

- Devlin et al. (2019) — BERT
- Reimers & Gurevych (2019) — Sentence-BERT
- Brown et al. (2020) — GPT-3 few-shot

### 2.3 SAPC Framework

Radwan et al. (2025) [CITE] propose Smart Agile Prioritization and Clustering (SAPC) combining:
- BERT embeddings (all-MiniLM-L6-v2)
- PageRank dependency scoring via spaCy
- UMAP dimensionality reduction
- PSO-optimised K-Means clustering

We adapt the SAPC feature fusion approach to sentence-level policy classification.

### 2.4 Environmental Policy NLP

[Review of: climate policy text analysis, ASPECT/MultiPeat related work]

---

## 3. Methodology

### 3.1 Data Collection

#### 3.1.1 Policy Documents

We collected [N] environmental policy documents from [countries/sources], covering:
- National peatland strategies (Ireland, Scotland, England)
- UNCCD strategic framework documents (COP13, COP14)
- EU nature restoration policies
- National climate action plans

Documents span [X] languages: English, [others].

#### 3.1.2 PDF Extraction

PyMuPDF (fitz) extracts text using the blocks method, which preserves reading order for both single-column and multi-column layouts. Each page's text blocks are sorted by vertical band (20px groups) then horizontal position. Extracted text undergoes:

- Hyphenated word split repair (`"afforesta-\ntion"` → `"afforestation"`)
- Apostrophe-s split repair (`"biodiversity\ns"` → `"biodiversity's"`)
- Single newline replacement with space (PDF line wraps)
- Double newline preservation (paragraph breaks)
- Section header stripping (e.g. "FORESTRY - ACTION A7")
- Trailing garbage truncation

#### 3.1.3 Sentence Chunking

Text is split into complete sentences at `.!?;` boundaries and paragraph breaks. Unlike fixed-size chunking, the KMA chunker never cuts mid-sentence: if a single sentence exceeds the maximum chunk size (350 characters), it is kept whole.

### 3.2 Annotation Protocol

#### 3.2.1 Bootstrap Annotation

Zephyr-7B (via Ollama) provides initial labels using a 10-shot prompt with examples from real peatlands policy documents. The prompt includes explicit class descriptions distinguishing common confusions (e.g. Emissions requires a specific % target, not merely mentioning climate change).

#### 3.2.2 Pre-filtering

Before classification, sentences are filtered by rule-based patterns to remove:
- Table of contents entries
- Footnotes and bibliography references
- Page headers (including UN document codes: ICCD/, FCCC/, CBD/)
- Contact information and photo credits
- Background science definitions
- Historical past-tense statements

#### 3.2.3 Human Review

Domain experts review each classified sentence in Excel, marking:
- `Y` (correct), `N` (wrong, discard), or `Partial` (one label needs fixing)
- Free-text notes parsed by a natural language correction extractor

The note parser handles expressions like:
- `"Policy action rather than outcome"` → corrects Level
- `"Should be Spending"` → corrects Class
- `"Environmental quality rather than miscellaneous"` → corrects Class

#### 3.2.4 Inter-annotator Agreement

[To be computed — Cohen's Kappa on overlapping subset]

### 3.3 Feature Engineering (SAPC-inspired)

Following Radwan et al. (2025), we fuse three feature types:

**1. Semantic embeddings**
```
E_BERT = f_SBERT(sentence)  ∈ ℝ^384
```
Using `all-MiniLM-L6-v2`, fine-tuned for sentence similarity.

**2. PageRank dependency score**
SpaCy extracts nominal subjects (nsubj), direct objects (dobj), and root verbs (ROOT) from each sentence. A directed graph connects sentences sharing dependency tokens. PageRank (α=0.85) scores each sentence's structural centrality.

```
G = (V, E)  where V = sentences, E = shared dependency tokens
PR(R_i) = (1-d)/N + d × Σ_{j∈In(R_i)} PR(R_j)/|Out(R_j)|
```

**3. KPI regex flag**
A binary feature (0/1) indicating presence of quantitative policy signals: hectare amounts, euro figures, percentage targets, named deadlines.

**4. UMAP dimensionality reduction**
```
X_reduced = UMAP(E_BERT, n_components=16, metric="cosine")
X_fused   = [X_reduced || PR_score || KPI_flag]  ∈ ℝ^18
```

n_components=16 chosen as ≈√(N_training) to avoid overfitting on small datasets.

### 3.4 Classifier Architecture

Two independent classifier heads (Level and Class) trained on the same fused features:

```
h1 = ReLU(BatchNorm(Linear(18 → 64)))
h1 = Dropout(0.4)(h1)
h2 = ReLU(Linear(64 → 32))
ŷ  = Linear(32 → n_classes)
```

**Loss function:** CrossEntropyLoss with sqrt inverse frequency weighting and label smoothing (ε=0.1) to handle class imbalance.

**Training:** AdamW (lr=5e-4, weight_decay=1e-4), CosineAnnealingWarmRestarts (T_0=20, T_mult=2), 80 epochs.

### 3.5 Post-processing

Rule-based class corrections override model predictions for clear cases:
- Sentences about "introducing guidance / plans / databases" → Knowledge Resource
- Sentences with specific CO2 % target AND future deadline → Emissions (otherwise → Environment Quality)
- Sentences with actual euro/dollar amounts → Spending
- Sentences referencing EU directives/regulations → Policy Action class

Hedge pattern detection forces Level=Unsure for sentences containing: "could", "may", "will be considered", "as of [year]", "can help", "it is noted that".

---

## 4. Experimental Evaluation

### 4.1 Dataset Statistics

| Policy | Language | Pages | Annotated | Y | N | Partial |
|---|---|---|---|---|---|---|
| National Peatlands Strategy (IE) | EN | 84 | 305 | 145 | 160 | - |
| COP21 Add.1 | EN | 9 | 30 | 8 | 15 | 7 |
| England Peat Action Plan | EN | 86 | 141 | 61 | 80 | - |
| Scotland Peatland Strategy | EN | - | 96 | 44 | 52 | - |
| Nature Recovery Strategy | EN | 86 | - | 157 | 374 | - |
| **Total** | | | **614** | | | |

### 4.2 Label Distribution

| Level | Count | % |
|---|---|---|
| Policy Action | 484 | 78.8% |
| Policy Outcome | 90 | 14.7% |
| Unsure | 40 | 6.5% |

| Class | Count | % |
|---|---|---|
| Environment Quality | 177 | 28.8% |
| Knowledge Resource | 149 | 24.3% |
| Miscellaneous | 99 | 16.1% |
| Site Status | 63 | 10.3% |
| Practical Resource | 50 | 8.1% |
| Area | 27 | 4.4% |
| Policy Action | 31 | 5.0% |
| Spending | 13 | 2.1% |
| Emissions | 5 | 0.8% |

### 4.3 Classification Results

[Table to be filled after final training run with full dataset]

| Model | Level F1 | Class F1 | Level Acc | Class Acc |
|---|---|---|---|---|
| Majority baseline | 0.26 | 0.09 | 78.8% | 28.8% |
| Zero-shot Zephyr | 0.41* | 0.32* | 65% | 42% |
| MiniLM only (no UMAP) | 0.36 | 0.27 | 39% | 20% |
| MiniLM + UMAP | 0.42 | 0.36 | 70% | 38% |
| **KMA (full pipeline)** | **TBD** | **TBD** | **TBD** | **TBD** |

*Estimated from bootstrap annotation accuracy

### 4.4 Ablation Study

[Compare: with/without PageRank, with/without KPI flag, different UMAP dims]

### 4.5 Per-class Analysis

[Report precision/recall/F1 per class for best model]

---

## 5. System Integration

### 5.1 FastAPI Inference Server

The trained classifier is served via a FastAPI REST API (step3) supporting:
- Batch classification (up to 128 sentences)
- Confidence scores for both Level and Class
- PageRank score per sentence
- Automatic flagging of low-confidence predictions for human review

### 5.2 Odoo ERP Integration

Validated KPIs are stored in a custom Odoo model (`kma.kpi`) linked to policy records. This enables:
- Structured querying of policy commitments by class, level, and confidence
- Dashboard visualisation of policy coverage
- Export for reporting to ASPECT project stakeholders

### 5.3 Qdrant Vector Database

Sentence embeddings are stored in Qdrant for semantic search:
- Find policies with similar commitments to a query
- Cluster related commitments across documents
- Identify policy gaps by searching for commitment types with few results

---

## 6. Limitations

1. **Class imbalance:** Emissions (5 examples) and Spending (13 examples) are severely underrepresented due to their genuine rarity in environmental policy documents.

2. **Language coverage:** Current training data is English-only. Cross-lingual transfer via multilingual MiniLM is planned but not yet evaluated.

3. **Domain specificity:** Training data focuses on peatland/wetland policies. Generalisation to other environmental domains (marine, forestry, agriculture) requires additional annotation.

4. **Chunk granularity:** Sentences exceeding the chunk size are kept whole, which may include multiple policy commitments.

5. **Bootstrap noise:** Zephyr's bootstrap annotations have ~50% accuracy on difficult UN policy documents; human review is essential.

---

## 7. Conclusion and Future Work

We present KMA, an open-source pipeline for automated extraction and classification of environmental policy commitments. The system combines state-of-the-art sentence embeddings with dependency-based PageRank scoring and UMAP dimensionality reduction, achieving [results] on [N] annotated policies.

**Future work:**
- Multilingual evaluation using `paraphrase-multilingual-MiniLM-L12-v2`
- Integration of DSPy for automated prompt optimisation once 500+ reviewed examples are available
- Extension to marine and agricultural policy domains
- Real-time policy monitoring dashboard via Odoo
- Comparison with GPT-4 zero-shot and fine-tuned approaches

---

## References

[To be completed]

- Devlin et al. (2019). BERT: Pre-training of deep bidirectional transformers.
- Reimers & Gurevych (2019). Sentence-BERT: Sentence embeddings using siamese BERT.
- Radwan et al. (2025). Smart Agile Prioritization and Clustering (SAPC). IEEE Access.
- McInnes et al. (2018). UMAP: Uniform manifold approximation and projection.
- Page et al. (1999). The PageRank citation ranking: Bringing order to the web.
- [ASPECT project reference]
- [EU LIFE MultiPeat reference]

---

## Appendix A — Label Taxonomy Detail

[Full definitions with examples and counter-examples — see annotation_guide.md]

## Appendix B — Prompt Template

[Full Zephyr prompt with 10 few-shot examples]

## Appendix C — Pre-filter Patterns

[Full regex patterns for all pre-filters]

## Appendix D — Code and Data Availability

Code: https://github.com/YOUR_USERNAME/kma-classifier
Data: [Zenodo DOI — to be added]
Model weights: [Zenodo DOI — to be added]
