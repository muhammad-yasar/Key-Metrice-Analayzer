"""
step5_push_validated.py
=======================
Run this AFTER reviewing your Excel files.

Reads validated annotations from Excel files and:
  1. Creates kma.kpi records in Odoo (one per unique KPI found)
  2. Pushes evidencing sentences to Qdrant kma_sentences collection
  3. Links Qdrant sentence IDs back to the Odoo KPI records

Run from your laptop (or any machine with access to Odoo + Qdrant):
    python step5_push_validated.py \
        --input ./exported_files \
        --odoo-url http://your-odoo-server \
        --odoo-db your_db \
        --odoo-user admin \
        --odoo-password yourpassword \
        --qdrant-url http://140.203.155.230:6333

Requirements:
    pip install xmlrpc-client qdrant-client sentence-transformers openpyxl pandas
"""

import argparse
import glob
import json
import os
import sys
import uuid
import xmlrpc.client
from collections import defaultdict
from pathlib import Path

import pandas as pd

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, Filter, FieldCondition, MatchValue,
        PointStruct, VectorParams,
    )
except ImportError:
    print("ERROR: pip install qdrant-client")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: pip install sentence-transformers")
    sys.exit(1)

KMA_SENTENCES_COLLECTION = "kma_sentences"
MPNET_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CONFIDENCE_THRESHOLD = 0.60


# ── Odoo XML-RPC helpers ──────────────────────────────────────────────────────
def odoo_connect(url: str, db: str, user: str, password: str):
    """Returns (uid, models_proxy)."""
    common = xmlrpc.client.ServerProxy(f"{url}/xmlrpc/2/common")
    uid = common.authenticate(db, user, password, {})
    if not uid:
        raise ValueError(f"Odoo authentication failed for user {user}")
    models = xmlrpc.client.ServerProxy(f"{url}/xmlrpc/2/object")
    return uid, models, db


def odoo_create_kpi(
    models, db, uid, password,
    policy_id: int,
    level_label: str,
    class_label: str,
    metric_text: str,
    metric_value: float,
    sentence_count: int,
) -> int:
    """Create one kma.kpi record. Returns the new record ID."""
    return models.execute_kw(
        db, uid, password,
        "kma.kpi", "create",
        [{
            "policy_id":      policy_id,
            "level_label":    level_label,
            "class_label":    class_label,
            "metric_text":    metric_text or "",
            "metric_value":   metric_value or 0.0,
            "sentence_count": sentence_count,
            "reviewed":       False,
        }],
    )


def odoo_write_qdrant_ids(
    models, db, uid, password,
    kpi_id: int,
    qdrant_ids: list,
):
    """Write the list of Qdrant sentence UUIDs back to kma.kpi."""
    models.execute_kw(
        db, uid, password,
        "kma.kpi", "write",
        [[kpi_id], {
            "qdrant_sentence_ids": json.dumps(qdrant_ids),
            "sentence_count": len(qdrant_ids),
        }],
    )


# ── Group sentences into KPIs ─────────────────────────────────────────────────
def group_into_kpis(rows: list) -> list:
    """
    Group labelled sentences into KPI buckets.
    Each unique (level, class) combination becomes one KPI.
    metric_text = the sentence with the highest confidence.
    Returns list of KPI dicts.
    """
    groups = defaultdict(list)
    for r in rows:
        key = (r["level_label"], r["class_label"])
        groups[key].append(r)

    kpis = []
    for (level, cls), sentences in groups.items():
        # Pick the highest-confidence sentence as the representative metric_text
        best = max(sentences,
                   key=lambda s: (s.get("lv_confidence", 0)
                                  + s.get("cl_confidence", 0)))
        kpis.append({
            "level_label":    level,
            "class_label":    cls,
            "metric_text":    best["text"][:200],
            "metric_value":   0.0,
            "sentence_count": len(sentences),
            "sentences":      sentences,
        })
    return kpis


# ── Main push function ────────────────────────────────────────────────────────
def push_policy(
    excel_path: Path,
    policy_id: int,
    models, db, uid, odoo_password,
    qclient: QdrantClient,
    mpnet,
    dry_run: bool = False,
):
    print(f"\n  Policy {policy_id}: {excel_path.name}")

    try:
        df = pd.read_excel(excel_path, sheet_name="Annotations")
    except Exception as e:
        print(f"    ERROR reading Excel: {e}")
        return 0, 0

    # Filter to verified direct sentences only
    verified = df[
        df["Correct? (Y/N/Partial)"].isin(["Y", "Partial"]) &
        (df["Direct?"] == "Y")
    ].copy()

    if verified.empty:
        print(f"    No verified direct sentences — skipping")
        return 0, 0

    print(f"    {len(verified)} verified direct sentences")

    # Build row dicts
    rows = []
    for _, row in verified.iterrows():
        rows.append({
            "text":          str(row.get("Text", "")).strip(),
            "level_label":   str(row.get("Level", "Unsure")).strip(),
            "class_label":   str(row.get("Class", "Miscellaneous")).strip(),
            "lv_confidence": float(row.get("lv_conf", 0.5) or 0.5),
            "cl_confidence": float(row.get("cl_conf", 0.5) or 0.5),
            "page":          str(row.get("Page", "")),
            "flagged_by":    str(row.get("Flagged by", "classifier")),
        })

    rows = [r for r in rows if r["text"] and r["text"] != "nan"]
    if not rows:
        return 0, 0

    kpis = group_into_kpis(rows)
    print(f"    Grouped into {len(kpis)} KPIs")

    kpis_created = 0
    sentences_stored = 0

    for kpi in kpis:
        # 1. Create Odoo kma.kpi record
        if not dry_run:
            try:
                kpi_id = odoo_create_kpi(
                    models, db, uid, odoo_password,
                    policy_id=policy_id,
                    level_label=kpi["level_label"],
                    class_label=kpi["class_label"],
                    metric_text=kpi["metric_text"],
                    metric_value=kpi["metric_value"],
                    sentence_count=kpi["sentence_count"],
                )
                print(f"    Created kma.kpi id={kpi_id}  "
                      f"[{kpi['level_label']} / {kpi['class_label']}]  "
                      f"{kpi['sentence_count']} sentences")
            except Exception as e:
                print(f"    ERROR creating kpi: {e}")
                continue
        else:
            kpi_id = 0
            print(f"    [DRY RUN] Would create kma.kpi "
                  f"[{kpi['level_label']} / {kpi['class_label']}]  "
                  f"{kpi['sentence_count']} sentences")

        kpis_created += 1

        # 2. Embed sentences and push to Qdrant
        sentences_text = [s["text"] for s in kpi["sentences"]]
        embeddings = mpnet.encode(
            sentences_text, batch_size=64, convert_to_numpy=True
        ).tolist()

        points = []
        qdrant_ids = []
        for s, vec in zip(kpi["sentences"], embeddings):
            point_id = str(uuid.uuid4())
            qdrant_ids.append(point_id)
            points.append(PointStruct(
                id=point_id,
                vector=vec,
                payload={
                    "span_text":    s["text"],
                    "level":        s["level_label"],
                    "class":        s["class_label"],
                    "lv_conf":      s["lv_confidence"],
                    "cl_conf":      s["cl_confidence"],
                    "page":         s["page"],
                    "flagged_by":   s["flagged_by"],
                    "is_direct":    True,
                    "active":       True,
                    "policy_id":    str(policy_id),
                    "kpi_id":       kpi_id,
                },
            ))

        if not dry_run:
            try:
                qclient.upsert(
                    collection_name=KMA_SENTENCES_COLLECTION,
                    points=points,
                )
                sentences_stored += len(points)

                # 3. Write Qdrant IDs back to Odoo
                odoo_write_qdrant_ids(
                    models, db, uid, odoo_password,
                    kpi_id, qdrant_ids,
                )
                print(f"      Stored {len(points)} sentences in Qdrant, "
                      f"linked to kpi_id={kpi_id}")
            except Exception as e:
                print(f"      ERROR storing sentences: {e}")
        else:
            sentences_stored += len(points)
            print(f"      [DRY RUN] Would store {len(points)} sentences")

    return kpis_created, sentences_stored


# ── Qdrant collection setup ───────────────────────────────────────────────────
def ensure_kma_collection(qclient: QdrantClient):
    if not qclient.collection_exists(KMA_SENTENCES_COLLECTION):
        qclient.create_collection(
            collection_name=KMA_SENTENCES_COLLECTION,
            vectors_config=VectorParams(
                size=768, distance=Distance.COSINE
            ),
        )
        print(f"Created Qdrant collection: {KMA_SENTENCES_COLLECTION}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Push validated KMA annotations to Odoo + Qdrant"
    )
    parser.add_argument("--input",         required=True,
                        help="exported_files folder")
    parser.add_argument("--odoo-url",      required=True)
    parser.add_argument("--odoo-db",       required=True)
    parser.add_argument("--odoo-user",     default="admin")
    parser.add_argument("--odoo-password", required=True)
    parser.add_argument("--qdrant-url",    default="http://127.0.0.1:6333")
    parser.add_argument("--dry-run",       action="store_true",
                        help="Print what would happen without writing anything")
    args = parser.parse_args()

    # Connect to services
    print("Connecting to Odoo...")
    try:
        uid, models, db = odoo_connect(
            args.odoo_url, args.odoo_db,
            args.odoo_user, args.odoo_password,
        )
        print(f"  Odoo: connected as uid={uid}")
    except Exception as e:
        print(f"  Odoo connection failed: {e}")
        sys.exit(1)

    print("Connecting to Qdrant...")
    qclient = QdrantClient(args.qdrant_url)
    ensure_kma_collection(qclient)
    print(f"  Qdrant: connected at {args.qdrant_url}")

    print("Loading mpnet for sentence embedding...")
    mpnet = SentenceTransformer(MPNET_MODEL, device="cpu")
    print("  mpnet: loaded")

    # Find all annotation Excels
    base = Path(args.input)
    excel_files = sorted(
        base.glob("**/*_annotations.xlsx")
    )
    print(f"\nFound {len(excel_files)} annotation Excel files\n")

    total_kpis = 0
    total_sentences = 0

    for excel_path in excel_files:
        policy_dir = excel_path.parent
        policy_id_str = policy_dir.name
        if not policy_id_str.isdigit():
            print(f"  Skipping non-numeric folder: {policy_id_str}")
            continue
        policy_id = int(policy_id_str)

        kpis_created, sents = push_policy(
            excel_path=excel_path,
            policy_id=policy_id,
            models=models,
            db=db,
            uid=uid,
            odoo_password=args.odoo_password,
            qclient=qclient,
            mpnet=mpnet,
            dry_run=args.dry_run,
        )
        total_kpis      += kpis_created
        total_sentences += sents

    print(f"\n{'='*50}")
    print(f"{'DRY RUN — ' if args.dry_run else ''}Complete")
    print(f"  KPIs created:       {total_kpis}")
    print(f"  Sentences stored:   {total_sentences}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
