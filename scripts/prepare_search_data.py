#!/usr/bin/env python3
"""Prepare WANDS query-product data for search model training."""
from __future__ import annotations
import json, random, sys
from pathlib import Path
from collections import defaultdict
import pandas as pd
from loguru import logger
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.wands_loader import WANDSLoader
from src.utils.logger import setup_logger

def build_product_text(row):
    parts = []
    for col in ["product_name", "product_class", "product_description"]:
        v = str(row.get(col, ""))
        if v and v != "nan":
            parts.append(v[:200] if col == "product_description" else v)
    return " | ".join(parts)

def prepare_search_data(wands_dir="data/raw/WANDS/dataset", output_dir="data/search", seed=42):
    setup_logger()
    random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loader = WANDSLoader(wands_dir)
    products = loader.products
    queries = loader.queries
    labels = loader.labels
    logger.info(f"Products: {len(products)}, Queries: {len(queries)}, Labels: {len(labels)}")
    product_text = {}
    product_names = {}
    for _, row in products.iterrows():
        pid = row["product_id"]
        product_text[pid] = build_product_text(row)
        product_names[pid] = str(row.get("product_name", ""))
    query_text = {}
    for _, row in queries.iterrows():
        query_text[row["query_id"]] = str(row["query"])
    query_products = defaultdict(lambda: {"exact": [], "partial": [], "irrelevant": []})
    for _, row in labels.iterrows():
        qid = row["query_id"]
        pid = row["product_id"]
        lbl = row["label"]
        if lbl == "Exact": query_products[qid]["exact"].append(pid)
        elif lbl == "Partial": query_products[qid]["partial"].append(pid)
        else: query_products[qid]["irrelevant"].append(pid)
    all_qids = sorted(query_products.keys())
    random.shuffle(all_qids)
    n = len(all_qids)
    train_qids = set(all_qids[:int(n*0.7)])
    val_qids = set(all_qids[int(n*0.7):int(n*0.85)])
    test_qids = set(all_qids[int(n*0.85):])
    logger.info(f"Query splits: train={len(train_qids)}, val={len(val_qids)}, test={len(test_qids)}")
    bi_encoder_train = []
    for qid in train_qids:
        qtext = query_text.get(qid, "")
        if not qtext: continue
        positives = query_products[qid]["exact"] + query_products[qid]["partial"]
        negatives = query_products[qid]["irrelevant"]
        if not positives or not negatives: continue
        for pos_pid in positives:
            pos_text = product_text.get(pos_pid, "")
            if not pos_text: continue
            neg_candidates = [p for p in negatives if p in product_text]
            if not neg_candidates: continue
            neg_pid = random.choice(neg_candidates)
            bi_encoder_train.append({"query": qtext, "positive": pos_text, "negative": product_text[neg_pid],
                                     "query_id": int(qid), "pos_product_id": int(pos_pid), "neg_product_id": int(neg_pid)})
    random.shuffle(bi_encoder_train)
    with open(output_dir / "bi_encoder_train.jsonl", "w") as f:
        for ex in bi_encoder_train:
            f.write(json.dumps(ex) + "\n")
    logger.info(f"Bi-encoder triplets: {len(bi_encoder_train)}")
    cross_encoder_train = []
    cross_encoder_val = []
    for qid_set, out_list in [(train_qids, cross_encoder_train), (val_qids, cross_encoder_val)]:
        for qid in qid_set:
            qtext = query_text.get(qid, "")
            if not qtext: continue
            for label_name, score in [("exact", 1.0), ("partial", 0.5), ("irrelevant", 0.0)]:
                for pid in query_products[qid][label_name]:
                    ptext = product_text.get(pid, "")
                    if not ptext: continue
                    out_list.append({"query": qtext, "product": ptext, "score": score,
                                     "label": label_name, "query_id": int(qid), "product_id": int(pid)})
    random.shuffle(cross_encoder_train)
    random.shuffle(cross_encoder_val)
    with open(output_dir / "cross_encoder_train.jsonl", "w") as f:
        for ex in cross_encoder_train:
            f.write(json.dumps(ex) + "\n")
    with open(output_dir / "cross_encoder_val.jsonl", "w") as f:
        for ex in cross_encoder_val:
            f.write(json.dumps(ex) + "\n")
    logger.info(f"Cross-encoder train: {len(cross_encoder_train)}, val: {len(cross_encoder_val)}")
    test_data = []
    for qid in test_qids:
        qtext = query_text.get(qid, "")
        if not qtext: continue
        for label_name, score in [("exact", 2), ("partial", 1), ("irrelevant", 0)]:
            for pid in query_products[qid][label_name]:
                ptext = product_text.get(pid, "")
                test_data.append({"query": qtext, "query_id": int(qid), "product_id": int(pid),
                                  "product_text": ptext, "product_name": product_names.get(pid, ""),
                                  "relevance": score, "label": label_name})
    with open(output_dir / "test_queries.jsonl", "w") as f:
        for ex in test_data:
            f.write(json.dumps(ex) + "\n")
    logger.info(f"Test pairs: {len(test_data)}")
    with open(output_dir / "product_corpus.jsonl", "w") as f:
        for pid, text in product_text.items():
            f.write(json.dumps({"product_id": int(pid), "product_name": product_names.get(pid, ""), "text": text}) + "\n")
    logger.info(f"Product corpus: {len(product_text)}")
    logger.info("SEARCH DATA PREPARATION COMPLETE")

if __name__ == "__main__":
    prepare_search_data()
