"""Search evaluation metrics: NDCG, MRR, Recall, latency."""
from __future__ import annotations
import math, time
from collections import defaultdict
import numpy as np
from loguru import logger

def dcg_at_k(rels, k):
    return sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels[:k]))

def ndcg_at_k(rels, k):
    dcg = dcg_at_k(rels, k)
    idcg = dcg_at_k(sorted(rels, reverse=True), k)
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_search(pipeline, test_data, ks=[5, 10, 20], threshold=1.0):
    query_labels = defaultdict(list)
    query_texts = {}
    for item in test_data:
        qid = item["query_id"]
        query_labels[qid].append({"product_id": item["product_id"], "relevance": item["relevance"]})
        query_texts[qid] = item["query"]
    logger.info(f"Evaluating on {len(query_labels)} queries")
    all_ndcg = {k: [] for k in ks}
    all_mrr = []
    latencies = []
    max_k = max(ks)
    for qid, labels in query_labels.items():
        query = query_texts[qid]
        rel_map = {l["product_id"]: l["relevance"] for l in labels}
        t0 = time.time()
        results = pipeline.search(query, top_k=max_k)
        latencies.append((time.time() - t0) * 1000)
        rels = [rel_map.get(r["product_id"], 0.0) for r in results]
        for k in ks:
            all_ndcg[k].append(ndcg_at_k(rels, k))
        for i, r in enumerate(rels):
            if r >= threshold:
                all_mrr.append(1.0 / (i + 1))
                break
        else:
            all_mrr.append(0.0)
    metrics = {}
    for k in ks:
        metrics[f"ndcg@{k}"] = round(np.mean(all_ndcg[k]), 4)
    metrics["mrr"] = round(np.mean(all_mrr), 4)
    metrics["num_queries"] = len(query_labels)
    lats = np.array(latencies)
    metrics["latency_p50_ms"] = round(np.percentile(lats, 50), 1)
    metrics["latency_p95_ms"] = round(np.percentile(lats, 95), 1)
    metrics["latency_p99_ms"] = round(np.percentile(lats, 99), 1)
    return metrics

def compare_search_models(results, primary="ndcg@10"):
    display = ["ndcg@5","ndcg@10","ndcg@20","mrr","latency_p50_ms","latency_p99_ms"]
    display = [m for m in display if any(m in r for r in results.values())]
    lines = ["| Model | " + " | ".join(display) + " |", "|" + "---|" * (len(display)+1)]
    for name, m in sorted(results.items(), key=lambda x: x[1].get(primary, 0), reverse=True):
        vals = [f"{m.get(d, '-'):.4f}" if isinstance(m.get(d), float) and "latency" not in d else str(m.get(d, "-")) for d in display]
        lines.append(f"| {name} | " + " | ".join(vals) + " |")
    return "\n".join(lines)
