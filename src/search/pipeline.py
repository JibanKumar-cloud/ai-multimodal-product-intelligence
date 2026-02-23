"""End-to-end search pipeline: bi-encoder, cross-encoder, attribute boost."""
from __future__ import annotations
import json, time
from pathlib import Path
from typing import Optional
from loguru import logger
from src.search.bi_encoder import BiEncoderRetriever
from src.search.cross_encoder import CrossEncoderReranker
from src.search.attribute_boost import AttributeBooster

class SearchPipeline:
    def __init__(self, bi_encoder=None, cross_encoder=None, attribute_booster=None, retrieval_top_k=50, rerank_top_k=10, boost_alpha=0.3):
        self.bi_encoder = bi_encoder
        self.cross_encoder = cross_encoder
        self.attribute_booster = attribute_booster
        self.retrieval_top_k = retrieval_top_k
        self.rerank_top_k = rerank_top_k
        self.boost_alpha = boost_alpha
        self.product_names = {}

    @classmethod
    def from_config(cls, config):
        bi_cfg = config.get("bi_encoder", {})
        bi_encoder = BiEncoderRetriever(model_name=bi_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"), model_path=bi_cfg.get("model_path"))
        emb_path = bi_cfg.get("embeddings_path", "outputs/search/embeddings")
        index_path = bi_cfg.get("index_path", "outputs/search/embeddings/product.index")
        if Path(emb_path).exists():
            bi_encoder.load_embeddings(emb_path)
        if Path(index_path).exists():
            bi_encoder.load_faiss_index(index_path)
        ce_cfg = config.get("cross_encoder", {})
        cross_encoder = CrossEncoderReranker(model_name=ce_cfg.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"), model_path=ce_cfg.get("model_path"))
        ab_cfg = config.get("attribute_boost", {})
        attribute_booster = AttributeBooster(enriched_catalog_path=ab_cfg.get("catalog_path", "data/enriched_catalog.jsonl"))
        pipeline = cls(bi_encoder=bi_encoder, cross_encoder=cross_encoder, attribute_booster=attribute_booster,
                       retrieval_top_k=config.get("retrieval_top_k", 50), rerank_top_k=config.get("rerank_top_k", 10), boost_alpha=ab_cfg.get("alpha", 0.3))
        corpus_path = bi_cfg.get("corpus_path", "data/search/product_corpus.jsonl")
        if Path(corpus_path).exists():
            pipeline.load_product_names(corpus_path)
        return pipeline

    def load_product_names(self, corpus_path):
        with open(corpus_path) as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                self.product_names[item["product_id"]] = item.get("product_name", "")
        logger.info(f"Loaded {len(self.product_names)} product names")

    def search(self, query, top_k=10, stages=None):
        if stages is None:
            stages = ["bi_encoder", "cross_encoder", "attribute_boost"]
        timings = {}
        results = []
        if "bi_encoder" in stages and self.bi_encoder:
            t0 = time.time()
            results = self.bi_encoder.retrieve(query, top_k=self.retrieval_top_k)
            timings["bi_encoder_ms"] = round((time.time() - t0) * 1000, 1)
        else:
            return []
        if "cross_encoder" in stages and self.cross_encoder and results:
            t0 = time.time()
            results = self.cross_encoder.rerank(query, results, top_k=self.rerank_top_k)
            timings["cross_encoder_ms"] = round((time.time() - t0) * 1000, 1)
        if "attribute_boost" in stages and self.attribute_booster and results:
            t0 = time.time()
            results = self.attribute_booster.boost_results(query, results, alpha=self.boost_alpha)
            timings["attribute_boost_ms"] = round((time.time() - t0) * 1000, 1)
        total_ms = sum(timings.values())
        for r in results[:top_k]:
            r["product_name"] = self.product_names.get(r["product_id"], "")
            r["timings"] = timings
            r["total_latency_ms"] = round(total_ms, 1)
        return results[:top_k]

class BM25Baseline:
    def __init__(self, corpus_path=None):
        self.corpus = []
        self.product_ids = []
        self.bm25 = None
        self.product_names = {}
        if corpus_path and Path(corpus_path).exists():
            self.build(corpus_path)

    def build(self, corpus_path):
        from rank_bm25 import BM25Okapi
        with open(corpus_path) as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                self.corpus.append(item["text"].lower().split())
                self.product_ids.append(item["product_id"])
                self.product_names[item["product_id"]] = item.get("product_name", "")
        self.bm25 = BM25Okapi(self.corpus)
        logger.info(f"BM25 index: {len(self.corpus)} products")

    def search(self, query, top_k=10):
        if self.bm25 is None: return []
        t0 = time.time()
        scores = self.bm25.get_scores(query.lower().split())
        top_idx = scores.argsort()[-top_k:][::-1]
        latency = (time.time() - t0) * 1000
        return [{"product_id": self.product_ids[i], "product_name": self.product_names.get(self.product_ids[i], ""),
                 "score": float(scores[i]), "rank": r+1, "text": " ".join(self.corpus[i]), "total_latency_ms": round(latency, 1)}
                for r, i in enumerate(top_idx)]
