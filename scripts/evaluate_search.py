#!/usr/bin/env python3
"""Evaluate search models: BM25 vs bi-encoder vs cross-encoder vs full pipeline."""
from __future__ import annotations
import argparse, json, sys, yaml
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from loguru import logger
from src.utils.logger import setup_logger
from src.search.metrics import evaluate_search, compare_search_models

class StageWrapper:
    def __init__(self, pipeline, stages):
        self.pipeline = pipeline
        self.stages = stages
    def search(self, query, top_k=10):
        return self.pipeline.search(query, top_k=top_k, stages=self.stages)

def main():
    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/search.yaml")
    parser.add_argument("--output-dir", default="outputs/search/evaluation")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test_path = config["data"]["test_queries"]
    if not Path(test_path).exists():
        logger.error(f"Test data not found: {test_path}")
        sys.exit(1)
    test_data = []
    with open(test_path) as f:
        for line in f:
            if line.strip(): test_data.append(json.loads(line))
    logger.info(f"Test pairs: {len(test_data)}")
    all_results = {}
    try:
        from src.search.pipeline import BM25Baseline
        bm25 = BM25Baseline(config["data"]["product_corpus"])
        bm25_m = evaluate_search(bm25, test_data)
        all_results["BM25"] = bm25_m
        logger.info(f"BM25 NDCG@10: {bm25_m['ndcg@10']}")
    except Exception as e:
        logger.warning(f"BM25 failed: {e}")
    from src.search.pipeline import SearchPipeline
    from src.search.bi_encoder import BiEncoderRetriever
    from src.search.cross_encoder import CrossEncoderReranker
    from src.search.attribute_boost import AttributeBooster
    bi_cfg = config.get("bi_encoder", {})
    ce_cfg = config.get("cross_encoder", {})
    ab_cfg = config.get("attribute_boost", {})
    bi_model_path = bi_cfg.get("model_path")
    bi_encoder = BiEncoderRetriever(model_path=bi_model_path if bi_model_path and Path(bi_model_path).exists() else None,
                                     model_name=bi_cfg.get("model_name"))
    emb_path = bi_cfg.get("embeddings_path", "outputs/search/embeddings")
    if Path(emb_path).exists():
        bi_encoder.load_embeddings(emb_path)
        idx_file = Path(emb_path) / "product.index"
        if idx_file.exists(): bi_encoder.load_faiss_index(str(idx_file))
        else: bi_encoder.build_faiss_index(save_path=emb_path)
    else:
        bi_encoder.encode_corpus(config["data"]["product_corpus"], save_path=emb_path)
        bi_encoder.build_faiss_index(save_path=emb_path)
    ce_model_path = ce_cfg.get("model_path")
    cross_encoder = CrossEncoderReranker(model_path=ce_model_path if ce_model_path and Path(ce_model_path).exists() else None,
                                          model_name=ce_cfg.get("model_name"))
    attribute_booster = AttributeBooster(enriched_catalog_path=ab_cfg.get("catalog_path"))
    pipeline = SearchPipeline(bi_encoder=bi_encoder, cross_encoder=cross_encoder, attribute_booster=attribute_booster,
                              retrieval_top_k=config.get("retrieval_top_k", 50), rerank_top_k=config.get("rerank_top_k", 10),
                              boost_alpha=ab_cfg.get("alpha", 0.3))
    corpus_path = config["data"]["product_corpus"]
    if Path(corpus_path).exists(): pipeline.load_product_names(corpus_path)
    for name, stages in [("Bi-Encoder", ["bi_encoder"]), ("Bi + Cross-Encoder", ["bi_encoder", "cross_encoder"]),
                          ("Full Pipeline", ["bi_encoder", "cross_encoder", "attribute_boost"])]:
        logger.info(f"\nEvaluating: {name}")
        wrapper = StageWrapper(pipeline, stages)
        m = evaluate_search(wrapper, test_data)
        all_results[name] = m
        logger.info(f"  NDCG@10: {m['ndcg@10']}, MRR: {m['mrr']}, Latency p99: {m['latency_p99_ms']}ms")
    table = compare_search_models(all_results)
    logger.info(f"\n{table}")
    with open(output_dir / "search_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    with open(output_dir / "search_comparison.md", "w") as f:
        f.write("# Search Relevance Evaluation\n\n" + table + "\n")
    logger.info(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()
