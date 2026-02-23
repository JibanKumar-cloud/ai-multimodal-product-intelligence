#!/usr/bin/env python3
"""Train search models: bi-encoder + cross-encoder."""
from __future__ import annotations
import argparse, sys, time, yaml
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from loguru import logger
from src.utils.logger import setup_logger

def train_bi_encoder(config):
    from src.search.bi_encoder import BiEncoderRetriever
    bi_cfg = config.get("bi_encoder", {})
    train_cfg = bi_cfg.get("training", {})
    retriever = BiEncoderRetriever(model_name=bi_cfg.get("model_name"))
    logger.info("=" * 50 + "\nTRAINING BI-ENCODER\n" + "=" * 50)
    start = time.time()
    retriever.train(train_path=config["data"]["bi_encoder_train"],
                    output_dir=bi_cfg.get("model_path", "outputs/checkpoints/bi-encoder"),
                    epochs=train_cfg.get("epochs", 5), batch_size=train_cfg.get("batch_size", 32),
                    warmup_ratio=train_cfg.get("warmup_ratio", 0.1), eval_path=config["data"].get("cross_encoder_val"))
    logger.info(f"Bi-encoder done in {(time.time()-start)/60:.1f} min")
    corpus_path = config["data"]["product_corpus"]
    emb_path = bi_cfg.get("embeddings_path", "outputs/search/embeddings")
    logger.info("Encoding product corpus...")
    retriever.encode_corpus(corpus_path, save_path=emb_path)
    logger.info("Building FAISS index...")
    retriever.build_faiss_index(save_path=emb_path)
    return retriever

def train_cross_encoder(config):
    from src.search.cross_encoder import CrossEncoderReranker
    ce_cfg = config.get("cross_encoder", {})
    train_cfg = ce_cfg.get("training", {})
    reranker = CrossEncoderReranker(model_name=ce_cfg.get("model_name"))
    logger.info("=" * 50 + "\nTRAINING CROSS-ENCODER\n" + "=" * 50)
    start = time.time()
    reranker.train(train_path=config["data"]["cross_encoder_train"],
                   val_path=config["data"].get("cross_encoder_val"),
                   output_dir=ce_cfg.get("model_path", "outputs/checkpoints/cross-encoder"),
                   epochs=train_cfg.get("epochs", 3), batch_size=train_cfg.get("batch_size", 32),
                   warmup_ratio=train_cfg.get("warmup_ratio", 0.1))
    logger.info(f"Cross-encoder done in {(time.time()-start)/60:.1f} min")
    return reranker

def main():
    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/search.yaml")
    parser.add_argument("--stage", choices=["bi-encoder", "cross-encoder", "both"], default="both")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    if not Path(config["data"]["search_dir"]).exists():
        logger.error("Search data not found. Run: python scripts/prepare_search_data.py")
        sys.exit(1)
    if args.stage in ("bi-encoder", "both"):
        train_bi_encoder(config)
    if args.stage in ("cross-encoder", "both"):
        train_cross_encoder(config)
    logger.info("SEARCH TRAINING COMPLETE. Next: python scripts/evaluate_search.py")

if __name__ == "__main__":
    main()
