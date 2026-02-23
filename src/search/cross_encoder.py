"""Cross-Encoder reranker for second-stage relevance scoring."""
from __future__ import annotations
import json, time
from pathlib import Path
from typing import Optional
import torch
from loguru import logger

class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", model_path=None, device=None):
        from sentence_transformers import CrossEncoder
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        if model_path and Path(model_path).exists():
            logger.info(f"Loading fine-tuned cross-encoder from {model_path}")
            self.model = CrossEncoder(model_path, device=device, max_length=256)
        else:
            logger.info(f"Loading pre-trained cross-encoder: {model_name}")
            self.model = CrossEncoder(model_name, device=device, max_length=256)

    def train(self, train_path, val_path=None, output_dir="outputs/checkpoints/cross-encoder", epochs=3, batch_size=32, warmup_ratio=0.1):
        from sentence_transformers import InputExample
        from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
        from torch.utils.data import DataLoader
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        train_examples = []
        with open(train_path) as f:
            for line in f:
                if not line.strip(): continue
                ex = json.loads(line)
                train_examples.append(InputExample(texts=[ex["query"], ex["product"]], label=float(ex["score"])))
        logger.info(f"Cross-encoder training examples: {len(train_examples)}")
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        evaluator = None
        if val_path and Path(val_path).exists():
            vq, vp, vs = [], [], []
            with open(val_path) as f:
                for line in f:
                    if not line.strip(): continue
                    ex = json.loads(line)
                    vq.append(ex["query"]); vp.append(ex["product"]); vs.append(float(ex["score"]))
            mx = min(3000, len(vq))
            pairs = [[q, p] for q, p in zip(vq[:mx], vp[:mx])]
            evaluator = CECorrelationEvaluator(sentence_pairs=pairs, scores=vs[:mx], name="val")
        warmup_steps = int(len(train_dataloader) * epochs * warmup_ratio)
        logger.info(f"Training cross-encoder: {epochs} epochs, warmup={warmup_steps}")
        self.model.fit(train_dataloader=train_dataloader, epochs=epochs, warmup_steps=warmup_steps,
                       evaluator=evaluator, evaluation_steps=500, output_path=str(output_dir), save_best_model=True, show_progress_bar=True)
        logger.info(f"Cross-encoder saved to {output_dir}")

    def rerank(self, query, candidates, top_k=10):
        if not candidates: return []
        pairs = [[query, c["text"]] for c in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)
        for c, score in zip(candidates, scores):
            c["ce_score"] = float(score)
        reranked = sorted(candidates, key=lambda x: x["ce_score"], reverse=True)
        for i, c in enumerate(reranked[:top_k]):
            c["final_rank"] = i + 1
        return reranked[:top_k]
