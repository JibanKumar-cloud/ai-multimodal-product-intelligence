"""Bi-Encoder retriever using SentenceTransformers + FAISS."""
from __future__ import annotations
import json, time
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from loguru import logger

class BiEncoderRetriever:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", model_path=None, device=None):
        from sentence_transformers import SentenceTransformer
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        if model_path and Path(model_path).exists():
            logger.info(f"Loading fine-tuned bi-encoder from {model_path}")
            self.model = SentenceTransformer(model_path, device=device)
        else:
            logger.info(f"Loading pre-trained bi-encoder: {model_name}")
            self.model = SentenceTransformer(model_name, device=device)
        self.product_embeddings = None
        self.product_ids = None
        self.product_texts = None
        self.faiss_index = None

    def train(self, train_path, output_dir="outputs/checkpoints/bi-encoder", epochs=5, batch_size=32, warmup_ratio=0.1, eval_path=None):
        from sentence_transformers import InputExample, losses, evaluation
        from torch.utils.data import DataLoader
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        train_examples = []
        with open(train_path) as f:
            for line in f:
                if not line.strip(): continue
                ex = json.loads(line)
                train_examples.append(InputExample(texts=[ex["query"], ex["positive"], ex["negative"]]))
        logger.info(f"Bi-encoder training examples: {len(train_examples)}")
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        train_loss = losses.TripletLoss(model=self.model, distance_metric=losses.TripletDistanceMetric.COSINE, triplet_margin=0.3)
        evaluator = None
        if eval_path and Path(eval_path).exists():
            eq, ep, es = [], [], []
            with open(eval_path) as f:
                for line in f:
                    if not line.strip(): continue
                    ex = json.loads(line)
                    eq.append(ex["query"]); ep.append(ex["product"]); es.append(ex["score"])
            mx = min(2000, len(eq))
            evaluator = evaluation.EmbeddingSimilarityEvaluator(eq[:mx], ep[:mx], es[:mx], name="val")
        warmup_steps = int(len(train_dataloader) * epochs * warmup_ratio)
        logger.info(f"Training bi-encoder: {epochs} epochs, warmup={warmup_steps}")
        self.model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=epochs, warmup_steps=warmup_steps,
                       evaluator=evaluator, evaluation_steps=500, output_path=str(output_dir), save_best_model=True, show_progress_bar=True)
        logger.info(f"Bi-encoder saved to {output_dir}")

    def encode_corpus(self, corpus_path, batch_size=128, save_path=None):
        logger.info(f"Encoding product corpus from {corpus_path}")
        products = []
        with open(corpus_path) as f:
            for line in f:
                if not line.strip(): continue
                products.append(json.loads(line))
        self.product_ids = np.array([p["product_id"] for p in products])
        self.product_texts = [p["text"] for p in products]
        start = time.time()
        self.product_embeddings = self.model.encode(self.product_texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
        elapsed = time.time() - start
        logger.info(f"Encoded {len(products)} products in {elapsed:.1f}s, dim={self.product_embeddings.shape[1]}")
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            np.save(save_path / "embeddings.npy", self.product_embeddings)
            np.save(save_path / "product_ids.npy", self.product_ids)
            with open(save_path / "product_texts.json", "w") as f:
                json.dump(self.product_texts, f)
            logger.info(f"Embeddings saved to {save_path}")

    def load_embeddings(self, load_path):
        load_path = Path(load_path)
        self.product_embeddings = np.load(load_path / "embeddings.npy")
        self.product_ids = np.load(load_path / "product_ids.npy")
        with open(load_path / "product_texts.json") as f:
            self.product_texts = json.load(f)
        logger.info(f"Loaded {len(self.product_ids)} embeddings from {load_path}")

    def build_faiss_index(self, save_path=None):
        import faiss
        if self.product_embeddings is None:
            raise ValueError("Encode corpus first")
        dim = self.product_embeddings.shape[1]
        n = len(self.product_embeddings)
        if n < 10000:
            self.faiss_index = faiss.IndexFlatIP(dim)
            idx_type = "FlatIP"
        else:
            nlist = min(int(np.sqrt(n)), 256)
            quantizer = faiss.IndexFlatIP(dim)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self.faiss_index.train(self.product_embeddings.astype(np.float32))
            self.faiss_index.nprobe = min(16, nlist)
            idx_type = f"IVF{nlist}"
        self.faiss_index.add(self.product_embeddings.astype(np.float32))
        logger.info(f"FAISS index: {idx_type}, {n} vectors, dim={dim}")
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.faiss_index, str(save_path / "product.index"))

    def load_faiss_index(self, index_path):
        import faiss
        self.faiss_index = faiss.read_index(str(index_path))
        logger.info(f"Loaded FAISS index: {self.faiss_index.ntotal} vectors")

    def retrieve(self, query, top_k=50):
        if self.faiss_index is None:
            raise ValueError("Build or load FAISS index first")
        query_emb = self.model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, indices = self.faiss_index.search(query_emb, top_k)
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < 0: continue
            results.append({"product_id": int(self.product_ids[idx]), "score": float(score),
                            "text": self.product_texts[idx] if self.product_texts else "", "rank": rank + 1})
        return results
