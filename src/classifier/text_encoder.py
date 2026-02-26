"""Text Encoder using frozen DistilBERT.

Encodes "[TITLE] ... [DESC] ..." into a single embedding vector.
Supports precomputing and caching embeddings to disk.
"""
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from loguru import logger


class TextEncoder(nn.Module):
    """Frozen DistilBERT text encoder with caching support."""

    def __init__(self, model_name: str = "distilbert-base-uncased",
                 max_length: int = 128, freeze: bool = True):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self._model = None
        self._tokenizer = None
        self.freeze = freeze
        self.embed_dim = 768

    def _load_model(self):
        """Lazy load model."""
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer
        logger.info(f"Loading text encoder: {self.model_name}")
        self._model = AutoModel.from_pretrained(self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.freeze:
            for param in self._model.parameters():
                param.requires_grad = False
            self._model.eval()
        logger.info(f"Text encoder loaded (frozen={self.freeze})")

    @property
    def model(self):
        self._load_model()
        return self._model

    @property
    def tokenizer(self):
        self._load_model()
        return self._tokenizer

    @staticmethod
    def build_text(product_name: str, product_description: str = "",
                   product_class: str = "") -> str:
        """Build standardized input text from product fields.

        Args:
            product_name: product title
            product_description: product description
            product_class: product category

        Returns:
            formatted text string
        """
        parts = []
        if product_name:
            parts.append(f"[TITLE] {product_name}")
        if product_class:
            parts.append(f"[CLASS] {product_class}")
        if product_description:
            # Truncate long descriptions
            desc = str(product_description)[:300]
            parts.append(f"[DESC] {desc}")
        return " ".join(parts)

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        """Encode a single text string to embedding.

        Args:
            text: input text string

        Returns:
            numpy array [768]
        """
        self._load_model()
        inputs = self._tokenizer(
            text, return_tensors="pt", max_length=self.max_length,
            truncation=True, padding="max_length"
        )
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self._model(**inputs)
        # Mean pooling over token embeddings (better than CLS for DistilBERT)
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        token_embeds = outputs.last_hidden_state
        summed = (token_embeds * attention_mask).sum(dim=1)
        counts = attention_mask.sum(dim=1)
        embedding = (summed / counts).squeeze(0).cpu().numpy()
        return embedding

    @torch.no_grad()
    def encode_batch(self, texts: list) -> np.ndarray:
        """Encode a batch of texts.

        Args:
            texts: list of text strings

        Returns:
            numpy array [B, 768]
        """
        self._load_model()
        inputs = self._tokenizer(
            texts, return_tensors="pt", max_length=self.max_length,
            truncation=True, padding="max_length"
        )
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self._model(**inputs)
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        token_embeds = outputs.last_hidden_state
        summed = (token_embeds * attention_mask).sum(dim=1)
        counts = attention_mask.sum(dim=1)
        embeddings = (summed / counts).cpu().numpy()
        return embeddings

    def precompute_product(self, product_id: str, text: str,
                           output_dir: str) -> str:
        """Precompute and cache text embedding for one product.

        Args:
            product_id: product identifier
            text: formatted input text
            output_dir: directory to save embeddings

        Returns:
            path to saved file
        """
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{product_id}_txt.npy")

        if os.path.exists(save_path):
            return save_path

        embedding = self.encode_text(text)
        np.save(save_path, embedding)
        return save_path

    @staticmethod
    def load_cached(product_id: str, embeddings_dir: str) -> np.ndarray:
        """Load cached text embedding.

        Args:
            product_id: product identifier
            embeddings_dir: directory with cached .npy files

        Returns:
            numpy array [768]
        """
        path = os.path.join(embeddings_dir, f"{product_id}_txt.npy")
        if os.path.exists(path):
            return np.load(path)
        return np.zeros(768, dtype=np.float32)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass for tokenized text.

        Args:
            input_ids: [B, T]
            attention_mask: [B, T]

        Returns:
            text embeddings [B, 768]
        """
        self._load_model()
        outputs = self._model(input_ids=input_ids,
                              attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1)
        token_embeds = outputs.last_hidden_state
        summed = (token_embeds * mask).sum(dim=1)
        counts = mask.sum(dim=1)
        return summed / counts
