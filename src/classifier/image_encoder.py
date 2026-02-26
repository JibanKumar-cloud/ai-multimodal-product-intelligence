"""Image Encoder using frozen CLIP ViT.

Extracts per-image embeddings. Does NOT pool — that's the attention pooler's job.
Supports precomputing and caching embeddings to disk.
"""
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from loguru import logger


class ImageEncoder(nn.Module):
    """Frozen CLIP ViT image encoder with caching support."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32",
                 freeze: bool = True):
        super().__init__()
        self.model_name = model_name
        self._model = None
        self._processor = None
        self.freeze = freeze
        self.embed_dim = 768  # CLIP ViT-B/32 output dim

    def _load_model(self):
        """Lazy load model (heavy, only when needed)."""
        if self._model is not None:
            return
        from transformers import CLIPVisionModel, CLIPImageProcessor
        logger.info(f"Loading image encoder: {self.model_name}")
        self._model = CLIPVisionModel.from_pretrained(self.model_name)
        self._processor = CLIPImageProcessor.from_pretrained(self.model_name)
        if self.freeze:
            for param in self._model.parameters():
                param.requires_grad = False
            self._model.eval()
        logger.info(f"Image encoder loaded (frozen={self.freeze})")

    @property
    def model(self):
        self._load_model()
        return self._model

    @property
    def processor(self):
        self._load_model()
        return self._processor

    @torch.no_grad()
    def encode_image(self, image) -> np.ndarray:
        """Encode a single PIL image to embedding vector.

        Args:
            image: PIL Image

        Returns:
            numpy array of shape [768]
        """
        self._load_model()
        inputs = self._processor(images=image, return_tensors="pt")
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self._model(**inputs)
        # Use pooler output (CLS token)
        embedding = outputs.pooler_output.squeeze(0).cpu().numpy()
        return embedding

    @torch.no_grad()
    def encode_images(self, images: list) -> np.ndarray:
        """Encode multiple PIL images to embedding matrix.

        Args:
            images: list of PIL Images

        Returns:
            numpy array of shape [N, 768]
        """
        if not images:
            return np.zeros((0, self.embed_dim), dtype=np.float32)

        self._load_model()
        inputs = self._processor(images=images, return_tensors="pt")
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self._model(**inputs)
        embeddings = outputs.pooler_output.cpu().numpy()
        return embeddings

    def precompute_product(self, product_id: str, image_paths: list,
                           output_dir: str) -> str:
        """Precompute and cache embeddings for one product.

        Args:
            product_id: product identifier
            image_paths: list of image file paths
            output_dir: directory to save embeddings

        Returns:
            path to saved embedding file
        """
        from PIL import Image

        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{product_id}_img.npy")

        if os.path.exists(save_path):
            return save_path

        # Load and encode images
        valid_images = []
        for p in image_paths:
            try:
                img = Image.open(p).convert("RGB")
                valid_images.append(img)
            except Exception as e:
                logger.warning(f"Failed to load {p}: {e}")

        if valid_images:
            embeddings = self.encode_images(valid_images)
        else:
            embeddings = np.zeros((0, self.embed_dim), dtype=np.float32)

        np.save(save_path, embeddings)
        return save_path

    @staticmethod
    def load_cached(product_id: str, embeddings_dir: str,
                    k_max: int = 5) -> tuple:
        """Load cached embeddings with padding to k_max.

        Args:
            product_id: product identifier
            embeddings_dir: directory with cached .npy files
            k_max: pad/truncate to this many images

        Returns:
            (embeddings [k_max, 768], mask [k_max]) as numpy arrays
        """
        path = os.path.join(embeddings_dir, f"{product_id}_img.npy")
        embed_dim = 768

        if os.path.exists(path):
            raw = np.load(path)
            n_images = raw.shape[0]
        else:
            raw = np.zeros((0, embed_dim), dtype=np.float32)
            n_images = 0

        # Pad or truncate to k_max
        padded = np.zeros((k_max, embed_dim), dtype=np.float32)
        mask = np.zeros(k_max, dtype=np.float32)

        if n_images > 0:
            n_use = min(n_images, k_max)
            # Random sample if more than k_max (during training)
            if n_images > k_max:
                indices = np.random.choice(n_images, k_max, replace=False)
                padded[:] = raw[indices]
                mask[:] = 1.0
            else:
                padded[:n_use] = raw[:n_use]
                mask[:n_use] = 1.0

        return padded, mask

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward pass for images already preprocessed.

        Args:
            pixel_values: [B, 3, 224, 224] or [B*K, 3, 224, 224]

        Returns:
            embeddings [B, 768] or [B*K, 768]
        """
        self._load_model()
        outputs = self._model(pixel_values=pixel_values)
        return outputs.pooler_output
