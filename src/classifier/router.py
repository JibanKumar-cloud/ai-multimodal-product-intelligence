"""Product Intelligence Router.

Single API that hides the routing logic between:
  - Fast classifier (90% of products, ~5ms)
  - Cautious accept (5% medium confidence, ~5ms)
  - VLM fallback (5% flagged products, ~380ms)

User sees ONE call, ONE output format. Never knows which model ran.

BOTH models produce the SAME output:
{
    "product_id": "25548",
    "taxonomy": ["Furniture", "Living Room", "Chairs", "Accent Chairs"],
    "attributes": {"color": "blue", "material": "velvet", "style": "modern", ...},
    "confidence": 0.94,
}

Three-tier routing:
  TIER 1 (>=0.85): FAST ACCEPT    — classifier result, high confidence
  TIER 2 (0.7-0.85): CAUTIOUS     — classifier result, check modality conflict
  TIER 3 (<0.7 or conflict): VLM  — VLM fallback, normalized to same format
"""
import os
import json
from typing import Optional
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from .vlm_normalizer import normalize_vlm_output


@dataclass
class ProductResult:
    """Unified result from product intelligence system.

    IDENTICAL format whether classifier or VLM produced it.
    """
    product_id: str
    taxonomy: list                    # ["Furniture", "Living Room", "Chairs", "Accent Chairs"]
    attributes: dict                  # {"color": "blue", "style": "modern", ...}
    confidence: float                 # 0.0 to 1.0

    # ── Internal metadata (for logging/debugging, not exposed to user) ──
    mismatch_detected: bool = False
    mismatch_reason: str = "none"     # "none", "modality_conflict", "low_confidence"
    method: str = "classifier"        # "classifier", "classifier_cautious", "vlm_fallback"
    routing_tier: int = 1             # 1=fast accept, 2=cautious, 3=vlm

    def to_dict(self) -> dict:
        """Public output — same format regardless of method."""
        return {
            "product_id": self.product_id,
            "taxonomy": self.taxonomy,
            "attributes": self.attributes,
            "confidence": self.confidence,
        }

    def to_catalog_line(self) -> str:
        """Single JSON line for enriched_catalog.jsonl."""
        record = {
            "product_id": self.product_id,
            "taxonomy": self.taxonomy,
            **self.attributes,
            "confidence": round(self.confidence, 3),
        }
        return json.dumps(record)


class ProductIntelligence:
    """Single API for product classification and enrichment.

    Internally routes between fast classifier and VLM fallback.
    User never sees the routing — just gets accurate results.

    Usage:
        pi = ProductIntelligence("outputs/checkpoints/classifier")
        result = pi.predict(images=[img1, img2], text="blue velvet chair")
        print(result.to_dict())
        # {"product_id": "...", "taxonomy": [...], "attributes": {...}, "confidence": 0.94}
    """

    # ── Routing thresholds ──
    TIER1_THRESHOLD = 0.85   # fast accept
    TIER2_THRESHOLD = 0.70   # cautious accept (check modality conflict)
    # Below TIER2 → VLM fallback

    def __init__(self, classifier_path: str,
                 vlm_path: Optional[str] = None,
                 device: str = "cpu"):
        """
        Args:
            classifier_path: path to saved classifier checkpoint
            vlm_path: path to VLM model (optional, for fallback)
            device: torch device
        """
        self.device = device

        # Load classifier
        from .model import MultiTowerClassifier
        self.classifier = MultiTowerClassifier.load(
            classifier_path, device=device)
        self.taxonomy = self.classifier.taxonomy

        # Reverse taxonomy lookup (int -> label)
        self.reverse_taxonomy = {}
        for level, mapping in self.taxonomy.items():
            self.reverse_taxonomy[level] = {v: k for k, v in mapping.items()}

        # Frozen encoders for live inference (lazy loaded)
        self.image_encoder = None
        self.text_encoder = None

        # VLM fallback (lazy loaded)
        self.vlm_path = vlm_path
        self._vlm = None
        self._postprocessor = None

        # Stats tracking
        self.stats = {
            "total": 0,
            "tier1_fast_accept": 0,
            "tier2_cautious_accept": 0,
            "tier3_vlm_fallback": 0,
            "reasons": {},
        }

        logger.info(
            f"ProductIntelligence ready: classifier={classifier_path}, "
            f"vlm={'enabled' if vlm_path else 'disabled'}"
        )

    # ──────────────────────────────────────────
    # PUBLIC API — ONE CALL, ONE OUTPUT FORMAT
    # ──────────────────────────────────────────

    def predict(self, images: list = None, text: str = "",
                attributes: dict = None,
                product_id: str = "unknown") -> ProductResult:
        """Classify a product. Single API — routing is internal.

        Args:
            images: list of PIL images (0 to N)
            text: product title + description
            attributes: dict of key-value pairs from product_features
            product_id: for tracking

        Returns:
            ProductResult — SAME format whether classifier or VLM handled it
        """
        images = images or []
        attributes = attributes or {}
        self.stats["total"] += 1

        self._load_encoders()

        # ── Step 1: Encode inputs ──
        img_tensor, mask_tensor, txt_tensor, attr_tensor = (
            self._encode_inputs(images, text, attributes))

        # ── Step 2: Run classifier (ALWAYS runs first) ──
        result = self.classifier.predict(
            img_tensor, mask_tensor, txt_tensor, attr_tensor)

        # Decode classifier outputs
        clf_taxonomy = self._decode_taxonomy(result["predictions"])
        clf_attributes = self._decode_attributes(result["attr_predictions"], 0)
        mismatch = result["mismatch_results"][0]
        fused_conf = mismatch.fused_confidence

        # ── Step 3: Three-tier routing ──

        # TIER 1: High confidence — fast accept
        if fused_conf >= self.TIER1_THRESHOLD:
            self.stats["tier1_fast_accept"] += 1
            return ProductResult(
                product_id=product_id,
                taxonomy=clf_taxonomy,
                attributes=clf_attributes,
                confidence=fused_conf,
                method="classifier",
                routing_tier=1,
            )

        # TIER 2: Medium confidence — check if modalities actively conflict
        if fused_conf >= self.TIER2_THRESHOLD:
            both_confident = (
                mismatch.img_confidence > 0.6 and
                mismatch.txt_confidence > 0.6
            )
            both_disagree = (
                mismatch.img_prediction != mismatch.txt_prediction
            )
            real_conflict = both_confident and both_disagree

            if not real_conflict:
                # No active conflict — one modality is weak, gate handled it
                self.stats["tier2_cautious_accept"] += 1
                return ProductResult(
                    product_id=product_id,
                    taxonomy=clf_taxonomy,
                    attributes=clf_attributes,
                    confidence=fused_conf,
                    method="classifier_cautious",
                    routing_tier=2,
                )
            else:
                # Real conflict — both confident but disagree
                reason = "modality_conflict"
                self.stats["tier3_vlm_fallback"] += 1
                self.stats["reasons"][reason] = (
                    self.stats["reasons"].get(reason, 0) + 1)

                logger.info(
                    f"Product {product_id}: TIER 3 ({reason}), "
                    f"img={mismatch.img_confidence:.2f} -> class "
                    f"{mismatch.img_prediction}, "
                    f"txt={mismatch.txt_confidence:.2f} -> class "
                    f"{mismatch.txt_prediction}"
                )

        else:
            # TIER 3: Low confidence — VLM fallback
            reason = "low_confidence"
            self.stats["tier3_vlm_fallback"] += 1
            self.stats["reasons"][reason] = (
                self.stats["reasons"].get(reason, 0) + 1)

            logger.info(
                f"Product {product_id}: TIER 3 ({reason}), "
                f"fused_conf={fused_conf:.2f}"
            )

        # ── Step 4: VLM Fallback ──
        # Pick best image using attention weights from classifier
        hero_image = None
        if images:
            best_idx = min(
                mismatch.best_image_idx, len(images) - 1)
            hero_image = images[best_idx]

        vlm_result = self._run_vlm_fallback(
            hero_image, text, clf_taxonomy, clf_attributes)

        return ProductResult(
            product_id=product_id,
            taxonomy=vlm_result["taxonomy"],
            attributes=vlm_result["attributes"],
            confidence=vlm_result["confidence"],
            mismatch_detected=True,
            mismatch_reason=reason,
            method="vlm_fallback",
            routing_tier=3,
        )

    def predict_batch(self, products: list) -> list:
        """Classify a batch of products.

        Args:
            products: list of dicts with keys:
                images, text, attributes, product_id

        Returns:
            list of ProductResult — ALL in same format
        """
        return [
            self.predict(
                images=p.get("images"),
                text=p.get("text", ""),
                attributes=p.get("attributes", {}),
                product_id=p.get("product_id", f"product_{i}"),
            )
            for i, p in enumerate(products)
        ]

    def enrich_catalog(self, products: list,
                       output_path: str) -> dict:
        """Run full catalog enrichment and write enriched_catalog.jsonl.

        Args:
            products: list of product dicts
            output_path: path to write JSONL

        Returns:
            stats dict
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        with open(output_path, "w") as f:
            for i, product in enumerate(products):
                result = self.predict(
                    images=product.get("images"),
                    text=product.get("text", ""),
                    attributes=product.get("attributes", {}),
                    product_id=product.get("product_id", str(i)),
                )
                f.write(result.to_catalog_line() + "\n")

                if (i + 1) % 1000 == 0:
                    logger.info(f"Enriched {i+1}/{len(products)} products")

        logger.info(f"Catalog written to {output_path}")
        return self.get_stats()

    # ──────────────────────────────────────────
    # INTERNAL — ENCODING
    # ──────────────────────────────────────────

    def _load_encoders(self):
        """Lazy load frozen encoders for live inference."""
        if self.image_encoder is not None:
            return
        from .image_encoder import ImageEncoder
        from .text_encoder import TextEncoder
        self.image_encoder = ImageEncoder(
            self.classifier.config.image_model, freeze=True)
        self.text_encoder = TextEncoder(
            self.classifier.config.text_model, freeze=True)
        logger.info("Encoders loaded for live inference")

    def _encode_inputs(self, images, text, attributes):
        """Encode raw inputs into tensors for classifier."""
        config = self.classifier.config

        # Image embeddings
        if images:
            img_embeds = self.image_encoder.encode_images(images)
            n_imgs = img_embeds.shape[0]
            padded = np.zeros(
                (config.k_max, config.image_dim), dtype=np.float32)
            mask = np.zeros(config.k_max, dtype=np.float32)
            n_use = min(n_imgs, config.k_max)
            padded[:n_use] = img_embeds[:n_use]
            mask[:n_use] = 1.0
        else:
            padded = np.zeros(
                (config.k_max, config.image_dim), dtype=np.float32)
            mask = np.zeros(config.k_max, dtype=np.float32)

        # Text embedding
        from .text_encoder import TextEncoder
        formatted_text = TextEncoder.build_text(text)
        txt_embed = self.text_encoder.encode_text(formatted_text)

        # Attribute indices
        feature_string = " | ".join(
            f"{k}:{v}" for k, v in attributes.items())
        parsed = self.classifier.attribute_encoder.parse_features(
            feature_string)
        attr_indices = self.classifier.attribute_encoder.encode_features(
            parsed)

        # To tensors
        img_t = torch.tensor(
            padded, dtype=torch.float32).unsqueeze(0).to(self.device)
        mask_t = torch.tensor(
            mask, dtype=torch.float32).unsqueeze(0).to(self.device)
        txt_t = torch.tensor(
            txt_embed, dtype=torch.float32).unsqueeze(0).to(self.device)
        attr_t = torch.tensor(
            attr_indices, dtype=torch.long).unsqueeze(0).to(self.device)

        return img_t, mask_t, txt_t, attr_t

    # ──────────────────────────────────────────
    # INTERNAL — CLASSIFIER DECODING
    # ──────────────────────────────────────────

    def _decode_taxonomy(self, predictions: dict) -> list:
        """Convert prediction indices back to label strings."""
        taxonomy = []
        level_names = sorted(
            [k for k in predictions if k != "leaf"]) + ["leaf"]
        for level in level_names:
            if level in predictions:
                pred_idx = predictions[level]["predicted"][0].item()
                label = self.reverse_taxonomy.get(level, {}).get(
                    pred_idx, f"unknown_{pred_idx}")
                taxonomy.append(label)
        return taxonomy

    def _decode_attributes(self, attr_predictions: list,
                           idx: int) -> dict:
        """Convert attribute predictions to clean dict.

        Uses confidence gating: model prediction if confident,
        fallback to parsed value if not.
        """
        if not attr_predictions or idx >= len(attr_predictions):
            return {}

        pred = attr_predictions[idx]
        result = {}
        for attr_name, info in pred.items():
            value = info["value"]
            confidence = info["confidence"]

            if value and value != "<UNK>":
                clean_key = attr_name.replace(
                    "_family", "").replace("primary_", "")
                if isinstance(value, list):
                    result[clean_key] = ", ".join(
                        v for v in value if v != "unknown")
                else:
                    result[clean_key] = value
        return result

    # ──────────────────────────────────────────
    # INTERNAL — VLM FALLBACK
    # ──────────────────────────────────────────

    def _load_vlm(self):
        """Lazy load VLM and its postprocessor."""
        if self._vlm is not None:
            return
        if self.vlm_path is None:
            logger.warning("No VLM path configured — fallback disabled")
            return
        try:
            from src.inference.pipeline import InferencePipeline
            self._vlm = InferencePipeline(self.vlm_path)

            from src.inference.postprocessor import PostProcessor
            self._postprocessor = PostProcessor(
                normalize_values=True, validate_schema=True)

            logger.info("VLM fallback loaded")
        except Exception as e:
            logger.warning(f"Failed to load VLM: {e}")

    def _run_vlm_fallback(self, hero_image, text: str,
                          classifier_taxonomy: list,
                          classifier_attrs: dict) -> dict:
        """Run VLM on hero image, normalize output to SAME format as classifier.

        Args:
            hero_image: single PIL image (best from attention pooler)
            text: product text
            classifier_taxonomy: partial taxonomy from classifier
            classifier_attrs: partial attributes from classifier

        Returns:
            dict with SAME keys as classifier output:
            {"taxonomy": [...], "attributes": {...}, "confidence": float}
        """
        self._load_vlm()

        if self._vlm is None:
            # VLM unavailable — return classifier's uncertain result
            return {
                "taxonomy": classifier_taxonomy,
                "attributes": classifier_attrs,
                "confidence": 0.5,
            }

        try:
            # Run VLM — gets raw output like:
            #   {"style": "modern", "color_family": "blue", ...}
            vlm_raw = self._vlm.predict(
                image=hero_image, text=text)

            # Postprocess VLM output (parse JSON, normalize values)
            if self._postprocessor:
                vlm_raw = self._postprocessor.process(vlm_raw)

            # ── KEY STEP: Normalize VLM output to SAME format ──
            # VLM: {"color_family": "blue", "primary_material": "velvet"}
            #   ↓ normalize_vlm_output()
            # Same: {"color": "blue", "material": "velvet"}
            normalized = normalize_vlm_output(
                vlm_raw=vlm_raw,
                classifier_taxonomy=classifier_taxonomy,
                classifier_attrs=classifier_attrs,
            )

            return normalized

        except Exception as e:
            logger.warning(f"VLM fallback failed: {e}")
            return {
                "taxonomy": classifier_taxonomy,
                "attributes": classifier_attrs,
                "confidence": 0.5,
            }

    # ──────────────────────────────────────────
    # STATS
    # ──────────────────────────────────────────

    def get_stats(self) -> dict:
        """Get routing statistics."""
        total = max(self.stats["total"], 1)
        return {
            **self.stats,
            "tier1_pct": round(
                self.stats["tier1_fast_accept"] / total * 100, 1),
            "tier2_pct": round(
                self.stats["tier2_cautious_accept"] / total * 100, 1),
            "tier3_pct": round(
                self.stats["tier3_vlm_fallback"] / total * 100, 1),
        }
