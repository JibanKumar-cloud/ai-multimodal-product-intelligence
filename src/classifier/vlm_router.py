"""VLM Router for Low-Confidence Predictions.

Production pipeline:
  1. Classifier predicts taxonomy + attributes  (~10ms)
  2. Check confidence per prediction
  3. High confidence -> use as-is
  4. Low confidence -> route to VLM for those specific fields  (~2s)
  5. Merge results

Handles per-attribute routing (not all-or-nothing):
  color=0.92 (high)    -> keep classifier result
  material=0.87 (high) -> keep classifier result
  style=0.31 (low)     -> route to VLM
  assembly=0.28 (low)  -> route to VLM
  level_3=0.22 (low)   -> route taxonomy to VLM

Usage:
    router = VLMRouter(
        classifier_model=model,
        vlm_endpoint="http://localhost:8000/predict",
        confidence_threshold=0.5,
    )
    results = router.predict(image, text)
"""
import json
import time
from typing import Optional
from dataclasses import dataclass, field

import torch
from PIL import Image


@dataclass
class PredictionResult:
    """Complete prediction for one product."""
    product_id: str = ""

    # Taxonomy
    taxonomy: list = field(default_factory=list)
    taxonomy_confidence: list = field(default_factory=list)
    product_class: str = ""
    product_class_confidence: str = ""  # "exact", "partial", "fallback"

    # Attributes
    attributes: dict = field(default_factory=dict)
    # {attr_name: {"value": str, "confidence": float,
    #              "source": "classifier"|"vlm", "gate_weights": [w_img, w_txt]}}

    # Routing info
    vlm_routed_fields: list = field(default_factory=list)
    classifier_time_ms: float = 0.0
    vlm_time_ms: float = 0.0


class VLMRouter:
    """Routes low-confidence predictions to VLM.

    Classifier handles 85-90% of products.
    VLM handles the remaining 10-15% (per-attribute, not per-product).
    """

    def __init__(
        self,
        classifier_model,
        taxonomy_lookup,
        vlm_client=None,
        confidence_threshold: float = 0.5,
        vlm_batch_size: int = 8,
        device: str = "cuda",
    ):
        """
        Args:
            classifier_model: trained ProductClassifier
            taxonomy_lookup: TaxonomyLookup instance
            vlm_client: VLM inference client (None = skip VLM)
            confidence_threshold: below this -> route to VLM
            vlm_batch_size: max products per VLM batch
        """
        self.model = classifier_model
        self.model.eval()
        self.lookup = taxonomy_lookup
        self.vlm = vlm_client
        self.threshold = confidence_threshold
        self.vlm_batch_size = vlm_batch_size
        self.device = device

    @torch.no_grad()
    def predict_single(self, image_paths: list, text: str,
                       product_id: str = "") -> PredictionResult:
        """Predict for a single product.

        Args:
            image_paths: list of image file paths
            text: product text (name + optional description)
            product_id: optional identifier

        Returns:
            PredictionResult with all fields populated
        """
        result = PredictionResult(product_id=product_id)

        # ── Step 1: Classifier prediction ──
        t0 = time.time()

        # Prepare batch of 1
        from src.classifier.dataset import get_image_transforms
        transform = get_image_transforms(train=False)

        images = []
        image_mask = []
        max_images = 2

        for path in image_paths[:max_images]:
            try:
                img = Image.open(path).convert("RGB")
                images.append(transform(img))
                image_mask.append(True)
            except Exception:
                pass

        # Pad
        while len(images) < max_images:
            images.append(torch.zeros(3, 224, 224))
            image_mask.append(False)

        batch = {
            "text_input": [text],
            "images": torch.stack(images).unsqueeze(0).to(self.device),
            "image_mask": torch.tensor(
                image_mask[:max_images]).unsqueeze(0).to(self.device),
        }

        outputs = self.model(batch)
        result.classifier_time_ms = (time.time() - t0) * 1000

        # ── Step 2: Extract taxonomy predictions ──
        taxonomy_path = []
        taxonomy_conf = []
        needs_vlm_taxonomy = False

        for level_key in sorted(self.model.taxonomy_heads.level_keys):
            head_out = outputs["taxonomy"][level_key]
            probs = torch.softmax(head_out["logits"], dim=-1)
            conf, pred = probs.max(dim=-1)
            conf_val = conf[0].item()
            pred_idx = pred[0].item()

            value = self.model.taxonomy_heads.idx_to_value[level_key].get(
                pred_idx, "<UNK>")

            if conf_val >= self.threshold and value != "<UNK>":
                taxonomy_path.append(value)
                taxonomy_conf.append(conf_val)
            else:
                needs_vlm_taxonomy = True
                break  # stop at first low-confidence level

        result.taxonomy = taxonomy_path
        result.taxonomy_confidence = taxonomy_conf

        # ── Step 3: Extract attribute predictions ──
        vlm_needed = []

        for attr_name, head_out in outputs["attributes"].items():
            probs = torch.softmax(head_out["logits"], dim=-1)
            conf, pred = probs.max(dim=-1)
            conf_val = conf[0].item()
            pred_idx = pred[0].item()

            i2v = self.model.attribute_heads.idx_to_value[attr_name]
            value = i2v.get(pred_idx, "<UNK>")
            gate_w = head_out["gate_weights"][0].tolist()

            if conf_val >= self.threshold and value != "<UNK>":
                result.attributes[attr_name] = {
                    "value": value,
                    "confidence": conf_val,
                    "source": "classifier",
                    "gate_weights": gate_w,
                }
            else:
                result.attributes[attr_name] = {
                    "value": None,
                    "confidence": conf_val,
                    "source": "pending_vlm",
                    "gate_weights": gate_w,
                }
                vlm_needed.append(attr_name)

        if needs_vlm_taxonomy:
            vlm_needed.append("taxonomy")

        # ── Step 4: Route to VLM if needed ──
        if vlm_needed and self.vlm is not None:
            t1 = time.time()
            vlm_results = self._call_vlm(
                image_paths, text, vlm_needed)
            result.vlm_time_ms = (time.time() - t1) * 1000

            # Merge VLM results
            for attr_name in vlm_needed:
                if attr_name == "taxonomy":
                    vlm_tax = vlm_results.get("taxonomy", [])
                    if vlm_tax:
                        result.taxonomy = vlm_tax
                        result.taxonomy_confidence = [0.8] * len(vlm_tax)
                else:
                    vlm_val = vlm_results.get(attr_name)
                    if vlm_val:
                        result.attributes[attr_name] = {
                            "value": vlm_val,
                            "confidence": 0.8,  # VLM default confidence
                            "source": "vlm",
                            "gate_weights": result.attributes[attr_name].get(
                                "gate_weights", [0.5, 0.5]),
                        }

            result.vlm_routed_fields = vlm_needed

        # ── Step 5: Resolve product_class from taxonomy ──
        if result.taxonomy:
            predicted_attrs = {
                k: v["value"] for k, v in result.attributes.items()
                if v.get("value")
            }
            class_result = self.lookup.resolve(
                result.taxonomy, predicted_attrs)
            result.product_class = class_result["product_class"] or ""
            result.product_class_confidence = class_result["confidence"]

        return result

    def _call_vlm(self, image_paths: list, text: str,
                  fields_needed: list) -> dict:
        """Call VLM for specific fields.

        Override this for your VLM backend (vLLM, API, etc).
        """
        if self.vlm is None:
            return {}

        # Build targeted prompt
        prompt_parts = [
            "Analyze this product image and provide:"
        ]
        for field in fields_needed:
            if field == "taxonomy":
                prompt_parts.append(
                    "- Product category hierarchy "
                    "(e.g., Furniture > Bedroom > Beds > Platform Beds)")
            elif field == "primary_color":
                prompt_parts.append("- Primary/dominant color")
            elif field == "secondary_color":
                prompt_parts.append("- Secondary/accent color")
            elif field == "primary_material":
                prompt_parts.append("- Primary material (what it's mostly made of)")
            elif field == "secondary_material":
                prompt_parts.append("- Secondary material")
            elif field == "style":
                prompt_parts.append(
                    "- Design style (modern, traditional, farmhouse, etc.)")
            elif field == "shape":
                prompt_parts.append("- Shape (rectangular, round, etc.)")
            elif field == "assembly":
                prompt_parts.append(
                    "- Assembly requirement (none, partial, full)")

        if text:
            prompt_parts.append(f"\nProduct text: {text}")

        prompt_parts.append(
            "\nRespond in JSON format with the requested fields only.")
        prompt = "\n".join(prompt_parts)

        try:
            vlm_response = self.vlm.predict(
                image_path=image_paths[0] if image_paths else None,
                prompt=prompt,
            )
            # Parse VLM response (assumes JSON)
            if isinstance(vlm_response, str):
                vlm_response = json.loads(vlm_response)
            return vlm_response
        except Exception as e:
            print(f"VLM call failed: {e}")
            return {}

    @torch.no_grad()
    def predict_batch(self, products: list) -> list:
        """Predict for a batch of products.

        Args:
            products: list of {"image_paths": [...], "text": "...",
                               "product_id": "..."}

        Returns:
            list of PredictionResult
        """
        results = []
        for product in products:
            result = self.predict_single(
                image_paths=product.get("image_paths", []),
                text=product.get("text", ""),
                product_id=product.get("product_id", ""),
            )
            results.append(result)
        return results

    def get_routing_stats(self, results: list) -> dict:
        """Analyze routing decisions across a batch.

        Returns:
            {"total": N, "classifier_only": M,
             "vlm_routed": K, "per_field_vlm_rate": {...}}
        """
        total = len(results)
        vlm_routed = sum(1 for r in results if r.vlm_routed_fields)
        classifier_only = total - vlm_routed

        field_vlm_count = {}
        for r in results:
            for field in r.vlm_routed_fields:
                field_vlm_count[field] = field_vlm_count.get(field, 0) + 1

        return {
            "total": total,
            "classifier_only": classifier_only,
            "classifier_only_pct": classifier_only / max(total, 1) * 100,
            "vlm_routed": vlm_routed,
            "vlm_routed_pct": vlm_routed / max(total, 1) * 100,
            "per_field_vlm_rate": {
                k: v / total * 100 for k, v in field_vlm_count.items()},
            "avg_classifier_ms": sum(
                r.classifier_time_ms for r in results) / max(total, 1),
            "avg_vlm_ms": sum(
                r.vlm_time_ms for r in results
                if r.vlm_time_ms > 0) / max(vlm_routed, 1),
        }


class DummyVLM:
    """Placeholder VLM client for testing.

    Replace with your actual VLM inference client:
    - vLLM endpoint
    - LLaVA API
    - Anthropic API with vision
    """

    def predict(self, image_path=None, prompt=""):
        return {
            "primary_color": "unknown",
            "style": "unknown",
            "taxonomy": ["Unknown"],
        }