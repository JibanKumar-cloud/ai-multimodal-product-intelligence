"""Unified Inference Pipeline for Classifier, VLM, and Hybrid modes.

Architecture:
    ProductPipeline
      ├── classifier_predict()  — fast multi-tower (~50ms)
      ├── vlm_predict()         — LLaVA QLoRA (~2-5s)
      └── hybrid_predict()      — classifier + VLM fallback

All outputs go through shared PostProcessors for consistent formatting.

Usage:
    pipeline = ProductPipeline(
        classifier_checkpoint="checkpoints/best_model.pt",
        taxonomy_path="data/processed/taxonomy_tree.json",
        vocab_path="data/processed/attribute_vocab.json",
        queue_path="data/processed/image_queue_with_images.json",
        vlm_adapter_path="outputs/checkpoints/qlora-multimodal/best_model",
    )

    # Single product
    result = pipeline.classifier_predict(name, image=img)
    result = pipeline.vlm_predict(name, desc, category, image=img)
    result = pipeline.hybrid_predict(name, desc, category, image=img, threshold=0.5)

    # Batch
    results = pipeline.batch_predict(products, mode="hybrid")
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import torch
from PIL import Image as PILImage

from src.inference.postprocessor import (
    ClassifierPostProcessor,
    VLMPostProcessor,
    HybridPostProcessor,
    COLOR_FAMILIES, MATERIAL_GROUPS, STYLE_GROUPS,
    SHAPE_GROUPS, ASSEMBLY_GROUPS, ATTR_KEYS,
)


# ════════════════════════════════════════════════════════════════
# VLM PROMPTS (with vocabulary families)
# ════════════════════════════════════════════════════════════════

def _compact_families(families, max_per_group=4):
    lines = []
    for group, keywords in sorted(families.items()):
        kws = ", ".join(keywords[:max_per_group])
        lines.append(f"  {group}: {kws}")
    return "\n".join(lines)


VOCAB_CONTEXT = f"""VALID VALUES (use ONLY these group names):

primary_color / secondary_color:
{_compact_families(COLOR_FAMILIES)}

primary_material / secondary_material:
{_compact_families(MATERIAL_GROUPS)}

style:
{_compact_families(STYLE_GROUPS)}

shape:
{_compact_families(SHAPE_GROUPS)}

assembly:
{_compact_families(ASSEMBLY_GROUPS)}"""

VLM_TEXT_PROMPT = (
    "You are a product catalog specialist. Extract structured attributes "
    "from the following product listing. Return ONLY valid JSON with these "
    "exact keys: primary_color, secondary_color, primary_material, "
    "secondary_material, style, shape, assembly. Use null for unknown.\n\n"
    f"{VOCAB_CONTEXT}\n\n"
    "{{input_text}}\n\nExtracted attributes (JSON):"
)

VLM_IMAGE_PROMPT = (
    "You are a product catalog specialist. You are given a product image "
    "and its text listing. Extract structured attributes by analyzing BOTH "
    "the image and the text. Return ONLY valid JSON with these exact keys: "
    "primary_color, secondary_color, primary_material, secondary_material, "
    "style, shape, assembly. Use null for unknown.\n\n"
    f"{VOCAB_CONTEXT}\n\n"
    "<image>\n{{input_text}}\n\nExtracted attributes (JSON):"
)


# ════════════════════════════════════════════════════════════════
# PIPELINE
# ════════════════════════════════════════════════════════════════

class ProductPipeline:
    """Unified inference for classifier, VLM, and hybrid modes."""

    def __init__(
        self,
        classifier_checkpoint: Optional[str] = None,
        taxonomy_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        queue_path: Optional[str] = None,
        vlm_adapter_path: Optional[str] = None,
        vlm_base_model: str = "llava-hf/llava-1.5-7b-hf",
        device: Optional[str] = None,
    ):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")

        # Paths
        self.classifier_checkpoint = classifier_checkpoint
        self.taxonomy_path = taxonomy_path
        self.vocab_path = vocab_path
        self.vlm_adapter_path = vlm_adapter_path
        self.vlm_base_model = vlm_base_model

        # Lazy-loaded models
        self._classifier = None
        self._classifier_meta = None
        self._vlm_model = None
        self._vlm_tokenizer = None
        self._vlm_processor = None

        # Post-processors
        self.classifier_pp = ClassifierPostProcessor(
            taxonomy_path=taxonomy_path, queue_path=queue_path)
        self.vlm_pp = VLMPostProcessor()
        self.hybrid_pp = HybridPostProcessor(
            self.classifier_pp, self.vlm_pp)

    # ── Model Loading (lazy) ──

    def _load_classifier(self):
        """Load classifier on first use."""
        if self._classifier is not None:
            return

        from scripts.train_classifier import ProductClassifier

        model = ProductClassifier(
            taxonomy_path=self.taxonomy_path,
            vocab_path=self.vocab_path,
        ).to(self.device)

        ckpt = torch.load(
            self.classifier_checkpoint,
            map_location=self.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        self._classifier = model
        self._classifier_meta = {
            "epoch": ckpt.get("epoch", "?"),
            "val_loss": ckpt.get("val_loss", 0),
            "accuracy": ckpt.get("accuracy", {}),
        }

    def _load_vlm(self):
        """Load LLaVA QLoRA on first use."""
        if self._vlm_model is not None:
            return

        from transformers import (
            AutoTokenizer, AutoProcessor,
            LlavaForConditionalGeneration, BitsAndBytesConfig)
        from peft import PeftModel

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        model = LlavaForConditionalGeneration.from_pretrained(
            self.vlm_base_model, quantization_config=bnb_config,
            device_map="auto", torch_dtype=torch.float16)
        model = PeftModel.from_pretrained(model, self.vlm_adapter_path)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(self.vlm_base_model)
        processor = AutoProcessor.from_pretrained(self.vlm_base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self._vlm_model = model
        self._vlm_tokenizer = tokenizer
        self._vlm_processor = processor

    # ── Properties ──

    @property
    def classifier_loaded(self):
        return self._classifier is not None

    @property
    def vlm_loaded(self):
        return self._vlm_model is not None

    @property
    def classifier_meta(self):
        if self._classifier_meta is None:
            self._load_classifier()
        return self._classifier_meta

    @property
    def has_classifier(self):
        return (self.classifier_checkpoint and
                Path(self.classifier_checkpoint).exists())

    @property
    def has_vlm(self):
        return (self.vlm_adapter_path and
                Path(self.vlm_adapter_path).exists())

    # ── Classifier Inference ──

    def classifier_predict(self, name, product_class="", description="",
                           image=None, confidence_threshold=0.5):
        """Run classifier and return post-processed result.

        Args:
            name: Product name
            product_class: Category
            description: Product description
            image: PIL Image or file-like object
            confidence_threshold: below this → flag for VLM

        Returns:
            Standardized result dict with taxonomy, attributes, confidence
        """
        self._load_classifier()
        from src.classifier.dataset import get_image_transforms

        transform = get_image_transforms(train=False)
        t0 = time.time()

        # Build text
        parts = [name]
        if product_class:
            parts.append(product_class)
        if description:
            parts.append(description)
        text = " [SEP] ".join(parts)

        # Build images
        images, mask = [], []
        if image is not None:
            try:
                if not isinstance(image, PILImage.Image):
                    img = PILImage.open(image).convert("RGB")
                else:
                    img = image.convert("RGB")
                images.append(transform(img))
                mask.append(True)
            except Exception:
                pass

        while len(images) < 2:
            images.append(torch.zeros(3, 224, 224))
            mask.append(False)

        batch = {
            "text_input": [text],
            "images": torch.stack(images).unsqueeze(0).to(self.device),
            "image_mask": torch.tensor(mask[:2]).unsqueeze(0).to(self.device),
        }

        with torch.inference_mode():
            out = self._classifier(batch)

        latency_ms = round((time.time() - t0) * 1000, 1)

        # Post-process
        result = self.classifier_pp.process(
            self._classifier, out, confidence_threshold)
        result["latency_ms"] = latency_ms
        result["model_epoch"] = self._classifier_meta.get("epoch", "?")

        return result

    # ── VLM Inference ──

    def vlm_predict(self, name, product_class="", description="",
                    image=None, max_tokens=300):
        """Run LLaVA VLM and return post-processed result.

        Args:
            name: Product name
            product_class: Category
            description: Product description
            image: PIL Image or file-like object
            max_tokens: Max generation tokens

        Returns:
            (standardized_result_dict, raw_output_string)
        """
        self._load_vlm()

        input_text = (f"Product: {name}\nCategory: {product_class}\n"
                      f"Description: {description}")[:300]

        if image is not None:
            if not isinstance(image, PILImage.Image):
                image = PILImage.open(image).convert("RGB")
            else:
                image = image.convert("RGB")

            prompt = VLM_IMAGE_PROMPT.format(input_text=input_text)
            inputs = self._vlm_processor(
                text=prompt, images=image,
                return_tensors="pt").to(self._vlm_model.device)
        else:
            prompt = VLM_TEXT_PROMPT.format(input_text=input_text)
            inputs = self._vlm_tokenizer(
                prompt, return_tensors="pt",
                max_length=512, truncation=True
            ).to(self._vlm_model.device)

        with torch.inference_mode():
            outputs = self._vlm_model.generate(
                **inputs, max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self._vlm_tokenizer.eos_token_id)

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        raw = self._vlm_tokenizer.decode(
            generated, skip_special_tokens=True)

        result = self.vlm_pp.process(raw)
        return result, raw

    # ── Hybrid Inference ──

    def hybrid_predict(self, name, product_class="", description="",
                       image=None, confidence_threshold=0.5,
                       max_tokens=300):
        """Classifier first, VLM fallback for low-confidence attributes.

        Taxonomy always from classifier.
        High-confidence attributes from classifier.
        Low-confidence attributes from VLM.

        Returns:
            (merged_result, classifier_ms, vlm_ms, vlm_attrs_list)
        """
        # Step 1: Classifier (fast)
        cls_result = self.classifier_predict(
            name, product_class, description, image, confidence_threshold)

        # Only attributes go to VLM — taxonomy stays from classifier
        vlm_attrs = self.hybrid_pp.get_vlm_attrs(cls_result)
        cls_ms = cls_result.get("latency_ms", 0)

        if not vlm_attrs:
            # All attributes above threshold
            return cls_result, cls_ms, 0, []

        # Step 2: VLM for low-confidence attributes
        if image is not None:
            # Reset file pointer if file-like
            if hasattr(image, 'seek'):
                image.seek(0)

        t0 = time.time()
        vlm_result, raw = self.vlm_predict(
            name, product_class, description, image, max_tokens)
        vlm_ms = round((time.time() - t0) * 1000, 1)

        # Step 3: Merge
        merged = self.hybrid_pp.merge(cls_result, vlm_result)
        merged["latency_ms"] = cls_ms
        merged["latency_vlm_ms"] = vlm_ms
        merged["model_epoch"] = cls_result.get("model_epoch", "?")

        return merged, cls_ms, vlm_ms, vlm_attrs

    # ── Batch Inference ──

    def batch_predict(self, products, mode="hybrid",
                      confidence_threshold=0.5):
        """Run inference on a list of products.

        Args:
            products: list of dicts with product_name, product_class,
                      product_description, image_path (optional)
            mode: "classifier", "vlm", or "hybrid"
            confidence_threshold: for hybrid mode

        Returns:
            list of result dicts
        """
        results = []
        t0 = time.time()

        for i, p in enumerate(products):
            name = p.get("product_name", "")
            cls = p.get("product_class", "")
            desc = p.get("product_description", "")

            # Load image if available
            image = None
            img_path = p.get("image_path")
            if img_path and Path(img_path).exists():
                try:
                    image = PILImage.open(img_path).convert("RGB")
                except Exception:
                    pass

            if mode == "classifier":
                result = self.classifier_predict(
                    name, cls, desc, image, confidence_threshold)
            elif mode == "vlm":
                result, _ = self.vlm_predict(name, cls, desc, image)
            elif mode == "hybrid":
                result, _, _, _ = self.hybrid_predict(
                    name, cls, desc, image, confidence_threshold)
            else:
                result = {"error": f"Unknown mode: {mode}"}

            result["product_id"] = p.get("product_id")
            results.append(result)

            if (i + 1) % 25 == 0:
                elapsed = time.time() - t0
                print(f"  [{i+1}/{len(products)}] "
                      f"{(i+1)/elapsed:.1f}/sec")

        elapsed = time.time() - t0
        print(f"\nDone: {len(results)} products in {elapsed:.1f}s "
              f"({len(results)/elapsed:.1f}/sec)")

        return results