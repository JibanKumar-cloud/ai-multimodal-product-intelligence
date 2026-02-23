"""LLaVA-based attribute extractor with QLoRA fine-tuning support.

Two modes:
1. Pretrained (no fine-tuning) — baseline VLM performance
2. Fine-tuned with QLoRA — production model adapted to Wayfair taxonomy

Handles text-only and image+text inference seamlessly.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import torch
from loguru import logger

from src.models.attribute_extractor import BaseAttributeExtractor


class LLaVAExtractor(BaseAttributeExtractor):
    """Extract product attributes using LLaVA VLM with optional QLoRA adapters."""

    EXPECTED_KEYS = [
        "style", "primary_material", "secondary_material",
        "color_family", "room_type", "product_type", "assembly_required",
    ]

    def __init__(
        self,
        model_name: str = "llava-hf/llava-1.5-7b-hf",
        adapter_path: Optional[str | Path] = None,
        quantization: str = "4bit",
        device: str = "auto",
        max_new_tokens: int = 300,
        temperature: float = 0.0,
    ):
        super().__init__(model_name=model_name)
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self.quantization = quantization
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.model = None
        self.tokenizer = None
        self.processor = None

    def _system_prompt(self, has_image: bool) -> str:
        if has_image:
            return (
                "You are a product catalog specialist. You are given a product image "
                "and its text listing. Extract structured attributes by analyzing BOTH "
                "the image and the text. Return ONLY valid JSON with fields: "
                "style, primary_material, secondary_material, color_family, room_type, "
                "product_type, assembly_required."
            )
        return (
            "You are a product catalog specialist. Extract structured attributes "
            "from the product listing below. Return ONLY valid JSON with fields: "
            "style, primary_material, secondary_material, color_family, room_type, "
            "product_type, assembly_required."
        )

    def load(self) -> None:
        from transformers import (
            AutoTokenizer, LlavaForConditionalGeneration,
            AutoProcessor, BitsAndBytesConfig,
        )

        logger.info(f"Loading {self.model_name} (quantization={self.quantization})")

        bnb_config = None
        if self.quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.quantization == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map=self.device,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        if self.adapter_path and self.adapter_path.exists():
            from peft import PeftModel
            logger.info(f"Loading QLoRA adapters from {self.adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, str(self.adapter_path))
            logger.info("QLoRA adapters loaded")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model.eval()
        self._is_loaded = True

        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Parameters: {total:,} total, {trainable:,} trainable")

    @torch.inference_mode()
    def extract(
        self,
        product_name: str,
        product_description: Optional[str] = None,
        product_class: Optional[str] = None,
        image_path: Optional[str] = None,
    ) -> dict:
        if not self._is_loaded:
            self.load()

        has_image = image_path is not None
        prompt = self._build_prompt(product_name, product_description, product_class, has_image)
        start = time.time()

        try:
            if has_image:
                from PIL import Image
                image = Image.open(image_path).convert("RGB")
                inputs = self.processor(
                    text=prompt, images=image, return_tensors="pt",
                ).to(self.model.device)
            else:
                inputs = self.tokenizer(
                    prompt, return_tensors="pt", max_length=512, truncation=True,
                ).to(self.model.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            generated = outputs[0][inputs["input_ids"].shape[1]:]
            response = self.tokenizer.decode(generated, skip_special_tokens=True)

            logger.debug(f"Inference: {(time.time()-start)*1000:.0f}ms, image={has_image}")
            return self._parse_response(response)

        except Exception as e:
            logger.error(f"LLaVA extraction failed: {e}")
            return self._empty_result()

    def _build_prompt(self, name, desc, cls, has_image):
        parts = []
        if has_image:
            parts.append("<image>")
        parts.append(self._system_prompt(has_image))
        parts.append("")
        parts.append(f"Product: {name}")
        if cls:
            parts.append(f"Category: {cls}")
        if desc:
            parts.append(f"Description: {str(desc)[:500]}")
        parts.append("")
        parts.append("Extracted attributes (JSON):")
        return "\n".join(parts)

    def _parse_response(self, text: str) -> dict:
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                parsed = json.loads(text[start:end])
                return {k: parsed.get(k) for k in self.EXPECTED_KEYS}
            except json.JSONDecodeError:
                pass
        try:
            return {k: json.loads(text).get(k) for k in self.EXPECTED_KEYS}
        except (json.JSONDecodeError, AttributeError):
            logger.warning(f"Parse failed: {text[:200]}")
            return self._empty_result()

    def _empty_result(self) -> dict:
        return {k: None for k in self.EXPECTED_KEYS}

    def extract_batch(self, products: list[dict], batch_size: int = 16) -> list[dict]:
        results = []
        for i, p in enumerate(products):
            results.append(self.extract(
                product_name=p.get("product_name", ""),
                product_description=p.get("product_description"),
                product_class=p.get("product_class"),
                image_path=p.get("image_path"),
            ))
            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i+1}/{len(products)}")
        return results
