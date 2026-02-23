"""PyTorch Dataset supporting two training modes:

Mode A: TEXT-ONLY    — all ~13K examples, text input only
Mode B: MULTIMODAL   — ~2.4K examples, image+text input (uniform batches)

Both produce models that can do multimodal inference (CLIP is pretrained).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from loguru import logger

try:
    from PIL import Image
except ImportError:
    Image = None


class AttributeExtractionDataset(Dataset):
    def __init__(
        self,
        data_path: str | Path,
        tokenizer=None,
        processor=None,
        max_length: int = 512,
        image_size: int = 336,
        mode: str = "text",
        prompt_template: Optional[str] = None,
    ):
        """
        Args:
            mode: "text" for text-only training, "multimodal" for image+text.
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        self.image_size = image_size
        self.mode = mode

        self.text_prompt = (
            "You are a product catalog specialist. Extract structured attributes "
            "from the following product listing. Return ONLY valid JSON.\n\n"
            "{input_text}\n\nExtracted attributes (JSON):"
        )
        self.image_prompt = (
            "You are a product catalog specialist. You are given a product image "
            "and its text listing. Extract structured attributes by analyzing BOTH "
            "the image and the text. Return ONLY valid JSON.\n\n"
            "<image>\n{input_text}\n\nExtracted attributes (JSON):"
        )

        self.examples = self._load_data()
        logger.info(f"Loaded {len(self.examples)} examples (mode={mode})")

    def _load_data(self) -> list[dict]:
        examples = []
        with open(self.data_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                if self.mode == "multimodal":
                    ip = ex.get("image_path")
                    if not ip or not Path(ip).exists():
                        continue
                examples.append(ex)
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def _load_image(self, path: str):
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return None

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        input_text = str(ex.get("input_text", ""))[:300]
        target_json = json.dumps(ex["target_attributes"], indent=None)

        if self.mode == "multimodal":
            return self._multimodal_item(ex, input_text, target_json)
        else:
            return self._text_item(ex, input_text, target_json)

    def _text_item(self, ex, input_text, target_json) -> dict:
        prompt = self.text_prompt.format(input_text=input_text)
        full_text = f"{prompt}\n{target_json}"

        enc = self.tokenizer(
            full_text, max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        prompt_enc = self.tokenizer(
            prompt + "\n", max_length=self.max_length,
            truncation=True, return_tensors="pt",
        )
        labels = enc["input_ids"].clone().squeeze(0)
        labels[:prompt_enc["input_ids"].shape[1]] = -100
        labels[enc["attention_mask"].squeeze(0) == 0] = -100
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels,
        }

    def _multimodal_item(self, ex, input_text, target_json) -> dict:
        prompt = self.image_prompt.format(input_text=input_text)
        full_text = f"{prompt}\n{target_json}"

        image = self._load_image(ex["image_path"])
        if image is None:
            # Fall back to text if image load fails
            return self._text_item(ex, input_text, target_json)

        enc = self.processor(
            text=full_text, images=image,
            return_tensors="pt", truncation=False,
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # Truncate end if too long (preserve image tokens at start)
        if input_ids.shape[0] > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]

        # Pad to max_length
        if input_ids.shape[0] < self.max_length:
            pad_len = self.max_length - input_ids.shape[0]
            pad_id = self.tokenizer.pad_token_id or 0
            input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_id, dtype=input_ids.dtype)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=attention_mask.dtype)])

        # Mask prompt in labels
        prompt_enc = self.processor(
            text=prompt + "\n", images=image,
            return_tensors="pt", truncation=False,
        )
        prompt_len = min(prompt_enc["input_ids"].shape[1], self.max_length)

        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": enc["pixel_values"].squeeze(0),
        }


class MultimodalCollateFunction:
    """Handles both text-only and uniform multimodal batches."""

    def __call__(self, batch: list[dict]) -> dict:
        collated = {}
        for key in ("input_ids", "attention_mask", "labels"):
            if key in batch[0] and isinstance(batch[0][key], torch.Tensor):
                collated[key] = torch.stack([ex[key] for ex in batch])

        if "pixel_values" in batch[0]:
            collated["pixel_values"] = torch.stack([ex["pixel_values"] for ex in batch])

        return collated
