#!/usr/bin/env python3
"""Batch inference using HuggingFace + BitsAndBytes 4-bit (fits T4 16GB).

Usage:
    # Multimodal model
    python scripts/run_inference_hf.py \
        --model-path outputs/checkpoints/qlora-multimodal/best_model \
        --test-data data/processed/test_multimodal.jsonl

    # Text-only model
    python scripts/run_inference_hf.py \
        --model-path outputs/checkpoints/qlora-text-only/best_model \
        --test-data data/processed/test.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from src.utils.logger import setup_logger
from src.inference.postprocessor import PostProcessor


def detect_mode(model_path):
    config_path = Path(model_path) / "training_config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f).get("training_mode", "text")
    return "text"


def main():
    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--base-model", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--test-data", type=str, default="data/processed/test.jsonl")
    parser.add_argument("--output-dir", type=str, default="outputs/predictions")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto","text","multimodal"])
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=300)
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
    from peft import PeftModel
    from PIL import Image

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = args.mode if args.mode != "auto" else detect_mode(args.model_path)
    logger.info(f"Inference mode: {mode}")

    # Load test data
    examples = []
    with open(args.test_data) as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            if mode == "multimodal":
                ip = ex.get("image_path")
                if not ip or not Path(ip).exists():
                    continue
            examples.append(ex)

    if args.sample_size and args.sample_size < len(examples):
        import random
        random.seed(42)
        examples = random.sample(examples, args.sample_size)

    n_img = sum(1 for e in examples if e.get("image_path") and Path(str(e.get("image_path",""))).exists())
    logger.info(f"Test examples: {len(examples)} ({n_img} with images)")

    # Load model with 4-bit quantization (fits T4)
    logger.info(f"Loading {args.base_model} with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = LlavaForConditionalGeneration.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    logger.info(f"Loading LoRA adapter from {args.model_path}...")
    model = PeftModel.from_pretrained(model, args.model_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    processor = AutoProcessor.from_pretrained(args.base_model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    postprocessor = PostProcessor()

    TEXT_PROMPT = (
        "You are a product catalog specialist. Extract structured attributes "
        "from the following product listing. Return ONLY valid JSON.\n\n"
        "{input_text}\n\nExtracted attributes (JSON):"
    )
    IMAGE_PROMPT = (
        "You are a product catalog specialist. You are given a product image "
        "and its text listing. Extract structured attributes by analyzing BOTH "
        "the image and the text. Return ONLY valid JSON.\n\n"
        "<image>\n{input_text}\n\nExtracted attributes (JSON):"
    )

    # Run inference
    results = []
    start = time.time()

    for i, ex in enumerate(examples):
        input_text = str(ex.get("input_text", ""))[:300]
        has_image = False

        try:
            if mode == "multimodal" and ex.get("image_path") and Path(ex["image_path"]).exists():
                image = Image.open(ex["image_path"]).convert("RGB")
                prompt = IMAGE_PROMPT.format(input_text=input_text)
                inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
                has_image = True
            else:
                prompt = TEXT_PROMPT.format(input_text=input_text)
                inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(model.device)

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated = outputs[0][inputs["input_ids"].shape[1]:]
            raw = tokenizer.decode(generated, skip_special_tokens=True)
            prediction = postprocessor.process(raw)

        except Exception as e:
            logger.debug(f"Error on {ex.get('product_id')}: {e}")
            raw = ""
            prediction = {}

        results.append({
            "product_id": ex.get("product_id"),
            "prediction": prediction,
            "ground_truth": ex.get("target_attributes"),
            "raw_output": raw,
            "has_image": has_image,
        })

        if (i + 1) % 25 == 0:
            elapsed = time.time() - start
            logger.info(f"Processed {i+1}/{len(examples)} ({(i+1)/elapsed:.1f}/sec)")

    elapsed = time.time() - start

    # Save results
    tag = f"hf-{mode}"
    eval_path = output_dir / f"{tag}_eval_ready.jsonl"
    pred_path = output_dir / f"{tag}_predictions.jsonl"

    with open(pred_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")

    with open(eval_path, "w") as f:
        for r in results:
            f.write(json.dumps({
                "product_id": r["product_id"],
                "prediction": r["prediction"],
                "ground_truth": r["ground_truth"],
                "model": tag,
                "has_image": r["has_image"],
            }, default=str) + "\n")

    logger.info("=" * 50)
    logger.info(f"INFERENCE COMPLETE")
    logger.info(f"  Mode:       {mode}")
    logger.info(f"  Examples:   {len(results)}")
    logger.info(f"  Time:       {elapsed:.1f}s")
    logger.info(f"  Throughput: {len(results)/elapsed:.2f} products/sec")
    logger.info(f"  Predictions: {pred_path}")
    logger.info(f"  Eval-ready:  {eval_path}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
