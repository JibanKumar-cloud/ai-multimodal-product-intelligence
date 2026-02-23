#!/usr/bin/env python3
"""Fast inference using vLLM with QLoRA adapters."""

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


def detect_mode(model_path):
    config_path = Path(model_path) / "training_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        mode = config.get("training_mode", "text")
        logger.info(f"Auto-detected training mode: {mode}")
        return mode
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

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = args.mode if args.mode != "auto" else detect_mode(args.model_path)

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

    n_img = sum(1 for e in examples if e.get("image_path") and Path(e.get("image_path","")).exists())
    logger.info(f"Test examples: {len(examples)} ({n_img} with images), mode={mode}")

    # Init vLLM
    logger.info(f"Loading vLLM: {args.base_model} + adapter {args.model_path}")
    llm = LLM(
        model=args.base_model,
        enable_lora=True,
        max_lora_rank=16,
        dtype="half",
        max_model_len=2048 if mode == "multimodal" else 512,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0, max_tokens=args.max_tokens,
        stop=["}\n\n", "\n\n\n"],
    )
    lora_req = LoRARequest("wayfair", 1, args.model_path)
    postprocessor = PostProcessor()

    # Run inference
    start = time.time()
    if mode == "text":
        prompts = [TEXT_PROMPT.format(input_text=str(e.get("input_text",""))[:400]) for e in examples]
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_req)
        results = []
        for ex, out in zip(examples, outputs):
            raw = out.outputs[0].text
            results.append({
                "product_id": ex.get("product_id"),
                "prediction": postprocessor.process(raw),
                "ground_truth": ex.get("target_attributes"),
                "raw_output": raw,
                "has_image": False,
            })
    else:
        from PIL import Image
        results = []
        for i, ex in enumerate(examples):
            input_text = str(ex.get("input_text",""))[:300]
            try:
                image = Image.open(ex["image_path"]).convert("RGB")
                prompt = IMAGE_PROMPT.format(input_text=input_text)
                out = llm.generate(
                    [{"prompt": prompt, "multi_modal_data": {"image": image}}],
                    sampling_params, lora_request=lora_req,
                )
                raw = out[0].outputs[0].text
            except Exception as e:
                logger.debug(f"Image failed for {ex.get('product_id')}: {e}")
                prompt = TEXT_PROMPT.format(input_text=input_text)
                out = llm.generate([prompt], sampling_params, lora_request=lora_req)
                raw = out[0].outputs[0].text
            results.append({
                "product_id": ex.get("product_id"),
                "prediction": postprocessor.process(raw),
                "ground_truth": ex.get("target_attributes"),
                "raw_output": raw,
                "has_image": True,
            })
            if (i+1) % 50 == 0:
                logger.info(f"Processed {i+1}/{len(examples)}")

    elapsed = time.time() - start

    # Save
    tag = f"vllm-{mode}"
    eval_path = output_dir / f"{tag}_eval_ready.jsonl"
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
    logger.info(f"DONE: {len(results)} examples in {elapsed:.1f}s ({len(results)/elapsed:.1f}/sec)")
    logger.info(f"Saved: {eval_path}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
