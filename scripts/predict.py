"""Inference Pipeline with VLM Fallback.

Production flow:
  1. Classifier predicts everything  (~10ms)
  2. Check confidence per head
  3. High confidence -> use as-is
  4. Low confidence -> route those fields to VLM  (~2s)
  5. Return complete prediction

Usage:
    # Single product
    PYTHONPATH=. python scripts/predict.py \
      --checkpoint checkpoints/best_model.pt \
      --taxonomy data/processed/taxonomy_tree.json \
      --vocab data/processed/attribute_vocab.json \
      --image data/images/wayfair/16868/hero.jpg \
      --text "wooden bed"

    # Batch from queue
    PYTHONPATH=. python scripts/predict.py \
      --checkpoint checkpoints/best_model.pt \
      --taxonomy data/processed/taxonomy_tree.json \
      --vocab data/processed/attribute_vocab.json \
      --queue data/processed/image_queue.json \
      --images data/images/wayfair \
      --limit 20
"""
import argparse
import json
import os
import time

import torch
from PIL import Image

from src.classifier.dataset import get_image_transforms


def load_model(checkpoint_path, taxonomy_path, vocab_path, device):
    from scripts.train_classifier import ProductClassifier
    model = ProductClassifier(
        taxonomy_path=taxonomy_path,
        vocab_path=vocab_path,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device,
                      weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded epoch {ckpt.get('epoch', '?')}, "
          f"val_loss={ckpt.get('val_loss', 0):.4f}")
    return model


@torch.no_grad()
def predict_single(model, image_paths, text, device,
                   confidence_threshold=0.5):
    """Predict for one product. Returns dict with all predictions."""
    transform = get_image_transforms(train=False)
    t0 = time.time()

    # Load images
    images, mask = [], []
    for path in image_paths[:2]:
        try:
            img = transform(Image.open(path).convert("RGB"))
            images.append(img)
            mask.append(True)
        except Exception:
            pass

    while len(images) < 2:
        images.append(torch.zeros(3, 224, 224))
        mask.append(False)

    batch = {
        "text_input": [text or ""],
        "images": torch.stack(images).unsqueeze(0).to(device),
        "image_mask": torch.tensor(mask[:2]).unsqueeze(0).to(device),
    }

    out = model(batch)
    classifier_ms = (time.time() - t0) * 1000

    result = {
        "classifier_ms": classifier_ms,
        "taxonomy": {},
        "product_class": None,
        "attributes": {},
        "vlm_needed": [],
    }

    # Taxonomy
    for lk in model.taxonomy_heads.level_keys:
        head_out = out["taxonomy"][lk]
        probs = torch.softmax(head_out["logits"], dim=-1)
        conf, pred = probs.max(-1)
        value = model.taxonomy_heads.level_i2v[lk].get(
            pred[0].item(), "<UNK>")
        c = conf[0].item()
        gw = head_out["gate_weights"][0].tolist()
        result["taxonomy"][lk] = {
            "value": value, "confidence": c, "gate": gw}
        if c < confidence_threshold:
            result["vlm_needed"].append(lk)

    # Product class
    if model.taxonomy_heads.has_class_head and "product_class" in out["taxonomy"]:
        head_out = out["taxonomy"]["product_class"]
        probs = torch.softmax(head_out["logits"], dim=-1)
        conf, pred = probs.max(-1)
        value = model.taxonomy_heads.class_i2v.get(
            pred[0].item(), "<UNK>")
        c = conf[0].item()
        result["product_class"] = {"value": value, "confidence": c}
        if c < confidence_threshold:
            result["vlm_needed"].append("product_class")

    # Attributes
    for attr, head_out in out["attributes"].items():
        probs = torch.softmax(head_out["logits"], dim=-1)
        conf, pred = probs.max(-1)
        value = model.attribute_heads.idx_to_value[attr].get(
            pred[0].item(), "<UNK>")
        c = conf[0].item()
        gw = head_out["gate_weights"][0].tolist()
        result["attributes"][attr] = {
            "value": value, "confidence": c, "gate": gw}
        if c < confidence_threshold:
            result["vlm_needed"].append(attr)

    return result


def print_result(result, text="", image=""):
    print(f"\n{'=' * 70}")
    print(f"INPUT: text='{text}' image={image}")
    print(f"{'=' * 70}")

    print(f"\n  TAXONOMY:")
    for lk, info in sorted(result["taxonomy"].items()):
        flag = " [VLM]" if lk in result["vlm_needed"] else ""
        print(f"    {lk:12s}: {info['value']:30s} "
              f"conf={info['confidence']:.3f} "
              f"(img={info['gate'][0]:.2f} txt={info['gate'][1]:.2f})"
              f"{flag}")

    if result["product_class"]:
        pc = result["product_class"]
        flag = " [VLM]" if "product_class" in result["vlm_needed"] else ""
        print(f"    {'class':12s}: {pc['value']:30s} "
              f"conf={pc['confidence']:.3f}{flag}")

    print(f"\n  ATTRIBUTES:")
    for attr, info in sorted(result["attributes"].items()):
        flag = " [VLM]" if attr in result["vlm_needed"] else ""
        print(f"    {attr:25s}: {info['value']:15s} "
              f"conf={info['confidence']:.3f} "
              f"(img={info['gate'][0]:.2f} txt={info['gate'][1]:.2f})"
              f"{flag}")

    n_vlm = len(result["vlm_needed"])
    total_heads = (len(result["taxonomy"]) +
                   (1 if result["product_class"] else 0) +
                   len(result["attributes"]))
    print(f"\n  Classifier: {result['classifier_ms']:.1f}ms | "
          f"VLM needed: {n_vlm}/{total_heads} fields")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--taxonomy", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--image", default=None)
    parser.add_argument("--text", default=None)

    parser.add_argument("--queue", default=None)
    parser.add_argument("--images", default=None)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, args.taxonomy, args.vocab, device)

    if args.image or args.text:
        # Single prediction
        image_paths = [args.image] if args.image else []
        result = predict_single(
            model, image_paths, args.text, device, args.threshold)
        print_result(result, text=args.text or "", image=args.image or "")

    elif args.queue:
        # Batch
        with open(args.queue) as f:
            products = json.load(f)[:args.limit]

        vlm_counts = {}
        for p in products:
            pid = p["product_id"]
            img_dir = os.path.join(args.images, pid)
            img_paths = []
            for ext in ("jpg", "png", "webp"):
                hero = os.path.join(img_dir, f"hero.{ext}")
                if os.path.exists(hero):
                    img_paths.append(hero)
                    break

            result = predict_single(
                model, img_paths, p["product_name"], device, args.threshold)

            # Summary line
            tax = " > ".join(
                info["value"] for info in result["taxonomy"].values()
                if info["value"] != "<UNK>")[:50]
            pc = result["product_class"]["value"] if result["product_class"] else "?"
            n_vlm = len(result["vlm_needed"])
            print(f"  {pid}: {tax} | {pc} | vlm={n_vlm}")

            for field in result["vlm_needed"]:
                vlm_counts[field] = vlm_counts.get(field, 0) + 1

        # Stats
        n = len(products)
        print(f"\n{'=' * 70}")
        print(f"VLM routing stats ({n} products):")
        for field, count in sorted(vlm_counts.items(), key=lambda x: -x[1]):
            print(f"  {field:25s}: {count}/{n} ({count/n*100:.1f}%)")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()