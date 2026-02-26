"""Evaluate Multi-Tower Classifier.

Reports:
  - Per-level accuracy (level1, level2, level3, leaf)
  - Confusion matrices for each level
  - Mismatch detection precision/recall
  - Gate weight analysis (image vs text vs attribute contribution)
  - Routing statistics (% going to VLM fallback)
  - Per-class accuracy for long tail analysis

Usage:
    python scripts/evaluate_classifier.py
    python scripts/evaluate_classifier.py --checkpoint outputs/checkpoints/classifier
"""
import os
import sys
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifier.model import MultiTowerClassifier
from src.classifier.dataset import (
    ClassifierDataset, build_taxonomy, create_splits, collate_fn
)


@torch.no_grad()
def full_evaluation(model, dataloader, device, taxonomy):
    """Run full evaluation with detailed metrics."""
    model.eval()

    # Reverse taxonomy for decoding
    reverse = {}
    for level, mapping in taxonomy.items():
        reverse[level] = {v: k for k, v in mapping.items()}

    # Collectors
    all_preds = {level: [] for level in ["level1", "level2", "level3", "leaf"]}
    all_labels = {level: [] for level in ["level1", "level2", "level3", "leaf"]}
    all_confs = []
    all_gate_weights = []
    mismatch_results = []
    has_images = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        img_emb = batch["image_embeddings"].to(device)
        img_mask = batch["image_mask"].to(device)
        txt_emb = batch["text_embeddings"].to(device)
        attr_idx = batch["attr_indices"].to(device)
        labels = {k: v.to(device) for k, v in batch["labels"].items()}

        result = model.predict(img_emb, img_mask, txt_emb, attr_idx)

        for level in ["level1", "level2", "level3", "leaf"]:
            if level in result["predictions"] and level in labels:
                preds = result["predictions"][level]["predicted"].cpu().numpy()
                lbls = labels[level].cpu().numpy()
                all_preds[level].extend(preds)
                all_labels[level].extend(lbls)

                if level == "leaf":
                    confs = result["predictions"][level]["confidence"].cpu().numpy()
                    all_confs.extend(confs)

        # Gate weights
        gw = result["gate_weights"].cpu().numpy()
        all_gate_weights.extend(gw)

        # Has image
        hi = (img_mask.sum(dim=1) > 0).cpu().numpy()
        has_images.extend(hi)

        # Mismatch
        mismatch_results.extend(result["mismatch_results"])

    return {
        "preds": all_preds,
        "labels": all_labels,
        "confs": all_confs,
        "gate_weights": np.array(all_gate_weights),
        "has_images": np.array(has_images),
        "mismatch_results": mismatch_results,
        "reverse_taxonomy": reverse,
    }


def compute_metrics(eval_data):
    """Compute all metrics from evaluation data."""
    metrics = {}

    # ── Per-level accuracy ──
    logger.info(f"\n{'='*60}")
    logger.info("PER-LEVEL ACCURACY")
    logger.info(f"{'='*60}")

    for level in ["level1", "level2", "level3", "leaf"]:
        preds = np.array(eval_data["preds"][level])
        labels = np.array(eval_data["labels"][level])
        valid = labels >= 0

        if not valid.any():
            continue

        acc = (preds[valid] == labels[valid]).mean() * 100
        metrics[f"{level}_accuracy"] = acc
        logger.info(f"  {level}: {acc:.1f}% ({valid.sum()} samples)")

    # ── Confidence analysis ──
    confs = np.array(eval_data["confs"])
    logger.info(f"\n{'='*60}")
    logger.info("CONFIDENCE ANALYSIS")
    logger.info(f"{'='*60}")
    logger.info(f"  Mean: {confs.mean():.3f}")
    logger.info(f"  Median: {np.median(confs):.3f}")
    logger.info(f"  <0.5: {(confs < 0.5).sum()} ({(confs < 0.5).mean()*100:.1f}%)")
    logger.info(f"  <0.8: {(confs < 0.8).sum()} ({(confs < 0.8).mean()*100:.1f}%)")
    logger.info(f"  >0.9: {(confs > 0.9).sum()} ({(confs > 0.9).mean()*100:.1f}%)")

    # ── Gate weight analysis ──
    gw = eval_data["gate_weights"]
    has_img = eval_data["has_images"]
    logger.info(f"\n{'='*60}")
    logger.info("GATE WEIGHT ANALYSIS (modality importance)")
    logger.info(f"{'='*60}")

    labels_gate = ["image", "text", "attributes"]
    logger.info("  Overall average:")
    for i, name in enumerate(labels_gate):
        logger.info(f"    {name}: {gw[:, i].mean():.3f}")

    if has_img.any():
        logger.info("  Products WITH images:")
        for i, name in enumerate(labels_gate):
            logger.info(f"    {name}: {gw[has_img][:, i].mean():.3f}")

    if (~has_img).any():
        logger.info("  Products WITHOUT images:")
        for i, name in enumerate(labels_gate):
            logger.info(f"    {name}: {gw[~has_img][:, i].mean():.3f}")

    metrics["gate_weights_avg"] = {
        name: float(gw[:, i].mean()) for i, name in enumerate(labels_gate)
    }

    # ── Mismatch detection ──
    mr = eval_data["mismatch_results"]
    n_mismatch = sum(1 for m in mr if m.mismatch_detected)
    reasons = Counter(m.reason for m in mr if m.mismatch_detected)

    logger.info(f"\n{'='*60}")
    logger.info("MISMATCH DETECTION & ROUTING")
    logger.info(f"{'='*60}")
    logger.info(f"  Total products: {len(mr)}")
    logger.info(f"  Classifier accepted: {len(mr) - n_mismatch} "
                f"({(1 - n_mismatch/max(len(mr),1))*100:.1f}%)")
    logger.info(f"  VLM fallback needed: {n_mismatch} "
                f"({n_mismatch/max(len(mr),1)*100:.1f}%)")
    logger.info(f"  Mismatch reasons:")
    for reason, count in reasons.most_common():
        logger.info(f"    {reason}: {count}")

    metrics["routing"] = {
        "total": len(mr),
        "classifier_accepted": len(mr) - n_mismatch,
        "vlm_fallback": n_mismatch,
        "vlm_pct": n_mismatch / max(len(mr), 1) * 100,
        "reasons": dict(reasons),
    }

    # ── Per-class accuracy (leaf) ──
    reverse = eval_data["reverse_taxonomy"]
    preds = np.array(eval_data["preds"]["leaf"])
    labels = np.array(eval_data["labels"]["leaf"])
    valid = labels >= 0

    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for p, l in zip(preds[valid], labels[valid]):
        class_name = reverse.get("leaf", {}).get(int(l), f"class_{l}")
        class_total[class_name] += 1
        if p == l:
            class_correct[class_name] += 1

    logger.info(f"\n{'='*60}")
    logger.info("PER-CLASS ACCURACY (leaf, top 20 + bottom 10)")
    logger.info(f"{'='*60}")

    class_accs = {
        k: class_correct[k] / class_total[k] * 100
        for k in class_total
    }
    sorted_accs = sorted(class_accs.items(), key=lambda x: -x[1])

    logger.info("  Top 20:")
    for name, acc in sorted_accs[:20]:
        n = class_total[name]
        logger.info(f"    {name:<40} {acc:>5.1f}% (n={n})")

    logger.info("  Bottom 10:")
    for name, acc in sorted_accs[-10:]:
        n = class_total[name]
        logger.info(f"    {name:<40} {acc:>5.1f}% (n={n})")

    metrics["per_class_accuracy"] = class_accs

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate classifier")
    parser.add_argument("--data-path",
                        default="data/raw/WANDS/dataset/product.csv")
    parser.add_argument("--embeddings-dir", default="data/embeddings")
    parser.add_argument("--checkpoint",
                        default="outputs/checkpoints/classifier")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="outputs/evaluation/classifier_eval.json")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Load model
    model = MultiTowerClassifier.load(args.checkpoint, device=device)
    taxonomy = model.taxonomy

    # Load data and create test split
    df = pd.read_csv(args.data_path, sep="\t")
    _, _, test_df = create_splits(df)

    test_ds = ClassifierDataset(
        test_df, taxonomy, args.embeddings_dir,
        k_max=model.config.k_max, mode="cached", split="test")
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2)

    # Run evaluation
    eval_data = full_evaluation(model, test_loader, device, taxonomy)

    # Compute metrics
    metrics = compute_metrics(eval_data)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"\nMetrics saved to {args.output}")


if __name__ == "__main__":
    main()
