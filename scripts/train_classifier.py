"""Train Multi-Tower Product Classifier.

Trains on precomputed embeddings (fast: ~5 min with cached, ~30 min without).
Only trains: attention pooler + attribute MLP + gated fusion + hierarchy heads
             + mismatch detector heads (~4.2M params)

Usage:
    # Step 1: precompute embeddings (run once)
    python scripts/precompute_embeddings.py --text-only

    # Step 2: train classifier
    python scripts/train_classifier.py

    # With custom settings
    python scripts/train_classifier.py --epochs 30 --lr 1e-3 --batch-size 64
"""
import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifier.config import ClassifierConfig
from src.classifier.model import MultiTowerClassifier
from src.classifier.dataset import (
    ClassifierDataset, build_taxonomy, create_splits, collate_fn
)

import pandas as pd


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_main = 0
    total_mismatch = 0
    n_batches = 0
    correct = {}
    total = {}

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        # Move to device
        img_emb = batch["image_embeddings"].to(device)
        img_mask = batch["image_mask"].to(device)
        txt_emb = batch["text_embeddings"].to(device)
        attr_idx = batch["attr_indices"].to(device)
        labels = {k: v.to(device) for k, v in batch["labels"].items()
                  if not isinstance(v, dict)}
        # Attribute labels nested dict
        if "attributes" in batch["labels"]:
            labels["attributes"] = {
                k: v.to(device) for k, v in batch["labels"]["attributes"].items()
            }

        # Forward
        output = model(img_emb, img_mask, txt_emb, attr_idx, labels=labels)

        loss = output["loss"]

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        total_main += output.get("main_loss", torch.tensor(0)).item()
        total_mismatch += output.get("mismatch_loss", torch.tensor(0)).item()
        n_batches += 1

        # Accuracy tracking (dynamic levels)
        with torch.no_grad():
            for level in output["logits"]:
                if level in labels:
                    logits = output["logits"][level]
                    lbl = labels[level]
                    valid_mask = lbl >= 0
                    if valid_mask.any():
                        pred = logits[valid_mask].argmax(dim=-1)
                        if level not in correct:
                            correct[level] = 0
                            total[level] = 0
                        correct[level] += (pred == lbl[valid_mask]).sum().item()
                        total[level] += valid_mask.sum().item()

        # Update progress bar
        leaf_acc = correct.get("leaf", 0) / max(total.get("leaf", 1), 1) * 100
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "leaf_acc": f"{leaf_acc:.1f}%",
        })

    avg_loss = total_loss / max(n_batches, 1)
    accs = {k: correct[k] / max(total[k], 1) * 100 for k in correct}

    return avg_loss, accs


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    n_batches = 0
    correct = {}
    total = {}
    mismatch_count = 0
    total_products = 0

    for batch in dataloader:
        img_emb = batch["image_embeddings"].to(device)
        img_mask = batch["image_mask"].to(device)
        txt_emb = batch["text_embeddings"].to(device)
        attr_idx = batch["attr_indices"].to(device)
        labels = {k: v.to(device) for k, v in batch["labels"].items()
                  if not isinstance(v, dict)}
        if "attributes" in batch["labels"]:
            labels["attributes"] = {
                k: v.to(device) for k, v in batch["labels"]["attributes"].items()
            }

        output = model(img_emb, img_mask, txt_emb, attr_idx, labels=labels)

        total_loss += output["loss"].item()
        n_batches += 1

        for level in output["logits"]:
            if level in labels:
                logits = output["logits"][level]
                lbl = labels[level]
                valid_mask = lbl >= 0
                if valid_mask.any():
                    pred = logits[valid_mask].argmax(dim=-1)
                    if level not in correct:
                        correct[level] = 0
                        total[level] = 0
                    correct[level] += (pred == lbl[valid_mask]).sum().item()
                    total[level] += valid_mask.sum().item()

        # Check mismatch detection
        result = model.predict(img_emb, img_mask, txt_emb, attr_idx)
        for mr in result["mismatch_results"]:
            total_products += 1
            if mr.mismatch_detected:
                mismatch_count += 1

    avg_loss = total_loss / max(n_batches, 1)
    accs = {k: correct[k] / max(total[k], 1) * 100 for k in correct}
    mismatch_rate = mismatch_count / max(total_products, 1) * 100

    return avg_loss, accs, mismatch_rate


def main():
    parser = argparse.ArgumentParser(description="Train classifier")
    parser.add_argument("--data-path",
                        default="data/raw/WANDS/dataset/product.csv")
    parser.add_argument("--embeddings-dir", default="data/embeddings")
    parser.add_argument("--checkpoint-dir",
                        default="outputs/checkpoints/classifier")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--k-max", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"Device: {device}")

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    logger.info(f"Loading products from {args.data_path}")
    df = pd.read_csv(args.data_path, sep="\t")
    logger.info(f"Loaded {len(df)} products")

    # Build taxonomy
    taxonomy_path = os.path.join(args.embeddings_dir, "taxonomy.json")
    if os.path.exists(taxonomy_path):
        with open(taxonomy_path) as f:
            taxonomy = json.load(f)
        logger.info(f"Loaded taxonomy from {taxonomy_path}")
    else:
        taxonomy = build_taxonomy(df)

    # Create splits
    train_df, val_df, test_df = create_splits(df)

    # Create datasets
    train_ds = ClassifierDataset(
        train_df, taxonomy, args.embeddings_dir,
        k_max=args.k_max, mode="cached", split="train")
    val_ds = ClassifierDataset(
        val_df, taxonomy, args.embeddings_dir,
        k_max=args.k_max, mode="cached", split="val")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True)

    # Create model
    config = ClassifierConfig(
        k_max=args.k_max,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        embeddings_dir=args.embeddings_dir,
        checkpoint_dir=args.checkpoint_dir,
    )
    model = MultiTowerClassifier(config, taxonomy).to(device)

    # Optimizer (only trainable params)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_acc = 0
    best_epoch = 0
    history = []

    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING MULTI-TOWER CLASSIFIER")
    logger.info(f"{'='*60}")
    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    logger.info(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    taxonomy_summary = ", ".join(
        f"{k}={len(v)}" for k, v in taxonomy.items())
    logger.info(f"Taxonomy: {taxonomy_summary}")

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_accs = train_epoch(
            model, train_loader, optimizer, device, epoch)

        # Evaluate
        val_loss, val_accs, mismatch_rate = evaluate(
            model, val_loader, device)

        scheduler.step()

        # Log
        logger.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | "
            f"Train leaf: {train_accs['leaf']:.1f}% | "
            f"Val leaf: {val_accs['leaf']:.1f}% | "
            f"Mismatch: {mismatch_rate:.1f}%"
        )

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accs": train_accs,
            "val_accs": val_accs,
            "mismatch_rate": mismatch_rate,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_result)

        # Save best model
        if val_accs["leaf"] > best_val_acc:
            best_val_acc = val_accs["leaf"]
            best_epoch = epoch
            model.save(args.checkpoint_dir)
            logger.info(f"  ★ New best: {best_val_acc:.1f}% leaf accuracy")

    # Save training history
    history_path = os.path.join(args.checkpoint_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Best val leaf accuracy: {best_val_acc:.1f}%")
    logger.info(f"Model saved: {args.checkpoint_dir}")
    logger.info(f"History: {history_path}")
    logger.info(f"\nNext: python scripts/evaluate_classifier.py")


if __name__ == "__main__":
    main()
