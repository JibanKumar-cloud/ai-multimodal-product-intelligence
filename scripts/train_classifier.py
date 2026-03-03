"""Training script for Multi-Tower Product Classifier.

Architecture (all 2-way gated):
  IMAGE -> CLIP ViT (frozen) -> attention pooler -> e_img [768]
  TEXT  -> DistilBERT (frozen) -> e_txt [768]

  TAXONOMY:      per-level gate(e_img, e_txt) -> level_1..5
  PRODUCT_CLASS: gate(e_img, e_txt) -> ~580 classes
  ATTRIBUTES:    per-attr gate(e_img, e_txt)  -> 7 predictions

  Low confidence at inference -> VLM fallback

Usage:
    PYTHONPATH=. python scripts/train_classifier.py \
      --queue data/processed/image_queue.json \
      --images data/images/wayfair \
      --vocab data/processed/attribute_vocab.json \
      --taxonomy data/processed/taxonomy_tree.json \
      --tsv data/processed/classifier_products.tsv \
      --epochs 20 --batch-size 32
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.classifier.dataset import build_dataloaders
from src.classifier.attribute_head import AttributePredictor
from src.classifier.taxonomy_head import TaxonomyPredictor


# ════════════════════════════════════════════════════════════════
# Encoders
# ════════════════════════════════════════════════════════════════

class TextEncoder(nn.Module):
    """Frozen DistilBERT -> CLS embedding."""

    def __init__(self, model_name="distilbert-base-uncased"):
        super().__init__()
        from transformers import DistilBertModel, DistilBertTokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name)
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, texts):
        # Handle empty strings gracefully
        texts = [t if t else "[PAD]" for t in texts]
        tokens = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=256, return_tensors="pt")
        tokens = {k: v.to(self.model.device) for k, v in tokens.items()}
        return self.model(**tokens).last_hidden_state[:, 0, :]


class ImageEncoder(nn.Module):
    """Frozen CLIP ViT + trainable attention pooler."""

    def __init__(self, model_name="openai/clip-vit-base-patch32",
                 output_dim=768):
        super().__init__()
        from transformers import CLIPVisionModel
        self.model = CLIPVisionModel.from_pretrained(model_name)
        for p in self.model.parameters():
            p.requires_grad = False

        clip_dim = self.model.config.hidden_size
        self.output_dim = output_dim
        self.attn_pool = nn.MultiheadAttention(
            embed_dim=clip_dim, num_heads=8, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, clip_dim) * 0.02)
        self.proj = (nn.Linear(clip_dim, output_dim)
                     if clip_dim != output_dim else nn.Identity())

    def forward(self, images, image_mask):
        B, N, C, H, W = images.shape
        device = images.device

        # Per-sample check: which samples have NO images at all
        has_any_image = image_mask.any(dim=-1)  # [B]

        if not has_any_image.any():
            return torch.zeros(B, self.output_dim, device=device)

        with torch.no_grad():
            features = self.model(
                pixel_values=images.view(B * N, C, H, W)).pooler_output
        features = features.view(B, N, -1)

        # For samples with no images: use mean pooling (safe, no attention)
        # For samples with images: use attention pooling
        result = torch.zeros(B, self.output_dim, device=device)

        # Process samples WITH images through attention
        valid_idx = has_any_image.nonzero(as_tuple=True)[0]
        if len(valid_idx) > 0:
            valid_features = features[valid_idx]           # [V, N, D]
            valid_mask = ~image_mask[valid_idx]            # [V, N] (key_padding)
            query = self.query.expand(len(valid_idx), -1, -1)
            pooled, _ = self.attn_pool(
                query, valid_features, valid_features,
                key_padding_mask=valid_mask)
            pooled = pooled.squeeze(1)                     # [V, D]
            result[valid_idx] = self.proj(pooled)

        return result


# ════════════════════════════════════════════════════════════════
# Full Model
# ════════════════════════════════════════════════════════════════

class ProductClassifier(nn.Module):
    """
    e_img + e_txt -> taxonomy + product_class + 7 attributes
    All 2-way gated. Handles missing modalities.
    """

    def __init__(self, taxonomy_path, vocab_path=None, input_dim=768):
        super().__init__()
        self.text_encoder = TextEncoder()
        self.image_encoder = ImageEncoder(output_dim=input_dim)
        self.taxonomy_heads = TaxonomyPredictor(
            input_dim=input_dim, taxonomy_path=taxonomy_path)
        self.attribute_heads = AttributePredictor(
            input_dim=input_dim, vocab_path=vocab_path)

    def forward(self, batch):
        device = next(self.parameters()).device
        e_txt = self.text_encoder(batch["text_input"])
        e_img = self.image_encoder(
            batch["images"].to(device),
            batch["image_mask"].to(device))

        tax_out = self.taxonomy_heads(e_img, e_txt)
        attr_out = self.attribute_heads(e_img, e_txt)

        return {
            "taxonomy": tax_out,
            "attributes": attr_out,
            "e_img": e_img, "e_txt": e_txt,
        }


# ════════════════════════════════════════════════════════════════
# Training
# ════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    sums = {"total": 0, "tax": 0, "attr": 0}
    n = 0

    for i, batch in enumerate(loader):
        optimizer.zero_grad()
        out = model(batch)

        # Taxonomy + product_class loss
        tax_labels = {k: batch[k].to(device)
                      for k in batch if k.startswith("tax_")}
        tax_labels["product_class"] = batch["product_class"].to(device)
        tax_loss, tax_det = model.taxonomy_heads.compute_loss(
            out["taxonomy"], tax_labels)

        # Attribute loss
        attr_labels = {k.replace("attr_", ""): batch[k].to(device)
                       for k in batch if k.startswith("attr_")}
        attr_loss, attr_det = model.attribute_heads.compute_loss(
            out["attributes"], attr_labels)

        loss = tax_loss + attr_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        sums["total"] += loss.item()
        sums["tax"] += tax_loss.item()
        sums["attr"] += attr_loss.item()
        n += 1

        if i % 50 == 0:
            print(f"  [{epoch}][{i}/{len(loader)}] "
                  f"loss={loss.item():.4f} "
                  f"tax={tax_loss.item():.4f} "
                  f"attr={attr_loss.item():.4f}")
            if attr_det:
                print(f"    attr: {' '.join(f'{k}={v:.3f}' for k, v in attr_det.items())}")
            if tax_det:
                print(f"    tax:  {' '.join(f'{k}={v:.3f}' for k, v in tax_det.items())}")

    return {k: v / max(n, 1) for k, v in sums.items()}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    sums = {"total": 0, "tax": 0, "attr": 0}
    n = 0
    correct, total_count = {}, {}

    for batch in loader:
        out = model(batch)

        tax_labels = {k: batch[k].to(device)
                      for k in batch if k.startswith("tax_")}
        tax_labels["product_class"] = batch["product_class"].to(device)
        tax_loss, _ = model.taxonomy_heads.compute_loss(
            out["taxonomy"], tax_labels)

        attr_labels = {k.replace("attr_", ""): batch[k].to(device)
                       for k in batch if k.startswith("attr_")}
        attr_loss, _ = model.attribute_heads.compute_loss(
            out["attributes"], attr_labels)

        sums["total"] += (tax_loss + attr_loss).item()
        sums["tax"] += tax_loss.item()
        sums["attr"] += attr_loss.item()
        n += 1

        # Attribute accuracy
        for name, head_out in out["attributes"].items():
            if name not in attr_labels:
                continue
            lbl = attr_labels[name]
            valid = lbl >= 0
            if not valid.any():
                continue
            preds = head_out["logits"][valid].argmax(-1)
            correct[name] = correct.get(name, 0) + (preds == lbl[valid]).sum().item()
            total_count[name] = total_count.get(name, 0) + valid.sum().item()

        # Taxonomy accuracy
        for name, head_out in out["taxonomy"].items():
            lk = f"tax_{name}" if name != "product_class" else "product_class"
            if lk not in tax_labels and lk not in batch:
                continue
            lbl = tax_labels.get(lk, batch.get(lk, torch.tensor([])).to(device))
            if lbl.numel() == 0:
                continue
            valid = lbl >= 0
            if not valid.any():
                continue
            preds = head_out["logits"][valid].argmax(-1)
            key = f"tax_{name}" if name != "product_class" else name
            correct[key] = correct.get(key, 0) + (preds == lbl[valid]).sum().item()
            total_count[key] = total_count.get(key, 0) + valid.sum().item()

    acc = {k: correct[k] / total_count[k]
           for k in correct if total_count.get(k, 0) > 0}

    return {
        "losses": {k: v / max(n, 1) for k, v in sums.items()},
        "accuracy": acc,
    }


def print_gates(model, loader, device):
    model.eval()
    batch = next(iter(loader))
    out = model(batch)

    print(f"\n  Attribute gates (w_img / w_txt):")
    ag = model.attribute_heads.get_gate_summary(out["e_img"], out["e_txt"])
    for name, w in ag.items():
        bi = "#" * int(w["w_img"] * 20)
        bt = "#" * int(w["w_txt"] * 20)
        print(f"    {name:25s} img={w['w_img']:.3f} [{bi:20s}] "
              f"txt={w['w_txt']:.3f} [{bt:20s}]")

    print(f"\n  Taxonomy gates (w_img / w_txt):")
    tg = model.taxonomy_heads.get_gate_summary(out["e_img"], out["e_txt"])
    for name, w in tg.items():
        bi = "#" * int(w["w_img"] * 20)
        bt = "#" * int(w["w_txt"] * 20)
        print(f"    {name:25s} img={w['w_img']:.3f} [{bi:20s}] "
              f"txt={w['w_txt']:.3f} [{bt:20s}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--taxonomy", required=True)
    parser.add_argument("--tsv", default=None)
    parser.add_argument("--output", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader = build_dataloaders(
        queue_path=args.queue,
        image_dir=args.images,
        vocab_path=args.vocab,
        taxonomy_path=args.taxonomy,
        tsv_path=args.tsv,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
    )

    model = ProductClassifier(
        taxonomy_path=args.taxonomy,
        vocab_path=args.vocab,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total")

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tm = train_one_epoch(model, train_loader, optimizer, device, epoch)
        vm = evaluate(model, val_loader, device)
        scheduler.step()

        vl = vm["losses"]
        print(f"\nEpoch {epoch}/{args.epochs} "
              f"({time.time()-t0:.1f}s, lr={optimizer.param_groups[0]['lr']:.6f})")
        print(f"  Train: loss={tm['total']:.4f} "
              f"(tax={tm['tax']:.4f} attr={tm['attr']:.4f})")
        print(f"  Val:   loss={vl['total']:.4f} "
              f"(tax={vl['tax']:.4f} attr={vl['attr']:.4f})")

        # Print accuracies grouped
        acc = vm["accuracy"]
        tax_acc = {k: v for k, v in acc.items()
                   if k.startswith("tax_") or k == "product_class"}
        attr_acc = {k: v for k, v in acc.items()
                    if k not in tax_acc}

        if tax_acc:
            print(f"  Taxonomy accuracy:")
            for k, v in sorted(tax_acc.items()):
                print(f"    {k:25s}: {v:.3f}")
        if attr_acc:
            print(f"  Attribute accuracy:")
            for k, v in sorted(attr_acc.items()):
                print(f"    {k:25s}: {v:.3f}")

        if epoch % 5 == 0:
            print_gates(model, val_loader, device)

        if vl["total"] < best_val:
            best_val = vl["total"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": vl["total"],
                "accuracy": vm["accuracy"],
            }, os.path.join(args.output, "best_model.pt"))
            print(f"  -> Saved best (val_loss={best_val:.4f})")

    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
    }, os.path.join(args.output, "final_model.pt"))
    print(f"\nDone. Best val_loss: {best_val:.4f}")


if __name__ == "__main__":
    main()