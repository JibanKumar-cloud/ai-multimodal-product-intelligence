"""Mismatch Detector.

Detects when image and text disagree about product classification.
Uses SEPARATE modality-specific heads (not shared embedding space).

Signals:
  1. Top-1 disagreement: argmax(p_img) != argmax(p_txt)
  2. KL divergence: KL(p_img || p_txt) > threshold
  3. Confidence margin: top1 - top2 < threshold
  4. Entropy: high entropy in fused prediction = uncertainty

When mismatch detected → route to VLM fallback with best image
(selected by attention pooler weights).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class MismatchResult:
    """Result from mismatch detection."""
    mismatch_detected: bool
    confidence: float
    reason: str  # "none", "top1_disagree", "kl_diverge", "low_confidence", "high_entropy"
    kl_divergence: float
    img_prediction: int
    img_confidence: float
    txt_prediction: int
    txt_confidence: float
    fused_confidence: float
    fused_margin: float
    fused_entropy: float
    best_image_idx: int  # from attention pooler, for VLM routing


class MismatchDetector(nn.Module):
    """Detects image-text mismatch using modality-specific classification heads.

    Key insight: we DON'T need contrastive learning or shared embedding space.
    Both heads output probabilities over the SAME label space (taxonomy),
    so we compare p_img(y) vs p_txt(y) directly.
    """

    def __init__(self, embed_dim: int = 768, num_leaf_classes: int = 100,
                 kl_threshold: float = 2.0,
                 confidence_threshold: float = 0.8,
                 margin_threshold: float = 0.1,
                 entropy_threshold: float = 2.0):
        super().__init__()

        # Modality-specific heads (separate from main classifier heads)
        self.image_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, num_leaf_classes),
        )
        self.text_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, num_leaf_classes),
        )

        # Thresholds
        self.kl_threshold = kl_threshold
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold
        self.entropy_threshold = entropy_threshold

    def forward(self, e_img: torch.Tensor, e_txt: torch.Tensor,
                fused_logits: torch.Tensor,
                attention_weights: torch.Tensor,
                has_image: torch.Tensor) -> list:
        """
        Args:
            e_img: [B, D] pooled image embedding
            e_txt: [B, D] text embedding
            fused_logits: [B, C] logits from main fused classifier (leaf level)
            attention_weights: [B, K_MAX] from attention pooler
            has_image: [B] bool tensor

        Returns:
            list of MismatchResult, one per batch item
        """
        B = e_img.shape[0]

        # Get modality-specific predictions
        img_logits = self.image_head(e_img)   # [B, C]
        txt_logits = self.text_head(e_txt)    # [B, C]

        p_img = F.softmax(img_logits, dim=-1)  # [B, C]
        p_txt = F.softmax(txt_logits, dim=-1)  # [B, C]
        p_fused = F.softmax(fused_logits, dim=-1)

        # Fused prediction stats
        fused_conf, fused_pred = p_fused.max(dim=-1)
        fused_top2 = p_fused.topk(min(2, p_fused.size(-1)), dim=-1)
        fused_margin = fused_top2.values[:, 0] - fused_top2.values[:, -1]
        fused_entropy = -(p_fused * (p_fused + 1e-8).log()).sum(dim=-1)

        # Image prediction stats
        img_conf, img_pred = p_img.max(dim=-1)

        # Text prediction stats
        txt_conf, txt_pred = p_txt.max(dim=-1)

        # KL divergence between image and text distributions
        kl_div = F.kl_div(
            (p_img + 1e-8).log(), p_txt, reduction="none"
        ).sum(dim=-1)  # [B]

        # Best image index from attention
        best_img_idx = attention_weights.argmax(dim=-1)  # [B]

        results = []
        for i in range(B):
            reason = "none"
            mismatch = False

            if has_image[i]:
                # Signal 1: Top-1 disagreement
                if img_pred[i] != txt_pred[i]:
                    reason = "top1_disagree"
                    mismatch = True

                # Signal 2: KL divergence too high
                elif kl_div[i] > self.kl_threshold:
                    reason = "kl_diverge"
                    mismatch = True

            # Signal 3: Low fused confidence (applies even without images)
            if fused_conf[i] < self.confidence_threshold:
                reason = "low_confidence"
                mismatch = True

            # Signal 4: Low margin between top 2 (model unsure)
            if fused_margin[i] < self.margin_threshold:
                reason = "low_margin"
                mismatch = True

            # Signal 5: High entropy
            if fused_entropy[i] > self.entropy_threshold:
                reason = "high_entropy"
                mismatch = True

            results.append(MismatchResult(
                mismatch_detected=mismatch,
                confidence=fused_conf[i].item(),
                reason=reason,
                kl_divergence=kl_div[i].item() if has_image[i] else 0.0,
                img_prediction=img_pred[i].item(),
                img_confidence=img_conf[i].item(),
                txt_prediction=txt_pred[i].item(),
                txt_confidence=txt_conf[i].item(),
                fused_confidence=fused_conf[i].item(),
                fused_margin=fused_margin[i].item(),
                fused_entropy=fused_entropy[i].item(),
                best_image_idx=best_img_idx[i].item(),
            ))

        return results

    def compute_loss(self, e_img: torch.Tensor, e_txt: torch.Tensor,
                     labels: torch.Tensor, has_image: torch.Tensor) -> torch.Tensor:
        """Auxiliary loss to train modality-specific heads.

        Both heads learn to predict the correct label independently.
        This is what makes mismatch detection meaningful — each head
        must be good enough on its own.

        Args:
            e_img: [B, D]
            e_txt: [B, D]
            labels: [B] leaf class labels
            has_image: [B] bool

        Returns:
            loss: scalar
        """
        txt_logits = self.text_head(e_txt)
        txt_loss = F.cross_entropy(txt_logits, labels)

        # Image loss only for products with images
        if has_image.any():
            img_logits = self.image_head(e_img[has_image])
            img_loss = F.cross_entropy(img_logits, labels[has_image])
        else:
            img_loss = torch.tensor(0.0, device=e_img.device)

        return 0.5 * txt_loss + 0.5 * img_loss
