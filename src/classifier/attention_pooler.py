"""Multi-Image Attention Pooler.

Handles variable number of images (0 to K_MAX) per product.
Learns which images are most informative for classification.

Cases:
  - Multiple images: attention-weighted combination
  - Single image: returns that embedding (weight=1.0)
  - Zero images: returns learned "no image" embedding
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooler(nn.Module):
    """Attention-based pooling over variable-length image sets."""

    def __init__(self, embed_dim: int = 768, num_heads: int = 4):
        """
        Args:
            embed_dim: dimension of input image embeddings
            num_heads: number of attention heads
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Multi-head attention scoring
        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim // num_heads),
                nn.Tanh(),
                nn.Linear(embed_dim // num_heads, 1)
            )
            for _ in range(num_heads)
        ])

        # Project multi-head output back to embed_dim
        self.head_projection = nn.Linear(embed_dim * num_heads, embed_dim)

        # Learned embedding for "no image available" case
        self.no_image_embedding = nn.Parameter(torch.randn(embed_dim) * 0.02)

        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, image_embeddings: torch.Tensor,
                image_mask: torch.Tensor) -> tuple:
        """
        Args:
            image_embeddings: [B, K_MAX, embed_dim] per-image embeddings
            image_mask: [B, K_MAX] binary mask (1=real image, 0=padded)

        Returns:
            pooled: [B, embed_dim] pooled image embedding
            attention_weights: [B, K_MAX] attention weights (for VLM routing)
        """
        B, K, D = image_embeddings.shape

        # Check which products have ANY images
        has_image = image_mask.sum(dim=1) > 0  # [B]

        # Compute attention from each head
        head_outputs = []
        all_weights = []

        for head in self.attention_heads:
            # Score each image
            scores = head(image_embeddings).squeeze(-1)  # [B, K]

            # Mask padded positions
            scores = scores.masked_fill(image_mask == 0, float("-inf"))

            # Softmax over valid images
            weights = F.softmax(scores, dim=1)  # [B, K]

            # Handle all-masked (no images): softmax of all -inf gives nan
            weights = weights.nan_to_num(0.0)

            # Weighted sum
            weighted = (weights.unsqueeze(-1) * image_embeddings).sum(dim=1)
            head_outputs.append(weighted)
            all_weights.append(weights)

        # Concatenate heads and project
        multi_head = torch.cat(head_outputs, dim=-1)  # [B, embed_dim * heads]
        pooled = self.head_projection(multi_head)  # [B, embed_dim]

        # Average attention weights across heads for routing
        attention_weights = torch.stack(all_weights, dim=0).mean(dim=0)  # [B, K]

        # Replace pooled embedding with no_image_embedding for products with 0 images
        no_img_expanded = self.no_image_embedding.unsqueeze(0).expand(B, -1)
        pooled = torch.where(has_image.unsqueeze(-1), pooled, no_img_expanded)

        # Normalize
        pooled = self.layer_norm(pooled)

        return pooled, attention_weights
