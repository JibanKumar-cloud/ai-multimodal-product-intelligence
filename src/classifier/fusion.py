"""Gated Fusion Layer.

Learns dynamic weights for each modality based on input quality.
Handles missing modalities gracefully (e.g., no images → w_img ≈ 0).

z = w_img * e_img + w_txt * e_txt + w_attr * e_attr
where [w_img, w_txt, w_attr] = softmax(gate_network([e_img; e_txt; e_attr]))
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFusion(nn.Module):
    """Gated fusion of image, text, and attribute embeddings."""

    def __init__(self, embed_dim: int = 768, dropout: float = 0.1):
        """
        Args:
            embed_dim: dimension of each modality embedding
            dropout: dropout rate
        """
        super().__init__()
        self.embed_dim = embed_dim

        # Gate network: takes concatenated modalities, outputs 3 weights
        self.gate_network = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 3),  # 3 modality weights
        )

        # Optional: project fused embedding through MLP for richer representation
        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, e_img: torch.Tensor, e_txt: torch.Tensor,
                e_attr: torch.Tensor, has_image: torch.Tensor = None):
        """
        Args:
            e_img:  [B, embed_dim] pooled image embedding
            e_txt:  [B, embed_dim] text embedding
            e_attr: [B, embed_dim] attribute embedding
            has_image: [B] bool tensor (optional, for monitoring)

        Returns:
            z: [B, embed_dim] fused embedding
            gate_weights: [B, 3] modality weights (for analysis)
        """
        # Concatenate all modalities
        concat = torch.cat([e_img, e_txt, e_attr], dim=-1)  # [B, 3*D]

        # Compute gate weights
        gate_logits = self.gate_network(concat)  # [B, 3]
        gate_weights = F.softmax(gate_logits, dim=-1)  # [B, 3]

        w_img = gate_weights[:, 0:1]   # [B, 1]
        w_txt = gate_weights[:, 1:2]   # [B, 1]
        w_attr = gate_weights[:, 2:3]  # [B, 1]

        # Weighted combination
        z = w_img * e_img + w_txt * e_txt + w_attr * e_attr  # [B, D]

        # Project for richer representation
        z = self.output_projection(z)

        return z, gate_weights
