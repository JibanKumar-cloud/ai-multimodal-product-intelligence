"""Configuration for Multi-Tower Product Classifier."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ClassifierConfig:
    """Full configuration for the multi-tower classifier."""

    # ── Image Encoder ──
    image_model: str = "openai/clip-vit-base-patch32"
    image_dim: int = 768
    freeze_image_encoder: bool = True

    # ── Text Encoder ──
    text_model: str = "distilbert-base-uncased"
    text_dim: int = 768
    max_text_length: int = 128
    freeze_text_encoder: bool = True

    # ── Attribute Encoder ──
    attr_input_dim: int = 64  # number of attribute features after encoding
    attr_hidden_dim: int = 256
    attr_output_dim: int = 768

    # ── Multi-Image Attention Pooler ──
    k_max: int = 5  # max images per product
    pooler_heads: int = 4  # multi-head attention

    # ── Gated Fusion ──
    fusion_dim: int = 768
    fusion_dropout: float = 0.1

    # ── Hierarchical Classification ──
    num_level1: int = 0  # set from data
    num_level2: int = 0
    num_level3: int = 0
    num_leaf: int = 0
    hierarchy_weights: list = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4]
    )  # loss weight per level (leaf gets most)
    label_smoothing: float = 0.1
    focal_loss_gamma: float = 2.0
    use_focal_loss: bool = True

    # ── Mismatch Detection ──
    mismatch_kl_threshold: float = 2.0
    mismatch_confidence_threshold: float = 0.8
    mismatch_margin_threshold: float = 0.1

    # ── Training ──
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 25
    warmup_steps: int = 100
    use_precomputed: bool = True  # use cached embeddings

    # ── Paths ──
    image_dir: str = "data/images"
    image_manifest: str = "data/images/manifest.json"
    embeddings_dir: str = "data/embeddings"
    checkpoint_dir: str = "outputs/checkpoints/classifier"

    # ── VLM Fallback ──
    vlm_model_path: Optional[str] = "outputs/checkpoints/qlora-vague-multimodal/best_model"
    vlm_confidence_threshold: float = 0.8
