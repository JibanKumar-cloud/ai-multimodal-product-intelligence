"""Multi-Tower Product Classifier.

Production classification system with:
- Multi-image attention pooling (CLIP ViT, frozen)
- Text encoding (DistilBERT, frozen)
- Structured attribute encoding (trainable MLP)
- Gated fusion (learns modality weights)
- Hierarchical classification heads
- Mismatch detection + VLM fallback routing
"""
from .config import ClassifierConfig
from .model import MultiTowerClassifier
from .router import ProductIntelligence, ProductResult

__all__ = [
    "ClassifierConfig",
    "MultiTowerClassifier",
    "ProductIntelligence",
    "ProductResult",
]
