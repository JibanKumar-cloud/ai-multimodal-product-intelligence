# Wayfair Product Intelligence Platform

A multimodal ML system for automated product classification, attribute extraction, and search relevance ranking. Built on Wayfair's 42K+ product catalog with images, structured metadata, and hierarchical taxonomy.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRODUCTION PIPELINE                          │
│                                                                 │
│  INPUT: product image + text ("wooden bed")                     │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────┐                            │
│  │   MULTI-TOWER CLASSIFIER       │  (~10ms, handles 85-90%)   │
│  │   CLIP ViT + DistilBERT        │                            │
│  │   gated prediction heads         │                            │
│  └──────────────┬──────────────────┘                            │
│                 │                                                │
│          confidence check                                       │
│          (per-attribute)                                         │
│           │            │                                        │
│        ≥ 0.5        < 0.5                                       │
│           │            │                                        │
│       USE AS-IS    ┌───▼───────────┐                            │
│           │        │  VLM FALLBACK │  (~2s, handles 10-15%)     │
│           │        │  (LLaVA)      │                            │
│           │        └───┬───────────┘                            │
│           │            │                                        │
│           ▼            ▼                                        │
│  ┌─────────────────────────────────┐                            │
│  │   MERGED PREDICTIONS            │                            │
│  │   taxonomy + product_class +    │                            │
│  │   color, material, style,       │                            │
│  │   shape, assembly               │                            │
│  └──────────────┬──────────────────┘                            │
│                 │                                                │
│                 ▼                                                │
│  ┌─────────────────────────────────┐                            │
│  │   SEARCH INDEX                  │                            │
│  │   WANDS-trained relevance       │                            │
│  │   Attribute-boosted ranking     │                            │
│  └─────────────────────────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

The system has three major components:

1. **Multi-Tower Classifier** — Fast multimodal model for taxonomy classification and attribute prediction
2. **VLM Fallback** — Vision-language model for low-confidence edge cases
3. **Search Relevance** — WANDS-trained ranking with attribute-boosted retrieval

### Multi-Tower Classifier — Detailed Architecture

```
  Product Images (1-N)            Product Text ("walnut bed with storage")
        │                                │
        ▼                                ▼
  CLIP ViT-B/32 (frozen)         DistilBERT (frozen)
        │                                │
        ▼                                │
  Attention Pooler (trainable)           │
  N images → 1 vector                   │
        │                                │
        ▼                                ▼
    e_img [768]                     e_txt [768]
        │                                │
        └───────────┬────────────────────┘
                    │
             concat(e_img, e_txt)
                    │
      ┌─────────────┼──────────────────────┐
      │             │                      │
      ▼             │                      ▼
  TAXONOMY          │                ATTRIBUTE HEADS
  (N levels,        │                (7 heads, each
  dynamic)          │                own GatedHead)
      │             │                      │
      │ softmax     │                      ▼
      │ probs       │                primary_color     → 28 cls
      │    │        │                secondary_color   → 29 cls
      │    ▼        │                primary_material  → 18 cls
      │  concat     │                secondary_material→ 18 cls
      │  project    │                style             → 12 cls
      │    │        │                shape             → 10 cls
      │    ▼        ▼                assembly          →  3 cls
      │  PRODUCT CLASS
      │  (~580 classes)
      │  gate(e_img,e_txt) + tax_emb
      │
      ▼
  L1: Furniture
  L2: Bedroom       product_class: "Platform Beds"
  L3: Beds          (conditioned on taxonomy probs,
  L4: Platform       gradients flow back to tax heads)
  L5+: deeper
    (if exists)

GATE (per head): Linear(1536→128) → ReLU → Linear(128→2) → Softmax
FUSION:          z = w_img · e_img + w_txt · e_txt
CLASSIFY:        MLP(z) → logits

EXPECTED GATE CONVERGENCE:
  primary_color     ████████████████░░░░  w_img=0.85
  shape             ██████████████████░░  w_img=0.90
  primary_material  █████████████░░░░░░░  w_img=0.65
  style             █████████░░░░░░░░░░░  w_img=0.45
  assembly          █░░░░░░░░░░░░░░░░░░░  w_img=0.05
  ████ = image weight    ░░░░ = text weight
```

---

## 1. Multi-Tower Classifier

### 1.1 Architecture Overview

The classifier uses a dual-encoder architecture with per-head learned gating. Both encoders are frozen pretrained models; only the gating networks and classification heads are trained.

```
IMAGE → CLIP ViT-B/32 (frozen) → attention pooler → e_img [768]
TEXT  → DistilBERT (frozen)     → [CLS] token     → e_txt [768]

┌──────────────────────────────────────────────────────────────┐
│  For each prediction head:                                    │
│                                                               │
│  concat(e_img, e_txt) → gate_net → softmax → [w_img, w_txt] │
│                                                               │
│  z = w_img · e_img + w_txt · e_txt                           │
│                                                               │
│  z → Linear(768→192) → ReLU → Dropout(0.1) → Linear(192→C)  │
└──────────────────────────────────────────────────────────────┘
```

All heads use independent 2-way gating. Each head learns its own modality weighting from data — visual attributes like color converge toward image-heavy weights, while text-only attributes like assembly converge toward text-heavy weights.

### 1.2 Prediction Heads

**Taxonomy heads (N levels, dynamic from data)** — Hierarchical product categorization. The number of levels is read from `taxonomy_tree.json` at init — loss weights are defined up to 8 levels:

| Head | Example Output | Num Classes |
|------|---------------|-------------|
| level_1 | "Furniture" | ~15 |
| level_2 | "Bedroom Furniture" | ~80 |
| level_3 | "Beds" | ~250 |
| level_4 | "Platform Beds" | ~500 |
| level_5 | "Wood Platform Beds" | ~700 |

Taxonomy levels use weighted loss: level_1=2.0, level_2=1.5, level_3=1.0, level_4=0.8, level_5=0.5, level_6=0.3, level_7=0.2, level_8=0.1. Higher levels get more weight because errors cascade downward.

**Product class** (~580 classes) is a `TaxonomyConditionedClassHead` — it receives the probability distributions from all taxonomy level heads as additional context before predicting. This means product_class *knows* where in the taxonomy tree the product sits, making contradictions (taxonomy says "Lighting" but class says "Sofas") much less likely. Gradient flows back from product_class loss through the taxonomy probs into the taxonomy heads, creating mutual reinforcement:

```python
# From taxonomy_head.py — product_class is conditioned on taxonomy
# Step 1: taxonomy heads predict → collect softmax probs
tax_probs = [softmax(level_head(e_img, e_txt)) for level in levels]

# Step 2: project concatenated probs into context vector
tax_probs_concat = torch.cat(tax_probs, dim=-1)  # [B, ~1545]
tax_emb = tax_projector(tax_probs_concat)         # [B, 256]

# Step 3: product_class uses gated fusion + taxonomy context
z = gate(e_img, e_txt)                            # [B, 768]
class_input = torch.cat([z, tax_emb], dim=-1)     # [B, 1024]
logits = class_mlp(class_input)                   # [B, 580]
```

**Attribute heads (7)** — Product properties with per-attribute gating:

| Head | Classes | Loss Weight | Expected Gate (w_img / w_txt) |
|------|---------|-------------|-------------------------------|
| primary_color | ~28 | 1.5 | 0.85 / 0.15 |
| secondary_color | ~29 (+multi) | 0.8 | 0.80 / 0.20 |
| primary_material | ~18 | 1.5 | 0.65 / 0.35 |
| secondary_material | ~18 | 0.8 | 0.60 / 0.40 |
| style | ~12 | 1.5 | 0.45 / 0.55 |
| shape | ~10 | 0.5 | 0.90 / 0.10 |
| assembly | 3 (full/partial/none) | 0.3 | 0.05 / 0.95 |

All attribute heads use single-label cross-entropy with label_smoothing=0.1 and masked loss (labels=-1 for missing data).

### 1.3 Encoders

**ImageEncoder** — Frozen CLIP ViT-B/32 with trainable attention pooling:

- Accepts up to N images per product (hero image + alternate views)
- Each image: 224x224x3, processed through frozen CLIP backbone
- Trainable `nn.MultiheadAttention` (8 heads) pools N image features into a single [768] vector via a learned query parameter
- Products with no valid images produce a zero vector; per-sample masking handles mixed batches safely

**TextEncoder** — Frozen DistilBERT (distilbert-base-uncased):

- Input format: `"[product name] [SEP] [product class] [SEP] [description]"`
- Truncated to 128 tokens (DistilBERT max 512, but shorter is faster)
- Uses [CLS] token output -> [768] embedding
- No attribute values in text input — that would be data leakage since mined attributes are the training labels

### 1.4 Gating Mechanism

Each head has an independent 2-way gate network:

```python
class GatedAttributeHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        self.gate = nn.Sequential(
            nn.Linear(input_dim * 2, 128),  # concat(e_img, e_txt)
            nn.ReLU(),
            nn.Linear(128, 2),              # [w_img, w_txt]
        )
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(input_dim // 4, num_classes),
        )

    def forward(self, e_img, e_txt):
        weights = F.softmax(self.gate(torch.cat([e_img, e_txt], -1)), -1)
        z = weights[:, 0:1] * e_img + weights[:, 1:2] * e_txt
        return {"logits": self.classifier(z), "gate_weights": weights}
```

The gate learns per-sample, per-attribute weighting. For a product like "brown leather sofa" with a clear image, the color gate might produce w_img=0.9 for that specific sample. For a product with a vague image but descriptive text, the same gate might produce w_img=0.3.

### 1.5 Data Leakage Prevention

A critical design decision: attribute heads receive only `e_img` and `e_txt`, never `e_attr` (structured attribute embeddings). The mined attribute values are training *labels*, not inputs.

```
WRONG (leakage):
  e_attr has: primary_color=brown (mined from description)
  attribute_head reads e_attr -> "brown" -> predicts brown
  Result: 99% accuracy from epoch 1 (copying, not learning)

CORRECT (our approach):
  Label: color=brown (mined from description, used as ground truth)
  Input: image of brown sofa + "leather sofa [SEP] Sofas"
  Model: learns that THIS visual pattern = brown
```

This was discovered when initial training hit 97-100% accuracy after a single epoch — a clear sign of data leakage through the text input containing explicit attribute tokens.

---

## 2. Training Pipeline

### 2.1 Data Preparation (`prepare_classifier_data.py`)

The data pipeline processes 42,994 raw Wayfair products into clean training data. Three major versions were developed to fix systematic extraction errors.

**V3 improvements over raw data:**

| Issue | Before | After |
|-------|--------|-------|
| Chair material=wood (should be fabric) | 26% | 21% |
| Color name vs label mismatch | 34% | 12% |
| Upholstered product material swaps | 0 | 219 corrected |
| Invalid shape assignments dropped | 0 | 6,556 removed |

**Key extraction logic:**

`parse_features_multi()` — Preserves duplicate keys in product_features as lists instead of dict overwrite (last-write-wins bug caused wrong materials on multi-material products).

`extract_colors_smart()` — Priority chain: product name > features > taxonomy > description. Name-derived colors override feature-parsed colors because product names like "Gray Velvet Sofa" are more reliable than internal feature keys which may reference frame color.

`extract_materials_smart()` — Surface vs frame material logic. Upholstered products (detected via class + keywords like "tufted", "cushion", "upholstered") use visible surface material (fabric, leather, velvet) instead of frame material (wood, metal).

`extract_shape_validated()` — Blacklist approach: shapes are dropped for classes where they're nonsensical (chairs="square" is meaningless) but preserved for classes where they matter (rugs, mirrors, tables, throw pillows, platters).

**Attribute granularity (fine-grained V3):**

Colors expanded from 16 -> ~28 classes to separate visually distinct shades:

| Family | Classes | Examples |
|--------|---------|----------|
| Brown | 5 | dark_brown (espresso), brown (walnut), light_brown (tan), mahogany, cognac |
| Blue | 4 | navy, blue, light_blue, teal |
| Gray | 3 | dark_gray (charcoal), gray, light_gray |
| Red | 2 | red, burgundy |
| Metallics | 2 | silver (chrome), gold_metal (brass) |
| Neutrals | 6 | white, cream, black, beige, natural, clear |

Materials expanded from 11 -> ~18 classes:

| Family | Classes | Visual Distinction |
|--------|---------|-------------------|
| Wood | 4 | light_wood (oak/pine), dark_wood (walnut/mahogany), wood (generic), manufactured_wood (MDF) |
| Fabric | 3 | velvet, linen (cotton/canvas), microfiber (polyester/chenille) |
| Metal | 3 | metal (steel), iron (wrought), brass_metal (copper/bronze) |
| Leather | 2 | leather, faux_leather |

This granularity was chosen as a middle ground: enough samples per class (~200-500) for reliable training, but visually distinct enough that grouping makes sense (unlike 16-class where espresso and tan are both "brown").

**Coverage after V3 extraction (41,414 valid products):**

| Attribute | Coverage | Unique Values |
|-----------|----------|---------------|
| primary_color | 86.5% | ~28 |
| secondary_color | 21.9% | ~29 |
| primary_material | 97.5% | ~18 |
| secondary_material | 50.9% | ~18 |
| style | 74.1% | ~12 |
| shape | ~20% | ~10 |
| assembly | 58.7% | 3 |

### 2.2 Image Sourcing (`source_wayfair_images.py`)

Downloads product images from Wayfair CDN using DuckDuckGo search (Bing APIs retired). Filters to products with at least one hero image, reducing 10,557 queued products to 7,431 with verified images.

Image pipeline:
1. Search DuckDuckGo for `"{product_name}" site:wayfair.com`
2. Extract image URLs from Wayfair CDN (`secure.img1-fg.wfcdn.com`)
3. Download and validate (min 200x200, valid JPEG/PNG)
4. Save to `data/images/wayfair/{product_id}/hero.jpg`

### 2.3 Dataset & Augmentation (`dataset.py`)

**Text augmentation strategy** — Simulates production reality where incoming products have varying text quality:

| Probability | Text Mode | Content |
|-------------|-----------|---------|
| 15% | Full | name + class + description (all text present) |
| 15% | Partial | name + class (description dropped) |
| 30% | Name only | product name only |
| 20% | Minimal | first 2-3 words of name |
| 20% | Empty | empty string (image-only) |

**Smart per-attribute word masking** — Targeted removal of attribute-revealing words to force visual learning:

| Attribute | Mask Rate | Rationale |
|-----------|-----------|-----------|
| Color words | 50% | Force model to learn color from images, not text |
| Shape words | 40% | Shape is primarily visual |
| Style words | 20% | Style needs both modalities |
| Material words | 15% | Some materials need text ("MDF" not visible) |
| Assembly phrases | 0% | Never masked — assembly is pure text, never in images |

Example: Input `"brown leather modern sofa"` might become `"[MASK] [MASK] modern sofa"` (color+material masked) in one training sample, forcing the color and material heads to learn from the image.

**Image handling** — No image dropout during training. At inference, images are always present (users upload product photos). Training focuses on text degradation instead.

### 2.4 Training Configuration

```
Optimizer:     AdamW (lr=1e-4, weight_decay=0.01)
Scheduler:     CosineAnnealingLR (eta_min=1e-6)
Batch size:    32
Epochs:        20
Grad clipping: max_norm=1.0
Val split:     10%

Frozen:        CLIP ViT (~87M params), DistilBERT (~66M params)
Trainable:     Attention pooler, gate networks, classification heads
               (~5M trainable / ~158M total)
```

**Loss function** — Multi-task weighted sum:

```
L_total = sum(taxonomy_weight * CE_level) + sum(attr_weight * CE_attr)

Taxonomy weights: level_1=2.0, level_2=1.5, level_3=1.0, level_4=0.8, level_5=0.5
Attribute weights: color=1.5, material=1.5, style=1.5, sec_color=0.8,
                   sec_material=0.8, shape=0.5, assembly=0.3
```

All losses use label_smoothing=0.1 and masked loss for missing labels (label=-1 excluded from gradient).

### 2.5 NaN Fix

Training with lr=1e-4 produced NaN at batch 50. Root cause: `ImageEncoder` attention pooling produces NaN for samples with zero valid images (all-masked key_padding_mask causes attention to produce undefined values).

Fix: Per-sample masking check in `ImageEncoder.forward`:
```python
all_masked = key_padding_mask.all(dim=-1)  # [B] bool
if all_masked.all():
    return torch.zeros(B, self.output_dim, device=images.device)
# ... normal attention pooling ...
pooled[all_masked] = 0.0  # zero out samples with no images
```

Additionally, training dataset was filtered to only products with verified hero images (10,557 -> 7,431).

---

## 3. VLM Fallback System

### 3.1 Purpose

The VLM handles the 10-15% of products where the classifier's confidence drops below threshold. It provides per-attribute routing — only low-confidence attributes are sent to the VLM, not the entire product.

### 3.2 Architecture

```
FROM CLASSIFIER (per-attribute confidence check):
  primary_color:  brown     conf=0.87  → KEEP
  material:       dark_wood conf=0.84  → KEEP
  style:          ???       conf=0.31  → ROUTE TO VLM
  assembly:       ???       conf=0.28  → ROUTE TO VLM
        │
        ▼
  VLM INPUT: product image + constrained prompt (only style, assembly)
        │
        ▼
  VLM OUTPUT: {"style": "modern", "assembly": "full"}
        │
        ▼
  MERGE:
    primary_color:  brown     (classifier)
    material:       dark_wood (classifier)
    style:          modern    (VLM)
    assembly:       full      (VLM)
```

### 3.3 Setup & Configuration

**Supported VLM backends:**

| Backend | Model | Speed | Best For |
|---------|-------|-------|----------|
| LLaVA (local) | llava-v1.6-34b | ~2s/product | Self-hosted, no API cost |
| OpenAI | gpt-4-vision | ~3s/product | Highest accuracy |
| Anthropic | claude-3-opus | ~3s/product | Strong reasoning |

**Configuration:**

```python
# src/vlm/catalog_extractor.py

VLM_CONFIG = {
    "backend": "llava",              # or "openai", "anthropic"
    "model": "llava-v1.6-34b",
    "confidence_threshold": 0.5,     # below this → route to VLM
    "max_tokens": 256,               # keep responses concise
    "temperature": 0.1,              # low temp for consistent output
    "timeout": 10,                   # seconds per request
    "max_retries": 2,
    "batch_size": 8,                 # concurrent VLM requests
}
```

### 3.4 Prompt Design

The VLM prompt is constrained to the same vocabulary the classifier uses, ensuring consistent outputs across tiers:

```python
def build_vlm_prompt(attributes_needed: list, product_text: str) -> str:
    """Build a structured prompt requesting only specific attributes."""

    # Map each attribute to its valid vocabulary
    VOCAB = {
        "primary_color": ["dark_brown", "brown", "light_brown", "mahogany",
                          "cognac", "navy", "blue", "light_blue", "teal",
                          "dark_gray", "gray", "light_gray", "red",
                          "burgundy", "pink", "coral", ...],
        "style": ["modern", "traditional", "farmhouse", "coastal",
                  "industrial", "mid_century", "bohemian", "glam",
                  "scandinavian", "transitional", "rustic", "contemporary"],
        "assembly": ["full", "partial", "none"],
        # ... all 7 attributes
    }

    attr_instructions = []
    for attr in attributes_needed:
        values = VOCAB[attr]
        attr_instructions.append(
            f'  "{attr}": one of {values}'
        )

    return f"""Analyze this product image. Product text: "{product_text}"

Return ONLY a JSON object with these attributes:
{chr(10).join(attr_instructions)}

Respond with valid JSON only, no explanation."""
```

### 3.5 Routing Logic

```python
def predict_with_fallback(classifier, vlm, image, text,
                          confidence_threshold=0.5):
    """Full prediction pipeline with per-attribute VLM routing."""

    # Step 1: Fast classifier prediction
    results = classifier.predict(e_img, e_txt)

    # Step 2: Identify low-confidence attributes
    vlm_needed = {}
    for attr, pred in results.items():
        if pred["confidence"] < confidence_threshold:
            vlm_needed[attr] = True

    # Step 3: Route only low-confidence attributes to VLM
    if vlm_needed:
        prompt = build_vlm_prompt(
            attributes_needed=list(vlm_needed.keys()),
            product_text=text
        )
        vlm_results = vlm.predict(image, prompt)

        # Merge: classifier for high-confidence, VLM for low
        for attr in vlm_needed:
            if attr in vlm_results:
                results[attr] = {
                    "value": vlm_results[attr],
                    "confidence": vlm_results.get(f"{attr}_conf", 0.85),
                    "source": "vlm",
                }

    # Step 4: Final safety net — parsed features for anything still missing
    for attr in ["assembly", "shape"]:
        if attr not in results or results[attr]["confidence"] < 0.3:
            parsed = parsed_features.get(attr)
            if parsed:
                results[attr] = {
                    "value": parsed,
                    "confidence": 1.0,
                    "source": "parsed_features",
                }

    return results
```

### 3.6 Three-Tier Prediction Strategy

| Tier | Source | Speed | Cost | When Used |
|------|--------|-------|------|-----------|
| 1 | Classifier model | ~10ms | Free (local GPU) | confidence ≥ 0.5 (85-90% of predictions) |
| 2 | VLM fallback | ~2-3s | API cost or GPU | confidence < 0.5, visual attributes |
| 3 | Parsed features | Free | None | assembly/shape with no text signal, safety net |

### 3.7 Running the VLM Pipeline

```bash
# Batch process products with low-confidence predictions
python scripts/run_vlm_fallback.py \
    --classifier-checkpoint checkpoints/best_model.pt \
    --queue data/processed/image_queue_with_images.json \
    --images data/images/wayfair \
    --vlm-backend llava \
    --confidence-threshold 0.5 \
    --output data/processed/enriched_products.json
```

```python
# Programmatic single-product inference
from src.vlm.catalog_extractor import VLMCatalogExtractor

extractor = VLMCatalogExtractor(backend="llava")

result = extractor.extract(
    image_path="product.jpg",
    text="modern accent chair",
    attributes=["style", "assembly", "primary_material"]
)
# → {"style": "modern", "assembly": "none", "primary_material": "velvet"}
```

---

## 4. Search Relevance System

### 4.1 Architecture Overview

```
USER QUERY: "navy velvet sofa under $500"
      │
      ▼
QUERY PARSER → text:"sofa", color:navy, material:velvet, price:<$500
      │
      ▼
CANDIDATE RETRIEVAL (BM25/embedding) ← PRODUCT INDEX (42K products
      │                                  with classifier predictions,
      ▼                                  taxonomy, attributes)
ATTRIBUTE BOOST (+0.3 color match, +0.3 material match, +0.2 taxonomy)
      │
      ▼
NEURAL RERANKER (WANDS-trained, query-product relevance scoring)
      │
      ▼
RANKED RESULTS: 1. Navy Velvet Sofa (0.97)  2. Blue Fabric Couch (0.82) ...
```

### 4.2 Data: WANDS Dataset

The Wayfair Annotated Dataset (WANDS) provides human-labeled query-product relevance judgments:

| Stat | Value |
|------|-------|
| Products | 42,994 with full metadata |
| Queries | 480 unique search queries |
| Annotations | 233,448 query-product pairs |
| Annotators | Professional human raters |

**Relevance distribution (3-point scale):**

| Label | Count | Percentage | Description |
|-------|-------|------------|-------------|
| Exact | ~24K | 10.4% | Product directly satisfies the query |
| Partial | ~77K | 33.0% | Related but not ideal match |
| Irrelevant | ~132K | 56.6% | Poor or no match to query |

The heavy class imbalance (56.6% irrelevant) requires careful handling — a baseline predicting all-irrelevant achieves 56.6% accuracy. The model must learn to distinguish exact from partial matches, which is the harder and more valuable task.

**Data cleaning pipeline:**
1. Remove duplicate query-product pairs
2. Handle conflicting labels (same pair, different annotators) via majority vote
3. Stratified train/val/test split preserving query distribution
4. Balance sampling during training (oversample Exact, undersample Irrelevant)

### 4.3 Attribute-Boosted Search

Classifier predictions feed directly into search ranking. The key insight: structured attribute predictions enable precise filtering that raw text matching cannot achieve.

**Query attribute extraction:**

```python
def parse_query_attributes(query: str) -> dict:
    """Extract structured attributes from search query text."""

    # Color extraction
    color = extract_color_from_text(query)  # reuses classifier's color logic

    # Material extraction
    material = extract_material_from_text(query)

    # Category intent
    category = match_taxonomy_keywords(query)  # "sofa" → Sofas

    # Price parsing
    price = parse_price_constraint(query)  # "under $500" → max=500

    return {
        "text": remove_attributes(query),  # cleaned query text
        "color": color,
        "material": material,
        "category": category,
        "price": price,
    }
```

**Attribute boosting logic:**

```python
def compute_attribute_boost(query_attrs: dict, product: dict) -> float:
    """Score boost based on attribute match between query and product."""
    boost = 0.0

    # Color match (check both primary and secondary)
    if query_attrs["color"]:
        q_color = query_attrs["color"]
        if product["primary_color"] == q_color:
            boost += 0.3
        elif product["secondary_color"] == q_color:
            boost += 0.15  # secondary match worth half

    # Material match
    if query_attrs["material"]:
        q_mat = query_attrs["material"]
        if product["primary_material"] == q_mat:
            boost += 0.3
        elif product["secondary_material"] == q_mat:
            boost += 0.15

    # Taxonomy match
    if query_attrs["category"]:
        if query_attrs["category"] in product["taxonomy_path"]:
            boost += 0.2

    return boost
```

**Multi-attribute queries:** Query "black and white chair" matches products where `primary_color=black OR secondary_color=white`, with full boost when both match.

### 4.4 Neural Reranker

The WANDS-trained reranker scores query-product pairs for relevance:

```python
class SearchRelevanceModel(nn.Module):
    """Score query-product pairs for search relevance."""

    def __init__(self, embed_dim=768):
        super().__init__()
        self.query_encoder = DistilBertModel.from_pretrained(
            "distilbert-base-uncased")
        self.product_encoder = DistilBertModel.from_pretrained(
            "distilbert-base-uncased")

        # Cross-attention between query and product
        self.cross_attn = nn.MultiheadAttention(embed_dim, 8)

        # Relevance scoring head
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim * 3, 256),  # concat + difference + product
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 3),  # exact / partial / irrelevant
        )

    def forward(self, query_text, product_text):
        q_emb = self.query_encoder(query_text).last_hidden_state[:, 0]
        p_emb = self.product_encoder(product_text).last_hidden_state[:, 0]

        # Features: concatenation + element-wise difference + product
        features = torch.cat([
            q_emb, p_emb,
            torch.abs(q_emb - p_emb)
        ], dim=-1)

        return self.scorer(features)
```

**Training the reranker:**

```bash
python scripts/train_search_relevance.py \
    --data data/wands/ \
    --epochs 10 \
    --batch-size 64 \
    --lr 2e-5 \
    --output checkpoints/search_model.pt
```

### 4.5 End-to-End Search Usage

```python
from src.search.relevance import SearchPipeline

# Initialize with classifier and search model
pipeline = SearchPipeline(
    classifier_path="checkpoints/best_model.pt",
    search_model_path="checkpoints/search_model.pt",
    product_index="data/processed/product_index.json",
    taxonomy_path="data/processed/taxonomy_tree.json",
    vocab_path="data/processed/attribute_vocab.json",
)

# Search with automatic attribute extraction and boosting
results = pipeline.search(
    query="navy velvet sofa",
    top_k=20,
    attribute_boost_weight=0.3
)

for rank, result in enumerate(results[:5], 1):
    print(f"{rank}. {result['product_name']}")
    print(f"   Score: {result['score']:.3f}")
    print(f"   Color: {result['primary_color']} | "
          f"Material: {result['primary_material']}")
    print(f"   Match: color={result['color_match']}, "
          f"material={result['material_match']}")
```

**Example output:**

```
1. Modway Engage Navy Velvet Sofa
   Score: 0.974
   Color: navy | Material: velvet
   Match: color=True, material=True

2. Corrigan Studio Navy Blue Velvet Loveseat
   Score: 0.891
   Color: navy | Material: velvet
   Match: color=True, material=True

3. Mercury Row Blue Fabric Sectional
   Score: 0.723
   Color: blue | Material: linen
   Match: color=partial (blue≈navy), material=False
```

### 4.6 DuckDuckGo Integration

After Bing APIs were retired, web-based product image sourcing pivoted to DuckDuckGo:

```python
# Image sourcing for products without catalog images
from scripts.source_wayfair_images import search_product_image

image_urls = search_product_image(
    product_name="Walnut Platform Bed",
    site="wayfair.com",
    max_results=5
)
# Returns CDN URLs: ["https://secure.img1-fg.wfcdn.com/..."]
```

---

## 5. Project Structure

```
project/
├── data/
│   ├── raw/                          # Original WANDS data
│   │   ├── product.csv               # 42,994 products
│   │   ├── query.csv                 # 480 queries
│   │   └── label.csv                 # 233,448 annotations
│   ├── processed/
│   │   ├── classifier_products.tsv   # Cleaned product data
│   │   ├── image_queue.json          # Products with extracted attributes
│   │   ├── image_queue_with_images.json  # Filtered to 7,431 with images
│   │   ├── attribute_vocab.json      # Fine-grained attribute vocabularies
│   │   └── taxonomy_tree.json        # Hierarchical category structure
│   └── images/
│       └── wayfair/{product_id}/     # Downloaded product images
│
├── src/
│   └── classifier/
│       ├── attribute_head.py         # 7 GatedAttributeHead modules
│       ├── attribute_encoder.py      # Attribute vocabulary definitions
│       └── dataset.py                # Smart augmentation, per-attribute masking
│
├── scripts/
│   ├── train_classifier.py           # Full training loop with evaluation
│   ├── prepare_classifier_data.py    # V3 data extraction pipeline
│   └── source_wayfair_images.py      # DuckDuckGo image downloader
│
├── checkpoints/
│   ├── best_model.pt                 # Best validation loss checkpoint
│   └── final_model.pt                # End-of-training checkpoint
│
└── README.md
```

---

## 6. Usage

### Training

```bash
# 1. Prepare training data (extract attributes, build vocab)
python scripts/prepare_classifier_data.py \
    --input data/raw/product.csv \
    --output data/processed/

# 2. Download product images
python scripts/source_wayfair_images.py \
    --queue data/processed/image_queue.json \
    --output data/images/wayfair/

# 3. Train classifier
PYTHONPATH=. python scripts/train_classifier.py \
    --queue data/processed/image_queue_with_images.json \
    --images data/images/wayfair \
    --vocab data/processed/attribute_vocab.json \
    --taxonomy data/processed/taxonomy_tree.json \
    --epochs 20 --batch-size 32 --lr 1e-4
```

### Inference

```python
model = ProductClassifier(
    taxonomy_path="data/processed/taxonomy_tree.json",
    vocab_path="data/processed/attribute_vocab.json"
)
model.load_state_dict(
    torch.load("checkpoints/best_model.pt")["model_state_dict"])

# Predict
outputs = model({
    "text_input": ["wooden bed"],
    "images": img_tensor,
    "image_mask": mask
})

# Per-attribute predictions with confidence
predictions = model.attribute_heads.predict(
    outputs["e_img"], outputs["e_txt"],
    confidence_threshold=0.5
)
# -> {"primary_color": {"value": "brown", "confidence": 0.87,
#     "source": "model", "gate_weights": [0.85, 0.15]}, ...}
```

### Gate Weight Analysis

Every 5 epochs, the training loop prints per-attribute gate weights to verify convergence:

```
Per-attribute gate weights (avg):
  primary_color:     img=0.847 [################    ] txt=0.153 [###                 ]
  assembly:          img=0.052 [#                   ] txt=0.948 [##################  ]
  style:             img=0.431 [########            ] txt=0.569 [###########         ]
```

---

## 7. Key Design Decisions

**Why frozen encoders?** CLIP and DistilBERT are already excellent feature extractors. Fine-tuning 153M params on 7K images would overfit. Instead, we train only ~5M params in gates and heads — fast, stable, and generalizes well.

**Why 2-way gates everywhere?** Initial design used 3-way gates (e_img + e_txt + e_attr) for taxonomy. But e_attr for attributes created data leakage (mined values = training labels fed as input). Switching to uniform 2-way gates simplified the architecture and eliminated leakage.

**Why fine-grained color/material classes?** With 16 colors, "espresso" and "tan" both map to "brown" despite looking completely different. With ~28 colors, the model learns visually meaningful distinctions. The tradeoff is fewer samples per class (~200-500 vs ~1000+), but still sufficient for CE training.

**Why text augmentation instead of image dropout?** At inference, users always upload an image — that's the whole point. But text quality varies wildly: some products come with full descriptions, others just a two-word name. Training with aggressive text degradation (20% empty, 30% name-only) simulates this reality.

**Why per-attribute VLM routing?** Sending every low-confidence product entirely to the VLM wastes compute. If only `style` is uncertain but `color` and `material` are confident, we only ask the VLM about style. This reduces VLM calls by ~60% while maintaining quality.

---

## 8. Dependencies

```
torch >= 2.0
transformers >= 4.30 (CLIP, DistilBERT)
Pillow
pandas
numpy
```

---

## 9. Data Quality Notes

Systematic issues identified and corrected in the Wayfair catalog:

1. **Duplicate feature key overwrite** — `parse_features()` dict overwrites duplicate keys; last value wins. Fixed with `parse_features_multi()` preserving all values as lists.

2. **Frame vs surface material** — Upholstered products reported frame material (wood) instead of visible surface (fabric/leather). Fixed with upholstery detection + material swap logic.

3. **Color source priority** — Original priority (features > taxonomy > text > name) was backwards. Product names like "Gray Velvet Sofa" are more reliable than internal feature tags. Fixed to name > features > taxonomy > description.

4. **Multi-color products** — "dark red/dark brown rug" picked one color randomly. Fixed to extract primary (first mentioned) and secondary (second mentioned) colors.

5. **Invalid shape assignments** — Chairs labeled "square", sofas labeled "round". Fixed with class-based blacklist that drops shapes for categories where they're meaningless.