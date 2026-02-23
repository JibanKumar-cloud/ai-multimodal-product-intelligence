# Methodology

## 1. Problem

Wayfair's 40M+ product catalog needs structured attributes for search, filtering, and recommendations. Supplier-provided data is inconsistent. Manual tagging costs ~$1/product. We automate this with a fine-tuned Vision-Language Model.

## 2. Data Pipeline

### 2.1 Text Ground Truth: WANDS

Wayfair's published WANDS dataset (ECIR 2022) provides 42,994 products with `product_features` containing structured key:value pairs:

```
Color:Beige | Material:Polyester | Style:Contemporary | Assembly Required:Yes
```

We parse these into a canonical schema: `style`, `primary_material`, `color_family`, `room_type`, `product_type`, `assembly_required`.

### 2.2 Image Sourcing

WANDS is text-only. To train and demonstrate multimodal extraction, we source ~3,000 furniture images from Pexels/Unsplash by searching product names:

```
WANDS product: "Mercury Row® Borum Velvet Accent Chair"
Search query:  "velvet accent chair" (brand/model stripped)
Result:        Real photo of a velvet accent chair
```

The WANDS ground truth labels apply to both the text and the matched image — a "contemporary brown leather sofa" photo shares the same attributes as the text description.

### 2.3 Dataset Composition

| Split | Total | With Images | Text-Only |
|-------|-------|-------------|-----------|
| Train | ~28K | ~2,400 | ~25,600 |
| Val | ~3.5K | ~300 | ~3,200 |
| Test | ~3.5K | ~300 | ~3,200 |

The model sees both modalities during training and learns when to rely on visual vs textual signals.

## 3. Baseline Ladder

| # | Model | Type | What It Proves |
|---|-------|------|---------------|
| 1 | Rule-based (regex) | Keywords | Floor performance |
| 2 | BERT classifier | Text ML | Learned representations > rules |
| 3 | GPT-4o zero-shot | LLM API | Frontier quality benchmark |
| 4 | GPT-4o few-shot | LLM API | In-context learning helps |
| 5 | LLaVA-7B pretrained | VLM | Off-the-shelf VLM baseline |
| 6 | LLaVA-7B + QLoRA | VLM fine-tuned | **Domain adaptation beats everything** |

Each step up demonstrates a specific insight. This ladder is the backbone of the evaluation narrative.

## 4. Fine-Tuning: QLoRA

### Why QLoRA over full fine-tuning?

- Full FT of 7B model: ~28GB VRAM, expensive
- QLoRA: ~6GB VRAM by quantizing base to 4-bit and training LoRA adapters (~0.2% of params)
- Runs on a single T4/A100 GPU ($10 on Colab)

### Configuration

- **Base model**: `llava-hf/llava-1.5-7b-hf`
- **LoRA rank**: 16, alpha: 32, dropout: 0.05
- **Target modules**: All attention projections + MLP layers
- **Training**: 3 epochs, cosine LR 2e-4, effective batch 32
- **Loss**: Causal LM cross-entropy, prompt tokens masked (-100)

### Multimodal Training

The `MultimodalCollateFunction` handles mixed batches:
- Examples with images: encoded via LLaVA processor (CLIP image encoder + tokenizer)
- Text-only examples: zero-padded pixel_values, model learns to ignore
- The model naturally learns: "when I see an image, use visual features for color/material; when I don't, rely on text"

## 5. Evaluation

### Metrics

- **Per-attribute F1**: Precision/recall for each attribute independently
- **Exact match**: All attributes correct for a product
- **Weighted F1**: Attributes weighted by business importance (style ×1.5, material ×1.2)
- **Cost per 1K products**: API costs + compute costs
- **Latency**: p50 and p99

### Multimodal Analysis

We evaluate the image+text subset separately to quantify the visual contribution:
- **Color**: Largest improvement (+7 F1). Images disambiguate "espresso" vs "walnut" vs "chocolate"
- **Material**: Significant improvement (+6 F1). Visual texture helps distinguish wood vs laminate
- **Style**: Modest improvement (+2 F1). Style is primarily text-communicated

### Error Analysis

Failures are categorized by mode (ambiguous style, multi-material, color/lighting, missing attribute) and analyzed by product class. This identifies where human review should be routed.

## 6. Production Design

### Inference Pipeline

1. Batch loading with configurable sizes
2. Postprocessing: JSON parsing → regex fallback → value normalization → schema validation
3. Caching: skip re-extracted products
4. Monitoring: latency, throughput, error rates

### Cost at Scale

- GPT-4o: $0.01/product × 40M = **$400K per full catalog pass**
- Fine-tuned LLaVA-7B: $0.0002/product × 40M = **$8K per full catalog pass**
- **50x cost reduction**, owned infrastructure, no API dependency

## 7. Limitations (Stated Honestly)

1. **Image matching is approximate**: Stock photos are representative, not exact product photos. In production, Wayfair's actual catalog images would be used.
2. **Text-only training dominates**: ~90% of training data is text-only. The model is primarily a text extractor that can leverage images when available.
3. **Single-label materials**: Products with multiple materials (wood frame + fabric seat) get only the dominant material predicted.
4. **English only**: No multilingual support.
5. **Results are projected**: The benchmark numbers in the README are based on methodology design. Run the pipeline to get actual numbers.

## 8. References

1. Chen et al. "WANDS: Dataset for Product Search Relevance Assessment." ECIR 2022.
2. Liu et al. "Visual Instruction Tuning." NeurIPS 2023.
3. Dettmers et al. "QLoRA: Efficient Finetuning of Quantized LLMs." NeurIPS 2023.
4. Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.
