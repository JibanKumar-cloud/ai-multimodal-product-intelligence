#!/usr/bin/env python3
"""Wayfair Product Intelligence — Demo App.

Three extraction modes:
  1. VLM Only (LLaVA)        — Full VLM extraction (~2s)
  2. Classifier Only          — Multi-tower model (~50ms)
  3. Classifier + VLM Fallback — Fast classifier, LLaVA for low confidence

Tab 2: Product Search (bi-encoder -> cross-encoder -> attribute boost)

Usage:
    streamlit run demo/app.py --server.port 8501 --server.address 0.0.0.0
"""
from __future__ import annotations
import json, sys, time, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import streamlit as st
except ImportError:
    print("pip install streamlit")
    sys.exit(1)

import torch
from PIL import Image

# ================================================================
# CONFIG
# ================================================================

LLAVA_ADAPTERS = {
    "LLaVA Multimodal": "outputs/checkpoints/qlora-multimodal/best_model",
    "LLaVA Text-Only": "outputs/checkpoints/qlora-text-only/best_model",
    "LLaVA Vague+Image": "outputs/checkpoints/qlora-vague-multimodal/best_model",
}

CLASSIFIER_PATHS = {
    "checkpoint": "checkpoints/best_model.pt",
    "taxonomy": "data/processed/taxonomy_tree.json",
    "vocab": "data/processed/attribute_vocab.json",
}

ATTR_ORDER = [
    "primary_color", "secondary_color",
    "primary_material", "secondary_material",
    "style", "shape", "assembly",
]

CONFIDENCE_COLORS = {"high": "#22c55e", "medium": "#f59e0b", "low": "#ef4444"}

# ================================================================
# TAXONOMY HIERARCHY VALIDATION
# ================================================================

@st.cache_resource
def build_taxonomy_hierarchy(queue_path="data/processed/image_queue_with_images.json"):
    """Build parent→valid_children map from training data.

    Returns:
        valid_children: {("level_1", "Furniture"): {"Living Room Furniture", ...}}
        max_depth: {product_class: typical_depth}  e.g. "Sectionals": 3
    """
    if not Path(queue_path).exists():
        return {}, {}

    with open(queue_path) as f:
        queue = json.load(f)

    valid_children = {}   # (parent_level, parent_value) → set of child values
    class_depths = {}     # product_class → max observed depth

    for p in queue:
        taxonomy = p.get("taxonomy", [])
        pc = p.get("product_class", "")

        # Track depths per product class
        if pc and len(taxonomy) > 0:
            class_depths[pc] = max(class_depths.get(pc, 0), len(taxonomy))

        # Build parent→child edges
        for i in range(len(taxonomy) - 1):
            parent_key = (f"level_{i+1}", taxonomy[i])
            child_val = taxonomy[i + 1]
            if parent_key not in valid_children:
                valid_children[parent_key] = set()
            valid_children[parent_key].add(child_val)

    # Also map product_class → valid parent level values
    pc_parents = {}
    for p in queue:
        taxonomy = p.get("taxonomy", [])
        pc = p.get("product_class", "")
        if pc and taxonomy:
            # Product class should be a child of the deepest level
            deepest = (f"level_{len(taxonomy)}", taxonomy[-1])
            if deepest not in pc_parents:
                pc_parents[deepest] = set()
            pc_parents[deepest].add(pc)

    return valid_children, class_depths


def validate_taxonomy_chain(taxonomy_dict, valid_children):
    """Validate taxonomy predictions and truncate where chain breaks.

    Rules:
        1. level_1 always kept (root)
        2. level_N kept only if its value is a valid child of level_(N-1)
        3. Stop at first break — everything after is invalid
        4. Also stop if confidence drops below 0.15 (random guessing)

    Returns:
        validated: dict of valid levels only
        depth: number of valid levels
    """
    if not taxonomy_dict:
        return {}, 0

    validated = {}
    sorted_levels = sorted(
        ((k, v) for k, v in taxonomy_dict.items() if k.startswith("level_")),
        key=lambda x: int(x[0].split("_")[1])
    )

    prev_level = None
    prev_value = None

    for lk, info in sorted_levels:
        value = info["value"]
        conf = info.get("confidence", 0)

        # Skip <UNK> predictions
        if value == "<UNK>":
            break

        # level_1: always accept
        if prev_level is None:
            validated[lk] = info
            prev_level = lk
            prev_value = value
            continue

        # Check if this value is a valid child of previous level
        parent_key = (prev_level, prev_value)
        children = valid_children.get(parent_key, set())

        if children and value not in children:
            # Invalid child — chain is broken, stop here
            break

        if not children:
            # No known children for this parent — taxonomy ends here.
            # Don't trust confidence — levels with 1 class show 99%.
            break

        # Confidence too low — probably random
        if conf < 0.15:
            break

        validated[lk] = info
        prev_level = lk
        prev_value = value

    return validated, len(validated)

# ================================================================
# CACHED MODEL LOADERS
# ================================================================

@st.cache_resource
def load_llava_model(adapter_path):
    from transformers import (AutoTokenizer, AutoProcessor,
                              LlavaForConditionalGeneration, BitsAndBytesConfig)
    from peft import PeftModel
    base_model = "llava-hf/llava-1.5-7b-hf"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    model = LlavaForConditionalGeneration.from_pretrained(
        base_model, quantization_config=bnb_config,
        device_map="auto", torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    processor = AutoProcessor.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, processor

@st.cache_resource
def load_classifier():
    try:
        from scripts.train_classifier import ProductClassifier
    except ImportError:
        return None, None, None
    paths = CLASSIFIER_PATHS
    if not all(Path(p).exists() for p in paths.values()):
        return None, None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProductClassifier(
        taxonomy_path=paths["taxonomy"], vocab_path=paths["vocab"]).to(device)
    ckpt = torch.load(paths["checkpoint"], map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    meta = {"epoch": ckpt.get("epoch", "?"), "val_loss": ckpt.get("val_loss", 0),
            "accuracy": ckpt.get("accuracy", {})}
    return model, device, meta

@st.cache_resource
def load_search_pipeline():
    import yaml
    config_path = Path("configs/search.yaml")
    if not config_path.exists():
        return None
    with open(config_path) as f:
        config = yaml.safe_load(f)
    from src.search.pipeline import SearchPipeline
    try:
        return SearchPipeline.from_config(config)
    except Exception:
        return None

@st.cache_resource
def load_bm25():
    corpus_path = "data/search/product_corpus.jsonl"
    if not Path(corpus_path).exists():
        return None
    try:
        from src.search.pipeline import BM25Baseline
        return BM25Baseline(corpus_path)
    except Exception:
        return None

# ================================================================
# EXTRACTION: VLM (LLaVA)
# ================================================================

def extract_vlm(name, desc, product_class, image_file, adapter_path):
    from src.inference.postprocessor import PostProcessor
    model, tokenizer, processor = load_llava_model(adapter_path)
    pp = PostProcessor()
    input_text = f"Product: {name}\nCategory: {product_class}\nDescription: {desc}"[:300]

    if image_file is not None:
        prompt = (
            "You are a product catalog specialist. You are given a product image "
            "and its text listing. Extract structured attributes by analyzing BOTH "
            "the image and the text.\n\n"
            "Example output format:\n"
            '{"style": "modern & contemporary", "primary_material": "wood", '
            '"secondary_material": null, "color_family": "brown", '
            '"room_type": "living room", "product_type": "table", '
            '"assembly_required": true}\n\n'
            f"<image>\n{input_text}\n\nExtracted attributes (JSON):")
        image = Image.open(image_file).convert("RGB")
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    else:
        prompt = (
            "You are a product catalog specialist. Extract structured attributes "
            "from the following product listing.\n\n"
            "Example output format:\n"
            '{"style": "modern & contemporary", "primary_material": "wood", '
            '"secondary_material": null, "color_family": "brown", '
            '"room_type": "living room", "product_type": "table", '
            '"assembly_required": true}\n\n'
            f"{input_text}\n\nExtracted attributes (JSON):")
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512,
                           truncation=True).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs, max_new_tokens=300, do_sample=False,
            pad_token_id=tokenizer.eos_token_id)
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(generated, skip_special_tokens=True)
    parsed = pp.process(raw)
    return parsed, raw

# ================================================================
# EXTRACTION: CLASSIFIER
# ================================================================

def extract_classifier(name, desc, product_class, image_file,
                       confidence_threshold=0.5):
    from src.classifier.dataset import get_image_transforms
    model, device, meta = load_classifier()
    if model is None:
        return None, None
    transform = get_image_transforms(train=False)
    t0 = time.time()

    parts = [name]
    if product_class:
        parts.append(product_class)
    if desc:
        parts.append(desc[:200])
    text = " [SEP] ".join(parts)

    images, mask = [], []
    if image_file is not None:
        try:
            img = transform(Image.open(image_file).convert("RGB"))
            images.append(img)
            mask.append(True)
        except Exception:
            pass
    while len(images) < 2:
        images.append(torch.zeros(3, 224, 224))
        mask.append(False)

    batch = {
        "text_input": [text],
        "images": torch.stack(images).unsqueeze(0).to(device),
        "image_mask": torch.tensor(mask[:2]).unsqueeze(0).to(device),
    }
    with torch.inference_mode():
        out = model(batch)
    classifier_ms = (time.time() - t0) * 1000

    result = {"_meta": {"latency_ms": round(classifier_ms, 1),
                        "model_epoch": meta.get("epoch", "?")},
              "taxonomy": {}, "product_class": None,
              "attributes": {}, "vlm_needed": []}

    for lk in model.taxonomy_heads.level_keys:
        if lk not in out["taxonomy"]:
            continue
        head_out = out["taxonomy"][lk]
        probs = torch.softmax(head_out["logits"], dim=-1)
        conf, pred = probs.max(-1)
        value = model.taxonomy_heads.level_i2v[lk].get(pred[0].item(), "<UNK>")
        c = conf[0].item()
        result["taxonomy"][lk] = {"value": value, "confidence": round(c, 3)}
        if c < confidence_threshold:
            result["vlm_needed"].append(lk)

    # Validate taxonomy chain — remove levels that break hierarchy
    valid_children, _ = build_taxonomy_hierarchy()
    validated_tax, depth = validate_taxonomy_chain(
        result["taxonomy"], valid_children)

    # Remove invalid levels from vlm_needed too
    removed = set(result["taxonomy"].keys()) - set(validated_tax.keys())
    result["vlm_needed"] = [f for f in result["vlm_needed"] if f not in removed]
    result["taxonomy"] = validated_tax
    result["_meta"]["taxonomy_depth"] = depth

    if model.taxonomy_heads.has_class_head and "product_class" in out["taxonomy"]:
        head_out = out["taxonomy"]["product_class"]
        probs = torch.softmax(head_out["logits"], dim=-1)
        conf, pred = probs.max(-1)
        value = model.taxonomy_heads.class_i2v.get(pred[0].item(), "<UNK>")
        c = conf[0].item()
        result["product_class"] = {"value": value, "confidence": round(c, 3)}
        if c < confidence_threshold:
            result["vlm_needed"].append("product_class")

    for attr, head_out in out["attributes"].items():
        probs = torch.softmax(head_out["logits"], dim=-1)
        conf, pred = probs.max(-1)
        value = model.attribute_heads.idx_to_value[attr].get(pred[0].item(), "<UNK>")
        c = conf[0].item()
        result["attributes"][attr] = {"value": value, "confidence": round(c, 3)}
        if c < confidence_threshold:
            result["vlm_needed"].append(attr)

    return result, out

# ================================================================
# EXTRACTION: HYBRID (Classifier + VLM Fallback)
# ================================================================

def extract_hybrid(name, desc, product_class, image_file,
                   adapter_path, confidence_threshold=0.5):
    cls_result, _ = extract_classifier(
        name, desc, product_class, image_file, confidence_threshold)
    if cls_result is None:
        vlm_result, raw = extract_vlm(name, desc, product_class, image_file, adapter_path)
        return {"mode": "vlm_only (classifier unavailable)",
                "attributes": vlm_result, "vlm_fields": "all",
                "latency_classifier_ms": 0, "latency_vlm_ms": None}

    vlm_fields = cls_result["vlm_needed"]
    hybrid = {"mode": "hybrid", "taxonomy": cls_result["taxonomy"],
              "product_class": cls_result["product_class"],
              "attributes": {}, "field_sources": {},
              "vlm_fields": vlm_fields,
              "latency_classifier_ms": cls_result["_meta"]["latency_ms"],
              "latency_vlm_ms": 0}

    for attr, info in cls_result["attributes"].items():
        hybrid["attributes"][attr] = info
        hybrid["field_sources"][attr] = "classifier"

    if vlm_fields:
        t0 = time.time()
        vlm_result, _ = extract_vlm(name, desc, product_class, image_file, adapter_path)
        hybrid["latency_vlm_ms"] = round((time.time() - t0) * 1000, 1)
        for field in vlm_fields:
            if field in vlm_result and vlm_result[field] is not None:
                if field in hybrid["attributes"]:
                    hybrid["attributes"][field] = {
                        "value": vlm_result[field], "confidence": None, "source": "vlm"}
                    hybrid["field_sources"][field] = "vlm"

    return hybrid

# ================================================================
# UI HELPERS
# ================================================================

def conf_color(conf):
    if conf is None: return "#6b7280"
    if conf >= 0.7: return CONFIDENCE_COLORS["high"]
    if conf >= 0.4: return CONFIDENCE_COLORS["medium"]
    return CONFIDENCE_COLORS["low"]

def conf_label(conf):
    if conf is None: return "VLM"
    if conf >= 0.7: return "HIGH"
    if conf >= 0.4: return "MED"
    return "LOW"

def _card(label, value, conf=None, source=None, vlm_flag=False):
    color = conf_color(conf)
    src_tag = ""
    if source == "vlm":
        src_tag = " — 🔄 VLM"
    elif source == "classifier":
        src_tag = " — ⚡ Fast"
    elif vlm_flag:
        src_tag = " 🔄"
    conf_text = f"{conf:.0%} {conf_label(conf)}" if conf else "VLM"
    if isinstance(value, str):
        value = value.replace("_", " ").title()
    return (
        f"<div style='padding:10px; border-radius:8px; "
        f"background:#1e293b; margin:4px 0; "
        f"border-left: 4px solid {color};'>"
        f"<small style='color:#94a3b8;'>{label}{src_tag}</small><br>"
        f"<b style='color:#f8fafc; font-size:1.1em;'>{value}</b>"
        f"<br><small style='color:{color};'>{conf_text}</small>"
        f"</div>"
    )

def show_vlm_result(result, raw_output=None):
    if not result or "error" in result:
        st.warning("No attributes extracted" if not result else result["error"])
        return
    cols = st.columns(3)
    i = 0
    for attr, value in result.items():
        if value is None or attr.startswith("_"):
            continue
        with cols[i % 3]:
            st.markdown(_card(attr.replace("_", " ").title(),
                              str(value)), unsafe_allow_html=True)
        i += 1
    with st.expander("Raw JSON"):
        st.json(result)
    if raw_output:
        with st.expander("Raw model output"):
            st.code(raw_output, language="text")

def show_classifier_result(result):
    if result is None:
        st.warning("Classifier not available")
        return
    meta = result.get("_meta", {})
    st.caption(f"⚡ {meta.get('latency_ms', '?')}ms | Epoch {meta.get('model_epoch', '?')}")

    st.markdown("##### Taxonomy")
    cols = st.columns(3)
    for i, (lk, info) in enumerate(sorted(result.get("taxonomy", {}).items())):
        with cols[i % 3]:
            vlm = lk in result.get("vlm_needed", [])
            st.markdown(_card(lk, info["value"], info["confidence"],
                              vlm_flag=vlm), unsafe_allow_html=True)

    pc = result.get("product_class")
    if pc:
        vlm = "product_class" in result.get("vlm_needed", [])
        st.markdown(_card("Product Class", pc["value"], pc["confidence"],
                          vlm_flag=vlm), unsafe_allow_html=True)

    st.markdown("##### Attributes")
    cols = st.columns(3)
    for i, attr in enumerate(ATTR_ORDER):
        info = result.get("attributes", {}).get(attr)
        if not info:
            continue
        with cols[i % 3]:
            vlm = attr in result.get("vlm_needed", [])
            st.markdown(_card(attr.replace("_", " ").title(),
                              info["value"], info["confidence"],
                              vlm_flag=vlm), unsafe_allow_html=True)

    n_vlm = len(result.get("vlm_needed", []))
    total = (len(result.get("taxonomy", {})) + (1 if pc else 0) +
             len(result.get("attributes", {})))
    if n_vlm > 0:
        st.info(f"🔄 VLM fallback recommended for **{n_vlm}/{total}** fields")
    else:
        st.success(f"✅ All {total} fields above confidence threshold")

    with st.expander("Raw JSON"):
        st.json(result)

def show_hybrid_result(result):
    if result is None:
        st.warning("No result")
        return
    cls_ms = result.get("latency_classifier_ms", 0)
    vlm_ms = result.get("latency_vlm_ms", 0)
    vlm_fields = result.get("vlm_fields", [])

    st.caption(f"⚡ Classifier: {cls_ms:.0f}ms | 🔄 VLM: {vlm_ms:.0f}ms | "
               f"Total: {cls_ms + vlm_ms:.0f}ms")

    st.markdown("##### Taxonomy")
    cols = st.columns(3)
    for i, (lk, info) in enumerate(sorted(result.get("taxonomy", {}).items())):
        with cols[i % 3]:
            src = "vlm" if lk in vlm_fields else "classifier"
            st.markdown(_card(lk, info["value"], info.get("confidence"),
                              source=src), unsafe_allow_html=True)

    pc = result.get("product_class")
    if pc:
        src = "vlm" if "product_class" in vlm_fields else "classifier"
        st.markdown(_card("Product Class", pc["value"], pc.get("confidence"),
                          source=src), unsafe_allow_html=True)

    st.markdown("##### Attributes")
    cols = st.columns(3)
    for i, attr in enumerate(ATTR_ORDER):
        info = result.get("attributes", {}).get(attr)
        if not info:
            continue
        with cols[i % 3]:
            src = result.get("field_sources", {}).get(attr, "classifier")
            val = info.get("value", "?")
            st.markdown(_card(attr.replace("_", " ").title(), val,
                              info.get("confidence"), source=src),
                        unsafe_allow_html=True)

    n_cls = sum(1 for v in result.get("field_sources", {}).values() if v == "classifier")
    n_vlm = sum(1 for v in result.get("field_sources", {}).values() if v == "vlm")
    if vlm_ms > 0:
        st.success(f"⚡ {n_cls} fields from classifier ({cls_ms:.0f}ms) + "
                   f"🔄 {n_vlm} fields from VLM ({vlm_ms:.0f}ms)")
    else:
        st.success(f"⚡ All {n_cls} fields from classifier ({cls_ms:.0f}ms) — no VLM needed!")

    with st.expander("Raw JSON"):
        st.json(result)

def show_search_results(results, query=""):
    if not results:
        st.warning("No results found")
        return
    for r in results:
        name = r.get("product_name", "Unknown")
        score = r.get("boosted_score", r.get("ce_score", r.get("score", 0)))
        rank = r.get("final_rank", r.get("rank", ""))
        with st.container():
            c1, c2, c3 = st.columns([1, 6, 2])
            with c1:
                st.markdown(f"### #{rank}")
            with c2:
                st.markdown(f"**{name}**")
                st.caption(f"ID: {r.get('product_id', '')}")
            with c3:
                st.metric("Score", f"{score:.3f}")
            st.divider()

# ================================================================
# MAIN
# ================================================================

def main():
    st.set_page_config(page_title="Wayfair Product Intelligence",
                       page_icon="🏠", layout="wide")
    st.title("🏠 Wayfair Product Intelligence")
    st.markdown("**Multi-Tower Classifier · VLM Fallback · Search Relevance**")

    tab1, tab2 = st.tabs(["🔍 Attribute Extraction", "🔎 Product Search"])

    # ── TAB 1: EXTRACTION ──
    with tab1:
        st.sidebar.header("⚙️ Settings")
        mode = st.sidebar.radio(
            "Extraction Mode",
            ["🔄 VLM Only (LLaVA)",
             "⚡ Classifier Only",
             "⚡+🔄 Classifier + VLM Fallback"],
            index=2)

        adapter_path = None
        if "VLM" in mode or "Fallback" in mode:
            available = {k: v for k, v in LLAVA_ADAPTERS.items() if Path(v).exists()}
            if available:
                sel = st.sidebar.selectbox("LLaVA Adapter", list(available.keys()))
                adapter_path = available[sel]
            else:
                st.sidebar.warning("No LLaVA adapters found")

        confidence_threshold = 0.5
        if "Classifier" in mode:
            confidence_threshold = st.sidebar.slider(
                "Confidence Threshold", 0.1, 0.9, 0.5, 0.05)
            cls_model, _, cls_meta = load_classifier()
            if cls_model is None:
                st.sidebar.error("Classifier not found")
            else:
                st.sidebar.success(f"Classifier loaded (epoch {cls_meta['epoch']})")

        st.sidebar.markdown("---")
        st.sidebar.markdown("**Latency:**\n- ⚡ Classifier: ~50ms\n- 🔄 VLM: ~2-5s\n"
                            "- ⚡+🔄 Hybrid: 50ms + VLM for low-conf")

        col_in, col_out = st.columns([1, 1])
        with col_in:
            st.subheader("📦 Product Input")
            uploaded_image = st.file_uploader(
                "Product Image", type=["jpg", "jpeg", "png", "webp"])
            if uploaded_image:
                st.image(uploaded_image, use_container_width=True)
            product_name = st.text_input(
                "Product Name", value="Modern Walnut Wood Dining Table with Metal Legs")
            product_class = st.text_input("Category", value="Dining Tables")
            product_desc = st.text_area(
                "Description",
                value="Solid walnut table top with sleek black metal hairpin legs. "
                      "Mid-century modern style. Assembly required.",
                height=100)
            btn = st.button("🚀 Extract Attributes", type="primary",
                            use_container_width=True)

        with col_out:
            st.subheader("📊 Results")
            if btn:
                if uploaded_image:
                    uploaded_image.seek(0)

                if "VLM Only" in mode:
                    if adapter_path is None:
                        st.error("No LLaVA adapter")
                    else:
                        with st.spinner("🔄 Running LLaVA..."):
                            t0 = time.time()
                            result, raw = extract_vlm(
                                product_name, product_desc, product_class,
                                uploaded_image, adapter_path)
                            ms = (time.time() - t0) * 1000
                        st.caption(f"🔄 {ms:.0f}ms")
                        show_vlm_result(result, raw)

                elif "Classifier Only" in mode:
                    cls_model, _, _ = load_classifier()
                    if cls_model is None:
                        st.error("Classifier not loaded")
                    else:
                        with st.spinner("⚡ Running classifier..."):
                            result, _ = extract_classifier(
                                product_name, product_desc, product_class,
                                uploaded_image, confidence_threshold)
                        show_classifier_result(result)

                elif "Fallback" in mode:
                    cls_model, _, _ = load_classifier()
                    if cls_model is None:
                        st.error("Classifier not loaded")
                    elif adapter_path is None:
                        st.warning("No VLM — running classifier only")
                        with st.spinner("⚡ Running classifier..."):
                            result, _ = extract_classifier(
                                product_name, product_desc, product_class,
                                uploaded_image, confidence_threshold)
                        show_classifier_result(result)
                    else:
                        with st.spinner("⚡ Classifier → 🔄 VLM fallback..."):
                            result = extract_hybrid(
                                product_name, product_desc, product_class,
                                uploaded_image, adapter_path,
                                confidence_threshold)
                        show_hybrid_result(result)

    # ── TAB 2: SEARCH ──
    with tab2:
        st.markdown("**Bi-encoder → Cross-encoder → Attribute-boosted scoring**")
        pipeline = load_search_pipeline()
        bm25 = load_bm25()
        if pipeline is None and bm25 is None:
            st.warning("Search not available. Run training scripts first.")
            return

        st.sidebar.header("🔎 Search")
        search_mode = st.sidebar.selectbox(
            "Approach", ["Full Pipeline", "Bi-Encoder + Cross-Encoder",
                         "Bi-Encoder Only", "BM25 Baseline"], key="s_mode")
        top_k = st.sidebar.slider("Results", 5, 20, 10, key="topk")

        if "search_query" not in st.session_state:
            st.session_state.search_query = "modern blue velvet sofa"
        if "run_search" not in st.session_state:
            st.session_state.run_search = False

        def set_q(q):
            st.session_state.search_query = q
            st.session_state.run_search = True
        def on_enter():
            st.session_state.run_search = True

        query = st.text_input("Search", key="search_query",
                              placeholder="rustic wooden dining table...",
                              on_change=on_enter)
        ex_cols = st.columns(6)
        for col, ex in zip(ex_cols, ["blue modern sofa", "rustic dining table",
                "metal bookshelf", "velvet accent chair",
                "marble coffee table", "outdoor patio set"]):
            with col:
                st.button(ex, key=f"ex_{ex}", use_container_width=True,
                          on_click=set_q, args=(ex,))

        go = st.button("🔎 Search", type="primary", use_container_width=True, key="sbtn")
        should = go or st.session_state.run_search
        st.session_state.run_search = False
        query = st.session_state.search_query

        if should and query:
            with st.spinner("Searching..."):
                t0 = time.time()
                if search_mode == "BM25 Baseline" and bm25:
                    results = bm25.search(query, top_k)
                elif pipeline:
                    stage_map = {"Full Pipeline": ["bi_encoder","cross_encoder","attribute_boost"],
                                 "Bi-Encoder + Cross-Encoder": ["bi_encoder","cross_encoder"],
                                 "Bi-Encoder Only": ["bi_encoder"]}
                    results = pipeline.search(query, top_k,
                                              stages=stage_map.get(search_mode, ["bi_encoder"]))
                else:
                    results = []
                lat = (time.time()-t0)*1000
            st.caption(f"{len(results)} results in {lat:.0f}ms")
            show_search_results(results, query)

if __name__ == "__main__":
    main()