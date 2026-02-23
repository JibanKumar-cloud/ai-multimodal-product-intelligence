#!/usr/bin/env python3
"""Wayfair Catalog AI - Unified Demo App.
Tab 1: Attribute Extraction (VLM, rule-based, hybrid)
Tab 2: Product Search (bi-encoder -> cross-encoder -> attribute boost)

Usage:
    streamlit run demo/app.py --server.port 8501 --server.address 0.0.0.0
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import streamlit as st
except ImportError:
    print("pip install streamlit"); sys.exit(1)

EXTRACTION_MODELS = {
    "Rule-based (instant)": {"type": "rule-based", "path": None},
    "LLaVA QLoRA [Multimodal]": {"type": "llava", "path": "outputs/checkpoints/qlora-multimodal/best_model"},
    "LLaVA QLoRA [Text-Only]": {"type": "llava", "path": "outputs/checkpoints/qlora-text-only/best_model"},
    "LLaVA QLoRA [Vague+Image]": {"type": "llava", "path": "outputs/checkpoints/qlora-vague-multimodal/best_model"},
    "Hybrid (Visual + Text)": {"type": "hybrid", "path": "outputs/checkpoints/qlora-vague-multimodal/best_model"},
    "GPT-4o (API key required)": {"type": "gpt4o", "path": None},
}

# ── Cached loaders ──

@st.cache_resource
def load_llava_model(adapter_path):
    import torch
    from transformers import AutoTokenizer, AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
    from peft import PeftModel
    base_model = "llava-hf/llava-1.5-7b-hf"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    model = LlavaForConditionalGeneration.from_pretrained(
        base_model, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    processor = AutoProcessor.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, processor

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
    except Exception as e:
        print(f"Search pipeline load failed: {e}")
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

# ── Extraction functions ──

def extract_rule_based(name, desc, cls):
    from src.models.rule_based import RuleBasedExtractor
    return RuleBasedExtractor().extract(product_name=name, product_description=desc, product_class=cls)

def extract_llava(name, desc, cls, adapter_path, image_file=None):
    import torch
    from PIL import Image
    from src.inference.postprocessor import PostProcessor
    model, tokenizer, processor = load_llava_model(adapter_path)
    pp = PostProcessor()
    input_text = f"Product: {name}\nCategory: {cls}\nDescription: {desc}"[:300]
    if image_file is not None:
        prompt = (
            "You are a product catalog specialist. You are given a product image "
            "and its text listing. Extract structured attributes by analyzing BOTH the image and the text.\n\n"
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
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs, max_new_tokens=300, do_sample=False,
            pad_token_id=tokenizer.eos_token_id)
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"\nRAW: {raw[:500]}\n")
    parsed = pp.process(raw)
    return parsed

def extract_hybrid(name, desc, cls, adapter_path, image_file=None):
    from src.models.rule_based import RuleBasedExtractor
    visual = extract_llava(name, desc, cls, adapter_path, image_file)
    text_attrs = RuleBasedExtractor().extract(
        product_name=name, product_description=desc, product_class=cls)
    VISUAL_KEYS = {"style", "primary_material", "color_family", "product_type"}
    TEXT_KEYS = {"secondary_material", "room_type", "assembly_required"}
    merged = {}
    for key in VISUAL_KEYS:
        merged[key] = visual.get(key) if visual.get(key) is not None else text_attrs.get(key)
    for key in TEXT_KEYS:
        merged[key] = text_attrs.get(key)
    return merged

def extract_gpt4o(name, desc, cls):
    import os
    if not os.getenv("OPENAI_API_KEY"):
        return {"error": "Set OPENAI_API_KEY"}
    from src.models.gpt4o_extractor import GPT4oExtractor
    return GPT4oExtractor().extract(product_name=name, product_description=desc, product_class=cls)

def _run_extraction(cfg, name, desc, cls, image):
    if cfg["type"] == "rule-based":
        return extract_rule_based(name, desc, cls)
    elif cfg["type"] == "llava":
        return extract_llava(name, desc, cls, cfg["path"], image)
    elif cfg["type"] == "hybrid":
        return extract_hybrid(name, desc, cls, cfg["path"], image)
    elif cfg["type"] == "gpt4o":
        return extract_gpt4o(name, desc, cls)
    return {"error": "Unknown model"}

# ── UI helpers ──

def show_extraction_result(result):
    if "error" in result:
        st.error(result["error"])
        return
    if not result:
        st.warning("No attributes extracted")
        return
    for attr, value in result.items():
        if value is not None:
            if isinstance(value, list):
                v = ", ".join(value)
            elif isinstance(value, bool):
                v = "Yes" if value else "No"
            else:
                v = str(value).title()
            st.metric(label=attr.replace("_", " ").title(), value=v)
    st.markdown("---")
    st.json(result)

def show_search_results(results, query=""):
    if not results:
        st.warning("No results found")
        return
    for r in results:
        pid = r.get("product_id", "")
        name = r.get("product_name", "Unknown")
        score = r.get("boosted_score", r.get("ce_score", r.get("score", 0)))
        rank = r.get("final_rank", r.get("rank", ""))
        with st.container():
            col_rank, col_info, col_score = st.columns([1, 6, 2])
            with col_rank:
                st.markdown(f"### #{rank}")
            with col_info:
                st.markdown(f"**{name}**")
                st.caption(f"ID: {pid}")
                matches = r.get("product_attributes", {})
                query_attrs = r.get("query_attributes", {})
                if matches and query_attrs:
                    tags = []
                    for key in ["color_family", "style", "primary_material", "product_type"]:
                        qv = query_attrs.get(key, "")
                        pv = str(matches.get(key, "") or "").lower()
                        if qv and pv:
                            if qv.lower() in pv:
                                tags.append(f":green[{key}: {pv}]")
                            else:
                                tags.append(f":red[{key}: {pv}]")
                    if tags:
                        st.markdown(" | ".join(tags))
            with col_score:
                st.metric("Score", f"{score:.3f}")
            st.divider()

# ── Main ──

def main():
    st.set_page_config(page_title="Product Intelligence System", page_icon="\U0001F3E0", layout="wide")
    st.title("\U0001F3E0 Product Intelligence System")
    st.markdown("**End-to-end Product Intelligence: Attribute Extraction + Search Relevance**")

    tab1, tab2 = st.tabs(["\U0001F50D Catatlog Attribute Extraction", "\U0001F50E Product Search"])

    # === TAB 1: EXTRACTION ===
    with tab1:
        st.markdown("Upload a photo and/or enter a description to extract structured attributes.")
        st.sidebar.header("Settings")
        available = {}
        for name, cfg in EXTRACTION_MODELS.items():
            if cfg["path"] is None or Path(cfg["path"]).exists():
                available[name] = cfg
            else:
                available[name + " [not trained]"] = {**cfg, "disabled": True}
        selected = st.sidebar.selectbox("Extraction model", list(available.keys()), key="ext_model")
        model_cfg = available[selected]
        if model_cfg.get("disabled"):
            st.sidebar.error("Model not trained yet")
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Cost per 1K:**\n- Rule-based: ~$0\n- LLaVA: ~$0.20\n- GPT-4o: ~$10")
        compare = st.sidebar.checkbox("Compare two models", key="ext_cmp")
        compare_model = None
        if compare:
            others = [n for n in available if n != selected and not available[n].get("disabled")]
            if others:
                compare_model = st.sidebar.selectbox("Compare with", others, key="ext_cmp2")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Input")
            uploaded_image = None
            if model_cfg["type"] in ("llava", "hybrid", "gpt4o") and not model_cfg.get("disabled"):
                uploaded_image = st.file_uploader("Product Image (optional)", type=["jpg", "jpeg", "png"])
                if uploaded_image:
                    st.image(uploaded_image, caption="Product image", use_container_width=True)
            product_name = st.text_input("Product Name", value="Modern Walnut Wood Dining Table with Metal Legs")
            product_class = st.text_input("Category", value="Dining Tables")
            product_description = st.text_area(
                "Description",
                value="Solid walnut table, black metal hairpin legs. Assembly required.",
                height=80)
            extract_btn = st.button("Extract Attributes", type="primary", use_container_width=True)

        with col2:
            st.subheader("Results")
            if extract_btn and not model_cfg.get("disabled"):
                results = {}
                with st.spinner(f"Running {selected}..."):
                    t0 = time.time()
                    results[selected] = _run_extraction(
                        model_cfg, product_name, product_description, product_class, uploaded_image)
                    st.caption(f"Latency: {(time.time()-t0)*1000:.0f}ms")
                if compare_model:
                    with st.spinner(f"Running {compare_model}..."):
                        results[compare_model] = _run_extraction(
                            available[compare_model], product_name, product_description,
                            product_class, uploaded_image)
                if compare_model:
                    import pandas as pd
                    models = list(results.keys())
                    all_attrs = sorted(set(k for r in results.values() for k in r.keys()))
                    rows = []
                    for a in all_attrs:
                        row = {"Attribute": a.replace("_", " ").title()}
                        for m in models:
                            label = m.split("(")[0].strip()
                            val = results[m].get(a)
                            row[label] = "\u2014" if val is None else str(val).title()
                        rows.append(row)
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    show_extraction_result(results[selected])

    # === TAB 2: SEARCH ===
    with tab2:
        st.markdown("**Bi-encoder retrieval \u2192 Cross-encoder reranking \u2192 Attribute-boosted scoring**")
        pipeline = load_search_pipeline()
        bm25 = load_bm25()
        if pipeline is None and bm25 is None:
            st.warning(
                "Search not available yet. Run:\n\n"
                "`python scripts/prepare_search_data.py && "
                "python scripts/enrich_catalog.py && "
                "python scripts/train_search.py`")
            return
        st.sidebar.header("Search Settings")
        search_mode = st.sidebar.selectbox(
            "Approach",
            ["Full Pipeline", "Bi-Encoder + Cross-Encoder", "Bi-Encoder Only", "BM25 Baseline"],
            key="s_mode")
        top_k = st.sidebar.slider("Results", 5, 20, 10, key="topk")
        show_comparison = st.sidebar.checkbox("Compare all approaches", key="s_cmp")

        query = st.text_input(
            "Search query", value="modern blue velvet sofa", key="sq",
            placeholder="Try: rustic wooden dining table, industrial bookshelf...")
        ex_cols = st.columns(6)
        examples = [
            "blue modern sofa", "rustic dining table", "metal bookshelf",
            "velvet accent chair", "marble coffee table", "outdoor patio set"]
        for col, ex in zip(ex_cols, examples):
            with col:
                if st.button(ex, key=f"ex_{ex}", use_container_width=True):
                    query = ex

        search_btn = st.button(
            "\U0001F50E Search", type="primary", use_container_width=True, key="sbtn")

        if search_btn and query:
            if show_comparison and pipeline:
                st.subheader("Approach Comparison")
                comparisons = {}
                if bm25:
                    t0 = time.time()
                    comparisons["BM25"] = (bm25.search(query, top_k), (time.time()-t0)*1000)
                if pipeline:
                    for sname, stages in [
                        ("Bi-Encoder", ["bi_encoder"]),
                        ("+ Cross-Encoder", ["bi_encoder", "cross_encoder"]),
                        ("+ Attr Boost", ["bi_encoder", "cross_encoder", "attribute_boost"]),
                    ]:
                        t0 = time.time()
                        comparisons[sname] = (
                            pipeline.search(query, top_k, stages=stages),
                            (time.time()-t0)*1000)
                cols = st.columns(len(comparisons))
                for col, (cname, (res, lat)) in zip(cols, comparisons.items()):
                    with col:
                        st.markdown(f"**{cname}** ({lat:.0f}ms)")
                        for r in res[:5]:
                            s = r.get("boosted_score", r.get("ce_score", r.get("score", 0)))
                            st.markdown(f"- {r.get('product_name', '')[:40]}")
                            st.caption(f"  Score: {s:.3f}")
            else:
                with st.spinner("Searching..."):
                    t0 = time.time()
                    if search_mode == "BM25 Baseline" and bm25:
                        results = bm25.search(query, top_k)
                    elif pipeline:
                        stage_map = {
                            "Full Pipeline": ["bi_encoder", "cross_encoder", "attribute_boost"],
                            "Bi-Encoder + Cross-Encoder": ["bi_encoder", "cross_encoder"],
                            "Bi-Encoder Only": ["bi_encoder"],
                        }
                        stages = stage_map.get(search_mode, ["bi_encoder"])
                        results = pipeline.search(query, top_k, stages=stages)
                    else:
                        results = []
                    latency = (time.time()-t0)*1000
                st.caption(f"{len(results)} results in {latency:.0f}ms | {search_mode}")
                if pipeline and pipeline.attribute_booster:
                    qa = pipeline.attribute_booster.parse_query_attributes(query)
                    if qa:
                        st.info("Detected: " + " | ".join(
                            f"**{k}**: {v}" for k, v in qa.items()))
                show_search_results(results, query)


if __name__ == "__main__":
    main()