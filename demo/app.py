#!/usr/bin/env python3
"""Wayfair Product Intelligence — Demo App.

Three extraction modes via unified ProductPipeline:
  1. VLM Only (LLaVA)        — Full VLM extraction (~2s)
  2. Classifier Only          — Multi-tower model (~50ms)
  3. Classifier + VLM Fallback — Fast classifier, LLaVA for low confidence

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

from src.inference.postprocessor import (
    COLOR_FAMILIES, MATERIAL_GROUPS, STYLE_GROUPS,
    SHAPE_GROUPS, ASSEMBLY_GROUPS,
)

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
    "queue": "data/processed/image_queue_with_images.json",
}

ATTR_ORDER = [
    "primary_color", "secondary_color",
    "primary_material", "secondary_material",
    "style", "shape", "assembly",
]


# ================================================================
# PIPELINE LOADER (cached)
# ================================================================

@st.cache_resource
def get_pipeline(adapter_path=None):
    """Create ProductPipeline (cached). Lazy-loads models on first use."""
    from src.inference.pipeline import ProductPipeline
    return ProductPipeline(
        classifier_checkpoint=CLASSIFIER_PATHS["checkpoint"],
        taxonomy_path=CLASSIFIER_PATHS["taxonomy"],
        vocab_path=CLASSIFIER_PATHS["vocab"],
        queue_path=CLASSIFIER_PATHS["queue"],
        vlm_adapter_path=adapter_path,
    )

@st.cache_resource
def load_search_pipeline():
    import yaml
    config_path = Path("configs/search.yaml")
    if not config_path.exists(): return None
    with open(config_path) as f:
        config = yaml.safe_load(f)
    from src.search.pipeline import SearchPipeline
    try: return SearchPipeline.from_config(config)
    except Exception: return None

@st.cache_resource
def load_bm25():
    corpus_path = "data/search/product_corpus.jsonl"
    if not Path(corpus_path).exists(): return None
    try:
        from src.search.pipeline import BM25Baseline
        return BM25Baseline(corpus_path)
    except Exception: return None

# ================================================================
# UI HELPERS
# ================================================================

TABLE_HEADER_BG = "#1e3a5f"
TABLE_HEADER_TEXT = "#ffffff"
TABLE_ROW_EVEN = "#f0f4f8"
TABLE_ROW_ODD = "#ffffff"
TABLE_BORDER = "#c8d6e5"
TABLE_TEXT = "#1e293b"


def _conf_html(conf, source=None):
    """Return confidence as inline-styled HTML span."""
    if conf == "HV":
        return '<span style="color:#059669; font-weight:600;">✅ HV</span>'
    if conf is None:
        if source == "vlm":
            return '<span style="color:#6366f1; font-weight:600;">🔄 VLM</span>'
        return '<span style="color:#8b5cf6; font-weight:600;">—</span>'
    if conf >= 0.7:
        return f'<span style="color:#16a34a; font-weight:600;">🟢 {conf:.0%}</span>'
    if conf >= 0.4:
        return f'<span style="color:#d97706; font-weight:600;">🟡 {conf:.0%}</span>'
    return f'<span style="color:#dc2626; font-weight:600;">🔴 {conf:.0%}</span>'


def _table_html(headers, rows, col_widths=None):
    """Build a professional HTML table with inline styles."""
    th_style = (f"background:{TABLE_HEADER_BG}; color:{TABLE_HEADER_TEXT}; "
                f"padding:10px 14px; text-align:left; font-weight:600; "
                f"font-size:13px; letter-spacing:0.3px;")
    th_right = th_style + " text-align:right;"

    header_html = "<tr>"
    for i, h in enumerate(headers):
        w = f" width:{col_widths[i]};" if col_widths and i < len(col_widths) else ""
        align = th_right if i == len(headers) - 1 else th_style
        header_html += f'<th style="{align}{w}">{h}</th>'
    header_html += "</tr>"

    rows_html = ""
    for idx, row in enumerate(rows):
        bg = TABLE_ROW_EVEN if idx % 2 == 0 else TABLE_ROW_ODD
        td_style = (f"padding:8px 14px; border-bottom:1px solid {TABLE_BORDER}; "
                    f"color:{TABLE_TEXT}; background:{bg};")
        rows_html += "<tr>"
        for i, cell in enumerate(row):
            align = " text-align:right;" if i == len(row) - 1 else ""
            rows_html += f'<td style="{td_style}{align}">{cell}</td>'
        rows_html += "</tr>"

    return (f'<table style="width:100%; border-collapse:collapse; '
            f'font-family:-apple-system,BlinkMacSystemFont,sans-serif; '
            f'font-size:14px; margin:8px 0; border-radius:8px; '
            f'overflow:hidden; box-shadow:0 1px 3px rgba(0,0,0,0.1);">'
            f'<thead>{header_html}</thead>'
            f'<tbody>{rows_html}</tbody></table>')


def show_taxonomy_table(result, key_prefix="tax"):
    """Display taxonomy as professional read-only table."""
    tax = result.get("taxonomy", {})
    if not tax:
        return result

    rows = []
    for lk, info in sorted(tax.items()):
        val = info.get("value", "").replace("_", " ").title()
        conf = _conf_html(info.get("confidence"))
        level_num = lk.replace("level_", "L")
        rows.append([
            f'<span style="color:#64748b; font-size:12px;">{level_num}</span>',
            f'<b style="color:{TABLE_TEXT};">{val}</b>',
            conf,
        ])

    html = _table_html(
        ["Level", "Category", "Confidence"],
        rows,
        col_widths=["60px", None, "100px"])
    st.markdown(html, unsafe_allow_html=True)

    return {"taxonomy": tax, "product_class": result.get("product_class")}


def show_product_class(result):
    """Display product class as single-row table."""
    pc = result.get("product_class")
    if not pc or not pc.get("value"):
        return

    val = pc["value"].replace("_", " ").title()
    conf = _conf_html(pc.get("confidence"))

    html = _table_html(
        ["Product Class", "Confidence"],
        [[f'<b style="color:{TABLE_TEXT};">{val}</b>', conf]],
        col_widths=[None, "100px"])
    st.markdown(html, unsafe_allow_html=True)


def _show_attrs_loading(result, sources=None, vlm_loading=None):
    """Show attributes as static HTML table with loading indicators.
    Used while VLM is processing — no interactive elements."""
    attrs = result.get("attributes", {})
    rows = []
    for attr in ATTR_ORDER:
        info = attrs.get(attr)
        if not info or not info.get("value"):
            continue
        val = str(info["value"]).replace("_", " ").title()
        is_loading = vlm_loading and attr in vlm_loading
        if is_loading:
            val_html = f'<span style="color:#94a3b8;">{val} ⏳</span>'
            conf_html = '<span style="color:#8b5cf6; font-weight:600;">⏳ Loading</span>'
        else:
            val_html = f'<b>{val}</b>'
            conf_html = _conf_html(info.get("confidence"),
                                   sources.get(attr) if sources else None)
        rows.append([
            f'<span style="font-weight:600;">{attr.replace("_"," ").title()}</span>',
            val_html,
            conf_html,
        ])
    html = _table_html(["Attribute", "Value", "Confidence"], rows,
                        col_widths=["150px", None, "100px"])
    st.markdown(html, unsafe_allow_html=True)


def show_attributes_table(result, sources=None, vlm_loading=None,
                          disabled=False, key_prefix="attr"):
    """Display attributes with dropdown values and professional styling."""
    attrs = result.get("attributes", {})

    VALID_OPTIONS = {
        "primary_color": sorted(COLOR_FAMILIES.keys()),
        "secondary_color": sorted(COLOR_FAMILIES.keys()),
        "primary_material": sorted(MATERIAL_GROUPS.keys()),
        "secondary_material": sorted(MATERIAL_GROUPS.keys()),
        "style": sorted(STYLE_GROUPS.keys()),
        "shape": sorted(SHAPE_GROUPS.keys()),
        "assembly": sorted(ASSEMBLY_GROUPS.keys()),
    }

    if not any(attrs.get(a, {}).get("value") for a in ATTR_ORDER):
        return {}

    # Header row matching taxonomy table style
    st.markdown(
        f'<div style="display:flex; padding:8px 14px; '
        f'background:{TABLE_HEADER_BG}; color:{TABLE_HEADER_TEXT}; '
        f'border-radius:8px 8px 0 0; font-weight:600; font-size:13px; '
        f'letter-spacing:0.3px; margin-top:8px; '
        f'font-family:-apple-system,BlinkMacSystemFont,sans-serif;">'
        f'<div style="flex:2;">Attribute</div>'
        f'<div style="flex:3;">Value</div>'
        f'<div style="flex:1.2; text-align:right;">Confidence</div>'
        f'</div>',
        unsafe_allow_html=True)

    edited_result = {}

    for idx, attr in enumerate(ATTR_ORDER):
        info = attrs.get(attr)
        if not info or not info.get("value"):
            continue

        original_val = str(info["value"]).lower().replace(" ", "_")
        conf = info.get("confidence")
        src = sources.get(attr) if sources else None
        is_loading = vlm_loading and attr in vlm_loading

        options = VALID_OPTIONS.get(attr, [])
        if original_val not in options:
            options = [original_val] + options

        bg = TABLE_ROW_EVEN if idx % 2 == 0 else TABLE_ROW_ODD

        col_name, col_val, col_conf = st.columns([2, 3, 1.2])

        with col_name:
            label = attr.replace("_", " ").title()
            st.markdown(
                f"<div style='padding:2px 0; color:{TABLE_TEXT}; "
                f"font-weight:600; font-size:13px; "
                f"font-family:-apple-system,BlinkMacSystemFont,sans-serif;'>"
                f"{label}</div>",
                unsafe_allow_html=True)

        with col_val:
            if is_loading:
                display = original_val.replace("_", " ").title()
                st.markdown(
                    f"<div style='padding:2px 0; color:#94a3b8; "
                    f"font-size:13px;'>{display} ⏳</div>",
                    unsafe_allow_html=True)
                selected = original_val
            else:
                display_options = [o.replace("_", " ").title()
                                   for o in options]
                current_idx = (options.index(original_val)
                               if original_val in options else 0)
                selected_display = st.selectbox(
                    f"val_{attr}",
                    display_options,
                    index=current_idx,
                    key=f"{key_prefix}_{attr}",
                    label_visibility="collapsed",
                )
                sel_idx = display_options.index(selected_display)
                selected = options[sel_idx]

        with col_conf:
            if is_loading:
                badge_text = '<span style="color:#8b5cf6; font-weight:600;">⏳ Loading</span>'
            elif selected != original_val:
                badge_text = '<span style="color:#059669; font-weight:600;">✅ HV</span>'
            else:
                badge_text = _conf_html(conf, src)

            st.markdown(
                f"<div style='padding:2px 0; text-align:right;'>"
                f"{badge_text}</div>",
                unsafe_allow_html=True)

        edited_result[attr] = {
            "value": selected,
            "confidence": "HV" if selected != original_val else conf,
        }

    # VLM loading message at the bottom
    if vlm_loading:
        loading_attrs = [a.replace("_", " ").title() for a in vlm_loading]
        st.markdown(
            f"<div style='padding:10px 14px; background:#ede9fe; "
            f"border-radius:0 0 8px 8px; color:#6d28d9; "
            f"font-family:-apple-system,BlinkMacSystemFont,sans-serif; font-size:13px;'>"
            f"🔄 Running VLM for: <b>{', '.join(loading_attrs)}</b>"
            f"</div>",
            unsafe_allow_html=True)

    return edited_result


def build_export_json(tax_result, attr_edited, original_result):
    """Build final JSON from edited tables."""
    export = {}

    # Taxonomy
    if tax_result and tax_result.get("taxonomy"):
        for lk, info in sorted(tax_result["taxonomy"].items()):
            export[lk] = info.get("value", "")

    # Product class
    pc = (tax_result or original_result or {}).get("product_class")
    if pc and pc.get("value"):
        export["product_class"] = pc["value"]

    # Attributes
    if attr_edited:
        for attr, info in attr_edited.items():
            export[attr] = info.get("value", "")
    else:
        for attr in ATTR_ORDER:
            info = original_result.get("attributes", {}).get(attr)
            if info and info.get("value"):
                export[attr] = info["value"]

    return export

def show_vlm_result(result, raw=None):
    """Display VLM-only result using attributes table."""
    sources = {attr: "vlm" for attr in ATTR_ORDER}
    attr_edited = show_attributes_table(result, sources, key_prefix="vlm_attr")

    export = build_export_json(None, attr_edited, result)
    with st.expander("📋 Export JSON (editable above)"):
        st.json(export)
    if raw:
        with st.expander("Raw VLM output"):
            st.code(raw, language="text")

def show_search_results(results, query=""):
    if not results:
        st.warning("No results found"); return
    for r in results:
        name = r.get("product_name", "Unknown")
        score = r.get("boosted_score", r.get("ce_score", r.get("score", 0)))
        rank = r.get("final_rank", r.get("rank", ""))
        with st.container():
            c1, c2, c3 = st.columns([1, 6, 2])
            with c1: st.markdown(f"### #{rank}")
            with c2:
                st.markdown(f"**{name}**")
                st.caption(f"ID: {r.get('product_id', '')}")
            with c3: st.metric("Score", f"{score:.3f}")
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
            index=1)

        # Adapter selection
        adapter_path = None
        if "VLM" in mode or "Fallback" in mode:
            available = {k: v for k, v in LLAVA_ADAPTERS.items()
                         if Path(v).exists()}
            if available:
                sel = st.sidebar.selectbox("LLaVA Adapter",
                                           list(available.keys()))
                adapter_path = available[sel]
            else:
                st.sidebar.warning("No LLaVA adapters found")

        # Confidence threshold
        confidence_threshold = 0.5
        if "Classifier" in mode:
            confidence_threshold = st.sidebar.slider(
                "Confidence Threshold", 0.1, 0.9, 0.5, 0.05)

        # Load pipeline
        pipeline = get_pipeline(adapter_path)

        if "Classifier" in mode and pipeline.has_classifier:
            try:
                meta = pipeline.classifier_meta
                st.sidebar.success(
                    f"Classifier loaded (epoch {meta['epoch']})")
            except Exception:
                st.sidebar.error("Classifier failed to load")

        st.sidebar.markdown("---")
        st.sidebar.markdown(
            "**Latency:**\n"
            "- ⚡ Classifier: ~50ms\n"
            "- 🔄 VLM: ~2-5s\n"
            "- ⚡+🔄 Hybrid: 50ms + VLM for low-conf")

        # ── Input ──
        col_in, col_out = st.columns([1, 1])
        with col_in:
            st.subheader("📦 Product Input")
            uploaded_image = st.file_uploader(
                "Product Image", type=["jpg", "jpeg", "png", "webp"])
            if uploaded_image:
                st.image(uploaded_image, use_container_width=True)
            product_name = st.text_input(
                "Product Name",
                value="Modern Walnut Wood Dining Table with Metal Legs")
            product_class = st.text_input("Category", value="Dining Tables")
            product_desc = st.text_area(
                "Description",
                value="Solid walnut table top with sleek black metal "
                      "hairpin legs. Mid-century modern style. "
                      "Assembly required.",
                height=100)
            btn = st.button("🚀 Extract Attributes", type="primary",
                            use_container_width=True)

        # ── Run extraction on button click ──
        if btn:
            img = uploaded_image
            if img: img.seek(0)

            if "VLM Only" in mode:
                if pipeline.has_vlm:
                    with st.spinner("🔄 Running LLaVA..."):
                        t0 = time.time()
                        result, raw = pipeline.vlm_predict(
                            product_name, product_class,
                            product_desc, img)
                        ms = (time.time() - t0) * 1000
                    st.session_state["extract_result"] = {
                        "mode": "vlm", "result": result,
                        "raw": raw, "ms": ms}

            elif "Classifier Only" in mode:
                if pipeline.has_classifier:
                    with st.spinner("⚡ Running classifier..."):
                        result = pipeline.classifier_predict(
                            product_name, product_class,
                            product_desc, img, confidence_threshold)
                    st.session_state["extract_result"] = {
                        "mode": "classifier", "result": result}

            elif "Fallback" in mode:
                if pipeline.has_classifier:
                    # Step 1: Classifier
                    with st.spinner("⚡ Running classifier..."):
                        cls_result = pipeline.classifier_predict(
                            product_name, product_class,
                            product_desc, img, confidence_threshold)

                    vlm_attrs = [f for f in cls_result.get("vlm_needed", [])
                                 if f in ATTR_ORDER]

                    if vlm_attrs and pipeline.has_vlm:
                        # Step 2: VLM
                        if img: img.seek(0)
                        with st.spinner(
                            f"🔄 VLM for {len(vlm_attrs)} attributes..."):
                            merged, cls_ms, vlm_ms, _ = \
                                pipeline.hybrid_predict(
                                    product_name, product_class,
                                    product_desc, img,
                                    confidence_threshold)
                        st.session_state["extract_result"] = {
                            "mode": "hybrid", "result": merged,
                            "cls_ms": cls_ms, "vlm_ms": vlm_ms,
                            "vlm_attrs": vlm_attrs}
                    else:
                        st.session_state["extract_result"] = {
                            "mode": "classifier", "result": cls_result}

        # ── Display results (persists across reruns) ──
        with col_out:
            st.subheader("📊 Results")

            if "extract_result" not in st.session_state:
                st.caption("Upload a product and click Extract")
            else:
                data = st.session_state["extract_result"]
                result = data["result"]
                mode_key = data["mode"]

                # Taxonomy + Product Class (all modes except VLM-only)
                if mode_key in ("classifier", "hybrid"):
                    ms = result.get("latency_ms", data.get("cls_ms", "?"))
                    epoch = result.get("model_epoch", "?")
                    st.caption(f"⚡ {ms}ms | Epoch {epoch}")

                    tax_edited = show_taxonomy_table(result, key_prefix="res_tax")
                    show_product_class(result)

                # Attributes
                if mode_key == "vlm":
                    st.caption(f"🔄 {data.get('ms', 0):.0f}ms")
                    sources = {a: "vlm" for a in ATTR_ORDER}
                    attr_edited = show_attributes_table(
                        result, sources, key_prefix="res_attr")
                    if data.get("raw"):
                        with st.expander("Raw VLM output"):
                            st.code(data["raw"], language="text")

                elif mode_key == "classifier":
                    attr_edited = show_attributes_table(
                        result, result.get("sources"),
                        key_prefix="res_attr")
                    n_vlm = len([f for f in result.get("vlm_needed", [])
                                 if f in ATTR_ORDER])
                    total = len([a for a in result.get("attributes", {}).values()
                                 if a and a.get("value")])
                    if n_vlm:
                        st.info(f"🔄 VLM recommended for "
                                f"**{n_vlm}/{total}** fields")
                    else:
                        st.success(f"✅ All {total} fields above threshold")

                elif mode_key == "hybrid":
                    attr_edited = show_attributes_table(
                        result, result.get("sources"),
                        key_prefix="res_attr")
                    cls_ms = data.get("cls_ms", 0)
                    vlm_ms = data.get("vlm_ms", 0)
                    sources = result.get("sources", {})
                    n_cls = sum(1 for v in sources.values()
                                if v == "classifier")
                    n_vlm = sum(1 for v in sources.values()
                                if v == "vlm")
                    st.success(
                        f"⚡ {n_cls} from classifier ({cls_ms:.0f}ms)"
                        f" + 🔄 {n_vlm} from VLM ({vlm_ms:.0f}ms)")

                # Export JSON
                if mode_key in ("classifier", "hybrid"):
                    export = build_export_json(tax_edited, attr_edited, result)
                else:
                    export = build_export_json(None, attr_edited, result)
                with st.expander("📋 Export JSON (editable above)"):
                    st.json(export)

    # ── TAB 2: SEARCH ──
    with tab2:
        st.markdown("**Bi-encoder → Cross-encoder → Attribute-boosted**")
        search_pipeline = load_search_pipeline()
        bm25 = load_bm25()
        if search_pipeline is None and bm25 is None:
            st.warning("Search not available. Run training scripts first.")
            return

        st.sidebar.header("🔎 Search")
        search_mode = st.sidebar.selectbox(
            "Approach",
            ["Full Pipeline", "Bi-Encoder + Cross-Encoder",
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
        for col, ex in zip(ex_cols, [
            "blue modern sofa", "rustic dining table",
            "metal bookshelf", "velvet accent chair",
            "marble coffee table", "outdoor patio set"]):
            with col:
                st.button(ex, key=f"ex_{ex}",
                          use_container_width=True,
                          on_click=set_q, args=(ex,))

        go = st.button("🔎 Search", type="primary",
                        use_container_width=True, key="sbtn")
        should = go or st.session_state.run_search
        st.session_state.run_search = False
        query = st.session_state.search_query

        if should and query:
            with st.spinner("Searching..."):
                t0 = time.time()
                if search_mode == "BM25 Baseline" and bm25:
                    results = bm25.search(query, top_k)
                elif search_pipeline:
                    stage_map = {
                        "Full Pipeline": [
                            "bi_encoder", "cross_encoder",
                            "attribute_boost"],
                        "Bi-Encoder + Cross-Encoder": [
                            "bi_encoder", "cross_encoder"],
                        "Bi-Encoder Only": ["bi_encoder"],
                    }
                    results = search_pipeline.search(
                        query, top_k,
                        stages=stage_map.get(search_mode, ["bi_encoder"]))
                else:
                    results = []
                lat = (time.time()-t0)*1000
            st.caption(f"{len(results)} results in {lat:.0f}ms")
            show_search_results(results, query)


if __name__ == "__main__":
    main()