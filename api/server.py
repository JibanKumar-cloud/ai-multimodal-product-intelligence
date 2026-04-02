"""FastAPI backend for Product Intelligence.

Endpoints:
    POST /predict/classifier   — fast multi-tower (~50ms)
    POST /predict/vlm          — LLaVA QLoRA (~2-5s)
    POST /predict/hybrid       — classifier + VLM fallback
    GET  /config               — available models and vocab
    POST /search               — product search

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import sys
import time
import json
import base64
import io
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))

app = FastAPI(title="Wayfair Product Intelligence API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    """Serve the React frontend."""
    html_path = Path(__file__).parent.parent / "frontend" / "index.html"
    if html_path.exists():
        return html_path.read_text()
    return HTMLResponse("<h1>Frontend not found</h1><p>Place index.html in frontend/</p>")

# ── Config ──

CLASSIFIER_PATHS = {
    "checkpoint": "checkpoints/best_model.pt",
    "taxonomy": "data/processed/taxonomy_tree.json",
    "vocab": "data/processed/attribute_vocab.json",
    "queue": "data/processed/image_queue_with_images.json",
}

LLAVA_ADAPTERS = {
    "multimodal": "outputs/checkpoints/qlora-multimodal/best_model",
    "text_only": "outputs/checkpoints/qlora-text-only/best_model",
    "vague_multimodal": "outputs/checkpoints/qlora-vague-multimodal/best_model",
}

# ── Lazy pipeline ──

_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from src.inference.pipeline import ProductPipeline
        available = {k: v for k, v in LLAVA_ADAPTERS.items() if Path(v).exists()}
        adapter = list(available.values())[0] if available else None
        _pipeline = ProductPipeline(
            classifier_checkpoint=CLASSIFIER_PATHS["checkpoint"],
            taxonomy_path=CLASSIFIER_PATHS["taxonomy"],
            vocab_path=CLASSIFIER_PATHS["vocab"],
            queue_path=CLASSIFIER_PATHS["queue"],
            vlm_adapter_path=adapter,
        )
    return _pipeline


# ── Schemas ──

class PredictRequest(BaseModel):
    product_name: str = ""
    product_class: str = ""
    description: str = ""
    confidence_threshold: float = 0.5
    image_base64: Optional[str] = None  # base64 encoded image


def _decode_image(b64: Optional[str]):
    if not b64:
        return None
    from PIL import Image
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")


# ── Endpoints ──

@app.get("/config")
def config():
    """Return available models and vocab."""
    from src.inference.postprocessor import (
        COLOR_FAMILIES, MATERIAL_GROUPS, STYLE_GROUPS,
        SHAPE_GROUPS, ASSEMBLY_GROUPS)

    available_adapters = {k: v for k, v in LLAVA_ADAPTERS.items()
                         if Path(v).exists()}

    return {
        "has_classifier": Path(CLASSIFIER_PATHS["checkpoint"]).exists(),
        "vlm_adapters": list(available_adapters.keys()),
        "vocab": {
            "primary_color": sorted(COLOR_FAMILIES.keys()),
            "secondary_color": sorted(COLOR_FAMILIES.keys()),
            "primary_material": sorted(MATERIAL_GROUPS.keys()),
            "secondary_material": sorted(MATERIAL_GROUPS.keys()),
            "style": sorted(STYLE_GROUPS.keys()),
            "shape": sorted(SHAPE_GROUPS.keys()),
            "assembly": sorted(ASSEMBLY_GROUPS.keys()),
        },
    }


@app.post("/predict/classifier")
def predict_classifier(req: PredictRequest):
    pipeline = get_pipeline()
    image = _decode_image(req.image_base64)
    t0 = time.time()
    result = pipeline.classifier_predict(
        req.product_name, req.product_class,
        req.description, image, req.confidence_threshold)
    result["total_ms"] = round((time.time() - t0) * 1000, 1)
    return result


@app.post("/predict/vlm")
def predict_vlm(req: PredictRequest):
    pipeline = get_pipeline()
    image = _decode_image(req.image_base64)
    t0 = time.time()
    result, raw = pipeline.vlm_predict(
        req.product_name, req.product_class,
        req.description, image)
    ms = round((time.time() - t0) * 1000, 1)
    result["raw_output"] = raw
    result["total_ms"] = ms
    return result


@app.post("/predict/hybrid")
def predict_hybrid(req: PredictRequest):
    """Run hybrid: returns classifier immediately, then VLM."""
    pipeline = get_pipeline()
    image = _decode_image(req.image_base64)

    # Step 1: Classifier
    cls_result = pipeline.classifier_predict(
        req.product_name, req.product_class,
        req.description, image, req.confidence_threshold)

    from src.inference.postprocessor import ATTR_KEYS
    vlm_attrs = [f for f in cls_result.get("vlm_needed", [])
                 if f in ATTR_KEYS]

    if not vlm_attrs or not pipeline.has_vlm:
        cls_result["vlm_attrs"] = []
        cls_result["mode"] = "classifier_only"
        return cls_result

    # Step 2: VLM for low-confidence
    if image:
        # Re-create image for VLM
        image = _decode_image(req.image_base64)

    merged, cls_ms, vlm_ms, _ = pipeline.hybrid_predict(
        req.product_name, req.product_class,
        req.description, image, req.confidence_threshold)

    merged["cls_ms"] = cls_ms
    merged["vlm_ms"] = vlm_ms
    merged["vlm_attrs"] = vlm_attrs
    merged["mode"] = "hybrid"
    return merged


@app.post("/predict/hybrid/stream")
async def predict_hybrid_stream(req: PredictRequest):
    """SSE stream: classifier result first, then VLM update."""
    import asyncio

    async def event_stream():
        pipeline = get_pipeline()
        image = _decode_image(req.image_base64)

        # Step 1: Classifier (instant)
        cls_result = pipeline.classifier_predict(
            req.product_name, req.product_class,
            req.description, image, req.confidence_threshold)

        from src.inference.postprocessor import ATTR_KEYS
        vlm_attrs = [f for f in cls_result.get("vlm_needed", [])
                     if f in ATTR_KEYS]

        cls_result["vlm_attrs"] = vlm_attrs
        cls_result["step"] = "classifier"

        yield f"data: {json.dumps(cls_result, default=str)}\n\n"

        if vlm_attrs and pipeline.has_vlm:
            # Step 2: VLM
            image2 = _decode_image(req.image_base64)
            merged, cls_ms, vlm_ms, _ = pipeline.hybrid_predict(
                req.product_name, req.product_class,
                req.description, image2, req.confidence_threshold)

            merged["cls_ms"] = cls_ms
            merged["vlm_ms"] = vlm_ms
            merged["vlm_attrs"] = vlm_attrs
            merged["step"] = "vlm_complete"

            yield f"data: {json.dumps(merged, default=str)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)