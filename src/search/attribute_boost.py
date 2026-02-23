"""Attribute-based relevance boosting using VLM-extracted attributes."""
from __future__ import annotations
import json, re
from pathlib import Path
from typing import Optional
from loguru import logger

COLOR_TERMS = {"red","blue","green","yellow","orange","purple","pink","black","white","gray","grey","brown","beige","ivory","cream","navy","teal","turquoise","gold","silver","bronze","copper","walnut","espresso","mahogany","cherry","oak","tan","charcoal"}
MATERIAL_TERMS = {"wood","wooden","metal","steel","iron","velvet","leather","fabric","cotton","polyester","linen","silk","glass","ceramic","marble","granite","stone","bamboo","rattan","wicker","plastic","resin","teak","pine","birch","upholstered","tufted"}
STYLE_TERMS = {"modern","contemporary","traditional","rustic","industrial","farmhouse","coastal","bohemian","scandinavian","mid-century","minimalist","vintage","retro","classic","transitional","glam","elegant","sleek"}
PRODUCT_TYPE_TERMS = {"sofa","couch","chair","table","desk","bed","dresser","bookshelf","bookcase","cabinet","nightstand","bench","ottoman","stool","rug","lamp","mirror","shelf","sectional","loveseat","recliner","futon","vanity","wardrobe"}

class AttributeBooster:
    def __init__(self, enriched_catalog_path=None, boost_weights=None):
        self.boost_weights = boost_weights or {"color_family": 0.15, "primary_material": 0.10, "style": 0.10, "product_type": 0.15}
        self.product_attributes = {}
        if enriched_catalog_path and Path(enriched_catalog_path).exists():
            self.load_catalog(enriched_catalog_path)

    def load_catalog(self, path):
        count = 0
        with open(path) as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                pid = item.get("product_id")
                attrs = item.get("attributes", {})
                if pid is not None and attrs:
                    self.product_attributes[int(pid)] = attrs
                    count += 1
        logger.info(f"Loaded attributes for {count} products")

    def parse_query_attributes(self, query):
        words = set(re.findall(r"\b\w+\b", query.lower()))
        parsed = {}
        cm = words & COLOR_TERMS
        if cm: parsed["color_family"] = sorted(cm)[0]
        mm = words & MATERIAL_TERMS
        if mm: parsed["primary_material"] = sorted(mm)[0]
        sm = words & STYLE_TERMS
        if sm: parsed["style"] = sorted(sm)[0]
        tm = words & PRODUCT_TYPE_TERMS
        if tm: parsed["product_type"] = sorted(tm)[0]
        return parsed

    def compute_boost(self, product_id, query_attrs):
        if not query_attrs: return 0.0
        product_attrs = self.product_attributes.get(product_id, {})
        if not product_attrs: return 0.0
        boost = 0.0
        for attr_key, weight in self.boost_weights.items():
            qv = query_attrs.get(attr_key, "")
            pv = str(product_attrs.get(attr_key, "") or "").lower()
            if not qv or not pv: continue
            if qv.lower() == pv or qv.lower() in pv:
                boost += weight
            elif any(qv.lower() in part.strip() for part in pv.split("&")):
                boost += weight * 0.7
        return boost

    def boost_results(self, query, candidates, alpha=0.3):
        query_attrs = self.parse_query_attributes(query)
        if not query_attrs: return candidates
        for c in candidates:
            pid = c.get("product_id")
            boost = self.compute_boost(pid, query_attrs)
            base_score = c.get("ce_score", c.get("score", 0))
            norm_base = max(0, min(1, (base_score + 1) / 2))
            c["attribute_boost"] = round(boost, 4)
            c["query_attributes"] = query_attrs
            c["product_attributes"] = self.product_attributes.get(pid, {})
            c["boosted_score"] = round((1 - alpha) * norm_base + alpha * (boost / max(sum(self.boost_weights.values()), 0.01)), 4)
        candidates.sort(key=lambda x: x.get("boosted_score", 0), reverse=True)
        for i, c in enumerate(candidates):
            c["final_rank"] = i + 1
        return candidates
