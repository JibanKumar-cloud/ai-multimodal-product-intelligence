"""Taxonomy-to-ProductClass Lookup Table.

Builds a mapping from taxonomy paths to product_class.
Used post-prediction to derive product_class from taxonomy.

Build once from training data:
    python -m src.classifier.taxonomy_lookup \
      --queue data/processed/image_queue.json \
      --output data/processed/taxonomy_to_class.json

At inference:
    lookup = TaxonomyLookup("data/processed/taxonomy_to_class.json")
    product_class = lookup.resolve(
        predicted_taxonomy=["Furniture", "Living Room", "Cabinets", "Brown Cabinets"],
        predicted_color="brown"
    )
"""
import json
import argparse
from collections import defaultdict


class TaxonomyLookup:
    """Resolve product_class from predicted taxonomy path."""

    def __init__(self, lookup_path: str):
        with open(lookup_path) as f:
            data = json.load(f)

        # Full path -> list of product_classes
        self.full_path_map = data.get("full_path", {})
        # Partial paths (level_1/2/3) -> list of product_classes
        self.partial_maps = data.get("partial", {})
        # product_class -> set of valid taxonomy paths
        self.class_to_paths = data.get("class_to_paths", {})

    def resolve(self, predicted_taxonomy: list,
                predicted_attrs: dict = None) -> dict:
        """Resolve product_class from taxonomy + optional attributes.

        Args:
            predicted_taxonomy: ["Furniture", "Living Room", "Cabinets", ...]
            predicted_attrs: {"primary_color": "brown", ...} for disambiguation

        Returns:
            {
                "product_class": "Accent Chests / Cabinets",
                "confidence": "exact"|"partial"|"fallback",
                "taxonomy_path": ["Furniture", "Living Room", ...],
            }
        """
        # Try full path match first
        full_key = "/".join(predicted_taxonomy)
        if full_key in self.full_path_map:
            classes = self.full_path_map[full_key]
            if len(classes) == 1:
                return {
                    "product_class": classes[0],
                    "confidence": "exact",
                    "taxonomy_path": predicted_taxonomy,
                }
            # Multiple classes for same path — disambiguate with attributes
            best = self._disambiguate(classes, predicted_attrs)
            return {
                "product_class": best,
                "confidence": "exact",
                "taxonomy_path": predicted_taxonomy,
            }

        # Try progressively shorter paths
        for depth in range(len(predicted_taxonomy) - 1, 0, -1):
            partial_key = "/".join(predicted_taxonomy[:depth])
            level_key = f"depth_{depth}"
            partial = self.partial_maps.get(level_key, {})
            if partial_key in partial:
                classes = partial[partial_key]
                best = self._disambiguate(classes, predicted_attrs)
                return {
                    "product_class": best,
                    "confidence": "partial",
                    "taxonomy_path": predicted_taxonomy[:depth],
                }

        return {
            "product_class": None,
            "confidence": "fallback",
            "taxonomy_path": predicted_taxonomy,
        }

    def _disambiguate(self, classes: list,
                      predicted_attrs: dict = None) -> str:
        """Pick best class when multiple match.

        Uses predicted attributes to pick the most specific match.
        E.g., color=brown + classes=["Brown Cabinets", "Grey Cabinets"]
              -> "Brown Cabinets"
        """
        if len(classes) == 1:
            return classes[0]
        if not predicted_attrs:
            return classes[0]  # no info, pick first

        # Score each class by attribute overlap
        best_score = -1
        best_class = classes[0]

        color = predicted_attrs.get("primary_color", "").lower()
        material = predicted_attrs.get("primary_material", "").lower()
        style = predicted_attrs.get("style", "").lower()

        for cls in classes:
            cls_lower = cls.lower()
            score = 0
            if color and color in cls_lower:
                score += 3  # color in class name is strong signal
            if material and material in cls_lower:
                score += 2
            if style and style in cls_lower:
                score += 1
            if score > best_score:
                best_score = score
                best_class = cls

        return best_class

    def get_valid_classes_for_path(self, taxonomy: list) -> list:
        """Get all valid product_classes for a taxonomy prefix."""
        results = set()
        for depth in range(len(taxonomy), 0, -1):
            key = "/".join(taxonomy[:depth])
            if key in self.full_path_map:
                results.update(self.full_path_map[key])
            level_key = f"depth_{depth}"
            partial = self.partial_maps.get(level_key, {})
            if key in partial:
                results.update(partial[key])
        return sorted(results)


def build_lookup(queue_path: str) -> dict:
    """Build lookup table from image_queue.json.

    Returns dict with:
      full_path: {"Furniture/Living Room/Cabinets/Brown Cabinets": ["Accent Chests"]}
      partial: {"depth_1": {"Furniture": ["Accent Chairs", ...]}, ...}
      class_to_paths: {"Accent Chairs": [["Furniture", "Living Room", ...]]}
    """
    with open(queue_path) as f:
        products = json.load(f)

    # Full path -> product_classes
    full_path_map = defaultdict(set)
    # Partial paths per depth
    partial_maps = defaultdict(lambda: defaultdict(set))
    # Class -> paths
    class_to_paths = defaultdict(list)

    for p in products:
        taxonomy = p.get("taxonomy", [])
        product_class = p.get("product_class")
        if not taxonomy or not product_class:
            continue

        # Full path
        full_key = "/".join(taxonomy)
        full_path_map[full_key].add(product_class)

        # Partial paths at each depth
        for depth in range(1, len(taxonomy) + 1):
            partial_key = "/".join(taxonomy[:depth])
            partial_maps[f"depth_{depth}"][partial_key].add(product_class)

        # Reverse: class -> paths
        class_to_paths[product_class].append(taxonomy)

    # Convert sets to sorted lists for JSON
    result = {
        "full_path": {k: sorted(v) for k, v in full_path_map.items()},
        "partial": {
            depth: {k: sorted(v) for k, v in paths.items()}
            for depth, paths in partial_maps.items()
        },
        "class_to_paths": {
            k: v for k, v in class_to_paths.items()
        },
    }

    # Stats
    n_full = len(result["full_path"])
    n_classes = len(result["class_to_paths"])
    n_ambiguous = sum(
        1 for v in result["full_path"].values() if len(v) > 1)
    print(f"Lookup table built:")
    print(f"  {n_full} unique taxonomy paths")
    print(f"  {n_classes} unique product classes")
    print(f"  {n_ambiguous} ambiguous paths (multiple classes)")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    lookup = build_lookup(args.queue)
    with open(args.output, "w") as f:
        json.dump(lookup, f, indent=2, default=str)
    print(f"Saved to {args.output}")