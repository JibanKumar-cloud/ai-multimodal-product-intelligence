"""Add product_classes to taxonomy_tree.json.

The product_class head needs a vocabulary of all unique product classes.
Run this once before training.

Usage:
    python scripts/build_class_vocab.py \
      --queue data/processed/image_queue.json \
      --taxonomy data/processed/taxonomy_tree.json
"""
import json
import argparse
from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", required=True)
    parser.add_argument("--taxonomy", required=True)
    args = parser.parse_args()

    # Load queue to get all product classes
    with open(args.queue) as f:
        products = json.load(f)

    class_counter = Counter()
    for p in products:
        pc = p.get("product_class")
        if pc:
            class_counter[pc] += 1

    classes = sorted(class_counter.keys())

    # Load taxonomy and add product_classes
    with open(args.taxonomy) as f:
        tax = json.load(f)

    tax["product_classes"] = classes

    with open(args.taxonomy, "w") as f:
        json.dump(tax, f, indent=2)

    print(f"Added {len(classes)} product classes to {args.taxonomy}")
    print(f"\nTop 20 classes:")
    for cls, count in class_counter.most_common(20):
        print(f"  {cls:40s}: {count}")


if __name__ == "__main__":
    main()