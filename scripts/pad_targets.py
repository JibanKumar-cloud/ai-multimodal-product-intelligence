#!/usr/bin/env python3
"""Pad all target_attributes to include all 7 keys with null defaults."""
import json
from pathlib import Path

ALL_KEYS = [
    "style", "primary_material", "secondary_material",
    "color_family", "room_type", "product_type", "assembly_required"
]

for split in ["train", "val", "test", "train_multimodal", "val_multimodal",
              "test_multimodal", "train_vague", "val_vague", "test_vague"]:
    path = Path(f"data/processed/{split}.jsonl")
    if not path.exists():
        continue
    lines = []
    with open(path) as f:
        for line in f:
            if not line.strip(): continue
            ex = json.loads(line)
            target = ex.get("target_attributes", {})
            ex["target_attributes"] = {k: target.get(k, None) for k in ALL_KEYS}
            lines.append(json.dumps(ex, default=str))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{split}: padded {len(lines)} examples with all 7 keys")
