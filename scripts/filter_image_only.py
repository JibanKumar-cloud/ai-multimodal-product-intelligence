"""Filter dataset to only image+text examples for multimodal training."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for split in ["train", "val", "test"]:
    src = Path(f"data/processed/{split}.jsonl")
    dst = Path(f"data/processed/{split}_multimodal.jsonl")
    if not src.exists():
        print(f"Skipping {split}: {src} not found")
        continue
    total, kept = 0, 0
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            total += 1
            ex = json.loads(line)
            if ex.get("image_path") and Path(ex["image_path"]).exists():
                fout.write(line)
                kept += 1
    print(f"{split}: {kept}/{total} examples with images -> {dst}")
