"""One-time fix: sanitize NaN values in processed data files."""
import json, math
from pathlib import Path

for f in Path("data/processed").glob("*.jsonl"):
    lines = []
    with open(f) as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            for k, v in d.items():
                if isinstance(v, float) and math.isnan(v):
                    d[k] = None
            lines.append(json.dumps(d, default=str))
    with open(f, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Fixed {f}")
