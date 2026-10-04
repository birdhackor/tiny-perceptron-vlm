import json
from pathlib import Path

for kind in ("toy-text", "attributes-sft"):
    directory = Path("data/generated") / kind
    manifest = json.loads((directory / "manifest.json").read_text())
    print(kind, "seed", manifest["seed"], "切分單位", manifest["split_unit"])
    for split, stats in manifest["splits"].items():
        print(split, "筆數", stats["records"], "家族", stats["families"], "sha256", stats["sha256"])
        print((directory / f"{split}.jsonl").read_text().splitlines()[0])
