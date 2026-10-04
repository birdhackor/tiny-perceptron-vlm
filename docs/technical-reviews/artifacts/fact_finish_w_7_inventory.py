"""Record repository directory and existing generated-page metadata without writes."""

import hashlib
import json
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main():
    result = {"environment": {"python": platform.python_version(), "device": "cpu"}, "directories": {}}
    for relative in ("course/chapters", "notebooks", "tiny_perceptron", "scripts", "data", "checkpoints", "outputs"):
        directory = ROOT / relative
        result["directories"][relative] = {
            "exists": directory.is_dir(),
            "immediate_entries": len(list(directory.iterdir())),
        }
    result["generated_pages"] = {}
    for relative in ("outputs/site/index.html", "outputs/site/1.1.html"):
        path = ROOT / relative
        raw = path.read_bytes()
        result["generated_pages"][relative] = {
            "exists": path.is_file(),
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "html_doctype_in_first_256_bytes": b"<!doctype html" in raw[:256].lower(),
        }
    result["scope"] = (
        "Metadata only; existing generated-page content, model quality, saved weights and exact resume were not audited."
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
