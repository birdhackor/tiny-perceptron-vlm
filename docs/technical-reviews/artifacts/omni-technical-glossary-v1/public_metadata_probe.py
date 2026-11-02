"""Read only public small manifests and one pinned model card; no weights."""

import hashlib
import json
import platform
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
CACHE = Path("/workspace/work/reference/technical-glossary-own")
targets = [
    ("a-inference-manifest.json", "https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/979cdfacc588ad0536f1c64fff96f264571cf054/selftrained/v2/moe-joint/inference-manifest.json", "f27856892b0c46ca81ba43bfadf7051ed6bd31d848e4868a05147e5799df1e5e"),
    ("b-artifact-manifest.json", "https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/2c6a1062f3e66a602cf83eda6c703456ec18bca0/lesson-products/v1-20261016/b/artifact-manifest.json", "dd077fd7a0f11fa5d95bdcf76a73bf6e035819ab135fa3191ff94f4e3643a1a3"),
    ("c-artifact-manifest.json", "https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/2c6a1062f3e66a602cf83eda6c703456ec18bca0/lesson-products/v1-20261016/c/artifact-manifest.json", "90ec535d1b6ad1461c40b4ee329ebf047917c9eccf65dbbc7f879fbf5879d506"),
    ("b-training-receipt.json", "https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/2c6a1062f3e66a602cf83eda6c703456ec18bca0/lesson-products/v1-20261016/b/training-receipt.json", "f6a38366729bafa369fc04196027e2486e08bce0be16253a3844ee925be2508d"),
    ("c-training-receipt.json", "https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/2c6a1062f3e66a602cf83eda6c703456ec18bca0/lesson-products/v1-20261016/c/training-receipt.json", "14d6f050979b7c2b9361085ea1cb56e34445d1a1daa171d78241afc26750704a"),
    ("qwen3vl-2b-card.md", "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/resolve/89644892e4d85e24eaac8bacfd4f463576704203/README.md", None),
]
receipts = []
for name, url, expected in targets:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read()
            status = response.status
        (CACHE / name).write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        item = {"name": name, "url": url, "status": status, "bytes": len(data), "sha256": digest, "expected_sha256": expected, "digest_matches": digest == expected if expected else "no preexisting expected digest"}
        if name.endswith(".json"):
            value = json.loads(data)
            item["manifest_keys"] = sorted(value)
            if len(data) < 4000:
                item["small_manifest_content"] = value
            else:
                item["selected_original_fields"] = {key: value[key] for key in ["schema", "stage", "selected_step", "selected_checkpoint_sha256", "files", "random_initialization", "product", "mode", "steps_this_attempt", "selected_step", "stop_reason"] if key in value}
                if "origin" in value:
                    item["selected_original_fields"]["origin.kind"] = value["origin"].get("kind")
                    item["selected_original_fields"]["origin.seed"] = value["origin"].get("seed")
        else:
            item["short_excerpt"] = data.decode()[:1200]
        receipts.append(item)
    except Exception as exc:
        receipts.append({"name": name, "url": url, "error": str(exc)})
result = {"created_at": datetime.now(timezone.utc).isoformat(), "environment": {"python": platform.python_version(), "device": "no model loaded; HTTPS read only"}, "receipts": receipts}
(OUT / "public-metadata-output.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"created_at": result["created_at"], "receipts": [{key: item[key] for key in ["name", "status", "bytes", "sha256", "digest_matches", "selected_original_fields", "error"] if key in item} for item in receipts]}, ensure_ascii=False, indent=2))
