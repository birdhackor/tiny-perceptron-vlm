"""Read-only HTTPS verification of exact official source bytes; no models/datasets."""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

base = Path(__file__).parent
commit = "8cb5963cc22174954e7dca2c0a3320b7dc2f4edc"
files = {
    "hf-stopping-criteria-v4.57.1.py": "src/transformers/generation/stopping_criteria.py",
    "hf-generation-config-v4.57.1.py": "src/transformers/generation/configuration_utils.py",
    "hf-generation-utils-v4.57.1.py": "src/transformers/generation/utils.py",
    "hf-tokenization-base-v4.57.1.py": "src/transformers/tokenization_utils_base.py",
}
receipt = {"accessed_on": "2026-10-05", "python": sys.version,
    "HF_tag": "v4.57.1", "HF_tag_commit": commit, "HTTPS_TLS_verification": "urllib default context",
    "scope": "Official original paper and pinned library source only; no training artifacts, credentials or model weights requested.",
    "requests": []}
urls = [(name, "https://raw.githubusercontent.com/huggingface/transformers/" + commit + "/" + path)
        for name, path in files.items()]
urls += [("gpt2-original-paper.pdf", "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf")]
for name, url in urls:
    request = urllib.request.Request(url, headers={"User-Agent": "Codex-independent-factual-audit/7.9"})
    with urllib.request.urlopen(request, timeout=20) as response:
        data = response.read(1_500_001)
        assert len(data) <= 1_500_000
        assert data == (base / name).read_bytes(), "Pinned source differs from previously fetched original snapshot"
        receipt["requests"].append({"url": url, "final_url": response.url, "HTTP_status": response.status,
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "local_snapshot": name,
            "content_type": response.headers.get("Content-Type"), "etag": response.headers.get("ETag"),
            "verified_equal_to_saved_original_snapshot": True})
assert json.loads((base / "hf-tag-object.json").read_text())["object"]["sha"] == commit
receipt["tag_ref_sha256"] = hashlib.sha256((base / "hf-tag-ref.json").read_bytes()).hexdigest()
receipt["tag_object_sha256"] = hashlib.sha256((base / "hf-tag-object.json").read_bytes()).hexdigest()
print(json.dumps(receipt, ensure_ascii=False, indent=2))
