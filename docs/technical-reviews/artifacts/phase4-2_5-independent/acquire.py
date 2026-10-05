"""Fresh 2.5 review: preserve current bytes, original sources and historical CPU evidence."""
from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
REV = "26f34ebb5d1e237611567697d2b3ea4d64669331"
TORCH = "5c4886908584029761b579af026dcfb627c84070"
def sha(raw):
    return hashlib.sha256(raw).hexdigest()
def save(path, raw):
    target = OUT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    return {"path": str(target.relative_to(ROOT)), "sha256": sha(raw), "bytes": len(raw)}
spec = importlib.util.spec_from_file_location("facts", ROOT / "docs/review-tools/section_facts.py")
facts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(facts)
body, whole, first = facts.original_section(ROOT / "course/chapters/02.md", "2.5")
metadata = {"section": save("original/section.md", body), "section_first_line": first,
            "chapter_sha256": sha(whole), "newline_policy": "original bytes, no newline normalization",
            "helper": save("original/section_facts.py", (ROOT / "docs/review-tools/section_facts.py").read_bytes()),
            "historical_revision": REV, "files": []}
for fence in facts.fences(body, first):
    metadata["files"].append(save("original/fence-1.py", fence["raw"]))
for name in ("rewrite-02-visible-window", "window_training"):
    metadata["files"].append(save(f"figures/{name}.svg", (ROOT / f"course/figures/{name}.svg").read_bytes()))
for path in ("tiny_perceptron/simple.py", "tiny_perceptron/data.py", "scripts/course_experiments/text.py",
             "scripts/course_experiments/common.py", "scripts/course_experiments/run.py"):
    raw = subprocess.run(["git", "show", f"{REV}:{path}"], cwd=ROOT, capture_output=True, check=True).stdout
    metadata["files"].append(save("historical-code/" + path, raw))
raw_result = json.loads((ROOT / "docs/course-experiments/results/simple_models.json").read_bytes())
metadata["files"].append(save("historical/simple_models.json", (ROOT / "docs/course-experiments/results/simple_models.json").read_bytes()))
original_dir = ROOT / "outputs/course-experiments/course-v1/simple_models"
expected_hashes = {item["path"]: item["sha256"] for item in raw_result["artifacts"]}
local_result_raw = (original_dir / "result.json").read_bytes()
local_result = json.loads(local_result_raw)
local_hashes = {item["path"]: item["sha256"] for item in local_result["artifacts"]}
metadata["files"].append(save("historical/existing-local-cpu-result.json", local_result_raw))
metadata["existing_local_cpu_revision"] = local_result["revision"]
for path in ("data/train.jsonl", "data/validation.jsonl", "data/test.jsonl", "data/manifest.json",
             "vocabulary.json", "mlp1.pt", "mlp3.pt", "mlp5.pt"):
    raw = (original_dir / path).read_bytes()
    assert sha(raw) == local_hashes[path], path
    if not path.endswith(".pt"):
        assert sha(raw) == expected_hashes[path], path
    if path.endswith(".pt"):
        metadata["files"].append({"path": str((original_dir / path).relative_to(ROOT)),
                                  "sha256": sha(raw), "bytes": len(raw), "snapshot_copied": False})
    else:
        metadata["files"].append(save("historical/" + path, raw))
metadata["checkpoint_scope"] = "Existing local CPU rerun at 5d60e35, not the published 26f34eb checkpoints; evaluation only, no retraining. Both raw receipts retained."
for path in ("tiny_perceptron/simple.py", "tiny_perceptron/data.py", "scripts/course_experiments/text.py"):
    raw = subprocess.run(["git", "show", f"{local_result['revision']}:{path}"], cwd=ROOT, capture_output=True, check=True).stdout
    metadata["files"].append(save("existing-local-code/" + path, raw))
for path in ("tiny_perceptron/simple.py", "tiny_perceptron/data.py", "scripts/train_simple.py",
             "scripts/course_experiments/text.py", "course/training.md"):
    metadata["files"].append(save("current-code/" + path, (ROOT / path).read_bytes()))
urls = [
    ("bengio03a.pdf", "https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf", "JMLR 3 (2003) 1137–1155; published February 2003"),
    ("python-3.13.5-stdtypes.rst", "https://raw.githubusercontent.com/python/cpython/v3.13.5/Doc/library/stdtypes.rst", "CPython v3.13.5"),
    ("pytorch-linear.py", f"https://raw.githubusercontent.com/pytorch/pytorch/{TORCH}/torch/nn/modules/linear.py", TORCH),
    ("pytorch-loss.py", f"https://raw.githubusercontent.com/pytorch/pytorch/{TORCH}/torch/nn/modules/loss.py", TORCH),
    ("pytorch-functional.py", f"https://raw.githubusercontent.com/pytorch/pytorch/{TORCH}/torch/nn/functional.py", TORCH),
]
receipts = []
for filename, url, version in urls:
    with urllib.request.urlopen(url, timeout=30) as response:
        raw = response.read()
        receipts.append({"url": url, "version": version, "accessed_on": "2026-10-05",
                         "http_status": response.status, **save("sources/" + filename, raw)})
(OUT / "source-receipts.json").write_text(json.dumps(receipts, ensure_ascii=False, indent=2) + "\n")
(OUT / "provenance.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
subprocess.run(["pdftotext", "-layout", str(OUT / "sources/bengio03a.pdf"), str(OUT / "sources/bengio03a.txt")], check=True)
print(json.dumps({"section_sha256": sha(body), "downloaded_originals": len(receipts),
                  "historical_data_hash_verified": True, "existing_local_cpu_checkpoint_hash_verified": True}, ensure_ascii=False))
