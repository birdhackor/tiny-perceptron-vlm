"""Package explicit small evidence files, preserving and verifying exact bytes."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ORIGINALS = ROOT / "outputs/natural-extension/recipe-diagnosis"
PUBLIC = ROOT / "docs/natural-assistant/evidence/recipe-diagnosis"
digest = lambda data: hashlib.sha256(data).hexdigest()
manifest_path = ROOT / "docs/natural-assistant/manifest.json"
manifest_sha = digest(manifest_path.read_bytes())
assert manifest_sha == "7604526c67da31a41940f16f9c87027c73cf2782c63522dbde8c7efbf015db91"

def write_new_or_identical(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_bytes() == content, f"Preserve existing different file: {path}"
    else:
        path.write_bytes(content)

historical = json.loads((ORIGINALS / "analysis-receipt.json").read_text())
historical_recommendation = next(a for a in historical["artifacts"] if a["path"].endswith("/RECOMMENDATION.md"))
old_sha = digest((ORIGINALS / "RECOMMENDATION.before-caption-boundary-correction.md").read_bytes())
assert old_sha == historical_recommendation["sha256"]
current_sha = digest((ORIGINALS / "RECOMMENDATION.md").read_bytes())
assert old_sha != current_sha
# Verify already-downloaded official metadata against its original receipts, no network refresh.
official = json.loads((ORIGINALS / "official-source-receipts.json").read_text())
for item in official["files"]:
    data = (ORIGINALS / item["path"]).read_bytes()
    assert digest(data) == item["sha256"] and len(data) == item["bytes"]
docci = json.loads((ORIGINALS / "docci-source-analysis.json").read_text())
for item in docci["sources"]:
    assert digest((ROOT / item["local_path"]).read_bytes()) == item["verified_sha256"] == item["sha256"]
boundary = json.loads((ORIGINALS / "caption-boundary-receipt.json").read_text())
assert boundary["manifest_sha256"] == manifest_sha and boundary["manifest_unchanged"]
assert boundary["computed_target_equals_frozen_target"]

publication = {
    "observed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "scope": "Publish exact copies of existing diagnosis, CPU scripts and pinned official small sources; no new model-version selection, network refresh, inference, GPU or test review",
    "manifest_sha256_before": manifest_sha,
    "caption_correction": {
        "row_id": boundary["row_id"], "wording": boundary["corrected_wording"],
        "previous_recommendation_sha256": old_sha, "current_recommendation_sha256": current_sha,
        "historical_receipt_old_recommendation_verified": True,
        "frozen_target_unchanged": True,
    },
    "official_metadata_receipt_checks_passed": len(official["files"]),
    "docci_document_source_receipt_checks_passed": len(docci["sources"]),
    "weights_full_corpus_images_large_tsv_copied": False,
    "test_evaluated_or_reviewed": False,
}
publication_path = ORIGINALS / "publication-receipt.json"
if publication_path.exists():
    publication = json.loads(publication_path.read_text())
    assert publication["caption_correction"]["current_recommendation_sha256"] == current_sha
else:
    publication_path.write_text(json.dumps(publication, ensure_ascii=False, indent=2) + "\n")

mapping = [
    ("PUBLIC-EVIDENCE.md", "README.md"),
    ("RECOMMENDATION.md", "RECOMMENDATION.md"),
    ("RECOMMENDATION.before-caption-boundary-correction.md", "history/RECOMMENDATION.before-caption-boundary-correction.md"),
    ("training-and-output-analysis.json", "analysis/training-and-output-analysis.json"),
    ("meta-4b-analysis.json", "analysis/meta-4b-analysis.json"),
    ("docci-source-analysis.json", "analysis/docci-source-analysis.json"),
    ("analysis-receipt.json", "receipts/analysis-receipt.json"),
    ("official-source-receipts.json", "receipts/official-source-receipts.json"),
    ("caption-boundary-receipt.json", "receipts/caption-boundary-receipt.json"),
    ("publication-receipt.json", "receipts/publication-receipt.json"),
    ("analyze_training.py", "scripts/analyze_training.py"),
    ("meta_count_4b.py", "scripts/meta_count_4b.py"),
    ("verify_caption_boundary.py", "scripts/verify_caption_boundary.py"),
    ("package_evidence.py", "scripts/package_evidence.py"),
    ("training-analysis.stdout.txt", "logs/training-analysis.stdout.txt"),
    ("meta-4b.stdout.txt", "logs/meta-4b.stdout.txt"),
    ("train_03125-original-line.jsonl", "originals/docci/train_03125-original-line.jsonl"),
]
for name in ["README.md", "config.json", "model.safetensors.index.json", "preprocessor_config.json",
             "generation_config.json", "official-4b-api.json"]:
    mapping.append((name, "originals/qwen-4b/" + name))
entries = []
for source_name, output_name in mapping:
    source = ORIGINALS / source_name
    entries.append((source, PUBLIC / output_name))
for name in ["sources/README-docci.md", "sources/docci-website.html", "LICENSE-DOCCI.txt"]:
    entries.append((ROOT / "outputs/natural-extension/data/vision" / name,
                    PUBLIC / "originals/docci" / Path(name).name))

files = []
for source, destination in entries:
    content = source.read_bytes()
    assert len(content) < 1024 * 1024, source
    assert source.suffix not in [".safetensors", ".pt", ".pth", ".bin", ".tsv"]
    write_new_or_identical(destination, content)
    copied = destination.read_bytes()
    assert copied == source.read_bytes()
    if destination.suffix == ".json":
        json.loads(copied)
    files.append({
        "source": str(source.relative_to(ROOT)), "output": str(destination.relative_to(ROOT)),
        "bytes": len(content), "source_sha256": digest(content), "copy_sha256": digest(copied),
        "sha256": digest(copied), "byte_identical": True,
    })
assert digest(manifest_path.read_bytes()) == manifest_sha
index = {
    "schema_version": 1,
    "scope": "Exact byte copies of small existing diagnosis artifacts, scripts, receipts and pinned official sources. Corrections are explicit; frozen dataset/model code unchanged; no model weights/full corpus/images/large TSV; no test review or new version decision.",
    "dataset_manifest_sha256": manifest_sha,
    "qwen4b_official_revision": official["revision"],
    "file_count": len(files), "total_bytes": sum(f["bytes"] for f in files),
    "all_source_copy_sha256_equal": all(f["source_sha256"] == f["copy_sha256"] for f in files),
    "historical_receipt_note": "The old analysis receipt's RECOMMENDATION artifact hash matches the archived history file. The corrected current RECOMMENDATION hash is recorded in publication receipt and this index.",
    "files": files,
}
write_new_or_identical(PUBLIC / "copy-index.json", (json.dumps(index, ensure_ascii=False, indent=2) + "\n").encode())
print(json.dumps({k: index[k] for k in ["file_count", "total_bytes", "all_source_copy_sha256_equal",
    "dataset_manifest_sha256", "qwen4b_official_revision"]}, indent=2))
