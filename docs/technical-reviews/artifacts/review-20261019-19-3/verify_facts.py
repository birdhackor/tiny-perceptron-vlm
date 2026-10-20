"""核對固定資料與讀檔邊界；完整下載內容只留在記憶體。"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import platform
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch
from PIL import Image
from tiny_perceptron.selftrained.dataset import read_records


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def conflicts(rows, key):
    registry = defaultdict(set)
    for row in rows:
        for value in key(row):
            registry[str(value)].add(row["split"])
    return {
        "unique_keys": len(registry),
        "cross_split_keys": sum(len(splits) > 1 for splits in registry.values()),
    }


def download(url):
    command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "180", url]
    completed = subprocess.run(command, capture_output=True, check=True)
    return completed.stdout, {"command": command, "bytes": len(completed.stdout), "sha256": sha(completed.stdout)}


def compact(value):
    if isinstance(value, list):
        return [compact(item) for item in value] if len(value) <= 6 else {"count": len(value), "first_two": [compact(item) for item in value[:2]]}
    if isinstance(value, dict):
        if len(value) > 30:
            return {"entry_count": len(value), "first_two": {k: compact(v) for k, v in list(value.items())[:2]}}
        return {k: compact(v) for k, v in value.items()}
    return value


def main():
    manifest = json.loads((ROOT / "docs/selftrained/v2-manifest.json").read_bytes())
    parent = json.loads((ROOT / "docs/selftrained/manifest.json").read_bytes())
    recipe = json.loads((ROOT / "docs/selftrained/v2-package-recipe.json").read_bytes())
    package = manifest["package"]
    url = f"https://media.githubusercontent.com/media/{package['repo_id']}/{package['revision']}/{package['path']}"
    raw, receipt = download(url)
    assert len(raw) == package["bytes"] and sha(raw) == package["sha256"]
    declarations = {item["path"]: item for item in manifest["records"] + manifest["assets"]}
    contents, file_hashes = {}, {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        for member in archive:
            assert member.isfile() and member.name in declarations
            assert member.name not in file_hashes
            value = archive.extractfile(member).read()
            expected = declarations[member.name]
            assert len(value) == expected["bytes"] and sha(value) == expected["sha256"]
            file_hashes[member.name] = sha(value)
            if member.name.endswith(".jsonl") or member.name.startswith(("provenance/", "images/vision/")):
                contents[member.name] = value
    assert file_hashes.keys() == declarations.keys()
    by_file = {item["path"]: [json.loads(line) for line in contents[item["path"]].splitlines() if line.strip()]
               for item in manifest["records"]}
    rows = [row for values in by_file.values() for row in values]
    memory_paths = {str(ROOT / "verified-package-memory" / name): contents[name].decode("utf-8") for name in by_file}
    original_read_text = Path.read_text
    def read_text(path, *args, **kwargs):
        if str(path) in memory_paths:
            return memory_paths[str(path)]
        return original_read_text(path, *args, **kwargs)
    with patch.object(Path, "read_text", read_text):
        parsed = read_records(list(memory_paths))
    assert len(parsed) == len(rows)
    assert [r["id"] for r in parsed] == sorted(r["id"] for r in rows)
    split_counts = dict(Counter(row["split"] for row in rows))
    assert split_counts == {"train": 28876, "validation": 2435, "test": 3734}
    vision = [row for row in rows if row["task"].startswith("vision")]
    ocr = [row for row in rows if row["task"] == "ocr"]
    voice = [row for row in rows if row["task"].startswith("voice")]
    text = [row for row in rows if not row["task"].startswith(("vision", "voice")) and row["task"] != "ocr"]
    source_splits = conflicts(vision, lambda r: r["supervision"]["source_ids"])
    original_pixels = {}
    for row in vision:
        if row["task"] == "vision_clothing":
            with Image.open(io.BytesIO(contents[row["image"]])) as image:
                original_pixels[row["supervision"]["source_ids"][0]] = image.crop((14, 42, 42, 70)).tobytes()
    pair_pixels_match = True
    for row in vision:
        if row["task"] == "vision_relation":
            with Image.open(io.BytesIO(contents[row["image"]])) as image:
                for box, source_id in zip(row["image_layout"]["slots"], row["supervision"]["source_ids"], strict=True):
                    x0, y0, x1, y1 = box
                    x, y = (x0 + x1 - 28) // 2, (y0 + y1 - 28) // 2
                    pair_pixels_match &= image.crop((x, y, x + 28, y + 28)).tobytes() == original_pixels[source_id]
    ocr_words = conflicts([r for r in ocr if len(r["messages"][-1]["content"]) > 1], lambda r: [r["messages"][-1]["content"]])
    schemas = {name: sorted(values[0]) for name, values in by_file.items()}
    voice_provenance = json.loads(contents["provenance/v2/voice-augmentation-v2.json"])
    ocr_provenance = json.loads(contents["provenance/v2/ocr-v2-augmentation.json"])
    text_provenance = json.loads(contents["provenance/v2/text-tools-v2-augmentation.json"])
    heldout_records = [item for item in parent["records"] if not item["path"].endswith("-train.jsonl")]
    heldout_match = all(file_hashes.get(item["path"]) == item["sha256"] for item in heldout_records)
    original_asset_match = all(file_hashes.get(item["path"]) == item["sha256"] for item in parent["assets"])
    producer_matches = {item["path"]: sha((ROOT / item["path"]).read_bytes()) == item["sha256"] for item in recipe["producers"]}
    original_media = [item for item in parent["assets"] if item["path"].startswith(("images/", "audio/"))]
    original_media_match = all(file_hashes.get(item["path"]) == item["sha256"] for item in original_media)
    asset_differences = [item["path"] for item in parent["assets"] if file_hashes.get(item["path"]) != item["sha256"]]
    report = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-19-3/verify_facts.py",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
        "package_receipt": receipt,
        "verified_file_count": len(file_hashes), "record_file_count": len(by_file),
        "split_counts": split_counts, "file_counts": {name: len(values) for name, values in by_file.items()},
        "read_records_all_files": {"parsed": len(parsed), "sorted": True, "io": "已核實壓縮包的十二份原始 UTF-8 內容，由 Path.read_text 的記憶體替身提供；實體檔案 I/O 另由下方小例驗證。"},
        "group_isolation": conflicts(rows, lambda r: [r["group_id"]]),
        "vision_source_isolation": source_splits,
        "vision_original_pixel_isolation": conflicts(vision, lambda r: [sha(original_pixels[s]) for s in r["supervision"]["source_ids"]]),
        "vision_pair_pixels_equal_declared_sources": pair_pixels_match,
        "vision_groups_per_source": dict(Counter(len({r["group_id"] for r in vision if source in r["supervision"]["source_ids"]}) for source in {s for r in vision for s in r["supervision"]["source_ids"]})),
        "ocr_composition_isolation": ocr_words,
        "ocr_single_characters": {s: "".join(sorted({r["messages"][-1]["content"] for r in ocr if r["split"] == s and len(r["messages"][-1]["content"]) == 1})) for s in split_counts},
        "ocr_fonts": {s: dict(Counter(r["supervision"]["font_family"] for r in ocr if r["split"] == s)) for s in split_counts},
        "ocr_lengths": {s: dict(Counter(len(r["messages"][-1]["content"]) for r in ocr if r["split"] == s)) for s in split_counts},
        "voice_audio_counts_including_derivatives": {s: len({r["audio"] for r in voice if r["split"] == s}) for s in split_counts},
        "voice_original_recording_counts": {s: len({r["audio"] for r in voice if r["split"] == s and r["audio"] in {a["path"] for a in parent["assets"]}}) for s in split_counts},
        "voice_group_isolation": conflicts(voice, lambda r: [r["group_id"]]),
        "text_group_isolation": conflicts(text, lambda r: [r["group_id"]]),
        "text_template_isolation": conflicts(text, lambda r: [r.get("supervision", {}).get("template_family", "absent")]),
        "text_operand_isolation": conflicts(text, lambda r: [r["supervision"]["operand_family"]] if "operand_family" in r.get("supervision", {}) else []),
        "heldout_record_hashes_equal_parent": heldout_match,
        "heldout_record_file_count": len(heldout_records),
        "all_original_asset_hashes_equal_parent": original_asset_match,
        "all_original_media_hashes_equal_parent": original_media_match,
        "original_media_count": len(original_media),
        "changed_original_asset_paths": asset_differences,
        "original_asset_count": len(parent["assets"]),
        "producer_hashes_match_recipe": producer_matches,
        "record_schemas": schemas,
        "provenance": {"voice": compact(voice_provenance), "ocr": compact(ocr_provenance), "text_tools": compact(text_provenance)},
    }
    probes = []
    def record(identifier, group, split, answer="左邊是包。"):
        return {"id": identifier, "group_id": group, "split": split, "task": "vision_relation", "messages": [{"role": "user", "content": "左邊是什麼？"}, {"role": "assistant", "content": answer}]}
    base = [record("r1", "bag-boot-07", "train"), record("r2", "bag-boot-07", "test", "左邊是短靴。")]
    variants = [("same_group_cross_split", base)]
    changed = copy.deepcopy(base); changed[1]["group_id"] = "bag-boot-08"
    variants.append(("different_group_cross_split", changed))
    alias = [record("r2", "g2", "val"), record("r1", "g1", "train")]
    variants.append(("validation_alias_and_sort", alias))
    duplicate = copy.deepcopy(base); duplicate[1]["id"] = "r1"
    variants.append(("duplicate_id", duplicate))
    missing = [record("r1", "g1", "train")]; del missing[0]["task"]
    variants.append(("missing_required", missing))
    variants.append(("unknown_split", [record("r1", "g1", "dev")]))
    bad_role = [record("r1", "g1", "train")]; bad_role[0]["messages"][0]["role"] = "unknown"
    variants.append(("invalid_message_role", bad_role))
    bad_content = [record("r1", "g1", "train")]; bad_content[0]["messages"][0]["content"] = 1
    variants.append(("nontext_content", bad_content))
    hidden = [record("r1", "g1", "train")]; hidden[0]["messages"][0]["intent"] = "answer"
    variants.append(("nonpublic_message_field", hidden))
    with tempfile.TemporaryDirectory(dir=ARTIFACT) as directory:
        path = Path(directory) / "records.jsonl"
        for name, values in variants:
            path.write_text("\n".join(json.dumps(v, ensure_ascii=False) for v in values), encoding="utf-8")
            try:
                result = read_records([path])
                probes.append({"case": name, "result": "accepted", "count": len(result), "ids": [r["id"] for r in result], "splits": [r["split"] for r in result]})
            except ValueError as error:
                message = str(error).replace(str(path), "records.jsonl")
                probes.append({"case": name, "result": "rejected", "error": message})
    report["read_records_probes"] = probes
    out = ARTIFACT / "verified-facts.json"
    with out.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2); handle.write("\n")
    print(json.dumps({"result": "completed", "artifact": str(out.relative_to(ROOT)), "split_counts": split_counts, "verified_files": len(file_hashes), "probes": probes}, ensure_ascii=False))


if __name__ == "__main__":
    main()
