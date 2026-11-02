"""Bounded byte/provenance recheck for the 20.14 i2/i3 corrections; no inference."""

import datetime
import hashlib
import json
import pathlib
import platform
import re
import sys


ROOT = pathlib.Path(__file__).resolve().parents[4]
ORIGINALS = pathlib.Path("/workspace/work/reference/omni-P5-20.14-originals")
ARTIFACTS = ROOT / "docs/technical-reviews/artifacts/omni-P5-20.14"
FREEZE = "docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v10.json"
CORRECTION = "docs/course-revision-20261029-omni/source-digest-corrections-v1.json"
OUTPUT = ARTIFACTS / "source10-correction-recheck-v1.json"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def ref(path):
    data = path.read_bytes()
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": len(data), "sha256": sha(data)}


def main():
    started = datetime.datetime.now(datetime.UTC).isoformat()
    assert not OUTPUT.exists(), "immutable output already exists"
    freeze = ROOT / FREEZE
    assert sha(freeze.read_bytes()) == "ddcc500db74f5df9ab3bb6fddcdc511ca177c98ce64e1a94160d1a7ccc1c35be"
    page = next(p for p in json.loads(freeze.read_bytes())["pages"] if p["page_id"] == "20.14")
    current = (ROOT / page["snapshot"]).read_bytes()
    assert sha(current) == page["source_sha256"] == "83405745e904fc54276141230e5b4dae751b37bf6628eafb34c29ee4fa2c6bb4"
    original = (ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/source-freeze-v1/sources/20.14.md").read_bytes()
    old = "其一般評測會逐題重算材料，服務路徑才共用prefix"
    new = "其一般Torch短列評測會逐題重算材料，服務路徑會共用prefix"
    assert original.count(old.encode()) == 1
    assert original.replace(old.encode(), new.encode(), 1) == current
    authored = (ROOT / "course/chapters/20.md").read_text()
    current_slice = re.search(r"^## 20\.14 .*?(?=^## |\Z)", authored, re.M | re.S)[0]
    assert sha(current_slice.encode()) == page["source_sha256"]
    assert not page["figures_sha256"]
    old_code = re.findall(r"^```python\n(.*?)^```", original.decode(), re.M | re.S)
    new_code = re.findall(r"^```python\n(.*?)^```", current.decode(), re.M | re.S)
    assert len(old_code) == len(new_code) == 1 and old_code == new_code
    execution_path = ARTIFACTS / "inline-cpu-execution-v1.json"
    assert sha(execution_path.read_bytes()) == "974c11e44a6f1cd24c8ef35504c37ff49c630b0c6925d3a9187b4fb192ca5cde"
    execution = json.loads(execution_path.read_bytes())
    assert sha(new_code[0].encode()) == execution["cases"][0]["code_sha256"]
    # This exact marker is the sole change assigned by the textbook exercise.
    variant = new_code[0].replace('("資訊足夠", 5)', '("資訊足夠", 1)', 1)
    assert sha(variant.encode()) == execution["cases"][1]["code_sha256"]
    assert all(c["all_assertions_passed"] for c in execution["cases"])
    notebook = ROOT / page["notebook"]
    assert sha(notebook.read_bytes()) == page["notebook_sha256"]
    correction_path = ROOT / CORRECTION
    assert sha(correction_path.read_bytes()) == "ff470dff6e2e7bcf56bd70c7935afd54d2b3a927e0783b4914a77c1e7c5ef7cf"
    correction = json.loads(correction_path.read_bytes())
    source_record = ROOT / correction["original_source_record"]["path"]
    assert sha(source_record.read_bytes()) == correction["original_source_record"]["sha256"]
    record = json.loads(source_record.read_bytes())
    tree = json.loads((ORIGINALS / "kev-git-tree.json").read_bytes())
    acquisition = json.loads((ORIGINALS / "acquisition-receipt-v1.json").read_bytes())
    names = {"kev-model": ("kev-model.py", "kev/model.py"), "kev-data": ("kev-data.py", "kev/data.py"), "kev-readme": ("kev-readme.md", "README.md")}
    assert {c["source_id"] for c in correction["corrections"]} == set(names)
    observations = []
    for item in correction["corrections"]:
        name, git_path = names[item["source_id"]]
        raw = (ORIGINALS / name).read_bytes()
        receipt = next(a for a in acquisition if a["name"] == name)
        original_entry = next(s for s in record["sources"] if s["id"] == item["source_id"])
        tree_entry = next(n for n in tree["tree"] if n["path"] == git_path)
        blob_sha1 = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        assert item["raw_url"] == receipt["url"]
        assert item["raw_upstream_bytes"] == len(raw) == receipt["bytes"]
        assert item["raw_upstream_sha256"] == sha(raw) == receipt["sha256"]
        assert item["original_recorded_bytes"] == len(raw) + 1 == original_entry["bytes"]
        assert item["original_recorded_sha256"] == sha(raw + b"\n") == original_entry["sha256"]
        assert item["version"] == original_entry["version"] == "2074cc44256200bb37540103fa4572901faa10f3"
        assert blob_sha1 == item["raw_git_blob_sha1"] == item["git_tree_blob_sha1"] == tree_entry["sha"]
        observations.append({"source_id": item["source_id"], "raw_url": item["raw_url"], "version": item["version"], "raw_bytes": len(raw), "raw_sha256": sha(raw), "raw_plus_lf_bytes": len(raw) + 1, "raw_plus_lf_sha256": sha(raw + b"\n"), "git_blob_sha1": blob_sha1, "matches_correction_and_original_record": True})
    sources_path = ROOT / "docs/course-revision-20261029-omni/SOURCES.md"
    errata = next(p for p in sources_path.read_text().split("\n\n") if p.startswith("來源位元組勘誤："))
    assert "(source-digest-corrections-v1.json)" in errata
    assert all(f"`{key}`" in errata for key in names)
    model = (ORIGINALS / "kev-model.py").read_bytes()
    predictors = (ORIGINALS / "kev-predictors.py").read_bytes()
    assert sha(model) == "2ff3b2e67c048e3bc5a0d1a32aafe50804ed315a5564fce06bb4b95697c5271c"
    assert sha(predictors) == "29263ae3297e5cde6cf03da5c5aa3ed377c9681c6a5a639ea4aa259acec7575b"
    original_report = ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/technical-20.14/technical-original-v1.json"
    current1 = ROOT / "docs/course-revision-20261029-omni/reviews/P5-actual/technical-20.14/technical-current-v1.json"
    assert sha(original_report.read_bytes()) == "b2f0c331d604de4c5fd687ac3e37c28394701daf4c6e6217384c5e6292eb671b"
    assert sha(current1.read_bytes()) == "260cf92039317085bcfdd30a071348f8f15c9ef55587f8381be3fc54a53c785a"
    value = {
        "schema_version": 1,
        "reviewer_task": "/root/omni_p5_technical_20_14",
        "started_at": started,
        "completed_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "command": "UV_PROJECT_ENVIRONMENT=/workspace/tiny-perceptron-vlm/.venv PYTHONPATH=/workspace/work/omni-book-integrated-v1 uv run --extra cpu --group notebook --no-sync python docs/technical-reviews/artifacts/omni-P5-20.14/verify_source10_corrections_v1.py",
        "environment": {"python": platform.python_version(), "executable": sys.executable, "device": "CPU; standard-library byte/hash comparisons only"},
        "inputs": {"canonical_freeze": ref(freeze), "source": ref(ROOT / page["snapshot"]), "notebook": ref(notebook), "correction": ref(correction_path), "sources_index": ref(sources_path), "preserved_original_source_record": ref(source_record), "preserved_original_report": ref(original_report), "preserved_current_v1": ref(current1), "previous_inline_execution": ref(execution_path)},
        "single_source_edit": {"old": old, "new": new, "all_other_bytes_identical": True, "code_and_numeric_text_unchanged": True, "figures": {}},
        "execution_reuse": {"method": "Current v10 inline code bytes and the increment-1 variant exactly match the two already executed code digests; no repeated execution.", "previous_completed_at": execution["completed_at"], "code_sha256s": [c["code_sha256"] for c in execution["cases"]]},
        "correction_observations": observations,
        "sources_erratum_paragraph": errata,
        "primary_dispatch_inspection": {"url": "https://raw.githubusercontent.com/jaredpalmer/kev/2074cc44256200bb37540103fa4572901faa10f3/kev/predictors.py", "sha256": sha(predictors), "locator": "LocalPredictor L93-106 and L133-140", "small_supporting_excerpt": "\n".join(predictors.decode().splitlines()[132:139]), "finding": "Torch short rows take model.forward; the long hybrid branch can use shared_prefix=True. The revised statement names general Torch short rows and no longer implies exclusive serving reuse. Model forward_rows_batch L364-383 and probs L396-414 were reread locally."},
        "result": "All byte/digest/reuse assertions passed. i2 raw-byte errata and i3 wording corrections are supported by pinned originals; original/current-v1 remain unchanged.",
        "limits": "Bounded same-reviewer follow-up, not a new blind read. No network, external service, neural forward, training, retuning, Modal, workflow, or publication. No independent long-document bootstrap or page-layout check.",
    }
    OUTPUT.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"path": OUTPUT.relative_to(ROOT).as_posix(), "bytes": OUTPUT.stat().st_size, "sha256": sha(OUTPUT.read_bytes()), "completed_at": value["completed_at"], "result": value["result"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
