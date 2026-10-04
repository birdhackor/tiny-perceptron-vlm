"""Bounded CPU audit of T.1; reads existing assets/results, never trains a model."""

import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import struct
import tarfile
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf
import torch
from PIL import Image

from scripts.course_experiments.common import split_records
from scripts.course_experiments.modalities import _resample_8_to_16
from scripts.course_experiments.text import _deduplicate_text
from scripts.prepare_data import generate_records
from tiny_perceptron.modal_data import modal_example

os.chdir(ROOT)
torch.set_num_threads(1)
digest = lambda content: hashlib.sha256(content).hexdigest()
manifest = json.loads(Path("assets/training/manifest.json").read_text())
assets = {a["id"]: a for a in manifest["assets"]}
out = {"environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "soundfile": sf.__version__}, "training_run": False}
extraction = Path("outputs/reviewer-tools/tool-choice-dep-t1-facts")
out["section_facts"] = {"extraction": json.loads((extraction / "extraction.json").read_text()), "execution": json.loads((extraction / "execution.json").read_text()), "tool_exit_code": 2}
out["commands"] = []
for command in ("python scripts/fetch_training_assets.py --list", "python scripts/fetch_training_assets.py --asset tinystories"):
    run = subprocess.run(["bash", "-c", "source .venv/bin/activate\n" + command], text=True, capture_output=True, timeout=40)
    assert run.returncode == 0, run.stderr
    rows = [json.loads(line) for line in run.stdout.splitlines()]
    if "--list" in command:
        assert {r["id"] for r in rows} == set(assets)
        assert all(r["archive_bytes"] == assets[r["id"]]["archive_bytes"] for r in rows)
    else:
        assert len(rows) == 1 and rows[0]["asset"] == "tinystories"
        assert rows[0]["files_verified"] == len(assets["tinystories"]["files"])
        assert Path(rows[0]["destination"]) == ROOT / "data/training"
    out["commands"].append({"command": command, "activation": "source .venv/bin/activate", "exit_code": run.returncode, "stdout": rows, "stderr": run.stderr})
out["download_scope"] = "Only TinyStories CLI invoked; its full archive already exists, so no LFS network pull was exercised. Other three named assets read directly from existing archives."
out["manifest_presence"] = {a["id"]: {"version": a["version"], "license": a["license"], "source_metadata_exists": Path(a["source_metadata"]).is_file(), "archive_sha256_present": len(a["archive_sha256"]) == 64} for a in assets.values()}
payloads = {}
out["asset_integrity"] = {}
for name in ("tinystories", "chinese-poetry", "fashion-mnist", "fsdd"):
    asset = assets[name]
    raw = Path(asset["archive"]).read_bytes()
    assert len(raw) == asset["archive_bytes"] and digest(raw) == asset["archive_sha256"]
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        payload = {entry.name: archive.extractfile(entry).read() for entry in archive.getmembers()}
    assert set(payload) == {row["path"] for row in asset["files"]}
    for row in asset["files"]:
        assert len(payload[row["path"]]) == row["bytes"] and digest(payload[row["path"]]) == row["sha256"]
    payloads[name] = payload
    out["asset_integrity"][name] = {"archive_sha256": digest(raw), "files_verified": len(payload), "training_records": asset["training_records"]}
parse_rows = lambda value: [json.loads(line) for line in value.decode().splitlines() if line]
stories = parse_rows(payloads["tinystories"]["text-initial/tinystories-train-512.jsonl"])
prefix = payloads["tinystories"]["text-initial/tinystories-train-prefix-complete.txt"].decode()
assert prefix.rstrip().endswith("<|endoftext|>")
complete = [s.strip() for s in prefix.split("<|endoftext|>") if s.strip()]
assert len(stories) == len(complete) == 512
assert [r["text"] for r in stories] == complete
assert all(r["source_split"] == "train" and "messages" not in r for r in stories)
poems = parse_rows(payloads["chinese-poetry"]["text-initial/chinese-classical-train-365.jsonl"])
original_poems = json.loads(payloads["chinese-poetry"]["text-initial/tang300-source.json"])
render = lambda r: "\n".join([r["title"], r["author"], *r["paragraphs"]])
unique_original = list(dict.fromkeys(render(r) for r in original_poems))
assert len(original_poems) == 366 and len(poems) == len(unique_original) == 365
assert [r["text"] for r in poems] == unique_original
assert all(r["source_split"] == "unpartitioned" for r in poems)
out["text_records"] = {"tinystories": {"count": 512, "complete_delimited_stories": 512, "source_split": "train"}, "chinese-poetry": {"count": 365, "original_count": 366, "complete_rendered_poems": 365, "source_split": "unpartitioned"}}
attributes = generate_records("attributes-sft")
expected = {"color=red;shape=square;pitch=low;shape?": ("square", "red:square:low"), "color=red;shape=square;pitch=low;describe": ("square", "red:square:low"), "color=blue;shape=circle;pitch=high;shape?": ("circle", "blue:circle:high")}
observed = {}
for row in attributes:
    question, answer = row["messages"]
    if question["content"] in expected:
        value = (answer["content"], row["family"])
        assert value == expected[question["content"]]
        observed[question["content"]] = value
assert len(observed) == 3
parts = split_records(attributes, seed=42)
sets = [{r["family"] for r in parts[key]} for key in ("train", "validation", "test")]
assert all(not (sets[i] & sets[j]) for i in range(3) for j in range(i + 1, 3))
out["attributes"] = {"ABC": observed, "seed": 42, "split_counts": {k: len(v) for k, v in parts.items()}, "family_disjoint": True, "table_assignment_is_illustrative": True}
results = {name: json.loads(Path(f"docs/course-experiments/results/{name}.json").read_text()) for name in ("real_text", "real_modal")}
out["original_results"] = {}
for name, report in results.items():
    assert report["status"] == "completed" and report["evidence_status"] == "complete_run" and report["step_scale"] == 1.0
    out["original_results"][name] = {"path": f"docs/course-experiments/results/{name}.json", "sha256": digest(Path(f"docs/course-experiments/results/{name}.json").read_bytes()), "revision": report["revision"], "python": report["python_version"], "torch": report["torch_version"], "device": report["device"], "gpu": report["gpu"], "seed": report["seed"], "new_training_execution": False}
for report_name, code in (("real_text", "scripts/course_experiments/text.py"), ("real_text", "scripts/course_experiments/common.py"), ("real_modal", "scripts/course_experiments/modalities.py"), ("real_modal", "tiny_perceptron/modal_data.py")):
    assert digest(Path(code).read_bytes()) == results[report_name]["code_sha256"][code]
for name, rows in (("tinystories", stories), ("chinese-poetry", poems)):
    report = results["real_text"]["results"]["runs"][name]
    parts = split_records(_deduplicate_text(rows), seed=42)
    for key, selected in parts.items():
        raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected).encode()
        assert len(selected) == report["data"][key]["records"]
        assert digest(raw) == report["data"][key]["sha256"]
    out["text_records"][name]["reproduced_split_counts"] = {k: len(v) for k, v in parts.items()}
    out["text_records"][name]["split_jsonl_sha256_match_original_run"] = True
    out["text_records"][name]["original_training_steps"] = report["training"]["steps"]
fashion_rows = parse_rows(payloads["fashion-mnist"]["vision-initial/train.jsonl"])
assert len(fashion_rows) == 50 and set(Counter(row["label"] for row in fashion_rows).values()) == {5}
assert all(row["source_split"] == "train" for row in fashion_rows)
groups = {}
for row in fashion_rows:
    groups.setdefault(row["label"], []).append(row)
    im = Image.open(io.BytesIO(payloads["fashion-mnist"]["vision-initial/" + row["image"]]))
    im.load()
    assert im.size == (28, 28) and im.mode == "L"
fashion_split = {k: [] for k in ("train", "validation", "test")}
for group in groups.values():
    for i, row in enumerate(group):
        fashion_split["train" if i < 3 else "validation" if i == 3 else "test"].append(row)
modal = results["real_modal"]["results"]
out["modal_splits"] = {}
for key, rows in fashion_split.items():
    original = modal["fashion-mnist"]["data"]["splits"][key]
    assert [Path(row["image"]).name for row in rows] == [row["family"] for row in original["records"]]
out["modal_splits"]["fashion-mnist"] = {k: len(v) for k, v in fashion_split.items()}
resampling = {r["source"]: r for r in modal["fsdd"]["resampling"]}
run_metadata = json.loads(Path("docs/technical-reviews/artifacts/tool-choice-dep-t1-original-sources.json").read_text())["original_modal_run_metadata"]["fields"]
assert run_metadata["head_sha"] == results["real_modal"]["revision"]
first_stamp = int(datetime.fromisoformat(run_metadata["run_started_at"].replace("Z", "+00:00")).timestamp())
last_stamp = int(datetime.fromisoformat(run_metadata["updated_at"].replace("Z", "+00:00")).timestamp())
out["initial_comparison_failure"] = {"command": "CUDA_VISIBLE_DEVICES='' .venv/bin/python docs/technical-reviews/artifacts/tool-choice-dep-t1-audit.py", "exit_code": 1, "file": "0_jackson_5.wav", "assertion": "fresh FLOAT WAV full SHA equals historical derived SHA", "observed": "AssertionError at initial audit line 148", "diagnosis": "RIFF PEAK chunk includes write time; the rerun independently reconstructs the four-byte historical timestamp without changing the expected SHA or audio payload."}
def compare_historical_wave(fresh, expected):
    candidate = bytearray(fresh)
    offset, stamp_offset, payload = 12, None, None
    while offset + 8 <= len(candidate):
        count = struct.unpack_from("<I", candidate, offset + 4)[0]
        if candidate[offset:offset + 4] == b"PEAK":
            assert struct.unpack_from("<I", candidate, offset + 8)[0] == 1
            stamp_offset = offset + 12
        elif candidate[offset:offset + 4] == b"data":
            payload = candidate[offset + 8:offset + 8 + count]
        offset += 8 + count + count % 2
    assert stamp_offset is not None and payload is not None
    matches = []
    for stamp in range(first_stamp, last_stamp + 1):
        struct.pack_into("<I", candidate, stamp_offset, stamp)
        if digest(candidate) == expected:
            matches.append(stamp)
    assert len(matches) == 1, "audio bytes cannot reproduce the recorded SHA by changing write time alone"
    return {"fresh_full_sha256": digest(fresh), "fresh_full_sha_matches_original": digest(fresh) == expected, "historical_PEAK_timestamp": matches[0], "audio_payload_sha256": digest(payload), "historical_reconstruction_matches_original_sha": True, "changed_field": "only four PEAK write-time bytes in an in-memory comparison; expected SHA and saved samples unchanged"}
audio_checks = []
for key, speaker in (("train", "jackson"), ("validation", "nicolas"), ("test", "theo")):
    rows = parse_rows(payloads["fsdd"][f"fsdd-initial/{key}.jsonl"])
    assert len(rows) == 20 and {r["speaker"] for r in rows} == {speaker}
    assert Counter(r["digit_label"] for r in rows) == Counter({i: 2 for i in range(10)})
    assert {r["recording_index"] for r in rows} == {5, 6}
    assert [Path(r["path"]).name for r in rows] == [r["family"] for r in modal["fsdd"]["data"]["splits"][key]["records"]]
    for row in rows:
        raw = payloads["fsdd"]["fsdd-initial/" + row["path"]]
        values, rate = sf.read(io.BytesIO(raw), dtype="float32")
        assert rate == 8000 and values.ndim == 1
        derived = _resample_8_to_16(values)
        saved = io.BytesIO()
        sf.write(saved, derived, 16000, subtype="FLOAT", format="WAV")
        logged = resampling[Path(row["path"]).name]
        assert digest(raw) == logged["source_sha256"]
        assert len(derived) == 2 * len(values) == logged["samples_after"]
        assert len(values) == logged["samples_before"]
        assert len(values) / rate == len(derived) / 16000
        comparison = compare_historical_wave(saved.getvalue(), logged["derived_sha256"])
        audio_checks.append({"source": Path(row["path"]).name, "samples_before": len(values), "samples_after": len(derived), "duration_seconds": len(values) / rate, **comparison})
    out["modal_splits"]["fsdd-" + key] = {"speaker": speaker, "records": len(rows), "digits": 10, "takes_per_digit": 2}
out["resampling"] = {"files_checked": len(audio_checks), "source_rate": 8000, "target_rate": 16000, "all_point_counts_doubled": True, "all_durations_equal": True, "all_historical_reconstruction_hashes_match_original_run": True, "fresh_writes_are_not_byte_identical": True, "historical_timestamp_search_metadata": run_metadata, "example": audio_checks[0]}
work = ROOT / "outputs/reviewer-tools/tool-choice-dep-t1-audio"
work.mkdir(parents=True, exist_ok=True)
first = payloads["fsdd"]["fsdd-initial/recordings/0_jackson_5.wav"]
(work / "raw.wav").write_bytes(first)
record = {"audio": "raw.wav", "question": "digit?", "answer": "0"}
try:
    modal_example(record, work, "audio", "cpu")
except ValueError as error:
    out["training_audio_reader"] = {"8000_Hz_rejected": True, "error": str(error)}
else:
    raise AssertionError("training reader unexpectedly accepted 8 kHz")
values, _ = sf.read(io.BytesIO(first), dtype="float32")
sf.write(work / "derived.wav", _resample_8_to_16(values), 16000, subtype="FLOAT")
record["audio"] = "derived.wav"
ids, labels, image, wave = modal_example(record, work, "audio", "cpu")
assert image is None and wave.shape == (9182,)
out["training_audio_reader"]["16000_Hz_accepted_shape"] = list(wave.shape)
out["heldout_result_recalculation"] = {}
for name, expected_examples in (("fashion-mnist", 10), ("fsdd", 20)):
    report = modal[name]
    test = report["test"]
    correct = sum(r["generated"] == r["target"] for r in test["samples"])
    assert len(test["samples"]) == test["examples"] == expected_examples
    assert correct == test["correct"] == 3
    assert test["exact_match"] == correct / expected_examples
    out["heldout_result_recalculation"][name] = {"correct": correct, "examples": expected_examples, "exact_match": test["exact_match"], "seed": results["real_modal"]["seed"], "original_steps": report["training"]["steps"], "original_effective_targets": report["training"]["effective_tokens"], "new_training_execution": False}
notebooks = sorted(Path("outputs/tool-choice-site-kernels").rglob("*.ipynb"))
assert len(notebooks) == 226
references = []
for path in notebooks:
    notebook = json.loads(path.read_text())
    code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    for term in ("fetch_training_assets", "data/training", "assets/training"):
        if term in code:
            references.append({"notebook": str(path), "term": term})
assert not references
out["notebook_asset_dependency_scan"] = {"input": "outputs/tool-choice-site-kernels", "notebooks": len(notebooks), "training_asset_code_references": references, "new_notebook_execution": False, "scope": "Static audit of this website input; bootstrap installation can still need network. No claim that all notebooks were rerun."}
out["result"] = "pass: commands, four named archive manifests, complete text rows, family rules, original split hashes, 50 image dimensions, 60 audio conversions, training reader and stored test denominators match T.1"
Path("docs/technical-reviews/artifacts/tool-choice-dep-t1-audit.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"result": out["result"], "environment": out["environment"], "resampling": out["resampling"], "scores": out["heldout_result_recalculation"]}, ensure_ascii=False))
