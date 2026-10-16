"""Fresh README checks: bounded CPU only; inspect named raw-data fields."""

import contextlib
import hashlib
import importlib.metadata
import io
import json
import math
import platform
import re
import shutil
import subprocess
import sys
import tarfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[5]
DEST = Path(__file__).resolve().parent
SCRATCH = ROOT / "work/engineering-readme-clean"
ORIGINAL = Path("/workspace/selftrained-v2")
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.file_digest(Path(path).open("rb"), "sha256").hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read(path):
    return json.loads(Path(path).read_bytes())


def raw_copy(path, target, expected=None):
    digest = sha(path)
    if expected:
        assert digest == expected, (path, digest, expected)
    target = DEST / "raw" / target
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)
    assert sha(target) == digest
    return {"original": str(path), "copy": str(target.relative_to(ROOT)), "sha256": digest,
            "bytes": path.stat().st_size}


def main():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    result = {"checked_at": datetime.now(UTC).isoformat(), "command": "PYTHONPATH=. .venv/bin/python "
              + str(Path(__file__).relative_to(ROOT)),
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "device": "cpu", "packages": {}},
              "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    source_files = ["README.md", "README_en.md", ".github/workflows/gpu-smoke.yml", "scripts/train.py",
                    "scripts/prepare_data.py", "tiny_perceptron/model.py", "tiny_perceptron/data.py",
                    "tiny_perceptron/training.py", "tiny_perceptron/selftrained/model.py",
                    "tiny_perceptron/selftrained/dataset.py", "scripts/lesson_products/modal_runner.py",
                    "tests/test_lesson_products_runner.py", "scripts/fetch_course_models.py",
                    "scripts/fetch_training_assets.py", "docs/selftrained/v2-manifest.json",
                    "docs/course-experiments/public-models.json", "docs/course-experiments/capstone-public.json",
                    "pyproject.toml", "uv.lock"]
    result['source_sha256'] = {name:sha(ROOT/name) for name in source_files}
    for name in ["torch", "numpy", "pillow", "soundfile", "tokenizers", "datasets", "huggingface-hub",
                 "ipykernel", "jupyterlab", "safetensors", "cairosvg", "uv"]:
        try:
            result["environment"]["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["environment"]["packages"][name] = "not-installed"
    import torch
    torch.set_num_threads(2)
    result["environment"]["torch"] = str(torch.__version__)
    result["environment"]["cuda_available"] = torch.cuda.is_available()
    assert not torch.cuda.is_available()
    result["snapshots"] = [raw_copy(ROOT / name, name) for name in
                           ["README.md", "README_en.md", ".github/workflows/gpu-smoke.yml",
                            "course/figures/rewrite-01-character-ids.svg"]]

    # Count headings directly, without taking a previous reviewer inventory as input.
    lessons = {}
    for path in sorted((ROOT / "course/chapters").glob("*.md")):
        for identifier in re.findall(r"^## ([\dABC]+\.\d+) ", path.read_text(), re.M):
            assert identifier not in lessons
            lessons[identifier] = str(path.relative_to(ROOT))
    front = {}
    for name in ["README.md", "first-steps.md", "training.md", "glossary.md"]:
        front[name] = re.findall(r"^## ([\dA-Z]+\.\d+) ", (ROOT / "course" / name).read_text(), re.M)
    notebooks = list((ROOT / "notebooks").glob("*/*.ipynb"))
    assert len(lessons) == len(notebooks) == 312
    assert sum(map(len, front.values())) == 26
    assert len(lessons) + sum(map(len, front.values())) == 338
    notebook_ids, code_notebooks, bootstrap = set(), 0, 0
    for path in notebooks:
        data = read(path)
        notebook_ids.add(data["metadata"]["lesson_id"])
        cells = [c for c in data["cells"] if c["cell_type"] == "code"]
        if cells:
            code_notebooks += 1
            assert cells[0]["metadata"].get("course_setup") is True, path
            source = "".join(cells[0]["source"])
            assert "import google.colab" in source and '"pip", "install"' in source
            bootstrap += 1
    assert notebook_ids == set(lessons)
    result["curriculum"] = {"chapter_files": sorted({Path(p).stem for p in lessons.values()}),
                            "lessons": len(lessons), "notebooks": len(notebooks), "front": front,
                            "numbered_total": 338, "code_notebooks": code_notebooks,
                            "first_code_bootstrap": bootstrap}

    # Run the original preparation and dry-run main functions. The sole changes are scratch paths,
    # explicit CPU selection, and spies that fail if an update or checkpoint is attempted.
    from scripts import prepare_data, train
    log = io.StringIO()
    with contextlib.redirect_stdout(log), patch.object(sys, "argv", ["prepare_data.py", "--kind", "toy-text",
                                                                     "--output", str(SCRATCH / "generated")]):
        prepare_data.main()
    result["toy_data"] = read(SCRATCH / "generated/toy-text/manifest.json")["splits"]
    dry_runs = []
    original_model = train.TinyLM
    for extra in [[], ["--max-length", "32", "--batch-size", "2", "--seed", "43"]]:
        constructed = []
        class ObservedModel(original_model):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                constructed.append((self, {k: v.detach().clone() for k, v in self.state_dict().items()}))
        output = SCRATCH / ("variation.pt" if extra else "default.pt")
        argv = ["train.py", "--task", "text", "--data", str(SCRATCH / "generated/toy-text/train.jsonl"),
                "--device", "cpu", "--output", str(output), *extra]
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), patch.object(sys, "argv", argv), \
             patch.object(train, "TinyLM", ObservedModel), \
             patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("unexpected weight update")) as step, \
             patch.object(train, "save_checkpoint", side_effect=AssertionError("unexpected checkpoint write")) as save:
            train.main()
        report = json.loads(stdout.getvalue().splitlines()[-1])
        assert report["mode"] == "dry-run-no-weight-update"
        assert len(report["history"]) == 1
        row = report["history"][0]
        assert math.isfinite(row["loss"]) and math.isfinite(row["grad_norm"]) and row["grad_norm"] > 0
        assert step.call_count == save.call_count == 0 and not output.exists()
        model, before = constructed[0]
        changed = [k for k,v in model.state_dict().items() if not torch.equal(v,before[k])]
        assert changed == []
        dry_runs.append({"argv": argv, "report": report, "changed_state_tensors": changed,
                         "optimizer_step_calls": step.call_count, "checkpoint_save_calls": save.call_count})
    result["dry_runs"] = dry_runs

    # Recompute released model capacities from the current architecture and raw configuration.
    from tiny_perceptron.selftrained.model import SelftrainedConfig, LimitedAssistant
    summary = read(ROOT / "docs/selftrained/results/v2-final-public-results.json")
    result["v2_capacity"] = {}
    for arch, entry in summary["architectures"].items():
        config = entry["capacity"]["config"]
        model = LimitedAssistant(SelftrainedConfig(**config))
        count = sum(p.numel() for p in model.parameters())
        expected = {"moe": 5447107, "dense": 2288067}[arch]
        assert count == expected == entry["capacity"]["parameters"]
        result["v2_capacity"][arch] = {"config": config, "parameters": count}

    # Inspect the existing original LFS archive, without downloading or extracting it.
    manifest = read(ROOT / "docs/selftrained/v2-manifest.json")
    archive = ORIGINAL / "assets/training/selftrained-v2.tar.gz"
    assert archive.stat().st_size == manifest["package"]["bytes"] == 67862431
    assert sha(archive) == manifest["package"]["sha256"]
    counters, tasks, ocr, voice, vision, file_rows = Counter(), Counter(), set(), set(), set(), {}
    ocr_lengths = set()
    with tarfile.open(archive) as tar:
        members = [m for m in tar if m.isfile()]
        assert len(members) == 8962
        by_name = {m.name: m for m in members}
        for spec in manifest["records"]:
            raw = tar.extractfile(by_name[spec["path"]]).read()
            assert hashlib.sha256(raw).hexdigest() == spec["sha256"] and len(raw) == spec["bytes"]
            rows = [json.loads(line) for line in raw.splitlines() if line]
            split = spec["path"].rsplit("-",1)[-1].split(".")[0]
            assert all(row["split"] == split for row in rows)
            counters[split] += len(rows)
            file_rows[spec["path"]] = len(rows)
            for row in rows:
                tasks[row["task"]] += 1
                supervision = row.get("supervision", {})
                if row["task"] == "ocr":
                    ocr.update(supervision["ocr_text"])
                    ocr_lengths.add(len(supervision["ocr_text"]))
                if row["task"].startswith("voice"):
                    voice.add(supervision["intent"])
                if row["task"].startswith("vision"):
                    vision.update(supervision["vision_labels"])
    assert counters == {"train": 28876, "validation": 2435, "test": 3734}
    # The model/data contract reserves -1 for an empty slot, not a fourth clothing class.
    assert len(ocr) == 12 and ocr_lengths == {1,2,3,4} and len(voice) == 3 and vision == {-1,0,1,2}
    result["v2_data"] = {"archive_bytes": archive.stat().st_size, "archive_sha256": sha(archive),
                         "canonical_source": "assets/training/selftrained-v2.tar.gz (Git LFS)",
                         "regular_files": len(members), "records_per_split": counters,
                         "records_per_file": file_rows, "tasks": tasks, "ocr_glyphs": sorted(ocr),
                         "ocr_lengths": sorted(ocr_lengths), "voice_intents": sorted(voice),
                         "vision_labels": sorted(vision), "vision_class_count": 3,
                         "read_scope": "12 manifest records: /split, /task, "
                         "/supervision/ocr_text, /supervision/intent, /supervision/vision_labels; count all tar regular members"}

    # Re-check original machine records; never read notes, review paths or authored conclusions.
    result["raw_provenance"], result["v2_final_test"], result["v2_training"] = [], {}, {}
    for arch, entry in summary["architectures"].items():
        original = ORIGINAL / entry["final_test"]["source_metrics_path"]
        result["raw_provenance"].append(raw_copy(original, f"{arch}-final-metrics.json", entry["final_test"]["metrics_sha256"]))
        metrics = read(original)
        total = sum(v["count"] for v in metrics["per_task_final_reply"].values())
        tool = metrics["per_task_final_reply"]["tool_call"]["tool_roundtrip"]
        assert total == metrics["count"] == metrics["expected_count"] == 3734
        assert metrics["evaluation_complete"] is True and metrics["interrupted"] is False
        assert tool["rate"] == tool["numerator"] / tool["denominator"]
        assert tool["rate"] < metrics["thresholds"]["tool_roundtrip"] == 0.95
        result["v2_final_test"][arch] = {"count": total, "tool_roundtrip": tool,
                                        "tool_threshold": metrics["thresholds"]["tool_roundtrip"],
                                        "complete": metrics["evaluation_complete"],
                                        "per_task_counts": {k:v["count"] for k,v in metrics["per_task_final_reply"].items()},
                                        "inspected_pointers": ["/count", "/expected_count", "/evaluation_complete", "/interrupted",
                                        "/per_task_final_reply/*/count", "/per_task_final_reply/tool_call/tool_roundtrip", "/thresholds/tool_roundtrip"]}
    for arch, index in [("moe", "v2-moe-native-balanced-4000-training-stage-index.json"),
                        ("dense", "v2-dense-weighted-10000-training-stage-index.json")]:
        idx = read(ROOT / "docs/selftrained/results" / index)
        receipt_path = idx.get("receipt", {}).get("path", idx.get("receipt_path"))
        expected = idx.get("receipt", {}).get("sha256", idx.get("receipt_sha256"))
        original = ORIGINAL / receipt_path
        result["raw_provenance"].append(raw_copy(original, f"{arch}-execution-receipt.json", expected))
        execution = read(original)
        assert execution["status"] == "completed" and execution["returncode"] == 0
        inspected = {}
        for name in ["train-receipt.json", "inference-manifest.json"]:
            raw = original.parent / "raw" / name
            expected_hash = next(f["sha256"] for f in execution["files"] if f["path"] == name)
            result["raw_provenance"].append(raw_copy(raw, f"{arch}-{name}", expected_hash))
            value = read(raw)
            inspected[name] = {k:value[k] for k in (["steps", "completed_requested_steps", "test_used_for_selection", "total_parameters"]
                                                  if name == "train-receipt.json" else
                                                  ["selected_step", "selection", "files", "selected_checkpoint_sha256"])}
        assert inspected["train-receipt.json"]["steps"] == {"moe":4000,"dense":10000}[arch]
        assert inspected["train-receipt.json"]["completed_requested_steps"] is True
        assert inspected["train-receipt.json"]["test_used_for_selection"] is False
        assert inspected["inference-manifest.json"]["selected_step"] == 1000
        assert inspected["inference-manifest.json"]["selection"] == "validation_loss"
        result["v2_training"][arch] = inspected

    # Count weight files only; companion files are not checkpoints.
    models = read(ROOT / "docs/course-experiments/public-models.json")["models"]
    weights = [f for m in models for f in m["files"] if f["path"].endswith((".pt", ".safetensors"))]
    assert len(models) == 30 and len(weights) == 120
    assert all(re.fullmatch(r"[a-f0-9]{40}", m["revision"]) for m in models)
    capstone = read(ROOT / "docs/course-experiments/capstone-public.json")
    assert len(capstone["models"]) == 11
    result["historical_public_files"] = {"experiment_groups":len(models), "weights":len(weights),
                                        "all_pinned":True, "capstone_models":len(capstone["models"])}
    workflow = (ROOT / ".github/workflows/gpu-smoke.yml").read_text()
    assert "if: ${{ inputs.curriculum_job == '' }}" in workflow
    assert "if: ${{ inputs.curriculum_job != '' }}" in workflow
    assert "default: ''" in workflow and "workflow_dispatch:" in workflow
    result["workflow"] = {"smoke_if":"inputs.curriculum_job == ''", "products_if":"inputs.curriculum_job != ''",
                           "default_curriculum_job":"", "trigger":"workflow_dispatch", "gpu_launched":False}
    # List commands are explicitly offline. The bounded job contract is exercised by its offline tests.
    result["subprocesses"] = []
    commands = [[sys.executable,"scripts/fetch_training_assets.py","--list"],
                [sys.executable,"scripts/fetch_course_models.py","--list"],
                [sys.executable,"-m","pytest","-q","tests/test_lesson_products_runner.py"]]
    for command in commands:
        ran = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=45)
        assert ran.returncode == 0, ran.stderr
        result["subprocesses"].append({"argv":command,"returncode":ran.returncode,"stdout":ran.stdout,"stderr":ran.stderr})
    assert result['source_sha256'] == {name:sha(ROOT/name) for name in source_files}
    write(DEST / "checks-result.json", result)
    print(json.dumps({"status":"passed", "results":str((DEST / "checks-result.json").relative_to(ROOT)),
                      "dry_runs":len(dry_runs),"v2_counts":dict(counters),"tests":result["subprocesses"][-1]["stdout"]}))


if __name__ == "__main__":
    main()
