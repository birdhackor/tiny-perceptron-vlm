"""真控制流+假訓練/HF：驗完整中途bundle及付費前資料來源鏈。"""

import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts.lesson_products import checkpoints as checkpoint
from scripts.lesson_products import modal_runner as runner


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started = datetime.now(UTC).isoformat()
    original_report = ROOT / "docs/lesson-products/reviews/engineering-runner-original.json"
    assert sha(original_report) == "368139cecf659abc40602b93f2ec66b3230499f2aaba23b72c382e5f843057e0"
    prior = json.loads(original_report.read_bytes())
    code = ROOT / next(x["path"] for x in prior["artifacts"] if x["id"] == "probe_code")
    spec = importlib.util.spec_from_file_location("original_fake_classes", code)
    classes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(classes)
    checked_names = runner.SOURCE_FILES + ["tests/test_lesson_products_runner.py", "tests/test_lesson_products_checkpoints.py"]
    checked_names += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "docs/lesson-products/jobs").glob("*.json"))]
    before = {name: sha(ROOT / name) for name in checked_names}
    workroot = ROOT / "work"
    workroot.mkdir(exist_ok=True)
    results = []
    with tempfile.TemporaryDirectory(dir=workroot, prefix="runner-recovery-callback-") as folder:
        work = Path(folder)
        remote = work / "remote"
        output = remote / "course/lesson-products/original"
        output.mkdir(parents=True)
        (output / "reservation.json").write_text('{"synthetic_preflight":true}')
        data_manifest = remote / "input-data/manifest.json"
        data_manifest.parent.mkdir(parents=True)
        data_manifest.write_text('{"dataset":"original fixed dataset; synthetic"}')
        job = json.loads((ROOT / "docs/lesson-products/jobs/train-b-v1.json").read_bytes())
        hashes = {name: sha(ROOT / name) for name in checkpoint.TRAIN_SOURCES}
        hashes[runner.DATA_MANIFEST] = sha(data_manifest)
        book = {"reserved_total_usd": "10.50", "reservations": [{"run_id": "history", "reserved_usd": "10.00", "status": "completed"},
                {"run_id": "original", "status": "reserved", "experiment_id": "selftrained_lesson_products", "reserved_usd": ".50",
                 "job_sha256": runner.canonical_sha(job), "source_sha256": hashes.copy(), "manifest_sha256": hashes[runner.DATA_MANIFEST],
                 "runtime_options": copy.deepcopy(job), "gross_quota_guard": {"synthetic": True}}]}
        ledger = remote / "course/budget.json"
        runner.write_json(ledger, book)
        events, uploads = [], []
        app = classes.FakeApp()
        volume = SimpleNamespace(reload=lambda: events.append("volume.reload"), commit=lambda: events.append("volume.commit"))
        sdk = SimpleNamespace(App=lambda _: app, Volume=SimpleNamespace(from_name=lambda *a, **k: volume),
                              Secret=SimpleNamespace(from_name=lambda *a, **k: "synthetic secret"),
                              Image=SimpleNamespace(debian_slim=lambda **k: classes.FakeImage()),
                              current_function_call_id=lambda: "fc-synthetic")
        with patch.dict(sys.modules, {"modal": sdk}):
            _, _, execute, _ = runner.register("run", remote / "input-data")

        class API:
            def __init__(self, token):
                pass

            def upload_folder(self, **kwargs):
                revision = f"{len(uploads) + 1:040x}"
                destination = work / "synthetic-hf" / revision
                destination.mkdir(parents=True)
                names = []
                for name in kwargs["allow_patterns"]:
                    path = Path(kwargs["folder_path"]) / name
                    if path.is_file():
                        shutil.copyfile(path, destination / name)
                        names.append(name)
                uploads.append({"repo": kwargs["repo_id"], "revision": revision, "files": names,
                                "prefix": kwargs["path_in_repo"], "bundle_sha256": sha(destination / "checkpoint-bundle.json")})
                events.append("HF.upload_folder paired commit")
                return SimpleNamespace(oid=revision)

        def fake_train(data, destination, product, mode, **kwargs):
            assert kwargs["resume"] is None and kwargs["resume_best"] is None
            destination = Path(destination)
            common = {"synthetic_only": True, "run_identity": "same-run", "data_manifest_sha256": hashes[runner.DATA_MANIFEST]}
            (destination / "best.pt").write_text(json.dumps({**common, "step": 200}))
            (destination / "last.pt").write_text(json.dumps({**common, "step": 1000, "best_step": 200}))
            (destination / "config.json").write_text('{"synthetic":true}')
            (destination / "tokenizer.json").write_text('{"synthetic":true}')
            (destination / "progress.json").write_text(json.dumps({"metadata": {**common, "step": 1000, "best_step": 200}}))
            kwargs["on_checkpoint"](destination / "last.pt", 1000)
            raise RuntimeError("synthetic failure after successful paired HF boundary")

        import torch
        import tiny_perceptron.lesson_products.train as training

        def mapped_path(value):
            value = str(value)
            return remote / value.lstrip("/") if value.startswith(("/course", "/input-data")) else Path(value)

        with patch.dict(sys.modules, {"modal": sdk, "huggingface_hub": SimpleNamespace(HfApi=API)}), patch.dict(os.environ, {"HF_TOKEN": "synthetic token"}), patch.object(runner, "LEDGER", ledger), patch.object(runner, "Path", side_effect=mapped_path), patch.object(runner, "check_source_files", return_value=None), patch.object(torch.cuda, "is_available", return_value=True), patch.object(torch.cuda, "get_device_name", return_value="synthetic L4; no hardware call"), patch.object(training, "train", side_effect=fake_train):
            receipt = execute("original", "a" * 40, job, hashes, "synthetic/private")
        book = json.loads(ledger.read_bytes())
        assert receipt["status"] == "failed" and book["reservations"][-1]["status"] == "failed"
        assert len(uploads) == 1
        backup = uploads[0]
        assert set(backup["files"]) == set(checkpoint.FILES) | {"checkpoint-bundle.json"}
        assert sha(output / "checkpoint-bundle.json") == backup["bundle_sha256"]
        assert json.loads((output / "checkpoint-backup.json").read_bytes())["revision"] == backup["revision"]
        results.append({"id": "failed_midrun_HF_has_complete_same_commit_bundle", "expected": "complete", "actual": "complete", "passed": True,
                        "HF_uploads": uploads, "ledger_status": book["reservations"][-1]["status"], "bound_retained": book["reserved_total_usd"],
                        "volume_files": sorted(x.name for x in output.iterdir()), "events": events.copy()})
        calls = []

        def download(repo, path, revision, token):
            assert repo == "synthetic/private" and revision == backup["revision"]
            calls.append({"repo": repo, "path": path, "revision": revision})
            return work / "synthetic-hf" / revision / path.rsplit("/", 1)[1]

        resume = {"run_id": "original", "source": "hf", "hf_revision": backup["revision"], "bundle_sha256": backup["bundle_sha256"]}
        requested = copy.deepcopy(job)
        requested["resume"] = resume
        runner.validate_job(requested)
        with patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(hf_hub_download=download)}), patch.dict(os.environ, {"HF_TOKEN": "synthetic token"}):
            destination = work / "hf-recovered"
            summary = checkpoint.prepare_resume(resume, destination, remote / "course", "synthetic/private", hashes, requested, book)
            assert summary["files_verified"] == 5
            assert len(calls) == 6 and all(c["revision"] == backup["revision"] for c in calls)
            assert all((destination / name).read_bytes() == (output / name).read_bytes() for name in (*checkpoint.FILES, "checkpoint-bundle.json"))
            results.append({"id": "fixed_HF_commit_restore_positive_control", "expected": "pass", "actual": "pass", "passed": True, "files_verified": 5, "downloads": calls.copy()})
            # 只換新attempt將使用的資料manifest，不改原checkpoint/ledger/四個training source/recipe。
            new_hashes = hashes.copy()
            new_hashes[runner.DATA_MANIFEST] = hashlib.sha256(b"different dataset with same generating code and recipe").hexdigest()
            caught = None
            try:
                value = checkpoint.prepare_resume(resume, work / "wrong-data-hf", remote / "course", "synthetic/private", new_hashes, requested, book)
            except Exception as error:
                caught = {"type": type(error).__name__, "message": str(error)}
            results.append({"id": "HF_preflight_binds_original_data_identity", "base_passed": True, "expected": "reject", "actual": "reject" if caught else "accepted",
                            "passed": caught is not None, "error": caught, "original_entry_manifest_sha256": book["reservations"][-1]["manifest_sha256"],
                            "requested_data_manifest_sha256": new_hashes[runner.DATA_MANIFEST], "training_sources_unchanged": True, "recipe_unchanged": True,
                            "files_verified_when_accepted": value["files_verified"] if not caught else None})
        volume_resume = {**resume, "source": "volume", "hf_revision": None}
        destination = work / "volume-recovered"
        summary = checkpoint.prepare_resume(volume_resume, destination, remote / "course", "synthetic/private", hashes, {**job, "resume": volume_resume}, book)
        assert summary["files_verified"] == 5
        results.append({"id": "Volume_restore_positive_control", "expected": "pass", "actual": "pass", "passed": True, "files_verified": 5})
        caught = None
        try:
            checkpoint.prepare_resume(volume_resume, work / "wrong-data-volume", remote / "course", "synthetic/private", new_hashes, {**job, "resume": volume_resume}, book)
        except Exception as error:
            caught = {"type": type(error).__name__, "message": str(error)}
        results.append({"id": "Volume_preflight_binds_original_data_identity", "base_passed": True, "expected": "reject", "actual": "reject" if caught else "accepted", "passed": caught is not None, "error": caught})
    assert before == {name: sha(ROOT / name) for name in checked_names}
    observed = [item["id"] for item in results if not item["passed"]]
    assert observed == ["HF_preflight_binds_original_data_identity", "Volume_preflight_binds_original_data_identity"]
    report = {"reviewer_task": "/root/tech_20261016_review_tools", "initial_context": "fresh", "current_context": "informed", "started_at": started, "completed_at": datetime.now(UTC).isoformat(),
              "environment": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "torch_distribution": importlib.metadata.version("torch")},
              "checked_files": [{"path": name, "sha256": digest} for name, digest in before.items()], "results": results, "counterexamples": observed,
              "verdict": "revise", "limits": ["Actual execute/persist/prepare_resume code, synthetic files and train/HF/GPU/API replacements; no model step, remote API or external write.", "Checkpoint files are explicit synthetic JSON bytes, not real trained weights. Pair metadata loader is separately reviewed by the model reviewer.", "Both changed-data counterexamples start with actually accepted corresponding HF/Volume controls; original ledger and source checkpoints remain unchanged."]}
    path = ART / "engineering-runner-recovery-probes-results-callback-01.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"results": len(results), "counterexamples": observed, "verdict": "revise", "result_sha256": sha(path)}))


if __name__ == "__main__":
    main()
