"""離線證明先核准整批、逐項隔離證據，且任何失敗都不重試或繼續。"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import release_course_batch as batch

SHA = "3" * 40
IDS = ["sft", "vqa", "audio"]


@pytest.fixture
def runner(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    monkeypatch.delenv("GITHUB_RUN_ATTEMPT", raising=False)
    blobs, events, controls = {}, [], {}

    def save(relative, document):
        content = (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode()
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        blobs[f"{SHA}:{relative}"] = content

    plan = {"schema_version": 1, "batch_id": "course-v1", "sequence": [{"id": name} for name in IDS]}
    save("docs/course-experiments/plan.json", plan)
    approvals = {}
    for name in IDS:
        approvals[name] = {
            "schema_version": 1,
            "approved": True,
            "reviewed": True,
            "experiment_id": name,
            "batch_id": "course-v1",
            "revision": "1" * 40,
            "private_source": {"repo": "owner/private", "revision": "2" * 40, "prefix": f"course/course-v1/{name}/run"},
            "files": [
                {
                    "path": "model.pt",
                    "sha256": "a" * 64,
                    "kind": "checkpoint",
                    "license": "MIT",
                    "redistribution_approved": True,
                }
            ],
            "model_card": {"summary": "Toy model", "scope": "Finite exercise", "limitations": ["Small"]},
            "inference": {"script": "scripts/infer.py", "checkpoint": "model.pt", "prompt": "x", "tokens": 2},
        }
        save(f"docs/course-experiments/releases/{name}.json", approvals[name])

    def fake_subprocess(command, **options):
        assert options["cwd"] == root
        if command[:2] == ["git", "show"]:
            events.append(("git", command[2]))
            assert options["check"] is True and options["capture_output"] is True
            if command[2] not in blobs:
                raise subprocess.CalledProcessError(1, command, stderr=b"missing committed approval")
            return subprocess.CompletedProcess(command, 0, stdout=blobs[command[2]], stderr=b"")
        assert command[:3] == ["modal", "run", "scripts/modal_course.py"]
        assert command[command.index("--mode") + 1] == "release"
        assert options["check"] is False and options["timeout"] == batch.ITEM_TIMEOUT_SECONDS
        name = command[command.index("--experiment-id") + 1]
        run_id = command[command.index("--run-id") + 1]
        events.append(("paid", name))
        destination = root / "outputs/modal-course"
        # 下一項必須看到前項的五個輸出已隔離，而非覆寫它們。
        assert all(not (destination / filename).exists() for filename in batch.GENERATED_FILES)
        approval = approvals[name]
        prefix = f"course/course-v1/{name}"
        public = {
            "id": name,
            "repo": "owner/public",
            "revision": "4" * 40,
            "inference": approval["inference"],
            "files": [
                {"path": f"{prefix}/{output}", "output": output, "sha256": "b" * 64, "bytes": 16}
                for output in ("model.pt", "LICENSE", "README.md", "export-manifest.json")
            ],
        }
        release = {
            "experiment_id": name,
            "repo": "owner/public",
            "revision": public["revision"],
            "weights_revision": "5" * 40,
            "anonymous_download_verified": True,
            "gpu_used": controls.get("invalid_gpu") == name,
            "public_manifest": public,
        }
        # 實際既有 release 沒有頂層 status；completed 記在 billing entry。
        result = {
            "experiment_id": name,
            "mode": "release",
            "preflight": {},
            "approval": {
                "approval_git_revision": SHA,
                "approval_sha256": hashlib.sha256(
                    blobs[f"{SHA}:docs/course-experiments/releases/{name}.json"]
                ).hexdigest(),
                "private_source": approval["private_source"],
                "files_verified": len(approval["files"]),
            },
            "release": release,
            "billing": {
                "entry": {
                    "run_id": run_id,
                    "experiment_id": name,
                    "batch_id": "course-v1",
                    "mode": "release",
                    "status": "completed",
                    "reserved_usd": "0.04",
                    "compute_guard": {"gpu_used": False},
                }
            },
        }
        if controls.get("wrong_run") == name:
            result["billing"]["entry"]["run_id"] = "gha-999-1-other"
        output = {
            "billing-before.json": {"scope": "account-wide", "test_run": name, "when": "before"},
            "billing-after.json": {"scope": "account-wide", "test_run": name, "when": "after"},
            "result.json": result,
            "release.json": release,
            "public-manifest.json": public,
        }
        if controls.get("fail") == name:
            result["status"] = "failed"
            result["exception_type"] = "RuntimeError"
            result["billing"]["entry"]["status"] = "failed"
        for filename, value in output.items():
            (destination / filename).write_text(json.dumps(value))
        options["stdout"].write(f"release output for {name}\n")
        options["stderr"].write(f"diagnostic for {name}\n")
        if controls.get("timeout") == name:
            raise subprocess.TimeoutExpired(command, options["timeout"])
        return subprocess.CompletedProcess(command, 1 if controls.get("fail") == name else 0)

    monkeypatch.setattr(batch.subprocess, "run", fake_subprocess)

    def run(ids=None):
        return batch.run_batch(
            ids or IDS,
            root=root,
            revision=SHA,
            batch_id="course-v1",
            checkpoint_repo="owner/private",
            release_repo="owner/public",
            run_prefix="gha-123-2",
        )

    return root, blobs, events, controls, approvals, save, run


@pytest.mark.parametrize("rejection", ["unreviewed", "pku", "changed_bytes", "missing_commit", "utf8_bom"])
def test_whole_batch_approval_rejection_precedes_every_paid_call(runner, rejection):
    root, blobs, events, _controls, approvals, save, run = runner
    relative = "docs/course-experiments/releases/audio.json"
    if rejection in ("unreviewed", "pku"):
        approvals["audio"]["reviewed" if rejection == "unreviewed" else "pku_derived"] = rejection != "unreviewed"
        save(relative, approvals["audio"])
    elif rejection == "changed_bytes":
        (root / relative).write_bytes(blobs[f"{SHA}:{relative}"] + b"\n")
    elif rejection == "utf8_bom":
        content = b"\xef\xbb\xbf" + blobs[f"{SHA}:{relative}"]
        blobs[f"{SHA}:{relative}"] = content
        (root / relative).write_bytes(content)
    else:
        del blobs[f"{SHA}:{relative}"]
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        run()
    assert not any(kind == "paid" for kind, _item in events)
    assert not (root / "outputs/course-release-batch/summary.json").exists()


def test_success_is_serial_with_unique_ids_and_separate_immutable_evidence(runner):
    root, _blobs, events, _controls, _approvals, _save, run = runner
    modal_output = root / "outputs/modal-course"
    modal_output.mkdir(parents=True)
    (modal_output / "assets.txt").write_text("unrelated existing file\n")
    summary = run()
    assert summary["status"] == "completed" and summary["not_started"] == []
    paid = [item for kind, item in events if kind == "paid"]
    assert paid == IDS
    first_paid = next(index for index, event in enumerate(events) if event[0] == "paid")
    assert [event[1] for event in events[:first_paid]] == [
        f"{SHA}:docs/course-experiments/plan.json",
        *[f"{SHA}:docs/course-experiments/releases/{name}.json" for name in IDS],
    ]
    for name, record in zip(IDS, summary["items"], strict=True):
        directory = root / f"outputs/course-release-batch/{name}"
        assert record["run_id"] == f"gha-123-2-{name}"
        assert record["status"] == "completed"
        assert json.loads((directory / "result.json").read_text())["experiment_id"] == name
        assert json.loads((directory / "billing-before.json").read_text())["test_run"] == name
        assert json.loads((directory / "public-manifest.json").read_text())["id"] == name
        assert (directory / "modal-stdout.txt").read_text() == f"release output for {name}\n"
        assert json.loads((directory / "validated-release.json").read_text())["release"]["gpu_used"] is False
        for evidence in record["evidence"]:
            raw = (directory / evidence["path"]).read_bytes()
            assert len(raw) == evidence["bytes"] and hashlib.sha256(raw).hexdigest() == evidence["sha256"]
    assert (modal_output / "assets.txt").read_text() == "unrelated existing file\n"
    assert all(not (modal_output / name).exists() for name in batch.GENERATED_FILES)


@pytest.mark.parametrize("failure", ["fail", "timeout", "invalid_gpu", "wrong_run"])
def test_first_failed_or_unverified_release_stops_without_retry_and_keeps_receipts(runner, failure):
    root, _blobs, events, controls, _approvals, _save, run = runner
    controls[failure] = "vqa"
    with pytest.raises(RuntimeError, match="已停止"):
        run()
    assert [item for kind, item in events if kind == "paid"] == ["sft", "vqa"]
    output = root / "outputs/course-release-batch"
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "failed" and summary["not_started"] == ["audio"]
    assert [item["status"] for item in summary["items"]] == ["completed", "failed"]
    assert json.loads((output / "sft/result.json").read_text())["experiment_id"] == "sft"
    assert json.loads((output / "vqa/result.json").read_text())["experiment_id"] == "vqa"
    assert not (output / "audio").exists()


def test_existing_modal_result_is_preserved_and_never_attributed_to_new_run(runner):
    root, _blobs, events, _controls, _approvals, _save, run = runner
    destination = root / "outputs/modal-course"
    destination.mkdir(parents=True)
    (destination / "result.json").write_text("previous run evidence")
    with pytest.raises(ValueError, match="既存"):
        run()
    assert (destination / "result.json").read_text() == "previous run evidence"
    assert not any(kind == "paid" for kind, _item in events)


@pytest.mark.parametrize("value", ['["sft","sft"]', "sft,,vqa", "sft;echo hello", "../sft", '["sft",2]', "[]"])
def test_malformed_duplicate_or_shell_shaped_id_list_is_rejected(value):
    with pytest.raises(ValueError):
        batch.parse_experiments(value)


def test_cli_and_workflow_are_release_only_without_training_dependencies():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "-I", "scripts/release_course_batch.py", "--help"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--validate-only" in completed.stdout and "--mode" not in completed.stdout
    workflow = (root / ".github/workflows/course-models-release.yml").read_text()
    assert "group: tiny-perceptron-course-gpu" in workflow and "cancel-in-progress: false" in workflow
    assert "contents: read" in workflow and "lfs: false" in workflow and "timeout-minutes: 180" in workflow
    assert "modal==1.6.0" in workflow and "scripts/select_modal_hf_secret.py" in workflow
    assert workflow.index("--validate-only") < workflow.index("pip install modal")
    assert "matrix:" not in workflow and "git lfs pull" not in workflow
    assert "if: always()" in workflow and "path: outputs/course-release-batch/" in workflow


def test_validate_only_cli_checks_all_approvals_without_a_paid_command(runner, monkeypatch, capsys):
    root, _blobs, events, _controls, _approvals, _save, _run = runner
    monkeypatch.setattr(batch, "ROOT", root)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "release_course_batch.py",
            "--experiments",
            "sft,vqa,audio",
            "--validate-only",
            "--revision",
            SHA,
            "--run-id-prefix",
            "gha-123-2",
            "--checkpoint-repo",
            "owner/private",
            "--release-repo",
            "owner/public",
        ],
    )
    assert batch.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "validated" and [item["experiment_id"] for item in result["items"]] == IDS
    assert len(events) == 4 and all(kind == "git" for kind, _item in events)


def test_registered_workflow_routes_batch_and_single_calls_exclusively():
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/course-experiments.yml").read_text()
    steps = {part.splitlines()[0]: part for part in workflow.split("      - name: ")[1:]}
    batch_condition = "if: inputs.mode == 'release' && (inputs.release_experiment_ids || '') != ''"
    single_condition = "if: inputs.mode != 'release' || (inputs.release_experiment_ids || '') == ''"
    assert batch_condition in steps["Validate all committed batch approvals before paid calls"]
    assert "--validate-only" in steps["Validate all committed batch approvals before paid calls"]
    assert batch_condition in steps["Release approved models one at a time"]
    assert single_condition in steps["Execute one bounded course experiment"]
    assert (
        "if: inputs.mode == 'release' && (inputs.release_experiment_ids || '') == ''"
        in steps["Validate the committed public release approval before paid calls"]
    )
    assert '"$EXPERIMENT_MODE" != "release"' in steps["Check settings without revealing credentials"]
    artifact = steps["Save experiment evidence and account-wide billing observations"]
    assert (
        "if: always()" in artifact
        and "outputs/course-release-batch/" in artifact
        and "outputs/modal-course/" in artifact
    )
    assert "group: tiny-perceptron-course-gpu" in workflow and "&& 180 || 35" in workflow
    assert workflow.index("--validate-only") < workflow.index("pip install modal")
