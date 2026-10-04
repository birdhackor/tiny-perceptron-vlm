"""Execute workflow preflight with mocked Git; stop before every upload boundary."""

import copy
import hashlib
import json
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/natural-data-v4.yml"
DOCUMENT = yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)
STEPS = DOCUMENT["jobs"]["publish-data"]["steps"]
UPLOAD_STEP = next(step for step in STEPS if step.get("name", "").startswith("Verify committed LFS pointers"))
UPLOAD_CODE = re.search(r"python - <<'PY'\n(.*?)\nPY(?:\n|$)", UPLOAD_STEP["run"], re.S).group(1)
EXPECTED_PATHS = {
    "assets/training/natural-vision-v4.tar.gz",
    "assets/training/natural-ocr-v4.tar.gz",
    "assets/training/natural-voice-v4.tar.gz",
}


class MockUploadBoundary(RuntimeError):
    """The subprocess mock intentionally stops before an upload could run."""


@pytest.fixture
def workflow_fixture(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_SHA", "c" * 40)
    archives, pointers, events = [], {}, []
    for name in sorted(EXPECTED_PATHS):
        raw = (name + " fixed byte fixture\n").encode()
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        archives.append({"path": name, "bytes": len(raw), "sha256": digest})
        pointers[name] = (
            f"version https://git-lfs.github.com/spec/v1\noid sha256:{digest}\nsize {len(raw)}\n"
        ).encode()
    manifest_path = tmp_path / "docs/natural-assistant/v4/manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest = {"archives": archives}
    manifest_path.write_text(json.dumps(manifest) + "\n")
    state = {"head_bad": None, "index_bad": None, "changed": []}

    def check_output(command):
        events.append(command)
        assert command[:2] == ["git", "show"] or command == ["git", "diff", "--cached", "--name-only"]
        if command[1] == "diff":
            return "\n".join(state["changed"]).encode()
        selector = command[2]
        head = selector.startswith("HEAD:")
        name = selector.removeprefix("HEAD:") if head else selector.removeprefix(":")
        assert name in EXPECTED_PATHS
        if name == state["head_bad" if head else "index_bad"]:
            return b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"0" * 64 + b"\nsize 1\n"
        return pointers[name]

    def run(command, *, check):
        events.append(command)
        assert check is True
        if command[:4] == ["git", "add", "--", command[-1]]:
            assert command[-1] in EXPECTED_PATHS
            return SimpleNamespace(returncode=0)
        assert command[:5] == ["git", "lfs", "push", "--object-id", "origin"]
        raise MockUploadBoundary("Mock stopped at upload boundary; no object was uploaded")

    monkeypatch.setattr(subprocess, "check_output", check_output)
    monkeypatch.setattr(subprocess, "run", run)
    return {"root": tmp_path, "manifest": manifest, "manifest_path": manifest_path, "state": state, "events": events}


def execute_embedded(fixture):
    namespace = {"__name__": "workflow_preflight_fixture"}
    exec(compile(UPLOAD_CODE, str(WORKFLOW) + ":embedded-python", "exec"), namespace)


def assert_no_upload_or_success_receipt(fixture):
    assert not any(command[:3] == ["git", "lfs", "push"] for command in fixture["events"])
    assert not (fixture["root"] / "outputs/natural-v4/data-publish/upload-receipt.json").exists()


def test_workflow_trigger_cpu_dependencies_and_commands_stay_in_authorized_scope():
    assert DOCUMENT["on"] == {
        "workflow_dispatch": "",
    }
    assert DOCUMENT["jobs"]["publish-data"]["runs-on"] == "ubuntu-latest"
    assert int(DOCUMENT["jobs"]["publish-data"]["timeout-minutes"]) == 40
    checkout = next(step for step in STEPS if step.get("uses", "").startswith("actions/checkout@"))
    assert checkout["with"]["lfs"] == "false"
    commands = "\n".join(step.get("run", "") for step in STEPS)
    assert "--verify-manifest docs/natural-assistant/v4/manifest.json" in commands
    for dependency in ("pillow==12.3.0", "h5py==3.16.0", "numpy==2.5.3", "soundfile==0.14.0"):
        assert dependency in commands
    assert "git lfs install --local" in commands
    assert not re.search(r"git\s+commit\b", commands)
    for disallowed in (
        "secrets.",
        "HF_TOKEN",
        "HUGGINGFACE_TOKEN",
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "modal run",
    ):
        assert disallowed not in WORKFLOW.read_text()


@pytest.mark.parametrize("location", ["head_bad", "index_bad"])
def test_embedded_preflight_rejects_mismatched_head_or_native_index_pointer(workflow_fixture, location):
    workflow_fixture["state"][location] = sorted(EXPECTED_PATHS)[1]
    with pytest.raises(RuntimeError, match="pointer"):
        execute_embedded(workflow_fixture)
    assert_no_upload_or_success_receipt(workflow_fixture)


def test_embedded_preflight_rejects_changed_actual_archive_bytes(workflow_fixture):
    path = workflow_fixture["root"] / sorted(EXPECTED_PATHS)[-1]
    raw = path.read_bytes()
    path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    with pytest.raises(RuntimeError, match="archive differs"):
        execute_embedded(workflow_fixture)
    assert_no_upload_or_success_receipt(workflow_fixture)


def test_embedded_preflight_rejects_unrelated_staged_paths(workflow_fixture):
    workflow_fixture["state"]["changed"] = ["docs/unrelated.txt"]
    with pytest.raises(RuntimeError, match="unrelated"):
        execute_embedded(workflow_fixture)
    assert_no_upload_or_success_receipt(workflow_fixture)


@pytest.mark.parametrize("change", ["v3", "duplicate", "missing", "unreviewed"])
def test_embedded_preflight_requires_exactly_three_unique_reviewed_v4_paths(workflow_fixture, change):
    manifest = copy.deepcopy(workflow_fixture["manifest"])
    if change == "duplicate":
        manifest["archives"].append(copy.deepcopy(manifest["archives"][0]))
    elif change == "missing":
        manifest["archives"].pop()
    elif change == "v3":
        manifest["archives"][0]["path"] = manifest["archives"][0]["path"].replace("-v4.", "-v3.")
    else:
        manifest["archives"][0]["path"] = "assets/training/unreviewed-v4.tar.gz"
    workflow_fixture["manifest_path"].write_text(json.dumps(manifest) + "\n")
    with pytest.raises(RuntimeError, match="three"):
        execute_embedded(workflow_fixture)
    assert_no_upload_or_success_receipt(workflow_fixture)


def test_valid_preflight_checks_all_head_and_index_pointers_before_mock_upload_boundary(workflow_fixture):
    with pytest.raises(MockUploadBoundary, match="no object was uploaded"):
        execute_embedded(workflow_fixture)
    events = workflow_fixture["events"]
    command = events[-1]
    assert command == [
        "git",
        "lfs",
        "push",
        "--object-id",
        "origin",
        *[item["sha256"] for item in workflow_fixture["manifest"]["archives"]],
    ]
    for name in EXPECTED_PATHS:
        assert ["git", "show", "HEAD:" + name] in events[:-1]
        assert ["git", "add", "--", name] in events[:-1]
        assert ["git", "show", ":" + name] in events[:-1]
        assert (
            events.index(["git", "show", "HEAD:" + name])
            < events.index(["git", "add", "--", name])
            < events.index(["git", "show", ":" + name])
            < len(events) - 1
        )
    assert not any("commit" in event for event in events)
    assert not (workflow_fixture["root"] / "outputs/natural-v4/data-publish/upload-receipt.json").exists()
