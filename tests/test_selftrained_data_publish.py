"""Native Git LFS publication gates without a remote push or credentials."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/selftrained/publish_data.py"
SPEC = importlib.util.spec_from_file_location("selftrained_publish_test", SOURCE)
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, capture_output=True, check=True, text=True).stdout.strip()


@pytest.fixture
def repository(tmp_path):
    git(tmp_path, "init", "--initial-branch=work")
    git(tmp_path, "config", "user.name", "Publication test")
    git(tmp_path, "config", "user.email", "test@example.invalid")
    # Match the checkout's LF contract even when Git inherits core.autocrlf.
    (tmp_path / ".gitattributes").write_text("* text=auto eol=lf\n", encoding="utf-8", newline="\n")
    git(tmp_path, "add", ".gitattributes")
    return tmp_path


def recipe():
    return {
        "schema_version": 1,
        "frozen": True,
        "python_version": "3.13.5",
        "dependencies": {"numpy": "2.5.3"},
        "producers": [{"path": "scripts/selftrained/prepare_voice.py", "sha256": "a" * 64, "args": []}],
        "source_files": [
            {"source": "outputs/selftrained/data/voice.jsonl", "path": "voice.jsonl", "sha256": "b" * 64, "bytes": 20}
        ],
        "archive": {
            "path": "assets/training/selftrained-v1.tar.gz",
            "sha256": "c" * 64,
            "bytes": 100,
            "unpacked_bytes": 20,
        },
    }


def commit(root, path, value, message="fixture"):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value) + "\n", encoding="utf-8", newline="\n")
    git(root, "add", "--", path)
    git(root, "commit", "-m", message)
    return git(root, "rev-parse", "HEAD")


def test_initial_and_workflow_only_push_cannot_publish_existing_control(repository):
    commit(repository, "docs/selftrained/package-recipe.json", recipe())
    before = commit(
        repository,
        publisher.CONTROL,
        {
            "operation": "publish",
            "approved": True,
            "recipe": "docs/selftrained/package-recipe.json",
            "recipe_sha256": publisher.digest(repository / "docs/selftrained/package-recipe.json"),
        },
    )
    revision = commit(repository, ".github/workflows/selftrained-data.yml", {"modified": True})
    selection = publisher.select_operation(repository, {"event_name": "push", "before": before}, {}, revision)
    assert selection["operation"] == "inspect"
    assert selection["publication_started"] is False and selection["modal_used"] is False
    assert (
        publisher.select_operation(repository, {"event_name": "push", "before": "0" * 40}, {}, revision)["operation"]
        == "inspect"
    )


def test_changed_approved_descriptor_binds_exact_recipe_before_publish(repository):
    before = commit(repository, "docs/selftrained/package-recipe.json", recipe())
    control = {
        "operation": "publish",
        "approved": True,
        "recipe": "docs/selftrained/package-recipe.json",
        "recipe_sha256": publisher.digest(repository / "docs/selftrained/package-recipe.json"),
    }
    revision = commit(repository, publisher.CONTROL, control)
    selection = publisher.select_operation(repository, {"event_name": "push", "before": before}, {}, revision)
    assert selection["operation"] == "publish" and selection["recipe_sha256"] == control["recipe_sha256"]
    changed = commit(repository, publisher.CONTROL, control | {"recipe_sha256": "f" * 64})
    with pytest.raises(ValueError, match="exact committed recipe"):
        publisher.select_operation(repository, {"event_name": "push", "before": revision}, {}, changed)


def test_unfrozen_or_url_dependency_and_oversized_packages_are_rejected():
    for item in (
        recipe() | {"frozen": False},
        recipe() | {"dependencies": {"numpy": "https://untrusted.invalid"}},
        recipe() | {"archive": recipe()["archive"] | {"bytes": 2**40}},
    ):
        with pytest.raises(ValueError):
            publisher.validate_recipe(item)


def test_publication_verifies_actual_archive_and_committed_native_pointer(repository):
    output = repository / "outputs/selftrained/data-publish"
    output.mkdir(parents=True)
    actual = output / "selftrained-v1.tar.gz"
    actual.write_bytes(b"verified frozen archive fixture")
    value = recipe()
    value["archive"].update(bytes=actual.stat().st_size, sha256=publisher.digest(actual))
    path = repository / value["archive"]["path"]
    path.parent.mkdir(parents=True)
    pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{value['archive']['sha256']}\nsize {value['archive']['bytes']}\n"
    path.write_text(pointer, encoding="utf-8", newline="\n")
    git(repository, "add", "--", value["archive"]["path"])
    git(repository, "commit", "-m", "exact pointer")
    revision = git(repository, "rev-parse", "HEAD")
    verified, selected, expected, pointer_present = publisher.verify_archive(repository, value, revision, output)
    assert verified == actual and selected.as_posix() == value["archive"]["path"] and expected == pointer.encode()
    assert pointer_present is True
    actual.write_bytes(b"different archive")
    with pytest.raises(ValueError, match="Reconstructed archive"):
        publisher.verify_archive(repository, value, revision, output)


def test_missing_pointer_bootstrap_uploads_native_local_lfs_object_without_commit(repository, monkeypatch):
    # Keep real local Git/LFS staging. Replace only the external push operation.
    with (repository / ".gitattributes").open("a", encoding="utf-8", newline="\n") as attributes:
        attributes.write("/assets/training/*.tar.gz filter=lfs diff=lfs merge=lfs -text\n")
    git(repository, "add", ".gitattributes")
    git(repository, "commit", "-m", "LFS attribute")
    output = repository / "outputs/selftrained/data-publish"
    output.mkdir(parents=True)
    actual = output / "selftrained-v1.tar.gz"
    actual.write_bytes(b"native verified LFS fixture")
    value = recipe()
    value["archive"].update(bytes=actual.stat().st_size, sha256=publisher.digest(actual))
    revision = commit(repository, "docs/selftrained/package-recipe.json", value)
    _, _, _, pointer_present = publisher.verify_archive(repository, value, revision, output)
    assert pointer_present is False
    original_run, pushed = subprocess.run, []

    def run(command, **kwargs):
        if command[:3] == ["git", "lfs", "push"]:
            pushed.append(command)
            return subprocess.CompletedProcess(command, 0)
        return original_run(command, **kwargs)

    monkeypatch.setattr(publisher.subprocess, "run", run)
    result = publisher.publish(repository, "docs/selftrained/package-recipe.json", revision, output)
    assert result["lfs_upload_completed"] is True and result["pointer_not_committed_yet"] is True
    assert pushed == [["git", "lfs", "push", "--object-id", "origin", value["archive"]["sha256"]]]
    assert git(repository, "rev-parse", "HEAD") == revision
    staged = git(repository, "show", ":" + value["archive"]["path"])
    assert f"oid sha256:{value['archive']['sha256']}" in staged


def test_existing_wrong_pointer_is_rejected_before_upload(repository):
    output = repository / "outputs/selftrained/data-publish"
    output.mkdir(parents=True)
    actual = output / "selftrained-v1.tar.gz"
    actual.write_bytes(b"reviewed object")
    value = recipe()
    value["archive"].update(bytes=actual.stat().st_size, sha256=publisher.digest(actual))
    path = repository / value["archive"]["path"]
    path.parent.mkdir(parents=True)
    path.write_text(
        f"version https://git-lfs.github.com/spec/v1\noid sha256:{'f' * 64}\nsize {value['archive']['bytes']}\n",
        encoding="utf-8",
        newline="\n",
    )
    git(repository, "add", "--", value["archive"]["path"])
    git(repository, "commit", "-m", "wrong pointer")
    revision = git(repository, "rev-parse", "HEAD")
    with pytest.raises(ValueError, match="exact reviewed LFS pointer"):
        publisher.verify_archive(repository, value, revision, output)
