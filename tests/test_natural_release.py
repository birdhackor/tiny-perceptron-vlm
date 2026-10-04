"""Public fetch failures, immutable pins and inert test-only LoRA file layouts."""

import copy
import json
import struct
from pathlib import Path

import pytest

from scripts import fetch_natural_release as release


def fixture_weights(path, *, extra=None):
    name = "base_model.model.model.language_model.layers.0.self_attn.q_proj"
    header = {
        "__metadata__": {"format": "pt"},
        name + ".lora_A.weight": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]},
        name + ".lora_B.weight": {"dtype": "F32", "shape": [3, 2], "data_offsets": [24, 48]},
    }
    if extra:
        header.update(extra)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(48))


@pytest.fixture
def public_fixture(tmp_path):
    directory = tmp_path / "public-source"
    directory.mkdir()
    versions = {}
    for line in release.REQUIREMENTS.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            name, version = line.split("==")
            versions[name.lower()] = version
    manifest = {
        "schema_version": 1,
        "release_id": "fixture",
        "reviewed": True,
        "anonymous_download_verified": True,
        "repo": "test-owner/public-fixture",
        "revision": "f" * 40,
        "prefix": "natural-v3/fixture",
        "git_revision": "a" * 40,
        "manifest_sha256": "b" * 64,
        "approval_sha256": "c" * 64,
        "requirements_sha256": release.sha256(release.REQUIREMENTS),
        "base_model": {"repo": "test-owner/base-fixture", "revision": "d" * 40},
        "asr_model": {"repo": "test-owner/asr-fixture", "revision": "e" * 40},
        "dependency_versions": versions,
        "runtime": {
            "python": "3.12",
            "min_pixels": 65536,
            "max_pixels": 524288,
            "max_tokens": 2048,
            "max_new_tokens": 96,
            "seed": 42,
        },
        "adapter_parameters": 12,
        "code_files": {path: release.sha256(release.ROOT / path) for path in release.CODE_FILES},
    }
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "inference_mode": True,
        "base_model_name_or_path": manifest["base_model"]["repo"],
        "revision": None,
        "r": 2,
        "target_modules": release.LORA_TARGETS,
        "modules_to_save": None,
        "auto_mapping": None,
        "bias": "none",
        "use_dora": False,
        "rank_pattern": {},
    }
    (directory / "adapter_config.json").write_text(json.dumps(config))
    fixture_weights(directory / "adapter_model.safetensors")
    (directory / "README.md").write_text("Test-generated tiny adapter fixture, not model-capability evidence.\n")
    provenance = {
        key: manifest[key] for key in ("approval_sha256", "git_revision", "manifest_sha256", "base_model", "asr_model")
    }
    provenance["source"] = {"repo": "test-owner/private-fixture", "revision": "1" * 40, "prefix": "natural-v3/fixture"}
    (directory / "release-provenance.json").write_text(json.dumps(provenance))
    manifest["files"] = [
        {
            "path": manifest["prefix"] + "/" + path.name,
            "output": path.name,
            "bytes": path.stat().st_size,
            "sha256": release.sha256(path),
            "license": "MIT",
            "redistribution_approved": True,
        }
        for path in sorted(directory.iterdir())
    ]
    calls = []

    def download(repo, filename, **options):
        calls.append((repo, filename, options))
        return directory / Path(filename).name

    return directory, manifest, download, calls


def refreshed(manifest, directory, filename):
    manifest = copy.deepcopy(manifest)
    item = next(item for item in manifest["files"] if item["output"] == filename)
    path = directory / filename
    item.update(bytes=path.stat().st_size, sha256=release.sha256(path))
    return manifest


def test_fetch_uses_only_exact_public_pin_and_verifies_complete_folder(public_fixture, tmp_path):
    directory, manifest, download, calls = public_fixture
    output = tmp_path / "download"
    result = release.fetch_release(manifest, output, downloader=download)
    assert result == output
    assert len(calls) == len(manifest["files"])
    assert all(
        repo == manifest["repo"] and options == {"revision": manifest["revision"], "token": False}
        for repo, _, options in calls
    )
    assert {name for _, name, _ in calls} == {item["path"] for item in manifest["files"]}
    assert release.verify_release(manifest, result) == output
    assert release.read_json(output / "verified-release.json") == manifest
    assert not {"training_state.pt", "optimizer.pt", "training.json"} & {path.name for path in output.iterdir()}
    for item in manifest["files"]:
        assert (output / item["output"]).read_bytes() == (directory / item["output"]).read_bytes()


@pytest.mark.parametrize("mutation", ["sha", "size", "interruption"])
def test_mismatch_or_interruption_never_leaves_partial_student_folder(public_fixture, tmp_path, mutation):
    directory, manifest, download, calls = public_fixture
    output = tmp_path / "download"
    if mutation == "sha":
        file = directory / "adapter_model.safetensors"
        content = file.read_bytes()
        file.write_bytes(content[:-1] + b"x")
    elif mutation == "size":
        with (directory / "README.md").open("a") as stream:
            stream.write("changed")
    else:
        original = download

        def download(*args, **kwargs):
            if calls:
                raise OSError("fixture network interruption")
            return original(*args, **kwargs)

    with pytest.raises((ValueError, OSError)):
        release.fetch_release(manifest, output, downloader=download)
    assert not output.exists()
    assert not list(tmp_path.glob(".natural-release-*"))


@pytest.mark.parametrize(
    "path",
    ["../adapter.json", "/tmp/adapter.json", "a\\b", "C:/model.json", "a/../b", "a//b", ".", "a\x00b", "%2e%2e/file"],
)
def test_path_traversal_is_rejected(path):
    with pytest.raises(ValueError):
        release.safe_relative(path)


@pytest.mark.parametrize(
    "change",
    [
        "mutable",
        "review",
        "anonymous",
        "outside",
        "traversal",
        "duplicate",
        "archive",
        "optimizer",
        "license",
        "base-pin",
        "count",
        "runtime",
        "repo-type",
    ],
)
def test_invalid_release_cannot_start_any_download(public_fixture, tmp_path, change):
    _, manifest, download, calls = public_fixture
    manifest = copy.deepcopy(manifest)
    if change == "mutable":
        manifest["revision"] = "main"
    elif change == "review":
        manifest["reviewed"] = False
    elif change == "anonymous":
        manifest["anonymous_download_verified"] = False
    elif change == "outside":
        manifest["files"][0]["path"] = "other-release/adapter_config.json"
    elif change == "traversal":
        manifest["files"][0]["output"] = "../adapter_config.json"
    elif change == "duplicate":
        manifest["files"].append(copy.deepcopy(manifest["files"][0]))
    elif change == "archive":
        manifest["files"][0].update(path=manifest["prefix"] + "/weights.tar", output="weights.tar")
    elif change == "optimizer":
        manifest["files"][0].update(path=manifest["prefix"] + "/optimizer.json", output="optimizer.json")
    elif change == "license":
        manifest["files"][0]["redistribution_approved"] = False
    elif change == "base-pin":
        manifest["base_model"]["revision"] = "latest"
    elif change == "count":
        manifest["adapter_parameters"] = True
    elif change == "runtime":
        manifest["runtime"] = []
    else:
        manifest["repo"] = False
    with pytest.raises(ValueError):
        release.fetch_release(manifest, tmp_path / "download", downloader=download)
    assert calls == []


@pytest.mark.parametrize(
    "change",
    ["pickle", "missing-pair", "bad-rank", "provenance", "config-base", "config-target", "empty-card", "bad-header"],
)
def test_reviewed_file_hash_alone_cannot_disguise_wrong_model_or_private_state(public_fixture, tmp_path, change):
    directory, manifest, download, _ = public_fixture
    filename = "adapter_model.safetensors"
    if change == "pickle":
        fixture_weights(
            directory / filename, extra={"optimizer.secret": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}}
        )
    elif change == "missing-pair":
        file = directory / filename
        data = file.read_bytes()
        length = struct.unpack("<Q", data[:8])[0]
        header = json.loads(data[8 : 8 + length])
        key = next(key for key in header if ".lora_B." in key)
        del header[key]
        encoded = json.dumps(header).encode()
        file.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(24))
        manifest["adapter_parameters"] = 6
    elif change == "bad-rank":
        filename = "adapter_config.json"
        config = release.read_json(directory / filename)
        config["r"] = 3
        (directory / filename).write_text(json.dumps(config))
    elif change == "provenance":
        filename = "release-provenance.json"
        provenance = release.read_json(directory / filename)
        provenance["base_model"]["revision"] = "2" * 40
        (directory / filename).write_text(json.dumps(provenance))
    elif change in {"config-base", "config-target"}:
        filename = "adapter_config.json"
        config = release.read_json(directory / filename)
        config["base_model_name_or_path" if change == "config-base" else "target_modules"] = "other-model"
        (directory / filename).write_text(json.dumps(config))
    elif change == "empty-card":
        filename = "README.md"
        (directory / filename).write_text("\n")
    else:
        (directory / filename).write_bytes(struct.pack("<Q", release.MAX_HEADER_BYTES + 1) + b"{}")
    manifest = refreshed(manifest, directory, filename)
    with pytest.raises(ValueError):
        release.fetch_release(manifest, tmp_path / "download", downloader=download)
    assert not (tmp_path / "download").exists()


def test_existing_or_symlink_destination_cannot_be_overwritten(public_fixture, tmp_path):
    _, manifest, download, calls = public_fixture
    output = tmp_path / "kept"
    output.mkdir()
    (output / "mine.txt").write_text("Keep my work")
    with pytest.raises(ValueError, match="exists"):
        release.fetch_release(manifest, output, downloader=download)
    assert (output / "mine.txt").read_text() == "Keep my work"
    link = tmp_path / "link"
    try:
        link.symlink_to(output, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("This machine cannot create directory symlinks")
    with pytest.raises(ValueError, match="symbolic"):
        release.fetch_release(manifest, link / "new", downloader=download)
    assert calls == []


def test_public_version_file_must_match_declared_versions_and_contain_no_extra_state(public_fixture, tmp_path):
    directory, manifest, download, _ = public_fixture
    file = directory / "runtime-versions.json"
    file.write_text(json.dumps({"python": "3.12", "dependencies": manifest["dependency_versions"]}))
    manifest["files"].append(
        {
            "path": manifest["prefix"] + "/" + file.name,
            "output": file.name,
            "sha256": release.sha256(file),
            "bytes": file.stat().st_size,
            "license": "MIT",
            "redistribution_approved": True,
        }
    )
    release.fetch_release(manifest, tmp_path / "correct", downloader=download)
    versions = release.read_json(file)
    versions["optimizer"] = {"state": "private fixture"}
    file.write_text(json.dumps(versions))
    manifest = refreshed(manifest, directory, file.name)
    with pytest.raises(ValueError, match="versions"):
        release.fetch_release(manifest, tmp_path / "rejected", downloader=download)
    assert not (tmp_path / "rejected").exists()


def test_verify_rejects_modified_or_extra_files_and_mismatched_receipt(public_fixture, tmp_path):
    _, manifest, download, _ = public_fixture
    output = release.fetch_release(manifest, tmp_path / "download", downloader=download)
    extra = output / "training_state.pt"
    extra.write_bytes(b"not published")
    with pytest.raises(ValueError, match="exactly"):
        release.verify_release(manifest, output)
    extra.unlink()
    receipt = copy.deepcopy(manifest)
    receipt["revision"] = "0" * 40
    (output / "verified-release.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="receipt"):
        release.verify_release(manifest, output)


def test_loader_uses_manifest_pins_limits_and_cpu_dtype(public_fixture):
    _, manifest, _, _ = public_fixture
    options = release.student_options(manifest, Path("adapter"), device="cpu")
    assert options.model == manifest["base_model"]["repo"]
    assert options.model_revision == manifest["base_model"]["revision"]
    assert options.asr_revision == manifest["asr_model"]["revision"]
    assert options.dtype == "float32"
    assert options.max_pixels == manifest["runtime"]["max_pixels"]
    assert options.max_new_tokens == manifest["runtime"]["max_new_tokens"]
    assert options.data_root != options.adapter
    with pytest.raises(ValueError):
        release.student_options(manifest, "adapter", device="cpu", dtype="float16")


def test_runtime_guard_catches_wrong_venv_stale_dependencies_or_source(public_fixture, monkeypatch):
    _, manifest, _, _ = public_fixture
    monkeypatch.setattr(release.sys, "version_info", (3, 13))
    with pytest.raises(ValueError, match="3.12"):
        release.check_runtime(manifest)
    monkeypatch.setattr(release.sys, "version_info", (3, 12))
    monkeypatch.setattr(
        release.importlib.metadata,
        "version",
        lambda name: manifest["dependency_versions"][name] + ("+cu128" if name == "torch" else ""),
    )
    release.check_runtime(manifest)
    monkeypatch.setattr(release.importlib.metadata, "version", lambda name: "0.0.1")
    with pytest.raises(ValueError, match="torch"):
        release.check_runtime(manifest)
    monkeypatch.setattr(release.importlib.metadata, "version", lambda name: manifest["dependency_versions"][name])
    changed = copy.deepcopy(manifest)
    changed["code_files"]["tiny_perceptron/natural_ui.py"] = "0" * 64
    with pytest.raises(ValueError, match="程式"):
        release.check_runtime(changed)


def test_cli_unpublished_release_refuses_to_guess_pins(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        release.sys, "argv", ["fetch_natural_release.py", "--manifest", str(tmp_path / "not-published.json")]
    )
    with pytest.raises(SystemExit) as error:
        release.main()
    assert error.value.code == 2
    assert "尚未發布" in capsys.readouterr().err
