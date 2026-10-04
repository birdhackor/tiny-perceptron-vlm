"""Base publication and student route contracts; mock network/model, real files."""

import ast
import copy
import hashlib
import json
import re
import shutil
import struct
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import fetch_natural_release as release
from tiny_perceptron import natural_ui


@pytest.fixture
def base_release(tmp_path):
    directory = tmp_path / "public-base"
    directory.mkdir()
    versions = {}
    for line in release.REQUIREMENTS.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            name, version = line.split("==")
            versions[name.lower()] = version
    manifest = {
        "schema_version": 1,
        "release_id": "base-fixture",
        "selected_variant": "base",
        "reviewed": True,
        "anonymous_download_verified": True,
        "repo": "fixture/public",
        "revision": "f" * 40,
        "prefix": "natural-v3/base-fixture",
        "git_revision": "a" * 40,
        "manifest_sha256": "b" * 64,
        "approval_sha256": "c" * 64,
        "requirements_sha256": release.sha256(release.REQUIREMENTS),
        "base_model": {"repo": "fixture/upstream-base", "revision": "d" * 40},
        "asr_model": {"repo": "fixture/upstream-asr", "revision": "e" * 40},
        "dependency_versions": versions,
        "runtime": {
            "python": "3.12",
            "min_pixels": 65536,
            "max_pixels": 524288,
            "max_tokens": 2048,
            "max_new_tokens": 384,
            "seed": 42,
        },
        "adapter_parameters": 0,
        "code_files": {name: release.sha256(release.ROOT / name) for name in release.CODE_FILES},
    }
    provenance = {
        name: manifest[name]
        for name in (
            "approval_sha256",
            "git_revision",
            "manifest_sha256",
            "base_model",
            "asr_model",
            "selected_variant",
        )
    }
    provenance["source"] = None
    (directory / "release-provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    (directory / "README.md").write_text("Test fixture: selected pinned base, no LoRA.\n", encoding="utf-8")
    manifest["files"] = [
        {
            "path": manifest["prefix"] + "/" + file.name,
            "output": file.name,
            "bytes": file.stat().st_size,
            "sha256": release.sha256(file),
            "redistribution_approved": True,
            "license": "MIT",
        }
        for file in sorted(directory.iterdir())
    ]
    calls = []

    def download(repo, filename, **options):
        calls.append((repo, filename, options))
        return directory / Path(filename).name

    return manifest, directory, download, calls


def test_base_fetch_and_real_ui_entry_load_pinned_core_without_peft(base_release, tmp_path, monkeypatch):
    manifest, _, download, calls = base_release
    fetched = release.fetch_release(manifest, tmp_path / "student", downloader=download)
    assert {file.name for file in fetched.iterdir()} == {
        "README.md",
        "release-provenance.json",
        "verified-release.json",
    }
    assert len(calls) == 2
    assert all(options == {"revision": manifest["revision"], "token": False} for _, _, options in calls)
    options = release.student_options(manifest, fetched, device="cpu", local_files_only=True)
    assert options.adapter is None
    assert options.asr_model == manifest["asr_model"]["repo"]
    assert options.asr_revision == manifest["asr_model"]["revision"]
    model_calls = []
    processor = object()

    class Model:
        def to(self, device):
            assert device == "cpu"
            return self

        def requires_grad_(self, enabled):
            assert enabled is False

        def eval(self):
            model_calls.append("eval")

    model = Model()

    def load_processor(repo, **kwargs):
        model_calls.append(("processor", repo, kwargs))
        return processor

    def load_model(repo, **kwargs):
        model_calls.append(("model", repo, kwargs))
        return model

    def unexpected_adapter(*args, **kwargs):
        pytest.fail("Selected base must never import/load a PEFT adapter")

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoProcessor=SimpleNamespace(from_pretrained=load_processor),
            Qwen3VLForConditionalGeneration=SimpleNamespace(from_pretrained=load_model),
        ),
    )
    monkeypatch.setitem(
        sys.modules, "peft", SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=unexpected_adapter))
    )
    server_calls = []

    class Server:
        server_address = ("127.0.0.1", 8766)
        server_port = 8766

        def serve_forever(self):
            server_calls.append("start")
            raise KeyboardInterrupt

        def server_close(self):
            server_calls.append("close")

    def create_server(actual_model, actual_processor, actual_options, *args):
        assert actual_model is model and actual_processor is processor
        assert actual_options is options and actual_options.adapter is None
        return Server()

    monkeypatch.setattr(natural_ui, "create_server", create_server)
    natural_ui.serve(options)
    assert server_calls == ["start", "close"]
    assert model_calls[-1] == "eval"
    for kind, repo, kwargs in model_calls[:-1]:
        assert repo == manifest["base_model"]["repo"]
        assert kwargs["revision"] == manifest["base_model"]["revision"]
        assert kwargs["local_files_only"] is True
        if kind == "processor":
            assert kwargs["max_pixels"] == manifest["runtime"]["max_pixels"]


@pytest.mark.parametrize(
    "change", ["parameters", "bool_parameters", "adapter_file", "unknown_variant", "implicit_base"]
)
def test_base_manifest_cannot_disguise_adapter_or_ambiguous_selection(base_release, change):
    original, _, _, _ = base_release
    manifest = copy.deepcopy(original)
    if change == "parameters":
        manifest["adapter_parameters"] = 12
    elif change == "bool_parameters":
        manifest["adapter_parameters"] = False
    elif change == "adapter_file":
        manifest["files"].append(
            manifest["files"][0]
            | {"output": "adapter_config.json", "path": manifest["prefix"] + "/adapter_config.json"}
        )
    elif change == "unknown_variant":
        manifest["selected_variant"] = {"base": True}
    else:
        del manifest["selected_variant"]
    with pytest.raises(ValueError):
        release.validate_manifest(manifest)


@pytest.mark.parametrize("change", ["private_source", "variant", "modified_card", "hidden_adapter"])
def test_base_verifier_keeps_sha_provenance_and_exact_file_guards(base_release, tmp_path, change):
    original, directory, download, _ = base_release
    manifest = copy.deepcopy(original)
    if change in {"private_source", "variant"}:
        provenance_file = directory / "release-provenance.json"
        provenance = release.read_json(provenance_file)
        if change == "private_source":
            provenance["source"] = {"repo": "fixture/private", "revision": "1" * 40, "prefix": "natural-v3/fake"}
        else:
            provenance["selected_variant"] = "adapter-step-001039"
        provenance_file.write_text(json.dumps(provenance), encoding="utf-8")
        item = next(item for item in manifest["files"] if item["output"] == provenance_file.name)
        item.update(bytes=provenance_file.stat().st_size, sha256=release.sha256(provenance_file))
    elif change == "modified_card":
        (directory / "README.md").write_text("Modified after approval", encoding="utf-8")
    else:
        output = release.fetch_release(manifest, tmp_path / "student", downloader=download)
        (output / "adapter_model.safetensors").write_bytes(b"hidden")
        with pytest.raises(ValueError, match="exactly"):
            release.verify_release(manifest, output)
        return
    with pytest.raises(ValueError):
        release.fetch_release(manifest, tmp_path / "student", downloader=download)
    assert not (tmp_path / "student").exists()


def test_explicit_archived_adapter_fetch_still_verifies_real_tensor_layout(base_release, tmp_path):
    original, directory, download, _ = base_release
    manifest = copy.deepcopy(original) | {"selected_variant": "adapter-step-001039", "adapter_parameters": 12}
    provenance = release.read_json(directory / "release-provenance.json")
    provenance.update(
        selected_variant=manifest["selected_variant"],
        source={"repo": "fixture/private", "revision": "1" * 40, "prefix": "natural-v3/v4/train/run-1"},
    )
    (directory / "release-provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "inference_mode": True,
        "base_model_name_or_path": manifest["base_model"]["repo"],
        "r": 2,
        "target_modules": release.LORA_TARGETS,
    }
    (directory / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    name = "base_model.model.model.language_model.layers.0.self_attn.q_proj"
    header = {
        "__metadata__": {"format": "pt"},
        name + ".lora_A.weight": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]},
        name + ".lora_B.weight": {"dtype": "F32", "shape": [3, 2], "data_offsets": [24, 48]},
    }
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    (directory / "adapter_model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(48))
    manifest["files"] = [
        {
            "path": manifest["prefix"] + "/" + file.name,
            "output": file.name,
            "bytes": file.stat().st_size,
            "sha256": release.sha256(file),
            "redistribution_approved": True,
            "license": "MIT",
        }
        for file in sorted(directory.iterdir())
    ]
    fetched = release.fetch_release(manifest, tmp_path / "student", downloader=download)
    assert release.verify_release(manifest, fetched) == fetched
    assert release.student_options(manifest, fetched, device="cpu").adapter == fetched


def publisher_helpers(tmp_path):
    """Execute actual undecorated CPU publisher without Modal, tokens or network."""
    names = {
        "safe_name",
        "safe_relative",
        "sha256",
        "write_json",
        "checkpoint_names",
        "validate_release",
        "release_remote",
    }
    source = release.ROOT / "scripts/modal_natural.py"
    nodes = []
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node = copy.deepcopy(node)
            node.decorator_list = []
            nodes.append(node)
    calls = []
    namespace = {
        "Path": Path,
        "json": json,
        "re": re,
        "hashlib": hashlib,
        "shutil": shutil,
        "tempfile": tempfile,
        "os": SimpleNamespace(environ={"HF_TOKEN": "test-placeholder"}),
        "MAX_PRIVATE_BACKUP_BYTES": 128 * 1024 * 1024,
        "NATURAL_ROOT": tmp_path / "course",
        "volume": SimpleNamespace(reload=lambda: calls.append("reload"), commit=lambda: calls.append("commit")),
        "require_reservation": lambda *args: calls.append(("reservation", args)),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace, calls


def release_approval(manifest):
    return {
        "approved": True,
        "reviewed": True,
        "release_id": manifest["release_id"],
        "selected_variant": "base",
        "source": None,
        "files": [],
        "model_card": "Selected upstream pinned base. Test fixture, not a trained adapter.",
        "base_model": manifest["base_model"],
        "asr_model": manifest["asr_model"],
    }


@pytest.mark.parametrize("variant", ["base", "legacy_adapter", "adapter-step-001039"])
def test_real_cpu_publisher_keeps_base_and_adapter_paths_distinct(base_release, tmp_path, monkeypatch, variant):
    manifest, _, _, _ = base_release
    approval = release_approval(manifest)
    namespace, calls = publisher_helpers(tmp_path)
    private_file = tmp_path / "reviewed-weights"
    private_file.write_bytes(b"reviewed-test-inference-bytes")
    if variant != "base":
        approval["source"] = {"repo": "fixture/private", "revision": "1" * 40, "prefix": "natural-v3/v4/train/run-1"}
        approval["files"] = [
            {
                "path": "adapter_model.safetensors",
                "source_path": "checkpoints/step-001039/adapter_model.safetensors",
                "sha256": release.sha256(private_file),
                "bytes": private_file.stat().st_size,
                "redistribution_approved": True,
                "license": "Apache-2.0",
            }
        ]
        if variant == "legacy_adapter":
            del approval["selected_variant"]
        else:
            approval["selected_variant"] = variant
    published = tmp_path / "published"
    published.mkdir()

    class Api:
        def __init__(self, token):
            self.token = token

        def repo_info(self, repo, **kwargs):
            calls.append(("repo", repo))
            return SimpleNamespace(private=repo == "fixture/private")

        def model_info(self, repo, revision):
            assert self.token is False
            calls.append(("dependency", repo, revision))
            return SimpleNamespace(private=False, sha=revision)

        def upload_folder(self, **kwargs):
            calls.append(("upload", kwargs["repo_id"], kwargs["path_in_repo"]))
            for file in Path(kwargs["folder_path"]).iterdir():
                shutil.copyfile(file, published / file.name)
            return SimpleNamespace(oid=manifest["revision"])

    def download(**kwargs):
        calls.append(("download", kwargs))
        if kwargs["repo_id"] == "fixture/private":
            assert kwargs["revision"] == approval["source"]["revision"]
            assert kwargs["filename"].endswith("/checkpoints/step-001039/adapter_model.safetensors")
            assert kwargs["token"] == "test-placeholder"
            return private_file
        assert kwargs["repo_id"] == manifest["repo"]
        assert kwargs["revision"] == manifest["revision"] and kwargs["token"] is False
        return published / Path(kwargs["filename"]).name

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=Api, hf_hub_download=download))
    approval_text = json.dumps(approval)
    approval_sha = hashlib.sha256(approval_text.encode()).hexdigest()
    result = namespace["release_remote"](
        manifest["repo"],
        approval_text,
        approval_sha,
        "release-1",
        "v4",
        manifest["git_revision"],
        manifest["manifest_sha256"],
    )
    provenance = release.read_json(published / "release-provenance.json")
    expected = {"README.md", "release-provenance.json"}
    if variant == "base":
        assert provenance["source"] is None and provenance["selected_variant"] == "base"
        assert ("repo", "fixture/private") not in calls
        assert all(call[1]["token"] is False for call in calls if isinstance(call, tuple) and call[0] == "download")
        # Publish -> pinned public manifest -> anonymous student fetch is the same actual contract.
        new_manifest = manifest | {"approval_sha256": approval_sha}
        new_manifest["files"] = [
            item | {"output": Path(item["path"]).name, "redistribution_approved": True, "license": "MIT"}
            for item in result["files"]
        ]
        fetched = release.fetch_release(
            new_manifest, tmp_path / "student", downloader=lambda repo, name, **kw: published / Path(name).name
        )
        assert release.student_options(new_manifest, fetched, device="cpu").adapter is None
    else:
        expected.add("adapter_model.safetensors")
        assert provenance["source"] == approval["source"]
        assert ("repo", "fixture/private") in calls
        assert (published / "adapter_model.safetensors").read_bytes() == private_file.read_bytes()
        if variant == "legacy_adapter":
            assert "selected_variant" not in provenance and "selected_variant" not in result
        else:
            assert provenance["selected_variant"] == variant and result["selected_variant"] == variant
    assert {file.name for file in published.iterdir()} == expected
    assert {Path(item["path"]).name for item in result["files"]} == expected
    assert result["anonymous_download_verified"] is True
    assert calls[-1] == "commit"


@pytest.mark.parametrize(
    "mutation", ["private_source", "adapter_files", "missing_files", "parameters", "unknown_variant", "unreviewed"]
)
def test_base_approval_refuses_private_weights_and_missing_review(base_release, tmp_path, mutation):
    manifest, _, _, _ = base_release
    approval = release_approval(manifest)
    if mutation == "private_source":
        approval["source"] = {"repo": "fixture/private", "revision": "1" * 40, "prefix": "natural-v3/fake"}
    elif mutation == "adapter_files":
        approval["files"] = [{"path": "adapter_config.json"}]
    elif mutation == "missing_files":
        del approval["files"]
    elif mutation == "parameters":
        approval["adapter_parameters"] = 12
    elif mutation == "unknown_variant":
        approval["selected_variant"] = ["base"]
    else:
        approval["reviewed"] = False
    namespace, _ = publisher_helpers(tmp_path)
    with pytest.raises(ValueError):
        namespace["validate_release"](approval)
