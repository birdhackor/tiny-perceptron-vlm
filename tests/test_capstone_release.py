"""Capstone artifacts retain inference only; published students load verified bytes."""

import copy
from dataclasses import asdict

import pytest
import torch

from scripts.capstone_release import (
    PUBLIC_REPO,
    build_public_manifest,
    clean_capstone_payload,
    export_capstone,
    file_sha256,
    prepare_approval,
    validate_capstone_payload,
    validate_manifest,
)
from scripts.course_release import validate_approval, validate_export
from scripts.fetch_capstone import fetch_capstone
from tiny_perceptron.capstone import CapstoneModel, load_capstone, save_capstone
from tiny_perceptron.capstone_quantization import (
    load_quantized_capstone,
    quantizable_weights,
    quantize_capstone,
    restore_quantized_payload,
    tensor_bytes,
)
from tiny_perceptron.model import ModelConfig


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def source(tmp_path):
    torch.manual_seed(17)
    model = CapstoneModel(ModelConfig(width=8, heads=2, kv_heads=1, experts=4, top_k=2, rotary=True, norm="rms"))
    path = tmp_path / "source.pt"
    save_capstone(
        path,
        model,
        stage="dpo",
        step=7,
        optimizer=torch.optim.AdamW(model.parameters()),
        metadata={"secret": "DO NOT EXPORT", "dataset": ["PRIVATE"], "seed": 17},
        reference=model,
        state={"samples": ["PRIVATE"]},
    )
    return path, model


def approved(source, directory, path="source.pt", stage="dpo"):
    draft = prepare_approval(
        directory,
        {stage: path},
        training_revision="a" * 40,
        private_repo="owner/private",
        private_revision="b" * 40,
        private_prefix="course/capstone-v1/capstone/run-1",
    )
    assert draft["approved"] is False and draft["reviewed"] is False
    assert draft["files"][0]["redistribution_approved"] is False
    with pytest.raises(ValueError, match="root"):
        validate_approval(draft, "capstone", "capstone-v1", "owner/private")
    # This test approval is deliberately limited to generated test tensors.
    draft.update(approved=True, reviewed=True)
    draft["files"][0]["redistribution_approved"] = True
    return draft


def test_export_removes_training_references_and_preserves_all_logits(source, tmp_path):
    path, model = source
    approval = approved(source, tmp_path)
    export_capstone(tmp_path, tmp_path / "public", approval, {"training_revision": "a" * 40, "secret": "PRIVATE"})
    target = tmp_path / "public/dpo/model.pt"
    loaded, clean = load_capstone(target)
    assert clean["architecture"] == {"type": "CapstoneModel"}
    assert clean["inference_only"] is True and clean["config"] == asdict(model.config)
    assert set(clean) == {
        "format_version",
        "inference_only",
        "config",
        "model",
        "stage",
        "step",
        "tokenizer",
        "data_version",
        "architecture",
        "metadata",
    }
    assert clean["metadata"] == {"seed": 17, "training_revision": "a" * 40}
    assert validate_export(target) == "capstone-v1"
    ids = torch.tensor([[1, 3, 5, 6, 8]])
    options = {"images": torch.rand(1, 3, 16, 16), "audio_features": torch.randn(1, 16)}
    model.eval(), loaded.eval()
    assert torch.equal(model(ids, **options)["logits"], loaded(ids, **options)["logits"])
    assert "MIT License" in (tmp_path / "public/LICENSE").read_text()
    assert (tmp_path / "public/THIRD_PARTY_NOTICES.md").is_file()
    assert file_sha256(path) == approval["files"][0]["sha256"]


@pytest.mark.parametrize("architecture", [{}, {"type": "TinyLM"}, {"type": "CapstoneModel", "secret": "PRIVATE"}])
def test_capstone_export_requires_exact_explicit_architecture(source, architecture):
    saved = torch.load(source[0], weights_only=True)
    with pytest.raises(ValueError, match="architecture"):
        clean_capstone_payload(saved, {}, {"architecture": architecture})


@pytest.mark.parametrize("where", ["config", "tokenizer", "model"])
def test_nested_training_payload_cannot_be_disguised_as_inference(source, where):
    saved = torch.load(source[0], weights_only=True)
    saved[where]["secret"] = "PRIVATE"
    with pytest.raises(ValueError):
        clean_capstone_payload(saved, {}, {"architecture": {"type": "CapstoneModel"}})


def test_metadata_cannot_be_renamed_into_checkpoint(source, tmp_path):
    (tmp_path / "notes.json").write_text("{}")
    approval = approved(source, tmp_path)
    approval["files"] = [
        {
            "path": "notes.json",
            "output": "model.pt",
            "kind": "metadata",
            "sha256": file_sha256(tmp_path / "notes.json"),
            "license": "MIT",
            "redistribution_approved": True,
        }
    ]
    with pytest.raises(ValueError, match="checkpoint"):
        validate_approval(approval, "capstone", "capstone-v1", "owner/private")


@pytest.mark.parametrize("bits", [4, 8])
def test_ptq_packs_real_weights_retains_router_and_reloads(source, tmp_path, bits):
    path, original = source
    output = tmp_path / f"int{bits}.pt"
    receipt = quantize_capstone(path, output, bits)
    loaded, payload = load_quantized_capstone(output)
    assert receipt["tensor_bytes"] < receipt["float_tensor_bytes"]
    assert receipt["tensor_bytes"] == tensor_bytes(payload["model"]) + tensor_bytes(payload["quantized"])
    assert receipt["file_bytes"] == output.stat().st_size
    assert receipt["sha256"] == file_sha256(output)
    assert not {"optimizer", "reference", "torch_rng", "python_rng", "training_state"} & set(payload)
    assert payload["metadata"] == {"source_checkpoint_sha256": file_sha256(path)}
    names = quantizable_weights(original)
    assert set(payload["quantized"]) == set(names) and all("router" not in name for name in names)
    for name, tensor in payload["model"].items():
        assert tensor.dtype == torch.float32
        assert torch.equal(tensor, original.state_dict()[name])
    for name, packed in payload["quantized"].items():
        expected = original.state_dict()[name]
        if bits == 4:
            assert packed["values"].dtype == torch.uint8
            assert packed["values"].numel() == (expected.numel() + 1) // 2
        else:
            assert packed["values"].dtype == torch.int8 and packed["values"].shape == expected.shape
        reconstructed = loaded.state_dict()[name]
        assert torch.all((expected - reconstructed).abs() <= packed["scale"] * 0.5001)
    logits = loaded(torch.tensor([[1, 3, 8]]))["logits"]
    assert logits.shape == (1, 3, 264) and torch.isfinite(logits).all()
    approval = approved(source, tmp_path, output.name, f"dpo-int{bits}")
    export_capstone(tmp_path, tmp_path / f"public-int{bits}", approval, {})
    exported = tmp_path / f"public-int{bits}/dpo-int{bits}/model.pt"
    clean_model, clean = load_quantized_capstone(exported)
    assert validate_capstone_payload(clean) == "capstone-ptq-v1"
    loaded.eval(), clean_model.eval()
    assert torch.equal(loaded(torch.tensor([[1, 3, 8]]))["logits"], clean_model(torch.tensor([[1, 3, 8]]))["logits"])


@pytest.mark.parametrize("change", ["router", "scale", "packed", "extra", "float"])
def test_ptq_loader_rejects_corrupted_or_smuggled_tensors(source, tmp_path, change):
    output = tmp_path / "int4.pt"
    quantize_capstone(source[0], output, 4)
    payload = torch.load(output, weights_only=True)
    name = next(iter(payload["quantized"]))
    if change == "router":
        payload["quantization"]["linear_weights"].append("language.blocks.0.ffn.router.weight")
    elif change == "scale":
        payload["quantized"][name]["scale"].fill_(float("nan"))
    elif change == "packed":
        payload["quantized"][name]["values"] = payload["quantized"][name]["values"][:-1]
    elif change == "extra":
        payload["quantized"][name]["secret"] = "PRIVATE"
    else:
        payload["model"]["language.embedding.weight"] = payload["model"]["language.embedding.weight"].half()
    with pytest.raises(ValueError):
        restore_quantized_payload(payload)


def public_fixture(source, tmp_path, monkeypatch):
    directory = tmp_path / "public"
    export_capstone(tmp_path, directory, approved(source, tmp_path), {})
    calls = []

    def download(repo, filename, **options):
        calls.append((repo, filename, options))
        return str(directory / filename.removeprefix("capstone/run-1/"))

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    manifest = build_public_manifest(directory, revision="c" * 40, prefix="capstone/run-1")
    assert all(options == {"revision": "c" * 40, "token": False} for _, _, options in calls)
    assert all(repo == PUBLIC_REPO for repo, _, _ in calls)
    return directory, manifest, calls


def test_public_verification_and_download_load_actual_checkpoint(source, tmp_path, monkeypatch):
    directory, manifest, calls = public_fixture(source, tmp_path, monkeypatch)
    assert len(calls) == 4 and manifest["models"][0]["id"] == "dpo"
    target = fetch_capstone(manifest, "dpo", tmp_path / "downloads")
    loaded, clean = load_capstone(target / "model.pt")
    assert loaded.config == source[1].config and clean["inference_only"] is True
    assert (target / "model.pt").read_bytes() == (directory / "dpo/model.pt").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        fetch_capstone(manifest, "dpo", tmp_path / "downloads")


def test_failed_download_never_installs_partial_stage(source, tmp_path, monkeypatch):
    directory, manifest, _ = public_fixture(source, tmp_path, monkeypatch)
    (directory / "LICENSE").write_text("corrupted after publication")
    with pytest.raises(ValueError, match="SHA/size"):
        fetch_capstone(manifest, "dpo", tmp_path / "downloads")
    assert not (tmp_path / "downloads/dpo").exists()
    assert not list((tmp_path / "downloads").iterdir())


def test_public_manifest_cannot_be_created_when_hf_bytes_do_not_match(source, tmp_path, monkeypatch):
    directory, _, _ = public_fixture(source, tmp_path, monkeypatch)
    wrong = tmp_path / "wrong"
    wrong.write_bytes(b"wrong")
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *args, **kwargs: str(wrong))
    with pytest.raises(ValueError, match="public HF bytes"):
        build_public_manifest(directory, revision="c" * 40, prefix="capstone/run-1")


@pytest.mark.parametrize("change", ["revision", "repo", "path", "output", "duplicate", "hash"])
def test_manifest_rejects_moving_revisions_and_unsafe_paths(source, tmp_path, monkeypatch, change):
    _, original, _ = public_fixture(source, tmp_path, monkeypatch)
    manifest = copy.deepcopy(original)
    if change == "revision":
        manifest["revision"] = "main"
    elif change == "repo":
        manifest["repo"] = "another/repo"
    elif change in ("path", "output"):
        manifest["models"][0]["files"][0][change] = "../escape.pt"
    elif change == "duplicate":
        manifest["models"].append(manifest["models"][0])
    else:
        manifest["models"][0]["files"][0]["sha256"] = "unknown"
    with pytest.raises(ValueError):
        validate_manifest(manifest)


def test_missing_manifest_explains_unpublished_status(tmp_path, capsys, monkeypatch):
    from scripts.fetch_capstone import main

    monkeypatch.setattr("sys.argv", ["fetch_capstone.py", "--manifest", str(tmp_path / "unpublished.json"), "--list"])
    with pytest.raises(SystemExit) as stopped:
        main()
    assert stopped.value.code == 2
    assert "have not been published" in capsys.readouterr().err
