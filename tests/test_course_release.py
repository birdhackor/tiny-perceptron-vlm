"""公開的是能實際載入的推論檔；來源、tokenizer 及授權皆固定。"""

import copy
import hashlib
import json
from dataclasses import asdict

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from torch import nn

from scripts.course_experiments.applications import _FinitePolicy
from scripts.course_experiments.modalities import _Contrastive
from scripts.course_release import build_export, file_sha256, inference_payload, validate_approval, validate_export
from scripts.infer_simple import load_simple_checkpoint
from tiny_perceptron.data import SPECIALS
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.multimodal import AudioEncoder, MultiModalLM, VisionEncoder
from tiny_perceptron.simple import BigramLM, ContextMLP
from tiny_perceptron.tokenization import load_tokenizer
from tiny_perceptron.training import load_checkpoint, save_checkpoint


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def approval(directory, names, **specification):
    return {
        "schema_version": 1,
        "approved": True,
        "reviewed": True,
        "experiment_id": "test",
        "batch_id": "course-v1",
        "revision": "1" * 40,
        "private_source": {"repo": "owner/private", "revision": "2" * 40, "prefix": "course/course-v1/test/run-1"},
        "files": [
            {
                "path": name,
                "sha256": file_sha256(directory / name),
                "kind": "checkpoint" if name.endswith(".pt") else "metadata",
                "license": "MIT",
                "redistribution_approved": True,
                **specification,
            }
            for name in names
        ],
        "model_card": {
            "summary": "A controlled experiment",
            "scope": "Toy only",
            "limitations": ["No general capabilities"],
            "training_data": [{"source": "Synthetic", "license": "MIT", "modifications": "Generated"}],
            "third_party_notices": "Upstream copyright retained verbatim.",
        },
    }


@pytest.mark.parametrize(
    "change",
    [
        {"reviewed": False},
        {"revision": "main"},
        {"private_source": {"repo": "owner/private", "revision": "main", "prefix": "course/course-v1/test/run-1"}},
        {"pku_derived": True},
        {"visibility": "private"},
    ],
)
def test_approval_rejects_unreviewed_or_unpinned_source(tmp_path, change):
    (tmp_path / "model.pt").write_bytes(b"placeholder")
    with pytest.raises(ValueError):
        validate_approval(approval(tmp_path, ["model.pt"]) | change, "test", "course-v1", "owner/private")


@pytest.mark.parametrize(
    "filename,output",
    [
        ("result.json", "result.json"),
        ("model.pt", "README.md"),
        ("model.pt", "../model.pt"),
        ("pku-pilot.pt", "model.pt"),
    ],
)
def test_private_data_and_generated_files_cannot_be_smuggled_into_release(tmp_path, filename, output):
    (tmp_path / filename).write_bytes(b"placeholder")
    approved = approval(tmp_path, [filename], output=output)
    with pytest.raises(ValueError):
        validate_approval(approved, "test", "course-v1", "owner/private")


@pytest.mark.parametrize("kind", ["bigram", "mlp"])
def test_simple_export_keeps_vocabulary_context_width_and_identical_logits(tmp_path, kind):
    context = 1 if kind == "bigram" else 3
    model = BigramLM(4) if kind == "bigram" else ContextMLP(4, context=context, width=8)
    saved = {
        "format_version": "simple-v1",
        "model": model.state_dict(),
        "vocabulary": {"甲": 2, "乙": 3},
        "context": context,
        "width": 8,
        "kind": kind,
        "optimizer": {"private": 1},
        "torch_rng": torch.get_rng_state(),
        "training_state": {"teacher": "private"},
        "metadata": {"dataset": "PRIVATE"},
    }
    torch.save(saved, tmp_path / "model.pt")
    build_export(tmp_path, tmp_path / "public", approval(tmp_path, ["model.pt"]), {"revision": "1" * 40})
    loaded, clean, vocabulary = load_simple_checkpoint(tmp_path / "public/model.pt")
    assert vocabulary == saved["vocabulary"] and clean["context"] == context and clean["width"] == 8
    assert not {"optimizer", "torch_rng", "training_state"} & set(clean)
    inputs = torch.tensor([[2] * context])
    assert torch.equal(model(inputs), loaded(inputs))
    assert "Upstream copyright retained" in (tmp_path / "public/THIRD_PARTY_NOTICES.md").read_text()
    assert "MIT License" in (tmp_path / "public/LICENSE").read_text()


def test_multimodal_export_preserves_actual_patch8_configuration(tmp_path):
    model = MultiModalLM(TinyLM(ModelConfig(width=8)), vision_width=12, audio_width=10)
    model.vision = VisionEncoder(width=12, image_size=16, patch_size=8)
    save_checkpoint(
        tmp_path / "model.pt",
        model,
        torch.optim.AdamW(model.parameters()),
        3,
        metadata={"task": "vision", "private_samples": ["PRIVATE"]},
    )
    build_export(tmp_path, tmp_path / "public", approval(tmp_path, ["model.pt"]), {"revision": "1" * 40})
    loaded, saved = load_checkpoint(tmp_path / "public/model.pt")
    assert loaded.vision.patch_size == 8 and loaded.vision.position.shape[1] == 4
    assert loaded.audio.projection.out_features == 10
    assert "optimizer" not in saved and "private_samples" not in saved["metadata"]
    model.eval(), loaded.eval()
    image = torch.rand(1, 3, 16, 16)
    assert torch.equal(model.vision(image), loaded.vision(image))


def bpe_file(path):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        ["文字 <user> emoji 🙂"],
        trainers.BpeTrainer(
            vocab_size=300,
            special_tokens=list(SPECIALS),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        ),
    )
    tokenizer.save(str(path))
    return tokenizer


def test_bpe_export_preserves_hash_and_requires_companion(tmp_path):
    tokenizer = bpe_file(tmp_path / "tokenizer.json")
    model = TinyLM(ModelConfig(vocab_size=tokenizer.get_vocab_size(), width=8))
    digest = file_sha256(tmp_path / "tokenizer.json")
    save_checkpoint(tmp_path / "model.pt", model, metadata={"tokenizer": "bpe", "tokenizer_sha256": digest})
    incomplete = approval(tmp_path, ["model.pt"])
    with pytest.raises(ValueError, match="tokenizer"):
        build_export(tmp_path, tmp_path / "incomplete", incomplete, {})
    approved = approval(tmp_path, ["model.pt", "tokenizer.json"])
    approved["files"][0].update(tokenizer_file="tokenizer.json", companions=["tokenizer.json"])
    approved["files"][1]["output"] = "vocab.json"
    files = build_export(tmp_path, tmp_path / "public", approved, {})
    loaded, saved = load_checkpoint(tmp_path / "public/model.pt")
    tokenizer = load_tokenizer(tmp_path / "public/vocab.json", loaded.config.vocab_size, saved)
    text = "<user> 🙂未見🦊"
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert saved["metadata"]["tokenizer_file"] == "vocab.json" and saved["metadata"]["tokenizer_sha256"] == digest
    assert files[0]["sha256"] != approved["files"][0]["sha256"]
    altered = copy.deepcopy(approved)
    altered["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA"):
        build_export(tmp_path, tmp_path / "altered", altered, {})
    saved["metadata"]["tokenizer_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="tokenizer SHA"):
        inference_payload(saved, {}, approved["files"][0], {"tokenizer.json": digest})


@pytest.mark.parametrize("modality", ["vision", "audio"])
def test_encoder_export_has_classification_config_not_training_state(tmp_path, modality):
    config = {"width": 8, "image_size": 16, "patch_size": 8} if modality == "vision" else {"width": 8, "bands": 12}
    encoder = VisionEncoder(**config) if modality == "vision" else AudioEncoder(**config)
    classifier = nn.Linear(8, 2)
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "classifier": classifier.state_dict(),
            "config": config,
            "classes": ["a", "b"],
            "optimizer": "PRIVATE",
            "report": {"samples": "PRIVATE"},
        },
        tmp_path / "model.pt",
    )
    approved = approval(
        tmp_path, ["model.pt"], architecture={"type": "VisionEncoder" if modality == "vision" else "AudioEncoder"}
    )
    build_export(tmp_path, tmp_path / "public", approved, {})
    saved = torch.load(tmp_path / "public/model.pt", weights_only=True)
    assert saved["config"] == config and saved["classes"] == ["a", "b"]
    assert "optimizer" not in saved and "report" not in saved
    assert validate_export(tmp_path / "public/model.pt") == "encoder-v1"


def test_custom_policy_and_contrastive_exports_need_complete_explicit_architecture(tmp_path):
    architectures = [
        (
            _Contrastive(),
            {
                "type": "Contrastive",
                "vision_config": {"width": 16, "image_size": 16, "patch_size": 4},
                "text_vocab_size": 264,
                "temperature": 0.1,
                "pooling": "mean_then_l2",
            },
        ),
        (
            _FinitePolicy(),
            {
                "type": "FinitePolicy",
                "actions": [str(i) for i in range(16)] + [" ".join(map(str, range(16)))],
                "input_features": "three six-way one-hot operands",
                "hidden_width": 48,
                "activation": "tanh",
            },
        ),
    ]
    for model, architecture in architectures:
        torch.save({"model": model.state_dict(), "optimizer": "PRIVATE", "rng": "PRIVATE"}, tmp_path / "model.pt")
        approved = approval(tmp_path, ["model.pt"], architecture=architecture)
        destination = tmp_path / architecture["type"]
        build_export(tmp_path, destination, approved, {})
        saved = torch.load(destination / "model.pt", weights_only=True)
        assert saved["architecture"] == architecture and set(saved) == {
            "format_version",
            "model",
            "architecture",
            "metadata",
        }
        incomplete = copy.deepcopy(approved)
        incomplete["files"][0]["architecture"] = {"type": architecture["type"]}
        with pytest.raises(ValueError):
            build_export(tmp_path, tmp_path / "bad", incomplete, {})


def test_lora_keeps_base_identity_and_adapter_scaling_and_refuses_unknown_formats(tmp_path):
    config = ModelConfig(width=8)
    base_model = TinyLM(config)
    base_file = tmp_path / "base.pt"
    save_checkpoint(base_file, base_model)
    base_digest = hashlib.sha256()
    for name, value in sorted(base_model.state_dict().items()):
        base_digest.update(name.encode())
        base_digest.update(str(tuple(value.shape)).encode())
        base_digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    saved = {
        "format_version": "lora-v1",
        "adapter": {"output": {"a": torch.rand(2, 8), "b": torch.rand(264, 2), "rank": 2, "alpha": 4}},
        "config": asdict(config),
        "base_sha256": base_digest.hexdigest(),
        "scaling": "alpha/rank",
        "optimizer": "PRIVATE",
        "reference": "PRIVATE",
    }
    torch.save(saved, tmp_path / "model.pt")
    pointer = {
        "repo": "owner/public",
        "revision": "3" * 40,
        "filename": "base/model.pt",
        "sha256": file_sha256(base_file),
    }
    bases = tmp_path / ".verified-bases"
    bases.mkdir()
    (bases / f"{pointer['sha256']}.pt").write_bytes(base_file.read_bytes())
    approved = approval(tmp_path, ["model.pt"], base_checkpoint=pointer)
    build_export(tmp_path, tmp_path / "public", approved, {})
    clean = torch.load(tmp_path / "public/model.pt", weights_only=True)
    assert (
        clean["base_checkpoint"] == pointer
        and clean["base_sha256"] == saved["base_sha256"]
        and clean["scaling"] == "alpha/rank"
    )
    assert "optimizer" not in clean and "reference" not in clean
    with pytest.raises(ValueError, match="base_checkpoint"):
        inference_payload(saved, {})
    with pytest.raises(ValueError, match="未知"):
        inference_payload({"model": {}, "teacher": "PRIVATE"}, {})
    saved["base_sha256"] = "0" * 64
    torch.save(saved, tmp_path / "model.pt")
    altered = approval(tmp_path, ["model.pt"], base_checkpoint=pointer)
    with pytest.raises(ValueError, match="base state identity"):
        build_export(tmp_path, tmp_path / "wrong-base", altered, {})


def test_export_manifest_hashes_are_export_bytes_and_not_private_weights(tmp_path):
    model = TinyLM(ModelConfig(width=8))
    save_checkpoint(tmp_path / "model.pt", model, torch.optim.AdamW(model.parameters()))
    approved = approval(tmp_path, ["model.pt"])
    approved["model_card"]["evaluation"] = {"accuracy": 0.5}
    build_export(tmp_path, tmp_path / "public", approved, {"revision": "1" * 40})
    item = json.loads((tmp_path / "public/export-manifest.json").read_text())["files"][0]
    assert item["sha256"] == file_sha256(tmp_path / "public/model.pt") and item["sha256"] != item["source_sha256"]
    assert json.loads((tmp_path / "public/evaluation.json").read_text()) == {"accuracy": 0.5}
    cc = approval(tmp_path, ["model.pt"], license="CC-BY-SA-4.0")
    build_export(tmp_path, tmp_path / "cc", cc, {})
    assert "https://creativecommons.org/licenses/by-sa/4.0/legalcode.en" in (tmp_path / "cc/LICENSE").read_text()
