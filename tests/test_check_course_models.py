"""學生下載檢查器真實重建各公開格式；測試完全離線。"""

import json
import sys
from dataclasses import asdict

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from torch import nn

from scripts import check_course_models as checks
from scripts.course_experiments.applications import _FinitePolicy
from scripts.course_experiments.modalities import _Contrastive
from scripts.course_release import inference_payload
from tiny_perceptron.adapters import base_state_sha256
from tiny_perceptron.data import SPECIALS
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.multimodal import AudioEncoder, MultiModalLM, VisionEncoder
from tiny_perceptron.quantization import replace_linear_layers
from tiny_perceptron.simple import BigramLM, ContextMLP
from tiny_perceptron.training import save_checkpoint


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def native(model, path, **metadata):
    save_checkpoint(path, model, metadata=metadata)
    saved = torch.load(path, weights_only=True)
    torch.save(inference_payload(saved, {"revision": "1" * 40}), path)
    return path


def published_spec(folder, names, inference=None):
    exported = {
        "schema_version": 1,
        "provenance": {"revision": "1" * 40, "experiment_id": "test"},
        "files": [
            {
                "output": name,
                "sha256": checks.sha256(folder / name),
                "bytes": (folder / name).stat().st_size,
                "source_sha256": "2" * 64,
                "license": "MIT",
            }
            for name in names
        ],
    }
    (folder / "export-manifest.json").write_text(json.dumps(exported))
    return {
        "id": "test",
        "repo": "owner/public",
        "revision": "3" * 40,
        "files": [
            {
                "path": "course/test/" + name,
                "output": name,
                "sha256": checks.sha256(folder / name),
                "bytes": (folder / name).stat().st_size,
            }
            for name in names + ["export-manifest.json"]
        ],
        "inference": inference or {},
    }


@pytest.mark.parametrize(
    "version", ["native", "quantized", "multimodal", "bigram", "mlp", "vision", "audio", "contrastive", "policy"]
)
def test_each_format_has_a_real_cpu_forward(tmp_path, version):
    checkpoint = tmp_path / "model.pt"
    if version == "native":
        native(TinyLM(ModelConfig(width=8)), checkpoint)
    elif version == "quantized":
        model = replace_linear_layers(TinyLM(ModelConfig(width=8)), 4)
        torch.save(
            {"format_version": "quantized-v1", "config": asdict(model.config), "model": model.state_dict(), "bits": 4},
            checkpoint,
        )
    elif version == "multimodal":
        model = MultiModalLM(TinyLM(ModelConfig(width=8)), vision_width=12)
        model.vision = VisionEncoder(width=12, image_size=16, patch_size=8)
        native(model, checkpoint, task="vision")
    elif version in ("bigram", "mlp"):
        context = 1 if version == "bigram" else 3
        model = BigramLM(4) if version == "bigram" else ContextMLP(4, context, 8)
        torch.save(
            {
                "format_version": "simple-v1",
                "model": model.state_dict(),
                "vocabulary": {"甲": 2, "乙": 3},
                "context": context,
                "width": 8,
                "kind": version,
            },
            checkpoint,
        )
    elif version in ("vision", "audio"):
        config = {"width": 8, "image_size": 16, "patch_size": 8} if version == "vision" else {"width": 8, "bands": 12}
        model = VisionEncoder(**config) if version == "vision" else AudioEncoder(**config)
        torch.save(
            {
                "format_version": "encoder-v1",
                "encoder": model.state_dict(),
                "config": config,
                "classifier": nn.Linear(8, 2).state_dict(),
                "classes": ["a", "b"],
                "architecture": {"type": "VisionEncoder" if version == "vision" else "AudioEncoder"},
            },
            checkpoint,
        )
    elif version == "contrastive":
        model = _Contrastive()
        torch.save(
            {
                "format_version": "contrastive-v1",
                "model": model.state_dict(),
                "architecture": {
                    "type": "Contrastive",
                    "vision_config": {"width": 16, "image_size": 16, "patch_size": 4},
                    "text_vocab_size": 264,
                    "temperature": 0.1,
                    "pooling": "mean_then_l2",
                },
            },
            checkpoint,
        )
    else:
        model = _FinitePolicy()
        torch.save(
            {
                "format_version": "finite-policy-v1",
                "model": model.state_dict(),
                "architecture": {
                    "type": "FinitePolicy",
                    "actions": [str(i) for i in range(16)] + [" ".join(map(str, range(16)))],
                    "input_features": "three six-way one-hot operands",
                    "hidden_width": 48,
                    "activation": "tanh",
                },
            },
            checkpoint,
        )
    result = checks.probe_checkpoint(checkpoint, tmp_path)
    assert result["device"] == "cpu" and result["result"]["finite"]
    assert result["result"]["shape"][0] in (1, 2) and "not accuracy evidence" in result["scope"]
    if version == "multimodal":
        assert result["visual_tokens"] == 4
    if version == "policy":
        assert result["chosen_action"] in [str(i) for i in range(16)] + [" ".join(map(str, range(16)))]


def test_bpe_probe_pairs_the_exact_public_companion(tmp_path):
    path = tmp_path / "tokenizer.json"
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        ["測試 <user> 🙂"],
        trainers.BpeTrainer(
            vocab_size=300,
            special_tokens=list(SPECIALS),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        ),
    )
    tokenizer.save(str(path))
    native(
        TinyLM(ModelConfig(width=8, vocab_size=tokenizer.get_vocab_size())),
        tmp_path / "model.pt",
        tokenizer="bpe",
        tokenizer_sha256=checks.sha256(path),
    )
    saved = torch.load(tmp_path / "model.pt", weights_only=True)
    saved["metadata"]["tokenizer_file"] = "tokenizer.json"
    torch.save(saved, tmp_path / "model.pt")
    spec = published_spec(tmp_path, ["model.pt", "tokenizer.json"])
    assert checks.validate_folder(spec, tmp_path)[0]["format_version"] == 1
    assert (
        checks.probe_checkpoint(tmp_path / "model.pt", tmp_path)["tokenizer_vocab_size"] == tokenizer.get_vocab_size()
    )
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="SHA-256"):
        checks.probe_checkpoint(tmp_path / "model.pt", tmp_path)
    missing = published_spec(tmp_path, ["model.pt"])
    with pytest.raises(ValueError, match="tokenizer 未列"):
        checks.validate_folder(missing, tmp_path)


def test_adapter_probe_downloads_pinned_anonymous_base_and_checks_identity(tmp_path, monkeypatch):
    import huggingface_hub

    base_model = TinyLM(ModelConfig(width=8))
    base = native(base_model, tmp_path / "base.pt")
    pointer = {
        "repo": "owner/public",
        "revision": "3" * 40,
        "filename": "course/base.pt",
        "sha256": checks.sha256(base),
    }
    adapter = {
        "format_version": "lora-v1",
        "config": asdict(base_model.config),
        "base_sha256": base_state_sha256(base_model.state_dict()),
        "base_checkpoint": pointer,
        "scaling": "alpha/rank",
        "adapter": {"output": {"a": torch.zeros(2, 8), "b": torch.zeros(264, 2), "rank": 2, "alpha": 4}},
    }
    torch.save(adapter, tmp_path / "adapter.pt")
    calls = []

    def download(repo, filename, *, revision, token):
        calls.append((repo, filename, revision, token))
        return str(base)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    result = checks.probe_checkpoint(tmp_path / "adapter.pt", tmp_path)
    assert calls == [(pointer["repo"], pointer["filename"], pointer["revision"], False)]
    assert result["adapter"]["base_sha256_verified"] is True and result["result"]["finite"]
    adapter["base_sha256"] = "0" * 64
    torch.save(adapter, tmp_path / "adapter.pt")
    with pytest.raises(ValueError, match="base_sha256 不匹配"):
        checks.probe_checkpoint(tmp_path / "adapter.pt", tmp_path)


@pytest.mark.parametrize("field", ["optimizer", "torch_rng", "training_state", "teacher", "reference"])
def test_public_payload_rejects_training_state_even_if_value_is_none(tmp_path, field):
    path = native(TinyLM(ModelConfig(width=8)), tmp_path / "model.pt")
    saved = torch.load(path, weights_only=True)
    saved[field] = None
    torch.save(saved, path)
    spec = published_spec(tmp_path, ["model.pt"])
    with pytest.raises(ValueError):
        checks.validate_folder(spec, tmp_path)


def test_unknown_format_or_nonfinite_weight_fails(tmp_path):
    torch.save({"format_version": "future-v9", "model": {}}, tmp_path / "future.pt")
    with pytest.raises(ValueError, match="未知"):
        checks.probe_checkpoint(tmp_path / "future.pt", tmp_path)
    model = TinyLM(ModelConfig(width=8))
    model.embedding.weight.data[0, 0] = float("nan")
    path = native(model, tmp_path / "model.pt")
    with pytest.raises(ValueError, match="非有限"):
        checks.probe_checkpoint(path, tmp_path)


def test_export_manifest_must_cover_actual_exported_weight_hashes(tmp_path):
    native(TinyLM(ModelConfig(width=8)), tmp_path / "model.pt")
    spec = published_spec(tmp_path, ["model.pt"])
    assert len(checks.validate_folder(spec, tmp_path)) == 1
    exported = json.loads((tmp_path / "export-manifest.json").read_text())
    exported["files"][0]["sha256"] = "0" * 64
    (tmp_path / "export-manifest.json").write_text(json.dumps(exported))
    spec["files"][-1].update(
        sha256=checks.sha256(tmp_path / "export-manifest.json"),
        bytes=(tmp_path / "export-manifest.json").stat().st_size,
    )
    with pytest.raises(ValueError, match="匯出後"):
        checks.validate_folder(spec, tmp_path)


def test_real_cli_subprocess_and_existing_root_evidence_are_preserved(tmp_path, monkeypatch):
    model = TinyLM(ModelConfig(width=8))
    with torch.no_grad():
        model.final_norm.weight.zero_()
        model.final_norm.bias.zero_()
        model.final_norm.bias[0] = 1
        model.output.weight.zero_()
        model.output.weight[2, 0] = 1
    native(model, tmp_path / "model.pt")
    spec = published_spec(
        tmp_path, ["model.pt"], {"script": "scripts/infer.py", "checkpoint": "model.pt", "prompt": "q", "tokens": 2}
    )
    monkeypatch.setattr(checks, "fetch_model", lambda spec, output: tmp_path)
    report = checks.check_model(spec, tmp_path, "1" * 64)
    assert report["status"] == "passed" and len(report["commands"]) == 2
    assert report["commands"][-1]["output"]["generated_ids"] == [2]
    assert report["commands"][-1]["command"][-1] == "--json"
    assert report["commands"][0]["output"]["result"]["finite"]
    proof = tmp_path / "proof"
    proof.mkdir()
    original = proof / "test.json"
    original.write_text('{"root_reviewed":true}\n')
    new = checks.save_evidence(proof, "test", report)
    assert new != original and original.read_text() == '{"root_reviewed":true}\n'
    assert json.loads(new.read_text())["status"] == "passed"


def test_unknown_cli_or_unpublished_companion_is_an_error(tmp_path):
    native(TinyLM(ModelConfig(width=8)), tmp_path / "model.pt")
    with pytest.raises(ValueError, match="未知學生"):
        checks.inference_command({"script": "scripts/arbitrary.py", "checkpoint": "model.pt"}, tmp_path, {"model.pt"})
    with pytest.raises(ValueError, match="未公開"):
        checks.inference_command(
            {"script": "scripts/infer.py", "checkpoint": "model.pt", "tokenizer": "missing.json"},
            tmp_path,
            {"model.pt"},
        )


def test_manifest_hash_binds_the_snapshot_even_if_next_release_updates_it(tmp_path, monkeypatch):
    manifest = tmp_path / "models.json"
    manifest.write_text(json.dumps({"schema_version": 1, "models": [{"id": "one"}, {"id": "two"}]}))
    snapshot = checks.sha256(manifest)
    hashes = []

    def check(spec, output, digest):
        hashes.append(digest)
        manifest.write_text(json.dumps({"schema_version": 1, "models": [{"id": "new-release"}]}))
        return {"status": "passed"}

    monkeypatch.setattr(checks, "check_model", check)
    monkeypatch.setattr(
        sys,
        "argv",
        ["check_course_models.py", "--manifest", str(manifest), "--all", "--evidence", str(tmp_path / "proof")],
    )
    checks.main()
    assert hashes == [snapshot, snapshot]


def test_empty_public_manifest_cannot_report_a_successful_all_check(tmp_path, monkeypatch):
    manifest = tmp_path / "models.json"
    manifest.write_text(json.dumps({"schema_version": 1, "models": []}))
    monkeypatch.setattr(sys, "argv", ["check_course_models.py", "--manifest", str(manifest), "--all"])
    with pytest.raises(ValueError, match="沒有可檢查"):
        checks.main()
