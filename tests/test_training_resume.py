"""中斷後的更新、固定教師與多模態兩階段交接，必須符合真實 CLI 行為。"""

import json
import sys

import pytest
import torch

from scripts import train
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.multimodal import AudioEncoder, MultiModalLM, VisionEncoder
from tiny_perceptron.training import load_checkpoint, save_checkpoint, seed_everything


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def run_cli(monkeypatch, capsys, task, output, *options):
    arguments = [
        "train.py",
        "--train",
        "--task",
        task,
        "--device",
        "cpu",
        "--width",
        "8",
        "--max-length",
        "64",
        "--batch-size",
        "1",
        "--steps",
        "4",
        "--save-every",
        "2",
        "--output",
        str(output),
        *map(str, options),
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    train.main()
    capsys.readouterr()
    return torch.load(output, weights_only=True)


@pytest.mark.parametrize("task", ["text", "sft", "dpo", "distill", "vision", "audio", "joint"])
def test_cli_interrupted_updates_match_uninterrupted(tmp_path, monkeypatch, capsys, task):
    options = []
    if task == "distill":
        teacher_path = tmp_path / "teacher.pt"
        seed_everything(91)
        save_checkpoint(teacher_path, TinyLM(ModelConfig(width=12, max_length=64)))
        options = ["--teacher", teacher_path]
    baseline = run_cli(monkeypatch, capsys, task, tmp_path / "baseline.pt", *options)
    interrupted_path = tmp_path / "interrupted.pt"
    interrupted = run_cli(monkeypatch, capsys, task, interrupted_path, *options, "--stop-after", "2")
    # teacher 已封存在 checkpoint 中，續訓不需要原始 teacher 檔。
    if options:
        teacher_path.unlink()
    resumed = run_cli(monkeypatch, capsys, task, tmp_path / "resumed.pt", "--checkpoint", interrupted_path, "--resume")
    assert interrupted["step"] == 2 and resumed["step"] == 4
    assert resumed["metadata"]["schedule_steps"] == 4
    assert resumed["metadata"]["peak_lr"] == 0.001
    assert resumed["optimizer"]["state"]
    assert all(torch.equal(value, resumed["model"][name]) for name, value in baseline["model"].items())
    assert any(not torch.equal(value, resumed["model"][name]) for name, value in interrupted["model"].items())
    if task in ("dpo", "distill"):
        key = "dpo_reference" if task == "dpo" else "distillation_teacher"
        frozen = interrupted["training_state"][key]["model"]
        assert all(torch.equal(value, resumed["training_state"][key]["model"][name]) for name, value in frozen.items())
        if task == "dpo":
            seed_everything(42)
            initial = TinyLM(ModelConfig(width=8, max_length=64)).state_dict()
            assert all(torch.equal(initial[name], value) for name, value in frozen.items())
            assert any(not torch.equal(value, interrupted["model"][name]) for name, value in frozen.items())


def test_multimodal_checkpoint_keeps_encoder_config_and_frozen_parameters(tmp_path):
    model = MultiModalLM(TinyLM(ModelConfig(width=8)), vision_width=12, audio_width=10)
    model.vision = VisionEncoder(width=12, image_size=8, patch_size=2)
    model.audio = AudioEncoder(bands=7, width=10)
    model.requires_grad_(False)
    model.image_projector.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.image_projector.parameters())
    path = tmp_path / "full.pt"
    save_checkpoint(path, model, optimizer, step=3, metadata={"task": "vision"})
    expected_random = torch.rand(5)
    restored, payload = load_checkpoint(path, restore_rng=True)
    assert torch.equal(torch.rand(5), expected_random)
    assert isinstance(restored, MultiModalLM)
    assert (restored.vision.image_size, restored.vision.patch_size, restored.audio.bands) == (8, 2, 7)
    assert restored.vision.projection.out_features == 12 and restored.audio.projection.out_features == 10
    assert payload["format_version"] == "multimodal-v1" and payload["step"] == 3
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in model.state_dict().items())
    assert all(
        parameter.requires_grad == name.startswith("image_projector.")
        for name, parameter in restored.named_parameters()
    )


def test_new_modal_stage_keeps_projector_weights(tmp_path, monkeypatch, capsys):
    first = tmp_path / "projector.pt"
    saved = run_cli(monkeypatch, capsys, "vision", first, "--freeze", "projector", "--stop-after", "2")
    original_loss = train.modal_loss
    seen = []

    def inspect_first_update(model, *args, **kwargs):
        if not seen:
            assert all(torch.equal(value, model.state_dict()[name]) for name, value in saved["model"].items())
            assert model.image_projector.weight.requires_grad
            assert model.language.blocks[0].attention.q.weight.requires_grad
            assert not model.language.embedding.weight.requires_grad
            seen.append(True)
        return original_loss(model, *args, **kwargs)

    monkeypatch.setattr(train, "modal_loss", inspect_first_update)
    second = run_cli(
        monkeypatch, capsys, "vision", tmp_path / "partial.pt", "--checkpoint", first, "--freeze", "partial"
    )
    assert seen and second["step"] == 4
    assert second["metadata"]["freeze"] == "partial"
    assert torch.equal(saved["model"]["language.embedding.weight"], second["model"]["language.embedding.weight"])


def test_resume_rejects_changed_schedule_reference_settings_or_teacher(tmp_path, monkeypatch, capsys):
    path = tmp_path / "dpo.pt"
    run_cli(monkeypatch, capsys, "dpo", path, "--stop-after", "2")
    for option, value, field in (
        ("--steps", "6", "schedule_steps"),
        ("--lr", "0.002", "peak_lr"),
        ("--beta", "0.2", "beta"),
    ):
        with pytest.raises(ValueError, match=field):
            run_cli(
                monkeypatch, capsys, "dpo", tmp_path / "invalid.pt", "--checkpoint", path, "--resume", option, value
            )
    teacher = tmp_path / "teacher.pt"
    save_checkpoint(teacher, TinyLM(ModelConfig(width=8, max_length=64)))
    path = tmp_path / "distill.pt"
    run_cli(monkeypatch, capsys, "distill", path, "--teacher", teacher, "--stop-after", "2")
    save_checkpoint(teacher, TinyLM(ModelConfig(width=8, max_length=64)))
    with pytest.raises(ValueError, match="teacher"):
        run_cli(
            monkeypatch,
            capsys,
            "distill",
            tmp_path / "invalid.pt",
            "--checkpoint",
            path,
            "--resume",
            "--teacher",
            teacher,
        )


def test_resume_rejects_changed_training_data(tmp_path, monkeypatch, capsys):
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({"text": "abc"}) + "\n")
    checkpoint = tmp_path / "data.pt"
    run_cli(monkeypatch, capsys, "text", checkpoint, "--data", data, "--stop-after", "2")
    data.write_text(json.dumps({"text": "changed"}) + "\n")
    with pytest.raises(ValueError, match="data_sha256"):
        run_cli(
            monkeypatch, capsys, "text", tmp_path / "invalid.pt", "--data", data, "--checkpoint", checkpoint, "--resume"
        )
