"""正式資料對照不補缺欄；AV配對選模不能被易答的缺線索題掩蓋。"""

import json
import shutil
from dataclasses import asdict

import pytest
import torch
from safetensors.torch import load_file

from scripts.omni_revision import formal_local
from tiny_perceptron.omni_products import training_runtime as trainer
from tiny_perceptron.omni_products.inference import load_trained
from tiny_perceptron.omni_products.training_runtime import av_development_metrics, choice_training_core


def test_compact_control_is_exact_original_variant_zero_subset():
    expanded, compact = choice_training_core(), choice_training_core("compact")
    assert len(expanded) == 10836 and len(compact) == 3612
    assert [asdict(row) for row in compact] == [asdict(row) for i, row in enumerate(expanded) if i % 12 < 4]
    assert any("unknown" in row.state for row in compact)


def test_paired_selection_beats_more_missing_cue_successes():
    class Fixed:
        def __init__(self, answers):
            self.answers = answers

        def __call__(self, row):
            return torch.nn.functional.one_hot(self.answers, 3).float()

    labels = torch.tensor([0, 2, 2, 1, 2, 2] * 2)
    bank = {"labels": labels}
    trivial = labels.clone()
    trivial[[0, 6]] = 1
    real = labels.clone()
    real[[1, 2, 4, 5, 7, 8, 10, 11]] = 0
    first = av_development_metrics(Fixed(trivial), bank)
    second = av_development_metrics(Fixed(real), bank)
    assert first["paired_accuracy"] == 0 and first["remove_image_accuracy"] == 1
    assert second["paired_accuracy"] == 1 and second["selection_score"] > first["selection_score"]


def test_actual_fresh_training_roundtrip_and_source_revision_rejection(tmp_path):
    config = {
        "seed": 11,
        "updates": 2,
        "training_seconds": 20,
        "batch_size": 2,
        "learning_rate": 0.001,
        "validate_every": 2,
        "representation": "time-rope",
        "variants": 1,
    }
    result = trainer.train_av(config, tmp_path, "cpu")
    model = load_trained(tmp_path, "av")
    assert result["complete"] and result["optimizer_updates_observed"]
    assert trainer.parameter_sha(model) != result["initial_parameter_sha256"]
    path = tmp_path / "training-result.json"
    record = json.loads(path.read_bytes())
    record["implementation_sha256"]["av.py"] = "0" * 64
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="revision"):
        load_trained(tmp_path, "av")


def test_duplicate_formal_attempt_preserves_original_receipt_and_never_trains(tmp_path, monkeypatch):
    old = tmp_path / "formal-job"
    old.mkdir()
    (old / "job-origin.json").write_text('{"original":true}')
    (old / "job-result.json").write_text('{"success":true}')
    selected = {"kind": "av", "training": {"training_seconds": 2}}
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    (frozen / "formal-recipes.json").write_text("{}")
    document = {"data_definition_sha256": "data", "metrics_sha256": "metrics"}
    monkeypatch.setattr(formal_local, "checked_job", lambda *args: (selected, document))
    monkeypatch.setattr(formal_local, "train", lambda *args: pytest.fail("duplicate must not train"))
    with pytest.raises(RuntimeError):
        formal_local.execute("formal-job", frozen, tmp_path)
    assert (old / "job-result.json").read_text() == '{"success":true}'
    assert not (tmp_path / ".formal-running").exists()


def test_missing_absolute_deadline_fails_before_any_origin_or_lock(tmp_path, monkeypatch):
    monkeypatch.delattr(formal_local.signal, "SIGALRM", raising=False)
    monkeypatch.setattr(
        formal_local, "checked_job", lambda *args: pytest.fail("must reject before entering formal work")
    )
    with pytest.raises(RuntimeError, match="POSIX"):
        formal_local.execute("job", tmp_path, tmp_path / "outputs")
    assert not (tmp_path / "outputs").exists()


def test_source_bound_resume_refuses_before_model_or_optimizer_mutation(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    generator = torch.Generator().manual_seed(1)
    before = trainer.parameter_sha(model)
    path = tmp_path / "checkpoint.pt"
    digest = trainer.save_checkpoint(path, model, optimizer, generator, 2, {"implementation": "old"})
    with pytest.raises(ValueError, match="來源"):
        trainer.restore_checkpoint(
            path, digest, model, optimizer, generator, expected_metadata={"implementation": "new"}
        )
    assert trainer.parameter_sha(model) == before and not optimizer.state


def test_new_directory_resume_keeps_original_best_without_dev_improvement(tmp_path, monkeypatch):
    config = {
        "seed": 11,
        "updates": 4,
        "training_seconds": 60,
        "batch_size": 2,
        "learning_rate": 0.01,
        "validate_every": 1,
    }
    saved = tmp_path / "step2.pt"
    original_save = trainer.save_checkpoint

    def copying_save(path, model, optimizer, generator, step, metadata, auxiliary=None):
        digest = original_save(path, model, optimizer, generator, step, metadata, auxiliary)
        if step == 2 and path.name == "latest-continuation.pt":
            shutil.copyfile(path, saved)
        return digest

    monkeypatch.setattr(trainer, "save_checkpoint", copying_save)

    def batch(generator):
        return torch.randn(2, 2), torch.randint(2, (2, 1), generator=generator).float()

    def loss(model, row):
        return {"loss": (model(row[0]) - row[1]).square().mean()}

    trainer.deterministic(11)
    full = trainer.run_updates(torch.nn.Linear(2, 1), batch, loss, lambda model: 0.5, config, tmp_path / "full")
    digest = trainer.file_sha(saved)
    trainer.deterministic(999)
    resumed = trainer.run_updates(
        torch.nn.Linear(2, 1), batch, loss, lambda model: 0.5, config, tmp_path / "resumed", resume=(saved, digest)
    )
    assert resumed["complete"] and resumed["best"]["update"] == 1
    assert resumed["latest_parameter_sha256"] == full["latest_parameter_sha256"]
    assert resumed["files"]["weights.safetensors"]["sha256"] == full["files"]["weights.safetensors"]["sha256"]
    assert (
        load_file(str(tmp_path / "resumed/weights.safetensors")).keys()
        == load_file(str(tmp_path / "full/weights.safetensors")).keys()
    )
