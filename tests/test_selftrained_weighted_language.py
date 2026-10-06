"""CPU objective/lineage wiring checks; synthetic corpus, no model generation."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
from PIL import Image

from scripts.selftrained import modal_runner, train
from tiny_perceptron.selftrained.dataset import RecordEncoder
from tiny_perceptron.selftrained.inference import verify_export
from tiny_perceptron.selftrained.model import LimitedAssistant, SelftrainedConfig
from tiny_perceptron.selftrained.tokenizer import CharacterTokenizer


def descriptor():
    return {"run_id": "own-joint", "stage": "joint", "path": "best.pt", "sha256": "a" * 64}


def job():
    return {"schema_version": 1, "stage": "joint", "init_checkpoint": descriptor()}


def test_numeric_union_mask_and_missing_digit_vocabulary():
    tokenizer = CharacterTokenizer(sorted(set("01279-,。.")))
    labels = torch.tensor(
        [
            tokenizer.encode("12-79。-。1.2")
            + [tokenizer.eos_id, -100, tokenizer.character_ids["-"], -100, tokenizer.character_ids["1"]]
        ]
    )
    assert train.numeric_run_loss_mask(labels, tokenizer).tolist() == [
        [True, True, True, True, True, True, False, False, True, True, True, True, False, False, False, True]
    ]
    empty = CharacterTokenizer(["a"])
    assert not train.numeric_run_loss_mask(torch.tensor([[empty.character_ids["a"], -100, empty.eos_id]]), empty).any()
    no_minus = CharacterTokenizer(["1"])
    assert train.numeric_run_loss_mask(
        torch.tensor([[no_minus.character_ids["1"], no_minus.eos_id, -100]]), no_minus
    ).tolist() == [[True, True, False]]


@pytest.mark.parametrize(
    "task,supervision,evaluation,eligible",
    [
        ("tool_call", {}, False, True),
        ("tool_call", {"protocol_fixture": True}, False, False),
        ("tool_call", {}, True, False),
        ("tool_reply", {"replay_kind": "actual_executor"}, False, True),
        ("tool_reply", {"replay_kind": "actual_executor", "protocol_fixture": True}, False, False),
        ("tool_reply", {"replay_kind": "counterfactual_value"}, False, False),
        ("tool_reply", {"replay_kind": "structured_value_reading_train_only"}, False, False),
        ("tool_missing", {}, False, False),
        ("text", {}, False, False),
    ],
)
def test_only_existing_targets_and_whitelisted_rows(task, supervision, evaluation, eligible):
    tokenizer = CharacterTokenizer(sorted(set("1a")))
    labels = torch.tensor([[-100, tokenizer.character_ids["a"], tokenizer.character_ids["1"], tokenizer.eos_id, -100]])
    record = {"task": task, "supervision": supervision, "evaluation_only": evaluation}
    weights = train.language_loss_weights(labels, [record], tokenizer, 4, 4)
    row = 4 if eligible else 1
    assert weights.tolist() == [[0, row, 4 * row, 4 * row, 0]]
    assert torch.equal(weights.ne(0), labels.ne(-100))


def test_fp16_weighted_reduction_is_fp32_and_aligned_without_second_shift():
    tokenizer = CharacterTokenizer(["1", "a"])
    labels = torch.full((1, 4097), tokenizer.character_ids["1"])
    labels[0, -1] = -100
    logits = torch.randn(1, 4097, tokenizer.vocab_size, dtype=torch.float16, requires_grad=True)
    records = [{"task": "tool_call", "supervision": {}}]
    output = {"logits": logits, "language_loss": torch.tensor(17.0)}
    assert train.weighted_language_loss(output, labels, records, tokenizer) is output["language_loss"]
    weighted = train.weighted_language_loss(output, labels, records, tokenizer, 4, 4)
    expected = torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, tokenizer.vocab_size), labels.reshape(-1), ignore_index=-100
    )
    assert weighted.dtype == torch.float32 and torch.isfinite(weighted)
    assert torch.allclose(weighted, expected, atol=1e-6)
    weights = train.language_loss_weights(labels, records, tokenizer, 4, 4)
    assert weights.sum().item() == 65536
    weighted.backward()
    assert torch.isfinite(logits.grad).all() and not logits.grad[0, -1].any()


@pytest.mark.parametrize("value", [True, False, None, "4", float("inf"), float("nan"), 0.99])
def test_invalid_loss_coefficients_rejected(value):
    with pytest.raises(ValueError):
        train.validate_loss_weights("joint", value, 1)
    with pytest.raises(ValueError):
        train.validate_loss_weights("joint", 1, value)
    for field in ("tool_loss_weight", "numeric_run_loss_weight"):
        with pytest.raises(ValueError):
            modal_runner.validate_job({**job(), field: value})


def test_nonjoint_rejected_before_data_loading_and_runner_options_are_joint_only():
    for field in ("tool_loss_weight", "numeric_run_loss_weight"):
        flag = "--" + field.replace("_", "-")
        with pytest.raises(ValueError, match="只可用於 joint"):
            train.main(
                [
                    "--stage",
                    "sft",
                    "--steps",
                    "1",
                    "--records",
                    "absent",
                    "--asset-dir",
                    "absent",
                    "--output-dir",
                    "absent",
                    flag,
                    "4",
                ]
            )
        for stage in ("sft", "pretrain", "validation", "prepare", "release"):
            with pytest.raises(ValueError, match=field):
                modal_runner.validate_job({**job(), "stage": stage, field: 1})
    assert not train.validate_loss_weights("pretrain", 1, 1)


def test_resume_effective_coefficients_full_policy_and_legacy_defaults():
    legacy = {"stage": "joint", "training_options": {}}
    defaults = {"stage": "joint", "tool_loss_weight": 1, "numeric_run_loss_weight": 1}
    weighted = {"stage": "joint", "tool_loss_weight": 4, "numeric_run_loss_weight": 4}
    train.validate_resume_language_objective(legacy, defaults)
    with pytest.raises(ValueError, match="resume 改變"):
        train.validate_resume_language_objective(legacy, weighted)
    metadata = train.language_objective_metadata(weighted)
    same = {"stage": "joint", "training_options": {**weighted, **metadata}, **metadata}
    train.validate_resume_language_objective(same, weighted)
    for change in ({"tool_loss_weight": 3}, {"numeric_run_loss_weight": 3}):
        with pytest.raises(ValueError):
            train.validate_resume_language_objective(same, {**weighted, **change})
    missing = {"stage": "joint", "training_options": weighted}
    with pytest.raises(ValueError, match="缺少"):
        train.validate_resume_language_objective(missing, weighted)
    for location in ("top", "options"):
        broken = copy.deepcopy(same)
        target = broken if location == "top" else broken["training_options"]
        target["language_objective_policy"]["version"] = "different-version"
        with pytest.raises(ValueError, match="version/policy"):
            train.validate_resume_language_objective(broken, weighted)


def test_nondefault_init_requires_own_selected_joint_and_unchanged_frozen_options():
    args = SimpleNamespace(
        init_checkpoint="own.pt",
        resume=None,
        freeze_perception_backbones=True,
        sampling_mode="task-family",
        context=128,
    )
    source = {
        "origin": {"kind": "all-neural-weights-random"},
        "stage": "joint",
        "checkpoint_kind": "selected_validation_best",
        "config": {"max_length": 128},
        "training_options": {"freeze_perception_backbones": True, "sampling_mode": "task-family"},
    }
    train.validate_weighted_source(source, args)
    for change in (
        {"checkpoint_kind": "latest"},
        {"stage": "sft"},
        {"origin": {"kind": "pretrained"}},
        {"config": {"max_length": 256}},
        {"training_options": {"freeze_perception_backbones": False, "sampling_mode": "task-family"}},
    ):
        with pytest.raises(ValueError):
            train.validate_weighted_source({**source, **change}, args)


def test_runner_preserves_old_job_and_omits_absent_flags(tmp_path, monkeypatch):
    old = job()
    monkeypatch.setattr(modal_runner, "EXPERIMENTS", tmp_path)
    prior = tmp_path / "batch/joint/own-joint/best.pt"
    prior.parent.mkdir(parents=True)
    prior.write_bytes(b"synthetic saved artifact")
    old["init_checkpoint"]["sha256"] = modal_runner.sha256(prior)
    saved = copy.deepcopy(old)
    assert modal_runner.validate_job(old) is old and old == saved
    manifest = {"records": [{"path": "train.jsonl"}], "model_config": {"width": 16}}
    command = modal_runner.trainer_command(old, manifest, tmp_path, tmp_path, "batch")
    assert "--tool-loss-weight" not in command and "--numeric-run-loss-weight" not in command
    new = {**old, "tool_loss_weight": 4, "numeric_run_loss_weight": 4}
    modal_runner.validate_job(new)
    changed = modal_runner.trainer_command(new, manifest, tmp_path, tmp_path, "batch")
    for flag in ("--tool-loss-weight", "--numeric-run-loss-weight"):
        assert changed[changed.index(flag) + 1] == "4"


@pytest.fixture
def synthetic_family_corpus(tmp_path):
    Image.fromarray(np.arange(64 * 128, dtype=np.uint8).reshape(64, 128)).save(tmp_path / "pixels.png")
    sf.write(tmp_path / "wave.wav", np.sin(np.arange(8000) * 0.05).astype("float32"), 8000)
    rows = []
    for split in ("train", "validation"):
        for task in ("text", "tool_call", "tool_reply", "vision_clothing", "ocr", "voice_qa"):
            row = {
                "id": f"{split}-{task}",
                "group_id": f"{split}-{task}",
                "split": split,
                "task": task,
                "messages": [{"role": "user", "content": "請算12。"}, {"role": "assistant", "content": "結果-7。"}],
                "supervision": {},
            }
            if task == "tool_reply":
                row["supervision"]["replay_kind"] = "actual_executor"
            if task == "vision_clothing":
                row.update(
                    image="pixels.png", image_layout={"axis": "horizontal", "slots": [[0, 0, 64, 64], [64, 0, 128, 64]]}
                )
                row["supervision"]["vision_labels"] = [0, 1]
            if task == "ocr":
                row.update(image="pixels.png", roi=[0, 0, 100, 32])
                row["supervision"]["ocr_text"] = "大小"
                row["messages"][-1]["content"] = "大小"
            if task == "voice_qa":
                row.update(audio="wave.wav", modality_message_index=0)
                row["supervision"].update(intent="address", intent_id=0)
            rows.append(row)
    records = tmp_path / "records.jsonl"
    records.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"width": 16, "layers": 1, "heads": 2, "kv_heads": 1, "ffn_hidden": 32, "experts": 2, "top_k": 1})
    )
    return tmp_path, records, config, rows


@pytest.mark.parametrize("architecture", ["moe", "dense"])
def test_new_joint_reset_exact_resume_receipts_export_and_unweighted_selection(
    synthetic_family_corpus, architecture, capsys
):
    root, records, config_path, rows = synthetic_family_corpus
    common = [
        "--records",
        str(records),
        "--asset-dir",
        str(root),
        "--stage",
        "joint",
        "--architecture",
        architecture,
        "--batch-size",
        "5",
        "--context",
        "128",
        "--config",
        str(config_path),
        "--eval-every",
        "10",
        "--save-every",
        "1",
        "--freeze-perception-backbones",
        "--sampling-mode",
        "task-family",
        "--export-inference",
    ]
    train.main(common + ["--output-dir", str(root / "baseline"), "--steps", "1"])
    source_path = root / "baseline/best.pt"
    source = torch.load(source_path, weights_only=False)
    old_history = copy.deepcopy(source["stage_history"])
    legacy = torch.load(root / "baseline/latest.pt", weights_only=False)
    for document in (legacy, legacy["selected_checkpoint"]):
        for key in ("tool_loss_weight", "numeric_run_loss_weight", "language_objective_policy"):
            document.pop(key, None)
            document["training_options"].pop(key, None)
    legacy_path = root / "legacy.pt"
    torch.save(legacy, legacy_path)
    train.main(common + ["--output-dir", str(root / "legacy-resumed"), "--steps", "2", "--resume", str(legacy_path)])
    legacy_metrics = [json.loads(line) for line in (root / "legacy-resumed/metrics.jsonl").read_text().splitlines()]
    assert all(
        "language_loss" in row and "weighted_language_loss" not in row and "unweighted_language_loss" not in row
        for row in legacy_metrics
        if row["event"] == "train"
    )
    weights = ["--tool-loss-weight", "4", "--numeric-run-loss-weight", "4"]
    for name, steps in (("whole", 2), ("part", 1)):
        train.main(
            common
            + weights
            + ["--output-dir", str(root / name), "--steps", str(steps), "--init-checkpoint", str(source_path)]
        )
    train.main(
        common
        + weights
        + ["--output-dir", str(root / "resumed"), "--steps", "2", "--resume", str(root / "part/latest.pt")]
    )
    whole = torch.load(root / "whole/latest.pt", weights_only=False)
    resumed = torch.load(root / "resumed/latest.pt", weights_only=False)
    part = torch.load(root / "part/latest.pt", weights_only=False)
    assert all(state["step"].item() == 1 for state in part["optimizer"]["state"].values())
    assert whole["step"] == resumed["step"] == 2
    assert whole["sampler"]["draws"] == resumed["sampler"]["draws"] == 10
    assert all(torch.equal(value, resumed["model"][name]) for name, value in whole["model"].items())
    assert whole["sampler"]["family_draws"] == resumed["sampler"]["family_draws"]
    assert torch.equal(whole["sampler"]["generator"], resumed["sampler"]["generator"])
    assert torch.equal(whole["rng"]["torch"], resumed["rng"]["torch"])
    first_baseline = json.loads((root / "baseline/metrics.jsonl").read_text().splitlines()[0])
    first_new = json.loads((root / "part/metrics.jsonl").read_text().splitlines()[0])
    assert first_new["step"] == 1 and first_new["sample_ids"] == first_baseline["sample_ids"]
    initialization = whole["origin"]["new_joint_initialization"]
    assert initialization["loaded_step"] == source["step"] == 1
    assert initialization["source_checkpoint_sha256"] == train.file_sha256(source_path)
    assert initialization["reset_state"]["stage_step"] == initialization["reset_state"]["sampler_draws"] == 0
    assert whole["stage_history"][:-1] == old_history
    assert whole["stage_history"][-1]["new_joint_initialization"] == initialization
    for field in ("data_sha256", "asset_sha256", "tokenizer_sha256", "config"):
        assert whole[field] == source[field]
    for directory in (root / "whole", root / "resumed"):
        receipt = json.loads((directory / "train-receipt.json").read_text())
        manifest, _ = verify_export(directory)
        selected = torch.load(directory / "best.pt", weights_only=False)
        metadata = train.language_objective_metadata(
            {"stage": "joint", "tool_loss_weight": 4, "numeric_run_loss_weight": 4}
        )
        for document in (receipt, manifest, selected, selected["training_options"]):
            assert all(document[key] == value for key, value in metadata.items())
        model = LimitedAssistant(SelftrainedConfig(**selected["config"]))
        model.load_state_dict(selected["model"])
        encoder = RecordEncoder(CharacterTokenizer.from_dict(selected["tokenizer"]), root, 128)
        validation = train.stage_records(train.read_records([records]), "joint", "validation")
        value, _ = train.validation_loss(model, encoder, validation, "joint", 5, 1.0, 0.01)
        assert value == receipt["best_validation_loss"]
        metrics = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
        assert all(
            "weighted_language_loss" in row and "unweighted_language_loss" in row and "language_loss" not in row
            for row in metrics
            if row["event"] == "train"
        )
    for source_file, options in ((source_path, weights), (root / "part/latest.pt", [])):
        with pytest.raises(ValueError, match="resume 改變"):
            train.main(
                common
                + options
                + ["--output-dir", str(root / "rejected"), "--steps", "2", "--resume", str(source_file)]
            )
    with pytest.raises(ValueError, match="selected_validation_best"):
        train.main(
            common
            + weights
            + [
                "--output-dir",
                str(root / "rejected-latest"),
                "--steps",
                "1",
                "--init-checkpoint",
                str(root / "baseline/latest.pt"),
            ]
        )
    capsys.readouterr()
