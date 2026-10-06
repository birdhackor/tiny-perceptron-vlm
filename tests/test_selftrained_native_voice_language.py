"""Native voice objective and actual trainer IO checks on synthetic CPU assets."""

import copy
import json
import signal

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


def assert_state_equal(left, right):
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_state_equal(a, b)
    else:
        assert left == right


def metrics(directory):
    return [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]


def completed_source_trace(directory, architecture, steps, run_id="gha-synthetic44", native=1, numeric=4):
    """Fixture only: actual synthetic main outputs with the real same-directory Volume schema."""
    job = {
        "schema_version": 1,
        "stage": "joint",
        "architecture": architecture,
        "steps": steps,
        "batch_size": 16,
        "context": 512,
        "learning_rate": 0.0002,
        "seed": 20261006,
        "eval_every": 1,
        "save_every": 1,
        "sampling_mode": "task-family",
        "freeze_perception_backbones": True,
        "tool_loss_weight": 4,
        "numeric_run_loss_weight": numeric,
    }
    if native != 1:
        job["native_voice_loss_weight"] = native
    training = json.loads((directory / "train-receipt.json").read_text())
    job["init_checkpoint"] = {
        "stage": "joint",
        "run_id": "gha-synthetic-parent",
        "path": "best.pt",
        "sha256": training["origin"]["new_joint_initialization"]["source_checkpoint_sha256"],
    }
    execution = {
        "status": "completed" if training["completed_requested_steps"] else "failed",
        "stage": "joint",
        "returncode": 0,
        "revision": "a" * 40,
        "run_id": run_id,
        "manifest_sha256": "b" * 64,
        "job": job,
        "synthetic_cpu_fixture": True,
    }
    modal_runner.write_json(directory / "execution.json", execution)
    receipt = {
        **execution,
        "files": [
            {"path": path.name, "sha256": modal_runner.sha256(path), "bytes": path.stat().st_size}
            for path in sorted(directory.iterdir())
            if path.is_file() and path.name != "receipt.json"
        ],
    }
    modal_runner.write_json(directory / "receipt.json", receipt)
    selected_name = "best.pt" if (directory / "best.pt").is_file() else "latest.pt"
    descriptor = {
        "run_id": run_id,
        "stage": "joint",
        "path": selected_name,
        "sha256": modal_runner.sha256(directory / selected_name),
    }
    return descriptor


def restamp_synthetic_trace(directory):
    receipt = json.loads((directory / "receipt.json").read_text())
    for item in receipt["files"]:
        path = directory / item["path"]
        item.update(sha256=modal_runner.sha256(path), bytes=path.stat().st_size)
    modal_runner.write_json(directory / "receipt.json", receipt)


@pytest.fixture
def native_corpus(tmp_path):
    Image.new("RGB", (128, 64), "gray").save(tmp_path / "pixels.png")
    sf.write(tmp_path / "wave.wav", np.sin(2 * np.pi * 300 * np.arange(8000) / 16000).astype(np.float32) * 0.1, 16000)
    rows = []
    tasks = (
        "text",
        "text_pretrain",
        "tool_call",
        "tool_reply",
        "vision_clothing",
        "ocr",
        "voice_qa",
        "voice_topic_continuation",
    )
    for split in ("train", "validation"):
        for i, task in enumerate(tasks):
            row = {
                "id": f"synthetic-{split}-{i}",
                "group_id": f"synthetic-{split}-{i}",
                "task": task,
                "split": split,
                "messages": [
                    {"role": "user", "content": "確認格式"},
                    {"role": "assistant", "content": "好1。"},
                    {"role": "user", "content": "問題"},
                    {"role": "assistant", "content": "1.第一\n2.第二"},
                ],
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
            if task.startswith("voice"):
                row.update(audio="wave.wav", modality_message_index=2)
                row["supervision"].update(intent="address", intent_id=0)
            rows.append(row)
            if task.startswith("voice"):
                aug = copy.deepcopy(row)
                aug["id"] += "-aug"
                aug["augmentation"] = {"variant": "synthetic"}
                rows.append(aug)
    record_path = tmp_path / "records.jsonl"
    record_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"width": 16, "layers": 1, "heads": 2, "kv_heads": 1, "ffn_hidden": 32, "experts": 2, "top_k": 1})
    )
    return tmp_path, record_path, config_path, rows


def common_args(corpus, architecture):
    root, record_path, config_path, _ = corpus
    return [
        "--records",
        str(record_path),
        "--asset-dir",
        str(root),
        "--config",
        str(config_path),
        "--stage",
        "joint",
        "--architecture",
        architecture,
        "--batch-size",
        "16",
        "--context",
        "512",
        "--learning-rate",
        "0.0002",
        "--seed",
        "20261006",
        "--eval-every",
        "1",
        "--save-every",
        "1",
        "--freeze-perception-backbones",
        "--sampling-mode",
        "task-family",
        "--export-inference",
    ]


def test_native_eligibility_shift_mask_fp32_and_graph_identity():
    tokenizer = CharacterTokenizer(sorted("1a-"))
    native = {"task": "voice_qa", "split": "train", "supervision": {}}
    for change in (
        {"augmentation": {}},
        {"augmentation": None},
        {"split": "validation"},
        {"task": "text"},
        {"evaluation_only": True},
        {"supervision": {"protocol_fixture": True}},
    ):
        assert not train.native_voice_loss_row({**native, **change})
    assert train.native_voice_loss_row(native)
    labels = torch.tensor([[-100, tokenizer.character_ids["a"], tokenizer.character_ids["1"], tokenizer.eos_id, -100]])
    assert train.language_loss_weights(labels, [native], tokenizer, 4, 4, 4).tolist() == [[0, 4, 16, 16, 0]]
    logits = torch.randn(1, 5, tokenizer.vocab_size, requires_grad=True)
    output = {"logits": logits, "language_loss": torch.tensor(19.0, requires_grad=True)}
    assert train.weighted_language_loss(output, labels, [native], tokenizer, 1, 1, 1) is output["language_loss"]
    plain = logits.detach().clone().requires_grad_()
    old = train.weighted_language_loss({"logits": plain}, labels, [native], tokenizer, 4, 4)
    new = train.weighted_language_loss(output, labels, [native], tokenizer, 4, 4, 4)
    old.backward()
    new.backward()
    assert torch.equal(logits.grad, plain.grad)  # Every row is native: uniform factor cancels.
    assert not logits.grad[labels.eq(-100)].any()
    long_labels = torch.full((1, 4096), tokenizer.character_ids["1"])
    half = torch.randn(1, 4096, tokenizer.vocab_size, dtype=torch.float16, requires_grad=True)
    finite = train.weighted_language_loss({"logits": half}, long_labels, [native], tokenizer, 4, 4, 4)
    assert finite.dtype == torch.float32 and torch.isfinite(finite)
    assert train.language_loss_weights(long_labels, [native], tokenizer, 4, 4, 4).sum().item() == 65536
    empty = CharacterTokenizer(["a"])
    assert not train.numeric_run_loss_mask(torch.tensor([[empty.character_ids["a"], -100]]), empty).any()


@pytest.mark.parametrize("value", [True, False, None, "4", 0.99, float("nan"), float("inf")])
def test_native_invalid_coefficients_and_job_bounds(value):
    with pytest.raises(ValueError):
        train.validate_loss_weights("joint", 4, 4, value)
    with pytest.raises(ValueError):
        modal_runner.validate_job({"schema_version": 1, "stage": "joint", "native_voice_loss_weight": value})


def test_native_full_policy_resume_and_default_metadata():
    default = {"stage": "joint", "tool_loss_weight": 4, "numeric_run_loss_weight": 4}
    old = train.language_objective_metadata(default)
    assert old == train.language_objective_metadata({**default, "native_voice_loss_weight": 1})
    assert old["language_objective_policy"]["version"] == "selftrained-language-objective-v1"
    assert old == modal_runner.native_objective_metadata(1, 4)
    assert (
        train.language_objective_metadata({**default, "numeric_run_loss_weight": 1, "native_voice_loss_weight": 4})
        == modal_runner.native_objective_metadata()
    )
    assert "native_voice_loss_weight" not in old
    new_options = {**default, "native_voice_loss_weight": 4}
    metadata = train.language_objective_metadata(new_options)
    new = {"stage": "joint", "training_options": {**new_options, **metadata}, **metadata}
    train.validate_resume_language_objective(new, new_options)
    for changed in (
        default,
        {**new_options, "native_voice_loss_weight": 2},
        {**new_options, "numeric_run_loss_weight": 2},
    ):
        with pytest.raises(ValueError):
            train.validate_resume_language_objective(new, changed)
    for field in metadata["language_objective_policy"]:
        broken = copy.deepcopy(new)
        broken["language_objective_policy"][field] = "changed"
        with pytest.raises(ValueError):
            train.validate_resume_language_objective(broken, new_options)
    with pytest.raises(ValueError):
        train.validate_resume_language_objective({"stage": "joint", "training_options": default, **old}, new_options)
    train.validate_resume_language_objective({"stage": "joint", "training_options": {}}, {"stage": "joint"})


@pytest.mark.parametrize("architecture", ["moe", "dense"])
def test_native_actual_main_fresh_resume_source_pins_export_and_selector(
    native_corpus, architecture, monkeypatch, capsys
):
    root, _, _, _ = native_corpus
    common = common_args(native_corpus, architecture)
    weights = ["--tool-loss-weight", "4", "--numeric-run-loss-weight", "4"]
    train.main(common + ["--steps", "2", "--output-dir", str(root / "old11")])
    source_dir = root / "experiments/selftrained-v2/joint/gha-synthetic44"
    train.main(
        common
        + weights
        + ["--steps", "2", "--output-dir", str(source_dir), "--init-checkpoint", str(root / "old11/best.pt")]
    )
    descriptor = completed_source_trace(source_dir, architecture, 2)
    source_path = source_dir / "best.pt"
    source = torch.load(source_path, weights_only=False)
    monkeypatch.setattr(modal_runner, "EXPERIMENTS", root / "experiments")
    path, _ = modal_runner.artifact_gate("selftrained-v2", descriptor, "b" * 64, architecture)
    evidence = modal_runner.native_joint_source_gate(path, architecture, descriptor, "b" * 64)
    target = {
        **evidence["job"],
        "steps": 4000,
        "numeric_run_loss_weight": 1,
        "native_voice_loss_weight": 4,
        "eval_every": 1000,
        "save_every": 1000,
    }
    manifest = {
        "records": [{"path": name, "sha256": digest} for name, digest in source["data_sha256"].items()],
        "assets": [{"path": name, "sha256": digest} for name, digest in source["asset_sha256"].items()],
        "model_config": source["config"],
    }
    modal_runner.native_joint_source_gate(path, architecture, descriptor, "b" * 64, target, manifest)
    for change in (
        {"seed": 42},
        {"batch_size": 32},
        {"learning_rate": 0.0003},
        {"context": 128},
        {"freeze_perception_backbones": False},
        {"sampling_mode": "bucket"},
        {"architecture": "dense" if architecture == "moe" else "moe"},
        {"router_weight": 0.02},
    ):
        with pytest.raises(ValueError, match="before reserve"):
            modal_runner.native_joint_source_gate(
                path, architecture, descriptor, "b" * 64, {**target, **change}, manifest
            )
    for changed_manifest in (
        {**manifest, "model_config": {**manifest["model_config"], "width": 32}},
        {**manifest, "records": [{"path": "records.jsonl", "sha256": "0" * 64}]},
    ):
        with pytest.raises(ValueError, match="before reserve"):
            modal_runner.native_joint_source_gate(path, architecture, descriptor, "b" * 64, target, changed_manifest)
    with pytest.raises(ValueError, match="objective/full policy"):
        modal_runner.native_resume_source_gate(
            source_dir / "latest.pt", target, manifest_sha="b" * 64, manifest=manifest
        )
    assert evidence["binding"]["source_checkpoint_sha256"] == modal_runner.sha256(source_path)
    with pytest.raises(ValueError):
        modal_runner.native_joint_source_gate(path, architecture, {**descriptor, "sha256": "c" * 64}, "b" * 64)
    native = ["--tool-loss-weight", "4", "--numeric-run-loss-weight", "1", "--native-voice-loss-weight", "4"]
    reset_seen = []
    original_objective = train.objective

    def observe_fresh(model, encoder, records, *args, **kwargs):
        if args[-1] == 4 and not reset_seen:
            assert all(torch.equal(value, source["model"][name]) for name, value in model.state_dict().items())
            assert [record["id"] for record in records] == [
                record["id"]
                for record in train.BalancedSampler(
                    train.stage_records(native_corpus[3], "joint", "train"), 20261006, "task-family"
                ).batch(16)
            ]
            reset_seen.append(True)
        return original_objective(model, encoder, records, *args, **kwargs)

    with monkeypatch.context() as patcher:
        patcher.setattr(train, "objective", observe_fresh)
        patcher.setattr(train, "restore_rng", lambda state: pytest.fail("fresh stage restored source RNG"))
        patcher.setattr(
            torch.optim.AdamW, "load_state_dict", lambda *args: pytest.fail("fresh stage restored source optimizer")
        )
        patcher.setattr(
            train.BalancedSampler, "load_state_dict", lambda *args: pytest.fail("fresh stage restored source sampler")
        )
        train.main(
            common
            + native
            + ["--steps", "2", "--output-dir", str(root / "native-whole"), "--init-checkpoint", str(source_path)]
        )
    assert reset_seen == [True]
    original_step = torch.optim.AdamW.step

    def interrupt_after_step(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        signal.raise_signal(signal.SIGTERM)
        return result

    with monkeypatch.context() as patcher:
        patcher.setattr(torch.optim.AdamW, "step", interrupt_after_step)
        train.main(
            common
            + native
            + ["--steps", "2", "--output-dir", str(root / "native-part"), "--init-checkpoint", str(source_path)]
        )
    part = torch.load(root / "native-part/latest.pt", weights_only=False)
    assert all(state["step"].item() == 1 for state in part["optimizer"]["state"].values())
    assert part["step"] == 1 and part["sampler"]["draws"] == 16
    part_receipt = json.loads((root / "native-part/train-receipt.json").read_text())
    assert part_receipt["interrupted"] is True and part_receipt["completed_requested_steps"] is False
    resume_descriptor = completed_source_trace(
        root / "native-part", architecture, 2, "gha-native-part", native=4, numeric=1
    )
    resume_descriptor.update(path="latest.pt", sha256=modal_runner.sha256(root / "native-part/latest.pt"))
    resume_target = {**target, "eval_every": 1, "save_every": 1, "steps": 2}
    modal_runner.native_resume_source_gate(
        root / "native-part/latest.pt", resume_target, resume_descriptor, "b" * 64, manifest
    )
    for change in (
        {"seed": 42},
        {"learning_rate": 0.0003},
        {"batch_size": 32},
        {"numeric_run_loss_weight": 4},
        {"native_voice_loss_weight": 1},
        {"eval_every": 1000},
        {"save_every": 1000},
    ):
        with pytest.raises(ValueError):
            modal_runner.native_resume_source_gate(
                root / "native-part/latest.pt", {**resume_target, **change}, resume_descriptor, "b" * 64, manifest
            )
    train.main(
        common
        + native
        + [
            "--steps",
            "2",
            "--output-dir",
            str(root / "native-resumed"),
            "--resume",
            str(root / "native-part/latest.pt"),
        ]
    )
    whole = torch.load(root / "native-whole/latest.pt", weights_only=False)
    resumed = torch.load(root / "native-resumed/latest.pt", weights_only=False)
    for field in (
        "model",
        "optimizer",
        "rng",
        "sampler",
        "step",
        "tokens",
        "target_tokens",
        "best_validation_loss",
        "stage_history",
        "origin",
        "language_objective_policy",
        "native_voice_loss_weight",
    ):
        assert_state_equal(whole[field], resumed[field])
    whole_train = [row for row in metrics(root / "native-whole") if row["event"] == "train"]
    resumed_train = [
        row
        for directory in (root / "native-part", root / "native-resumed")
        for row in metrics(directory)
        if row["event"] == "train"
    ]
    assert whole_train == resumed_train
    assert whole_train[0]["step"] == 1 and whole_train[0]["tokens"] == part["tokens"]
    assert whole_train[0]["native_voice_target_tokens"] > 0
    initialization = whole["origin"]["new_joint_initialization"]
    assert initialization["source_integrity"] == evidence["binding"]
    assert initialization["loaded_step"] == source["step"]
    assert initialization["source_completed_steps"] == 2
    assert initialization["reset_state"] == {
        "optimizer": "new",
        "rng": "fresh_stage_seed_not_source_rng",
        "sampler_draws": 0,
        "family_draws": 0,
        "stage_step": 0,
        "tokens": 0,
        "target_tokens": 0,
    }
    assert whole["stage_history"][:-1] == source["stage_history"]
    for directory in (root / "native-whole", root / "native-resumed"):
        receipt = json.loads((directory / "train-receipt.json").read_text())
        manifest, _ = verify_export(directory)
        selected = torch.load(directory / "best.pt", weights_only=False)
        metadata = train.language_objective_metadata(
            {"stage": "joint", "tool_loss_weight": 4, "numeric_run_loss_weight": 1, "native_voice_loss_weight": 4}
        )
        for document in (receipt, manifest, selected, selected["training_options"]):
            assert all(document[key] == value for key, value in metadata.items())
        assert manifest["selected_checkpoint_sha256"] == modal_runner.sha256(directory / "best.pt")
        model = LimitedAssistant(SelftrainedConfig(**selected["config"]))
        model.load_state_dict(selected["model"])
        encoder = RecordEncoder(CharacterTokenizer.from_dict(selected["tokenizer"]), root, 512)
        value, _ = train.validation_loss(
            model, encoder, train.stage_records(native_corpus[3], "joint", "validation"), "joint", 16, 1.0, 0.01
        )
        assert value == receipt["best_validation_loss"]
        train.set_trainable(model, "joint", True)
        frozen = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]
        assert frozen and all(torch.equal(source["model"][name], whole["model"][name]) for name in frozen)
    from safetensors.torch import load_file

    assert_state_equal(
        load_file(str(root / "native-whole/model.safetensors")),
        load_file(str(root / "native-resumed/model.safetensors")),
    )
    for change in ({"stage": "sft"}, {"tool_loss_weight": 1}, {"native_voice_loss_weight": 1}):
        with pytest.raises(ValueError):
            train.validate_resume_language_objective(whole, {**whole["training_options"], **change})
    # Restamped synthetic metadata still cannot bypass mismatching actual checkpoint/job settings.
    receipt_file = source_dir / "receipt.json"
    saved_receipt = receipt_file.read_bytes()
    outer = json.loads(saved_receipt)
    outer["job"]["learning_rate"] = 0.0003
    modal_runner.write_json(receipt_file, outer)
    with pytest.raises(ValueError):
        modal_runner.native_joint_source_gate(source_path, architecture)
    receipt_file.write_bytes(saved_receipt)
    training_file = source_dir / "train-receipt.json"
    saved_training = training_file.read_bytes()
    broken = json.loads(saved_training)
    broken["batch_size"] = 32
    modal_runner.write_json(training_file, broken)
    restamp_synthetic_trace(source_dir)
    with pytest.raises(ValueError, match="batch_size"):
        train.main(
            common
            + native
            + ["--steps", "1", "--output-dir", str(root / "bad-receipt"), "--init-checkpoint", str(source_path)]
        )
    training_file.write_bytes(saved_training)
    receipt_file.write_bytes(saved_receipt)
    for name, value in (("batch_size", 32), ("learning_rate", 0.0003)):
        changed = json.loads(saved_training)
        changed[name] = value
        modal_runner.write_json(training_file, changed)
        restamp_synthetic_trace(source_dir)
        with pytest.raises(ValueError, match="before reserve"):
            modal_runner.native_joint_source_gate(source_path, architecture, descriptor, "b" * 64, target, manifest)
        training_file.write_bytes(saved_training)
        receipt_file.write_bytes(saved_receipt)
    with pytest.raises(ValueError, match="selected_validation_best"):
        train.main(
            common
            + native
            + [
                "--steps",
                "1",
                "--output-dir",
                str(root / "bad-latest"),
                "--init-checkpoint",
                str(source_dir / "latest.pt"),
            ]
        )
    capsys.readouterr()
