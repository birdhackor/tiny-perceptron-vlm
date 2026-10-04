"""Bounded v4 knobs and real historical CPU checkpoints, without remote calls."""

import ast
import contextlib
import hashlib
import json
import re
import struct
import threading
import time
from copy import deepcopy
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tiny_perceptron import natural_assistant as assistant

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/modal_natural.py"
TREE = ast.parse(SOURCE.read_text())


def modal_helpers():
    constants = {
        "SPEC",
        "ASR_VARIANTS",
        "TOTAL_CAP_USD",
        "PRIOR_RESERVED_FLOOR_USD",
        "MAX_JOB_USD",
        "AUXILIARY_SECONDS",
        "BUILD_AND_RETAINED_STORAGE_ALLOWANCE_USD",
        "MAX_EGRESS_GIB",
        "STAGES",
        "SELECTION_STAGES",
        "EXTERNAL_STAGES",
        "MAX_PRIVATE_BACKUP_BYTES",
    }
    functions = {
        "safe_name",
        "safe_relative",
        "sha256",
        "write_json",
        "checkpoint_names",
        "runtime_options",
        "runtime_contract",
        "verify_runtime_contract",
        "validate_release",
        "adapter_descriptor",
        "validate_selection",
        "selection_gate",
        "reservation_guard",
        "reservation",
        "require_reservation",
        "execute_stage",
        "main",
    }
    nodes = []
    for original in TREE.body:
        if isinstance(original, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in constants for target in original.targets
        ):
            nodes.append(original)
        elif isinstance(original, ast.FunctionDef) and original.name in functions:
            node = deepcopy(original)
            node.decorator_list = []
            nodes.append(node)
    namespace = {
        "Path": Path,
        "Decimal": Decimal,
        "ROUND_CEILING": ROUND_CEILING,
        "json": json,
        "re": re,
        "hashlib": hashlib,
        "threading": threading,
        "time": time,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


def settings(helper, **changed):
    values = dict(
        stage="train",
        steps=4,
        seed=42,
        max_seconds=3300,
        max_pixels=524288,
        learning_rate=0.0001,
        asr_variant="small",
        checkpoints="2,4",
        checkpoint="",
        compare="",
    )
    return helper["runtime_options"](**(values | changed))


@pytest.mark.parametrize(
    "changed",
    [
        {"steps": 3001},
        {"steps": 0},
        {"max_seconds": 3301},
        {"max_seconds": 0},
        {"max_pixels": 1048577},
        {"learning_rate": 0.0002},
        {"learning_rate": float("nan")},
        {"asr_variant": "latest"},
        {"checkpoints": "2,1"},
        {"checkpoints": "2,2"},
        {"checkpoints": "1,2,3"},
        {"checkpoints": "2,5"},
        {"checkpoints": "../../2"},
        {"stage": "validation"},
        {"checkpoint": "../adapter"},
        {"compare": "step-000002,step-000004"},
    ],
)
def test_reject_unbounded_or_ambiguous_settings_before_remote_call(changed):
    with pytest.raises(ValueError):
        settings(modal_helpers(), **changed)


def test_explicit_lr_asr_pins_and_checkpoint_comparison():
    helper = modal_helpers()
    assert helper["ASR_VARIANTS"] == assistant.ASR_VARIANTS
    config = settings(helper, asr_variant="turbo")
    assert config["learning_rate"] == 1e-4
    assert config["asr_model"] == "openai/whisper-large-v3-turbo"
    assert config["asr_revision"] == "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
    assert config["checkpoint_steps"] == [2, 4]
    other = settings(helper, stage="validation", checkpoints="", compare="step-000002,step-000004")
    assert other["adapter_checkpoints"] == ["step-000002", "step-000004"]
    with pytest.raises(ValueError):
        settings(helper, stage="validation", checkpoints="", compare="step-000002", checkpoint="step-000004")


def test_existing_defaults_remain_small_and_latest():
    main = next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    names = [arg.arg for arg in main.args.args][-len(main.args.defaults) :]
    defaults = dict(zip(names, [ast.literal_eval(value) for value in main.args.defaults], strict=True))
    assert defaults["learning_rate"] == 3e-5 and defaults["asr_variant"] == "small"
    assert defaults["adapter_checkpoint"] == defaults["adapter_checkpoints"] == defaults["checkpoint_steps"] == ""


def test_reservation_binds_runtime_settings_and_preserves_prior_spend(tmp_path):
    helper = modal_helpers()
    ledger_path = tmp_path / "budget.json"
    config = settings(helper)
    contract_hash = hashlib.sha256(helper["runtime_contract"](config).encode()).hexdigest()
    ledger = {
        "reservations": [{"run_id": "historical", "reserved_usd": "24.81", "status": "failed"}],
        "reserved_total_usd": "24.81",
    }
    rates = {"cpu_hour_cost": "0.0473", "mem_gib_hour_cost": "0.008", "gpu_hour_cost_l4": "0.8", "egress_gib_cost": "0"}
    entry = helper["reservation"](ledger, "new", "v4", "train", "a" * 40, "b" * 64, {"rates": rates})
    entry["runtime_options_sha256"] = contract_hash
    helper["write_json"](ledger_path, ledger)
    helper.update(LEDGER_PATH=ledger_path, volume=SimpleNamespace(commit=lambda: None))
    with pytest.raises(ValueError, match="Runtime settings differ"):
        helper["require_reservation"]("new", "v4", "train", "a" * 40, "b" * 64, runtime_sha="c" * 64)
    assert json.loads(ledger_path.read_text())["reservations"][-1]["status"] == "reserved"
    helper["require_reservation"]("new", "v4", "train", "a" * 40, "b" * 64, runtime_sha=contract_hash)
    result = json.loads(ledger_path.read_text())
    assert result["reservations"][0] == ledger["reservations"][0]
    assert Decimal(result["reserved_total_usd"]) > Decimal("24.81")


class TinyCPUAdapter(torch.nn.Module):
    """Real AdamW updates; minimal valid one-tensor safetensors for the fixture."""

    def __init__(self):
        super().__init__()
        self.lora_weight = torch.nn.Parameter(torch.tensor([0.5]))
        self.frozen = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=False)

    def forward(self, **batch):
        return SimpleNamespace(loss=((self.lora_weight - 0.9) ** 2).mean())

    def save_pretrained(self, path, *, safe_serialization):
        assert safe_serialization
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        header = json.dumps({"lora_weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
        header += b" " * (-len(header) % 8)
        value = struct.pack("<f", self.lora_weight.detach().item())
        (path / "adapter_model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + value)
        (path / "adapter_config.json").write_text('{"fixture": "one real trained CPU tensor"}\n')


def saved_value(path):
    data = path.read_bytes()
    header_length = struct.unpack("<Q", data[:8])[0]
    return struct.unpack("<f", data[8 + header_length :])[0]


def trained_fixture(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "rows": [
                    {"id": "one", "split": "train", "family": "one", "task": "text_chat", "user": "q", "answer": "a"}
                ],
                "sources": [],
            }
        )
    )
    options = SimpleNamespace(
        manifest=manifest,
        data_root=tmp_path,
        output=tmp_path / "train",
        adapter=None,
        model=assistant.MODEL_ID,
        model_revision=assistant.MODEL_REVISION,
        asr_model=assistant.ASR_ID,
        asr_revision=assistant.ASR_REVISION,
        device="cpu",
        dtype="float32",
        min_pixels=65536,
        max_pixels=524288,
        max_tokens=2048,
        max_seconds=100,
        seed=42,
        lora_rank=8,
        gradient_accumulation=2,
        learning_rate=1e-4,
        steps=4,
        checkpoint_steps=(2, 4),
        checkpoint_every=25,
    )
    model = TinyCPUAdapter()
    monkeypatch.setattr(assistant, "load_core", lambda *args, **kwargs: (model, object()))
    monkeypatch.setattr(
        assistant,
        "encode_training_row",
        lambda *args: {"input_ids": torch.tensor([[0, 1, 2]]), "labels": torch.tensor([[-100, 1, 2]])},
    )
    result = assistant.run_train(options)
    return options, model, result


def test_archives_are_actual_distinct_updates_and_resume_state(tmp_path, monkeypatch):
    options, model, result = trained_fixture(tmp_path, monkeypatch)
    first = options.output / "checkpoints/step-000002"
    second = options.output / "checkpoints/step-000004"
    assert result["status"] == "completed" and result["completed_steps"] == 4
    assert saved_value(first / "adapter_model.safetensors") != saved_value(second / "adapter_model.safetensors")
    assert saved_value(second / "adapter_model.safetensors") == model.lora_weight.detach().item()
    assert saved_value(second / "adapter_model.safetensors") == saved_value(
        options.output / "adapter/adapter_model.safetensors"
    )
    for directory, step in [(first, 2), (second, 4)]:
        record = json.loads((directory / "training.json").read_text())
        assert record["completed_steps"] == step and len(record["history"]) == step
        state = torch.load(directory / "training_state.pt", weights_only=True)
        assert state["optimizer"]["state"][0]["step"].item() == step
        metadata = json.loads((directory / "checkpoint.json").read_text())
        assert metadata["completed_steps"] == step
        for item in metadata["files"]:
            file = directory / item["path"]
            assert file.stat().st_size == item["bytes"]
            assert hashlib.sha256(file.read_bytes()).hexdigest() == item["sha256"]
    with pytest.raises(ValueError, match="immutable"):
        assistant.archive_checkpoint(model, torch.optim.AdamW([model.lora_weight]), options.output, result)


def test_selection_accepts_only_the_checkpoint_actually_validated(tmp_path, monkeypatch):
    options, _, trained = trained_fixture(tmp_path, monkeypatch)
    helper = modal_helpers()
    helper["NATURAL_ROOT"] = tmp_path / "course"
    train_root = helper["NATURAL_ROOT"] / "v4/train/run-1"
    train_root.parent.mkdir(parents=True)
    import shutil

    shutil.copytree(options.output, train_root)
    actual = helper["adapter_descriptor"]("v4", "run-1", "step-000002", trained["manifest_sha256"])
    report = {
        "status": "completed",
        "split": "validation",
        "variants": {actual["variant"]: {"completed": True}},
        "execution": {
            "stage": "validation",
            "run_id": "val-1",
            "batch_id": "v4",
            "manifest_sha256": trained["manifest_sha256"],
            "adapters": [actual],
        },
    }
    report_path = helper["NATURAL_ROOT"] / "v4/validation/val-1/result.json"
    helper["write_json"](report_path, report)
    selected = {
        "dataset_manifest_sha256": trained["manifest_sha256"],
        "validation_run_id": "val-1",
        "validation_result_sha256": helper["sha256"](report_path),
        "selected_variant": actual["variant"],
        "adapter_run_id": "run-1",
        "adapter_sha256": actual["adapter_sha256"],
        "adapter_checkpoint": "step-000002",
        "criterion": "frozen",
        "decision": "first epoch",
    }
    assert (
        helper["selection_gate"](selected, trained["manifest_sha256"], "run-1", "v4", "step-000002")[
            "adapter_checkpoint"
        ]
        == "step-000002"
    )
    with pytest.raises(ValueError, match="exact archived checkpoint"):
        helper["selection_gate"](selected, trained["manifest_sha256"], "run-1", "v4", "step-000004")
    (train_root / "checkpoints/step-000002/adapter_model.safetensors").write_bytes(b"modified")
    with pytest.raises(ValueError, match="exact saved update"):
        helper["selection_gate"](selected, trained["manifest_sha256"], "run-1", "v4", "step-000002")


def test_paired_validation_switches_two_adapters_without_reloading_asr(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "rows": [
                    {"id": "one", "split": "validation", "family": "one", "task": "ocr", "user": "q", "answer": "a"}
                ],
                "audio_rows": [
                    {
                        "id": "speech",
                        "split": "validation",
                        "family": "voice",
                        "audio": "a.wav",
                        "user": "ref",
                        "answer": None,
                    }
                ],
            }
        )
    )
    (tmp_path / "a.wav").write_bytes(b"fixture not decoded")
    options = SimpleNamespace(
        manifest=manifest,
        data_root=tmp_path,
        output=tmp_path / "evaluation",
        adapter=tmp_path / "first",
        comparison_adapters=[("adapter-step-000002", tmp_path / "first"), ("adapter-step-000004", tmp_path / "second")],
        model=assistant.MODEL_ID,
        model_revision=assistant.MODEL_REVISION,
        asr_model=assistant.ASR_ID,
        asr_revision=assistant.ASR_REVISION,
        device="cpu",
        dtype="float32",
        min_pixels=65536,
        max_pixels=524288,
        max_tokens=2048,
        max_seconds=100,
        seed=42,
        split="validation",
    )

    class Model(torch.nn.Linear):
        active_adapter = "default"
        disabled = False

        def set_adapter(self, name):
            self.active_adapter = name

        def load_adapter(self, path, *, adapter_name, is_trainable):
            assert path == tmp_path / "second" and not is_trainable

        @contextlib.contextmanager
        def disable_adapter(self):
            self.disabled = True
            try:
                yield
            finally:
                self.disabled = False

    model = Model(1, 1)
    monkeypatch.setattr(assistant, "load_core", lambda *args, **kwargs: (model, object()))
    asr_calls = []
    monkeypatch.setattr(assistant, "load_asr", lambda *args: (asr_calls.append("load") or model, object()))
    monkeypatch.setattr(assistant, "transcribe", lambda *args: {"transcript": "actual hypothesis"})
    seen = []

    def generate(model, processor, row, root, options):
        seen.append(("base" if model.disabled else model.active_adapter, row["user"]))
        return {"id": row["id"], "task": row["task"], "prediction": row["user"], "score": None}

    monkeypatch.setattr(assistant, "generate", generate)
    result = assistant.run_evaluate(options)
    assert list(result["variants"]) == ["base", "adapter-step-000002", "adapter-step-000004"]
    assert result["status"] == "completed" and asr_calls == ["load"]
    assert [variant for variant, _ in seen] == ["base"] * 3 + ["default"] * 3 + ["adapter-step-000004"] * 3
    assert [utterance for _, utterance in seen] == ["q", "actual hypothesis", "ref"] * 3
    assert model.active_adapter == "default"


def test_real_client_transmits_bounded_settings_instead_of_old_hardcoded_lr(tmp_path):
    helper = modal_helpers()
    captured = []
    manifest_text = json.dumps({"schema_version": 1, "rows": []})
    helper.update(
        ROOT=tmp_path,
        PHASE="execute",
        committed_text=lambda path, revision, prefix: manifest_text,
        gpu_remote=SimpleNamespace(remote=lambda *args: captured.append(args[-1]) or {"status": "completed"}),
        backup_remote=SimpleNamespace(remote=lambda *args: {"files": []}),
        download_review=lambda *args: None,
        snapshot=lambda: {},
        finish_remote=SimpleNamespace(remote=lambda *args: {"reserved_total_usd": "26.60"}),
    )
    helper["main"](
        "train",
        "run-1",
        "a" * 40,
        batch_id="v4",
        manifest="docs/natural-assistant/v4/manifest.json",
        steps=4,
        max_seconds=3300,
        learning_rate=1e-4,
        asr_variant="turbo",
        checkpoint_steps="2,4",
    )
    config = captured[0]
    assert config["learning_rate"] == 1e-4 and config["asr_model"] == assistant.ASR_VARIANTS["turbo"][0]
    assert config["checkpoint_steps"] == [2, 4]
    assert helper["verify_runtime_contract"]("train", config) == config["runtime_options_sha256"]
    with pytest.raises(ValueError, match="Actual runtime arguments"):
        helper["verify_runtime_contract"]("train", config | {"max_pixels": 1048576})


def test_real_runner_passes_lr_asr_and_archive_updates_to_cli(tmp_path):
    helper = modal_helpers()
    app_root = tmp_path / "app"
    manifest_relative = Path("docs/natural-assistant/v4/manifest.json")
    file = app_root / manifest_relative
    file.parent.mkdir(parents=True)
    file.write_text('{"schema_version":1,"rows":[]}\n')
    manifest_sha = helper["sha256"](file)
    natural_root = tmp_path / "course"
    data = natural_root / "data" / manifest_sha
    data.mkdir(parents=True)
    (data / "archive-receipt.json").write_text("{}")
    config = settings(helper, asr_variant="turbo")
    options = config | {
        "runtime_options": config,
        "runtime_options_sha256": hashlib.sha256(helper["runtime_contract"](config).encode()).hexdigest(),
        "adapter_run_id": "",
        "selection": None,
        "selection_sha256": None,
        "external_metadata_sha256": None,
    }
    captured = []

    def run(args, **kwargs):
        captured.append(args)
        output = Path(args[args.index("--output") + 1])
        helper["write_json"](output / "result.json", {"status": "completed"})

    helper.update(
        Path=lambda *args: app_root if args == ("/app",) else Path(*args),
        manifest_path=manifest_relative,
        NATURAL_ROOT=natural_root,
        volume=SimpleNamespace(reload=lambda: None, commit=lambda: None),
        require_reservation=lambda *args: None,
        subprocess=SimpleNamespace(run=run, STDOUT=-2),
    )
    helper["execute_stage"]("train", "v4", "run-1", "a" * 40, manifest_sha, options)
    args = captured[0]
    assert args[args.index("--learning-rate") + 1] == "0.0001"
    assert args[args.index("--checkpoint-steps") + 1] == "2,4"
    assert args[args.index("--asr-model") + 1] == assistant.ASR_VARIANTS["turbo"][0]
    assert args[args.index("--asr-revision") + 1] == assistant.ASR_VARIANTS["turbo"][1]
    execution = json.loads((natural_root / "v4/train/run-1/execution.json").read_text())
    assert execution["runtime_options"] == config
    assert execution["secret_injected_into_gpu"] is False


def test_release_maps_an_archived_adapter_without_exporting_resume_state():
    helper = modal_helpers()
    approved = {
        "approved": True,
        "reviewed": True,
        "release_id": "assistant-2b-v4",
        "model_card": "reviewed card",
        "source": {"repo": "owner/private", "revision": "a" * 40, "prefix": "natural-v3/v4/train/run-1"},
        "base_model": {"repo": assistant.MODEL_ID, "revision": assistant.MODEL_REVISION},
        "asr_model": {"repo": assistant.ASR_ID, "revision": assistant.ASR_REVISION},
        "files": [
            {
                "path": "adapter/adapter_model.safetensors",
                "source_path": "checkpoints/step-000002/adapter_model.safetensors",
                "bytes": 10,
                "sha256": "b" * 64,
                "redistribution_approved": True,
                "license": "Apache-2.0",
            }
        ],
    }
    assert helper["validate_release"](approved) is approved
    for bad in (
        "checkpoints/step-000002/training_state.pt",
        "../adapter_model.safetensors",
        "unknown/adapter_model.safetensors",
    ):
        changed = approved | {"files": [approved["files"][0] | {"source_path": bad}]}
        with pytest.raises(ValueError):
            helper["validate_release"](changed)


def test_transcription_only_audio_has_no_chat_and_keeps_source_metrics(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    audio_rows = [
        {
            "id": "read",
            "split": "validation",
            "family": "read",
            "task": "speech_transcription",
            "source": "fleurs",
            "audio": "a.wav",
            "user": "原句",
            "answer": None,
        },
        {
            "id": "chat",
            "split": "validation",
            "family": "chat",
            "task": "speech_chat",
            "source": {"id": "aishell"},
            "audio": "b.wav",
            "user": "你好",
            "answer": None,
        },
    ]
    for name in ("a.wav", "b.wav"):
        (tmp_path / name).write_bytes(b"mock decoder fixture")
    manifest.write_text(json.dumps({"schema_version": 1, "rows": [], "audio_rows": audio_rows}))
    options = SimpleNamespace(
        manifest=manifest,
        data_root=tmp_path,
        output=tmp_path / "output",
        adapter=None,
        model=assistant.MODEL_ID,
        model_revision=assistant.MODEL_REVISION,
        asr_model=assistant.ASR_ID,
        asr_revision=assistant.ASR_REVISION,
        device="cpu",
        dtype="float32",
        min_pixels=65536,
        max_pixels=524288,
        max_tokens=2048,
        max_seconds=100,
        seed=42,
        split="validation",
    )
    model = torch.nn.Linear(1, 1)
    monkeypatch.setattr(assistant, "load_core", lambda *args, **kwargs: (model, object()))
    monkeypatch.setattr(assistant, "load_asr", lambda *args: (model, object()))
    monkeypatch.setattr(
        assistant,
        "transcribe",
        lambda model, processor, path: {"transcript": "原句" if path.name == "a.wav" else "你好"},
    )
    users = []

    def generate(model, processor, row, root, options):
        users.append(row["user"])
        return {"id": row["id"], "task": row["task"], "prediction": "reply", "score": None}

    monkeypatch.setattr(assistant, "generate", generate)
    result = assistant.run_evaluate(options)
    assert users == ["你好", "你好"]
    assert result["status"] == "completed"
    assert result["requested_audio_rows"] == 2 and result["requested_audio_chat_rows"] == 1
    assert result["variants"]["base"]["generation_count"] == 2
    assert result["asr"]["by_task"]["speech_transcription"]["count"] == 1
    assert result["asr"]["by_source"]["fleurs"]["raw_micro_cer"] == 0
    assert result["asr"]["by_source"]["aishell"]["count"] == 1
    transcripts = json.loads((options.output / "transcripts.json").read_text())
    assert transcripts[0]["reference_transcript"] == "原句" and transcripts[0]["source"] == "fleurs"
    assert transcripts[0]["task"] == "speech_transcription"
    manifest.write_text(json.dumps({"schema_version": 1, "audio_rows": [audio_rows[0] | {"task": "unknown"}]}))
    with pytest.raises(ValueError, match="Audio task"):
        assistant.load_manifest(manifest)


def test_incomplete_transcription_only_rows_do_not_report_complete(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    (tmp_path / "a.wav").write_bytes(b"mock decoder fixture")
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "audio_rows": [
                    {
                        "id": "read",
                        "split": "validation",
                        "family": "read",
                        "task": "speech_transcription",
                        "audio": "a.wav",
                        "user": "原句",
                        "answer": None,
                    }
                ],
            }
        )
    )
    options = SimpleNamespace(
        manifest=manifest,
        data_root=tmp_path,
        output=tmp_path / "output",
        adapter=None,
        model=assistant.MODEL_ID,
        model_revision=assistant.MODEL_REVISION,
        asr_model=assistant.ASR_ID,
        asr_revision=assistant.ASR_REVISION,
        device="cpu",
        dtype="float32",
        min_pixels=65536,
        max_pixels=524288,
        max_tokens=2048,
        max_seconds=1,
        seed=42,
        split="validation",
    )
    model = torch.nn.Linear(1, 1)
    monkeypatch.setattr(assistant, "load_core", lambda *args, **kwargs: (model, object()))
    monkeypatch.setattr(assistant, "load_asr", lambda *args: (model, object()))
    clock = iter([0, 2, 2])
    monkeypatch.setattr(assistant.time, "monotonic", lambda: next(clock))
    result = assistant.run_evaluate(options)
    assert result["requested_audio_chat_rows"] == 0
    assert result["variants"]["base"]["completed"] is True
    assert result["asr"]["completed"] is False and result["status"] == "time_limit_partial"
