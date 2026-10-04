"""One selected test model versus legacy paired validation, without GPU work."""

import ast
import contextlib
import copy
import hashlib
import json
import re
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import natural_assistant as cli
from tiny_perceptron import natural_assistant as assistant


@pytest.fixture
def evaluation(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "rows": [
                    {"id": "text", "split": "test", "family": "text", "task": "ocr", "user": "text q", "answer": "a"}
                ],
                "audio_rows": [
                    {
                        "id": "chat",
                        "split": "test",
                        "family": "chat",
                        "task": "speech_chat",
                        "audio": "chat.wav",
                        "user": "reference q",
                        "answer": None,
                    },
                    {
                        "id": "read",
                        "split": "test",
                        "family": "read",
                        "task": "speech_transcription",
                        "audio": "read.wav",
                        "user": "reference text",
                        "answer": None,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    for name in ("chat.wav", "read.wav"):
        (tmp_path / name).write_bytes(b"fixture: decoder is mocked")
    options = SimpleNamespace(
        manifest=manifest,
        data_root=tmp_path,
        output=tmp_path / "result",
        max_seconds=100,
        model=assistant.MODEL_ID,
        model_revision=assistant.MODEL_REVISION,
        asr_model=assistant.ASR_ID,
        asr_revision=assistant.ASR_REVISION,
        device="cpu",
        dtype="float32",
        min_pixels=65536,
        max_pixels=524288,
        max_tokens=2048,
        seed=42,
        split="test",
        adapter=None,
        adapter_label="adapter-step-001039",
        comparison_adapters=[],
        selected_only=True,
    )
    calls = {"load_core": [], "load_asr": [], "generate": [], "disable_adapter": []}

    class Model(torch.nn.Linear):
        active = "base"

        @contextlib.contextmanager
        def disable_adapter(self):
            previous = self.active
            calls["disable_adapter"].append("enter")
            self.active = "base"
            try:
                yield
            finally:
                self.active = previous

    model = Model(1, 1)

    def load_core(options, *, adapter):
        calls["load_core"].append(adapter)
        model.active = "adapter" if adapter else "base"
        return model, object()

    def generate(model, processor, row, root, options):
        calls["generate"].append((model.active, row["id"], row["task"], row["user"]))
        return {"id": row["id"], "task": row["task"], "prediction": model.active, "score": None}

    monkeypatch.setattr(assistant, "load_core", load_core)
    monkeypatch.setattr(assistant, "load_asr", lambda options: (calls["load_asr"].append("load") or model, object()))
    monkeypatch.setattr(assistant, "transcribe", lambda *args: {"transcript": "actual ASR"})
    monkeypatch.setattr(assistant, "generate", generate)
    return options, calls


@pytest.mark.parametrize("adapter", [False, True])
def test_selected_only_runs_one_exact_model_and_one_shared_asr(evaluation, adapter):
    options, calls = evaluation
    if adapter:
        options.adapter = options.data_root / "selected-archive"
    result = assistant.run_evaluate(options)
    variant = options.adapter_label if adapter else "base"
    active = "adapter" if adapter else "base"
    assert result["status"] == "completed" and result["selected_only"] is True
    assert list(result["variants"]) == [variant]
    assert result["variants"][variant]["generation_count"] == 3
    assert result["requested_audio_rows"] == 2 and result["requested_audio_chat_rows"] == 1
    assert result["asr"]["count"] == 2 and calls["load_asr"] == ["load"]
    assert calls["load_core"] == [options.adapter] and calls["disable_adapter"] == []
    assert calls["generate"] == [
        (active, "text", "ocr", "text q"),
        (active, "chat", "speech_chat", "actual ASR"),
        (active, "chat", "typed_chat", "reference q"),
    ]
    raw = json.loads((options.output / "generations.json").read_text())
    assert len(raw) == 3 and {row["variant"] for row in raw} == {variant}
    assert {path.name for path in options.output.glob("generations-*.json")} == {f"generations-{variant}.json"}


def test_legacy_default_still_compares_base_and_adapter(evaluation):
    options, calls = evaluation
    del options.selected_only  # Existing API callers never had this option.
    options.adapter = options.data_root / "adapter"
    result = assistant.run_evaluate(options)
    assert result["selected_only"] is False and list(result["variants"]) == ["base", options.adapter_label]
    assert [call[0] for call in calls["generate"]] == ["base"] * 3 + ["adapter"] * 3
    assert calls["disable_adapter"] == ["enter"] and calls["load_asr"] == ["load"]
    assert all(row["completed"] and row["generation_count"] == 3 for row in result["variants"].values())


def test_selected_only_comparisons_rejected_before_manifest_or_model_load(evaluation, monkeypatch):
    options, _ = evaluation
    options.comparison_adapters = [(options.adapter_label, options.data_root / "archive")]
    monkeypatch.setattr(
        assistant, "load_manifest", lambda *args: pytest.fail("Invalid combination must fail before loading")
    )
    monkeypatch.setattr(assistant, "load_core", lambda *args, **kw: pytest.fail("No model may be loaded"))
    with pytest.raises(ValueError, match="cannot include comparison"):
        assistant.run_evaluate(options)


def test_cli_opt_in_is_false_by_default_and_passes_selected_choice(tmp_path, monkeypatch):
    defaults = cli.parser().parse_args(["evaluate", "--output", str(tmp_path)])
    assert defaults.selected_only is False
    seen = []
    monkeypatch.setattr(
        cli.assistant, "run_evaluate", lambda options, **kw: seen.append(options) or {"status": "fixture"}
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "natural_assistant.py",
            "evaluate",
            "--output",
            str(tmp_path),
            "--manifest",
            "fixture.json",
            "--split",
            "test",
            "--selected-only",
            "--adapter",
            "selected-archive",
        ],
    )
    cli.main()
    assert len(seen) == 1 and seen[0].selected_only is True
    assert seen[0].adapter == Path("selected-archive") and seen[0].split == "test"


@pytest.mark.parametrize("stage,comparison", [("validation", False), ("validation", True), ("train", False)])
def test_cli_keeps_selected_only_out_of_validation_and_training(tmp_path, monkeypatch, stage, comparison):
    args = ["natural_assistant.py", stage, "--output", str(tmp_path), "--manifest", "fixture.json", "--selected-only"]
    if comparison:
        args.extend(["--comparison-adapter", "adapter-step-001039=archive"])
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(ValueError, match="Selected-only|--selected-only"):
        cli.main()


@pytest.mark.parametrize(
    "stage,adapter", [("evaluate", False), ("evaluate", True), ("validation", True), ("baseline", False)]
)
def test_actual_modal_runner_selects_only_for_final_evaluate(tmp_path, stage, adapter):
    source = Path(__file__).resolve().parents[1] / "scripts/modal_natural.py"
    names = {"execute_stage", "sha256", "write_json", "safe_name"}
    nodes = []
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node = copy.deepcopy(node)
            node.decorator_list = []
            nodes.append(node)
    app_root = tmp_path / "app"
    manifest_relative = Path("docs/natural-assistant/v4/manifest.json")
    file = app_root / manifest_relative
    file.parent.mkdir(parents=True)
    file.write_text('{"schema_version":1,"rows":[]}\n')
    manifest_sha = hashlib.sha256(file.read_bytes()).hexdigest()
    natural_root = tmp_path / "course"
    data = natural_root / "data" / manifest_sha
    data.mkdir(parents=True)
    (data / "archive-receipt.json").write_text("{}")
    args_seen = []
    namespace = {
        "Path": lambda *args: app_root if args == ("/app",) else Path(*args),
        "manifest_path": manifest_relative,
        "NATURAL_ROOT": natural_root,
        "hashlib": hashlib,
        "json": json,
        "re": re,
        "threading": threading,
        "time": time,
        "SPEC": {stage: {}},
        "SELECTION_STAGES": ("evaluate",),
        "EXTERNAL_STAGES": (),
        "volume": SimpleNamespace(reload=lambda: None, commit=lambda: None),
        "verify_runtime_contract": lambda *args: "b" * 64,
        "require_reservation": lambda *args: None,
        "selection_gate": lambda *args: {"selected_variant": "adapter-step-001039" if adapter else "base"},
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)

    def descriptor(batch_id, run_id, checkpoint, manifest_sha):
        return {
            "path": str(tmp_path / checkpoint),
            "adapter_sha256": "d" * 64,
            "variant": "adapter-" + checkpoint,
            "adapter_run_id": run_id,
        }

    def run(args, **kwargs):
        args_seen.append(args)
        output = Path(args[args.index("--output") + 1])
        namespace["write_json"](output / "result.json", {"status": "completed"})

    namespace.update(adapter_descriptor=descriptor, subprocess=SimpleNamespace(run=run, STDOUT=-2))
    options = {
        "max_seconds": 100,
        "max_pixels": 524288,
        "seed": 42,
        "asr_model": assistant.ASR_ID,
        "asr_revision": assistant.ASR_REVISION,
        "adapter_run_id": "train-1" if adapter else "",
        "adapter_checkpoint": "step-001039" if adapter else "",
        "adapter_checkpoints": ["step-001039", "step-002077"] if stage == "validation" else [],
        "runtime_options": {},
        "runtime_options_sha256": "b" * 64,
        "selection": {},
        "selection_sha256": "c" * 64,
        "external_metadata_sha256": None,
    }
    namespace["execute_stage"](stage, "v4", "run-1", "a" * 40, manifest_sha, options)
    assert len(args_seen) == 1
    args = args_seen[0]
    assert ("--selected-only" in args) == (stage == "evaluate")
    assert args[args.index("--split") + 1] == ("test" if stage == "evaluate" else "validation")
    assert ("--adapter" in args) == adapter
    if stage == "validation":
        assert args.count("--comparison-adapter") == 2
    else:
        assert "--comparison-adapter" not in args
