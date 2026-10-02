"""離線核對參照的因果性／梯度、量測完整度及supporting入口；不證明CUDA Flash。"""

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from scripts.course_experiments.common import Context

PROJECT = Path(__file__).resolve().parents[1]


def load_candidate(name, relative):
    # 準備階段從ignored副本載入；套用後同一測試從repo實際檔案載入。
    spec = importlib.util.spec_from_file_location(name, PROJECT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def architecture():
    return load_candidate("flash_architecture_candidate", "scripts/course_experiments/architecture.py")


@pytest.fixture
def runner():
    return load_candidate("flash_runner_candidate", "scripts/course_experiments/run.py")


def inputs(dtype=torch.float32):
    generator = torch.Generator().manual_seed(42)
    return tuple(torch.randn(1, 2, 5, 4, generator=generator).to(dtype).requires_grad_() for _ in range(3))


@pytest.mark.parametrize("dtype,atol", [(torch.float32, 2e-6), (torch.float16, 3e-3), (torch.bfloat16, 3e-2)])
def test_manual_same_dtype_reference_has_correct_outputs_and_gradients(architecture, dtype, atol):
    qkv = inputs(dtype)
    manual = architecture._flash_manual_attention(*qkv)
    with sdpa_kernel(backends=[SDPBackend.MATH]):
        math = F.scaled_dot_product_attention(*qkv, attn_mask=None, dropout_p=0.0, is_causal=True)
    upstream = torch.linspace(-0.25, 0.25, manual.numel()).reshape_as(manual).to(dtype)
    manual_gradients = torch.autograd.grad(manual, qkv, upstream)
    math_gradients = torch.autograd.grad(math, qkv, upstream)
    assert manual.dtype == dtype
    assert torch.allclose(manual, math, atol=atol, rtol=atol)
    for actual, expected in zip(manual_gradients, math_gradients, strict=True):
        assert actual.dtype == dtype and bool(torch.isfinite(actual).all())
        assert torch.allclose(actual, expected, atol=atol, rtol=atol)


def test_manual_reference_masks_future_values_and_future_gradients(architecture):
    qkv = inputs()
    output = architecture._flash_manual_attention(*qkv)
    changed_v = qkv[2].detach().clone()
    changed_v[:, :, -1] += 100
    changed = architecture._flash_manual_attention(qkv[0], qkv[1], changed_v)
    assert torch.equal(output[:, :, :-1], changed[:, :, :-1])
    assert torch.equal(output[:, :, 0], qkv[2][:, :, 0])
    q_gradient, k_gradient, v_gradient = torch.autograd.grad(output[:, :, 0].sum(), qkv)
    assert torch.count_nonzero(q_gradient) == torch.count_nonzero(k_gradient) == 0
    assert torch.count_nonzero(v_gradient[:, :, 1:]) == 0
    with pytest.raises(ValueError, match="同形狀"):
        architecture._flash_manual_attention(qkv[0], qkv[1][:, :, :-1], qkv[2])


@pytest.mark.parametrize("backward", [False, True])
def test_measurement_completes_warmup_and_all_calls_without_leaf_grad_accumulation(architecture, backward):
    qkv = inputs()
    calls = []

    def reference(*values):
        calls.append(torch.is_grad_enabled())
        return architecture._flash_manual_attention(*values)

    report = architecture._flash_measure(reference, qkv, torch.ones_like(qkv[0]), "cpu", backward)
    assert report["status"] == "completed" and not report["budget_exhausted"]
    assert report["warmup_calls"] == 3 and report["measured_calls"] == len(report["samples_seconds"]) == 9
    assert calls == [backward] * 12
    assert all(tensor.grad is None for tensor in qkv)
    assert report["allocated_before_bytes"] is report["peak_allocated_bytes"] is None
    assert report["median_seconds"] > 0 and not report["synchronized"]


def test_deadline_cannot_report_completed_measurement(architecture, runner):
    qkv = inputs()
    calls = []
    report = architecture._flash_measure(
        lambda *values: calls.append(values), qkv, torch.ones_like(qkv[0]), "cpu", False, deadline=0
    )
    assert calls == []
    assert report["status"] == "budget_exhausted" and report["budget_exhausted"]
    assert report["requested_warmup_calls"] == 3 and report["requested_calls"] == 9
    assert report["warmup_calls"] == report["measured_calls"] == 0
    assert report["median_seconds"] is None
    assert runner.unfinished_schedules({"route": report})[0]["path"] == "results.route"


def test_cpu_sdpa_profiler_is_never_labeled_cuda_flash(architecture):
    qkv = inputs()
    report = architecture._flash_profile(architecture._flash_sdpa_attention, qkv, torch.ones_like(qkv[0]), "cpu", True)
    assert "aten::scaled_dot_product_attention" in report["operator_names"]
    assert report["includes_backward"]
    assert not report["flash_cuda_verified"] and report["cuda_kernel_names"] == []


def test_nonfinite_comparison_is_reportable_and_fails_tolerance(architecture):
    report = architecture._flash_error(torch.tensor([float("nan")]), torch.tensor([1.0]), 0.01, 0.01)
    assert not report["finite"] and not report["within_declared_tolerance"]
    assert report["max_absolute_error"] is None
    json.dumps(report, allow_nan=False)


def test_cpu_entry_creates_reproducible_input_fixture_without_model_or_fake_flash(architecture, tmp_path, monkeypatch):
    monkeypatch.setattr(architecture, "FLASH_PROBE_SHAPE", (1, 2, 5, 4))
    fixtures = []
    for name in ("first", "second"):
        output = tmp_path / name
        output.mkdir()
        ctx = Context("cpu", output, tmp_path, tmp_path, seed=42)
        report = architecture.run_flash_probe(ctx)
        assert report["status"] == "not_run" and report["supported_routes"] == []
        assert report["experiment_kind"] == "mechanism_probe"
        assert report["fixture"]["kind"] == "input_fixture" and not report["fixture"]["contains_model_weights"]
        assert list(output.iterdir()) == [output / "fixture.pt"]
        saved = torch.load(output / "fixture.pt", weights_only=True)
        assert saved["format"] == "course-flash-probe-input-v1"
        assert set(saved["tensors"]) == {"q", "k", "v", "upstream"}
        assert "model" not in saved and "optimizer" not in saved
        fixtures.append(saved["tensors"])
    assert all(torch.equal(fixtures[0][key], fixtures[1][key]) for key in fixtures[0])


def test_supporting_id_preserves_formal_thirty_and_rejects_ambiguous_lookup(runner, tmp_path, monkeypatch):
    plan = json.loads(runner.PLAN_PATH.read_text())
    assert len(plan["sequence"]) == 30
    assert "flash_probe" not in {entry["id"] for entry in plan["sequence"]}
    assert [entry["id"] for entry in plan["supporting_experiments"]] == ["flash_probe"]
    assert runner.experiment_spec("flash_probe")["kind"] == "mechanism_probe"
    with pytest.raises(ValueError, match="Unknown"):
        runner.experiment_spec("unknown")
    duplicate = tmp_path / "plan.json"
    plan["sequence"].append(plan["supporting_experiments"][0])
    duplicate.write_text(json.dumps(plan))
    monkeypatch.setattr(runner, "PLAN_PATH", duplicate)
    with pytest.raises(ValueError, match="Unknown"):
        runner.experiment_spec("flash_probe")


def test_list_assets_supporting_cli_needs_neither_torch_nor_training_archives(tmp_path):
    relative = "scripts/course_experiments/run.py"
    candidate = tmp_path / relative
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes((PROJECT / relative).read_bytes())
    plan = tmp_path / "docs/course-experiments/plan.json"
    plan.parent.mkdir(parents=True)
    plan.write_bytes((PROJECT / "docs/course-experiments/plan.json").read_bytes())
    manifest = tmp_path / "assets/training/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"assets": []}')
    program = (
        "import runpy,sys; sys.modules['torch']=None; "
        "sys.argv=[sys.argv[1],'--list-assets','flash_probe']; runpy.run_path(sys.argv[0],run_name='__main__')"
    )
    result = subprocess.run([sys.executable, "-c", program, str(candidate)], capture_output=True, text=True, check=True)
    assert result.stdout == "" and result.stderr == ""


def test_runner_cpu_supporting_is_smoke_without_changing_formal_cpu_status(runner, architecture, tmp_path, monkeypatch):
    monkeypatch.setattr(architecture, "FLASH_PROBE_SHAPE", (1, 2, 5, 4))
    original_import = runner.importlib.import_module
    monkeypatch.setattr(
        runner.importlib,
        "import_module",
        lambda name: architecture if name == "scripts.course_experiments.architecture" else original_import(name),
    )
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "manifest.json").write_text('{"assets": []}')
    report = runner.execute("flash_probe", "cpu", tmp_path / "probe", tmp_path, assets, revision="a" * 40)
    assert report["evidence_status"] == "interface_smoke_only" and report["unfinished_schedules"] == []
    assert report["experiment_kind"] == "mechanism_probe" and "no model training" in report["timing_scope"]
    assert [entry["path"] for entry in report["artifacts"]] == ["fixture.pt"]
    text = original_import("scripts.course_experiments.text")
    monkeypatch.setattr(text, "run_simple_models", lambda _ctx: {"steps": 1, "requested_steps": 1})
    formal = runner.execute("simple_models", "cpu", tmp_path / "formal", tmp_path, assets, revision="a" * 40)
    assert formal["evidence_status"] == "complete_run" and "experiment_kind" not in formal


def test_supporting_inventory_is_separate_and_cannot_raise_formal_counts(tmp_path, monkeypatch):
    import scripts.course_experiments.common as common

    monkeypatch.syspath_prepend(str(Path(common.__file__).parents[1]))
    builder = load_candidate("flash_progress_candidate", "scripts/update_course_progress.py")
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    plan = tmp_path / "docs/course-experiments/plan.json"
    plan.parent.mkdir(parents=True)
    plan.write_bytes((PROJECT / "docs/course-experiments/plan.json").read_bytes())
    for name in ("README.md", "first-steps.md", "training.md", "glossary.md"):
        file = tmp_path / "course" / name
        file.parent.mkdir(exist_ok=True)
        file.write_text("")
    results = plan.parent / "results"
    results.mkdir()
    (results / "simple_models.json").write_text('{"evidence_status":"complete_run"}')
    fixture_result = results / "flash_probe.json"
    fixture_result.write_text(
        json.dumps(
            {
                "evidence_status": "complete_run",
                "revision": "b" * 40,
                "results": {"status": "completed", "supported_routes": ["fp16", "bf16"]},
                "hf": {"verified_checkpoints": ["fixture.pt"]},
            }
        )
    )
    # 只執行這份候選builder、空的tmp ROOT；不讀寫正在執行的正式progress/ledger。
    builder.main()
    report = json.loads((plan.parent / "progress.json").read_text())
    assert report["counts"] == {"experiments": 30, "complete_runs": 1, "sections": 0}
    assert len(report["experiments"]) == 30 and len(report["supporting_evidence"]) == 1
    supporting = report["supporting_evidence"][0]
    assert supporting["id"] == "flash_probe" and not supporting["student_model_release"]
    assert supporting["supported_routes"] == ["fp16", "bf16"] and "training_revision" not in supporting
    assert supporting["code_revision"] == "b" * 40
    assert supporting["evidence_sha256"] == hashlib.sha256(fixture_result.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "completed,outcome,expected",
    [
        (False, "failed_verification", "incomplete_run"),
        (True, "unsupported", "complete_run"),
        (True, "completed", "complete_run"),
    ],
)
def test_simulated_cuda_completion_distinguishes_finished_support_decision_from_failed_checks(
    runner, architecture, tmp_path, monkeypatch, completed, outcome, expected
):
    import scripts.course_experiments.common as common

    # CPU-only contract test. Every CUDA method is replaced; no tensors/kernels/hardware checks execute.
    monkeypatch.setattr(common, "seed", lambda _value: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "simulated contract fixture")
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(
        architecture, "run_flash_probe", lambda _ctx: {"schedule_completed": completed, "status": outcome}
    )
    monkeypatch.setattr(runner.importlib, "import_module", lambda _name: architecture)
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "manifest.json").write_text('{"assets": []}')
    report = runner.execute("flash_probe", "cuda", tmp_path / "mock-run", tmp_path, assets, revision="c" * 40)
    assert report["evidence_status"] == expected
    assert report["results"]["status"] == outcome
