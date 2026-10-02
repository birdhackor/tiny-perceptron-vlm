"""縮步與到時停止必須保留真實更新次數，不能標成完整排程。"""

import copy
from types import SimpleNamespace

import pytest
import torch

from scripts.course_experiments import architecture
from scripts.course_experiments.common import Context, new_lm, write_json
from tiny_perceptron.training import load_checkpoint, save_checkpoint


@pytest.fixture
def setup(tmp_path):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    ctx = Context("cpu", tmp_path / "result", tmp_path / "dependencies", tmp_path / "assets")
    ctx.output.mkdir()
    records = [
        {
            "family": str(index),
            "text": f"A bird saw {index} cats.",
            "messages": [
                {"role": "user", "content": f"color {index}?"},
                {"role": "assistant", "content": "red" if index % 2 else "blue"},
            ],
        }
        for index in range(4)
    ]
    data = {"train": records[:2], "validation": records[2:3], "test": records[3:]}
    model = new_lm(ctx, width=8, layers=1, max_length=32)
    dependency = ctx.dependencies / "sft"
    write_json(dependency / "dataset.json", data)
    save_checkpoint(dependency / "model.pt", model)
    yield ctx, model, data
    torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("stop_after_first", [False, True])
def test_expired_deadline_preserves_actual_attempts_and_checkpoint(setup, monkeypatch, stop_after_first):
    ctx, model, data = setup
    before = copy.deepcopy(model.state_dict())
    now, calls = [0.0], [0]
    monkeypatch.setattr(architecture, "time", SimpleNamespace(perf_counter=lambda: now[0]))

    def sync(_device):
        calls[0] += 1
        # 計時開始、第一步開始、第一步完成；在完成一次真實更新後到時。
        if stop_after_first and calls[0] == 3:
            now[0] = 2.0

    monkeypatch.setattr(architecture, "_sync", sync)
    result = architecture._train(
        model,
        data["train"],
        ctx,
        name="stopped",
        steps=3,
        batch_size=2,
        deadline=1.0 if stop_after_first else 0.0,
    )
    expected = int(stop_after_first)
    assert result["requested_steps"] == 3
    assert result["steps"] == result["optimizer_updates"] == expected
    assert result["skipped_updates"] == 0
    assert result["budget_exhausted"] is True
    assert result["all_requested_attempts_completed"] is False
    assert result["status"] == "budget_exhausted"
    changed = any(not torch.equal(before[name], value) for name, value in model.state_dict().items())
    assert changed is stop_after_first
    restored, payload = load_checkpoint(ctx.output / "stopped.pt", "cpu")
    assert payload["step"] == payload["metadata"]["completed_attempts"] == expected
    assert payload["metadata"]["requested_steps"] == 3
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in model.state_dict().items())


def test_completed_attempts_remain_separate_from_successful_updates(setup):
    ctx, model, data = setup
    result = architecture._train(model, data["train"], ctx, name="complete", steps=2, batch_size=2)
    assert result["status"] == "completed"
    assert result["requested_steps"] == result["steps"] == 2
    assert result["all_requested_attempts_completed"] is True
    assert result["budget_exhausted"] is False
    assert result["optimizer_updates"] + result["skipped_updates"] == result["steps"]


def test_packing_deadline_cannot_hide_incomplete_branches(setup):
    ctx, model, data = setup
    result = architecture._packing_updates(model, data, ctx, steps=2, deadline=0.0)
    for name in ("padded", "packed"):
        branch = result[name]
        assert branch["requested_steps"] == 2
        assert branch["steps"] == branch["optimizer_updates"] == 0
        assert branch["budget_exhausted"] is True
        assert branch["status"] == "budget_exhausted"
        assert branch["all_requested_attempts_completed"] is False
        _, payload = load_checkpoint(ctx.output / f"{name}.pt", "cpu")
        assert payload["step"] == 0
        assert payload["metadata"]["requested_steps"] == 2


@pytest.mark.parametrize(
    ("name", "expected"),
    [("modern", [2] * 6), ("moe", [2] * 7), ("efficiency", [1] * 6), ("precision", [2] * 2)],
)
def test_public_entries_honor_step_scale_and_return_partial_status(setup, monkeypatch, name, expected):
    ctx, _model, data = setup
    ctx.step_scale = 0.01
    requested, packing_steps = [], []
    original_train, original_packing = architecture._train, architecture._packing_updates

    def train(*args, **kwargs):
        requested.append(kwargs["steps"])
        kwargs["deadline"] = 0.0
        return original_train(*args, **kwargs)

    def packing(*args, **kwargs):
        packing_steps.append(kwargs["steps"])
        kwargs["deadline"] = 0.0
        return original_packing(*args, **kwargs)

    def small_model(ctx, **kwargs):
        return new_lm(ctx, width=8, layers=1, max_length=32, **kwargs)

    # 只檢查正式入口的縮步與partial傳遞；不在此重跑架構／後端／編譯實驗。
    monkeypatch.setattr(architecture, "new_lm", small_model)
    monkeypatch.setattr(architecture, "_text_dataset", lambda _ctx: data)
    monkeypatch.setattr(architecture, "_matched_width", lambda config, _target: (config.width, 0))
    monkeypatch.setattr(architecture, "_train", train)
    monkeypatch.setattr(architecture, "_packing_updates", packing)
    monkeypatch.setattr(architecture, "_heldout", lambda *_args, **_kwargs: {})
    for function in (
        "_router_gradients",
        "_routing",
        "_cache_probe",
        "_sdpa_probe",
        "_packing_probe",
        "_accumulation_probe",
        "_checkpoint_probe",
        "_benchmark",
    ):
        monkeypatch.setattr(architecture, function, lambda *_args, **_kwargs: {})
    monkeypatch.setattr(architecture, "_compile_probe", lambda *_args, **_kwargs: {"status": "not_run"})
    result = getattr(architecture, f"run_{name}")(ctx)
    assert requested == expected
    assert result["step_scale"] == 0.01
    if name == "precision":
        for branch in ("fp32", "bf16"):
            assert result["variants"][branch]["status"] == "budget_exhausted"
            assert result["variants"][branch]["training"]["steps"] == 0
        assert result["variants"]["fp16"]["status"] == "not_run"
    elif name == "efficiency":
        assert packing_steps == [1]
        for branch in ("padded", "packed"):
            report = result["packing"]["actual_updates"][branch]
            assert report["requested_steps"] == 1
            assert report["budget_exhausted"] is True
    else:
        assert result["all_requested_updates_completed"] is False
