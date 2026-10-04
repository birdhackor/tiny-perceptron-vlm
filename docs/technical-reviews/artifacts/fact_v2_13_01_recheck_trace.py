"""Trace current run_dpo routing with training/evaluation leaves replaced, without training."""

import hashlib
import inspect
import json
import platform
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from scripts.course_experiments import behavior
from scripts.course_experiments.common import split_records
from scripts.course_experiments.text import arithmetic_records

OUT = Path(__file__).parent
ROOT = OUT.parents[2]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def run_trace(score):
    arithmetic = split_records(arithmetic_records(), seed=42)
    pairs = behavior._preference_parts(arithmetic)
    split_by_prompts = {tuple(row["prompt"] for row in rows): split for split, rows in pairs.items()}
    events = []
    defaults = inspect.signature(behavior._dpo_train).parameters
    base = torch.nn.Linear(2, 2)
    base_state = {key: tensor.clone() for key, tensor in base.state_dict().items()}

    def evaluate(model, reference, rows):
        split = split_by_prompts[tuple(row["prompt"] for row in rows)]
        assert split in {"validation", "test"}
        assert all(not parameter.requires_grad for parameter in reference.parameters())
        events.append({"event": "preference_evaluate", "split": split, "rows": len(rows), "score": score})
        return {"records": len(rows), "score": score}

    def train(model, reference, rows, ctx, name, beta=defaults["beta"].default, count=defaults["count"].default):
        assert tuple(row["prompt"] for row in rows) == tuple(row["prompt"] for row in pairs["train"])
        assert all(not parameter.requires_grad for parameter in reference.parameters())
        events.append({"event": "train_call", "name": name, "rows": len(rows), "beta": beta, "requested_steps": count})
        return {"steps": count, "score_unread_by_trainer": score, "execution": "training leaf replaced; no update performed"}

    def arithmetic_evaluation(model, rows):
        assert rows is arithmetic or rows == arithmetic
        events.append({"event": "arithmetic_evaluation", "splits": ["validation", "test"]})
        return {"validation": {"score": score}, "test": {"score": score}}

    def save_splits(ctx, rows, name="data"):
        return {split: {"records": len(records)} for split, records in rows.items()}

    ctx = SimpleNamespace(seed=42, device="cpu", dependency=lambda experiment, filename: Path("unused-controlflow-only.pt"))
    with (
        patch.object(behavior, "load_lm", return_value=base),
        patch.object(behavior, "_preference_evaluate", side_effect=evaluate),
        patch.object(behavior, "_dpo_train", side_effect=train),
        patch.object(behavior, "_evaluations", side_effect=arithmetic_evaluation),
        patch.object(behavior, "_save_splits", side_effect=save_splits),
        patch.object(behavior, "_natural_dpo_pilot", return_value={"not_run": "unrelated external-data leaf"}),
    ):
        result = behavior.run_dpo(ctx)
    assert result["reference_unchanged"]
    assert all(torch.equal(base_state[key], tensor) for key, tensor in base.state_dict().items())
    assert [event["event"] for event in events] == [
        "preference_evaluate", "preference_evaluate",
        "train_call", "preference_evaluate", "preference_evaluate", "arithmetic_evaluation",
        "train_call", "preference_evaluate", "preference_evaluate", "arithmetic_evaluation",
        "train_call", "preference_evaluate", "preference_evaluate", "arithmetic_evaluation",
    ]
    calls = [event for event in events if event["event"] == "train_call"]
    assert [(call["name"], call["rows"], call["beta"], call["requested_steps"]) for call in calls] == [
        ("model", 49, 0.1, 250), ("beta1", 49, 1.0, 250), ("format-model", 49, 0.1, 200),
    ]
    assert events[0]["split"] == "validation" and events[1]["split"] == "test"
    return {"score_sentinel": score, "events": events, "train_calls": calls}


def main():
    traces = [run_trace(-1_000_000.0), run_trace(1_000_000.0)]
    assert traces[0]["train_calls"] == traces[1]["train_calls"]
    report = {
        "command": "PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/fact_v2_13_01_recheck_trace.py",
        "result": "Actual current run_dpo control flow traced twice. Train calls and fixed settings are identical for opposite evaluation-score sentinels; all calls use the 49 training rows. Baseline validation/test evaluation precedes branch training, with validation/test evaluation after each fixed branch.",
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "scope": "control flow with train/eval leaves substituted; no training or quality reproduction"},
        "source_sha256": sha((ROOT / "scripts/course_experiments/behavior.py").read_bytes()),
        "inspected_functions": {
            name: inspect.getsource(getattr(behavior, name))
            for name in ["run_dpo", "_dpo_train", "_preference_evaluate"]
        },
        "traces": traces,
        "limitations": "Only current routing and absence of programmatic feedback into fixed settings are demonstrated. No claim about human choice history, withheld-until-end testing, GPU execution, or learned quality. The real training and model evaluations were not executed by this trace.",
    }
    target = OUT / "fact_v2_13_01_recheck_trace.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"result": report["result"], "command": report["command"], "train_calls": traces[0]["train_calls"], "output": str(target)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
