"""B.1 independent CPU audit; no full training and no committed binary weights."""

import argparse
import contextlib
import copy
import hashlib
import io
import json
import platform
import random
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts.check_technical_reviews import sections  # noqa: E402
from scripts.course_experiments.applications import (  # noqa: E402
    _parse_json_action,
    _prompt_ids,
    _sft_nll,
    _tool_episode,
    _tool_records,
)
from scripts.course_experiments.common import (  # noqa: E402
    Context,
    new_lm,
    records_sha256,
    split_records,
    text_examples,
    write_json,
)
from tiny_perceptron.data import IGNORE, ByteTokenizer, pad_batch  # noqa: E402
from tiny_perceptron.retrieval import call_tool  # noqa: E402
from tiny_perceptron.training import load_checkpoint  # noqa: E402

PREFIX = ROOT / "docs/technical-reviews/artifacts/fact_finish_b_1"
SECTION_HASH = "21ac28796f17cbc9a4d9f35e59f71d4fcfd2603eb2c6bc4d414b3bd31992d843"
INTRO_HASH = "e60ddfb6588851795e6ddcb307577b49d7816ba604271f252a29735253f3adc0"
WEIGHT_HASH = "c8789df280e4a4cba06994ee94f1e96c7199b30b9806dc005dd9c32d32d4bdbf"
WEIGHT_URL = (
    "https://huggingface.co/birdhackor/tiny-perceptron-course-models/resolve/"
    "973d02736f4ebbfd7188e7f032fff1dfeddbc66c/course/course-v1/tools/model.pt"
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(suffix, value):
    write_json(Path(str(PREFIX) + suffix), value)


def inspect_episode(episode, record):
    tok = ByteTokenizer()
    messages = copy.deepcopy(record["messages"][:2])
    inspected = []
    for event in episode["trace"]:
        generation = event["generation"]
        sample = generation["samples"][0]
        assert generation["input_ids"] == _prompt_ids(messages, tok)
        assert generation["input_tokens"] == len(generation["input_ids"])
        ids = sample["generated_ids"]
        raw = ids[:-1] if ids and ids[-1] == tok.eos_id else ids
        assert tok.decode(raw) == sample["generated"]
        assert sample["invalid_special_tokens"] == [i for i in raw if i < 8]
        assert sample["generated_tokens"] == generation["generated_tokens"] == len(ids)
        assert generation["temperature"] == 0 and generation["candidate_count"] == 1
        assert generation["max_new_tokens"] == 64
        try:
            action = _parse_json_action(sample["generated"])
        except (ValueError, TypeError):
            assert not event["executed"] and "error" in event
            inspected.append({"generated": sample["generated"], "parse": "rejected", "executed": False})
            continue
        assert action == event["parsed"]
        if "done" in action:
            assert not event["executed"] and episode["answer"] == action["answer"]
            inspected.append({"generated": sample["generated"], "parse": "done", "executed": False})
        else:
            result = call_tool(action)
            assert event["executed"] and result == event["tool_result"]
            expected = {"name": record["operation"], "arguments": {"a": record["a"], "b": record["b"]}}
            assert event["request_matches_question"] == (action == expected)
            messages.extend(
                [
                    {"role": "assistant", "content": sample["generated"]},
                    {"role": "user", "content": f"TOOL_RESULT:{result}"},
                ]
            )
            inspected.append({"generated": sample["generated"], "parse": "tool", "executed": True, "result": result})
    assert episode["question"] == record["messages"][1]["content"]
    assert episode["expected"] == record["answer"]
    return {"question": episode["question"], "status": episode["status"], "events": inspected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/course/tools/model.pt")
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    body = dict(sections(ROOT / "course/chapters/0B.md"))["B.1"]
    chapter = (ROOT / "course/chapters/0B.md").read_text()
    intro = chapter[: re.search(r"^## ", chapter, re.M).start()]
    assert hashlib.sha256(body.encode()).hexdigest() == SECTION_HASH
    assert hashlib.sha256(intro.encode()).hexdigest() == INTRO_HASH
    assert body == Path(str(PREFIX) + "_section.md").read_text()
    assert intro == Path(str(PREFIX) + "_intro.md").read_text()
    report_path = ROOT / "docs/course-experiments/results/tools.json"
    pinned_report = Path(str(PREFIX) + "_original_tools.json")
    if not pinned_report.exists():
        pinned_report.write_bytes(report_path.read_bytes())
    assert pinned_report.read_bytes() == report_path.read_bytes()
    report = json.loads(pinned_report.read_text())
    result = report["results"]
    source_paths = [
        "scripts/course_experiments/applications.py",
        "scripts/course_experiments/common.py",
        "scripts/course_experiments/run.py",
        "tiny_perceptron/retrieval.py",
        "tiny_perceptron/data.py",
        "tiny_perceptron/model.py",
        "tiny_perceptron/training.py",
    ]
    code_identity = {path: digest(ROOT / path) for path in source_paths}
    assert all(report["code_sha256"][path] == value for path, value in code_identity.items())
    out = io.StringIO()
    namespace = {}
    snippet = re.findall(r"```python\n(.*?)```", body, re.S)[0]
    with contextlib.redirect_stdout(out):
        exec(compile(snippet, "B.1 original snippet", "exec"), namespace)
    assert json.loads(namespace["text"]) == {"name": "multiply", "arguments": {"a": 123, "b": 45}}
    assert "尚未執行乘法" in out.getvalue()
    exercise = []
    for question, b in [("123乘46", 45), ("123乘46", 46)]:
        request = {"name": "multiply", "arguments": {"a": 123, "b": b}}
        modified = snippet.replace('"123乘45"', json.dumps(question, ensure_ascii=False)).replace(
            '"b": 45', f'"b": {b}'
        )
        exercise_stdout = io.StringIO()
        exercise_namespace = {}
        with contextlib.redirect_stdout(exercise_stdout):
            exec(compile(modified, "B.1 exercise", "exec"), exercise_namespace)
        assert exercise_namespace["training_example"] == {"user": question, "assistant_tool_request": request}
        exercise.append(
            {
                "training_example": exercise_namespace["training_example"],
                "stdout": exercise_stdout.getvalue(),
                "aligned": b == 46,
            }
        )
    arithmetic_demo = {
        "original_request_result": call_tool(namespace["request"]),
        "wrong_parameter_result": call_tool({"name": "multiply", "arguments": {"a": 123, "b": 54}}),
    }
    assert arithmetic_demo == {"original_request_result": 5535, "wrong_parameter_result": 6642}
    records = _tool_records()
    splits = split_records(records, 42)
    assert len(records) == 210 and len({r["family"] for r in records}) == 65
    save("_dataset.json", splits)
    expected_dataset_hash = next(a["sha256"] for a in report["artifacts"] if a["path"] == "dataset.json")
    assert digest(Path(str(PREFIX) + "_dataset.json")) == expected_dataset_hash
    family_sets = {name: {r["family"] for r in rows} for name, rows in splits.items()}
    assert all(
        not family_sets[a] & family_sets[b]
        for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]
    )
    record_inspection = {}
    token_counts = {}
    for name, rows in splits.items():
        examples = text_examples(rows, mode="sft", max_length=384)
        rows_inspected = []
        for row, (x, y) in zip(rows, examples, strict=True):
            assert all(message["content"].isascii() for message in row["messages"])
            assistant = [m["content"] for m in row["messages"] if m["role"] == "assistant"]
            independent_count = sum(len(content.encode("ascii")) + 1 for content in assistant)
            assert independent_count == int((y != IGNORE).sum())
            assert x.dtype == y.dtype == torch.int64 and len(x) == len(y) <= 384
            action = json.loads(assistant[0])
            if row["operation"] == "copy":
                assert action == {"done": True, "answer": row["answer"]}
            else:
                assert action == {"name": row["operation"], "arguments": {"a": row["a"], "b": row["b"]}}
                assert call_tool(action) == row["answer"]
                assert json.loads(assistant[1]) == {"done": True, "answer": row["answer"]}
            rows_inspected.append(
                {
                    "family": row["family"],
                    "question": row["messages"][1]["content"],
                    "assistant_targets": independent_count,
                    "x_length": len(x),
                    "answer": row["answer"],
                }
            )
        record_inspection[name] = rows_inspected
        token_counts[name] = [row["assistant_targets"] for row in rows_inspected]
        assert result["split"][name] == {
            "records": len(rows),
            "families": len(family_sets[name]),
            "sha256": records_sha256(rows),
        }
    sampler = random.Random(42)
    step_counts = [
        sum(token_counts["train"][i] for i in sampler.choices(range(len(splits["train"])), k=16)) for _ in range(1200)
    ]
    assert sum(step_counts) == result["training"]["effective_tokens"] == 1296064
    for name in ["validation", "test"]:
        assert sum(token_counts[name]) == result[name]["effective_tokens"]
    x, y, valid = pad_batch(text_examples(splits["train"][:16], mode="sft", max_length=384))
    assert valid.dtype == torch.bool and x.dtype == y.dtype == torch.int64
    ctx = Context(
        "cpu", Path(tempfile.mkdtemp(prefix="fact_finish_b_1_random_")), ROOT, ROOT / "assets/training", seed=42
    )
    fresh = new_lm(ctx, width=64, layers=2, max_length=384, heads=2, backend="sdpa")
    assert fresh.description() == result["model"]
    assert torch.isfinite(fresh(x, valid=valid)["logits"]).all()
    episode_audits = [
        inspect_episode(episode, row) for episode, row in zip(result["samples"], splits["test"], strict=True)
    ]
    arithmetic = [r for r in splits["test"] if r["operation"] != "copy"]
    capped_audits = [
        inspect_episode(episode, row) for episode, row in zip(result["one_step_cap_samples"], arithmetic, strict=True)
    ]
    assert len(result["samples"]) == 20 and len(result["one_step_cap_samples"]) == 18
    assert sum(e["actual_tool_calls"] for e in result["samples"]) == 18
    assert sum(e["generated_tokens"] for e in result["samples"] + result["one_step_cap_samples"]) == 2072
    checkpoint = args.checkpoint
    if not checkpoint.exists():
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(WEIGHT_URL, timeout=60) as response:
            checkpoint.write_bytes(response.read())
    assert digest(checkpoint) == WEIGHT_HASH
    model, payload = load_checkpoint(checkpoint, "cpu")
    tensor_inspection = []
    for name, tensor in payload["model"].items():
        assert tensor.dtype == torch.float32 and torch.isfinite(tensor).all()
        tensor_inspection.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "elements": tensor.numel(),
                "sha256": hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest(),
            }
        )
    assert sum(t["elements"] for t in tensor_inspection) == 157952
    cpu_ctx = SimpleNamespace(device="cpu")
    cpu_episodes = [_tool_episode(model, row, cpu_ctx) for row in splits["test"]]
    cpu_capped = [_tool_episode(model, row, cpu_ctx, max_steps=1) for row in arithmetic]
    exact_token_matches = [
        [e["generation"]["samples"][0]["generated_ids"] for e in cpu["trace"]]
        == [e["generation"]["samples"][0]["generated_ids"] for e in gpu["trace"]]
        for cpu, gpu in zip(cpu_episodes, result["samples"], strict=True)
    ]
    cpu_nll = {name: _sft_nll(model, rows, cpu_ctx) for name, rows in splits.items() if name != "train"}
    save("_cpu_episodes.json", {"episodes": cpu_episodes, "one_step_cap": cpu_capped, "nll": cpu_nll})
    smoke = {"executed": False}
    if not args.skip_smoke:
        smoke_dir = Path(tempfile.mkdtemp(prefix="fact_finish_b_1_smoke_"))
        command = [
            str(ROOT / ".venv/bin/python"),
            "scripts/course_experiments/run.py",
            "--experiment",
            "tools",
            "--device",
            "cpu",
            "--step-scale",
            "0.000834",
            "--output",
            str(smoke_dir),
        ]
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
        raw = (smoke_dir / "tools/result.json").read_bytes()
        Path(str(PREFIX) + "_smoke.json").write_bytes(raw)
        smoke_report = json.loads(raw)
        assert smoke_report["results"]["training"]["steps"] == 1
        assert smoke_report["evidence_status"] == "interface_smoke_only"
        smoke = {
            "executed": True,
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "steps": 1,
            "scope": "One CPU optimizer update and actual controller generation; not the 1200-step GPU result.",
        }
    audit = {
        "reviewer_task": "/root/fact_finish_b_1",
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "device": "cpu",
            "cuda_available": str(torch.cuda.is_available()),
            "threads": "2",
        },
        "snapshot": {
            "source_sha256": SECTION_HASH,
            "intro_sha256": INTRO_HASH,
            "intro_summary": "本章由123乘45的明確請求、驗證、執行、返回與完成證據開始，再討論自然語言下直接回答、呼叫或求助的策略與成本；這是路線導覽，不能把手寫格式等同已習得能力。",
            "figure_references": re.findall(r"!\[[^\]]*\]\(([^)]+)\)", body),
        },
        "snippet_stdout": out.getvalue(),
        "exercise": exercise,
        "reviewer_extra_arithmetic_execution": arithmetic_demo,
        "hand_calculation": {
            "123_times_45": "123*(40+5)=4920+615=5535",
            "wrong_54": "123*(50+4)=6150+492=6642, which differs from 5535",
            "pair_families": "10*11/2=55 unordered pairs with repetition; plus 10 COPY families =65",
            "record_total": "10*10*2+10=210",
            "split_families": "floor(65*0.8)=52; floor(65*0.9)=58; 58-52=6; 65-58=7",
        },
        "record_inspection": record_inspection,
        "target_count_by_training_step": step_counts,
        "effective_targets": sum(step_counts),
        "split_summary": result["split"],
        "code_identity": code_identity,
        "gpu_report_environment": {
            key: report[key]
            for key in [
                "revision",
                "device",
                "seed",
                "torch_version",
                "python_version",
                "gpu",
                "timing_scope",
                "peak_memory_scope",
            ]
        },
        "gpu_episode_audits": episode_audits,
        "gpu_capped_audits": capped_audits,
        "weight_inspection": {
            "sha256": WEIGHT_HASH,
            "download_url": WEIGHT_URL,
            "public_bytes": checkpoint.stat().st_size,
            "source_sha256_declared": "11a36d5a864e355e4c3f81a05325fc295844e62a7631ba8577262bf7d077d851",
            "config": payload["config"],
            "format_version": payload["format_version"],
            "tokenizer": payload["tokenizer"],
            "tensors": tensor_inspection,
            "limitations": "Public inference export omits source optimizer/RNG and training step; source checkpoint bytes are not independently inspected.",
        },
        "cpu_replay": {
            "full_episodes": len(cpu_episodes),
            "correct": sum(e["correct"] for e in cpu_episodes),
            "actual_calls": sum(e["actual_tool_calls"] for e in cpu_episodes),
            "gpu_record_token_identical_episodes": sum(exact_token_matches),
            "capped_episodes": len(cpu_capped),
            "capped_correct": sum(e["correct"] for e in cpu_capped),
            "scope": "CPU inference from hash-pinned public export, no GPU speed/memory verification and no full retraining.",
        },
        "command_smoke": smoke,
        "assertions": "All listed assertions passed; full 1200-step training was not run.",
    }
    save("_audit.json", audit)
    print(
        json.dumps(
            {
                "effective_targets": sum(step_counts),
                "records": len(records),
                "families": 65,
                "cpu_replay": audit["cpu_replay"],
                "one_step_smoke": smoke["executed"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
