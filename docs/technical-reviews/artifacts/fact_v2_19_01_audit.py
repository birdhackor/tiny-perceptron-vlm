"""Independent 19.1 review: source binding, frozen row audit and CPU replay."""

import hashlib
import importlib.metadata
import json
import platform
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts.check_technical_reviews import sections  # noqa: E402
from tiny_perceptron.capstone import (  # noqa: E402
    TOK,
    build_dataset,
    calculator_runtime,
    evaluate_rows,
    expected_final,
    load_capstone,
    modality_tensors,
    parse_action,
    prompt_ids,
)

PREFIX = ROOT / "docs/technical-reviews/artifacts/fact_v2_19_01"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, obj):
    path = Path(str(PREFIX) + name)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    checkpoint = Path(sys.argv[1])
    torch.set_num_threads(2)
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "huggingface_hub": importlib.metadata.version("huggingface-hub"),
        "device": "cpu",
        "threads": str(torch.get_num_threads()),
        "platform": platform.platform(),
    }
    source_paths = [
        "course/chapters/19.md",
        "docs/technical-review-guide.md",
        "docs/environment.md",
        "tiny_perceptron/capstone.py",
        "tiny_perceptron/data.py",
        "tiny_perceptron/model.py",
        "tiny_perceptron/multimodal.py",
        "tiny_perceptron/modern.py",
        "tiny_perceptron/attention.py",
        "tiny_perceptron/capstone_ui.py",
        "scripts/capstone.py",
        "scripts/fetch_capstone.py",
        "scripts/capstone_release.py",
        "scripts/course_experiments/capstone.py",
        "scripts/course_experiments/capstone_deployment.py",
        "pyproject.toml",
        "uv.lock",
        "scripts/check_technical_reviews.py",
    ]
    binding = {p: sha(ROOT / p) for p in source_paths}
    chapter = (ROOT / "course/chapters/19.md").read_bytes()
    intro = chapter[: re.search(rb"(?m)^## ", chapter).start()]
    Path(str(PREFIX) + "_intro.md").write_bytes(intro)
    section = dict(sections(ROOT / "course/chapters/19.md"))["19.1"].encode()
    Path(str(PREFIX) + "_section.md").write_bytes(section)
    prerequisites = {}
    prereq_text = []
    for name, lesson in [("07.md", "7.1"), ("10.md", "10.1"), ("12.md", "12.1"), ("0B.md", "B.5"), ("19.md", "19.11")]:
        body = dict(sections(ROOT / "course/chapters" / name))[lesson]
        prerequisites[f"course/chapters/{name}#{lesson}"] = hashlib.sha256(body.encode()).hexdigest()
        prereq_text.append(body)
    Path(str(PREFIX) + "_prerequisites.md").write_text("\n".join(prereq_text))
    data_path = ROOT / "docs/course-experiments/capstone-evidence/deployment/data.json"
    formal_path = ROOT / "docs/course-experiments/capstone-evidence/deployment/test-joint.json"
    data = json.loads(data_path.read_text())
    formal = json.loads(formal_path.read_text())
    splits, manifest = build_dataset(seed=42)
    assert data["splits"] == splits
    assert data["manifest"] == manifest
    assert len(splits["test"]) == 90
    by_id = {r["id"]: r for r in splits["test"]}
    assert len(by_id) == 90
    assert set(by_id) == {r["id"] for r in formal["records"]}
    assert len(formal["records"]) == 90
    audited = []
    for record in formal["records"]:
        row = by_id[record["id"]]
        first = record["action_trace"]
        assert first["prompt_ids"] == prompt_ids(row)
        assert first["raw"] == TOK.decode(first["generated_ids"])
        assert first["generated_ids"] == TOK.encode(first["raw"]) + [TOK.eos_id]
        assert first["eos"] is True and first["stop_reason"] == "eos"
        action = parse_action(first)
        assert record["parsed_action"] == action
        assert record["family"] == row["family"] and record["task"] == row["task"]
        assert record["expected_action"] == row["answer"]
        assert record["expected_final"] == expected_final(row)
        action_ok = first["eos"] and first["raw"] == row["answer"]
        answer = None
        if action["status"] in ("direct", "ask"):
            answer = action["content"]
            assert record["runtime"] is None and record["final_trace"] is None
        elif action["status"] == "tool":
            runtime = calculator_runtime(action, row["available"])
            assert runtime == record["runtime"]
            if runtime["status"] == "ok":
                final = record["final_trace"]
                followup = dict(row, image=None, audio=None)
                followup["user"] = f"原題：{action['a']}+{action['b']}。計算器回報：{runtime['result']}。請回答。"
                assert final["prompt_ids"] == prompt_ids(followup)
                assert final["generated_ids"] == TOK.encode(final["raw"]) + [TOK.eos_id]
                assert final["eos"] is True and final["stop_reason"] == "eos"
                parsed = parse_action(final)
                if parsed["status"] == "direct":
                    answer = parsed["content"]
        final_ok = bool(action_ok and answer == expected_final(row))
        assert record["answer"] == answer
        assert record["action_correct"] == action_ok
        assert record["end_to_end_correct"] == final_ok
        image, audio = modality_tensors(row)
        audited.append(
            {
                "id": row["id"],
                "row": row,
                "raw": first["raw"],
                "generated_ids": first["generated_ids"],
                "eos": first["eos"],
                "runtime": record["runtime"],
                "final_trace": record["final_trace"],
                "answer": answer,
                "action_correct_recomputed": action_ok,
                "end_to_end_correct_recomputed": final_ok,
                "image_shape": None if image is None else list(image.shape),
                "image_sha256": None if image is None else tensor_sha(image),
                "audio_feature_shape": None if audio is None else list(audio.shape),
                "audio_feature_sha256": None if audio is None else tensor_sha(audio),
            }
        )
    counts = Counter(r["row"]["task"] for r in audited)
    by_task = {
        task: {
            "count": count,
            "action_correct": sum(r["action_correct_recomputed"] for r in audited if r["row"]["task"] == task),
            "end_to_end_correct": sum(r["end_to_end_correct_recomputed"] for r in audited if r["row"]["task"] == task),
        }
        for task, count in counts.items()
    }
    assert by_task == formal["by_task"]
    assert sum(r["action_correct_recomputed"] for r in audited) == formal["action_correct"] == 80
    assert sum(r["end_to_end_correct_recomputed"] for r in audited) == formal["end_to_end_correct"] == 78
    model, public = load_capstone(checkpoint, "cpu")
    assert model.description()["parameters"] == 328128
    original_path = ROOT / "outputs/integration-runs/v2-joint/capstone-review/capstone_joint/gha-37168518451-1/model.pt"
    original_model, original = load_capstone(original_path, "cpu")
    assert sha(original_path) == formal["checkpoint_sha256"]
    tensors = []
    for name, value in model.state_dict().items():
        source = original_model.state_dict()[name]
        assert torch.equal(value, source)
        assert value.dtype == torch.float32
        tensors.append(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "numel": value.numel(),
                "public_sha256": tensor_sha(value),
                "original_sha256": tensor_sha(source),
                "exact_equal": True,
            }
        )
    assert sum(v["numel"] for v in tensors) == 328128
    selection = json.loads((ROOT / "docs/course-experiments/capstone-selection.json").read_text())
    assert selection["selected_stage"] == "joint" and selection["selected_before_test_generation"] is True
    candidates = {}
    for stage, candidate in selection["candidates"].items():
        path = ROOT / candidate["evidence"]
        assert sha(path) == candidate["evidence_sha256"]
        report = json.loads(path.read_text())
        validation = json.loads(
            (ROOT / f"docs/course-experiments/capstone-evidence/{stage}/validation.json").read_text()
        )
        assert validation["count"] == candidate["validation_count"] == 84
        assert validation["end_to_end_correct"] == candidate["validation_end_to_end_correct"]
        assert validation["end_to_end_correct"] == sum(r["end_to_end_correct"] for r in validation["records"])
        candidates[stage] = {
            "validation_count": 84,
            "end_to_end_correct": validation["end_to_end_correct"],
            "evidence_sha256": sha(path),
            "gpu": report["gpu"],
            "seed": report["seed"],
        }
    started = time.perf_counter()
    replay = evaluate_rows(model, splits["test"], max_new_tokens=64, batch_size=24)
    elapsed = time.perf_counter() - started
    write("_cpu_replay.json", replay)
    comparisons = []
    for old, new in zip(formal["records"], replay["records"], strict=True):
        comparisons.append(
            {
                "id": old["id"],
                "whole_record_equal": old == new,
                "action_ids_equal": old["action_trace"]["generated_ids"] == new["action_trace"]["generated_ids"],
                "final_ids_equal": (old["final_trace"] or {}).get("generated_ids")
                == (new["final_trace"] or {}).get("generated_ids"),
            }
        )
    deployment = json.loads((ROOT / "docs/course-experiments/results/capstone_deployment.json").read_text())
    frozen_code = {
        p: {
            "observed": binding[p],
            "formal": deployment["code_sha256"].get(p),
            "equal": binding[p] == deployment["code_sha256"].get(p),
        }
        for p in binding
        if p in deployment["code_sha256"]
    }
    snippets = re.findall(r"```python\n(.*?)```", section.decode(), flags=re.S)
    assert len(snippets) == 1
    Path(str(PREFIX) + "_snippet.py").write_text(snippets[0])
    actions = [
        parse_action({"raw": value, "eos": True})
        for value in ["DIRECT:red", "ASK:請提供數量", "TOOL:calculator:1+2", "DIRECT:"]
    ]
    assert [a["status"] for a in actions] == ["direct", "ask", "tool", "invalid"]
    result = {
        "command": " ".join(sys.argv),
        "environment": environment,
        "intro_sha256": hashlib.sha256(intro).hexdigest(),
        "source_sha256": hashlib.sha256(section).hexdigest(),
        "read_and_executed_source_sha256": binding,
        "prerequisite_sha256": prerequisites,
        "formal_data_sha256": sha(data_path),
        "formal_evaluation_sha256": sha(formal_path),
        "split_manifest": manifest,
        "formal_environment": {
            k: deployment[k]
            for k in [
                "revision",
                "device",
                "seed",
                "torch_version",
                "python_version",
                "gpu",
                "elapsed_seconds",
                "timing_scope",
            ]
        },
        "frozen_code_comparison": frozen_code,
        "formal_count": 90,
        "action_correct_recomputed": 80,
        "end_to_end_correct_recomputed": 78,
        "by_task_recomputed": by_task,
        "audited_rows": audited,
        "selection": selection,
        "selection_candidate_checks": candidates,
        "public_checkpoint": {
            "path": str(checkpoint),
            "sha256": sha(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "format_version": public["format_version"],
            "stage": public["stage"],
            "step": public["step"],
            "inference_only": public["inference_only"],
            "config": public["config"],
            "tokenizer": public["tokenizer"],
        },
        "original_checkpoint": {
            "path": str(original_path.relative_to(ROOT)),
            "sha256": sha(original_path),
            "metadata": original["metadata"],
            "stage": original["stage"],
            "step": original["step"],
        },
        "tensor_receipts": tensors,
        "parameters": model.description(),
        "cpu_replay_count": replay["count"],
        "cpu_replay_end_to_end_correct": replay["end_to_end_correct"],
        "cpu_replay_elapsed_seconds": elapsed,
        "cpu_replay_timing_scope": "one CPU evaluate_rows after checkpoint load and data generation; excludes download, model load, audit and report writes; no GPU speed claim",
        "cpu_comparisons": comparisons,
        "all_cpu_records_equal_formal": all(r["whole_record_equal"] for r in comparisons),
        "parser_examples_including_exercise": actions,
    }
    assert all(sha(ROOT / p) == h for p, h in binding.items()), "source changed while executing"
    write("_audit.json", result)
    print(
        json.dumps(
            {
                "environment": environment,
                "parameters": model.description()["parameters"],
                "formal_count": 90,
                "formal_end_to_end": 78,
                "cpu_count": replay["count"],
                "cpu_end_to_end": replay["end_to_end_correct"],
                "all_cpu_records_equal_formal": result["all_cpu_records_equal_formal"],
                "cpu_elapsed_seconds": elapsed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
