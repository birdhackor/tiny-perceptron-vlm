"""Bounded CPU checks of the exact lesson fence and published raw records.

No optimizer, training, model checkpoint load, or new model score evaluation.
"""
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
import hashlib
import io
import json
import os
import sys

ART = Path(__file__).resolve().parent
ROOT = ART.parents[3]
sys.path.insert(0, str(ROOT))
os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
import torch
from tiny_perceptron.capstone import TOK, CapstoneModel, build_dataset, calculator_runtime, default_config, expected_final, parse_action, prepare_batch, prompt_ids
from tiny_perceptron.data import IGNORE
from tiny_perceptron.model import masked_loss

assert torch.version.cuda is None and not torch.cuda.is_available()
torch.set_num_threads(1)
torch.set_default_device("cpu")
environment = {"python": sys.version, "python_executable": sys.executable, "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": "cpu", "cuda_build": str(torch.version.cuda), "cuda_available": str(torch.cuda.is_available()), "threads": str(torch.get_num_threads()), "offline": "CUDA disabled; HF offline; no inference checkpoints read"}
environment["repository_modules"] = {name: {"path": str(Path(module.__file__).resolve().relative_to(ROOT)), "sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()} for name, module in sorted(sys.modules.items()) if name.startswith("tiny_perceptron.") and getattr(module, "__file__", None)}
(ART / "cpu-environment.json").write_text(json.dumps(environment, indent=2) + "\n")
namespace = {"__name__": "__main__"}
buffer = io.StringIO()
raw = (ART / "fence-1.py").read_bytes()
with redirect_stdout(buffer):
    exec(compile(raw, "course/chapters/19.md#19.4:fence-1", "exec"), namespace)
output = buffer.getvalue()
print("EXACT ORIGINAL FENCE")
print(output, end="")
assert output == "問題 照抄數字29，只要答案。 示範回答 DIRECT:29\n預訓練 有效目標 43 誤差有限 True\nSFT 有效目標 10 誤差有限 True\n"
splits, manifest = build_dataset()
row = namespace["row"]
model = namespace["model"]
assert model.config.experts == 0
before = {name: p.detach().clone() for name, p in model.named_parameters()}
variant_results = []
for name, changed in (("original", row), ("longer_question", dict(row, user=row["user"] + "甲")), ("longer_answer", dict(row, answer=row["answer"] + "9"))):
    counts = []
    for pretrain in (True, False):
        batch, labels = prepare_batch([changed], pretrain=pretrain)
        observed = int((labels != IGNORE).sum())
        expected = len((changed["user"] + "\n" + changed["answer"]).encode("utf-8")) + 1 if pretrain else len(changed["answer"].encode("utf-8")) + 1
        assert observed == expected
        assert int(batch["valid"].sum()) == batch["ids"].numel()
        counts.append(observed)
    variant_results.append({"variant": name, "pretrain_targets": counts[0], "sft_targets": counts[1]})
assert variant_results == [{"variant": "original", "pretrain_targets": 43, "sft_targets": 10}, {"variant": "longer_question", "pretrain_targets": 46, "sft_targets": 10}, {"variant": "longer_answer", "pretrain_targets": 44, "sft_targets": 11}]
batch, labels = prepare_batch([row], pretrain=False)
result = model(**batch)
result["logits"].retain_grad()
loss = masked_loss(result["logits"], labels)
loss.backward()
ignored = labels == IGNORE
assert torch.count_nonzero(result["logits"].grad[ignored]) == 0
assert torch.count_nonzero(result["logits"].grad[~ignored]) > 0
assert all(torch.equal(before[name], p.detach()) for name, p in model.named_parameters())
changed_batch, changed_labels = prepare_batch([dict(row, user=row["user"].replace("29", "28"))], pretrain=False)
assert torch.equal(labels, changed_labels)
assert not torch.equal(batch["ids"], changed_batch["ids"])
with torch.no_grad():
    delta = (model(**changed_batch)["logits"][~ignored] - result["logits"].detach()[~ignored]).abs().max().item()
assert delta > 0
try:
    masked_loss(result["logits"], torch.full_like(labels, IGNORE))
except ValueError:
    no_targets_error = True
else:
    raise AssertionError("all ignored must be rejected by repository helper")
toy = torch.tensor([[[1.0, 2.0], [3.0, -1.0], [0.0, 0.0]]])
toy_labels = torch.tensor([[1, IGNORE, 0]])
manual = (-torch.log_softmax(toy[0, 0], -1)[1] - torch.log_softmax(toy[0, 2], -1)[0]) / 2
assert torch.allclose(masked_loss(toy, toy_labels), manual, atol=1e-7, rtol=0)
print("LABEL VARIANTS", json.dumps(variant_results))
print("IGNORE context preserved: all input positions valid; changed question changes answer logits", delta)
print("No parameter updates; direct logit gradient is zero at ignored positions; all-IGNORE raises ValueError")
print("Manual CE denominator=2 valid target positions; loss", float(manual))

mapping = {row["id"]: row for row in splits["validation"]}
stages = [("pretrain", "pretrain", 300, 0), ("sft", "sft", 1400, 42), ("joint", "joint", 600, 75), ("dpo", "preference", 100, 71)]
summaries = []
previous = None
ids_reference = None
for stage, name, steps, score in stages:
    d = json.loads((ART / "original" / f"docs/course-experiments/results/capstone_{name}.json").read_bytes())
    validation_path = ART / "original" / f"docs/course-experiments/capstone-evidence/{stage}/validation.json"
    v = json.loads(validation_path.read_bytes())
    r = d["results"]
    assert v["protocol"] == r["validation_summary"]["protocol"] == "exact generated action + EOS; requested calculator arguments must match; runtime return is input to a second generation"
    assert d["device"] == "cuda" and d["gpu"] == "NVIDIA L4" and d["seed"] == 42
    assert r["data_version"] == "capstone-small-world-v2" and r["seed"] == 42
    assert r["stage"] == stage and r["steps"] == r["requested_steps"] == r["new_steps"] == steps
    assert r["schedule_completed"] is True and r["test_evaluated"] is False
    assert r["data_manifest"]["sha256"] == manifest["sha256"]
    assert r["data_manifest"]["counts"] == manifest["counts"] == {"train": 552, "validation": 84, "test": 90}
    assert r["parent_checkpoint_sha256"] == previous
    if previous is not None:
        assert len(previous) == 64
    previous = r["inference_export"]["sha256"]
    for source, recorded in r["code_sha256"].items():
        assert hashlib.sha256((ART / "original" / source).read_bytes()).hexdigest() == recorded
    receipt = next(a for a in d["artifacts"] if a["path"] == "validation.json")
    assert hashlib.sha256(validation_path.read_bytes()).hexdigest() == receipt["sha256"]
    ids = [rec["id"] for rec in v["records"]]
    assert len(ids) == len(set(ids)) == 84 and set(ids) == set(mapping)
    if ids_reference is None:
        ids_reference = ids
    else:
        assert ids == ids_reference
    by_task = {}
    errors = Counter()
    for rec in v["records"]:
        row = mapping[rec["id"]]
        assert rec["family"] == row["family"] and rec["task"] == row["task"]
        assert rec["expected_action"] == row["answer"] and rec["expected_final"] == expected_final(row)
        assert rec["action_trace"]["prompt_ids"] == prompt_ids(row)
        trace = rec["action_trace"]
        assert trace["raw"] == TOK.decode(trace["generated_ids"])
        assert trace["eos"] == (TOK.eos_id in trace["generated_ids"])
        assert trace["eos"] == (trace["stop_reason"] == "eos")
        if trace["eos"]:
            assert trace["generated_ids"][-1] == TOK.eos_id and all(token >= 8 for token in trace["generated_ids"][:-1])
        parsed = parse_action(trace)
        assert parsed == rec["parsed_action"]
        action_correct = trace["eos"] and trace["raw"] == row["answer"]
        answer = None
        if parsed["status"] in ("direct", "ask"):
            answer = parsed["content"]
        elif parsed["status"] == "tool":
            assert rec["runtime"] == calculator_runtime(parsed, row["available"])
            if rec["runtime"]["status"] == "ok":
                final = rec["final_trace"]
                assert final["raw"] == TOK.decode(final["generated_ids"])
                assert final["eos"] == (TOK.eos_id in final["generated_ids"])
                assert final["eos"] == (final["stop_reason"] == "eos")
                if final["eos"]:
                    assert final["generated_ids"][-1] == TOK.eos_id and all(token >= 8 for token in final["generated_ids"][:-1])
                final_parsed = parse_action(final)
                if final_parsed["status"] == "direct":
                    answer = final_parsed["content"]
        end_correct = bool(action_correct and answer == expected_final(row))
        assert answer == rec["answer"] and bool(action_correct) == rec["action_correct"] and end_correct == rec["end_to_end_correct"]
        t = by_task.setdefault(rec["task"], {"count": 0, "action_correct": 0, "end_to_end_correct": 0})
        t["count"] += 1
        t["action_correct"] += int(action_correct)
        t["end_to_end_correct"] += int(end_correct)
        if not end_correct:
            errors[rec["task"]] += 1
    assert by_task == v["by_task"] == r["validation_summary"]["by_task"]
    assert sum(t["count"] for t in by_task.values()) == v["count"] == r["validation_summary"]["count"] == 84
    assert sum(t["action_correct"] for t in by_task.values()) == v["action_correct"] == r["validation_summary"]["action_correct"] == score
    assert sum(t["end_to_end_correct"] for t in by_task.values()) == v["end_to_end_correct"] == r["validation_summary"]["end_to_end_correct"] == score
    modality_tasks = {"image_color", "image_shape", "joint", "audio"}
    modalities = {"count": sum(t["count"] for name, t in by_task.items() if name in modality_tasks), "correct": sum(t["end_to_end_correct"] for name, t in by_task.items() if name in modality_tasks)}
    text = {"count": 84 - modalities["count"], "correct": score - modalities["correct"]}
    summaries.append({"stage": stage, "completed_updates": steps, "score": score, "denominator": 84, "text": text, "modalities": modalities, "failed_tasks": dict(errors), "effective_ce_targets": r["effective_tokens"], "parent_checkpoint_sha256": r["parent_checkpoint_sha256"], "inference_export_sha256": previous})
assert summaries[1]["text"] == {"count": 42, "correct": 42} and summaries[1]["modalities"] == {"count": 42, "correct": 0}
assert summaries[2]["text"] == {"count": 42, "correct": 42} and summaries[2]["modalities"] == {"count": 42, "correct": 33}
assert summaries[2]["failed_tasks"] == {"image_shape": 9}
print("RECALCULATED PUBLISHED RECORDS (not model reevaluation)")
print(json.dumps(summaries, ensure_ascii=False, indent=2))
result = {"environment": environment, "original_fence_sha256": hashlib.sha256(raw).hexdigest(), "original_stdout": output, "label_variants": variant_results, "parameters_unchanged": True, "ignored_direct_logit_gradients_zero": True, "prompt_change_answer_logit_max_abs_delta": delta, "manual_cross_entropy": {"denominator": 2, "value": float(manual), "absolute_tolerance": 1e-7}, "dataset_sha256": manifest["sha256"], "counts": manifest["counts"], "published_records_recalculated": summaries, "record_count_recalculated": 336}
(ART / "cpu-results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print("ALL CHECKS PASSED")
