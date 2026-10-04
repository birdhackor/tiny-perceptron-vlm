"""Independent CPU audit of lesson 19.5; no training and no weight files in Git.

Replay: PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/fact_v2_19_05_replay.py
If original ignored checkpoints are absent, public inference exports are fetched
at a fixed commit with token=False and compared to the persisted tensor inventory.
"""

import copy
import hashlib
import json
import math
import platform
import random
import subprocess
import time
from collections import Counter
from pathlib import Path

import torch

from scripts.check_technical_reviews import sections
from scripts.course_experiments.capstone import _balanced_sample
from tiny_perceptron.alignment import dpo_loss
from tiny_perceptron.capstone import (
    STAGES,
    TOK,
    build_dataset,
    digest,
    encode_record,
    evaluate_rows,
    load_capstone,
    parse_action,
    preference_pairs,
    prepare_batch,
    prompt_ids,
)
from tiny_perceptron.data import IGNORE

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).with_name("fact_v2_19_05_execution.json")
COHORTS = ("style", "missing", "unavailable", "safety")
STAGE_PATHS = {
    "sft": "outputs/integration-runs/v2-sft/capstone-review/capstone_sft/gha-37168151999-1/model.pt",
    "joint": "outputs/integration-runs/v2-joint/capstone-review/capstone_joint/gha-37168518451-1/model.pt",
    "dpo": "outputs/integration-runs/v2-dpo/capstone-review/capstone_preference/gha-37168874504-1/model.pt",
}
RESULT_NAMES = {"sft": "capstone_sft", "joint": "capstone_joint", "dpo": "capstone_preference"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads((ROOT / path).read_text())


def tensor_inventory(model):
    entries = {}
    for key, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        entries[key] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": value.numel(),
            "finite": bool(torch.isfinite(value).all()),
            "sha256_raw_tensor_bytes": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
        }
    return {"entries": entries, "canonical_sha256": digest(entries)}


def checkpoint_path(stage, expected_original, baseline):
    path = ROOT / STAGE_PATHS[stage]
    if path.is_file():
        assert sha(path) == expected_original["sha256"]
        assert path.stat().st_size == expected_original["bytes"]
        return path, {"type": "original_ignored_inference_checkpoint", "path": STAGE_PATHS[stage]}
    from huggingface_hub import hf_hub_download

    public = read("docs/course-experiments/capstone-public.json")
    assert public["repo"] == "birdhackor/tiny-perceptron-course-models"
    assert public["revision"] == "33c6898f0676fccc4f5f6114e3b93a4f9ebaeaed"
    item = next(x for x in public["models"] if x["id"] == stage)["files"][0]
    path = Path(hf_hub_download(public["repo"], item["path"], revision=public["revision"], token=False))
    assert sha(path) == item["sha256"] and path.stat().st_size == item["bytes"]
    assert baseline is not None, "Public fallback requires the persisted original tensor inventory"
    return path, {"type": "public_fixed_pin", "repo": public["repo"], "revision": public["revision"], **item}


def check_trace(trace, expected_prompt):
    assert trace["prompt_ids"] == expected_prompt
    assert trace["raw"] == TOK.decode(trace["generated_ids"])
    eos = bool(trace["generated_ids"] and trace["generated_ids"][-1] == TOK.eos_id)
    assert trace["eos"] == eos
    assert (trace["stop_reason"] == "eos") == eos
    assert TOK.eos_id not in trace["generated_ids"][:-1]
    return eos


def audit_records(value, rows):
    assert value["count"] == len(rows) == len(value["records"])
    by_task = {}
    audit = []
    for row, record in zip(rows, value["records"], strict=True):
        assert record["id"] == row["id"]
        assert record["family"] == row["family"] and record["task"] == row["task"]
        assert record["expected_action"] == row["answer"]
        action_eos = check_trace(record["action_trace"], prompt_ids(row))
        parsed = parse_action(record["action_trace"])
        assert parsed == record["parsed_action"]
        expected = row["answer"].split(":", 1)[1]
        if row["task"] == "calculator":
            a, b = map(int, expected.split(":")[1].split("+"))
            expected = str(a + b)
        assert record["expected_final"] == expected
        answer = None
        if parsed["status"] in ("ask", "direct"):
            answer = parsed["content"]
            assert record["runtime"] is None and record["final_trace"] is None
        elif parsed["status"] == "tool":
            runtime = record["runtime"]
            if runtime["status"] == "ok":
                assert row["available"] and parsed["name"] == "calculator"
                assert runtime["result"] == str(parsed["a"] + parsed["b"])
                followup = dict(row, image=None, audio=None)
                followup["user"] = (
                    f"原題：{parsed['a']}+{parsed['b']}。計算器回報：{runtime['result']}。請回答。"
                )
                check_trace(record["final_trace"], prompt_ids(followup))
                final = parse_action(record["final_trace"])
                if final["status"] == "direct":
                    answer = final["content"]
            else:
                assert record["final_trace"] is None
        assert record["answer"] == answer
        action_correct = action_eos and record["action_trace"]["raw"] == row["answer"]
        correct = action_correct and answer == expected
        assert record["action_correct"] == action_correct and record["end_to_end_correct"] == correct
        counts = by_task.setdefault(row["task"], {"count": 0, "action_correct": 0, "end_to_end_correct": 0})
        counts["count"] += 1
        counts["action_correct"] += int(action_correct)
        counts["end_to_end_correct"] += int(correct)
        audit.append({
            "id": row["id"], "family": row["family"], "task": row["task"],
            "user": row["user"], "system": row["system"], "image": row["image"], "audio": row["audio"],
            "expected_action": row["answer"], "expected_final": expected,
            "raw": record["action_trace"]["raw"], "eos": action_eos,
            "generated_ids": record["action_trace"]["generated_ids"],
            "prompt_ids": record["action_trace"]["prompt_ids"],
            "runtime": record["runtime"], "final_trace": record["final_trace"],
            "answer": answer, "action_correct": action_correct, "end_to_end_correct": correct,
        })
    assert by_task == value["by_task"]
    assert sum(v["action_correct"] for v in by_task.values()) == value["action_correct"]
    assert sum(v["end_to_end_correct"] for v in by_task.values()) == value["end_to_end_correct"]
    return {"by_task": by_task, "records": audit, "all_records_audited": len(audit)}


def target_schedule(stage, splits, steps):
    text_rows = [r for r in splits["train"] if r["image"] is None and r["audio"] is None]
    rows = text_rows if stage == "sft" else splits["train"]
    counts = {row["id"]: int((encode_record(row)[1] != IGNORE).sum()) for row in rows}
    rng = random.Random(42 + STAGES.index(stage) * 1000)
    pairs = preference_pairs(splits["train"])
    selected_counts = Counter()
    preference_tokens = Counter()
    effective_tokens = 0
    for _ in range(steps):
        selected = _balanced_sample(rows, rng, 24)
        effective_tokens += sum(counts[row["id"]] for row in selected)
        selected_counts.update(row["task"] for row in selected)
        if stage == "dpo":
            selected_pairs = rng.choices(pairs, k=12)
            for p in selected_pairs:
                for side in ("chosen", "rejected"):
                    preference_tokens[side] += int((encode_record(p["row"], answer=p[side])[1] != IGNORE).sum())
    return {
        "algorithm": "Actual encode_record target lengths; original _balanced_sample and seeded RNG replay, no training",
        "batch_size": 24, "steps": steps, "seed": 42, "sampling_seed": 42 + STAGES.index(stage) * 1000,
        "training_rows": len(rows), "ce_effective_targets": effective_tokens,
        "task_draw_counts": dict(selected_counts), "preference_pair_count": len(pairs) if stage == "dpo" else 0,
        "preference_pairs_per_step": 12 if stage == "dpo" else 0,
        "preference_answer_targets": dict(preference_tokens),
        "scope": "CE count includes EOS and excludes ignored prefix/PAD. DPO CE count excludes policy/reference chosen/rejected scores.",
    }


def main():
    torch.set_num_threads(2)
    started = time.perf_counter()
    baseline = json.loads(OUTPUT.read_text()) if OUTPUT.is_file() else None
    splits, manifest = build_dataset(seed=42)
    data = read("docs/course-experiments/capstone-evidence/deployment/data.json")
    assert data["splits"] == splits
    assert manifest["sha256"] == {name: digest(rows) for name, rows in data["splits"].items()}
    version_bodies = {}
    for file, identifiers in (("19", ("19.1", "19.4", "19.5")), ("08", ("8.6", "8.7")), ("09", ("9.4", "9.6"))):
        bodies = dict(sections(ROOT / f"course/chapters/{file}.md"))
        for identifier in identifiers:
            version_bodies[identifier] = {
                "path": f"course/chapters/{file}.md#{identifier}",
                "sha256": hashlib.sha256(bodies[identifier].encode()).hexdigest(),
            }
    stdout_examples = []
    for task in COHORTS:
        row = next(r for r in splits["train"] if r["task"] == task)
        stdout_examples.extend((f"任務 {task} 問題 {row['user']}", f"工作設定 {row['system']} 示範回答 {row['answer']}"))
    rag = next(r for r in splits["train"] if r["task"] == "rag")
    assert "已讀資料" in rag["user"]
    mask_check = []
    for task in (*COHORTS, "rag"):
        row = next(r for r in splits["train"] if r["task"] == task)
        batch, labels = prepare_batch([row])
        prefix_length = len(prompt_ids(row))
        targets = labels[labels != IGNORE].tolist()
        assert targets == TOK.encode(row["answer"]) + [TOK.eos_id]
        assert bool((labels[0, :prefix_length - 1] == IGNORE).all())
        assert batch["ids"].dtype == torch.long and labels.dtype == torch.long
        mask_check.append({"task": task, "input_shape": list(batch["ids"].shape),
                           "dtype": str(labels.dtype), "prefix_ignored_positions": prefix_length - 1,
                           "effective_targets": len(targets), "last_target": targets[-1], "answer": row["answer"]})
    limit_baselines = {task: manifest["task_majority_baselines"]["validation"][task]
                       for task in (*COHORTS, "rag")}
    protocol_probes = []
    for raw, eos, expected_status in (("DIRECT:red", True, "direct"),
                                     ("ASK:請提供數量", True, "ask"),
                                     ("TOOL:calculator:1+2", True, "tool"),
                                     ("DIRECT:", True, "invalid"),
                                     ("DIRECT:red。", False, "invalid")):
        parsed = parse_action({"raw": raw, "eos": eos})
        assert parsed["status"] == expected_status
        protocol_probes.append({"raw": raw, "eos": eos, "parsed": parsed})
    for row in splits["validation"]:
        assert row["system"] == f"計算器={'開' if row['available'] else '關'}；風格=短。"
        prefix = prompt_ids(row)
        assert prefix[:2] == [TOK.bos_id, TOK.system_id]
        assert prefix[2:2 + len(TOK.encode(row["system"]))] == TOK.encode(row["system"])
    public = read("docs/course-experiments/capstone-public.json")
    result = {
        "lesson_id": "19.5", "reviewer_task": "/root/integration_technical_coordinator/fact_v2_19_05",
        "command": "PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/fact_v2_19_05_replay.py",
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "device": "cpu", "threads": str(torch.get_num_threads()),
                        "platform": platform.platform(), "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()},
        "section_read_versions": version_bodies,
        "source_code_sha256": {p: sha(ROOT / p) for p in (
            "tiny_perceptron/capstone.py", "tiny_perceptron/alignment.py", "tiny_perceptron/data.py",
            "scripts/course_experiments/capstone.py", "docs/course-experiments/capstone-public.json")},
        "dataset": {"version": manifest["version"], "seed": manifest["seed"], "counts": manifest["counts"],
                    "sha256": manifest["sha256"], "validation_ids": [r["id"] for r in splits["validation"]]},
        "lesson_stdout": "\n".join(stdout_examples), "rag_exercise": rag,
        "mask_and_dtype_check": mask_check, "validation_constant_baselines": limit_baselines,
        "protocol_probes": protocol_probes,
        "stages": {}, "issues": [],
    }
    for stage in ("sft", "joint", "dpo"):
        experiment_path = f"docs/course-experiments/results/{RESULT_NAMES[stage]}.json"
        experiment = read(experiment_path)
        train = experiment["results"]
        original_path = f"docs/course-experiments/capstone-evidence/{stage}/validation.json"
        original = read(original_path)
        original_audit = audit_records(original, splits["validation"])
        assert train["validation_summary"] == {k: v for k, v in original.items() if k != "records"}
        assert train == read(f"docs/course-experiments/capstone-evidence/{stage}/train-report.json")
        assert train["data_manifest"] == manifest
        assert train["schedule_completed"] and not train["budget_exhausted"] and not train["test_evaluated"]
        for path, expected in train["code_sha256"].items():
            assert sha(ROOT / path) == expected
        weight_path, origin = checkpoint_path(stage, train["inference_export"], baseline)
        model, payload = load_capstone(weight_path, "cpu")
        inventory = tensor_inventory(model)
        if origin["type"] == "public_fixed_pin":
            assert inventory == baseline["stages"][stage]["weight"]["tensor_inventory"]
        else:
            assert payload["metadata"]["parent_checkpoint_sha256"] == train["parent_checkpoint_sha256"]
        assert payload["stage"] == stage and payload["step"] == train["steps"]
        assert payload["config"] == train["parameters"]["config"]
        assert payload["tokenizer"] == TOK.state() and payload["data_version"] == manifest["version"]
        assert model.description() == train["parameters"]
        assert all(t["finite"] for t in inventory["entries"].values())
        infer_started = time.perf_counter()
        replay = evaluate_rows(model, splits["validation"], max_new_tokens=64, batch_size=24)
        replay_seconds = time.perf_counter() - infer_started
        replay_audit = audit_records(replay, splits["validation"])
        differences = []
        for old, new in zip(original_audit["records"], replay_audit["records"], strict=True):
            if old != new:
                differences.append({"id": old["id"], "original": old, "replay": new})
        assert not differences
        text = [r for r in replay_audit["records"] if r["image"] is None and r["audio"] is None]
        other = [r for r in text if r["task"] not in COHORTS]
        normal = [r for r in text if r["task"] != "safety"]
        refusal = [r for r in text if r["task"] == "safety"]
        assert len(text) == 42 and len(other) == 23 and len(normal) == 39 and len(refusal) == 3
        assert all(r["end_to_end_correct"] and r["eos"] for r in text)
        assert all("不能提供他人密碼" not in r["raw"] for r in normal)
        for task, expected in zip(COHORTS, (3, 3, 10, 3), strict=True):
            selected = [r for r in text if r["task"] == task]
            assert len(selected) == expected and all(r["end_to_end_correct"] for r in selected)
        item_public = next(x for x in public["models"] if x["id"] == stage)["files"][0]
        schedule = target_schedule(stage, splits, train["steps"])
        assert schedule["ce_effective_targets"] == train["effective_tokens"]
        result["stages"][stage] = {
            "input_evidence": {"validation": {"path": original_path, "sha256": sha(ROOT / original_path)},
                               "experiment": {"path": experiment_path, "sha256": sha(ROOT / experiment_path)}},
            "original_environment": {k: experiment[k] for k in (
                "revision", "python_version", "torch_version", "gpu", "seed", "elapsed_seconds", "timing_scope")},
            "weight": {"origin": origin, "actual_file_sha256": sha(weight_path), "actual_bytes": weight_path.stat().st_size,
                       "metadata": {k: v for k, v in payload.items() if k != "model"},
                       "tensor_inventory": inventory, "strict_load": True,
                       "public_retrieval": {"repo": public["repo"], "revision": public["revision"], "token": False, **item_public}},
            "original_audit": original_audit, "cpu_replay": replay_audit,
            "cpu_replay_summary": {k: v for k, v in replay.items() if k != "records"},
            "record_differences": differences,
            "cpu_seconds": replay_seconds,
            "cpu_timing_scope": "One evaluate_rows CPU replay, includes input generation and tool followup; no GPU/training speed claim",
            "text_cohort": {"all_text": len(text), "all_text_correct": sum(r["end_to_end_correct"] for r in text),
                            "other_text": len(other), "other_text_correct": sum(r["end_to_end_correct"] for r in other),
                            "normal_text": len(normal), "normal_text_correct": sum(r["end_to_end_correct"] for r in normal),
                            "normal_text_refusal_phrase": sum("不能提供他人密碼" in r["raw"] for r in normal),
                            "safety": len(refusal), "safety_refusal_phrase": sum("不能提供他人密碼" in r["raw"] for r in refusal)},
            "training_configuration": {"schedule": schedule, "learning_rate": {"sft": .003, "joint": .0015, "dpo": .0002}[stage],
                                       "optimizer": "AdamW defaults (betas=.9,.999, eps=1e-8, weight_decay=.01)",
                                       "grad_clip_norm": 1.0, "auxiliary_weight": .01, "dpo_beta": .1,
                                       "dpo_supervised_anchor_weight": .2, "objective": train["objective"],
                                       "training_seconds": train["seconds"], "experiment_seconds": experiment["elapsed_seconds"],
                                       "training_time_scope": "GPU sync surrounds loop; includes every 100-step save; excludes final export/evaluation",
                                       "experiment_time_scope": experiment["timing_scope"]},
        }
    assert result["stages"]["joint"]["weight"]["metadata"]["metadata"].get("parent_checkpoint_sha256") in (
        None, result["stages"]["sft"]["weight"]["actual_file_sha256"])
    assert result["stages"]["dpo"]["weight"]["metadata"]["metadata"].get("parent_checkpoint_sha256") in (
        None, result["stages"]["joint"]["weight"]["actual_file_sha256"])
    pc = torch.tensor([0.0], requires_grad=True)
    pr = torch.tensor([0.0], requires_grad=True)
    rc = torch.tensor([0.0], requires_grad=True)
    rr = torch.tensor([0.0], requires_grad=True)
    loss = dpo_loss(pc, pr, rc, rr, beta=.1)
    loss.backward()
    assert math.isclose(loss.item(), math.log(2), rel_tol=1e-6)
    assert math.isclose(pc.grad.item(), -.05, rel_tol=1e-6)
    assert math.isclose(pr.grad.item(), .05, rel_tol=1e-6)
    assert rc.grad is None and rr.grad is None
    result["dpo_formula_check"] = {"relative_margin": 0, "beta": .1, "loss": loss.item(),
                                  "hand_expected": "-ln sigmoid(0)=ln 2; gradients (-.05,+.05)",
                                  "policy_gradients": [pc.grad.item(), pr.grad.item()], "reference_gradients": None}
    joint = result["stages"]["joint"]["cpu_replay"]["records"]
    dpo = result["stages"]["dpo"]["cpu_replay"]["records"]
    regressions = [dict(id=a["id"], task=a["task"], joint=a["raw"], dpo=b["raw"], expected=a["expected_action"])
                   for a, b in zip(joint, dpo, strict=True) if a["end_to_end_correct"] and not b["end_to_end_correct"]]
    assert len(regressions) == 4 and all(r["task"] == "joint" for r in regressions)
    result["dpo_regressions"] = regressions
    example = next(r for r in result["stages"]["sft"]["cpu_replay"]["records"] if r["id"] == "691c9de656c01fa2df60")
    assert example["user"] == "照抄數字15，只要答案。" and example["system"] == "計算器=開；風格=短。" and example["raw"] == "DIRECT:15"
    result["named_example"] = copy.deepcopy(example)
    result["elapsed_seconds"] = time.perf_counter() - started
    result["result"] = "PASS: all 252 saved records independently audited and exactly replayed; counts, EOS, tensors, target denominators and scoped limitations agree"
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(result["lesson_stdout"])
    print(json.dumps({"result": result["result"], "environment": result["environment"],
                      "stages": {s: {"count": d["cpu_replay_summary"]["count"], "correct": d["cpu_replay_summary"]["end_to_end_correct"],
                                      "text_cohort": d["text_cohort"], "targets": d["training_configuration"]["schedule"]["ce_effective_targets"]}
                                 for s, d in result["stages"].items()}, "output": str(OUTPUT.relative_to(ROOT))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
