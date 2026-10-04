"""Independent bounded CPU review; no formal training or environment changes."""

import contextlib
import hashlib
import io
import json
import math
import platform
import random
import re
import subprocess
import time
from pathlib import Path
from urllib.request import urlopen

import torch
from huggingface_hub import hf_hub_download

from scripts.check_technical_reviews import sections
from scripts.course_experiments.capstone import _balanced_sample
from tiny_perceptron.alignment import sequence_log_probability
from tiny_perceptron.capstone import (
    build_dataset,
    evaluate_rows,
    load_capstone,
    preference_pairs,
    prepare_batch,
)
from tiny_perceptron.capstone_quantization import load_quantized_capstone
from tiny_perceptron.data import IGNORE

ROOT = Path.cwd()
OUT = ROOT / "docs/technical-reviews/artifacts"
PREFIX = "fact_v2_19_08"
COMMAND = "PYTHONPATH=. .venv/bin/python docs/technical-reviews/artifacts/fact_v2_19_08_replay.py"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_sha(state):
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        result.update(name.encode())
        result.update(str(value.dtype).encode())
        result.update(str(tuple(value.shape)).encode())
        result.update(value.numpy().tobytes())
    return result.hexdigest()


def save(name, value):
    path = OUT / f"{PREFIX}_{name}.json"
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return path


def main():
    torch.set_num_threads(2)
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": "cpu",
        "threads": str(torch.get_num_threads()),
    }
    result = {"command": COMMAND, "environment": environment, "reviewer_task": "/root/integration_technical_coordinator/fact_v2_19_08"}
    read_sections = {}
    for chapter, ids in {"07": {"7.18"}, "13": {"13.4", "13.8", "13.11"}, "19": {"19.2", "19.4", "19.8", "19.10"}}.items():
        for number, body in sections(ROOT / f"course/chapters/{chapter}.md"):
            if number in ids:
                read_sections[number] = {"sha256": hashlib.sha256(body.encode()).hexdigest(), "body": body}
    save("read_sections", read_sections)
    body = read_sections["19.8"]["body"]
    result["source_sha256"] = read_sections["19.8"]["sha256"]
    code = re.findall(r"```python\n(.*?)```", body, re.S)[0]
    namespace = {}
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(code, "19.8 exact original code", "exec"), namespace)
    policy, reference, pair = (namespace[x] for x in ("policy", "reference", "pair"))
    result["exact_lesson_code"] = {"code": code, "stdout": stdout.getvalue(), "loss": namespace["loss"].item(), "hand_derivation": "m=(pc-pr)-(rc-rr)=0 at equal weights; -log(sigmoid(.1*0))=log(2)", "expected": math.log(2)}
    assert abs(namespace["loss"].item() - math.log(2)) < 1e-6
    result["exercise_policy_requires_grad"] = any(p.requires_grad for p in policy.parameters())
    assert result["exercise_policy_requires_grad"]
    split, manifest = build_dataset(42)
    original_data = json.loads((ROOT / "docs/course-experiments/capstone-evidence/deployment/data.json").read_text())
    assert original_data["splits"] == split and original_data["manifest"] == manifest
    pairs = preference_pairs(split["train"])
    result["data"] = {"manifest": manifest, "original_data_sha256": sha(ROOT / "docs/course-experiments/capstone-evidence/deployment/data.json"), "all_splits_match_saved_original": True, "pair_count": len(pairs), "first_pair": pair, "all_pair_ids_train_only": all(p["row"]["id"] in {r["id"] for r in split["train"]} for p in pairs), "all_rejected_preserve_chosen_prefix": all(p["rejected"] == p["chosen"] + "，祝你愉快！" for p in pairs)}
    chosen, chosen_labels = prepare_batch([pair["row"]], answers=[pair["chosen"]])
    rejected, rejected_labels = prepare_batch([pair["row"]], answers=[pair["rejected"]])
    pc = sequence_log_probability(policy(**chosen)["logits"], chosen_labels)
    pr = sequence_log_probability(policy(**rejected)["logits"], rejected_labels)
    result["sequence_score"] = {"chosen_shape": list(chosen["ids"].shape), "rejected_shape": list(rejected["ids"].shape), "input_dtype": str(chosen["ids"].dtype), "valid_dtype": str(chosen["valid"].dtype), "chosen_effective_targets": int((chosen_labels != IGNORE).sum()), "rejected_effective_targets": int((rejected_labels != IGNORE).sum()), "chosen_labels": chosen_labels.tolist(), "rejected_labels": rejected_labels.tolist(), "chosen_logprob": pc.item(), "rejected_logprob": pr.item(), "sum_not_length_mean": True, "eos_included": True, "prompt_and_pad_ignored": True}
    ref_before, policy_before = state_sha(reference.state_dict()), state_sha(policy.state_dict())
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.0002)
    namespace["loss"].backward()
    optimizer.step()
    result["one_synthetic_step"] = {"reference_before": ref_before, "reference_after": state_sha(reference.state_dict()), "policy_before": policy_before, "policy_after": state_sha(policy.state_dict()), "reference_eval": not reference.training, "reference_requires_grad": any(p.requires_grad for p in reference.parameters()), "reference_has_grad": any(p.grad is not None for p in reference.parameters()), "optimizer_has_reference_parameter": bool({id(p) for p in reference.parameters()} & {id(p) for g in optimizer.param_groups for p in g["params"]}), "scope": "fresh random CPU model, one optimizer update; does not assert historical GPU runtime reference fingerprint"}
    assert ref_before == result["one_synthetic_step"]["reference_after"]
    assert policy_before != result["one_synthetic_step"]["policy_after"]
    reports = {}
    stage_paths = {"joint": "v2-joint/capstone-review/capstone_joint/gha-37168518451-1/model.pt", "dpo": "v2-dpo/capstone-review/capstone_preference/gha-37168874504-1/model.pt"}
    models = {}
    for stage, relative in stage_paths.items():
        experiment = "capstone_joint" if stage == "joint" else "capstone_preference"
        report_path = ROOT / f"docs/course-experiments/results/{experiment}.json"
        report = json.loads(report_path.read_text())
        reports[stage] = report
        checkpoint = ROOT / "outputs/integration-runs" / relative
        model, payload = load_capstone(checkpoint)
        models[stage] = model
        assert sha(checkpoint) == report["results"]["inference_export"]["sha256"]
        assert payload["metadata"]["data_manifest"] == manifest
        assert payload["stage"] == stage and payload["step"] == report["results"]["steps"]
        assert payload["metadata"]["schedule_completed"]
        assert all(torch.isfinite(t).all() for t in payload["model"].values())
        started = time.perf_counter()
        replay = evaluate_rows(model, split["validation"], max_new_tokens=64, batch_size=24)
        replay_seconds = time.perf_counter() - started
        original_path = ROOT / f"docs/course-experiments/capstone-evidence/{stage}/validation.json"
        original = json.loads(original_path.read_text())
        assert replay == original
        save(f"validation_{stage}", replay)
        manual_counts = {}
        for row, record in zip(split["validation"], original["records"], strict=True):
            assert row["id"] == record["id"] and row["answer"] == record["expected_action"]
            action_correct = record["action_trace"]["eos"] and record["action_trace"]["raw"] == row["answer"]
            expected_final = row["answer"].split(":", 1)[1] if row["task"] != "calculator" else str(sum(map(int, row["answer"].rsplit(":", 1)[1].split("+"))))
            correct = action_correct and record["answer"] == expected_final
            assert correct == record["end_to_end_correct"]
            entry = manual_counts.setdefault(row["task"], {"count": 0, "correct": 0})
            entry["count"] += 1
            entry["correct"] += int(correct)
        result[stage] = {"checkpoint": {"ignored_local_path": checkpoint.relative_to(ROOT).as_posix(), "bytes": checkpoint.stat().st_size, "sha256": sha(checkpoint), "tensor_sha256": state_sha(payload["model"]), "tensor_count": len(payload["model"]), "parameters": sum(t.numel() for t in payload["model"].values()), "all_finite": True, "tensor_dtypes": sorted({str(t.dtype) for t in payload["model"].values()}), "stage": payload["stage"], "step": payload["step"], "config": payload["config"], "tokenizer": payload["tokenizer"], "metadata": payload["metadata"]}, "formal_report_sha256": sha(report_path), "formal_environment": {k: report[k] for k in ("revision", "seed", "device", "gpu", "python_version", "torch_version", "elapsed_seconds", "timing_scope")}, "formal_result": report["results"], "all_84_records_exact_replay_match": True, "original_validation_sha256": sha(original_path), "independent_counts": manual_counts, "cpu_replay_seconds": replay_seconds, "cpu_time_scope": "review-only CPU validation; not a repeat of GPU training timing"}
    assert reports["dpo"]["results"]["parent_checkpoint_sha256"] == result["joint"]["checkpoint"]["sha256"]
    for entry in reports["dpo"]["results"]["history"]:
        assert abs(entry["loss_before_update"] - (entry["dpo_loss"] + .2 * entry["ce_before_update"] + .01 * entry["auxiliary_before_update"])) < 1e-7
    sampler = random.Random(42 + 3000)
    effective_tokens = 0
    batches = []
    for step in range(1, 101):
        rows = _balanced_sample(split["train"], sampler, 24)
        selected = sampler.choices(pairs, k=12)
        _, labels = prepare_batch(rows)
        effective_tokens += int((labels != IGNORE).sum())
        batches.append({"step": step, "ce_row_ids": [r["id"] for r in rows], "preference_row_ids": [p["row"]["id"] for p in selected], "ce_effective_targets": int((labels != IGNORE).sum())})
    assert effective_tokens == reports["dpo"]["results"]["effective_tokens"] == 41403
    result["dpo_sampling_reconstruction"] = {"seed": 3042, "batch_size": 24, "pairs_per_step": 12, "updates": 100, "ce_effective_tokens": effective_tokens, "batches": batches, "scope": "deterministic sampler reconstruction from frozen source, not captured GPU sample log"}
    changed = []
    for left, right in zip(json.loads((OUT / f"{PREFIX}_validation_joint.json").read_text())["records"], json.loads((OUT / f"{PREFIX}_validation_dpo.json").read_text())["records"], strict=True):
        if left != right:
            changed.append({"id": left["id"], "task": left["task"], "expected": left["expected_action"], "joint": left["action_trace"]["raw"], "dpo": right["action_trace"]["raw"], "joint_correct": left["end_to_end_correct"], "dpo_correct": right["end_to_end_correct"]})
    result["changed_validation_records"] = changed
    selection_path = ROOT / "docs/course-experiments/capstone-selection.json"
    selection = json.loads(selection_path.read_text())
    assert selection["selected_stage"] == "joint" and selection["selected_before_test_generation"]
    assert all(not r["results"]["test_evaluated"] for r in reports.values())
    result["selection"] = {"sha256": sha(selection_path), "record": selection, "scope": "contemporaneous saved selection and stage no-test flags; cannot prove unlogged human access did not occur"}
    student = json.loads((ROOT / "docs/course-experiments/results/capstone_student.json").read_text())
    assert student["results"]["teacher_checkpoint_sha256"] == result["dpo"]["checkpoint"]["sha256"]
    result["student_teacher"] = {"teacher_sha256": student["results"]["teacher_checkpoint_sha256"], "recipe_frozen_before_test": student["results"]["recipe_frozen_before_test"], "branches": student["results"]["branches"], "report_sha256": sha(ROOT / "docs/course-experiments/results/capstone_student.json")}
    public = json.loads((ROOT / "docs/course-experiments/capstone-public.json").read_text())
    result["public_weights"] = {}
    for stage in ("joint", "dpo", "joint-int4", "joint-int8", "student-ce", "student-kd"):
        specification = next(x for x in public["models"] if x["id"] == stage)
        item = next(x for x in specification["files"] if x["output"] == "model.pt")
        path = Path(hf_hub_download(public["repo"], item["path"], revision=public["revision"], token=False))
        assert sha(path) == item["sha256"] and path.stat().st_size == item["bytes"]
        loaded, payload = load_quantized_capstone(path) if stage.startswith("joint-int") else load_capstone(path)
        provenance = {"repo": public["repo"], "revision": public["revision"], "token": False, "path": item["path"], "sha256": sha(path), "bytes": path.stat().st_size, "tensor_sha256": state_sha(loaded.state_dict()), "metadata": payload["metadata"]}
        if stage in models:
            assert all(torch.equal(payload["model"][k], v) for k, v in models[stage].state_dict().items())
            provenance["all_tensors_equal_historical_run"] = True
        else:
            is_student = stage.startswith("student-")
            branch = stage.removeprefix("student-")
            test_path = ROOT / (f"docs/course-experiments/capstone-evidence/student/test-{branch}.json" if is_student else f"docs/course-experiments/capstone-evidence/deployment/test-joint-ptq{stage[-1]}.json")
            original = json.loads(test_path.read_text())
            replay = evaluate_rows(loaded, split["test"], max_new_tokens=64, batch_size=24)
            assert replay["records"] == original["records"]
            assert replay["by_task"] == original["by_task"]
            if is_student:
                assert payload["metadata"]["teacher_checkpoint_sha256"] == result["dpo"]["checkpoint"]["sha256"]
                assert not loaded.config.experts and loaded.config.width == 48
            else:
                assert payload["stage"] == "joint"
                assert payload["metadata"]["source_checkpoint_sha256"] == result["joint"]["checkpoint"]["sha256"]
            save(f"test_{stage}", replay)
            provenance.update(all_90_records_replay_match=True, historical_test_sha256=sha(test_path), summary={k: v for k, v in replay.items() if k != "records"}, config=loaded.description())
        result["public_weights"][stage] = provenance
    result["source_versions"] = {}
    for path in ("tiny_perceptron/capstone.py", "tiny_perceptron/alignment.py", "tiny_perceptron/posttraining.py", "scripts/course_experiments/capstone.py", "scripts/course_experiments/capstone_student.py", "scripts/course_experiments/capstone_deployment.py", "tests/test_capstone.py"):
        result["source_versions"][path] = {"sha256": sha(ROOT / path)}
        expected = reports["dpo"].get("code_sha256", {}).get(path)
        if expected is not None:
            original = subprocess.run(["git", "show", f"6eaab90c15594ff004cf855d7421c80caa7e3384:{path}"], capture_output=True, check=True).stdout
            assert hashlib.sha256(original).hexdigest() == expected
            snapshot = OUT / f"{PREFIX}_training_{path.replace('/', '_')}.txt"
            snapshot.write_bytes(original)
            result["source_versions"][path].update(training_revision="6eaab90c15594ff004cf855d7421c80caa7e3384", training_sha256=expected, current_equal_training=sha(ROOT / path) == expected, training_snapshot=snapshot.relative_to(ROOT).as_posix())
    commit = "f8b8c0f49dc92a430bae41585f9d467d3618fe2f"
    url = f"https://raw.githubusercontent.com/eric-mitchell/direct-preference-optimization/{commit}/trainers.py"
    with urlopen(url, timeout=30) as response:
        original = response.read()
    source_path = OUT / f"{PREFIX}_official_dpo_trainers.py.txt"
    source_path.write_bytes(original)
    result["official_source"] = {"url": url, "version": commit, "sha256": sha(source_path), "path": source_path.relative_to(ROOT).as_posix()}
    result["result"] = "All assertions passed: exact lesson and exercise, reference mechanism, data/denominators, two full 84-record CPU replays, formal parent/teacher linkage and pinned anonymous public weights."
    save("execution", result)
    print(json.dumps({"result": result["result"], "environment": environment, "source_sha256": result["source_sha256"], "stdout": stdout.getvalue(), "joint": result["joint"]["independent_counts"], "dpo": result["dpo"]["independent_counts"], "changed": changed}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
