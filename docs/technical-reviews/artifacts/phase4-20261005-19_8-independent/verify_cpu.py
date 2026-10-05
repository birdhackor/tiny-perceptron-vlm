"""Fresh 19.8 checks: one local update, never score or train a full model."""
import hashlib
import json
import math
import platform
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch

from tiny_perceptron.alignment import dpo_loss, sequence_log_probability
from tiny_perceptron.capstone import (
    CapstoneModel, build_dataset, frozen_reference, preference_loss, preference_pairs,
)
from tiny_perceptron.data import IGNORE

HERE = Path(__file__).resolve().parent
BASE = HERE / "sources"


def read(relative):
    return json.loads((BASE / relative).read_bytes())


def tensor_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


torch.set_num_threads(1)
torch.set_default_device("cpu")
assert torch.version.cuda is None and not torch.cuda.is_available()
torch.manual_seed(42)
splits, manifest = build_dataset()
pairs = preference_pairs(splits["train"])
first = pairs[0]
assert first["row"]["user"] == "原題：3+6。計算器回報：9。請回答。"
assert first["chosen"] == "DIRECT:9"
assert first["rejected"] == "DIRECT:9，祝你愉快！"
assert all(p["rejected"] == p["chosen"] + "，祝你愉快！" for p in pairs)

policy = CapstoneModel()
reference = frozen_reference(policy)
before_policy, before_reference = tensor_digest(policy), tensor_digest(reference)
assert before_policy == before_reference
assert not reference.training and not any(p.requires_grad for p in reference.parameters())
loss, details = preference_loss(policy, reference, [first])
assert abs(loss.item() - math.log(2)) < 1e-6
assert abs(details["policy_margin"] - details["reference_margin"]) < 1e-6
assert before_policy == tensor_digest(policy) and before_reference == tensor_digest(reference)
initial = {"loss": loss.item(), "policy_margin": details["policy_margin"],
           "reference_margin": details["reference_margin"], "weights_unchanged": True}

# A necessary small variation verifies gradient/update ownership, not task scores.
optimizer = torch.optim.AdamW(policy.parameters(), lr=0.0002)
optimizer.zero_grad(set_to_none=True)
loss.backward()
policy_gradients = sum(p.grad is not None for p in policy.parameters())
reference_gradients = sum(p.grad is not None for p in reference.parameters())
assert policy_gradients > 0 and reference_gradients == 0
optimizer.step()
assert tensor_digest(policy) != before_policy
assert tensor_digest(reference) == before_reference
reference.requires_grad_(True)
try:
    preference_loss(policy, reference, [first])
except ValueError as error:
    assert "frozen" in str(error)
else:
    raise AssertionError("Unfrozen reference was not rejected")

# Check response-token sum and vocabulary axis independently of model outputs.
logits = torch.tensor([[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]])
labels = torch.tensor([[0, IGNORE, 2]])
sequence = sequence_log_probability(logits, labels).item()
assert abs(sequence - (-2 * math.log(3))) < 1e-6
numeric = []
for beta in (0.1, 0.2, 1.0):
    chosen = torch.tensor([2.], requires_grad=True)
    rejected = torch.tensor([0.], requires_grad=True)
    observed = dpo_loss(chosen, rejected, torch.tensor([0.]), torch.tensor([0.]), beta)
    expected = math.log1p(math.exp(-2 * beta))
    assert abs(observed.item() - expected) < 1e-6
    observed.backward()
    assert chosen.grad.item() < 0 < rejected.grad.item()
    numeric.append({"beta": beta, "expected": expected, "observed": observed.item(),
                    "chosen_gradient": chosen.grad.item(), "rejected_gradient": rejected.grad.item()})

# Recompute existing measurements only; no checkpoint load or model generation.
evaluations = {}
records_by_stage = {}
for stage in ("joint", "dpo"):
    result = read(f"docs/course-experiments/capstone-evidence/{stage}/validation.json")
    records = result["records"]
    aggregate = defaultdict(lambda: {"count": 0, "action_correct": 0, "end_to_end_correct": 0})
    assert len(records) == len({r["id"] for r in records}) == 84
    assert {r["id"] for r in records} == {r["id"] for r in splits["validation"]}
    for record in records:
        action = record["action_trace"]["eos"] and record["action_trace"]["raw"] == record["expected_action"]
        final = action and record["answer"] == record["expected_final"]
        assert record["action_correct"] == action
        assert record["end_to_end_correct"] == final
        total = aggregate[record["task"]]
        total["count"] += 1
        total["action_correct"] += int(action)
        total["end_to_end_correct"] += int(final)
    recomputed = {"count": len(records),
                  "action_correct": sum(t["action_correct"] for t in aggregate.values()),
                  "end_to_end_correct": sum(t["end_to_end_correct"] for t in aggregate.values()),
                  "by_task": dict(aggregate)}
    for key, value in recomputed.items():
        assert result[key] == value
    summary_name = "preference" if stage == "dpo" else stage
    summary = read(f"docs/course-experiments/results/capstone_{summary_name}.json")["results"]["validation_summary"]
    for key, value in recomputed.items():
        assert summary[key] == value
    evaluations[stage] = recomputed
    records_by_stage[stage] = {r["id"]: r for r in records}
assert evaluations["joint"]["end_to_end_correct"] == 75
assert evaluations["dpo"]["end_to_end_correct"] == 71
assert evaluations["joint"]["by_task"]["style"] == evaluations["dpo"]["by_task"]["style"] == {
    "count": 3, "action_correct": 3, "end_to_end_correct": 3}
assert evaluations["joint"]["by_task"]["joint"]["end_to_end_correct"] == 18
assert evaluations["dpo"]["by_task"]["joint"]["end_to_end_correct"] == 14
changed = []
for identifier, joint in records_by_stage["joint"].items():
    dpo = records_by_stage["dpo"][identifier]
    assert joint["task"] == dpo["task"] and joint["expected_action"] == dpo["expected_action"]
    if joint["action_trace"]["raw"] != dpo["action_trace"]["raw"]:
        changed.append({"id": identifier, "task": joint["task"], "expected": joint["expected_action"],
                        "joint": joint["action_trace"]["raw"], "dpo": dpo["action_trace"]["raw"]})
assert len(changed) == 4 and all(r["task"] == "joint" for r in changed)

train = read("docs/course-experiments/capstone-evidence/dpo/train-report.json")
assert train["requested_steps"] == train["steps"] == train["new_steps"] == 100
assert train["schedule_completed"] and train["test_evaluated"] is False
history = train["history"]
assert [r["step"] for r in history] == [1,10,20,30,40,50,60,70,80,90,100]
objective_residuals = []
for row in history:
    expected = row["dpo_loss"] + 0.2 * row["ce_before_update"] + 0.01 * row["auxiliary_before_update"]
    residual = abs(row["loss_before_update"] - expected)
    assert residual < 1e-7
    objective_residuals.append(residual)
assert abs(history[0]["dpo_loss"] - math.log(2)) < 1e-6
assert abs(history[-1]["dpo_loss"] - 0.00004602140688803047) < 1e-14
assert history[0]["reference_margin"] != history[-1]["reference_margin"]

# Reproduce only the documented sampling schedule, without model execution.
sampler = random.Random(42 + 3 * 1000)
task_rows = {}
for row in splits["train"]:
    task_rows.setdefault(row["task"], []).append(row)
task_names = sorted(task_rows)
sampled = {}
for step in range(1, 101):
    _ = [sampler.choice(task_rows[sampler.choice(task_names)]) for _ in range(24)]
    selected = sampler.choices(pairs, k=12)
    if step in (1, 100):
        sampled[step] = [pair["row"]["id"] for pair in selected]
assert Counter(sampled[1]) != Counter(sampled[100])

joint = read("docs/course-experiments/results/capstone_joint.json")["results"]
dpo = read("docs/course-experiments/results/capstone_preference.json")["results"]
selection = read("docs/course-experiments/capstone-selection.json")
assert selection["selected_stage"] == "joint" and selection["selected_before_test_generation"] is True
assert joint["test_evaluated"] is False and dpo["test_evaluated"] is False
assert train["parent_checkpoint_sha256"] == joint["inference_export"]["sha256"]
for result in (joint, dpo, train):
    assert result["data_manifest"]["version"] == manifest["version"]
    assert result["data_manifest"]["sha256"] == manifest["sha256"]
    assert result["data_manifest"]["counts"] == {"train": 552, "validation": 84, "test": 90}
    for path in ("tiny_perceptron/capstone.py", "scripts/course_experiments/capstone.py"):
        assert result["code_sha256"][path] == hashlib.sha256((BASE / path).read_bytes()).hexdigest()
student = read("docs/course-experiments/results/capstone_student.json")["results"]
assert student["teacher_checkpoint_sha256"] == dpo["inference_export"]["sha256"]
assert student["teacher_checkpoint_sha256"] != joint["inference_export"]["sha256"]
for mode in ("ce", "kd"):
    branch = student["branches"][mode]
    assert branch["steps"] == branch["requested_steps"] == student["steps_per_branch"] == 350
    assert branch["schedule_completed"]
    assert branch["teacher_checkpoint_sha256"] == student["teacher_checkpoint_sha256"]
deployment = read("docs/course-experiments/results/capstone_deployment.json")["results"]
assert deployment["recommended_stage"] == "joint"
ptq = {}
for identity in ("joint-int4", "joint-int8"):
    row = deployment["joint_ptq"][identity]
    assert row["diagnostic_stage"] == "joint"
    assert row["source_checkpoint_sha256"] == joint["inference_export"]["sha256"]
    ptq[identity] = {key: row[key] for key in ("count", "diagnostic_stage", "source_checkpoint_sha256")}

results = {
    "scope": "Exact original fence separate; one local CPU preference update only; original JSON recomputation, no weights loaded or generated model score.",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                    "torch_git_version": str(torch.version.git_version), "cuda_build": str(torch.version.cuda),
                    "cuda_available": str(torch.cuda.is_available()), "device": "cpu"},
    "first_pair": {"user": first["row"]["user"], "chosen": first["chosen"], "rejected": first["rejected"]},
    "pair_count": len(pairs), "initial": initial,
    "small_update": {"policy_gradient_tensors": policy_gradients, "reference_gradient_tensors": reference_gradients,
                     "policy_changed": tensor_digest(policy) != before_policy,
                     "reference_unchanged": tensor_digest(reference) == before_reference,
                     "not_evidence_of_historical_GPU_fingerprint_comparison": True},
    "axes": {"logits": "batch,sequence,vocabulary", "token_reduction": "sum over unmasked answer tokens including EOS",
             "dpo_reduction": "mean over preference pairs", "sum_log_probability": sequence},
    "numeric_variations": numeric, "existing_validation": evaluations, "changed_records": changed,
    "existing_training": {"steps": train["steps"], "effective_tokens": train["effective_tokens"],
                          "history_first": history[0], "history_last": history[-1],
                          "sampling_only_first_last_pair_ids": sampled,
                          "preference_pairs_per_update": 12,
                          "replay_rows_per_update": 24,
                          "max_objective_residual": max(objective_residuals)},
    "provenance": {"selection_stage": selection["selected_stage"], "before_test_generation": selection["selected_before_test_generation"],
                   "joint_checkpoint": joint["inference_export"]["sha256"], "dpo_checkpoint": dpo["inference_export"]["sha256"],
                   "student_teacher": student["teacher_checkpoint_sha256"], "student_steps": student["steps_per_branch"], "joint_ptq": ptq},
}
(HERE / "verification.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(results, ensure_ascii=False, indent=2))
print("ALL ASSERTIONS PASSED")
