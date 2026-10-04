"""Independent bounded CPU probes and complete fixed-record arithmetic for 13.15."""
import contextlib
import hashlib
import io
import json
import math
import platform
import re
from pathlib import Path

import torch

from scripts.course_experiments.posttraining import CONFIG, build_records, split_records
from tiny_perceptron.posttraining import FiniteResponsePolicy, FiniteRewardModel, FiniteValueModel

ROOT = Path(__file__).resolve().parents[5]
ARTIFACT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


raw = (ARTIFACT / "initial-section.md").read_text()
snippet = re.search(r"```python\n(.*?)```", raw, re.S).group(1)
probes = []
for explanation in [0.0, 1.0]:
    code = snippet.replace("[[0.1, 0.2, 0.0, 0.0]]", f"[[0.1, 0.2, {explanation}, 0.0]]")
    code = code.replace("optimizer.zero_grad()", "all_before = {name: p.detach().clone() for name, p in policy.named_parameters()}\ncritic_before = {name: p.detach().clone() for name, p in critic.named_parameters()}\nreward_before = {name: p.detach().clone() for name, p in reward_model.named_parameters()}\nreference_before = {name: p.detach().clone() for name, p in reference.named_parameters()}\noptimizer.zero_grad()")
    namespace = {}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(code, "chapter-13.15-exact-snippet-with-observation-only", "exec"), namespace)
    policy = namespace["policy"]
    errors = {name: float((p.detach() - (namespace["all_before"][name] - 0.1 * p.grad)).abs().max()) for name, p in policy.named_parameters()}
    critic_errors = {name: float((p.detach() - (namespace["critic_before"][name] - 0.1 * p.grad)).abs().max()) for name, p in namespace["critic"].named_parameters()}
    probes.append({
        "explanation_feature": explanation,
        "stdout": output.getvalue(),
        "action": namespace["action"].item(),
        "ratio": namespace["terms"]["ratio"].item(),
        "context_shape": list(namespace["context"].shape),
        "logits_shape": list(namespace["new_logits"].shape),
        "advantage_requires_grad": namespace["advantage"].requires_grad,
        "reward_gradients": any(p.grad is not None for p in namespace["reward_model"].parameters()),
        "reference_gradients": any(p.grad is not None for p in namespace["reference"].parameters()),
        "reward_parameters_unchanged": all(torch.equal(p, namespace["reward_before"][name]) for name, p in namespace["reward_model"].named_parameters()),
        "reference_parameters_unchanged": all(torch.equal(p, namespace["reference_before"][name]) for name, p in namespace["reference"].named_parameters()),
        "sgd_policy_parameter_max_error": errors,
        "sgd_critic_parameter_max_error": critic_errors,
        "frozen_reference_kl_before_update": namespace["exact_kl"](namespace["new_logits"], namespace["reference"](namespace["context"])).item(),
    })
    assert probes[-1]["ratio"] == 1.0
    assert probes[-1]["reward_parameters_unchanged"] and probes[-1]["reference_parameters_unchanged"]
    assert max(errors.values()) < 3e-8
    assert max(critic_errors.values()) < 3e-8

logits = torch.tensor([[0.2, -0.4, 1.1, 0.6]], dtype=torch.float64, requires_grad=True)
target = torch.tensor([[0.85, 0.05, 0.05, 0.05]], dtype=torch.float64)
smoothed_loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0]), label_smoothing=0.2)
manual_loss = -(target * logits.log_softmax(-1)).sum()
smoothed_loss.backward()
smooth_loss_error = abs(smoothed_loss.item() - manual_loss.item())
smooth_gradient_error = float((logits.grad - (logits.detach().softmax(-1) - target)).abs().max())
assert smooth_loss_error < 1e-14 and smooth_gradient_error < 1e-14

record_path = ROOT / "docs/course-experiments/results/posttraining.json"
record = json.loads(record_path.read_text())
result = record["results"]
splits = split_records(build_records(), 42)
split_summary = {}
families = {}
for split, rows in splits.items():
    expected_record_rows = result["evaluations"][split]["rows"]
    assert len(rows) == len(expected_record_rows)
    owners = sorted({row["family"] for row in rows})
    families[split] = set(owners)
    assert owners == result["splits"][split]["families"]
    expected_hash = hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    assert expected_hash == result["splits"][split]["sha256"]
    pairs = sum(len(row["preference_pairs"]) for row in rows)
    assert pairs == result["splits"][split]["preference_pairs"]
    successes = {}
    for policy in ["sft", "ppo", "dpo"]:
        by_mode = {mode: 0 for mode in ["number", "explain", "missing"]}
        total = 0
        for base, row in zip(rows, expected_record_rows):
            for field, value in base.items():
                assert row[field] == value
            pred = row["policies"][policy]
            probs = pred["probabilities"]
            assert len(probs) == 4 and abs(sum(probs) - 1) < 2e-7
            chosen = max(range(4), key=probs.__getitem__)
            success = chosen == {"number": 0, "explain": 1, "missing": 3}[row["mode"]]
            assert chosen == pred["chosen_action"]
            assert success == pred["full_request_success"]
            assert row["candidates"][chosen] == pred["chosen_response"]
            total += success
            by_mode[row["mode"]] += success
        metric = result["evaluations"][split]["policies"][policy]["greedy_full_request_success"]
        assert metric["numerator"] == total and metric["denominator"] == len(rows)
        assert abs(metric["rate"] - total / len(rows)) < 4e-8
        successes[policy] = {"numerator": total, "denominator": len(rows), "by_mode": by_mode}
    split_summary[split] = {"families": owners, "family_count": len(owners), "contexts": len(rows), "preference_pairs": pairs, "record_sha256_recomputed": expected_hash, "successes": successes}
assert not (families["train"] & families["validation"] or families["train"] & families["test"] or families["validation"] & families["test"])
assert len(set.union(*families.values())) == 55
assert result["config"] == CONFIG
parameter_counts = {name: sum(p.numel() for p in model.parameters()) for name, model in [("policy", FiniteResponsePolicy()), ("reward_model", FiniteRewardModel()), ("value_model", FiniteValueModel())]}
assert parameter_counts == result["parameters"]
assert result["reward"]["processed_pair_draws"] == 300 * 64 == 19200
assert result["ppo"]["sampled_actions"] == 120 * 64 == 7680
assert result["ppo"]["reused_action_draws"] == 120 * 64 * 3 == 23040
assert result["ppo"]["policy_updates"] == result["ppo"]["value_updates"] == 120 * 3 == 360
assert result["sft"]["demonstrations"] == sum(row["mode"] == "number" for row in splits["train"]) == 44
assert result["effective_tokens"] == 0
assert result["reference_state_sha256_before"] == result["reference_state_sha256_after"] == result["ppo"]["initial_state_sha256"]
assert all(sha(ROOT / file) == digest for file, digest in record["code_sha256"].items())
trace = result["ppo"]["first_rollout_trace"]
for epoch, entry in enumerate(trace, 1):
    assert entry["epoch"] == epoch
    assert len(entry["sampled_actions"]) == len(entry["train_context_indices"]) == 64
    for i in range(64):
        advantage = entry["normalized_rm_rewards"][i] - entry["old_values"][i]
        ratio = math.exp(entry["log_probabilities_before_update"][i] - entry["old_selected_log_probabilities"][i])
        assert abs(advantage - entry["fixed_advantages"][i]) < 2e-7
        assert abs(ratio - entry["ratios_before_update"][i]) < 2e-7
    for fixed in ["sampled_actions", "train_context_indices", "old_selected_log_probabilities", "old_values", "normalized_rm_rewards", "fixed_advantages", "old_log_probability_sha256", "reference_state_sha256"]:
        assert entry[fixed] == trace[0][fixed]

summary = {
    "reviewer_task": "/root/v4_review_coordinator/factual_v4_13_15",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "torch_git_version": torch.version.git_version, "device": "cpu", "initial_threads": torch.get_num_threads()},
    "exact_snippet_probes": probes,
    "label_smoothing_execution": {"dtype": "float64", "logits": logits.detach().tolist(), "target": target.tolist(), "torch_cross_entropy": smoothed_loss.item(), "manual_target_cross_entropy": manual_loss.item(), "loss_absolute_error": smooth_loss_error, "gradient_absolute_error": smooth_gradient_error},
    "fixed_record_sha256": sha(record_path),
    "full_fixed_record_inspection": "Parsed every row in all three splits; regenerated every original record, checked every candidate/feature/label, recomputed every policy argmax and success, and checked all 192 first-rollout ratio/advantage observations. This is record validation, not replication of the author's timing.",
    "config": result["config"],
    "parameter_counts": parameter_counts,
    "splits": split_summary,
    "accounting": {"sft_draws": 60 * 32, "reward_pair_draws": 300 * 64, "ppo_new_actions": 120 * 64, "ppo_reused_actions": 120 * 64 * 3, "policy_updates": 120 * 3, "value_updates": 120 * 3, "effective_tokens": result["effective_tokens"]},
    "timing_from_original_record_only": {"experiment_total_seconds": record["elapsed_seconds"], "ppo_seconds": result["ppo"]["seconds"], "cpu_threads": result["config"]["cpu_threads"], "scope": "Original main() timer includes fixed training, evaluation and checkpoint saves; imports/environment setup precede timer. Single CPU run; not a portable speed guarantee."},
    "arithmetic": {"families": "10+9+8+7+6+5+4+3+2+1 = 10*11/2 = 55", "train_pairs": "44 families * (6 number + 6 explain + 3 missing comparisons) = 660", "label_smoothing_target": [0.8 + 0.2 / 4, 0.2 / 4, 0.2 / 4, 0.2 / 4], "policy_parameters": "4*16+16+16*4+4=148", "reward_parameters": "(4+4)*24+24+24*1+1=241", "value_parameters": "4*16+16+16*1+1=97"},
    "passed": True,
}
(ARTIFACT / "cpu-verification.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(summary, ensure_ascii=False, indent=2))
