"""Independent, bounded 13.16 checks; complete finite experiment stays ignored."""
import contextlib
import hashlib
import io
import json
import math
import platform
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from scripts.course_experiments.common import Context
from scripts.course_experiments.posttraining import run_posttraining
from tiny_perceptron.posttraining import exact_kl, ppo_clipped_objective

OUT = Path(__file__).resolve().parent
RESEARCH = ROOT / "outputs/natural-v4/factual-research/13.16"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_rows(result):
    summaries = {}
    families_by_split = {}
    examples = []
    for split_name, split in result["evaluations"].items():
        rows = split["rows"]
        families_by_split[split_name] = {r["family"] for r in rows}
        mode_counts = Counter(r["mode"] for r in rows)
        wins = 0
        pair_count = 0
        failures_below_expected_reward = Counter()
        success = {p: Counter() for p in ["sft", "ppo", "dpo"]}
        for row in rows:
            a, b = row["operands"]
            assert 1 <= a <= b <= 10
            assert row["family"] == f"pair:{a}:{b}"
            mode = row["mode"]
            assert row["features"] == [a / 10, b / 10, float(mode == "explain"), float(mode == "missing")]
            if mode == "missing":
                expected = 3
                assert row["candidates"][3] == "請補充購買數量。"
                pairs = [(3, i) for i in [0, 1, 2]]
            else:
                assert row["candidates"] == [str(a + b), f"{a} 加 {b} 是 {a + b}。", str(a + b + 1), "請再提供更多資訊。"]
                expected = 0 if mode == "number" else 1
                ranking = [0, 1, 3, 2] if mode == "number" else [1, 0, 3, 2]
                pairs = [(ranking[i], ranking[j]) for i in range(4) for j in range(i + 1, 4)]
            assert row["expected_action"] == expected
            assert row["candidate_meets_full_request"] == [i == expected for i in range(4)]
            assert row["preference_pairs"] == [list(x) for x in pairs]
            scores = row["reward_model_raw_scores"]
            assert row["reward_model_best_action"] == max(range(4), key=lambda i: scores[i])
            wins += sum(scores[w] > scores[l] for w, l in pairs)
            pair_count += len(pairs)
            for policy_name, policy in row["policies"].items():
                probs = policy["probabilities"]
                assert len(probs) == 4 and all(0 <= p <= 1 for p in probs)
                assert abs(sum(probs) - 1) < 2e-7
                action = max(range(4), key=lambda i: probs[i])
                assert action == policy["chosen_action"]
                assert policy["chosen_response"] == row["candidates"][action]
                assert policy["full_request_success"] == (action == expected)
                success[policy_name][mode] += int(action == expected)
                if action != expected and scores[action] < scores[expected]:
                    failures_below_expected_reward[policy_name] += 1
            if split_name == "test" and row["family"] == "pair:1:2" and mode == "explain":
                examples.append({"candidates": row["candidates"], "rm_raw_scores": scores,
                                 "policy_probabilities": {p: row["policies"][p]["probabilities"] for p in success}})
        metrics = {p: {m: {"numerator": n[m], "denominator": mode_counts[m]} for m in mode_counts}
                   for p, n in success.items()}
        for p, by_mode in metrics.items():
            assert by_mode == {m: {k: split["policies"][p]["by_mode"][m][k] for k in ["numerator", "denominator"]} for m in mode_counts}
            total = split["policies"][p]["greedy_full_request_success"]
            assert total["numerator"] == sum(success[p].values()) and total["denominator"] == len(rows)
        rm = split["reward_model_pairwise_accuracy"]
        assert (wins, pair_count) == (rm["numerator"], rm["denominator"])
        summaries[split_name] = {"contexts": len(rows), "families": len(families_by_split[split_name]),
                                "mode_counts": dict(mode_counts), "policy_full_request_success": metrics,
                                "rm_pairwise": {"numerator": wins, "denominator": pair_count},
                                "failed_choices_with_lower_rm_score_than_requested_card": dict(failures_below_expected_reward)}
    names = list(families_by_split)
    assert all(families_by_split[names[i]].isdisjoint(families_by_split[names[j]]) for i in range(3) for j in range(i + 1, 3))
    assert len(set.union(*families_by_split.values())) == 55
    assert "pair:1:2" in families_by_split["test"]
    return {"splits": summaries, "test_1_plus_2_explain": examples, "family_split_disjoint": True}


report_path = ROOT / "docs/course-experiments/results/posttraining.json"
original = json.loads(report_path.read_text(encoding="utf-8"))
old = original["results"]
recomputed = inspect_rows(old)
code_paths = ["scripts/course_experiments/posttraining.py", "tiny_perceptron/posttraining.py", "tiny_perceptron/alignment.py"]
code_hashes = {p: sha(ROOT / p) for p in code_paths}
assert all(original["code_sha256"][p] == h for p, h in code_hashes.items())
assert old["sft"]["demonstrations"] == 44
assert old["sft"]["processed_demonstration_draws"] == 60 * 32 == 1920
assert old["reward"]["unique_training_pairs"] == 44 * (6 + 6 + 3) == 660
assert old["reward"]["processed_pair_draws"] == 300 * 64 == 19200
assert old["ppo"]["sampled_actions"] == 120 * 64 == 7680
assert old["ppo"]["reused_action_draws"] == 120 * 64 * 3 == 23040
assert old["ppo"]["policy_updates"] == old["ppo"]["value_updates"] == 120 * 3 == 360
assert old["dpo"]["processed_pair_draws"] == 360 * 64 == 23040
assert old["ppo"]["initial_state_sha256"] == old["dpo"]["initial_state_sha256"] == old["reference_state_sha256_before"] == old["reference_state_sha256_after"]

# Execute the exact original primary snippet; changed syntax prose cannot affect it.
primary = (OUT / "source-original.md").read_text(encoding="utf-8")
snippet = re.findall(r"```python\n(.*?)```", primary, re.S)[0]
buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    namespace = {}
    exec(compile(snippet, "13.16-original-snippet", "exec"), namespace)
assert namespace["scores"] == [1, 14] and namespace["chosen"] == 1
answers = namespace["answers"]
correct_scores = [int(answer == str(1 + 2)) for answer in answers]
correct_choice = max(range(len(answers)), key=lambda index: correct_scores[index])
assert correct_scores == [1, 0] and answers[correct_choice] == "3"

# A counterexample: clipping and an exact fixed-reference KL preserve bad reward incentives.
logits = torch.zeros((1, 2), dtype=torch.float64, requires_grad=True)
old_log = torch.full((2,), math.log(0.5), dtype=torch.float64)
advantage = torch.tensor([-6.5, 6.5], dtype=torch.float64)
terms = ppo_clipped_objective(logits.log_softmax(-1)[0], old_log, advantage)
loss = terms["policy_loss"] + 0.1 * exact_kl(logits, torch.zeros_like(logits)).mean()
loss.backward()
gradient = logits.grad.detach().tolist()[0]
new_probs = (logits.detach() - 0.1 * logits.grad).softmax(-1).tolist()[0]
assert all(abs(x-y) < 1e-12 for x,y in zip(gradient,[3.25,-3.25]))
assert new_probs[1] > 0.5

# Existing-source full short CPU training, no GPU, external dataset, model download, or Git query.
fresh_dir = RESEARCH / "cpu-run"
context = Context("cpu", fresh_dir, RESEARCH, ROOT / "assets/training", seed=42)
started = time.perf_counter()
fresh = run_posttraining(context)
elapsed = time.perf_counter() - started
fresh_summary = inspect_rows(fresh)
assert fresh_summary["splits"] == recomputed["splits"]
assert fresh["config"] == old["config"]
state_hashes = {c["file"]: c["state_sha256"] for c in fresh["checkpoints"]}
old_hashes = {c["file"]: c["state_sha256"] for c in old["checkpoints"]}
max_probability_difference = 0.0
max_centered_reward_difference = 0.0
reward_offsets = []
for split in old["evaluations"]:
    for old_row, new_row in zip(old["evaluations"][split]["rows"], fresh["evaluations"][split]["rows"]):
        old_scores, new_scores = old_row["reward_model_raw_scores"], new_row["reward_model_raw_scores"]
        offset = sum(n - o for o, n in zip(old_scores, new_scores)) / 4
        reward_offsets.append(offset)
        max_centered_reward_difference = max(max_centered_reward_difference, max(abs((n - o) - offset) for o, n in zip(old_scores, new_scores)))
        for p in ["sft", "ppo", "dpo"]:
            max_probability_difference = max(max_probability_difference, max(abs(a - b) for a, b in zip(old_row["policies"][p]["probabilities"], new_row["policies"][p]["probabilities"])))
assert max_probability_difference < 2e-6
assert max_centered_reward_difference < 1e-5

result = {
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu",
                    "cpu_threads": torch.get_num_threads(), "cuda_available": torch.cuda.is_available(), "platform": platform.platform()},
    "original_record": {"path": str(report_path.relative_to(ROOT)), "sha256": sha(report_path),
                        "seed": original["seed"], "python": original["python_version"], "torch": original["torch_version"],
                        "device": original["device"], "recorded_elapsed_seconds": original["elapsed_seconds"],
                        "recorded_ppo_seconds": old["ppo"]["seconds"], "timing_scope": "Original finite training/evaluation/checkpoint run; no installation/startup/GPU timing."},
    "source_code_sha256": code_hashes,
    "config": old["config"],
    "data_and_updates": {"families": 55, "training_demonstrations": 44, "sft_updates": 60, "sft_draws": 1920,
                         "rm_training_pairs": 660, "rm_updates": 300, "rm_pair_draws": 19200,
                         "ppo_sampled_actions": 7680, "ppo_reused_action_draws": 23040,
                         "ppo_policy_updates": 360, "critic_updates": 360, "dpo_updates": 360,
                         "dpo_pair_draws": 23040, "effective_tokens": old["effective_tokens"]},
    "independent_original_recomputation": recomputed,
    "primary_snippet": {"sha256": hashlib.sha256(snippet.encode()).hexdigest(), "stdout": buffer.getvalue(), "length_scores": namespace["scores"], "chosen": namespace["chosen"],
                        "exercise_scores": correct_scores, "exercise_chosen_text": answers[correct_choice]},
    "incorrect_reward_counterexample": {"reward": [1,14], "baseline": 7.5, "advantage": [-6.5,6.5], "reference": [0.5,0.5],
                                        "clip_range": 0.2, "kl_coefficient": 0.1, "loss_gradient_logits": gradient,
                                        "learning_rate": 0.1, "probability_after_step": new_probs,
                                        "interpretation": "Artificial length reward still makes the wrong long answer more probable with clipping and fixed-reference KL."},
    "fresh_bounded_cpu": {"elapsed_seconds": elapsed, "ppo_seconds": fresh["ppo"]["seconds"],
                          "output_path": str(fresh_dir.relative_to(ROOT)), "configuration_identical": True,
                          "all_recomputed_discrete_metrics_identical": True, "state_hashes_identical": state_hashes == old_hashes,
                          "fresh_state_sha256": state_hashes, "original_state_sha256": old_hashes,
                          "max_absolute_probability_difference": max_probability_difference,
                          "max_absolute_centered_reward_difference": max_centered_reward_difference,
                          "raw_reward_offset_range": [min(reward_offsets), max(reward_offsets)],
                          "fresh_test_1_plus_2_explain": fresh_summary["test_1_plus_2_explain"],
                          "schedule_completed": fresh["schedule_completed"],
                          "timing_scope": "This invocation includes run_posttraining training/evaluation/checkpoint writes, after module imports; this is one CPU run, no GPU benchmark."},
    "limitations": ["One seed42 and one split; no quality guarantee for other seeds/tasks.",
                    "Only four Python-prewritten candidates and structured features; no arithmetic/Chinese/autoregressive generation training.",
                    "Original RM rankings and policy acceptance are recomputed separately; no learned-RM exploit is established.",
                    "Fresh weights/raw scores are not bitwise identical to the recorded run. Original printed raw values are validated from the original record; current CPU validates discrete metrics and close policy probabilities. Reward-pair loss is invariant to same-context additive score shifts."]
}
(OUT / "cpu-results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
