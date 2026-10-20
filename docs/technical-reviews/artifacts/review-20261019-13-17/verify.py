"""核對偏好代價、資料切分與既有實驗紀錄；不訓練模型。"""

import copy
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments.common import records_sha256
from scripts.course_experiments.posttraining import CONFIG, build_records, split_records
from tiny_perceptron.alignment import dpo_loss
from tiny_perceptron.posttraining import FiniteResponsePolicy


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def main():
    record_path = "docs/course-experiments/results/posttraining.json"
    record = json.loads((ROOT / record_path).read_text())
    result = record["results"]
    paths = [
        record_path,
        "tiny_perceptron/alignment.py",
        "tiny_perceptron/posttraining.py",
        "scripts/course_experiments/posttraining.py",
        "scripts/course_experiments/common.py",
        "scripts/selftrained/train.py",
        "docs/selftrained/v2-training-stage-index.json",
        "docs/selftrained/results/v2-moe-weighted-10000-training-stage-index.json",
        "docs/selftrained/results/v2-dense-weighted-10000-training-stage-index.json",
        "docs/selftrained/results/v2-moe-native-balanced-4000-training-stage-index.json",
    ]
    output = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-13-17/verify.py",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"},
        "input_sha256": {path: digest(path) for path in paths},
        "historical_environment": {key: record[key] for key in ["device", "seed", "torch_version", "python_version"]},
        "historical_code_hashes": {},
    }
    for path, expected in record["code_sha256"].items():
        actual = digest(path)
        assert actual == expected, path
        output["historical_code_hashes"][path] = {"recorded": expected, "current": actual, "equal": True}

    reference = torch.tensor([0.5, 0.5]).log()
    examples = []
    for probabilities in [[0.5, 0.5], [0.7, 0.3], [0.3, 0.7]]:
        policy = torch.tensor(probabilities).log()
        actual = dpo_loss(policy[:1], policy[1:], reference[:1], reference[1:], beta=1.0)
        margin = math.log(probabilities[0] / probabilities[1])
        expected = math.log1p(math.exp(-margin))
        assert abs(actual.item() - expected) < 1e-7
        assert policy[:1].shape == policy[1:].shape == (1,)
        assert policy[:1].item() == policy[0].item()
        assert policy[1:].item() == policy[1].item()
        assert not actual.requires_grad
        examples.append({"probabilities": probabilities, "reference": [0.5, 0.5], "beta": 1.0,
                         "shape": list(policy.shape), "dtype": str(policy.dtype), "margin": margin,
                         "formula_loss": expected, "torch_loss": actual.item(), "rounded": round(actual.item(), 4),
                         "chosen_slice": policy[:1].tolist(), "rejected_slice": policy[1:].tolist(),
                         "requires_grad": actual.requires_grad})
    output["dpo_examples"] = examples

    splits = split_records(build_records(), record["seed"])
    output["splits"] = {}
    for name, rows in splits.items():
        saved = result["splits"][name]
        families = sorted({row["family"] for row in rows})
        assert families == saved["families"]
        assert records_sha256(rows) == saved["sha256"]
        assert len(rows) == saved["contexts"]
        output["splits"][name] = {"families": len(families), "contexts": len(rows), "records_sha256": records_sha256(rows)}
        for fresh, historical in zip(rows, result["evaluations"][name]["rows"], strict=True):
            for key, value in fresh.items():
                assert historical[key] == value, (name, key)
    assert not ({r["family"] for r in splits["train"]} & {r["family"] for r in splits["test"]})
    output["example_family"] = [r for r in build_records() if r["family"] == "pair:1:2"]
    output["test_recount"] = {}
    test = result["evaluations"]["test"]
    for policy_name in ["sft", "ppo", "dpo"]:
        by_mode = {}
        for mode in ["number", "explain", "missing"]:
            rows = [r for r in test["rows"] if r["mode"] == mode]
            numerator = 0
            for row in rows:
                policy = row["policies"][policy_name]
                chosen = max(range(4), key=lambda i: policy["probabilities"][i])
                assert chosen == policy["chosen_action"]
                correct = chosen == row["expected_action"]
                assert correct == policy["full_request_success"]
                numerator += correct
            by_mode[mode] = {"numerator": numerator, "denominator": len(rows)}
            saved = test["policies"][policy_name]["by_mode"][mode]
            assert numerator == saved["numerator"] and len(rows) == saved["denominator"]
        total = sum(v["numerator"] for v in by_mode.values())
        saved = test["policies"][policy_name]["greedy_full_request_success"]
        assert total == saved["numerator"] and len(test["rows"]) == saved["denominator"]
        output["test_recount"][policy_name] = {"numerator": total, "denominator": len(test["rows"]), "by_mode": by_mode}

    policy = FiniteResponsePolicy()
    first, second = copy.deepcopy(policy), copy.deepcopy(policy)
    assert all(torch.equal(first.state_dict()[key], value) for key, value in second.state_dict().items())
    assert sum(p.numel() for p in policy.parameters()) == result["parameters"]["policy"] == 148
    assert result["ppo"]["initial_state_sha256"] == result["dpo"]["initial_state_sha256"]
    assert result["ppo"]["initial_state_sha256"] == result["reference_state_sha256_before"] == result["reference_state_sha256_after"]
    output["initialization"] = {"parameters": 148, "historical_initial_state_sha256": result["ppo"]["initial_state_sha256"],
                                "historical_reference_unchanged": True, "deepcopy_tensor_comparison": True}
    assert CONFIG == result["config"]
    output["config"] = CONFIG
    counts = {"ppo_policy_updates": CONFIG["ppo_rollout_batches"] * CONFIG["ppo_epochs_per_rollout"],
              "ppo_value_updates": CONFIG["ppo_rollout_batches"] * CONFIG["ppo_epochs_per_rollout"],
              "ppo_sampled_actions": CONFIG["ppo_rollout_batches"] * CONFIG["ppo_rollout_size"],
              "dpo_policy_updates": CONFIG["dpo_steps"], "dpo_pair_draws": CONFIG["dpo_steps"] * CONFIG["dpo_batch_size"],
              "reward_updates": CONFIG["reward_steps"]}
    assert counts == {"ppo_policy_updates": result["ppo"]["policy_updates"], "ppo_value_updates": result["ppo"]["value_updates"],
                      "ppo_sampled_actions": result["ppo"]["sampled_actions"], "dpo_policy_updates": result["dpo"]["policy_updates"],
                      "dpo_pair_draws": result["dpo"]["processed_pair_draws"], "reward_updates": 300}
    output["schedule_recount"] = counts
    output["historical_seconds"] = {name: {"recorded": result[name]["seconds"], "rounded_2dp": round(result[name]["seconds"], 2)} for name in ["ppo", "dpo", "reward"]}
    output["sft_scope"] = {key: result["sft"][key] for key in ["demonstrations", "demonstration_mode"]}
    output["comparison_limits"] = result["comparison_limits"]
    baseline = json.loads((ROOT / paths[6]).read_text())
    output["public_product_stages"] = [stage["stage_key"] for stage in baseline["stages"]]
    output["public_product_objectives"] = {}
    for path in paths[7:]:
        item = json.loads((ROOT / path).read_text())
        output["public_product_objectives"][path] = item.get("full_language_objective", item.get("full_native_objective"))
    output["limitations"] = ["未重跑歷史訓練；計時只核對紀錄值與程式量測區間。", "公開成品以 repo 的階段索引與訓練程式核對，未下載權重或存取禁止的原始 outputs。"]
    output["status"] = "pass"
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
