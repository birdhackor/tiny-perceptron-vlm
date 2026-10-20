"""核對凍結算例、KL 契約與有限候選 PPO 的更新資料流。"""

import ast
import copy
import hashlib
import json
import math
import platform
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts.course_experiments import posttraining as experiment  # noqa: E402
from tiny_perceptron.posttraining import (  # noqa: E402
    FiniteResponsePolicy,
    FiniteRewardModel,
    FiniteValueModel,
    exact_kl,
)


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def main():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    torch.set_default_device("cpu")
    output = {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-13-14/verify_facts.py",
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torch_git_version": str(torch.version.git_version),
            "device": "cpu",
            "seed": "42",
            "dtype": "float32；KL 解析對照另用 float64",
        },
        "inputs": {
            path: sha(path)
            for path in (
                "docs/course-review-20261019/freeze-01/sources/13.14.md",
                "docs/course-review-20261019/freeze-04.json",
                "course/figures/rewrite-13-model-roles.svg",
                "tiny_perceptron/posttraining.py",
                "scripts/course_experiments/posttraining.py",
            )
        },
    }
    values = []
    for prediction, expected_loss, expected_gradient in ((0.4, 0.36, -1.2), (1.4, 0.16, 0.8)):
        value = torch.tensor([prediction], requires_grad=True)
        loss = (value - torch.tensor([1.0])).square().mean()
        loss.backward()
        assert abs(loss.item() - expected_loss) < 1e-6
        assert abs(value.grad.item() - expected_gradient) < 1e-6
        assert value.item() == torch.tensor([prediction]).item()
        values.append({"prediction": prediction, "target": 1.0, "loss": loss.item(), "gradient": value.grad.item(), "value_after_backward": value.item(), "sgd_direction": "增加" if value.grad.item() < 0 else "降低"})
    output["value_examples"] = values

    probabilities = torch.tensor([[0.8, 0.2], [0.5, 0.5], [0.3, 0.7]], dtype=torch.float64)
    reference = torch.tensor([[0.5, 0.5], [0.5, 0.5], [0.6, 0.4]], dtype=torch.float64)
    logits = probabilities.log().requires_grad_()
    reference_logits = reference.log().requires_grad_()
    result = exact_kl(logits, reference_logits)
    manual = [sum(p * math.log(p / q) for p, q in zip(ps, qs, strict=True)) for ps, qs in zip(probabilities.tolist(), reference.tolist(), strict=True)]
    torch.testing.assert_close(logits.softmax(-1), probabilities, rtol=0, atol=1e-15)
    torch.testing.assert_close(result, torch.tensor(manual, dtype=torch.float64), rtol=0, atol=1e-12)
    result.sum().backward()
    analytic_gradient = probabilities * (probabilities.log() - reference.log() - result.detach()[:, None])
    torch.testing.assert_close(logits.grad, analytic_gradient, rtol=0, atol=1e-12)
    assert reference_logits.grad is None
    assert result[1].item() == 0.0
    output["kl_examples"] = {"probabilities": probabilities.tolist(), "reference_probabilities": reference.tolist(), "manual": manual, "executed": result.tolist(), "output_shape": list(result.shape), "policy_logit_gradient": logits.grad.tolist(), "reference_gradient_is_none": reference_logits.grad is None, "roundtrip_probabilities": logits.detach().softmax(-1).tolist()}
    errors = []
    for shapes in (((2,), (2,)), ((2, 3), (2, 4)), ((1, 2, 3), (1, 2, 3))):
        try:
            exact_kl(torch.zeros(shapes[0]), torch.zeros(shapes[1]))
        except ValueError as error:
            errors.append({"input_shapes": shapes, "exception_type": type(error).__name__, "message": str(error)})
        else:
            raise AssertionError("無效形狀未被拒絕")
    output["shape_validation"] = errors

    source = (ROOT / "docs/course-review-20261019/freeze-01/sources/13.14.md").read_text()
    fences = re.findall(r"```python\n(.*?)```", source, re.S)
    lesson_function = ast.parse(fences[-1]).body[0]
    implementation_tree = ast.parse((ROOT / "tiny_perceptron/posttraining.py").read_text())
    implementation_function = next(node for node in implementation_tree.body if isinstance(node, ast.FunctionDef) and node.name == "exact_kl")
    assert ast.dump(lesson_function, include_attributes=False) == ast.dump(implementation_function, include_attributes=False)
    output["excerpt_matches_implementation_ast"] = True

    # 只執行現行原始碼的收集／更新迴圈；不跑完整訓練或寫權重。
    script_path = ROOT / "scripts/course_experiments/posttraining.py"
    tree = ast.parse(script_path.read_text())
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_posttraining")
    loop = next(node for node in run.body if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "rollout")
    namespace = vars(experiment).copy()
    config = {**experiment.CONFIG, "ppo_rollout_batches": 2, "ppo_rollout_size": 4, "ppo_epochs_per_rollout": 2}
    policy = FiniteResponsePolicy()
    reference_model = copy.deepcopy(policy).requires_grad_(False).eval()
    reward_model = FiniteRewardModel().requires_grad_(False).eval()
    value_model = FiniteValueModel()
    train = experiment.build_records()[:6]
    before = {"policy": experiment._state_sha256(policy), "reference": experiment._state_sha256(reference_model), "reward": experiment._state_sha256(reward_model), "value": experiment._state_sha256(value_model)}
    namespace.update(CONFIG=config, ctx=SimpleNamespace(device="cpu"), ppo=policy, reference=reference_model, reward_model=reward_model, value=value_model, train=train, x=experiment._features(train, "cpu"), scale=1.0, sampler=random.Random(42), action_generator=torch.Generator(device="cpu").manual_seed(44), optimizer=torch.optim.Adam(policy.parameters(), lr=config["ppo_learning_rate"]), value_optimizer=torch.optim.Adam(value_model.parameters(), lr=config["value_learning_rate"]), sft_hash=before["reference"], ppo_history=[], first_rollout=[])
    exec(compile(ast.Module(body=[loop], type_ignores=[]), str(script_path), "exec"), namespace)
    after = {"policy": experiment._state_sha256(policy), "reference": experiment._state_sha256(reference_model), "reward": experiment._state_sha256(reward_model), "value": experiment._state_sha256(value_model)}
    changed = {name: before[name] != after[name] for name in before}
    assert changed == {"policy": True, "reference": False, "reward": False, "value": True}
    rows = namespace["first_rollout"]
    for key in ("old_selected_log_probabilities", "old_values", "fixed_advantages", "old_log_probability_sha256"):
        assert rows[0][key] == rows[1][key]
    assert all(parameter.grad is None for model in (reference_model, reward_model) for parameter in model.parameters())
    torch.testing.assert_close(namespace["advantages"], namespace["rewards"] - namespace["old_values"])
    expected_value_loss = (value_model(namespace["features"]) - namespace["rewards"]).square().mean()
    output["ppo_dataflow"] = {"configuration": {key: config[key] for key in ("ppo_rollout_batches", "ppo_rollout_size", "ppo_epochs_per_rollout", "reference_kl_coefficient")}, "initialization": "隨機小網路；僅核對原始收集／更新迴圈，不作 SFT 或獎勵品質主張", "changed_parameters": changed, "before": before, "after": after, "first_rollout": rows, "reference_and_reward_gradients_absent": True, "latest_value_loss_after_update_against_rm_only": expected_value_loss.item(), "latest_advantages_require_grad": namespace["advantages"].requires_grad}
    receipt = json.loads((ROOT / "docs/course-review-20261019/figure-renders-04.json").read_text())
    figure = "course/figures/rewrite-13-model-roles.svg"
    def matches(value):
        if isinstance(value, dict):
            if figure in value.values():
                return value
            for nested in value.values():
                found = matches(nested)
                if found is not None:
                    return found
        if isinstance(value, list):
            for nested in value:
                found = matches(nested)
                if found is not None:
                    return found
        return None
    output["figure_render_receipt"] = matches(receipt)
    output["result"] = "所有解析數值、形狀、梯度與更新資料流核對通過；不測量回答品質。"
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
