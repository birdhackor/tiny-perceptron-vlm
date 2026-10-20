"""核對偏好算例、有限候選模型與資料切分；不執行完整訓練。"""

import ast
import hashlib
import inspect
import json
import math
import platform
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.nn import functional as F

from scripts.course_experiments.posttraining import (
    _evaluate,
    _features,
    _pairs,
    build_records,
    run_posttraining,
    split_records,
)
from tiny_perceptron.alignment import dpo_loss
from tiny_perceptron.posttraining import FiniteResponsePolicy, FiniteRewardModel, preference_loss

torch.set_num_threads(1)
torch.manual_seed(20261020)
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-13-10/verify.py",
    "environment": {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version),
        "device": "cpu",
        "seed": "20261020",
    },
}

numeric = []
for rejected_score in (0.0, 1.0):
    for chosen_score in (0.0, 2.0):
        chosen = torch.tensor([chosen_score], requires_grad=True)
        rejected = torch.tensor([rejected_score], requires_grad=True)
        gap = chosen - rejected
        probability = gap.sigmoid()
        loss = preference_loss(chosen, rejected)
        expected_probability = 1 / (1 + math.exp(-(chosen_score - rejected_score)))
        expected_loss = math.log1p(math.exp(-(chosen_score - rejected_score)))
        assert abs(probability.item() - expected_probability) < 1e-7
        assert abs(loss.item() - expected_loss) < 1e-7
        shifted_gap = (chosen + 10) - (rejected + 10)
        shifted_loss = preference_loss(chosen + 10, rejected + 10)
        assert torch.equal(gap, shifted_gap)
        assert torch.equal(loss, shifted_loss)
        loss.backward()
        assert chosen.grad.item() < 0 < rejected.grad.item()
        numeric.append({
            "chosen": chosen_score,
            "rejected": rejected_score,
            "gap": gap.item(),
            "probability": probability.item(),
            "expected_probability": expected_probability,
            "loss": loss.item(),
            "expected_loss": expected_loss,
            "shifted_loss": shifted_loss.item(),
            "chosen_gradient": chosen.grad.item(),
            "rejected_gradient": rejected.grad.item(),
        })
result["numeric"] = numeric
batch_chosen = torch.tensor([0.0, 2.0])
batch_rejected = torch.zeros(2)
batch_loss = preference_loss(batch_chosen, batch_rejected)
assert torch.allclose(batch_loss, -F.logsigmoid(batch_chosen - batch_rejected).mean())
assert torch.equal(batch_loss, dpo_loss(batch_chosen, batch_rejected, torch.zeros(2), torch.zeros(2), beta=1.0))
result["batch_mean"] = batch_loss.item()

invalid = []
for chosen, rejected in [(torch.ones(2), torch.ones(1)), (torch.empty(0), torch.empty(0))]:
    try:
        preference_loss(chosen, rejected)
    except ValueError as error:
        invalid.append({"chosen_shape": list(chosen.shape), "rejected_shape": list(rejected.shape), "exception": str(error)})
    else:
        raise AssertionError("不合法的輸入必須停止計算")
result["invalid_inputs"] = invalid
assert torch.tensor([[1, 2], [3, 4]]).numel() == 4

frozen = (ROOT / "docs/course-review-20261019/freeze-01/sources/13.10.md").read_text()
fences = re.findall(r"```python\n(.*?)```", frozen, re.S)
namespace = {"torch": torch, "nn": nn, "F": F}
for block in fences[1:]:
    exec(compile(block, "frozen-excerpt", "exec"), namespace)
for name, implementation in [("preference_loss", preference_loss), ("FiniteRewardModel", FiniteRewardModel)]:
    excerpt = next(node for block in fences[1:] for node in ast.parse(block).body if getattr(node, "name", None) == name)
    actual = ast.parse(inspect.getsource(implementation)).body[0]
    assert ast.dump(excerpt, include_attributes=False) == ast.dump(actual, include_attributes=False)
result["excerpt_matches"] = ["preference_loss", "FiniteRewardModel"]
assert namespace["preference_loss"].__doc__ is not None
namespace["preference_loss"].__doc__ = None
assert torch.equal(namespace["preference_loss"](batch_chosen, batch_rejected), batch_loss)
result["docstring_mutation_preserves_loss"] = True

model = FiniteRewardModel()
assert torch.equal(model.candidate_identity, torch.eye(4))
assert "candidate_identity" in dict(model.named_buffers())
assert "candidate_identity" in model.state_dict()
assert "candidate_identity" not in dict(model.named_parameters())
assert not model.candidate_identity.requires_grad
result["buffer"] = {
    "identity": model.candidate_identity.tolist(),
    "persistent": True,
    "parameter": False,
    "requires_grad": False,
}
shape_records = []
for count in (1, 3):
    features = torch.tensor([[0.1, 0.2, 0.0, 0.0]] * count)
    contexts = features[:, None, :].expand(-1, len(model.candidate_identity), -1)
    identities = model.candidate_identity[None, :, :].expand(len(features), -1, -1)
    combined = torch.cat((contexts, identities), -1)
    linear = model.network[0](combined)
    activated = model.network[1](linear)
    scalar = model.network[2](activated)
    scores = model(features)
    assert list(contexts.shape) == [count, 4, 4]
    assert list(identities.shape) == [count, 4, 4]
    assert list(combined.shape) == [count, 4, 8]
    assert list(linear.shape) == [count, 4, 24]
    assert list(scalar.shape) == [count, 4, 1]
    assert list(scores.shape) == [count, 4]
    assert torch.equal(scores, scalar.squeeze(-1))
    assert (activated.abs() <= 1).all()
    for question in range(count):
        for card in range(4):
            single = model.network(torch.cat((features[question], model.candidate_identity[card])))
            torch.testing.assert_close(scores[question, card], single.squeeze(-1))
    shape_records.append({"features": list(features.shape), "contexts": list(contexts.shape), "identities": list(identities.shape), "combined": list(combined.shape), "hidden": list(linear.shape), "scalar": list(scalar.shape), "output": list(scores.shape)})
result["shapes"] = shape_records
restored = FiniteRewardModel()
restored.load_state_dict(model.state_dict())
torch.testing.assert_close(restored(features), model(features))

records = build_records()
splits = split_records(records)
owners = {}
for split, rows in splits.items():
    for row in rows:
        owners.setdefault(row["family"], set()).add(split)
assert all(len(value) == 1 for value in owners.values())
for a in range(1, 11):
    for b in range(a, 11):
        rows = [r for r in records if r["operands"] == [a, b]]
        assert {r["mode"] for r in rows} == {"number", "explain", "missing"}
        for row in rows:
            assert row["features"] == [a / 10, b / 10, float(row["mode"] == "explain"), float(row["mode"] == "missing")]
            assert row["expected_action"] == {"number": 0, "explain": 1, "missing": 3}[row["mode"]]
            assert row["candidates"][2] == str(a + b + 1)
pair = [r for r in records if r["family"] == "pair:1:2"]
assert next(r for r in pair if r["mode"] == "number")["candidates"] == ["3", "1 加 2 是 3。", "4", "請再提供更多資訊。"]
missing = next(r for r in pair if r["mode"] == "missing")
assert missing["candidates"][3] == "請補充購買數量。"
assert missing["preference_pairs"] == [[3, 0], [3, 1], [3, 2]]
result["records"] = {"count": len(records), "families": len(owners), "split_contexts": {k: len(v) for k, v in splits.items()}, "split_pairs": {k: len(_pairs(v)) for k, v in splits.items()}, "family_disjoint": True, "example_family": pair}

train_features = _features(splits["train"], "cpu")
context, winner, loser = torch.tensor(_pairs(splits["train"])[0]).tolist()
scores = model(train_features[context:context + 1])
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer.zero_grad()
loss = preference_loss(scores[:, winner], scores[:, loser])
loss.backward()
assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
optimizer.step()
model.requires_grad_(False).eval()
assert not any(p.requires_grad for p in model.parameters())
result["one_pair_update_then_freeze"] = {"training_pair": [context, winner, loser], "loss": loss.item(), "all_parameters_frozen": True, "quality_claim": "未量測訓練品質；僅核對資料流與參數凍結"}
policy = FiniteResponsePolicy()
evaluation = _evaluate(splits["test"], {"sft": policy}, model, policy, 1.0, "cpu")
assert evaluation["reward_model_pairwise_accuracy"]["denominator"] == len(_pairs(splits["test"]))
result["evaluation_contract"] = {"contexts": evaluation["contexts"], "pairwise_metric_fields": sorted(evaluation["reward_model_pairwise_accuracy"])}
pipeline = inspect.getsource(run_posttraining)
assert 'train = splits["train"]' in pipeline
assert 'x = _features(train, ctx.device)' in pipeline
assert 'pairs = _pairs(train)' in pipeline
assert pipeline.index('reward_model.requires_grad_(False).eval()') < pipeline.index('ppo = copy.deepcopy(reference)')
assert '"history": reward_history' in pipeline
result["pipeline_inspection"] = {"training_split_only": True, "reward_frozen_before_ppo": True, "loss_history_stored_separately_from_pairwise_accuracy": True, "method": "執行來源文字斷言，另直接呼叫資料建構、模型更新與評估函式；未執行完整訓練"}
result["repository_sources"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ["tiny_perceptron/posttraining.py", "tiny_perceptron/alignment.py", "scripts/course_experiments/posttraining.py"]}
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
