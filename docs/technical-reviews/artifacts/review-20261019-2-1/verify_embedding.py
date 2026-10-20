import json
import platform

import torch
from torch import nn


torch.manual_seed(41)
torch.set_num_threads(1)
table = torch.tensor(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
     [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
)
embedding = nn.Embedding(5, 3)
random_initial = embedding.weight.detach().clone()
assert not torch.equal(random_initial, table)
assert embedding.weight.grad is None
with torch.no_grad():
    embedding.weight.copy_(table)
ids = torch.tensor([[1, 2, 1]])
x = embedding(ids)
assert tuple(x.shape) == (1, 3, 3)
assert torch.equal(x, table[ids])
assert torch.equal(x[0, 0], x[0, 2])
assert torch.equal(embedding.weight, table)
exercise = embedding(torch.tensor([[1, 1, 1]]))
assert torch.equal(exercise, table[torch.tensor([[1, 1, 1]])])
assert exercise.shape == x.shape

equal_checks = {
    "same_shape_and_values": torch.equal(x[0, 0], x[0, 2]),
    "different_shape": torch.equal(torch.tensor([1.0]), torch.tensor([[1.0]])),
    "different_value": torch.equal(torch.tensor([1.0]), torch.tensor([2.0])),
    "different_dtype_same_values": torch.equal(torch.tensor([1]), torch.tensor([1.0])),
    "nan_compared_with_nan": torch.equal(torch.tensor([float("nan")]), torch.tensor([float("nan")])),
}
assert equal_checks == {
    "same_shape_and_values": True, "different_shape": False,
    "different_value": False, "different_dtype_same_values": True,
    "nan_compared_with_nan": False,
}
coefficient = nn.Parameter(torch.tensor([2.0, 0.0, 1.0]))
scores = x @ coefficient
assert scores.tolist() == [[2.0, 0.0, 2.0]]
changed_table = table.clone()
changed_table[1, 0] = 1.5
assert (changed_table[1] @ coefficient).item() == 3.0
changed_coefficient = torch.tensor([3.0, 0.0, 1.0])
assert (table[1] @ changed_coefficient).item() == 3.0

# 重複使用同一列時，梯度會累加；求導本身不會更新參數。
scores.sum().backward()
gradient = embedding.weight.grad.detach().clone()
expected_gradient = torch.tensor(
    [[0.0, 0.0, 0.0], [4.0, 0.0, 2.0], [2.0, 0.0, 1.0],
     [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
)
assert torch.equal(gradient, expected_gradient)
assert torch.equal(coefficient.grad, torch.tensor([2.0, 1.0, 0.0]))
assert torch.equal(embedding.weight, table)

# 用答案代價驗證兩組參數都能被更新，不推論訓練品質。
output = nn.Linear(3, 5, bias=False)
with torch.no_grad():
    output.weight.zero_()
    output.weight[3].copy_(torch.tensor([2.0, 0.0, 1.0]))
optimizer = torch.optim.SGD(list(embedding.parameters()) + list(output.parameters()), lr=0.1)
optimizer.zero_grad()
before_embedding = embedding.weight.detach().clone()
before_output = output.weight.detach().clone()
input_cat = torch.tensor([1])
target = torch.tensor([3])
logits = output(embedding(input_cat))
probability_before = logits.softmax(-1)[0, 3].item()
loss_before = nn.functional.cross_entropy(logits, target)
loss_before.backward()
assert torch.equal(embedding.weight, before_embedding)
assert torch.equal(output.weight, before_output)
optimizer.step()
with torch.no_grad():
    after_logits = output(embedding(input_cat))
    probability_after = after_logits.softmax(-1)[0, 3].item()
    loss_after = nn.functional.cross_entropy(after_logits, target).item()
assert not torch.equal(embedding.weight, before_embedding)
assert not torch.equal(output.weight, before_output)
assert torch.equal(embedding.weight[[0, 2, 3, 4]], before_embedding[[0, 2, 3, 4]])
assert probability_after > probability_before
assert loss_after < loss_before.item()

# 改 ID 時同步交換表列，查出的數字不變。
permutation = torch.tensor([0, 3, 2, 1, 4])
assert torch.equal(table[ids], table[permutation][permutation[ids]])
# 特徵軸與配方權重同步重排，分數仍相同。
axes = torch.tensor([2, 1, 0])
assert torch.equal(table @ coefficient.detach(), table[:, axes] @ coefficient.detach()[axes])

bigram = nn.Embedding(3, 3)
with torch.no_grad():
    bigram.weight.zero_()
    bigram.weight[0, 1] = 2.0
bigram_scores = bigram(torch.tensor([0, 2]))
assert bigram_scores.tolist() == [[0.0, 2.0, 0.0], [0.0, 0.0, 0.0]]
assert sum(p.numel() for p in embedding.parameters()) == 15

print(json.dumps({
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-2-1/verify_embedding.py",
    "environment": {"python": platform.python_version(), "torch": torch.__version__,
                    "torch_git_version": torch.version.git_version, "device": "cpu", "seed": "41"},
    "ids_dtype": str(ids.dtype), "weight_dtype": str(table.dtype),
    "table": table.tolist(), "ids": ids.tolist(), "shape": list(x.shape), "lookup": x.tolist(),
    "exercise_shape": list(exercise.shape), "exercise": exercise.tolist(),
    "equal_checks": equal_checks, "candidate_scores": scores.detach().tolist(),
    "score_after_table_change": 3.0, "score_after_coefficient_change": 3.0,
    "embedding_gradient": gradient.tolist(), "coefficient_gradient": coefficient.grad.tolist(),
    "unchanged_after_lookup_and_backward": True,
    "probability_before": probability_before, "probability_after": probability_after,
    "loss_before": loss_before.item(), "loss_after": loss_after,
    "embedding_and_output_changed_after_step": True, "unused_rows_unchanged_with_plain_sgd": True,
    "relabeling_preserves_lookup": True, "axis_permutation_preserves_score": True,
    "random_initialized_before_manual_copy": True, "parameter_count": 15,
    "previous_bigram_scores": bigram_scores.tolist(), "result": "所有斷言通過"
}, ensure_ascii=False, indent=2))
