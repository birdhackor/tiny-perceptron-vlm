"""獨立核對查詢與匹配的算例、矩陣尺寸及限制條件。"""

import hashlib
import json
import platform
import sys
from pathlib import Path

import torch
from torch import nn


def product(left, right):
    """以純 Python 逐項內積計算左表與右表轉置的乘積。"""
    return [[sum(a * b for a, b in zip(row, key, strict=True)) for key in right] for row in left]


q_rows = [[1, 0], [0, 1]]
k_rows = [[1, 0], [0, 1], [1, 1]]
has_rows = [[1, 0], [0, 1]]
need_rows = [[0, 1], [1, 0]]
q = torch.tensor(q_rows, dtype=torch.float32)
k = torch.tensor(k_rows, dtype=torch.float32)
has = torch.tensor(has_rows, dtype=torch.float32)
need = torch.tensor(need_rows, dtype=torch.float32)
scores = q @ k.T
shared = has @ has.T
separate = need @ has.T
assert scores.tolist() == product(q_rows, k_rows) == [[1, 0, 1], [0, 1, 1]]
assert shared.tolist() == product(has_rows, has_rows) == [[1, 0], [0, 1]]
assert separate.tolist() == product(need_rows, has_rows) == [[0, 1], [1, 0]]
assert tuple(scores.shape) == (2, 3)
assert tuple(k.T.shape) == (2, 3)
assert k.T.T.equal(k)

extended_k = torch.cat([k, torch.tensor([[2.0, 0.0]])], dim=0)
extended_scores = q @ extended_k.T
assert tuple(extended_scores.shape) == (2, 4)
assert extended_scores[:, -1].tolist() == [2.0, 0.0]
assert extended_scores.tolist() == product(q_rows, k_rows + [[2, 0]])

wide_q = torch.arange(12, dtype=torch.float64).reshape(3, 4)
wide_k = torch.arange(20, dtype=torch.float64).reshape(5, 4)
assert tuple((wide_q @ wide_k.T).shape) == (3, 5)
try:
    wide_q @ torch.zeros(5, 6, dtype=torch.float64).T
except RuntimeError as error:
    mismatch = str(error)
else:
    raise AssertionError("匹配特徵數不同時應拒絕相乘")

query_layer = nn.Linear(2, 2, bias=False)
key_layer = nn.Linear(2, 2, bias=False)
with torch.no_grad():
    query_layer.weight.copy_(torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
    key_layer.weight.copy_(torch.eye(2))
assert query_layer(has).equal(need)
assert key_layer(has).equal(has)
assert query_layer.weight.data_ptr() != key_layer.weight.data_ptr()

x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
gram = x @ x.T
assert gram.equal(gram.T)
assert torch.all(gram.diag().unsqueeze(1) >= gram)
unequal = torch.tensor([[1.0, 0.0], [2.0, 0.0]], dtype=torch.float64)
unequal_gram = unequal @ unequal.T
assert unequal_gram[0, 1] > unequal_gram[0, 0]

rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
assert ((q @ rotation) @ (k @ rotation).T).equal(scores)

weights = scores.softmax(dim=-1)
column_weights = scores.softmax(dim=0)
flat_weights = scores.flatten().softmax(dim=0).reshape_as(scores)
torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2), rtol=0, atol=1e-7)
changed_scores = scores.clone()
changed_scores[1, 0] = 10
changed_weights = changed_scores.softmax(dim=-1)
assert changed_weights[0].equal(weights[0])
assert not changed_scores.softmax(dim=0)[0].equal(column_weights[0])
assert not changed_scores.flatten().softmax(dim=0).reshape_as(scores)[0].equal(flat_weights[0])

source = Path('docs/course-review-20261019/freeze-04/sources/3.3.md')
print(json.dumps({
    'command': 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-3-3/verify.py',
    'environment': {'python': sys.version, 'torch': str(torch.__version__), 'torch_git': str(torch.version.git_version), 'device': str(scores.device), 'dtype': str(scores.dtype), 'platform': platform.platform()},
    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'scores': scores.tolist(),
    'scores_shape': list(scores.shape),
    'transposed_k_shape': list(k.T.shape),
    'shared_scores': shared.tolist(),
    'separate_scores': separate.tolist(),
    'shared_argmax': shared.argmax(dim=-1).tolist(),
    'separate_argmax': separate.argmax(dim=-1).tolist(),
    'extended_scores': extended_scores.tolist(),
    'extended_shape': list(extended_scores.shape),
    'general_shape_example': list((wide_q @ wide_k.T).shape),
    'feature_mismatch_error': mismatch,
    'query_linear_output': query_layer(has).tolist(),
    'key_linear_output': key_layer(has).tolist(),
    'equal_norm_gram': gram.tolist(),
    'unequal_norm_gram': unequal_gram.tolist(),
    'row_softmax': weights.tolist(),
    'row_softmax_sums': weights.sum(dim=-1).tolist(),
    'column_softmax': column_weights.tolist(),
    'column_softmax_row_sums': column_weights.sum(dim=-1).tolist(),
    'flat_softmax_row_sums': flat_weights.sum(dim=-1).tolist(),
    'other_query_change_preserves_first_row_softmax': changed_weights[0].equal(weights[0]),
    'orthogonal_change_preserves_scores': True,
    'result': '全部斷言通過；僅核對此 CPU 配置的確定性運算，沒有訓練或品質實測。',
}, ensure_ascii=False, indent=2))
