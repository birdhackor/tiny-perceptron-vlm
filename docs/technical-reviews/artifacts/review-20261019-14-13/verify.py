"""核對共享加法、末端監督與截斷反向傳播的確定性算例。"""

import json
import math
import platform

import torch
from torch import nn


def close(observed, expected):
    assert math.isclose(observed, expected, abs_tol=1e-6), (observed, expected)


def shared_addition(depth, truncate_before=None):
    weight = torch.tensor(0.2, requires_grad=True)
    before = weight.item()
    state = torch.tensor(0.0)
    states = []
    uses = []
    for step in range(depth):
        if step == truncate_before:
            state = state.detach()
        use = weight * 1
        use.retain_grad()
        uses.append(use)
        state = state + use
        state.retain_grad()
        states.append(state)
    loss = (state - 1).square()
    loss.backward()
    result = {
        'depth': depth,
        'truncate_before': truncate_before,
        'state': state.item(),
        'loss': loss.item(),
        'weight_gradient': weight.grad.item(),
        'state_gradients': [None if s.grad is None else s.grad.item() for s in states],
        'use_gradients': [None if u.grad is None else u.grad.item() for u in uses],
        'weight_before_backward': before,
        'weight_after_backward': weight.item(),
    }
    assert result['weight_before_backward'] == result['weight_after_backward']
    close(result['state'], 0.2 * depth)
    close(result['loss'], (0.2 * depth - 1) ** 2)
    retained_uses = depth if truncate_before is None else depth - truncate_before
    close(result['weight_gradient'], 2 * (0.2 * depth - 1) * retained_uses)
    return result


def terminal_classification():
    weight = nn.Parameter(torch.tensor(0.2))
    optimizer = torch.optim.SGD([weight], lr=0.1)
    state = torch.tensor(0.0)
    states = []
    for _ in range(3):
        state = state + weight
        state.retain_grad()
        states.append(state)
    head = torch.stack([state * i for i in range(5)]).unsqueeze(0)
    target = torch.tensor([4])
    loss = nn.functional.cross_entropy(head, target)
    expected = -head[0, 4] + torch.logsumexp(head[0], dim=0)
    close(loss.item(), expected.item())
    before = weight.item()
    loss.backward()
    after_backward = weight.item()
    grad = weight.grad.item()
    close(before, after_backward)
    assert all(s.grad is not None for s in states)
    close(grad, sum(s.grad.item() for s in states))
    optimizer.step()
    close(weight.item(), before - 0.1 * grad)
    return {
        'classes': 5,
        'target': target.item(),
        'head_shape': list(head.shape),
        'loss': loss.item(),
        'manual_negative_log_probability': expected.item(),
        'state_gradients': [s.grad.item() for s in states],
        'weight_gradient': grad,
        'weight_before': before,
        'weight_after_backward': after_backward,
        'weight_after_one_optimizer_step': weight.item(),
        'optimizer_steps': 1,
        'directly_supervised_depths': [3],
    }


full = shared_addition(3)
two = shared_addition(2)
truncated = shared_addition(3, truncate_before=2)
assert truncated['state_gradients'][:2] == [None, None]
assert truncated['use_gradients'][:2] == [None, None]
close(full['weight_gradient'], -2.4)
close(truncated['weight_gradient'], -0.8)
for contribution in full['use_gradients']:
    close(contribution, -0.8)

# 達到末端最小代價仍不代表其他深度有相同代價。
depth_counterexample = {
    str(depth): {'state': depth / 3, 'loss': (depth / 3 - 1) ** 2}
    for depth in (2, 3, 4)
}
assert depth_counterexample['3']['loss'] == 0
assert depth_counterexample['2']['loss'] > 0
assert depth_counterexample['4']['loss'] > 0

print(json.dumps({
    'environment': {
        'python': platform.python_version(),
        'torch': str(torch.__version__),
        'torch_git_version': str(torch.version.git_version),
        'device': 'cpu',
        'dtype': 'torch.float32',
    },
    'arithmetic_answer': 2 + 3 - 1,
    'three_rounds': full,
    'two_rounds': two,
    'last_round_only_backward': truncated,
    'terminal_classification': terminal_classification(),
    'exact_terminal_optimum_counterexample': {
        'shared_weight': '1/3',
        'initial_state': 0,
        'target': 1,
        'depths': depth_counterexample,
    },
    'scope': '確定性機制與數值核對；未訓練 Transformer，未量測跨深度品質。',
    'result': '全部斷言通過。',
}, ensure_ascii=False, indent=2))
