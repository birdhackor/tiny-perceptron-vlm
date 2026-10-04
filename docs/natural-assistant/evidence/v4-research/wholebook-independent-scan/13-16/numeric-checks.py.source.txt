import sys, json, datetime, torch

torch.set_num_threads(1)

print("CHECK_ENV", json.dumps({"python":sys.version,"torch":str(torch.__version__),"device":"cpu","task_identity":"/root/v4_wholebook_scan_13_16"}))

print("BEGIN_SECTION", '13.3')

import torch
from tiny_perceptron.alignment import sequence_log_probability

logits = torch.tensor([[[0.0, 2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 2.0]]])
labels = torch.tensor([[-100, 0, 2]])
score = sequence_log_probability(logits, labels)
print("整段log分數", round(score.item(), 4))
print("還原機率", round(score.exp().item(), 4))

print("END_SECTION", '13.3')

print("BEGIN_SECTION", '13.6')

import torch
from tiny_perceptron.alignment import dpo_loss

for beta in [0.1, 1.0, 5.0]:
    chosen = torch.tensor([-2.0], requires_grad=True)
    loss = dpo_loss(chosen, torch.tensor([-4.0]), torch.tensor([-3.0]), torch.tensor([-3.0]), beta=beta)
    loss.backward()
    print(beta, "代價", round(loss.item(), 6), "梯度", round(chosen.grad.item(), 6))

print("END_SECTION", '13.6')

print("BEGIN_SECTION", '13.13')

import torch
from tiny_perceptron.posttraining import ppo_clipped_objective

ratio = torch.tensor([0.7, 1.0, 1.3, 0.7, 1.0, 1.3], requires_grad=True)
advantage = torch.tensor([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
old_log_probability = torch.full((6,), 0.5).log()
new_log_probability = old_log_probability + ratio.log()
result = ppo_clipped_objective(new_log_probability, old_log_probability, advantage, clip_range=0.2)
loss = -result["surrogate"].sum()
loss.backward()
print("待提高的值", [round(value, 2) for value in result["surrogate"].tolist()])
print("下降代價時對ratio的梯度", [round(value, 2) for value in ratio.grad.tolist()])

print("END_SECTION", '13.13')

print("BEGIN_SECTION", '13.15')

import copy
import torch
from tiny_perceptron.posttraining import (
    FiniteResponsePolicy,
    FiniteRewardModel,
    FiniteValueModel,
    bandit_advantage,
    exact_kl,
    ppo_clipped_objective,
)

torch.manual_seed(42)
context = torch.tensor([[0.1, 0.2, 0.0, 0.0]])
policy, critic = FiniteResponsePolicy(), FiniteValueModel()
reward_model = FiniteRewardModel().requires_grad_(False)
reference = copy.deepcopy(policy).requires_grad_(False)
optimizer = torch.optim.SGD(list(policy.parameters()) + list(critic.parameters()), lr=0.1)

with torch.no_grad():
    old_distribution = torch.distributions.Categorical(logits=policy(context))
    action = old_distribution.sample()
    old_log_probability = old_distribution.log_prob(action)
    reward = reward_model(context).gather(1, action[:, None]).squeeze(1)
    advantage = bandit_advantage(reward, critic(context))

new_logits = policy(context)
new_log_probability = torch.distributions.Categorical(logits=new_logits).log_prob(action)
terms = ppo_clipped_objective(new_log_probability, old_log_probability, advantage)
value_loss = (critic(context) - reward).square().mean()
loss = terms["policy_loss"] + 0.1 * exact_kl(new_logits, reference(context)).mean() + 0.5 * value_loss
before = next(policy.parameters()).detach().clone()
optimizer.zero_grad()
loss.backward()
optimizer.step()
print("選中的卡", action.item(), "更新前ratio", terms["ratio"].item())
print("策略參數改變", not torch.equal(before, next(policy.parameters())))
print("評分員有梯度", any(p.grad is not None for p in reward_model.parameters()))
print("輸入與選卡分數形狀", list(context.shape), list(new_logits.shape))
print("固定參考有梯度", any(p.grad is not None for p in reference.parameters()))

print("END_SECTION", '13.15')

print("BEGIN_SECTION", '15.7')

import math
import torch

for renormalize in (False, True):
    scores = torch.tensor([math.log(2), 0.0], requires_grad=True)
    prob = scores.softmax(-1)
    p = prob.topk(1).values[0]
    gate = p / p if renormalize else p
    output = gate * 2
    loss = output.square()
    loss.backward()
    print("重新正規化", renormalize, "輸出", round(output.item(), 4), "梯度", scores.grad.round(decimals=4).tolist())

print("END_SECTION", '15.7')

print("BEGIN_SECTION", '16.3')

import torch
from tiny_perceptron.model import TinyLM, ModelConfig

torch.manual_seed(0)
model = TinyLM(ModelConfig(width=8)).eval()
ids = torch.tensor([[1, 2, 3, 4]])
with torch.no_grad():
    full = model(ids)["logits"][:, -1]
    prefix = model(ids[:, :3])
    step = model(ids[:, 3:], cache=prefix["cache"])
    cached = step["logits"][:, 0]
error = (full - cached).abs().max().item()
print("兩路分數形狀", tuple(full.shape), tuple(cached.shape))
print("最大差異", error)
assert torch.allclose(full, cached, atol=1e-6)

print("END_SECTION", '16.3')

print("BEGIN_SECTION", '16.4')

import torch
from tiny_perceptron.model import TinyLM, ModelConfig

ids = torch.tensor([[1, 2, 3]])
with torch.no_grad():
    for kv_heads in (4, 1):
        model = TinyLM(ModelConfig(width=16, heads=4, kv_heads=kv_heads)).eval()
        cache = model(ids)["cache"]
        size = sum(t.numel() * t.element_size() for pair in cache for t in pair)
        print("KV頭數", kv_heads, "K形狀", tuple(cache[0][0].shape), "bytes", size)

print("END_SECTION", '16.4')

print("BEGIN_SECTION", '16.6')

import torch

x = torch.tensor([1.0, 2.0, 3.0])
whole = torch.tensor(1.0, requires_grad=True)
(whole * x).square().mean().backward()
accumulated = torch.tensor(1.0, requires_grad=True)
wrong = torch.tensor(1.0, requires_grad=True)
for part in (x[:1], x[1:]):
    ((accumulated * part).square().sum() / len(x)).backward()
    ((wrong * part).square().mean() / 2).backward()
print("整批梯度", round(whole.grad.item(), 4))
print("正確累積", round(accumulated.grad.item(), 4))
print("錯誤平均", round(wrong.grad.item(), 4))

print("END_SECTION", '16.6')

print("BEGIN_SECTION", '16.8')

import torch
import torch.nn.functional as F

torch.manual_seed(0)
q, k, v = [torch.randn(1, 1, 4, 3) for _ in range(3)]
allowed = torch.ones(4, 4, dtype=torch.bool).tril()[None, None]
scores = q @ k.transpose(-2, -1) / (3**0.5)
weights = scores.masked_fill(~allowed, float("-inf")).softmax(-1)
manual = weights @ v
optimized = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0.0)
print("輸出形狀", tuple(optimized.shape))
print("最大差異", (manual - optimized).abs().max().item())
assert torch.allclose(manual, optimized, atol=1e-6)

print("END_SECTION", '16.8')

print("BEGIN_SECTION", '16.9')

import math
import torch

scores = torch.tensor([0.0, math.log(2), math.log(4)])
values = torch.tensor([10.0, 20.0, 30.0])
m = torch.tensor(float("-inf"))
denominator = torch.tensor(0.0)
numerator = torch.tensor(0.0)
for start in range(0, 3, 2):
    block = scores[start : start + 2]
    new_m = torch.maximum(m, block.max())
    factor = torch.exp(m - new_m)
    weight = torch.exp(block - new_m)
    denominator = denominator * factor + weight.sum()
    numerator = numerator * factor + (weight * values[start : start + 2]).sum()
    m = new_m
    print("分母/分子", denominator.item(), numerator.item())
print("分塊結果", round((numerator / denominator).item(), 4))
print("完整結果", round((scores.softmax(0) * values).sum().item(), 4))

print("END_SECTION", '16.9')
