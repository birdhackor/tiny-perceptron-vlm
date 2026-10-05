"""Independent bounded CPU verification of lesson 13.14; no model downloads or full recipe."""
import ast
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
import torch
from tiny_perceptron.posttraining import (
    FiniteResponsePolicy, FiniteRewardModel, FiniteValueModel,
    exact_kl, bandit_advantage, ppo_clipped_objective,
)

torch.set_num_threads(1)
torch.manual_seed(7)
out = {"environment": {"python": sys.version, "torch": torch.__version__,
                       "torch_git_version": torch.version.git_version,
                       "device": "cpu", "cpu_threads": "1"}}
numeric = []
for initial in [0.4, 1.4]:
    value = torch.tensor([initial], requires_grad=True)
    loss = (value - torch.tensor([1.0])).square().mean()
    loss.backward()
    expected_loss = (initial - 1.0) ** 2
    expected_gradient = 2 * (initial - 1.0)
    assert abs(loss.item() - expected_loss) < 1e-6
    assert abs(value.grad.item() - expected_gradient) < 1e-6
    assert value.item() == torch.tensor([initial]).item()
    numeric.append({"initial": initial, "loss": loss.item(), "gradient": value.grad.item(),
                    "expected_loss": expected_loss, "expected_gradient": expected_gradient,
                    "value_unchanged_after_backward": True})
out["mse"] = numeric
probabilities = torch.tensor([[0.8, 0.2], [0.5, 0.5]], dtype=torch.float64)
logits = probabilities.log().requires_grad_()
reference = torch.full_like(logits, 0.5).log().requires_grad_()
kl = exact_kl(logits, reference)
expected_kl = 0.8 * math.log(0.8/0.5) + 0.2 * math.log(0.2/0.5)
assert kl.shape == (2,)
assert abs(kl[0].item() - expected_kl) < 1e-12 and abs(kl[1].item()) < 1e-12
assert torch.allclose(exact_kl(logits + 10, reference - 3), kl, atol=1e-12)
kl.sum().backward()
assert reference.grad is None and logits.grad is not None
out["kl"] = {"values": kl.detach().tolist(), "hand_calculated": expected_kl,
             "axis": "two contexts, two candidates; sum candidates to one KL per context",
             "unit": "nats (natural logarithm)", "reference_grad": None,
             "logits_grad": logits.grad.tolist(), "shift_invariance": True}
invalid = []
for a, b in [(torch.zeros(2), torch.zeros(2)), (torch.zeros(1, 2), torch.zeros(1, 3))]:
    try:
        exact_kl(a, b)
    except ValueError as e:
        invalid.append(str(e))
    else:
        raise AssertionError("invalid input accepted")
out["invalid_shapes"] = invalid

# Extract only the original optimizer method; do not execute the long experiment runner.
implementation = ROOT / 'scripts/course_experiments/posttraining.py'
tree = ast.parse(implementation.read_bytes())
update = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_update')
namespace = {"torch": torch}
exec(compile(ast.Module(body=[update], type_ignores=[]), str(implementation), "exec"), namespace)
policy = FiniteResponsePolicy()
fixed_reference = copy.deepcopy(policy).requires_grad_(False).eval()
reward_model = FiniteRewardModel().requires_grad_(False).eval()
critic = FiniteValueModel()
features = torch.tensor([[0., 1., 0., 1.], [1., 0., 1., 0.], [1., 1., 0., 0.]])
actions = torch.tensor([0, 1, 2])
with torch.no_grad():
    old_selected = policy(features).log_softmax(-1).gather(1, actions[:, None]).squeeze(1).clone()
    old_values = critic(features).clone()
    raw_scores = reward_model(features)
    rewards = (raw_scores - raw_scores.mean(-1, keepdim=True)).gather(1, actions[:, None]).squeeze(1)
    advantages = bandit_advantage(rewards, old_values)
    reference_logits = fixed_reference(features)
before = {"policy": copy.deepcopy(policy.state_dict()), "critic": copy.deepcopy(critic.state_dict()),
          "reference": copy.deepcopy(fixed_reference.state_dict()), "reward": copy.deepcopy(reward_model.state_dict())}
old_selected_before, old_values_before, advantages_before = old_selected.clone(), old_values.clone(), advantages.clone()
optimizer = torch.optim.SGD(policy.parameters(), lr=0.01)
value_optimizer = torch.optim.SGD(critic.parameters(), lr=0.01)
epochs = []
for epoch in range(2):
    new_logits = policy(features)
    selected = new_logits.log_softmax(-1).gather(1, actions[:, None]).squeeze(1)
    terms = ppo_clipped_objective(selected, old_selected, advantages)
    policy_loss = terms['policy_loss'] + 0.1 * exact_kl(new_logits, reference_logits).mean()
    namespace['_update'](optimizer, policy.parameters(), policy_loss)
    value_loss = torch.nn.functional.mse_loss(critic(features), rewards)
    namespace['_update'](value_optimizer, critic.parameters(), value_loss)
    epochs.append({"epoch": epoch+1, "policy_loss": policy_loss.item(), "value_mse": value_loss.item()})
def changed(module, snapshot):
    return any(not torch.equal(module.state_dict()[k], v) for k, v in snapshot.items())
assert changed(policy, before['policy']) and changed(critic, before['critic'])
assert not changed(fixed_reference, before['reference']) and not changed(reward_model, before['reward'])
assert torch.equal(old_selected, old_selected_before) and torch.equal(old_values, old_values_before)
assert torch.equal(advantages, advantages_before) and not advantages.requires_grad
out['bounded_recipe'] = {"contexts": 3, "candidates": 4, "epochs": epochs,
    "policy_changed": True, "critic_changed": True, "reference_unchanged": True,
    "reward_unchanged": True, "old_log_probabilities_unchanged": True,
    "old_values_unchanged": True, "advantages_unchanged": True,
    "critic_target": "sampled centered RM reward only; KL was added to policy loss separately",
    "parameters": {"policy": sum(p.numel() for p in policy.parameters()),
                   "reward_model": sum(p.numel() for p in reward_model.parameters()),
                   "value_model": sum(p.numel() for p in critic.parameters())}}

# Read explicit raw measurements and provenance only; never read attached commentary.
raw_path = ROOT / 'docs/course-experiments/results/posttraining.json'
raw = raw_path.read_bytes()
j = json.loads(raw)
trace = j['results']['ppo']['first_rollout_trace']
fields = ['epoch', 'train_context_indices', 'sampled_actions', 'old_selected_log_probabilities',
          'old_values', 'normalized_rm_rewards', 'fixed_advantages', 'log_probabilities_before_update',
          'ratios_before_update', 'log_probabilities_after_update', 'old_log_probability_sha256',
          'reference_state_sha256']
selected_trace = [{k: row[k] for k in fields} for row in trace]
assert len(selected_trace) == 3
assert all(len(row['sampled_actions']) == 64 for row in selected_trace)
assert all(row['old_values'] == selected_trace[0]['old_values'] for row in selected_trace)
assert all(row['old_selected_log_probabilities'] == selected_trace[0]['old_selected_log_probabilities'] for row in selected_trace)
assert all(row['fixed_advantages'] == selected_trace[0]['fixed_advantages'] for row in selected_trace)
max_adv_error = max(abs(a - (r-v)) for row in selected_trace for a, r, v in zip(
    row['fixed_advantages'], row['normalized_rm_rewards'], row['old_values']))
max_ratio_error = max(abs(r - math.exp(new-old)) for row in selected_trace for r, new, old in zip(
    row['ratios_before_update'], row['log_probabilities_before_update'], row['old_selected_log_probabilities']))
assert max_adv_error < 1e-6 and max_ratio_error < 1e-5
assert j['results']['reference_state_sha256_before'] == j['results']['reference_state_sha256_after']
assert j['results']['parameters'] == out['bounded_recipe']['parameters']
out['existing_measurements'] = {"source_sha256": hashlib.sha256(raw).hexdigest(),
    "provenance": {k:j[k] for k in ['experiment_id','device','seed','torch_version','python_version','code_sha256']},
    "pointers_read": ['/experiment_id','/device','/seed','/torch_version','/python_version','/code_sha256',
        '/results/parameters','/results/ppo/first_rollout_trace/*/{'+','.join(fields)+'}',
        '/results/reference_state_sha256_before','/results/reference_state_sha256_after'],
    "denominators": {"rollout_context_action_samples": 64, "reuse_epochs": 3, "candidate_options": 4},
    "fixed_old_values": True, "fixed_old_log_probabilities": True, "fixed_advantages": True,
    "reference_hash_unchanged": True, "max_advantage_error": max_adv_error, "max_ratio_error": max_ratio_error,
    "first_sample": {k:selected_trace[0][k][0] for k in fields if isinstance(selected_trace[0][k], list)},
    "parameters": j['results']['parameters']}
print(json.dumps(out, ensure_ascii=False, indent=2))
