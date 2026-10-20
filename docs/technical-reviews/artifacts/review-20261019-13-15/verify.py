"""核對有限選卡範例、既有實驗紀錄與原始來源收據。"""

import ast
import contextlib
import copy
import hashlib
import io
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from scripts.course_experiments.posttraining import CONFIG, build_records, split_records
from scripts.course_experiments.common import records_sha256
from tiny_perceptron.posttraining import FiniteResponsePolicy, FiniteRewardModel, FiniteValueModel


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    with (ARTIFACT / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def fetch(url):
    return subprocess.check_output(["curl", "-fsSL", url])


environment = {"python": platform.python_version(), "torch": str(torch.__version__),
               "torch_git_version": torch.version.git_version, "device": "cpu",
               "platform": platform.platform()}
source = ROOT / "docs/course-review-20261019/freeze-01/sources/13.15.md"
body = source.read_text()
assert sha(source) == "b1612a3b2c22def41bfae9ca6417d0407542da7d28ecd286e80a1b42b09ed98d"
freeze = ROOT / "docs/course-review-20261019/freeze-04.json"
assert sha(freeze) == "a2ad51a05f23d12eaa65cc2e90828302843274a20496624c570426afc611d412"
blocks = re.findall(r"```python\n(.*?)```", body, re.S)
assert len(blocks) == 3
names = {"FiniteResponsePolicy", "FiniteValueModel"}
excerpt = ast.parse(blocks[1])
module = ast.parse((ROOT / "tiny_perceptron/posttraining.py").read_text())
excerpt_classes = {n.name: ast.dump(n, include_attributes=False) for n in excerpt.body if isinstance(n, ast.ClassDef)}
module_classes = {n.name: ast.dump(n, include_attributes=False) for n in module.body if isinstance(n, ast.ClassDef)}
assert all(excerpt_classes[name] == module_classes[name] for name in names)
namespace = {}
capture = io.StringIO()
with contextlib.redirect_stdout(capture):
    for block in blocks:
        exec(compile(block, str(source), "exec"), namespace)
assert namespace["action"].item() == 0
assert abs(namespace["advantage"].item() - (-0.2658)) < 0.00005
assert namespace["terms"]["ratio"].item() == 1.0
assert not namespace["old_log_probability"].requires_grad
assert not namespace["reward"].requires_grad
assert not namespace["advantage"].requires_grad
assert namespace["new_logits"].requires_grad

torch.manual_seed(42)
policy, critic = FiniteResponsePolicy(), FiniteValueModel()
reward_model = FiniteRewardModel().requires_grad_(False)
reference = copy.deepcopy(policy).requires_grad_(False)
x = torch.tensor([[0.1, 0.2, 0.0, 0.0]])
initial = {name: [p.detach().clone() for p in net.parameters()]
           for name, net in (("policy", policy), ("critic", critic), ("reward_model", reward_model), ("reference", reference))}
optimizer = torch.optim.SGD(list(policy.parameters()) + list(critic.parameters()), lr=0.1)
with torch.no_grad():
    distribution = torch.distributions.Categorical(logits=policy(x))
    action = distribution.sample()
    old_log = distribution.log_prob(action)
    rewards = reward_model(x).gather(1, action[:, None]).squeeze(1)
    old_value = critic(x)
advantage = namespace["bandit_advantage"](rewards, old_value)
logits = policy(x)
torch.testing.assert_close(torch.distributions.Categorical(logits=logits).probs, logits.softmax(-1))
torch.testing.assert_close(action[:, None], action.unsqueeze(-1))
terms = namespace["ppo_clipped_objective"](torch.distributions.Categorical(logits=logits).log_prob(action), old_log, advantage)
kl = namespace["exact_kl"](logits, reference(x)).mean()
value_loss = (critic(x) - rewards).square().mean()
torch.testing.assert_close(terms["policy_loss"], -advantage.mean())
torch.testing.assert_close(kl, torch.tensor(0.0))
loss = terms["policy_loss"] + 0.1 * kl + 0.5 * value_loss
optimizer.zero_grad()
loss.backward()
optimizer.step()
changed = {name: [not torch.equal(before, after) for before, after in zip(initial[name], net.parameters())]
           for name, net in (("policy", policy), ("critic", critic), ("reward_model", reward_model), ("reference", reference))}
assert any(changed["policy"]) and any(changed["critic"])
assert not any(changed["reward_model"]) and not any(changed["reference"])
assert all(p.grad is None for net in (reward_model, reference) for p in net.parameters())

critic_a, critic_b = FiniteValueModel(), FiniteValueModel()
critic_b.load_state_dict(critic_a.state_dict())
opt_a = torch.optim.SGD(critic_a.parameters(), lr=0.1)
opt_b = torch.optim.SGD(critic_b.parameters(), lr=0.05)
opt_a.zero_grad()
(0.5 * (critic_a(x) - rewards).square().mean()).backward()
opt_a.step()
opt_b.zero_grad()
((critic_b(x) - rewards).square().mean()).backward()
opt_b.step()
assert all(torch.equal(a, b) for a, b in zip(critic_a.parameters(), critic_b.parameters()))
variant = torch.tensor([[0.1, 0.2, 1.0, 0.0]])
assert list(policy(variant).shape) == [1, 4]
assert list(critic(variant).shape) == [1]
assert list(reward_model(variant).shape) == [1, 4]
target = torch.tensor([[0.85, 0.05, 0.05, 0.05]], dtype=torch.float64)
example_logits = torch.tensor([[0.4, -0.6, 1.2, 0.9]], dtype=torch.float64)
smooth = F.cross_entropy(example_logits, torch.tensor([0]), label_smoothing=0.2)
explicit = -(target * example_logits.log_softmax(-1)).sum(-1).mean()
torch.testing.assert_close(smooth, explicit, atol=1e-14, rtol=0)

record_path = ROOT / "docs/course-experiments/results/posttraining.json"
record = json.loads(record_path.read_bytes())
result = record["results"]
assert result["config"] == CONFIG
bound_code = {name: {"recorded": digest, "observed": sha(ROOT / name)}
              for name, digest in record["code_sha256"].items()}
assert all(v["recorded"] == v["observed"] for v in bound_code.values())
splits = split_records(build_records())
split_summary = {}
families = {}
for name, rows in splits.items():
    selected = {r["family"] for r in rows}
    families[name] = selected
    summary = {"families": sorted(selected), "contexts": len(rows),
               "preference_pairs": sum(len(r["preference_pairs"]) for r in rows),
               "sha256": records_sha256(rows)}
    assert summary == result["splits"][name]
    split_summary[name] = summary
assert not (families["train"] & families["test"] or families["train"] & families["validation"] or families["test"] & families["validation"])
assert len(set.union(*families.values())) == 55
assert [len(splits[n]) for n in ("train", "validation", "test")] == [132, 15, 18]
assert len([r for r in splits["train"] if r["mode"] == "number"]) == result["sft"]["demonstrations"] == 44
sample_rows = {r["mode"]: r for r in splits["test"] if r["family"] == "pair:1:2"}
assert sample_rows["number"]["features"] == [0.1, 0.2, 0, 0]
assert sample_rows["number"]["expected_action"] == 0
assert sample_rows["number"]["candidates"][0] == "3"
assert sample_rows["explain"]["features"] == [0.1, 0.2, 1, 0]
assert sample_rows["explain"]["expected_action"] == 1
parameters = {"policy": sum(p.numel() for p in policy.parameters()),
              "reward_model": sum(p.numel() for p in reward_model.parameters()),
              "value_model": sum(p.numel() for p in critic.parameters())}
assert parameters == result["parameters"] == {"policy": 148, "reward_model": 241, "value_model": 97}
counts = {"reward_pair_draws": CONFIG["reward_steps"] * CONFIG["reward_batch_size"],
          "sampled_actions": CONFIG["ppo_rollout_batches"] * CONFIG["ppo_rollout_size"],
          "policy_updates": CONFIG["ppo_rollout_batches"] * CONFIG["ppo_epochs_per_rollout"],
          "value_updates": CONFIG["ppo_rollout_batches"] * CONFIG["ppo_epochs_per_rollout"],
          "reused_action_draws": CONFIG["ppo_rollout_batches"] * CONFIG["ppo_rollout_size"] * CONFIG["ppo_epochs_per_rollout"]}
assert counts["reward_pair_draws"] == result["reward"]["processed_pair_draws"] == 19200
assert all(counts[k] == result["ppo"][k] for k in counts if k != "reward_pair_draws")
assert result["effective_tokens"] == 0
evaluation = result["evaluations"]["test"]
assert len(evaluation["rows"]) == 18
assert [{k: row[k] for k in splits["test"][i]} for i, row in enumerate(evaluation["rows"])] == splits["test"]
successes = {}
for name in ("sft", "ppo", "dpo"):
    chosen = [max(range(4), key=row["policies"][name]["probabilities"].__getitem__) for row in evaluation["rows"]]
    assert all(a == row["policies"][name]["chosen_action"] for a, row in zip(chosen, evaluation["rows"]))
    count = sum(a == row["expected_action"] for a, row in zip(chosen, evaluation["rows"]))
    assert count == evaluation["policies"][name]["greedy_full_request_success"]["numerator"]
    successes[name] = {"numerator": count, "denominator": len(chosen)}
assert successes["sft"]["numerator"] == 6 and successes["ppo"]["numerator"] == 12
trace = result["ppo"]["first_rollout_trace"]
assert len(trace) == 3
assert all(len(epoch["sampled_actions"]) == 64 for epoch in trace)
assert all(epoch["sampled_actions"] == trace[0]["sampled_actions"] for epoch in trace)
assert all(epoch["old_selected_log_probabilities"] == trace[0]["old_selected_log_probabilities"] for epoch in trace)
assert all(epoch["fixed_advantages"] == trace[0]["fixed_advantages"] for epoch in trace)
assert all(epoch["reference_state_sha256"] == result["reference_state_sha256_before"] for epoch in trace)
assert result["reference_state_sha256_before"] == result["reference_state_sha256_after"]
assert round(record["elapsed_seconds"], 2) == 2.12
assert round(result["ppo"]["seconds"], 2) == 0.66
help_run = subprocess.run([sys.executable, "-m", "scripts.course_experiments.run", "--help"], cwd=ROOT, capture_output=True, text=True, check=True)
assert "--experiment" in help_run.stdout and "--device" in help_run.stdout

run = ROOT / "outputs/reviewer-tools/runs/20261020T060607652968Z"
section_record = {"command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/13.md#13.15" --execute',
                  "copied_from": run.relative_to(ROOT).as_posix(),
                  "stdout": (run / "stdout.txt").read_text(), "stderr": (run / "stderr.txt").read_text(),
                  **{name: json.loads((run / (name + ".json")).read_bytes()) for name in ("environment", "execution", "extraction")}}
assert section_record["execution"]["exit_code"] == 0
save("section-execution.json", section_record)

pin = "5c4886908584029761b579af026dcfb627c84070"
base = "https://raw.githubusercontent.com/pytorch/pytorch/" + pin + "/"
requests = [
    ("categorical", base + "torch/distributions/categorical.py", "Categorical.__init__, sample, log_prob", "self.logits = logits - logits.logsumexp(dim=-1, keepdim=True)"),
    ("no_grad", base + "torch/autograd/grad_mode.py", "no_grad", "Context-manager that disables gradient calculation."),
    ("sgd", base + "torch/optim/sgd.py", "SGD, _single_tensor_sgd", "param.add_(grad, alpha=-lr)"),
    ("smoothing", base + "torch/nn/modules/loss.py", "CrossEntropyLoss.label_smoothing", "The targets\n            become a mixture of the original ground truth and a uniform distribution"),
    ("rlhf", "https://arxiv.org/html/2203.02155v1", "Models; Details of the initialization models for RLHF; Details of RLHF training", "We initialize the RLHF models from a pretrained GPT-3 model"),
]
receipts = []
for name, url, locator, excerpt_text in requests:
    raw = fetch(url)
    text = raw.decode()
    if name == "smoothing":
        assert " ".join(excerpt_text.split()) in " ".join(text.split())
    else:
        assert excerpt_text in text
    receipts.append({"id": name, "url": url, "version": "PyTorch 2.14.1, commit " + pin if name != "rlhf" else "arXiv:2203.02155v1",
                     "accessed_on": "2026-10-20", "command": "curl -fsSL " + url,
                     "full_response_sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                     "locator": locator, "excerpt": excerpt_text})
paper_url = "https://arxiv.org/pdf/1707.06347v2"
raw = fetch(paper_url)
assert raw.startswith(b"%PDF")
receipts.append({"id": "ppo", "url": paper_url, "version": "arXiv:1707.06347v2, 2017-08-28",
                 "accessed_on": "2026-10-20", "command": "curl -fsSL " + paper_url,
                 "full_response_sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                 "locator": "Clipped Surrogate Objective; Algorithm; Comparison of Surrogate Objectives",
                 "excerpt": "so r(θold) = 1",
                 "inspection": "PDF 原文以 web open 讀取：ratio 定義、min 與 clip 目標、平方 value loss、固定舊策略採樣再重用；curl 僅在記憶體保留全文以產生原檔收據。"})
save("original-sources.json", {"sources": receipts, "full_text_saved": False})
out = {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-13-15/verify.py",
       "environment": environment, "result": "所有斷言通過；未執行完整訓練。", "source_sha256": sha(source),
       "source_freeze_sha256": sha(freeze), "excerpt_ast_matches": True,
       "section_stdout": capture.getvalue(), "advantage": namespace["advantage"].item(),
       "reward": rewards.item(), "old_value": old_value.item(), "ratio": terms["ratio"].item(),
       "initial_exact_kl": kl.item(), "changed_parameter_groups": changed,
       "critic_half_gradient_equals_half_sgd_lr": True,
       "label_smoothing": {"target": target.tolist(), "loss": smooth.item(), "explicit_loss": explicit.item()},
       "variant_shapes": {"policy": list(policy(variant).shape), "critic": list(critic(variant).shape), "reward_model": list(reward_model(variant).shape)},
       "bound_code": bound_code, "experiment_record_sha256": sha(record_path), "split_summary": split_summary,
       "parameters": parameters, "counts": counts, "test_successes": successes,
       "fixed_first_rollout_reuse": True, "recorded_environment": {k: record[k] for k in ("python_version", "torch_version", "device", "seed")},
       "historical_timing": {"elapsed_seconds": record["elapsed_seconds"], "ppo_seconds": result["ppo"]["seconds"], "cpu_threads": result["config"]["cpu_threads"],
                             "scope": "既有單次 CPU 紀錄，含 DPO；未重新量測，不能外推到其他機器。"},
       "cli_help": {"command": ".venv/bin/python -m scripts.course_experiments.run --help", "returncode": help_run.returncode,
                    "stdout": help_run.stdout, "stderr": help_run.stderr},
       "derivations": ["policy 參數 = (4×16+16)+(16×4+4)=148；reward = (8×24+24)+(24+1)=241；value = (4×16+16)+(16+1)=97。",
                       "家族為允許重複的無序正整數對，10+9+…+1=55；每家族三題；每家族的偏好對 6+6+3=15。",
                       "value 以同情境採樣 reward 的平方誤差訓練；E[(v-R)^2|x]=(v-E[R|x])^2+Var(R|x)，最小值在條件平均；未訓練的網路不保證等於平均。",
                       "參數互不共用，策略項對 critic 的梯度為零；∇critic(0.5 Lvalue)=0.5∇critic Lvalue，無 momentum 與 weight decay 的 SGD 可把 0.5 併入學習率。",
                       "q=(1-0.2)one_hot+0.2/4=[0.85,0.05,0.05,0.05]；最小化交叉熵會以 q 為目標，不保證有限步模型輸出恰為 q。"],
       "limitations": ["未重跑完整訓練或取得既有 checkpoint；逐題概率與統計只能驗證已保存紀錄的內部一致性。",
                       "只在目前 CPU/Python/PyTorch 環境重現隨機範例；seed 不構成跨版本與跨裝置逐位元一致的契約。",
                       "不驗證中文理解、算術能力、逐 token 生成、跨 seed 品質或加速；本節也未提出這些結論。"]}
save("verification.json", out)
print(json.dumps({"result": out["result"], "parameters": parameters, "counts": counts, "test_successes": successes,
                  "changed_parameter_groups": changed, "artifact_files": ["verify.py", "section-execution.json", "original-sources.json", "verification.json"]}, ensure_ascii=False))
