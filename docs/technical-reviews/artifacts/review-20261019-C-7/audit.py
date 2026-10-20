"""核對凍結算例、既有測試軌跡與有限策略的單步更新；不執行完整訓練。"""

import ast
import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from html.parser import HTMLParser
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch
from scripts.course_experiments import applications as app
from scripts.course_experiments.common import split_records


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_record(path):
    raw = (ROOT / path).read_bytes()
    return {"path": path, "sha256": sha(raw), "bytes": len(raw)}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, value):
        self.parts.append(value)


def original(url, version, markers, excerpt, note, html=False):
    command = ["curl", "-fsSL", url]
    raw = subprocess.check_output(command)
    body = raw.decode()
    if html:
        parser = Text()
        parser.feed(body)
        body = " ".join(parser.parts)
    body = re.sub(r"\s+", " ", body)
    for marker in markers:
        require(marker in body, "原始來源缺少指定定位文字：" + marker)
    require(excerpt in body, "短摘錄與原文不符")
    return {"url": url, "version": version, "accessed_on": "2026-10-20",
            "command": command, "exit_code": 0, "download_sha256": sha(raw),
            "download_bytes": len(raw), "checked_original": True,
            "inspected_markers": markers, "excerpt": excerpt, "inspection_note": note}


result = {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-C-7/audit.py",
          "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                          "torch_git_version": torch.version.git_version, "device": "cpu"}}
command_receipt = json.loads((ARTIFACT / "section-facts-command.txt").read_text())
run = Path(command_receipt["artifact_directory"])
result["section_run"] = {
    "copied_from": str(run.relative_to(ROOT)),
    "files": {name: {"sha256": sha((run / name).read_bytes()), "text": (run / name).read_text()}
              for name in ("stdout.txt", "stderr.txt", "environment.json", "execution.json", "extraction.json", "fence-1.py")}}
require(command_receipt["execution_exit_code"] == 0, "小節程式執行失敗")
source = (ROOT / "docs/course-review-20261019/freeze-01/sources/C.7.md").read_bytes()
require(sha(source) == command_receipt["source_sha256"], "實跑原文與凍結稿不同")
result["toy"] = []
for reward in (1.0, 0.0):
    logits = torch.tensor([0.0, 0.0], requires_grad=True)
    before = logits.detach().softmax(0)
    loss = -(reward - 0.5) * logits.log_softmax(0)[1]
    loss.backward()
    with torch.no_grad():
        updated = logits - 0.1 * logits.grad
        after = updated.softmax(0)
    advantage = reward - 0.5
    expected_gradient = [advantage * 0.5, advantage * -0.5]
    expected_logits = [-0.1 * v for v in expected_gradient]
    expected_probability = [math.exp(v) / sum(map(math.exp, expected_logits)) for v in expected_logits]
    require(logits.grad.tolist() == expected_gradient, "算例梯度不同")
    require(max(abs(a-b) for a,b in zip(after.tolist(), expected_probability)) < 1e-7, "算例機率不同")
    require(not before.requires_grad and not updated.requires_grad and logits.tolist() == [0.0, 0.0], "計算圖狀態不同")
    result["toy"].append({"reward": reward, "baseline": 0.5, "advantage": advantage,
                          "dtype": str(logits.dtype), "gradient": logits.grad.tolist(),
                          "updated_logits": updated.tolist(), "probabilities": after.tolist(),
                          "rounded": [round(v, 6) for v in after.tolist()],
                          "independent_math_probability": expected_probability})

record_path = "docs/course-experiments/results/reasoning.json"
record = json.loads((ROOT / record_path).read_bytes())
result["record"] = {**file_record(record_path),
                    "metadata": {k: record[k] for k in ("revision", "device", "seed", "gpu", "torch_version", "python_version", "elapsed_seconds", "timing_scope", "step_scale", "evidence_status")},
                    "split": record["results"]["split"], "branches": {}}
splits = split_records(app._reasoning_records(), 42)
require(app._split_summary(splits) == record["results"]["split"], "重建資料切分不同")
for label in ("strict", "weak_proxy"):
    branch = record["results"]["reinforce"][label]
    checked = {}
    for stage in ("before", "after"):
        evaluation = branch[stage]
        rows = evaluation["samples"]
        require([(r["a"], r["b"], r["c"]) for r in rows] == [(r["a"], r["b"], r["c"]) for r in splits["test"]], "測試題不同")
        greedy = strict = proxy = enumeration = samples = 0
        for row in rows:
            require(row["truth"] == row["a"] + row["b"] + row["c"], "算術真值不同")
            index = max(range(17), key=lambda i: row["probabilities"][i])
            require(row["greedy_action"] == app._POLICY_ACTIONS[index], "最高機率動作不同")
            correct = int(app._policy_reward(index, row["truth"]))
            require(correct == row["greedy_strict_correct"], "最高機率動作評分不同")
            greedy += correct
            require(len(row["samples"]) == 16, "每題抽樣數不同")
            for sample in row["samples"]:
                action = sample["action_id"]
                require(sample["generated"] == app._POLICY_ACTIONS[action], "動作文字不同")
                sr = int(app._policy_reward(action, row["truth"]))
                pr = int(app._policy_reward(action, row["truth"], True))
                require(sr == sample["strict_reward"] and pr == sample["proxy_reward"], "抽樣評分不同")
                strict += sr
                proxy += pr
                enumeration += action == 16
                samples += 1
        counts = {"greedy_accuracy": (greedy, len(rows)), "sample_accuracy": (strict, samples),
                  "sample_proxy_reward": (proxy, samples), "enumeration_action_rate": (enumeration, samples)}
        for key, (n, d) in counts.items():
            require(evaluation[key] == {"numerator": n, "denominator": d, "rate": n / d}, "重新統計與紀錄不同")
        checked[stage] = {"recounted": counts, "first_question": rows[0]["question"],
                          "first_truth": rows[0]["truth"], "first_greedy": rows[0]["greedy_action"]}
    training = branch["training"]
    require(training["steps"] == 1200 and training["sampled_actions"] == 1200 * 64, "訓練分母不同")
    require([v["step"] for v in training["reward_history"]] == [1] + list(range(100, 1201, 100)), "獎勵紀錄頻率不同")
    checked["training"] = {k: training[k] for k in ("steps", "sampled_actions", "reward_history", "effective_tokens", "checkpoint", "planned_steps", "step_scale")}
    result["record"]["branches"][label] = checked
require(record["results"]["reinforce"]["strict"]["before"] == record["results"]["reinforce"]["weak_proxy"]["before"], "兩支訓練前測試不同")
result["repository_files"] = [file_record(p) for p in (
    "scripts/course_experiments/applications.py", "scripts/course_experiments/common.py", "scripts/course_experiments/run.py",
    "docs/course-experiments/plan.json", "assets/training/manifest.json", ".gitattributes")]
result["record_code_matches_current"] = {p: sha((ROOT / p).read_bytes()) == record["code_sha256"][p]
    for p in ("scripts/course_experiments/applications.py", "scripts/course_experiments/common.py", "scripts/course_experiments/run.py")}
historical_command = ["git", "show", record["revision"] + ":scripts/course_experiments/applications.py"]
historical = subprocess.run(historical_command, cwd=ROOT, capture_output=True, text=True)
result["local_historical_source"] = {"command": historical_command, "exit_code": historical.returncode, "stderr": historical.stderr}
original_url = "https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/" + record["revision"] + "/scripts/course_experiments/applications.py"
original_command = ["curl", "-fsSL", original_url]
original_app = subprocess.check_output(original_command)
require(sha(original_app) == record["code_sha256"]["scripts/course_experiments/applications.py"], "實驗當時原始碼雜湊不符")
current_app = (ROOT / "scripts/course_experiments/applications.py").read_bytes()
def definitions(raw):
    return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(raw).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
old_definitions, current_definitions = definitions(original_app), definitions(current_app)
changed = [name for name, value in old_definitions.items() if value != current_definitions.get(name)]
policy_names = ["_FinitePolicy", "_policy_features", "_policy_reward", "_evaluate_policy", "_reinforce", "_reasoning_records", "run_reasoning", "_fit"]
require(not set(changed).intersection(policy_names), "有限策略相關定義已改變")
result["historical_source"] = {"command": original_command, "url": original_url, "sha256": sha(original_app),
                               "matches_recorded_source": True, "changed_definitions": changed,
                               "policy_definitions_ast_identical": policy_names,
                               "scope": "僅 _sample 的函式 AST 有變；有限動作策略不用該逐 token 生成函式。"}

probes = []
def one_step(policy, loss_fn, ctx, *, steps, lr, name, metadata):
    initial = b"".join(p.detach().numpy().tobytes() for p in policy.parameters())
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)
    loss = loss_fn(0)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    updated = b"".join(p.detach().numpy().tobytes() for p in policy.parameters())
    require(initial != updated, "單步探針沒有更新參數")
    probes.append({"name": name, "requested_steps": steps, "actual_steps": 1, "lr": lr,
                   "initial_parameter_sha256": sha(initial), "updated_parameter_sha256": sha(updated),
                   "loss": float(loss.detach()), "gradient_norm_before_clip": float(gradient_norm), "metadata": metadata})
    return {"probe_steps": 1}

fit_original = app._fit
app._fit = one_step
try:
    probe = app._reinforce(splits, SimpleNamespace(seed=42, device="cpu", output=ARTIFACT))
finally:
    app._fit = fit_original
require(probes[0]["initial_parameter_sha256"] == probes[1]["initial_parameter_sha256"], "策略初始權重不同")
for label in ("strict", "weak_proxy"):
    history = probe[label]["training"]["reward_history"]
    require(history[0]["moving_baseline"] == 0.05 * history[0]["sampled_mean_reward"], "基準值單步更新不同")
result["one_step_probes"] = probes
result["policy_shapes"] = {"input": list(app._policy_features(splits["test"], SimpleNamespace(device="cpu")).shape),
                           "output": list(app._FinitePolicy()(app._policy_features(splits["test"], SimpleNamespace(device="cpu"))).shape)}
help_command = [sys.executable, "scripts/course_experiments/run.py", "--help"]
asset_command = [sys.executable, "scripts/course_experiments/run.py", "--list-assets", "reasoning"]
result["cli"] = {"help_command": help_command, "help_stdout": subprocess.check_output(help_command, cwd=ROOT, text=True),
                  "asset_command": asset_command, "asset_stdout": subprocess.check_output(asset_command, cwd=ROOT, text=True)}
require(result["cli"]["asset_stdout"].strip() == "assets/training/gsm8k-v1.tar.gz", "實驗資料包不同")
result["original_sources"] = [
    original("https://raw.githubusercontent.com/openai/spinningup/20921137141b154454c0a2698709d9f9a0302101/docs/spinningup/rl_intro3.rst",
             "20921137141b154454c0a2698709d9f9a0302101", ["Baselines in Policy Gradients", "only depends on state", "reducing variance", "The Advantage Function"],
             "Any function :math:`b` used in this way is called a **baseline**.",
             "已讀策略梯度推導、負號代價、基準值與 advantage；基準值不依賴所選動作時不改變期望梯度，合適基準值可減少抽樣變異。"),
    original("https://docs.pytorch.org/docs/2.14/distributions.html", "PyTorch 2.14 文件；本機 2.14.1",
             ["Score function", "implement REINFORCE", "optimizers use gradient descent", "Returns entropy of distribution"],
             "Returns entropy of distribution, batched over batch_shape.",
             "已讀 score function 的取樣、log_prob 與負號更新規則，以及 entropy API；軟體細節另以本機 CPU 探針核對。", True),
    original("https://arxiv.org/html/2411.15124v2", "arXiv:2411.15124v2",
             ["Reinforcement Learning on Verifiable Rewards", "deterministic function", "verification function", "We then train against this reward using PPO"],
             "We then train against this reward using PPO.",
             "已讀 RLVR 方法、驗證函式及目標式；支持可驗證答案產生訓練獎勵的術語，不將有限動作策略等同大型語言模型的 PPO 配方或成績。", True),
    original("https://raw.githubusercontent.com/git-lfs/git-lfs/v3.7.1/docs/man/git-lfs-pull.adoc", "Git LFS v3.7.1",
             ["--include=<paths>", "--exclude=<paths>", "Setting either option to an empty string clears the value"],
             "Setting either option to an empty string clears the value.",
             "已讀 pull 的下載與工作樹更新、include/exclude 限制及空字串清除設定；只檢查語法，不下載資料。"),
    original("https://raw.githubusercontent.com/git-lfs/git-lfs/v3.7.1/docs/man/git-lfs-install.adoc", "Git LFS v3.7.1",
             ["--local", "local repository's git config"],
             "git-lfs-install - Install Git LFS configuration.",
             "已讀 install --local 的 repo 設定與 hook 用途；沒有修改本地 Git 設定。"),
    original("https://arxiv.org/html/2305.04388v2", "arXiv:2305.04388v2",
             ["Evaluating Systematic Unfaithfulness", "Counterfactual Simulatability", "only evaluating the plausibility"],
             "CoT explanations can be plausible yet misleading",
             "已讀 Abstract 與 Introduction 的反例及推論限制；可見且合理的解釋不足以辨認輸出的因果依據，未引用其模型成績作本課能力證據。", True)]
result["limitations"] = ["沒有重跑完整訓練、重新取得權重或驗證 GPU 計時；實測數字是對既有原始逐題紀錄的重新統計。",
                         "沒有執行 Git LFS install/pull；以官方手冊、repo attributes、資料 manifest 與 CLI 資產清單核對。",
                         "沒有檢查網站版面或初學讀者理解。"]
print(json.dumps(result, ensure_ascii=False, indent=2))
