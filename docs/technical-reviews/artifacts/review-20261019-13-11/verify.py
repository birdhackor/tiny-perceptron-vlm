"""核對單步優勢的數值、梯度與程式邊界；來源全文只在記憶體處理。"""

import argparse
import ast
import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))


class OriginalText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ("script", "style"):
            self.skip.append(attrs.get("type", "") not in ("math/tex", "math/tex; mode=display"))
        if tag == "img" and attrs.get("alt"):
            self.parts.append(attrs["alt"])

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self.skip:
            self.skip.pop()

    def handle_data(self, data):
        if not any(self.skip) and data.strip():
            self.parts.append(data.strip())


def sources():
    specifications = [
        {
            "id": "policy-gradient",
            "url": "https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html",
            "version": "Spinning Up revision 038665d62d569055401d91856abb287263096178",
            "locator": "Baselines in Policy Gradients; Other Forms of the Policy Gradient / The Advantage Function",
            "quotes": ["which only depends on state", "The most common choice of baseline is the on-policy value function"],
            "required": ["038665d6", "mean-squared-error objective", "Q^{\\pi}(s_t,a_t) - V^{\\pi}(s_t)"],
            "inspection": "已讀 baseline 的 action-independent 條件、扣除 baseline 不改變期望梯度、V 的平均回報定義、value network 的平方誤差回歸與 A=Q−V。單步回報是 R，故 R−V_old 是採樣優勢估計；真實期望優勢仍為 E[R|s,a]−V(s)。",
        },
        {
            "id": "ppo",
            "url": "https://spinningup.openai.com/en/latest/algorithms/ppo.html",
            "version": "Spinning Up revision 038665d62d569055401d91856abb287263096178",
            "locator": "Key Equations; Advantage is positive; Advantage is negative; Pseudocode",
            "quotes": ["typically taking multiple steps of (usually minibatch) SGD to maximize the objective."],
            "required": ["038665d6", "Compute advantage estimates", "\\pi_{\\theta}(a|s)}{\\pi_{\\theta_k}(a|s)", "Fit value function by regression on mean-squared error"],
            "inspection": "已讀同一 s,a 的新舊策略比、舊策略下的固定 A、正負優勢對單樣本目標的方向、同批資料多次更新與 value 回歸。本文只說處理重用時的變化，沒有宣稱 clipping 強制保證策略距離。",
        },
        {
            "id": "instructgpt",
            "url": "https://arxiv.org/html/2203.02155v1",
            "version": "arXiv:2203.02155v1, 2022-03-04; arXiv HTML rendering accessed 2026-10-20",
            "locator": "Models / Reward modeling (RM), Reinforcement learning (RL); Details of RLHF training",
            "quotes": ["a scalar reward", "we add a per-token KL penalty", "is a bandit environment"],
            "required": ["2203.02155v1", "generalized advantage", "normalize the reward model using a bias"],
            "inspection": "已讀偏好 RM 的純量輸出與分數差損失、RM 平移平均值、prompt-response bandit 與 token KL，以及 RLHF 的 GAE 設定。純量獎勵不受二元標籤的 0/1 範圍限制。論文本身也稱完整回答環境為 bandit，因此不能把 bandit 這個名稱當作排除逐 token RLHF 的條件；此頁的限制在預寫四卡與沒有逐 token 決策。",
        },
        {
            "id": "detach",
            "url": "https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.detach.html",
            "version": "PyTorch 2.14 documentation; runtime torch 2.14.1",
            "locator": "torch.Tensor.detach API description and Note",
            "quotes": ["The result will never require gradient."],
            "required": ["shares the same storage", "forward mode AD gradients"],
            "inspection": "已讀 detach 不再要求梯度、也不保留 forward-mode AD 梯度與共享原儲存體的條件。本函式對相減的新結果 detach，並沒有宣稱整個 critic 被永久凍結，也沒有把 detach 當成獨立複製。",
        },
    ]
    records = []
    for spec in specifications:
        command = ["curl", "-L", "--fail", "--max-time", "40", "-sS", spec["url"]]
        result = subprocess.run(command, capture_output=True, check=True)
        parser = OriginalText()
        parser.feed(result.stdout.decode("utf-8"))
        text = " ".join(" ".join(parser.parts).split())
        for phrase in spec.pop("required") + spec["quotes"]:
            if phrase not in text:
                raise AssertionError("原文定位字串不存在：" + phrase)
        records.append({**spec, "command_argv": command, "accessed_at_utc": datetime.now(UTC).isoformat(), "fetched_bytes": len(result.stdout), "fetched_sha256": hashlib.sha256(result.stdout).hexdigest(), "curl_exit_code": result.returncode, "checked_original": True})
    return {"records": records, "storage_policy": "僅保存定位、短摘錄與人工核對說明；第三方全文未寫入檔案。"}


def verify():
    import torch
    from scripts.course_experiments.posttraining import build_records
    from tiny_perceptron.posttraining import (
        FiniteResponsePolicy,
        FiniteRewardModel,
        FiniteValueModel,
        bandit_advantage,
        ppo_clipped_objective,
    )

    torch.manual_seed(42)
    torch.set_num_threads(1)
    rows = []
    for baseline in (0.4, 0.9):
        rewards = torch.tensor([1.0, 0.0], requires_grad=True)
        old_values = torch.full((2,), baseline, requires_grad=True)
        advantage = bandit_advantage(rewards, old_values)
        expected = [1 - baseline, -baseline]
        torch.testing.assert_close(advantage, torch.tensor(expected), atol=1e-7, rtol=0)
        assert not advantage.requires_grad and advantage.grad_fn is None
        rows.append({"baseline": baseline, "raw": advantage.tolist(), "rounded": [round(v, 2) for v in advantage.tolist()], "requires_grad": advantage.requires_grad, "dtype": str(advantage.dtype)})

    error_cases = []
    for rewards, values in [(torch.ones(2), torch.ones(1)), (torch.empty(0), torch.empty(0)), (torch.ones(2, 1), torch.ones(2))]:
        try:
            bandit_advantage(rewards, values)
        except ValueError as error:
            error_cases.append({"reward_shape": list(rewards.shape), "value_shape": list(values.shape), "exception": type(error).__name__, "message": str(error)})
        else:
            raise AssertionError("無效形狀未被拒絕")

    directions = []
    for reward, baseline in [(0.0, 0.0), (0.0, 0.4), (1.0, 0.4)]:
        logits = torch.zeros(4, requires_grad=True)
        before = logits.softmax(-1)[0].item()
        coefficient = bandit_advantage(torch.tensor(reward), torch.tensor(baseline))
        loss = -logits.log_softmax(-1)[0] * coefficient
        loss.backward()
        with torch.no_grad():
            after = (logits - 0.1 * logits.grad).softmax(-1)[0].item()
        assert (after == before) if coefficient.item() == 0 else ((after < before) if coefficient.item() < 0 else (after > before))
        directions.append({"reward": reward, "baseline": baseline, "coefficient": coefficient.item(), "selected_logit_gradient": logits.grad[0].item(), "probability_before": before, "probability_after": after})

    rewards = torch.tensor([1.0, 0.0], requires_grad=True)
    critic = torch.tensor([0.4, 0.4], requires_grad=True)
    old_log_prob = torch.full((2,), -1.0, requires_grad=True)
    new_log_prob = torch.full((2,), -1.0, requires_grad=True)
    advantages = bandit_advantage(rewards, critic)
    terms = ppo_clipped_objective(new_log_prob, old_log_prob, advantages)
    terms["policy_loss"].backward()
    gradient_boundary = {"advantage_requires_grad": advantages.requires_grad, "reward_grad": rewards.grad, "critic_grad_from_policy_loss": critic.grad, "old_log_prob_grad": old_log_prob.grad, "new_log_prob_grad": new_log_prob.grad.tolist(), "ratio": terms["ratio"].tolist()}
    assert rewards.grad is None and critic.grad is None and old_log_prob.grad is None
    value_loss = (critic - rewards.detach()).square().mean()
    value_loss.backward()
    gradient_boundary["critic_grad_from_value_loss"] = critic.grad.tolist()
    assert critic.grad is not None

    records = build_records()
    assert all(len(row["features"]) == 4 and len(row["candidates"]) == 4 for row in records)
    features = torch.tensor([[0.1, 0.2, 0.0, 0.0]])
    policy, value, reward_model = FiniteResponsePolicy(), FiniteValueModel(), FiniteRewardModel()
    logits, predicted_values, reward_scores = policy(features), value(features), reward_model(features)
    probabilities = logits.softmax(-1)
    assert logits.shape == (1, 4) and predicted_values.shape == (1,) and reward_scores.shape == (1, 4)
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(1))
    selected = next(row for row in records if row["operands"] == [1, 2] and row["mode"] == "number")

    fixed = bandit_advantage(torch.tensor([1.0]), torch.tensor([0.4]))
    old = torch.tensor([0.25]).log()
    reuse = []
    for probability in (0.25, 0.275):
        current = torch.tensor([probability], requires_grad=True).log()
        objective = ppo_clipped_objective(current, old, fixed)
        reuse.append({"new_probability": probability, "ratio": objective["ratio"].item(), "advantage": fixed.item(), "loss": objective["policy_loss"].item()})
    torch.testing.assert_close(torch.tensor([item["ratio"] for item in reuse]), torch.tensor([1.0, 1.1]))

    code_receipts = []
    for path, identifiers in [("tiny_perceptron/posttraining.py", ["bandit_advantage", "FiniteResponsePolicy", "FiniteValueModel"]), ("scripts/course_experiments/posttraining.py", ["build_records", "run_posttraining"])]:
        raw = (ROOT / path).read_bytes()
        tree = ast.parse(raw)
        excerpts = {node.name: ast.get_source_segment(raw.decode(), node) for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in identifiers}
        if path.startswith("scripts/"):
            function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and any(isinstance(item, ast.Call) and isinstance(item.func, ast.Name) and item.func.id == "bandit_advantage" for item in ast.walk(node)))
            code = ast.get_source_segment(raw.decode(), function)
            lines = code.splitlines()
            start = next(i for i, line in enumerate(lines) if 'ppo = copy.deepcopy' in line)
            end = next(i for i, line in enumerate(lines[start:], start) if 'if rollout == 0:' in line)
            excerpts = {function.name + " / PPO採樣與兩種更新": "\n".join(lines[start:end])}
        code_receipts.append({"path": path, "sha256": hashlib.sha256(raw).hexdigest(), "excerpts": excerpts})

    frozen = (ROOT / "docs/course-review-20261019/freeze-01/sources/13.11.md").read_bytes()
    current = (ROOT / "course/chapters/13.md").read_bytes()
    start = current.index(b"## 13.11 ")
    end = current.index(b"\n## ", start)
    # 下一標題前的換行仍屬於本節。
    assert current[start:end + 1] == frozen
    return {"environment": {"python": sys.version, "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": "cpu", "platform": platform.platform()}, "numeric_cases": rows, "shape_errors": error_cases, "single_sample_directions": directions, "gradient_boundary": gradient_boundary, "finite_task": {"all_records_have_four_features_and_cards": True, "example": selected, "policy_logits_shape": list(logits.shape), "value_shape": list(predicted_values.shape), "reward_shape": list(reward_scores.shape), "probabilities": probabilities.tolist()}, "fixed_advantage_reuse": reuse, "repository_receipts": code_receipts, "freeze_matches_current_section": True, "result": "全部斷言通過"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", action="store_true")
    args = parser.parse_args()
    print(json.dumps(sources() if args.sources else verify(), ensure_ascii=False, indent=2))
