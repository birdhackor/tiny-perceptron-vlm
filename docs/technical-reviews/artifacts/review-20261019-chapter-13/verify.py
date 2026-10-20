"""核對凍結章首的連結、函式、選卡流程與原始來源收據。"""

import ast
import contextlib
import hashlib
import io
import json
import math
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

import torch

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from tiny_perceptron import alignment, posttraining
from tiny_perceptron.data import IGNORE


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sections(raw):
    headings = list(re.finditer(r"^## (\S+) (.+)$", raw, re.M))
    return {
        item[1]: {
            "title": item[2],
            "body": raw[item.start() : headings[i + 1].start() if i + 1 < len(headings) else len(raw)],
        }
        for i, item in enumerate(headings)
    }


chapter_path = Path("course/chapters/13.md")
chapter = chapter_path.read_text()
frozen_path = Path("docs/course-review-20261019/freeze-01/sources/chapter-13.md")
frozen = frozen_path.read_bytes()
assert digest(frozen) == "0cd233032a85525bf85a753d35f23384d83223920d303b58ab2c7b6c583ca2a9"
assert chapter.split("## ", 1)[0].encode() == frozen
freeze = json.loads(Path("docs/course-review-20261019/freeze-04.json").read_text())
page = next(p for p in freeze["pages"] if p["page_id"] == "chapter-13")
assert page["figures_sha256"] == {}
assert not re.search(r"!\[", frozen.decode())

links = []
for href in re.findall(r"\[[^\]]*\]\(([^)]+)\)", frozen.decode()):
    relative, _, anchor = href.partition("#")
    target = (chapter_path.parent / relative).resolve() if relative else chapter_path.resolve()
    assert target.is_file(), href
    target_text = target.read_text()
    target_sections = sections(target_text)
    assert not anchor or anchor in target_sections, href
    links.append({"href": href, "path": target.relative_to(ROOT).as_posix(), "title": target_sections[anchor]["title"] if anchor else target_text.splitlines()[0]})

chapter_sections = sections(chapter)
route_titles = {key: value["title"] for key, value in chapter_sections.items()}
prior = sections(Path("course/chapters/07.md").read_text())
prior_match = next(s for s in prior.values() if s["title"] == "後訓練有哪些不同教法？")
assert all(term in prior_match["body"] for term in ["DPO", "RLHF", "PPO"])
warmup = sections(Path("course/chapters/0C.md").read_text())
assert "等讀完" in Path("course/chapters/0C.md").read_text().split("## ", 1)[0]
warmup_titles = {key: value["title"] for key, value in warmup.items()}
glossary = Path("course/glossary.md").read_text()
assert all(term in glossary for term in ["DPO", "PPO", "reward", "advantage", "KL"])

definitions = {}
for code in re.findall(r"```python\n(.*?)```", chapter, re.S):
    for node in ast.parse(code).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            definitions[node.name] = ast.dump(node, include_attributes=False)
matched = []
for module, names in [
    (alignment, ["sequence_log_probability", "dpo_loss"]),
    (posttraining, ["preference_loss", "FiniteRewardModel", "bandit_advantage", "ppo_clipped_objective", "exact_kl", "FiniteResponsePolicy", "FiniteValueModel"]),
]:
    implementation = ast.parse(Path(module.__file__).read_text())
    for name in names:
        node = next(node for node in implementation.body if getattr(node, "name", None) == name)
        assert definitions[name] == ast.dump(node, include_attributes=False), name
        matched.append(name)

torch.set_num_threads(1)
logits = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [2.0, 0.0]]], dtype=torch.float64)
labels = torch.tensor([[0, 1, IGNORE]])
sequence = alignment.sequence_log_probability(logits, labels)
assert torch.allclose(sequence, torch.tensor([-2 * math.log(2)], dtype=torch.float64))
chosen = torch.tensor([-4.0], requires_grad=True)
rejected = torch.tensor([-3.0], requires_grad=True)
reference_chosen = torch.tensor([-4.0], requires_grad=True)
reference_rejected = torch.tensor([-3.0], requires_grad=True)
dpo = alignment.dpo_loss(chosen, rejected, reference_chosen, reference_rejected)
dpo.backward()
assert abs(dpo.item() - math.log(2)) < 1e-6
assert abs(chosen.grad.item() + 0.05) < 1e-6
assert abs(rejected.grad.item() - 0.05) < 1e-6
assert reference_chosen.grad is None and reference_rejected.grad is None
preference = posttraining.preference_loss(torch.tensor([2.0]), torch.tensor([0.0]))
assert abs(preference.item() - math.log1p(math.exp(-2))) < 1e-6
rewards = torch.tensor([1.0, 0.0], requires_grad=True)
values = torch.tensor([0.4, 0.5], requires_grad=True)
advantages = posttraining.bandit_advantage(rewards, values)
assert torch.allclose(advantages, torch.tensor([0.6, -0.5])) and not advantages.requires_grad
ratio_input = torch.tensor([1.5, 0.5, 0.5, 1.5], dtype=torch.float64)
terms = posttraining.ppo_clipped_objective(ratio_input.log(), torch.zeros(4, dtype=torch.float64), torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.float64))
expected_surrogate = torch.tensor([1.2, -0.8, 0.5, -1.5], dtype=torch.float64)
assert torch.allclose(terms["surrogate"], expected_surrogate)
p = torch.tensor([[0.8, 0.2]], dtype=torch.float64)
q = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
kl = posttraining.exact_kl(p.log(), q.log())
assert torch.allclose(kl, (p * (p / q).log()).sum(-1))

complete = next(item for item in chapter_sections.values() if item["title"] == "一輪選卡與更新，如何接起來？")
runtime = {"__name__": "__review_probe__"}
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    for code in re.findall(r"```python\n(.*?)```", complete["body"], re.S):
        exec(compile(code, str(chapter_path), "exec"), runtime)
assert not torch.equal(runtime["before"], next(runtime["policy"].parameters()))
assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in runtime["critic"].parameters())
assert all(parameter.grad is None for name in ["reward_model", "reference"] for parameter in runtime[name].parameters())
assert runtime["new_logits"].shape == (1, 4)
assert runtime["terms"]["ratio"].item() == 1.0


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.bits = []

    def handle_data(self, data):
        self.bits.append(data)


paper_inputs = [
    ("dpo", "https://arxiv.org/html/2305.18290v3", "2305.18290v3, 2024-07-29", "4 Direct Preference Optimization", "our goal is to derive a simple approach for policy optimization using preferences directly.", "已讀 Preliminaries、Direct Preference Optimization 與目標式：同題偏好對可學獎勵，也可改寫為直接調整策略的代價；兩者不必依序執行。"),
    ("ppo", "https://arxiv.org/pdf/1707.06347v2", "1707.06347v2, 2017-08-28", "Clipped Surrogate Objective; Algorithm; equations (7), (9)-(12); Algorithm 1", None, "curl 取得原始 PDF 並計算原檔 SHA；另以 web PDF 文字擷取讀取 Clipped Surrogate Objective 與 Algorithm。機率比乘優勢與裁切取小、價值平方誤差、採樣後更新相符；單步終止時優勢化為 reward 減舊 value。"),
    ("reward", "https://arxiv.org/html/2210.10760v1", "2210.10760v1, 2022-10-19", "Abstract; Introduction", "Because the reward model is an imperfect proxy, optimizing its value too much can hinder ground truth performance", "已讀摘要與導論：獎勵模型是代理目標，過度最佳化可降低另一個 gold 模型的評分。實驗使用合成 gold 評分，不把它誤作所有真實答案的正確率定理。"),
    ("compute", "https://arxiv.org/html/2407.21787v3", "2407.21787v3, 2024-12-30", "Abstract; Discussion and Limitations", "fail to fully scale with the sample budget.", "已讀摘要與 Discussion and Limitations：增加抽樣能提高候選集合命中率，能否辨識正解是另一件事。此為回答時多花計算的一種方式，不是任何服務的 thinking effort 參數測試。"),
]
papers = []
for identifier, url, version, locator, quote, note in paper_inputs:
    command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "30", url]
    response = subprocess.run(command, capture_output=True, check=True)
    if quote:
        parser = Text()
        parser.feed(response.stdout.decode())
        normalized = " ".join(" ".join(parser.bits).split())
        assert quote in normalized, identifier
        assert len(quote.split()) <= 25
    else:
        assert response.stdout.startswith(b"%PDF-")
    papers.append({"id": identifier, "url": url, "version": version, "accessed_on": datetime.now(UTC).date().isoformat(), "command": command, "exit_code": response.returncode, "original_sha256": digest(response.stdout), "bytes": len(response.stdout), "locator": locator, "short_excerpt": quote, "inspection_note": note})

paths = ["course/chapters/13.md", "course/chapters/07.md", "course/chapters/0C.md", "course/glossary.md", "tiny_perceptron/alignment.py", "tiny_perceptron/posttraining.py", str(frozen_path), "docs/course-review-20261019/freeze-04.json"]
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python docs/technical-reviews/artifacts/review-20261019-chapter-13/verify.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "platform": platform.platform(), "device": "cpu", "dtype": "float32; formula probes also float64", "torch_threads": str(torch.get_num_threads()), "uv": subprocess.run(["uv", "--version"], capture_output=True, text=True, check=True).stdout.strip()},
    "repository_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
    "file_sha256": {path: digest(Path(path).read_bytes()) for path in paths},
    "frozen_intro_matches_current": True,
    "figures": [],
    "links": links,
    "route_titles": route_titles,
    "warmup_titles": warmup_titles,
    "previous_teaching_methods_include": [term for term in ["DPO", "RLHF", "PPO"] if term in prior_match["body"]],
    "glossary_terms_found": [term for term in ["DPO", "PPO", "reward", "advantage", "KL"] if term in glossary],
    "matching_implementation_definitions": matched,
    "function_probes": {"sequence_log_probability": sequence.tolist(), "dpo_loss": dpo.item(), "dpo_gradients": [chosen.grad.item(), rejected.grad.item()], "reference_gradients_absent": True, "preference_loss": preference.item(), "bandit_advantage": advantages.tolist(), "ppo_surrogate": terms["surrogate"].tolist(), "exact_kl": kl.tolist()},
    "complete_selection_update": {"stdout": captured.getvalue(), "policy_changed": True, "critic_received_nonzero_gradient": True, "reward_and_reference_gradients_absent": True, "selected_action": runtime["action"].tolist(), "loss_finite": bool(torch.isfinite(runtime["loss"]))},
    "original_sources": papers,
    "result": "所有斷言通過；未執行完整訓練，未驗證模型品質或跨後端一致性。",
}
print(json.dumps(result, ensure_ascii=False, indent=2))
