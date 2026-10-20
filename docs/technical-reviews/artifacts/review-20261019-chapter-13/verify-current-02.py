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
freeze = json.loads(Path("docs/course-review-20261019/freeze-05.json").read_text())
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

paths = ["course/chapters/13.md", "course/chapters/07.md", "course/chapters/0C.md", "course/glossary.md", "tiny_perceptron/alignment.py", "tiny_perceptron/posttraining.py", str(frozen_path), "docs/course-review-20261019/freeze-05.json"]
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python docs/technical-reviews/artifacts/review-20261019-chapter-13/verify-current-02.py",
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


original_path = Path("docs/course-review-20261019/reviews/rv1019_technical_chapter-13/technical-chapter-13-original-01.json")
assert digest(original_path.read_bytes()) == "1f1029ae251fe3d8175369f4f7b0839e5ebafe0230c67690e97eb1ca5d3d9978"
original = json.loads(original_path.read_text())
prior_evidence = {}
for artifact in original["artifacts"]:
    actual_digest = digest(Path(artifact["path"]).read_bytes())
    assert actual_digest == artifact["sha256"]
    prior_evidence[artifact["path"]] = actual_digest
prior_self_check = Path("docs/technical-reviews/artifacts/review-20261019-chapter-13/self-check.json")
assert digest(prior_self_check.read_bytes()) == "4942ded41a4d9f627a029b7453ac69df06bc4256949d74fe50916f2216d4a967"
assert digest(Path("docs/course-review-20261019/freeze-05.json").read_bytes()) == "e3dbd6ee0acaf92438950504795acda172b72ac17dd610a02122e2bfdfcba6ba"
assert not original["issues"]

changed_paths = ["course/chapters/13.md", "course/chapters/07.md", "course/chapters/0C.md", "course/glossary.md"]
diff_command = ["git", "diff", "5ef56bc756f3", "3af7f13ef959", "--", *changed_paths]
difference = subprocess.run(diff_command, capture_output=True, text=True, check=True)
versions = {}
for path in changed_paths:
    old_blob = subprocess.run(["git", "show", f"5ef56bc756f3:{path}"], capture_output=True, check=True).stdout
    new_blob = subprocess.run(["git", "show", f"3af7f13ef959:{path}"], capture_output=True, check=True).stdout
    old_source = next(source for source in original["sources"] if source.get("path") == path)
    assert digest(old_blob) == old_source["sha256"]
    assert new_blob == Path(path).read_bytes()
    versions[path] = {"old_sha256": digest(old_blob), "new_sha256": digest(new_blob), "current_matches_diff_target": True}


def execute_heading(collection, title, replacements=()):
    body = next(item["body"] for item in collection.values() if item["title"] == title)
    namespace = {"__name__": "__review_probe__"}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        for block in re.findall(r"```python\n(.*?)```", body, re.S):
            for before, after in replacements:
                block = block.replace(before, after)
            exec(compile(block, title, "exec"), namespace)
    return namespace, output.getvalue()


score_namespace, score_stdout = execute_heading(chapter_sections, "逐字機率如何合成整篇答案的分數？")
ratio_namespace, ratio_stdout = execute_heading(chapter_sections, "同一批作答重用時，如何記住它來自哪個策略？")
alignment_namespace, alignment_stdout = execute_heading(prior, "回答第一個 token 在哪個位置被預測？")
padding_namespace, padding_stdout = execute_heading(prior, "填充的位置會被當線索嗎？")
assert score_namespace["score"].shape == (1,)
assert torch.allclose(ratio_namespace["ratio"], torch.tensor([2.0, 0.5]))
from tiny_perceptron.model import TinyLM, ModelConfig, masked_loss
from torch.nn import functional as F
ids = alignment_namespace["x"].unsqueeze(0)
targets = alignment_namespace["y"].unsqueeze(0)
model = padding_namespace["model"]
out = TinyLM(ModelConfig(vocab_size=264, width=8))(ids)["logits"]
assert out.shape[:2] == ids.shape
assert torch.allclose(masked_loss(out, targets), F.cross_entropy(out.reshape(-1, out.shape[-1]), targets.reshape(-1), ignore_index=IGNORE))
padding_results = []
for position in [0, model.config.max_length - 1]:
    test_positions = padding_namespace["positions"].clone()
    test_positions[:, :2] = position
    actual = model(padding_namespace["ids"], valid=padding_namespace["valid"], positions=test_positions)["logits"][:, 2:]
    maximum_difference = (actual - padding_namespace["base"]).abs().max().item()
    assert torch.allclose(actual, padding_namespace["base"], atol=1e-6, rtol=0)
    padding_results.append({"padding_position": position, "maximum_real_token_difference": maximum_difference})
invalid_positions = padding_namespace["positions"].clone()
invalid_positions[:, :2] = -1
try:
    model(padding_namespace["ids"], valid=padding_namespace["valid"], positions=invalid_positions)
except (IndexError, RuntimeError) as error:
    invalid_position_error = type(error).__name__ + ": " + str(error)
else:
    raise AssertionError("位置表應拒絕範圍外的索引")

reasoning_namespace, reasoning_stdout = execute_heading(warmup, "最後答對代表過程正確嗎？", [("steps = [(2, 2, 5), (5, -1, 4)]", "steps = [(2, 2, 4), (4, 3, 7), (7, -3, 4)]")])
assert all(reasoning_namespace["valid"])
assert all(reasoning_namespace["linked"])
assert reasoning_namespace["steps"][-1][2] == reasoning_namespace["truth"]
assert sorted(reasoning_namespace["used"]) != sorted(reasoning_namespace["question"])
applications_path = Path("scripts/course_experiments/applications.py")
verifier = next(node for node in ast.parse(applications_path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "_verify_reasoning")
verifier_namespace = {"re": re}
exec(compile(ast.Module(body=[verifier], type_ignores=[]), str(applications_path), "exec"), verifier_namespace)
verify_reasoning = verifier_namespace["_verify_reasoning"]
question = {"a": 5, "b": 0, "c": 3, "truth": 8}
verifier_results = {}
for text in ["5+0=5;5+3=8;answer=8", "5+0=5;5+3=10;answer=10", "5+1=6;6+2=8;answer=8"]:
    verifier_results[text] = verify_reasoning({"generated": text, "invalid_special_tokens": []}, question, "steps")
assert verifier_results["5+0=5;5+3=8;answer=8"]["fully_verified"]
assert not verifier_results["5+0=5;5+3=10;answer=10"]["equations_valid"]
assert not verifier_results["5+1=6;6+2=8;answer=8"]["task_operands_valid"]

clipped_example = posttraining.ppo_clipped_objective(torch.tensor([2.0]).log(), torch.zeros(1), torch.tensor([0.6]))
assert abs(clipped_example["surrogate"].item() - 0.72) < 1e-6
large_update_logits = torch.nn.Parameter(torch.zeros(2))
old_log = large_update_logits.log_softmax(0)[0].detach()
optimizer_probe = torch.optim.SGD([large_update_logits], lr=10.0)
objective = posttraining.ppo_clipped_objective(large_update_logits.log_softmax(0)[0:1], old_log.reshape(1), torch.ones(1))
optimizer_probe.zero_grad()
objective["policy_loss"].backward()
optimizer_probe.step()
ratio_after = (large_update_logits.log_softmax(0)[0] - old_log).exp().item()
assert ratio_after > 1.2
coverage_endpoints = {str(p): [1 - (1 - p) ** count for count in [1, 2, 8]] for p in [0, 1]}
assert coverage_endpoints == {"0": [0, 0, 0], "1": [1, 1, 1]}
round_results = {"four_places": round(1.23456, 4), "ties": [round(0.5), round(1.5), round(2.5)]}
assert round_results == {"four_places": 1.2346, "ties": [0, 2, 2]}
emoji = "👍🏽"
assert [ord(character) for character in emoji] == [0x1F44D, 0x1F3FD]
scalar_values = [0, 0x7F, 0x80, 0x7FF, 0x800, 0xD7FF, 0xE000, 0xFFFF, 0x10000, 0x10FFFF]
encoding_lengths = {f"U+{value:04X}": len(chr(value).encode("utf-8")) for value in scalar_values}
surrogate_errors = []
for value in [0xD800, 0xDFFF]:
    try:
        chr(value).encode("utf-8")
    except UnicodeEncodeError:
        surrogate_errors.append(f"U+{value:04X}")
    else:
        raise AssertionError("UTF-8 不應接受 surrogate 碼點")

new_documents = []
for identifier, url, version, locator, quote in [
    ("python_round", "https://docs.python.org/3.13/library/functions.html#round", "Python 3.13.16 documentation", "round", "if two multiples are equally close, rounding is done toward the even choice"),
    ("unicode", "https://www.unicode.org/versions/Unicode17.0.0/core-spec/chapter-3/", "Unicode 17.0.0", "Grapheme cluster; Unicode scalar value; UTF-8 encoding form", "Because surrogate code points are not Unicode scalar values"),
]:
    command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "30", url]
    response = subprocess.run(command, capture_output=True, check=True)
    parser = Text()
    parser.feed(response.stdout.decode())
    normalized = " ".join(" ".join(parser.bits).split())
    assert quote in normalized
    new_documents.append({"id": identifier, "url": url, "version": version, "accessed_on": datetime.now(UTC).date().isoformat(), "command": command, "exit_code": response.returncode, "original_sha256": digest(response.stdout), "bytes": len(response.stdout), "locator": locator, "short_excerpt": quote})

new_paths = ["tiny_perceptron/model.py", "tiny_perceptron/data.py", "tiny_perceptron/attention.py", str(applications_path)]
result["file_sha256"].update({path: digest(Path(path).read_bytes()) for path in new_paths})
result["recheck"] = {
    "original_report": {"path": str(original_path), "sha256": digest(original_path.read_bytes())},
    "prior_artifacts_unchanged": prior_evidence,
    "prior_self_check_sha256": digest(prior_self_check.read_bytes()),
    "previous_issues": original["issues"],
    "diff": {"command": diff_command, "exit_code": difference.returncode, "stdout": difference.stdout},
    "dependency_versions": versions,
    "changed_example_stdout": {"sequence_score": score_stdout, "probability_ratio": ratio_stdout, "answer_alignment": alignment_stdout, "padding": padding_stdout, "reasoning_extra_steps": reasoning_stdout},
    "padding_boundary_probes": padding_results,
    "invalid_padding_position": invalid_position_error,
    "reasoning_verifier_probes": verifier_results,
    "clipped_example": {key: value.detach().tolist() for key, value in clipped_example.items()},
    "ratio_after_one_large_sgd_step": ratio_after,
    "degenerate_sampling_coverage": coverage_endpoints,
    "round": round_results,
    "unicode": {"emoji_code_points": [f"U+{ord(character):04X}" for character in emoji], "emoji_utf8_lengths": [len(character.encode("utf-8")) for character in emoji], "scalar_utf8_lengths": encoding_lengths, "surrogates_rejected": surrogate_errors},
    "new_original_documents": new_documents,
}
result["result"] = "所有原始探針與新增複查斷言通過；章首未變，引用內容的修訂不推翻章首主張。"
print(json.dumps(result, ensure_ascii=False, indent=2))
