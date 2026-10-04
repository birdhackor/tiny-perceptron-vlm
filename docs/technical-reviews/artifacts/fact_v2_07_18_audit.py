"""Execute section examples, preserve inputs, and render its actual SVG."""

import contextlib
import copy
import hashlib
import io
import json
import platform
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from playwright.sync_api import sync_playwright

from scripts.check_technical_reviews import sections
from tiny_perceptron.alignment import dpo_loss
from tiny_perceptron.data import ByteTokenizer, render_chat
from tiny_perceptron.model import masked_loss

ROOT = Path.cwd()
OUT = ROOT / "docs/technical-reviews/artifacts"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    records = {}
    for chapter, lesson in (("07", "7.18"), ("07", "7.1"), ("05", "5.8"), ("13", "13.3"), ("0C", "C.6")):
        body = dict(sections(ROOT / f"course/chapters/{chapter}.md"))[lesson]
        path = OUT / f"fact_v2_07_18_source_{lesson}.txt"
        path.write_text(body, encoding="utf-8")
        records[lesson] = {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}
    body = dict(sections(ROOT / "course/chapters/07.md"))["7.18"]
    code = re.findall(r"```python\n(.*?)```", body, re.S)[0]
    code_path = OUT / "fact_v2_07_18_snippet.py"
    code_path.write_text(code, encoding="utf-8")
    buffer = io.StringIO()
    namespace = {}
    with contextlib.redirect_stdout(buffer):
        exec(compile(code, str(code_path), "exec"), namespace)
    stdout = buffer.getvalue()
    expected = "示範給什麼 3\n偏好比較什麼 3 勝過 答案是3喔！\n作答後得到什麼 4 0\n"
    assert stdout == expected
    assert namespace["feedback"]["sampled_answer"] == "4"
    assert namespace["feedback"]["reward"] == 0
    assert 1 + 2 == 3 and 4 != 1 + 2
    (OUT / "fact_v2_07_18_stdout.txt").write_text(stdout, encoding="utf-8")
    inputs, targets = render_chat([
        {"role": "user", "content": namespace["demonstration"]["question"]},
        {"role": "assistant", "content": "3"},
    ], ByteTokenizer())
    assert targets[targets != -100].tolist() == [59, 2]
    logits = torch.zeros((1, len(inputs), 264), requires_grad=True)
    loss = masked_loss(logits, targets[None])
    loss.backward()
    assert torch.count_nonzero(logits.grad[0, targets == -100]).item() == 0
    chosen = torch.tensor([-0.2], requires_grad=True)
    rejected = torch.tensor([-1.2], requires_grad=True)
    ref_chosen = torch.tensor([-0.3], requires_grad=True)
    ref_rejected = torch.tensor([-1.0], requires_grad=True)
    preference_loss = dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=0.1)
    preference_loss.backward()
    assert chosen.grad.item() < 0 < rejected.grad.item()
    assert ref_chosen.grad is None and ref_rejected.grad is None
    policy = torch.nn.Linear(2, 2)
    reference = copy.deepcopy(policy).requires_grad_(False).eval()
    before = {key: value.clone() for key, value in reference.state_dict().items()}
    with torch.no_grad():
        policy.weight.add_(1)
    assert all(torch.equal(value, before[key]) for key, value in reference.state_dict().items())
    direct_checks = {
        "answer_target_ids": targets[targets != -100].tolist(),
        "answer_target_count": int((targets != -100).sum()),
        "input_shape": list(inputs.shape),
        "target_shape": list(targets.shape),
        "logits_shape": list(logits.shape),
        "logits_dtype": str(logits.dtype),
        "target_dtype": str(targets.dtype),
        "loss_denominator": "2 assistant targets: byte for 3 and EOS; prompt/roles ignored",
        "uniform_answer_cross_entropy": float(loss.detach()),
        "prompt_logit_gradient_nonzeros": int(torch.count_nonzero(logits.grad[0, targets == -100])),
        "dpo_beta": 0.1,
        "dpo_relative_margin": 0.3,
        "dpo_loss": float(preference_loss.detach()),
        "dpo_chosen_gradient": float(chosen.grad),
        "dpo_rejected_gradient": float(rejected.grad),
        "dpo_reference_gradients": "None, None",
        "frozen_reference_copy_unchanged": True,
        "scope": "Direct CPU API/gradient checks, no language model quality or trained-policy benchmark.",
    }
    exercise = {
        "question": "用一句話解釋1+2",
        "answer": "把一個和兩個合在一起，共有三個，所以1+2=3。",
        "original_chosen": "3",
        "original_rejected": "答案是3喔！",
        "inspection": "Both original answers state the result, neither explains addition; swapping labels is insufficient.",
    }
    figure = ROOT / "course/figures/posttrain_signals.svg"
    tree = ET.parse(figure)
    ns = {"s": "http://www.w3.org/2000/svg"}
    labels = [element.text for element in tree.findall(".//s:text", ns)]
    paths = [element.attrib for element in tree.findall(".//s:path", ns)]
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path="/usr/bin/chromium", args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 820, "height": 490}, device_scale_factor=1)
        page.set_content("<style>body{margin:0}svg{display:block;width:820px;height:490px}</style>" + figure.read_text(encoding="utf-8"))
        page.evaluate("document.fonts.ready")
        page.locator("svg").screenshot(path=str(OUT / "fact_v2_07_18_posttrain_signals.png"))
        browser_version = browser.version
        browser.close()
    result = {
        "command": ".venv/bin/python docs/technical-reviews/artifacts/fact_v2_07_18_audit.py",
        "result": "Section snippet output matches exactly; literal dictionaries do not sample or train a model; arithmetic checks pass; original SVG rendered.",
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "CPU", "chromium": browser_version},
        "sources": records,
        "snippet_sha256": digest(code_path),
        "expected_stdout": expected,
        "observed_stdout": stdout,
        "direct_software_checks": direct_checks,
        "exercise": exercise,
        "figure": {"path": str(figure.relative_to(ROOT)), "sha256": digest(figure), "viewBox": tree.getroot().attrib["viewBox"], "labels": labels, "paths": paths},
    }
    (OUT / "fact_v2_07_18_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
