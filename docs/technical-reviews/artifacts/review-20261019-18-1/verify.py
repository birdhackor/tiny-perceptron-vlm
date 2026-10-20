"""核對候選比例、目標梯度、教師來源紀錄與原始來源收據。"""

import hashlib
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class VisibleText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skipped = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skipped += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skipped -= 1

    def handle_data(self, data):
        if not self.skipped:
            self.parts.append(data)


def external_receipts():
    specifications = [
        (
            "https://arxiv.org/html/1503.02531v1",
            "arXiv:1503.02531v1, 2015-03-09",
            "Introduction / transfer set",
            "For this transfer stage, we could use the same training set or a separate",
        ),
        (
            "https://arxiv.org/html/2306.08543v6",
            "arXiv:2306.08543v6, 2026-01-31",
            "Introduction / black-box KD",
            "black-box KD, where only the teacher-generated texts are accessible",
        ),
        (
            "https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.softmax.html",
            "PyTorch 2.14 documentation",
            "torch.nn.functional.softmax / normalization",
            "lie in the range [0, 1] and sum to 1.",
        ),
        (
            "https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html",
            "PyTorch 2.14 documentation",
            "CrossEntropyLoss / ignore_index",
            "Specifies a target value that is ignored and does not contribute to the input gradient.",
        ),
        (
            "https://docs.pytorch.org/docs/2.14/generated/torch.argmax.html",
            "PyTorch 2.14 documentation",
            "torch.argmax / return value",
            "Returns the indices of the maximum value of all elements in the input tensor.",
        ),
    ]
    receipts = []
    for url, version, locator, excerpt in specifications:
        command = ["curl", "--fail", "--location", "--silent", "--show-error", url]
        raw = subprocess.check_output(command)
        parser = VisibleText()
        parser.feed(raw.decode("utf-8"))
        text = re.sub(r"\s+", " ", " ".join(parser.parts))
        assert excerpt in text, "原文摘錄未出現在這次讀取的內容"
        receipts.append(
            {
                "url": url,
                "version": version,
                "accessed_at": datetime.now(timezone.utc).isoformat(),
                "command_argv": command,
                "returncode": 0,
                "original_bytes": len(raw),
                "original_sha256": hashlib.sha256(raw).hexdigest(),
                "locator": locator,
                "excerpt": excerpt,
                "retention": "只保存短摘錄與原檔收據；全文只在記憶體中讀取。",
            }
        )
    return {"retrievals": receipts}


def local_verification():
    import torch
    from torch.nn import functional as F

    from tiny_perceptron.alignment import distillation_kl
    from tiny_perceptron.data import IGNORE, ByteTokenizer, CharTokenizer

    torch.set_default_device("cpu")
    distributions = [
        (["4", "3", "100"], [0.7, 0.2, 0.1]),
        (["4", "3", "100"], [0.4, 0.35, 0.25]),
        (["貓", "狗", "鳥"], [0.8, 0.15, 0.05]),
        (["貓", "狗", "鳥"], [0.8, 0.05, 0.15]),
    ]
    numeric = []
    for candidates, values in distributions:
        probabilities = torch.tensor(values)
        assert abs(sum(values) - 1) < 1e-12, "人工比例未正規化"
        numeric.append(
            {
                "candidates": candidates,
                "specified": values,
                "observed_float32": probabilities.tolist(),
                "sum": float(probabilities.sum()),
                "answer": candidates[probabilities.argmax().item()],
                "descending": [candidates[i] for i in probabilities.argsort(descending=True).tolist()],
                "top_margin": sorted(values, reverse=True)[0] - sorted(values, reverse=True)[1],
            }
        )
    hard_runs = []
    for _ in range(2):
        logits = torch.tensor([[0.1, -0.2, 0.3]], dtype=torch.float64, requires_grad=True)
        loss = F.cross_entropy(logits, torch.tensor([0]))
        gradient = torch.autograd.grad(loss, logits)[0]
        hard_runs.append({"loss": loss.item(), "gradient": gradient.tolist()})
    assert hard_runs[0] == hard_runs[1], "相同完整目標的確定性梯度不相同"
    soft_gradients = []
    for values in ([0.7, 0.2, 0.1], [0.4, 0.35, 0.25]):
        logits = torch.tensor([[0.1, -0.2, 0.3]], dtype=torch.float64, requires_grad=True)
        target = torch.tensor([values], dtype=torch.float64)
        loss = -(target * logits.log_softmax(-1)).sum()
        gradient = torch.autograd.grad(loss, logits)[0]
        soft_gradients.append(gradient.tolist())
    tok = ByteTokenizer()
    text = "100"
    char_tok = CharTokenizer(text)
    assert tok.decode(tok.encode(text)) == char_tok.decode(char_tok.encode(text)) == text
    labels = torch.tensor([[0, IGNORE]])
    student = torch.zeros(1, 2, 3, requires_grad=True)
    teacher = torch.tensor([[[0.7, 0.2, 0.1], [0.2, 0.3, 0.5]]]).log().requires_grad_()
    kl = distillation_kl(student, teacher, labels, temperature=1.0)
    p = teacher.detach().softmax(-1)[0, 0]
    expected_kl = (p * (p.log() - student.detach()[0, 0].log_softmax(-1))).sum()
    kl.backward()
    assert torch.allclose(kl.detach(), expected_kl, atol=1e-7), "KL 方向或遮罩不符"
    assert teacher.grad is None, "教師梯度沒有切斷"
    assert torch.equal(student.grad[0, 1], torch.zeros(3)), "忽略位置產生梯度"
    result_dir = ROOT / "docs/course-experiments/results"
    distill = json.loads((result_dir / "distillation.json").read_bytes())
    teachers = {}
    for name, task in distill["results"]["tasks"].items():
        provenance = task["teacher_provenance"]
        original = json.loads((result_dir / (provenance["experiment"] + ".json")).read_bytes())
        checkpoint = next(a for a in original["artifacts"] if a["path"] == "model.pt")
        assert checkpoint["sha256"] == provenance["sha256"], "教師來源與原訓練產物的雜湊不符"
        assert task["teacher_frozen_and_unchanged"], "紀錄未確認教師固定"
        audit = (task.get("hard_target_generation") or {}).get("audit", [])
        selected = [
            row
            for row in audit
            if row["question"]
            in (
                "color=green;shape=square;pitch=low;color?",
                "style=concise; 1+2=?",
                "style=json; 1+2=?",
            )
        ]
        teachers[name] = {
            "teacher_provenance": provenance,
            "source_checkpoint_manifest": checkpoint,
            "frozen": task["teacher_frozen_and_unchanged"],
            "data": task["data"],
            "examples": selected,
            "original_training_steps": original["results"].get("training", {}).get("steps"),
            "teacher_variant": original["results"].get("teacher_variant"),
            "assets": [
                {k: v for k, v in a.items() if k in ("asset", "name", "id", "source", "sha256")}
                for a in original.get("assets", [])
            ],
        }
    modal = json.loads((result_dir / "multimodal_distillation.json").read_bytes())
    modal_teachers = {}
    for name, task in modal["results"]["tasks"].items():
        p = task["teacher_provenance"]
        modal_teachers[name] = {k: p[k] for k in ("experiment", "sha256", "format_version", "steps")}
        modal_teachers[name]["task"] = p["metadata"]["task"]
    svg_path = ROOT / "course/figures/p7-18-three-targets.svg"
    svg = ET.fromstring(svg_path.read_bytes())
    bars = [
        float(node.attrib["width"])
        for node in svg.findall("{http://www.w3.org/2000/svg}rect")
        if node.attrib.get("x") == "163"
    ]
    assert bars == [133.0, 38.0, 19.0], "候選比例的長條寬度不符"
    render = next(
        item
        for item in json.loads((ROOT / "docs/course-review-20261019/figure-renders-04.json").read_bytes())["records"]
        if item["page_id"] == "18.1"
    )
    assert sha(ROOT / render["png"]["path"]) == render["png"]["sha256"], "算繪圖的雜湊不符"
    inputs = [
        "scripts/course_experiments/compression.py",
        "tiny_perceptron/alignment.py",
        "tiny_perceptron/data.py",
        "tiny_perceptron/model.py",
        "course/training.md",
        "docs/course-experiments/results/distillation.json",
        "docs/course-experiments/results/sft.json",
        "docs/course-experiments/results/style.json",
        "docs/course-experiments/results/moe.json",
        "docs/course-experiments/results/multimodal_distillation.json",
        "course/figures/p7-18-three-targets.svg",
    ]
    return {
        "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + str(Path(__file__).relative_to(ROOT)),
        "environment": {"python": sys.version, "torch": str(torch.__version__), "device": "cpu"},
        "result": "全部斷言通過",
        "arithmetic": {"expression": "2+2", "observed": 2 + 2},
        "distributions": numeric,
        "same_hard_target": hard_runs,
        "different_soft_targets": soft_gradients,
        "tokenization": {"text": text, "byte_ids": tok.encode(text), "char_ids": char_tok.encode(text)},
        "masked_forward_kl": {
            "labels": labels.tolist(),
            "observed": kl.item(),
            "expected": expected_kl.item(),
            "teacher_gradient": None,
            "student_gradient": student.grad.tolist(),
        },
        "teachers": teachers,
        "modal_teachers": modal_teachers,
        "experiment_teacher_signal": distill["results"]["teacher_signal"],
        "figure": {"bar_widths": bars, "normalized_widths": [b / sum(bars) for b in bars], "render_receipt": render},
        "input_sha256": {path: sha(ROOT / path) for path in inputs},
        "scope": "原實驗只讀既有紀錄並交叉核對產物清單；沒有下載教師權重或重訓。",
    }


if __name__ == "__main__":
    value = external_receipts() if "--sources" in sys.argv else local_verification()
    print(json.dumps(value, ensure_ascii=False, indent=2))
