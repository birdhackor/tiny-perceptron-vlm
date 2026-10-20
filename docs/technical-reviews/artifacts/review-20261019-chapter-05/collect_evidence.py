"""唯讀核對凍結導言、必要前置與原始來源，僅保存短摘錄及本機核對結果。"""

import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime
from html import unescape
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def section(path, title):
    text = (ROOT / path).read_text(encoding="utf-8")
    start = text.index(title)
    end = text.find("\n## ", start + len(title))
    return text[start:] if end < 0 else text[start:end]


def fetch(url, markers):
    command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "45", url]
    response = subprocess.run(command, capture_output=True, check=True)
    html = response.stdout.decode("utf-8")
    visible = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    visible = re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", visible)))
    title = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
    excerpts = []
    for label, needle, width in markers:
        position = visible.lower().find(needle.lower())
        if position < 0:
            raise ValueError("原文未找到指定定位：" + label)
        excerpts.append({"locator": label, "text": visible[position:position + width]})
    return {
        "url": url,
        "command": command,
        "returncode": response.returncode,
        "retrieved_at": datetime.now().astimezone().isoformat(),
        "response_sha256": digest(response.stdout),
        "response_bytes": len(response.stdout),
        "title": unescape(title[1]).strip() if title else "",
        "excerpts": excerpts,
        "storage_note": "全文僅在記憶體內處理；只保存支撐主張的短摘錄與全文指紋。",
    }


def main():
    freeze_path = "docs/course-review-20261019/freeze-04.json"
    freeze = json.loads((ROOT / freeze_path).read_bytes())
    page = next(item for item in freeze["pages"] if item["page_id"] == "chapter-05")
    source = (ROOT / page["snapshot"]).read_bytes()
    current = (ROOT / "course/chapters/05.md").read_bytes().split(b"## ", 1)[0]
    assert digest(source) == "3b90377c05db9463ff2a179fcae4405300f731384d0c7bb4f0ba55e888abe2c4"
    assert source == current
    assert page["figures_sha256"] == {}
    assert not re.search(rb"!\[[^\]]*\]\([^)]*\)", source)
    contexts = {
        "effective_targets": section("course/chapters/05.md", "## 5.2 一次看多少例子？"),
        "effort_comparison": section("course/chapters/07.md", "## 7.25 怎樣判斷多花計算值得？"),
        "nonmonotonic_depth": section("course/chapters/14.md", "## 14.16 能保證多一輪，loss 就下降嗎？"),
    }
    navigation = {}
    for path in ("course/chapters/05.md", "course/chapters/07.md", "course/chapters/14.md", "course/glossary.md"):
        raw = (ROOT / path).read_bytes()
        navigation[path] = {"sha256": digest(raw), "headings": re.findall(r"^## .+$", raw.decode("utf-8"), re.M)}
    labels = torch.tensor([[1, 2, 3, 4], [1, -100, -100, -100]])
    from tiny_perceptron.model import loss_sum
    from tiny_perceptron.lesson_products.model import Config, LoopedSolver

    logits = torch.zeros(2, 4, 5, requires_grad=True)
    total, count = loss_sum(logits, labels)
    (total / count).backward()
    assert int(count) == 5
    assert torch.equal(logits.grad.abs().sum(-1) > 0, labels != -100)
    torch.manual_seed(42)
    model = LoopedSolver(Config(vocab_size=8, width=8, heads=2, ff=16, max_length=8)).eval()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    calls = []
    hook = model.shared.register_forward_hook(lambda *_: calls.append(1))
    recurrence = []
    with torch.no_grad():
        for loops in (1, 3, 5):
            calls.clear()
            output = model(torch.tensor([[1, 2, 3]]), loops=loops)
            assert len(calls) == loops
            assert all(torch.equal(before[name], parameter) for name, parameter in model.named_parameters())
            recurrence.append({"loops": loops, "shared_block_calls": len(calls), "parameter_elements": sum(p.numel() for p in model.parameters()), "output_shape": list(output.shape), "parameters_unchanged": True})
    hook.remove()
    sources = {
        "cross_entropy": fetch("https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html", [
            ("CrossEntropyLoss input shape", "The input is expected", 480),
            ("CrossEntropyLoss ignore_index parameter", "ignore_index (", 530),
        ]),
        "numel": fetch("https://docs.pytorch.org/docs/2.14/generated/torch.numel.html", [
            ("torch.numel contract", "Returns the total number", 100),
        ]),
        "timer": fetch("https://docs.python.org/3.13/library/time.html", [
            ("time.perf_counter", "Return the value (in fractional seconds) of a performance counter", 510),
        ]),
        "evaluation": fetch("https://scikit-learn.org/stable/modules/cross_validation.html", [
            ("Cross-validation introduction", "Learning the parameters of a prediction function", 820),
            ("Validation and final test separation", "When evaluating different settings", 860),
        ]),
        "recurrence": fetch("https://arxiv.org/html/2502.05171v2", [
            ("Abstract", "Our model works by iterating a recurrent block", 260),
            ("Macroscopic Design", "Given a number of recurrent iterations", 1100),
            ("Standard Benchmarks: parameter and compute budgets", "While our model has only", 570),
            ("Where does recurrence help most?", "We find that saturation is highly task-dependent", 240),
            ("Baseline comparison table column names", "Model Tokens ARC-E ARC-C HellaSwag MMLU OBQA PiQA SciQ WinoGrande GSM8K CoT", 85),
            ("Baseline comparison: early checkpoint with recurrent depth", "Ours, early ckpt, ( r = 32", 150),
            ("Baseline comparison: same early checkpoint with one iteration", "Ours, early ckpt, ( r = 1", 150),
        ]),
    }
    result = {
        "command": "PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python docs/technical-reviews/artifacts/review-20261019-chapter-05/collect_evidence.py > docs/technical-reviews/artifacts/review-20261019-chapter-05/evidence.json",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform(), "executable": sys.executable},
        "source_audit": {"snapshot": page["snapshot"], "source_sha256": digest(source), "source_freeze": {"path": freeze_path, "sha256": digest((ROOT / freeze_path).read_bytes())}, "current_intro_matches": True, "figures": {}, "code_fences": source.count(b"```"), "page_text": source.decode("utf-8")},
        "navigation": navigation,
        "local_code_sha256": {path: digest((ROOT / path).read_bytes()) for path in ("tiny_perceptron/model.py", "tiny_perceptron/lesson_products/model.py")},
        "context_excerpt": {key: value.split("```", 1)[0] for key, value in contexts.items()},
        "effective_target_probe": {"labels": labels.tolist(), "samples": labels.shape[0], "target_positions": labels.numel(), "effective_targets": int(count), "nonzero_gradient_positions": (logits.grad.abs().sum(-1) > 0).tolist()},
        "recurrence_probe": recurrence,
        "probe_scope": "只核對忽略位置與共享參數的機制；不做訓練，不量測速度，不以隨機模型聲稱品質或泛化效果。",
        "sources": sources,
        "result": "凍結來源、無圖、定義連結及機制核對全部通過；外部原文均以 curl 讀取成功。",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
