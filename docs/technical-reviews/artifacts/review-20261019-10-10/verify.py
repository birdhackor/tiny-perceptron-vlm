"""以解析公式、CPU 執行及原始來源核對配對分類範例。"""

import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(name, value):
    path = ARTIFACT / name
    serialized = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text() == serialized:
            return
        raise FileExistsError("證據檔已存在，請另用新檔名")
    path.write_text(serialized)


class TextOnly(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"style", "script"}:
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in {"style", "script"}:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip and data.strip():
            self.parts.append(data.strip())


def fetch(url, version, selections, html=False):
    argv = ["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "45", url]
    result = subprocess.run(argv, capture_output=True, check=True)
    raw = result.stdout
    text = raw.decode()
    if html:
        parser = TextOnly()
        parser.feed(text)
        text = " ".join(parser.parts)
    excerpts = []
    for locator, start, end in selections:
        begin = text.index(start)
        finish = text.index(end, begin) + len(end)
        excerpts.append({"locator": locator, "excerpt": text[begin:finish]})
    return {
        "url": url,
        "version": version,
        "accessed_at_utc": datetime.now(UTC).isoformat(),
        "command_argv": argv,
        "exit_code": result.returncode,
        "stderr": result.stderr.decode(),
        "response_bytes": len(raw),
        "response_sha256": sha(raw),
        "excerpts": excerpts,
        "storage_policy": "完整回應只在記憶體；本檔只保存支持主張的小段原文與收據。",
    }


torch.set_num_threads(1)
environment = {
    "python": platform.python_version(),
    "torch": str(torch.__version__),
    "torch_git_version": str(torch.version.git_version),
    "device": "cpu",
    "platform": platform.platform(),
}
run = ROOT / "outputs/reviewer-tools/runs/20261020T070502556141Z"
bundle = {
    "command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/10.md#10.10" --execute',
    "provenance": str(run.relative_to(ROOT)),
    "files": {},
}
for name in ("extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt", "fence-1.py", "section.md"):
    raw = (run / name).read_bytes()
    bundle["files"][name] = {"sha256": sha(raw), "content": raw.decode()}
extraction = json.loads(bundle["files"]["extraction.json"]["content"])
assert extraction["source_sha256"] == "4c7aee038deaa644459fea2c50231a1ad8291cb3b5d55198733971b1502f32c0", "原文雜湊不符"
assert json.loads(bundle["files"]["execution.json"]["content"])["exit_code"] == 0, "小節執行未成功"
assert bundle["files"]["stderr.txt"]["content"] == "", "小節執行出現標準錯誤輸出"
save("section-execution.json", bundle)

tau, batch_size = 0.5, 2
p = 1 / (1 + math.exp(-4))
analytic_loss = math.log1p(math.exp(-4))
analytic_gradient = [[(p - 1) / (batch_size * tau), (1 - p) / (batch_size * tau)], [(1 - p) / (batch_size * tau), (p - 1) / (batch_size * tau)]]
values = {}
for dtype in (torch.float32, torch.float64):
    cases = {}
    for name, labels in (("aligned", [0, 1]), ("reversed", [1, 0])):
        scores = torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=dtype, requires_grad=True)
        old_scores = scores.detach().clone()
        targets = torch.tensor(labels, dtype=torch.int64)
        loss = F.cross_entropy(scores / tau, targets)
        loss.backward()
        expected_loss = analytic_loss if name == "aligned" else 4 + analytic_loss
        probabilities = (scores.detach() / tau).softmax(-1)
        expected_grad = (probabilities - F.one_hot(targets, 2)) / (batch_size * tau)
        assert torch.equal(scores.detach(), old_scores), "backward 不應更新分數"
        assert abs(loss.item() - expected_loss) < 2e-7, "代價與解析公式不符"
        assert torch.allclose(scores.grad, expected_grad, atol=1e-7, rtol=0), "梯度與解析公式不符"
        cases[name] = {
            "scores": scores.detach().tolist(),
            "scores_after_backward_equal_input": torch.equal(scores.detach(), old_scores),
            "targets": targets.tolist(),
            "target_dtype": str(targets.dtype),
            "score_dtype": str(dtype),
            "logits": (scores.detach() / tau).tolist(),
            "probabilities": probabilities.tolist(),
            "loss": loss.item(),
            "gradient": scores.grad.tolist(),
            "hypothetical_score_sgd_lr_0_1": (scores.detach() - 0.1 * scores.grad).tolist(),
            "expected_loss": expected_loss,
            "max_gradient_error": (scores.grad - expected_grad).abs().max().item(),
        }
    values[str(dtype)] = cases

from scripts.course_experiments.modalities import _Contrastive

torch.manual_seed(42)
model = _Contrastive().cpu()
images = torch.rand(2, 3, 16, 16)
texts = ["red circle", "blue square"]
scores = model.scores(images, texts)
scores.retain_grad()
loss = F.cross_entropy(scores, torch.arange(2))
vision_parameter = model.vision.projection.weight
text_parameter = model.text.weight
score_jacobians = [torch.autograd.grad(scores[i, j], (vision_parameter, text_parameter), retain_graph=True) for i in range(2) for j in range(2)]
# retain_grad 也收集逐格求導的中間梯度；代價反傳前清空，避免混入別次求導。
scores.grad = None
loss.backward()
assert torch.allclose(scores.grad, (scores.detach().softmax(-1) - torch.eye(2)) / 2, atol=1e-7, rtol=0), "分數梯度與解析式不符"
chain_errors = {}
for k, (name, parameter) in enumerate((("vision_projection", vision_parameter), ("text_embedding", text_parameter))):
    recombined = sum(scores.grad[i, j] * score_jacobians[i * 2 + j][k] for i in range(2) for j in range(2))
    error = (recombined - parameter.grad).abs().max().item()
    assert torch.allclose(recombined, parameter.grad, atol=2e-6, rtol=2e-5), "參數梯度與逐格鏈式法則不符"
    chain_errors[name] = {
        "gradient_norm": parameter.grad.norm().item(),
        "entries_with_nonzero_score_derivative": sum(jacobian[k].abs().sum().item() > 0 for jacobian in score_jacobians),
        "max_chain_rule_error": error,
    }

record_path = ROOT / "docs/course-experiments/results/contrastive.json"
record_raw = record_path.read_bytes()
record = json.loads(record_raw)
module_path = ROOT / "scripts/course_experiments/modalities.py"
module_raw = module_path.read_bytes()
assert record["results"]["config"]["temperature"] == 0.1, "實驗紀錄溫度不符"
assert record["code_sha256"]["scripts/course_experiments/modalities.py"] == sha(module_raw), "實驗程式與紀錄版本不符"
verification = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-10-10/verify.py",
    "environment": environment,
    "formula": {
        "probability": "p=exp(4)/(exp(4)+1)",
        "mean_cross_entropy": "L=-sum_i log p[i,y_i]/N，N=2",
        "score_gradient": "dL/dS[i,j]=(p[i,j]-1[j=y_i])/(N*tau)，tau=0.5，N*tau=1",
        "aligned_loss": analytic_loss,
        "aligned_gradient": analytic_gradient,
        "reversed_loss": 4 + analytic_loss,
        "score_sgd": "S_new=S_old-0.1*dL/dS；只是獨立分數的假設更新，不等於編碼器參數更新後的分數",
    },
    "score_cases": values,
    "encoder_chain_rule": {"seed": 42, "images_shape": list(images.shape), "texts": texts, "scores_shape": list(scores.shape), "loss": loss.item(), "results": chain_errors},
    "experiment_record": {
        "path": str(record_path.relative_to(ROOT)),
        "sha256": sha(record_raw),
        "revision": record["revision"],
        "evidence_status": record["evidence_status"],
        "status": record["status"],
        "results.config": record["results"]["config"],
        "module_sha256": sha(module_raw),
        "module_sha256_matches_record": True,
    },
    "result": "所有解析公式、梯度、未更新狀態、編碼器鏈式法則與紀錄溫度核對通過。未執行長時間訓練。",
}
save("verification.json", verification)

commit = str(torch.version.git_version)
base = f"https://raw.githubusercontent.com/pytorch/pytorch/{commit}/"
receipts = {
    "torch_functional": fetch(base + "torch/nn/functional.py", f"PyTorch {torch.__version__}, commit {commit}", [("cross_entropy signature and input/target arguments", "def cross_entropy(", "        weight (Tensor, optional):")]),
    "torch_loss": fetch(base + "torch/nn/modules/loss.py", f"PyTorch {torch.__version__}, commit {commit}", [("CrossEntropyLoss class-index formula and reduction", "      .. math::\n          \\ell(x, y)", "      Note that this case is equivalent to applying :class:`~torch.nn.LogSoftmax`\n      on an input, followed by :class:`~torch.nn.NLLLoss`.")]),
    "torch_backward": fetch(base + "torch/autograd/__init__.py", f"PyTorch {torch.__version__}, commit {commit}", [("backward chain rule and leaf accumulation", '    r"""Compute the sum of gradients', "    ``.grad`` attributes or set them to ``None`` before calling it.")]),
    "torch_sgd": fetch(base + "torch/optim/sgd.py", f"PyTorch {torch.__version__}, commit {commit}", [("SGD parameter update", "            &\\hspace{5mm}\\theta_t", "            &\\bf{return} \\:  \\theta_t")]),
    "clip": fetch("https://arxiv.org/html/2103.00020v1", "arXiv:2103.00020v1, 2021-02-26", [("Selecting an Efficient Pre-Training Method, in-batch objective", "Given a batch of", "We optimize a symmetric cross entropy loss over these similarity scores.")], html=True),
    "supcon": fetch("https://arxiv.org/html/2004.11362v5", "arXiv:2004.11362v5, 2021-03-10", [("Introduction, multiple positives", "Our technical novelty", "These positives are drawn from samples of the same class as the anchor, rather than being data augmentations of the anchor, as done in self-supervised learning."), ("Related Work, false negatives", "These are usually chosen uniformly", "very low probability of false negatives.")], html=True),
}
save("source-excerpts.json", receipts)
print(json.dumps({"environment": environment, "aligned_loss": values["torch.float32"]["aligned"]["loss"], "aligned_gradient": values["torch.float32"]["aligned"]["gradient"], "reversed_loss": values["torch.float32"]["reversed"]["loss"], "reversed_gradient": values["torch.float32"]["reversed"]["gradient"], "encoder_chain_rule": chain_errors, "temperature": record["results"]["config"]["temperature"], "result": verification["result"]}, ensure_ascii=False, indent=2))
