"""核對 top-k、梯度、稀疏執行路徑與分派統計；第三方全文只留在記憶體。"""

import hashlib
import io
import json
import platform
import re
import subprocess
import sys
import tarfile
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments.architecture import _routing
from scripts.course_experiments.common import records_sha256, split_records, text_examples
from tiny_perceptron.data import pad_batch
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.modern import MoEFFN


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, value):
        self.parts.append(value)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fetch(url):
    return subprocess.run(
        ["curl", "-f", "-s", "-S", "-L", "--max-time", "40", url],
        check=True,
        capture_output=True,
    ).stdout


def receipt(url, version, locator, excerpt, note):
    raw = fetch(url)
    parser = Text()
    parser.feed(raw.decode())
    text = re.sub(r"\s+", " ", " ".join(parser.parts))
    assert excerpt in text, "來源摘錄必須與本次取得的原文相同"
    return {
        "url": url,
        "version": version,
        "accessed_at": datetime.now(UTC).isoformat(),
        "command": ["curl", "-f", "-s", "-S", "-L", "--max-time", "40", url],
        "response_bytes": len(raw),
        "response_sha256": sha(raw),
        "locator": locator,
        "excerpt": excerpt,
        "inspection_note": note,
    }


torch.set_num_threads(1)
torch.manual_seed(42)
report = {
    "environment": {
        "python": sys.version,
        "torch": str(torch.__version__),
        "torch_git_version": str(torch.version.git_version),
        "platform": platform.platform(),
        "device": "cpu",
        "threads": torch.get_num_threads(),
        "seed": 42,
    },
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-15-4/verify.py",
    "sources": {},
}

report["sources"]["paper"] = receipt(
    "https://arxiv.org/html/1701.06538v1",
    "arXiv:1701.06538v1, 2017-01-23",
    "The Structure of the Mixture-of-Experts layer; Gating Network; Strictly Balanced Gating",
    "We train the gating network by simple back-propagation, along with the rest of the model.",
    "已讀 MoE 加權和、零 gate 不必算 expert、softmax gate、top-k、分段梯度、每個樣本的 k 與 batchwise mask 的區別。論文的正規化 top-1 不提供任務 gate 梯度；本頁尚未正規化，不能套用該限制。",
)
report["sources"]["topk"] = receipt(
    "https://docs.pytorch.org/docs/2.14/generated/torch.topk.html",
    "PyTorch 2.14 documentation; installed torch 2.14.1",
    "torch.topk signature, return value, sorted and tie note",
    "the indices of tied elements are not guaranteed to be stable and may vary across different invocations.",
    "已讀 dim、largest=True、sorted=True、回傳 values 與原欄索引、LongTensor 索引，以及平手順序的限制。API 是選值，沒有執行 expert 或自動正規化。",
)
git_version = str(torch.version.git_version)
report["sources"]["autograd"] = receipt(
    f"https://raw.githubusercontent.com/pytorch/pytorch/{git_version}/tools/autograd/derivatives.yaml",
    f"PyTorch commit {git_version}",
    "topk(Tensor self, SymInt k, int dim=-1, bool largest=True, bool sorted=True)",
    "output_differentiability: [True, False]",
    "已讀 topk 的 backward：values 可微，indices 不可微；values 的切向以 gather 按已選索引傳遞。",
)
report["sources"]["float"] = receipt(
    "https://docs.python.org/3.13/tutorial/floatingpoint.html",
    "Python 3.13 documentation",
    "Floating-Point Arithmetic: Issues and Limitations",
    "most decimal fractions cannot be represented exactly as binary fractions.",
    "已讀二進位有限精度與十進位近似表示；本頁實際 tensor 為 float32，未以 Python float 的 53-bit 精度替代 tensor 精度。",
)
report["sources"]["tuple"] = receipt(
    "https://docs.python.org/3.13/reference/expressions.html#expression-lists",
    "Python 3.13 language reference",
    "Expression lists",
    "A trailing comma is required only to create a one-item tuple",
    "已讀單元素 tuple 要有逗號，單獨括號不會建立 tuple。",
)

prob = torch.tensor([[0.1, 0.7, 0.2], [0.6, 0.3, 0.1]])
cases = {}
for name, values in (("original", prob), ("permuted", torch.tensor([[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]]))):
    cases[name] = {}
    for k in (2, 1, 3):
        weights, chosen = values.topk(k, dim=-1)
        assert torch.equal(weights, values.gather(-1, chosen)), "回傳值必須是所選輸入值"
        cases[name][str(k)] = {
            "chosen": chosen.tolist(),
            "weights": weights.tolist(),
            "sum": weights.sum(-1).tolist(),
            "index_dtype": str(chosen.dtype),
        }
assert cases["original"]["2"]["chosen"] == [[1, 2], [0, 1]], "原算例索引不符"
assert cases["permuted"]["2"]["chosen"][0] == [2, 1], "交換後的索引不符"
assert cases["permuted"]["1"]["chosen"][0] == [2], "交換後的 top-1 不符"
assert isinstance((3,), tuple) and not isinstance((3), tuple), "單元素 tuple 語法不符"
report["topk_cases"] = cases
report["tuple"] = {"with_comma_type": type((3,)).__name__, "without_comma_type": type((3)).__name__}

expert_values = torch.tensor([2.0, 3.0, 5.0], dtype=torch.float64)
gradient_cases = []
for k in (1, 2):
    logits = torch.log(torch.tensor([0.1, 0.7, 0.2], dtype=torch.float64)).requires_grad_()

    def objective(scores, normalized=False):
        weights, chosen = scores.softmax(-1).topk(k)
        if normalized:
            weights = weights / weights.sum()
        return (weights * expert_values[chosen]).sum()

    loss = objective(logits)
    gradient = torch.autograd.grad(loss, logits)[0]
    epsilon = 1e-6
    finite_difference = []
    stable = []
    for j in range(len(logits)):
        delta = torch.zeros_like(logits)
        delta[j] = epsilon
        finite_difference.append(float(((objective(logits.detach() + delta) - objective(logits.detach() - delta)) / (2 * epsilon))))
        stable.append(torch.equal((logits.detach() + delta).topk(k).indices, logits.detach().topk(k).indices))
    error = float((gradient - torch.tensor(finite_difference, dtype=torch.float64)).abs().max())
    normalized_gradient = torch.autograd.grad(objective(logits, normalized=True), logits)[0]
    assert error < 1e-7, "解析梯度與有限差分不符"
    assert all(stable), "微擾跨過了排名邊界"
    gradient_cases.append({
        "k": k,
        "probabilities": logits.detach().softmax(-1).tolist(),
        "expert_values": expert_values.tolist(),
        "loss": float(loss.detach()),
        "gradient": gradient.tolist(),
        "finite_difference": finite_difference,
        "epsilon": epsilon,
        "max_error": error,
        "chosen_stable": stable,
        "values_require_grad": logits.softmax(-1).topk(k).values.requires_grad,
        "indices_require_grad": logits.softmax(-1).topk(k).indices.requires_grad,
        "normalized_gradient": normalized_gradient.tolist(),
    })
report["gradients"] = gradient_cases
report["ranking_boundary"] = {
    "left": torch.tensor([0.1, 0.500001, 0.5]).topk(1).indices.tolist(),
    "right": torch.tensor([0.1, 0.499999, 0.5]).topk(1).indices.tolist(),
}

dispatch = []
for k in (1, 2):
    module = MoEFFN(2, experts=3, top_k=k)
    with torch.no_grad():
        module.router.weight.copy_(torch.tensor([[-2.0, 0.0], [2.0, 0.0], [1.0, 0.0]]))
    rows = [0, 0, 0]
    def make_hook(index):
        def hook(module, args, output):
            rows[index] += len(args[0])
        return hook
    hooks = [expert.register_forward_hook(make_hook(i)) for i, expert in enumerate(module.experts)]
    with torch.no_grad():
        _, _, chosen = module(torch.tensor([[1.0, 0.0], [2.0, 0.0]]))
    for hook in hooks:
        hook.remove()
    assert rows[0] == 0 and sum(rows) == 2 * k, "未選的 expert 不應處理 token"
    dispatch.append({"k": k, "tokens": 2, "experts": 3, "chosen": chosen.tolist(), "executed_rows_per_expert": rows})
report["dispatch"] = dispatch

short_examples = [(torch.tensor([1, 2, 3]), torch.tensor([2, 3, 4])), (torch.tensor([1]), torch.tensor([2]))]
mask_checks = []
for k in (1, 2):
    model = TinyLM(ModelConfig(width=8, heads=2, layers=2, max_length=8, experts=4, top_k=k))
    routing = _routing(model, short_examples, SimpleNamespace(device="cpu"))
    assert routing["effective_input_tokens"] == 4, "PAD 排除數目不符"
    assert all(layer["dispatch_denominator"] == 4 * k for layer in routing["layers"]), "每層分母不符"
    mask_checks.append({"k": k, "padded_slots": 6, "valid_slots": 4, "routing": routing})
report["padding_probe"] = mask_checks

experiment_path = ROOT / "docs/course-experiments/results/moe.json"
experiment = json.loads(experiment_path.read_bytes())
report["experiment"] = {
    "path": str(experiment_path.relative_to(ROOT)),
    "sha256": sha(experiment_path.read_bytes()),
    "revision": experiment["revision"],
    "device": experiment["device"],
    "seed": experiment["seed"],
    "python": experiment["python_version"],
    "torch": experiment["torch_version"],
    "dataset": experiment["results"]["dataset"],
    "variants": {},
}
for name, variant in experiment["results"]["variants"].items():
    config = variant["model"]["config"]
    routing = variant["validation_routing"]
    if routing is None:
        continue
    assert config["layers"] == len(routing["layers"]) == 2, "記錄中的層數不符"
    assert config["experts"] == 4, "記錄中的 expert 數不符"
    assert routing["effective_input_tokens"] == 39256 and routing["padding_excluded"], "有效 token 記錄不符"
    for layer in routing["layers"]:
        denominator = layer["dispatch_denominator"]
        assert sum(layer["dispatch_counts"]) == denominator == 39256 * config["top_k"], "分派次數與分母不符"
        assert max(abs(count / denominator - fraction) for count, fraction in zip(layer["dispatch_counts"], layer["load_fraction"], strict=True)) < 1e-15, "分派份額不符"
    report["experiment"]["variants"][name] = {"layers": config["layers"], "experts": config["experts"], "top_k": config["top_k"], "validation_routing": routing}

data_url = "https://media.githubusercontent.com/media/birdhackor/tiny-perceptron-vlm/90ebf972aecddd555f18443624974a3f8ddc60a6/assets/training/tinystories-v1.tar.gz"
archive = fetch(data_url)
assert sha(archive) == "1ac125d427040eb3311c237fbbc65102adfefe62e0a4ecafd78c395a5db59e23", "資料包 hash 不符實驗紀錄"
with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
    member = next(item for item in bundle.getmembers() if item.name.endswith("tinystories-train-512.jsonl"))
    raw_records = bundle.extractfile(member).read()
records = [json.loads(line) for line in raw_records.splitlines() if line.strip()]
for record in records:
    record["family"] = record.get("text_sha256", records_sha256([{"text": record["text"]}]))
data = split_records(records, seed=42)
data_checks = {}
for split, entries in data.items():
    actual = {"records": len(entries), "sha256": records_sha256(entries)}
    assert actual == experiment["results"]["dataset"][split], "重建的資料切分不符實驗紀錄"
    data_checks[split] = actual
examples = text_examples(data["validation"], max_length=128)
valid_count = sum(len(x) for x, _ in examples)
padded_count = 0
masked_count = 0
for start in range(0, len(examples), 16):
    inputs, _, valid = pad_batch(examples[start : start + 16])
    padded_count += inputs.numel()
    masked_count += int(valid.sum())
assert valid_count == masked_count == 39256, "重建的非填充輸入數不符"
report["dataset_reconstruction"] = {
    "url": data_url,
    "accessed_at": datetime.now(UTC).isoformat(),
    "command": ["curl", "-f", "-s", "-S", "-L", "--max-time", "40", data_url],
    "archive_bytes": len(archive),
    "archive_sha256": sha(archive),
    "member": member.name,
    "member_sha256": sha(raw_records),
    "source_records": len(records),
    "splits": data_checks,
    "validation_chunks": len(examples),
    "batch_size": 16,
    "max_length": 128,
    "effective_input_tokens": valid_count,
    "valid_mask_sum": masked_count,
    "padded_slots": padded_count,
    "padding_slots_excluded": padded_count - masked_count,
    "scope": "只重建資料與分母；未重新訓練、下載權重或重現原 CUDA 分派給各 expert 的次數。資料包與第三方全文未寫入工作樹。",
}
report["repository_files"] = {
    path: sha((ROOT / path).read_bytes()) for path in (
        "tiny_perceptron/modern.py",
        "tiny_perceptron/data.py",
        "tiny_perceptron/model.py",
        "scripts/course_experiments/common.py",
        "scripts/course_experiments/architecture.py",
        "docs/course-experiments/results/moe.json",
    )
}
report["experiment_code_hash_matches"] = {
    path: {"recorded": experiment["code_sha256"].get(path), "current": value,
           "equal": experiment["code_sha256"].get(path) == value}
    for path, value in report["repository_files"].items()
    if path in experiment["code_sha256"]
}
historical = subprocess.run(
    ["git", "show", experiment["revision"] + ":scripts/course_experiments/architecture.py"],
    capture_output=True,
)
report["historical_code_availability"] = {
    "command": ["git", "show", experiment["revision"] + ":scripts/course_experiments/architecture.py"],
    "exit_code": historical.returncode,
    "stderr": historical.stderr.decode(),
    "scope": "此查核不影響資料與分母的獨立重建；未聲稱重現歷史訓練或原 GPU 的 expert 分派。",
}
report["result"] = "所有算例、梯度、執行路徑、分母、資料切分及來源摘錄核對通過"
print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
