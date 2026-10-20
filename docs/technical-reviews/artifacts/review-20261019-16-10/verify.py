"""核對重算算例、資料分母、既有量測與官方契約；不重新訓練。"""

import ast
import copy
import hashlib
import json
import platform
import random
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from scripts.course_experiments.architecture import _checkpoint_forward, _checkpoint_probe
from scripts.course_experiments.common import records_sha256, split_records, text_examples
from scripts.prepare_data import generate_records
from tiny_perceptron.data import IGNORE, ByteTokenizer, pad_batch, render_chat
from tiny_perceptron.model import ModelConfig, TinyLM


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fetch(url):
    command = ["curl", "--fail", "--location", "--silent", "--show-error", url]
    return subprocess.run(command, check=True, capture_output=True).stdout


class Article(HTMLParser):
    def __init__(self):
        super().__init__()
        self.inside = False
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag == "article":
            self.inside = True

    def handle_endtag(self, tag):
        if tag == "article":
            self.inside = False

    def handle_data(self, text):
        if self.inside:
            self.parts.append(text)


def function(tree, name):
    node = next(n for n in ast.parse(tree).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.dump(node, include_attributes=False)


torch.set_num_threads(1)
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-16-10/verify.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                    "torch_git_version": str(torch.version.git_version), "device": "cpu"},
    "repository_inputs": {},
    "ffn": [],
}
paths = ["scripts/course_experiments/architecture.py", "scripts/course_experiments/common.py",
         "scripts/prepare_data.py", "tiny_perceptron/data.py", "tiny_perceptron/model.py",
         "tiny_perceptron/attention.py", "tiny_perceptron/modern.py",
         "docs/course-experiments/results/efficiency.json"]
for name in paths:
    result["repository_inputs"][name] = sha((ROOT / name).read_bytes())

for width in (8, 16):
    torch.manual_seed(0)
    layer = nn.Sequential(nn.Linear(4, width), nn.GELU(), nn.Linear(width, 4))
    independent = copy.deepcopy(layer)
    base = torch.randn(2, 4)
    x = base.clone().requires_grad_()
    z = base.clone().requires_grad_()
    original_parameters = [p.detach().clone() for p in layer.parameters()]
    a = layer(x)
    b = checkpoint(layer, z, use_reentrant=False)
    a.square().sum().backward()
    first_gradients = [p.grad.clone() for p in layer.parameters()]
    b.square().sum().backward()
    independent.zero_grad(set_to_none=True)
    independent(base).square().sum().backward()
    observed = {
        "hidden_width": width, "input_shape": list(base.shape), "output_shape": list(b.shape),
        "output_max_error": (a - b).abs().max().item(),
        "input_gradient_max_error": (x.grad - z.grad).abs().max().item(),
        "loss_max_error": abs(a.square().sum().item() - b.square().sum().item()),
        "parameter_max_change": max((p - old).abs().max().item() for p, old in zip(layer.parameters(), original_parameters)),
        "accumulated_parameter_gradient_error": max((p.grad - 2 * g).abs().max().item() for p, g in zip(layer.parameters(), first_gradients)),
        "independent_parameter_gradient_error": max((p.grad - g).abs().max().item() for p, g in zip(independent.parameters(), first_gradients)),
        "distinct_input_storage": x.data_ptr() != z.data_ptr(),
    }
    assert observed["output_shape"] == [2, 4]
    assert all(observed[k] == 0 for k in observed if k.endswith("error") or k == "parameter_max_change")
    assert observed["distinct_input_storage"]
    result["ffn"].append(observed)

data = split_records(generate_records("attributes-sft"), seed=42)
raw = json.loads((ROOT / "docs/course-experiments/results/efficiency.json").read_bytes())
reported = raw["results"]
result["dataset_reconstruction"] = {}
for split, rows in data.items():
    measured = {"records": len(rows), "sha256": records_sha256(rows)}
    assert measured == reported["dataset"][split]
    result["dataset_reconstruction"][split] = measured
examples = text_examples(data["train"], mode="sft", max_length=128)
sampler = random.Random(42)
token_counts = [sum(int((y != IGNORE).sum()) for _, y in sampler.choices(examples, k=8)) for _ in range(40)]
assert sum(token_counts) == 2328
result["effective_target_counts"] = {"per_update": token_counts, "sum": sum(token_counts),
                                      "updates": len(token_counts), "sample_draws": 40 * 8}
mask_example = [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}]
_, labels = render_chat(mask_example)
assert labels[labels != IGNORE].tolist() == ByteTokenizer().encode("answer") + [ByteTokenizer.eos_id]
result["answer_mask_probe"] = {"messages": mask_example, "supervised_ids": labels[labels != IGNORE].tolist(),
                               "expected": ByteTokenizer().encode("answer") + [ByteTokenizer.eos_id]}

model = TinyLM(ModelConfig(**reported["models"]["mha"]["model"]["config"]))
probe = _checkpoint_probe(model, examples, type("Context", (), {"device": "cpu"})())
assert probe["logit_max_error"] == probe["gradient_max_error"] == 0
result["random_model_cpu_probe"] = probe
result["random_model_cpu_probe"]["scope"] = "隨機初值的兩層四頭模型；不是既有 L4 權重或品質的重現。"
result["model_modules"] = sorted({type(m).__name__ for m in model.modules()})
assert not any(isinstance(m, nn.Dropout) for m in model.modules())

result["historical_measurement"] = {key: raw[key] for key in
    ("revision", "device", "seed", "torch_version", "python_version", "gpu", "evidence_status")}
result["historical_measurement"].update(runtime=reported["runtime"],
    config=reported["models"]["mha"]["model"]["config"],
    checkpoint_probe=reported["activation_checkpoint"], branches={})
tok = ByteTokenizer()
for name in ("ordinary", "activation_checkpoint"):
    branch = reported["update_variants"][name]
    train = branch["training"]
    assert train["steps"] == train["optimizer_updates"] == 40
    assert train["skipped_updates"] == 0 and train["effective_tokens"] == 2328
    assert train["peak_memory_allocated_bytes"] - train["memory_allocated_before_bytes"] == train["peak_additional_allocated_bytes"]
    heldout = {}
    for split in ("validation", "test"):
        evaluation = branch["heldout"][split]
        recalculated = []
        for sample in evaluation["samples"]:
            generated = sample["generated_ids"]
            answer = generated[:generated.index(tok.eos_id)] if tok.eos_id in generated else generated
            exact = answer == tok.encode(sample["expected"])
            assert exact == sample["exact"]
            recalculated.append(exact)
        matches = sum(recalculated)
        assert len(recalculated) == evaluation["records"]
        assert matches == evaluation["matches"]
        assert matches / len(recalculated) == evaluation["exact_match"]
        heldout[split] = {"matches": matches, "records": len(recalculated),
                          "exact_match": evaluation["exact_match"], "samples": evaluation["samples"]}
    summary = {"training": train, "heldout": heldout,
        "calculated": {"step_ms": train["warm_step_median_seconds"] * 1000,
            "baseline_MiB": train["memory_allocated_before_bytes"] / 2**20,
            "peak_MiB": train["peak_memory_allocated_bytes"] / 2**20,
            "additional_MiB": train["peak_additional_allocated_bytes"] / 2**20}}
    result["historical_measurement"]["branches"][name] = summary
    assert heldout["validation"]["matches"] == 3 and heldout["test"]["matches"] == 5
result["historical_measurement"]["final_parameter_error"] = reported["update_variants"]["activation_checkpoint"]["weight_max_error_from_ordinary_after_updates"]
assert result["historical_measurement"]["final_parameter_error"] == 0
assert reported["activation_checkpoint"]["logit_max_error"] == reported["activation_checkpoint"]["gradient_max_error"] == 0

revision = raw["revision"]
result["historical_implementation"] = []
for path, names in (
    ("scripts/course_experiments/architecture.py", ["_checkpoint_forward", "_checkpoint_probe", "_train", "run_efficiency"]),
    ("scripts/course_experiments/common.py", ["evaluate_lm", "text_examples", "split_records"]),
    ("scripts/prepare_data.py", ["generate_records"]),
    ("tiny_perceptron/data.py", ["render_chat", "pad_batch"]),
):
    url = f"https://raw.githubusercontent.com/birdhackor/tiny-perceptron-vlm/{revision}/{path}"
    historical = fetch(url)
    current = (ROOT / path).read_bytes()
    comparison = {name: function(historical, name) == function(current, name) for name in names}
    assert all(comparison.values())
    result["historical_implementation"].append({"url": url, "version": revision,
        "accessed_on": "2026-10-20", "retrieval_command": ["curl", "--fail", "--location", "--silent", "--show-error", url],
        "original_sha256": sha(historical), "ast_matches_current": comparison})

sources = [
    ("checkpoint_source", f"https://raw.githubusercontent.com/pytorch/pytorch/{torch.version.git_version}/torch/utils/checkpoint.py",
     ["Activation checkpointing is a technique that trades compute for memory.",
      "fwd_cpu_state = torch.get_rng_state()", "torch.set_rng_state(fwd_cpu_state)"],
     "checkpoint 的說明核對前向保留輸入、反向按需重算、非重入記錄 graph 與提早結束；_checkpoint_without_reentrant_generator 核對 RNG 恢復。"),
    ("checkpoint_docs", "https://docs.pytorch.org/docs/2.14/checkpoint.html",
     ["Default: True"],
     "讀取正文 Note 與 checkpoint API：預設保留 RNG，CPU 加一種裝置類型；函數內搬到新的裝置無法保證與普通執行相同；前後依賴外部狀態不同會造成錯誤梯度。"),
    ("autograd_docs", "https://docs.pytorch.org/docs/2.14/notes/autograd.html",
     ["Some operations need intermediary results to be saved during the forward pass in order to execute the backward pass."],
     "讀取 How autograd encodes the history 與 Saved tensors：記錄求導關係及保存中間結果的用途。"),
    ("dropout_docs", "https://docs.pytorch.org/docs/2.14/generated/torch.nn.Dropout.html",
     ["During training, randomly zeroes some of the elements of the input tensor"],
     "讀取 Dropout 類別正文：訓練時逐次抽 Bernoulli 遮罩，保留元素另外縮放；課文只定義清零機制，未宣稱完整公式。"),
]
result["original_source_receipts"] = []
for identifier, url, quotes, note in sources:
    original = fetch(url)
    text = original.decode()
    if url.endswith(".html"):
        parser = Article()
        parser.feed(text)
        text = " ".join(" ".join(parser.parts).split())
    assert all(quote in text for quote in quotes)
    if identifier == "checkpoint_docs":
        start = text.index("Note")
        print_text = text[start:text.index("torch.utils.checkpoint.checkpoint", start)]
    elif identifier == "autograd_docs":
        start = text.index("Saved tensors")
        print_text = text[start:start + 3000]
    elif identifier == "dropout_docs":
        start = text.index("During training")
        print_text = text[start:start + 1050]
    else:
        start = text.index("Activation checkpointing is a technique")
        print_text = text[start:start + 4200]
    result["original_source_receipts"].append({"id": identifier, "url": url,
        "version": str(torch.version.git_version) if identifier == "checkpoint_source" else "PyTorch 2.14 documentation",
        "accessed_on": "2026-10-20", "original_sha256": sha(original),
        "retrieval_command": ["curl", "--fail", "--location", "--silent", "--show-error", url],
        "short_original_quotes": quotes, "inspection_note": note})
    print(print_text, file=sys.stderr)

result["limitations"] = ["既有 L4 數字由原始結果、逐筆留出結果與量測程式核對，未重新執行 GPU 訓練。",
    "CPU 探針未宣稱大型模型速度或實際峰值記憶體收益；未執行 dropout 或跨裝置的訓練重現。"]
result["result"] = "全部斷言通過"
print(json.dumps(result, ensure_ascii=False, indent=2))
