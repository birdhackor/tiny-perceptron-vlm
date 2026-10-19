"""核對抽樣流程、資料切分，以及接字表的欄位混淆。"""

import contextlib
import hashlib
import io
import json
import platform
import re
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from scripts.course_experiments.common import seed
from scripts.course_experiments.text import _simple_examples, _simple_sample
from tiny_perceptron.data import split_documents, toy_documents
from tiny_perceptron.model import generate
from tiny_perceptron.simple import BigramLM


ROOT = Path(__file__).resolve().parents[4]
torch.set_num_threads(1)
torch.set_default_device("cpu")
frozen = ROOT / "docs/course-review-20261019/freeze-01/sources/1.14.md"
raw = frozen.read_bytes()
code = re.search(r"```python\n(.*?)```", raw.decode(), re.S)[1]
original = ROOT / "course/chapters/01.md"
section = re.search(r"(?m)^## 1\.14 .*?(?=^## |\Z)", original.read_text(), re.S)[0]
assert raw == section.encode()


def run_example(text):
    namespace = {"__name__": "__main__"}
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(text, str(frozen), "exec"), namespace)
    return namespace, stdout.getvalue()


base, printed = run_example(code)
repeat, printed_repeat = run_example(code)
changed_seed, printed_seed = run_example(code.replace("manual_seed(42)", "manual_seed(7)"))
short, printed_short = run_example(code.replace("range(12)", "range(3)").replace("== 13", "== 4"))
assert printed == printed_repeat
assert base["chars"] == ["貓", "看", "狗", "。"]
assert base["p"].shape == (4, 4)
assert base["p"].dtype == torch.float32
assert torch.allclose(base["p"].sum(-1), torch.ones(4), atol=1e-7, rtol=0)
assert abs(base["p"][0, 1].item() - 0.85) < 1e-7
assert abs(base["p"][0, 0].item() - 0.05) < 1e-7
assert abs(base["p"][1, 1].item() - 0.05) < 1e-7
assert len(base["output"]) == len(changed_seed["output"]) == 13
assert len(short["output"]) == 4
assert isinstance(base["current"], int)
sorted_chars = sorted(set("貓看狗，狗看貓。"))
assert sorted_chars == ["。", "狗", "看", "貓", "，"]

generator = torch.Generator().manual_seed(42)
draws = torch.multinomial(base["p"][0], 100000, replacement=True, generator=generator)
counts = torch.bincount(draws, minlength=4)
frequencies = counts / counts.sum()
assert counts[1] > counts[0] > 0
assert torch.all(torch.abs(frequencies - base["p"][0]) < 0.01)

trace_generator = torch.Generator().manual_seed(42)
current, trace = 0, []
for _ in range(12):
    previous = current
    current = torch.multinomial(base["p"][previous], 1, generator=trace_generator).item()
    trace.append({"from_id": previous, "to_id": current, "weight": base["p"][previous, current].item()})
assert "貓" + "".join(base["chars"][step["to_id"]] for step in trace) == printed.strip()
early_period_seed = None
for candidate in range(100):
    namespace, text = run_example(code.replace("manual_seed(42)", f"manual_seed({candidate})"))
    if "。" in text.strip()[:-1]:
        early_period_seed = {"seed": candidate, "text": text.strip(), "length": len(namespace["output"])}
        break
assert early_period_seed is not None
assert early_period_seed["length"] == 13

documents = toy_documents()
parts = split_documents(documents, seed=42)
assert len(documents) == len(set(documents)) == 12
assert {name: len(rows) for name, rows in parts.items()} == {"train": 9, "validation": 1, "test": 2}
vocabulary = {char: index + 2 for index, char in enumerate(sorted(set("".join(parts["train"]))))}
batches = {name: _simple_examples(rows, vocabulary, 1) for name, rows in parts.items()}
seed(42)
model = BigramLM(len(vocabulary) + 2)
with torch.no_grad():
    before = {name: float(F.cross_entropy(model(x), y)) for name, (x, y) in batches.items()}
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
history = []
for step in range(200):
    optimizer.zero_grad(set_to_none=True)
    x, y = batches["train"]
    loss = F.cross_entropy(model(x), y)
    assert torch.isfinite(loss)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    assert torch.isfinite(norm)
    optimizer.step()
    if step in (0, 199):
        history.append({"step": step + 1, "loss_before_update": float(loss.detach())})
with torch.no_grad():
    after = {name: float(F.cross_entropy(model(x), y)) for name, (x, y) in batches.items()}
samples = [_simple_sample(model, vocabulary, 1, prompt, "cpu") for prompt in ("顏色=", "顏色=紅；形狀=")]
assert all(sample["generated"] == "三角。" for sample in samples)
assert after["train"] < before["train"]
with torch.no_grad():
    ids = [torch.tensor([[vocabulary[char] for char in prompt]]) for prompt in ("顏色=", "顏色=紅；形狀=")]
    logits = [model(context) for context in ids]
    assert torch.equal(logits[0], logits[1])
    probabilities = logits[0].softmax(-1)[0]
reverse = {value: char for char, value in vocabulary.items()}
top_scores = [{"id": int(index), "char": reverse.get(int(index), "<boundary/unknown>"), "probability": float(probabilities[index])} for index in probabilities.argsort(descending=True)[:5]]
record_path = ROOT / "docs/course-experiments/results/simple_models.json"
record = json.loads(record_path.read_bytes())
record_run = record["results"]["runs"]["bigram"]
assert record_run["samples"] == samples
assert abs(before["train"] - record_run["before_nll"]["train"]) < 1e-5
assert abs(after["train"] - record_run["after_nll_same_post_update_time"]["train"]) < 1e-5
split_hashes = {}
for name, rows in parts.items():
    text = "".join(json.dumps({"text": text, "family": text}, ensure_ascii=False) + "\n" for text in rows)
    digest = hashlib.sha256(text.encode()).hexdigest()
    assert digest == record["results"]["data"][name]["sha256"]
    split_hashes[name] = digest


class FixedTokenModel(torch.nn.Module):
    def __init__(self, answer):
        super().__init__()
        self.answer = answer
        self.config = SimpleNamespace(max_length=32)

    def forward(self, ids, cache=None):
        logits = torch.full((*ids.shape, 4), -10.0)
        logits[..., self.answer] = 10.0
        return {"logits": logits, "cache": None}


eos_model = FixedTokenModel(2)
eos_sequence = generate(eos_model, torch.tensor([[1]]), max_new_tokens=12, eos_id=2)
ordinary_sequence = generate(FixedTokenModel(3), torch.tensor([[1]]), max_new_tokens=12, eos_id=2)
assert eos_sequence.tolist() == [[1, 2]]
assert ordinary_sequence.shape == (1, 13)
assert eos_model.training
source_files = [
    "tiny_perceptron/simple.py", "tiny_perceptron/data.py", "tiny_perceptron/model.py",
    "tiny_perceptron/training.py", "scripts/course_experiments/text.py", "scripts/course_experiments/common.py",
    "docs/course-experiments/results/simple_models.json", "course/training.md",
    "docs/course-review-20261019/freeze-01/sources/1.14.md", "course/chapters/01.md", "course/chapters/06.md",
]
source_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_files}
code_hashes_match = {name: source_hashes[name] == record["code_sha256"][name] for name in source_files if name in record["code_sha256"]}
assert all(code_hashes_match.values())
output = {
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version), "device": "cpu", "dtype": "torch.float32", "threads": str(torch.get_num_threads())},
    "sampling": {"seed_42_output": printed.strip(), "seed_42_repeat_equal": printed == printed_repeat, "seed_7_output": printed_seed.strip(), "short_output": printed_short.strip(), "lengths": [len(base["output"]), len(changed_seed["output"]), len(short["output"])], "row_sums": base["p"].sum(-1).tolist(), "trace": trace, "single_row_draws": 100000, "single_row_counts": counts.tolist(), "single_row_frequencies": frequencies.tolist(), "early_period_example": early_period_seed, "untrained_table_requires_grad": base["p"].requires_grad, "current_python_type": type(base["current"]).__name__},
    "bigram": {"seed": 42, "context": 1, "steps": 200, "optimizer": "AdamW", "lr": 0.01, "weight_decay": 0.01, "gradient_clip_norm": 1.0, "documents": documents, "splits": parts, "split_sha256": split_hashes, "effective_targets_per_split": {name: int(y.numel()) for name, (x, y) in batches.items()}, "vocabulary": vocabulary, "before_nll": before, "after_nll": after, "history": history, "samples": samples, "equal_logits_for_two_prompts": torch.equal(logits[0], logits[1]), "equal_sign_top_candidates": top_scores, "saved_record_samples": record_run["samples"], "saved_record_environment": {key: record[key] for key in ("revision", "device", "seed", "torch_version", "python_version")}, "source_hashes_equal_saved_code": code_hashes_match},
    "termination": {"eos_sequence": eos_sequence.tolist(), "ordinary_candidate_length": ordinary_sequence.shape[1], "training_mode_restored": eos_model.training},
    "source_sha256": source_hashes,
    "result": "全部斷言通過；短訓練只核對固定玩具配置，不構成自然語言品質或跨平台重現保證。",
}
print(json.dumps(output, ensure_ascii=False, indent=2))
