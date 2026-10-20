"""核對固定題的梯度、更新、空目標，以及已保存實驗的分母。"""

import contextlib
import difflib
import hashlib
import io
import json
import math
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch

from scripts.course_experiments.common import split_records, text_examples
from scripts.course_experiments.run import experiment_spec
from scripts.prepare_data import generate_records
from tiny_perceptron.data import IGNORE, pad_batch
from tiny_perceptron.model import ModelConfig, TinyLM, masked_loss


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def command(args):
    run = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=True)
    return {"argv": args, "stdout": run.stdout, "stderr": run.stderr, "exit_code": run.returncode}


torch.set_num_threads(1)
torch.set_default_device("cpu")
freeze_path = "docs/course-review-20261019/freeze-01/sources/5.1.md"
source = (ROOT / freeze_path).read_bytes()
fence = re.search(rb"```python\n(.*?)```", source, re.S)[1].decode()
run_dir = ROOT / "outputs/reviewer-tools/runs/20261020T055735752261Z"
section_run = {
    "command": 'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py "course/chapters/05.md#5.1" --execute',
    "copied_from": str(run_dir.relative_to(ROOT)),
    "files": {},
}
for name in ("extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt", "fence-1.py"):
    raw = (run_dir / name).read_bytes()
    section_run["files"][name] = {"sha256": sha(raw), "content": json.loads(raw) if name.endswith(".json") else raw.decode()}
assert section_run["files"]["extraction.json"]["content"]["source_sha256"] == sha(source)
assert section_run["files"]["execution.json"]["content"]["exit_code"] == 0
assert section_run["files"]["fence-1.py"]["content"] == fence
assert section_run["files"]["stdout.txt"]["content"].count("第一步embedding梯度大小") == 1

torch.manual_seed(42)
model = TinyLM(ModelConfig(vocab_size=5, width=8, max_length=8))
x = torch.tensor([[1, 2, 3]])
y = torch.tensor([[2, 3, 4]])
optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.0)
parameter_ids = {id(p) for p in model.parameters()}
optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
assert parameter_ids == optimizer_ids
initial_parameters = {name: p.detach().clone() for name, p in model.named_parameters()}
logits = model(x)["logits"]
before = masked_loss(logits, y).item()
manual_loss = -logits.log_softmax(-1).gather(-1, y.unsqueeze(-1)).mean().item()
assert math.isclose(before, manual_loss, rel_tol=1e-6)
partial_y = torch.tensor([[2, -100, 4]])
partial = masked_loss(logits, partial_y).item()
partial_manual = -logits.log_softmax(-1)[0, [0, 2], [2, 4]].mean().item()
assert math.isclose(partial, partial_manual, rel_tol=1e-6)
first = {}
for step in range(40):
    optimizer.zero_grad()
    assert all(p.grad is None for p in model.parameters())
    loss = masked_loss(model(x)["logits"], y)
    loss.backward()
    if step == 0:
        grad = model.embedding.weight.grad.detach().clone()
        first = {
            "gradient_shape": list(grad.shape),
            "gradient_norm": grad.norm().item(),
            "sqrt_sum_squares": grad.square().sum().sqrt().item(),
            "gradient_by_row": grad.norm(dim=1).tolist(),
            "parameters_receiving_gradient": sum(p.grad is not None for p in model.parameters()),
        }
        assert first["gradient_norm"] > 0
        assert math.isclose(first["gradient_norm"], first["sqrt_sum_squares"], rel_tol=1e-6)
    optimizer.step()
after = masked_loss(model(x)["logits"], y).item()
deltas = {name: (p.detach() - initial_parameters[name]).abs().max().item() for name, p in model.named_parameters()}
assert after < before
assert max(deltas.values()) > 0
assert abs(before - 2.27) < 0.005
assert abs(after - 0.0004) < 0.00005
assert list(model.embedding.weight.shape) == [5, 8]
lookup_same = torch.equal(model.embedding(x), model.embedding.weight[x])
assert lookup_same
steps = sorted({int(state["step"].item()) for state in optimizer.state.values()})
assert steps == [40]
loop_probe = {
    "config": vars(model.config), "input": x.tolist(), "labels": y.tolist(),
    "input_dtype": str(x.dtype), "logits_shape": list(logits.shape),
    "logits_dtype": str(logits.dtype), "effective_targets": int((y != IGNORE).sum()),
    "before": before, "manual_cross_entropy": manual_loss, "after": after,
    "partial_mask": {"labels": partial_y.tolist(), "effective_targets": 2,
                     "observed_loss": partial, "manual_loss": partial_manual},
    "first_backward": first, "embedding_shape": list(model.embedding.weight.shape),
    "embedding_lookup_equals_indexing": lookup_same,
    "parameter_tensors": len(parameter_ids), "optimizer_parameters_match": parameter_ids == optimizer_ids,
    "parameter_max_absolute_changes": deltas, "optimizer_state_steps": steps,
    "optimizer_weight_decay": optimizer.param_groups[0]["weight_decay"],
}

modified = fence.replace("y = torch.tensor([[2, 3, 4]])", "y = torch.tensor([[-100, -100, -100]])")
assert modified != fence
output = io.StringIO()
try:
    with contextlib.redirect_stdout(output):
        exec(compile(modified, freeze_path, "exec"), {"__name__": "__main__"})
except ValueError as exc:
    ignored_probe = {"error_type": type(exc).__name__, "message": str(exc), "stdout": output.getvalue()}
else:
    raise AssertionError("空目標應停止計算")
assert "所有 labels 都被忽略" in ignored_probe["message"]
assert ignored_probe["stdout"] == ""

p = torch.nn.Parameter(torch.tensor([2.0]))
opt = torch.optim.AdamW([p], lr=0.03, weight_decay=0.0)
(p * p).sum().backward()
one_backward = p.grad.item()
(p * p).sum().backward()
two_backwards = p.grad.item()
opt.zero_grad()
cleared = p.grad is None
old = p.detach().clone()
opt.step()
unchanged_without_gradient = torch.equal(old, p)
assert one_backward == 4 and two_backwards == 8 and cleared and unchanged_without_gradient

result_path = "docs/course-experiments/results/text_foundation.json"
record_raw = (ROOT / result_path).read_bytes()
record = json.loads(record_raw)
parts = split_records(generate_records("toy-text"), seed=42)
split_probes = {}
for split, rows in parts.items():
    raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    examples = text_examples(rows, mode="text", max_length=128)
    effective = sum(int((labels != IGNORE).sum()) for _, labels in examples)
    x_part, y_part, valid = pad_batch(examples)
    assert sha(raw) == record["results"]["data"][split]["sha256"]
    split_probes[split] = {
        "records": len(rows), "examples": len(examples), "effective_tokens": effective,
        "split_sha256": sha(raw), "texts": [row["text"] for row in rows],
        "padded_shape": list(x_part.shape), "valid_positions": int(valid.sum()),
        "counted_labels": int((y_part != IGNORE).sum()),
    }
assert set(row["family"] for row in parts["train"]).isdisjoint(row["family"] for row in parts["validation"])
training = record["results"]["training"]
validation = record["results"]["after"]["validation"]
ratio = validation["nll_sum"] / validation["effective_tokens"]
assert ratio == validation["nll"]
assert validation["effective_tokens"] == split_probes["validation"]["effective_tokens"]
assert round(training["initial_loss"], 5) == 5.79243
assert round(training["final_loss"], 5) == 0.06352
assert round(validation["nll"], 5) == 0.96282
assert training["steps"] == 600 and record["gpu"] == "NVIDIA L4"
recipe_model = TinyLM(ModelConfig(width=64, layers=2))
recipe_description = recipe_model.description()
assert recipe_description["parameters"] == training["parameters"]
recipe_spec = experiment_spec("text_foundation")
assert recipe_spec["module"] == "text" and recipe_spec["function"] == "run_text_foundation"
code_comparison = {}
historical_revision_receipt = subprocess.run(
    ["git", "show", record["revision"] + ":scripts/prepare_data.py"],
    cwd=ROOT, capture_output=True,
)
for path in ("tiny_perceptron/model.py", "tiny_perceptron/data.py", "scripts/course_experiments/text.py", "scripts/course_experiments/common.py"):
    digest = sha((ROOT / path).read_bytes())
    code_comparison[path] = {"current_sha256": digest, "recorded_sha256": record["code_sha256"][path]}
    if digest != record["code_sha256"][path]:
        commits = subprocess.check_output(["git", "log", "--all", "--format=%H", "--", path], cwd=ROOT, text=True).splitlines()
        for commit in commits:
            args = ["git", "show", commit + ":" + path]
            historical = subprocess.run(args, cwd=ROOT, capture_output=True)
            if historical.returncode == 0 and sha(historical.stdout) == record["code_sha256"][path]:
                code_comparison[path].update({
                    "matching_commit": commit, "command": args, "exit_code": historical.returncode,
                    "historical_sha256": sha(historical.stdout),
                    "diff_from_recorded_to_current": "".join(difflib.unified_diff(
                        historical.stdout.decode().splitlines(True), (ROOT / path).read_text().splitlines(True),
                        fromfile="recorded-content", tofile="current-content",
                    )),
                })
                break
        else:
            raise AssertionError("未找到實驗記錄綁定的程式內容：" + path)
t4 = (ROOT / "course/training.md").read_text().split("## T.4 ", 1)[1].split("\n## ", 1)[0]
assert "--experiment text_foundation --device cuda" in t4 and "--steps 200" in t4 and "layers=1" in t4
cli_help = command([str(ROOT / ".venv/bin/python"), "-m", "scripts.course_experiments.run", "--help"])
evidence = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-5-1/review_probe.py",
    "environment": {"python": sys.version, "torch": torch.__version__, "torch_git_version": torch.version.git_version,
                    "device": "cpu", "dtype": str(torch.get_default_dtype()), "threads": str(torch.get_num_threads()),
                    "platform": platform.platform()},
    "head_receipt": command(["git", "rev-parse", "HEAD"]),
    "source": {"path": freeze_path, "sha256": sha(source)},
    "section_execution": section_run, "loop_probe": loop_probe,
    "ignored_labels": ignored_probe,
    "gradient_lifecycle": {"one_backward": one_backward, "two_backwards": two_backwards,
                           "cleared_to_none": cleared, "step_without_gradient_unchanged": unchanged_without_gradient},
    "record_inspection": {
        "path": result_path, "sha256": sha(record_raw),
        "metadata": {k: record[k] for k in ("revision", "device", "seed", "torch_version", "python_version", "gpu", "step_scale", "evidence_status", "status")},
        "training": training,
        "recipe_model_description": recipe_description, "recipe_spec": recipe_spec,
        "validation_after": {k: v for k, v in validation.items() if k != "samples"},
        "split_probes": split_probes, "validation_nll_recomputed_from_saved_sum": ratio,
        "implementation_hash_comparison": code_comparison,
        "recorded_revision_lookup": {"command": historical_revision_receipt.args,
                                     "exit_code": historical_revision_receipt.returncode,
                                     "stdout": historical_revision_receipt.stdout.decode(),
                                     "stderr": historical_revision_receipt.stderr.decode()},
        "scope": "只核對已保存實驗紀錄、實作與資料分母；未重新執行 GPU 訓練。",
    },
    "experiment_cli_help": cli_help,
    "result": "所有斷言通過；固定題更新、空目標與紀錄中的數字均符合原文。",
}
print(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False))
