"""只讀複查來源、既有證據與算例，新增證據使用不覆寫的寫法。"""
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
import torch
from scripts.course_experiments.run import experiment_spec
from tiny_perceptron.model import ModelConfig, TinyLM
from tiny_perceptron.modern import DenseFFN


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def command(argv):
    result = subprocess.run(argv, cwd=ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), capture_output=True, text=True)
    value = {"argv": argv, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    if result.returncode:
        raise RuntimeError(json.dumps(value, ensure_ascii=False))
    return value


def section(raw, identifier):
    headings = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    index = next(i for i, match in enumerate(headings) if match[0].startswith(("## " + identifier + " ").encode()))
    end = headings[index + 1].start() if index + 1 < len(headings) else len(raw)
    return raw[headings[index].start():end]


original_path = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_15-1/technical-15-1-original-01.json"
assert sha(original_path.read_bytes()) == "3ec08b10e0eb30125e4a6155f8abf29d80183c4a98bc7240aafb855fe7708398"
original = json.loads(original_path.read_text())
old_artifacts = []
for item in original["artifacts"]:
    actual = sha((ROOT / item["path"]).read_bytes())
    assert actual == item["sha256"]
    old_artifacts.append({"id": item["id"], "path": item["path"], "sha256": actual, "unchanged": True})
old_selfcheck = HERE / "self-check-01.txt"
assert sha(old_selfcheck.read_bytes()) == "f7594c230b04dde648959c7e4bc062e1e7ce07ebfa62f5bb51711629f2491472"
repo_sources = []
for item in original["sources"]:
    if item["kind"] == "repository_code":
        actual = sha((ROOT / item["path"]).read_bytes())
        repo_sources.append({"id": item["id"], "path": item["path"], "original_sha256": item["sha256"], "current_sha256": actual, "unchanged": actual == item["sha256"]})
assert {item["path"] for item in repo_sources if not item["unchanged"]} == {"course/training.md", "course/chapters/15.md"}

file_diffs = []
baselines = {}
for path in ("course/training.md", "course/chapters/15.md"):
    diff = command(["git", "diff", "5ef56bc756f3", "6a7aa1de5227", "--", path])
    baseline = command(["git", "show", "5ef56bc756f3:" + path])["stdout"].encode()
    committed = command(["git", "show", "6a7aa1de5227:" + path])["stdout"].encode()
    current = (ROOT / path).read_bytes()
    expected = next(item["sha256"] for item in original["sources"] if item.get("path") == path)
    assert sha(baseline) == expected
    assert committed == current
    baselines[path] = baseline
    file_diffs.append({"path": path, "baseline_sha256": sha(baseline), "current_sha256": sha(current), "current_matches_named_commit": True, "diff_command": diff})

frozen = (ROOT / "docs/course-review-20261019/freeze-01/sources/15.1.md").read_bytes()
current_chapter = (ROOT / "course/chapters/15.md").read_bytes()
current_training = (ROOT / "course/training.md").read_bytes()
assert frozen == section(current_chapter, "15.1") == section(baselines["course/chapters/15.md"], "15.1")
assert sha(frozen) == original["source_sha256"]
old_comparison = section(baselines["course/chapters/15.md"], "15.13")
new_comparison = section(current_chapter, "15.13")
assert old_comparison == new_comparison


def recipe(raw):
    marker = b'<details id="fixed-moe-recipe" markdown="1">'
    start = raw.index(marker)
    end = raw.index(b"</details>", start) + len(b"</details>")
    return raw[start:end]


assert recipe(current_training) == recipe(baselines["course/training.md"])
linked_blocks = [
    {"path": "course/chapters/15.md", "locator": "15.1", "sha256": sha(frozen), "unchanged": True},
    {"path": "course/chapters/15.md", "locator": "15.13", "sha256": sha(new_comparison), "unchanged": True},
    {"path": "course/training.md", "locator": "fixed-moe-recipe", "sha256": sha(recipe(current_training)), "unchanged": True},
]
freeze_path = ROOT / "docs/course-review-20261019/freeze-06.json"
assert sha(freeze_path.read_bytes()) == "9f5489d6e94946c6aa8690a1672b7f931b02e8c61ebc59f84f179eba5e44a5c9"
freeze = json.loads(freeze_path.read_text())
page = next(item for item in freeze["pages"] if item["page_id"] == "15.1")
assert page["source_sha256"] == sha(frozen) and page["figures_sha256"] == {}

section_command = command([str(ROOT / ".venv/bin/python"), "docs/review-tools/section_facts.py", "course/chapters/15.md#15.1", "--execute"])
receipt = json.loads(section_command["stdout"])
assert receipt["execution_exit_code"] == 0 and receipt["source_sha256"] == sha(frozen)
run = Path(receipt["artifact_directory"])
section_bundle = {"command": section_command, "receipt": receipt, "files": {}}
for name in ("extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt", "fence-1.py"):
    raw = (run / name).read_bytes()
    section_bundle["files"][name] = {"sha256": sha(raw), "text": raw.decode()}
assert section_bundle["files"]["stderr.txt"]["text"] == ""

up = torch.tensor([[1., 0., 1.], [0., 1., 1.]])
down = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
x = torch.tensor([[1., 2.], [2., 1.]])
positive_up = x @ up
positive_relu = torch.relu(positive_up)
positive_output = positive_relu @ down
assert positive_up.tolist() == [[1., 2., 3.], [2., 1., 3.]]
assert positive_output.tolist() == [[4., 5.], [5., 4.]]
assert up.numel() + down.numel() == 12
repeated = {}
for copies in (2, 3):
    batch = torch.cat([x] * copies, dim=0)
    output = torch.relu(batch @ up) @ down
    assert output.tolist() == positive_output.tolist() * copies
    repeated[str(copies)] = {"tokens": len(batch), "parameters": up.numel() + down.numel(), "output": output.tolist()}
negative = torch.tensor([[1., -2.]]) @ up
negative_relu = torch.relu(negative)
negative_output = negative_relu @ down
assert negative.tolist() == [[1., -2., -1.]]
assert negative_relu.tolist() == [[1., 0., 0.]]
assert negative_output.tolist() == [[1., 0.]]
small = DenseFFN(2, hidden=3)
assert small.activation == "gelu" and small.up.bias is not None and small.down.bias is not None
assert sum(p.numel() for p in small.parameters()) == 17

record = json.loads((ROOT / "docs/course-experiments/results/moe.json").read_text())
dense = record["results"]["variants"]["dense_active_top1"]
model = TinyLM(ModelConfig(**dense["model"]["config"]))
parameters = {name: p.numel() for name, p in model.named_parameters()}
assert sum(parameters.values()) == 141568
ffn_counts = [sum(p.numel() for p in block.ffn.parameters()) for block in model.blocks]
assert ffn_counts == [33088, 33088]
heldout = {}
for name, rounded, records in (("validation", "2.29999", 51), ("test", "2.32093", 52)):
    measured = dense["heldout"][name]
    calculated = measured["nll_sum"] / measured["effective_tokens"]
    assert calculated == measured["nll"] and f"{calculated:.5f}" == rounded
    assert measured["records"] == records
    heldout[name] = {key: val for key, val in measured.items() if key != "samples"}
    heldout[name]["calculated"] = calculated
assert dense["training"]["optimizer_updates"] == dense["training"]["steps"] == 180

help_command = command([str(ROOT / ".venv/bin/python"), "-m", "scripts.course_experiments.run", "--help"])
assert "--experiment ID" in help_command["stdout"]
spec = experiment_spec("moe")
assert spec["module"] == "architecture" and spec["function"] == "run_moe"
comparison_text = new_comparison.decode()
full_table = comparison_text.split("| 模型 |", 1)[1]
rows = [line for line in full_table.splitlines() if line.startswith("| Dense，匹配") or line.startswith("| MoE top-")]
assert len(rows) == 7
assert all(term in comparison_text for term in ("39,256", "41,914", "337,761"))

result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-15-1/recheck-current-01.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
    "original_report": {"path": str(original_path.relative_to(ROOT)), "sha256": sha(original_path.read_bytes()), "issues": original["issues"]},
    "original_artifacts": old_artifacts,
    "repository_sources": repo_sources,
    "file_diffs": file_diffs,
    "unchanged_linked_blocks": linked_blocks,
    "section_execution": section_bundle,
    "positive": {"up": positive_up.tolist(), "relu": positive_relu.tolist(), "output": positive_output.tolist()},
    "repeated": repeated,
    "negative": {"up": negative.tolist(), "relu": negative_relu.tolist(), "output": negative_output.tolist()},
    "library_parameters": sum(p.numel() for p in small.parameters()),
    "dense": {"total_parameters": sum(parameters.values()), "ffn_parameters": ffn_counts, "parameter_breakdown": parameters, "heldout": heldout, "training_steps": dense["training"]["steps"], "optimizer_updates": dense["training"]["optimizer_updates"]},
    "recipe": {"help": help_command, "spec": spec, "comparison_rows": rows, "default_device": "cpu，明示 --device cuda 的固定 MoE 命令未改"},
    "claims_rechecked": [item["id"] for item in original["claims"]],
    "scope": "重審凍結頁面全部實質主張與其明確連結的 MoE 配方／七組比較表；其他小節的異動僅作影響分析，不構成替那些小節作完整技術審閱。",
    "result": "凍結小節與兩段明確引用內容均未變；既有證據未漂移；本次算例、參數量、loss 算術與命令核對均通過。",
}
with (HERE / "recheck-current-01.json").open("x", encoding="utf-8") as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(json.dumps({"result": result["result"], "claims_rechecked": len(result["claims_rechecked"]), "unchanged_linked_blocks": linked_blocks, "section_stdout": section_bundle["files"]["stdout.txt"]["text"]}, ensure_ascii=False))
