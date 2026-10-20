"""核對現行來源與既有證據，重新執行指定程式，僅新增工作檔。"""
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


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def command(argv):
    p = subprocess.run(argv, cwd=ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), capture_output=True, text=True)
    record = {"argv": argv, "exit_code": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    if p.returncode:
        raise RuntimeError(json.dumps(record, ensure_ascii=False))
    return record


def section(raw, identifier):
    headings = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    i = next(i for i, h in enumerate(headings) if h[0].startswith(("## " + identifier + " ").encode()))
    return raw[headings[i].start():headings[i + 1].start() if i + 1 < len(headings) else len(raw)]


def recipe(raw):
    start = raw.index(b'<details id="fixed-moe-recipe" markdown="1">')
    return raw[start:raw.index(b"</details>", start) + len(b"</details>")]


previous_path = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_15-1/technical-15-1-current-01.json"
assert digest(previous_path.read_bytes()) == "a6782454931be6d0c378e17a6156eb837f88ab9f6ae9cd469d7fce76c8c04ae6"
previous = json.loads(previous_path.read_text())
previous_artifacts = []
for item in previous["artifacts"]:
    actual = digest((ROOT / item["path"]).read_bytes())
    assert actual == item["sha256"]
    previous_artifacts.append({"id": item["id"], "path": item["path"], "sha256": actual, "unchanged": True})
source_checks = []
for item in previous["sources"]:
    if item["kind"] == "repository_code":
        actual = digest((ROOT / item["path"]).read_bytes())
        source_checks.append({"id": item["id"], "path": item["path"], "previous_sha256": item["sha256"], "current_sha256": actual, "unchanged": actual == item["sha256"]})
assert [item["path"] for item in source_checks if not item["unchanged"]] == ["course/training.md"]
training = (ROOT / "course/training.md").read_bytes()
assert digest(training) == "1b1dd4657b898ce0cfe15911d8bd2eaa0d19cfb88a47b91ae980ce773826615b"
baseline_command = command(["git", "show", "ecf6dec71280:course/training.md"])
current_commit_command = command(["git", "show", "c6b9339d2595:course/training.md"])
baseline = baseline_command["stdout"].encode()
assert digest(baseline) == next(s["sha256"] for s in previous["sources"] if s.get("path") == "course/training.md")
assert training == current_commit_command["stdout"].encode()
assert recipe(training) == recipe(baseline)
diff = command(["git", "diff", "ecf6dec71280", "c6b9339d2595", "--", "course/training.md"])
prior_run_path = ROOT / next(item["path"] for item in previous["artifacts"] if item["id"] == "current_recheck")
prior_run = json.loads(prior_run_path.read_text())
blocks = []
for item in prior_run["unchanged_linked_blocks"]:
    raw = (ROOT / item["path"]).read_bytes()
    fragment = recipe(raw) if item["locator"] == "fixed-moe-recipe" else section(raw, item["locator"])
    assert digest(fragment) == item["sha256"]
    blocks.append({**item, "unchanged": True})
freeze_path = ROOT / "docs/course-review-20261019/freeze-07.json"
assert digest(freeze_path.read_bytes()) == "67358baeb4833b9b08445fb06bb5ccae0f31fbb7727163f93e1265bbcb420753"
page = next(p for p in json.loads(freeze_path.read_text())["pages"] if p["page_id"] == previous["lesson_id"])
frozen = (ROOT / page["snapshot"]).read_bytes()
assert digest(frozen) == previous["source_sha256"]
assert page["figures_sha256"] == {}
assert section((ROOT / page["source"]).read_bytes(), previous["lesson_id"]) == frozen

execution_command = command([str(ROOT / ".venv/bin/python"), "docs/review-tools/section_facts.py", previous["source"], "--execute"])
receipt = json.loads(execution_command["stdout"])
assert receipt["execution_exit_code"] == 0
assert receipt["source_sha256"] == previous["source_sha256"]
run = Path(receipt["artifact_directory"])
execution = {"command": execution_command, "receipt": receipt, "files": {}}
for name in ("extraction.json", "execution.json", "environment.json", "stdout.txt", "stderr.txt", "fence-1.py"):
    raw = (run / name).read_bytes()
    execution["files"][name] = {"sha256": digest(raw), "text": raw.decode()}
assert execution["files"]["stderr.txt"]["text"] == ""

up = torch.tensor([[1., 0., 1.], [0., 1., 1.]])
down = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
x = torch.tensor([[1., 2.], [2., 1.]])
examples = []
for copies in (1, 2, 3):
    batch = torch.cat([x] * copies, dim=0)
    output = torch.relu(batch @ up) @ down
    assert output.tolist() == [[4., 5.], [5., 4.]] * copies
    examples.append({"input_rows": len(batch), "output": output.tolist(), "weights": up.numel() + down.numel()})
negative = torch.tensor([[1., -2.]]) @ up
negative_relu = torch.relu(negative)
negative_output = negative_relu @ down
assert negative_output.tolist() == [[1., 0.]]
record = json.loads((ROOT / "docs/course-experiments/results/moe.json").read_text())
dense = record["results"]["variants"]["dense_active_top1"]
model = TinyLM(ModelConfig(**dense["model"]["config"]))
parameters = sum(p.numel() for p in model.parameters())
ffn_parameters = [sum(p.numel() for p in block.ffn.parameters()) for block in model.blocks]
assert parameters == 141568 and ffn_parameters == [33088, 33088]
losses = {}
for split, expected in (("validation", "2.29999"), ("test", "2.32093")):
    measured = dense["heldout"][split]
    calculated = measured["nll_sum"] / measured["effective_tokens"]
    assert calculated == measured["nll"] and f"{calculated:.5f}" == expected
    losses[split] = {k: v for k, v in measured.items() if k != "samples"}
    losses[split]["recomputed_nll"] = calculated
help_command = command([str(ROOT / ".venv/bin/python"), "-m", "scripts.course_experiments.run", "--help"])
assert "--experiment ID" in help_command["stdout"]
spec = experiment_spec("moe")
assert spec["module"] == "architecture" and spec["function"] == "run_moe"
result = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-15-1/recheck-current-02.py",
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu", "platform": platform.platform()},
    "previous_report": {"path": str(previous_path.relative_to(ROOT)), "sha256": digest(previous_path.read_bytes()), "issues": previous["issues"]},
    "previous_artifacts": previous_artifacts,
    "repository_sources": source_checks,
    "training_diff": diff,
    "training_versions": {"baseline_command": baseline_command["argv"], "baseline_exit_code": baseline_command["exit_code"], "baseline_sha256": digest(baseline), "current_commit_command": current_commit_command["argv"], "current_commit_exit_code": current_commit_command["exit_code"], "current_sha256": digest(training), "worktree_equals_current_commit": True},
    "unchanged_linked_blocks": blocks,
    "section_execution": execution,
    "examples": examples,
    "negative": {"up": negative.tolist(), "relu": negative_relu.tolist(), "output": negative_output.tolist()},
    "parameters": {"model": parameters, "ffn": ffn_parameters},
    "losses": losses,
    "recipe": {"help": help_command, "spec": spec},
    "claims_rechecked": [claim["id"] for claim in previous["claims"]],
    "change_scope": "指定 diff 只改多模態練習的抽查／測試題與文字能力比較說明，未改本頁及它明確連結的 fixed-moe-recipe 或完整七組比較表。多模態段落本身不納入本頁的逐句審閱。",
    "result": "目標原文與被引用配方／比較表未變；全部既有證據 hash 相同，本次小節程式、算例、參數計數、loss 算術與命令解析均符合原主張。",
}
with (HERE / "recheck-current-02.json").open("x", encoding="utf-8") as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(json.dumps({"result": result["result"], "claims_rechecked": len(result["claims_rechecked"]), "section_stdout": execution["files"]["stdout.txt"]["text"], "parameters": result["parameters"]}, ensure_ascii=False))
