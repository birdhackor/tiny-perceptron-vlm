"""Same-owner narrow 11.2 context callback. Hash/AST inspection only, no training."""
import ast
from datetime import datetime, UTC
import hashlib
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[3]
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def raw_hash(raw):
    return hashlib.sha256(raw).hexdigest()
prior_path = HERE / "prior-report.opaque.json"
prior_raw = prior_path.read_bytes()
# This is the same owner's own retained scope/evidence, not another reviewer's answer.
prior = json.loads(prior_raw)
assert prior["reviewer_task"] == "/root/phase4_factual_coordinator/factual_11_2"
records = []
for source, lesson, cutoff in [("course/chapters/11.md", "11.2", None), ("course/chapters/05.md", "5.1", b"<details>")]:
    raw = (ROOT / source).read_bytes()
    hs = list(re.finditer(rb"(?m)^## ([\w.]+) [^\r\n]+", raw))
    i = next(i for i, h in enumerate(hs) if h[1].decode() == lesson)
    start = hs[i].start()
    end = hs[i + 1].start() if i + 1 < len(hs) else len(raw)
    whole_section = raw[start:end]
    if cutoff is not None:
        marker = whole_section.find(cutoff)
        assert marker >= 0
        end = start + marker
    body = raw[start:end]
    snapshot = HERE / ("current-" + lesson + ".md")
    assert snapshot.read_bytes() == body
    records.append({"source": source + "#" + lesson, "first_line": raw[:start].count(b"\n") + 1,
        "end_line": raw[:end].count(b"\n"), "raw_byte_start": start, "raw_byte_end_exclusive": end,
        "sha256": raw_hash(body), "full_section_sha256": raw_hash(whole_section),
        "snapshot": snapshot.relative_to(ROOT).as_posix(),
        "inspection_scope": "complete current primary section" if lesson == "11.2" else "main explanatory text, Python fence and exercise before details; historical model measurements excluded"})
assert records[0]["sha256"] == prior["source_sha256"] == "5b6d0b4819041b601df9a10e2a535477096af071d2d2122d93c4251520e0e019"
assert (HERE / "current-11.2.md").read_bytes() == (BASE / "section.md").read_bytes()
frozen_raw = (BASE / "inputs/course/chapters/11.md").read_bytes()
frozen_match = re.search(rb"(?ms)^## 11\.2 .*?(?=^## 11\.3 )", frozen_raw)
assert frozen_match and frozen_match[0] == (HERE / "current-11.2.md").read_bytes()
artifact_checks = [{"id": item["id"], "path": item["path"], "expected_sha256": item["sha256"],
    "observed_sha256": sha(ROOT / item["path"]), "matches": sha(ROOT / item["path"]) == item["sha256"]}
    for item in prior["artifacts"]]
assert all(item["matches"] for item in artifact_checks)
code_checks = [{"id": item["id"], "path": item["path"], "expected_sha256": item["sha256"],
    "observed_sha256": sha(ROOT / item["path"]), "matches": sha(ROOT / item["path"]) == item["sha256"]}
    for item in prior["sources"] if item["kind"] == "repository_code"]
assert all(item["matches"] for item in code_checks)
current_11 = (HERE / "current-11.2.md").read_bytes()
fence_11 = re.search(rb"(?s)```python\n(.*?)```", current_11)[1]
assert fence_11 == (BASE / "fence-1.py").read_bytes()
current_5 = (HERE / "current-5.1.md").read_text()
fence_5 = re.search(r"(?s)```python\n(.*?)```", current_5)[1]
(HERE / "current-5.1-fence.py").write_text(fence_5)
tree_5 = ast.parse(fence_5)
tree_11 = ast.parse(fence_11)
def method_calls(tree):
    return [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
calls_5, calls_11 = method_calls(tree_5), method_calls(tree_11)
assert all(method in calls_5 for method in ["zero_grad", "backward", "step"])
assert all(method not in calls_11 for method in ["zero_grad", "backward", "step"])
loop = next(node for node in tree_5.body if isinstance(node, ast.For))
assert isinstance(loop.iter, ast.Call) and isinstance(loop.iter.func, ast.Name) and loop.iter.func.id == "range"
assert loop.iter.args[0].value == 40
loop_stmts = [ast.unparse(stmt) for stmt in loop.body]
assert loop_stmts[0] == "optimizer.zero_grad()" and loop_stmts[1].startswith("loss = masked_loss(")
assert loop_stmts[2] == "loss.backward()" and loop_stmts[-1] == "optimizer.step()"
assert "梯度大小大於0，只說這張表收到訊號；代價下降才說明更新改善了固定題。" in current_5
assert "[5.1](05.md#5.1)示範step前後的數值更新" in current_11.decode()
import torch
environment = {"python": sys.version, "python_executable": sys.executable,
    "torch": str(torch.__version__), "torch_git_version": str(torch.version.git_version),
    "cuda_build": str(torch.version.cuda), "device": "CPU", "execution_kind": "hash/AST/source inspection; no model forward/backward/step"}
old_environment = json.loads((BASE / "probe-results.json").read_bytes())["environment"]
assert environment["torch"] == old_environment["torch"] and environment["torch_git_version"] == old_environment["torch_git_version"]
assert torch.version.cuda is None
result = {"checked_at": datetime.now(UTC).isoformat(), "reviewer_task": prior["reviewer_task"],
    "environment": environment, "prior_report": {"path": prior_path.relative_to(ROOT).as_posix(), "sha256": raw_hash(prior_raw)},
    "current_context_locators": records, "primary_matches_original_raw_and_frozen_input": True,
    "reused_artifact_hash_checks": artifact_checks, "reused_repository_code_hash_checks": code_checks,
    "ast_inspection": {"current_11_2_fence_sha256": raw_hash(fence_11), "current_5_1_fence_sha256": raw_hash(fence_5.encode()),
        "current_5_1_loop_range": 40, "current_5_1_loop_statements": loop_stmts,
        "current_11_2_has_no_backward_or_optimizer_step": True, "executed_current_5_1_training_fence": False},
    "findings": [
        "Current 5.1 main text distinguishes gradient norm greater than zero from signed gradient and from loss decrease; this is consistent with the unchanged 11.2 distinction between gradient, optimizer membership and numerical update.",
        "The linked 5.1 code explicitly clears grads, computes loss, calls backward then step. The current 11.2 fence only creates/lists optimizer membership. This reference adds no unsupported 11.2 performance claim.",
        "All original 24 registered artifacts and 4 repository-code sources match exact SHA-256. Original authority snapshots, executed CPU proof and their original dates/results are reused, not redownloaded or reexecuted.",
        "No historical raw 5.1 snapshot was saved in the initial 11.2 review; this callback does not invent an earlier 5.1 SHA or claim a byte-complete diff. The current necessary 5.1 slice is saved with raw bytes and exact locators.",
        "No figures or spatial visual claims occur in the current primary section or necessary 5.1 slice. No render is required for this narrow technical callback."],
    "limits": "This is a same-owner context impact callback, not a fresh independent full review or 5.1 lesson acceptance. No model training, scoring, download or GPU work was performed. The prior full report/issues/history remain opaque-preserved. The schema checker cannot decide factual truth."}
(HERE / "inspection-results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
(HERE / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
