"""Read-only callback for a diagnostic path-normalization change."""

import ast
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[6]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


previous_path = "docs/course-revision-20261016/reviews/technical-order-chapter-19-callback-02.json"
previous = json.loads((ROOT / previous_path).read_bytes())
guard_path = "scripts/check_lesson_order.py"
old_pin = next(s["sha256"] for s in previous["sources"] if s.get("path") == guard_path)
head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
old = subprocess.check_output(["git", "show", "HEAD:" + guard_path], cwd=ROOT)
current = (ROOT / guard_path).read_bytes()
assert sha(old) == old_pin
before = '            failures.append(f"{path.relative_to(root)}: 章內節號應依序為1至{len(ids)}，實際為{\', \'.join(ids)}")\n'
after = '            failures.append(\n                f"{path.relative_to(root).as_posix()}: 章內節號應依序為1至{len(ids)}，實際為{\', \'.join(ids)}"\n            )\n'
assert old.count(before.encode()) == 1
assert old.replace(before.encode(), after.encode()) == current


class RemoveDiagnosticPosix(ast.NodeTransformer):
    def __init__(self):
        self.changed = 0

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "as_posix":
            assert not node.args and not node.keywords
            assert isinstance(node.func.value, ast.Call)
            assert isinstance(node.func.value.func, ast.Attribute)
            assert node.func.value.func.attr == "relative_to"
            self.changed += 1
            return node.func.value
        return node


transform = RemoveDiagnosticPosix()
normalized = transform.visit(ast.parse(current))
assert transform.changed == 1
assert ast.dump(normalized, include_attributes=False) == ast.dump(ast.parse(old), include_attributes=False)
positive = subprocess.run([sys.executable, guard_path], cwd=ROOT, capture_output=True, text=True)
assert positive.returncode == 0, positive.stderr
positive_result = json.loads(positive.stdout)
assert positive_result["status"] == "passed" and positive_result["failures"] == []
source_pins = []
for source in previous["sources"]:
    if source["kind"] == "repository_code":
        observed = sha((ROOT / source["path"]).read_bytes())
        changed = observed != source["sha256"]
        assert changed == (source["path"] == guard_path)
        source_pins.append({"id": source["id"], "path": source["path"], "old_sha256": source["sha256"], "current_sha256": observed, "changed": changed})
artifacts = []
for artifact in previous["artifacts"]:
    assert sha((ROOT / artifact["path"]).read_bytes()) == artifact["sha256"]
    artifacts.append({"id": artifact["id"], "path": artifact["path"], "sha256": artifact["sha256"]})
freeze_path = "docs/course-revision-20261016/freeze-19.json"
freeze = json.loads((ROOT / freeze_path).read_bytes())
page = next(p for p in freeze["pages"] if p["page_id"] == "chapter-19")
assert page["source_sha256"] == previous["source_sha256"]
assert sha((ROOT / page["snapshot"]).read_bytes()) == previous["source_sha256"]
result = {
    "reviewer_task": "/root/backfill_tech_chapter_19",
    "current_context": "informed",
    "recorded_at": datetime.now(timezone.utc).isoformat(),
    "command": ".venv/bin/python docs/technical-reviews/artifacts/revision-20261016/order-review/chapter-19/guard-callback-03.py",
    "environment": {"python": platform.python_version(), "platform": platform.platform(), "device": "cpu", "native_windows_run": "not performed"},
    "result": "all assertions passed",
    "previous_callback": {"path": previous_path, "sha256": sha((ROOT / previous_path).read_bytes())},
    "diagnostic_only_diff": {"source": guard_path, "old_git_revision": head, "old_sha256": sha(old), "current_sha256": sha(current), "before": before, "after": after, "whole_file_exact_replacement": True, "ast_identical_after_removing_one_diagnostic_as_posix_call": True},
    "current_positive_command": [sys.executable, guard_path],
    "current_positive_exit_code": positive.returncode,
    "current_positive_stdout": positive.stdout,
    "current_positive_stderr": positive.stderr,
    "current_positive_result": positive_result,
    "current_repository_pins": source_pins,
    "previous_artifacts_unchanged": artifacts,
    "freeze19_sha256": sha((ROOT / freeze_path).read_bytes()),
    "chapter19_source_sha256": page["source_sha256"],
    "scope": "Unique diagnostic formatting change only: wrong-order acceptance/rejection criteria are unchanged, while the error label now formats its relative path with forward slashes on Windows too. Actual positive audit was executed in this Linux environment. Native Windows behavior is left to CI; this callback does not claim to have executed native Windows, nor rerun any model, formal test, HF download, HTML build or shared review seal.",
}
out = Path(__file__).with_suffix(".json")
assert not out.exists()
out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"result": result["result"], "guard_sha256": sha(current), "positive_failures": positive_result["failures"], "changed_repository_sources": [s["id"] for s in source_pins if s["changed"]], "output": str(out.relative_to(ROOT))}))
