"""Bounded source/fixture callback; never run the evaluator main or a model."""
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace
import ast
import hashlib
import json
import os
import platform
import re
import tempfile

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent
sha = lambda raw: hashlib.sha256(raw).hexdigest()
before = (ART / "evaluate-original-frozen-before-windows.py").read_bytes()
current = (ROOT / "scripts/selftrained/evaluate.py").read_bytes()
assert sha(before) == "828b7cbbbb70ae4f40598c632a440a24bfdd46f6b8f392042a01a051fc0c0111"
assert before.replace(b"str(path.relative_to(root))", b"path.relative_to(root).as_posix()").replace(b'journal.receipt_path.open("rb")', b'journal.receipt_path.open("r+b")') == current
trees = [ast.parse(raw) for raw in (before, current)]
def node(tree, name):
    return next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
identical = {}
for name in ["normalize", "positive_visual_answer", "numeric_reply_value", "edit_distance", "format_pass", "score_reply", "protocol_contents", "validate_protocol", "Generator", "summarize"]:
    identical[name] = ast.dump(node(trees[0], name), include_attributes=False) == ast.dump(node(trees[1], name), include_attributes=False)
    assert identical[name]
thresholds = [next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "THRESHOLDS" for t in n.targets)) for tree in trees]
assert ast.dump(thresholds[0], include_attributes=False) == ast.dump(thresholds[1], include_attributes=False)

class FixtureWindowsPath(PureWindowsPath):
    def resolve(self):
        return self
    def rglob(self, pattern):
        assert pattern == "*.py"
        return [self / "nested" / "probe.py", self / "__init__.py"]

namespace = {"Path": FixtureWindowsPath, "__file__": r"C:\fixture\scripts\selftrained\evaluate.py", "file_sha256": lambda path: sha(str(path).encode())}
fn = node(trees[1], "code_fingerprints")
exec(compile(ast.Module(body=[fn], type_ignores=[]), "current code_fingerprints fixture", "exec"), namespace)
keys = list(namespace["code_fingerprints"]())
assert keys == ["tiny_perceptron/__init__.py", "tiny_perceptron/nested/probe.py", "scripts/selftrained/train.py", "scripts/selftrained/evaluate.py", "scripts/selftrained/chat.py"]
assert all("\\" not in key for key in keys)

main = node(trees[1], "main")
sync_with = next(n for n in ast.walk(main) if isinstance(n, ast.With) and any(isinstance(item.context_expr, ast.Call) and isinstance(item.context_expr.func, ast.Attribute) and ast.unparse(item.context_expr.func.value) == "journal.receipt_path" for item in n.items))
modes = []
sync_calls = []
with tempfile.TemporaryDirectory(prefix="p6-19-12-fsync-") as temp:
    receipt = Path(temp) / "receipt.json"
    receipt.write_bytes(b'{"status":"fixture-only"}\n')
    original = receipt.read_bytes()
    class ReceiptPath:
        def open(self, mode):
            modes.append(mode)
            return receipt.open(mode)
    def fixture_sync(fd):
        sync_calls.append(fd)
        os.fsync(fd)
    namespace = {"journal": SimpleNamespace(receipt_path=ReceiptPath()), "os": SimpleNamespace(fsync=fixture_sync)}
    exec(compile(ast.Module(body=[sync_with], type_ignores=[]), "current receipt-fsync fixture", "exec"), namespace)
    assert modes == ["r+b"] and len(sync_calls) == 1 and receipt.read_bytes() == original

prior_report = json.loads((ART / "before-windows-source-callback-report.json").read_bytes())
hashes = []
for item in prior_report["artifacts"] + prior_report["sources"]:
    if "path" not in item or item.get("id") == "evaluation":
        continue
    digest = sha((ROOT / item["path"]).read_bytes())
    assert digest == item["sha256"], item["path"]
    hashes.append({"path": item["path"], "sha256": digest, "unchanged": True})
chapter = (ROOT / "course/chapters/19.md").read_bytes()
heading = re.search(rb"^## 19\.12 .+$", chapter, re.M)
n = re.search(rb"^## ", chapter[heading.end():], re.M)
end = heading.end() + n.start() if n else len(chapter)
section_sha = sha(chapter[heading.start():end])
assert section_sha == prior_report["source_sha256"]
out = {"reviewer_task": "/root/p6_fact_19_12", "original_evaluator_sha256": sha(before), "current_evaluator_sha256": sha(current), "section_sha256_unchanged": section_sha, "exact_two_source_changes_only": True, "unchanged_methods_AST": identical, "thresholds_AST_identical": True, "Windows_path_fixture_keys": keys, "receipt_fsync_fixture": {"mode": modes[0], "calls": len(sync_calls), "receipt_bytes_unchanged": True}, "unchanged_other_report_evidence": hashes, "data_archive_sha256": sha((ROOT / "assets/training/selftrained-v2.tar.gz").read_bytes()), "preserved_original_report_sha256": sha((ART / "before-windows-source-callback-report.json").read_bytes()), "environment": {"python": platform.python_version(), "OS": platform.platform(), "device": "cpu"}, "support_scope": "Scoring, aggregation, thresholds, prompt construction and continuation calculation are byte-equivalent or AST-identical. Fixture exercises Windows-path serialization via PureWindowsPath and real disposable receipt fsync on this Linux environment. Native Windows runtime was not executed.", "model_loading": False, "training": False, "heldout_model_evaluation": False, "neural_generation": False}
(ART / "windows-source-callback-output.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({k: out[k] for k in ["current_evaluator_sha256", "section_sha256_unchanged", "exact_two_source_changes_only", "Windows_path_fixture_keys", "receipt_fsync_fixture", "support_scope"]}, ensure_ascii=False, indent=2))
