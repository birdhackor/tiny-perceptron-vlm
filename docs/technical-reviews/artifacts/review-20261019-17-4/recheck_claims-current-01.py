"""核對現行原文、引用依賴、既有證據與 tensor 型別轉換契約。"""

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "docs/technical-reviews/artifacts/review-20261019-17-4"
ORIGINAL = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_17-4/technical-17-4-original-01.json"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def section(raw, lesson):
    headings = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    for index, heading in enumerate(headings):
        if heading[0].startswith(f"## {lesson} ".encode()):
            stop = headings[index + 1].start() if index + 1 < len(headings) else len(raw)
            return raw[heading.start():stop]
    raise ValueError("找不到指定小節")


def run(command):
    completed = subprocess.run(command, cwd=ROOT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, check=True)
    return completed


class VisibleText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


original_raw = ORIGINAL.read_bytes()
assert sha(original_raw) == "202b7b1a1af94d64b840a4ee3aaf35767b12771afd9877f5f21895f43e1ed0c6"
original = json.loads(original_raw)
old_artifacts = []
for artifact in original["artifacts"]:
    digest = sha((ROOT / artifact["path"]).read_bytes())
    assert digest == artifact["sha256"]
    old_artifacts.append({"path": artifact["path"], "sha256": digest, "unchanged": True})

chapter = (ROOT / "course/chapters/17.md").read_bytes()
old_command = ["git", "show", "5ef56bc756f3:course/chapters/17.md"]
new_command = ["git", "show", "3af7f13ef959:course/chapters/17.md"]
old = run(old_command).stdout
new = run(new_command).stdout
assert sha(old) == "4328608fc1cfb9f9ac6febf838df9b9f60c2e9ffe1912e957fe3fc55824e365a"
assert sha(chapter) == "3a1053817afd9934216d8b7eed62dd129ed230af0d56d4e546f52493c46486c4"
assert chapter == new
comparisons = []
for lesson in ("17.2", "17.3", "17.4", "17.5", "17.8"):
    before, after = section(old, lesson), section(chapter, lesson)
    assert before == after
    comparisons.append({"lesson": lesson, "old_sha256": sha(before), "current_sha256": sha(after), "byte_identical": True})
snapshot = (ROOT / "docs/course-review-20261019/freeze-01/sources/17.4.md").read_bytes()
assert section(chapter, "17.4") == snapshot
freeze_path = ROOT / "docs/course-review-20261019/freeze-05.json"
assert sha(freeze_path.read_bytes()) == "e3dbd6ee0acaf92438950504795acda172b72ac17dd610a02122e2bfdfcba6ba"
freeze_page = next(page for page in json.loads(freeze_path.read_bytes())["pages"] if page["page_id"] == "17.4")
assert freeze_page["source_sha256"] == sha(snapshot)
assert freeze_page["figures_sha256"] == {}
diff_command = ["git", "diff", "5ef56bc756f3", "3af7f13ef959", "--", "course/chapters/17.md"]
difference = run(diff_command)

probe_command = [str(ROOT / ".venv/bin/python"), str(BASE / "verify_claims.py")]
probe = run(probe_command)
old_probe_path = BASE / "verification-output.json"
assert probe.stdout == old_probe_path.read_bytes()

run_directory = ROOT / "outputs/reviewer-tools/runs/20261020T085300359863Z"
extraction = json.loads((run_directory / "extraction.json").read_bytes())
execution = json.loads((run_directory / "execution.json").read_bytes())
environment = json.loads((run_directory / "environment.json").read_bytes())
old_section = json.loads((BASE / "section-execution.json").read_bytes())
stdout = (run_directory / "stdout.txt").read_bytes()
stderr = (run_directory / "stderr.txt").read_bytes()
assert stdout.decode() == old_section["files"]["stdout.txt"]
assert stderr == b""
assert execution["exit_code"] == 0
assert extraction["source_sha256"] == sha(snapshot)
assert extraction["figure_sha256"] == {}
assert extraction["source_file_sha256"] == sha(chapter)

url = "https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.to.html"
curl_command = ["curl", "-fsSL", url]
response = run(curl_command)
parser = VisibleText()
parser.feed(response.stdout.decode())
visible = " ".join(" ".join(parser.parts).split())
start = visible.index("If the self Tensor already has the correct torch.dtype and torch.device")
stop = visible.index("Note", start)
contract = visible[start:stop].strip()
assert "then self is returned." in contract
assert "the returned tensor is a copy of self" in contract
assert "copy = False" in visible

x = torch.zeros(32, 64, dtype=torch.float32, device="cpu")
conversions = []
for dtype in (torch.float32, torch.float16, torch.int8):
    converted = x.to(dtype)
    same_dtype = dtype == x.dtype
    assert (converted is x) == same_dtype
    assert (converted.data_ptr() == x.data_ptr()) == same_dtype
    conversions.append({"target_dtype": str(dtype), "same_object": converted is x, "shares_data_pointer": converted.data_ptr() == x.data_ptr(), "numel": converted.numel(), "element_bytes": converted.element_size(), "total_bytes": converted.numel() * converted.element_size()})
assert x.to(dtype=x.dtype, device=x.device) is x
assert x.to(dtype=x.dtype, copy=True) is not x

record = {
    "command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/technical-reviews/artifacts/review-20261019-17-4/recheck_claims-current-01.py",
    "accessed_on": datetime.now(UTC).date().isoformat(),
    "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "device": "cpu"},
    "old_artifacts": old_artifacts,
    "repository_code": [{"path": source["path"], "original_sha256": source["sha256"], "current_sha256": sha((ROOT / source["path"]).read_bytes())} for source in original["sources"] if source["kind"] == "repository_code"],
    "chapter_versions": {"old_command": old_command, "old_sha256": sha(old), "current_commit_command": new_command, "current_sha256": sha(chapter), "current_commit_matches_worktree": chapter == new},
    "diff": {"command": diff_command, "exit_code": difference.returncode, "stdout": difference.stdout.decode(), "stderr": difference.stderr.decode()},
    "sections": comparisons,
    "snapshot_sha256": sha(snapshot),
    "source_freeze": {"path": freeze_path.relative_to(ROOT).as_posix(), "sha256": sha(freeze_path.read_bytes()), "page": freeze_page},
    "section_execution": {"command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/review-tools/section_facts.py 'course/chapters/17.md#17.4' --execute", "tool_run_directory": run_directory.relative_to(ROOT).as_posix(), "entrypoint_stdout": {"artifact_directory": str(run_directory), "source_sha256": sha(snapshot), "python_fences": 1, "svg_references": 0, "execution_exit_code": 0}, "extraction": extraction, "execution": execution, "environment": environment, "stdout_sha256": sha(stdout), "stdout_reference": {"path": "docs/technical-reviews/artifacts/review-20261019-17-4/section-execution.json", "sha256": sha((BASE / "section-execution.json").read_bytes()), "locator": "files.stdout.txt", "byte_identical": True}, "stderr": stderr.decode()},
    "supplementary_execution": {"command_argv": probe_command, "exit_code": probe.returncode, "stdout_reference": {"path": old_probe_path.relative_to(ROOT).as_posix(), "sha256": sha(probe.stdout), "byte_identical": True}, "stderr": probe.stderr.decode(), "result": json.loads(probe.stdout)["result"]},
    "changed_dependency_sentence": {
        "url": url, "version": "PyTorch 2.14 versioned Tensor.to documentation; runtime 2.14.1", "command": curl_command, "exit_code": response.returncode, "response_sha256": sha(response.stdout), "locator": "Tensor.to opening contract and to(dtype, ..., copy=False, ...) overload", "excerpts": ["then self is returned.", "Otherwise, the returned tensor is a copy of self"], "inspection_note": "實讀開頭 dtype/device 相同即回傳 self、否則回傳指定 dtype/device 的副本，以及 dtype overload 預設 copy=False。正文呼叫 to(dtype) 沒有要求 copy=True，因此修正句成立。", "conversions": conversions, "explicit_same_dtype_same_device_returns_self": True, "explicit_copy_true_returns_new_tensor": True, "status": "verified"
    },
    "original_issues": original["issues"],
    "result": "所有核對通過；本頁與引用段落未變，原有全部主張維持成立。",
}
print(json.dumps(record, ensure_ascii=False, indent=2))
