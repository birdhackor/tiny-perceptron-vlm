"""核對既有證據完整性、章檔差異與合成圖片行為。"""

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
ORIGINAL = ROOT / "docs/course-review-20261019/reviews/rv1019_technical_10-1/technical-10-1-original-01.json"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def execute(arguments):
    completed = subprocess.run(arguments, cwd=ROOT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, text=True, timeout=180)
    return {"command_argv": arguments, "exit_code": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}


def section(raw):
    headers = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    index = next(index for index, item in enumerate(headers) if item[0].startswith(b"## 10.1 "))
    return raw[headers[index].start():headers[index + 1].start()]


original_bytes = ORIGINAL.read_bytes()
assert digest(original_bytes) == "96eeb197ad4fad3251c75c08e93fc2132415d3137f8799f9b5cea0d5fd006793"
original = json.loads(original_bytes)
assert original["issues"] == []
integrity = []
for artifact in original["artifacts"]:
    actual = digest((ROOT / artifact["path"]).read_bytes())
    assert actual == artifact["sha256"]
    integrity.append({"path": artifact["path"], "expected_sha256": artifact["sha256"], "actual_sha256": actual, "unchanged": True})
repo_sources = []
for source in original["sources"]:
    if source["kind"] != "repository_code":
        continue
    actual = digest((ROOT / source["path"]).read_bytes())
    if source["path"] == "course/chapters/10.md":
        assert actual == "a90293b8deaec91d8b634525c1ff6174d55420cab8dfb7b13b7bd906e01877eb"
    else:
        assert actual == source["sha256"]
    repo_sources.append({"path": source["path"], "previous_sha256": source["sha256"], "current_sha256": actual, "unchanged": actual == source["sha256"]})

change = execute(["git", "diff", "5ef56bc756f3", "3af7f13ef959", "--", "course/chapters/10.md"])
assert change["exit_code"] == 0
old = execute(["git", "show", "5ef56bc756f3:course/chapters/10.md"])
new = execute(["git", "show", "3af7f13ef959:course/chapters/10.md"])
assert old["exit_code"] == new["exit_code"] == 0
old_raw, new_raw = old["stdout"].encode(), new["stdout"].encode()
assert digest(old_raw) == "0661467822545c12eae64a568a9bb9f417989f7c9757eae6deb22a336a1de3a8"
assert digest(new_raw) == "a90293b8deaec91d8b634525c1ff6174d55420cab8dfb7b13b7bd906e01877eb"
current = (ROOT / "course/chapters/10.md").read_bytes()
frozen = (ROOT / "docs/course-review-20261019/freeze-01/sources/10.1.md").read_bytes()
assert current == new_raw
assert section(old_raw) == section(new_raw) == section(current) == frozen
removed = [line[1:] for line in change["stdout"].splitlines() if line.startswith("-") and not line.startswith("---")]
added = [line[1:] for line in change["stdout"].splitlines() if line.startswith("+") and not line.startswith("+++")]
assert len(removed) == len(added) == 1
assert "F.normalize" in added[0] and "整數 tensor 不行" in added[0]

verification = execute([str(ROOT / ".venv/bin/python"), str(BASE / "verify_scene.py")])
assert verification["exit_code"] == 0 and not verification["stderr"]
result = json.loads(verification["stdout"])
assert result["result"] == "全部核對成立"
previous_result = json.loads((BASE / "verification-output.json").read_bytes())
assert result == previous_result

facts = execute([str(ROOT / ".venv/bin/python"), "docs/review-tools/section_facts.py", "course/chapters/10.md#10.1", "--execute"])
assert facts["exit_code"] == 0 and not facts["stderr"]
facts_summary = json.loads(facts["stdout"])
assert facts_summary["source_sha256"] == digest(frozen)
assert facts_summary["python_fences"] == 2 and facts_summary["execution_exit_code"] == 0
run_directory = Path(facts_summary["artifact_directory"])
section_files = {}
for name in ("stdout.txt", "stderr.txt", "environment.json", "execution.json", "extraction.json"):
    raw = (run_directory / name).read_bytes()
    section_files[name] = {"sha256": digest(raw), "bytes": len(raw), "content": raw.decode("utf-8")}
assert section_files["stdout.txt"]["content"] == (BASE / "section-stdout.txt").read_text()
assert section_files["stderr.txt"]["content"] == ""

import torch
from torch.nn import functional as F

integer_input = torch.tensor([[0, 3], [2, 0]])
try:
    F.normalize(integer_input, dim=-1)
except RuntimeError as error:
    integer_result = {"type": type(error).__name__, "message": str(error)}
else:
    raise AssertionError("整數輸入未觸發預期例外")
float_result = F.normalize(integer_input.float(), dim=-1)
assert torch.equal(float_result, torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
complex_input = torch.tensor([[1.0 + 1.0j, 0.0 + 0.0j]])
complex_result = F.normalize(complex_input, dim=-1)
assert torch.allclose(complex_result.abs().square().sum(-1), torch.ones(1))

assert digest(ORIGINAL.read_bytes()) == digest(original_bytes)
for artifact in integrity:
    assert digest((ROOT / artifact["path"]).read_bytes()) == artifact["expected_sha256"]
head = execute(["git", "rev-parse", "HEAD"])
assert head["exit_code"] == 0
result["recheck"] = {
    "original_report": {"path": str(ORIGINAL.relative_to(ROOT)), "sha256": digest(original_bytes), "unchanged_after_execution": True},
    "previous_artifacts": integrity,
    "repository_sources": repo_sources,
    "head": head["stdout"].strip(),
    "diff": change,
    "old_version_sha256": digest(old_raw),
    "new_version_sha256": digest(new_raw),
    "source_section_sha256": digest(frozen),
    "section_bytes_identical_in_old_new_current_and_freeze": True,
    "removed_paragraphs": removed,
    "added_paragraphs": added,
    "previous_issue_count": len(original["issues"]),
    "verification_command": verification,
    "verification_equals_previous_result": result == previous_result,
    "section_command": facts,
    "section_summary": facts_summary,
    "section_files": section_files,
    "changed_statement_verification": {
        "scope": "僅核對章檔改寫句的 dtype 行為，未審閱其他小節全文。",
        "api": "torch.nn.functional.normalize(input, dim=-1)",
        "integer_input": integer_input.tolist(),
        "integer_dtype": str(integer_input.dtype),
        "integer_result": integer_result,
        "float_output": float_result.tolist(),
        "complex_dtype": str(complex_result.dtype),
        "complex_output_abs_squared_sum": complex_result.abs().square().sum(-1).tolist(),
        "conclusion": "整數 tensor 會報錯；浮點例可執行，複數也可執行，因此移除只接受浮點數的限制符合本次環境。"
    },
    "original_evidence_files_explicitly_read": [artifact["path"] for artifact in original["artifacts"]] + ["docs/course-review-20261019/figure-renders-04.json", result["figure"]["inspected_png"]],
    "all_original_artifacts_preserved": True
}
print(json.dumps(result, ensure_ascii=False, indent=2))
