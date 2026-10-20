"""比對前置段落並重跑原本的算術核對；不改寫既有證據。"""

import hashlib
import io
import json
import os
import re
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = "docs/technical-reviews/artifacts/review-20261019-C-4"
ORIGINAL = "docs/course-review-20261019/reviews/rv1019_technical_C-4/technical-C-4-original-01.json"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return (ROOT / path).read_bytes()


def section(raw, identifier):
    text = raw.decode("utf-8")
    match = re.search(r"^## " + re.escape(identifier) + r" .+$", text, re.M)
    assert match is not None
    following = re.search(r"^## ", text[match.end():], re.M)
    end = match.end() + following.start() if following else len(text)
    return text[match.start():end].encode("utf-8")


def command(argv):
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, check=True)
    return {"argv": argv, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def execute_fence(raw):
    fence = re.search(r"```python\n(.*?)\n```", raw.decode("utf-8"), re.S)[1]
    output = io.StringIO()
    with redirect_stdout(output):
        exec(compile(fence, "prerequisite-example", "exec"), {})
    return {"code_sha256": sha(fence.encode()), "stdout": output.getvalue()}


original = json.loads(read(ORIGINAL))
assert sha(read(ORIGINAL)) == "59bbba6cc61af28d486e74563714e5f9d6e79d1969217502ae3b4c2eaca275d1"
preserved = []
for item in [{"path": ORIGINAL, "sha256": sha(read(ORIGINAL))}, *original["artifacts"],
             {"path": ARTIFACT + "/self-check.txt", "sha256": "f08b4503e20bbd576b80d5cf848194134b75d76038bad69e51ce3dda9ae82fde"}]:
    actual = sha(read(item["path"]))
    assert actual == item["sha256"]
    preserved.append({"path": item["path"], "recorded_sha256": item["sha256"], "current_sha256": actual, "unchanged": True})

prerequisites = []
for path, revision, identifier, expected_sha, expression in [
    ("course/chapters/01.md", "f47d47b91432", "1.4", "97047e1172975518ac6bad6cf847f911ebccde73f4ab10b9349246e36951ed14", "from collections import Counter"),
    ("course/chapters/06.md", "5ef56bc756f3", "6.2", "e5ba622295b957ac094f3c39e76a6a29beaf89e5db2b96c750732f4b837e4820", "pairs.most_common(1)[0][0]"),
]:
    current = read(path)
    assert sha(current) == expected_sha
    previous_result = command(["git", "show", revision + ":" + path])
    previous = previous_result["stdout"].encode("utf-8")
    recorded = next(s["sha256"] for s in original["sources"] if s.get("path") == path)
    assert sha(previous) == recorded
    target_result = command(["git", "show", "6a7aa1de5227:" + path])
    assert target_result["stdout"].encode("utf-8") == current
    before = section(previous, identifier)
    after = section(current, identifier)
    assert before == after
    assert expression in after.decode("utf-8")
    old_execution = execute_fence(before)
    new_execution = execute_fence(after)
    assert old_execution == new_execution
    prerequisites.append({
        "path": path,
        "recorded_sha256": recorded,
        "current_sha256": sha(current),
        "previous_revision": revision,
        "compared_revision": "6a7aa1de5227",
        "current_equals_compared_revision": True,
        "git_diff": command(["git", "diff", revision, "6a7aa1de5227", "--", path]),
        "referenced_heading": after.decode("utf-8").splitlines()[0],
        "previous_section_sha256": sha(before),
        "current_section_sha256": sha(after),
        "section_unchanged": True,
        "relevant_expression": expression,
        "execution": new_execution,
        "execution_unchanged": True,
        "assessment": "全檔的改動沒有觸及此處被引用的段落；語法出處與實跑結果仍支持本節的前置引用主張。其他小節的改寫不在本節逐主張判斷的範圍。",
    })

frozen = read("docs/course-review-20261019/freeze-01/sources/C.4.md")
assert section(read("course/chapters/0C.md"), "C.4") == frozen
freeze_path = "docs/course-review-20261019/freeze-06.json"
assert sha(read(freeze_path)) == "9f5489d6e94946c6aa8690a1672b7f931b02e8c61ebc59f84f179eba5e44a5c9"
page = next(p for p in json.loads(read(freeze_path))["pages"] if p["page_id"] == "C.4")
assert page["source_sha256"] == sha(frozen) == original["source_sha256"]
assert page["figures_sha256"] == original["figure_sha256"]
for path, digest in original["figure_sha256"].items():
    assert sha(read(path)) == digest

old_audit = json.loads(read(ARTIFACT + "/verification-output.json"))
for path, digest in old_audit["inputs"].items():
    if path not in {p["path"] for p in prerequisites}:
        assert sha(read(path)) == digest

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
rerun = command([sys.executable, ARTIFACT + "/verify.py"])
audit = json.loads(rerun["stdout"])
assert audit["checked_candidates"] == 720
assert audit["all_replayed_budgets_equal_record"]
for field in ("budgets", "examples", "toy", "tie_rule", "generator", "verifier_probes", "linked_example_checks"):
    assert audit[field] == old_audit[field]
audit["original_verifier_command"] = audit["command"]
audit["command"] = "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python " + ARTIFACT + "/recheck-current-01.py"
audit["verification_subprocess"] = {key: rerun[key] for key in ("argv", "returncode", "stderr")}
audit["inputs"][freeze_path] = sha(read(freeze_path))
audit["preserved_files"] = preserved
audit["prerequisite_rechecks"] = prerequisites
audit["current_section_equals_frozen"] = True
audit["changed_sentences_in_reviewed_section"] = []
audit["original_issues"] = original["issues"]
audit["resolved_issue_ids"] = []
audit["rechecked_claim_ids"] = [c["id"] for c in original["claims"]]
audit["result"] = "全部核對成立；相關前置段落逐位元組相同且程式結果相同，所有固定候選重算結果與原報告一致。"
print(json.dumps(audit, ensure_ascii=False, indent=2))
