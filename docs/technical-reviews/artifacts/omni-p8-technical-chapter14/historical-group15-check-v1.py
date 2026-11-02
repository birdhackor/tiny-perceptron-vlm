"""Read-only identity, quotation and AST checks for the three released c3 occurrences.

This does not import project/model code or execute any neural operation.
"""

import ast
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
TASK = "/root/omni_p8_technical_chapter14"
BASE = "docs/course-revision-20261029-omni/reviews/P5-actual/wave06/technical-chapter14"
REQUEST = f"{BASE}/prompted-history-request-v1.json"
REQUEST_SHA = "0f4011f9be7f5ef9e73e0a095b18828b9d19e91a0c84602a47aa7f1bdb242155"
CURRENT_SNAPSHOT = (
    "docs/course-revision-20261029-omni/reviews/"
    "source-freeze-technical-fix15/sources/chapter-14.md"
)
CURRENT_SHA = "6da5445696429a1bee5a5612397083f68a3ac1f4f8a1dbb31042f89ad3c03e44"
CONDITION = "若函式庫已有對應的設定"
UNCONDITIONAL = "每節先用固定數字追蹤一個改動，再打開本課函式庫的實作"
PRESERVED = {
    f"{BASE}/original-v1.json": "b8738dbf72a14d0200c5d55c9bfa0011691adc87034c0c549d546c7a8f00285a",
    f"{BASE}/current-v1.json": "dbeb9e6954c2608683d605c50b596b28cdb39b47d831343c144c50c846ee90c6",
    "docs/technical-reviews/artifacts/omni-p8-technical-chapter14/source-receipts-v1.json":
        "290b5b46861e7999b8a7a99b3b6f7a27e6f5fdccfeb6b3f4e34b498b9da0d632",
    "docs/technical-reviews/artifacts/omni-p8-technical-chapter14/peer-comparison-v1.json":
        "a2fd421d5072fedb71e49aac76118d9d13d7ad03064a9fb35a29d7898d55b397",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked(ref):
    data = (ROOT / ref["path"]).read_bytes()
    assert digest(data) == ref["sha256"], ref["path"]
    return json.loads(data)


def pointer(value, where):
    if not where:
        return value
    for part in where.strip("/").split("/"):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def old_source(report):
    freeze = checked(report["source_freeze"])
    matches = [(i, p) for i, p in enumerate(freeze["pages"]) if p["page_id"] == "chapter-14"]
    assert len(matches) == 1
    index, page = matches[0]
    assert page["selector"] == "before-first-section"
    assert page["source_sha256"] == report["source_sha256"]
    data = (ROOT / page["snapshot"]).read_bytes()
    assert digest(data) == page["source_sha256"]
    assert page["figures_sha256"] == report["figure_sha256"] == {}
    text = data.decode("utf-8")
    sentence = re.search(r"每節先用固定數字[^。]+。", text)[0]
    return {
        "freeze": {**report["source_freeze"], "json_pointer": f"/pages/{index}"},
        "snapshot": {"path": page["snapshot"], "sha256": page["source_sha256"]},
        "figure_sha256": {},
        "library_sentence": sentence,
        "conditional_present": CONDITION in sentence,
        "unconditional_old_phrase_present": UNCONDITIONAL in sentence,
    }


for name, expected in PRESERVED.items():
    assert digest((ROOT / name).read_bytes()) == expected, name

request = checked({"path": REQUEST, "sha256": REQUEST_SHA})
group = request["group"]
assert group["group_index"] == 15 and group["page_id"] == "chapter-14"
assert group["old_role"] == "technical" and len(group["items"]) == 3
unavailability = group["unavailability"]
unavailable = pointer(checked(unavailability["report"]), unavailability["json_pointer"])
assert unavailable["reviewer_task"] == group["old_owner"]
assert unavailable["unavailable"] is True

chapter_path = "course/chapters/14.md"
chapter_bytes = (ROOT / chapter_path).read_bytes()
boundary = re.search(rb"^## ", chapter_bytes, re.M)
intro_bytes = chapter_bytes[: boundary.start()]
assert digest(intro_bytes) == CURRENT_SHA
assert intro_bytes == (ROOT / CURRENT_SNAPSHOT).read_bytes()
intro = intro_bytes.decode("utf-8")
assert CONDITION in intro and UNCONDITIONAL not in intro
assert len(intro.splitlines()) == 6 and len(intro_bytes) == 1265
assert not re.search(r"!\[[^\]]*\]\([^)]+\)", intro)
current_sentence = re.search(r"每節先用固定數字[^。]+。", intro)[0]

chapter = chapter_bytes.decode("utf-8")
qk_match = re.search(r"^## 14\.3 [^\n]+\n.*?(?=^## |\Z)", chapter, re.M | re.S)
assert qk_match is not None
qk = qk_match[0]
local_quote = "本節只有局部方向與尺度的CPU示範。"
no_switch_quote = "TinyLM也沒有對應的整模型開關。"
assert local_quote in qk and no_switch_quote in qk
assert "tiny_perceptron" not in qk
qk_base_line = chapter[: qk_match.start()].count("\n") + 1
scope_line = chapter[: chapter.index(local_quote)].count("\n") + 1

model_path = "tiny_perceptron/model.py"
model_bytes = (ROOT / model_path).read_bytes()
model_text = model_bytes.decode("utf-8")
model_ast = ast.parse(model_text)
config = next(n for n in model_ast.body if isinstance(n, ast.ClassDef) and n.name == "ModelConfig")
fields = [n.target.id for n in config.body if isinstance(n, ast.AnnAssign)]
assert not any("qk" in name.lower() or "query_norm" in name.lower() for name in fields)
attention_constructors = [
    ast.get_source_segment(model_text, n)
    for n in ast.walk(model_ast)
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "CausalAttention"
]
assert attention_constructors == [
    "CausalAttention(config.width, config.heads, config.kv_heads, config.backend, config.rotary)"
]

attention_path = "tiny_perceptron/attention.py"
attention_bytes = (ROOT / attention_path).read_bytes()
attention_text = attention_bytes.decode("utf-8")
attention_ast = ast.parse(attention_text)
attention_class = next(n for n in attention_ast.body if isinstance(n, ast.ClassDef) and n.name == "CausalAttention")
initializer = next(n for n in attention_class.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
forward = next(n for n in attention_class.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
signature = [a.arg for a in initializer.args.args]
assert signature == ["self", "width", "heads", "kv_heads", "backend", "rotary"]
normalization_calls = [
    ast.unparse(n.func)
    for n in ast.walk(forward)
    if isinstance(n, ast.Call) and "norm" in ast.unparse(n.func).lower()
]
assert normalization_calls == []
qk_assignments = []
for node in ast.walk(forward):
    if isinstance(node, ast.Assign):
        names = [n.id for target in node.targets for n in ast.walk(target) if isinstance(n, ast.Name)]
        if set(names) & {"q", "k"}:
            qk_assignments.append({"line": node.lineno, "text": ast.get_source_segment(attention_text, node)})

occurrences = []
for item in group["items"]:
    disposition = pointer(checked(item["disposition_ref"]), item["disposition_ref"]["json_pointer"])
    assert disposition["issue_id"] == item["issue_id"]
    assert disposition["report"] == item["original_report"]
    assert disposition["json_pointer"] == item["json_pointer"]
    assert disposition["final_necessity"] == item["necessity"] == "necessary"
    report = checked(item["original_report"])
    assert report["reviewer_task"] == group["old_owner"]
    issue = pointer(report, item["json_pointer"])
    assert issue["claim_id"] == "c3" and issue["necessary"] is True
    fix = item["historical_fix_verification"]
    fix_report = pointer(checked(fix["report"]), fix["json_pointer"])
    assert fix_report["reviewer_task"] == group["old_owner"]
    assert fix_report["issues"][0]["claim_id"] == "c3"
    assert fix_report["issues"][0]["status"] == "resolved"
    at_occurrence = old_source(report)
    at_fix = old_source(fix_report)
    assert at_fix["conditional_present"]
    assert not at_fix["unconditional_old_phrase_present"]
    assert at_fix["library_sentence"] == current_sentence
    if item["issue_id"] == "TECH-chapter-14-c3":
        assert at_occurrence["unconditional_old_phrase_present"]
        assert not at_occurrence["conditional_present"]
    else:
        assert at_occurrence["conditional_present"]
    occurrences.append({
        "issue_id": item["issue_id"],
        "identity_fields_match_request": True,
        "original_status": issue["status"],
        "original_issue_locator": item["json_pointer"],
        "old_source_at_occurrence": at_occurrence,
        "old_source_at_fix": at_fix,
        "current_conditional_sentence_matches_old_fix": True,
        "current_counterexample_scope_and_ast_rechecked": True,
        "current_figure_sha256": {},
    })

for name, expected in PRESERVED.items():
    assert digest((ROOT / name).read_bytes()) == expected, name

result = {
    "schema_version": 1,
    "kind": "small-offline-historical-identity-quote-ast-check",
    "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "reviewer_task": TASK,
    "command": "UV_PROJECT_ENVIRONMENT=/workspace/tiny-perceptron-vlm/.venv PYTHONPATH=/workspace/work/omni-book-integrated-v1 uv run --extra cpu --group notebook --no-sync python docs/technical-reviews/artifacts/omni-p8-technical-chapter14/historical-group15-check-v1.py",
    "environment": {"python": sys.version.split()[0], "device": "CPU; standard-library JSON/hash/AST only, project code not imported"},
    "script": {"path": str(Path(__file__).resolve().relative_to(ROOT)), "sha256": digest(Path(__file__).read_bytes())},
    "request": {"path": REQUEST, "sha256": REQUEST_SHA},
    "unavailability_record": {**unavailability, "actual_record_unavailable": True},
    "current_intro": {
        "source": chapter_path, "selector": "before-first-section",
        "snapshot": {"path": CURRENT_SNAPSHOT, "sha256": CURRENT_SHA},
        "bytes": len(intro_bytes), "line_range": "L1–L6", "figure_sha256": {},
        "conditional_library_sentence": current_sentence,
        "unconditional_old_phrase_present": False,
        "full_introduction_read_by_reviewer": True,
    },
    "current_qk_counterexample": {
        "path": chapter_path, "sha256": digest(chapter_bytes),
        "section": "14.3", "section_sha256": digest(qk.encode("utf-8")),
        "section_start_line": qk_base_line, "scope_line": scope_line,
        "quotes": [local_quote, no_switch_quote],
        "tiny_perceptron_import_in_section": False,
        "scope": "Read only the issue-specific 14.3 section as a bounded prerequisite; no numerical example or model executed.",
    },
    "current_static_implementation": {
        "model": {"path": model_path, "sha256": digest(model_bytes), "config_fields": fields,
                  "attention_constructor": attention_constructors[0]},
        "attention": {"path": attention_path, "sha256": digest(attention_bytes),
                      "constructor_parameters": signature, "normalization_calls_in_forward": normalization_calls,
                      "qk_assignments": qk_assignments},
        "scope": "Only the visible ModelConfig → Block → CausalAttention path; no generic claim about every repository component or future plugin.",
    },
    "occurrences": occurrences,
    "preserved_files": PRESERVED,
    "limits": "The old eighteen-section fixed-number inventory, old forwards and historical accepted grades were not reused as current factual proof. No full-chapter review or historical chain constructor run.",
    "all_checks_passed": True,
}
print(json.dumps(result, ensure_ascii=False, indent=2))
