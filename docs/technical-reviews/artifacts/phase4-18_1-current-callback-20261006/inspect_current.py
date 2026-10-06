"""Original 18.1 reviewer's narrow current-context callback; read-only source inspection."""
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
PRIOR = ROOT / "docs/technical-reviews/artifacts/phase4-18_1-factual-fresh"
TASK = "/root/phase4_factual_coordinator/factual_18_1"
EXPECTED_SECTION = "2d50fee86657a86a9471cdf1713b8b73f54f729b6c8957a1e04730f72a4d67ae"
EXPECTED_INTRO = "803ef51cbe69ad31a5d1222179c56cff83160643ba928ae43e02787e0e6e284e"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def section(raw, lesson):
    raw.decode("utf-8")
    headings = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
    positions = [i for i, h in enumerate(headings) if re.match(rb"^## " + re.escape(lesson.encode()) + rb" ", h[0])]
    assert len(positions) == 1
    i = positions[0]
    start = headings[i].start()
    end = headings[i + 1].start() if i + 1 < len(headings) else len(raw)
    return raw[start:end], raw[:start].count(b"\n") + 1, raw[:start]


inputs = HERE / "inputs"
inputs.mkdir(exist_ok=True)
environment = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "torch_package_version": importlib.metadata.version("torch"), "tokenizers_package_version": importlib.metadata.version("tokenizers"), "execution_kind": "read-only UTF8/hash/AST/context inspection", "device": "No tensor/model execution; CPU interpreter", "downloads": "none", "model_runs": "none"}
(HERE / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2) + "\n")
raw = (ROOT / "course/chapters/18.md").read_bytes()
own, own_line, intro = section(raw, "18.1")
assert digest(own) == EXPECTED_SECTION
assert digest(intro) == EXPECTED_INTRO
for filename, data in [("current-chapter18-frozen.md", raw), ("current-section18_1.md", own), ("current-intro18.md", intro)]:
    (inputs / filename).write_bytes(data)
context, context_line, _ = section((ROOT / "course/chapters/07.md").read_bytes(), "7.11")
(inputs / "current-context7_11.md").write_bytes(context)
old_context = (PRIOR / "inputs/context-7.11.md").read_bytes()

print("CURRENT CHAPTER INTRO UTF8 BYTES SHA", digest(intro))
print(intro.decode("utf8"), end="")
print("CURRENT COMPLETE 18.1 UTF8 BYTES SHA", digest(own))
for lineno, line in enumerate(own.decode("utf8").splitlines(), own_line):
    print(f"{lineno}: {line}")
print("CURRENT NECESSARY 7.11 UTF8 BYTES SHA", digest(context))
for lineno, line in enumerate(context.decode("utf8").splitlines(), context_line):
    print(f"{lineno}: {line}")

# Prior proof/history was archived opaquely before this inspection. Verify all bytes.
archival = json.loads((HERE / "history/opaque-history-manifest.json").read_bytes())
proof_matches = []
for item in archival["prior_files"]:
    actual = digest((ROOT / item["path"]).read_bytes())
    assert actual == item["sha256"]
    proof_matches.append({"path": item["path"], "sha256": actual, "matched_opaque_preinspection_manifest": True})

# Exact version comparison to the previously personally inspected original bytes.
versions = []
for name in ["tiny_perceptron/alignment.py", "tiny_perceptron/training.py", "tiny_perceptron/model.py", "tiny_perceptron/data.py", "scripts/course_experiments/common.py"]:
    old = (PRIOR / "inputs" / name.replace("/", "__")).read_bytes()
    current = (ROOT / name).read_bytes()
    assert digest(old) == digest(current)
    versions.append({"source": name, "old_sha256": digest(old), "current_sha256": digest(current), "exact_bytes_unchanged": True})

# Full compression-file hash may differ from the recorded experiment; only the
# previously inspected distillation methods control this narrow claim.
current_compression = (ROOT / "scripts/course_experiments/compression.py").read_bytes()
original_compression = (PRIOR / "inputs/compression-recorded.py").read_bytes()
tree_current, tree_original = ast.parse(current_compression), ast.parse(original_compression)
method_matches = []
for name in ["_teacher", "_cache_text", "_fit_text", "_distill_case", "run_distillation"]:
    new = next(n for n in tree_current.body if isinstance(n, ast.FunctionDef) and n.name == name)
    old = next(n for n in tree_original.body if isinstance(n, ast.FunctionDef) and n.name == name)
    new_ast, old_ast = ast.dump(new, include_attributes=False), ast.dump(old, include_attributes=False)
    assert new_ast == old_ast
    method_matches.append({"method": name, "current_lines": [new.lineno, new.end_lineno], "recorded_lines": [old.lineno, old.end_lineno], "ast_sha256": digest(new_ast.encode()), "AST_unchanged": True})

source_versions = []
for name in ["distillation", "sft", "style", "moe"]:
    original = (PRIOR / "inputs" / (name + "-original.json")).read_bytes()
    current = (ROOT / "docs/course-experiments/results" / (name + ".json")).read_bytes()
    # Only unchanged bytes are checked here: no original result comments are read.
    assert digest(original) == digest(current)
    source_versions.append({"source": "docs/course-experiments/results/" + name + ".json", "original_sha256": digest(original), "current_sha256": digest(current), "exact_bytes_unchanged": True, "current_content_pointers_read": []})

instruction_versions = []
for name in ["docs/review-tools/factual-reviewer-instructions.md", ".agents/skills/clear-tutorial/references/review-protocol.md", "scripts/check_technical_reviews.py", "docs/review-tools/section_facts.py"]:
    original = (ROOT / name).read_bytes()
    (inputs / name.replace("/", "__")).write_bytes(original)
    instruction_versions.append({"path": name, "sha256": digest(original)})

result = {"reviewer_task": TASK, "current_own_source": "course/chapters/18.md#18.1", "current_source_sha256": digest(own), "current_source_lines": [own_line, own_line + own.count(b"\n") - 1], "current_intro_sha256": digest(intro), "intro_lines": [1, own_line - 1], "current_whole_chapter_frozen_sha256": digest(raw), "current_context_source": "course/chapters/07.md#7.11", "current_context_sha256": digest(context), "current_context_lines": [context_line, context_line + context.count(b"\n") - 1], "prior_context_sha256": digest(old_context), "context_bytes_changed": context != old_context, "figure_sha256": {}, "versioned_helpers": versions, "original_compression_full_sha256": digest(original_compression), "current_compression_full_sha256": digest(current_compression), "distillation_method_AST_matches": method_matches, "raw_result_versions": source_versions, "prior_artifact_byte_verifications": proof_matches, "instruction_versions": instruction_versions, "prior_canonical_opaque_path": archival["prior_canonical_opaque_path"], "prior_canonical_sha256": archival["prior_canonical_sha256"], "prior_proof_archive_path": archival["prior_proof_archive_path"], "prior_proof_archive_sha256": archival["prior_proof_archive_sha256"], "actual_execution_scope": "Current source/context and prior proof/code/source version inspection only; prior original CPU/paper/official-source evidence is reused after exact hash/AST checks, not re-executed or fetched."}
(HERE / "inspection-facts.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print("INSPECTION FACTS:", json.dumps({k: v for k, v in result.items() if k != "prior_artifact_byte_verifications"}, ensure_ascii=False))
print("ALL CURRENT SOURCE/INTRO/CTX/PRIORPROOF VERSION CHECKS PASSED")
