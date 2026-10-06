"""Current 19.2 callback: exact diff, code contract, inherited evidence, page view.

This performs no model forward, training, checkpoint loading or score evaluation.
"""

import ast
import difflib
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PRIOR = ROOT / "docs/technical-reviews/artifacts/phase4-factual-19_2-independent"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def emit(name, value):
    print(json.dumps({"check": name, "observed": value}, ensure_ascii=False))


environment = {
    "python": platform.python_version(),
    "python_executable": sys.executable,
    "torch_distribution_version": importlib.metadata.version("torch"),
    "playwright": importlib.metadata.version("playwright"),
    "compute_scope": "CPU AST/hash inspection only; no model forward or training",
    "browser": "system /usr/bin/chromium, headless, CPU",
    "cwd": str(Path.cwd()),
}
save("environment.json", environment)

opaque = json.loads((HERE / "opaque-preservation.json").read_text())
assert sha(HERE / "prior-canonical-report.opaque.json") == opaque["prior_report"]["sha256"]
prior_report = json.loads((HERE / "prior-canonical-report.opaque.json").read_text())
assert prior_report["reviewer_task"] == "/root/phase4_factual_coordinator/factual_19_2"
assert prior_report["lesson_id"] == "19.2"
# Only this original reviewer's own report/evidence is used, never a peer report.
checked_prior_files = []
for entry in opaque["prior_evidence_files_preserved_in_place"]:
    path = ROOT / entry["path"]
    assert sha(path) == entry["sha256"]
    checked_prior_files.append(entry)
for artifact in prior_report["artifacts"]:
    assert sha(ROOT / artifact["path"]) == artifact["sha256"]
checked_current_code = []
for source in prior_report["sources"]:
    if source["kind"] == "repository_code":
        assert sha(ROOT / source["path"]) == source["sha256"]
        checked_current_code.append({"path": source["path"], "sha256": source["sha256"]})
save("inherited-evidence-fingerprints.json", {
    "reviewer_task": prior_report["reviewer_task"],
    "all_prior_artifacts_checked_against_own_recorded_SHA": True,
    "unchanged_current_repository_code": checked_current_code,
    "prior_proof_files": checked_prior_files,
    "retained_external_source_access_date": "2026-10-05",
    "reuse_scope": "All unchanged routing/capacity, parameter/fp32, PAD/load balance, L4 recorded measurement, allocator and design-limit claims; original CPU/paper evidence preserved and hash checked, not re-executed or re-fetched.",
})
emit("prior_evidence_fingerprint_reuse", {
    "prior_report_SHA": opaque["prior_report"]["sha256"],
    "prior_proof_file_count": len(checked_prior_files),
    "all_prior_artifact_SHA_matches": True,
    "unchanged_current_repository_code": checked_current_code,
    "no_model_or_pipeline_rerun": True,
})

raw = (ROOT / "course/chapters/19.md").read_bytes()
headers = list(re.finditer(rb"(?m)^## [^\r\n]+", raw))
i = next(i for i, h in enumerate(headers) if h[0].startswith(b"## 19.2 "))
current = raw[headers[i].start():headers[i + 1].start()]
assert hashlib.sha256(current).hexdigest() == "e5fb93785bd2ddaa0948e2c58bc563179f742195e56e27262106618bfedda9d1"
assert current == (HERE / "current-section.md").read_bytes()
old = (PRIOR / "section.md").read_bytes()
old_lines, current_lines = old.splitlines(keepends=True), current.splitlines(keepends=True)
changes = [op for op in difflib.SequenceMatcher(a=old_lines, b=current_lines).get_opcodes() if op[0] != "equal"]
assert changes == [("replace", 4, 5, 4, 5)]
assert current_lines[4].decode().startswith("共同核心的注意力、字元嵌入與輸出表由各位 expert 共用；這裡指專家之間共用這些零件，輸入嵌入表與輸出表仍各自儲存。")
fence_pattern = rb"(?ms)^```python\r?\n(.*?)^```"
assert re.findall(fence_pattern, old) == re.findall(fence_pattern, current)
assert re.findall(rb"!\[[^\]]*\]\(([^)]+)\)", current) == []
emit("complete_current_section_diff", {
    "source_SHA": hashlib.sha256(current).hexdigest(),
    "prior_source_SHA": hashlib.sha256(old).hexdigest(),
    "changed_opcodes": changes,
    "current_first_line": raw[:headers[i].start()].count(b"\n") + 1,
    "only_change": "Clarifies expert-shared modules versus untied input embedding/output tables.",
    "all_other_text_and_python_fences_exactly_unchanged": True,
    "new_long_recipes_or_commands": False,
    "course_figures": [],
})

def parsed(name):
    text = (ROOT / name).read_text()
    return text, ast.parse(text)


def class_named(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)


def method_named(cls, name):
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)


def assignments(fn):
    return {
        ast.unparse(target): ast.unparse(node.value)
        for node in ast.walk(fn)
        if isinstance(node, ast.Assign)
        for target in node.targets
    }


cap_text, cap_tree = parsed("tiny_perceptron/capstone.py")
model_text, model_tree = parsed("tiny_perceptron/model.py")
moe_text, moe_tree = parsed("tiny_perceptron/modern.py")
config = next(n for n in cap_tree.body if isinstance(n, ast.FunctionDef) and n.name == "default_config")
returned_config = next(n.value for n in config.body if isinstance(n, ast.Return))
assert next(k.value.value for k in returned_config.keywords if k.arg == "tied") is False
lm = class_named(model_tree, "TinyLM")
lm_init = method_named(lm, "__init__")
lm_assignments = assignments(lm_init)
assert lm_assignments["self.embedding"].startswith("nn.Embedding(")
assert lm_assignments["self.output"].startswith("nn.Linear(")
tying = next(n for n in lm_init.body if isinstance(n, ast.If) and ast.unparse(n.test) == "c.tied")
assert assignments(tying)["self.output.weight"] == "self.embedding.weight"
block_init = method_named(class_named(model_tree, "Block"), "__init__")
assert assignments(block_init)["self.attention"].startswith("CausalAttention(")
assert "MoEFFN(" in assignments(block_init)["self.ffn"]
moe_init = method_named(class_named(moe_tree, "MoEFFN"), "__init__")
assert assignments(moe_init)["self.experts"].startswith("nn.ModuleList([DenseFFN(")
assert not any(key in assignments(moe_init) for key in ["self.attention", "self.embedding", "self.output"])
cap_init = method_named(class_named(cap_tree, "CapstoneModel"), "__init__")
assert assignments(cap_init)["self.language"] == "TinyLM(self.config)"
excerpts = []
for name, text, fn in [
    ("tiny_perceptron/capstone.py", cap_text, config),
    ("tiny_perceptron/capstone.py", cap_text, cap_init),
    ("tiny_perceptron/model.py", model_text, block_init),
    ("tiny_perceptron/model.py", model_text, lm_init),
    ("tiny_perceptron/model.py", model_text, method_named(lm, "forward")),
    ("tiny_perceptron/modern.py", moe_text, moe_init),
]:
    excerpts.append({"path": name, "source_SHA": sha(ROOT / name), "method": fn.name, "first_line": fn.lineno, "last_line": fn.end_lineno, "original_excerpt": "\n".join(text.splitlines()[fn.lineno - 1:fn.end_lineno])})
save("current-code-contract-excerpts.json", excerpts)
# Reinspect the exact original CPU observation relevant to this clarification.
cpu_checks = {}
for line in (PRIOR / "cpu-stdout.txt").read_text().splitlines():
    if line.startswith("{"):
        item = json.loads(line)
        cpu_checks[item["check"]] = item["observed"]
counts = cpu_checks["independent_parameter_and_byte_count"]
assert counts["MoE64"]["embedding_and_output_untied"] is True
assert counts["Dense64"]["embedding_and_output_untied"] is True
emit("changed_shared_vs_tied_claim_inspection", {
    "default_config_tied": False,
    "language_embedding_and_output_separately_constructed": True,
    "tying_branch_requires_c_tied": True,
    "one_attention_per_layer_outside_expert_FFN": True,
    "experts_contain_DenseFFN_not_embedding_attention_output": True,
    "prior_CPU_object_and_storage_identity_check_reused_after_SHA_verification": {
        "MoE64_untied": counts["MoE64"]["embedding_and_output_untied"],
        "Dense64_untied": counts["Dense64"]["embedding_and_output_untied"],
        "stdout_SHA": sha(PRIOR / "cpu-stdout.txt"),
    },
    "scope": "Shared across experts in the old configured core; embedding/output are not tied to each other. No assertion that every possible model configuration is untied.",
})

served = (HERE / "current-served-19.2.html").read_bytes()
assert served == (ROOT / "outputs/site/19.2.html").read_bytes()
page_checks = []
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/chromium", headless=True, args=["--no-sandbox"])
    environment["chromium_version"] = browser.version
    save("environment.json", environment)
    for label, width, height in [("desktop", 1280, 800), ("mobile", 390, 844)]:
        page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
        response = page.goto("http://127.0.0.1:8765/19.2.html", wait_until="networkidle", timeout=15000)
        assert response is not None and response.status == 200
        page.locator("article details").evaluate_all("els => els.forEach(el => el.open = true)")
        page.evaluate("document.fonts.ready")
        article = page.locator("article")
        visible_text = article.inner_text()
        assert "共同核心的注意力、字元嵌入與輸出表由各位 expert 共用" in visible_text
        assert "輸入嵌入表與輸出表仍各自儲存" in visible_text
        assert "沒有綁成同一份權重" in visible_text
        assert "MoE中位時間卻為21.464毫秒，Dense為11.338毫秒" in visible_text
        assert article.locator("img").count() == 0
        screenshot = HERE / (label + "-current.png")
        page.screenshot(path=str(screenshot), full_page=True)
        (HERE / (label + "-article-text.txt")).write_text(visible_text + "\n")
        page_checks.append({"viewport": [width, height], "HTTP_status": response.status, "expanded_details": article.locator("details[open]").count(), "article_image_count": 0, "clarified_text_and_old_numbers_present": True, "screenshot": str(screenshot.relative_to(ROOT)), "screenshot_SHA": sha(screenshot)})
        page.close()
    browser.close()
save("site-inspection.json", {"url": "http://127.0.0.1:8765/19.2.html", "served_HTML_SHA": sha(HERE / "current-served-19.2.html"), "matches_current_outputs_site_file": True, "pages": page_checks, "visual_view_image_status": "Screenshots captured; reviewer must then actually view them."})
emit("current_site_render", page_checks)
emit("context_version", {"intro": None, "fig": {}, "context_file": str((HERE / "callback-context.json").relative_to(ROOT)), "context_file_SHA": sha(HERE / "callback-context.json")})
print("ALL CURRENT CALLBACK INSPECTION ASSERTIONS PASSED")
