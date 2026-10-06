"""Inspect only the current 19.11 page; no model or lesson code runs."""
import ast
import hashlib
import importlib.metadata
import json
import re
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

A = Path(__file__).resolve().parent
ROOT = Path.cwd()
URL = "http://127.0.0.1:8765/19.11.html"
sha = lambda b: hashlib.sha256(b).hexdigest()
source = (A / "section-current.md").read_bytes()
fences = re.findall(rb"(?ms)^```([^\n]*)\n(.*?)^```\s*$", source)
current_html = (ROOT / "outputs/site/19.11.html").read_bytes()
http_html = (A / "http-19.11.html").read_bytes()
assert current_html == http_html
changed_table = "四份階段／分支的FP32推論檔"
changed_recipe = "如果要自己重做四份階段／分支模型"

# Inspect only actual changed-claim methods, not author report strings.
module = ast.parse((ROOT / "tiny_perceptron/capstone.py").read_bytes())
stage_assignment = next(n for n in module.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "STAGES" for t in n.targets))
stages = ast.literal_eval(stage_assignment.value)
assert stages == ("pretrain", "sft", "joint", "dpo")
trainer = ast.parse((ROOT / "scripts/course_experiments/capstone.py").read_bytes())
train_stage = next(n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == "train_stage")
predecessor_expr = next(n for n in ast.walk(train_stage) if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "expected" for t in n.targets))
assert ast.unparse(predecessor_expr.value) == "STAGES[STAGES.index(stage) - 1] if stage != 'pretrain' else None"
predecessors = {s: stages[stages.index(s) - 1] if s != "pretrain" else None for s in stages}
assert predecessors["dpo"] == "joint"

environment = {"python": sys.version, "python_executable": sys.executable,
               "torch_installed_version": importlib.metadata.version("torch"),
               "playwright": importlib.metadata.version("playwright"),
               "browser_executable": "/usr/bin/chromium", "device": "CPU/browser; no model construction",
               "cwd": str(ROOT)}
out = {"source_sha256": sha(source), "html_sha256": sha(current_html),
       "http_url": URL, "http_html_equal_current_file": True,
       "actual_changed_claims": [changed_table, changed_recipe],
       "stage_predecessors_from_original_ast": predecessors,
       "ast_locators": {"STAGES": [stage_assignment.lineno, stage_assignment.end_lineno],
                        "stage_predecessor": [predecessor_expr.lineno, predecessor_expr.end_lineno]},
       "screens": [], "scope": "Current desktop/mobile page and two changed labels only. No execution buttons clicked, lesson code/model/training/download/generation not run.",
       "new_model_cpu_runs": 0, "new_recipe_runs": 0, "new_source_fetches": 0}

with sync_playwright() as pw:
    browser = pw.chromium.launch(executable_path="/usr/bin/chromium", headless=True, args=["--no-sandbox"])
    environment["chromium"] = browser.version
    for label, width, height in [("desktop", 1280, 800), ("mobile", 390, 844)]:
        page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        response = page.goto(URL, wait_until="networkidle", timeout=20000)
        assert response and response.status == 200
        page.locator("article details").evaluate("el => el.open = true")
        article = page.locator("article.md-content__inner")
        text = article.inner_text()
        assert changed_table in text and changed_recipe in text
        assert "四站FP32推論檔" not in text and "如果要自己重做四站，" not in text
        codes = article.locator("pre code").all_inner_texts()
        assert len(codes) == len(fences) == 5
        for displayed, (_, original) in zip(codes, fences, strict=True):
            assert displayed.rstrip("\n") == original.decode().rstrip("\n")
        assert article.locator("img").count() == 0
        for target, locator in [
            ("table", article.locator("table")),
            ("recipe", article.locator("p").filter(has_text=changed_recipe).first),
        ]:
            locator.scroll_into_view_if_needed()
            screenshot = A / f"{label}-{target}.png"
            page.screenshot(path=str(screenshot), full_page=False)
            dimensions = page.evaluate("({viewport:innerWidth, body:document.body.scrollWidth, document:document.documentElement.scrollWidth})")
            assert dimensions["document"] <= width, dimensions
            out["screens"].append({"viewport": [width, height], "target": target,
                                   "path": screenshot.name, "sha256": sha(screenshot.read_bytes()),
                                   "main_text_sha256": sha(text.encode()), "code_fences_match_current_raw": True,
                                   "heading": article.locator("h1").inner_text(), "page_errors": list(errors),
                                   "layout_dimensions": dimensions,
                                   "visual_inspection": "Screenshot saved; reviewer must actually view this file separately."})
        page.close()
    browser.close()
(A / "current-inspection-environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2) + "\n")
(A / "current-inspection-results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"environment": environment, "inspection": out}, ensure_ascii=False))
