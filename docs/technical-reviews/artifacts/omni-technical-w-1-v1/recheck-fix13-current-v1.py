"""W.1 same-owner callback: current raw bundle, actual preview, new Colab guards.

Complete CLI logs, the isolated site and UI PNG originals remain outside Git.
"""
import copy
import hashlib
import html
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
A = Path(__file__).resolve().parent
D = ROOT / "docs/course-revision-20261029-omni/reviews/technical/technical_w_1"
F = ROOT / "docs/course-revision-20261029-omni/reviews/numbered-freeze-v11.json"
C = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix13.json"
RECEIPT = ROOT / "docs/course-revision-20261029-omni/evidence/notebooks-technical-fix13-cumulative.json"
BUNDLE = ROOT / "outputs/omni-notebooks-technical-fix13-current"
SITE = ROOT / "outputs/omni-review-site-technical-fix13"
BASE = "http://127.0.0.1:8774/"
ISOLATED = Path("/workspace/work/omni-w1-fix13-export-v1")
sys.path.insert(0, str(ROOT))
from scripts.export_course import DOCUMENTS, checked_notebook, output_html, section_footer


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ref(p):
    return {"path": str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p), "sha256": sha(p)}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class Actions(HTMLParser):
    def __init__(self):
        super().__init__()
        self.depth = 0
        self.action_depth = None
        self.links = []
        self.ids = set()
        self.text = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "div":
            self.depth += 1
            if "actions" in a.get("class", "").split():
                self.action_depth = self.depth
        if a.get("id"):
            self.ids.add(a["id"])
        if tag == "a" and self.action_depth is not None:
            self.links.append({"href": a.get("href"), "download_attribute": "download" in a})

    def handle_endtag(self, tag):
        if tag == "div":
            if self.depth == self.action_depth:
                self.action_depth = None
            self.depth -= 1

    def handle_data(self, value):
        self.text.append(value)


def preview_check(body, item):
    p = Actions()
    p.feed(body)
    assert {x["href"] for x in p.links} == {item["notebook"], "https://colab.research.google.com/"}, item["id"]
    assert any(x["href"] == item["notebook"] and x["download_attribute"] for x in p.links)
    assert "colab-upload-instructions" in p.ids
    text = " ".join(p.text)
    assert all(x in text for x in ["尚未發布的預覽版", "先按「下載本節 .ipynb」", "在首頁按", "Upload notebook", "File → Upload notebook", "雲端環境未經本課實測"]), item["id"]
    return hashlib.sha256(body.encode()).hexdigest()


started = datetime.now(UTC).isoformat()
assert not ISOLATED.exists()
ISOLATED.mkdir()
prior_path = D / "technical-current-v2.json"
assert sha(prior_path) == "44e72896b96d11db3a06beb15bc64567b6b08a6e9ac9be01f7336b1a83839ca3"
prior = json.loads(prior_path.read_bytes())
preserved = [ref(D / n) for n in ["technical-original-v1.json", "technical-current-v1.json", "technical-current-v2.json", "recheck-observation-v1.json"]]
for a in prior["artifacts"]:
    assert sha(ROOT / a["path"]) == a["sha256"], a["path"]
dependencies = []
for s in prior["sources"]:
    if s["kind"] == "repository_code":
        actual = sha(ROOT / s["path"])
        if s["id"] != "r_export":
            assert actual == s["sha256"], s["path"]
        dependencies.append({"source_id": s["id"], "path": s["path"], "previous_sha256": s["sha256"], "current_sha256": actual, "changed": actual != s["sha256"]})
dependencies.append({"source_id": "r_site_check", "path": "scripts/check_site.py", "current_sha256": sha(ROOT / "scripts/check_site.py"), "newly_inspected": True})
freeze, canonical = json.loads(F.read_bytes()), json.loads(C.read_bytes())
assert freeze["canonical_source_freeze"] == ref(C)
w1 = next(p for p in freeze["pages"] if p["page_id"] == "W.1")
raw = (ROOT / w1["snapshot"]).read_bytes()
assert hashlib.sha256(raw).hexdigest() == w1["source_sha256"] == prior["source_sha256"]
old_freeze = json.loads((ROOT / prior["source_freeze"]["path"]).read_bytes())
old_w1 = next(p for p in old_freeze["pages"] if p["page_id"] == "W.1")
assert raw == (ROOT / old_w1["snapshot"]).read_bytes()
assert w1["figures_sha256"] == {}
assert re.search(rb"^## W\.1 .*?(?=^## |\Z)", (ROOT / "course/first-steps.md").read_bytes(), re.M | re.S)[0] == raw
index = json.loads((ROOT / "course/lesson-index.json").read_bytes())
pages = {p["page_id"]: p for p in canonical["pages"] if p.get("notebook")}
author = json.loads(RECEIPT.read_bytes())
assert author["source_freeze"] == ref(C)
assert sha(Path(author["script"]["path"])) == author["script"]["sha256"]
rows = {r["lesson"]: r for r in author["records"]}
assert len(index) == len(rows) == len(pages) == 331
assert set(rows) == set(pages) == {x["id"] for x in index}
verified = []
code_sections = teaching_count = display_count = 0
for item in index:
    row = rows[item["id"]]
    nb_path = ROOT / item["notebook"]
    target = BUNDLE / Path(item["notebook"]).relative_to("notebooks")
    assert Path(row["bundle_path"]) == target
    assert sha(nb_path) == row["notebook_sha256"] == pages[item["id"]]["notebook_sha256"]
    assert sha(target) == row["executed_sha256"]
    assert target.read_bytes() == Path(row["source"]).read_bytes()
    actual = json.loads(nb_path.read_bytes())
    executed = checked_notebook(actual, target)
    code = [c for c in executed["cells"] if c["cell_type"] == "code" and not c.get("metadata", {}).get("course_setup") and "course_figure" not in c.get("metadata", {})]
    displays = sum(bool(output_html(c.get("outputs", []))) for c in code)
    assert not code or displays > 0
    code_sections += bool(code)
    teaching_count += len(code)
    display_count += displays
    assert (SITE / item["notebook"]).read_bytes() == nb_path.read_bytes()
    body = (SITE / (item["id"] + ".html")).read_text()
    body_sha = preview_check(body, item)
    if code:
        assert "實際執行結果 · CPU" in body
    verified.append({"lesson_id": item["id"], "notebook_sha256": sha(nb_path), "raw_sha256": sha(target), "preview_html_sha256": body_sha})

def http_check(item):
    with urllib.request.urlopen(BASE + item["notebook"], timeout=15) as response:
        data = response.read()
        assert response.status == 200
    assert data == (ROOT / item["notebook"]).read_bytes(), item["id"]
    return {"lesson_id": item["id"], "sha256": hashlib.sha256(data).hexdigest(), "status": 200}

with ThreadPoolExecutor(max_workers=6) as pool:
    http_downloads = list(pool.map(http_check, index))
info = json.loads((SITE / "build-info.json").read_bytes())
assert info["colab_mode"] == "upload" and info["publication_ready"] is False and info["colab_git_objects_verified"] is False
lineages = []
for fix, lesson in [("06", "9.11"), ("09", "9.18"), ("11", "12.17"), ("12", "9.12"), ("13", "9.15")]:
    receipt_path = ROOT / f"docs/course-revision-20261029-omni/evidence/notebooks-technical-fix{fix}-cumulative.json"
    receipt = json.loads(receipt_path.read_bytes())
    row = next(x for x in receipt["records"] if x["lesson"] == lesson)
    validation_path = ROOT / f"outputs/omni-notebooks-technical-fix{fix}-run/validation-kernel.json"
    validation = json.loads(validation_path.read_bytes())
    assert validation["total"] == validation["passed"] == 1 and validation["failed"] == 0 and not validation["failures"]
    assert row["method"] == "new_independent_kernel"
    assert sha(Path(row["source"])) == row["executed_sha256"]
    assert Path(row["source"]).read_bytes() == Path(rows[lesson]["bundle_path"]).read_bytes()
    lineages.append({"lesson_id": lesson, "author_receipt": ref(receipt_path), "kernel_validation": ref(validation_path), "new_kernel_raw": ref(Path(row["source"])), "latest_bundle_same_bytes": True, "result": "1/1 passed,0 failed"})

def run_logged(name, args, cwd=ROOT, env=None):
    done = subprocess.run(args, cwd=cwd, env=env, text=True, capture_output=True, timeout=120)
    path = ISOLATED / (name + ".log")
    path.write_text("STDOUT\n" + done.stdout + "\nSTDERR\n" + done.stderr)
    return done, {"command": args, "cwd": str(cwd), "exit_code": done.returncode, "complete_cli_log": ref(path)}

root_check, root_check_receipt = run_logged("actual-root-site-check", [sys.executable, str(ROOT / "scripts/check_site.py"), "--site", str(SITE)])
assert root_check.returncode == 0
for folder in ["course", "notebooks"]:
    shutil.copytree(ROOT / folder, ISOLATED / folder)
for relative in ["scripts/export_course.py", "scripts/check_site.py", "scripts/build_course.py", "scripts/reading_time.py", "scripts/stage_mathjax.py", "zensical.toml", *DOCUMENTS.values()]:
    target = ISOLATED / relative
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    assert target.read_bytes() == (ROOT / relative).read_bytes()
if (ROOT / "outputs/mathjax-cache").is_dir():
    shutil.copytree(ROOT / "outputs/mathjax-cache", ISOLATED / "outputs/mathjax-cache")
export_command = [sys.executable, str(ISOLATED / "scripts/export_course.py"), "--output", str(ISOLATED / "outputs/site"), "--executed", str(BUNDLE), "--revision", "W1-fix13-preview-test", "--reading-times", str(ISOLATED / "intentionally-absent-reading-times.json")]
exported, export_receipt = run_logged("isolated-export", export_command, cwd=ISOLATED)
assert exported.returncode == 0
isolated_site = ISOLATED / "outputs/site"
isolation_check, isolation_receipt = run_logged("isolated-site-check", [sys.executable, str(ISOLATED / "scripts/check_site.py"), "--site", str(isolated_site)], cwd=ISOLATED)
assert isolation_check.returncode == 0
for item in index:
    assert (isolated_site / item["notebook"]).read_bytes() == (ROOT / item["notebook"]).read_bytes()
    preview_check((isolated_site / (item["id"] + ".html")).read_text(), item)

# Existing real Git objects only. No commit, checkout, object creation or cloud call.
head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT, text=True).strip()
git_dir = subprocess.check_output(["git", "rev-parse", "--absolute-git-dir"], cwd=ROOT, text=True).strip()
git_env = {**os.environ, "GIT_DIR": git_dir, "GIT_WORK_TREE": str(ISOLATED)}
gate_code = '''import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from scripts.export_course import verify_colab_notebooks
items=json.loads(Path("course/lesson-index.json").read_bytes())
mode,revision=sys.argv[1:]
if mode=="one":items=[next(x for x in items if x["id"]=="1.1")]
if mode=="missing":items=[{"notebook":"notebooks/not-existing-for-W1-check.ipynb"}]
if mode=="tree":items=[{"notebook":"notebooks"}]
try:
 verify_colab_notebooks(items,revision)
 result={"accepted":True,"index_count":len(items)}
except Exception as e:
 result={"accepted":False,"index_count":len(items),"error_type":type(e).__name__,"message":str(e)}
print(json.dumps(result,ensure_ascii=False))
'''
gate_results = []
for name, mode, revision, expected in [("reject-short-ref", "one", "main", False), ("reject-missing-commit", "one", "0" * 40, False), ("reject-tree-as-commit", "one", tree, False), ("accept-existing-one-matching-blob", "one", head, True), ("reject-missing-blob", "missing", head, False), ("reject-tree-as-blob", "tree", head, False), ("current331-publication-comparison", "all", head, False)]:
    done, receipt = run_logged(name, [sys.executable, "-c", gate_code, mode, revision], cwd=ISOLATED, env=git_env)
    assert done.returncode == 0
    value = json.loads(done.stdout)
    assert value["accepted"] is expected, name
    gate_results.append({"case": name, "revision": revision, "accepted": value["accepted"], "index_count": value["index_count"], "message_first_line": value.get("message", "").splitlines()[0] if value.get("message") else "", "log": receipt["complete_cli_log"]})
path = ISOLATED / "notebooks/01/1.1.ipynb"
old = path.read_bytes()
try:
    path.write_bytes(old + b"\n")
    done, receipt = run_logged("reject-byte-only-Notebook-drift", [sys.executable, "-c", gate_code, "one", head], cwd=ISOLATED, env=git_env)
    value = json.loads(done.stdout)
    assert done.returncode == 0 and value["accepted"] is False
    gate_results.append({"case": "reject-byte-only-Notebook-drift", "accepted": False, "JSON_semantics_unchanged": True, "log": receipt["complete_cli_log"]})
finally:
    path.write_bytes(old)
assert path.read_bytes() == (ROOT / "notebooks/01/1.1.ipynb").read_bytes()
build_info_path = isolated_site / "build-info.json"
saved_info = build_info_path.read_bytes()
try:
    bad_info = json.loads(saved_info)
    bad_info["publication_ready"] = True
    build_info_path.write_text(json.dumps(bad_info))
    done, receipt = run_logged("reject-upload-mode-as-publication", [sys.executable, str(ISOLATED / "scripts/check_site.py"), "--site", str(isolated_site)], cwd=ISOLATED)
    assert done.returncode != 0 and "正式網站尚未核對Colab同版本Notebook" in done.stderr
    gate_results.append({"case": "reject-upload-mode-as-publication", "accepted": False, "log": receipt["complete_cli_log"]})
finally:
    build_info_path.write_bytes(saved_info)

nb11 = json.loads((ROOT / "notebooks/01/1.1.ipynb").read_bytes())
direct_footer = section_footer(index[0], nb11, head, True, direct_colab=True)
assert f"/blob/{head}/notebooks/01/1.1.ipynb" in direct_footer and "在 Colab 開啟" in direct_footer
assert "尚未發布的預覽版" not in direct_footer
assert not json.loads(saved_info)["publication_ready"]
ui_shots = [{"url": "https://colab.research.google.com/", "path": "/workspace/work/reader-ui-colab-preview-fix01/colab-desktop-initial.png", "sha256": "d68b97191fd4dff44d2234b97b19085252390ce9379e5170944899164b38c3f5", "observed_label": "Upload notebook"}, {"url": "https://colab.research.google.com/notebooks/intro.ipynb", "path": "/workspace/work/reader-ui-colab-preview-fix01/colab-desktop-file-menu.png", "sha256": "5f9830d0127322af0cfa68a16b8c42f5bed18ec0ac1764c6839a1eeb9eefbf50", "observed_label": "File -> Upload notebook"}]
for s in ui_shots:
    assert sha(Path(s["path"])) == s["sha256"]
    s.update(personally_viewed_this_callback_with_view_image=True, new_browser_capture=False)
git_doc_receipt = Path("/workspace/work/reference/omni-technical-w-1-v3/git-cat-file-receipt.json")
git_doc = json.loads(git_doc_receipt.read_bytes())
assert sha(Path(git_doc["path"])) == git_doc["sha256"]
environment = {"python": platform.python_version(), "platform": platform.platform(), **{p: importlib.metadata.version(p) for p in ["torch", "jupyterlab", "ipykernel", "nbclient", "zensical"]}, "device": "cpu", "git": subprocess.check_output(["git", "--version"], cwd=ROOT, text=True).strip()}
result = {"started_at": started, "created_at": datetime.now(UTC).isoformat(), "reviewer_task": prior["reviewer_task"], "lesson_id": "W.1", "verdict": "pass", "preserved_reports": preserved, "source_freeze": ref(F), "canonical_source_freeze": ref(C), "w1_snapshot": {"path": w1["snapshot"], "sha256": w1["source_sha256"]}, "W1_exact_bytes_same_as_previous": True, "figure_sha256": {}, "current_dependencies": dependencies, "author_bundle": ref(RECEIPT), "raw_bundle_root": str(BUNDLE), "counts": {"canonical_notebooks": 331, "matched_raw": 331, "teaching_code_sections": code_sections, "teaching_code_cells": teaching_count, "displayed_cells": display_count, "actual_preview_downloads_identical": 331, "HTTP200_matching_downloads": len(http_downloads), "isolated_downloads_identical": 331, "actual_preview_pages": len(list(SITE.glob('*.html')))}, "all331_source_raw_preview_rows_fingerprint": fingerprint(verified), "HTTP331_download_fingerprint": fingerprint(http_downloads), "kernel_lineage": lineages, "actual_preview": {"root": str(SITE), "url": BASE, "build_info": info, "build_info_ref": ref(SITE / "build-info.json"), "check_site": root_check_receipt, "all331_manual_flow_footers_checked": True}, "isolated_export": {"root": str(ISOLATED), "export": export_receipt, "check_site": isolation_receipt, "build_info": json.loads(saved_info)}, "new_formal_colab_gate_cases": gate_results, "formal_gate_scope": "A real matching1.1 blob passed; byte-only drift, missing/incorrect object types and the current331/HEAD comparison were rejected. No commit was created; no full formal publication or remote availability is claimed.", "official_UI_originals_actually_viewed": ui_shots, "official_UI_limit": "Only preserved official UI screenshots/menu labels personally viewed; no new browser session, upload, sign-in, Connect, runtime or cell execution.", "git_batch_original_documentation": git_doc, "environment": environment, "existing_issue_continuation": {"id": "w1-current-dependency-drift", "status": "resolved", "note": "Retain every previous issue field. The current source/output/download evidence is the actual fix13 preview plus current unmodified exporter/checker, not the historical792f implementation."}, "limitations": ["Only W.1 and necessary global software dependencies; not a fresh subject-matter review of other W or numbered lessons.", "331 source/code/output/author records, file downloads and HTTP bytes were checked; not331 kernels freshly executed by this reviewer.", "Preview upload workflow is explicitly different from the future direct GitHub Colab workflow. W.1 direct-button wording is interpreted in publication mode; the actual preview footer supplies the manual exception at its use point.", "No cloud upload/login/Connect/cell operation, no course-figure visual review, no deployment, no fullreader or overall seal.", "New source662355... is current; historical792f source only remains in immutable prior evidence. Complete CLI and site stay outside Git."]}
target = A / "fix13-current-recheck-v1.json"
with target.open("x", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(json.dumps({"receipt": str(target.relative_to(ROOT)), "sha256": sha(target), "created_at": result["created_at"], "verdict": result["verdict"], "counts": result["counts"], "formal_gate_cases": [{"case": x["case"], "accepted": x["accepted"]} for x in gate_results]}, ensure_ascii=False, indent=2))
