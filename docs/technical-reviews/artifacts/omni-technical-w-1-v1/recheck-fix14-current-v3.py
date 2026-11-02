"""W.1 same-owner source14 callback. Full site, CLI and per-page rows stay outside Git."""
import contextlib
import hashlib
import importlib.metadata
import io
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
F = ROOT / "docs/course-revision-20261029-omni/reviews/numbered-freeze-v12.json"
C = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix14.json"
R = ROOT / "docs/course-revision-20261029-omni/evidence/notebooks-technical-fix14-cumulative.json"
BUNDLE = ROOT / "outputs/omni-notebooks-technical-fix14-current"
SITE = ROOT / "outputs/omni-review-site-technical-fix14"
BASE = "http://127.0.0.1:8775/"
PRIVATE = Path("/workspace/work/omni-w1-fix14-export-v3")
TIMES = ROOT / "course/reading-times.json"
PREVIEW_TIMES = ROOT / "outputs/omni-reading-times-technical-fix14-preview.json"
TIME_RECEIPT = ROOT / "docs/course-revision-20261029-omni/evidence/preview-estimates-technical-fix14-v1.json"
SCOPE = ROOT / "docs/course-revision-20261029-omni/reviews/history-v2-review-scope-v4.json"
sys.path.insert(0, str(ROOT))
from scripts.export_course import DOCUMENTS, checked_notebook, home_introduction, output_html
from scripts.reading_time import build_inventory, format_range, load_estimates
from output_dom_v1 import outputs as output_dom


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path):
    return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), "sha256": sha(path)}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


class Actions(HTMLParser):
    def __init__(self):
        super().__init__()
        self.depth = 0
        self.actions = None
        self.links = []
        self.ids = set()
        self.text = []

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag == "div":
            self.depth += 1
            if "actions" in attrs.get("class", "").split():
                self.actions = self.depth
        if attrs.get("id"):
            self.ids.add(attrs["id"])
        if tag == "a" and self.actions is not None:
            self.links.append((attrs.get("href"), "download" in attrs))

    def handle_endtag(self, tag):
        if tag == "div":
            if self.actions == self.depth:
                self.actions = None
            self.depth -= 1

    def handle_data(self, text):
        self.text.append(text)


def footer(body, item):
    parser = Actions()
    parser.feed(body)
    assert {x[0] for x in parser.links} == {item["notebook"], "https://colab.research.google.com/"}, item["id"]
    assert (item["notebook"], True) in parser.links
    assert "colab-upload-instructions" in parser.ids
    text = " ".join(parser.text)
    for phrase in ["尚未發布的預覽版", "先按「下載本節 .ipynb」", "在首頁按", "Upload notebook", "File → Upload notebook", "雲端環境未經本課實測"]:
        assert phrase in text, (item["id"], phrase)
    return hashlib.sha256(body.encode()).hexdigest()


def run(name, args, cwd=ROOT, env=None):
    completed = subprocess.run(args, cwd=cwd, env=env, text=True, capture_output=True, timeout=120)
    log = PRIVATE / (name + ".log")
    log.write_text("STDOUT\n" + completed.stdout + "\nSTDERR\n" + completed.stderr)
    return completed, {"command": args, "cwd": str(cwd), "exit_code": completed.returncode, "complete_cli_log": ref(log)}


started = datetime.now(UTC).isoformat()
PRIVATE.mkdir(exist_ok=False)
assert not (A / "fix14-current-recheck-v3.json").exists()
prior_path = D / "technical-current-v3.json"
assert sha(prior_path) == "d9d6fb79a655735922903eb6617cbb452f3a8f75a9b0a62721fa71b6b814ea39"
prior = json.loads(prior_path.read_bytes())
preserved = [ref(D / name) for name in ["technical-original-v1.json", "recheck-observation-v1.json", "technical-current-v1.json", "technical-current-v2.json", "technical-current-v3.json"]]
for artifact in prior["artifacts"]:
    assert sha(ROOT / artifact["path"]) == artifact["sha256"], artifact["path"]
dependencies = []
for source in prior["sources"]:
    if source["kind"] == "repository_code":
        current = sha(ROOT / source["path"])
        assert current == source["sha256"], source["path"]
        dependencies.append({"source_id": source["id"], "path": source["path"], "sha256": current, "same_as_current3": True})
for name in ["scripts/reading_time.py", "scripts/stage_mathjax.py", "course/reading-time-routes.json", "zensical.toml"]:
    assert (ROOT / name).read_bytes() == (Path("/workspace/work/omni-w1-fix13-export-v1") / name).read_bytes(), name
    dependencies.append({**ref(ROOT / name), "same_as_current3_private_export": True})
scope = json.loads(SCOPE.read_bytes())
scope_rows = {x["path"]: x for x in scope["checked_file_scope"]}
assert len(scope_rows) == 210
scope_subset = ["scripts/export_course.py", "scripts/check_site.py", "scripts/build_course.py", "scripts/reading_time.py", ".github/workflows/windows-student-commands.yml", "docs/course-revision-20261029-omni/REVIEW-PROTOCOL.md"]
for name in scope_subset:
    assert sha(ROOT / name) == scope_rows[name]["sha256"], name

freeze, canonical = json.loads(F.read_bytes()), json.loads(C.read_bytes())
assert freeze["canonical_source_freeze"] == ref(C)
w1 = next(x for x in freeze["pages"] if x["page_id"] == "W.1")
w1_raw = (ROOT / w1["snapshot"]).read_bytes()
assert hashlib.sha256(w1_raw).hexdigest() == w1["source_sha256"] == prior["source_sha256"]
old_freeze = json.loads((ROOT / prior["source_freeze"]["path"]).read_bytes())
old_w1 = next(x for x in old_freeze["pages"] if x["page_id"] == "W.1")
assert (ROOT / old_w1["snapshot"]).read_bytes() == w1_raw
assert re.search(rb"^## W\.1 .*?(?=^## |\Z)", (ROOT / "course/first-steps.md").read_bytes(), re.M | re.S)[0] == w1_raw
assert w1["figures_sha256"] == {}
index = json.loads((ROOT / "course/lesson-index.json").read_bytes())
inventory = build_inventory(ROOT, index, DOCUMENTS, home_introduction(index, True))
inventory_pages = {x["page_id"]: x for x in inventory["pages"]}
canonical_pages = {x["page_id"]: x for x in canonical["pages"]}
assert set(inventory_pages) == set(canonical_pages)
for page_id, page in inventory_pages.items():
    assert page["source_sha256"] == canonical_pages[page_id]["source_sha256"], page_id
    assert page["figures_sha256"] == canonical_pages[page_id]["figures_sha256"], page_id
old_c = json.loads((ROOT / prior["canonical_source_freeze"]["path"]).read_bytes())
old_pages = {x["page_id"]: x for x in old_c["pages"]}
delta = []
for page_id, page in canonical_pages.items():
    changed = {k: {"previous": old_pages[page_id].get(k), "current": page.get(k)} for k in ["source_sha256", "figures_sha256", "notebook_sha256"] if old_pages[page_id].get(k) != page.get(k)}
    if changed:
        delta.append({"page_id": page_id, "changes": changed})
assert [x["page_id"] for x in delta] == ["12.2"]
author = json.loads(R.read_bytes())
assert author["source_freeze"] == ref(C)
assert sha(Path(author["script"]["path"])) == author["script"]["sha256"]
assert author["reexecuted_this_fix"] == ["12.2"] and author["reused_matching_cells"] == 330
records = {x["lesson"]: x for x in author["records"]}
assert len(index) == len(records) == 331
assert set(records) == {x["id"] for x in index}
rows = []
wrapper_normalizations = []
code_sections = code_cells = display_cells = reused = new = 0
for item in index:
    lesson = item["id"]
    record = records[lesson]
    notebook_path = ROOT / item["notebook"]
    target = BUNDLE / Path(item["notebook"]).relative_to("notebooks")
    source = Path(record["source"])
    assert Path(record["bundle_path"]) == target
    assert sha(notebook_path) == record["notebook_sha256"] == canonical_pages[lesson]["notebook_sha256"]
    assert sha(target) == record["executed_sha256"] == sha(source)
    assert target.read_bytes() == source.read_bytes()
    expected_method = "new_independent_kernel" if lesson == "12.2" else "reuse_matching_cells"
    assert record["method"] == expected_method
    expected_root = "omni-notebooks-technical-fix14-run" if lesson == "12.2" else "omni-notebooks-technical-fix13-current"
    assert source == ROOT / "outputs" / expected_root / Path(item["notebook"]).relative_to("notebooks")
    new += lesson == "12.2"
    reused += lesson != "12.2"
    notebook = json.loads(notebook_path.read_bytes())
    executed = checked_notebook(notebook, target)
    teaching = [c for c in executed["cells"] if c["cell_type"] == "code" and not c.get("metadata", {}).get("course_setup") and "course_figure" not in c.get("metadata", {})]
    displays = sum(bool(output_html(c.get("outputs", []))) for c in teaching)
    assert not teaching or displays > 0
    code_sections += bool(teaching)
    code_cells += len(teaching)
    display_cells += displays
    assert (SITE / item["notebook"]).read_bytes() == notebook_path.read_bytes()
    body = (SITE / (lesson + ".html")).read_text()
    if teaching:
        assert "實際執行結果 · CPU" in body
        expected_html = [output_html(cell.get("outputs", [])) for cell in teaching]
        expected_dom = output_dom("".join(expected_html))
        actual_dom = output_dom(body)
        assert expected_dom == actual_dom, (lesson, "exact output DOM content/order mismatch")
        if any(value and value not in body for value in expected_html):
            wrapper_normalizations.append({"lesson_id": lesson, "DOM_content_and_order_exactly_equal": True, "expected_DOM_sha256": digest(expected_dom), "actual_DOM_sha256": digest(actual_dom)})
    rows.append({"lesson_id": lesson, "authored_source_sha256": inventory_pages[lesson]["source_sha256"], "notebook_sha256": sha(notebook_path), "raw_sha256": sha(target), "actual_html_sha256": footer(body, item), "method": record["method"]})
assert (new, reused) == (1, 330)
row_path = PRIVATE / "all331-source-raw-preview-rows.json"
row_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")

def http_download(item):
    with urllib.request.urlopen(BASE + item["notebook"], timeout=15) as response:
        raw = response.read()
        assert response.status == 200
    assert raw == (ROOT / item["notebook"]).read_bytes(), item["id"]
    return {"lesson_id": item["id"], "status": 200, "sha256": hashlib.sha256(raw).hexdigest()}

with ThreadPoolExecutor(max_workers=6) as executor:
    http_rows = list(executor.map(http_download, index))
info = json.loads((SITE / "build-info.json").read_bytes())
with urllib.request.urlopen(BASE + "build-info.json", timeout=15) as response:
    assert response.status == 200 and response.read() == (SITE / "build-info.json").read_bytes()
assert info["colab_mode"] == "upload" and info["publication_ready"] is False and info["colab_git_objects_verified"] is False
assert info["reading_time_pages"] == 367 and info["reading_time_complete"] is False
lineage = []
for fix, lesson in [("06", "9.11"), ("09", "9.18"), ("11", "12.17"), ("12", "9.12"), ("13", "9.15"), ("14", "12.2")]:
    receipt_path = ROOT / f"docs/course-revision-20261029-omni/evidence/notebooks-technical-fix{fix}-cumulative.json"
    receipt = json.loads(receipt_path.read_bytes())
    record = next(x for x in receipt["records"] if x["lesson"] == lesson)
    validation_path = ROOT / f"outputs/omni-notebooks-technical-fix{fix}-run/validation-kernel.json"
    validation = json.loads(validation_path.read_bytes())
    assert validation["total"] == validation["passed"] == 1 and validation["failed"] == 0 and not validation["failures"]
    assert record["method"] == "new_independent_kernel"
    assert sha(Path(record["source"])) == record["executed_sha256"]
    assert Path(record["source"]).read_bytes() == Path(records[lesson]["bundle_path"]).read_bytes()
    lineage.append({"lesson_id": lesson, "author_receipt": ref(receipt_path), "kernel_validation": ref(validation_path), "new_kernel_raw": ref(Path(record["source"])), "latest_bundle_same_bytes": True, "result": "1/1 pass,0 fail"})
nb_old = json.loads((ROOT / "outputs/omni-notebooks-technical-fix13-current/12/12.2.ipynb").read_bytes())
nb_new = json.loads((ROOT / "notebooks/12/12.2.ipynb").read_bytes())
changed_cells = [i for i, (a, b) in enumerate(zip(nb_old["cells"], nb_new["cells"], strict=True)) if a["cell_type"] != b["cell_type"] or a["source"] != b["source"]]
assert changed_cells == [6] and nb_new["cells"][6]["cell_type"] == "markdown"
assert "也把兩行print中的頻率名稱改成9500Hz、6500Hz" in "".join(nb_new["cells"][6]["source"])
raw12 = json.loads((BUNDLE / "12/12.2.ipynb").read_bytes())
outputs12 = [{"cell": i, "execution_count": c["execution_count"], "stdout": "".join("".join(o["text"]) if isinstance(o["text"], list) else o["text"] for o in c.get("outputs", []) if o["output_type"] == "stream" and o["name"] == "stdout")} for i, c in enumerate(raw12["cells"]) if c["cell_type"] == "code" and not c.get("metadata", {}).get("course_setup") and "course_figure" not in c.get("metadata", {})]
assert outputs12[-1]["stdout"].splitlines()[-1] == "每一點都只差正負號 True"

# Metadata is checked as data binding, not as reader answers or independent estimates.
time_refs = [ref(TIMES), ref(PREVIEW_TIMES), ref(TIME_RECEIPT)]
base_times = json.loads(TIMES.read_bytes())
preview_times = json.loads(PREVIEW_TIMES.read_bytes())
base_rows = {x["page_id"]: x for x in base_times["pages"]}
preview_rows = {x["page_id"]: x for x in preview_times["pages"]}
assert len(base_rows) == 369 and len(preview_rows) == 367
stale = sorted(page_id for page_id, row in base_rows.items() if row["source_sha256"] != inventory_pages[page_id]["source_sha256"] or row["figures_sha256"] != inventory_pages[page_id]["figures_sha256"])
assert stale == ["12.2", "first-steps"]
assert preview_times["pages"] == [x for x in base_times["pages"] if x["page_id"] not in stale]
assert set(base_rows) - set(preview_rows) == set(stale)
assert set(preview_times) - set(base_times) == {"preview_scope"}
assert not set(base_times) - set(preview_times)
assert preview_times["preview_scope"] == "Only exact current estimates retained. Omitted stale estimates are pending actual reviewer estimates. This preview metadata cannot satisfy a formal complete-publication gate."
for key in base_times:
    if key != "pages":
        assert base_times[key] == preview_times[key]
time_receipt = json.loads(TIME_RECEIPT.read_bytes())
assert time_receipt["original_metadata_sha256"] == sha(TIMES)
assert time_receipt["preview_metadata_sha256"] == sha(PREVIEW_TIMES)
estimates = load_estimates(PREVIEW_TIMES, inventory)
assert len(estimates) == 367
time_guards = []
for page_id in stale:
    sample = PRIVATE / ("stale-" + page_id + "-metadata.json")
    sample.write_text(json.dumps({"schema_version": 1, "estimate_kind": "ai_estimate", "pages": [base_rows[page_id]]}, ensure_ascii=False))
    try:
        load_estimates(sample, inventory)
    except ValueError as error:
        assert "不是目前正文" in str(error)
        time_guards.append({"case": "reject-stale-row", "page_id": page_id, "message": str(error)})
    else:
        raise AssertionError(page_id + " stale metadata accepted")
try:
    load_estimates(PREVIEW_TIMES, inventory, require_complete=True)
except ValueError as error:
    assert "缺少 2 頁閱讀時間" in str(error)
    time_guards.append({"case": "reject-incomplete-formal-metadata", "message": str(error)})
else:
    raise AssertionError("Incomplete preview metadata accepted as complete")
rendered_times = set()
for page_id, page in inventory_pages.items():
    body = (SITE / Path(page["page_path"]).with_suffix(".html")).read_text()
    present = 'class="reading-time"' in body
    assert present == (page_id in estimates), page_id
    if present:
        assert "閱讀約 " + format_range((estimates[page_id]["minutes_min"], estimates[page_id]["minutes_max"])) + " · AI 估計" in body
        rendered_times.add(page_id)
assert len(rendered_times) == 367
sync, sync_receipt = run("current-source-notebook-sync", [sys.executable, str(ROOT / "scripts/build_course.py"), "--check"])
assert sync.returncode == 0
site_check, site_check_receipt = run("actual-site-check", [sys.executable, str(ROOT / "scripts/check_site.py"), "--site", str(SITE)])
assert site_check.returncode == 0
env_check, env_receipt = run("current-environment-check", [sys.executable, str(ROOT / "scripts/check_env.py")])
assert env_check.returncode == 0 and "OK" in env_check.stdout and "未安裝" not in env_check.stdout
smoke_code = '''import contextlib,io,json
from pathlib import Path
n=json.loads(Path(%r).read_bytes())
setup=next(c for c in n["cells"] if c.get("metadata",{}).get("course_setup"))
teaching=[c for c in n["cells"] if c["cell_type"]=="code" and not c.get("metadata",{}).get("course_setup") and "course_figure" not in c.get("metadata",{})]
result=[]
for sentence in ["貓看狗，狗看貓。","鳥看狗，狗看鳥。"]:
 ns={}; output=io.StringIO()
 with contextlib.redirect_stdout(output):
  exec("".join(setup["source"]),ns)
  for c in teaching:exec("".join(c["source"]).replace("貓看狗，狗看貓。",sentence),ns)
 assert output.getvalue().splitlines()[-1]==sentence
 result.append({"input":sentence,"restored":output.getvalue().splitlines()[-1]})
print(json.dumps(result,ensure_ascii=False))
''' % str(ROOT / "notebooks/01/1.1.ipynb")
smoke, smoke_receipt = run("W1-current-cat-bird-bootstrap", [sys.executable, "-c", smoke_code], cwd=ROOT / "notebooks/01")
assert smoke.returncode == 0
smoke_receipt["result"] = json.loads(smoke.stdout)

# Use unmodified current tools and exact preview metadata in a fresh private root.
for folder in ["course", "notebooks"]:
    shutil.copytree(ROOT / folder, PRIVATE / folder)
for name in ["scripts/export_course.py", "scripts/check_site.py", "scripts/build_course.py", "scripts/reading_time.py", "scripts/stage_mathjax.py", "zensical.toml", *DOCUMENTS.values()]:
    destination = PRIVATE / name
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    assert destination.read_bytes() == (ROOT / name).read_bytes()
if (ROOT / "outputs/mathjax-cache").is_dir():
    shutil.copytree(ROOT / "outputs/mathjax-cache", PRIVATE / "outputs/mathjax-cache")
private_times = PRIVATE / "preview-estimates.json"
shutil.copyfile(PREVIEW_TIMES, private_times)
exported, export_receipt = run("isolated-export", [sys.executable, str(PRIVATE / "scripts/export_course.py"), "--output", str(PRIVATE / "outputs/site"), "--executed", str(BUNDLE), "--revision", "W1-fix14-preview-test", "--reading-times", str(private_times)], cwd=PRIVATE)
assert exported.returncode == 0
private_site = PRIVATE / "outputs/site"
isolated_check, isolated_check_receipt = run("isolated-site-check", [sys.executable, str(PRIVATE / "scripts/check_site.py"), "--site", str(private_site)], cwd=PRIVATE)
assert isolated_check.returncode == 0
for item in index:
    assert (private_site / item["notebook"]).read_bytes() == (ROOT / item["notebook"]).read_bytes()
    footer((private_site / (item["id"] + ".html")).read_text(), item)
private_info = json.loads((private_site / "build-info.json").read_bytes())
assert private_info["reading_time_pages"] == 367 and private_info["publication_ready"] is False
head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
git_dir = subprocess.check_output(["git", "rev-parse", "--absolute-git-dir"], cwd=ROOT, text=True).strip()
git_env = {**os.environ, "GIT_DIR": git_dir, "GIT_WORK_TREE": str(PRIVATE)}
gate_code = '''import json,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from scripts.export_course import verify_colab_notebooks
index=json.loads(Path("course/lesson-index.json").read_bytes())
try:
 verify_colab_notebooks(index,sys.argv[1])
except ValueError as error:
 print(json.dumps({"accepted":False,"count":len(index),"message_first_line":str(error).splitlines()[0]},ensure_ascii=False))
else:print(json.dumps({"accepted":True,"count":len(index)}))
'''
gate, gate_receipt = run("current331-formal-Git-comparison", [sys.executable, "-c", gate_code, head], cwd=PRIVATE, env=git_env)
assert gate.returncode == 0
gate_value = json.loads(gate.stdout)
assert gate_value["accepted"] is False and gate_value["count"] == 331
gate_receipt.update(result=gate_value, revision=head, prior_exact_implementation_gate_cases_reused=ref(A / "fix13-current-recheck-v1.json"))

# Fixed originals and UI bytes are reused with explicit provenance, not new live UI claims.
originals = json.loads((A / "official-source-inspection.json").read_bytes())
for item in originals:
    assert sha(Path(item["raw_path"])) == item["raw_sha256"]
ui = json.loads((A / "fix13-current-recheck-v1.json").read_bytes())["official_UI_originals_actually_viewed"]
for item in ui:
    assert sha(Path(item["path"])) == item["sha256"]
for item in time_refs:
    assert sha(ROOT / item["path"]) == item["sha256"]
for item in dependencies:
    assert sha(ROOT / item["path"]) == item["sha256"]
assert sha(prior_path) == "d9d6fb79a655735922903eb6617cbb452f3a8f75a9b0a62721fa71b6b814ea39"
environment = {"python": platform.python_version(), "platform": platform.platform(), "device": "cpu", **{p: importlib.metadata.version(p) for p in ["torch", "jupyterlab", "ipykernel", "nbclient", "zensical"]}}
result = {"started_at": started, "created_at": datetime.now(UTC).isoformat(), "reviewer_task": prior["reviewer_task"], "lesson_id": "W.1", "verdict": "pass", "source_freeze": ref(F), "canonical_source_freeze": ref(C), "W1_snapshot": {"path": w1["snapshot"], "sha256": w1["source_sha256"]}, "W1_exact_bytes_and_empty_figure_map_same_as_current3": True, "canonical_delta_from13": delta, "preserved_reports": preserved, "preserved_prior_artifacts": len(prior["artifacts"]), "current_dependencies": dependencies, "scope_inventory": {**ref(SCOPE), "listed_paths": 210, "personally_checked_hash_subset": scope_subset, "limit": "Inventory is not a tool audit or approval. W.1 relevant export/build/check/reading-time code and Windows workflow context only; no210-file subject review."}, "author_bundle": ref(R), "raw_bundle_root": str(BUNDLE), "counts": {"canonical_source_inventory": len(inventory_pages), "notebooks_current_source_raw_manifest": 331, "new_kernel_this_fix": new, "source_matched_reuse": reused, "teaching_code_sections": code_sections, "teaching_code_cells": code_cells, "displayed_teaching_cells": display_cells, "actual_preview_pages": len(list(SITE.glob('*.html'))), "actual_preview_same_byte_downloads": 331, "HTTP200_same_byte_downloads": len(http_rows), "actual_rendered_current_estimates": len(rendered_times), "isolated_same_byte_downloads": 331}, "all331_per_page_rows": ref(row_path), "all331_rows_fingerprint": digest(rows), "HTTP331_fingerprint": digest(http_rows), "kernel_lineage": lineage, "new_12_2": {"changed_cells": changed_cells, "change_kind": "Markdown exercise print-label clarification only; all code sources unchanged", "current_raw": ref(BUNDLE / "12/12.2.ipynb"), "actual_teaching_stdout": outputs12}, "actual_preview": {"url": BASE, "root": str(SITE), "build_info": info, "build_info_ref": ref(SITE / "build-info.json"), "HTTP_build_info_exact_bytes": True, "all331_manual_upload_footer_checks": True, "all_teaching_output_DOM_content_and_order_match_raw": True, "wrapper_formatting_only_normalizations": wrapper_normalizations, "check_site": site_check_receipt}, "preview_time_binding": {"metadata_refs": time_refs, "formal_original_rows": len(base_rows), "unchanged_retained_rows": len(preview_rows), "omitted_stale": stale, "no_minutes_or_reviewer_values_synthesized": True, "only_added_top_level_field": {"preview_scope": preview_times["preview_scope"]}, "rendered_rows_exactly_match367_current_rows": True, "guards_actually_called": time_guards, "limit": "Source/figure binding and preview display only, not estimating reading time or authenticating reader completion. first-steps and12.2 have no current estimate here."}, "current_source_notebook_sync": sync_receipt, "current_environment_smoke": env_receipt, "W1_bootstrap_cat_bird_smoke": smoke_receipt, "isolated_export": {"root": str(PRIVATE), "export": export_receipt, "check_site": isolated_check_receipt, "build_info": private_info, "current_unmodified_tools": True, "reading_time_metadata": ref(private_times)}, "current331_formal_Git_gate": gate_receipt, "stable_evidence_reuse": {"official_originals_rehashed": len(originals), "official_UI_originals_rehashed": 2, "UI_actual_visual_read": "Previous current-v3 callback; no new image viewing, browser capture, upload/login/Connect/runtime/cell in this callback.", "prior_nine_gate_cases": "Same exact current exporter/checker bytes as current-v3; retain original positive/negative tests. Only the changed current331 comparison repeated here."}, "new_test_failure_and_diagnosis": {"initial_failure": ref(A / "fix14-initial-output-comparison-v1.json"), "diagnosis": ref(A / "fix14-output-rendering-diagnosis-v1.json"), "DOM_comparison_source": ref(A / "output_dom_v1.py"), "DOM_negative_controls": ref(A / "fix14-output-DOM-negative-controls-v1.json"), "second_failure": ref(A / "fix14-metadata-shape-observation-v1.json"), "second_diagnosis": "The only extra top-level key is preview_scope, explicitly disclosing omission and nonformal status. All retained row values and every existing top-level value are exact. Existing load_estimates ignores this descriptive key; the source/figure/completeness guards are tested unchanged.", "scope": "No original test/checker altered. Old strict wrapper-string driver/failure retained; new driver requires all output labels, exact pre text, exact image attributes and order to match. Only outside-block formatting whitespace is ignored."}, "existing_issue_continued_closure": {"id": "w1-current-dependency-drift", "status": "resolved", "original_issue": False, "details": "Original report had zero issues. All current1 discovery fields and current2/3 closure fields retained; current14 source/raw/downloads newly checked, not relabeled13 evidence."}, "environment": environment, "limitations": ["Only W.1 and necessary global dependencies; no fresh review of other numbered lessons or all210 tools.", "All331 current sources/Notebook/raw/author pointers and HTTP downloads checked; this reviewer did not rerun331 kernels.", "Root genuinely reran12.2 only for this checkpoint; current raw chain of earlier five reruns is also retained and rechecked.", "Preview367 is a filtered exact-current copy, not a new estimate or formal complete metadata. first-steps and12.2 fullreader still pending; no whole-page reader or stage seal claim.", "No new Colab UI viewing/browser session/cloud execution; unchanged original screenshots only rehashed and reused from actual current-v3 visual read.", "Formal331 comparison against existing HEAD fails as expected. No commit, publication, deployment, GPU, long training or paid operation.", "No historical source792f exemption applied to current exporter662355 or check_site084027. All complete CLI, full rows and site remain outside Git."]}
target = A / "fix14-current-recheck-v3.json"
with target.open("x", encoding="utf-8") as file:
    json.dump(result, file, ensure_ascii=False, indent=2)
    file.write("\n")
print(json.dumps({"path": str(target.relative_to(ROOT)), "sha256": sha(target), "created_at": result["created_at"], "verdict": result["verdict"], "counts": result["counts"], "time_guards": time_guards}, ensure_ascii=False, indent=2))
