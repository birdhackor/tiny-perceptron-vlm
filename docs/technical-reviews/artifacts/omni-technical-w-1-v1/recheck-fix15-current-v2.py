"""Bounded source15 W.1 dependency callback; all331 code/output/downloads checked.

No export, kernel execution, cloud operation or shared-output mutation. Full
per-page rows and CLI stay outside Git. Existing output parser is reused verbatim.
"""
import hashlib
import importlib.metadata
import json
import platform
import re
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
C = ROOT / "docs/course-revision-20261029-omni/reviews/source-freeze-technical-fix15.json"
F = ROOT / "docs/course-revision-20261029-omni/reviews/numbered-freeze-v13.json"
R = ROOT / "docs/course-revision-20261029-omni/evidence/notebooks-technical-fix15-cumulative.json"
SITE_RECEIPT = ROOT / "docs/course-revision-20261029-omni/evidence/preview-site-technical-fix15-v1.json"
TIME_RECEIPT = ROOT / "docs/course-revision-20261029-omni/evidence/preview-estimates-technical-fix15-v1.json"
TIMES = ROOT / "course/reading-times.json"
PREVIEW_TIMES = ROOT / "outputs/omni-reading-times-technical-fix15-preview.json"
BUNDLE = ROOT / "outputs/omni-notebooks-technical-fix15-current"
SITE = ROOT / "outputs/omni-review-site-technical-fix15"
BASE = "http://127.0.0.1:8776/"
PRIVATE = Path("/workspace/work/omni-w1-fix15-recheck-v2")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path):
    return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), "sha256": sha(path)}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


assert sha(A / "output_dom_v1.py") == "55e67c3e848cb5f7aad28da80915cfb16f87ac5bec1271eb64a46691323c4597"
sys.path.insert(0, str(ROOT))
from scripts.export_course import DOCUMENTS, checked_notebook, home_introduction, output_html
from scripts.reading_time import build_inventory, format_range, load_estimates
from output_dom_v1 import outputs as output_dom


class LessonHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.depth = 0
        self.action_depth = None
        self.python_depth = None
        self.code_parts = None
        self.python_blocks = []
        self.links = []
        self.ids = set()
        self.text = []

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag == "div":
            self.depth += 1
            classes = attrs.get("class", "").split()
            if "actions" in classes:
                self.action_depth = self.depth
            if "language-python" in classes:
                self.python_depth = self.depth
        if attrs.get("id"):
            self.ids.add(attrs["id"])
        if tag == "a" and self.action_depth is not None:
            self.links.append((attrs.get("href"), "download" in attrs))
        if tag == "code" and self.python_depth is not None:
            assert self.code_parts is None
            self.code_parts = []

    def handle_endtag(self, tag):
        if tag == "code" and self.code_parts is not None:
            self.python_blocks.append("".join(self.code_parts).rstrip("\n"))
            self.code_parts = None
        if tag == "div":
            if self.depth == self.action_depth:
                self.action_depth = None
            if self.depth == self.python_depth:
                self.python_depth = None
            self.depth -= 1

    def handle_data(self, value):
        self.text.append(value)
        if self.code_parts is not None:
            self.code_parts.append(value)


def run(name, args):
    done = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, timeout=120)
    path = PRIVATE / (name + ".log")
    path.write_text("STDOUT\n" + done.stdout + "\nSTDERR\n" + done.stderr)
    return done, {"command": args, "cwd": str(ROOT), "exit_code": done.returncode, "complete_cli_log": ref(path)}


started = datetime.now(UTC).isoformat()
PRIVATE.mkdir(exist_ok=False)
prior_path = D / "technical-current-v4-schema-corrected-v1.json"
assert sha(prior_path) == "e80a87a8dd2f53bdc6eb479b157b63346e9649e55a52c7103f9b52617e169a82"
prior = json.loads(prior_path.read_bytes())
preserved_reports = [ref(D / name) for name in ["technical-original-v1.json", "recheck-observation-v1.json", "technical-current-v1.json", "technical-current-v2.json", "technical-current-v3.json", "technical-current-v4.json", "technical-current-v4-schema-corrected-v1.json"]]
for artifact in prior["artifacts"]:
    assert sha(ROOT / artifact["path"]) == artifact["sha256"], artifact["path"]
dependencies = []
for source in prior["sources"]:
    if source["kind"] == "repository_code":
        assert sha(ROOT / source["path"]) == source["sha256"], source["path"]
        dependencies.append({"source_id": source["id"], **ref(ROOT / source["path"]), "same_as_previous": True})
auth = prior["preserved_original_authorized_implementation_snapshot"]["snapshot"]
assert sha(ROOT / auth["path"]) == auth["sha256"] == "792f6b269d0c1afd34619d2ea10e1db52a6c19a67ce3365539a612f6ef625874"
scope_path = ROOT / "docs/course-revision-20261029-omni/reviews/history-v2-review-scope-v4.json"
scope = json.loads(scope_path.read_bytes())
assert len(scope["checked_file_scope"]) == 210
for row in scope["checked_file_scope"]:
    assert sha(ROOT / row["path"]) == row["sha256"], row["path"]
canonical, numbered = json.loads(C.read_bytes()), json.loads(F.read_bytes())
assert numbered["canonical_source_freeze"] == ref(C)
w1 = next(p for p in numbered["pages"] if p["page_id"] == "W.1")
w1_raw = (ROOT / w1["snapshot"]).read_bytes()
old_numbered = json.loads((ROOT / prior["source_freeze"]["path"]).read_bytes())
old_w1 = next(p for p in old_numbered["pages"] if p["page_id"] == "W.1")
assert w1_raw == (ROOT / old_w1["snapshot"]).read_bytes()
assert hashlib.sha256(w1_raw).hexdigest() == w1["source_sha256"] == prior["source_sha256"]
assert w1["figures_sha256"] == {}
assert re.search(rb"^## W\.1 .*?(?=^## |\Z)", (ROOT / "course/first-steps.md").read_bytes(), re.M | re.S)[0] == w1_raw
assert len(numbered["pages"]) == len(old_numbered["pages"])
old_sections = {p["page_id"]: p for p in old_numbered["pages"]}
for page in numbered["pages"]:
    assert page["source_sha256"] == old_sections[page["page_id"]]["source_sha256"]
    assert page["figures_sha256"] == old_sections[page["page_id"]]["figures_sha256"]
index = json.loads((ROOT / "course/lesson-index.json").read_bytes())
inventory = build_inventory(ROOT, index, DOCUMENTS, home_introduction(index, True))
inventory_pages = {p["page_id"]: p for p in inventory["pages"]}
pages = {p["page_id"]: p for p in canonical["pages"]}
assert set(inventory_pages) == set(pages)
for page_id, page in inventory_pages.items():
    assert page["source_sha256"] == pages[page_id]["source_sha256"], page_id
    assert page["figures_sha256"] == pages[page_id]["figures_sha256"], page_id
old_c = json.loads((ROOT / prior["canonical_source_freeze"]["path"]).read_bytes())
old_pages = {p["page_id"]: p for p in old_c["pages"]}
delta = []
for page_id, page in pages.items():
    changes = {k: {"previous": old_pages[page_id].get(k), "current": page.get(k)} for k in ["source_sha256", "figures_sha256", "notebook_sha256"] if old_pages[page_id].get(k) != page.get(k)}
    if changes:
        delta.append({"page_id": page_id, "changes": changes})
assert {p["page_id"] for p in delta} == {"5.7", "chapter-10", "chapter-14"}
author = json.loads(R.read_bytes())
assert author["source_freeze"] == ref(C)
assert sha(Path(author["script"]["path"])) == author["script"]["sha256"]
assert author["reexecuted_this_fix"] == ["5.7"] and author["reused_matching_cells"] == 330
records = {r["lesson"]: r for r in author["records"]}
assert len(index) == len(records) == 331
assert set(records) == {p["id"] for p in index}
rows = []
sections = cells = displayed = wrapper_only = 0
for item in index:
    lesson = item["id"]
    notebook_path = ROOT / item["notebook"]
    relative = Path(item["notebook"]).relative_to("notebooks")
    target = BUNDLE / relative
    record = records[lesson]
    expected_source = ROOT / "outputs" / ("omni-notebooks-technical-fix15-run" if lesson == "5.7" else "omni-notebooks-technical-fix14-current") / relative
    assert Path(record["source"]) == expected_source and Path(record["bundle_path"]) == target
    assert record["method"] == ("new_independent_kernel" if lesson == "5.7" else "reuse_matching_cells")
    assert sha(notebook_path) == record["notebook_sha256"] == pages[lesson]["notebook_sha256"]
    assert sha(target) == record["executed_sha256"] == sha(expected_source)
    assert target.read_bytes() == expected_source.read_bytes()
    notebook = json.loads(notebook_path.read_bytes())
    executed = checked_notebook(notebook, target)
    teaching = [c for c in executed["cells"] if c["cell_type"] == "code" and not c.get("metadata", {}).get("course_setup") and "course_figure" not in c.get("metadata", {})]
    expected_output_html = [output_html(c.get("outputs", [])) for c in teaching]
    display_count = sum(bool(s) for s in expected_output_html)
    assert not teaching or display_count > 0
    body_path = SITE / (lesson + ".html")
    body = body_path.read_text()
    parser = LessonHTML()
    parser.feed(body)
    assert parser.python_blocks == ["".join(c["source"]).rstrip("\n") for c in teaching], (lesson, "visible code mismatch")
    assert output_dom(body) == output_dom("".join(expected_output_html)), (lesson, "output content/order mismatch")
    wrapper_only += any(s and s not in body for s in expected_output_html)
    assert {p[0] for p in parser.links} == {item["notebook"], "https://colab.research.google.com/"}
    assert (item["notebook"], True) in parser.links
    assert "colab-upload-instructions" in parser.ids
    text = " ".join(parser.text)
    assert all(s in text for s in ["尚未發布的預覽版", "先按「下載本節 .ipynb」", "Upload notebook", "File → Upload notebook", "雲端環境未經本課實測"])
    assert (SITE / item["notebook"]).read_bytes() == notebook_path.read_bytes()
    sections += bool(teaching)
    cells += len(teaching)
    displayed += display_count
    rows.append({"lesson_id": lesson, "authored_source_sha256": inventory_pages[lesson]["source_sha256"], "notebook_sha256": sha(notebook_path), "raw_sha256": sha(target), "HTML_sha256": sha(body_path), "code_DOM_sha256": digest(parser.python_blocks), "output_DOM_sha256": digest(output_dom(body)), "method": record["method"]})


def actual_http(item):
    values = {"lesson_id": item["id"]}
    for kind, path in [("HTML", item["id"] + ".html"), ("notebook", item["notebook"])]:
        with urllib.request.urlopen(BASE + path, timeout=15) as response:
            data = response.read()
            assert response.status == 200
        assert data == (SITE / path).read_bytes(), (item["id"], kind)
        if kind == "notebook":
            assert data == (ROOT / path).read_bytes()
        values[kind + "_sha256"] = hashlib.sha256(data).hexdigest()
    return values


with ThreadPoolExecutor(max_workers=6) as pool:
    http_rows = list(pool.map(actual_http, index))
row_path = PRIVATE / "all331-source-code-output-HTML-download-rows.json"
row_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
info_path = SITE / "build-info.json"
info = json.loads(info_path.read_bytes())
with urllib.request.urlopen(BASE + "build-info.json", timeout=15) as response:
    assert response.status == 200 and response.read() == info_path.read_bytes()
assert info["colab_mode"] == "upload" and info["publication_ready"] is False and info["colab_git_objects_verified"] is False
site_author = json.loads(SITE_RECEIPT.read_bytes())
assert site_author["source_freeze"] == ref(C) and site_author["notebook_execution_receipt"] == ref(R)
assert site_author["reading_estimate_filter_receipt"] == ref(TIME_RECEIPT)
assert site_author["build_info"] == info and site_author["build_info_sha256"] == sha(info_path)
old57 = json.loads((ROOT / "outputs/omni-notebooks-technical-fix14-current/05/5.7.ipynb").read_bytes())
new57 = json.loads((ROOT / "notebooks/05/5.7.ipynb").read_bytes())
changed_cells = [i for i, (l, r) in enumerate(zip(old57["cells"], new57["cells"], strict=True)) if l["cell_type"] != r["cell_type"] or l["source"] != r["source"]]
assert changed_cells == [4] and new57["cells"][4]["cell_type"] == "markdown"
assert "降低載入時執行任意程式的風險" in "".join(new57["cells"][4]["source"])
new57_raw = json.loads((BUNDLE / "05/5.7.ipynb").read_bytes())
stdout57 = ["".join(o["text"]) if isinstance(o["text"], list) else o["text"] for c in new57_raw["cells"] if c["cell_type"] == "code" and not c.get("metadata", {}).get("course_setup") for o in c.get("outputs", []) if o["output_type"] == "stream" and o["name"] == "stdout"]
assert "保存步數 1" in "".join(stdout57) and "trainable_parameters" in "".join(stdout57)
lineage = []
for fix, lesson in [("06", "9.11"), ("09", "9.18"), ("11", "12.17"), ("12", "9.12"), ("13", "9.15"), ("14", "12.2"), ("15", "5.7")]:
    receipt_path = ROOT / f"docs/course-revision-20261029-omni/evidence/notebooks-technical-fix{fix}-cumulative.json"
    receipt = json.loads(receipt_path.read_bytes())
    record = next(x for x in receipt["records"] if x["lesson"] == lesson)
    validation_path = ROOT / f"outputs/omni-notebooks-technical-fix{fix}-run/validation-kernel.json"
    validation = json.loads(validation_path.read_bytes())
    assert validation["total"] == validation["passed"] == 1 and validation["failed"] == 0 and not validation["failures"]
    assert record["method"] == "new_independent_kernel"
    assert sha(Path(record["source"])) == record["executed_sha256"]
    assert Path(record["source"]).read_bytes() == Path(records[lesson]["bundle_path"]).read_bytes()
    lineage.append({"lesson_id": lesson, "author_receipt": ref(receipt_path), "kernel_validation": ref(validation_path), "actual_new_kernel_raw": ref(Path(record["source"])), "current_bundle_same_bytes": True})
base_times = json.loads(TIMES.read_bytes())
preview_times = json.loads(PREVIEW_TIMES.read_bytes())
base_rows = {p["page_id"]: p for p in base_times["pages"]}
preview_rows = {p["page_id"]: p for p in preview_times["pages"]}
stale = sorted(page_id for page_id, row in base_rows.items() if row["source_sha256"] != inventory_pages[page_id]["source_sha256"] or row["figures_sha256"] != inventory_pages[page_id]["figures_sha256"])
assert stale == ["12.2", "5.7", "chapter-10", "chapter-14", "first-steps"]
assert preview_times["pages"] == [p for p in base_times["pages"] if p["page_id"] not in stale]
assert set(preview_times) - set(base_times) == {"preview_scope"} and not set(base_times) - set(preview_times)
for key in base_times:
    if key != "pages":
        assert preview_times[key] == base_times[key]
assert preview_times["preview_scope"] == {"source_freeze": ref(C), "kind": "ignored_nonformal_review_preview_only", "omitted_stale_or_missing": stale, "new_estimates_fabricated": False, "formal_publication_ready": False}
estimates = load_estimates(PREVIEW_TIMES, inventory)
assert len(base_rows) == 369 and len(estimates) == len(preview_rows) == info["reading_time_pages"] == 364
assert info["reading_time_complete"] is False
filter_receipt = json.loads(TIME_RECEIPT.read_bytes())
assert filter_receipt["preview_metadata_sha256"] == sha(PREVIEW_TIMES)
assert filter_receipt["all_pending_current_reader_estimates"] == stale
rendered = []
for page_id, page in inventory_pages.items():
    body = (SITE / Path(page["page_path"]).with_suffix(".html")).read_text()
    present = 'class="reading-time"' in body
    assert present == (page_id in estimates), page_id
    if present:
        assert "閱讀約 " + format_range((estimates[page_id]["minutes_min"], estimates[page_id]["minutes_max"])) + " · AI 估計" in body
        rendered.append(page_id)
guards = []
for page_id in stale:
    sample = PRIVATE / ("stale-" + page_id + ".json")
    sample.write_text(json.dumps({"schema_version": 1, "estimate_kind": "ai_estimate", "pages": [base_rows[page_id]]}, ensure_ascii=False))
    try:
        load_estimates(sample, inventory)
    except ValueError as error:
        assert "不是目前正文" in str(error)
        guards.append({"case": "reject-stale", "page_id": page_id, "message": str(error)})
    else:
        raise AssertionError("Accepted stale estimate: " + page_id)
try:
    load_estimates(PREVIEW_TIMES, inventory, require_complete=True)
except ValueError as error:
    assert "缺少 5 頁閱讀時間" in str(error)
    guards.append({"case": "reject-incomplete-formal-estimates", "message": str(error)})
else:
    raise AssertionError("Incomplete preview accepted as formal complete")
sync, sync_receipt = run("current-source-notebook-sync", [sys.executable, str(ROOT / "scripts/build_course.py"), "--check"])
assert sync.returncode == 0
site_check, site_check_receipt = run("actual-site-check", [sys.executable, str(ROOT / "scripts/check_site.py"), "--site", str(SITE)])
assert site_check.returncode == 0
fixed_originals = json.loads((A / "official-source-inspection.json").read_bytes())
for original in fixed_originals:
    assert sha(Path(original["raw_path"])) == original["raw_sha256"]
stable = json.loads((A / "fix14-stable-authorities-v1.json").read_bytes())
assert sha(ROOT / stable["installed_shortcut_schema"]["path"]) == stable["installed_shortcut_schema"]["sha256"]
git_doc = stable["Git_cat_file_original"]
assert sha(Path(git_doc["path"])) == git_doc["sha256"]
for item in preserved_reports + dependencies + [ref(C), ref(F), ref(R), ref(SITE_RECEIPT), ref(TIME_RECEIPT)]:
    assert sha(ROOT / item["path"]) == item["sha256"]
environment = {"python": platform.python_version(), "platform": platform.platform(), "device": "cpu", **{p: importlib.metadata.version(p) for p in ["torch", "jupyterlab", "ipykernel", "nbclient", "zensical"]}}
result = {"started_at": started, "created_at": datetime.now(UTC).isoformat(), "reviewer_task": prior["reviewer_task"], "lesson_id": "W.1", "verdict": "pass", "source_freeze": ref(F), "canonical_source_freeze": ref(C), "W1_snapshot": {"path": w1["snapshot"], "sha256": w1["source_sha256"]}, "W1_full_slice_same_bytes_empty_figure_map": True, "numbered_sections_hashes_unchanged": len(numbered["pages"]), "canonical_delta_from14": delta, "preserved_reports": preserved_reports, "preserved_prior_artifacts": len(prior["artifacts"]), "current_direct_dependencies": dependencies, "scope_inventory_fingerprint_recheck": {**ref(scope_path), "paths_rehashed": 210, "all_same": True, "scope": "Fingerprint inventory only, not a fresh210-file tool audit; necessary W1 software semantics remain unchanged."}, "author_notebook_bundle": ref(R), "author_preview_site_receipt": ref(SITE_RECEIPT), "raw_bundle_root": str(BUNDLE), "counts": {"canonical_inventory": len(inventory_pages), "source_notebook_raw_manifest_matches": 331, "teaching_code_sections": sections, "teaching_code_cells": cells, "displayed_teaching_cells": displayed, "visible_code_DOM_exact_matches": 331, "output_DOM_content_order_exact_matches": 331, "wrapper_only_formatting_pages": wrapper_only, "actual_preview_pages": len(list(SITE.glob('*.html'))), "actual_download_byte_matches": 331, "HTTP200_exact_HTML": len(http_rows), "HTTP200_current_downloads": len(http_rows), "rendered_current_estimates": len(rendered)}, "all331_per_page_rows": ref(row_path), "all331_fingerprint": digest(rows), "HTTP331_HTML_and_download_fingerprint": digest(http_rows), "output_parser_reused_verbatim": ref(A / "output_dom_v1.py"), "output_parser_prior_negative_controls": ref(A / "fix14-output-DOM-negative-controls-v1.json"), "new5_7_bounded_dependency": {"changed_cells": changed_cells, "cell_type": "markdown", "teaching_code_source_unchanged": True, "new_notebook": ref(ROOT / "notebooks/05/5.7.ipynb"), "new_kernel_raw": ref(BUNDLE / "05/5.7.ipynb"), "teaching_stdout": stdout57, "scope": "Source/code/output consistency and genuine kernel provenance only; not a fresh technical sign-off on5.7 or chapters10/14."}, "kernel_lineage": lineage, "actual_preview": {"root": str(SITE), "url": BASE, "build_info": info, "build_info_ref": ref(info_path), "HTTP_build_info_same_bytes": True, "all331_visible_code_and_output_content_order_checked": True, "all331_download_and_manual_upload_footers_checked": True, "check_site": site_check_receipt}, "preview_estimate_binding": {"original_metadata": ref(TIMES), "preview_metadata": ref(PREVIEW_TIMES), "author_filter_receipt": ref(TIME_RECEIPT), "original_rows": 369, "unchanged_retained_rows": len(estimates), "omitted_stale": stale, "all_existing_metadata_and_retained_rows_exact": True, "only_extra_key": "preview_scope", "current_descriptive_scope": preview_times["preview_scope"], "guards_actually_run": guards, "scope": "Metadata source/figure bindings and display only; no estimate synthesis or authentication of reader answers/completion."}, "source_notebook_sync": sync_receipt, "unchanged_evidence_reuse": {"authority_originals_rehashed": len(fixed_originals), "installed_shortcut_and_Git_docs_rehashed": True, "prior_CPU_smoke": ref(A / "fix14-current-recheck-v3.json"), "prior_fixed_authorities": ref(A / "fix14-stable-authorities-v1.json"), "note": "Current1.1/bootstrap/check_env/project/lock and exporter/Git gate bytes are unchanged; prior true executions reused honestly. No new kernel or cloud/UI/OS runtime claimed."}, "review_driver_scope_observation": {"failure": ref(A / "fix15-first-scope-observation-v1.json"), "first_driver": ref(A / "recheck-fix15-current-v1.py"), "resolution": "Optional preview_scope changed from descriptive string to exact source15-bound object with five omitted IDs and false fabrication/publication flags. All existing top-level metadata and retained rows remain exact; old equality assumption rejected, existing data guards unchanged. All331 checks repeated in this successful driver."}, "existing_issue_continued_closure": {"id": "w1-current-dependency-drift", "status": "resolved", "original_issue": False, "note": "Keep every issue field from accepted current4; new15 raw/site/HTTP are personally checked here, not relabeled14 evidence."}, "no_new_export": "Existing actual source15 preview reused; no shared output/zensical lock or site mutated.", "environment": environment, "limitations": ["Only W1 and required global source/Notebook/output/download/footer dependencies; not fresh subject review of5.7, chapters10/14, other W or the210 tools.", "All331 sources/raw/visible code/output DOM/HTML/downloads actually checked. This reviewer did not execute331 new kernels; root newly ran5.7, others are330 source-matched reused copies.", "Actual preview364 current estimates, five missing IDs disclosed; formal complete guard rejected. No new minutes, reader authentication, wave02 completion or stage seal.", "No new export, Colab UI viewing/upload/login/Connect/runtime, Windows/macOS execution, GPU, long/formal training, paid operation, commit, deployment or publication.", "Original792f snapshot only authenticates original source. Current exporter662355/check_site084027 and all direct dependencies were rehashed and their unchanged semantics reviewed.", "All original/current/schema-corrected versions and prior diagnostics/failures retained. FullCLI and per-page rows remain outside Git."]}
target = A / "fix15-current-recheck-v2.json"
with target.open("x", encoding="utf-8") as file:
    json.dump(result, file, ensure_ascii=False, indent=2)
    file.write("\n")
print(json.dumps({"receipt": str(target.relative_to(ROOT)), "sha256": sha(target), "created_at": result["created_at"], "verdict": result["verdict"], "counts": result["counts"], "pending_estimate_ids": stale}, ensure_ascii=False, indent=2))
